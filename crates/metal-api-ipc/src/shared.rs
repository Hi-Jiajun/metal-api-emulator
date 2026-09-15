//! Cross-process shared mappings for no-copy provider leases.
//!
//! An owner creates a shared-memory object and passes it to a provider
//! process. Unix transports can pass the descriptor with `SCM_RIGHTS`; a
//! transport without descriptor passing instead sends a named mapping, which
//! works on Unix (`shm_open`) and Windows (`CreateFileMappingW`). Both
//! processes map the same physical pages, so a provider that imports the
//! mapping through `metal_api_core::provider::NoCopyLeaseImporter` reads and
//! writes the owner's bytes in place instead of staging a copy.
//!
//! On Linux the anonymous object is an `memfd_create` descriptor; on other
//! Unix targets the module falls back to `shm_open` and unlinks the name
//! immediately, so no filesystem entry survives the mapping. Named mappings
//! keep their name until the creator drops the handle, because the provider
//! opens the object by name.
//!
//! # Safety boundary
//!
//! A named mapping is visible to every process in the same session. The
//! generated name includes the process id, a monotonic sequence and the
//! current time, and the creator keeps the object alive until the provider
//! releases the lease, so an unrelated process cannot reuse the name while it
//! is in use. The owner must keep the mapping alive until
//! `release_borrowed_lease` returns.

use std::io::{self, Read, Write};
use std::ptr::NonNull;

#[cfg(unix)]
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
#[cfg(unix)]
use std::os::unix::net::UnixStream;

/// Largest accepted mapping name in the named-mapping handshake.
pub const MAX_MAPPING_NAME: usize = 4096;

/// A writable shared mapping backed by a process-shared memory object.
///
/// The mapping starts at a page boundary, which satisfies the
/// `VK_EXT_external_memory_host` alignment reported by the Vulkan provider on
/// systems with a 4096-byte `minImportedHostPointerAlignment`.
#[derive(Debug)]
pub struct SharedMemory {
    #[cfg(unix)]
    descriptor: OwnedFd,
    #[cfg(windows)]
    handle: *mut std::ffi::c_void,
    pointer: NonNull<u8>,
    len: usize,
    /// Name exported by [`SharedMemory::create_named`], or the name used by
    /// [`SharedMemory::from_name`] to open the object.
    name: Option<String>,
    /// Unix shared-memory name to remove when the creator drops the mapping.
    #[cfg(unix)]
    unlink: Option<std::ffi::CString>,
}

// SAFETY: the mapping is process-shared memory owned by this handle. Sending
// the handle moves that ownership, and reads through `as_slice` are safe from
// any thread while no writer holds `&mut self`.
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create and map an anonymous shared object of `len` bytes.
    ///
    /// On Linux the object is an anonymous `memfd_create` descriptor; on other
    /// Unix targets the `shm_open` name is unlinked before this function
    /// returns. Windows creates an unnamed pagefile-backed section.
    pub fn create(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(invalid_input("shared mapping length is zero"));
        }
        #[cfg(unix)]
        {
            Self::map_owned(create_object(len)?, len, None, None)
        }
        #[cfg(windows)]
        {
            Self::map_section(create_section(len, None)?, len, None)
        }
    }

    /// Create and map a named shared object, returning the exported name.
    ///
    /// The provider opens the same object with [`SharedMemory::from_name`].
    /// The creator keeps the name alive until this handle is dropped, so the
    /// mapping must outlive every provider import.
    pub fn create_named(len: usize) -> io::Result<(Self, String)> {
        if len == 0 {
            return Err(invalid_input("shared mapping length is zero"));
        }
        #[cfg(unix)]
        {
            let (descriptor, name, unlink) = create_named_object(len)?;
            let memory = Self::map_owned(descriptor, len, Some(name.clone()), Some(unlink))?;
            Ok((memory, name))
        }
        #[cfg(windows)]
        {
            let name = generate_name();
            let handle = create_section(len, Some(&name))?;
            let memory = Self::map_section(handle, len, Some(name.clone()))?;
            Ok((memory, name))
        }
    }

    /// Open and map a named object created by [`SharedMemory::create_named`].
    ///
    /// `len` must not exceed the creator's mapping length. The returned handle
    /// never unlinks the name; only the creator owns it.
    pub fn from_name(name: &str, len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(invalid_input("shared mapping length is zero"));
        }
        #[cfg(unix)]
        {
            let descriptor = open_named_object(name)?;
            if descriptor_len(descriptor.as_raw_fd())? < len {
                return Err(invalid_input("named mapping is smaller than requested"));
            }
            Self::map_owned(descriptor, len, Some(name.to_string()), None)
        }
        #[cfg(windows)]
        {
            Self::map_section(open_section(name)?, len, Some(name.to_string()))
        }
    }

    /// Take ownership of a descriptor received from [`recv_fd`] and map it.
    ///
    /// The length comes from `fstat`, so the two processes do not need to
    /// exchange it separately.
    #[cfg(unix)]
    pub fn from_owned_fd(descriptor: OwnedFd) -> io::Result<Self> {
        let len = descriptor_len(descriptor.as_raw_fd())?;
        Self::map_owned(descriptor, len, None, None)
    }

    #[cfg(unix)]
    fn map_owned(
        descriptor: OwnedFd,
        len: usize,
        name: Option<String>,
        unlink: Option<std::ffi::CString>,
    ) -> io::Result<Self> {
        if len == 0 {
            return Err(invalid_input("shared mapping length is zero"));
        }
        // SAFETY: `descriptor` is an open shared-memory object of at least
        // `len` bytes and the protection and flags are valid for `mmap`.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                descriptor.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let pointer = NonNull::new(pointer.cast::<u8>())
            .ok_or_else(|| io::Error::other("shared mapping returned a null pointer"))?;
        Ok(Self {
            descriptor,
            pointer,
            len,
            name,
            unlink,
        })
    }

    #[cfg(windows)]
    fn map_section(
        handle: *mut std::ffi::c_void,
        len: usize,
        name: Option<String>,
    ) -> io::Result<Self> {
        // SAFETY: `handle` is an open file mapping and the access and offset
        // arguments are valid.
        let pointer =
            unsafe { win32::MapViewOfFile(handle, win32::FILE_MAP_ALL_ACCESS, 0, 0, len) };
        let pointer = match NonNull::new(pointer.cast::<u8>()) {
            Some(pointer) => pointer,
            None => {
                let error = io::Error::last_os_error();
                // SAFETY: `handle` is owned by this function and not used
                // after the failed map.
                unsafe {
                    win32::CloseHandle(handle);
                }
                return Err(error);
            }
        };
        Ok(Self {
            handle,
            pointer,
            len,
            name,
        })
    }

    /// Exported name for a named mapping, if this handle has one.
    pub fn mapping_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Borrow the descriptor for [`send_fd`].
    #[cfg(unix)]
    pub fn descriptor(&self) -> BorrowedFd<'_> {
        self.descriptor.as_fd()
    }

    /// Number of mapped bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is empty, which [`SharedMemory::create`] rejects.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Base address of the mapping. It is aligned to the system page size.
    pub fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    /// Read the mapped bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the pointer covers `len` initialized bytes that stay valid
        // until this handle is dropped.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    /// Mutate the mapped bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` excludes concurrent readers and the mapping is
        // writable for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the pointer and length are exactly what `mmap` returned.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.len);
        }
        #[cfg(unix)]
        if let Some(name) = self.unlink.take() {
            // SAFETY: `name` is a NUL-terminated name previously created by
            // `shm_open`; the provider keeps its own mapping alive even after
            // the name disappears.
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
        }
        #[cfg(windows)]
        // SAFETY: the view and handle are exactly what `MapViewOfFile` and
        // `CreateFileMappingW` returned.
        unsafe {
            win32::UnmapViewOfFile(self.pointer.as_ptr().cast());
            win32::CloseHandle(self.handle);
        }
    }
}

/// Write one named mapping after the current command frame.
///
/// The receiver must call [`recv_mapping`] exactly once before reading any
/// other bytes from the stream. The name identifies an object the creator
/// keeps alive; the length bounds the provider's view.
pub fn send_mapping<W: Write>(writer: &mut W, name: &str, len: usize) -> io::Result<()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_MAPPING_NAME {
        return Err(invalid_input("mapping name length is out of range"));
    }
    if len == 0 {
        return Err(invalid_input("mapping length is zero"));
    }
    writer.write_all(&(len as u64).to_le_bytes())?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()
}

/// Read one named mapping written by [`send_mapping`].
pub fn recv_mapping<R: Read>(reader: &mut R) -> io::Result<(String, usize)> {
    let mut raw_len = [0_u8; 8];
    reader.read_exact(&mut raw_len)?;
    let len = usize::try_from(u64::from_le_bytes(raw_len))
        .map_err(|_| invalid_input("mapping length is too large"))?;
    if len == 0 {
        return Err(invalid_input("mapping length is zero"));
    }
    let mut raw_name = [0_u8; 4];
    reader.read_exact(&mut raw_name)?;
    let name_len = u32::from_le_bytes(raw_name) as usize;
    if name_len == 0 || name_len > MAX_MAPPING_NAME {
        return Err(invalid_input("mapping name length is out of range"));
    }
    let mut bytes = vec![0_u8; name_len];
    reader.read_exact(&mut bytes)?;
    let name = String::from_utf8(bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "mapping name is not valid UTF-8",
        )
    })?;
    Ok((name, len))
}

/// Send one descriptor over a connected Unix socket.
///
/// The control message carries a single payload byte, so the receiver must
/// call [`recv_fd`] exactly once before reading any other bytes from the
/// stream. The descriptor is duplicated into the receiver's table; the sender
/// keeps its own handle.
#[cfg(unix)]
pub fn send_fd(socket: &UnixStream, descriptor: BorrowedFd<'_>) -> io::Result<()> {
    let mut payload = 0_u8;
    let mut vector = libc::iovec {
        iov_base: (&mut payload as *mut u8).cast(),
        iov_len: 1,
    };
    // SAFETY: the control buffer is sized for exactly one descriptor.
    let control_len = unsafe { libc::CMSG_SPACE(size_of::<RawFd>() as u32) } as usize;
    let mut control = vec![0_u8; control_len];
    // SAFETY: `msghdr` is a plain C struct that accepts an all-zero value.
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut vector;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    // The field is `size_t` on Linux and `socklen_t` on macOS.
    #[allow(clippy::unnecessary_cast)]
    {
        message.msg_controllen = control.len() as _;
    }
    // SAFETY: `message` points at the iovec and control buffer above.
    let header = unsafe { libc::CMSG_FIRSTHDR(&message) };
    if header.is_null() {
        return Err(io::Error::other(
            "control buffer is too small for a descriptor",
        ));
    }
    // SAFETY: `header` points into `control`, which holds one descriptor after
    // `CMSG_SPACE`, and `CMSG_DATA` returns its aligned payload address.
    unsafe {
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        // The field is `size_t` on Linux and `socklen_t` on macOS.
        #[allow(clippy::unnecessary_cast)]
        {
            (*header).cmsg_len = libc::CMSG_LEN(size_of::<RawFd>() as u32) as _;
        }
        std::ptr::write_unaligned(
            libc::CMSG_DATA(header).cast::<RawFd>(),
            descriptor.as_raw_fd(),
        );
    }
    // SAFETY: `socket` is a connected Unix socket and `message` is fully
    // initialized with one iovec and one control message.
    let written = unsafe { libc::sendmsg(socket.as_raw_fd(), &message, 0) };
    if written < 0 {
        return Err(io::Error::last_os_error());
    }
    if written != 1 {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "descriptor message payload was truncated",
        ));
    }
    Ok(())
}

/// Receive one descriptor sent by [`send_fd`].
///
/// The returned handle owns the descriptor; dropping it closes only the
/// receiver's copy.
#[cfg(unix)]
pub fn recv_fd(socket: &UnixStream) -> io::Result<OwnedFd> {
    let mut payload = 0_u8;
    let mut vector = libc::iovec {
        iov_base: (&mut payload as *mut u8).cast(),
        iov_len: 1,
    };
    // SAFETY: the control buffer is sized for exactly one descriptor.
    let control_len = unsafe { libc::CMSG_SPACE(size_of::<RawFd>() as u32) } as usize;
    let mut control = vec![0_u8; control_len];
    // SAFETY: `msghdr` is a plain C struct that accepts an all-zero value.
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut vector;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    // The field is `size_t` on Linux and `socklen_t` on macOS.
    #[allow(clippy::unnecessary_cast)]
    {
        message.msg_controllen = control.len() as _;
    }
    #[cfg(target_os = "linux")]
    let flags = libc::MSG_CMSG_CLOEXEC;
    #[cfg(not(target_os = "linux"))]
    let flags = 0;
    // SAFETY: `message` points at the iovec and control buffer above.
    let received = unsafe { libc::recvmsg(socket.as_raw_fd(), &mut message, flags) };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }
    if received == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "socket closed before a descriptor arrived",
        ));
    }
    if message.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "descriptor control message was truncated",
        ));
    }
    // SAFETY: `message` was filled by `recvmsg` and points at `control`.
    let header = unsafe { libc::CMSG_FIRSTHDR(&message) };
    if header.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "message carried no control data",
        ));
    }
    // SAFETY: `header` points into `control`.
    let (level, kind, length) = unsafe {
        (
            (*header).cmsg_level,
            (*header).cmsg_type,
            (*header).cmsg_len,
        )
    };
    if level != libc::SOL_SOCKET || kind != libc::SCM_RIGHTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "message control data is not SCM_RIGHTS",
        ));
    }
    // SAFETY: the length of one descriptor is a valid control message size.
    let expected = unsafe { libc::CMSG_LEN(size_of::<RawFd>() as u32) } as u64;
    if (length as u64) < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "control message is too short to hold a descriptor",
        ));
    }
    // SAFETY: the header carries at least one descriptor payload.
    let raw = unsafe { std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<RawFd>()) };
    if raw < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SCM_RIGHTS carried an invalid descriptor",
        ));
    }
    #[cfg(not(target_os = "linux"))]
    // SAFETY: `raw` is a freshly received, valid descriptor.
    unsafe {
        libc::fcntl(raw, libc::F_SETFD, libc::FD_CLOEXEC);
    }
    // SAFETY: `raw` is a valid descriptor owned by this function and is not
    // used elsewhere after the conversion.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(unix)]
fn descriptor_len(descriptor: RawFd) -> io::Result<usize> {
    // SAFETY: `stat` is a plain C struct that accepts an all-zero value.
    let mut status: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `status` is writable and `descriptor` is open.
    if unsafe { libc::fstat(descriptor, &mut status) } != 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(status.st_size).map_err(|_| invalid_input("shared object is too large to map"))
}

#[cfg(unix)]
fn set_len(descriptor: RawFd, len: usize) -> io::Result<()> {
    let len = libc::off_t::try_from(len)
        .map_err(|_| invalid_input("shared mapping length overflows off_t"))?;
    // SAFETY: `descriptor` is an open shared-memory object.
    if unsafe { libc::ftruncate(descriptor, len) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(all(unix, target_os = "linux"))]
fn create_object(len: usize) -> io::Result<OwnedFd> {
    // SAFETY: the name is a valid NUL-terminated literal.
    let raw = unsafe { libc::memfd_create(c"metal-api-shared".as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `raw` is a fresh descriptor owned by this function.
    let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
    set_len(descriptor.as_raw_fd(), len)?;
    Ok(descriptor)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn create_object(len: usize) -> io::Result<OwnedFd> {
    use std::ffi::CString;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_NAME: AtomicU64 = AtomicU64::new(0);
    let sequence = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
    let name = CString::new(format!("/metal-api-{}-{sequence}", std::process::id()))
        .map_err(|_| invalid_input("shared object name contains a NUL byte"))?;
    // SAFETY: `name` is NUL-terminated and the flags and mode are valid.
    let raw = unsafe {
        libc::shm_open(
            name.as_ptr(),
            libc::O_CREAT | libc::O_EXCL | libc::O_RDWR | libc::O_CLOEXEC,
            0o600,
        )
    };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `raw` is a fresh descriptor owned by this function.
    let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: unlinking an open shared-memory object is supported by POSIX.
    // The descriptor keeps the object alive for every mapper.
    unsafe {
        libc::shm_unlink(name.as_ptr());
    }
    set_len(descriptor.as_raw_fd(), len)?;
    Ok(descriptor)
}

/// Create a named object and return its descriptor, name and unlink name.
///
/// The caller owns the name and must unlink it after the last mapper is gone.
#[cfg(unix)]
fn create_named_object(len: usize) -> io::Result<(OwnedFd, String, std::ffi::CString)> {
    use std::ffi::CString;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_NAME: AtomicU64 = AtomicU64::new(0);
    // Keep the name under the 31-byte macOS `PSHMNAMLEN` limit.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64 & 0xffff_ffff)
        .unwrap_or(0);
    for _ in 0..16 {
        let sequence = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
        let name = format!("/m{}-{sequence:x}-{stamp:x}", std::process::id());
        let unlink = CString::new(name.clone())
            .map_err(|_| invalid_input("shared object name contains a NUL byte"))?;
        // SAFETY: `unlink` is NUL-terminated and the flags and mode are valid.
        let raw = unsafe {
            libc::shm_open(
                unlink.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR | libc::O_CLOEXEC,
                0o600,
            )
        };
        if raw < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EEXIST) {
                continue;
            }
            return Err(error);
        }
        // SAFETY: `raw` is a fresh descriptor owned by this function.
        let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
        if let Err(error) = set_len(descriptor.as_raw_fd(), len) {
            // SAFETY: the object was created by this call and has no other
            // mapper yet.
            unsafe {
                libc::shm_unlink(unlink.as_ptr());
            }
            return Err(error);
        }
        return Ok((descriptor, name, unlink));
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "named shared mapping name is in use",
    ))
}

#[cfg(unix)]
fn open_named_object(name: &str) -> io::Result<OwnedFd> {
    let name = std::ffi::CString::new(name)
        .map_err(|_| invalid_input("shared object name contains a NUL byte"))?;
    // SAFETY: `name` is NUL-terminated and the flags are valid.
    let raw = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC, 0) };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `raw` is a fresh descriptor owned by this function.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::ffi::OsStrExt;

    pub const INVALID_HANDLE_VALUE: *mut c_void = -1_isize as *mut c_void;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const FILE_MAP_ALL_ACCESS: u32 = 0x000f_001f;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateFileMappingW(
            h_file: *mut c_void,
            file_mapping_attributes: *mut c_void,
            protect: u32,
            maximum_size_high: u32,
            maximum_size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn OpenFileMappingW(
            desired_access: u32,
            inherit_handle: i32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn MapViewOfFile(
            file_mapping_object: *mut c_void,
            desired_access: u32,
            file_offset_high: u32,
            file_offset_low: u32,
            number_of_bytes_to_map: usize,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(base_address: *const c_void) -> i32;
        pub fn CloseHandle(object: *mut c_void) -> i32;
    }

    /// Encode a section name as a NUL-terminated UTF-16 string.
    pub fn wide(name: &str) -> io::Result<Vec<u16>> {
        if name.is_empty() || name.contains('\0') {
            return Err(super::invalid_input("shared object name is invalid"));
        }
        let mut wide: Vec<u16> = std::ffi::OsStr::new(name).encode_wide().collect();
        wide.push(0);
        Ok(wide)
    }
}

#[cfg(windows)]
fn generate_name() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_NAME: AtomicU64 = AtomicU64::new(0);
    let sequence = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("Local\\metal-api-{}-{sequence}-{stamp}", std::process::id())
}

#[cfg(windows)]
fn create_section(len: usize, name: Option<&str>) -> io::Result<*mut std::ffi::c_void> {
    let wide = match name {
        Some(name) => Some(win32::wide(name)?),
        None => None,
    };
    let len =
        u64::try_from(len).map_err(|_| invalid_input("shared mapping length is too large"))?;
    // SAFETY: the name is NUL-terminated (or null) and the size arguments are
    // derived from `len`.
    let handle = unsafe {
        win32::CreateFileMappingW(
            win32::INVALID_HANDLE_VALUE,
            std::ptr::null_mut(),
            win32::PAGE_READWRITE,
            (len >> 32) as u32,
            len as u32,
            wide.as_ref().map_or(std::ptr::null(), |wide| wide.as_ptr()),
        )
    };
    if handle.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(handle)
}

#[cfg(windows)]
fn open_section(name: &str) -> io::Result<*mut std::ffi::c_void> {
    let wide = win32::wide(name)?;
    // SAFETY: `wide` is NUL-terminated and the access flags are valid.
    let handle = unsafe { win32::OpenFileMappingW(win32::FILE_MAP_ALL_ACCESS, 0, wide.as_ptr()) };
    if handle.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn passes_a_mapping_between_sockets() {
        let mut owner = SharedMemory::create(4096).unwrap();
        owner.as_mut_slice()[..4].copy_from_slice(&0xaaaa_aaaa_u32.to_le_bytes());
        let (sender, receiver) = UnixStream::pair().unwrap();
        send_fd(&sender, owner.descriptor()).unwrap();
        let descriptor = recv_fd(&receiver).unwrap();
        let mut provider = SharedMemory::from_owned_fd(descriptor).unwrap();
        assert_eq!(provider.len(), 4096);
        assert_eq!(&provider.as_slice()[..4], &0xaaaa_aaaa_u32.to_le_bytes());

        provider.as_mut_slice()[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        assert_eq!(&owner.as_slice()[..4], &0x1234_5678_u32.to_le_bytes());
    }

    #[test]
    fn passes_a_mapping_by_name() {
        let (mut owner, name) = SharedMemory::create_named(4096).unwrap();
        assert_eq!(owner.mapping_name(), Some(name.as_str()));
        owner.as_mut_slice()[..4].copy_from_slice(&0xaaaa_aaaa_u32.to_le_bytes());
        let mut provider = SharedMemory::from_name(&name, 4096).unwrap();
        assert_eq!(provider.len(), 4096);
        assert_eq!(&provider.as_slice()[..4], &0xaaaa_aaaa_u32.to_le_bytes());

        provider.as_mut_slice()[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
        assert_eq!(&owner.as_slice()[..4], &0x1234_5678_u32.to_le_bytes());
    }

    #[test]
    fn round_trips_a_mapping_handshake() {
        let mut bytes = Vec::new();
        send_mapping(&mut bytes, "example-mapping", 4096).unwrap();
        let (name, len) = recv_mapping(&mut bytes.as_slice()).unwrap();
        assert_eq!(name, "example-mapping");
        assert_eq!(len, 4096);
    }

    #[test]
    fn refuses_an_empty_mapping() {
        let error = SharedMemory::create(0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn reports_a_closed_socket() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        drop(sender);
        let error = recv_fd(&receiver).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }
}
