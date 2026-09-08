//! Cross-process shared mappings for no-copy provider leases.
//!
//! An owner creates a shared-memory object and passes its descriptor to a
//! provider process over an existing Unix-domain socket with `SCM_RIGHTS`.
//! Both processes map the same physical pages, so a provider that imports the
//! mapping through `metal_api_core::provider::NoCopyLeaseImporter` reads and
//! writes the owner's bytes in place instead of staging a copy.
//!
//! On Linux the object is an anonymous `memfd_create` descriptor; on other
//! Unix targets the module falls back to `shm_open` and unlinks the name
//! immediately, so no filesystem entry survives the mapping.

use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::ptr::NonNull;

/// A writable shared mapping backed by a process-shared memory object.
///
/// The mapping starts at a page boundary, which satisfies the
/// `VK_EXT_external_memory_host` alignment reported by the Vulkan provider on
/// systems with a 4096-byte `minImportedHostPointerAlignment`.
#[derive(Debug)]
pub struct SharedMemory {
    descriptor: OwnedFd,
    pointer: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is process-shared memory owned by this handle. Sending
// the handle moves that ownership, and reads through `as_slice` are safe from
// any thread while no writer holds `&mut self`.
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create and map a shared object of `len` bytes.
    ///
    /// The object is anonymous: on Linux it never gets a filesystem name, and
    /// elsewhere the name is unlinked before this function returns.
    pub fn create(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Err(invalid_input("shared mapping length is zero"));
        }
        let descriptor = create_object(len)?;
        Self::map_owned(descriptor, len)
    }

    /// Take ownership of a descriptor received from [`recv_fd`] and map it.
    ///
    /// The length comes from `fstat`, so the two processes do not need to
    /// exchange it separately.
    pub fn from_owned_fd(descriptor: OwnedFd) -> io::Result<Self> {
        let len = descriptor_len(descriptor.as_raw_fd())?;
        Self::map_owned(descriptor, len)
    }

    fn map_owned(descriptor: OwnedFd, len: usize) -> io::Result<Self> {
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
        })
    }

    /// Borrow the descriptor for [`send_fd`].
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
        // SAFETY: the pointer and length are exactly what `mmap` returned.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.len);
        }
    }
}

/// Send one descriptor over a connected Unix socket.
///
/// The control message carries a single payload byte, so the receiver must
/// call [`recv_fd`] exactly once before reading any other bytes from the
/// stream. The descriptor is duplicated into the receiver's table; the sender
/// keeps its own handle.
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

fn descriptor_len(descriptor: RawFd) -> io::Result<usize> {
    // SAFETY: `stat` is a plain C struct that accepts an all-zero value.
    let mut status: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `status` is writable and `descriptor` is open.
    if unsafe { libc::fstat(descriptor, &mut status) } != 0 {
        return Err(io::Error::last_os_error());
    }
    usize::try_from(status.st_size).map_err(|_| invalid_input("shared object is too large to map"))
}

fn set_len(descriptor: RawFd, len: usize) -> io::Result<()> {
    let len = libc::off_t::try_from(len)
        .map_err(|_| invalid_input("shared mapping length overflows off_t"))?;
    // SAFETY: `descriptor` is an open shared-memory object.
    if unsafe { libc::ftruncate(descriptor, len) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
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

#[cfg(not(target_os = "linux"))]
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn refuses_an_empty_mapping() {
        let error = SharedMemory::create(0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn reports_a_closed_socket() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        drop(sender);
        let error = recv_fd(&receiver).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }
}
