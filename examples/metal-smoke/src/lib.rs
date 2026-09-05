//! Shared offline compute fixtures for the standalone and reims executors.

use metal_api_core::{ComputeExecutor, Device, Library, Size};
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

pub fn run_suite(
    label: &'static str,
    device_name: &str,
    executor: Arc<dyn ComputeExecutor>,
) -> Result<(), Box<dyn Error>> {
    println!("Metal API executor: {label}");
    println!("Metal API Vulkan device: {device_name}");
    let device = Device::new(executor);
    run_copy_word(&device)?;
    run_binary_air_copy_word(&device)?;
    run_indexed_boundary_dispatch(&device)?;
    Ok(())
}

fn run_copy_word(device: &Device) -> Result<(), Box<dyn Error>> {
    let library = device.new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?;
    let word = execute_copy_word(device, library)?;
    println!("PASS copy_word output={word:#010x}");
    Ok(())
}

fn run_binary_air_copy_word(device: &Device) -> Result<(), Box<dyn Error>> {
    let raw = assemble_owned_air(include_str!("../shaders/kernel_copy_word.ll"))?;
    let wrapped = wrap_air_bitcode(&raw)?;
    for (encoding, air) in [("raw", raw), ("wrapper", wrapped)] {
        let library = device.new_library_with_binary_air(air)?;
        let word = execute_copy_word(device, library)?;
        println!("PASS binary_air_copy_word encoding={encoding} output={word:#010x}");
    }
    Ok(())
}

fn execute_copy_word(device: &Device, library: Library) -> Result<u32, Box<dyn Error>> {
    let function = library.function("copy_word")?;
    let pipeline = device.new_compute_pipeline_state(&function)?;
    let input = device.new_buffer_with_bytes(0x6745_2301_u32.to_le_bytes())?;
    let output = device.new_buffer_with_bytes(0xabab_abab_u32.to_le_bytes())?;
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    let mut encoder = command.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline)?;
    encoder.set_buffer(0, &input, 0)?;
    encoder.set_buffer(1, &output, 0)?;
    encoder.dispatch_threads(Size::new(1, 1, 1)?, Size::new(1, 1, 1)?)?;
    encoder.end_encoding()?;
    command.commit()?;
    command.wait_until_completed()?;

    let bytes = output.read()?;
    let word = u32::from_le_bytes(bytes.as_slice().try_into()?);
    if word != 0x6745_2301 {
        return Err(format!("copy_word returned {word:#010x}, expected 0x67452301").into());
    }
    Ok(word)
}

fn assemble_owned_air(source: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let tool = std::env::var_os("METAL_API_LLVM_AS").unwrap_or_else(|| "llvm-as".into());
    let output_path = std::env::temp_dir().join(format!(
        "metal-api-smoke-{}-copy-word.air",
        std::process::id()
    ));
    let _cleanup = TemporaryFile(output_path.clone());
    let mut child = Command::new(&tool)
        .arg("-")
        .arg("-o")
        .arg(&output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("start {:?}: {error}", tool))?;
    child
        .stdin
        .take()
        .ok_or("llvm-as stdin was not piped")?
        .write_all(source.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!(
            "llvm-as failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let air = std::fs::read(&output_path)?;
    if !air.starts_with(&[0x42, 0x43, 0xc0, 0xde]) {
        return Err("llvm-as did not produce raw LLVM bitcode".into());
    }
    Ok(air)
}

fn wrap_air_bitcode(bitcode: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let size = u32::try_from(bitcode.len()).map_err(|_| "binary AIR is larger than u32")?;
    let mut wrapper = vec![0_u8; 0x14];
    wrapper[0..4].copy_from_slice(&[0xde, 0xc0, 0x17, 0x0b]);
    wrapper[8..12].copy_from_slice(&0x14_u32.to_le_bytes());
    wrapper[12..16].copy_from_slice(&size.to_le_bytes());
    wrapper.extend_from_slice(bitcode);
    Ok(wrapper)
}

struct TemporaryFile(PathBuf);

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run_indexed_boundary_dispatch(device: &Device) -> Result<(), Box<dyn Error>> {
    let library = device.new_library_with_air(include_str!(
        "../shaders/kernel_dispatch_threads_boundary_barrier.ll"
    ))?;
    let function = library.function("kernel_dispatch_threads_boundary_barrier")?;
    let pipeline = device.new_compute_pipeline_state(&function)?;
    let output = device.new_buffer_with_bytes(vec![0xaa; 30 * size_of::<u32>()])?;
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    let mut encoder = command.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline)?;
    encoder.set_buffer(0, &output, 0)?;
    encoder.dispatch_threads(Size::new(10, 3, 1)?, Size::new(8, 2, 1)?)?;
    encoder.end_encoding()?;
    command.commit()?;
    command.wait_until_completed()?;

    let mut expected = Vec::with_capacity(30 * size_of::<u32>());
    for y in 0..3 {
        for x in 0..10 {
            let local_x = if x < 8 { 8_u32 } else { 2 };
            let local_y = if y < 2 { 2_u32 } else { 1 };
            expected.extend_from_slice(&(local_y * 100 + local_x).to_le_bytes());
        }
    }
    let actual = output.read()?;
    if actual != expected {
        return Err("indexed boundary dispatch did not match the Metal golden output".into());
    }
    println!("PASS indexed_boundary_dispatch words=30 regions=4");
    Ok(())
}
