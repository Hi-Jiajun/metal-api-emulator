use metal_api_core::{ComputeExecutor, Device, Size};
use metal_api_reims_vulkan::ReimsVulkanExecutor;
use metal_api_vulkan::VulkanExecutor;
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let choice = match arguments.as_slice() {
        [] => "standalone",
        [flag, value] if flag == "--executor" => value,
        _ => return Err("usage: metal-smoke [--executor standalone|reims]".into()),
    };
    let name = match choice {
        "standalone" => {
            let executor = VulkanExecutor::new()?;
            let name = executor.device_name().to_string();
            run_suite("standalone", &name, executor)?;
            "standalone"
        }
        "reims" => {
            let executor = ReimsVulkanExecutor::new();
            run_suite("reims", executor.device_name(), executor)?;
            "reims"
        }
        _ => {
            return Err(format!("unknown executor {choice:?}; expected standalone or reims").into())
        }
    };
    println!("PASS suite executor={name}");
    Ok(())
}

fn run_suite(
    label: &'static str,
    device_name: &str,
    executor: Arc<dyn ComputeExecutor>,
) -> Result<(), Box<dyn Error>> {
    println!("Metal API executor: {label}");
    println!("Metal API Vulkan device: {device_name}");
    let device = Device::new(executor);
    run_copy_word(&device)?;
    run_indexed_boundary_dispatch(&device)?;
    Ok(())
}

fn run_copy_word(device: &Device) -> Result<(), Box<dyn Error>> {
    let library = device.new_library_with_air(include_str!("../shaders/kernel_copy_word.ll"))?;
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
    println!("PASS copy_word output={word:#010x}");
    Ok(())
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
