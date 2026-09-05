use metal_api_reims_vulkan::ReimsVulkanExecutor;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let executor = ReimsVulkanExecutor::new();
    metal_smoke::run_suite("reims", executor.device_name(), executor)?;
    println!("PASS suite executor=reims");
    Ok(())
}
