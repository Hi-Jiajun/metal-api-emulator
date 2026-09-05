use metal_api_vulkan::VulkanExecutor;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args_os().len() != 1 {
        return Err("usage: provider-smoke".into());
    }
    metal_smoke::run_provider_suite(VulkanExecutor::new()?)
}
