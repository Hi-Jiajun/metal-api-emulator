use metal_api_vulkan::VulkanExecutor;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [] => {}
        [flag, value] if flag == "--executor" && value == "standalone" => {}
        [flag, value] if flag == "--executor" && value == "reims" => {
            return Err("reims smoke moved to integration/reims; see its README.md".into());
        }
        _ => return Err("usage: metal-smoke [--executor standalone]".into()),
    }
    let executor = VulkanExecutor::new()?;
    let name = executor.device_name().to_string();
    metal_smoke::run_suite("standalone", &name, executor)?;
    println!("PASS suite executor=standalone");
    Ok(())
}
