use metal_api_vulkan::VulkanExecutor;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    match args.next() {
        None => metal_smoke::run_provider_suite(VulkanExecutor::new()?),
        Some(flag) if flag == "--completion-child" => {
            let socket = args
                .next()
                .ok_or("usage: provider-smoke --completion-child <socket>")?;
            if args.next().is_some() {
                return Err("usage: provider-smoke --completion-child <socket>".into());
            }
            metal_smoke::run_completion_child(&socket)
        }
        Some(other) => Err(format!("usage: provider-smoke (unknown argument {:?})", other).into()),
    }
}
