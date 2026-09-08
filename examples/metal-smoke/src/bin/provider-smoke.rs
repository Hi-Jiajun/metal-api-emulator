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
        Some(flag) if flag == "--borrowed-shared-child" => {
            let socket = args
                .next()
                .ok_or("usage: provider-smoke --borrowed-shared-child <socket>")?;
            if args.next().is_some() {
                return Err("usage: provider-smoke --borrowed-shared-child <socket>".into());
            }
            metal_smoke::run_borrowed_shared_child(&socket)
        }
        Some(flag) if flag == "--command-child" => {
            let command_socket = args.next().ok_or(
                "usage: provider-smoke --command-child <command-socket> <completion-socket>",
            )?;
            let completion_socket = args.next().ok_or(
                "usage: provider-smoke --command-child <command-socket> <completion-socket>",
            )?;
            if args.next().is_some() {
                return Err(
                    "usage: provider-smoke --command-child <command-socket> <completion-socket>"
                        .into(),
                );
            }
            metal_smoke::run_provider_command_child(&command_socket, &completion_socket)
        }
        Some(flag) if flag == "--named-command-child" => {
            let command_addr = args.next().ok_or(
                "usage: provider-smoke --named-command-child <command-addr> <completion-addr>",
            )?;
            let completion_addr = args.next().ok_or(
                "usage: provider-smoke --named-command-child <command-addr> <completion-addr>",
            )?;
            if args.next().is_some() {
                return Err(
                    "usage: provider-smoke --named-command-child <command-addr> <completion-addr>"
                        .into(),
                );
            }
            metal_smoke::run_named_command_child(&command_addr, &completion_addr)
        }
        Some(other) => Err(format!("usage: provider-smoke (unknown argument {:?})", other).into()),
    }
}
