//! Echo one completion frame at a time from stdin to stdout.
//!
//! This helper exists for the cross-process integration test. It validates
//! every decoded message and exits cleanly on end of stream, so the test
//! exercises the same framing two provider processes will use.

use metal_api_ipc::codec::{CodecError, CompletionCodec};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();
    loop {
        match CompletionCodec::read(&mut reader) {
            Ok(message) => {
                message.validate()?;
                CompletionCodec::write(&mut writer, &message)?;
                writer.flush()?;
            }
            Err(CodecError::Eof) => return Ok(()),
            Err(error) => return Err(error.into()),
        }
    }
}
