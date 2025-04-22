use clap::{Args, Parser, Subcommand};
use ruminant::Engine;
use std::io::Read;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct RumArgs {
    #[arg(long, short)]
    file: String,
}

fn main() -> Result<(), String> {
    let args = RumArgs::parse();

    let mut engine = Engine::new();

    let mut source = String::new();
    let mut file = std::fs::File::open(&args.file).map_err(|e| format!("{:?}", e))?;
    file.read_to_string(&mut source)
        .map_err(|e| format!("{:?}", e))?;

    engine.run(source.as_str())?;

    Ok(())
}
