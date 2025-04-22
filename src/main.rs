use {
    clap::Parser,
    ruminant::{DisplayMode, Engine, EngineOpts},
    std::io::Read,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct RumArgs {
    #[arg(long, short)]
    file: String,

    #[clap(short, long, default_value_t, value_enum)]
    display_mode: DisplayMode,
}

fn main() -> Result<(), String> {
    let args = RumArgs::parse();

    println!("Display mode is {:?}", args.display_mode);

    let mut engine = Engine::new(EngineOpts {
        display_mode: args.display_mode,
    });

    let mut source = String::new();
    let mut file = std::fs::File::open(&args.file).map_err(|e| format!("{:?}", e))?;
    file.read_to_string(&mut source)
        .map_err(|e| format!("{:?}", e))?;

    engine.run(source.as_str())?;

    Ok(())
}
