use clap::Parser;
use log::info;
use serde::{Deserialize, Serialize};
use std::{env, io::Write as _};

use log::LevelFilter;
use log4rs::{append::console::ConsoleAppender, config::{Appender, Root}, encode::pattern::PatternEncoder, Config as Log4rsConfig};

use mistral_test::rag::RAG;

pub fn default_log4rs_config() -> Log4rsConfig {
    let level = {
        let lvar = env::var("LOG_LEVEL");
        
        let lvl = match lvar.unwrap_or(String::from("info")).as_str() {
            "info" => LevelFilter::Info,
            "warn" => LevelFilter::Warn,
            "debug" => LevelFilter::Debug,
            "error" => LevelFilter::Error,
            "off" => LevelFilter::Off,
            "trace" => LevelFilter::Trace,
            _ => LevelFilter::Info
        };

        lvl

    };
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{h([{l}][{M}][{d}])} {m}{n}"     // [INFO][module][timestamp] message
        )))
        .build();

    Log4rsConfig::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(level))
        .unwrap()
}

#[derive(Parser)]
#[command(name = "app", version = "0.1.0", author = "Shubham Mishra <grapheo12@gmail.com>", about = "A sample RAG application")]
struct Cli {
    #[arg(short, long, long_help = "Path to the config file")]
    pub config: String,

    #[arg(short, long, long_help = "Query to ask the model")]
    pub query: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
enum DataSource {
    #[serde(rename = "path")]
    Local(String),
    #[serde(rename = "url")]
    Url(String),
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    pub embedding_model: String,
    pub query_model: String,
    pub data_source: DataSource,
}


#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let _ = log4rs::init_config(default_log4rs_config())?;
    let cli = Cli::parse();
    let config_text = std::fs::read_to_string(&cli.config)?;
    let config = toml::from_str::<Config>(&config_text)?;

    info!("Initializing RAG pipeline");

    let mut rag = RAG::new(config.embedding_model, config.query_model);
    rag.init().await?;

    info!("Loading data");
    match config.data_source {
        DataSource::Local(path) => {
            info!("Loading data from local file: {}", path);
            let data = std::fs::read_to_string(path)?;
            rag.load_data(data).await?;
        }
        DataSource::Url(url) => {
            info!("Loading data from URL: {}", url);
            let data = reqwest::get(url).await?;
            let data = data.text().await?;

            info!("Feeding data to RAG");
            rag.load_data(data).await?;
        }
    }

    if let Some(query) = cli.query {
        info!("Querying model: {}", query);
        let response = rag.query(query).await?;
        info!("Response:\n{}", response);
    } else {
        info!("No query provided, starting interactive mode");
        loop {
            let mut query = String::new();
            print!("> ");
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut query)?;
            let response = rag.query(query).await?;
            println!("{}", response);
        }
    }

    Ok(())
}