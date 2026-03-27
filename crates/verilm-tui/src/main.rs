//! VeriLM TUI - Interactive verification debugger and demo tool
//!
//! Modes:
//! - Live: Connect to running VeriLM server and visualize verification in real-time
//! - Replay: Load a receipt file and step through verification
//! - Demo: Run a standalone toy example with visualization

use anyhow::Result;
use clap::{Parser, Subcommand};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    crossterm::{
        event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    Terminal,
};
use std::io;
use tracing::{debug, info};

mod app;
mod components;
mod events;
mod views;

use app::{App, Mode};

#[derive(Parser)]
#[command(name = "verilm-tui")]
#[command(about = "Interactive TUI for VeriLM verification")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    
    /// Server URL for live mode
    #[arg(short, long, default_value = "ws://localhost:8000")]
    server: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Connect to live server and visualize real-time verification
    Live {
        /// Server WebSocket URL
        #[arg(short, long, default_value = "ws://localhost:8000")]
        url: String,
    },
    /// Replay a saved receipt file
    Replay {
        /// Path to receipt file
        receipt: String,
        /// Path to verifier key
        #[arg(short, long)]
        key: String,
    },
    /// Run standalone demo with toy model
    Demo {
        /// Demo scenario to run
        #[arg(short, long, default_value = "fibonacci")]
        scenario: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup tracing
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    // Create app state
    let app = App::new(match cli.command {
        Some(Commands::Live { url }) => Mode::Live { url },
        Some(Commands::Replay { receipt, key }) => Mode::Replay { receipt, key },
        Some(Commands::Demo { scenario }) => Mode::Demo { scenario },
        None => Mode::Demo { scenario: "fibonacci".to_string() },
    });
    
    // Run app
    let res = run_app(&mut terminal, app).await;
    
    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    
    if let Err(err) = res {
        eprintln!("Error: {:?}", err);
    }
    
    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> Result<()> {
    let mut last_tick = std::time::Instant::now();
    let tick_rate = std::time::Duration::from_millis(250);
    
    loop {
        // Draw UI
        terminal.draw(|f| views::draw(f, &app))?;
        
        // Handle events
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| std::time::Duration::from_secs(0));
        
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char(' ') => app.toggle_pause(),
                    KeyCode::Char('n') => app.next_step(),
                    KeyCode::Char('p') => app.prev_step(),
                    KeyCode::Char('v') => app.toggle_view(),
                    KeyCode::Char('a') => app.show_attestation(),
                    KeyCode::Up => app.scroll_up(),
                    KeyCode::Down => app.scroll_down(),
                    _ => {}
                }
            }
        }
        
        // Update on tick
        if last_tick.elapsed() >= tick_rate {
            app.update().await?;
            last_tick = std::time::Instant::now();
        }
    }
}
