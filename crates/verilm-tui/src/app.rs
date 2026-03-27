//! Application state management

use anyhow::Result;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub enum Mode {
    Live { url: String },
    Replay { receipt: String, key: String },
    Demo { scenario: String },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View {
    Overview,      // High-level pipeline view
    Shell,         // 7 matrix verification details
    Attention,     // Attention replay visualization
    FinalToken,    // Final boundary and sampling
    Log,           // Event log
}

#[derive(Debug, Clone)]
pub struct VerificationEvent {
    pub timestamp: chrono::DateTime<chrono::Local>,
    pub layer: usize,
    pub token: usize,
    pub stage: VerificationStage,
    pub status: VerificationStatus,
    pub details: String,
}

#[derive(Debug, Clone)]
pub enum VerificationStage {
    Embedding,
    ShellMatrix { mat_type: ShellMatrix },
    Bridge,
    Attention,
    FinalNorm,
    LmHead,
    Decode,
}

#[derive(Debug, Clone)]
pub enum ShellMatrix {
    Wq, Wk, Wv, Wo, Wg, Wu, Wd,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationStatus {
    Pending,
    InProgress,
    Success,
    Warning,  // For approximate stages within tolerance
    Failed,
    Skipped,
}

pub struct App {
    pub mode: Mode,
    pub current_view: View,
    pub paused: bool,
    pub current_token: usize,
    pub current_layer: usize,
    pub total_tokens: usize,
    pub total_layers: usize,
    pub events: VecDeque<VerificationEvent>,
    pub selected_event: usize,
    pub shell_status: [VerificationStatus; 7],  // One per matrix
    pub attention_match_pct: f64,
    pub final_token_verified: bool,
}

impl App {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            current_view: View::Overview,
            paused: false,
            current_token: 0,
            current_layer: 0,
            total_tokens: 128,  // Example
            total_layers: 80,   // Example
            events: VecDeque::with_capacity(1000),
            selected_event: 0,
            shell_status: [VerificationStatus::Pending; 7],
            attention_match_pct: 0.0,
            final_token_verified: false,
        }
    }
    
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
    
    pub fn next_step(&mut self) {
        if self.current_layer < self.total_layers - 1 {
            self.current_layer += 1;
        } else if self.current_token < self.total_tokens - 1 {
            self.current_token += 1;
            self.current_layer = 0;
        }
    }
    
    pub fn prev_step(&mut self) {
        if self.current_layer > 0 {
            self.current_layer -= 1;
        } else if self.current_token > 0 {
            self.current_token -= 1;
            self.current_layer = self.total_layers - 1;
        }
    }
    
    pub fn toggle_view(&mut self) {
        self.current_view = match self.current_view {
            View::Overview => View::Shell,
            View::Shell => View::Attention,
            View::Attention => View::FinalToken,
            View::FinalToken => View::Log,
            View::Log => View::Overview,
        };
    }
    
    pub fn show_attestation(&mut self) {
        // Generate attestation report
    }
    
    pub fn scroll_up(&mut self) {
        if self.selected_event > 0 {
            self.selected_event -= 1;
        }
    }
    
    pub fn scroll_down(&mut self) {
        if self.selected_event < self.events.len().saturating_sub(1) {
            self.selected_event += 1;
        }
    }
    
    pub async fn update(&mut self) -> Result<()> {
        if self.paused {
            return Ok(());
        }
        
        // In live/replay modes, fetch new events
        // In demo mode, simulate progress
        match &self.mode {
            Mode::Demo { .. } => self.simulate_demo_step(),
            _ => {}
        }
        
        Ok(())
    }
    
    fn simulate_demo_step(&mut self) {
        // Simulate verification progress for demo
        let mat_idx = self.current_layer % 7;
        self.shell_status[mat_idx] = VerificationStatus::InProgress;
        
        // Add event
        self.events.push_back(VerificationEvent {
            timestamp: chrono::Local::now(),
            layer: self.current_layer,
            token: self.current_token,
            stage: VerificationStage::ShellMatrix { 
                mat_type: match mat_idx {
                    0 => ShellMatrix::Wq,
                    1 => ShellMatrix::Wk,
                    2 => ShellMatrix::Wv,
                    3 => ShellMatrix::Wo,
                    4 => ShellMatrix::Wg,
                    5 => ShellMatrix::Wu,
                    6 => ShellMatrix::Wd,
                    _ => ShellMatrix::Wq,
                }
            },
            status: VerificationStatus::Success,
            details: format!("Freivalds check passed for layer {}", self.current_layer),
        });
        
        // Mark complete
        self.shell_status[mat_idx] = VerificationStatus::Success;
    }
}
