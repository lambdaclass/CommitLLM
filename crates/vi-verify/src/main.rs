use anyhow::{Context, Result};
use clap::Parser;
use std::fs;

use vi_core::serialize;
use vi_verify::{Verdict, verify_batch, verify_trace};

#[derive(Parser)]
#[command(name = "vi-verify", about = "Verify token traces against a verifier key")]
struct Args {
    /// Path to the verifier key file (.vkey)
    #[arg(long, short)]
    key: String,

    /// Path to a single token trace file (.vtrace)
    #[arg(long, short)]
    trace: Option<String>,

    /// Path to a batch proof file (.vbatch)
    #[arg(long, short)]
    batch: Option<String>,

    /// Verifier seed for challenge derivation (hex, 32 bytes). Required for --batch.
    #[arg(long)]
    seed: Option<String>,

    /// Number of tokens to challenge. Required for --batch.
    #[arg(long, default_value = "5")]
    challenge_k: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let key_data = fs::read(&args.key).context("reading key file")?;
    let key = serialize::deserialize_key(&key_data).map_err(|e| anyhow::anyhow!(e))?;

    if let Some(trace_path) = &args.trace {
        let trace_data = fs::read(trace_path).context("reading trace file")?;
        let trace = serialize::deserialize_trace(&trace_data).map_err(|e| anyhow::anyhow!(e))?;

        eprintln!(
            "model: {}, layers: {}, token: {}",
            key.config.name, key.config.n_layers, trace.token_index
        );

        let report = verify_trace(&key, &trace);
        println!("{}", report);
        if report.verdict == Verdict::Fail {
            std::process::exit(1);
        }
    } else if let Some(batch_path) = &args.batch {
        let seed_hex = args
            .seed
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--seed required for batch mode"))?;
        let seed_bytes = hex::decode(seed_hex).context("invalid hex seed")?;
        if seed_bytes.len() != 32 {
            anyhow::bail!("seed must be 32 bytes (64 hex chars)");
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_bytes);

        let batch_data = fs::read(batch_path).context("reading batch file")?;
        let proof =
            serialize::deserialize_batch(&batch_data).map_err(|e| anyhow::anyhow!(e))?;

        eprintln!(
            "model: {}, layers: {}, batch: {} tokens, {} opened",
            key.config.name,
            key.config.n_layers,
            proof.commitment.n_tokens,
            proof.traces.len()
        );

        let report = verify_batch(&key, &proof, seed, args.challenge_k);
        eprintln!("challenges (k={}): {:?}", report.challenges.len(), report.challenges);
        println!("{}", report);
        if report.verdict == Verdict::Fail {
            std::process::exit(1);
        }
    } else {
        anyhow::bail!("specify --trace for single token or --batch for batch verification");
    }

    Ok(())
}
