use anyhow::{Context, Result};
use clap::Parser;
use std::fs;

use verilm_core::serialize;
use verilm_verify::{Verdict, verify_batch, verify_trace};

#[derive(Parser)]
#[command(name = "verilm-verify", about = "Verify token traces against a verifier key")]
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

    /// Verifier-generated challenge seed for batch challenge expansion (hex, 32 bytes).
    #[arg(long = "challenge-seed", alias = "seed")]
    challenge_seed: Option<String>,

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
        let challenge_seed_hex = args
            .challenge_seed
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--challenge-seed required for batch mode"))?;
        let challenge_seed_bytes =
            hex::decode(challenge_seed_hex).context("invalid hex challenge seed")?;
        if challenge_seed_bytes.len() != 32 {
            anyhow::bail!("challenge seed must be 32 bytes (64 hex chars)");
        }
        let mut challenge_seed = [0u8; 32];
        challenge_seed.copy_from_slice(&challenge_seed_bytes);

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

        let report = verify_batch(&key, &proof, challenge_seed, args.challenge_k);
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
