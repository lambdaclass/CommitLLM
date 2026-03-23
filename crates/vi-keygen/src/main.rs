use anyhow::Result;
use clap::Parser;

use vi_core::serialize;

#[derive(Parser)]
#[command(
    name = "vi-keygen",
    about = "Generate a verifier-secret key from model weights",
    long_about = "Generate a verifier-secret key from model weights.\n\n\
        SECURITY: The output .vkey file contains secret random vectors (r_j)\n\
        and precomputed values (v_j = r^T W) that must NEVER be shared with\n\
        the prover. If the prover learns r (or v, since it knows W), it can\n\
        forge traces that pass Freivalds checks without running correct inference.\n\n\
        The verifier generates this key from its own copy of the public model\n\
        weights. The prover needs no key material — it just runs inference\n\
        and records intermediates."
)]
struct Args {
    /// Path to directory containing safetensors model shards
    #[arg(long)]
    model_dir: String,

    /// Output path for the verifier key file (VERIFIER-SECRET)
    #[arg(long, short)]
    output: String,

    /// Hex seed for deterministic r_j generation (32 bytes).
    /// If omitted, a random seed is used.
    #[arg(long)]
    seed: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let seed: [u8; 32] = if let Some(hex) = &args.seed {
        let bytes = hex::decode(hex)?;
        if bytes.len() != 32 {
            anyhow::bail!("seed must be exactly 32 bytes (64 hex chars)");
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        arr
    } else {
        let mut arr = [0u8; 32];
        getrandom::getrandom(&mut arr)?;
        arr
    };

    eprintln!("vi-keygen: generating verifier-secret key from {}", args.model_dir);
    let key = vi_keygen::generate_key(std::path::Path::new(&args.model_dir), seed)?;

    let data = serialize::serialize_key(&key);
    std::fs::write(&args.output, &data)?;
    eprintln!(
        "wrote {} ({:.1} MB) — VERIFIER-SECRET, do not share with prover",
        args.output,
        data.len() as f64 / 1_048_576.0
    );

    Ok(())
}
