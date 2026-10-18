mod fair;
mod report;
mod stats;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::env;
use std::io::Read;
use std::sync::mpsc::{self, SyncSender};
use std::hint::black_box;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Cpu,
    Gpu,
    Hybrid,
    Probability,
    CollisionTest,
    CollisionStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CollisionSource {
    Cpu,
    Gpu,
    Hybrid,
}

impl CollisionSource {
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err("--source must be cpu, gpu, or hybrid".into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Producer {
    Cpu,
    Gpu,
}

impl Producer {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }
}

enum CollisionMsg {
    Batch(Producer, Vec<u128>),
    Error(String),
}

struct Options {
    mode: Mode,
    seconds: u64,
    threads: usize,
    blocks: u32,
    tpb: u32,
    iterations: u32,
    rate: Option<f64>,
    source: CollisionSource,
    samples: u64,
    batch: usize,
    entropy_bits: u8,
    seed: u64,
    timestamp_ms: u64,
    generator: String,
    output: Option<String>,
    volumes: Vec<usize>,
    trials: usize,
}

struct ResultData {
    count: u64,
    seconds: f64,
    checksum: u64,
    device: String,
}

fn usage() -> &'static str {
    "Usage: uuid-v7-bench <cpu|gpu|hybrid|probability|collision-test|collision-stats> [--seconds N] [--threads N] [--blocks N] [--tpb N] [--iterations N] [--rate N] [--seed N] [--timestamp-ms N] [--output PATH]\n\
     Example: uuid-v7-bench cpu --seconds 10 --threads 12\n\
     Example: uuid-v7-bench gpu --seconds 10\n\
     Example: uuid-v7-bench hybrid --seconds 10 --threads 6\n\
     Example: uuid-v7-bench probability --rate 145000000000 --seconds 1200\n\
     Example: uuid-v7-bench collision-test --source hybrid --samples 250000 --threads 6\n\
     Collision options: --source cpu|gpu|hybrid --samples N (per device) --batch N --entropy-bits 1..74\n\\
     Monte Carlo: collision-stats --volumes 1000,10000,100000 --trials 10 --entropy-bits 16\n\\
     Fair compute workload: --generator splitmix (default). CPU library: cpu --generator uuid"
}

fn parse() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mode = match args.next().as_deref() {
        Some("cpu") => Mode::Cpu,
        Some("gpu") => Mode::Gpu,
        Some("hybrid") => Mode::Hybrid,
        Some("probability") => Mode::Probability,
        Some("collision-test") => Mode::CollisionTest,
        Some("collision-stats") => Mode::CollisionStats,
        _ => return Err(usage().to_owned()),
    };
    let mut opt = Options {
        mode,
        seconds: 10,
        threads: thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        blocks: 256,
        tpb: 256,
        iterations: 256,
        rate: None,
        source: CollisionSource::Cpu,
        samples: 250_000,
        batch: 65_536,
        entropy_bits: 74,
        seed: fair::DEFAULT_SEED,
        timestamp_ms: SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_millis() as u64,
        generator: "splitmix".to_string(),
        output: None,
        volumes: vec![1_000, 10_000, 100_000],
        trials: 10,
    };
    while let Some(arg) = args.next() {
        let value = args.next().ok_or_else(|| format!("Missing value for {arg}"))?;
        match arg.as_str() {
            "--seconds" => opt.seconds = value.parse().map_err(|_| "Invalid --seconds")?,
            "--threads" => opt.threads = value.parse().map_err(|_| "Invalid --threads")?,
            "--blocks" => opt.blocks = value.parse().map_err(|_| "Invalid --blocks")?,
            "--tpb" => opt.tpb = value.parse().map_err(|_| "Invalid --tpb")?,
            "--iterations" => opt.iterations = value.parse().map_err(|_| "Invalid --iterations")?,
            "--rate" => opt.rate = Some(value.parse::<f64>().map_err(|_| "Invalid --rate")?),
            "--source" => opt.source = CollisionSource::from_str(&value)?,
            "--samples" => opt.samples = value.parse().map_err(|_| "Invalid --samples")?,
            "--batch" => opt.batch = value.parse().map_err(|_| "Invalid --batch")?,
            "--entropy-bits" => opt.entropy_bits = value.parse().map_err(|_| "Invalid --entropy-bits")?,
            "--seed" => opt.seed = value.parse().map_err(|_| "Invalid --seed")?,
            "--timestamp-ms" => opt.timestamp_ms = value.parse().map_err(|_| "Invalid --timestamp-ms")?,
            "--generator" => opt.generator = value,
            "--output" => opt.output = Some(value),
            "--volumes" => opt.volumes = value.split(',')
                .map(|s| s.parse::<usize>().map_err(|_| format!("Invalid volume: {s}")))
                .collect::<Result<Vec<_>, _>>()?,
            "--trials" => opt.trials = value.parse().map_err(|_| "Invalid --trials")?,
            _ => return Err(format!("Unknown option {arg}\n{}", usage())),
        }
    }
    if opt.seconds == 0 || opt.threads == 0 || opt.blocks == 0 || opt.tpb == 0 || opt.iterations == 0 || opt.tpb > 1024 {
        return Err("All numeric arguments must be > 0; --tpb must be <= 1024".into());
    }
    if matches!(opt.mode, Mode::Probability) && !matches!(opt.rate, Some(x) if x.is_finite() && x > 0.0) {
        return Err("probability requires a finite positive --rate (identifiers/second)".into());
    }
    if matches!(opt.mode, Mode::CollisionTest) {
        if opt.samples == 0 || opt.samples > 2_000_000 {
            return Err("--samples must be 1..=2000000 (per device, RAM safety cap)".into());
        }
        if opt.threads > 64 {
            return Err("--threads must be <= 64 in collision-test".into());
        }
        if opt.batch == 0 || opt.batch > 262_144 {
            return Err("--batch must be 1..=262144".into());
        }
        if !(1..=74).contains(&opt.entropy_bits) {
            return Err("--entropy-bits must be 1..=74".into());
        }
    }
    if opt.generator != "splitmix" && opt.generator != "uuid" {
        return Err("--generator must be splitmix or uuid".into());
    }
    if opt.generator == "uuid" && !matches!(opt.mode, Mode::Cpu) {
        return Err("--generator uuid is only supported in cpu mode".into());
    }
    if !(1..=74).contains(&opt.entropy_bits) {
        return Err("--entropy-bits must be 1..=74".into());
    }
    if matches!(opt.mode, Mode::CollisionStats) &&
        (opt.trials == 0 || opt.trials > 100 ||
         opt.volumes.is_empty() || opt.volumes.len() > 20 ||
         opt.volumes.iter().any(|&n| n == 0 || n > 2_000_000)) {
        return Err("--trials must be 1..100; --volumes must contain 1..20 values each 1..2000000".into());
    }
    if opt.output.as_deref().is_some_and(|p| p.is_empty() ||
       !(p.to_ascii_lowercase().ends_with(".json") || p.to_ascii_lowercase().ends_with(".csv"))) {
        return Err("--output must end in .json or .csv".into());
    }
    if opt.rate.is_some() && !matches!(opt.mode, Mode::Probability) {
        return Err("--rate is only valid in probability mode".into());
    }
    Ok(opt)
}

fn run_cpu(seconds: u64, threads: usize, seed: u64, timestamp_ms: u64, generator: &str) -> ResultData {
    let deadline = Duration::from_secs(seconds);
    let start = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let sequence = Arc::new(AtomicU64::new(0));
    let library = generator == "uuid";
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let stop = Arc::clone(&stop);
        let sequence = Arc::clone(&sequence);
        handles.push(thread::spawn(move || {
            let mut count = 0_u64;
            let mut checksum = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                // A real UUID v7 with the crate's timestamp/counter/RNG strategy.
                // Batch the stop check so it does not dominate throughput.
                let base = sequence.fetch_add(1024, Ordering::Relaxed);
                for index in base..base + 1024 {
                    let bits = if library {
                        black_box(Uuid::now_v7().as_u128())
                    } else {
                        black_box(fair::synthetic_uuid(seed, index, timestamp_ms))
                    };
                    checksum ^= (bits as u64) ^ ((bits >> 64) as u64);
                    count += 1;
                }
            }
            black_box(checksum);
            (count, checksum)
        }));
    }
    // Timer includes worker creation/join; both are included in the rate.
    while start.elapsed() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    stop.store(true, Ordering::Relaxed);
    let (mut count, mut checksum) = (0_u64, 0_u64);
    for h in handles {
        let (c, x) = h.join().expect("CPU thread panicked");
        count += c;
        checksum ^= x;
    }
    ResultData { count, seconds: start.elapsed().as_secs_f64(), checksum, device: "CPU".into() }
}

fn gpu_program() -> Result<PathBuf, String> {
    let exe = env::current_exe().map_err(|e| e.to_string())?;
    let name = if cfg!(windows) { "uuid-gpu.exe" } else { "uuid-gpu" };
    let sibling = exe.with_file_name(name);
    if sibling.is_file() { Ok(sibling) } else {
        Err(format!("GPU executable not found: {}. Run the GPU build step (README).", sibling.display()))
    }
}

fn run_gpu(opt: &Options) -> Result<ResultData, String> {
    let output = Command::new(gpu_program()?)
        .args([
            opt.seconds.to_string(),
            opt.blocks.to_string(),
            opt.tpb.to_string(),
            opt.iterations.to_string(),
            opt.seed.to_string(),
            opt.timestamp_ms.to_string(),
        ])
        .stdout(Stdio::piped())
        .output()
        .map_err(|e| format!("Could not launch CUDA executable: {e}"))?;
    if !output.status.success() {
        return Err(format!("CUDA failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let record = stdout.lines().find(|line| line.starts_with("GPU_RESULT|"))
        .ok_or_else(|| format!("Unexpected GPU output: {stdout}"))?;
    let fields: Vec<&str> = record.split('|').collect();
    if fields.len() != 5 { return Err(format!("Invalid GPU output: {record}")); }
    println!("GPU: {}", fields[4]);
    Ok(ResultData {
        count: fields[1].parse().map_err(|_| "Invalid GPU count")?,
        seconds: fields[2].parse().map_err(|_| "Invalid GPU elapsed time")?,
        checksum: fields[3].parse().map_err(|_| "Invalid GPU checksum")?,
        device: fields[4].to_string(),
    })
}

fn print_result(label: &str, r: &ResultData) {
    println!("{label}: {:.3} M UUID/s | total={} | {:.3} s | checksum={:016x}",
        r.count as f64 / r.seconds / 1e6, r.count, r.seconds, r.checksum);
}

// Poisson approximation to the birthday bound for independently uniformly random
// 74-bit suffixes, partitioned into 1-ms UUIDv7 timestamp buckets.
// This is NOT the empirical collision rate of any specific generator.
fn run_probability(rate: f64, seconds: u64) -> f64 {
    let buckets_per_second = 1000.0;
    let identifiers_per_bucket = rate / buckets_per_second;
    let possible_suffixes = 2_f64.powi(74);
    let expected_collisions_per_bucket =
        identifiers_per_bucket * (identifiers_per_bucket - 1.0).max(0.0)
        / (2.0 * possible_suffixes);
    let expected_collisions = expected_collisions_per_bucket * buckets_per_second * seconds as f64;
    let probability = -(-expected_collisions).exp_m1();
    let median_seconds = if expected_collisions_per_bucket > 0.0 {
        std::f64::consts::LN_2 / (expected_collisions_per_bucket * buckets_per_second)
    } else {
        f64::INFINITY
    };
    println!("MODEL: ideal independent uniform 74-bit random suffix, exact millisecond timestamps");
    println!("Rate: {rate:.0} identifiers/s | {identifiers_per_bucket:.3} identifiers/ms");
    println!("Window: {seconds} s | P(at least one collision) ~= {:.9}%", probability * 100.0);
    println!("Median waiting time for >=1 collision ~= {median_seconds:.3} s ({:.3} min)", median_seconds / 60.0);
    println!("WARNING: theoretical model only; not the empirical rate of SplitMix64 or uuid::now_v7().");
    probability
}


/// Keep the 48-bit timestamp, v7 and RFC variant fixed and optionally cut
/// the random portion to fewer bits. This is a DELIBERATE entropy degradation
/// for demonstration; only entropy_bits=74 tests original identifiers.
fn reduce_entropy(id: u128, entropy_bits: u8) -> u128 {
    if entropy_bits == 74 {
        return id;
    }
    let mut hi = (id >> 64) as u64;
    let mut lo = id as u64;
    let payload_mask = (1u64 << 62) - 1;
    if entropy_bits <= 62 {
        hi &= !0xFFF;
        let mask = (1u64 << entropy_bits) - 1;
        lo = (lo & !payload_mask) | (lo & mask);
    } else {
        let hi_bits = entropy_bits - 62;
        let mask = (1u64 << hi_bits) - 1;
        hi = (hi & !0xFFF) | (hi & mask);
    }
    ((hi as u128) << 64) | u128::from(lo)
}

fn is_uuid_v7(id: u128) -> bool {
    let hi = (id >> 64) as u64;
    let lo = id as u64;
    ((hi >> 12) & 0xF) == 7 && (lo >> 62) == 0b10
}

fn cpu_collision_worker(
    samples: u64,
    batch: usize,
    seed: u64,
    timestamp_ms: u64,
    cursor: Arc<AtomicU64>,
    tx: SyncSender<CollisionMsg>,
) {
    loop {
        let start = cursor.fetch_add(batch as u64, Ordering::Relaxed);
        if start >= samples {
            break;
        }
        let amount = std::cmp::min(batch as u64, samples - start) as usize;
        let mut ids = Vec::with_capacity(amount);
        for i in start..start + amount as u64 {
            ids.push(fair::synthetic_uuid(seed, i, timestamp_ms));
        }
        if tx.send(CollisionMsg::Batch(Producer::Cpu, ids)).is_err() {
            break;
        }
    }
}

fn gpu_collision_worker(opt_samples: u64, opt_batch: usize, blocks: u32, tpb: u32,
    seed: u64, timestamp_ms: u64, tx: SyncSender<CollisionMsg>) {
    let run = || -> Result<(), String> {
        let mut child = Command::new(gpu_program()?)
            .arg("stream")
            .arg(opt_samples.to_string())
            .arg(blocks.to_string())
            .arg(tpb.to_string())
            .arg(opt_batch.to_string())
            .arg(seed.to_string())
            .arg(timestamp_ms.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Cannot start CUDA stream: {e}"))?;
        let mut output = child.stdout.take().ok_or("CUDA stdout unavailable")?;
        let mut remaining = opt_samples;
        let mut error = None;
        while remaining > 0 {
            let amount = std::cmp::min(remaining, opt_batch as u64) as usize;
            let mut bytes = vec![0u8; amount * 16];
            if let Err(e) = output.read_exact(&mut bytes) {
                error = Some(format!("GPU binary stream ended early: {e}"));
                break;
            }
            let mut ids = Vec::with_capacity(amount);
            for raw in bytes.chunks_exact(16) {
                let hi = u64::from_le_bytes(raw[0..8].try_into().expect("8 bytes"));
                let lo = u64::from_le_bytes(raw[8..16].try_into().expect("8 bytes"));
                ids.push((u128::from(hi) << 64) | u128::from(lo));
            }
            if tx.send(CollisionMsg::Batch(Producer::Gpu, ids)).is_err() {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            remaining -= amount as u64;
        }
        // Drop pipe before waiting so CUDA cannot block on the OS pipe.
        drop(output);
        let status = child.wait().map_err(|e| format!("CUDA wait failed: {e}"))?;
        if let Some(e) = error {
            return Err(e);
        }
        if !status.success() {
            return Err(format!("CUDA stream exited with {status}"));
        }
        Ok(())
    };
    if let Err(e) = run() {
        let _ = tx.send(CollisionMsg::Error(e));
    }
}

fn run_collision_test(opt: &Options) -> Result<report::Record, String> {
    let cpu_enabled = opt.source != CollisionSource::Gpu;
    let gpu_enabled = opt.source != CollisionSource::Cpu;
    if gpu_enabled {
        gpu_program()?; // Fail fast if CUDA binary is missing.
    }
    let sources = u64::from(cpu_enabled) + u64::from(gpu_enabled);
    let total_expected = opt.samples * sources;
    println!("COLLISION TEST (synthetic SplitMix64, same algorithm on CPU/GPU): {:?}, {} IDs per device, {} total, batch={}, threads={}",
        opt.source, opt.samples, total_expected, opt.batch, opt.threads);
    if opt.entropy_bits == 74 {
        println!("ENTROPY: unchanged UUIDs (74 variable bits in RFC layout; CPU library may use counters)");
    } else {
        println!("SIMULATION: entropy artificially reduced to {} bits (not the original IDs!)", opt.entropy_bits);
    }
    println!("IMPORTANT: exact duplicate detection across ALL batches in a bounded HashMap.");
    println!("RAM grows with distinct IDs (up to {} total). This is not a 145-billion/s benchmark.", total_expected);

    let start = Instant::now();
    let (tx, rx) = mpsc::sync_channel::<CollisionMsg>((opt.threads + 2).min(64));
    let mut handles = Vec::new();
    if cpu_enabled {
        let cursor = Arc::new(AtomicU64::new(0));
        for _ in 0..opt.threads {
            let cursor = Arc::clone(&cursor);
            let tx = tx.clone();
            let (samples, batch, seed, ms) = (opt.samples, opt.batch, opt.seed, opt.timestamp_ms);
            handles.push(thread::spawn(move || cpu_collision_worker(samples, batch, seed, ms, cursor, tx)));
        }
    }
    if gpu_enabled {
        let tx = tx.clone();
        let (samples, batch, blocks, tpb, seed, ms) =
            (opt.samples, opt.batch, opt.blocks, opt.tpb, opt.seed ^ 0xA5A5_A5A5_A5A5_A5A5, opt.timestamp_ms);
        handles.push(thread::spawn(move || gpu_collision_worker(samples, batch, blocks, tpb, seed, ms, tx)));
    }
    drop(tx);

    // Stores every distinct UUID seen across ALL CPU and GPU batches so that
    // cross-batch and cross-device duplicate detection is exact in this sample.
    let mut first_seen = HashMap::<u128, Producer>::with_capacity(total_expected as usize);
    let (mut generated_cpu, mut generated_gpu) = (0u64, 0u64);
    let (mut duplicates, mut cross_source, mut invalid) = (0u64, 0u64, 0u64);
    let mut first_five = Vec::<(Producer, Producer, u128)>::new();
    let mut error = None;
    for message in rx {
        match message {
            CollisionMsg::Error(e) => {
                if error.is_none() { error = Some(e); }
            }
            CollisionMsg::Batch(source, ids) => {
                if source == Producer::Cpu { generated_cpu += ids.len() as u64; }
                else { generated_gpu += ids.len() as u64; }
                for original in ids {
                    if !is_uuid_v7(original) { invalid += 1; }
                    let id = reduce_entropy(original, opt.entropy_bits);
                    if let Some(&first) = first_seen.get(&id) {
                        duplicates += 1;
                        if first != source { cross_source += 1; }
                        if first_five.len() < 5 { first_five.push((first, source, id)); }
                    } else {
                        first_seen.insert(id, source);
                    }
                }
            }
        }
    }
    for handle in handles {
        if handle.join().is_err() && error.is_none() {
            error = Some("Producer thread panicked".to_owned());
        }
    }
    if let Some(e) = error { return Err(e); }
    let count = generated_cpu + generated_gpu;
    if count != total_expected {
        return Err(format!("Expected {total_expected} generated IDs, got {count}"));
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("CPU: {generated_cpu} | GPU: {generated_gpu} | total: {count}");
    println!("Unique: {} | duplicates (repeated occurrences): {} | cross-device repeats: {}",
        first_seen.len(), duplicates, cross_source);
    println!("Invalid version/variant UUIDs: {invalid}");
    println!("End-to-end speed (generation + transfer + HashMap): {:.3} M UUID/s ({elapsed:.3}s)",
        count as f64 / elapsed / 1e6);
    for (first, later, id) in first_five {
        println!("DUPLICATE: first={} again={} UUID={}", first.name(), later.name(), Uuid::from_u128(id));
    }
    if duplicates == 0 {
        println!("No collisions OBSERVED in this finite sample. This is NOT a mathematical guarantee.");
    }
    if opt.entropy_bits < 74 {
        println!("SIMULATION ONLY: collisions reported above are after intentionally discarding entropy bits.");
    } else if gpu_enabled {
        println!("GPU: deterministic SplitMix64 v7-shaped IDs, NOT cryptographically secure.");
    }
    Ok(report::Record {
        mode: "collision-test".into(),
        device: format!("{:?}", opt.source),
        generator: "splitmix".into(),
        count,
        elapsed_seconds: elapsed,
        collisions: duplicates as usize,
        volume: total_expected as usize,
        ..make_record_base(opt)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduction_preserves_rfc_fields() {
        let id = Uuid::now_v7().as_u128();
        for bits in [1, 16, 62, 63, 73, 74] {
            let reduced = reduce_entropy(id, bits);
            assert!(is_uuid_v7(reduced));
            assert_eq!((reduced >> 80) & 0xffff_ffff_ffff, (id >> 80) & 0xffff_ffff_ffff);
            assert_eq!(reduce_entropy(reduced, bits), reduced);
        }
        assert_eq!(reduce_entropy(id, 74), id);
    }
    #[test]
    fn collision_demo_works_on_same_timestamp() {
        let x = Uuid::now_v7().as_u128();
        let y = x ^ (1u128 << 40); // change a random bit, not timestamp/version/variant
        assert_ne!(x, y);
        assert_eq!(reduce_entropy(x, 16), reduce_entropy(y, 16));
    }
}

fn main() {
    let opt = match parse() {
        Ok(o) => o,
        Err(e) => { eprintln!("{e}"); std::process::exit(2); }
    };
    println!("Mode={:?}, duration={}s, CPU threads={}, CUDA blocks={} x threads={} x iterations={}",
        opt.mode, opt.seconds, opt.threads, opt.blocks, opt.tpb, opt.iterations);
    let result: Result<(), String> = match opt.mode {
        Mode::Cpu => {
            let r = run_cpu(opt.seconds, opt.threads);
            print_result("CPU", &r);
            Ok(())
        }
        Mode::Gpu => run_gpu(&opt).map(|r| {
            print_result("GPU (synthetic v7-shaped)", &r);
            eprintln!("NOTE: CUDA Philox is not a cryptographic generator; no collision test was performed.");
        }),
        Mode::CollisionTest => run_collision_test(&opt),
        Mode::Probability => {
            run_probability(opt.rate.expect("validated --rate"), opt.seconds);
            Ok(())
        }
        Mode::Hybrid => {
            // Both measurements run concurrently, not sequentially.
            let started = Instant::now();
            let seconds = opt.seconds;
            let threads = opt.threads;
            let cpu = thread::spawn(move || run_cpu(seconds, threads));
            let gpu = run_gpu(&opt);
            let cpu = cpu.join().expect("CPU worker panicked");
            print_result("CPU (hybrid)", &cpu);
            match gpu {
                Ok(g) => {
                    print_result("GPU (hybrid, synthetic v7-shaped)", &g);
                    let wall = started.elapsed().as_secs_f64();
                    print_result("COMBINED", &ResultData {
                        count: cpu.count + g.count,
                        seconds: wall,
                        checksum: cpu.checksum ^ g.checksum,
                    });
                    eprintln!("NOTE: CPU and GPU measure different generators. No collision test was performed.");
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }
    };
    if let Err(e) = result {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
}
