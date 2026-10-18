use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default)]
pub struct Record {
    pub mode: String,
    pub device: String,
    pub generator: String,
    pub count: u64,
    pub elapsed_seconds: f64,
    pub checksum: u64,
    pub seed: u64,
    pub timestamp_ms: u64,
    pub threads: usize,
    pub volume: usize,
    pub trials: usize,
    pub collisions: usize,
    pub probability: f64,
    pub gpu_name: String,
}

fn csv_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}
fn json_quote(value: &str) -> String {
    let mut result = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}
fn cpu_model() -> String {
    if cfg!(target_os = "linux") {
        if let Ok(data) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(line) = data.lines().find(|line| line.starts_with("model name")) {
                if let Some((_, model)) = line.split_once(':') { return model.trim().to_owned(); }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into())
}

/// Exports one or more benchmark records, with machine metadata.
/// JSON is an array; CSV is a flat table with the same fields.
pub fn export(path: &str, records: &[Record]) -> Result<(), String> {
    let is_csv = Path::new(path).extension().and_then(|x| x.to_str())
        .is_some_and(|x| x.eq_ignore_ascii_case("csv"));
    let cpu = cpu_model();
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let mut output = String::new();
    if is_csv {
        output.push_str("mode,device,generator,count,elapsed_seconds,rate_per_second,checksum,seed,timestamp_ms,threads,volume,trials,collisions,probability,gpu,os,arch,cpu,available_parallelism,created_at_unix\n");
        for r in records {
            let rate = if r.elapsed_seconds > 0.0 { r.count as f64 / r.elapsed_seconds } else { 0.0 };
            output.push_str(&format!("{},{},{},{},{:.9},{:.3},{},{},{},{},{},{},{},{:.12},{},{},{},{},{},{}\n",
                csv_quote(&r.mode), csv_quote(&r.device), csv_quote(&r.generator),
                r.count, r.elapsed_seconds, rate, r.checksum, r.seed, r.timestamp_ms,
                r.threads, r.volume, r.trials, r.collisions, r.probability,
                csv_quote(&r.gpu_name), csv_quote(os), csv_quote(arch), csv_quote(&cpu),
                available, created_at));
        }
    } else {
        output.push_str("[\n");
        for (i, r) in records.iter().enumerate() {
            let rate = if r.elapsed_seconds > 0.0 { r.count as f64 / r.elapsed_seconds } else { 0.0 };
            if i != 0 { output.push_str(",\n"); }
            output.push_str(&format!(
                "  {{\"mode\":{},\"device\":{},\"generator\":{},\"count\":{},\"elapsed_seconds\":{:.9},\"rate_per_second\":{:.3},\"checksum\":{},\"seed\":{},\"timestamp_ms\":{},\"threads\":{},\"volume\":{},\"trials\":{},\"collisions\":{},\"probability\":{:.12},\"gpu\":{},\"os\":{},\"arch\":{},\"cpu\":{},\"available_parallelism\":{},\"created_at_unix\":{}}}",
                json_quote(&r.mode), json_quote(&r.device), json_quote(&r.generator),
                r.count, r.elapsed_seconds, rate, r.checksum, r.seed, r.timestamp_ms,
                r.threads, r.volume, r.trials, r.collisions, r.probability,
                json_quote(&r.gpu_name), json_quote(os), json_quote(arch),
                json_quote(&cpu), available, created_at
            ));
        }
        output.push_str("\n]\n");
    }
    fs::write(path, output).map_err(|e| format!("Could not write {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escaped_values() {
        assert_eq!(csv_quote("a\"b"), "\"a\"\"b\"");
        assert_eq!(json_quote("a\nb"), "\"a\\nb\"");
    }
}