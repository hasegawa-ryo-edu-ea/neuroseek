//! A deliberately read-only dashboard for durable NEUROSEEK JSONL events.
//!
//! It never opens a control socket or sends a signal to the trainer.  The
//! display can therefore be detached at any time without changing training.
//! The implementation is std-only on purpose: a dashboard must still attach
//! on a minimal aarch64 recovery environment.
use std::{
    collections::BTreeMap,
    env,
    fs::File,
    io::{self, IsTerminal, Read, Seek, SeekFrom, Write},
    path::Path,
    thread,
    time::Duration,
};

const MISSING: &str = "n/a";
const TAIL_BYTES: u64 = 64 * 1024;

#[derive(Debug, Default, Clone)]
struct Event {
    fields: BTreeMap<String, String>,
}

impl Event {
    fn get(&self, key: &str) -> &str {
        self.fields.get(key).map(String::as_str).unwrap_or(MISSING)
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.fields.get(key)?.parse().ok()
    }
}

/// Split a JSON object body without being confused by quoted commas or nested
/// telemetry objects. This is deliberately a small *viewer* parser: compound
/// values are skipped below, while the durable scalar fields that drive the
/// dashboard remain visible.
fn top_level_items(body: &str) -> Option<Vec<&str>> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut escaped = false;
    let mut nested = 0_i32;
    for (index, character) in body.char_indices() {
        if quoted {
            if escaped { escaped = false; }
            else if character == '\\' { escaped = true; }
            else if character == '"' { quoted = false; }
            continue;
        }
        match character {
            '"' => quoted = true,
            '{' | '[' => nested += 1,
            '}' | ']' => { nested -= 1; if nested < 0 { return None; } }
            ',' if nested == 0 => { result.push(&body[start..index]); start = index + 1; }
            _ => {}
        }
    }
    if quoted || nested != 0 { return None; }
    result.push(&body[start..]);
    Some(result)
}

/// A malformed/incomplete last write is ignored until its next durable event.
/// Nested telemetry maps are deliberately retained outside `Event`; rejecting
/// the whole line would hide otherwise real training metrics.
fn parse_event(line: &str) -> Option<Event> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    let body = &trimmed[1..trimmed.len() - 1];
    let mut fields = BTreeMap::new();
    for item in top_level_items(body)? {
        let (raw_key, raw_value) = item.split_once(':')?;
        let key = raw_key.trim().strip_prefix('"')?.strip_suffix('"')?;
        let value = raw_value.trim();
        if value.starts_with(['{', '[']) {
            continue;
        }
        let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value);
        fields.insert(key.to_owned(), value.to_owned());
    }
    Some(Event { fields })
}

fn current_metrics_path() -> io::Result<String> {
    let content = std::fs::read_to_string("runs/current.json")?;
    // `current.json` is owned by the launcher and has one scalar path field.
    // Parse exactly that field rather than assuming a fragile symlink exists.
    let marker = "\"path\"";
    let start = content.find(marker).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run has no path"))? + marker.len();
    let after_colon = content[start..].split_once(':').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run path is malformed"))?.1.trim_start();
    let value = after_colon.strip_prefix('"').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run path is not a string"))?;
    let end = value.find('"').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run path is unterminated"))?;
    Ok(format!("{}/metrics.jsonl", &value[..end]))
}

/// Reads only a bounded tail, which matters for a long-running 50-hour run.
fn tail_events(path: &Path) -> io::Result<Vec<Event>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let start = len.saturating_sub(TAIL_BYTES);
    file.seek(SeekFrom::Start(start))?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    let mut lines = text.lines();
    if start > 0 {
        // The first line can start in the middle of a JSON record.
        lines.next();
    }
    Ok(lines.filter_map(parse_event).collect())
}

fn latest<'a>(events: &'a [Event], category: &str) -> Option<&'a Event> {
    events.iter().rev().find(|event| event.get("category") == category)
}

fn latest_display_event(events: &[Event]) -> Option<&Event> {
    latest(events, "TrainingEvent").or_else(|| events.last())
}

fn terminal_dimension(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|value| value.parse().ok()).filter(|value: &usize| *value > 0).unwrap_or(default)
}

fn paint(text: &str, code: &str, color: bool) -> String {
    if color { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_owned() }
}

fn sparkline(events: &[Event], field: &str, invert: bool) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let values: Vec<f64> = events.iter()
        .filter(|event| event.get("category") == "TrainingEvent")
        .filter_map(|event| event.number(field))
        .filter(|value| value.is_finite())
        .rev().take(32).collect::<Vec<_>>().into_iter().rev().collect();
    if values.len() < 2 { return MISSING.to_owned(); }
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (high - low).abs() < f64::EPSILON { return "▅".repeat(values.len()); }
    values.into_iter().map(|value| {
        let normalized = if invert { (high - value) / (high - low) } else { (value - low) / (high - low) };
        GLYPHS[(normalized * 7.0).round().clamp(0.0, 7.0) as usize]
    }).collect()
}

fn line(label: &str, value: &str, width: usize) {
    let available = width.saturating_sub(label.len() + 3);
    let visible: String = value.chars().take(available.max(1)).collect();
    println!("{label:<18} {visible}");
}

fn draw(events: &[Event], color: bool, tty: bool) {
    let width = terminal_dimension("COLUMNS", 100);
    let height = terminal_dimension("LINES", 28);
    let compact = width < 80 || height < 20;
    if tty { print!("\x1b[2J\x1b[H"); }
    let Some(event) = latest_display_event(events) else {
        println!("NEUROSEEK // waiting for durable metrics event");
        return;
    };
    let title = paint("NEUROSEEK // LEARNED GPU KNOWLEDGE PROCESSOR", "1;36", color);
    println!("{title}");
    println!("read-only dashboard · trainer remains detached · source: metrics.jsonl");
    println!("{}", "─".repeat(width.min(100)));
    line("PHASE", event.get("phase"), width);
    line("GLOBAL STEP", event.get("global_step"), width);
    line("ELAPSED", event.get("elapsed_seconds"), width);
    line("LAST EVENT", event.get("category"), width);
    if compact {
        line("REWARD", event.get("reward"), width);
        line("SUCCESS", event.get("success_rate"), width);
        line("PROOF VALID", event.get("proof_validity"), width);
        line("NODES / QUERY", event.get("nodes_per_query"), width);
        line("EDGES / QUERY", event.get("edges_per_query"), width);
        line("TEMPERATURE C", event.get("temperature_c"), width);
        println!("compact mode · Ctrl-C detaches display only");
    } else {
        println!("\nSEARCH INTELLIGENCE");
        line("REWARD", event.get("reward"), width);
        line("SUCCESS", event.get("success_rate"), width);
        line("PROOF VALIDITY", event.get("proof_validity"), width);
        line("NODES / QUERY", event.get("nodes_per_query"), width);
        line("EDGES / QUERY", event.get("edges_per_query"), width);
        line("CREDITS / QUERY", event.get("credits_per_query"), width);
        println!("REWARD TREND       {}", sparkline(events, "reward", false));
        println!("SEARCH COST TREND  {}", sparkline(events, "nodes_per_query", true));
        println!("\nPOLICY");
        line("POLICY LOSS", event.get("policy_loss"), width);
        line("VALUE LOSS", event.get("value_loss"), width);
        line("ENTROPY", event.get("entropy"), width);
        line("KL", event.get("kl"), width);
        line("GRADIENT NORM", event.get("gradient_norm"), width);
        line("OPERATOR SAMPLE", event.get("operator_sample"), width);
        println!("\nLIVE SEARCH (durable SearchTraceEvent)");
        let trace = latest(events, "SearchTraceEvent").unwrap_or(event);
        line("TASK", trace.get("task_id"), width);
        line("FAMILY", trace.get("family"), width);
        line("RESULT", trace.get("result"), width);
        line("PROGRAM", trace.get("trace"), width);
        println!("\nOPERATOR DISTRIBUTION (current rollout batch)");
        line("OPERATORS", event.get("operator_distribution"), width);
        println!("\nJETSON (only fields emitted by telemetry)");
        line("CUDA ACTIVE", event.get("cuda"), width);
        line("RAM USED GiB", event.get("ram_used_gib"), width);
        line("RAM TOTAL GiB", event.get("ram_total_gib"), width);
        line("TEMPERATURE C", event.get("temperature_c"), width);
        if let Some(checkpoint) = latest(events, "CheckpointEvent") {
            println!("\nCHECKPOINT");
            line("LAST SAVED", checkpoint.get("checkpoint"), width);
            line("SAVE REASON", checkpoint.get("reason"), width);
        }
        println!("\nCtrl-C detaches display only; training continues in its detached service.");
    }
    let _ = io::stdout().flush();
}

fn usage() -> ! {
    eprintln!("usage: neuroseek-tui <metrics.jsonl> [--once] [--no-color]\n       neuroseek-tui --attach-current [--once] [--no-color]");
    std::process::exit(2)
}

fn main() {
    let mut args = env::args().skip(1);
    let first = args.next().unwrap_or_else(|| usage());
    let path = if first == "--attach-current" {
        current_metrics_path().unwrap_or_else(|error| { eprintln!("cannot attach current run: {error}"); std::process::exit(1) })
    } else if first.starts_with('-') { usage() } else { first };
    let flags: Vec<String> = args.collect();
    if flags.iter().any(|flag| flag != "--once" && flag != "--no-color") { usage(); }
    let once = flags.iter().any(|flag| flag == "--once");
    let tty = io::stdout().is_terminal();
    let color = tty && env::var_os("NO_COLOR").is_none() && !flags.iter().any(|flag| flag == "--no-color");
    loop {
        match tail_events(Path::new(&path)) {
            Ok(events) if !events.is_empty() => draw(&events, color, tty),
            Ok(_) => eprintln!("waiting for complete durable metrics event: {path}"),
            Err(error) => eprintln!("waiting for {path}: {error}"),
        }
        if once || !tty { break; }
        thread::sleep(Duration::from_secs(1));
    }
}

#[cfg(test)]
mod tests {
    use super::{current_metrics_path, parse_event, sparkline, tail_events};
    use std::{fs, time::{SystemTime, UNIX_EPOCH}};

    #[test]
    fn parses_flat_trainer_events_and_rejects_partial_json() {
        let event = parse_event(r#"{"category":"TrainingEvent","global_step":12,"reward":3.5,"cuda":true}"#).unwrap();
        assert_eq!(event.get("category"), "TrainingEvent");
        assert_eq!(event.get("global_step"), "12");
        assert_eq!(event.number("reward"), Some(3.5));
        assert!(parse_event(r#"{"category":"TrainingEvent""#).is_none());
        let nested = parse_event(r#"{"category":"TrainingEvent","reward":2,"thermal_sensors_c":{"gpu":50.0}}"#).unwrap();
        assert_eq!(nested.number("reward"), Some(2.0));
    }

    #[test]
    fn bounded_tail_ignores_split_first_record_and_keeps_latest() {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("neuroseek-tui-{unique}.jsonl"));
        let padding = "x".repeat(70 * 1024);
        fs::write(&path, format!("{{\"category\":\"Ignored\",\"padding\":\"{padding}\"}}\n{{\"category\":\"TrainingEvent\",\"reward\":1.0}}\n")).unwrap();
        let events = tail_events(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].get("category"), "TrainingEvent");
    }

    #[test]
    fn trends_are_based_on_real_training_events_only() {
        let events = [
            parse_event(r#"{"category":"WarningEvent","reward":999}"#).unwrap(),
            parse_event(r#"{"category":"TrainingEvent","reward":1}"#).unwrap(),
            parse_event(r#"{"category":"TrainingEvent","reward":2}"#).unwrap(),
        ];
        assert_eq!(sparkline(&events, "reward", false).chars().count(), 2);
    }

    #[test]
    fn attach_current_uses_launcher_metadata() {
        let original = std::fs::read_to_string("runs/current.json").ok();
        std::fs::create_dir_all("runs").unwrap();
        std::fs::write("runs/current.json", r#"{"path":"runs/example-run"}"#).unwrap();
        assert_eq!(current_metrics_path().unwrap(), "runs/example-run/metrics.jsonl");
        if let Some(content) = original { std::fs::write("runs/current.json", content).unwrap(); }
        else { std::fs::remove_file("runs/current.json").unwrap(); }
    }
}
