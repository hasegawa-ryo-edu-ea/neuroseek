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
    process::Command,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

const MISSING: &str = "n/a";
const TAIL_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    English,
    Japanese,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Explore,
    Trace,
    System,
    Model,
}

impl Screen {
    fn from_key(key: u8) -> Option<Self> {
        match key {
            b'1' => Some(Self::Explore),
            b'2' => Some(Self::Trace),
            b'3' => Some(Self::System),
            b'4' => Some(Self::Model),
            _ => None,
        }
    }
}

impl Language {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "en" | "english" => Some(Self::English),
            "ja" | "jp" | "japanese" | "日本語" => Some(Self::Japanese),
            _ => None,
        }
    }
}

fn t(language: Language, english: &'static str, japanese: &'static str) -> &'static str {
    match language {
        Language::English => english,
        Language::Japanese => japanese,
    }
}

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
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        match character {
            '"' => quoted = true,
            '{' | '[' => nested += 1,
            '}' | ']' => {
                nested -= 1;
                if nested < 0 {
                    return None;
                }
            }
            ',' if nested == 0 => {
                result.push(&body[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if quoted || nested != 0 {
        return None;
    }
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
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(value);
        fields.insert(key.to_owned(), value.to_owned());
    }
    Some(Event { fields })
}

fn current_metrics_path() -> io::Result<String> {
    let content = std::fs::read_to_string("runs/current.json")?;
    // `current.json` is owned by the launcher and has one scalar path field.
    // Parse exactly that field rather than assuming a fragile symlink exists.
    let marker = "\"path\"";
    let start = content
        .find(marker)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run has no path"))?
        + marker.len();
    let after_colon = content[start..]
        .split_once(':')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "current run path is malformed"))?
        .1
        .trim_start();
    let value = after_colon.strip_prefix('"').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "current run path is not a string",
        )
    })?;
    let end = value.find('"').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "current run path is unterminated",
        )
    })?;
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
    events
        .iter()
        .rev()
        .find(|event| event.get("category") == category)
}

fn latest_display_event(events: &[Event]) -> Option<&Event> {
    latest(events, "TrainingEvent").or_else(|| events.last())
}

fn terminal_dimension(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or(default)
}

fn paint(text: &str, code: &str, color: bool) -> String {
    if color {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

fn sparkline(events: &[Event], field: &str, invert: bool) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let values: Vec<f64> = events
        .iter()
        .filter(|event| event.get("category") == "TrainingEvent")
        .filter_map(|event| event.number(field))
        .filter(|value| value.is_finite())
        .rev()
        .take(32)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if values.len() < 2 {
        return MISSING.to_owned();
    }
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (high - low).abs() < f64::EPSILON {
        return "▅".repeat(values.len());
    }
    values
        .into_iter()
        .map(|value| {
            let normalized = if invert {
                (high - value) / (high - low)
            } else {
                (value - low) / (high - low)
            };
            GLYPHS[(normalized * 7.0).round().clamp(0.0, 7.0) as usize]
        })
        .collect()
}

fn line(label: &str, value: &str, width: usize) {
    let available = width.saturating_sub(label.len() + 3);
    let visible: String = value.chars().take(available.max(1)).collect();
    println!("{label:<18} {visible}");
}

fn clipped(value: &str, width: usize) -> String {
    let mut result: String = value.chars().take(width.saturating_sub(1)).collect();
    if value.chars().count() > result.chars().count() {
        result.push('…');
    }
    result
}

fn meter(value: Option<f64>, maximum: f64, width: usize, color: bool, good_high: bool) -> String {
    let ratio = value
        .map(|value| (value / maximum).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let filled = (ratio * width as f64).round() as usize;
    let raw = format!(
        "{}{}",
        "█".repeat(filled),
        "░".repeat(width.saturating_sub(filled))
    );
    let code = if (ratio >= 0.65) == good_high {
        "1;92"
    } else if (ratio >= 0.35) == good_high {
        "1;93"
    } else {
        "1;91"
    };
    paint(&raw, code, color)
}

fn phase_index(phase: &str) -> usize {
    [
        "cuda_search_microbenchmarks",
        "hardware_cost_model",
        "behavior_cloning",
        "rl_2_3_hop",
        "rl_multihop_distractor",
        "rl_intersection",
        "rl_semantic_hybrid",
        "rl_robustness",
        "jetson_specialization",
        "deterministic_final_evaluation",
    ]
    .iter()
    .position(|known| *known == phase)
    .unwrap_or(0)
}

fn phase_rail(phase: &str, color: bool) -> String {
    let index = phase_index(phase);
    (0..10)
        .map(|position| {
            if position < index {
                paint("◆", "1;36", color)
            } else if position == index {
                paint("◉", "1;95", color)
            } else {
                paint("◇", "2;37", color)
            }
        })
        .collect::<Vec<_>>()
        .join("─")
}

#[allow(dead_code)]
fn showcase_row(label: &str, value: &str, accent: &str, width: usize, color: bool) {
    let raw_label = format!("{label:<14}");
    let raw_value = clipped(value, width.saturating_sub(22));
    let padding = width.saturating_sub(5 + raw_label.chars().count() + raw_value.chars().count());
    println!(
        "║  {} {}{}║",
        paint(&raw_label, "2;37", color),
        paint(&raw_value, accent, color),
        " ".repeat(padding)
    );
}

/// The showcase is intentionally still a read-only JSONL visualizer.  It is
/// designed for an audience-facing terminal without adding a control path to
/// the detached trainer.
#[allow(dead_code)]
fn draw_showcase(events: &[Event], color: bool, tty: bool, language: Language) {
    let width = terminal_dimension("COLUMNS", 120).clamp(80, 150);
    if tty {
        print!("\x1b[2J\x1b[H");
    }
    let Some(event) = latest_display_event(events) else {
        println!(
            "NEUROSEEK // {}",
            t(
                language,
                "SIGNAL ACQUISITION — waiting for durable metrics",
                "信号取得中 — 永続メトリクスを待機しています"
            )
        );
        return;
    };
    let trace = latest(events, "SearchTraceEvent").unwrap_or(event);
    let title = paint("N E U R O S E E K", "1;96", color);
    let subtitle = paint(
        t(
            language,
            "LEARNED GPU KNOWLEDGE PROCESSOR  //  LIVE SEARCH TELEMETRY",
            "学習型GPU知識プロセッサ  //  ライブ探索テレメトリ",
        ),
        "1;35",
        color,
    );
    println!("╔{}╗", "═".repeat(width.saturating_sub(2)));
    println!("║  {title:<width$}║", width = width.saturating_sub(4));
    println!("║  {subtitle:<width$}║", width = width.saturating_sub(4));
    println!("╠{}╣", "═".repeat(width.saturating_sub(2)));
    println!(
        "║  {}  {}  {}  {}{}║",
        paint(t(language, "MODE", "モード"), "2;37", color),
        paint(t(language, "READ-ONLY", "読み取り専用"), "1;92", color),
        paint(t(language, "SOURCE", "ソース"), "2;37", color),
        paint("metrics.jsonl", "1;37", color),
        " ".repeat(width.saturating_sub(45))
    );
    println!("╠{}╣", "─".repeat(width.saturating_sub(2)));
    println!(
        "║  {}  {}{}║",
        paint(
            t(language, "CURRICULUM VECTOR", "カリキュラム進行"),
            "1;36",
            color
        ),
        phase_rail(event.get("phase"), color),
        " ".repeat(width.saturating_sub(47))
    );
    println!(
        "║  {}{}{}║",
        paint(
            t(language, "ACTIVE PHASE  ", "現在のフェーズ  "),
            "2;37",
            color
        ),
        paint(event.get("phase"), "1;95", color),
        " ".repeat(width.saturating_sub(18 + event.get("phase").chars().count()))
    );
    println!(
        "║  {}{}  {}{}{}║",
        paint(t(language, "GLOBAL STEP ", "総ステップ "), "2;37", color),
        paint(event.get("global_step"), "1;97", color),
        paint(t(language, "ELAPSED ", "経過時間 "), "2;37", color),
        paint(event.get("elapsed_seconds"), "1;97", color),
        " ".repeat(width.saturating_sub(
            42 + event.get("global_step").len() + event.get("elapsed_seconds").len()
        ))
    );
    println!("╠{}╣", "─".repeat(width.saturating_sub(2)));
    println!(
        "║  {}{}║",
        paint(
            t(language, "SEARCH INTELLIGENCE", "探索インテリジェンス"),
            "1;96",
            color
        ),
        " ".repeat(width.saturating_sub(23))
    );
    let success = event.number("success_rate");
    let proof = event.number("proof_validity");
    println!(
        "║  {}  {}  {:>7}    {}  {}  {:>7}{}║",
        t(language, "SUCCESS", "成功率"),
        meter(success, 1.0, 18, color, true),
        event.get("success_rate"),
        t(language, "PROOF", "証明"),
        meter(proof, 1.0, 18, color, true),
        event.get("proof_validity"),
        " ".repeat(width.saturating_sub(80))
    );
    println!(
        "║  {}   {}  {}{}║",
        t(language, "REWARD", "報酬"),
        paint(&sparkline(events, "reward", false), "1;92", color),
        paint(event.get("reward"), "1;97", color),
        " ".repeat(width.saturating_sub(45 + event.get("reward").len()))
    );
    println!(
        "║  {}     {}  {}{}║",
        t(language, "COST", "コスト"),
        paint(&sparkline(events, "credits_per_query", true), "1;93", color),
        paint(event.get("credits_per_query"), "1;97", color),
        " ".repeat(width.saturating_sub(45 + event.get("credits_per_query").len()))
    );
    println!(
        "║  {} {:>11}     {} {:>11}     {} {:>11}{}║",
        t(language, "NODES", "ノード"),
        event.get("nodes_per_query"),
        t(language, "EDGES", "エッジ"),
        event.get("edges_per_query"),
        t(language, "CREDITS", "クレジット"),
        event.get("credits_per_query"),
        " ".repeat(width.saturating_sub(
            57 + event.get("nodes_per_query").len()
                + event.get("edges_per_query").len()
                + event.get("credits_per_query").len()
        ))
    );
    println!("╠{}╣", "─".repeat(width.saturating_sub(2)));
    println!(
        "║  {}{}║",
        paint(t(language, "POLICY FIELD", "方策フィールド"), "1;95", color),
        " ".repeat(width.saturating_sub(16))
    );
    println!(
        "║  {} {:>10}    {} {:>10}    {} {:>10}{}║",
        t(language, "ENTROPY", "エントロピー"),
        event.get("entropy"),
        t(language, "VALUE LOSS", "価値損失"),
        event.get("value_loss"),
        t(language, "GRAD NORM", "勾配ノルム"),
        event.get("gradient_norm"),
        " ".repeat(width.saturating_sub(
            54 + event.get("entropy").len()
                + event.get("value_loss").len()
                + event.get("gradient_norm").len()
        ))
    );
    println!(
        "║  {} {}{}║",
        t(language, "OPERATOR", "演算子"),
        paint(event.get("operator_sample"), "1;95", color),
        " ".repeat(width.saturating_sub(13 + event.get("operator_sample").len()))
    );
    println!("╠{}╣", "─".repeat(width.saturating_sub(2)));
    println!(
        "║  {}{}║",
        paint(
            t(language, "LIVE PROOF PROGRAM", "ライブ証明プログラム"),
            "1;33",
            color
        ),
        " ".repeat(width.saturating_sub(22))
    );
    showcase_row(
        t(language, "TASK", "タスク"),
        trace.get("task_id"),
        "1;97",
        width,
        color,
    );
    showcase_row(
        t(language, "FAMILY", "系統"),
        trace.get("family"),
        "1;36",
        width,
        color,
    );
    showcase_row(
        t(language, "RESULT", "結果"),
        trace.get("result"),
        if trace.get("result") == "VALID" {
            "1;92"
        } else {
            "1;93"
        },
        width,
        color,
    );
    showcase_row(
        t(language, "TRACE", "トレース"),
        trace.get("trace"),
        "1;37",
        width,
        color,
    );
    println!("╠{}╣", "─".repeat(width.saturating_sub(2)));
    let temp = event.number("temperature_c");
    let ram = match (event.number("ram_used_gib"), event.number("ram_total_gib")) {
        (Some(used), Some(total)) if total > 0.0 => Some(used / total),
        _ => None,
    };
    println!(
        "║  {}{}║",
        paint(
            t(
                language,
                "JETSON THERMAL / MEMORY ENVELOPE",
                "JETSON 熱 / メモリ状況"
            ),
            "1;91",
            color
        ),
        " ".repeat(width.saturating_sub(38))
    );
    println!(
        "║  {} {}  {:>7}°C    RAM {}  {}/{} GiB{}║",
        t(language, "TEMP", "温度"),
        meter(temp, 85.0, 16, color, false),
        event.get("temperature_c"),
        meter(ram, 1.0, 16, color, false),
        event.get("ram_used_gib"),
        event.get("ram_total_gib"),
        " ".repeat(width.saturating_sub(
            76 + event.get("temperature_c").len()
                + event.get("ram_used_gib").len()
                + event.get("ram_total_gib").len()
        ))
    );
    println!(
        "║  CUDA {}  {} {}{}║",
        paint(event.get("cuda"), "1;92", color),
        t(language, "GRAPH SESSION", "グラフセッション"),
        paint(event.get("cuda_graph_session"), "1;92", color),
        " ".repeat(
            width.saturating_sub(
                31 + event.get("cuda").len() + event.get("cuda_graph_session").len()
            )
        )
    );
    println!("╚{}╝", "═".repeat(width.saturating_sub(2)));
    println!(
        "{}  {}",
        paint("● LIVE", "1;92", color),
        t(
            language,
            "Ctrl-C closes this read-only display; the trainer has no TUI control channel.",
            "Ctrl-Cは読み取り専用表示を閉じるだけです。学習器への制御経路はありません。"
        )
    );
    let _ = io::stdout().flush();
}

fn draw(events: &[Event], color: bool, tty: bool, language: Language) {
    let width = terminal_dimension("COLUMNS", 100);
    let height = terminal_dimension("LINES", 28);
    let compact = width < 80 || height < 20;
    if tty {
        print!("\x1b[2J\x1b[H");
    }
    let Some(event) = latest_display_event(events) else {
        println!(
            "NEUROSEEK // {}",
            t(
                language,
                "waiting for durable metrics event",
                "永続メトリクスイベントを待機しています"
            )
        );
        return;
    };
    let title = paint(
        t(
            language,
            "NEUROSEEK // LEARNED GPU KNOWLEDGE PROCESSOR",
            "NEUROSEEK // 学習型GPU知識プロセッサ",
        ),
        "1;36",
        color,
    );
    println!("{title}");
    println!(
        "{}",
        t(
            language,
            "read-only dashboard · trainer remains detached · source: metrics.jsonl",
            "読み取り専用ダッシュボード · 学習器は分離済み · ソース: metrics.jsonl"
        )
    );
    println!("{}", "─".repeat(width.min(100)));
    line(t(language, "PHASE", "フェーズ"), event.get("phase"), width);
    line(
        t(language, "GLOBAL STEP", "総ステップ"),
        event.get("global_step"),
        width,
    );
    line(
        t(language, "ELAPSED", "経過時間"),
        event.get("elapsed_seconds"),
        width,
    );
    line(
        t(language, "LAST EVENT", "最新イベント"),
        event.get("category"),
        width,
    );
    if compact {
        line(t(language, "REWARD", "報酬"), event.get("reward"), width);
        line(
            t(language, "SUCCESS", "成功率"),
            event.get("success_rate"),
            width,
        );
        line(
            t(language, "PROOF VALID", "有効な証明"),
            event.get("proof_validity"),
            width,
        );
        line(
            t(language, "NODES / QUERY", "ノード / 問合せ"),
            event.get("nodes_per_query"),
            width,
        );
        line(
            t(language, "EDGES / QUERY", "エッジ / 問合せ"),
            event.get("edges_per_query"),
            width,
        );
        line(
            t(language, "TEMPERATURE C", "温度 C"),
            event.get("temperature_c"),
            width,
        );
        println!(
            "{}",
            t(
                language,
                "compact mode · Ctrl-C detaches display only",
                "コンパクト表示 · Ctrl-Cは表示を閉じるだけです"
            )
        );
    } else {
        println!(
            "\n{}",
            t(language, "SEARCH INTELLIGENCE", "探索インテリジェンス")
        );
        line(t(language, "REWARD", "報酬"), event.get("reward"), width);
        line(
            t(language, "SUCCESS", "成功率"),
            event.get("success_rate"),
            width,
        );
        line(
            t(language, "PROOF VALIDITY", "証明有効性"),
            event.get("proof_validity"),
            width,
        );
        line(
            t(language, "NODES / QUERY", "ノード / 問合せ"),
            event.get("nodes_per_query"),
            width,
        );
        line(
            t(language, "EDGES / QUERY", "エッジ / 問合せ"),
            event.get("edges_per_query"),
            width,
        );
        line(
            t(language, "CREDITS / QUERY", "クレジット / 問合せ"),
            event.get("credits_per_query"),
            width,
        );
        println!(
            "{}  {}",
            t(language, "REWARD TREND", "報酬トレンド"),
            sparkline(events, "reward", false)
        );
        println!(
            "{}  {}",
            t(language, "SEARCH COST TREND", "探索コストトレンド"),
            sparkline(events, "nodes_per_query", true)
        );
        println!("\n{}", t(language, "POLICY", "方策"));
        line(
            t(language, "POLICY LOSS", "方策損失"),
            event.get("policy_loss"),
            width,
        );
        line(
            t(language, "VALUE LOSS", "価値損失"),
            event.get("value_loss"),
            width,
        );
        line(
            t(language, "ENTROPY", "エントロピー"),
            event.get("entropy"),
            width,
        );
        line("KL", event.get("kl"), width);
        line(
            t(language, "GRADIENT NORM", "勾配ノルム"),
            event.get("gradient_norm"),
            width,
        );
        line(
            t(language, "OPERATOR SAMPLE", "演算子サンプル"),
            event.get("operator_sample"),
            width,
        );
        println!(
            "\n{}",
            t(
                language,
                "LIVE SEARCH (durable SearchTraceEvent)",
                "ライブ探索（永続SearchTraceEvent）"
            )
        );
        let trace = latest(events, "SearchTraceEvent").unwrap_or(event);
        line(t(language, "TASK", "タスク"), trace.get("task_id"), width);
        line(t(language, "FAMILY", "系統"), trace.get("family"), width);
        line(t(language, "RESULT", "結果"), trace.get("result"), width);
        line(
            t(language, "PROGRAM", "プログラム"),
            trace.get("trace"),
            width,
        );
        println!(
            "\n{}",
            t(
                language,
                "OPERATOR DISTRIBUTION (current rollout batch)",
                "演算子分布（現在のロールアウトバッチ）"
            )
        );
        line(
            t(language, "OPERATORS", "演算子"),
            event.get("operator_distribution"),
            width,
        );
        println!(
            "\n{}",
            t(
                language,
                "JETSON (only fields emitted by telemetry)",
                "JETSON（テレメトリが出力した項目のみ）"
            )
        );
        line(
            t(language, "CUDA ACTIVE", "CUDA有効"),
            event.get("cuda"),
            width,
        );
        line(
            t(language, "RAM USED GiB", "使用RAM GiB"),
            event.get("ram_used_gib"),
            width,
        );
        line(
            t(language, "RAM TOTAL GiB", "総RAM GiB"),
            event.get("ram_total_gib"),
            width,
        );
        line(
            t(language, "TEMPERATURE C", "温度 C"),
            event.get("temperature_c"),
            width,
        );
        if let Some(checkpoint) = latest(events, "CheckpointEvent") {
            println!("\n{}", t(language, "CHECKPOINT", "チェックポイント"));
            line(
                t(language, "LAST SAVED", "最終保存"),
                checkpoint.get("checkpoint"),
                width,
            );
            line(
                t(language, "SAVE REASON", "保存理由"),
                checkpoint.get("reason"),
                width,
            );
        }
        println!(
            "\n{}",
            t(
                language,
                "Ctrl-C detaches display only; training continues in its detached service.",
                "Ctrl-Cは表示を閉じるだけです。学習は分離済みサービスで継続します。"
            )
        );
    }
    let _ = io::stdout().flush();
}

fn label(screen: Screen, language: Language) -> &'static str {
    match screen {
        Screen::Explore => t(language, "EXPLORE", "探索"),
        Screen::Trace => t(language, "TRACE", "経路"),
        Screen::System => t(language, "SYSTEM", "システム"),
        Screen::Model => t(language, "MODEL", "モデル"),
    }
}

fn operator_steps(trace: &str) -> Vec<&str> {
    trace
        .split(" -> ")
        .filter(|step| !step.trim().is_empty())
        .take(10)
        .collect()
}

fn operation_kind(step: &str) -> &'static str {
    if step.starts_with("SEED") {
        "SEED"
    } else if step.starts_with("EXPAND") {
        "EXPAND"
    } else if step.starts_with("VERIFY") {
        "VERIFY"
    } else if step.starts_with("FILTER") {
        "FILTER"
    } else if step.starts_with("INTERSECT") {
        "INTERSECT"
    } else if step.starts_with("BACKTRACK") {
        "BACKTRACK"
    } else {
        "OP"
    }
}

/// Draw a fixed-height rooted program tree.  Search traces change on every
/// refresh, but its 10 reserved leaf rows do not: that keeps the telemetry
/// panel below from jumping as operators are added or removed.
fn render_program_tree(trace: &str, color: bool, language: Language, width: usize) {
    let steps = operator_steps(trace);
    println!(
        "  {}",
        paint(
            t(
                language,
                "SEARCH PROGRAM TREE  ·  durable program path",
                "探索プログラム樹形図  ·  永続プログラム経路"
            ),
            "1;96",
            color
        )
    );
    println!(
        "  {}",
        paint(
            t(
                language,
                "Fixed tree slots: only durable SearchTraceEvent operators are shown.",
                "固定スロットの樹形図です。SearchTraceEventに保存済みの演算子だけを表示します。"
            ),
            "2;37",
            color
        )
    );
    println!(
        "  {} {}",
        paint("◉", "1;95", color),
        paint(t(language, "QUERY ROOT", "問合せルート"), "1;97", color)
    );
    for index in 0..10 {
        let last_slot = index == 9;
        let connector = if last_slot { "└─" } else { "├─" };
        let (marker, text, style) = match steps.get(index) {
            Some(step) => (
                if index + 1 == steps.len() {
                    "◆"
                } else {
                    "●"
                },
                clipped(step, width.saturating_sub(22)),
                if index + 1 == steps.len() {
                    "1;95"
                } else {
                    "1;36"
                },
            ),
            None => ("·", "—".to_owned(), "2;37"),
        };
        println!(
            "  {} {} {}  {}",
            paint(connector, "2;36", color),
            paint(marker, style, color),
            paint(&format!("{:02}", index + 1), "2;37", color),
            paint(&text, if text == "—" { "2;37" } else { "1;97" }, color)
        );
    }
}

fn render_explore(
    events: &[Event],
    event: &Event,
    trace: &Event,
    color: bool,
    language: Language,
    width: usize,
) {
    println!(
        "\n  {}  {}",
        paint(t(language, "LIVE QUERY", "ライブクエリ"), "1;97", color),
        paint(trace.get("task_id"), "1;36", color)
    );
    println!(
        "  {}  {}     {}  {}     {}  {}",
        t(language, "FAMILY", "系統"),
        paint(trace.get("family"), "1;96", color),
        t(language, "RESULT", "結果"),
        paint(
            trace.get("result"),
            if trace.get("result") == "VALID" {
                "1;92"
            } else {
                "1;93"
            },
            color
        ),
        t(language, "REWARD", "報酬"),
        paint(event.get("reward"), "1;97", color)
    );
    println!();
    render_program_tree(trace.get("trace"), color, language, width);
    println!();
    println!(
        "  {}  {}  {}  {}  {}  {}",
        t(language, "SUCCESS", "成功率"),
        meter(event.number("success_rate"), 1.0, 20, color, true),
        t(language, "PROOF", "証明"),
        meter(event.number("proof_validity"), 1.0, 20, color, true),
        t(language, "CREDITS", "クレジット"),
        paint(event.get("credits_per_query"), "1;93", color)
    );
    println!(
        "  {}  {}",
        t(language, "REWARD PULSE", "報酬パルス"),
        paint(&sparkline(events, "reward", false), "1;92", color)
    );
}

fn render_trace(event: &Event, trace: &Event, color: bool, language: Language, width: usize) {
    println!(
        "\n  {}",
        paint(
            t(language, "PROGRAM INSPECTOR", "プログラムインスペクタ"),
            "1;97",
            color
        )
    );
    println!(
        "  {}  {}",
        t(language, "TASK", "タスク"),
        paint(trace.get("task_id"), "1;36", color)
    );
    println!(
        "  {}  {}",
        t(language, "FAMILY", "系統"),
        trace.get("family")
    );
    println!(
        "  {}  {}",
        t(language, "PHASE", "フェーズ"),
        paint(event.get("phase"), "1;95", color)
    );
    println!();
    for (index, step) in operator_steps(trace.get("trace")).iter().enumerate() {
        let kind = operation_kind(step);
        println!(
            "  {} {}  {:<11}  {}",
            paint(
                if index + 1 == operator_steps(trace.get("trace")).len() {
                    "◆"
                } else {
                    "◇"
                },
                "1;36",
                color
            ),
            paint(&format!("{:02}", index + 1), "2;37", color),
            paint(kind, "1;95", color),
            clipped(step, width.saturating_sub(28))
        );
    }
    println!();
    println!(
        "  {}",
        paint(t(language, "OPERATOR MIX", "演算子ミックス"), "1;96", color)
    );
    println!(
        "  {}",
        clipped(event.get("operator_distribution"), width.saturating_sub(4))
    );
    println!();
    println!(
        "  {}",
        paint(
            t(
                language,
                "The viewer uses only durable events already appended by the trainer.",
                "このビューアは学習器が追記済みの永続イベントだけを使用します。"
            ),
            "2;37",
            color
        )
    );
}

fn render_system(event: &Event, color: bool, language: Language) {
    println!(
        "\n  {}",
        paint(
            t(language, "JETSON TELEMETRY", "JETSON テレメトリ"),
            "1;97",
            color
        )
    );
    println!("\n  CUDA   {}", paint(event.get("cuda"), "1;92", color));
    println!(
        "  TEMP   {}  {} °C",
        meter(event.number("temperature_c"), 85.0, 32, color, false),
        paint(event.get("temperature_c"), "1;97", color)
    );
    let ram_ratio = match (event.number("ram_used_gib"), event.number("ram_total_gib")) {
        (Some(used), Some(total)) if total > 0.0 => Some(used / total),
        _ => None,
    };
    println!(
        "  RAM    {}  {}/{} GiB",
        meter(ram_ratio, 1.0, 32, color, false),
        event.get("ram_used_gib"),
        event.get("ram_total_gib")
    );
    println!();
    println!(
        "  {}",
        paint(t(language, "SEARCH COST", "探索コスト"), "1;96", color)
    );
    println!(
        "  {}  {}     {}  {}     {}  {}",
        t(language, "NODES", "ノード"),
        event.get("nodes_per_query"),
        t(language, "EDGES", "エッジ"),
        event.get("edges_per_query"),
        t(language, "CREDITS", "クレジット"),
        event.get("credits_per_query")
    );
    println!();
    println!("  {}", paint(t(language, "Only telemetry emitted in metrics.jsonl is displayed; CPU/GPU utilization is never inferred.", "metrics.jsonlが出力したテレメトリだけを表示します。CPU/GPU使用率は推測しません。"), "2;37", color));
}

fn render_model(events: &[Event], event: &Event, color: bool, language: Language) {
    println!(
        "\n  {}",
        paint(
            t(language, "POLICY / VALUE FIELD", "方策 / 価値フィールド"),
            "1;97",
            color
        )
    );
    println!(
        "\n  {}  {}     {}  {}",
        t(language, "POLICY LOSS", "方策損失"),
        event.get("policy_loss"),
        t(language, "VALUE LOSS", "価値損失"),
        event.get("value_loss")
    );
    println!(
        "  {}  {}     KL  {}     {}  {}",
        t(language, "ENTROPY", "エントロピー"),
        event.get("entropy"),
        event.get("kl"),
        t(language, "GRADIENT", "勾配"),
        event.get("gradient_norm")
    );
    println!();
    println!(
        "  {}  {}",
        t(language, "CURRICULUM", "カリキュラム"),
        phase_rail(event.get("phase"), color)
    );
    println!(
        "  {}  {}",
        t(language, "ACTIVE", "現在"),
        paint(event.get("phase"), "1;95", color)
    );
    println!();
    println!(
        "  {}  {}",
        t(language, "REWARD HISTORY", "報酬履歴"),
        paint(&sparkline(events, "reward", false), "1;92", color)
    );
    println!(
        "  {}  {}",
        t(language, "COST HISTORY", "コスト履歴"),
        paint(&sparkline(events, "credits_per_query", true), "1;93", color)
    );
}

fn draw_cinematic(
    events: &[Event],
    screen: Screen,
    input: &str,
    notice: &str,
    color: bool,
    tty: bool,
    language: Language,
) {
    let width = terminal_dimension("COLUMNS", 120).clamp(90, 180);
    if tty {
        print!("\x1b[H\x1b[2J");
    }
    let Some(event) = latest_display_event(events) else {
        println!(
            "NEUROSEEK // {}",
            t(
                language,
                "waiting for durable metrics",
                "永続メトリクスを待機しています"
            )
        );
        return;
    };
    let trace = latest(events, "SearchTraceEvent").unwrap_or(event);
    println!(
        " {}  {}",
        paint("NEUROSEEK", "1;96", color),
        paint(
            t(language, "KNOWLEDGE SEARCH CONSOLE", "知識探索コンソール"),
            "2;37",
            color
        )
    );
    println!(
        " {}  {}  ·  {}  {}  ·  {}  {}",
        paint("●", "1;92", color),
        t(language, "LIVE", "ライブ"),
        t(language, "STEP", "ステップ"),
        event.get("global_step"),
        t(language, "PHASE", "フェーズ"),
        paint(event.get("phase"), "1;95", color)
    );
    println!(
        " {}",
        paint(&"─".repeat(width.saturating_sub(2)), "2;36", color)
    );
    for (number, tab) in [
        (1, Screen::Explore),
        (2, Screen::Trace),
        (3, Screen::System),
        (4, Screen::Model),
    ] {
        let active = tab == screen;
        print!(
            " {} {}",
            paint(
                &format!("[{number}]"),
                if active { "1;97" } else { "2;37" },
                color
            ),
            paint(
                label(tab, language),
                if active { "1;96" } else { "2;37" },
                color
            )
        );
        if number < 4 {
            print!("   ");
        }
    }
    println!(
        "     {}",
        paint(
            t(
                language,
                "l language  / command  q quit",
                "l 言語  / コマンド  q 終了"
            ),
            "2;37",
            color
        )
    );
    println!(
        " {}",
        paint(&"─".repeat(width.saturating_sub(2)), "2;36", color)
    );
    match screen {
        Screen::Explore => render_explore(events, event, trace, color, language, width),
        Screen::Trace => render_trace(event, trace, color, language, width),
        Screen::System => render_system(event, color, language),
        Screen::Model => render_model(events, event, color, language),
    }
    let min_lines = terminal_dimension("LINES", 40).saturating_sub(22);
    for _ in 0..min_lines {
        println!();
    }
    println!(
        " {}",
        paint(&"─".repeat(width.saturating_sub(2)), "2;36", color)
    );
    let prompt = if input.is_empty() {
        t(
            language,
            "type /help for viewer commands",
            " /help でビューアコマンドを表示",
        )
    } else {
        input
    };
    println!(
        " {} {}",
        paint("›", "1;96", color),
        paint(
            prompt,
            if input.is_empty() { "2;37" } else { "1;97" },
            color
        )
    );
    println!(" {}", paint(notice, "2;37", color));
    let _ = io::stdout().flush();
}

struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    fn enter(tty: bool) -> Self {
        if !tty {
            return Self { active: false };
        }
        let active = Command::new("stty")
            .args(["-icanon", "-echo", "min", "0", "time", "0"])
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if active {
            print!("\x1b[?1049h\x1b[?25l");
            let _ = io::stdout().flush();
        }
        Self { active }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = Command::new("stty").arg("sane").status();
            print!("\x1b[?25h\x1b[?1049l");
            let _ = io::stdout().flush();
        }
    }
}

fn key_channel() -> mpsc::Receiver<u8> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut stdin = io::stdin();
        let mut byte = [0_u8; 1];
        loop {
            if stdin
                .read(&mut byte)
                .ok()
                .filter(|count| *count > 0)
                .is_some()
                && sender.send(byte[0]).is_err()
            {
                break;
            }
        }
    });
    receiver
}

fn apply_viewer_command(
    command: &str,
    screen: &mut Screen,
    language: &mut Language,
    notice: &mut String,
) -> bool {
    match command.trim().trim_start_matches('/') {
        "q" | "quit" | "exit" => true,
        "help" => {
            *notice = "1 explore · 2 trace · 3 system · 4 model · l language · /lang ja|en · /quit"
                .into();
            false
        }
        "explore" | "1" => {
            *screen = Screen::Explore;
            false
        }
        "trace" | "2" => {
            *screen = Screen::Trace;
            false
        }
        "system" | "3" => {
            *screen = Screen::System;
            false
        }
        "model" | "4" => {
            *screen = Screen::Model;
            false
        }
        "lang ja" | "ja" => {
            *language = Language::Japanese;
            *notice = "表示言語: 日本語".into();
            false
        }
        "lang en" | "en" => {
            *language = Language::English;
            *notice = "Display language: English".into();
            false
        }
        _ => {
            *notice = "Unknown local viewer command. Type /help.".into();
            false
        }
    }
}

fn usage() -> ! {
    eprintln!("usage: neuroseek-tui <metrics.jsonl> [--showcase] [--lang en|ja] [--once] [--no-color]\n       neuroseek-tui --attach-current [--showcase] [--lang en|ja] [--once] [--no-color]");
    std::process::exit(2)
}

fn parse_flags(flags: &[String]) -> Option<(bool, bool, bool, Language)> {
    let mut once = false;
    let mut showcase = false;
    let mut no_color = false;
    let mut language = Language::English;
    let mut index = 0;
    while index < flags.len() {
        match flags[index].as_str() {
            "--once" => once = true,
            "--showcase" => showcase = true,
            "--no-color" => no_color = true,
            "--lang" => {
                index += 1;
                language = Language::parse(flags.get(index)?.as_str())?;
            }
            value if value.starts_with("--lang=") => {
                language = Language::parse(value.trim_start_matches("--lang="))?
            }
            _ => return None,
        }
        index += 1;
    }
    Some((once, showcase, no_color, language))
}

fn main() {
    let mut args = env::args().skip(1);
    let first = args.next().unwrap_or_else(|| usage());
    let path = if first == "--attach-current" {
        current_metrics_path().unwrap_or_else(|error| {
            eprintln!("cannot attach current run: {error}");
            std::process::exit(1)
        })
    } else if first.starts_with('-') {
        usage()
    } else {
        first
    };
    let flags: Vec<String> = args.collect();
    let (once, showcase, no_color, language) = parse_flags(&flags).unwrap_or_else(|| usage());
    let tty = io::stdout().is_terminal();
    let color = tty && env::var_os("NO_COLOR").is_none() && !no_color;
    if showcase && tty && !once {
        let _terminal = TerminalGuard::enter(true);
        let receiver = key_channel();
        let mut screen = Screen::Explore;
        let mut language = language;
        let mut input = String::new();
        let mut notice = t(
            language,
            "1–4 switch views · / opens the command bar",
            "1〜4で表示切替 · /でコマンドバーを開きます",
        )
        .to_owned();
        let mut last_draw = Instant::now() - Duration::from_secs(1);
        loop {
            if last_draw.elapsed() >= Duration::from_millis(180) {
                match tail_events(Path::new(&path)) {
                    Ok(events) if !events.is_empty() => {
                        draw_cinematic(&events, screen, &input, &notice, color, true, language)
                    }
                    Ok(_) => eprintln!("waiting for complete durable metrics event: {path}"),
                    Err(error) => eprintln!("waiting for {path}: {error}"),
                }
                last_draw = Instant::now();
            }
            match receiver.recv_timeout(Duration::from_millis(40)) {
                Ok(b'\x03') | Ok(b'q') if input.is_empty() => break,
                Ok(b'l') if input.is_empty() => {
                    language = if language == Language::English {
                        Language::Japanese
                    } else {
                        Language::English
                    };
                    notice =
                        t(language, "Display language: English", "表示言語: 日本語").to_owned();
                }
                Ok(key) if input.is_empty() && Screen::from_key(key).is_some() => {
                    screen = Screen::from_key(key).unwrap()
                }
                Ok(b'\r') | Ok(b'\n') if !input.is_empty() => {
                    if apply_viewer_command(&input, &mut screen, &mut language, &mut notice) {
                        break;
                    }
                    input.clear();
                }
                Ok(b'\x1b') => input.clear(),
                Ok(b'\x7f') | Ok(b'\x08') => {
                    input.pop();
                }
                Ok(byte @ b' '..=b'~') => input.push(byte as char),
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        return;
    }
    loop {
        match tail_events(Path::new(&path)) {
            Ok(events) if !events.is_empty() => {
                if showcase {
                    draw_cinematic(&events, Screen::Explore, "", "", color, tty, language)
                } else {
                    draw(&events, color, tty, language)
                }
            }
            Ok(_) => eprintln!("waiting for complete durable metrics event: {path}"),
            Err(error) => eprintln!("waiting for {path}: {error}"),
        }
        if once || !tty {
            break;
        }
        thread::sleep(Duration::from_secs(1));
    }
}

#[cfg(test)]
mod tests {
    use super::{current_metrics_path, parse_event, parse_flags, sparkline, tail_events, Language};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn parses_flat_trainer_events_and_rejects_partial_json() {
        let event = parse_event(
            r#"{"category":"TrainingEvent","global_step":12,"reward":3.5,"cuda":true}"#,
        )
        .unwrap();
        assert_eq!(event.get("category"), "TrainingEvent");
        assert_eq!(event.get("global_step"), "12");
        assert_eq!(event.number("reward"), Some(3.5));
        assert!(parse_event(r#"{"category":"TrainingEvent""#).is_none());
        let nested = parse_event(
            r#"{"category":"TrainingEvent","reward":2,"thermal_sensors_c":{"gpu":50.0}}"#,
        )
        .unwrap();
        assert_eq!(nested.number("reward"), Some(2.0));
    }

    #[test]
    fn bounded_tail_ignores_split_first_record_and_keeps_latest() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
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
    fn language_flags_select_english_or_japanese() {
        let japanese = parse_flags(&["--showcase".into(), "--lang".into(), "ja".into()]).unwrap();
        assert_eq!(japanese, (false, true, false, Language::Japanese));
        let english = parse_flags(&["--lang=en".into(), "--once".into()]).unwrap();
        assert_eq!(english, (true, false, false, Language::English));
        assert!(parse_flags(&["--lang".into(), "fr".into()]).is_none());
    }

    #[test]
    fn attach_current_uses_launcher_metadata() {
        let original = std::fs::read_to_string("runs/current.json").ok();
        std::fs::create_dir_all("runs").unwrap();
        std::fs::write("runs/current.json", r#"{"path":"runs/example-run"}"#).unwrap();
        assert_eq!(
            current_metrics_path().unwrap(),
            "runs/example-run/metrics.jsonl"
        );
        if let Some(content) = original {
            std::fs::write("runs/current.json", content).unwrap();
        } else {
            std::fs::remove_file("runs/current.json").unwrap();
        }
    }
}
