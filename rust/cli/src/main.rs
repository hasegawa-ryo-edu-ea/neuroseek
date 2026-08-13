//! JSONL acceptance runner for the deterministic native Search VM.
//!
//! It intentionally has no hidden fixture or implicit CPU fallback. The input
//! describes a bounded CSR and program, and output is one structured event per
//! input record. This makes it suitable for smoke/preflight diagnostics and
//! for retaining exact native traces alongside Python training metrics.

use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use neuroseek_core::{CsrGraph, Edge, Instruction, QuerySpec, SearchVm, VmError, VmStep, validate_proof};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputRecord {
    Graph { nodes: usize, edges: Vec<JsonEdge> },
    Program { budget: u64, instructions: Vec<JsonInstruction>, #[serde(default)] query: Option<JsonQuery> },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct JsonEdge { src: u32, relation: u16, dst: u32 }
impl From<JsonEdge> for Edge { fn from(edge: JsonEdge) -> Self { Edge { src: edge.src, relation: edge.relation, dst: edge.dst } } }
impl From<Edge> for JsonEdge { fn from(edge: Edge) -> Self { Self { src: edge.src, relation: edge.relation, dst: edge.dst } } }

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "SCREAMING_SNAKE_CASE")]
enum JsonInstruction {
    Seed { node: u32 }, Ann { k: usize }, ExpandRel { relation: u16 }, ExpandAny,
    Filter { relation: u16 }, Intersect { register: u8 }, Union { register: u8 },
    Prune { k: usize }, TopK { k: usize }, Verify, Backtrack, Prefetch, Evict, Stop,
}
impl From<JsonInstruction> for Instruction {
    fn from(op: JsonInstruction) -> Self {
        match op {
            JsonInstruction::Seed { node } => Self::Seed(node), JsonInstruction::Ann { k } => Self::Ann { k },
            JsonInstruction::ExpandRel { relation } => Self::ExpandRel(relation), JsonInstruction::ExpandAny => Self::ExpandAny,
            JsonInstruction::Filter { relation } => Self::FilterRelation(relation), JsonInstruction::Intersect { register } => Self::Intersect(register),
            JsonInstruction::Union { register } => Self::Union(register), JsonInstruction::Prune { k } => Self::Prune(k),
            JsonInstruction::TopK { k } => Self::TopK(k), JsonInstruction::Verify => Self::Verify,
            JsonInstruction::Backtrack => Self::Backtrack, JsonInstruction::Prefetch => Self::Prefetch,
            JsonInstruction::Evict => Self::Evict, JsonInstruction::Stop => Self::Stop,
        }
    }
}

#[derive(Debug, Deserialize)]
struct JsonQuery { #[serde(default)] required_edges: Vec<JsonEdge>, answers: Vec<u32> }

#[derive(Serialize)]
struct StatsOut {
    attempted_instructions: u64, instructions: u64, failed_instructions: u64,
    nodes_visited: u64, edges_examined: u64, ann_calls: u64, credits: u64,
    bytes_estimated: u64, max_depth: u32, backtracks: u64, prefetches: u64,
    evictions: u64, frontier_peak: u64, proof_edges: u64,
}
impl From<&neuroseek_core::VmStats> for StatsOut {
    fn from(stats: &neuroseek_core::VmStats) -> Self {
        Self { attempted_instructions: stats.attempted_instructions, instructions: stats.instructions, failed_instructions: stats.failed_instructions, nodes_visited: stats.nodes_visited, edges_examined: stats.edges_examined, ann_calls: stats.ann_calls, credits: stats.credits, bytes_estimated: stats.bytes_estimated, max_depth: stats.max_depth, backtracks: stats.backtracks, prefetches: stats.prefetches, evictions: stats.evictions, frontier_peak: stats.frontier_peak, proof_edges: stats.proof_edges }
    }
}

#[derive(Serialize)]
struct StepOut { event: &'static str, index: usize, opcode: String, frontier_before: usize, frontier_after: usize, answer: Option<u32>, proof_edges: usize, budget_remaining: u64, stats: StatsOut }
impl StepOut {
    fn from_step(index: usize, step: VmStep) -> Self {
        Self { event: "vm_step", index, opcode: format!("{:?}", step.opcode), frontier_before: step.frontier_before, frontier_after: step.frontier_after, answer: step.answer, proof_edges: step.proof_edges, budget_remaining: step.budget_remaining, stats: StatsOut::from(&step.stats) }
    }
}

#[derive(Serialize)]
struct RunOut { event: &'static str, ok: bool, answer: Option<u32>, stopped: bool, proof_edges: Vec<JsonEdge>, proof_valid: Option<bool>, budget_remaining: u64, stats: StatsOut }
#[derive(Serialize)]
struct ErrorOut { event: &'static str, ok: bool, line: usize, error: String }

fn write_json<T: Serialize>(writer: &mut dyn Write, value: &T) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, value).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn vm_error(error: VmError) -> String { error.to_string() }

fn run_program(writer: &mut dyn Write, graph: &CsrGraph, budget: u64, instructions: Vec<JsonInstruction>, query: Option<JsonQuery>) -> io::Result<()> {
    let mut vm = SearchVm::new(graph, budget);
    for (index, raw) in instructions.into_iter().enumerate() {
        match vm.execute_traced(raw.into()) {
            Ok(step) => write_json(writer, &StepOut::from_step(index, step))?,
            Err(error) => {
                write_json(writer, &ErrorOut { event: "vm_error", ok: false, line: index, error: vm_error(error) })?;
                return Ok(());
            }
        }
        if vm.is_stopped() { break; }
    }
    let proof_valid = query.map(|query| validate_proof(graph, &QuerySpec { required_edges: query.required_edges.into_iter().map(Into::into).collect(), answers: query.answers.into_iter().collect::<BTreeSet<_>>() }, &vm.proof));
    let proof_edges = vm.proof.edges.iter().copied().map(Into::into).collect();
    write_json(writer, &RunOut { event: "vm_result", ok: true, answer: vm.proof.answer, stopped: vm.is_stopped(), proof_edges, proof_valid, budget_remaining: vm.budget, stats: StatsOut::from(&vm.stats) })
}

fn usage() -> &'static str { "usage: neuroseek-native --jsonl <fixture.jsonl|- >\nrecords: graph then program; output: JSONL vm_step/vm_result" }

fn run_jsonl(reader: impl BufRead, writer: &mut dyn Write) -> io::Result<()> {
    let mut graph: Option<CsrGraph> = None;
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let record: InputRecord = match serde_json::from_str(&line) {
            Ok(record) => record,
            Err(error) => { write_json(writer, &ErrorOut { event: "input_error", ok: false, line: line_index + 1, error: error.to_string() })?; continue; }
        };
        match record {
            InputRecord::Graph { nodes, edges } => match CsrGraph::from_edges(nodes, &edges.into_iter().map(Into::into).collect::<Vec<Edge>>()) {
                Ok(new_graph) => {
                    let edge_count = new_graph.edge_count();
                    graph = Some(new_graph);
                    write_json(writer, &serde_json::json!({"event":"graph_loaded", "ok":true, "nodes":nodes, "edges":edge_count}))?;
                }
                Err(error) => write_json(writer, &ErrorOut { event: "graph_error", ok: false, line: line_index + 1, error: error.to_string() })?,
            },
            InputRecord::Program { budget, instructions, query } => match graph.as_ref() {
                Some(graph) => run_program(writer, graph, budget, instructions, query)?,
                None => write_json(writer, &ErrorOut { event: "input_error", ok: false, line: line_index + 1, error: "program requires a preceding valid graph record".to_owned() })?,
            },
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.as_slice() == ["--help"] || args.as_slice() == ["-h"] { println!("{}", usage()); return; }
    if args.len() != 2 || args[0] != "--jsonl" { eprintln!("{}", usage()); std::process::exit(2); }
    let result = if args[1] == "-" {
        run_jsonl(BufReader::new(io::stdin().lock()), &mut io::stdout().lock())
    } else {
        File::open(Path::new(&args[1])).map(BufReader::new).and_then(|reader| run_jsonl(reader, &mut io::stdout().lock()))
    };
    if let Err(error) = result { eprintln!("neuroseek-native: {error}"); std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_fixture_runs_bounded_program_and_validates_proof() {
        let input = include_str!("../tests/fixtures/path_proof.jsonl");
        let mut output = Vec::new();
        run_jsonl(BufReader::new(input.as_bytes()), &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"event\":\"graph_loaded\""));
        assert_eq!(output.matches("\"event\":\"vm_step\"").count(), 5);
        assert!(output.contains("\"proof_valid\":true"));
        assert!(output.contains("\"answer\":2"));
    }

    #[test]
    fn malformed_or_unordered_input_is_reported_not_panicked() {
        let mut output = Vec::new();
        run_jsonl(BufReader::new(b"{\"type\":\"program\",\"budget\":1,\"instructions\":[]}\nnot-json\n".as_slice()), &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.matches("\"event\":\"input_error\"").count(), 2);
    }
}
