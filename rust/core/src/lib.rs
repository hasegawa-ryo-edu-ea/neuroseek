//! Native, deterministic semantics for NEUROSEEK learned graph search.
//!
//! This crate is deliberately the CPU semantic authority. CUDA may replace a
//! bulk operation only after parity validation; it must not change accounting,
//! proof construction, or instruction outcomes.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

pub type NodeId = u32;
pub type RelationId = u16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Edge {
    pub src: NodeId,
    pub relation: RelationId,
    pub dst: NodeId,
}

/// Compact CSR graph. Relation IDs are adjacent to compact target IDs and no
/// string processing is allowed on the search path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrGraph {
    pub offsets: Vec<u64>,
    pub targets: Vec<NodeId>,
    pub relations: Vec<RelationId>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum GraphError {
    InvalidOffsets,
    MismatchedEdges,
    NodeOutOfRange(NodeId),
}
impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for GraphError {}

impl CsrGraph {
    pub fn from_edges(nodes: usize, edges: &[Edge]) -> Result<Self, GraphError> {
        let mut sorted = edges.to_vec();
        sorted.sort();
        let mut offsets = vec![0u64; nodes + 1];
        for edge in &sorted {
            if edge.src as usize >= nodes || edge.dst as usize >= nodes {
                return Err(GraphError::NodeOutOfRange(if edge.src as usize >= nodes { edge.src } else { edge.dst }));
            }
            offsets[edge.src as usize + 1] += 1;
        }
        for index in 1..offsets.len() { offsets[index] += offsets[index - 1]; }
        let mut cursor = offsets[..nodes].to_vec();
        let mut targets = vec![0; sorted.len()];
        let mut relations = vec![0; sorted.len()];
        for edge in sorted {
            let index = cursor[edge.src as usize] as usize;
            targets[index] = edge.dst;
            relations[index] = edge.relation;
            cursor[edge.src as usize] += 1;
        }
        let graph = Self { offsets, targets, relations };
        graph.validate()?;
        Ok(graph)
    }

    pub fn node_count(&self) -> usize { self.offsets.len().saturating_sub(1) }
    pub fn edge_count(&self) -> usize { self.targets.len() }

    pub fn validate(&self) -> Result<(), GraphError> {
        if self.offsets.is_empty() || self.targets.len() != self.relations.len() {
            return Err(GraphError::MismatchedEdges);
        }
        let final_offset = *self.offsets.last().ok_or(GraphError::InvalidOffsets)?;
        if self.offsets[0] != 0
            || final_offset as usize != self.targets.len()
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(GraphError::InvalidOffsets);
        }
        if let Some(&bad) = self.targets.iter().find(|&&node| node as usize >= self.node_count()) {
            return Err(GraphError::NodeOutOfRange(bad));
        }
        Ok(())
    }

    pub fn neighbors(&self, node: NodeId) -> Result<impl Iterator<Item = (NodeId, RelationId)> + '_, GraphError> {
        if node as usize >= self.node_count() { return Err(GraphError::NodeOutOfRange(node)); }
        let start = self.offsets[node as usize] as usize;
        let end = self.offsets[node as usize + 1] as usize;
        Ok(self.targets[start..end].iter().copied().zip(self.relations[start..end].iter().copied()))
    }

    pub fn reverse(&self) -> Result<Self, GraphError> {
        let mut edges = Vec::with_capacity(self.edge_count());
        for src in 0..self.node_count() as u32 {
            for (dst, relation) in self.neighbors(src)? {
                edges.push(Edge { src: dst, relation, dst: src });
            }
        }
        Self::from_edges(self.node_count(), &edges)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredNode { pub node: NodeId, pub score: f32 }

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Instruction {
    Seed(NodeId), Ann { k: usize }, ExpandRel(RelationId), ExpandAny,
    FilterRelation(RelationId), Intersect(u8), Union(u8), Prune(usize),
    TopK(usize), Verify, Backtrack, Prefetch, Evict, Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Opcode {
    Seed, Ann, ExpandRel, ExpandAny, Filter, Intersect, Union, Prune, TopK,
    Verify, Backtrack, Prefetch, Evict, Stop,
}
impl Instruction {
    pub fn opcode(&self) -> Opcode {
        match self {
            Self::Seed(_) => Opcode::Seed, Self::Ann { .. } => Opcode::Ann,
            Self::ExpandRel(_) => Opcode::ExpandRel, Self::ExpandAny => Opcode::ExpandAny,
            Self::FilterRelation(_) => Opcode::Filter, Self::Intersect(_) => Opcode::Intersect,
            Self::Union(_) => Opcode::Union, Self::Prune(_) => Opcode::Prune,
            Self::TopK(_) => Opcode::TopK, Self::Verify => Opcode::Verify,
            Self::Backtrack => Opcode::Backtrack, Self::Prefetch => Opcode::Prefetch,
            Self::Evict => Opcode::Evict, Self::Stop => Opcode::Stop,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuerySpec { pub required_edges: Vec<Edge>, pub answers: BTreeSet<NodeId> }

/// The proof is evidence actually observed by the VM, never a hidden task
/// generator path. It may contain extra evidence, but every required edge and
/// returned answer must be grounded in the graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Proof { pub edges: Vec<Edge>, pub answer: Option<NodeId> }

pub fn validate_proof(graph: &CsrGraph, query: &QuerySpec, proof: &Proof) -> bool {
    let listed: BTreeSet<Edge> = proof.edges.iter().copied().collect();
    let answer = match proof.answer { Some(answer) if query.answers.contains(&answer) => answer, _ => return false };
    if !query.required_edges.iter().all(|edge| listed.contains(edge)) { return false; }
    if !proof.edges.iter().all(|edge| {
        graph.neighbors(edge.src)
            .map(|mut edges| edges.any(|(dst, relation)| dst == edge.dst && relation == edge.relation))
            .unwrap_or(false)
    }) { return false; }
    listed.iter().any(|edge| edge.dst == answer) || query.required_edges.is_empty()
}

/// Stable per-instruction accounting. `instructions` counts successful state
/// transitions only; failed instructions have no partial VM-state effects.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct VmStats {
    pub attempted_instructions: u64,
    pub instructions: u64,
    pub failed_instructions: u64,
    pub nodes_visited: u64,
    pub edges_examined: u64,
    pub ann_calls: u64,
    pub credits: u64,
    pub bytes_estimated: u64,
    pub max_depth: u32,
    pub backtracks: u64,
    pub prefetches: u64,
    pub evictions: u64,
    pub frontier_peak: u64,
    pub proof_edges: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VmStep {
    pub opcode: Opcode,
    pub frontier_before: usize,
    pub frontier_after: usize,
    pub answer: Option<NodeId>,
    pub proof_edges: usize,
    pub budget_remaining: u64,
    pub stats: VmStats,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum VmError { BudgetExhausted, InvalidRegister(u8), Stopped, MissingAnnProvider }
impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for VmError {}

pub trait AnnProvider: Send + Sync { fn search(&self, k: usize) -> Result<Vec<ScoredNode>, VmError>; }

#[derive(Clone)]
struct VmSnapshot {
    registers: [Vec<ScoredNode>; 4], answers: Vec<ScoredNode>, proof: Proof,
    budget: u64, depth: u32, stopped: bool, history: Vec<HistoryEntry>, stats: VmStats,
}

/// A backtrack boundary includes evidence as well as frontiers. Otherwise a
/// proof could accidentally retain edges from a branch the VM has abandoned.
#[derive(Clone)]
struct HistoryEntry {
    registers: [Vec<ScoredNode>; 4],
    answers: Vec<ScoredNode>,
    proof: Proof,
    depth: u32,
}

pub struct SearchVm<'a> {
    pub graph: &'a CsrGraph,
    /// F0..F3. Current ISA always writes F0; register operations expose F1..F3
    /// for native callers and future explicit register-placement extensions.
    pub registers: [Vec<ScoredNode>; 4],
    pub answers: Vec<ScoredNode>,
    pub proof: Proof,
    pub stats: VmStats,
    pub budget: u64,
    pub depth: u32,
    stopped: bool,
    ann: Option<&'a dyn AnnProvider>,
    history: Vec<HistoryEntry>,
}

impl<'a> SearchVm<'a> {
    pub fn new(graph: &'a CsrGraph, budget: u64) -> Self {
        Self {
            graph, registers: std::array::from_fn(|_| Vec::new()), answers: Vec::new(),
            proof: Proof { edges: Vec::new(), answer: None }, stats: VmStats::default(),
            budget, depth: 0, stopped: false, ann: None, history: Vec::new(),
        }
    }
    pub fn with_ann(mut self, ann: &'a dyn AnnProvider) -> Self { self.ann = Some(ann); self }
    pub fn is_stopped(&self) -> bool { self.stopped }
    fn snapshot(&self) -> VmSnapshot {
        VmSnapshot { registers: self.registers.clone(), answers: self.answers.clone(), proof: self.proof.clone(), budget: self.budget, depth: self.depth, stopped: self.stopped, history: self.history.clone(), stats: self.stats.clone() }
    }
    fn restore(&mut self, snapshot: VmSnapshot) {
        self.registers = snapshot.registers; self.answers = snapshot.answers; self.proof = snapshot.proof;
        self.budget = snapshot.budget; self.depth = snapshot.depth; self.stopped = snapshot.stopped;
        self.history = snapshot.history; self.stats = snapshot.stats;
    }
    fn charge(&mut self, amount: u64) -> Result<(), VmError> {
        if self.budget < amount { return Err(VmError::BudgetExhausted); }
        self.budget -= amount;
        self.stats.credits += amount;
        Ok(())
    }
    fn current_mut(&mut self) -> &mut Vec<ScoredNode> { &mut self.registers[0] }
    fn push_history(&mut self) {
        self.history.push(HistoryEntry {
            registers: self.registers.clone(), answers: self.answers.clone(),
            proof: self.proof.clone(), depth: self.depth,
        });
    }
    fn normalized(nodes: Vec<ScoredNode>) -> Vec<ScoredNode> {
        let mut best = BTreeMap::<NodeId, f32>::new();
        for node in nodes { best.entry(node.node).and_modify(|score| *score = score.max(node.score)).or_insert(node.score); }
        best.into_iter().map(|(node, score)| ScoredNode { node, score }).collect()
    }
    fn update_peaks(&mut self) {
        self.stats.frontier_peak = self.stats.frontier_peak.max(self.registers.iter().map(Vec::len).max().unwrap_or(0) as u64);
        self.stats.proof_edges = self.proof.edges.len() as u64;
    }

    fn execute_inner(&mut self, instruction: &Instruction) -> Result<(), VmError> {
        match instruction {
            Instruction::Seed(node) => {
                if *node as usize >= self.graph.node_count() { return Err(VmError::InvalidRegister(0)); }
                self.charge(1)?;
                self.push_history();
                self.registers[0] = vec![ScoredNode { node: *node, score: 1.0 }];
            }
            Instruction::Ann { k } => {
                let candidates = self.ann.ok_or(VmError::MissingAnnProvider)?.search(*k)?;
                self.charge(*k as u64)?;
                self.push_history();
                self.registers[0] = Self::normalized(candidates);
                self.stats.ann_calls += 1;
            }
            Instruction::ExpandAny | Instruction::ExpandRel(_) => {
                let wanted = match instruction { Instruction::ExpandRel(relation) => Some(*relation), _ => None };
                let old = self.registers[0].clone();
                let mut next = Vec::new();
                let mut observed_edges = Vec::new();
                let mut examined = 0u64;
                for scored in old {
                    let edges = self.graph.neighbors(scored.node).map_err(|_| VmError::InvalidRegister(0))?;
                    for (dst, relation) in edges {
                        examined += 1;
                        if wanted.is_none_or(|required| required == relation) {
                            next.push(ScoredNode { node: dst, score: scored.score });
                            observed_edges.push(Edge { src: scored.node, relation, dst });
                        }
                    }
                }
                self.charge(next.len() as u64)?;
                self.push_history();
                self.stats.edges_examined += examined;
                self.stats.nodes_visited += next.len() as u64;
                self.stats.bytes_estimated += examined * (std::mem::size_of::<NodeId>() + std::mem::size_of::<RelationId>()) as u64;
                self.depth += 1;
                self.stats.max_depth = self.stats.max_depth.max(self.depth);
                self.proof.edges.extend(observed_edges);
                self.registers[0] = Self::normalized(next);
            }
            Instruction::FilterRelation(relation) => {
                let old = self.registers[0].clone();
                let keep: HashSet<NodeId> = old.iter().filter_map(|scored| {
                    self.graph.neighbors(scored.node).ok().and_then(|mut edges| edges.any(|(_, found)| found == *relation).then_some(scored.node))
                }).collect();
                self.charge(old.len() as u64)?;
                self.current_mut().retain(|scored| keep.contains(&scored.node));
            }
            Instruction::Intersect(register) | Instruction::Union(register) => {
                if *register > 3 { return Err(VmError::InvalidRegister(*register)); }
                let current = self.registers[0].clone();
                let other = self.registers[*register as usize].clone();
                self.charge((current.len() + other.len()) as u64)?;
                if matches!(instruction, Instruction::Intersect(_)) {
                    let set: HashSet<_> = other.iter().map(|node| node.node).collect();
                    self.registers[0] = current.into_iter().filter(|node| set.contains(&node.node)).collect();
                } else {
                    let mut union = current;
                    union.extend(other);
                    self.registers[0] = Self::normalized(union);
                }
            }
            Instruction::Prune(k) | Instruction::TopK(k) => {
                let charge = self.registers[0].len() as u64;
                self.charge(charge)?;
                self.current_mut().sort_by(|left, right| right.score.total_cmp(&left.score).then_with(|| left.node.cmp(&right.node)));
                self.current_mut().truncate(*k);
            }
            Instruction::Verify => {
                self.charge(1)?;
                self.answers = self.registers[0].clone();
                self.proof.answer = self.answers.first().map(|node| node.node);
            }
            Instruction::Backtrack => {
                self.charge(1)?;
                if let Some(previous) = self.history.pop() {
                    self.registers = previous.registers;
                    self.answers = previous.answers;
                    self.proof = previous.proof;
                    self.depth = previous.depth;
                } else {
                    self.registers = std::array::from_fn(|_| Vec::new());
                    self.answers.clear();
                    self.proof = Proof { edges: Vec::new(), answer: None };
                    self.depth = 0;
                }
                self.stats.backtracks += 1;
            }
            Instruction::Prefetch => { self.charge(1)?; self.stats.prefetches += 1; }
            Instruction::Evict => { self.charge(1)?; self.stats.evictions += 1; }
            Instruction::Stop => { self.charge(1)?; self.stopped = true; }
        }
        self.update_peaks();
        Ok(())
    }

    /// Execute one instruction atomically. On an error, all state other than
    /// failed-attempt observability is restored, so a scheduler can recover
    /// deterministically without a partly expanded frontier.
    pub fn execute_traced(&mut self, instruction: Instruction) -> Result<VmStep, VmError> {
        let before = self.registers[0].len();
        let opcode = instruction.opcode();
        if self.stopped {
            self.stats.attempted_instructions += 1;
            self.stats.failed_instructions += 1;
            return Err(VmError::Stopped);
        }
        let snapshot = self.snapshot();
        self.stats.attempted_instructions += 1;
        match self.execute_inner(&instruction) {
            Ok(()) => {
                self.stats.instructions += 1;
                Ok(VmStep { opcode, frontier_before: before, frontier_after: self.registers[0].len(), answer: self.proof.answer, proof_edges: self.proof.edges.len(), budget_remaining: self.budget, stats: self.stats.clone() })
            }
            Err(error) => {
                self.restore(snapshot);
                self.stats.attempted_instructions += 1;
                self.stats.failed_instructions += 1;
                Err(error)
            }
        }
    }

    pub fn execute(&mut self, instruction: Instruction) -> Result<(), VmError> { self.execute_traced(instruction).map(|_| ()) }
    pub fn run(&mut self, program: impl IntoIterator<Item = Instruction>) -> Result<(), VmError> {
        for instruction in program {
            self.execute(instruction)?;
            if self.stopped { break; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> CsrGraph {
        CsrGraph::from_edges(5, &[
            Edge { src: 0, relation: 1, dst: 1 }, Edge { src: 1, relation: 2, dst: 2 },
            Edge { src: 1, relation: 1, dst: 3 }, Edge { src: 3, relation: 2, dst: 4 },
        ]).unwrap()
    }

    #[test]
    fn csr_reverse_and_validation() {
        let graph = graph();
        assert_eq!(graph.edge_count(), 4);
        assert_eq!(graph.reverse().unwrap().neighbors(1).unwrap().count(), 1);
        assert!(CsrGraph { offsets: vec![1], targets: vec![], relations: vec![] }.validate().is_err());
    }

    #[test]
    fn vm_expands_and_proves() {
        let graph = graph();
        let mut vm = SearchVm::new(&graph, 100);
        vm.run([Instruction::Seed(0), Instruction::ExpandRel(1), Instruction::ExpandRel(2), Instruction::Verify, Instruction::Stop]).unwrap();
        assert_eq!(vm.proof.answer, Some(2));
        assert_eq!(vm.stats.instructions, 5);
        assert_eq!(vm.stats.edges_examined, 3);
        assert!(validate_proof(&graph, &QuerySpec { required_edges: vec![Edge { src: 0, relation: 1, dst: 1 }, Edge { src: 1, relation: 2, dst: 2 }], answers: BTreeSet::from([2]) }, &vm.proof));
    }

    #[test]
    fn failed_expansion_is_atomic_and_accounted() {
        let graph = graph();
        let mut vm = SearchVm::new(&graph, 1);
        vm.execute(Instruction::Seed(0)).unwrap();
        let before = vm.registers.clone();
        assert_eq!(vm.execute(Instruction::ExpandAny), Err(VmError::BudgetExhausted));
        assert_eq!(vm.registers, before);
        assert_eq!(vm.proof.edges.len(), 0);
        assert_eq!(vm.stats.instructions, 1);
        assert_eq!(vm.stats.failed_instructions, 1);
        assert_eq!(vm.stats.credits, 1);
    }

    #[test]
    fn backtrack_restores_frontier_and_records_trace() {
        let graph = graph();
        let mut vm = SearchVm::new(&graph, 20);
        vm.run([Instruction::Seed(0), Instruction::ExpandAny]).unwrap();
        let step = vm.execute_traced(Instruction::Backtrack).unwrap();
        assert_eq!(vm.registers[0][0].node, 0);
        assert_eq!(step.opcode, Opcode::Backtrack);
        assert_eq!(step.frontier_before, 1);
        assert_eq!(step.frontier_after, 1);
        assert_eq!(step.stats.backtracks, 1);
    }

    #[test]
    fn backtrack_discards_abandoned_proof_evidence() {
        let graph = graph();
        let mut vm = SearchVm::new(&graph, 20);
        vm.run([Instruction::Seed(0), Instruction::ExpandAny, Instruction::Backtrack]).unwrap();
        assert!(vm.proof.edges.is_empty());
        assert_eq!(vm.proof.answer, None);
        assert_eq!(vm.depth, 0);
    }

    #[test]
    fn proof_rejects_generator_only_or_ungrounded_edges() {
        let graph = graph();
        let query = QuerySpec { required_edges: vec![Edge { src: 0, relation: 1, dst: 1 }], answers: BTreeSet::from([1]) };
        assert!(!validate_proof(&graph, &query, &Proof { edges: vec![Edge { src: 0, relation: 9, dst: 1 }], answer: Some(1) }));
        assert!(!validate_proof(&graph, &query, &Proof { edges: vec![], answer: Some(1) }));
    }
}
