#!/usr/bin/env python3
"""Cinematic, read-only console for real NEUROSEEK learned-policy searches.

The process owns neither a trainer socket nor a CUDA context.  It loads one
immutable checkpoint on CPU and explores the immutable mmap graph only after
the operator requests a task.  Keyboard input controls this viewer alone.
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import QuerySpec, load_task_jsonl
from neuroseek.models.policy import NavigatorPolicy, OP_NAMES
from neuroseek.search.environment import GraphSearchEnv, SearchResult
from neuroseek.training.checkpoint import load_checkpoint


DEFAULT_CHECKPOINT = Path("runs/presentation-stabilized-20260812T1520EDT/checkpoints/latest.ckpt")
DEFAULT_TASKS = Path("data/processed/task_splits/validation_v2.jsonl")


@dataclass
class Step:
    operator: str
    probability: float | None
    value: float | None
    trace: str
    frontier: list[int]
    credits: int
    edges: int


@dataclass
class SearchRun:
    query: QuerySpec
    steps: list[Step]
    result: SearchResult
    elapsed_ms: float
    answer: int | None
    proof_path: tuple[int, ...]


@dataclass
class WordCandidate:
    identifier: str
    label: str
    description: str
    local_id: int | None


@dataclass
class WordSearch:
    term: str
    candidates: list[WordCandidate]
    selected: int | None = None
    relation_term: str = ""
    relation: WordCandidate | None = None
    neighbors: list[tuple[int, int]] | None = None
    elapsed_ms: float = 0.0


def wikidata_candidates(term: str, language: str, kind: str) -> list[dict[str, str]]:
    """Resolve a human word to public Wikidata IDs; graph traversal stays local."""
    params = urllib.parse.urlencode({"action": "wbsearchentities", "search": term,
                                     "language": language, "format": "json", "limit": 6,
                                     "type": "property" if kind == "property" else "item"})
    request = urllib.request.Request(
        f"https://www.wikidata.org/w/api.php?{params}",
        headers={"User-Agent": "NEUROSEEK-Jetson-demo/0.1 (local read-only knowledge graph viewer)"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [{"id": str(row["id"]), "label": str(row.get("label", row["id"])),
             "description": str(row.get("description", ""))} for row in payload.get("search", [])]


def wikidata_labels(identifiers: list[str], language: str) -> dict[str, str]:
    """Fetch display labels in one bounded request; IDs stay the local truth."""
    unique = list(dict.fromkeys(identifiers))[:24]
    if not unique:
        return {}
    params = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(unique),
                                     "props": "labels", "languages": f"{language}|en", "format": "json"})
    request = urllib.request.Request(
        f"https://www.wikidata.org/w/api.php?{params}",
        headers={"User-Agent": "NEUROSEEK-Jetson-demo/0.1 (local read-only knowledge graph viewer)"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result: dict[str, str] = {}
    for identifier, entity in payload.get("entities", {}).items():
        labels = entity.get("labels", {})
        chosen = labels.get(language) or labels.get("en")
        if isinstance(chosen, dict) and isinstance(chosen.get("value"), str):
            result[str(identifier)] = chosen["value"]
    return result


def cells(value: str) -> int:
    """Terminal-cell width for Japanese tab labels; curses uses cells, not len."""
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 1 for char in value)


def clip_cells(value: str, maximum: int) -> str:
    """Clip by terminal cells so CJK copy never wraps into the footer."""
    if cells(value) <= maximum:
        return value
    if value and len(set(value)) == 1:
        return value[0] * maximum
    result: list[str] = []
    used = 0
    content_limit = max(0, maximum - 1)
    for char in value:
        size = 2 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 1
        if used + size > content_limit:
            return "".join(result) + ("…" if maximum else "")
        result.append(char)
        used += size
    return "".join(result)


class Console:
    def __init__(self, screen: Any, graph: GraphMmap, model: NavigatorPolicy, tasks: list[tuple[QuerySpec, tuple[int, ...] | None]], checkpoint: Path, language: str):
        self.screen, self.graph, self.model, self.tasks, self.checkpoint = screen, graph, model, tasks, checkpoint
        self.language = language
        self.task_index = 0
        self.tab = 1
        self.run: SearchRun | None = None
        self.word: WordSearch | None = None
        self.label_cache: dict[str, str] = {}
        self.command = ""
        self.command_mode = False
        self.notice = self.text("Press f to search a word. Press 2 for the learned-model demo.", "fで単語を検索します。学習済みモデルのデモは2番です。")
        self._colors()

    def text(self, english: str, japanese: str) -> str:
        return japanese if self.language == "ja" else english

    def _colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        for pair, fg in ((1, curses.COLOR_CYAN), (2, curses.COLOR_GREEN), (3, curses.COLOR_MAGENTA), (4, curses.COLOR_YELLOW), (5, curses.COLOR_RED), (6, curses.COLOR_WHITE)):
            curses.init_pair(pair, fg, -1)

    def write(self, row: int, column: int, value: str, pair: int = 0, bold: bool = False) -> None:
        height, width = self.screen.getmaxyx()
        if not 0 <= row < height or column >= width:
            return
        try:
            style = curses.color_pair(pair) | (curses.A_BOLD if bold else 0)
            self.screen.addstr(row, max(0, column), clip_cells(value, max(0, width - column - 1)), style)
        except curses.error:
            pass

    def entity(self, identifier: int) -> str:
        wikidata_id = self.graph.entity_identifier(identifier)
        return f"{self.label_cache.get(wikidata_id, self.graph.entity_label(identifier))}  [{wikidata_id}]"

    def relation(self, identifier: int) -> str:
        wikidata_id = self.graph.relation_identifier(identifier)
        return f"{self.label_cache.get(wikidata_id, self.graph.relation_label(identifier))}  [{wikidata_id}]"

    def word_find(self, term: str) -> None:
        started = time.perf_counter()
        direct = term.strip().upper()
        if direct.startswith("Q") and direct[1:].isdigit():
            candidates = [WordCandidate(direct, direct, self.text("direct local Q-ID", "ローカルQ-ID直接指定"), self.graph.find_entity_identifier(direct))]
        else:
            remote = wikidata_candidates(term, self.language, "item")
            candidates = [WordCandidate(row["id"], row["label"], row["description"], self.graph.find_entity_identifier(row["id"])) for row in remote]
        self.word = WordSearch(term, candidates, elapsed_ms=(time.perf_counter() - started) * 1000.0)
        usable = next((index for index, candidate in enumerate(candidates) if candidate.local_id is not None), None)
        if usable is None:
            self.notice = self.text("Name resolved, but none of its candidates are in the local Wikidata5M graph.", "名称は解決できましたが、候補はローカルWikidata5Mグラフにありません。")
        else:
            self.word.selected = usable
            self.expand_word()
            self.notice = self.text("Word resolved. The first local graph candidate is selected; use :use N to change it.", "語を解決しました。最初のローカル候補を選択中です。:use Nで変更できます。")

    def select_word(self, index: int) -> None:
        if self.word is None or not 1 <= index <= len(self.word.candidates):
            raise ValueError("candidate index is unavailable")
        if self.word.candidates[index - 1].local_id is None:
            raise ValueError("candidate is not present in local graph")
        self.word.selected = index - 1
        self.expand_word()

    def word_relation(self, term: str) -> None:
        if self.word is None or self.word.selected is None:
            raise ValueError("find a local graph entity first")
        direct = term.strip().upper()
        remote = ([{"id": direct, "label": direct, "description": self.text("direct local P-ID", "ローカルP-ID直接指定")}]
                  if direct.startswith("P") and direct[1:].isdigit()
                  else wikidata_candidates(term, self.language, "property"))
        for row in remote:
            local_id = self.graph.find_relation_identifier(row["id"])
            if local_id is not None:
                self.word.relation_term = term
                self.word.relation = WordCandidate(row["id"], row["label"], row["description"], local_id)
                self.expand_word()
                self.notice = self.text("Relation filter applied to real local graph edges.", "実際のローカルグラフエッジに関係フィルタを適用しました。")
                return
        raise ValueError("resolved relation is absent from local graph")

    def expand_word(self) -> None:
        assert self.word is not None and self.word.selected is not None
        entity = self.word.candidates[self.word.selected]
        assert entity.local_id is not None
        nodes, relations = self.graph.neighbors(entity.local_id)
        edges = [(int(node), int(relation)) for node, relation in zip(nodes, relations)]
        if self.word.relation is not None:
            edges = [(node, relation) for node, relation in edges if relation == self.word.relation.local_id]
        self.word.neighbors = edges[:12]
        identifiers = [self.graph.entity_identifier(node) for node, _relation in self.word.neighbors]
        identifiers.extend(self.graph.relation_identifier(relation) for _node, relation in self.word.neighbors)
        try:
            self.label_cache.update(wikidata_labels(identifiers, self.language))
        except (OSError, urllib.error.URLError, ValueError):
            # Names are a display enhancement. Local Q/P IDs and graph edges
            # remain fully usable if the optional public lookup is unavailable.
            pass

    def local_word_candidates(self) -> list[tuple[int, WordCandidate]]:
        if self.word is None:
            return []
        return [(index, candidate) for index, candidate in enumerate(self.word.candidates) if candidate.local_id is not None]

    def execute(self) -> None:
        query, _ = self.tasks[self.task_index]
        env = GraphSearchEnv(self.graph, query, (), cuda_session=None)
        steps: list[Step] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(12):
                observation = torch.as_tensor(env.observation(), dtype=torch.float32).unsqueeze(0)
                logits, value = self.model(observation)
                probabilities = torch.softmax(logits[0], dim=0)
                action = int(torch.argmax(probabilities).item())
                result = env.step(action)
                steps.append(Step(OP_NAMES[action], float(probabilities[action]), float(value[0]), result.trace[-1], sorted(env.frontier)[:6], result.credits, result.edges_examined))
                if result.done:
                    break
            if not env.done:
                result = env.step(OP_NAMES.index("STOP"))
                steps.append(Step("STOP", None, None, result.trace[-1], sorted(env.frontier)[:6], result.credits, result.edges_examined))
        self.run = SearchRun(query, steps, result, (time.perf_counter() - started) * 1000.0, env.answer, env.proof_path)
        self.notice = self.text("Real policy execution complete — see proof status below.", "実方策の探索が完了しました — 下部の証明状態を確認してください。")

    def header(self) -> int:
        height, width = self.screen.getmaxyx()
        self.write(0, 1, "NEUROSEEK", 1, True)
        self.write(0, 13, self.text("LIVE MODEL SEARCH", "ライブモデル検索"), 6, True)
        self.write(1, 1, "●", 2, True)
        self.write(1, 3, self.text("CPU-ONLY · READ-ONLY · TRAINER ISOLATED", "CPU専用 · 読み取り専用 · 学習器から分離"), 6)
        self.write(1, max(1, width - 38), f"TASK {self.task_index + 1}/{len(self.tasks)}", 4)
        self.write(2, 1, "─" * max(1, width - 2), 1)
        tabs = [(1, self.text("WORDS", "単語検索")), (2, self.text("MODEL", "モデル")), (3, self.text("PATH", "経路")), (4, self.text("PROOF", "証明")), (5, self.text("SYSTEM", "システム"))]
        column = 2
        for number, name in tabs:
            active = number == self.tab
            self.write(3, column, f"[{number}] {name}", 1 if active else 6, active)
            column += cells(name) + 7
        controls = self.text("r model run · n next · : command · q quit", "r モデル実行 · n 次問 · : コマンド · q 終了")
        if width - column > cells(controls) + 3:
            self.write(3, width - cells(controls) - 2, controls, 6)
        elif width - column > 10:
            self.write(3, column + 1, ":help", 6)
        self.write(4, 1, "─" * max(1, width - 2), 1)
        return 6

    def page_words(self, row: int) -> None:
        height, _width = self.screen.getmaxyx()
        compact = height < 32
        self.write(row, 2, self.text("SEARCH THE KNOWLEDGE GRAPH", "知識グラフを検索"), 1, True)
        self.write(row + 1, 2, self.text("1 Find a word   →   2 Pick an available result   →   3 Explore its real facts", "1 語を探す   →   2 探索できる候補を選ぶ   →   3 実際の事実をたどる"), 6)
        if self.word is None:
            self.write(row + 4, 4, self.text("START A SEARCH", "検索をはじめる"), 4, True)
            self.write(row + 6, 4, self.text("Press  f  and type a word", "f を押して、調べたい語を入力"), 6)
            self.write(row + 8, 7, self.text(":find Japan", ":find 日本"), 3, True)
            self.write(row + 11, 4, self.text("You can also use English, Japanese, or a Wikidata ID such as Q17.", "日本語・英語・Q17のようなWikidata IDを使えます。"), 6)
            self.write(row + 13, 4, self.text("No model is run in this page: it is a direct read-only walk of real graph facts.", "このページではモデルを実行せず、実グラフの事実を読み取り専用で直接たどります。"), 6)
            return
        self.write(row + 3, 2, f"{self.text('SEARCHED FOR', '検索語')}  {self.word.term}", 3, True)
        available = self.local_word_candidates()
        skipped = len(self.word.candidates) - len(available)
        self.write(row + 5, 2, self.text("CHOOSE A RESULT READY TO EXPLORE", "探索できる候補を選ぶ"), 4, True)
        if not available:
            self.write(row + 7, 4, self.text("This word exists in Wikidata, but its results were not downloaded into this demo graph.", "この語はWikidataにありますが、該当候補はこのデモ用グラフに収録されていません。"), 5)
            self.write(row + 9, 4, self.text("Try a different candidate, or enter a known Q-ID directly.", "別の語を試すか、既知のQ-IDを直接入力してください。"), 6)
            return
        candidate_limit = 2 if compact else 4
        for display_index, (index, candidate) in enumerate(available[:candidate_limit]):
            selected = index == self.word.selected
            marker = "▶" if selected else "·"
            candidate_row = row + 6 + display_index * (1 if compact else 2)
            self.write(candidate_row, 4, f"{marker} [{index + 1}] {candidate.label} [{candidate.identifier}]", 2 if selected else 6, selected)
            if not compact:
                self.write(candidate_row + 1, 8, candidate.description or self.text("Ready to explore in this graph", "このグラフで探索できます"), 6)
        candidate_bottom = row + 6 + min(len(available), candidate_limit) * (1 if compact else 2)
        more_ready = len(available) - min(len(available), candidate_limit)
        if more_ready:
            self.write(candidate_bottom, 4, self.text(f"{more_ready} more results are ready to explore — choose them with :use N.", f"ほかに {more_ready} 件の探索可能な候補があります。:use Nで選択できます。"), 6)
            candidate_bottom += 1
        if skipped and not compact:
            self.write(candidate_bottom, 4, self.text(f"{skipped} name matches are known by Wikidata but were not downloaded into this demo graph.", f"{skipped} 件はWikidataにありますが、このデモ用グラフには収録されていません。"), 6)
        if self.word.selected is None:
            return
        selected = self.word.candidates[self.word.selected]
        after = max(row + (10 if compact else 15), candidate_bottom + 1)
        relation = self.word.relation.label if self.word.relation else self.text("all facts", "すべての事実")
        self.write(after, 2, f"{self.text('REAL FACTS ABOUT', '実グラフ上の事実')}  {selected.label}  ·  {relation}", 1, True)
        if not compact:
            self.write(after + 1, 4, self.text("To narrow this list, press : and type :rel capital", "絞り込むには : を押して :rel 首都 と入力"), 6)
        edges = self.word.neighbors or []
        edge_start = after + (2 if compact else 3)
        edge_limit = max(1, min(7, height - 4 - edge_start))
        if not edges:
            self.write(edge_start, 4, self.text("No matching facts in the local graph.", "ローカルグラフに一致する事実はありません。"), 5)
        for index, (node, relation_id) in enumerate(edges[:edge_limit]):
            self.write(edge_start + index, 5, f"{index + 1:02}", 4)
            self.write(edge_start + index, 10, self.relation(relation_id), 4)
            self.write(edge_start + index, 38, "→", 1, True)
            self.write(edge_start + index, 42, self.entity(node), 3)
        status_row = edge_start + edge_limit + 1
        if status_row < height - 3:
            self.write(status_row, 2, f"{self.text('SAFE MODE', '安全モード')}  CPU · {self.text('GRAPH DATA', 'グラフデータ')} read-only · CUDA off · {self.text('NAME LOOKUP', '名称解決')} {self.word.elapsed_ms:.1f} ms", 6)

    def page_model(self, row: int) -> None:
        query, _ = self.tasks[self.task_index]
        self.write(row, 2, self.text("QUERY DEFINITION", "クエリ定義"), 1, True)
        self.write(row + 2, 2, self.text("SOURCE", "始点"), 6)
        self.write(row + 2, 16, self.entity(query.source), 3, True)
        self.write(row + 4, 2, self.text("RELATION PROGRAM", "関係プログラム"), 6)
        for index, relation in enumerate(query.relations):
            self.write(row + 5 + index, 6, f"{index + 1:02}  ── {self.relation(relation)} ──▶", 1 if index % 2 == 0 else 3)
        after = row + 7 + len(query.relations)
        if self.run is None:
            self.write(after, 2, self.text("The reference answer remains hidden from the model. Press r to execute the learned policy.", "正解は方策に与えられません。rで学習済み方策を実行します。"), 4)
            return
        result = self.run.result
        state = "VALID PROOF" if result.valid_proof else ("ANSWER UNVERIFIED" if result.answer_correct else "NO ANSWER")
        self.write(after, 2, state, 2 if result.valid_proof else 5, True)
        answer = self.entity(self.run.answer) if self.run.answer is not None else self.text("none", "なし")
        self.write(after + 1, 2, f"{self.text('ANSWER', '回答')}  {answer}", 6)
        self.write(after + 2, 2, f"{self.text('LATENCY', 'レイテンシ')}  {self.run.elapsed_ms:.2f} ms CPU policy+search  ·  {self.text('CREDITS', 'クレジット')} {result.credits}  ·  {self.text('EDGES', 'エッジ')} {result.edges_examined}", 6)

    def page_path(self, row: int) -> None:
        self.write(row, 2, self.text("LEARNED OPERATOR LATTICE", "学習済み演算子ラティス"), 1, True)
        self.write(row + 1, 2, self.text("This is the operation sequence actually selected by the loaded checkpoint.", "これはロード済みチェックポイントが実際に選択した演算子列です。"), 6)
        if self.run is None:
            self.write(row + 4, 2, self.text("No execution yet. Press r.", "まだ実行されていません。rを押してください。"), 4)
            return
        lanes = (4, 28, 52, 76)
        previous = 0
        for index, step in enumerate(self.run.steps):
            lane = sum(step.operator.encode()) % len(lanes)
            line = row + 3 + index * 2
            if index:
                low, high = sorted((lanes[previous], lanes[lane]))
                bridge = "│" if lane == previous else ("╲" if lane > previous else "╱")
                self.write(line - 1, low, bridge + "─" * max(0, high - low - 1), 1)
            confidence = "" if step.probability is None else f"  p={step.probability:.3f}"
            self.write(line, lanes[lane], f"{'◉' if index + 1 == len(self.run.steps) else '●'} {index + 1:02} {step.operator:<11}{confidence}", 3 if step.operator in ('SEED', 'EXPAND_REL') else 1, True)
            self.write(line, min(lanes[lane] + 31, self.screen.getmaxyx()[1] - 12), step.trace, 6)
            previous = lane

    def page_proof(self, row: int) -> None:
        self.write(row, 2, self.text("INDEPENDENT PROOF CHECK", "独立証明チェック"), 1, True)
        if self.run is None:
            self.write(row + 3, 2, self.text("A proof can only appear after a real policy execution.", "証明は実方策を実行した後だけ表示されます。"), 4)
            return
        result = self.run.result
        self.write(row + 2, 2, self.text("VALID", "有効") if result.valid_proof else self.text("NOT VALID", "無効"), 2 if result.valid_proof else 5, True)
        self.write(row + 2, 18, self.text("Graph evidence reconstructed independently from the executed frontier.", "実行済みフロンティアからグラフ証拠を独立再構築しています。"), 6)
        if not self.run.result.valid_proof:
            self.write(row + 5, 2, self.text("No valid proof path was produced. This outcome is shown as-is.", "有効な証明経路は生成されませんでした。この結果をそのまま表示しています。"), 5)
            return
        for index, entity in enumerate(self.run.proof_path):
            self.write(row + 5 + index * 2, 6, f"{index + 1:02}  {self.entity(entity)}", 3 if index else 1, True)
            if index + 1 < len(self.run.proof_path):
                relation = self.run.query.relations[index]
                self.write(row + 6 + index * 2, 11, f"└── {self.relation(relation)}", 6)
        sample_row = row + 7 + len(self.run.proof_path) * 2
        self.write(sample_row, 2, self.text("LAST FRONTIER SAMPLE", "最終フロンティアの例"), 4, True)
        for index, entity in enumerate(self.run.steps[-1].frontier[:4]):
            self.write(sample_row + 1 + index, 6, self.entity(entity), 6)

    def page_system(self, row: int) -> None:
        self.write(row, 2, self.text("ISOLATION / PROVENANCE", "隔離 / 来歴"), 1, True)
        rows = [
            (self.text("MODEL", "モデル"), "NavigatorPolicy (CPU inference)"),
            (self.text("CHECKPOINT", "チェックポイント"), str(self.checkpoint)),
            (self.text("CHECKPOINT STEP", "チェックポイントステップ"), str(self.model_step)),
            (self.text("GRAPH", "グラフ"), "data/processed (mmap, read-only)"),
            (self.text("CUDA", "CUDA"), self.text("disabled: no context created", "無効: コンテキストを作成しません")),
            (self.text("TRAINER", "学習器"), self.text("no socket, no signal, no writes", "ソケットなし・シグナルなし・書込みなし")),
        ]
        for index, (name, value) in enumerate(rows):
            self.write(row + 2 + index * 2, 3, name, 4, True)
            self.write(row + 2 + index * 2, 24, value, 6)

    def draw(self) -> None:
        self.screen.erase()
        row = self.header()
        if self.tab == 1:
            self.page_words(row)
        elif self.tab == 2:
            self.page_model(row)
        elif self.tab == 3:
            self.page_path(row)
        elif self.tab == 4:
            self.page_proof(row)
        else:
            self.page_system(row)
        height, width = self.screen.getmaxyx()
        self.write(height - 3, 1, "─" * max(1, width - 2), 1)
        prompt = self.command if self.command_mode else self.text("f search a word · :help commands", "fで単語を検索 · :helpでコマンド一覧")
        self.write(height - 2, 2, "›", 1, True)
        self.write(height - 2, 4, prompt, 6 if not self.command_mode else 3)
        self.write(height - 1, 2, self.notice, 6)
        self.screen.refresh()

    @property
    def model_step(self) -> int:
        return int(getattr(self.model, "_neuroseek_step", 0))

    def apply(self, command: str) -> bool:
        words = command.strip().lstrip(":/").split()
        if not words:
            return False
        try:
            if words[0] in {"q", "quit", "exit"}:
                return True
            if words[0] in {"find", "search"} and len(words) > 1:
                self.word_find(" ".join(words[1:]))
                self.tab = 1
            elif words[0] in {"use", "select"} and len(words) == 2:
                self.select_word(int(words[1]))
                self.notice = self.text("Local graph candidate selected.", "ローカルグラフ候補を選択しました。")
            elif words[0] in {"rel", "relation"} and len(words) > 1:
                self.word_relation(" ".join(words[1:]))
                self.tab = 1
            elif words[0] in {"clear", "reset"}:
                self.word = None
                self.tab = 1
                self.notice = self.text("Word-search workspace cleared.", "単語検索ワークスペースを消去しました。")
            elif words[0] in {"r", "run"}:
                if len(words) > 1:
                    self.task_index = int(words[1]) % len(self.tasks)
                self.execute()
            elif words[0] in {"n", "next"}:
                self.task_index = (self.task_index + 1) % len(self.tasks)
                self.run = None
                self.notice = self.text("Loaded next immutable validation task.", "次の不変検証タスクを読み込みました。")
            elif words[0] in {"lang", "language"} and len(words) > 1 and words[1] in {"ja", "en"}:
                self.language = words[1]
            elif words[0] in {"help", "?"}:
                self.notice = ":find WORD · :use N · :rel WORD · r/run [index] · n/next · 1-5 tabs · l · :quit"
            else:
                self.notice = self.text("Unknown viewer command. Use :help.", "不明なビューアコマンドです。:helpを使用してください。")
        except (OSError, ValueError, urllib.error.URLError) as error:
            self.notice = self.text(f"Search could not run: {error}", f"検索を実行できませんでした: {error}")
        return False

    def loop(self) -> None:
        self.screen.nodelay(False)
        self.screen.keypad(True)
        while True:
            self.draw()
            key = self.screen.get_wch()
            if self.command_mode:
                if key in ("\n", "\r"):
                    self.command_mode = False
                    if self.apply(self.command):
                        return
                    self.command = ""
                elif key in ("\x1b",):
                    self.command_mode, self.command = False, ""
                elif key in ("\x7f", "\b"):
                    self.command = self.command[:-1]
                elif isinstance(key, str) and key.isprintable():
                    self.command += key
                continue
            if key in ("q", "Q", "\x03"):
                return
            if key == ":":
                self.command_mode, self.command = True, ":"
            elif key in ("f", "F"):
                self.command_mode, self.command = True, ":find "
            elif key in ("r", "R"):
                self.execute()
            elif key in ("n", "N"):
                self.apply("next")
            elif key in ("l", "L"):
                self.language = "ja" if self.language == "en" else "en"
            elif isinstance(key, str) and key in "12345":
                self.tab = int(key)


def run(screen: Any, graph: GraphMmap, model: NavigatorPolicy, tasks: list[tuple[QuerySpec, tuple[int, ...] | None]], checkpoint: Path, language: str) -> None:
    Console(screen, graph, model, tasks, checkpoint, language).loop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("en", "ja"), default="en")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.tasks.is_file():
        raise SystemExit("required immutable checkpoint or task artifact is absent")
    device = torch.device("cpu")
    state = load_checkpoint(args.checkpoint, device)
    model = NavigatorPolicy().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    model._neuroseek_step = int(state["global_step"])
    graph = GraphMmap("data/processed")
    tasks = load_task_jsonl(args.tasks)
    curses.wrapper(run, graph, model, tasks, args.checkpoint, args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
