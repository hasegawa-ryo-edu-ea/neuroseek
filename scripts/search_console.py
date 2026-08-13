#!/usr/bin/env python3
"""Cinematic, read-only console for real NEUROSEEK learned-policy searches.

The process owns neither a trainer socket nor a CUDA context.  It loads one
immutable checkpoint on CPU and explores the immutable mmap graph only after
the operator requests a task.  Keyboard input controls this viewer alone.
"""
from __future__ import annotations

import argparse
import curses
import os
import time
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


class Console:
    def __init__(self, screen: Any, graph: GraphMmap, model: NavigatorPolicy, tasks: list[tuple[QuerySpec, tuple[int, ...] | None]], checkpoint: Path, language: str):
        self.screen, self.graph, self.model, self.tasks, self.checkpoint = screen, graph, model, tasks, checkpoint
        self.language = language
        self.task_index = 0
        self.tab = 1
        self.run: SearchRun | None = None
        self.command = ""
        self.command_mode = False
        self.notice = self.text("r: execute the loaded policy", "r: 学習済み方策を実行")
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
            self.screen.addnstr(row, max(0, column), value, max(0, width - column - 1), style)
        except curses.error:
            pass

    def entity(self, identifier: int) -> str:
        return f"{self.graph.entity_label(identifier)}  [{self.graph.entity_identifier(identifier)}]"

    def relation(self, identifier: int) -> str:
        return f"{self.graph.relation_label(identifier)}  [{self.graph.relation_identifier(identifier)}]"

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
        tabs = [(1, self.text("SEARCH", "探索")), (2, self.text("PATH", "経路")), (3, self.text("PROOF", "証明")), (4, self.text("SYSTEM", "システム"))]
        column = 2
        for number, name in tabs:
            active = number == self.tab
            self.write(3, column, f"[{number}] {name}", 1 if active else 6, active)
            column += len(name) + 7
        self.write(3, max(column + 2, width - 42), self.text("r run · n next · : command · q quit", "r 実行 · n 次問 · : コマンド · q 終了"), 6)
        self.write(4, 1, "─" * max(1, width - 2), 1)
        return 6

    def page_search(self, row: int) -> None:
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
            self.page_search(row)
        elif self.tab == 2:
            self.page_path(row)
        elif self.tab == 3:
            self.page_proof(row)
        else:
            self.page_system(row)
        height, width = self.screen.getmaxyx()
        self.write(height - 3, 1, "─" * max(1, width - 2), 1)
        prompt = self.command if self.command_mode else self.text("type :help for local commands", ":help でローカルコマンドを表示")
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
        if words[0] in {"q", "quit", "exit"}:
            return True
        if words[0] in {"r", "run"}:
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
            self.notice = "r/run [index] · n/next · 1-4 tabs · l · lang ja|en · q/quit"
        else:
            self.notice = self.text("Unknown viewer command. Use :help.", "不明なビューアコマンドです。:helpを使用してください。")
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
            elif key in ("r", "R"):
                self.execute()
            elif key in ("n", "N"):
                self.apply("next")
            elif key in ("l", "L"):
                self.language = "ja" if self.language == "en" else "en"
            elif isinstance(key, str) and key in "1234":
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
