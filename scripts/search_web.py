#!/usr/bin/env python3
"""Loopback-only, CPU-only NEUROSEEK evidence-search web server.

The viewer reads the same immutable graph and presentation checkpoint as the
terminal search console.  It deliberately owns no CUDA context, does not write
run data, and never signals the detached trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from neuroseek.data.graph import GraphMmap
from search_console import DEFAULT_TASKS, wikidata_candidates, wikidata_labels


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"


class App:
    def __init__(self) -> None:
        self.graph = GraphMmap(ROOT / "data/processed")

    def search(self, term: str, language: str) -> dict[str, Any]:
        if not term.strip():
            return {"term": term, "candidates": []}
        direct = term.strip().upper()
        if direct.startswith("Q") and direct[1:].isdigit():
            remote = [{"id": direct, "label": direct, "description": "direct Wikidata Q-ID"}]
        else:
            remote = wikidata_candidates(term, language, "item")
        candidates = []
        for row in remote:
            local_id = self.graph.find_entity_identifier(row["id"])
            candidates.append({
                "identifier": row["id"], "label": row.get("label", row["id"]),
                "description": row.get("description", ""), "local": local_id is not None,
                "local_id": local_id,
                "wikidata_url": f"https://www.wikidata.org/wiki/{row['id']}",
            })
        return {"term": term, "candidates": candidates}

    def graph_view(self, identifier: str, relation_term: str, language: str) -> dict[str, Any]:
        entity = self.graph.find_entity_identifier(identifier.upper())
        if entity is None:
            raise ValueError("This entity is not contained in the local graph snapshot.")
        relation_id: int | None = None
        relation_data: dict[str, str] | None = None
        if relation_term.strip():
            direct = relation_term.strip().upper()
            choices = ([{"id": direct, "label": direct}]
                       if direct.startswith("P") and direct[1:].isdigit()
                       else wikidata_candidates(relation_term, language, "property"))
            for row in choices:
                found = self.graph.find_relation_identifier(row["id"])
                if found is not None:
                    relation_id = found
                    relation_data = {"identifier": row["id"], "label": row.get("label", row["id"])}
                    break
            if relation_id is None:
                raise ValueError("The requested relation is not contained in the local graph snapshot.")
        nodes, relations = self.graph.neighbors(entity)
        edges = [(int(node), int(rel)) for node, rel in zip(nodes, relations)
                 if relation_id is None or int(rel) == relation_id][:18]
        ids = [self.graph.entity_identifier(entity)]
        for node, rel in edges:
            ids.extend((self.graph.entity_identifier(node), self.graph.relation_identifier(rel)))
        try:
            labels = wikidata_labels(ids, language)
        except OSError:
            labels = {}
        return {
            "root": {"id": self.graph.entity_identifier(entity), "label": labels.get(self.graph.entity_identifier(entity), self.graph.entity_label(entity))},
            "relation_filter": relation_data,
            "edges": [{
                "source": self.graph.entity_identifier(entity),
                "relation": {"id": self.graph.relation_identifier(rel), "label": labels.get(self.graph.relation_identifier(rel), self.graph.relation_label(rel))},
                "target": {"id": self.graph.entity_identifier(node), "label": labels.get(self.graph.entity_identifier(node), self.graph.entity_label(node))},
            } for node, rel in edges],
            "graph": {"entities": self.graph.manifest.entity_count, "relations": self.graph.manifest.relation_count, "triples": self.graph.manifest.original_triples},
        }

    def policy(self, task: int) -> dict[str, Any]:
        command = [sys.executable, "scripts/live_query.py", "--index", str(max(0, task))]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120, check=True)
        return json.loads(result.stdout)


def make_handler(app: App, language: str):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                return super().do_GET()
            query = urllib.parse.parse_qs(parsed.query)
            chosen_language = query.get("lang", [language])[0] if query.get("lang", [language])[0] in {"ja", "en"} else language
            try:
                if parsed.path == "/api/health":
                    self.send_json({"status": "ready", "mode": "cpu_read_only", "cuda": False})
                elif parsed.path == "/api/search":
                    self.send_json(app.search(query.get("q", [""])[0], chosen_language))
                elif parsed.path == "/api/graph":
                    self.send_json(app.graph_view(query.get("entity", [""])[0], query.get("relation", [""])[0], chosen_language))
                elif parsed.path == "/api/policy":
                    self.send_json(app.policy(int(query.get("task", ["0"])[0])))
                else:
                    self.send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--lang", choices=("ja", "en"), default="ja")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("NEUROSEEK web viewer must bind to loopback only")
    app = App()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app, args.lang))
    print(f"NEUROSEEK Web UI: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
