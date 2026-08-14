#!/usr/bin/env python3
"""NEUROSEEK's dependency-free, stdio Model Context Protocol server.

The server itself uses only the Python standard library.  Each knowledge tool
starts the existing ``search`` Compose profile, which is CPU-only and mounts
this repository read-only.  Consequently an MCP client cannot change graph
evidence, checkpoints, telemetry, or the concurrent GPU trainer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SERVER_NAME = "neuroseek"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_OUTPUT_BYTES = 1_000_000


def repository_root() -> Path:
    configured = os.environ.get("NEUROSEEK_ROOT")
    root = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not (root / "compose.yaml").is_file():
        raise RuntimeError(f"NEUROSEEK_ROOT is not a repository root: {root}")
    return root


def _tool_definitions() -> list[dict[str, object]]:
    return [
        {
            "name": "neuroseek_get_capabilities",
            "description": "Describe the local NEUROSEEK graph and proof boundaries before using its results.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "neuroseek_get_local_facts",
            "description": "Return up to 50 direct outgoing facts from the immutable local Wikidata5M graph for a Q-ID. Results are graph evidence, not generated text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "pattern": "^[Qq][0-9]+$", "description": "Wikidata entity ID, for example Q17."},
                    "relation_id": {"type": "string", "pattern": "^[Pp][0-9]+$", "description": "Optional Wikidata property ID filter."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": ["entity_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "neuroseek_run_validation_search",
            "description": "Execute the immutable learned policy on one local validation task and return its path, answer, and independently checked proof status. This is not a natural-language QA model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_index": {"type": "integer", "minimum": 0, "default": 0},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 12, "default": 12},
                },
                "additionalProperties": False,
            },
        },
    ]


def _text_result(payload: object, *, error: bool = False) -> dict[str, object]:
    text = json.dumps(payload, ensure_ascii=False, indent=2) if not isinstance(payload, str) else payload
    result: dict[str, object] = {"content": [{"type": "text", "text": text}]}
    if error:
        result["isError"] = True
    return result


def _require_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("arguments must be an object")
    return value


def _positive_int(arguments: dict[str, Any], name: str, default: int, maximum: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def _only(arguments: dict[str, Any], *allowed: str) -> None:
    unexpected = sorted(set(arguments).difference(allowed))
    if unexpected:
        raise ValueError(f"unsupported argument(s): {', '.join(unexpected)}")


def _run_worker(root: Path, script: str, arguments: list[str]) -> object:
    # Match the repository's existing query launcher: a Jetson operator may
    # have passwordless Docker through sudo but not direct socket membership.
    # ``-n`` guarantees that an MCP client can never block waiting for a
    # password or receive one through its stdio channel.
    docker = ["docker"]
    try:
        direct = subprocess.run(["docker", "info"], text=True, capture_output=True, timeout=10, check=False)
        if direct.returncode:
            docker = ["sudo", "-n", "docker"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        docker = ["sudo", "-n", "docker"]
    command = [*docker, "compose", "-f", "compose.yaml", "run", "--rm", "--no-deps", "search", "python3", script, *arguments]
    try:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=120, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("Docker is required for NEUROSEEK MCP queries but was not found") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("NEUROSEEK read-only query exceeded its 120-second limit") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(f"read-only NEUROSEEK worker failed (exit {completed.returncode}): {detail}")
    output = completed.stdout.strip()
    if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise RuntimeError("NEUROSEEK worker response exceeded the MCP output limit")
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("NEUROSEEK worker did not return valid JSON") from error


def call_tool(name: str, raw_arguments: object) -> dict[str, object]:
    arguments = _require_object(raw_arguments)
    root = repository_root()
    if name == "neuroseek_get_capabilities":
        _only(arguments)
        return _text_result({
            "system": "NEUROSEEK",
            "knowledge_source": "immutable local Wikidata5M CSR graph",
            "model": "immutable learned graph-navigation policy",
            "evidence_rule": "Use direct_local_csr_edge facts as graph evidence. Treat learned-policy answers as usable only when valid_proof is true.",
            "isolation": "Every tool call is CPU-only, read-only, bounded to 120 seconds, and cannot modify a trainer, checkpoint, graph, or telemetry.",
            "limitations": "The graph is a finite local snapshot. No public-Wikidata lookup or generative answer synthesis is performed by this MCP server.",
        })
    if name == "neuroseek_get_local_facts":
        _only(arguments, "entity_id", "relation_id", "limit")
        entity = arguments.get("entity_id")
        relation = arguments.get("relation_id")
        if not isinstance(entity, str) or not entity.upper().startswith("Q") or not entity[1:].isdigit():
            raise ValueError("entity_id must be a Wikidata Q-ID")
        if relation is not None and (not isinstance(relation, str) or not relation.upper().startswith("P") or not relation[1:].isdigit()):
            raise ValueError("relation_id must be a Wikidata P-ID")
        limit = _positive_int(arguments, "limit", 10, 50)
        worker_arguments = ["--entity", entity.upper(), "--limit", str(limit)]
        if relation:
            worker_arguments.extend(["--relation", relation.upper()])
        return _text_result(_run_worker(root, "scripts/mcp_graph_query.py", worker_arguments))
    if name == "neuroseek_run_validation_search":
        _only(arguments, "task_index", "max_steps")
        task_index = arguments.get("task_index", 0)
        if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index < 0:
            raise ValueError("task_index must be a non-negative integer")
        steps = _positive_int(arguments, "max_steps", 12, 12)
        return _text_result(_run_worker(root, "scripts/live_query.py", ["--index", str(task_index), "--max-steps", str(steps)]))
    raise ValueError(f"unknown tool: {name}")


def _response(identifier: object, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def _error(identifier: object, code: int, message: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


def handle_message(message: object) -> dict[str, object] | None:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(message.get("id") if isinstance(message, dict) else None, -32600, "invalid JSON-RPC request")
    method, identifier, params = message["method"], message.get("id"), message.get("params", {})
    if method == "notifications/initialized":
        return None
    try:
        if method == "initialize":
            parameters = _require_object(params)
            requested = parameters.get("protocolVersion")
            protocol = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return _response(identifier, {"protocolVersion": protocol, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}})
        if method == "tools/list":
            return _response(identifier, {"tools": _tool_definitions()})
        if method == "tools/call":
            parameters = _require_object(params)
            name = parameters.get("name")
            if not isinstance(name, str):
                raise ValueError("tools/call requires a tool name")
            return _response(identifier, call_tool(name, parameters.get("arguments", {})))
        return _error(identifier, -32601, f"method not found: {method}")
    except ValueError as error:
        return _error(identifier, -32602, str(error))
    except RuntimeError as error:
        return _response(identifier, _text_result(str(error), error=True))
    except Exception as error:  # Keep stdio framing intact even on an unexpected host failure.
        return _response(identifier, _text_result(f"internal NEUROSEEK MCP error: {error}", error=True))


def main() -> int:
    for line in sys.stdin:
        try:
            response = handle_message(json.loads(line))
        except json.JSONDecodeError:
            response = _error(None, -32700, "invalid JSON")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
