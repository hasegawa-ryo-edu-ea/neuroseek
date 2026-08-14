from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek import mcp_server


def test_initialize_and_tool_discovery_are_valid_json_rpc():
    initialized = mcp_server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert initialized is not None
    assert initialized["result"]["protocolVersion"] == "2024-11-05"
    assert initialized["result"]["capabilities"] == {"tools": {}}
    listed = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed is not None
    assert {tool["name"] for tool in listed["result"]["tools"]} == {
        "neuroseek_get_capabilities", "neuroseek_get_local_facts", "neuroseek_run_validation_search",
    }


def test_notification_has_no_response_and_invalid_tool_arguments_are_rejected():
    assert mcp_server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    response = mcp_server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "neuroseek_get_local_facts", "arguments": {"entity_id": "Japan"},
    }})
    assert response is not None
    assert response["error"]["code"] == -32602


def test_graph_tool_normalizes_ids_and_uses_bounded_read_only_worker():
    observed: dict[str, object] = {}

    def worker(root: Path, script: str, arguments: list[str]):
        observed.update(root=root, script=script, arguments=arguments)
        return {"event": "local_graph_facts", "returned_count": 1}

    with patch.object(mcp_server, "_run_worker", worker):
        response = mcp_server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "neuroseek_get_local_facts", "arguments": {"entity_id": "q17", "relation_id": "p36", "limit": 3},
        }})
    assert response is not None
    assert observed["script"] == "scripts/mcp_graph_query.py"
    assert observed["arguments"] == ["--entity", "Q17", "--limit", "3", "--relation", "P36"]
    assert json.loads(response["result"]["content"][0]["text"])["returned_count"] == 1


def test_validation_tool_never_accepts_checkpoint_or_unbounded_steps():
    response = mcp_server.handle_message({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "neuroseek_run_validation_search", "arguments": {"task_index": 0, "max_steps": 13},
    }})
    assert response is not None
    assert response["error"]["code"] == -32602
    forbidden = mcp_server.handle_message({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
        "name": "neuroseek_run_validation_search", "arguments": {"checkpoint": "/tmp/other.ckpt"},
    }})
    assert forbidden is not None
    assert forbidden["error"]["code"] == -32602
