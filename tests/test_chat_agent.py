"""Tests for the agentic chat runner (network, web search, file, and shell tools)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
BUGHUNTER_DIR = ROOT / "bughunter"
if str(BUGHUNTER_DIR) not in sys.path:
    sys.path.insert(0, str(BUGHUNTER_DIR))

from tools import chat_agent  # noqa: E402


def test_parse_tool_call_extracts_valid_block():
    reply = 'Let me check that.\n<tool_call>\n{"tool": "http_fetch", "args": {"url": "https://example.com"}}\n</tool_call>'
    parsed = chat_agent.parse_tool_call(reply)
    assert parsed == ("http_fetch", {"url": "https://example.com"})
    assert chat_agent.parse_tool_call("Just plain text") is None
    assert chat_agent.parse_tool_call("<tool_call>{bad json}</tool_call>") is None

    # DeepSeek DSML closing tag variant
    dsml = '<tool_call>\n{"tool": "run_command", "args": {"command": "curl -sI https://example.com"}}\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>'
    assert chat_agent.parse_tool_call(dsml) == (
        "run_command",
        {"command": "curl -sI https://example.com"},
    )

    # Full native DeepSeek DSML XML invoke
    full_dsml = (
        '<｜DSML｜ calls>\n'
        '<｜DSML｜ invoke name="run_command">\n'
        '<｜DSML｜ parameter name="command" string="true">dig +short edlink.id A</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n'
        '</｜DSML｜ calls>'
    )
    assert chat_agent.parse_tool_call(full_dsml) == (
        "run_command",
        {"command": "dig +short edlink.id A"},
    )

    # Multi-parameter DSML with non-string integer
    multi_dsml = (
        '<|DSML| invoke name="web_search">\n'
        '<|DSML| parameter name="query" string="true">edlink.id CVE</|DSML| parameter>\n'
        '<|DSML| parameter name="max_results" string="false">5</|DSML| parameter>\n'
        '</|DSML| invoke>'
    )
    assert chat_agent.parse_tool_call(multi_dsml) == (
        "web_search",
        {"query": "edlink.id CVE", "max_results": 5},
    )

    # Fenced JSON fallback
    fenced = '```json\n{"tool": "list_dir", "args": {"path": "."}}\n```'
    assert chat_agent.parse_tool_call(fenced) == ("list_dir", {"path": "."})


def test_tool_web_search_parses_ddg_html(monkeypatch):
    fake_html = """
    <html><body>
      <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fnvd.nist.gov%2Fvuln%2Fdetail%2FCVE-2024-1234">CVE-2024-1234 Details</a>
      <a class="result__snippet">Critical auth bypass vulnerability.</a>
    </body></html>
    """
    fake_resp = MagicMock()
    fake_resp.text = fake_html
    fake_resp.raise_for_status.return_value = None

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **kw: fake_resp)

    out = chat_agent.tool_web_search("CVE-2024-1234", max_results=3)
    assert "CVE-2024-1234 Details" in out
    assert "https://nvd.nist.gov/vuln/detail/CVE-2024-1234" in out
    assert "Critical auth bypass vulnerability." in out


def test_tool_http_fetch_returns_status_headers_body(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.reason = "OK"
    fake_resp.url = "https://example.com/"
    fake_resp.headers = {"Server": "nginx", "X-Frame-Options": "SAMEORIGIN"}
    fake_resp.text = "<html>Hello BugHunter</html>"

    import requests
    monkeypatch.setattr(requests, "request", lambda *a, **kw: fake_resp)

    out = chat_agent.tool_http_fetch("https://example.com")
    assert "HTTP 200 OK" in out
    assert "Server: nginx" in out
    assert "Hello BugHunter" in out


def test_tool_read_file_and_list_dir(tmp_path):
    sample = tmp_path / "urls.txt"
    sample.write_text("https://a.test\nhttps://b.test\n")

    ls_out = chat_agent.tool_list_dir(str(tmp_path))
    assert "urls.txt" in ls_out

    read_out = chat_agent.tool_read_file(str(sample), start_line=1, end_line=2)
    assert "1: https://a.test" in read_out
    assert "2: https://b.test" in read_out


def test_execute_tool_confirmation_flow(monkeypatch):
    monkeypatch.setitem(chat_agent.TOOLS, "http_fetch", lambda url: f"fetched:{url}")

    # Decline
    res, auto = chat_agent.execute_tool("http_fetch", {"url": "https://x.test"}, False, lambda *_: "n")
    assert "declined" in res.lower()
    assert auto is False

    # Always approve
    res, auto = chat_agent.execute_tool("http_fetch", {"url": "https://x.test"}, False, lambda *_: "a")
    assert res == "fetched:https://x.test"
    assert auto is True


def test_run_chat_turn_executes_tool_then_answers(monkeypatch):
    monkeypatch.setitem(chat_agent.TOOLS, "web_search", lambda query: "1. Example CVE result")

    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        '<tool_call>{"tool": "web_search", "args": {"query": "test cve"}}</tool_call>',
        "Found 1 result: Example CVE result.",
    ]

    reply, auto = chat_agent.run_chat_turn(fake_client, "cariin info test cve", [], auto_approve=True)
    assert reply == "Found 1 result: Example CVE result."
    assert auto is True
    assert fake_client.chat.call_count == 2


def test_session_persistence_and_target_context(tmp_path):
    sessions_dir = tmp_path / "sessions"
    recon_root = tmp_path / "recon"
    findings_root = tmp_path / "findings"

    (recon_root / "edlink.id" / "live").mkdir(parents=True)
    (recon_root / "edlink.id" / "live" / "urls.txt").write_text("https://edlink.id\nhttps://api.edlink.id\n")
    (findings_root / "edlink.id").mkdir(parents=True)
    (findings_root / "edlink.id" / "summary.txt").write_text("TOTAL FINDINGS: 1\n- CORS on api.edlink.id\n")

    ctx = chat_agent.build_target_context("edlink.id", recon_root, findings_root)
    assert "live/urls.txt: 2 entries" in ctx
    assert "https://api.edlink.id" in ctx
    assert "CORS on api.edlink.id" in ctx

    # Save twice — must overwrite single file, not create clutter
    p1 = chat_agent.save_chat_session("edlink.id", 2, [{"user": "u1", "assistant": "a1"}], sessions_dir)
    p2 = chat_agent.save_chat_session("edlink.id", 3, [{"user": "u2", "assistant": "a2"}], sessions_dir)
    assert p1 == p2
    assert len(list(sessions_dir.iterdir())) == 1

    loaded = chat_agent.load_chat_session("edlink.id", sessions_dir)
    assert loaded["phase"] == 3
    assert loaded["history"] == [{"user": "u2", "assistant": "a2"}]


def test_run_chat_turn_injects_phase_and_target_context():
    fake_client = MagicMock()
    fake_client.chat.return_value = "Testing IDOR on api.edlink.id"

    reply, _ = chat_agent.run_chat_turn(
        fake_client,
        "lanjut hunt",
        [],
        phase=3,
        target="edlink.id",
        target_context="live/urls.txt: 2 entries",
    )
    assert reply == "Testing IDOR on api.edlink.id"
    sent_prompt = fake_client.chat.call_args.args[2]
    assert "[Active Target: edlink.id]" in sent_prompt
    assert "[Active Phase: Phase 3 — Targeted Hunting." in sent_prompt
    assert "live/urls.txt: 2 entries" in sent_prompt
