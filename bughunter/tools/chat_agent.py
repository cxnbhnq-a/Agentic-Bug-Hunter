"""Interactive Agentic Chat Runner with network, web search, file, and shell tools."""

from __future__ import annotations

import html as _html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

MAX_OUTPUT_CHARS = 6000
MAX_TOOL_STEPS = 8

CHAT_AGENT_SYSTEM = """You are BugHunter AI, an expert Bug Bounty & Penetration Testing assistant running inside the user's terminal.
You DO have real network access, web search, HTTP request capability, local file access, and shell command execution via tools.
Never claim "I have no network access" — use your tools whenever you need live data, CVE research, recon, or endpoint testing.

Available tools:
1. web_search(query, max_results=6)
   Search the web (DuckDuckGo) for CVEs, exploit writeups, HackerOne disclosures, tech docs, or dorks.
2. http_fetch(url, method="GET", headers=None, data=None)
   Send a live HTTP request and inspect status code, response headers, and body.
3. run_command(command)
   Execute a shell command on the user's machine (e.g. subfinder, httpx, nuclei, nmap, curl, dig, whois, ffuf).
4. read_file(path, start_line=1, end_line=200)
   Read a local file (e.g. recon output, findings, JS files).
5. list_dir(path=".")
   List files and subdirectories in a local path.

To call a tool, output ONLY a <tool_call> block with valid JSON:
<tool_call>
{"tool": "http_fetch", "args": {"url": "https://example.com"}}
</tool_call>

Rules:
- Emit at most ONE <tool_call> per response, then wait for <tool_result>.
- Always close with </tool_call>. Keep run_command short and focused (1-3 commands max per call, never giant multi-section scripts).
- Never fabricate tool outputs, URLs, or findings you did not observe in <tool_result>.
- Be concise, practical, and security-focused. Match the user's language."""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _decode_ddg_url(href: str) -> str:
    href = _html.unescape(href)
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        if qs.get("uddg"):
            return unquote(qs["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def tool_web_search(query: str, max_results: int = 6) -> str:
    """Search the web via DuckDuckGo HTML endpoint (no API key required)."""
    import requests

    if not query or not query.strip():
        return "Error: query is required"
    max_results = max(1, min(int(max_results or 6), 12))
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    }
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query.strip()},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"web_search error: {exc}"

    body = resp.text
    link_matches = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    snippet_matches = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not link_matches:
        return f"No search results found for: {query}"

    lines = [f"Search results for: {query}"]
    for idx, (href, raw_title) in enumerate(link_matches[:max_results]):
        title = re.sub(r"<[^>]+>", "", _html.unescape(raw_title)).strip()
        url = _decode_ddg_url(href)
        snippet = ""
        if idx < len(snippet_matches):
            snippet = re.sub(r"<[^>]+>", "", _html.unescape(snippet_matches[idx])).strip()
        lines.append(f"{idx + 1}. {title}\n   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
    return _truncate("\n".join(lines))


def tool_http_fetch(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    data: str | dict | None = None,
) -> str:
    """Perform a live HTTP request and return status, headers, and body."""
    import requests

    if not url or not str(url).strip():
        return "Error: url is required"
    method = (method or "GET").upper()
    req_headers = dict(headers) if isinstance(headers, dict) else {}
    req_headers.setdefault("User-Agent", "BugHunter-Chat/1.0")

    kwargs: dict = {"headers": req_headers, "timeout": 20, "allow_redirects": True}
    if data is not None:
        if isinstance(data, dict):
            kwargs["json"] = data
        else:
            kwargs["data"] = str(data)

    try:
        resp = requests.request(method, url.strip(), **kwargs)
    except Exception as exc:
        return f"http_fetch error: {exc}"

    hdr_lines = "\n".join(f"  {k}: {v}" for k, v in list(resp.headers.items())[:25])
    out = (
        f"HTTP {resp.status_code} {resp.reason}\n"
        f"Final URL: {resp.url}\n"
        f"Headers:\n{hdr_lines}\n\n"
        f"Body:\n{_truncate(resp.text, 4500)}"
    )
    return _truncate(out)


def tool_run_command(command: str, timeout: int = 60) -> str:
    """Execute a local shell command and capture stdout/stderr."""
    if not command or not str(command).strip():
        return "Error: command is required"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = ""
        if proc.stdout:
            combined += proc.stdout
        if proc.stderr:
            combined += ("\n[stderr]\n" if combined else "[stderr]\n") + proc.stderr
        if not combined.strip():
            combined = "(no output)"
        return _truncate(f"Exit code: {proc.returncode}\n{combined}")
    except subprocess.TimeoutExpired:
        return f"run_command error: timed out after {timeout}s"
    except Exception as exc:
        return f"run_command error: {exc}"


def tool_read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read lines from a local file."""
    if not path or not str(path).strip():
        return "Error: path is required"
    p = Path(os.path.expanduser(path.strip()))
    if not p.is_file():
        return f"Error: file not found: {p}"
    try:
        start = max(1, int(start_line or 1))
        end = max(start, int(end_line or start + 199))
        lines = p.read_text(errors="replace").splitlines()
        selected = lines[start - 1 : end]
        numbered = "\n".join(f"{start + i}: {line}" for i, line in enumerate(selected))
        return _truncate(f"File: {p} (lines {start}-{min(end, len(lines))} of {len(lines)})\n{numbered}")
    except Exception as exc:
        return f"read_file error: {exc}"


def tool_list_dir(path: str = ".") -> str:
    """List files and directories at path."""
    p = Path(os.path.expanduser((path or ".").strip()))
    if not p.is_dir():
        return f"Error: directory not found: {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = [f"{'[DIR] ' if e.is_dir() else '      '}{e.name}" for e in entries[:100]]
        extra = f"\n... ({len(entries) - 100} more)" if len(entries) > 100 else ""
        return f"Directory: {p}\n" + ("\n".join(lines) + extra if lines else "(empty)")
    except Exception as exc:
        return f"list_dir error: {exc}"


TOOLS = {
    "web_search": tool_web_search,
    "http_fetch": tool_http_fetch,
    "run_command": tool_run_command,
    "read_file": tool_read_file,
    "list_dir": tool_list_dir,
}

# Tools that execute external commands or active target HTTP requests ask confirmation unless auto-approved.
CONFIRM_REQUIRED_TOOLS = {"run_command", "http_fetch"}


def _parse_dsml_tool_call(reply: str) -> tuple[str, dict] | None:
    """Parse DeepSeek DSML tool invocation XML (<｜DSML｜ invoke name="...">...)."""
    inv_match = re.search(
        r"<[｜|]DSML[｜|]\s*invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)(?:</[｜|]DSML[｜|]\s*invoke>|$)",
        reply,
        flags=re.DOTALL,
    )
    if not inv_match:
        return None
    tool_name = inv_match.group(1).strip()
    body = inv_match.group(2)
    args: dict = {}
    for p_match in re.finditer(
        r"<[｜|]DSML[｜|]\s*parameter\s+name=[\"']([^\"']+)[\"'](?:\s+string=[\"'](true|false)[\"'])?\s*>(.*?)</[｜|]DSML[｜|]\s*parameter>",
        body,
        flags=re.DOTALL,
    ):
        pname = p_match.group(1).strip()
        is_str = p_match.group(2)
        pval = p_match.group(3)
        if is_str == "false":
            try:
                args[pname] = json.loads(pval.strip())
                continue
            except ValueError:
                pass
        elif pval.strip().startswith("{") and pname == "headers":
            try:
                args[pname] = json.loads(pval.strip())
                continue
            except ValueError:
                pass
        args[pname] = pval.strip() if pname != "data" else pval
    if tool_name not in TOOLS:
        return None
    return tool_name, args


def parse_tool_call(reply: str) -> tuple[str, dict] | None:
    """Extract a tool call from <tool_call>, DeepSeek DSML XML, or fenced JSON."""
    if not reply:
        return None

    # 1. Standard <tool_call>{...} (tolerates non-standard closing tags)
    marker = re.search(r"<tool_call>\s*", reply)
    if marker:
        tail = reply[marker.end():]
        brace_idx = tail.find("{")
        if brace_idx != -1:
            try:
                data, _ = json.JSONDecoder().raw_decode(tail[brace_idx:])
                if isinstance(data, dict) and data.get("tool"):
                    tool_name = str(data["tool"]).strip()
                    args = data.get("args") if isinstance(data.get("args"), dict) else {}
                    return tool_name, args
            except ValueError:
                pass

    # 2. DeepSeek DSML native XML (<｜DSML｜ invoke name="...">...)
    dsml = _parse_dsml_tool_call(reply)
    if dsml:
        return dsml

    # 3. Fenced JSON {"tool": "...", "args": {...}}
    for fence in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", reply, flags=re.DOTALL):
        try:
            data = json.loads(fence.group(1))
            if isinstance(data, dict) and data.get("tool") in TOOLS:
                tool_name = str(data["tool"]).strip()
                args = data.get("args") if isinstance(data.get("args"), dict) else {}
                return tool_name, args
        except ValueError:
            continue

    return None


def execute_tool(
    tool_name: str,
    args: dict,
    auto_approve: bool = False,
    confirm_fn: Callable[[str, dict], str] | None = None,
) -> tuple[str, bool]:
    """Execute a parsed tool call, returning (result_text, new_auto_approve)."""
    fn = TOOLS.get(tool_name)
    if not fn:
        return f"Unknown tool '{tool_name}'. Available: {', '.join(TOOLS)}", auto_approve

    if tool_name in CONFIRM_REQUIRED_TOOLS and not auto_approve and confirm_fn is not None:
        decision = (confirm_fn(tool_name, args) or "").strip().lower()
        if decision == "a":
            auto_approve = True
        elif decision not in ("", "y", "yes"):
            return f"User declined tool execution ({tool_name}). Proceed without it or ask.", auto_approve

    try:
        return str(fn(**args)), auto_approve
    except TypeError as exc:
        return f"Invalid arguments for {tool_name}: {exc}", auto_approve
    except Exception as exc:
        return f"Tool {tool_name} failed: {exc}", auto_approve


def run_chat_turn(
    client,
    user_input: str,
    history: list[dict],
    auto_approve: bool = False,
    confirm_fn: Callable[[str, dict], str] | None = None,
    on_tool_event: Callable[[str, str, dict, str | None], None] | None = None,
    max_tokens: int = 4096,
) -> tuple[str, bool]:
    """Run one user turn with multi-step tool calling, returning (final_reply, auto_approve)."""
    ctx_parts = []
    for turn in history[-4:]:
        ctx_parts.append(f"[you] {turn['user']}")
        ctx_parts.append(f"[assistant] {turn['assistant']}")

    conversation = ("\n".join(ctx_parts) + "\n\n" if ctx_parts else "") + f"[you] {user_input}"

    for _ in range(MAX_TOOL_STEPS):
        reply = client.chat(None, CHAT_AGENT_SYSTEM, conversation, max_tokens=max_tokens)
        if not reply:
            return "", auto_approve

        parsed = parse_tool_call(reply)
        if not parsed:
            return reply, auto_approve

        tool_name, args = parsed
        if on_tool_event:
            on_tool_event("start", tool_name, args, None)

        result, auto_approve = execute_tool(
            tool_name,
            args,
            auto_approve=auto_approve,
            confirm_fn=confirm_fn,
        )
        if on_tool_event:
            on_tool_event("end", tool_name, args, result)

        conversation += (
            f"\n\n[assistant]\n{reply}\n\n"
            f"<tool_result tool=\"{tool_name}\">\n{result}\n</tool_result>\n\n"
            f"Continue based on the tool_result above."
        )

    # One last synthesis if tool budget was reached
    final = client.chat(
        None,
        CHAT_AGENT_SYSTEM,
        conversation + "\n\nTool step limit reached. Summarize what you found for the user now.",
        max_tokens=max_tokens,
    )
    return final or "", auto_approve
