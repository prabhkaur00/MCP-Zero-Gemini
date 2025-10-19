# tools/reembed_with_gemini.py
import os, json, time, argparse
from typing import Dict, Any, List
from tqdm import tqdm

try:
    import google.generativeai as genai
except ImportError as e:
    raise SystemExit("pip install google-generativeai tqdm") from e

EMBED_MODEL = "text-embedding-004"

def embed_text(txt: str) -> List[float]:
    if not txt:
        return None
    r = genai.embed_content(model=EMBED_MODEL, content=txt)
    return r["embedding"]

def reembed(src_path: str, dst_path: str, overwrite: bool = False, sleep: float = 0.0,
            max_servers: int = 0, allowlist: str = "", name_regex: str = "",
            max_tools_per_server: int = 0):

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # --- NEW: optional server filtering before embedding ---
        names_allow = set()
        if allowlist:
            with open(allowlist, "r", encoding="utf-8") as af:
                names_allow = {line.strip() for line in af if line.strip()}

        if name_regex:
            import re
            pat = re.compile(name_regex)
            data = [s for s in data if pat.search(s.get("name", ""))]

        if names_allow:
            data = [s for s in data if s.get("name", "") in names_allow]

        if max_servers and max_servers > 0:
            data = data[:max_servers]

        print(f"[subset] servers to embed: {len(data)}")
        # -----------------------------------------------

    # If resuming, load existing output to skip completed work
    existing = {}
    if os.path.exists(dst_path) and not overwrite:
        try:
            with open(dst_path, "r", encoding="utf-8") as f:
                existing = {s.get("name",""): s for s in json.load(f)}
        except Exception:
            existing = {}

    out = []
    for server in tqdm(data, desc="Re-embedding servers/tools"):
        sname = server.get("name", "")
        prev = existing.get(sname, {})

        # copy base
        new_server: Dict[str, Any] = {k: v for k, v in server.items() if k not in (
            "description_embedding", "summary_embedding")}

        # server-level embeddings (reuse if present in existing file)
        if "description_embedding" in prev and prev["description_embedding"]:
            new_server["description_embedding"] = prev["description_embedding"]
        else:
            new_server["description_embedding"] = embed_text(server.get("description", ""))

        if "summary" in server:
            if "summary_embedding" in prev and prev["summary_embedding"]:
                new_server["summary_embedding"] = prev["summary_embedding"]
            else:
                new_server["summary_embedding"] = embed_text(server.get("summary", ""))

        # tool-level embeddings
        tools = []
        prev_tools_by_name = {t.get("name",""): t for t in prev.get("tools", [])} if prev else {}
        for tool in server.get("tools", []):
            tname = tool.get("name", "")
            new_tool = {k: v for k, v in tool.items() if k != "description_embedding"}
            if tname in prev_tools_by_name and prev_tools_by_name[tname].get("description_embedding"):
                new_tool["description_embedding"] = prev_tools_by_name[tname]["description_embedding"]
            else:
                new_tool["description_embedding"] = embed_text(tool.get("description", ""))
            tools.append(new_tool)
            if sleep: time.sleep(sleep)  # gentle throttle if you hit RPM limits

        if max_tools_per_server and max_tools_per_server > 0:
            tools = tools[:max_tools_per_server]
        # --------------------------------------------

        new_server["tools"] = tools
        out.append(new_server)

        # checkpoint every 50 servers to be safe
        if len(out) % 50 == 0:
            with open(dst_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
    # final write
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Done. Wrote {len(out)} servers → {dst_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="MCP-tools/mcp_tools_with_embedding.json",
                    help="Input JSON")
    ap.add_argument("--dst", default="MCP-tools/mcp_tools_with_embedding_gemini.json",
                    help="Output JSON with Gemini embeddings")
    ap.add_argument("--overwrite", action="store_true", help="Ignore existing dst and rebuild")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between tool embeds")

    # NEW: subset controls
    ap.add_argument("--max-servers", type=int, default=0,
                    help="Embed at most this many servers (0 = all)")
    ap.add_argument("--allowlist", type=str, default="",
                    help="Path to a newline-delimited list of server names to include")
    ap.add_argument("--name-regex", type=str, default="",
                    help="Regex to include servers by name (applied before max-servers limit)")
    ap.add_argument("--max-tools-per-server", type=int, default=0,
                    help="Only embed up to this many tools per server (0 = all)")
    args = ap.parse_args()
    reembed(args.src, args.dst, overwrite=args.overwrite, sleep=args.sleep,
            max_servers=args.max_servers,
            allowlist=args.allowlist,
            name_regex=args.name_regex,
            max_tools_per_server=args.max_tools_per_server)

