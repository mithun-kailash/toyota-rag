"""Toyota RAG - LITE version. Works on locked-down Windows laptops.

Needs only `pypdf` (pure Python). No torch, numpy, faiss or streamlit, so nothing
compiled that Windows "Application Control" could block.

Search = BM25 keyword ranking (written here in plain Python).
Answers = Claude through a plain HTTPS call, only if ANTHROPIC_API_KEY is set (in .env or the environment).

Run:  python rag_lite.py
"""
import json
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

BASE = Path(__file__).parent
PDF_PATH = BASE / "Toyota_Cars_Complete_Knowledge_Base.pdf"
CHUNK_CHARS, OVERLAP_LINES, TOP_K = 900, 2, 5

HEADER_RE = re.compile(r"^TOYOTA CARS \| COMPLETE KNOWLEDGE BASE Page \d+$")
SECTION_RE = re.compile(r"^(\d{1,2})\. (\S.+)$")
STOP = set("a an the of and or to in on for is are was were be it its this that with as by at from what which how do does did can i you".split())


# ---------- 1. Read the PDF and make chunks ----------
def load_pages():
    pages = []
    for i, page in enumerate(PdfReader(str(PDF_PATH)).pages, start=1):
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
        pages.append((i, [ln for ln in lines if ln and not HEADER_RE.match(ln)]))
    return pages


def build_chunks():
    sections, expected = [{"title": "Introduction", "lines": []}], 1
    for page_no, lines in load_pages():
        for ln in lines:
            m = SECTION_RE.match(ln)
            if page_no >= 3 and m and len(ln) < 90 and not ln.endswith("?") and int(m.group(1)) == expected:
                sections.append({"title": f"{m.group(1)}. {m.group(2)}", "lines": []})
                expected += 1
                continue
            sections[-1]["lines"].append((page_no, ln))

    chunks = []
    for sec in sections:
        if sec["title"].startswith("20."):  # question bank: only repeats questions, would pollute search
            continue
        groups, cur, size = [], [], 0
        for page_no, ln in sec["lines"]:
            if cur and size + len(ln) > CHUNK_CHARS:
                groups.append(cur)
                cur = cur[-OVERLAP_LINES:]
                size = sum(len(x[1]) for x in cur)
            cur.append((page_no, ln))
            size += len(ln) + 1
        if cur and (not groups or len(cur) > OVERLAP_LINES):
            groups.append(cur)
        for g in groups:
            chunks.append(
                {
                    "section": sec["title"],
                    "pages": sorted({p for p, _ in g}),
                    "text": "\n".join(ln for _, ln in g),
                }
            )
    return chunks


# ---------- 2. BM25 keyword search ----------
def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOP]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # section title is searched too, so "India" finds the India section
        self.docs = [Counter(tokenize(c["section"] + " " + c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=TOP_K):
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [dict(self.chunks[i], score=s) for s, i in scored[:k]]


# ---------- 3. Optional: ask Claude to write the answer ----------
SYSTEM = (
    "You answer questions about Toyota using ONLY the numbered context passages. "
    "If they do not contain the answer, say the knowledge base does not cover it. "
    "Keep country/model-year qualifications. Cite passages like [1][2]. Be concise."
)


def load_env():
    env = BASE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ask_claude(question, results):
    ctx = "\n\n".join(
        f"[{n}] (Section: {r['section']} | page {', '.join(map(str, r['pages']))})\n{r['text']}"
        for n, r in enumerate(results, start=1)
    )
    body = {
        "model": os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 800,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Context passages:\n\n{ctx}\n\nQuestion: {question}"}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["content"][0]["text"]


def main():
    load_env()
    chunks = build_chunks()
    bm25 = BM25(chunks)
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"Loaded {len(chunks)} chunks. Mode: {'Claude answers' if use_llm else 'passages only (no API key)'}")
    print("Ask a question about Toyota. Press Enter on an empty line to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            break
        results = bm25.search(q)
        if not results:
            print("\nNo matching passage found in the knowledge base.\n")
            continue
        if use_llm:
            try:
                print(f"\n{ask_claude(q, results)}\n")
            except Exception as e:  # network/key problems should not crash the chat
                print(f"\n(Could not reach Claude: {e}. Showing passages instead.)\n")
        print("Sources:")
        for n, r in enumerate(results, start=1):
            print(f"\n[{n}] {r['section']} (page {', '.join(map(str, r['pages']))})")
            if not use_llm:
                print(r["text"])
        print()


if __name__ == "__main__":
    main()
