#!/usr/bin/env python3
"""
Pure-HTML Scraper → DRY-RUN test dataset for GraphRAG (no MediaWiki API required).

- Reads your curated seed graph (combined_accelerators_clean.json) and visits each page URL directly.
- Parses HTML to extract sections, text, tables, figures, and internal links.
- Chunks text per section with configurable size/overlap.
- Embeddings:
    • Default: REAL embeddings via embeddings.py (thellert/accphysbert_cased or ENV EMBED_MODEL)
    • Optional: --fake-embeddings for deterministic hash-based vectors (no ML deps)
    • Optional: --embed-title to prepend page title to passage before embedding

- Emits NDJSON files in --out:
    manifest.json
    articles.ndjson
    chunks.ndjson
    figures.ndjson
    links.ndjson
    assets_manifest.ndjson
    hierarchy.ndjson

Auth:
- If your wiki requires login, set an HTTP Cookie header via env var: WIKI_COOKIE="key1=val1; key2=val2;"
- Optionally pass extra headers via --headers-json (JSON dict).

Examples:
python scrape_wiki_html_testset.py \
  --seeds combined_accelerators_clean.json \
  --out ./out_testset_html_real \
  --all \
  --sleep 0.4 \
  --use-cuda auto \
  --embed-model thellert/accphysbert_cased \
  --embed-title
"""

import argparse, json, os, re, sys, time, hashlib, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple
from urllib.parse import urlparse, urljoin, urldefrag
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from pathlib import Path

# -------- logging helpers --------
def log(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------
# HTML utilities
# ----------------------
def strip_fragment(url: str) -> str:
    base, _ = urldefrag(url or "")
    return base

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ").strip())

def normalize_tables(html: str) -> List[Dict]:
    soup = BeautifulSoup(html or "", "lxml")
    out: List[Dict] = []
    for tbl in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in tbl.find_all("th")]
        rows: List[List[str]] = []
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if cells: rows.append(cells)
        if not rows:
            continue
        md = ""
        if headers:
            md += "| " + " | ".join(headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in rows:
            md += "| " + " | ".join(r) + " |\n"
        out.append({"headers": headers, "rows": rows, "markdown": md})
    return out

def sectionize_rendered_html(doc_html: str) -> List[Dict]:
    """
    Split by headings inside the main content wrapper.
    Works for MediaWiki-like pages that use <div class="mw-parser-output">.
    Falls back to whole document if not found.
    """
    soup = BeautifulSoup(doc_html or "", "lxml")
    root = soup.select_one("div.mw-parser-output") or soup.body or soup
    sections: List[Dict] = []
    current = {"title": "Intro", "anchor": None, "html": ""}
    # Iterate child nodes to keep order
    for el in getattr(root, "children", []):
        name = getattr(el, "name", None)
        if name in ("h1", "h2", "h3"):
            sections.append(current)
            anchor = el.get("id") or None
            title = el.get_text(" ", strip=True) or "Section"
            current = {"title": title, "anchor": anchor, "html": ""}
        elif hasattr(el, "decode"):
            current["html"] += el.decode()
    sections.append(current)
    sections = [s for s in sections if (s["html"] or "").strip()]
    return sections or [{"title": "Intro", "anchor": None, "html": doc_html or ""}]

def resolve_url(base_url: str, href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    return urljoin(base_url, href)

def is_same_host(base_url: str, link_url: str) -> bool:
    try:
        b = urlparse(base_url or "")
        l = urlparse(link_url or "")
        return (b.scheme, b.netloc) == (l.scheme, l.netloc)
    except Exception:
        return False

# ----------------------
# Data classes
# ----------------------
@dataclass
class ArticleRow:
    article_id: str
    accelerator: str
    tab: str
    section: Optional[str]
    category: Optional[str]
    title: str
    url: str
    lang: str
    revid: Optional[int]          # unknown in pure HTML; keep None
    last_modified: Optional[str]  # from HTTP header if available
    valid: bool = True
    # Graph structure metadata
    node_type: Optional[str] = None      # root, accelerator, tab, section, category, article
    order: Optional[int] = None          # hierarchical ordering
    parent_id: Optional[str] = None      # parent node in the graph
    path_from_root: Optional[str] = None # full path from root for GraphRAG
    depth_level: Optional[int] = None    # depth in the hierarchy

@dataclass
class ChunkRow:
    chunk_id: str
    article_id: str
    section_title: str
    section_anchor: Optional[str]
    text: str
    html: Optional[str]
    tables_md: Optional[str]
    embedding: List[float]

@dataclass
class FigureRow:
    asset_id: str
    article_id: str
    chunk_id: Optional[str]
    caption: Optional[str]
    url: Optional[str]
    mime: Optional[str]
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class LinkRow:
    src_article_id: str
    dst_article_id: str   # mapped to seed ID when possible, else absolute URL
    anchor_text: Optional[str] = None

@dataclass
class HierarchyRow:
    """Represents hierarchical relationships in the knowledge graph"""
    parent_id: str
    child_id: str
    relationship_type: str  # parent_child, contains, belongs_to
    parent_type: str       # root, accelerator, tab, section, category
    child_type: str        # accelerator, tab, section, category, article
    parent_title: str
    child_title: str
    depth_difference: int = 1

# ----------------------
# Seeds
# ----------------------
def iter_seed_articles_with_context(seeds: Dict) -> Iterable[Dict]:
    """Walk the tree structure and yield articles with full context"""
    def walk(node, path=[], parent_id=None, depth=0, parent_title=""):
        if isinstance(node, dict):
            node_id = node.get("id", f"unknown_{depth}")
            node_title = node.get("title", node_id)
            
            # Add current node to path  
            current_path = path + [node_title]
            current_node = dict(node)
            current_node["parent_id"] = parent_id
            current_node["parent_title"] = parent_title
            current_node["path_from_root"] = " > ".join(current_path)
            current_node["depth_level"] = depth
            
            # If this node has a URL, it's an article/page we want to scrape
            if "id" in node and ("url" in node or "path" in node or "page" in node):
                yield current_node
            
            # Process children
            if "children" in node and isinstance(node["children"], list):
                for child in node["children"]:
                    yield from walk(child, current_path, node_id, depth + 1, node_title)
            
            # Also check other dict values that might contain children (for legacy support)
            for k, v in node.items():
                if k != "children" and isinstance(v, (dict, list)):
                    yield from walk(v, current_path, node_id, depth + 1, node_title)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item, path, parent_id, depth, parent_title)
    
    yield from walk(seeds)

def extract_hierarchy_relationships(seeds: Dict) -> List[HierarchyRow]:
    """Extract all hierarchical relationships from the seed structure"""
    relationships = []
    
    def walk_for_hierarchy(node, parent_node=None, depth=0):
        if isinstance(node, dict):
            # If we have a parent, create a relationship
            if parent_node is not None:
                relationship = HierarchyRow(
                    parent_id=parent_node.get("id", f"unknown_parent_{depth-1}"),
                    child_id=node.get("id", f"unknown_child_{depth}"),
                    relationship_type="parent_child",
                    parent_type=parent_node.get("type", "unknown"),
                    child_type=node.get("type", "unknown"),
                    parent_title=parent_node.get("title", parent_node.get("id", "Unknown")),
                    child_title=node.get("title", node.get("id", "Unknown")),
                    depth_difference=1
                )
                relationships.append(relationship)
            
            # Process children
            if "children" in node and isinstance(node["children"], list):
                for child in node["children"]:
                    walk_for_hierarchy(child, node, depth + 1)
                    
        elif isinstance(node, list):
            for item in node:
                walk_for_hierarchy(item, parent_node, depth)
    
    walk_for_hierarchy(seeds)
    return relationships

def iter_seed_articles(seeds: Dict) -> Iterable[Dict]:
    """Backward compatibility wrapper"""
    for article in iter_seed_articles_with_context(seeds):
        yield article

def parse_context_from_id(seed_id: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    parts = (seed_id or "").split(":")
    accel = parts[0] if parts else "unknown"
    tab = parts[1] if len(parts) > 1 else None
    return accel, tab, None, parts[-1] if parts else None

# ----------------------
# HTTP Client
# ----------------------
class HtmlClient:
    def __init__(self, timeout: int = 25, sleep: float = 0.5, headers: Optional[Dict] = None, retries: int = 2):
        self.sess = requests.Session()
        self.timeout = timeout
        self.sleep = sleep
        self.retries = retries
        self.sess.headers.update({"User-Agent": "PSI-GraphRAG-HTML/0.1"})
        if headers:
            self.sess.headers.update(headers)
        cookie = os.environ.get("WIKI_COOKIE")
        if cookie:
            self.sess.headers.update({"Cookie": cookie})

    def get_html(self, url: str):
        url = strip_fragment(url)
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self.sess.get(url, timeout=self.timeout)
                r.raise_for_status()
                time.sleep(self.sleep)
                return r.text, r  # return response for headers (Last-Modified)
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise last_err

# ----------------------
# Chunking
# ----------------------
def chunk_section_text(section_text: str, max_words=600, overlap_words=50) -> List[str]:
    chunks: List[str] = []
    text = section_text or ""
    words = text.split()
    if not words:
        return []
    
    start = 0
    total_words = len(words)
    
    while start < total_words:
        end = min(total_words, start + max_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        if end == total_words:
            break
        start = max(0, end - overlap_words)
    
    return chunks

# ----------------------
# Writer
# ----------------------
def write_ndjson(path: Path, rows: Iterable[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="Path to combined_accelerators_clean.json")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--max-per-accel", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="Process ALL valid seeds")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--overlap-words", type=int, default=50)
    ap.add_argument("--headers-json", type=str, default=None, help="Path to JSON dict of extra headers")
    ap.add_argument("--embed-model", type=str, default=None, help="HF model name (defaults to ENV EMBED_MODEL or accphysBERT)")
    ap.add_argument("--use-cuda", type=str, default=None, choices=["true", "false", "auto"], help="CUDA preference")
    ap.add_argument("--hf-token", type=str, default=None, help="Hugging Face token if needed")
    ap.add_argument("--no-embed-title", dest="embed_title", action="store_false", help="Do NOT prepend title to passage text before embedding")
    ap.set_defaults(embed_title=True)

    args = ap.parse_args()

    outdir = Path(args.out)
    ensure_dir(outdir)

    # Load seeds
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)
    seed_articles = list(iter_seed_articles_with_context(seeds))

    # URL → seed-id map for resolving internal links back to our graph IDs
    seed_url_to_id: Dict[str, str] = {}
    for a in seed_articles:
        u = a.get("url") or a.get("path") or a.get("page")
        if u:
            seed_url_to_id[strip_fragment(u)] = a.get("id", u)

    # Group seeds by accelerator
    by_accel: Dict[str, List[Dict]] = defaultdict(list)
    for a in seed_articles:
        acc, *_ = parse_context_from_id(a.get("id", ""))
        by_accel[acc].append(a)

    # Select pages
    if args.all:
        selected = [x for x in seed_articles if x.get("valid", True)]
    else:
        selected = []
        for acc, items in by_accel.items():
            valids = [x for x in items if x.get("valid", True)]
            random.shuffle(valids)
            selected.extend(valids[:args.max_per_accel])
        if args.limit is not None:
            selected = selected[:args.limit]

    # De-duplicate by normalized URL
    def norm_url(u): return strip_fragment((u or "").strip())
    seen = set()
    uniq = []
    for s in selected:
        u = norm_url(s.get("url") or s.get("path") or s.get("page"))
        if u and u not in seen:
            seen.add(u)
            uniq.append(s)
    selected = uniq
    log(f"Selected {len(selected)} seed pages.")

    # Extra headers
    extra_headers = None
    if args.headers_json:
        with open(args.headers_json, "r", encoding="utf-8") as hf:
            extra_headers = json.load(hf)

    client = HtmlClient(timeout=args.timeout, sleep=args.sleep, headers=extra_headers)

    # Embedding setup
    embedder = None
    embed_dim_runtime = 384  # default dimension
    # Make sure we can import your embeddings.py next to this script
    sys.path.append(str(Path(__file__).resolve().parent))
    from embeddings import get_embedder  # type: ignore
    embedder = get_embedder(
        model_name=args.embed_model,
        use_cuda=args.use_cuda,
        hf_token=args.hf_token
    )
    embed_dim_runtime = embedder.vector_dim  # e.g., 768 for accphysBERT

    articles_out: List[dict] = []
    chunks_out: List[dict] = []
    figures_out: List[dict] = []
    links_out: List[dict] = []
    assets_manifest_out: List[dict] = []
    
    # Extract hierarchical relationships from the seed structure
    hierarchy_relationships = extract_hierarchy_relationships(seeds)
    hierarchy_out = [asdict(rel) for rel in hierarchy_relationships]

    for i, seed in enumerate(selected, 1):
        url = seed.get("url") or seed.get("path") or seed.get("page")
        if not url:
            log(f"[SKIP] missing url in seed: {seed.get('id')}")
            continue
        try:
            log(f"[{i}/{len(selected)}] {seed.get('id')} → {url}")
            html, resp = client.get_html(url)
            soup = BeautifulSoup(html, "lxml")

            # Title: prefer MediaWiki's firstHeading, else <title>, else seed
            title_el = soup.select_one("#firstHeading") or soup.find("h1")
            if title_el:
                title = title_el.get_text(" ", strip=True)
            elif soup.title:
                title = soup.title.get_text(" ", strip=True)
            else:
                title = seed.get("title") or url

            accel, tab, _, _leaf = parse_context_from_id(seed.get("id", ""))

            # article row with enhanced graph metadata
            article = ArticleRow(
                article_id=seed.get("id", url),
                accelerator=accel,
                tab=tab or seed.get("tab") or "",
                section=None,
                category=seed.get("category") or "",
                title=title,
                url=url,
                lang=seed.get("lang") or "de",
                revid=None,
                last_modified=resp.headers.get("Last-Modified"),
                valid=seed.get("valid", True),
                # Enhanced graph structure metadata
                node_type=seed.get("type", "article"),
                order=seed.get("order"),
                parent_id=seed.get("parent_id"),
                path_from_root=seed.get("path_from_root"),
                depth_level=seed.get("depth_level", 0)
            )
            articles_out.append(asdict(article))

            # main content and sections
            content_root = soup.select_one("div.mw-parser-output") or soup.body or soup
            sections = sectionize_rendered_html(str(content_root))

            base = strip_fragment(url)

            # internal links → map to seed IDs when possible
            # (dedupe per page)
            link_set = set()
            for a_tag in content_root.find_all("a", href=True):
                target = resolve_url(base, a_tag["href"])
                if not target or not is_same_host(base, target):
                    continue
                tnorm = strip_fragment(target)
                if tnorm in link_set:
                    continue
                link_set.add(tnorm)
                dst_id = seed_url_to_id.get(tnorm, tnorm)
                links_out.append(asdict(LinkRow(
                    src_article_id=article.article_id,
                    dst_article_id=dst_id,
                    anchor_text=a_tag.get_text(" ", strip=True) or None
                )))

            # figures (img tags)
            fig_set = set()
            for img in content_root.find_all("img"):
                src = resolve_url(base, img.get("src"))
                if not src: continue
                if src in fig_set: continue
                fig_set.add(src)
                mime = None
                low = src.lower()
                if low.endswith(".png"): mime = "image/png"
                elif low.endswith(".jpg") or low.endswith(".jpeg"): mime = "image/jpeg"
                elif low.endswith(".pdf"): mime = "application/pdf"
                asset_id = hashlib.sha1(src.encode("utf-8")).hexdigest()
                figures_out.append(asdict(FigureRow(
                    asset_id=asset_id,
                    article_id=article.article_id,
                    chunk_id=None,
                    caption=img.get("alt") or None,
                    url=src,
                    mime=mime,
                    width=None, height=None
                )))
                if mime in ("image/png", "image/jpeg", "application/pdf"):
                    assets_manifest_out.append({"asset_id": asset_id, "url": src, "mime": mime})

            # Build chunks per section (batch embed per article)
            article_chunk_texts: List[str] = []
            article_chunk_refs: List[int] = []

            for sec_idx, sec in enumerate(sections):
                sec_text = html_to_text(sec["html"])
                sec_tables = normalize_tables(sec["html"])
                tables_md = "\n\n".join(t["markdown"] for t in sec_tables) if sec_tables else None

                parts = chunk_section_text(sec_text, max_words=args.chunk_words, overlap_words=args.overlap_words)
                for j, t in enumerate(parts):
                    plain_id = f"{article.article_id}::s{sec_idx}::c{j}"
                    chunk_id = hashlib.sha1(plain_id.encode("utf-8")).hexdigest()

                    # text for embedding (optionally prepend title)
                    to_embed = f"{title}\n\n{t}" if args.embed_title else t
                    article_chunk_texts.append(to_embed)

                    chunks_out.append(asdict(ChunkRow(
                        chunk_id=chunk_id,
                        article_id=article.article_id,
                        section_title=sec["title"],
                        section_anchor=sec.get("anchor"),
                        text=t,
                        html=None,  # keep small; store full HTML only if really needed
                        tables_md=tables_md,
                        embedding=[]  # fill after batch encode
                    )))
                    article_chunk_refs.append(len(chunks_out) - 1)

            # Embeddings for all chunks of this article
            X = embedder.encode_passages(
                article_chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            for loc, vec in zip(article_chunk_refs, X):
                chunks_out[loc]["embedding"] = vec.tolist()

        except Exception as e:
            log(f"[WARN] Failed {seed.get('id')}: {e}")

    # manifest + write files
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "html",
        "seeds": os.path.abspath(args.seeds),
        "articles": len(articles_out),
        "chunks": len(chunks_out),
        "figures": len(figures_out),
        "links": len(links_out),
        "hierarchy_relationships": len(hierarchy_out),
        "embed_dim": embed_dim_runtime,
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "max_per_accel": None if args.all else args.max_per_accel,
        "embed_title": bool(args.embed_title),
    }
    with (Path(args.out) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_ndjson(Path(args.out) / "articles.ndjson", articles_out)
    write_ndjson(Path(args.out) / "chunks.ndjson", chunks_out)
    write_ndjson(Path(args.out) / "figures.ndjson", figures_out)
    write_ndjson(Path(args.out) / "links.ndjson", links_out)
    write_ndjson(Path(args.out) / "assets_manifest.ndjson", assets_manifest_out)
    write_ndjson(Path(args.out) / "hierarchy.ndjson", hierarchy_out)

    log(f"Done. Wrote to: {args.out}")

if __name__ == "__main__":
    main()
