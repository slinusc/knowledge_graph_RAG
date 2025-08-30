#!/usr/bin/env python3
"""
Ingest scraped wiki data into Qdrant (vector database) and Neo4j (graph database).

This script takes the JSON output from your wiki scraper and:
1) Chunks the text content
2) Creates embeddings and stores them in Qdrant
3) Inserts a graph into Neo4j (batched UNWIND upserts)
   - Optional hierarchy creation either from titles with "/" or a provided JSON tree.

Usage:
    python ingest_to_databases.py scraped_results.json [flags]

Key flags:
    --qdrant-only / --neo4j-only
    --skip-qdrant / --skip-neo4j
    --derive-parent-from-title           # Build parent from 'A/B/C' -> parent 'A/B'
    --tree /path/to/SwissFEL_tree.json   # Build hierarchy from predefined JSON tree
    --clear-neo4j                        # Wipes only your wiki nodes (page:, img:) before ingest

Environment:
    CHUNK_WORDS (default 300), CHUNK_OVERLAP (default 50)
    NEO4J_URI (default bolt://localhost:7687)
    NEO4J_USER (default neo4j)
    NEO4J_PASS (default password)
    QDRANT_URL (default http://localhost:6333)
    QDRANT_COLLECTION (default swissfel_wiki)

Dependencies:
    pip install qdrant-client neo4j sentence-transformers tqdm
"""

import os
import sys
import json
import uuid
import hashlib
import argparse
import logging
from math import ceil
from typing import Dict, Any, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Your embedder factory; must expose .encode_passages(...) and .vector_dim
from embeddings import get_embedder

# ========== CONFIG ==========
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CHUNK_WORDS   = int(os.environ.get("CHUNK_WORDS", 300))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "swissfel_wiki")

# Global clients
_qdrant_client = None
_neo4j_driver = None

# ========== TEXT CHUNKING & IDS ==========

def chunk_text(text: str, max_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Split a string into overlapping word chunks."""
    words = text.split()
    chunks = []
    if not words:
        return chunks

    start = 0
    idx = 1
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_txt = " ".join(chunk_words).strip()
        if chunk_txt:
            chunks.append({"index": idx, "text": chunk_txt, "word_count": len(chunk_words)})
        idx += 1
        if end == len(words):
            break
        start = end - overlap_words if end - overlap_words > start else end
    return chunks

def create_page_id(title: str) -> str:
    page_id = title.lower().replace(" ", "_").replace("/", "_")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    page_id = "".join(c for c in page_id if c in allowed)
    return f"page:{page_id}"

def create_chunk_id(page_id: str, chunk_index: int) -> str:
    namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')
    return str(uuid.uuid5(namespace, f"{page_id}#chunk{chunk_index}"))

def create_image_id(image_url: str) -> str:
    return f"img:{hashlib.md5(image_url.encode('utf-8')).hexdigest()}"

# ========== QDRANT ==========
def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            _qdrant_client = QdrantClient(url=QDRANT_URL)
            embedder = get_embedder()
            expected_dim = embedder.vector_dim

            if _qdrant_client.collection_exists(QDRANT_COLLECTION):
                info = _qdrant_client.get_collection(QDRANT_COLLECTION)
                current_dim = info.config.params.vectors.size
                if current_dim != expected_dim:
                    logging.warning(f"Existing collection {QDRANT_COLLECTION} has dim {current_dim}, expected {expected_dim}. Recreating.")
                    _qdrant_client.delete_collection(QDRANT_COLLECTION)
                    _qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE)
                    )
                else:
                    logging.info(f"Using existing Qdrant collection: {QDRANT_COLLECTION} ({current_dim}D)")
            else:
                logging.info(f"Creating Qdrant collection: {QDRANT_COLLECTION} ({expected_dim}D)")
                _qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE)
                )
        except ImportError:
            logging.error("qdrant-client not installed. pip install qdrant-client")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to connect to Qdrant at {QDRANT_URL}: {e}")
            sys.exit(1)
    return _qdrant_client

def ingest_to_qdrant(scraped_data: List[Dict[str, Any]]):
    from qdrant_client.models import PointStruct

    logging.info("Starting Qdrant ingestion...")
    client = get_qdrant_client()
    embedder = get_embedder()

    # Flatten chunks
    all_chunks = []
    for page in scraped_data:
        if not page.get("text") or page["text"].startswith("Error:"):
            continue
        page_id = page.get("page_id", create_page_id(page["title"]))
        chunks = chunk_text(page["text"])
        for ch in chunks:
            all_chunks.append({
                "chunk_id": create_chunk_id(page_id, ch["index"]),
                "page_id": page_id,
                "title": page["title"],
                "url": page["url"],
                "index": ch["index"],
                "text": ch["text"],
                "word_count": ch["word_count"],
                "original_chunk_ref": f"{page_id}#chunk{ch['index']}",
            })

    if not all_chunks:
        logging.warning("No chunks to ingest into Qdrant.")
        return

    texts = [c["text"] for c in all_chunks]
    logging.info(f"Creating {len(texts)} embeddings...")
    embeddings = embedder.encode_passages(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    points = []
    for ch, emb in zip(all_chunks, embeddings):
        points.append(PointStruct(
            id=ch["chunk_id"],
            vector=emb.astype("float32").tolist(),
            payload={
                "page_id": ch["page_id"],
                "title": ch["title"],
                "url": ch["url"],
                "index": ch["index"],
                "text": ch["text"],
                "word_count": ch["word_count"],
                "original_chunk_ref": ch["original_chunk_ref"],
            }
        ))

    batch_size = 100
    logging.info(f"Uploading {len(points)} points to Qdrant in batches of {batch_size}...")
    for i in tqdm(range(0, len(points), batch_size), desc="Qdrant upsert"):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+batch_size])

    logging.info(f"Qdrant ingestion complete: {len(points)} points in '{QDRANT_COLLECTION}'.")

# ========== NEO4J (BATCHED) ==========
def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            with _neo4j_driver.session() as session:
                session.run("RETURN 1").single()
            logging.info(f"Connected to Neo4j at {NEO4J_URI}")
        except ImportError:
            logging.error("neo4j driver not installed. pip install neo4j")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j at {NEO4J_URI}: {e}")
            sys.exit(1)
    return _neo4j_driver

CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Page)  REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
    # Helpful lookups:
    "CREATE INDEX IF NOT EXISTS FOR (p:Page) ON (p.title)",
    "CREATE INDEX IF NOT EXISTS FOR (p:Page) ON (p.url)"
]

UPSERT_BATCH = """
UNWIND $pages AS p
MERGE (pg:Page {id: p.id})
  ON CREATE SET pg.title = p.title, pg.url = p.url, pg.text_length = p.text_length, pg.created_at = datetime()
  ON MATCH  SET pg.title = p.title, pg.url = p.url, pg.text_length = p.text_length, pg.updated_at = datetime()

FOREACH (_ IN CASE WHEN p.parent_id IS NULL THEN [] ELSE [1] END |
  MERGE (parent:Page {id: p.parent_id})
  MERGE (parent)-[:HAS_CHILD]->(pg)
  MERGE (pg)-[:HAS_PARENT]->(parent)
)

FOREACH (ch IN p.chunks |
  MERGE (c:Chunk {id: ch.id})
    ON CREATE SET c.index = ch.index, c.text = ch.text, c.word_count = ch.word_count
    ON MATCH  SET c.index = ch.index, c.text = ch.text, c.word_count = ch.word_count
  MERGE (pg)-[:HAS_CHUNK]->(c)
)

FOREACH (im IN p.images |
  MERGE (i:Image {id: im.id})
    ON CREATE SET i.url = im.url
    ON MATCH  SET i.url = im.url
  MERGE (pg)-[:HAS_IMAGE]->(i)
)
"""

def ensure_constraints(session):
    for stmt in CREATE_CONSTRAINTS:
        session.run(stmt)

def parent_from_title(title: str) -> Optional[str]:
    """Compute parent page id from a title with slashes: 'A/B/C' -> 'A/B'."""
    if "/" not in title:
        return None
    parts = title.split("/")
    if len(parts) <= 1:
        return None
    parent_title = "/".join(parts[:-1])
    return create_page_id(parent_title)

def build_parent_map_from_tree(tree: dict) -> Dict[str, str]:
    """Return dict mapping child_title -> parent_page_id from a nested tree with 'title' and 'children' keys."""
    parent_map: Dict[str, str] = {}
    def walk(node: dict, parent_title: Optional[str] = None):
        title = node.get("title")
        if title and parent_title:
            parent_map[title] = create_page_id(parent_title)
        for ch in node.get("children", []):
            walk(ch, title)
    walk(tree, None)
    return parent_map

def build_pages_payload(
    scraped_data: List[Dict[str, Any]],
    use_tree_parent: bool,
    parent_map: Optional[Dict[str, str]],
    derive_parent_from_title: bool
) -> List[Dict[str, Any]]:
    pages = []
    for page in scraped_data:
        if not page.get("text") or page["text"].startswith("Error:"):
            continue

        page_id = page.get("page_id", create_page_id(page["title"]))

        # Choose parent_id priority: explicit in data > JSON tree > title-derivation
        parent_id = page.get("parent_id")
        if parent_id is None and use_tree_parent and parent_map:
            parent_id = parent_map.get(page["title"])
        if parent_id is None and derive_parent_from_title:
            parent_id = parent_from_title(page["title"])

        # Chunks
        chunk_nodes = []
        for ch in chunk_text(page["text"]):
            chunk_nodes.append({
                "id": create_chunk_id(page_id, ch["index"]),
                "index": ch["index"],
                "text": ch["text"],
                "word_count": ch["word_count"],
            })

        # Images
        image_nodes = []
        for u in page.get("images", []):
            image_nodes.append({"id": create_image_id(u), "url": u})

        pages.append({
            "id": page_id,
            "title": page["title"],
            "url": page["url"],
            "text_length": len(page["text"]),
            "parent_id": parent_id,
            "chunks": chunk_nodes,
            "images": image_nodes
        })
    return pages

def ingest_to_neo4j_batched(
    scraped_data: List[Dict[str, Any]],
    use_tree_parent: bool = False,
    tree_path: Optional[str] = None,
    derive_parent_from_title: bool = False,
    clear: bool = False,
    batch_size: int = 100
):
    from neo4j import GraphDatabase

    driver = get_neo4j_driver()
    with driver.session() as session:
        ensure_constraints(session)

        if clear:
            logging.info("Clearing existing wiki data in Neo4j (page:/img: only)...")
            session.run("""
                MATCH (n)
                WHERE n.id STARTS WITH 'page:' OR n.id STARTS WITH 'img:'
                DETACH DELETE n
            """)

        parent_map = None
        if use_tree_parent:
            if not tree_path:
                raise ValueError("--tree must be provided when --use-tree-parent is enabled")
            with open(tree_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
            parent_map = build_parent_map_from_tree(tree)
            logging.info(f"Loaded parent map from tree: {len(parent_map)} entries")

        pages_payload = build_pages_payload(
            scraped_data=scraped_data,
            use_tree_parent=use_tree_parent,
            parent_map=parent_map,
            derive_parent_from_title=derive_parent_from_title
        )

        if not pages_payload:
            logging.warning("No valid pages to ingest into Neo4j.")
            return

        logging.info(f"Ingesting {len(pages_payload)} pages into Neo4j in batches of {batch_size}...")
        total = len(pages_payload)
        # use execute_write for automatic retries on transient errors
        for i in tqdm(range(0, total, batch_size), desc="Neo4j upsert"):
            batch = pages_payload[i:i+batch_size]
            session.execute_write(lambda tx: tx.run(UPSERT_BATCH, pages=batch))

    logging.info("Neo4j ingestion completed.")

# ========== MAIN ==========
def parse_args():
    p = argparse.ArgumentParser(description="Ingest scraped wiki data into Qdrant and Neo4j")
    p.add_argument("input_file", help="JSON file from wiki scraper")
    p.add_argument("--qdrant-only", action="store_true", help="Only ingest into Qdrant")
    p.add_argument("--neo4j-only", action="store_true", help="Only ingest into Neo4j")
    p.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant ingestion")
    p.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j ingestion")

    # Hierarchy options
    p.add_argument("--derive-parent-from-title", action="store_true", help="Infer parent from slashes in title")
    p.add_argument("--tree", default=None, help="Path to SwissFEL_tree.json to define canonical hierarchy")
    p.add_argument("--clear-neo4j", action="store_true", help="Clear existing wiki nodes (page:/img:) before ingest")
    p.add_argument("--neo4j-batch-size", type=int, default=100, help="Batch size for Neo4j upsert")

    return p.parse_args()

def main():
    args = parse_args()

    # Load scraped JSON
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            scraped_data = json.load(f)
        logging.info(f"Loaded {len(scraped_data)} pages from {args.input_file}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {args.input_file}: {e}")
        sys.exit(1)

    ingest_qdrant = not args.skip_qdrant and not args.neo4j_only
    ingest_neo4j = not args.skip_neo4j and not args.qdrant_only

    try:
        if ingest_qdrant:
            ingest_to_qdrant(scraped_data)
        else:
            logging.info("Skipping Qdrant ingestion")

        if ingest_neo4j:
            ingest_to_neo4j_batched(
                scraped_data=scraped_data,
                use_tree_parent=bool(args.tree),
                tree_path=args.tree,
                derive_parent_from_title=args.derive_parent_from_title,
                clear=args.clear_neo4j,
                batch_size=args.neo4j_batch_size
            )
        else:
            logging.info("Skipping Neo4j ingestion")

        logging.info("Ingestion completed successfully!")

    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        sys.exit(1)
    finally:
        global _neo4j_driver
        if _neo4j_driver:
            _neo4j_driver.close()

if __name__ == "__main__":
    main()
