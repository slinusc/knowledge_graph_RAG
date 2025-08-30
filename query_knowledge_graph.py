#!/usr/bin/env python3
"""
Query the knowledge graph using both Qdrant (vector search) and Neo4j (graph traversal).

This script demonstrates how to reuse the embedding model for both
vector similarity search and structured graph queries.
"""

import os
import argparse
import logging
from typing import List, Dict, Any, Optional
import json

from embeddings import get_embedder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "swissfel_wiki")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "password")

# Global connections
_qdrant_client = None
_neo4j_driver = None


def get_qdrant_client():
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(url=QDRANT_URL)
            logging.info(f"Connected to Qdrant at {QDRANT_URL}")
        except ImportError:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Qdrant: {e}")
    return _qdrant_client


def get_neo4j_driver():
    """Get or create Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            # Test connection
            with _neo4j_driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logging.info(f"Connected to Neo4j at {NEO4J_URI}")
        except ImportError:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Neo4j: {e}")
    return _neo4j_driver


def vector_search(query: str, limit: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search using Qdrant.
    
    Args:
        query: Search query text
        limit: Maximum number of results
        score_threshold: Minimum similarity score
        
    Returns:
        List of search results with metadata
    """
    client = get_qdrant_client()
    embedder = get_embedder()
    
    # Create query embedding
    query_vector = embedder.encode_query(query)[0].tolist()
    
    # Search in Qdrant
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True
    )
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            'chunk_id': result.id,
            'score': result.score,
            'page_id': result.payload.get('page_id'),
            'title': result.payload.get('title'),
            'url': result.payload.get('url'),
            'text': result.payload.get('text'),
            'chunk_index': result.payload.get('index'),
            'word_count': result.payload.get('word_count')
        })
    
    return formatted_results


def get_page_hierarchy(page_id: str) -> Dict[str, Any]:
    """
    Get the hierarchical context of a page using Neo4j.
    
    Args:
        page_id: ID of the page to get hierarchy for
        
    Returns:
        Dictionary containing parent and child pages
    """
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Get parent pages (ancestors)
        parents_query = """
        MATCH path = (ancestor:Page)-[:HAS_CHILD*]->(page:Page {id: $page_id})
        RETURN ancestor.id as id, ancestor.title as title, ancestor.url as url, length(path) as depth
        ORDER BY depth DESC
        """
        
        # Get child pages (descendants)
        children_query = """
        MATCH path = (page:Page {id: $page_id})-[:HAS_CHILD*]->(descendant:Page)
        RETURN descendant.id as id, descendant.title as title, descendant.url as url, length(path) as depth
        ORDER BY depth ASC
        """
        
        # Get direct siblings
        siblings_query = """
        MATCH (page:Page {id: $page_id})-[:HAS_PARENT]->(parent:Page)-[:HAS_CHILD]->(sibling:Page)
        WHERE sibling.id <> $page_id
        RETURN sibling.id as id, sibling.title as title, sibling.url as url
        """
        
        parents = list(session.run(parents_query, page_id=page_id))
        children = list(session.run(children_query, page_id=page_id))
        siblings = list(session.run(siblings_query, page_id=page_id))
        
        return {
            'page_id': page_id,
            'parents': [dict(record) for record in parents],
            'children': [dict(record) for record in children],
            'siblings': [dict(record) for record in siblings]
        }


def hybrid_search(query: str, limit: int = 5, score_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Perform hybrid search combining vector similarity and graph structure.
    
    Args:
        query: Search query text
        limit: Maximum number of vector search results
        score_threshold: Minimum similarity score
        
    Returns:
        Dictionary containing vector results and hierarchical context
    """
    # First, get vector search results
    vector_results = vector_search(query, limit, score_threshold)
    
    # Then, for each result, get its hierarchical context
    enhanced_results = []
    for result in vector_results:
        page_id = result['page_id']
        hierarchy = get_page_hierarchy(page_id)
        
        enhanced_result = result.copy()
        enhanced_result['hierarchy'] = hierarchy
        enhanced_results.append(enhanced_result)
    
    return {
        'query': query,
        'vector_results': vector_results,
        'enhanced_results': enhanced_results
    }


def search_by_topic(topic_keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for pages related to specific topics using graph traversal.
    
    Args:
        topic_keywords: List of keywords to search for in page titles
        limit: Maximum number of results
        
    Returns:
        List of related pages with their hierarchical context
    """
    driver = get_neo4j_driver()
    
    # Create a regex pattern for the keywords
    pattern = '(?i).*(' + '|'.join(topic_keywords) + ').*'
    
    with driver.session() as session:
        query = """
        MATCH (page:Page)
        WHERE page.title =~ $pattern
        OPTIONAL MATCH (page)-[:HAS_PARENT]->(parent:Page)
        OPTIONAL MATCH (page)-[:HAS_CHILD]->(child:Page)
        RETURN page.id as page_id, page.title as title, page.url as url,
               collect(DISTINCT parent.title) as parents,
               collect(DISTINCT child.title) as children
        LIMIT $limit
        """
        
        results = session.run(query, pattern=pattern, limit=limit)
        return [dict(record) for record in results]


def main():
    parser = argparse.ArgumentParser(description='Query the SwissFEL knowledge graph')
    parser.add_argument('query', help='Search query text')
    parser.add_argument('--limit', '-l', type=int, default=5, help='Maximum number of results')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Minimum similarity score')
    parser.add_argument('--vector-only', action='store_true', help='Only perform vector search')
    parser.add_argument('--graph-only', action='store_true', help='Only perform graph search')
    parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    try:
        if args.graph_only:
            # Extract keywords from query for graph search
            keywords = args.query.lower().split()
            results = search_by_topic(keywords, args.limit)
            search_type = "Graph Search"
        elif args.vector_only:
            results = vector_search(args.query, args.limit, args.threshold)
            search_type = "Vector Search"
        else:
            results = hybrid_search(args.query, args.limit, args.threshold)
            search_type = "Hybrid Search"
        
        # Output results
        output_data = {
            'search_type': search_type,
            'query': args.query,
            'results': results
        }
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
            
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return 1
    
    finally:
        # Cleanup connections
        if _neo4j_driver:
            _neo4j_driver.close()
    
    return 0


if __name__ == "__main__":
    exit(main())