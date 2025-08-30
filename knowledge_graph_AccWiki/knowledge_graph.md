# PSI Accelerator Knowledge Graph Documentation

## Overview

This document describes the structure and integration strategy for the PSI Accelerator Knowledge Graph, designed for GraphRAG (Graph Retrieval-Augmented Generation) applications. The knowledge graph preserves the hierarchical structure from the original Neo4j database while providing semantic search capabilities through embeddings.

## Data Generation

**Source**: `combined_accelerators_clean.json` - Neo4j knowledge graph export  
**Scraper**: `scrape_wiki_html_testset.py` - Enhanced HTML scraper with graph structure preservation  
**Output Directory**: `out_graph_structure_full/`

### Generation Statistics
- **Articles**: 186 scraped pages
- **Chunks**: 1,081 semantic text blocks  
- **Figures**: 701 multimedia assets
- **Links**: 2,023 cross-references
- **Hierarchy Relationships**: 397 explicit parent-child relationships
- **Embedding Dimension**: 768 (AccPhysBERT model)

## Data Structure

### 1. Articles (`articles.ndjson`)
**Purpose**: Primary entities in the knowledge graph

```json
{
  "article_id": "hipa:uebersicht:article:einleitung",
  "accelerator": "hipa", 
  "tab": "uebersicht",
  "section": null,
  "category": "",
  "title": "Einführung Hochstromprotonenbeschleuniger (Anfänger)",
  "url": "https://acceleratorwiki.psi.ch/wiki/Einf%C3%BChrung_Hochstromprotonenbeschleuniger_(Anf%C3%A4nger)",
  "lang": "de",
  "revid": null,
  "last_modified": "Tue, 25 Feb 2025 01:35:48 GMT",
  "valid": true,
  // Enhanced Graph Metadata
  "node_type": "article",        // root, accelerator, tab, section, category, article
  "order": 1,                    // Hierarchical ordering
  "parent_id": "hipa:category:theorie", // Direct parent reference
  "path_from_root": "PSI Accelerators > HIPA > Übersicht > EINFÜHRUNG > Theorie > Einleitung",
  "depth_level": 4               // Depth in hierarchy (0 = root)
}
```

**Key Features**:
- **Hierarchical Context**: `path_from_root` provides full organizational context
- **Parent References**: Direct links to parent nodes via `parent_id`
- **Type Classification**: Distinguishes between different node types
- **Depth Tracking**: Enables level-based queries

### 2. Content Chunks (`chunks.ndjson`)
**Purpose**: Searchable semantic content blocks with embeddings

```json
{
  "chunk_id": "7380bb86a67ff6698b169b9bf3e0107dde1b1026",
  "article_id": "hipa:uebersicht:article:einleitung",
  "section_title": "Strahlführungskomponenten",
  "section_anchor": "strahlfuehrung",
  "text": "Die Protonen werden durch verschiedene Magnete und Fokussierungselemente geleitet...",
  "html": null,
  "tables_md": "| Component | Function | Location |\n|-----------|----------|----------|\n| Quadrupole | Focusing | Section A |",
  "embedding": [0.02427, -0.04871, 0.07220, ...]  // 768-dimensional vector
}
```

**Key Features**:
- **Semantic Embeddings**: 768-dimensional vectors for similarity search
- **Structured Data**: Tables preserved in Markdown format
- **Section Context**: Links to specific sections within articles
- **Article References**: Connected to parent articles via `article_id`

### 3. Hierarchy Relationships (`hierarchy.ndjson`) ⭐
**Purpose**: Explicit graph structure preservation from Neo4j

```json
{
  "parent_id": "hipa:category:theorie",
  "child_id": "hipa:uebersicht:article:einleitung",
  "relationship_type": "parent_child",
  "parent_type": "category",
  "child_type": "article", 
  "parent_title": "Theorie",
  "child_title": "Einleitung",
  "depth_difference": 1
}
```

**Key Features**:
- **Explicit Edges**: Direct parent-child relationships
- **Type Awareness**: Relationship context between different node types
- **Traversal Support**: Enables hierarchical navigation
- **Multi-hop Queries**: Foundation for complex graph traversals

### 4. Content Links (`links.ndjson`)
**Purpose**: Wiki-style cross-references and semantic relationships

```json
{
  "src_article_id": "hipa:uebersicht:article:einleitung",
  "dst_article_id": "hipa:uebersicht:article:beschleunigerphysik",
  "anchor_text": "Beschleunigerphysik Grundlagen"
}
```

**Key Features**:
- **Cross-References**: Links between related articles
- **Semantic Context**: Anchor text provides relationship meaning
- **Topic Networks**: Creates thematic clusters beyond hierarchy

### 5. Multimedia Assets (`figures.ndjson`)
**Purpose**: Visual elements and their relationships to content

```json
{
  "asset_id": "a4f7b2c8d1e5f9b3c7a8e2f4d6b9c5a1e8f7d3b2",
  "article_id": "hipa:uebersicht:article:einleitung",
  "chunk_id": null,
  "caption": "HIPA Beschleuniger Übersicht Diagramm",
  "url": "https://acceleratorwiki.psi.ch/images/hipa_overview.png",
  "mime": "image/png",
  "width": null,
  "height": null
}
```

**Key Features**:
- **Multimodal Support**: Images, diagrams, PDFs
- **Content Association**: Links to specific articles/chunks  
- **Metadata Preservation**: Captions and technical specifications

## Knowledge Graph Integration Strategy

### Phase 1: Node Creation

#### Article Nodes
```python
def create_article_nodes(articles_data):
    for article in articles_data:
        graph.create_node(
            id=article["article_id"],
            type="ARTICLE",
            properties={
                "title": article["title"],
                "node_type": article["node_type"],
                "accelerator": article["accelerator"],
                "path_from_root": article["path_from_root"],
                "depth_level": article["depth_level"],
                "url": article["url"],
                "last_modified": article["last_modified"]
            }
        )
```

#### Content Nodes
```python
def create_content_nodes(chunks_data):
    for chunk in chunks_data:
        graph.create_node(
            id=chunk["chunk_id"],
            type="CONTENT",
            properties={
                "text": chunk["text"],
                "section_title": chunk["section_title"],
                "embedding": chunk["embedding"],
                "tables_md": chunk.get("tables_md")
            }
        )
```

### Phase 2: Relationship Creation

#### Hierarchical Relationships
```python
def create_hierarchy_edges(hierarchy_data):
    for rel in hierarchy_data:
        graph.create_edge(
            source=rel["parent_id"],
            target=rel["child_id"],
            type="CONTAINS",
            properties={
                "relationship_type": rel["relationship_type"],
                "parent_type": rel["parent_type"],
                "child_type": rel["child_type"],
                "context": f"{rel['parent_title']} contains {rel['child_title']}"
            }
        )
```

#### Content-Article Links
```python
def create_content_article_edges(chunks_data):
    for chunk in chunks_data:
        graph.create_edge(
            source=chunk["chunk_id"],
            target=chunk["article_id"],
            type="PART_OF",
            properties={
                "section": chunk["section_title"]
            }
        )
```

#### Cross-Reference Links
```python
def create_cross_reference_edges(links_data):
    for link in links_data:
        graph.create_edge(
            source=link["src_article_id"],
            target=link["dst_article_id"], 
            type="REFERENCES",
            properties={
                "anchor_text": link["anchor_text"],
                "context": "wiki_reference"
            }
        )
```

### Phase 3: Asset Integration
```python
def create_asset_relationships(figures_data):
    for figure in figures_data:
        # Create asset node
        graph.create_node(
            id=figure["asset_id"],
            type="ASSET",
            properties={
                "caption": figure["caption"],
                "url": figure["url"],
                "mime": figure["mime"]
            }
        )
        
        # Link to article
        graph.create_edge(
            source=figure["asset_id"],
            target=figure["article_id"],
            type="ILLUSTRATES"
        )
```

## GraphRAG Query Patterns

### 1. Hierarchical Context Queries

**Find all content under a specific accelerator section:**
```cypher
MATCH (root:ARTICLE {accelerator: "hipa"})-[:CONTAINS*]->(articles:ARTICLE)
MATCH (articles)-[:PART_OF]-(content:CONTENT)
WHERE content.text CONTAINS "Raumladung"
RETURN articles.path_from_root, content.text, content.section_title
ORDER BY articles.depth_level
```

**Get organizational context for a specific topic:**
```cypher
MATCH (content:CONTENT)
WHERE content.text CONTAINS "Buncher"
MATCH (content)-[:PART_OF]->(article:ARTICLE)
RETURN article.path_from_root, article.title, content.text
```

### 2. Semantic Similarity Queries

**Find related content using embeddings:**
```python
def find_similar_content(query_embedding, threshold=0.8):
    return graph.query(f"""
        MATCH (content:CONTENT)
        WHERE gds.similarity.cosine(content.embedding, {query_embedding}) > {threshold}
        MATCH (content)-[:PART_OF]->(article:ARTICLE)
        RETURN content.text, article.path_from_root, 
               gds.similarity.cosine(content.embedding, {query_embedding}) as similarity
        ORDER BY similarity DESC
        LIMIT 10
    """)
```

### 3. Multi-Modal Queries

**Find articles with both textual content and visual assets:**
```cypher
MATCH (article:ARTICLE)-[:PART_OF]-(content:CONTENT)
MATCH (article)-[:ILLUSTRATES]-(asset:ASSET)
WHERE content.text CONTAINS "Zyklotron" 
  AND asset.caption CONTAINS "diagram"
RETURN article.title, article.path_from_root, content.text, asset.url
```

### 4. Cross-System Analysis

**Find connections between different accelerators:**
```cypher
MATCH (hipa_article:ARTICLE {accelerator: "hipa"})-[:REFERENCES]->(other_article:ARTICLE)
WHERE other_article.accelerator <> "hipa"
RETURN hipa_article.title, other_article.accelerator, other_article.title
```

### 5. Depth-Based Queries

**Get overview vs. detailed information:**
```cypher
// High-level overview (depth 0-2)
MATCH (article:ARTICLE)
WHERE article.depth_level <= 2
MATCH (article)-[:PART_OF]-(content:CONTENT)
RETURN article.path_from_root, content.text

// Detailed technical content (depth 3+)
MATCH (article:ARTICLE) 
WHERE article.depth_level >= 3
MATCH (article)-[:PART_OF]-(content:CONTENT)
WHERE content.text CONTAINS "Prozedur"
RETURN article.path_from_root, content.text
```

## Advanced GraphRAG Capabilities

### 1. Context-Aware Retrieval
- **Hierarchical Context**: Use `path_from_root` to provide organizational context in responses
- **Depth-Aware Responses**: Adjust detail level based on query specificity
- **Multi-Level Reasoning**: Traverse hierarchy for broader/narrower concepts

### 2. Semantic Enhancement
- **Embedding-Based Search**: Leverage 768-dimensional AccPhysBERT embeddings
- **Contextual Similarity**: Combine semantic similarity with graph structure
- **Cross-Domain Connections**: Find related concepts across accelerator systems

### 3. Multimodal Integration
- **Visual Context**: Include relevant diagrams and images in responses
- **Technical Documentation**: Link procedures to visual aids
- **Comprehensive Understanding**: Combine text, tables, and images

### 4. Procedural Knowledge
- **Step-by-Step Guidance**: Navigate procedural content with hierarchy
- **Safety Protocols**: Access safety information with proper context
- **Troubleshooting**: Follow diagnostic paths through the knowledge structure

## Implementation Recommendations

### 1. Graph Database Setup
- Use Neo4j or similar graph database for optimal performance
- Index key properties: `accelerator`, `node_type`, `path_from_root`
- Store embeddings in vector format for similarity search

### 2. Query Optimization
- Cache frequent hierarchical patterns
- Pre-compute common path traversals
- Use graph algorithms for complex relationship queries

### 3. Real-time Updates
- Monitor wiki changes for content updates
- Implement incremental embedding updates
- Maintain graph consistency during updates

### 4. Quality Assurance
- Validate hierarchy consistency
- Monitor embedding quality
- Track query performance and relevance

This knowledge graph structure provides a robust foundation for GraphRAG applications, enabling both precise technical queries and broad conceptual exploration across the PSI accelerator systems.