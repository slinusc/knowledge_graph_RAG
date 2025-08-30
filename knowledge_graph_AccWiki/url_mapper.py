#!/usr/bin/env python3
"""
URL mapper that only adds 'url' and 'valid' (bool) to nodes.
"""

import json
import requests
import time
import copy
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

# Portal entry points
PORTAL_URLS = {
    'hipa': 'https://acceleratorwiki.psi.ch/wiki/Portal:HIPA',
    'proscan': 'https://acceleratorwiki.psi.ch/wiki/Portal:PROSCAN',
    'swissfel': 'https://acceleratorwiki.psi.ch/wiki/Portal:SwissFEL',
    'sls': 'https://acceleratorwiki.psi.ch/wiki/Portal:SLS'
}

BASE_WIKI_URL = 'https://acceleratorwiki.psi.ch/wiki/'

class CleanMapper:
    def __init__(self, delay=0.3):
        self.session = requests.Session()
        self.delay = delay
        self.discovered_links = {}  # accelerator -> {title -> url}
        self.visited_urls = set()
        self.failed_urls = set()
        
    def normalize_title_for_matching(self, title):
        """Normalize title for matching against tree nodes."""
        variations = [
            title.lower().strip(),
            title.lower().strip().replace(' ', ''),
            title.lower().strip().replace('-', ' '),
            title.lower().strip().replace('_', ' '),
            title.lower().strip().replace('/', ' '),
            title.lower().strip().replace('&', 'und'),
            # German character replacements
            title.lower().strip().replace('ü', 'u').replace('ä', 'a').replace('ö', 'o').replace('ß', 's'),
            # Remove common prefixes
            re.sub(r'^(portal:|kategorie:|category:)', '', title.lower().strip()),
            # Clean up common patterns
            re.sub(r'\s+', ' ', title.lower().strip()),
        ]
        return list(set(variations))  # Remove duplicates
    
    def extract_wiki_links_from_page(self, url, base_accelerator):
        """Extract all internal wiki links from a page."""
        if url in self.visited_urls or url in self.failed_urls:
            return []
        
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(self.delay)
            
            if response.status_code != 200:
                self.failed_urls.add(url)
                return []
            
            # Check for dead link patterns
            content_lower = response.text.lower()
            dead_indicators = [
                'diese seite enthält momentan noch keinen text',
                'du bist auch nicht dazu berechtigt'
            ]
            
            if any(indicator in content_lower for indicator in dead_indicators):
                self.failed_urls.add(url)
                return []
            
            self.visited_urls.add(url)
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find main content div
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if not content_div:
                content_div = soup
            
            # Extract all internal wiki links
            links = []
            for a_tag in content_div.find_all('a', href=True):
                href = a_tag['href']
                
                # Only internal wiki links
                if href.startswith('/wiki/'):
                    full_url = 'https://acceleratorwiki.psi.ch' + href
                    link_text = a_tag.get_text(strip=True)
                    
                    if link_text and len(link_text) > 1:
                        # Skip navigation and special pages
                        if any(skip in href.lower() for skip in [
                            'spezial:', 'special:', 'datei:', 'file:', 
                            'kategorie:', 'category:', 'diskussion:', 'talk:',
                            'hilfe:', 'help:', 'benutzer:', 'user:'
                        ]):
                            continue
                        
                        links.append({
                            'url': full_url,
                            'text': link_text,
                            'href': href
                        })
                        
                        # Store in discovered links
                        if base_accelerator not in self.discovered_links:
                            self.discovered_links[base_accelerator] = {}
                        
                        # Store multiple variations of the title
                        title_variations = self.normalize_title_for_matching(link_text)
                        for variation in title_variations:
                            self.discovered_links[base_accelerator][variation] = full_url
            
            return links
            
        except Exception as e:
            self.failed_urls.add(url)
            return []
    
    def crawl_accelerator_recursively(self, accelerator_id, start_url, max_depth=3):
        """Recursively crawl an accelerator portal to discover all links."""
        urls_to_crawl = [(start_url, 0)]  # (url, depth)
        crawled_urls = set()
        
        while urls_to_crawl:
            current_url, depth = urls_to_crawl.pop(0)
            
            if depth >= max_depth or current_url in crawled_urls:
                continue
                
            crawled_urls.add(current_url)
            links = self.extract_wiki_links_from_page(current_url, accelerator_id)
            
            # Add sub-portal links for deeper crawling
            for link in links:
                link_url = link['url']
                link_href = link['href']
                
                # Only crawl portal subpages and direct accelerator pages
                if (f"portal:{accelerator_id}" in link_href.lower() or 
                    accelerator_id.lower() in link_href.lower() or
                    any(keyword in link_href.lower() for keyword in [
                        'einführung', 'training', 'procedure', 'prozedur', 
                        'betrieb', 'operation', 'system'
                    ])):
                    if link_url not in crawled_urls and depth < max_depth - 1:
                        urls_to_crawl.append((link_url, depth + 1))
        
        return len(crawled_urls)
    
    def find_best_match_for_node(self, node, accelerator_id):
        """Find the best URL match for a node using discovered links."""
        node_title = node.get('title', '').lower().strip()
        node_type = node.get('type', 'unknown')
        
        # Get discovered links for this accelerator
        acc_links = self.discovered_links.get(accelerator_id, {})
        
        # Generate title variations to match against
        title_variations = self.normalize_title_for_matching(node_title)
        
        # Try exact matches first
        for variation in title_variations:
            if variation in acc_links:
                return acc_links[variation]
        
        # For articles, try partial matching
        if node_type == 'article':
            node_words = set(node_title.split())
            best_match = None
            best_score = 0
            
            for discovered_title, url in acc_links.items():
                discovered_words = set(discovered_title.split())
                overlap = len(node_words & discovered_words)
                
                if overlap > best_score and overlap > 0:
                    best_score = overlap
                    best_match = url
            
            if best_score >= 1:  # At least one word match
                return best_match
        
        # For sections/categories, look for containing matches
        if node_type in ['section', 'category']:
            for discovered_title, url in acc_links.items():
                # Check if any words from node title appear in discovered title
                if any(word in discovered_title for word in node_title.split() if len(word) > 2):
                    return url
        
        return None
    
    def validate_url(self, url):
        """Validate a single URL and return True/False."""
        try:
            response = self.session.get(url, timeout=10)
            time.sleep(self.delay)
            
            if response.status_code != 200:
                return False
            
            # Check for dead link content
            content_lower = response.text.lower()
            dead_indicators = [
                'diese seite enthält momentan noch keinen text',
                'du bist auch nicht dazu berechtigt',
                'does not exist',
                'page not found'
            ]
            
            return not any(indicator in content_lower for indicator in dead_indicators)
            
        except Exception as e:
            return False
    
    def map_urls_to_graph(self, graph):
        """Map discovered URLs to all nodes in the graph with only 'url' and 'valid' fields."""
        def process_node(node, parent=None):
            processed_node = copy.deepcopy(node)
            
            # Extract accelerator ID
            accelerator_id = node['id'].split(':')[0] if ':' in node['id'] else None
            node_type = node.get('type', 'unknown')
            
            # Handle root node
            if node_type == 'root':
                processed_node['url'] = BASE_WIKI_URL
                processed_node['valid'] = True
            
            # Handle accelerator nodes
            elif node_type == 'accelerator':
                portal_url = PORTAL_URLS.get(accelerator_id, BASE_WIKI_URL)
                processed_node['url'] = portal_url
                processed_node['valid'] = self.validate_url(portal_url)
                
            # Handle other node types
            else:
                if accelerator_id and accelerator_id in self.discovered_links:
                    best_url = self.find_best_match_for_node(node, accelerator_id)
                    
                    if best_url:
                        processed_node['url'] = best_url
                        processed_node['valid'] = self.validate_url(best_url)
                    # If no URL found, don't add url/valid fields
            
            # Process children
            if 'children' in processed_node:
                processed_children = []
                for child in processed_node['children']:
                    processed_child = process_node(child, node)
                    processed_children.append(processed_child)
                processed_node['children'] = processed_children
            
            return processed_node
        
        return process_node(graph)

def load_graph(filepath):
    """Load knowledge graph from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def save_graph(graph, filepath):
    """Save knowledge graph to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving graph: {e}")
        return False

def main():
    print("Clean URL Mapper - Adding only 'url' and 'valid' fields")
    print("=" * 50)
    
    # File paths
    input_file = Path(__file__).parent / "combined_accelerators.json"
    output_file = Path(__file__).parent / "combined_accelerators_clean.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    # Load graph
    print(f"Loading graph from: {input_file}")
    graph = load_graph(input_file)
    if not graph:
        return 1
    
    # Create mapper and crawl all accelerators
    mapper = CleanMapper()
    
    print("🚀 Crawling accelerator portals...")
    
    total_pages_crawled = 0
    for acc_id, portal_url in PORTAL_URLS.items():
        print(f"  Crawling {acc_id.upper()}...")
        pages_crawled = mapper.crawl_accelerator_recursively(acc_id, portal_url)
        total_pages_crawled += pages_crawled
        print(f"  ✅ {len(mapper.discovered_links.get(acc_id, {}))} links discovered")
    
    # Map URLs to graph
    print("\n🔗 Mapping URLs and validating...")
    mapped_graph = mapper.map_urls_to_graph(graph)
    
    # Count results
    def count_stats(node):
        stats = {'total': 0, 'with_urls': 0, 'valid': 0}
        stats['total'] += 1
        
        if 'url' in node:
            stats['with_urls'] += 1
            if node.get('valid', False):
                stats['valid'] += 1
        
        for child in node.get('children', []):
            child_stats = count_stats(child)
            for key in stats:
                stats[key] += child_stats[key]
        
        return stats
    
    stats = count_stats(mapped_graph)
    
    # Save clean graph
    if save_graph(mapped_graph, output_file):
        print(f"✅ Clean graph saved to: {output_file}")
        print(f"📊 Results:")
        print(f"   Total nodes: {stats['total']}")
        print(f"   Nodes with URLs: {stats['with_urls']}")
        print(f"   Valid URLs: {stats['valid']}")
        print(f"   Success rate: {stats['valid']/stats['with_urls']*100:.1f}%" if stats['with_urls'] > 0 else "   Success rate: 0%")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())