#!/usr/bin/env python3
"""
Combine all PSI accelerator knowledge graphs into one unified tree structure.
Reads HIPA, ProScan, SLS, and SwissFEL JSON files and creates a single combined graph.
"""

import json
import os
import sys
from pathlib import Path

def load_accelerator_files(directory):
    """Load all accelerator JSON files from the specified directory."""
    accelerators = []
    
    # Define expected files in order
    expected_files = ['HIPA.json', 'ProScan.json', 'SLS.json', 'SwissFEL.json']
    
    for filename in expected_files:
        filepath = Path(directory) / filename
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    accelerators.append(data)
                    print(f"✓ Loaded {filename}: {data.get('title', 'Unknown')} ({len(data.get('children', []))} top-level sections)")
            except json.JSONDecodeError as e:
                print(f"✗ Error loading {filename}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"✗ Error reading {filename}: {e}", file=sys.stderr)
        else:
            print(f"⚠ File not found: {filename}", file=sys.stderr)
    
    return accelerators

def count_total_nodes(node):
    """Recursively count all nodes in a tree structure."""
    count = 1  # Count current node
    for child in node.get('children', []):
        count += count_total_nodes(child)
    return count

def create_combined_graph(accelerators):
    """Create a unified graph structure combining all accelerators."""
    
    # Create root node
    combined_graph = {
        "id": "psi_accelerators",
        "title": "PSI Accelerators",
        "type": "root",
        "description": "Combined knowledge graph of all PSI accelerator facilities",
        "lang": "de",
        "source": "BeschleunigerWiki",
        "version": "1.0",
        "created": "2025",
        "accelerators": len(accelerators),
        "children": []
    }
    
    # Add each accelerator as a top-level child
    total_nodes = 1  # Root node
    for accelerator in accelerators:
        combined_graph["children"].append(accelerator)
        accelerator_nodes = count_total_nodes(accelerator)
        total_nodes += accelerator_nodes
        print(f"  Added {accelerator['title']}: {accelerator_nodes} nodes")
    
    combined_graph["total_nodes"] = total_nodes
    
    return combined_graph

def save_combined_graph(graph, output_file):
    """Save the combined graph to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        print(f"✓ Combined graph saved to: {output_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving combined graph: {e}", file=sys.stderr)
        return False

def print_summary(graph):
    """Print a summary of the combined graph."""
    print("\n" + "="*60)
    print("COMBINED ACCELERATOR KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    print(f"Title: {graph['title']}")
    print(f"Total Accelerators: {graph['accelerators']}")
    print(f"Total Nodes: {graph['total_nodes']}")
    print(f"Language: {graph['lang']}")
    print(f"Source: {graph['source']}")
    print()
    
    print("Accelerators included:")
    for i, accelerator in enumerate(graph['children'], 1):
        accelerator_nodes = count_total_nodes(accelerator)
        sections = len(accelerator.get('children', []))
        print(f"  {i}. {accelerator['title']} ({accelerator['id']})")
        print(f"     └─ {sections} main sections, {accelerator_nodes} total nodes")
    
    print("="*60)

def main():
    # Directory containing the accelerator JSON files
    input_directory = Path(__file__).parent / "knowledge_graph_AccWiki"
    output_file = Path(__file__).parent / "combined_accelerators.json"
    
    print("PSI Accelerator Knowledge Graph Combiner")
    print("-" * 50)
    
    # Check if input directory exists
    if not input_directory.exists():
        print(f"✗ Input directory not found: {input_directory}", file=sys.stderr)
        return 1
    
    print(f"Reading accelerator files from: {input_directory}")
    
    # Load all accelerator files
    accelerators = load_accelerator_files(input_directory)
    
    if not accelerators:
        print("✗ No accelerator files found or loaded successfully", file=sys.stderr)
        return 1
    
    print(f"\nSuccessfully loaded {len(accelerators)} accelerator(s)")
    
    # Create combined graph
    print("\nCombining accelerators into unified graph...")
    combined_graph = create_combined_graph(accelerators)
    
    # Save combined graph
    if save_combined_graph(combined_graph, output_file):
        print_summary(combined_graph)
        
        # Also create a pretty-printed version
        pretty_output = output_file.with_stem(output_file.stem + "_pretty")
        try:
            with open(pretty_output, 'w', encoding='utf-8') as f:
                json.dump(combined_graph, f, indent=4, ensure_ascii=False, sort_keys=True)
            print(f"✓ Pretty-printed version saved to: {pretty_output}")
        except Exception as e:
            print(f"⚠ Could not save pretty version: {e}", file=sys.stderr)
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())