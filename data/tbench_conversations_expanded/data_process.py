#!/usr/bin/env python3
"""
Process JSONL files from tbench_conversations_expanded format to LLaMA-Factory training format.

This script processes three JSONL files:
- all_traces.jsonl (335 entries)
- train.jsonl (284 entries)  
- val.jsonl (51 entries)

Transforms from:
{
    "task_id": "...",
    "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}

To:
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}

Output files:
- all_traces_processed.jsonl
- train_processed.jsonl
- val_processed.jsonl
"""

import json
import sys
from pathlib import Path


def process_jsonl_file(input_path: Path, output_path: Path):
    """
    Process a single JSONL file, converting from tbench format to messages format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the input JSON
                data = json.loads(line)
                
                # Extract conversations array
                if 'conversations' in data:
                    conversations = data['conversations']
                elif 'messages' in data:
                    # Already in the right format
                    conversations = data['messages']
                else:
                    print(f"Warning: Line {line_num} has no 'conversations' or 'messages' field", file=sys.stderr)
                    continue
                
                # Create the output format
                output_data = {
                    "messages": conversations
                }
                
                # Write to output file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    return processed_count


def main():
    # Define paths
    base_dir = Path(__file__).parent
    
    input_files = [
        base_dir / "all_traces.jsonl",
        base_dir / "train.jsonl",
        base_dir / "val.jsonl"
    ]
    
    output_files = [
        base_dir / "all_traces_processed.jsonl",
        base_dir / "train_processed.jsonl",
        base_dir / "val_processed.jsonl"
    ]
    
    # Process each file
    for input_path, output_path in zip(input_files, output_files):
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}", file=sys.stderr)
            continue
        
        print(f"Processing {input_path.name}...")
        count = process_jsonl_file(input_path, output_path)
        print(f"  ✓ Processed {count} entries → {output_path.name}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

