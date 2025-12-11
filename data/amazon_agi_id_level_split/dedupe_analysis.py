#!/usr/bin/env python3
"""
Deduplication Analysis Report for train.jsonl and val.jsonl
Analyzes duplicates using 4 strategies.
"""

import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

def load_jsonl(path):
    """Load JSONL file and return list of (line_num, raw_line, parsed_obj)"""
    entries = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                entries.append((i, line, json.loads(line)))
    return entries

def get_base_task_id(task_id):
    """Extract base UUID from task_id like 'UUID_variation_N'"""
    if '_variation_' in task_id:
        return task_id.rsplit('_variation_', 1)[0]
    return task_id

def hash_content(obj):
    """Hash the conversations content (ignoring task_id)"""
    content = json.dumps(obj.get('conversations', []), sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def hash_line(line):
    """Hash the entire line for exact duplicate detection"""
    return hashlib.md5(line.encode()).hexdigest()

def analyze_exact_duplicates(train_entries, val_entries):
    """Strategy 1: Find exact duplicate lines"""
    print("\n" + "="*70)
    print("STRATEGY 1: EXACT DUPLICATES (identical JSON lines)")
    print("="*70)
    
    train_hashes = {hash_line(line): (i, obj) for i, line, obj in train_entries}
    val_hashes = {hash_line(line): (i, obj) for i, line, obj in val_entries}
    
    # Within train
    train_line_counts = Counter(hash_line(line) for i, line, obj in train_entries)
    train_internal_dupes = sum(c - 1 for c in train_line_counts.values() if c > 1)
    
    # Within val
    val_line_counts = Counter(hash_line(line) for i, line, obj in val_entries)
    val_internal_dupes = sum(c - 1 for c in val_line_counts.values() if c > 1)
    
    # Cross-file
    cross_dupes = set(train_hashes.keys()) & set(val_hashes.keys())
    
    print(f"\n  Train file ({len(train_entries)} entries):")
    print(f"    - Internal exact duplicates: {train_internal_dupes}")
    if train_internal_dupes > 0:
        for h, c in train_line_counts.items():
            if c > 1:
                print(f"      • task_id '{train_hashes[h][1]['task_id']}' appears {c} times")
    
    print(f"\n  Val file ({len(val_entries)} entries):")
    print(f"    - Internal exact duplicates: {val_internal_dupes}")
    if val_internal_dupes > 0:
        for h, c in val_line_counts.items():
            if c > 1:
                print(f"      • task_id '{val_hashes[h][1]['task_id']}' appears {c} times")
    
    print(f"\n  Cross-file (train ∩ val):")
    print(f"    - Exact duplicate lines: {len(cross_dupes)}")
    if cross_dupes:
        for h in list(cross_dupes)[:5]:
            print(f"      • task_id '{train_hashes[h][1]['task_id']}'")
        if len(cross_dupes) > 5:
            print(f"      ... and {len(cross_dupes) - 5} more")
    
    return {
        'train_internal': train_internal_dupes,
        'val_internal': val_internal_dupes,
        'cross_file': len(cross_dupes)
    }

def analyze_task_id_duplicates(train_entries, val_entries):
    """Strategy 2: Find duplicate task_ids"""
    print("\n" + "="*70)
    print("STRATEGY 2: TASK_ID DUPLICATES")
    print("="*70)
    
    train_ids = {obj['task_id']: i for i, line, obj in train_entries}
    val_ids = {obj['task_id']: i for i, line, obj in val_entries}
    
    # Within train
    train_id_counts = Counter(obj['task_id'] for i, line, obj in train_entries)
    train_internal_dupes = sum(c - 1 for c in train_id_counts.values() if c > 1)
    
    # Within val
    val_id_counts = Counter(obj['task_id'] for i, line, obj in val_entries)
    val_internal_dupes = sum(c - 1 for c in val_id_counts.values() if c > 1)
    
    # Cross-file
    cross_dupes = set(train_ids.keys()) & set(val_ids.keys())
    
    print(f"\n  Train file ({len(train_entries)} entries, {len(set(train_ids.keys()))} unique task_ids):")
    print(f"    - Internal task_id duplicates: {train_internal_dupes}")
    if train_internal_dupes > 0:
        for tid, c in train_id_counts.items():
            if c > 1:
                print(f"      • '{tid}' appears {c} times")
    
    print(f"\n  Val file ({len(val_entries)} entries, {len(set(val_ids.keys()))} unique task_ids):")
    print(f"    - Internal task_id duplicates: {val_internal_dupes}")
    if val_internal_dupes > 0:
        for tid, c in val_id_counts.items():
            if c > 1:
                print(f"      • '{tid}' appears {c} times")
    
    print(f"\n  Cross-file (train ∩ val):")
    print(f"    - Overlapping task_ids: {len(cross_dupes)}")
    if cross_dupes:
        for tid in list(cross_dupes)[:5]:
            print(f"      • '{tid}'")
        if len(cross_dupes) > 5:
            print(f"      ... and {len(cross_dupes) - 5} more")
    
    return {
        'train_internal': train_internal_dupes,
        'val_internal': val_internal_dupes,
        'cross_file': len(cross_dupes),
        'cross_file_ids': cross_dupes
    }

def analyze_base_task_duplicates(train_entries, val_entries):
    """Strategy 3: Find duplicate base task UUIDs (ignoring _variation_N suffix)"""
    print("\n" + "="*70)
    print("STRATEGY 3: BASE TASK UUID DUPLICATES (ignoring _variation_N)")
    print("="*70)
    
    train_base_ids = defaultdict(list)
    for i, line, obj in train_entries:
        base = get_base_task_id(obj['task_id'])
        train_base_ids[base].append(obj['task_id'])
    
    val_base_ids = defaultdict(list)
    for i, line, obj in val_entries:
        base = get_base_task_id(obj['task_id'])
        val_base_ids[base].append(obj['task_id'])
    
    # Cross-file overlap
    cross_dupes = set(train_base_ids.keys()) & set(val_base_ids.keys())
    
    print(f"\n  Train file:")
    print(f"    - Total entries: {len(train_entries)}")
    print(f"    - Unique base task UUIDs: {len(train_base_ids)}")
    print(f"    - Entries that are variations of same base task:")
    variation_counts = Counter(len(v) for v in train_base_ids.values())
    for count, num_tasks in sorted(variation_counts.items()):
        if count > 1:
            print(f"      • {num_tasks} base tasks have {count} variations each")
    
    print(f"\n  Val file:")
    print(f"    - Total entries: {len(val_entries)}")
    print(f"    - Unique base task UUIDs: {len(val_base_ids)}")
    variation_counts = Counter(len(v) for v in val_base_ids.values())
    for count, num_tasks in sorted(variation_counts.items()):
        if count > 1:
            print(f"      • {num_tasks} base tasks have {count} variations each")
    
    print(f"\n  Cross-file (train ∩ val by base UUID):")
    print(f"    - Overlapping base task UUIDs: {len(cross_dupes)}")
    if cross_dupes:
        total_train_affected = sum(len(train_base_ids[b]) for b in cross_dupes)
        total_val_affected = sum(len(val_base_ids[b]) for b in cross_dupes)
        print(f"    - Train entries affected: {total_train_affected}")
        print(f"    - Val entries affected: {total_val_affected}")
        print(f"\n    Examples of overlap:")
        for base in list(cross_dupes)[:3]:
            print(f"      Base UUID: {base}")
            print(f"        Train variations: {train_base_ids[base]}")
            print(f"        Val variations: {val_base_ids[base]}")
    
    return {
        'train_unique_base': len(train_base_ids),
        'val_unique_base': len(val_base_ids),
        'cross_file': len(cross_dupes),
        'cross_file_base_ids': cross_dupes,
        'train_base_ids': train_base_ids,
        'val_base_ids': val_base_ids
    }

def analyze_content_duplicates(train_entries, val_entries):
    """Strategy 4: Find duplicate conversation content (ignoring task_id)"""
    print("\n" + "="*70)
    print("STRATEGY 4: CONTENT-BASED DUPLICATES (same conversations, ignoring task_id)")
    print("="*70)
    
    train_content = defaultdict(list)
    for i, line, obj in train_entries:
        h = hash_content(obj)
        train_content[h].append((i, obj['task_id']))
    
    val_content = defaultdict(list)
    for i, line, obj in val_entries:
        h = hash_content(obj)
        val_content[h].append((i, obj['task_id']))
    
    # Within train
    train_internal_dupes = sum(len(v) - 1 for v in train_content.values() if len(v) > 1)
    
    # Within val
    val_internal_dupes = sum(len(v) - 1 for v in val_content.values() if len(v) > 1)
    
    # Cross-file
    cross_dupes = set(train_content.keys()) & set(val_content.keys())
    
    print(f"\n  Train file ({len(train_entries)} entries):")
    print(f"    - Unique conversation hashes: {len(train_content)}")
    print(f"    - Internal content duplicates: {train_internal_dupes}")
    if train_internal_dupes > 0:
        print(f"    - Entries with identical content (different task_ids):")
        for h, entries in train_content.items():
            if len(entries) > 1:
                task_ids = [tid for _, tid in entries]
                print(f"      • {len(entries)} entries share content: {task_ids[:3]}{'...' if len(task_ids) > 3 else ''}")
    
    print(f"\n  Val file ({len(val_entries)} entries):")
    print(f"    - Unique conversation hashes: {len(val_content)}")
    print(f"    - Internal content duplicates: {val_internal_dupes}")
    if val_internal_dupes > 0:
        for h, entries in val_content.items():
            if len(entries) > 1:
                task_ids = [tid for _, tid in entries]
                print(f"      • {len(entries)} entries share content: {task_ids[:3]}{'...' if len(task_ids) > 3 else ''}")
    
    print(f"\n  Cross-file (train ∩ val by content):")
    print(f"    - Entries with identical conversation content: {len(cross_dupes)}")
    if cross_dupes:
        total_train = sum(len(train_content[h]) for h in cross_dupes)
        total_val = sum(len(val_content[h]) for h in cross_dupes)
        print(f"    - Train entries affected: {total_train}")
        print(f"    - Val entries affected: {total_val}")
        print(f"\n    Examples:")
        for h in list(cross_dupes)[:3]:
            print(f"      Content hash: {h[:12]}...")
            print(f"        Train task_ids: {[tid for _, tid in train_content[h]]}")
            print(f"        Val task_ids: {[tid for _, tid in val_content[h]]}")
    
    return {
        'train_unique_content': len(train_content),
        'val_unique_content': len(val_content),
        'train_internal': train_internal_dupes,
        'val_internal': val_internal_dupes,
        'cross_file': len(cross_dupes)
    }

def main():
    base_path = Path(__file__).parent
    train_path = base_path / 'train.jsonl'
    val_path = base_path / 'val.jsonl'
    
    print("\n" + "#"*70)
    print("# DEDUPLICATION ANALYSIS REPORT")
    print(f"# Train file: {train_path}")
    print(f"# Val file: {val_path}")
    print("#"*70)
    
    train_entries = load_jsonl(train_path)
    val_entries = load_jsonl(val_path)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Train entries: {len(train_entries)}")
    print(f"   Val entries: {len(val_entries)}")
    print(f"   Total: {len(train_entries) + len(val_entries)}")
    
    results = {}
    results['exact'] = analyze_exact_duplicates(train_entries, val_entries)
    results['task_id'] = analyze_task_id_duplicates(train_entries, val_entries)
    results['base_task'] = analyze_base_task_duplicates(train_entries, val_entries)
    results['content'] = analyze_content_duplicates(train_entries, val_entries)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
┌─────────────────────────────────────┬────────────────┬────────────────┬────────────────┐
│ Strategy                            │ Train Internal │ Val Internal   │ Cross-file     │
├─────────────────────────────────────┼────────────────┼────────────────┼────────────────┤
│ 1. Exact duplicates                 │ {results['exact']['train_internal']:>14} │ {results['exact']['val_internal']:>14} │ {results['exact']['cross_file']:>14} │
│ 2. Task ID duplicates               │ {results['task_id']['train_internal']:>14} │ {results['task_id']['val_internal']:>14} │ {results['task_id']['cross_file']:>14} │
│ 3. Base task UUID overlap           │            N/A │            N/A │ {results['base_task']['cross_file']:>14} │
│ 4. Content duplicates               │ {results['content']['train_internal']:>14} │ {results['content']['val_internal']:>14} │ {results['content']['cross_file']:>14} │
└─────────────────────────────────────┴────────────────┴────────────────┴────────────────┘
""")
    
    print("KEY FINDINGS:")
    if results['base_task']['cross_file'] > 0:
        print(f"  ⚠️  {results['base_task']['cross_file']} base tasks have variations in BOTH train and val")
        print(f"      This could cause data leakage during training!")
    if results['content']['cross_file'] > 0:
        print(f"  ⚠️  {results['content']['cross_file']} entries have identical content in both files")
    if results['exact']['cross_file'] > 0:
        print(f"  ⚠️  {results['exact']['cross_file']} lines are exactly identical in both files")
    if all(v == 0 for k, v in results['exact'].items()):
        print(f"  ✅ No exact duplicates found")
    if results['base_task']['cross_file'] == 0 and results['content']['cross_file'] == 0:
        print(f"  ✅ Clean train/val split - no data leakage detected")
    
    print()

if __name__ == '__main__':
    main()





