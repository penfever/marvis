#!/usr/bin/env python3
"""
Quick script to validate semantic data structures across all CC18 datasets.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_semantic_structures():
    """Analyze all semantic data files to understand their structures."""
    
    data_dir = Path("data/cc18_semantic")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Statistics
    total_files = 0
    structures_found = defaultdict(list)
    field_presence = defaultdict(int)
    
    # Fields we care about for TabLLM
    target_fields = [
        'feature_description',
        'feature_descriptions', 
        'columns',
        'feature_names'
    ]
    
    print("🔍 Scanning CC18 semantic data files...")
    
    for json_file in data_dir.glob("*.json"):
        total_files += 1
        task_id = json_file.stem
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check which target fields are present
            present_fields = []
            for field in target_fields:
                if field in data:
                    field_presence[field] += 1
                    present_fields.append(field)
            
            # Categorize by structure pattern
            structure_key = tuple(sorted(present_fields))
            structures_found[structure_key].append(task_id)
            
            # Quick validation for feature count
            feature_count = None
            if 'features' in data:
                feature_count = data['features']
            
            # Count actual semantic features
            semantic_count = 0
            if 'feature_description' in data:
                semantic_count = len(data['feature_description'])
            elif 'feature_descriptions' in data:
                semantic_count = len(data['feature_descriptions'])
            elif 'columns' in data:
                semantic_count = len(data['columns'])
            elif 'feature_names' in data:
                semantic_count = len(data['feature_names'])
            
            # Check for mismatches
            if feature_count and semantic_count and feature_count != semantic_count:
                print(f"⚠️  Task {task_id}: Feature count mismatch - dataset has {feature_count}, semantic has {semantic_count}")
                
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
    
    print(f"\n📊 Analysis Results:")
    print(f"Total files analyzed: {total_files}")
    
    print(f"\n📈 Field Presence:")
    for field, count in sorted(field_presence.items()):
        percentage = (count / total_files) * 100
        print(f"  {field}: {count}/{total_files} ({percentage:.1f}%)")
    
    print(f"\n🏗️  Structure Patterns:")
    for structure, task_ids in sorted(structures_found.items()):
        print(f"  {structure if structure else '(no semantic fields)'}: {len(task_ids)} datasets")
        if len(task_ids) <= 10:  # Show task IDs for small groups
            print(f"    Task IDs: {sorted(task_ids)}")
        else:
            print(f"    Task IDs: {sorted(task_ids)[:5]}... (and {len(task_ids)-5} more)")
    
    # Identify problematic cases
    print(f"\n🚨 Issues to Address:")
    
    # No semantic fields at all
    no_semantic = structures_found.get((), [])
    if no_semantic:
        print(f"  ❌ {len(no_semantic)} datasets have NO semantic feature fields")
        print(f"     Task IDs: {sorted(no_semantic)[:10]}{'...' if len(no_semantic) > 10 else ''}")
    
    # Multiple conflicting fields
    multiple_fields = [structure for structure in structures_found.keys() if len(structure) > 1]
    if multiple_fields:
        print(f"  ⚠️  {len(multiple_fields)} structure patterns have multiple semantic fields")
        for structure in multiple_fields:
            task_ids = structures_found[structure]
            print(f"     {structure}: {len(task_ids)} datasets")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    # Find the most common structure
    if structures_found:
        most_common_structure = max(structures_found.keys(), key=lambda k: len(structures_found[k]))
        most_common_count = len(structures_found[most_common_structure])
        print(f"  🎯 Most common structure: {most_common_structure} ({most_common_count} datasets)")
        
        if most_common_structure:
            print(f"  ✅ Standardize all datasets to use: {most_common_structure[0]}")
        
        # Check if we need normalization
        if len(structures_found) > 1:
            print(f"  🔧 Need to normalize {total_files - most_common_count} datasets to common structure")
    
    return structures_found, field_presence, total_files

if __name__ == "__main__":
    analyze_semantic_structures()