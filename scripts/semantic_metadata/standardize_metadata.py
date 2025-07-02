#!/usr/bin/env python3
"""
Universal metadata standardization script for MARVIS.

This script consolidates and replaces:
- scripts/enhance_metadata_columns.py 
- The validation script we just wrote

It standardizes all semantic metadata files across different benchmarks
to use a consistent 'columns' structure for both classification and regression tasks.

Universal Column Structure:
{
    "columns": [
        {
            "name": "feature_name",
            "semantic_description": "human readable description", 
            "data_type": "numerical|categorical|ordinal|text",
            "is_target": false
        },
        ...
    ],
    "target": {
        "name": "target_column_name",
        "semantic_description": "target description",
        "data_type": "numerical|categorical|ordinal", 
        "task_type": "classification|regression",
        "classes": [...] or null,
        "is_target": true
    },
    "metadata": {...}  # Preserve original benchmark-specific metadata
}
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

def discover_all_benchmarks(data_dir: Path) -> Dict[str, Path]:
    """Discover all benchmark directories in the data folder."""
    benchmarks = {}
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return benchmarks
    
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if it contains metadata files
            json_files = list(item.glob("*.json"))
            if json_files:
                benchmarks[item.name] = item
                print(f"📁 Found benchmark: {item.name} ({len(json_files)} files)")
    
    return benchmarks

def analyze_current_structure(metadata_file: Path) -> Dict[str, Any]:
    """Analyze the current structure of a metadata file."""
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        analysis = {
            'file': metadata_file.name,
            'has_columns': 'columns' in data,
            'has_feature_description': 'feature_description' in data,
            'has_feature_descriptions': 'feature_descriptions' in data,
            'has_feature_names': 'feature_names' in data,
            'has_target_info': any(key in data for key in ['target', 'target_type', 'target_values']),
            'feature_count': data.get('features', 0),
            'raw_keys': list(data.keys())
        }
        
        # Count semantic features
        semantic_count = 0
        if 'columns' in data:
            semantic_count = len(data['columns'])
        elif 'feature_description' in data:
            semantic_count = len(data['feature_description'])
        elif 'feature_descriptions' in data:
            semantic_count = len(data['feature_descriptions'])
        elif 'feature_names' in data:
            semantic_count = len(data['feature_names'])
        
        analysis['semantic_count'] = semantic_count
        
        return analysis
        
    except Exception as e:
        return {'file': metadata_file.name, 'error': str(e)}

def standardize_metadata_file(metadata_file: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Standardize a single metadata file to universal structure."""
    
    try:
        with open(metadata_file, 'r') as f:
            original_data = json.load(f)
        
        # Create standardized structure
        standardized = {
            # Preserve original metadata
            'dataset': original_data.get('dataset', metadata_file.stem),
            'description': original_data.get('description', ''),
            'source': original_data.get('source', ''),
            'task_type': determine_task_type(original_data),
            'instances': original_data.get('instances', 0),
            'features': original_data.get('features', 0),
            
            # Standardized columns structure
            'columns': [],
            'target': None,
            
            # Preserve benchmark-specific metadata
            'metadata': {k: v for k, v in original_data.items() 
                        if k not in ['columns', 'feature_description', 'feature_descriptions', 'feature_names']}
        }
        
        # Extract and standardize feature information
        feature_info = extract_feature_info(original_data)
        target_info = extract_target_info(original_data)
        
        # Build standardized columns
        for i, (name, description, data_type) in enumerate(feature_info):
            column = {
                'name': name,
                'semantic_description': description,
                'data_type': data_type,
                'is_target': False,
                'index': i
            }
            standardized['columns'].append(column)
        
        # Add target information
        if target_info:
            standardized['target'] = target_info
        
        # Validation
        validation_result = validate_standardized_metadata(standardized, original_data)
        
        if not dry_run and validation_result['valid']:
            # Create backup
            backup_file = metadata_file.with_suffix('.json.backup')
            if not backup_file.exists():
                shutil.copy2(metadata_file, backup_file)
            
            # Write standardized version
            with open(metadata_file, 'w') as f:
                json.dump(standardized, f, indent=2, ensure_ascii=False)
        
        return {
            'file': metadata_file.name,
            'success': True,
            'validation': validation_result,
            'features_standardized': len(standardized['columns']),
            'has_target': standardized['target'] is not None,
            'dry_run': dry_run
        }
        
    except Exception as e:
        return {
            'file': metadata_file.name,
            'success': False,
            'error': str(e),
            'dry_run': dry_run
        }

def determine_task_type(data: Dict[str, Any]) -> str:
    """Determine if this is a classification or regression task."""
    
    # Check explicit indicators
    if 'target_type' in data:
        target_type = data['target_type'].lower()
        if 'class' in target_type or 'multiclass' in target_type or 'binary' in target_type:
            return 'classification'
        elif 'regr' in target_type or 'numeric' in target_type:
            return 'regression'
    
    # Check for class-related fields
    if any(key in data for key in ['target_values', 'instances_per_class', 'classes']):
        return 'classification'
    
    # Default to classification (most common)
    return 'classification'

def extract_feature_info(data: Dict[str, Any]) -> List[tuple]:
    """Extract feature information from various metadata structures."""
    
    feature_info = []
    
    # Priority 1: columns structure (if present and well-formed)
    if 'columns' in data and isinstance(data['columns'], list):
        for col in data['columns']:
            if isinstance(col, dict) and 'name' in col:
                name = col['name']
                description = col.get('semantic_description', col.get('description', name))
                data_type = infer_data_type(col, name)
                feature_info.append((name, description, data_type))
        
        if feature_info:  # If we got good data from columns, use it
            return feature_info
    
    # Priority 2: feature_description structure
    if 'feature_description' in data and isinstance(data['feature_description'], dict):
        for name, description in data['feature_description'].items():
            data_type = infer_data_type_from_description(description)
            feature_info.append((name, description, data_type))
        return feature_info
    
    # Priority 3: feature_descriptions structure  
    if 'feature_descriptions' in data and isinstance(data['feature_descriptions'], dict):
        for name, description in data['feature_descriptions'].items():
            data_type = infer_data_type_from_description(description)
            feature_info.append((name, description, data_type))
        return feature_info
    
    # Priority 4: feature_names structure
    if 'feature_names' in data and isinstance(data['feature_names'], dict):
        for name, description in data['feature_names'].items():
            data_type = infer_data_type_from_description(description)
            feature_info.append((name, description, data_type))
        return feature_info
    
    # Fallback: create generic features based on feature count
    feature_count = data.get('features', 0)
    if feature_count > 0:
        for i in range(feature_count):
            name = f"feature_{i}"
            description = f"Feature {i}"
            data_type = "numerical"  # Default assumption
            feature_info.append((name, description, data_type))
    
    return feature_info

def extract_target_info(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract target/label information."""
    
    target_info = None
    
    # Get target name
    target_name = data.get('target', 'target')
    
    # Get target description
    target_description = ""
    if 'description' in data and 'target' in data.get('description', '').lower():
        target_description = data['description']
    
    # Determine task type and classes
    task_type = determine_task_type(data)
    classes = None
    
    if task_type == 'classification':
        # Extract class information
        if 'target_values' in data:
            if isinstance(data['target_values'], dict):
                classes = list(data['target_values'].values())
            elif isinstance(data['target_values'], list):
                classes = data['target_values']
        elif 'instances_per_class' in data:
            classes = list(data['instances_per_class'].keys())
        elif 'classes' in data:
            classes = data['classes']
    
    # Build target info
    if target_name or task_type:
        target_info = {
            'name': target_name,
            'semantic_description': target_description or f"Target variable for {task_type}",
            'data_type': 'categorical' if task_type == 'classification' else 'numerical',
            'task_type': task_type,
            'classes': classes,
            'is_target': True
        }
    
    return target_info

def infer_data_type(col_data: Dict[str, Any], name: str) -> str:
    """Infer data type from column information."""
    
    # Check explicit type information
    if 'data_type' in col_data:
        return col_data['data_type']
    if 'type' in col_data:
        return col_data['type']
    
    # Infer from description
    description = col_data.get('semantic_description', col_data.get('description', name)).lower()
    return infer_data_type_from_description(description)

def infer_data_type_from_description(description: str) -> str:
    """Infer data type from textual description."""
    
    description = description.lower()
    
    # Categorical indicators
    if any(word in description for word in ['category', 'class', 'type', 'kind', 'group', 'label']):
        return 'categorical'
    
    # Ordinal indicators  
    if any(word in description for word in ['level', 'grade', 'rank', 'order', 'rating', 'scale']):
        return 'ordinal'
    
    # Numerical indicators
    if any(word in description for word in ['age', 'year', 'count', 'number', 'amount', 'size', 'length', 'weight', 'price', 'cost', 'income', 'salary']):
        return 'numerical'
    
    # Pattern-based inference
    if any(indicator in description for indicator in ['(numerical)', '(float)', '(integer)', '(int)']):
        return 'numerical'
    
    if any(indicator in description for indicator in ['(categorical)', '(string)', '(text)']):
        return 'categorical'
    
    # Default assumption
    return 'numerical'

def validate_standardized_metadata(standardized: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that standardization preserved essential information."""
    
    validation = {'valid': True, 'warnings': [], 'errors': []}
    
    # Check that we have columns
    if not standardized.get('columns'):
        validation['errors'].append("No columns in standardized metadata")
        validation['valid'] = False
    
    # Check feature count consistency
    original_features = original.get('features', 0)
    standardized_features = len(standardized.get('columns', []))
    
    if original_features > 0 and standardized_features != original_features:
        validation['warnings'].append(f"Feature count mismatch: original {original_features}, standardized {standardized_features}")
    
    # Check that all columns have required fields
    for i, col in enumerate(standardized.get('columns', [])):
        if not col.get('name'):
            validation['errors'].append(f"Column {i} missing name")
            validation['valid'] = False
        if not col.get('semantic_description'):
            validation['warnings'].append(f"Column {i} ({col.get('name')}) missing semantic description")
    
    return validation

def main():
    """Main standardization function."""
    
    print("🔧 Universal Metadata Standardization for MARVIS")
    print("=" * 50)
    
    # Discover benchmarks
    data_dir = Path("data")
    benchmarks = discover_all_benchmarks(data_dir)
    
    if not benchmarks:
        print("❌ No benchmarks found in data directory")
        return
    
    # Ask user for confirmation
    total_files = sum(len(list(path.glob("*.json"))) for path in benchmarks.values())
    print(f"\n📊 Found {len(benchmarks)} benchmarks with {total_files} total metadata files")
    
    print("\n🔍 Running analysis phase...")
    
    # Analyze current state
    all_analyses = []
    for benchmark_name, benchmark_path in benchmarks.items():
        print(f"\n--- Analyzing {benchmark_name} ---")
        
        json_files = list(benchmark_path.glob("*.json"))
        for json_file in json_files[:5]:  # Sample first 5 files
            analysis = analyze_current_structure(json_file)
            analysis['benchmark'] = benchmark_name
            all_analyses.append(analysis)
            
            if 'error' in analysis:
                print(f"❌ {analysis['file']}: {analysis['error']}")
            else:
                print(f"📄 {analysis['file']}: {analysis['semantic_count']} semantic features, has_columns={analysis['has_columns']}")
    
    # Get user decision
    print(f"\n❓ Proceed with standardization?")
    print("   'dry' - Dry run (validate only)")
    print("   'yes' - Execute standardization") 
    print("   'no' - Exit")
    
    choice = input("Choice: ").lower().strip()
    
    if choice == 'no':
        print("👋 Exiting without changes")
        return
    
    dry_run = choice != 'yes'
    
    if dry_run:
        print("\n🧪 Running DRY RUN (no files will be modified)")
    else:
        print("\n⚡ Running STANDARDIZATION (files will be modified)")
        print("   Backups will be created as .json.backup")
    
    # Process all files
    all_results = []
    for benchmark_name, benchmark_path in benchmarks.items():
        print(f"\n--- Standardizing {benchmark_name} ---")
        
        json_files = list(benchmark_path.glob("*.json"))
        for json_file in json_files:
            if json_file.suffix == '.backup':  # Skip backup files
                continue
                
            result = standardize_metadata_file(json_file, dry_run=dry_run)
            result['benchmark'] = benchmark_name
            all_results.append(result)
            
            if result['success']:
                status = "✅ DRY RUN" if dry_run else "✅ UPDATED"
                print(f"{status} {result['file']}: {result['features_standardized']} features, target={result['has_target']}")
                
                if result['validation']['warnings']:
                    for warning in result['validation']['warnings']:
                        print(f"   ⚠️  {warning}")
                        
                if result['validation']['errors']:
                    for error in result['validation']['errors']:
                        print(f"   ❌ {error}")
            else:
                print(f"❌ FAILED {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Summary
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\n📈 SUMMARY:")
    print(f"  ✅ Successful: {len(successful)}/{len(all_results)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if dry_run and successful:
        print(f"\n💡 To execute standardization, run with 'yes' option")
    elif not dry_run and successful:
        print(f"\n🎉 Standardization complete! Backup files created with .backup extension")

if __name__ == "__main__":
    main()