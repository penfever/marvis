#!/usr/bin/env python3
"""
Script to synthesize TabLLM notes from actual OpenML CC18 data.
This script loads real datasets and creates notes from actual data rows.
"""

import json
import os
import glob
import yaml
import uuid
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to import from tabllm_baseline
sys.path.append(str(Path(__file__).parent.parent))
from tabllm_baseline import expand_semantic_features

# Optional: Import OpenML if available
try:
    import openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False
    print("Warning: OpenML not installed. Will generate synthetic examples only.")

# Define paths using resource manager
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))

# Initialize resource manager and metadata loader
try:
    import sys
    sys.path.insert(0, project_root)
    from marvis.utils.resource_manager import get_resource_manager
    from marvis.utils.metadata_loader import get_metadata_loader
    
    RESOURCE_MANAGER = get_resource_manager()
    METADATA_LOADER = get_metadata_loader()
    
    # Use resource manager for output directory
    OUTPUT_DIR = RESOURCE_MANAGER.path_resolver.get_configs_dir() / 'tabllm_like'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using resource manager output directory: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"Warning: Could not initialize resource manager: {e}")
    RESOURCE_MANAGER = None
    METADATA_LOADER = None
    # Fallback to current directory
    OUTPUT_DIR = current_dir
    print(f"Using fallback output directory: {OUTPUT_DIR}")
NOTES_PER_DATASET = 100  # Number of example notes to generate per dataset


def load_openml_dataset(dataset_id: int) -> Optional[pd.DataFrame]:
    """Load a dataset from OpenML."""
    if not OPENML_AVAILABLE:
        return None
    
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=attribute_names)
        df['target'] = y
        
        return df
    except Exception as e:
        print(f"Error loading OpenML dataset {dataset_id}: {e}")
        return None


# expand_semantic_features is now imported from tabllm_baseline


def generate_note_from_row(row: pd.Series, semantic_info: Dict[str, Any], exclude_target: bool = True) -> str:
    """Generate a TabLLM-style note from a data row."""
    note_parts = []
    
    # Get feature descriptions based on the structure of semantic JSON
    if 'columns' in semantic_info:
        # Structure like kr-vs-kp dataset
        for col in semantic_info['columns']:
            col_name = col['name']
            if col_name in row.index and (not exclude_target or col_name != 'target'):
                semantic_desc = col['semantic_description']
                value = row[col_name]
                
                # Handle different data types
                if pd.isna(value):
                    note_parts.append(f"The {semantic_desc} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {semantic_desc} is {int(value)}.")
                    else:
                        note_parts.append(f"The {semantic_desc} is {value:.2f}.")
                else:
                    note_parts.append(f"The {semantic_desc} is {value}.")
    
    elif 'feature_descriptions' in semantic_info:
        # Structure like letter dataset
        for feat_name, feat_desc in semantic_info['feature_descriptions'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                # Clean up description for better readability
                clean_desc = feat_desc.replace(' (integer)', '').replace(' (float)', '')
                
                if pd.isna(value):
                    note_parts.append(f"The {clean_desc} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {clean_desc} is {int(value)}.")
                    else:
                        note_parts.append(f"The {clean_desc} is {value:.2f}.")
                else:
                    note_parts.append(f"The {clean_desc} is {value}.")
    
    elif 'feature_description' in semantic_info:
        # Handle adult dataset format (singular feature_description)
        for feat_name, feat_desc in semantic_info['feature_description'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                # Create more natural descriptions
                feature_label = feat_name.replace('-', ' ').replace('_', ' ')
                
                if pd.isna(value):
                    note_parts.append(f"The {feature_label} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {feature_label} is {int(value)}.")
                    else:
                        note_parts.append(f"The {feature_label} is {value:.2f}.")
                else:
                    note_parts.append(f"The {feature_label} is {value}.")
    
    elif 'feature_names' in semantic_info:
        # Fallback to feature_names if descriptions not available
        for feat_name, feat_meaning in semantic_info['feature_names'].items():
            if feat_name in row.index and (not exclude_target or feat_name != 'target'):
                value = row[feat_name]
                
                if pd.isna(value):
                    note_parts.append(f"The {feat_meaning} is missing.")
                elif isinstance(value, (int, float)):
                    if value == int(value):
                        note_parts.append(f"The {feat_meaning} is {int(value)}.")
                    else:
                        note_parts.append(f"The {feat_meaning} is {value:.2f}.")
                else:
                    note_parts.append(f"The {feat_meaning} is {value}.")
    
    return " ".join(note_parts)


def process_dataset_to_notes(json_file: str, max_notes: int = NOTES_PER_DATASET) -> List[Dict[str, Any]]:
    """Process a single dataset and generate notes."""
    notes_data = []
    
    try:
        with open(json_file, 'r') as f:
            semantic_info = json.load(f)
        
        # Extract dataset info from metadata - both task_id and dataset_id are available
        dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', Path(json_file).stem))
        task_id = semantic_info.get('openml_task') or semantic_info.get('_metadata', {}).get('task_id') or Path(json_file).stem
        dataset_id = semantic_info.get('openml_dataset')
        
        # Log what we found
        if dataset_id and task_id:
            print(f"Processing dataset {dataset_name} (Task ID: {task_id}, Dataset ID: {dataset_id})...")
        elif task_id:
            # No dataset_id in metadata, try the old resolution method as fallback
            if RESOURCE_MANAGER:
                try:
                    task_id_int = int(task_id)
                    identifiers = RESOURCE_MANAGER.resolve_openml_identifiers(task_id=task_id_int)
                    dataset_id = identifiers.get('dataset_id')
                    if dataset_id:
                        print(f"Processing dataset {dataset_name} (Task ID: {task_id}, Dataset ID: {dataset_id} - resolved)...")
                    else:
                        print(f"Processing dataset {dataset_name} (Task ID: {task_id}, could not resolve to dataset ID)...")
                except (ValueError, TypeError) as e:
                    print(f"Processing dataset {dataset_name} (Task ID: {task_id}, error resolving: {e})...")
            else:
                # Ultimate fallback: assume task_id is dataset_id
                try:
                    dataset_id = int(task_id)
                    print(f"Processing dataset {dataset_name} (ID: {dataset_id}, assuming task_id=dataset_id)...")
                except (ValueError, TypeError):
                    print(f"Processing dataset {dataset_name} (invalid ID: {task_id})...")
        else:
            print(f"Processing dataset {dataset_name} (no valid IDs found)...")
        
        # Try to load actual data if we have a valid dataset_id
        df = None
        if dataset_id:
            df = load_openml_dataset(dataset_id)
        
        if df is not None and len(df) > 0:
            # Get actual feature count (excluding target column)
            feature_columns = [col for col in df.columns if col not in ['target', 'class']]
            actual_feature_count = len(feature_columns)
            
            # Expand semantic features if needed to match actual dataset features
            semantic_info = expand_semantic_features(semantic_info, actual_feature_count, feature_columns)
            
            # Sample rows for notes
            n_samples = min(max_notes, len(df))
            sampled_df = df.sample(n=n_samples, random_state=42)
            
            for idx, row in sampled_df.iterrows():
                note = generate_note_from_row(row, semantic_info, exclude_target=True)
                target = row.get('target', row.get('class', None))
                
                notes_data.append({
                    'dataset': dataset_name,
                    'task_id': task_id,
                    'dataset_id': dataset_id,
                    'note': note,
                    'target': target,
                    'row_index': idx
                })
        
        else:
            # Generate synthetic examples if real data not available
            if dataset_id is None:
                print(f"  Could not resolve task_id {task_id} to dataset_id - generating synthetic examples for {dataset_name}...")
            else:
                print(f"  Could not load OpenML dataset {dataset_id} - generating synthetic examples for {dataset_name}...")
            
            for i in range(min(10, max_notes)):  # Generate fewer synthetic examples
                example_features = {}
                
                if 'columns' in semantic_info:
                    for col in semantic_info['columns']:
                        col_name = col['name']
                        data_type = col.get('data_type', 'unknown')
                        if 'binary' in data_type:
                            example_features[col_name] = np.random.choice(['yes', 'no'])
                        elif 'categorical' in data_type:
                            if '(' in data_type and ')' in data_type:
                                values = data_type[data_type.find('(')+1:data_type.find(')')].split('/')
                                example_features[col_name] = np.random.choice(values)
                            else:
                                example_features[col_name] = f'category_{np.random.randint(1, 4)}'
                        else:
                            example_features[col_name] = np.random.randint(0, 100)
                
                elif 'feature_description' in semantic_info:
                    # Handle adult dataset format
                    for feat_name, feat_desc in semantic_info['feature_description'].items():
                        if feat_desc.lower() == 'continuous':
                            if 'age' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(17, 90)
                            elif 'fnlwgt' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(10000, 1000000)
                            elif 'education-num' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(1, 16)
                            elif 'capital' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(0, 10000)
                            elif 'hours' in feat_name.lower():
                                example_features[feat_name] = np.random.randint(1, 80)
                            else:
                                example_features[feat_name] = np.random.randint(0, 100)
                        elif ',' in feat_desc:
                            # Categorical with comma-separated values
                            values = [v.strip() for v in feat_desc.split(',')]
                            example_features[feat_name] = np.random.choice(values)
                        else:
                            example_features[feat_name] = f'value_{np.random.randint(1, 5)}'
                
                elif 'feature_names' in semantic_info or 'feature_descriptions' in semantic_info:
                    feature_list = semantic_info.get('feature_descriptions', semantic_info.get('feature_names', {}))
                    for feat_name in feature_list.keys():
                        example_features[feat_name] = np.random.randint(0, 16)  # Based on letter dataset scale
                
                # Create Series for note generation
                row_series = pd.Series(example_features)
                note = generate_note_from_row(row_series, semantic_info, exclude_target=True)
                
                # Generate synthetic target
                if 'target_classes' in semantic_info:
                    target = np.random.choice([tc['name'] for tc in semantic_info['target_classes']])
                elif 'target_values' in semantic_info:
                    target = np.random.choice(list(semantic_info['target_values'].keys()))
                else:
                    target = np.random.choice(['0', '1'])
                
                notes_data.append({
                    'dataset': dataset_name,
                    'task_id': task_id,
                    'dataset_id': dataset_id,
                    'note': note,
                    'target': target,
                    'row_index': i,
                    'synthetic': True
                })
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
    
    return notes_data


def write_yaml_template(template_data: Dict[str, Any], output_path: str):
    """Write a template to a YAML file with custom tags."""
    # Custom YAML representers for our tags
    def template_representer(dumper, data):
        # Remove the __tag__ key before dumping
        data_copy = {k: v for k, v in data.items() if k != '__tag__'}
        return dumper.represent_mapping('!Template', data_copy)
    
    def template_metadata_representer(dumper, data):
        # Remove the __tag__ key before dumping
        data_copy = {k: v for k, v in data.items() if k != '__tag__'}
        return dumper.represent_mapping('!TemplateMetadata', data_copy)
    
    # Create a custom YAML dumper
    yaml.add_representer(dict, lambda dumper, data: (
        template_representer(dumper, data) if data.get('__tag__') == '!Template' else
        template_metadata_representer(dumper, data) if data.get('__tag__') == '!TemplateMetadata' else
        dumper.represent_dict(data)
    ))
    
    with open(output_path, 'w') as f:
        yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)


def create_notes_bank_files():
    """Create note bank files and YAML templates for all datasets."""
    # Get all JSON files using the new general search approach
    json_files = []
    
    if METADATA_LOADER:
        # Use the metadata loader to find all semantic files recursively
        try:
            data_dir = Path(project_root) / "data"
            if data_dir.exists():
                json_files = list(data_dir.rglob("*.json"))
                print(f"Found {len(json_files)} JSON files using recursive search")
        except Exception as e:
            print(f"Error with metadata loader search: {e}")
    
    # Fallback to old method if metadata loader failed
    if not json_files and 'SEMANTIC_DIR' in globals():
        json_files = glob.glob(os.path.join(SEMANTIC_DIR, "*.json"))
        print(f"Using fallback search, found {len(json_files)} files")
    
    all_notes = []
    dataset_notes = {}  # Store notes by dataset
    templates_created = 0
    
    for json_file in sorted(json_files):
        # Load semantic info for template creation
        try:
            with open(json_file, 'r') as f:
                semantic_info = json.load(f)
            
            dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', Path(json_file).stem))
            task_id = semantic_info.get('_metadata', {}).get('task_id') or Path(json_file).stem
            dataset_id = semantic_info.get('openml_dataset')
            
            # Try to resolve dataset_id if not available
            if not dataset_id and task_id:
                if RESOURCE_MANAGER:
                    try:
                        task_id_int = int(task_id)
                        identifiers = RESOURCE_MANAGER.resolve_openml_identifiers(task_id=task_id_int)
                        dataset_id = identifiers.get('dataset_id')
                    except (ValueError, TypeError):
                        pass
                
                # Ultimate fallback: assume task_id is dataset_id
                if not dataset_id:
                    try:
                        dataset_id = int(task_id)
                    except (ValueError, TypeError):
                        pass
            
            # Try to get actual feature count for template expansion
            template_semantic_info = semantic_info
            if dataset_id:
                df = load_openml_dataset(dataset_id)
                if df is not None and len(df) > 0:
                    feature_columns = [col for col in df.columns if col not in ['target', 'class']]
                    actual_feature_count = len(feature_columns)
                    # Expand semantic features for template creation too
                    template_semantic_info = expand_semantic_features(semantic_info, actual_feature_count, feature_columns)
            
            # Create and save YAML template using task_id for consistent lookup
            template_data = create_template_for_dataset(template_semantic_info, dataset_name)
            template_filename = f"templates_task_{task_id}.yaml"
            template_path = OUTPUT_DIR / template_filename
            
            write_yaml_template(template_data, template_path)
            templates_created += 1
            
        except Exception as e:
            print(f"Error creating template for {json_file}: {e}")
        
        # Process notes as before
        notes = process_dataset_to_notes(json_file)
        if notes:
            dataset_name = notes[0]['dataset']
            dataset_notes[dataset_name] = notes
            all_notes.extend(notes)
    
    # Write individual dataset note files
    for dataset_name, notes in dataset_notes.items():
        notes_filename = f"notes_{dataset_name}.jsonl"
        notes_dir = OUTPUT_DIR / "notes"
        notes_dir.mkdir(exist_ok=True)
        notes_path = notes_dir / notes_filename
        
        with open(notes_path, 'w') as f:
            for note_item in notes:
                f.write(json.dumps(note_item) + '\n')
        
        # Also create simple text version
        text_filename = f"notes_{dataset_name}.txt"
        text_path = notes_dir / text_filename
        
        with open(text_path, 'w') as f:
            for note_item in notes:
                f.write(f"['{note_item['note']}'] -> {note_item['target']}\n")
    
    # Write combined notes file
    combined_path = notes_dir / "all_notes.jsonl"
    with open(combined_path, 'w') as f:
        for note_item in all_notes:
            f.write(json.dumps(note_item) + '\n')
    
    print(f"\nCreated note files for {len(dataset_notes)} datasets")
    print(f"Total notes generated: {len(all_notes)}")
    print(f"Notes directory: {notes_dir}")
    print(f"Created {templates_created} YAML template files")
    print(f"Templates directory: {OUTPUT_DIR}")


def generate_note_from_semantic_info(semantic_info: Dict[str, Any]) -> str:
    """Generate a note example from semantic information."""
    try:
        # Extract feature descriptions from semantic info
        features = []
        if 'columns' in semantic_info:
            for col in semantic_info['columns']:
                if col.get('name') != 'target':
                    name = col.get('name', 'feature')
                    desc = col.get('semantic_description', name)
                    # Create a synthetic example value
                    if 'type' in col and col['type'] == 'categorical':
                        value = 'example_category'
                    else:
                        value = '42'
                    features.append(f"The {desc} is {value}")
        elif 'feature_descriptions' in semantic_info:
            for name, desc in semantic_info['feature_descriptions'].items():
                features.append(f"The {desc} is example_value")
        elif 'feature_description' in semantic_info:
            for name, desc in semantic_info['feature_description'].items():
                features.append(f"The {desc} is example_value")
        else:
            # Fallback
            features = ["The feature is example_value"]
        
        return ". ".join(features[:10])  # Limit to first 10 features
        
    except Exception as e:
        # Fallback note
        return "This is an example tabular data instance with various features"


def create_template_for_dataset(semantic_info: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Create a YAML template for a dataset from semantic information."""
    try:
        # Extract class information
        classes = []
        class_descriptions = {}
        
        # Try to get target classes from semantic info
        if 'target_classes' in semantic_info:
            # Handle structure like: [{"name": "Class 0", "description": "..."}, ...]
            classes = [tc['name'] for tc in semantic_info['target_classes']]
            for tc in semantic_info['target_classes']:
                class_descriptions[tc['name']] = tc.get('description', tc['name'])
        elif 'target_values' in semantic_info:
            # Handle structure like: {"0": "no", "1": "yes"}
            classes = list(semantic_info['target_values'].values())
            class_descriptions = {v: v for v in classes}
        elif 'target_description' in semantic_info:
            # Fallback: use generic class names with target description
            target_desc = semantic_info['target_description']
            if 'class_names' in semantic_info:
                classes = semantic_info['class_names']
            elif 'classes' in semantic_info:
                classes = semantic_info['classes']
            else:
                classes = ['Class 0', 'Class 1']  # Default binary
            
            # Create class descriptions
            for cls in classes:
                class_descriptions[str(cls)] = f"{cls} ({target_desc})"
        else:
            # Ultimate fallback
            classes = ['Class 0', 'Class 1']
            class_descriptions = {'Class 0': 'Class 0', 'Class 1': 'Class 1'}
        
        # Generate a unique ID for the template
        template_id = str(uuid.uuid4())
        
        # Create answer choices string
        answer_choices = ' ||| '.join(classes)
        
        # Create jinja template with appropriate question
        if len(classes) == 2 and any(keyword in ' '.join(classes).lower() for keyword in ['yes', 'no', 'true', 'false', 'positive', 'negative']):
            # Binary classification with clear yes/no, true/false, etc.
            jinja_template = f"{{{{note}}}}\n\nWhat is the class of this instance?\nAnswer: \n|||\n{{{{ answer_choices[label] }}}}"
        else:
            # Multi-class or less clear binary
            classes_str = ', '.join(classes)
            jinja_template = f"{{{{note}}}}\n\nWhich of the following classes does this instance belong to: {classes_str}?\nAnswer: \n|||\n{{{{ answer_choices[label] }}}}"
        
        # Create template structure matching the YAML format
        template_data = {
            'dataset': dataset_name,
            'templates': {
                template_id: {
                    '__tag__': '!Template',  # Will be converted to YAML tag
                    'name': dataset_name,
                    'id': template_id,
                    'reference': '',
                    'answer_choices': answer_choices,
                    'jinja': jinja_template,
                    'metadata': {
                        '__tag__': '!TemplateMetadata',  # Will be converted to YAML tag
                        'choices_in_prompt': True,
                        'metrics': ['accuracy'],
                        'original_task': True
                    }
                }
            }
        }
        
        return template_data
        
    except Exception as e:
        print(f"Error creating template for {dataset_name}: {e}")
        # Fallback template
        template_id = str(uuid.uuid4())
        return {
            'dataset': dataset_name,
            'templates': {
                template_id: {
                    '__tag__': '!Template',
                    'name': dataset_name,
                    'id': template_id,
                    'reference': '',
                    'answer_choices': 'Class 0 ||| Class 1',
                    'jinja': "{{note}}\n\nWhat is the class of this instance?\nAnswer: \n|||\n{{ answer_choices[label] }}",
                    'metadata': {
                        '__tag__': '!TemplateMetadata',
                        'choices_in_prompt': True,
                        'metrics': ['accuracy'],
                        'original_task': True
                    }
                }
            }
        }


def main():
    """Main function."""
    # Ensure output directories exist (OUTPUT_DIR is already created in initialization)
    notes_dir = OUTPUT_DIR / "notes"
    notes_dir.mkdir(exist_ok=True)
    
    print("Synthesizing TabLLM notes and YAML templates from OpenML CC18 datasets...")
    create_notes_bank_files()
    print("\nDone!")


if __name__ == "__main__":
    main()