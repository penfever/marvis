#!/usr/bin/env python3
"""
Script to synthesize JOLT templates and prefixes from semantic JSON files.
JOLT uses simple feature-value pairs with optional task context prefixes.
"""

import json
import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

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
    OUTPUT_DIR = RESOURCE_MANAGER.path_resolver.get_configs_dir() / 'jolt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using resource manager output directory: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"Warning: Could not initialize resource manager: {e}")
    RESOURCE_MANAGER = None
    METADATA_LOADER = None
    # Fallback to current directory
    OUTPUT_DIR = current_dir
    print(f"Using fallback output directory: {OUTPUT_DIR}")


def extract_column_descriptions(semantic_info: Dict[str, Any]) -> Dict[str, str]:
    """Extract column descriptions from semantic information."""
    column_descriptions = {}
    
    # Handle the improved semantic file format first
    if 'feature_description' in semantic_info and isinstance(semantic_info['feature_description'], str):
        # New detailed format - use feature_description string and features count
        feat_desc_string = semantic_info['feature_description']
        dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', 'unknown'))
        num_features = semantic_info.get('features', 5)
        
        if isinstance(num_features, str):
            try:
                num_features = int(num_features)
            except:
                num_features = 5
        
        # Generate column descriptions based on the feature description
        if 'fourier' in feat_desc_string.lower():
            for i in range(min(20, num_features)):
                column_descriptions[f'fourier_coeff_{i+1}'] = f'Fourier coefficient {i+1}'
        elif 'pixel' in feat_desc_string.lower():
            for i in range(min(20, num_features)):
                column_descriptions[f'pixel_{i+1}'] = f'Pixel value {i+1}'
        elif 'karhunen' in feat_desc_string.lower() or 'pca' in feat_desc_string.lower():
            for i in range(min(20, num_features)):
                column_descriptions[f'kl_coeff_{i+1}'] = f'Karhunen-Loève coefficient {i+1}'
        elif 'coordinate' in feat_desc_string.lower():
            # For coordinate-based data like pendigits
            num_points = num_features // 2
            for i in range(num_points):
                column_descriptions[f'x{i+1}'] = f'X coordinate of point {i+1}'
                column_descriptions[f'y{i+1}'] = f'Y coordinate of point {i+1}'
        elif 'protein' in feat_desc_string.lower():
            for i in range(min(20, num_features)):
                column_descriptions[f'protein_{i+1}'] = f'Protein expression level {i+1}'
        elif 'sensor' in feat_desc_string.lower() or 'acceleration' in feat_desc_string.lower():
            # For sensor/activity data like HAR
            sensor_types = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            for i in range(min(20, num_features)):
                if i < len(sensor_types):
                    column_descriptions[f'feature_{i+1}'] = f'{sensor_types[i % len(sensor_types)]} measurement'
                else:
                    column_descriptions[f'feature_{i+1}'] = f'Sensor feature {i+1}'
        else:
            # Generic feature descriptions
            for i in range(min(20, num_features)):
                column_descriptions[f'feature_{i+1}'] = f'Feature {i+1}'
    
    elif 'columns' in semantic_info:
        # Structure like kr-vs-kp dataset
        for col in semantic_info['columns']:
            col_name = col['name']
            semantic_desc = col['semantic_description']
            
            # Handle case where semantic_desc might be a dict
            if isinstance(semantic_desc, dict):
                # Extract the description from dict if possible
                clean_desc = str(semantic_desc.get('description', col_name))
            elif isinstance(semantic_desc, str):
                # Clean up semantic description for JOLT
                if semantic_desc.lower().startswith('whether'):
                    clean_desc = semantic_desc.replace('Whether', '').replace('whether', '').strip()
                    if clean_desc.startswith('the '):
                        clean_desc = clean_desc[4:]
                    clean_desc = clean_desc.capitalize()
                else:
                    clean_desc = semantic_desc
            else:
                # Fallback to column name
                clean_desc = col_name.replace('_', ' ').replace('-', ' ')
                
            column_descriptions[col_name] = clean_desc
    
    elif 'feature_descriptions' in semantic_info:
        # Structure like banknote dataset
        for feat_name, feat_desc in semantic_info['feature_descriptions'].items():
            clean_desc = feat_desc.replace(' (integer)', '').replace(' (float)', '')
            column_descriptions[feat_name] = clean_desc
    
    elif 'feature_description' in semantic_info and isinstance(semantic_info['feature_description'], dict):
        feat_desc_data = semantic_info['feature_description']
        # Dictionary format (like adult dataset)
        for feat_name, feat_desc in feat_desc_data.items():
            if isinstance(feat_desc, str):
                # For categorical features with comma-separated values, use the feature name
                if ',' in feat_desc:
                    clean_desc = feat_name.replace('-', ' ').replace('_', ' ')
                else:
                    clean_desc = feat_desc if feat_desc.lower() != 'continuous' else feat_name.replace('-', ' ').replace('_', ' ')
            else:
                clean_desc = feat_name.replace('-', ' ').replace('_', ' ')
            column_descriptions[feat_name] = clean_desc
    
    # NEW: Handle cases where we have input_features information but no detailed descriptions
    elif 'input_features' in semantic_info:
        input_features = semantic_info['input_features']
        num_features = input_features.get('number_of_features', 5)
        feature_desc = input_features.get('feature_description', '')
        dataset_name = semantic_info.get('dataset_name', 'unknown').lower()
        
        # Create brief summary descriptions based on available information
        if 'phishing' in dataset_name and 'url' in feature_desc.lower():
            # Phishing website detection features
            feature_types = ['URL structure', 'domain registration', 'page content', 'technical indicator']
            for i in range(num_features):
                feature_type = feature_types[i % len(feature_types)]
                column_descriptions[f'feature_{i+1}'] = f'{feature_type} feature'
        elif 'url' in feature_desc.lower():
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'URL-based feature {i+1}'
        elif 'website' in feature_desc.lower():
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Website characteristic {i+1}'
        else:
            # Generic based on feature description
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Dataset feature {i+1}'
    
    # Fallback: if still no column descriptions, try to infer from dataset info
    if not column_descriptions:
        # Get number of features from various sources
        num_features = (semantic_info.get('features') or 
                       semantic_info.get('dataset_size', {}).get('features') or
                       semantic_info.get('input_features', {}).get('number_of_features') or
                       5)
        
        if isinstance(num_features, str):
            try:
                num_features = int(num_features)
            except:
                num_features = 5
        
        dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', 'unknown')).lower()
        dataset_desc = semantic_info.get('dataset_description', '').lower()
        
        # Create dataset-specific brief descriptions
        if 'phishing' in dataset_name or 'phishing' in dataset_desc:
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Website security feature {i+1}'
        elif 'spam' in dataset_name or 'spam' in dataset_desc:
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Email/text feature {i+1}'
        elif 'image' in dataset_desc or 'pixel' in dataset_desc:
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Image feature {i+1}'
        elif 'medical' in dataset_desc or 'patient' in dataset_desc:
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Medical measurement {i+1}'
        elif 'network' in dataset_desc or 'intrusion' in dataset_desc:
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Network feature {i+1}'
        else:
            # Most generic fallback
            for i in range(num_features):
                column_descriptions[f'feature_{i+1}'] = f'Dataset feature {i+1}'
    
    return column_descriptions


def extract_class_information(semantic_info: Dict[str, Any]) -> Tuple[List[str], str]:
    """Extract class names and class description from semantic information."""
    class_names = []
    class_description = ""
    
    # Check for regression tasks first
    if 'target_description' in semantic_info:
        # This is a regression task with continuous target
        target_desc = semantic_info['target_description']
        target_name = target_desc.get('name', 'target')
        target_meaning = target_desc.get('meaning', 'continuous target variable')
        target_units = target_desc.get('units', '')
        target_range = target_desc.get('range', '')
        
        # For regression, we don't have discrete classes
        # Instead, we describe the continuous target variable
        class_names = ["continuous_value"]
        
        description_parts = [f"Target: {target_name} - {target_meaning}"]
        if target_units:
            description_parts.append(f"Units: {target_units}")
        if target_range:
            description_parts.append(f"Range: {target_range}")
        
        class_description = ". ".join(description_parts)
        
    elif 'target_values' in semantic_info:
        # Multi-class classification - this covers both old and new formats
        target_values = semantic_info['target_values']
        
        if isinstance(target_values, dict):
            # Could be {class_id: class_name} or {class_name: class_description}
            class_names = sorted(target_values.keys())
            
            # Create class description
            class_meanings = [f"'{cls}' ({meaning})" for cls, meaning in target_values.items()]
            class_description = f"Classes: {', '.join(class_meanings)}"
        else:
            # Handle other formats
            class_names = ["Class_0", "Class_1"]
            class_description = "Classes: Class_0, Class_1"
    
    elif 'target_variable' in semantic_info:
        # New format with target_variable structure
        target_var = semantic_info['target_variable']
        
        if 'values' in target_var and isinstance(target_var['values'], list):
            # Handle numeric values like [-1, 1]
            class_names = [str(val) for val in target_var['values']]
            description = target_var.get('description', '')
            
            if description:
                # Try to extract class meanings from description
                if 'phishing' in description.lower() and 'legitimate' in description.lower():
                    class_description = f"Classes: '{class_names[0]}' (phishing website), '{class_names[1]}' (legitimate website)"
                else:
                    class_list = ', '.join([f"'{cls}'" for cls in class_names])
                    class_description = f"Classes: {class_list} - {description}"
            else:
                class_list = ', '.join([f"'{cls}'" for cls in class_names])
                class_description = f"Classes: {class_list}"
        else:
            # Fallback for target_variable format
            class_names = ["Class_0", "Class_1"]
            class_description = "Classes: Class_0, Class_1"
            
    elif 'target_classes' in semantic_info:
        # Binary classification like kr-vs-kp (old format)
        target_classes = semantic_info['target_classes']
        class_names = [tc['name'] for tc in target_classes]
        
        # Create class description
        class_meanings = [f"'{tc['name']}' ({tc['meaning']})" for tc in target_classes]
        class_description = f"Classes: {', '.join(class_meanings)}"
    
    else:
        # Default fallback - try to infer from target_type
        target_type = semantic_info.get('target_type', 'binary')
        
        if target_type == 'multiclass':
            # Make reasonable guess about number of classes
            dataset_name = semantic_info.get('dataset', '').lower()
            if 'digit' in dataset_name or 'mnist' in dataset_name:
                class_names = [str(i) for i in range(10)]
                class_description = "Classes: " + ", ".join([f"'{i}' (Digit {i})" for i in range(10)])
            elif 'letter' in dataset_name:
                class_names = [chr(ord('A') + i) for i in range(26)]
                class_description = "Classes: " + ", ".join([f"'{chr(ord('A') + i)}' (Letter {chr(ord('A') + i)})" for i in range(26)])
            else:
                # Generic multiclass
                class_names = [f"Class_{i}" for i in range(3)]
                class_description = "Classes: " + ", ".join([f"'Class_{i}'" for i in range(3)])
        else:
            # Binary classification
            class_names = ["Class_0", "Class_1"]
            class_description = "Classes: Class_0, Class_1"
    
    return class_names, class_description


def create_task_context_prefix(semantic_info: Dict[str, Any], column_descriptions: Dict[str, str], class_description: str) -> str:
    """Create a task context prefix for JOLT prompts."""
    dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', 'unknown'))
    
    # Use rich description if available
    if 'description' in semantic_info:
        base_description = semantic_info['description']
        use_case = semantic_info.get('use_case', '')
        
        # Create column list description
        if column_descriptions:
            num_cols = len(column_descriptions)
        else:
            num_cols = semantic_info.get('features', 'several')
        
        # Check if this is a regression task
        is_regression = 'target_description' in semantic_info
        
        # Create task prefix based on description and use case
        if use_case:
            return f"{base_description} Each example contains {num_cols} features. {use_case.split(',')[0].strip()}."
        else:
            if is_regression:
                target_desc = semantic_info.get('target_description', {})
                target_name = target_desc.get('name', 'target value')
                return f"{base_description} Each example contains {num_cols} features. Predict the {target_name}."
            else:
                return f"{base_description} Each example contains {num_cols} features. Predict the target class."
    
    # Fallback to original logic
    # Create column list description
    if column_descriptions:
        column_list = ", ".join(list(column_descriptions.values())[:10])  # Limit to first 10 for readability
        if len(column_descriptions) > 10:
            column_list += "..."
        num_cols = len(column_descriptions)
    else:
        column_list = "various features"
        num_cols = semantic_info.get('features', 'several')
    
    # Create dataset-specific context
    context_templates = {
        'adult': f"Each example contains {num_cols} columns: {column_list}. Predict whether a person's income is above or below $50K based on demographic and employment information.",
        'kr-vs-kp': f"Each example contains {num_cols} columns: {column_list}. Predict whether White can force a win in this chess King-Rook vs King-Pawn endgame position.",
        'letter': f"Each example contains {num_cols} columns: {column_list}. Predict which capital letter of the English alphabet this pixel display represents.",
        'diabetes': f"Each example contains {num_cols} columns: {column_list}. Predict whether a patient has diabetes based on diagnostic measurements.",
        'car': f"Each example contains {num_cols} columns: {column_list}. Predict the overall evaluation of a car based on its characteristics.",
        'vehicle': f"Each example contains {num_cols} columns: {column_list}. Predict the type of vehicle based on silhouette features.",
        'banknote-authentication': f"Each example contains {num_cols} columns: {column_list}. Predict whether a banknote is authentic or forged based on wavelet features.",
        'breast-w': f"Each example contains {num_cols} columns: {column_list}. Predict whether a breast mass is benign or malignant based on cell characteristics.",
        'credit-approval': f"Each example contains {num_cols} columns: {column_list}. Predict whether a credit application should be approved based on applicant information.",
        'heart-disease': f"Each example contains {num_cols} columns: {column_list}. Predict the presence of heart disease based on medical measurements.",
        'ionosphere': f"Each example contains {num_cols} columns: {column_list}. Predict whether radar signals show structure in the ionosphere.",
        'sonar': f"Each example contains {num_cols} columns: {column_list}. Predict whether sonar signals bounced off a metal cylinder or rock.",
        'har': f"Each example contains {num_cols} sensor features from smartphone accelerometer and gyroscope. Predict the human activity being performed.",
        'miceprotein': f"Each example contains {num_cols} protein expression levels from mice brain tissue. Predict the experimental condition (genotype, treatment, learning).",
        'pendigits': f"Each example contains {num_cols} coordinate features from pen trajectories. Predict which digit (0-9) was written."
    }
    
    # Check for dataset-specific context
    for dataset_key, template in context_templates.items():
        if dataset_key.lower() in dataset_name.lower():
            return template
    
    # Generic fallback based on feature types
    feat_desc = semantic_info.get('feature_description', '')
    if 'fourier' in feat_desc.lower():
        return f"Each example contains {num_cols} Fourier coefficient features. Predict the class based on the frequency domain representation."
    elif 'pixel' in feat_desc.lower():
        return f"Each example contains {num_cols} pixel intensity features. Predict the class based on the image representation."
    elif 'protein' in feat_desc.lower():
        return f"Each example contains {num_cols} protein expression features. Predict the biological condition or outcome."
    elif 'coordinate' in feat_desc.lower():
        return f"Each example contains {num_cols} coordinate features. Predict the target class based on spatial patterns."
    elif 'sensor' in feat_desc.lower() or 'acceleration' in feat_desc.lower():
        return f"Each example contains {num_cols} sensor measurement features. Predict the activity or state being measured."
    elif 'medical' in str(semantic_info).lower() or 'patient' in str(semantic_info).lower():
        # Check if regression for medical data
        if 'target_description' in semantic_info:
            target_desc = semantic_info.get('target_description', {})
            target_name = target_desc.get('name', 'target value')
            return f"Each example contains {num_cols} medical features: {column_list}. Predict the {target_name}."
        else:
            return f"Each example contains {num_cols} medical features: {column_list}. Predict the patient outcome or diagnosis."
    else:
        # Final fallback - check if regression
        if 'target_description' in semantic_info:
            target_desc = semantic_info.get('target_description', {})
            target_name = target_desc.get('name', 'target value')
            return f"Each example contains {num_cols} features: {column_list}. Predict the {target_name}."
        else:
            return f"Each example contains {num_cols} features: {column_list}. {class_description}"

def create_jolt_config_for_dataset(semantic_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JOLT configuration for a dataset."""
    dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', 'unknown'))
    
    # Extract information
    column_descriptions = extract_column_descriptions(semantic_info)
    class_names, class_description = extract_class_information(semantic_info)
    task_prefix = create_task_context_prefix(semantic_info, column_descriptions, class_description)
    
    return {
        'dataset_name': dataset_name,
        'task_prefix': task_prefix,
        'context': task_prefix,  # Add context field for backward compatibility
        'column_descriptions': column_descriptions,
        'class_names': class_names,
        'class_description': class_description,
        'num_features': len(column_descriptions),
        'num_classes': len(class_names)
    }


def process_semantic_files():
    """Process all semantic JSON files and create JOLT configurations."""
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
    
    all_configs = {}
    dataset_count = 0
    empty_count = 0
    
    for json_file in sorted(json_files):
        print(f"Processing {os.path.basename(json_file)}...")
        
        try:
            with open(json_file, 'r') as f:
                semantic_info = json.load(f)
            
            # Skip empty JSON files
            if not semantic_info:
                print(f"  Skipping empty file: {os.path.basename(json_file)}")
                empty_count += 1
                continue
            
            # Extract dataset name and task ID with better error handling
            dataset_name = semantic_info.get('dataset_name', semantic_info.get('dataset', Path(json_file).stem))
            task_id = semantic_info.get('_metadata', {}).get('task_id') or Path(json_file).stem
            
            # Create JOLT config for this dataset with defensive programming
            try:
                jolt_config = create_jolt_config_for_dataset(semantic_info)
                
                # Add task_id to the config for easier lookup
                jolt_config['task_id'] = task_id
                
                # Validate that we got reasonable results
                if jolt_config['num_features'] == 0 and jolt_config['num_classes'] <= 1:
                    print(f"  Warning: Generated config has no features or classes - using fallback")
                    # Create minimal fallback config
                    jolt_config = {
                        'task_id': task_id,
                        'dataset_name': dataset_name,
                        'task_prefix': f"Dataset {dataset_name}: Predict the target class based on the input features.",
                        'column_descriptions': {"feature_1": "Feature 1", "feature_2": "Feature 2"},
                        'class_names': ["Class_0", "Class_1"],
                        'class_description': "Classes: Class_0, Class_1",
                        'example_prompt': "[feature_1=0.5,feature_2=1.2] → Class_0",
                        'num_features': 2,
                        'num_classes': 2
                    }
                
                # Save individual config using task_id for consistent lookup
                config_filename = f"jolt_config_task_{task_id}.json"
                config_path = OUTPUT_DIR / config_filename
                
                with open(config_path, 'w') as f:
                    json.dump(jolt_config, f, indent=2)
                
                all_configs[str(task_id)] = jolt_config
                dataset_count += 1
                
                # Print summary for this dataset
                print(f"  Created config: {jolt_config['num_features']} features, {jolt_config['num_classes']} classes")
                
            except Exception as config_error:
                import traceback
                traceback.print_exc()
                print(f"  Error creating config for {dataset_name}: {config_error}")
                # Try to create a minimal config as fallback
                try:
                    fallback_config = {
                        'task_id': task_id,
                        'dataset_name': dataset_name,
                        'task_prefix': f"Dataset {dataset_name}: Predict the target class.",
                        'column_descriptions': {"feature_1": "Feature 1"},
                        'class_names': ["Class_0", "Class_1"],
                        'class_description': "Classes: Class_0, Class_1",
                        'example_prompt': "[feature_1=0.5] → Class_0",
                        'num_features': 1,
                        'num_classes': 2
                    }
                    
                    config_filename = f"jolt_config_task_{task_id}.json"
                    config_path = OUTPUT_DIR / config_filename
                    
                    with open(config_path, 'w') as f:
                        json.dump(fallback_config, f, indent=2)
                    
                    all_configs[str(task_id)] = fallback_config
                    dataset_count += 1
                    print(f"  Created fallback config: 1 feature, 2 classes")
                    
                except Exception as fallback_error:
                    print(f"  Failed to create even fallback config: {fallback_error}")
                    continue
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            # Try to create a minimal config based on filename
            try:
                dataset_name = Path(json_file).stem
                task_id = dataset_name  # Use filename as task_id fallback
                minimal_config = {
                    'task_id': task_id,
                    'dataset_name': dataset_name,
                    'task_prefix': f"Dataset {dataset_name}: Predict the target class.",
                    'column_descriptions': {"feature_1": "Feature 1"},
                    'class_names': ["Class_0", "Class_1"],
                    'class_description': "Classes: Class_0, Class_1",
                    'example_prompt': "[feature_1=0.5] → Class_0",
                    'num_features': 1,
                    'num_classes': 2
                }
                
                config_filename = f"jolt_config_task_{task_id}.json"
                config_path = OUTPUT_DIR / config_filename
                
                with open(config_path, 'w') as f:
                    json.dump(minimal_config, f, indent=2)
                
                all_configs[str(task_id)] = minimal_config
                dataset_count += 1
                print(f"  Created minimal config from filename: 1 feature, 2 classes")
                
            except Exception as minimal_error:
                print(f"  Failed to create minimal config: {minimal_error}")
                continue
    
    # Save aggregated config
    aggregated_path = OUTPUT_DIR / "all_jolt_configs.json"
    with open(aggregated_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    
    print(f"\nSuccessfully processed {dataset_count} datasets")
    print(f"Skipped {empty_count} empty files")
    print(f"Created {dataset_count} JOLT config files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Aggregated config: {aggregated_path}")


def main():
    """Main function."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Synthesizing JOLT configurations from semantic information...")
    process_semantic_files()
    print("\nDone!")


if __name__ == "__main__":
    main()