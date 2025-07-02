#!/usr/bin/env python3
"""
Script to generate semantic information for OpenML tasks using Claude API.

This script:
1. Queries OpenML API for basic task information
2. Uses Claude API to retrieve enhanced semantic information by searching multiple sources
3. Saves structured JSON files compatible with MARVIS's semantic metadata format

The script generates comprehensive semantic descriptions including:
- Dataset domain and application context
- Feature meanings and relationships
- Target variable interpretation (classification classes or regression target)
- Data collection methodology and historical context
- Usage recommendations and inference notes

Usage:
    # Single task
    python generate_openml_semantic_info.py --task_id 3 --output_dir data/semantic_info

    # Multiple tasks
    python generate_openml_semantic_info.py --task_ids "3,23,53" --output_dir data/semantic_info

    # Batch processing from file
    python generate_openml_semantic_info.py --task_ids_file task_list.txt --output_dir data/semantic_info

Requirements:
- OpenML Python package (pip install openml)
- Anthropic Claude API access and API key
- MARVIS repository structure for semantic metadata integration
"""

import os
import argparse
import json
import logging
import time
import openml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import sys

# Try to import anthropic, provide helpful error if not available
try:
    import anthropic
except ImportError:
    print("Error: anthropic package not found. Please install it with:")
    print("pip install anthropic")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate semantic information for OpenML tasks using Claude API")
    
    # Task specification (mutually exclusive group)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task_id",
        type=int,
        help="Single OpenML task ID to process"
    )
    task_group.add_argument(
        "--task_ids",
        type=str,
        help="Comma-separated list of OpenML task IDs"
    )
    task_group.add_argument(
        "--task_ids_file",
        type=str,
        help="Path to text file containing task IDs (one per line)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./semantic_info",
        help="Directory to save semantic information JSON files"
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing semantic information files"
    )
    
    # Claude API configuration
    parser.add_argument(
        "--claude_model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use for semantic information generation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="Maximum tokens for Claude API response"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Anthropic API key (if not set as environment variable)"
    )
    
    # Processing options
    parser.add_argument(
        "--delay_between_requests",
        type=float,
        default=1.0,
        help="Delay in seconds between Claude API requests to respect rate limits"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API requests"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Fetch OpenML data but don't make Claude API calls"
    )
    
    return parser.parse_args()

def setup_claude_client(api_key: Optional[str] = None) -> anthropic.Anthropic:
    """Setup Claude API client."""
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
    else:
        # Will use ANTHROPIC_API_KEY environment variable
        try:
            client = anthropic.Anthropic()
        except anthropic.AuthenticationError:
            logger.error("No valid Anthropic API key found. Please set ANTHROPIC_API_KEY environment variable or use --api_key argument.")
            sys.exit(1)
    
    logger.info("Claude API client initialized successfully")
    return client

def fetch_openml_task_info(task_id: int) -> Dict[str, Any]:
    """Fetch comprehensive information about an OpenML task."""
    logger.info(f"Fetching OpenML task {task_id} information...")
    
    try:
        # Get task
        task = openml.tasks.get_task(task_id)
        
        # Get dataset
        dataset = task.get_dataset()
        
        # Extract basic information
        task_info = {
            "task_id": task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset.name,
            "task_type": task.task_type,
            "target_names": task.target_name if hasattr(task, 'target_name') else None,
            "num_instances": None,
            "num_features": None,
            "num_missing_values": None,
            "class_labels": None,
            "feature_names": None,
            "feature_types": None
        }
        
        # Extract dataset qualities if available
        if hasattr(dataset, 'qualities') and dataset.qualities:
            task_info.update({
                "num_instances": dataset.qualities.get("NumberOfInstances"),
                "num_features": dataset.qualities.get("NumberOfFeatures"),
                "num_missing_values": dataset.qualities.get("NumberOfMissingValues"),
                "num_classes": dataset.qualities.get("NumberOfClasses")
            })
        
        # Extract feature information
        if hasattr(dataset, 'features') and isinstance(dataset.features, dict):
            features = dataset.features
            task_info["feature_names"] = list(features.keys())
            task_info["feature_types"] = {name: feat.data_type for name, feat in features.items()}
        
        # Extract class labels for classification tasks
        if hasattr(task, 'class_labels') and task.class_labels:
            task_info["class_labels"] = task.class_labels
        
        # Get dataset description and additional metadata
        task_info["dataset_description"] = dataset.description if hasattr(dataset, 'description') else None
        task_info["dataset_url"] = dataset.url if hasattr(dataset, 'url') else None
        task_info["citation"] = dataset.citation if hasattr(dataset, 'citation') else None
        task_info["creator"] = dataset.creator if hasattr(dataset, 'creator') else None
        task_info["collection_date"] = dataset.collection_date if hasattr(dataset, 'collection_date') else None
        
        logger.info(f"Successfully fetched information for task {task_id}: {dataset.name}")
        return task_info
        
    except Exception as e:
        logger.error(f"Error fetching OpenML task {task_id}: {e}")
        raise

def create_claude_prompt(task_info: Dict[str, Any]) -> str:
    """Create a comprehensive prompt for Claude to generate semantic information."""
    
    task_type = task_info.get('task_type', '').lower()
    is_regression = 'regression' in task_type
    is_classification = 'classification' in task_type
    
    # Base prompt
    prompt = f"""You are an expert data scientist and domain specialist. I need you to research and provide comprehensive semantic information about an OpenML machine learning task.

**Task Information:**
- Task ID: {task_info['task_id']}
- Dataset Name: {task_info['dataset_name']}
- Task Type: {task_info['task_type']}
- Number of Instances: {task_info.get('num_instances', 'Unknown')}
- Number of Features: {task_info.get('num_features', 'Unknown')}
- Target Variable(s): {task_info.get('target_names', 'Unknown')}"""

    if task_info.get('feature_names'):
        feature_list = ', '.join(str(name) for name in task_info['feature_names'][:20])  # Limit to first 20 features
        if len(task_info['feature_names']) > 20:
            feature_list += f", ... ({len(task_info['feature_names'])} total features)"
        prompt += f"\n- Feature Names: {feature_list}"
    
    if task_info.get('class_labels') and is_classification:
        prompt += f"\n- Target Classes: {', '.join(map(str, task_info['class_labels']))}"
    
    if task_info.get('dataset_description'):
        prompt += f"\n- Dataset Description: {task_info['dataset_description'][:500]}..."
    
    prompt += f"""

**Instructions:**
Please search for information about this dataset from multiple sources including:
1. OpenML database and documentation
2. UCI Machine Learning Repository  
3. Academic papers and publications
4. Domain-specific databases and repositories
5. General web sources with credible information

Provide a comprehensive analysis in the following JSON format:

```json
{{
  "dataset_name": "{task_info['dataset_name']}",
  "description": "Comprehensive description of the dataset, its domain, and purpose",
  "original_source": {{
    "creator": "Name of creator/institution",
    "institution": "Originating institution or organization", 
    "date": "Creation or publication date",
    "publication": "Associated publication or paper (if any)"
  }},
  "columns": [
    {{
      "name": "feature_name",
      "semantic_description": "Detailed explanation of what this feature represents and measures",
      "data_type": "data type and possible values"
    }}
  ],"""

    if is_classification:
        prompt += '''
  "target_classes": [
    {
      "name": "class_name",
      "meaning": "Detailed explanation of what this class represents"
    }
  ],'''
    elif is_regression:
        prompt += '''
  "target_description": {
    "name": "target_variable_name",
    "meaning": "Detailed explanation of what the target variable represents and measures",
    "units": "Units of measurement (if applicable)",
    "range": "Typical or theoretical range of values"
  },'''

    prompt += '''
  "dataset_history": "Historical context, how the dataset was created, why it was collected, and its significance in the field",
  "inference_notes": "Additional insights about the data collection process, known biases, interpretation guidelines, or other important considerations for analysis"
}
```

**Requirements:**
1. Be thorough and accurate - research the dataset comprehensively
2. Provide semantic meaning, not just technical descriptions
3. Include domain-specific context and interpretation
4. Note any known limitations, biases, or considerations
5. If information is not readily available, clearly indicate uncertainty
6. Focus on practical insights that would help in machine learning analysis
7. Ensure all feature descriptions are meaningful and contextual

Please provide the complete JSON response with all requested fields filled out based on your research.'''

    return prompt

def query_claude_for_semantic_info(client: anthropic.Anthropic, prompt: str, 
                                 model: str, max_tokens: int, max_retries: int = 3) -> Dict[str, Any]:
    """Query Claude API for semantic information with retry logic."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Querying Claude API (attempt {attempt + 1}/{max_retries})...")
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual, consistent responses
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in Claude response")
            
            json_text = response_text[json_start:json_end]
            semantic_info = json.loads(json_text)
            
            logger.info("Successfully received and parsed Claude response")
            return semantic_info
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to parse JSON after {max_retries} attempts: {e}")
        
        except Exception as e:
            logger.warning(f"Claude API error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Claude API failed after {max_retries} attempts: {e}")
        
        # Wait before retry
        time.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError("Unexpected error in Claude API query")

def save_semantic_info(semantic_info: Dict[str, Any], task_id: int, output_dir: str):
    """Save semantic information to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{task_id}.json")
    
    # Add metadata
    semantic_info["_metadata"] = {
        "task_id": task_id,
        "generated_at": datetime.now().isoformat(),
        "generator": "generate_openml_semantic_info.py",
        "claude_model": "claude-3-5-sonnet-20241022"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(semantic_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved semantic information to {output_path}")

def load_task_ids_from_file(file_path: str) -> List[int]:
    """Load task IDs from a text file."""
    task_ids = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        task_ids.append(int(line))
                    except ValueError:
                        logger.warning(f"Skipping invalid task ID: {line}")
        
        logger.info(f"Loaded {len(task_ids)} task IDs from {file_path}")
        return task_ids
        
    except Exception as e:
        logger.error(f"Error reading task IDs from file {file_path}: {e}")
        raise

def process_task(client: anthropic.Anthropic, task_id: int, args) -> bool:
    """Process a single task to generate semantic information."""
    logger.info(f"Processing task {task_id}...")
    
    # Check if output already exists
    output_path = os.path.join(args.output_dir, f"{task_id}.json")
    if os.path.exists(output_path) and not args.force_overwrite:
        logger.info(f"Semantic information for task {task_id} already exists. Skipping.")
        return True
    
    try:
        # Fetch OpenML task information
        task_info = fetch_openml_task_info(task_id)
        
        if args.dry_run:
            logger.info(f"Dry run: Would generate semantic info for task {task_id} ({task_info['dataset_name']})")
            return True
        
        # Create Claude prompt
        prompt = create_claude_prompt(task_info)
        
        # Query Claude for semantic information
        semantic_info = query_claude_for_semantic_info(
            client, prompt, args.claude_model, args.max_tokens, args.max_retries
        )
        
        # Save semantic information
        save_semantic_info(semantic_info, task_id, args.output_dir)
        
        logger.info(f"Successfully processed task {task_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        return False

def main():
    args = parse_args()
    
    # Setup Claude client
    if not args.dry_run:
        client = setup_claude_client(args.api_key)
    else:
        client = None
    
    # Determine task IDs to process
    task_ids = []
    
    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids:
        # Handle newlines and extra whitespace in task_ids string
        clean_ids = args.task_ids.replace('\n', '').replace(' ', '')
        task_ids = [int(id.strip()) for id in clean_ids.split(',') if id.strip()]
    elif args.task_ids_file:
        task_ids = load_task_ids_from_file(args.task_ids_file)
    
    if not task_ids:
        logger.error("No task IDs specified")
        return
    
    logger.info(f"Processing {len(task_ids)} tasks: {task_ids}")
    
    # Process each task
    successful_tasks = 0
    failed_tasks = []
    
    for i, task_id in enumerate(task_ids):
        logger.info(f"Processing task {i+1}/{len(task_ids)}: {task_id}")
        
        try:
            if process_task(client, task_id, args):
                successful_tasks += 1
            else:
                failed_tasks.append(task_id)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing task {task_id}: {e}")
            failed_tasks.append(task_id)
        
        # Rate limiting delay
        if i < len(task_ids) - 1 and not args.dry_run:
            time.sleep(args.delay_between_requests)
    
    # Summary
    logger.info(f"Processing complete: {successful_tasks}/{len(task_ids)} tasks successful")
    if failed_tasks:
        logger.warning(f"Failed tasks: {failed_tasks}")
    
    # Save processing summary
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_tasks": len(task_ids),
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "task_ids_processed": task_ids,
        "arguments": vars(args)
    }
    
    summary_path = os.path.join(args.output_dir, "processing_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processing summary saved to {summary_path}")

if __name__ == "__main__":
    main()