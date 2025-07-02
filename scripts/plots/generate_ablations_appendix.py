#!/usr/bin/env python3
"""
Script to generate LaTeX-formatted documentation of MARVIS ablation results
for inclusion in the appendix.
"""

import os
import json
import glob
from pathlib import Path
import re
from collections import defaultdict, Counter

# Configuration
ABLATIONS_PATH = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/results/ablations"
OUTPUT_PATH = "/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/marvis/marvis_claude/ablations_appendix_content.tex"

# Method categorization and descriptions
METHOD_CATEGORIES = {
    "Basic Visualizations": {
        "basic_tsne": "Standard t-SNE visualization with default parameters",
        "tsne_3d": "Three-dimensional t-SNE visualization for enhanced spatial understanding",
        "tsne_high_dpi": "High-resolution t-SNE with increased image quality",
        "tsne_high_perplexity": "t-SNE with modified perplexity parameter for different clustering",
    },
    "Enhanced Single Methods": {
        "tsne_knn": "t-SNE with k-nearest neighbor information overlay",
        "tsne_perturbation_axes": "t-SNE with perturbation analysis for uncertainty quantification",
        "tsne_semantic_axes": "t-SNE with semantic class labels and axes descriptions",
        "tsne_3d_knn": "3D t-SNE visualization with k-NN connections displayed",
        "tsne_3d_perturbation": "3D t-SNE with perturbation analysis for spatial uncertainty",
    },
    "Multi-Visualization Methods": {
        "multi_comprehensive": "PCA + t-SNE + Spectral + Isomap comprehensive view",
        "multi_pca_tsne": "Combined PCA and t-SNE dual visualization",
        "multi_pca_tsne_spectral": "Triple visualization: PCA + t-SNE + Spectral embedding",
        "multi_linear_nonlinear": "Linear and nonlinear dimensionality reduction comparison",
        "multi_local_global": "Local and global structure preservation methods",
        "multi_with_umap": "Multi-method visualization including UMAP",
        "multi_grid_layout": "Grid-based layout for systematic method comparison",
    },
    "Specialized Methods": {
        "decision_regions_svm": "SVM decision boundary visualization with regions",
        "frequent_patterns": "Pattern mining visualization for feature relationships",
        "metadata_comprehensive": "Metadata-enhanced comprehensive visualization approach",
    }
}

def escape_latex(text):
    """Escape special LaTeX characters in text."""
    if not isinstance(text, str):
        return str(text)
    
    # Basic LaTeX character escaping
    latex_chars = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '\\': '\\textbackslash{}',
    }
    
    for char, escape in latex_chars.items():
        text = text.replace(char, escape)
    
    return text

def format_method_name(method_name):
    """Format method name for LaTeX with proper underscore handling."""
    return method_name.replace('_', '\\_')

def truncate_text(text, max_length=200):
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def analyze_test_directory(test_dir):
    """Analyze a single test directory and extract information."""
    test_info = {
        'test_id': os.path.basename(test_dir).replace('test_vlm_outputs_', ''),
        'methods': {},
        'metadata': {}
    }
    
    # Load metadata if available
    metadata_dir = os.path.join(test_dir, '_METADATA')
    if os.path.exists(metadata_dir):
        summary_file = os.path.join(metadata_dir, 'test_summary.json')
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    test_info['metadata'] = json.load(f)
            except:
                pass
    
    # Analyze each method directory
    for item in os.listdir(test_dir):
        method_dir = os.path.join(test_dir, item)
        if os.path.isdir(method_dir) and item != '_METADATA':
            method_info = analyze_method_directory(method_dir, item)
            if method_info:
                test_info['methods'][item] = method_info
    
    return test_info

def analyze_method_directory(method_dir, method_name):
    """Analyze a single method directory."""
    method_info = {
        'name': method_name,
        'prompts': [],
        'responses': [],
        'visualizations': [],
        'performance': {}
    }
    
    # Collect prompts
    prompt_files = glob.glob(os.path.join(method_dir, 'prompt_*.txt'))
    for prompt_file in sorted(prompt_files)[:3]:  # Limit to first 3 examples
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                method_info['prompts'].append({
                    'file': os.path.basename(prompt_file),
                    'content': content
                })
        except:
            pass
    
    # Collect responses
    response_files = glob.glob(os.path.join(method_dir, 'response_*.txt'))
    for response_file in sorted(response_files)[:3]:  # Limit to first 3 examples
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                method_info['responses'].append({
                    'file': os.path.basename(response_file),
                    'content': content
                })
        except:
            pass
    
    # Collect visualization info
    viz_files = glob.glob(os.path.join(method_dir, '*.png'))
    method_info['visualizations'] = [os.path.basename(f) for f in viz_files[:3]]
    
    # Load detailed outputs for performance info
    detailed_file = os.path.join(method_dir, 'detailed_vlm_outputs.json')
    if os.path.exists(detailed_file):
        try:
            with open(detailed_file, 'r') as f:
                detailed_data = json.load(f)
                if 'performance_summary' in detailed_data:
                    method_info['performance'] = detailed_data['performance_summary']
        except:
            pass
    
    return method_info

def generate_method_summary_table():
    """Generate LaTeX table summarizing all MARVIS method variants."""
    latex_content = []
    
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\begin{tabular}{l|l|p{7cm}}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Category} & \\textbf{Method} & \\textbf{Description} \\\\")
    latex_content.append("\\midrule")
    
    for category, methods in METHOD_CATEGORIES.items():
        first_method = True
        for method, description in methods.items():
            if first_method:
                latex_content.append(f"\\multirow{{{len(methods)}}}{{*}}{{{escape_latex(category)}}} & {format_method_name(method)} & {escape_latex(description)} \\\\")
                first_method = False
            else:
                latex_content.append(f" & {format_method_name(method)} & {escape_latex(description)} \\\\")
        latex_content.append("\\midrule")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\caption{\\textbf{MARVIS Method Variants Overview.} Comprehensive summary of visualization approaches evaluated in ablation studies, categorized by methodology type and complexity level.}")
    latex_content.append("\\label{tab:marvis_methods}")
    latex_content.append("\\end{table}")
    
    return '\n'.join(latex_content)

def generate_method_documentation(test_data, method_name, max_examples=2):
    """Generate LaTeX documentation for a specific method."""
    latex_content = []
    
    # Find method category
    category = "Other"
    for cat, methods in METHOD_CATEGORIES.items():
        if method_name in methods:
            category = cat
            break
    
    latex_content.append(f"\\subsubsection{{{format_method_name(method_name)}}}")
    latex_content.append(f"\\label{{sec:{method_name.replace('_', '-')}}}")
    latex_content.append("")
    
    # Add method description
    description = "Custom visualization method"
    for cat, methods in METHOD_CATEGORIES.items():
        if method_name in methods:
            description = methods[method_name]
            break
    
    latex_content.append(f"\\textbf{{Method Category}}: {escape_latex(category)}")
    latex_content.append("")
    latex_content.append(f"\\textbf{{Description}}: {escape_latex(description)}")
    latex_content.append("")
    
    # Collect examples from different tests
    examples_collected = 0
    for test_id, test_info in test_data.items():
        if method_name in test_info['methods'] and examples_collected < max_examples:
            method_info = test_info['methods'][method_name]
            
            latex_content.append(f"\\textbf{{Example {examples_collected + 1} (Test {test_id}):}}")
            latex_content.append("")
            
            # Add prompt example
            if method_info['prompts']:
                prompt = method_info['prompts'][0]
                prompt_text = truncate_text(prompt['content'], 300)
                latex_content.append("\\textbf{Prompt}:")
                latex_content.append("\\begin{quote}")
                latex_content.append("\\footnotesize")
                latex_content.append(escape_latex(prompt_text))
                latex_content.append("\\end{quote}")
                latex_content.append("")
            
            # Add response example
            if method_info['responses']:
                response = method_info['responses'][0]
                response_text = truncate_text(response['content'], 200)
                latex_content.append("\\textbf{VLM Response}:")
                latex_content.append("\\begin{quote}")
                latex_content.append("\\footnotesize")
                latex_content.append(f"\\textit{{{escape_latex(response_text)}}}")
                latex_content.append("\\end{quote}")
                latex_content.append("")
            
            # Add performance info if available
            if method_info['performance']:
                perf = method_info['performance']
                if 'accuracy' in perf:
                    latex_content.append(f"\\textbf{{Performance}}: {perf.get('accuracy', 'N/A'):.1%} accuracy")
                    latex_content.append("")
            
            examples_collected += 1
            latex_content.append("")
    
    return '\n'.join(latex_content)

def main():
    """Main function to generate the ablations appendix content."""
    print("Analyzing ablation results...")
    
    # Collect data from all test directories
    test_data = {}
    test_dirs = glob.glob(os.path.join(ABLATIONS_PATH, "test_vlm_outputs_*"))
    
    for test_dir in sorted(test_dirs)[:5]:  # Limit to first 5 tests for manageable output
        print(f"Processing {os.path.basename(test_dir)}...")
        test_info = analyze_test_directory(test_dir)
        test_data[test_info['test_id']] = test_info
    
    # Generate LaTeX content
    latex_content = []
    
    # Header
    latex_content.append("\\section{MARVIS Method Variants: Detailed Ablation Documentation}")
    latex_content.append("\\label{sec:marvis_ablations}")
    latex_content.append("")
    latex_content.append("This section provides comprehensive documentation of all MARVIS visualization method variants evaluated in our ablation studies. Each method variant represents a different approach to transforming embedding spaces into visual representations for VLM reasoning.")
    latex_content.append("")
    
    # Method summary table
    latex_content.append("\\subsection{Method Variants Overview}")
    latex_content.append("")
    latex_content.append(generate_method_summary_table())
    latex_content.append("")
    
    # Detailed method documentation
    latex_content.append("\\subsection{Detailed Method Documentation}")
    latex_content.append("")
    latex_content.append("The following subsections provide detailed examples of prompts, VLM responses, and performance characteristics for each method variant, demonstrating how different visualization approaches elicit different reasoning patterns from the VLM.")
    latex_content.append("")
    
    # Get all unique methods across tests
    all_methods = set()
    for test_info in test_data.values():
        all_methods.update(test_info['methods'].keys())
    
    # Generate documentation for each method
    categorized_methods = {}
    for category, methods in METHOD_CATEGORIES.items():
        categorized_methods[category] = []
        for method in methods:
            if method in all_methods:
                categorized_methods[category].append(method)
    
    # Add uncategorized methods
    uncategorized = [m for m in all_methods if not any(m in methods for methods in METHOD_CATEGORIES.values())]
    if uncategorized:
        categorized_methods["Other Methods"] = uncategorized
    
    for category, methods in categorized_methods.items():
        if not methods:
            continue
            
        latex_content.append(f"\\subsubsection{{{escape_latex(category)}}}")
        latex_content.append("")
        
        for method in sorted(methods):
            method_doc = generate_method_documentation(test_data, method)
            latex_content.append(method_doc)
            latex_content.append("")
    
    # Write to file
    full_content = '\n'.join(latex_content)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"LaTeX content generated: {OUTPUT_PATH}")
    print(f"Processed {len(test_data)} test directories")
    print(f"Documented {len(all_methods)} method variants")
    
    # Print summary statistics
    method_counts = Counter()
    for test_info in test_data.values():
        method_counts.update(test_info['methods'].keys())
    
    print("\nMost common methods:")
    for method, count in method_counts.most_common(10):
        print(f"  {method}: {count} tests")

if __name__ == "__main__":
    main()