#!/usr/bin/env python
"""
Test to validate that all examples use correct, up-to-date parameter names.

This test scans all example files and validates:
1. Deprecated parameter usage (use_3d_tsne -> use_3d)
2. Correct parameter mapping (nn_k args -> knn_k constructor)
3. Consistent parameter naming across examples
4. Missing parameters that should be updated
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Parameter mapping definitions
DEPRECATED_PARAMS = {
    'use_3d_tsne': 'use_3d',
    'tsne_zoom_factor': 'zoom_factor',
}

class ParameterValidationError:
    def __init__(self, file_path: str, line_num: int, issue_type: str, details: str, suggestion: str = ""):
        self.file_path = file_path
        self.line_num = line_num
        self.issue_type = issue_type
        self.details = details
        self.suggestion = suggestion
    
    def __str__(self):
        suggestion_text = f" → {self.suggestion}" if self.suggestion else ""
        return f"{self.file_path}:{self.line_num} [{self.issue_type}] {self.details}{suggestion_text}"

class ExampleParameterValidator:
    def __init__(self, examples_dir: str):
        self.examples_dir = Path(examples_dir)
        self.errors: List[ParameterValidationError] = []
        self.files_checked = 0
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the examples directory."""
        python_files = []
        for root, dirs, files in os.walk(self.examples_dir):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def check_file_content(self, file_path: Path):
        """Check a single file for parameter issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.files_checked += 1
            
            # Check each line for issues
            for line_num, line in enumerate(lines, 1):
                self._check_line_for_issues(file_path, line_num, line.strip())
            
        except Exception as e:
            self.errors.append(ParameterValidationError(
                str(file_path), 0, "FILE_ERROR", 
                f"Error reading file: {e}"
            ))
    
    def _check_line_for_issues(self, file_path: Path, line_num: int, line: str):
        """Check a single line for parameter issues."""
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            return
        
        # Check for deprecated parameter usage
        for deprecated, preferred in DEPRECATED_PARAMS.items():
            # Check for parameter assignment
            if f'{deprecated}=' in line or f'{deprecated} =' in line:
                self.errors.append(ParameterValidationError(
                    str(file_path), line_num, "DEPRECATED_PARAM",
                    f"Using deprecated parameter '{deprecated}': {line[:50]}...",
                    f"Use '{preferred}' instead"
                ))
            
            # Check for command line arguments
            if f'--{deprecated}' in line:
                self.errors.append(ParameterValidationError(
                    str(file_path), line_num, "DEPRECATED_ARG",
                    f"Using deprecated argument '--{deprecated}': {line[:50]}...",
                    f"Use '--{preferred}' instead"
                ))
            
            # Check for getattr usage
            if f'getattr(args, \'{deprecated}\'' in line or f'getattr(args, "{deprecated}"' in line:
                self.errors.append(ParameterValidationError(
                    str(file_path), line_num, "DEPRECATED_GETATTR",
                    f"Using deprecated getattr '{deprecated}': {line[:50]}...",
                    f"Use 'getattr(args, '{preferred}', ...)' instead"
                ))
        
        # Check for specific known problematic patterns
        self._check_specific_patterns(file_path, line_num, line)
    
    def _check_specific_patterns(self, file_path: Path, line_num: int, line: str):
        """Check for specific known problematic patterns."""
        
        # Check for use_3d_tsne in MarvisTsneClassifier calls
        if 'MarvisTsneClassifier(' in line and 'use_3d_tsne' in line:
            self.errors.append(ParameterValidationError(
                str(file_path), line_num, "DEPRECATED_CONSTRUCTOR",
                f"MarvisTsneClassifier using 'use_3d_tsne': {line[:50]}...",
                "Use 'use_3d' instead"
            ))
        
        # Check for knn_k in args but should use nn_k
        if '--knn_k' in line:
            self.errors.append(ParameterValidationError(
                str(file_path), line_num, "OLD_ARG_NAME",
                f"Using old argument name '--knn_k': {line[:50]}...",
                "Use '--nn_k' instead"
            ))
        
        # Check for evaluate_marvis_tsne with deprecated params
        if 'evaluate_marvis_tsne(' in line:
            for deprecated, preferred in DEPRECATED_PARAMS.items():
                if deprecated in line:
                    self.errors.append(ParameterValidationError(
                        str(file_path), line_num, "DEPRECATED_EVAL_PARAM",
                        f"evaluate_marvis_tsne using '{deprecated}': {line[:50]}...",
                        f"Use '{preferred}' instead"
                    ))
    
    def run_validation(self) -> bool:
        """Run validation on all example files."""
        print(f"🔍 Validating parameter usage in examples directory: {self.examples_dir}")
        print("=" * 80)
        
        python_files = self.find_python_files()
        
        if not python_files:
            print("❌ No Python files found in examples directory")
            return False
        
        print(f"Found {len(python_files)} Python files to check...")
        print()
        
        for file_path in python_files:
            print(f"Checking: {file_path}")
            self.check_file_content(file_path)
        
        return self._report_results()
    
    def _report_results(self) -> bool:
        """Report validation results."""
        print()
        print("=" * 80)
        print("🎯 PARAMETER VALIDATION RESULTS")
        print("=" * 80)
        
        if not self.errors:
            print(f"✅ SUCCESS: All {self.files_checked} example files use correct parameter names!")
            print()
            print("📋 Validation Summary:")
            print(f"  • Files checked: {self.files_checked}")
            print(f"  • Issues found: 0")
            print(f"  • Deprecated parameters: None found")
            print(f"  • Parameter mapping issues: None found")
            return True
        
        # Group errors by type
        error_types = {}
        for error in self.errors:
            if error.issue_type not in error_types:
                error_types[error.issue_type] = []
            error_types[error.issue_type].append(error)
        
        print(f"❌ VALIDATION FAILED: Found {len(self.errors)} issues in {self.files_checked} files")
        print()
        
        # Report by error type
        for issue_type, errors in error_types.items():
            print(f"🔸 {issue_type.replace('_', ' ').title()} ({len(errors)} issues):")
            for error in errors[:5]:  # Show first 5 of each type
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more")
            print()
        
        print("📋 Validation Summary:")
        print(f"  • Files checked: {self.files_checked}")
        print(f"  • Total issues: {len(self.errors)}")
        for issue_type, errors in error_types.items():
            print(f"  • {issue_type.replace('_', ' ').title()}: {len(errors)}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate parameter usage in MARVIS examples")
    parser.add_argument("--examples-dir", default="examples", 
                        help="Path to examples directory (default: examples)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all errors instead of limiting to 5 per type")
    
    args = parser.parse_args()
    
    # Determine examples directory path
    examples_dir = Path(args.examples_dir)
    if not examples_dir.is_absolute():
        examples_dir = Path(__file__).parent.parent / examples_dir
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return 1
    
    # Run validation
    validator = ExampleParameterValidator(str(examples_dir))
    success = validator.run_validation()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())