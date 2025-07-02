#!/usr/bin/env python
"""
Test script to verify that the marvis package is installed correctly.
"""

import os
import sys

def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Python path:", sys.path)
    
    print("\nTrying to import marvis...")
    try:
        import marvis
        print("✅ Successfully imported marvis package")
        print("marvis location:", marvis.__file__)
        print("marvis version:", marvis.__version__)
        
        print("\nChecking submodules...")
        modules = ["data", "models", "train", "utils"]
        for module_name in modules:
            try:
                module = __import__(f"marvis.{module_name}", fromlist=[""])
                print(f"✅ Successfully imported marvis.{module_name} from {module.__file__}")
            except ImportError as e:
                print(f"❌ Failed to import marvis.{module_name}: {e}")
        
        print("\nChecking specific functions...")
        try:
            from marvis.data import load_dataset
            print("✅ Successfully imported load_dataset")
        except ImportError as e:
            print(f"❌ Failed to import load_dataset: {e}")
            
        try:
            from marvis.data import get_tabpfn_embeddings
            print("✅ Successfully imported get_tabpfn_embeddings")
        except ImportError as e:
            print(f"❌ Failed to import get_tabpfn_embeddings: {e}")
            
        try:
            from marvis.data import create_llm_dataset
            print("✅ Successfully imported create_llm_dataset")
        except ImportError as e:
            print(f"❌ Failed to import create_llm_dataset: {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import marvis: {e}")
    
    print("\nListing all installed packages:")
    import pkg_resources
    for package in pkg_resources.working_set:
        print(f"  {package.project_name}=={package.version}")
    
    print("\nListing all Python path directories to check that the install location is in Python's path:")
    for path in sys.path:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} (does not exist)")
            
if __name__ == "__main__":
    main()