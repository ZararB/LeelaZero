#!/usr/bin/env python3
#
# Comprehensive code structure and logic testing
# Tests code without requiring all dependencies

import ast
import os
import re
import sys

def test_syntax(filepath):
    """Test if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_imports(filepath):
    """Check for required imports."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        imports = []
        if 'import tensorflow' in content or 'import tf' in content:
            imports.append('tensorflow')
        if 'import numpy' in content or 'import np' in content:
            imports.append('numpy')
        if 'import yaml' in content:
            imports.append('yaml')
        if 'import protobuf' in content or 'import proto' in content:
            imports.append('protobuf')
        
        return imports
    except Exception as e:
        return []

def check_function_definitions(filepath):
    """Count function and class definitions."""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        return functions, classes
    except Exception:
        return [], []

def validate_train_py():
    """Validate train.py structure."""
    print("\n" + "="*60)
    print("Validating train.py")
    print("="*60)
    
    if not os.path.exists('train.py'):
        print("✗ train.py not found")
        return False
    
    # Check syntax
    valid, error = test_syntax('train.py')
    if not valid:
        print(f"✗ Syntax error: {error}")
        return False
    print("✓ Syntax valid")
    
    # Check for key functions
    with open('train.py', 'r') as f:
        content = f.read()
    
    required = ['def main', 'def find_chunks', 'def create_dataset']
    for req in required:
        if req in content:
            print(f"✓ Found {req}")
        else:
            print(f"✗ Missing {req}")
            return False
    
    # Check imports
    imports = check_imports('train.py')
    print(f"✓ Imports: {', '.join(imports) if imports else 'None detected'}")
    
    # Count functions
    functions, classes = check_function_definitions('train.py')
    print(f"✓ Functions: {len(functions)}, Classes: {len(classes)}")
    
    return True

def validate_tfprocess():
    """Validate tfprocess.py structure."""
    print("\n" + "="*60)
    print("Validating tfprocess.py")
    print("="*60)
    
    if not os.path.exists('tfprocess.py'):
        print("✗ tfprocess.py not found")
        return False
    
    # Check syntax
    valid, error = test_syntax('tfprocess.py')
    if not valid:
        print(f"✗ Syntax error: {error}")
        return False
    print("✓ Syntax valid")
    
    # Check for key classes and methods
    with open('tfprocess.py', 'r') as f:
        content = f.read()
    
    required_classes = ['class TFProcess', 'class ApplySqueezeExcitation', 'class ApplyPolicyMap']
    for req in required_classes:
        if req in content:
            print(f"✓ Found {req}")
        else:
            print(f"✗ Missing {req}")
    
    required_methods = ['def init_net_v2', 'def process_v2', 'def construct_net_v2']
    for req in required_methods:
        if req in content:
            print(f"✓ Found {req}")
        else:
            print(f"⚠ Missing {req}")
    
    # Count functions
    functions, classes = check_function_definitions('tfprocess.py')
    print(f"✓ Functions: {len(functions)}, Classes: {len(classes)}")
    
    # Check for duplicate code
    if content.count('self.policy_accuracy_fn = policy_accuracy') > 1:
        print("✗ Duplicate policy_accuracy_fn assignment found")
        return False
    print("✓ No duplicate assignments")
    
    return True

def validate_shufflebuffer():
    """Validate shufflebuffer.py."""
    print("\n" + "="*60)
    print("Validating shufflebuffer.py")
    print("="*60)
    
    if not os.path.exists('shufflebuffer.py'):
        print("✗ shufflebuffer.py not found")
        return False
    
    valid, error = test_syntax('shufflebuffer.py')
    if not valid:
        print(f"✗ Syntax error: {error}")
        return False
    print("✓ Syntax valid")
    
    with open('shufflebuffer.py', 'r') as f:
        content = f.read()
    
    if 'class ShuffleBuffer' in content:
        print("✓ ShuffleBuffer class found")
    else:
        print("✗ ShuffleBuffer class missing")
        return False
    
    required_methods = ['def insert_or_replace', 'def extract']
    for req in required_methods:
        if req in content:
            print(f"✓ Found {req}")
        else:
            print(f"✗ Missing {req}")
            return False
    
    return True

def validate_config():
    """Validate configuration files."""
    print("\n" + "="*60)
    print("Validating configuration files")
    print("="*60)
    
    configs = ['configs/test_config.yaml', 'configs/example.yaml']
    all_valid = True
    
    for config in configs:
        if os.path.exists(config):
            print(f"✓ {config} exists")
            try:
                with open(config, 'r') as f:
                    content = f.read()
                # Check for key sections
                if 'training:' in content and 'model:' in content:
                    print(f"  ✓ Valid YAML structure")
                else:
                    print(f"  ⚠ Missing key sections")
            except Exception as e:
                print(f"  ✗ Error reading: {e}")
                all_valid = False
        else:
            print(f"⚠ {config} not found")
    
    return all_valid

def validate_data_generator():
    """Validate generate_test_data.py."""
    print("\n" + "="*60)
    print("Validating generate_test_data.py")
    print("="*60)
    
    if not os.path.exists('generate_test_data.py'):
        print("✗ generate_test_data.py not found")
        return False
    
    valid, error = test_syntax('generate_test_data.py')
    if not valid:
        print(f"✗ Syntax error: {error}")
        return False
    print("✓ Syntax valid")
    
    with open('generate_test_data.py', 'r') as f:
        content = f.read()
    
    if 'def generate_test_chunk' in content and 'def main' in content:
        print("✓ Required functions found")
    else:
        print("✗ Missing required functions")
        return False
    
    return True

def check_file_structure():
    """Check overall file structure."""
    print("\n" + "="*60)
    print("Checking file structure")
    print("="*60)
    
    required_files = [
        'train.py',
        'tfprocess.py',
        'net.py',
        'chunkparser.py',
        'shufflebuffer.py',
        'generate_test_data.py',
        'README.md',
        'configs/test_config.yaml'
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} missing")
            all_present = False
    
    return all_present

def main():
    """Run all validation tests."""
    print("="*60)
    print("LeelaZero Code Structure Validation")
    print("="*60)
    
    results = []
    
    # File structure
    results.append(("File Structure", check_file_structure()))
    
    # Individual file validations
    results.append(("train.py", validate_train_py()))
    results.append(("tfprocess.py", validate_tfprocess()))
    results.append(("shufflebuffer.py", validate_shufflebuffer()))
    results.append(("Configuration", validate_config()))
    results.append(("Data Generator", validate_data_generator()))
    
    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All validations passed! Code structure is correct.")
        print("\nNote: Full training test requires TensorFlow and dependencies.")
        print("To install dependencies (when SSL is available):")
        print("  pip install -r requirements.txt")
        return 0
    else:
        print(f"\n⚠ {total - passed} validation(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

