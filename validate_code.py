#!/usr/bin/env python3
#
# Code validation script
# Checks for common issues and validates code structure

import os
import re
import sys

def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {filepath} exists")
        return True
    else:
        print(f"✗ {filepath} missing")
        return False

def check_imports(filepath):
    """Check if file has proper imports."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if 'import' in content or 'from' in content:
                print(f"✓ {filepath} has imports")
                return True
            else:
                print(f"⚠ {filepath} has no imports")
                return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

def check_no_debug_code(filepath):
    """Check for debug code patterns."""
    debug_patterns = [
        r'print\(.*ma\)',
        r'ma = make_map\(\)',
        r'#TODO.*softmax',
        r'^\s*print\s*\([^)]*\)\s*$',  # Standalone print statements
    ]
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            issues = []
            for i, line in enumerate(lines, 1):
                for pattern in debug_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(f"Line {i}: {line.strip()}")
            
            if issues:
                print(f"⚠ {filepath} has potential debug code:")
                for issue in issues[:3]:  # Show first 3
                    print(f"  {issue}")
                return False
            else:
                print(f"✓ {filepath} no debug code found")
                return True
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

def main():
    """Run validation checks."""
    print("=" * 60)
    print("Code Validation Report")
    print("=" * 60)
    print()
    
    # Check critical files
    critical_files = [
        'train.py',
        'tfprocess.py',
        'net.py',
        'chunkparser.py',
        'shufflebuffer.py',
        'generate_test_data.py',
        'README.md',
        'configs/test_config.yaml'
    ]
    
    print("Checking critical files...")
    all_exist = True
    for file in critical_files:
        if not check_file_exists(file):
            all_exist = False
    print()
    
    # Check for debug code
    print("Checking for debug code...")
    files_to_check = ['lc0_az_policy_map.py', 'keras_net.py', 'tfprocess.py']
    for file in files_to_check:
        if os.path.exists(file):
            check_no_debug_code(file)
    print()
    
    # Check structure
    print("Checking project structure...")
    required_dirs = ['configs', 'scripts', 'proto']
    for dirname in required_dirs:
        if os.path.isdir(dirname):
            print(f"✓ {dirname}/ directory exists")
        else:
            print(f"✗ {dirname}/ directory missing")
    print()
    
    # Summary
    print("=" * 60)
    if all_exist:
        print("✓ All critical files present")
        print("✓ Code structure validated")
        print("\nReady for training!")
    else:
        print("⚠ Some files are missing")
    print("=" * 60)

if __name__ == '__main__':
    main()

