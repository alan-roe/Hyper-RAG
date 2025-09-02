#!/usr/bin/env python3
"""
Find and fix Chinese characters in both backend and frontend code.
This tool helps identify Chinese text that needs localization.
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def is_chinese_char(char):
    """Check if a character is Chinese (CJK Unified Ideographs)."""
    return (
        '\u4e00' <= char <= '\u9fff' or
        '\u3400' <= char <= '\u4dbf' or
        '\u20000' <= char <= '\u2a6df' or
        '\u2a700' <= char <= '\u2b73f' or
        '\u2b740' <= char <= '\u2b81f' or
        '\u2b820' <= char <= '\u2ceaf' or
        '\uf900' <= char <= '\ufaff'
    )

def find_chinese_in_line(line: str, line_num: int) -> List[Dict]:
    """Find all Chinese text occurrences in a line."""
    results = []
    chinese_matches = []
    current_chinese = []
    start_pos = 0
    
    for i, char in enumerate(line):
        if is_chinese_char(char):
            if not current_chinese:
                start_pos = i
            current_chinese.append(char)
        else:
            if current_chinese:
                chinese_text = ''.join(current_chinese)
                chinese_matches.append({
                    'text': chinese_text,
                    'start': start_pos,
                    'end': i,
                    'context': line.strip()
                })
                current_chinese = []
    
    # Handle Chinese at end of line
    if current_chinese:
        chinese_text = ''.join(current_chinese)
        chinese_matches.append({
            'text': chinese_text,
            'start': start_pos,
            'end': len(line),
            'context': line.strip()
        })
    
    for match in chinese_matches:
        results.append({
            'line': line_num,
            'text': match['text'],
            'context': match['context'],
            'start_pos': match['start'],
            'end_pos': match['end']
        })
    
    return results

def find_chinese_in_file(filepath: Path) -> List[Dict]:
    """Find all Chinese text in a file."""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                chinese_instances = find_chinese_in_line(line, line_num)
                results.extend(chinese_instances)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return results

def scan_directory(root_path: Path, target: str = 'all') -> Dict[str, List]:
    """
    Scan directory for files with Chinese characters.
    
    Args:
        root_path: Root directory to scan
        target: 'backend', 'frontend', or 'all'
    """
    findings = defaultdict(list)
    
    # Define file extensions based on target
    if target in ['backend', 'all']:
        backend_extensions = ['.py', '.json', '.yaml', '.yml']
    else:
        backend_extensions = []
    
    if target in ['frontend', 'all']:
        frontend_extensions = ['.tsx', '.ts', '.jsx', '.js', '.css', '.scss']
    else:
        frontend_extensions = []
    
    extensions = backend_extensions + frontend_extensions
    
    # Add common text files
    if target == 'all':
        extensions.extend(['.md', '.txt'])
    
    for ext in extensions:
        for filepath in root_path.rglob(f'*{ext}'):
            # Skip certain directories
            skip_dirs = [
                '__pycache__', '.git', 'node_modules', '.venv', 'venv',
                'dist', 'build', '.next', 'coverage', '.pytest_cache',
                'tools'  # Skip the tools directory itself
            ]
            
            if any(skip in str(filepath) for skip in skip_dirs):
                continue
            
            # Skip translation files themselves
            if 'translations' in filepath.name.lower() or 'i18n' in filepath.name.lower():
                continue
            
            chinese_instances = find_chinese_in_file(filepath)
            if chinese_instances:
                relative_path = filepath.relative_to(root_path.parent) if root_path.parent.exists() else filepath
                findings[str(relative_path)] = chinese_instances
    
    return findings

def categorize_findings(findings: Dict[str, List]) -> Tuple[List, List, List]:
    """Categorize findings into actionable groups."""
    unique_texts = defaultdict(list)
    
    for filepath, instances in findings.items():
        for instance in instances:
            key = instance['text']
            unique_texts[key].append({
                'file': filepath,
                'line': instance['line'],
                'context': instance['context']
            })
    
    # Sort by frequency
    sorted_texts = sorted(unique_texts.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Categorize
    likely_needs_i18n = []
    already_in_i18n = []
    comments_or_data = []
    
    for text, locations in sorted_texts:
        first_context = locations[0]['context']
        
        # Check if it's likely already using i18n
        i18n_patterns = ['t(', '_t(', 'i18n', 'translate', 'gettext', 'intl', 'formatMessage']
        if any(pattern in first_context for pattern in i18n_patterns):
            already_in_i18n.append((text, locations))
        # Check if it's in a comment or data structure
        elif any(pattern in first_context for pattern in ['#', '//', '/*', '*/', 'json.dumps']):
            comments_or_data.append((text, locations))
        # Check for console.log or print statements (usually debug)
        elif any(pattern in first_context for pattern in ['console.log', 'console.warn', 'console.error', 'print(']):
            comments_or_data.append((text, locations))
        else:
            likely_needs_i18n.append((text, locations))
    
    return likely_needs_i18n, already_in_i18n, comments_or_data

def generate_report(findings: Dict[str, List], target: str, output_file: Path = None):
    """Generate a detailed report of findings."""
    if not findings:
        print("\n✅ No Chinese characters found!")
        return
    
    likely_needs_i18n, already_in_i18n, comments_or_data = categorize_findings(findings)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"CHINESE CHARACTER ANALYSIS REPORT - {target.upper()}")
    report_lines.append("=" * 80)
    report_lines.append(f"\nFound {len(likely_needs_i18n) + len(already_in_i18n) + len(comments_or_data)} unique Chinese text strings in {len(findings)} files")
    report_lines.append("=" * 80)
    
    # Section 1: Needs Localization
    if likely_needs_i18n:
        report_lines.append("\n🔴 NEEDS LOCALIZATION:")
        report_lines.append("-" * 40)
        for text, locations in likely_needs_i18n[:30]:  # Limit to top 30
            report_lines.append(f"\n  Text: '{text}'")
            report_lines.append(f"  Found in {len(locations)} location(s):")
            for loc in locations[:5]:  # Show first 5 locations
                report_lines.append(f"    - {loc['file']}:{loc['line']}")
                if len(loc['context']) > 100:
                    report_lines.append(f"      Context: {loc['context'][:100]}...")
                else:
                    report_lines.append(f"      Context: {loc['context']}")
    
    # Section 2: Possibly Already Using i18n
    if already_in_i18n:
        report_lines.append("\n🟡 POSSIBLY ALREADY USING I18N:")
        report_lines.append("-" * 40)
        for text, locations in already_in_i18n[:15]:
            report_lines.append(f"\n  Text: '{text}'")
            report_lines.append(f"  Found in {len(locations)} location(s):")
            for loc in locations[:3]:
                report_lines.append(f"    - {loc['file']}:{loc['line']}")
    
    # Section 3: Comments or Data
    if comments_or_data:
        report_lines.append("\n🟢 IN COMMENTS/DEBUG (probably OK):")
        report_lines.append("-" * 40)
        for text, locations in comments_or_data[:15]:
            report_lines.append(f"\n  Text: '{text}'")
            report_lines.append(f"  Example: {locations[0]['file']}:{locations[0]['line']}")
    
    # Summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUMMARY:")
    report_lines.append(f"  - Needs localization: {len(likely_needs_i18n)} unique strings")
    report_lines.append(f"  - Possibly already localized: {len(already_in_i18n)} unique strings")
    report_lines.append(f"  - In comments/debug: {len(comments_or_data)} unique strings")
    report_lines.append("=" * 80)
    
    # Print report
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    # Save detailed JSON output if requested
    if output_file:
        output_data = {
            'summary': {
                'total_files': len(findings),
                'needs_localization': len(likely_needs_i18n),
                'possibly_localized': len(already_in_i18n),
                'in_comments': len(comments_or_data)
            },
            'needs_localization': [
                {
                    'text': text,
                    'occurrences': len(locations),
                    'locations': locations[:10]  # Limit locations for readability
                }
                for text, locations in likely_needs_i18n
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Detailed results saved to {output_file}")
    
    return likely_needs_i18n

def generate_fix_suggestions(needs_i18n: List[Tuple[str, List]], target: str):
    """Generate suggestions for fixing the Chinese strings."""
    if not needs_i18n:
        return
    
    print("\n" + "=" * 80)
    print("SUGGESTED FIXES:")
    print("=" * 80)
    
    if target == 'backend':
        print("\n📌 For Backend (Python):")
        print("1. Import the translation function: from translations import t")
        print("2. Replace Chinese strings with t() calls:")
        print("\nExamples:")
        for text, locations in needs_i18n[:5]:
            print(f"\n  # Original: '{text}'")
            # Generate a suggested key from the Chinese text
            key = 'key_' + str(hash(text) % 10000)
            print(f"  # Replace with: t('{key}')")
            print(f"  # Add to translations.py:")
            print(f"    'en-US': {{ '{key}': 'English translation here' }}")
            print(f"    'zh-CN': {{ '{key}': '{text}' }}")
    
    elif target == 'frontend':
        print("\n📌 For Frontend (React/TypeScript):")
        print("1. Import the translation hook: import { useTranslation } from 'react-i18next';")
        print("2. Use the t function in components:")
        print("\nExamples:")
        for text, locations in needs_i18n[:5]:
            print(f"\n  // Original: '{text}'")
            key = 'key_' + str(hash(text) % 10000)
            print(f"  // Replace with: {{t('{key}')}}")
            print(f"  // Add to translation JSON files:")
            print(f"    en-US.json: {{ '{key}': 'English translation here' }}")
            print(f"    zh-CN.json: {{ '{key}': '{text}' }}")

def main():
    parser = argparse.ArgumentParser(description='Find and analyze Chinese characters in code')
    parser.add_argument(
        '--target',
        choices=['backend', 'frontend', 'all'],
        default='all',
        help='Target directory to scan (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Show fix suggestions'
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Custom path to scan (default: auto-detect web-ui directory)'
    )
    
    args = parser.parse_args()
    
    # Determine the path to scan
    if args.path:
        base_path = Path(args.path)
    else:
        # Auto-detect based on script location
        script_path = Path(__file__).resolve()
        repo_root = script_path.parent.parent  # Go up from tools/ to repo root
        
        if args.target == 'backend':
            base_path = repo_root / 'web-ui' / 'backend'
        elif args.target == 'frontend':
            base_path = repo_root / 'web-ui' / 'frontend'
        else:
            base_path = repo_root / 'web-ui'
    
    if not base_path.exists():
        print(f"❌ Error: Path {base_path} does not exist!")
        return 1
    
    print(f"🔍 Scanning {base_path} for Chinese characters...")
    print(f"   Target: {args.target}")
    
    # Scan for Chinese characters
    findings = scan_directory(base_path, args.target)
    
    # Generate report
    output_file = Path(args.output) if args.output else None
    needs_i18n = generate_report(findings, args.target, output_file)
    
    # Show fix suggestions if requested
    if args.fix and needs_i18n:
        generate_fix_suggestions(needs_i18n, args.target)
    
    return 0

if __name__ == "__main__":
    exit(main())