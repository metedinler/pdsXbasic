#!/usr/bin/env python3
# Dosya analizi scripti

import os

def analyze_files():
    files_to_check = ['auto_importer.py', 'core2_6.py', 'pdsXuv14.py', 'pdsx_repl.py', 'pdsx_simple_repl.py']
    
    for filename in files_to_check:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                import_count = content.count('import')
                from_import_count = content.count('from ')
                
            print(f"{filename}:")
            print(f"  Size: {os.path.getsize(filename)//1024}KB")
            print(f"  Lines: {len(lines)}")
            print(f"  Import statements: {import_count}")
            print(f"  From imports: {from_import_count}")
            print(f"  Total imports: {import_count + from_import_count}")
            print()
        else:
            print(f"{filename}: NOT FOUND")

if __name__ == "__main__":
    analyze_files()
