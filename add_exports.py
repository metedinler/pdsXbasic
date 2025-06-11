"""
add_exports.py - PDS-X BASIC Modül Export Ekleyici
Version: 1.0.0
Date: June 9, 2025
"""

import os
import re
import ast
from typing import Dict, List, Set

def find_classes_and_functions(content: str) -> tuple[Set[str], Set[str]]:
    """Dosya içeriğindeki sınıf ve fonksiyon isimlerini bulur."""
    tree = ast.parse(content)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    return classes, functions

def add_exports_to_file(file_path: str) -> None:
    """Dosyaya __pdsX_exports__ tanımı ekler."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Eğer zaten exports tanımı varsa, atla
    if "__pdsX_exports__" in content:
        print(f"[ATLA] {file_path} zaten exports tanımı içeriyor")
        return
    
    # Sınıf ve fonksiyonları bul
    try:
        classes, functions = find_classes_and_functions(content)
    except Exception as e:
        print(f"[HATA] {file_path}: {str(e)}")
        return
    
    # Modül adını al
    module_name = os.path.basename(file_path).replace('.py', '')
    
    # Export tanımını oluştur
    exports_def = f'''
if __name__ == "__main__":
    print("{module_name}.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")

# Dinamik yükleme için ihraç edilecek öğeler
__pdsX_exports__ = {{
    "classes": {{
        {", ".join(f'"{cls}": {cls}' for cls in classes)}
    }},
    "functions": {{
        {", ".join(f'"{func}": {func}' for func in functions if not func.startswith("_"))}
    }},
    "variables": {{
        "version": "1.0.0",
        "dependencies": []  # Her modül kendi bağımlılıklarını eklemelidir
    }}
}}
'''
    
    # Tanımı dosyanın sonuna ekle
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(exports_def)
    
    print(f"[BAŞARILI] {file_path} için exports tanımı eklendi")

def process_all_modules():
    """Tüm .py dosyalarına exports tanımı ekler."""
    modules = [
        "bytecode_engine(core2duo).py",
        "functional.py",
        "graph.py",
        "libx_ml.py",
        "libx_nlp.py",
        "libx_network.py",
        "pipe.py",
        "tree.py"
    ]
    
    for module in modules:
        if os.path.exists(module):
            add_exports_to_file(module)
        else:
            print(f"[HATA] {module} bulunamadı")

if __name__ == "__main__":
    process_all_modules()
