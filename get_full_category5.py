#!/usr/bin/env python3
"""
🔍 PDS-X Kategori 5 Tam Liste - Test Edilmemiş Modüller
"""

import os
import glob

def get_full_category5_list():
    print("📄 KATEGORİ 5: Yedekte yok + Ana dizinde var + Import edilmemiş - TAM LİSTE")
    print("="*70)
    
    # Ana dizindeki modüller
    current_modules = set()
    for f in glob.glob('*.py'):
        current_modules.add(f[:-3])
    
    # pdsXuv14eski klasöründeki modüller
    backup_path = r"dislananlar\pdsXuv14eski"
    backup_modules = set()
    if os.path.exists(backup_path):
        for f in os.listdir(backup_path):
            if f.endswith('.py'):
                backup_modules.add(f[:-3])
    
    # pdsXuv14.py'de import edilen modüller
    imported_modules = set()
    if os.path.exists('pdsXuv14.py'):
        with open('pdsXuv14.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.split('\n'):
            line = line.strip()
            if (line.startswith('import ') or line.startswith('from ')) and not line.startswith('#'):
                if line.startswith('import '):
                    module = line.replace('import ', '').split()[0].split('.')[0]
                elif line.startswith('from '):
                    module = line.split(' import ')[0].replace('from ', '').split('.')[0]
                
                if module not in ['sys', 'os', 'subprocess', 'json', 'atexit', 'pathlib', 'logging', 'asyncio', 'site', 'psutil', 're', 'time', 'threading', 'io', 'pickle', 'platform', 'functools', 'collections', 'datetime', 'traceback', 'gc', 'signal', 'copy']:
                    imported_modules.add(module)
    
    # Kategori 5: Yedekte yok + Ana dizinde var + Import edilmemiş
    category5 = (current_modules - backup_modules) - imported_modules
    
    print(f"📊 Toplam Kategori 5 modülü: {len(category5)}")
    print()
    
    # Alfabetik sırayla tam liste
    for i, module in enumerate(sorted(category5), 1):
        print(f"  {i:3d}. ⚪ {module}")
    
    return sorted(category5)

if __name__ == "__main__":
    full_list = get_full_category5_list()
