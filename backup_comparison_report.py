#!/usr/bin/env python3
"""
🔍 PDS-X Yedek Modül Karşılaştırma Raporu
"""

import os
import glob
from pathlib import Path

def get_pdsx_imports():
    """pdsXuv14.py'deki import'ları çıkar"""
    imports = []
    if os.path.exists('pdsXuv14.py'):
        with open('pdsXuv14.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.split('\n'):
            line = line.strip()
            if (line.startswith('import ') or line.startswith('from ')) and not line.startswith('#'):
                # Modül adını çıkar
                if line.startswith('import '):
                    module = line.replace('import ', '').split()[0].split('.')[0]
                elif line.startswith('from '):
                    module = line.split(' import ')[0].replace('from ', '').split('.')[0]
                
                # Sistem modülleri hariç
                if module not in ['sys', 'os', 'subprocess', 'json', 'atexit', 'pathlib', 'logging', 'asyncio', 'site', 'psutil', 're', 'time', 'threading', 'io', 'pickle', 'platform', 'functools', 'collections', 'datetime', 'traceback', 'gc', 'signal', 'copy']:
                    imports.append(module)
    
    return list(set(imports))

def analyze_modules():
    print("🔍 PDS-X YEDEK MODÜL KARŞILAŞTIRMA RAPORU")
    print("="*60)
    
    # Ana dizindeki modüller
    current_modules = set()
    for f in glob.glob('*.py'):
        current_modules.add(f[:-3])  # .py uzantısını kaldır
    
    # pdsXuv14eski klasöründeki modüller
    backup_path = r"dislananlar\pdsXuv14eski"
    backup_modules = set()
    if os.path.exists(backup_path):
        for f in os.listdir(backup_path):
            if f.endswith('.py'):
                backup_modules.add(f[:-3])
    
    # pdsXuv14.py'de import edilen modüller
    imported_modules = set(get_pdsx_imports())
    
    print(f"📊 TEMEL İSTATİSTİKLER:")
    print(f"  📁 Ana dizindeki modül sayısı: {len(current_modules)}")
    print(f"  💾 Yedekteki modül sayısı: {len(backup_modules)}")
    print(f"  🔗 pdsXuv14.py'de import edilen: {len(imported_modules)}")
    
    # Kategorize et
    # 1. Yedekte var, ana dizinde var, import edilmiş
    category1 = backup_modules & current_modules & imported_modules
    
    # 2. Yedekte var, ana dizinde var, import edilmemiş
    category2 = backup_modules & current_modules - imported_modules
    
    # 3. Yedekte var, ana dizinde yok
    category3 = backup_modules - current_modules
    
    # 4. Yedekte yok, ana dizinde var, import edilmiş
    category4 = (current_modules - backup_modules) & imported_modules
    
    # 5. Yedekte yok, ana dizinde var, import edilmemiş
    category5 = (current_modules - backup_modules) - imported_modules
    
    # 6. Import edilmiş ama ana dizinde yok
    category6 = imported_modules - current_modules
    
    print(f"\n🎯 KATEGORİ ANALİZİ:")
    print("="*40)
    
    print(f"\n✅ KATEGORİ 1: Yedekte var + Ana dizinde var + Import edilmiş ({len(category1)}):")
    for mod in sorted(category1):
        print(f"  🟢 {mod}")
    
    print(f"\n⚠️ KATEGORİ 2: Yedekte var + Ana dizinde var + Import edilmemiş ({len(category2)}):")
    for mod in sorted(category2):
        print(f"  🟡 {mod}")
    
    print(f"\n❌ KATEGORİ 3: Yedekte var + Ana dizinde yok ({len(category3)}):")
    for mod in sorted(category3):
        print(f"  🔴 {mod}")
    
    print(f"\n🆕 KATEGORİ 4: Yedekte yok + Ana dizinde var + Import edilmiş ({len(category4)}):")
    for mod in sorted(category4):
        print(f"  🟦 {mod}")
    
    print(f"\n📄 KATEGORİ 5: Yedekte yok + Ana dizinde var + Import edilmemiş ({len(category5)}):")
    for mod in sorted(category5)[:15]:
        print(f"  ⚪ {mod}")
    if len(category5) > 15:
        print(f"  ... ve {len(category5)-15} modül daha")
    
    print(f"\n⚠️ KATEGORİ 6: Import edilmiş ama dosya yok ({len(category6)}):")
    for mod in sorted(category6):
        print(f"  🟠 {mod}")
    
    # Öneriler
    print(f"\n\n💡 ÖNERİLER:")
    print("="*30)
    
    if category2:
        print(f"🟡 KATEGORİ 2 modülleri pdsXuv14.py'ye eklenebilir:")
        for mod in sorted(category2)[:5]:
            print(f"   • {mod}")
    
    if category3:
        print(f"🔴 KATEGORİ 3 modülleri yedekten geri yüklenebilir:")
        for mod in sorted(category3)[:5]:
            print(f"   • {mod}")
    
    if category6:
        print(f"🟠 KATEGORİ 6 import'ları temizlenebilir:")
        for mod in sorted(category6):
            print(f"   • {mod}")
    
    print(f"\n📋 DURUM ÖZET:")
    print(f"  ✅ Başarılı entegrasyon: {len(category1)} modül")
    print(f"  ⚠️ Eksik entegrasyon: {len(category2)} modül")
    print(f"  ❌ Kayıp modül: {len(category3)} modül")
    print(f"  🆕 Yeni modül: {len(category4)} modül")
    print(f"  📄 Test edilmemiş: {len(category5)} modül")
    print(f"  🟠 Hatalı import: {len(category6)} modül")
    
    return {
        'category1': category1,
        'category2': category2,
        'category3': category3,
        'category4': category4,
        'category5': category5,
        'category6': category6,
        'backup_modules': backup_modules,
        'current_modules': current_modules,
        'imported_modules': imported_modules
    }

if __name__ == "__main__":
    results = analyze_modules()
