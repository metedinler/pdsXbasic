#!/usr/bin/env python3
"""
PDS-X Modül Analizi ve Ekosistem Raporu
"""

import os
import glob
from pathlib import Path

def analyze_pdsx_modules():
    print("🔍 PDS-X MODÜL ANALİZİ")
    print("="*60)
    
    # Python dosyalarını bul
    py_files = glob.glob('*.py')
    print(f"📊 Toplam Python dosyası: {len(py_files)}")
    
    # Ana modülleri kategorize et
    core_modules = []
    libx_modules = []
    utility_modules = []
    manager_modules = []
    repl_modules = []
    bytecode_modules = []
    test_modules = []
    other_modules = []
    
    for file in sorted(py_files):
        name = file.replace('.py', '')
        if name.startswith('libx'):
            libx_modules.append(name)
        elif 'manager' in name.lower():
            manager_modules.append(name)
        elif name in ['core2_6', 'core_system']:
            core_modules.append(name)
        elif 'bytecode' in name.lower():
            bytecode_modules.append(name)
        elif 'repl' in name.lower() or name in ['reply_extension', 'program_manager']:
            repl_modules.append(name)
        elif name in ['auto_importer_lite', 'auto_importer_heavy', 'memory_manager', 'data_structures']:
            utility_modules.append(name)
        elif 'test' in name.lower() or name.startswith('demo'):
            test_modules.append(name)
        else:
            other_modules.append(name)
    
    print(f"\n🔥 ÇEKİRDEK MODÜLLER ({len(core_modules)}):")
    for mod in core_modules:
        print(f"  ⚡ {mod}")
    
    print(f"\n🧠 BYTECODE MODÜLLER ({len(bytecode_modules)}):")
    for mod in bytecode_modules:
        print(f"  💾 {mod}")
    
    print(f"\n📚 LIBX MODÜLLER ({len(libx_modules)}):")
    for mod in libx_modules:
        print(f"  🔧 {mod}")
    
    print(f"\n⚙️ YÖNETİCİ MODÜLLER ({len(manager_modules)}):")
    for mod in manager_modules:
        print(f"  🎛️ {mod}")
    
    print(f"\n💬 REPL MODÜLLER ({len(repl_modules)}):")
    for mod in repl_modules:
        print(f"  🖥️ {mod}")
    
    print(f"\n🛠️ YARDIMCI MODÜLLER ({len(utility_modules)}):")
    for mod in utility_modules:
        print(f"  🔨 {mod}")
    
    print(f"\n🧪 TEST/DEMO MODÜLLER ({len(test_modules)}):")
    for mod in test_modules[:8]:
        print(f"  🔬 {mod}")
    if len(test_modules) > 8:
        print(f"  ... ve {len(test_modules)-8} test modülü daha")
    
    print(f"\n📁 DİĞER MODÜLLER ({len(other_modules)}):")
    for mod in other_modules[:20]:
        print(f"   📄 {mod}")
    if len(other_modules) > 20:
        print(f"  ... ve {len(other_modules)-20} modül daha")
    
    # pdsXuv14.py'deki importları kontrol et
    print(f"\n🎯 PDSX ENTEGRASYON ANALİZİ")
    print("="*40)
    
    if os.path.exists('pdsXuv14.py'):
        with open('pdsXuv14.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        imported_modules = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                if not line.strip().startswith('#'):
                    imported_modules.append(line.strip())
        
        print(f"📈 pdsXuv14.py'de import edilen modül satırı: {len(imported_modules)}")
        print("\n🔗 İMPORT EDİLEN MODÜLLER:")
        for imp in imported_modules[:15]:
            print(f"  ➤ {imp}")
        if len(imported_modules) > 15:
            print(f"  ... ve {len(imported_modules)-15} import daha")
    
    return {
        'core': core_modules,
        'libx': libx_modules,
        'utility': utility_modules,
        'manager': manager_modules,
        'repl': repl_modules,
        'bytecode': bytecode_modules,
        'test': test_modules,
        'other': other_modules
    }

def create_git_integration_plan():
    print(f"\n\n🚀 GIT ENTEGRASYON PLANI")
    print("="*50)
    
    print("1️⃣ Git repository başlatma:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial PDS-X ecosystem commit'")
    
    print("\n2️⃣ Çalışan modülleri test etme:")
    print("   python pdsXuv14.py")
    print("   python pdsx_repl.py")
    
    print("\n3️⃣ Modül bağımlılıklarını kontrol etme:")
    print("   python -c 'import pdsXuv14; print(\"OK\")'")
    
    print("\n4️⃣ GitHub'a push etme:")
    print("   git remote add origin <repo_url>")
    print("   git push -u origin main")

if __name__ == "__main__":
    modules = analyze_pdsx_modules()
    create_git_integration_plan()
    
    print(f"\n\n✅ ANALİZ TAMAMLANDI")
    print(f"📊 Toplam modül kategorisi: {len(modules)}")
    print(f"🎯 Sonraki adım: Git repository kurma")
