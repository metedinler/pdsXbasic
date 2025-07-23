#!/usr/bin/env python3
"""
PDS-X Ecosystem Integration Manager
Tüm çalışan modülleri pdsXuv14.py'ye entegre eder ve git repository'sinde saklar.
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path

def detect_key_modules():
    """Ana modülleri tespit et"""
    
    print("🔍 PDS-X ANA MODÜL TESPİTİ")
    print("=" * 40)
    
    # Kritik modüller
    critical_modules = {
        'core': ['core2_6', 'core_system', 'bytecode_compiler', 'bytecode_manager'],
        'auto_importers': ['auto_importer_lite', 'auto_importer_heavy'],
        'managers': ['memory_manager', 'module_manager', 'program_manager'],
        'repls': ['pdsx_repl', 'reply_extension'],
        'libx': ['libxcore', 'libx_data', 'libx_gui', 'libx_network'],
        'utilities': ['data_structures', 'command_executor']
    }
    
    existing_modules = {}
    
    for category, modules in critical_modules.items():
        existing_modules[category] = []
        for module in modules:
            module_file = f"{module}.py"
            if os.path.exists(module_file):
                existing_modules[category].append(module)
                print(f"  ✅ {module} - Mevcut")
            else:
                print(f"  ❌ {module} - Eksik")
    
    return existing_modules

def create_integration_plan():
    """Entegrasyon planı oluştur"""
    
    plan = {
        "phase_1": {
            "description": "Çekirdek Sistem",
            "modules": ["auto_importer_lite", "auto_importer_heavy"],
            "priority": "CRITICAL"
        },
        "phase_2": {
            "description": "REPL Sistemi", 
            "modules": ["pdsx_repl", "reply_extension"],
            "priority": "HIGH"
        },
        "phase_3": {
            "description": "Core Modüller",
            "modules": ["core2_6", "data_structures", "memory_manager"],
            "priority": "HIGH"
        },
        "phase_4": {
            "description": "LibX Kütüphaneleri",
            "modules": ["libxcore", "libx_data", "libx_gui"],
            "priority": "MEDIUM"
        },
        "phase_5": {
            "description": "Yönetici Modüller",
            "modules": ["module_manager", "program_manager", "command_executor"],
            "priority": "MEDIUM"
        }
    }
    
    return plan

def initialize_git_repo():
    """Git repository'si başlat"""
    
    print("\n🔧 GIT REPOSITORY BAŞLATILIYOR")
    print("=" * 35)
    
    try:
        # Git init
        result = subprocess.run(['git', 'init'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("  ✅ Git repository başlatıldı")
        else:
            print(f"  ⚠️ Git zaten var: {result.stderr}")
        
        # Gitignore oluştur
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PDS-X Specific
.pdsx_cache/
.pdsx_isolated_env/
disabled_modules/
sealed_versions/
temp/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# Test files
*test*.py
*_test.py
test_*/
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        
        print("  ✅ .gitignore oluşturuldu")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Git hatası: {e}")
        return False

def backup_current_state():
    """Mevcut durumu yedekle"""
    
    print("\n💾 MEVCUT DURUM YEDEKLENİYOR")
    print("=" * 30)
    
    backup_files = [
        'pdsXuv14.py',
        'auto_importer_lite.py', 
        'auto_importer_heavy.py',
        'pdsx_repl.py'
    ]
    
    # Backup dizini oluştur
    os.makedirs('backup_before_integration', exist_ok=True)
    
    for file in backup_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as src:
                    content = src.read()
                
                backup_path = f"backup_before_integration/{file}"
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                
                print(f"  ✅ {file} yedeklendi")
            except Exception as e:
                print(f"  ❌ {file} yedeklenemedi: {e}")

def commit_initial_state():
    """İlk commit'i gerçekleştir"""
    
    print("\n🚀 İLK COMMIT GERÇEKLEŞTİRİLİYOR")
    print("=" * 30)
    
    try:
        # Add files
        subprocess.run(['git', 'add', '.'], cwd='.')
        
        # Initial commit
        commit_msg = "Initial PDS-X ecosystem commit - Base modules ready for integration"
        result = subprocess.run(['git', 'commit', '-m', commit_msg], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("  ✅ İlk commit başarılı")
            print(f"  📝 Commit mesajı: {commit_msg}")
            return True
        else:
            print(f"  ❌ Commit hatası: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Git commit hatası: {e}")
        return False

def prepare_integration_script():
    """Entegrasyon script'ini hazırla"""
    
    integration_script = '''#!/usr/bin/env python3
"""
PDS-X Module Integration Script
Bu script tüm modülleri pdsXuv14.py'ye adım adım entegre eder.
"""

def phase1_core_imports():
    """Faz 1: Çekirdek import sistemi"""
    
    core_imports = """
# PDS-X Core Imports - Phase 1
try:
    from auto_importer_lite import AutoImporterLite
    from auto_importer_heavy import AutoImporterHeavy
    print("✅ AutoImporter modülleri yüklendi")
except ImportError as e:
    print(f"❌ AutoImporter hatası: {e}")
    AutoImporterLite = None
    AutoImporterHeavy = None

# Environment Detection
def detect_environment():
    \"""Sistem durumunu tespit et\"""
    import os
    import site
    
    try:
        site_packages = site.getsitepackages()[0]
        package_count = len([d for d in os.listdir(site_packages) 
                           if os.path.isdir(os.path.join(site_packages, d))])
        
        # 50+ paket varsa lite mode
        if package_count > 50:
            return "lite"
        else:
            return "heavy"
    except:
        return "heavy"

# Dynamic AutoImporter Selection
ENV_MODE = detect_environment()
if ENV_MODE == "lite" and AutoImporterLite:
    auto_importer = AutoImporterLite()
    print("🚀 AutoImporter Lite aktif")
elif AutoImporterHeavy:
    auto_importer = AutoImporterHeavy()
    print("🚀 AutoImporter Heavy aktif")
else:
    auto_importer = None
    print("⚠️ AutoImporter devre dışı")
"""
    
    return core_imports

if __name__ == "__main__":
    print("📦 PDS-X Integration Script Hazırlandı")
    print("Bu script pdsXuv14.py güncellemesi için kullanılacak")
'''
    
    with open('integration_script.py', 'w', encoding='utf-8') as f:
        f.write(integration_script)
    
    print("  ✅ integration_script.py oluşturuldu")

def main():
    """Ana entegrasyon süreci"""
    
    print("🔥 PDS-X ECOSYSTEM INTEGRATION MANAGER")
    print("=" * 50)
    print("Ekosistemi canlandırmak için tüm modülleri entegre ediyor...")
    
    # 1. Modül tespiti
    existing_modules = detect_key_modules()
    
    # 2. Entegrasyon planı
    plan = create_integration_plan()
    
    # 3. Git repository başlat
    git_success = initialize_git_repo()
    
    # 4. Mevcut durumu yedekle
    backup_current_state()
    
    # 5. Entegrasyon script'i hazırla
    prepare_integration_script()
    
    # 6. İlk commit
    if git_success:
        commit_success = commit_initial_state()
    
    # 7. Özet rapor
    print("\n📊 ENTEGRASYON HAZIRLIK RAPORU")
    print("=" * 35)
    
    total_available = sum(len(modules) for modules in existing_modules.values())
    print(f"✅ Mevcut Modül Sayısı: {total_available}")
    print(f"🔧 Git Repository: {'Hazır' if git_success else 'Hatalı'}")
    print(f"💾 Backup: Tamamlandı")
    print(f"📋 Entegrasyon Script: Hazır")
    
    print("\n🚀 SONRAKİ ADIMLAR:")
    print("1. integration_script.py ile modülleri pdsXuv14.py'ye ekle")
    print("2. Her faz sonrası test et")
    print("3. git add . && git commit -m 'Phase X integration'")
    print("4. Tam entegrasyon sonrası final test")
    
    # Plan dosyasını kaydet
    with open('integration_plan.json', 'w', encoding='utf-8') as f:
        json.dump({
            'available_modules': existing_modules,
            'integration_phases': plan,
            'status': 'ready_for_integration'
        }, f, ensure_ascii=False, indent=2)
    
    print("\n💾 integration_plan.json kaydedildi")
    print("✨ PDS-X Ecosystem Integration Manager tamamlandı!")

if __name__ == "__main__":
    main()
