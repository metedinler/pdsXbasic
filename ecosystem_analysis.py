#!/usr/bin/env python3
"""
🚀 PDS-X Ekosistem Canlandırma ve Git Entegrasyon Raporu
"""

import os
import glob
from pathlib import Path

def main():
    print("🔍 PDS-X MODÜL ANALİZİ VE EKOSİSTEM CANLANDIRMA")
    print("="*60)
    
    # Python dosyalarını bul (ana dizinde)
    py_files = [f for f in glob.glob('*.py') if not f.startswith('disabled_')]
    
    print(f"📊 Ana dizindeki toplam Python dosyası: {len(py_files)}")
    
    # Modülleri kategorize et
    categories = {
        'core': ['pdsXuv14', 'core2_6', 'core_system'],
        'bytecode': ['bytecode_compiler', 'bytecode_manager', 'bytecode_engine_core2duo', 'bytecode_engine_core2duo_2', 'bytecode_engine_wd658160_', 'bytecode_engine(core2duo)'],
        'libx': ['libxcore', 'libx_jit', 'libx_data', 'libx_logic', 'libx_gui', 'libx_concurrency', 'libx_nlp', 'libx_network', 'libx_ml'],
        'managers': ['module_manager', 'module_manager_simple', 'memory_manager', 'base_module_manager', 'offline_manager'],
        'auto_import': ['auto_importer_heavy', 'auto_importer_lite', 'autoinstaller', 'minimal_auto_importer'],
        'repl': ['pdsx_repl', 'reply_extension', 'program_manager', 'pdsx_clean_launcher', 'fast_launcher'],
        'utils': ['data_structures', 'f11_backtrace_logger', 'f12_timer_manager', 'save', 'lowlevel'],
        'db': ['lib_db', 'database_sql_isam', 'sqlite'],
        'graphics': ['pipe3', 'tree3', 'graph2', 'functional2'],
        'system': ['multithreading_process', 'parallel_processor'],
        'commands': ['command_executor', 'command_executor2', 'command_executorx1', 'command_executorx2', 'command_executorx2z', 'command_executorx2z1'],
        'oop': ['oop_and_class2', 'clazz'],
        'gui': ['pipe_monitor_gui'],
        'reports': ['export_report_doc'],
        'tests': ['minimal_repl_test', 'direct_repl_test', 'demo_multiline', 'heavy_module_test'],
        'analysis': ['module_analyzer', 'analyze_imports', 'format_analizi', 'format_support_report', 'final_optimization_report', 'optimization_success_report', 'pdsx_complete_module_analysis', 'module_analysis']
    }
    
    # Dosyaları kategorilere göre sınıflandır
    categorized = {}
    uncategorized = []
    
    for category, modules in categories.items():
        categorized[category] = []
        for module in modules:
            if f"{module}.py" in py_files:
                categorized[category].append(module)
    
    # Kategorize edilmeyenleri bul
    all_categorized = set()
    for cat_modules in categorized.values():
        all_categorized.update(cat_modules)
    
    for py_file in py_files:
        module_name = py_file[:-3]  # .py uzantısını kaldır
        if module_name not in all_categorized:
            uncategorized.append(module_name)
    
    # Kategorileri göster
    for category, modules in categorized.items():
        if modules:
            print(f"\n🔧 {category.upper()} MODÜLLER ({len(modules)}):")
            for module in modules:
                print(f"  ✅ {module}")
    
    if uncategorized:
        print(f"\n❓ KATEGORİZE EDİLMEYEN MODÜLLER ({len(uncategorized)}):")
        for module in uncategorized[:15]:
            print(f"  📄 {module}")
        if len(uncategorized) > 15:
            print(f"  ... ve {len(uncategorized)-15} modül daha")
    
    # pdsXuv14.py analizi
    print(f"\n🎯 PDSX ENTEGRASİYON ANALİZİ")
    print("="*40)
    
    if os.path.exists('pdsXuv14.py'):
        with open('pdsXuv14.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Import satırlarını bul
        import_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if (line.startswith('import ') or line.startswith('from ')) and not line.startswith('#'):
                import_lines.append(line)
        
        print(f"📈 pdsXuv14.py'de import satırı sayısı: {len(import_lines)}")
        
        # İlk 10 import'u göster
        print("\n🔗 MEVCUT İMPORT'LAR (İlk 10):")
        for imp in import_lines[:10]:
            print(f"  ➤ {imp}")
        if len(import_lines) > 10:
            print(f"  ... ve {len(import_lines)-10} import daha")
    
    # Git entegrasyon planı
    print(f"\n\n🚀 GIT ENTEGRASİYON PLANI")
    print("="*50)
    
    print("1️⃣ Git repository başlatma:")
    print("   git init")
    print("   git config user.name 'PDS-X Developer'")
    print("   git config user.email 'pdsx@developer.local'")
    
    print("\n2️⃣ .gitignore dosyası oluşturma:")
    gitignore_content = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/

# PDS-X Cache
.pdsx_cache/
.pdsx_isolated_env/
last_args.json
learned_dependencies.json
pdsx_dependencies.json

# Disabled modules
disabled_modules/
dislananlar/

# Logs and temp files
*.log
*.tmp
optimization_results.json
integration_test_report.json
FINAL_OPTIMIZATION_RESULTS.json
    """
    print("   .gitignore dosyası içeriği hazırlandı")
    
    print("\n3️⃣ İlk commit için dosya seçimi:")
    essential_files = [
        'pdsXuv14.py',
        'auto_importer_heavy.py',
        'auto_importer_lite.py',
        'core2_6.py',
        'reply_extension.py',
        'program_manager.py',
        'pdsx_repl.py',
        'module_manager_simple.py'
    ]
    
    print("   ✅ TEMEL DOSYALAR:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"     📄 {file}")
        else:
            print(f"     ❌ {file} (bulunamadı)")
    
    print("\n4️⃣ Commit stratejisi:")
    print("   git add pdsXuv14.py auto_importer_*.py core2_6.py reply_extension.py")
    print("   git commit -m '🚀 Initial PDS-X ecosystem commit - Core modules'")
    print("   git add libx*.py module_manager*.py data_structures.py")
    print("   git commit -m '📚 Add LibX modules and managers'")
    print("   git add *repl*.py program_manager.py")
    print("   git commit -m '💬 Add REPL and interaction modules'")
    
    print("\n5️⃣ Remote repository ekleme:")
    print("   git remote add origin <repository_url>")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    # Çalışabilirlik testi önerileri
    print(f"\n\n🧪 ÇALIŞAN MODÜL TESTİ")
    print("="*40)
    
    test_commands = [
        "python -c 'import pdsXuv14; print(\"✅ pdsXuv14 import OK\")'",
        "python -c 'import auto_importer_lite; print(\"✅ AutoImporter Lite OK\")'",
        "python -c 'import core2_6; print(\"✅ Core2_6 OK\")'",
        "python -c 'import reply_extension; print(\"✅ Reply Extension OK\")'",
        "python pdsx_repl.py --help 2>/dev/null || echo '⚠️ REPL test needs manual check'"
    ]
    
    print("Aşağıdaki komutları çalıştırarak modülleri test edin:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"{i}. {cmd}")
    
    print(f"\n\n✅ ANALİZ TAMAMLANDI")
    print(f"📊 Toplam kategorize edilen modül: {sum(len(mods) for mods in categorized.values())}")
    print(f"🎯 Sonraki adım: Git repository kurma ve test etme")
    print(f"🔗 Ekosistem durumu: Hazır")

if __name__ == "__main__":
    main()
