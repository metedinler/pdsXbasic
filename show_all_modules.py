#!/usr/bin/env python3
"""
🔍 PDS-X Ana Dizin Modül Tam Listesi
"""

import os
import glob

def show_all_modules():
    print("📋 PDS-X ANA DİZİN - TÜM MODÜLLER")
    print("="*60)
    
    # Ana dizindeki tüm Python dosyalarını al
    py_files = sorted([f for f in glob.glob('*.py') if not f.startswith('disabled_')])
    
    print(f"📊 Toplam Python dosyası: {len(py_files)}")
    print(f"🔗 Ana dizin: {os.getcwd()}")
    print()
    
    # Kategorize edilmiş modüller
    categories = {
        'CORE': ['pdsXuv14', 'core2_6', 'core_system'],
        'BYTECODE': ['bytecode_compiler', 'bytecode_manager', 'bytecode_engine_core2duo_2', 'bytecode_engine_wd658160_', 'bytecode_engine(core2duo)'],
        'LIBX': ['libxcore', 'libx_jit', 'libx_data', 'libx_logic', 'libx_gui', 'libx_concurrency', 'libx_nlp', 'libx_network', 'libx_ml'],
        'MANAGERS': ['module_manager', 'module_manager_simple', 'memory_manager', 'base_module_manager', 'offline_manager'],
        'AUTO_IMPORT': ['auto_importer_heavy', 'auto_importer_lite', 'autoinstaller', 'minimal_auto_importer'],
        'REPL': ['pdsx_repl', 'reply_extension', 'program_manager', 'pdsx_clean_launcher', 'fast_launcher'],
        'UTILS': ['data_structures', 'f11_backtrace_logger', 'f12_timer_manager', 'save', 'lowlevel'],
        'DB': ['lib_db', 'database_sql_isam', 'sqlite'],
        'GRAPHICS': ['pipe3', 'tree3', 'graph2', 'functional2'],
        'SYSTEM': ['multithreading_process', 'parallel_processor'],
        'COMMANDS': ['command_executor', 'command_executor2', 'command_executorx1', 'command_executorx2', 'command_executorx2z', 'command_executorx2z1'],
        'OOP': ['oop_and_class2', 'clazz'],
        'GUI': ['pipe_monitor_gui'],
        'REPORTS': ['export_report_doc'],
        'TESTS': ['minimal_repl_test', 'direct_repl_test', 'demo_multiline', 'heavy_module_test'],
        'ANALYSIS': ['module_analyzer', 'analyze_imports', 'format_analizi', 'format_support_report', 'final_optimization_report', 'optimization_success_report', 'pdsx_complete_module_analysis', 'module_analysis']
    }
    
    # Kategorize edilmiş modülleri topla
    categorized_modules = set()
    for cat_modules in categories.values():
        categorized_modules.update(cat_modules)
    
    # Kategorize edilmemiş modülleri bul
    uncategorized = []
    for py_file in py_files:
        module_name = py_file[:-3]  # .py uzantısını kaldır
        if module_name not in categorized_modules:
            uncategorized.append(module_name)
    
    # Kategorize edilmiş modülleri göster
    total_categorized = 0
    for category, modules in categories.items():
        existing_modules = [m for m in modules if f"{m}.py" in py_files]
        if existing_modules:
            total_categorized += len(existing_modules)
            print(f"🔧 {category} ({len(existing_modules)}):")
            for module in existing_modules:
                print(f"  ✅ {module}")
            print()
    
    # Kategorize edilmemiş modülleri göster
    print(f"❓ KATEGORİZE EDİLMEYEN MODÜLLER ({len(uncategorized)}):")
    for i, module in enumerate(uncategorized, 1):
        print(f"  {i:2d}. 📄 {module}")
    
    print(f"\n📊 ÖZET:")
    print(f"  ✅ Kategorize edilmiş: {total_categorized}")
    print(f"  ❓ Kategorize edilmemiş: {len(uncategorized)}")
    print(f"  📁 Toplam: {len(py_files)}")
    
    return py_files, uncategorized

if __name__ == "__main__":
    all_files, uncategorized = show_all_modules()
