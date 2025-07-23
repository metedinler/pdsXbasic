#!/usr/bin/env python3
"""
PDS-X v14u Çalışma Ortamı Düzenleme Scripti
Kullanılacak modüller ana dizinde kalacak, geri kalanlar yigin/ klasörüne taşınacak
"""

import os
import shutil
import glob

# Ana dizinde kalacak aktif modüller
ACTIVE_MODULES = {
    # ANA SİSTEM
    "pdsXuv14.py",
    
    # COMMAND EXECUTORS  
    "command_executor.py", "command_executorx1.py", "command_executorx2z1.py",
    
    # LIBX EKOSİSTEMİ
    "libxcore.py", "libx_concurrency.py", "libx_data.py", "libx_gui.py", 
    "libx_jit.py", "libx_logic.py", "libx_ml.py", "libx_network.py", "libx_nlp.py",
    
    # DATABASE
    "lib_db.py", "sqlite.py", "database_sql_isam.py",
    
    # PIPE & BUS
    "pipe3.py", "bus3.py", "pipe_monitor_gui.py",
    
    # BYTECODE
    "bytecode_compiler.py", "bytecode_manager.py", "bytecode_engine_core2duo_2.py",
    
    # DATA & ALGORITHMS
    "data_structures.py", "tree3.py", "graph2.py", "functional2.py",
    
    # SİSTEM
    "core2-6.py", "memory_manager.py", "lowlevel.py", "multithreading_process.py", 
    "offline_manager.py", "parallel_processor.py",
    
    # MODÜL YÖNETİMİ
    "module_manager.py", "module_analyzer.py", "module_validator.py", "base_module_manager.py",
    
    # SAVE/LOAD
    "save.py", "save_load_system2.py", "export_report_doc.py",
    
    # REPL
    "reply_extension.py",
    
    # EVENT & TIMER
    "event3.py", "f11_backtrace_logger.py", "f12_timer_manager.py",
    
    # SCIENTIFIC
    "scientific_utils.py", "quantum_pdsX.py",
    
    # OOP
    "oop_and_class2.py", "clazz.py",
    
    # EXCEPTION
    "exception_manager3.py",
    
    # AUTO SYSTEMS
    "auto_importer.py", "autoinstaller.py", "add_exports.py",
    
    # UTILITIES
    "program_manager.py", "ai.py"
}

# İNCELENECEKLER (geçici ana dizinde kalacak)
INVESTIGATION_MODULES = {
    "command_executor2.py", "command_executorx2.py", "command_executorx2z.py",
    "bytecode_engine(core2duo).py", "bytecode_engine_wd658160_.py",
    "eventx.py", "eventxkullanim.py", "quantum.py",
    "pdsx_exception.py", "pdsx_exception2.py", "pdsx_pipe.py"
}

def main():
    # Tüm Python dosyalarını bul
    all_py_files = glob.glob('*.py')
    moved_count = 0
    kept_count = 0
    
    print(f"📁 Toplam {len(all_py_files)} Python dosyası bulundu")
    print(f"📋 {len(ACTIVE_MODULES)} aktif modül ana dizinde kalacak")
    print(f"🔍 {len(INVESTIGATION_MODULES)} modül inceleme için geçici kalacak")
    
    for file in all_py_files:
        if file == "cleanup_script.py":  # Bu scripti kendisini taşıma
            continue
            
        if file not in ACTIVE_MODULES and file not in INVESTIGATION_MODULES:
            if os.path.exists(file):
                try:
                    shutil.move(file, f'yigin/{file}')
                    print(f"📦 Taşındı: {file}")
                    moved_count += 1
                except Exception as e:
                    print(f"❌ Hata: {file} - {e}")
        else:
            print(f"✅ Kalıyor: {file}")
            kept_count += 1
    
    print(f"\n📊 ÖZET:")
    print(f"✅ Ana dizinde kalan: {kept_count} dosya")
    print(f"📦 Yığına taşınan: {moved_count} dosya")
    print(f"🔍 İnceleme bekleyen: {len(INVESTIGATION_MODULES)} dosya")
    
    # Ana dizinde kalan dosyaları listele
    remaining_files = glob.glob('*.py')
    print(f"\n📋 ANA DİZİNDE KALAN DOSYALAR:")
    for file in sorted(remaining_files):
        if file != "cleanup_script.py":
            status = "🔍" if file in INVESTIGATION_MODULES else "✅"
            print(f"  {status} {file}")

if __name__ == "__main__":
    main()
