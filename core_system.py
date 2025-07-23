# core_system.py - PDS-X Core System Wrapper
# Version: 1.0.0
# Date: December 21, 2024
# Wrapper for core2-6.py, libxcore.py, memory_manager.py systems

"""
Core System Wrapper for PDS-X Framework

Bu modül core2-6.py, libxcore.py ve memory_manager.py dosyalarındaki 
karmaşık import'ları ve class'ları daha temiz bir interface ile sağlar.
"""

import logging
from typing import Any, Dict, Optional

# Core system logger
log = logging.getLogger("core_system")

def get_core_manager(interpreter=None):
    """
    CoreManager instance'ını güvenli bir şekilde döndür
    """
    try:
        # core2-6.py dosyasından CoreManager'ı import et
        import importlib.util
        import os
        
        # Dosya path'ini kontrol et
        core_file = "core2_6.py"
        if not os.path.exists(core_file):
            log.warning(f"Core system dosyası bulunamadı: {core_file}")
            return None
        
        # Dynamic import
        spec = importlib.util.spec_from_file_location("core2_6", core_file)
        if spec is None or spec.loader is None:
            log.error("Core system modülü yüklenemedi")
            return None
            
        core2_6_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core2_6_module)
        
        # CoreManager class'ını al
        CoreManager = getattr(core2_6_module, 'CoreManager', None)
        if CoreManager is None:
            log.error("CoreManager class'ı bulunamadı")
            return None
        
        # CoreManager instance oluştur
        core_manager = CoreManager(interpreter)
        log.info("✅ CoreManager başarıyla yüklendi")
        return core_manager
        
    except Exception as e:
        log.error(f"Core system yükleme hatası: {e}")
        return None

def get_libx_core(interpreter=None):
    """
    LibXCore instance'ını güvenli bir şekilde döndür
    """
    try:
        from libxcore import LibXCore
        libx_core = LibXCore(interpreter)
        log.info("✅ LibXCore başarıyla yüklendi")
        return libx_core
    except Exception as e:
        log.error(f"LibXCore yükleme hatası: {e}")
        return None

def get_memory_manager(interpreter=None):
    """
    MemoryManager instance'ını güvenli bir şekilde döndür
    """
    try:
        from memory_manager import MemoryManager
        memory_manager = MemoryManager(interpreter)
        log.info("✅ MemoryManager başarıyla yüklendi")
        return memory_manager
    except Exception as e:
        log.error(f"MemoryManager yükleme hatası: {e}")
        return None

def get_core2_instance():
    """
    Core2 instance'ını güvenli bir şekilde döndür
    """
    try:
        import importlib.util
        import os
        
        core_file = "core2_6.py"
        if not os.path.exists(core_file):
            return None
            
        spec = importlib.util.spec_from_file_location("core2_6", core_file)
        if spec is None or spec.loader is None:
            return None
            
        core2_6_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core2_6_module)
        
        Core2 = getattr(core2_6_module, 'Core2', None)
        if Core2 is None:
            return None
        
        core2_instance = Core2()
        log.info("✅ Core2 instance başarıyla oluşturuldu")
        return core2_instance
        
    except Exception as e:
        log.error(f"Core2 instance oluşturma hatası: {e}")
        return None

def is_core_system_available():
    """
    Core system'in kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("core2_6.py")

def is_libx_available():
    """
    LibXCore'un kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("libxcore.py")

def is_memory_manager_available():
    """
    MemoryManager'ın kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("memory_manager.py")

def get_hybrid_executor(interpreter=None, auto_installer=None):
    """
    HybridCommandExecutor instance'ını güvenli bir şekilde döndür
    """
    try:
        from hybrid_command_executor import get_hybrid_executor as get_hybrid
        hybrid_executor = get_hybrid(interpreter, auto_installer)
        log.info("✅ HybridCommandExecutor başarıyla yüklendi")
        return hybrid_executor
    except Exception as e:
        log.error(f"HybridCommandExecutor yükleme hatası: {e}")
        return None

def is_hybrid_executor_available():
    """
    HybridCommandExecutor'ın kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("hybrid_command_executor.py")

def get_auto_installer(workspace_path=None):
    """
    AutoInstaller instance'ını güvenli bir şekilde döndür
    """
    try:
        from autoinstaller import AutoInstaller
        import os
        
        # Workspace path belirleme
        if workspace_path is None:
            workspace_path = os.getcwd()
            
        auto_installer = AutoInstaller(workspace_path)
        log.info("✅ AutoInstaller başarıyla yüklendi")
        return auto_installer
    except Exception as e:
        log.error(f"AutoInstaller yükleme hatası: {e}")
        return None

def is_auto_installer_available():
    """
    AutoInstaller'ın kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("autoinstaller.py")

def get_auto_importer():
    """
    AutoImporter instance'ını güvenli bir şekilde döndür
    """
    try:
        from auto_importer import AutoImporter
        auto_importer = AutoImporter()
        log.info("✅ AutoImporter başarıyla yüklendi")
        return auto_importer
    except Exception as e:
        log.error(f"AutoImporter yükleme hatası: {e}")
        return None

def is_auto_importer_available():
    """
    AutoImporter'ın kullanılabilir olup olmadığını kontrol et
    """
    import os
    return os.path.exists("auto_importer.py")

# Module level initialization
CORE_SYSTEM_AVAILABLE = is_core_system_available()
LIBX_AVAILABLE = is_libx_available()
MEMORY_MANAGER_AVAILABLE = is_memory_manager_available()
AUTO_IMPORTER_AVAILABLE = is_auto_importer_available()
AUTO_INSTALLER_AVAILABLE = is_auto_installer_available()
HYBRID_EXECUTOR_AVAILABLE = is_hybrid_executor_available()

# Status report
status_items = []
if CORE_SYSTEM_AVAILABLE:
    status_items.append("Core2_6")
if LIBX_AVAILABLE:
    status_items.append("LibX")
if MEMORY_MANAGER_AVAILABLE:
    status_items.append("MemoryManager")
if AUTO_IMPORTER_AVAILABLE:
    status_items.append("AutoImporter")
if AUTO_INSTALLER_AVAILABLE:
    status_items.append("AutoInstaller")
if HYBRID_EXECUTOR_AVAILABLE:
    status_items.append("HybridExecutor")

if status_items:
    log.info(f"🔧 Core system wrapper hazır: {', '.join(status_items)}")
else:
    log.warning("⚠️ Hiçbir core system bulunamadı")
