"""
Simple Module Manager for PDS-X
Just handles basic module loading without complex dependencies
"""

import os
import sys
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

class SimpleModuleManager:
    """Basit modül yöneticisi - sadece temel yükleme yapar"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.modules: Dict[str, object] = {}
        
    def load_module(self, module_name: str) -> Optional[object]:
        """Modülü yükle"""
        try:
            if module_name in self.modules:
                return self.modules[module_name]
                
            # Python builtin modülleri
            if module_name in sys.builtin_module_names:
                module = importlib.import_module(module_name)
                self.modules[module_name] = module
                return module
                
            # Dosya sistemden yükle
            module_path = self.base_path / f"{module_name}.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    self.modules[module_name] = module
                    return module
                    
            # Normal import deneme
            try:
                module = importlib.import_module(module_name)
                self.modules[module_name] = module
                return module
            except ImportError:
                pass
                
            log.warning(f"Modül bulunamadı: {module_name}")
            return None
            
        except Exception as e:
            log.error(f"Modül yükleme hatası {module_name}: {e}")
            return None
            
    def unload_module(self, module_name: str):
        """Modülü kaldır"""
        if module_name in self.modules:
            del self.modules[module_name]
        if module_name in sys.modules:
            del sys.modules[module_name]
            
    def get_loaded_modules(self):
        """Yüklü modülleri döndür"""
        return list(self.modules.keys())
