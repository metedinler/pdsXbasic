"""
module_validator.py - PDS-X BASIC v14u Modül Doğrulama Aracı
Version: 1.0.0
Date: June 9, 2025
"""

import importlib
import sys
import logging
from typing import Dict, List, Any
from pdsx_unified_exception import PdsXException

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_validation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("module_validator")

def validate_exports(module_name: str) -> Dict[str, Any]:
    """Bir modülün __pdsX_exports__ yapısını doğrular."""
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, "__pdsX_exports__"):
            raise PdsXException(f"{module_name} modülünde __pdsX_exports__ tanımı yok", "MISSING_EXPORTS")
        
        exports = module.__pdsX_exports__
        required_keys = ["classes", "functions", "variables"]
        
        for key in required_keys:
            if key not in exports:
                raise PdsXException(f"{module_name} modülünün __pdsX_exports__ yapısında {key} anahtarı eksik", "INVALID_EXPORTS")
        
        # Sınıfları kontrol et
        for class_name, class_obj in exports["classes"].items():
            if not isinstance(class_obj, type):
                raise PdsXException(f"{module_name} modülünde {class_name} geçerli bir sınıf değil", "INVALID_CLASS")
        
        # Fonksiyonları kontrol et
        for func_name, func_obj in exports["functions"].items():
            if not callable(func_obj):
                raise PdsXException(f"{module_name} modülünde {func_name} geçerli bir fonksiyon değil", "INVALID_FUNCTION")
        
        return exports
    except Exception as e:
        log.error(f"Modül doğrulama hatası ({module_name}): {str(e)}")
        raise

def check_syntax(module_name: str) -> bool:
    """Bir modülün sözdizimini kontrol eder."""
    try:
        with open(f"{module_name}.py", "r", encoding="utf-8") as f:
            content = f.read()
        compile(content, module_name + ".py", "exec")
        return True
    except SyntaxError as e:
        log.error(f"Sözdizimi hatası ({module_name}): {str(e)}")
        return False

def check_dependencies(module_name: str) -> bool:
    """Bir modülün bağımlılıklarını kontrol eder."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "__pdsX_exports__"):
            metadata = module.__pdsX_exports__.get("variables", {})
            dependencies = metadata.get("dependencies", [])
            
            for dep in dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    log.error(f"Bağımlılık hatası ({module_name}): {dep} yüklenemedi")
                    return False
        return True
    except Exception as e:
        log.error(f"Bağımlılık kontrolü hatası ({module_name}): {str(e)}")
        return False

def validate_all_modules(modules: List[str]) -> Dict[str, bool]:
    """Tüm modülleri doğrular."""
    results = {}
    for module_name in modules:
        try:
            syntax_ok = check_syntax(module_name)
            deps_ok = check_dependencies(module_name)
            exports_ok = bool(validate_exports(module_name))
            results[module_name] = all([syntax_ok, deps_ok, exports_ok])
        except Exception as e:
            log.error(f"Modül doğrulama hatası ({module_name}): {str(e)}")
            results[module_name] = False
    return results

class ModuleVersionValidator:
    """PDS-X modül versiyon kontrolü ve uyumluluk sınıfı."""
    
    def __init__(self):
        self.required_versions = {
            "core": "2.6",
            "tree": "3.0",
            "oop_and_class": "2.0", 
            "save_load_system": "2.0",
            "pdsx_exception": "2.0",
            "functional": "2.0",
            "graph": "2.0"
        }
    
    def check_module_version(self, module_name, module):
        """Modül versiyonunu kontrol eder."""
        try:
            if not hasattr(module, "__version__"):
                return False
            
            required = self.required_versions.get(module_name.split("_")[0], "1.0")
            current = module.__version__
            
            return self._compare_versions(current, required)
        except Exception as e:
            log.error(f"Modül versiyon kontrolü hatası: {module_name} - {str(e)}")
            return False
            
    def _compare_versions(self, current, required):
        """Versiyon karşılaştırması yapar."""
        try:
            current = [int(x) for x in current.split(".")]
            required = [int(x) for x in required.split(".")]
            
            for c, r in zip(current, required):
                if c > r:
                    return True
                if c < r:
                    return False
            return True
        except:
            return False

if __name__ == "__main__":
    # Test modüllerini doğrula
    test_modules = [
        "core2-5",
        "database_sql_isam",
        "bytecode_engine",
        "functional",
        "graph",
        "libx_ml",
        "libx_nlp",
        "libx_network",
        "pipe",
        "tree"
    ]
    
    results = validate_all_modules(test_modules)
    for module, is_valid in results.items():
        print(f"{module}: {'BAŞARILI' if is_valid else 'BAŞARISIZ'}")
