# auto_importer.py - PDS-X BASIC v15 Dinamik Modül Yükleyici
# Version: 1.5.0
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import os
import sys
import importlib.util
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set
from collections import defaultdict
from packaging import version
import threading
import time
import random
from typing import List

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("auto_importer")

REQUIRED_PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), 
    ("sklearn", "sklearn"), ("tensorflow", "tensorflow"),
    ("pdfplumber", "pdfplumber"), ("requests", "requests"), ("packaging", "packaging"), 
    ("psutil", "psutil"), ("pyyaml", "yaml"), ("graphviz", "graphviz"), ("aiofiles", "aiofiles"),
    ("seaborn", "seaborn"), ("plotly", "plotly"), ("dash", "dash"), ("aiohttp", "aiohttp"),
    ("grpcio", "grpc"), ("pyzmq", "zmq"), ("websocket-client", "websocket"),
    ("paho-mqtt", "paho.mqtt.client"), ("kafka-python", "kafka"),
    ("torch", "torch"), ("torch-geometric", "torch_geometric"), ("river", "river"),
    ("qiskit", "qiskit"), ("networkx", "networkx"),
    ("tk", "tkinter"), ("boto3", "boto3"), ("botocore", "botocore"),
    ("websockets", "websockets"), ("restrictedpython", "RestrictedPython"),
    ("rich", "rich"), ("colorama", "colorama"),
    ("textblob", "textblob"),
    # Standart kütüphaneler pip ile yüklenmez ama eksikse hata alınmasın diye eklenebilir
    # ("csv", "csv"), ("xml", "xml"), ("gzip", "gzip"), ("zlib", "zlib"), ("functools", "functools"), ("multiprocessing", "multiprocessing"), ("ctypes", "ctypes"), ("decimal", "decimal")
]

def install_missing_packages():
    """Gerekli pip paketlerini yükler."""
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=False)

install_missing_packages()

from pdsx_exception2 import PdsXException, PdsXSyntaxError, PdsXRuntimeError

class AutoImporter:
    """Dinamik modül yükleme ve bağımlılık yönetimi sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.loaded_modules: Dict[str, Any] = {}
        self.module_cache: Dict[str, Any] = {}
        self.imported_files: Set[str] = set()
        self.aliases: Dict[str, str] = {}
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        self.secure_mode: bool = False
        self.metadata: Dict = {"auto_importer": {"version": "1.5.0", "dependencies": []}}
        self.lock = threading.Lock()

    def load_module(self, file_path: str, alias: Optional[str] = None) -> Any:
        """Modül dosyasını yükler."""
        with self.lock:
            abs_path = os.path.abspath(file_path)
            if abs_path in self.imported_files:
                log.debug(f"Modül zaten yüklü: {abs_path}")
                return self.module_cache.get(abs_path)
            
            if self.secure_mode and not self._is_allowed_path(abs_path):
                raise PdsXRuntimeError(f"Güvenli modda dış modül yüklenemez: {file_path}", context={"source": "load_module"})

            try:
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                if alias:
                    module_name = alias
                    self.aliases[alias] = abs_path
                
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                if not spec:
                    raise PdsXSyntaxError(f"Modül spec oluşturulamadı: {file_path}", context={"source": "load_module"})
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                self.loaded_modules[module_name] = module
                self.module_cache[abs_path] = module
                self.imported_files.add(abs_path)
                
                # Bağımlılıkları kontrol et
                self._check_dependencies(module, module_name)
                
                log.info(f"Modül yüklendi: {module_name} ({abs_path})")
                return module
            except Exception as e:
                log.error(f"Modül yükleme hatası: {file_path}, {str(e)}")
                raise PdsXRuntimeError(f"Modül yükleme hatası: {file_path}, {str(e)}", context={"source": "load_module"})

    def unload_module(self, module_name: str) -> None:
        """Modülü bellekten kaldırır."""
        with self.lock:
            if module_name not in self.loaded_modules:
                raise PdsXRuntimeError(f"Modül yüklü değil: {module_name}", context={"source": "unload_module"})
            
            try:
                module = self.loaded_modules[module_name]
                abs_path = self.aliases.get(module_name, module.__file__)
                
                del sys.modules[module_name]
                del self.loaded_modules[module_name]
                self.module_cache.pop(abs_path, None)
                self.imported_files.discard(abs_path)
                self.aliases.pop(module_name, None)
                
                log.info(f"Modül kaldırıldı: {module_name}")
            except Exception as e:
                log.error(f"Modül kaldırma hatası: {module_name}, {str(e)}")
                raise PdsXRuntimeError(f"Modül kaldırma hatası: {module_name}, {str(e)}", context={"source": "unload_module"})

    def _is_allowed_path(self, path: str) -> bool:
        """Güvenli modda yolun izinli olup olmadığını kontrol eder."""
        allowed_dirs = [os.path.abspath("."), os.path.abspath("libs")]
        return any(path.startswith(d) for d in allowed_dirs)

    def _check_dependencies(self, module: Any, module_name: str) -> None:
        """Modül bağımlılıklarını kontrol eder."""
        required_deps = getattr(module, "metadata", {}).get("dependencies", [])
        for dep in required_deps:
            if dep not in self.loaded_modules:
                try:
                    self.load_module(f"libs/{dep}.py")
                    self.dependencies[module_name].append(dep)
                except Exception as e:
                    raise PdsXRuntimeError(f"Bağımlılık yüklenemedi: {dep}, {str(e)}", context={"source": "_check_dependencies"})

    def get_module_stats(self, module_name: str) -> Dict:
        """Modül istatistiklerini döndürür."""
        with self.lock:
            if module_name not in self.loaded_modules:
                raise PdsXRuntimeError(f"Modül yüklü değil: {module_name}", context={"source": "get_module_stats"})
            
            module = self.loaded_modules[module_name]
            abs_path = self.aliases.get(module_name, module.__file__)
            stats = {
                "name": module_name,
                "path": abs_path,
                "size": os.path.getsize(abs_path) / 1024,  # KB cinsinden
                "load_time": getattr(module, "load_time", time.time()),
                "dependencies": self.dependencies[module_name],
                "version": getattr(module, "metadata", {}).get("version", "unknown")
            }
            log.debug(f"Modül istatistikleri: {stats}")
            return stats

    def secure_mode_enable(self) -> None:
        """Güvenli modu etkinleştirir."""
        with self.lock:
            self.secure_mode = True
            log.info("Güvenli mod etkinleştirildi")

    def secure_mode_disable(self) -> None:
        """Güvenli modu devre dışı bırakır."""
        with self.lock:
            self.secure_mode = False
            log.info("Güvenli mod devre dışı bırakıldı")

    # Deneysel/Bilimsel İşlevler
    def quantum_load_simulation(self, module_name: str) -> Dict:
        """Kuantum simülasyonu ile modül yükleme performansı analizi."""
        # Deneysel: Kuantum simülasyonu tabanlı yükleme tahmini
        start_time = time.time()
        module = self.load_module(module_name)
        end_time = time.time()
        return {
            "module": module_name,
            "load_time": end_time - start_time,
            "simulated_quantum_efficiency": random.uniform(0.8, 0.95)  # Mock kuantum verimliliği
        }

    def chaos_load_prediction(self, module_name: str) -> float:
        """Kaotik sistem analizi ile yükleme süresi tahmini."""
        # Deneysel: Kaotik dinamikler kullanılarak yükleme süresi tahmini
        return random.uniform(0.1, 1.0)  # Mock kaotik tahmin

    def genetic_dependency_optimizer(self, module_name: str) -> List[str]:
        """Genetik algoritmalarla bağımlılık optimizasyonu."""
        # Deneysel: Genetik algoritmalarla bağımlılık sıralama
        return self.dependencies.get(module_name, [])[::-1]  # Mock sıralama

    def neural_load_balancer(self, module_name: str) -> float:
        """Nöral ağ tabanlı yük dengeleme."""
        # Deneysel: Nöral ağ ile yükleme yükü tahmini
        return random.uniform(0.5, 0.9)  # Mock dengeleme skoru

    def blockchain_module_validation(self, module_name: str) -> bool:
        """Blockchain tabanlı modül doğrulama."""
        # Deneysel: Modül doğruluğu için blockchain simülasyonu
        return True  # Mock doğrulama

if __name__ == "__main__":
    print("auto_importer.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")