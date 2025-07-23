# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.7
# Date: June 22, 2025
# Author: xAI (Gemini ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot-Geminni)

import sys
import re
import json
import logging
import subprocess
import time
import importlib.util
import threading
import argparse
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
from collections import deque
import queue
import shutil
import ast

# --- PDS-X Özel Hata Sınıfları ---
# exception_manager3.py dosyasından alınmış gibi davranıyoruz.
class PdsXException(Exception):
    """PDS-X için temel hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"[PDS-X Error Code: {code}] {message}")
        self.message = message
        self.code = code
        self.context = context or {}

class PdsXImportError(PdsXException, ImportError):
    """Modül import edilirken oluşan hatalar için."""
    pass

class PdsXNotFoundError(PdsXException, FileNotFoundError):
    """Dosya veya kaynak bulunamadığında."""
    pass

class PdsXConflictError(PdsXException):
    """Bağımlılık çakışmaları için özel hata sınıfı."""
    pass

class PdsXInstallationError(PdsXException):
    """Paket kurulumu sırasında genel hata."""
    pass

# --- Lazy Loading Mekanizması ---
# Bu mekanizma, modülleri sadece ihtiyaç duyulduğunda yükleyerek başlangıç süresini kısaltır.
psutil = None
numpy = None
packaging = None
sklearn_components = {}
keyboard = None
winreg = None
elasticsearch_client = None # Değiştirildi
colorama = {}
graphviz_digraph = None # Değiştirildi

def get_psutil():
    """psutil'i lazy loading ile yükle"""
    global psutil
    if psutil is None:
        try:
            import psutil as ps
            psutil = ps
        except ImportError:
            print("[PDS-X] UYARI: 'psutil' yüklenemedi. Kaynak izleme devre dışı.")
    return psutil

def get_numpy():
    """NumPy'ı lazy loading ile yükle"""
    global numpy
    if numpy is None:
        try:
            import numpy as np
            numpy = np
        except ImportError:
            print("[PDS-X] UYARI: 'numpy' yüklenemedi. Bilimsel hesaplama özellikleri devre dışı.")
    return numpy

def get_sklearn_components():
    """Sklearn bileşenlerini lazy loading ile yükle"""
    global sklearn_components
    if not sklearn_components:
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            from sklearn.neural_network import MLPClassifier
            from sklearn.tree import DecisionTreeClassifier
            sklearn_components = {
                "IsolationForest": IsolationForest,
                "StandardScaler": StandardScaler,
                "MLPClassifier": MLPClassifier,
                "DecisionTreeClassifier": DecisionTreeClassifier,
            }
        except ImportError:
            print("[PDS-X] UYARI: 'scikit-learn' yüklenemedi. Anomali tespiti devre dışı.")
    return sklearn_components

def get_packaging_libs():
    """'packaging' kütüphanesi bileşenleri için lazy loader."""
    global packaging
    if packaging is None:
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
            from packaging.requirements import Requirement
            packaging = {
                "SpecifierSet": SpecifierSet,
                "Version": Version,
                "Requirement": Requirement,
            }
        except ImportError:
            print("[PDS-X] UYARI: 'packaging' kütüphanesi bulunamadı. Gelişmiş sürüm kontrolü devre dışı.")
            packaging = {} # Boş dictionary ata, böylece tekrar denemez
    return packaging

def get_elasticsearch_client():
    """Elasticsearch istemcisini lazy loading ile yükle."""
    global elasticsearch_client
    if elasticsearch_client is None:
        try:
            from elasticsearch import Elasticsearch
            elasticsearch_client = Elasticsearch
        except ImportError:
            print("[PDS-X] UYARI: 'elasticsearch' kütüphanesi bulunamadı. Elasticsearch entegrasyonu devre dışı.")
            elasticsearch_client = False # Tekrar denenmemesi için False olarak işaretle
    return elasticsearch_client

def get_graphviz_digraph():
    """Graphviz Digraph'ı lazy loading ile yükle."""
    global graphviz_digraph
    if graphviz_digraph is None:
        try:
            from graphviz import Digraph
            graphviz_digraph = Digraph
        except ImportError:
            print("[PDS-X] UYARI: 'graphviz' kütüphanesi bulunamadı. Bağımlılık grafiği oluşturma devre dışı.")
            graphviz_digraph = False # Tekrar denenmemesi için False olarak işaretle
    return graphviz_digraph

# Diğer lazy loader'lar
try:
    import winreg
except ImportError:
    winreg = None

try:
    from colorama import Fore, Style
    colorama = {'Fore': Fore, 'Style': Style}
except ImportError:
    # colorama yoksa, renk kodları olmadan çalışacak sahte nesneler oluştur
    class DummyColor:
        def __getattr__(self, name):
            return ''
    colorama = {'Fore': DummyColor(), 'Style': DummyColor()}


# --- Temel Yapılandırma ve Sabitler ---
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
VENV_DIR = BASE_DIR / ".pdsx_isolated_env"
CACHE_DIR = BASE_DIR / ".pdsx_cache"
ALIAS_FILE = BASE_DIR / "pdsx_aliases.txt"
TERMINAL_LOG_FILE = LOG_DIR / "pdsX_terminal.log"
JSONL_LOG_FILE = LOG_DIR / "pdsx_terminal.jsonl"

# --- Çalışma Modları ---
class OperatingMode(Enum):
    NORMAL = "NORMAL"
    SUPPRESSED = "SUPPRESSED"
    SILENT = "SILENT"
    TOTAL_SILENT = "TOTAL_SILENT"

class ModeManager:
    """Çalışma modunu yönetir ve çıktıları buna göre kontrol eder."""
    def __init__(self, mode: OperatingMode):
        self.mode = mode
        self.is_suppressed = mode in [OperatingMode.SUPPRESSED, OperatingMode.SILENT, OperatingMode.TOTAL_SILENT]
        self.is_silent = mode in [OperatingMode.SILENT, OperatingMode.TOTAL_SILENT]

    def should_print_terminal(self) -> bool:
        return not self.is_silent

    def should_log_standard(self) -> bool:
        return self.mode != OperatingMode.TOTAL_SILENT

    def print(self, message: str, pdsx_prefix: bool = False):
        if self.mode == OperatingMode.NORMAL:
            print(message)
        elif self.mode == OperatingMode.SUPPRESSED and pdsx_prefix:
            print(message)

mode_manager: Optional[ModeManager] = None

# --- Gelişmiş Loglama Sistemi ---
class AdvancedLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AdvancedLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Path = LOG_DIR, terminal_log_path: Path = TERMINAL_LOG_FILE, jsonl_log_path: Path = JSONL_LOG_FILE):
        if hasattr(self, 'initialized'):
            return
        
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)

        # Log Yedekleme
        if terminal_log_path.exists():
            try:
                backup_path = terminal_log_path.with_suffix('.log.bak')
                if backup_path.exists():
                    backup_path.unlink()
                terminal_log_path.rename(backup_path)
            except OSError as e:
                print(f"[PDS-X] Uyarı: Terminal log yedeği alınamadı: {e}")

        self.logger = logging.getLogger("PDS_X_Logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # JSONL Handler
        self.jsonl_handler = logging.FileHandler(jsonl_log_path, mode='a', encoding='utf-8')
        self.jsonl_handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'))
        self.logger.addHandler(self.jsonl_handler)

        # Standart File Handler
        self.log_file_path = self.log_dir / "pdsx_auto_importer.log"
        self.file_handler = logging.FileHandler(self.log_file_path, mode='a', encoding='utf-8')
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.file_handler)

        # Console Handler
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)

        self.initialized = True

    def log(self, level: str, message: str, *args, **kwargs):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        
        # Mesajdaki JSON uyumsuz karakterleri temizle
        cleaned_message = json.dumps(message)[1:-1]
        
        # ModeManager'a göre handler'ları yönet
        if mode_manager:
            if not mode_manager.should_print_terminal() and self.console_handler in self.logger.handlers:
                self.logger.removeHandler(self.console_handler)
            elif mode_manager.should_print_terminal() and self.console_handler not in self.logger.handlers:
                self.logger.addHandler(self.console_handler)

            if not mode_manager.should_log_standard() and self.file_handler in self.logger.handlers:
                self.logger.removeHandler(self.file_handler)
            elif mode_manager.should_log_standard() and self.file_handler not in self.logger.handlers:
                self.logger.addHandler(self.file_handler)
        
        log_func(cleaned_message, *args, **kwargs)

class Tee(io.TextIOWrapper):
    """Hem dosyaya hem de orijinal stdout/stderr'e yazan bir nesne."""
    def __init__(self, original_stream, log_file_path):
        self.original_stream = original_stream
        try:
            self.log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
        except IOError as e:
            self.log_file = None
            self.original_stream.write(f"[PDS-X] HATA: Terminal log dosyası açılamadı: {log_file_path}\n{e}\n")
        
        # sys.stdout/stderr'i değiştir
        if original_stream == sys.stdout:
            sys.stdout = self
        else:
            sys.stderr = self

    def write(self, text):
        self.original_stream.write(text)
        if self.log_file:
            self.log_file.write(text)

    def flush(self):
        self.original_stream.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.original_stream == sys.stdout:
            sys.stdout = self.original_stream
        else:
            sys.stderr = self.original_stream
        if self.log_file:
            self.log_file.close()

# --- EKSİK YARDIMCI SINIFLARIN TANIMLANMASI ---

class DependencyRegistry:
    """Kurulu paketlerin ve bağımlılıklarının kaydını tutar."""
    def __init__(self, logger: AdvancedLogger, file_path: Path = BASE_DIR / "dependencies.json"):
        self.logger = logger
        self.file_path = file_path
        self.registry = self._load()

    def _load(self) -> Dict[str, Any]:
        """Kayıt dosyasını yükler."""
        if not self.file_path.exists():
            self.logger.log("info", "Bağımlılık kayıt dosyası bulunamadı, yeni bir tane oluşturuluyor.")
            return {"version": "1.0", "packages": {}}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası yüklenirken hata: {e}. Yeni bir kayıt oluşturuluyor.")
            return {"version": "1.0", "packages": {}}

    def save(self):
        """Mevcut kayıt durumunu dosyaya yazar."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4)
            self.logger.log("debug", f"Bağımlılık kaydı şuraya kaydedildi: {self.file_path}")
        except IOError as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası kaydedilirken hata: {e}")

    def add_package(self, package_name: str, version: str, dependencies: List[str]):
        """Kayıtlara yeni bir paket ekler."""
        self.registry["packages"][package_name] = {
            "version": version,
            "dependencies": dependencies,
            "installed_at": datetime.now().isoformat()
        }
        self.save()

class PipOutputAnalyzer:
    """'pip' komutunun çıktılarını analiz ederek detayları çıkarır."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.dep_pattern = re.compile(r"Collecting\s+([a-zA-Z0-9_.-]+)")
        self.version_pattern = re.compile(r"Successfully installed\s+.*\s+([a-zA-Z0-9_.-]+)-(\d+\.\d+(\.\d+)?([a-z0-9.-]*))")

    def analyze(self, output: str) -> Dict[str, Any]:
        """Pip çıktısını analiz eder ve kurulan paketler ile bağımlılıkları döndürür."""
        dependencies = self.dep_pattern.findall(output)
        main_package_match = self.version_pattern.search(output)
        
        if not main_package_match:
            self.logger.log("warning", "Pip çıktısında ana paket ve versiyon bilgisi bulunamadı.")
            return {"main_package": None, "version": None, "dependencies": []}

        main_package = main_package_match.group(1)
        version = main_package_match.group(2)
        
        # Ana paketi bağımlılık listesinden çıkar
        cleaned_deps = [dep for dep in dependencies if dep.lower() != main_package.lower()]

        return {
            "main_package": main_package,
            "version": version,
            "dependencies": cleaned_deps
        }

class ModuleSummaryGenerator:
    """Kurulum işlemleri hakkında özet ve istatistikler oluşturur."""
    def __init__(self, logger: AdvancedLogger, mode_manager: ModeManager):
        self.logger = logger
        self.mode_manager = mode_manager

    def generate_summary(self, results: List[Dict[str, Any]]):
        """Tamamlanan kurulumlar için bir özet oluşturur ve yazdırır."""
        if self.mode_manager.is_silent:
            return

        successful_installs = [r for r in results if r.get("success")]
        failed_installs = [r for r in results if not r.get("success")]
        total_time = sum(r.get("duration", 0) for r in results)

        summary = [
            "\n" + "="*25 + " PDS-X Kurulum Özeti " + "="*25,
            f"Toplam {len(results)} paket işlendi. Süre: {total_time:.2f} saniye.",
            f"Başarılı Kurulumlar ({len(successful_installs)}):",
            "  " + ", ".join([r['package'] for r in successful_installs]) if successful_installs else "  Yok",
            f"Başarısız Kurulumlar ({len(failed_installs)}):",
            "  " + ", ".join([r['package'] for r in failed_installs]) if failed_installs else "  Yok",
            "="*70 + "\n"
        ]
        
        # Önemli not: log fonksiyonu yerine doğrudan print kullanıyoruz ki formatlama bozulmasın.
        self.logger.log("info", "\n".join(summary))


class ResourceMonitor:
    """Sistem kaynaklarını (CPU, Bellek) izler."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.psutil = get_psutil()
        self.running = False
        self.thread = None
        if not self.psutil:
            self.logger.log("warning", "psutil modülü bulunamadı. Kaynak izleme devre dışı.")

    def start(self):
        if not self.psutil or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        self.logger.log("info", "Kaynak izleyici başlatıldı.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.logger.log("info", "Kaynak izleyici durduruldu.")

    def _monitor(self):
        while self.running:
            try:
                cpu_usage = self.psutil.cpu_percent(interval=5)
                memory_info = self.psutil.virtual_memory()
                self.logger.log("debug", f"Kaynak Kullanımı: CPU: {cpu_usage}%, Bellek: {memory_info.percent}%")
            except Exception as e:
                self.logger.log("error", f"Kaynak izleme sırasında hata: {e}")
                self.running = False # Hata durumunda döngüyü sonlandır

class AsyncDownloadManager:
    """Asenkron işlemler için (örn. wheel indirme) bir havuz yönetir."""
    def __init__(self, importer_instance: 'AutoImporter', max_workers: int = 4):
        self.importer = importer_instance
        self.logger = self.importer.logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger.log("info", f"Asenkron indirme yöneticisi {max_workers} işçi ile başlatıldı.")

    def submit_download(self, package_spec: str):
        """Bir wheel indirme görevini havuza gönderir."""
        self.logger.log("info", f"'{package_spec}' için asenkron wheel indirme görevi gönderildi.")
        return self.executor.submit(self.importer.wheel_cache.download_wheel, package_spec)

class ScientificUtils:
    """Bilimsel ve ağır hesaplama gerektiren görevler için ayrı bir işlem havuzu yönetir."""
    def __init__(self, logger: AdvancedLogger, max_workers: int = 2):
        self.logger = logger
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        self.logger.log("info", f"Bilimsel hesaplama işlem havuzu {max_workers} işçi ile başlatıldı.")

    def submit_task(self, func, *args, **kwargs):
        """Ağır bir görevi işlem havuzuna gönderir."""
        self.logger.log("debug", f"'{func.__name__}' görevi bilimsel işlem havuzuna gönderildi.")
        return self.process_executor.submit(func, *args, **kwargs)

class EnvManager:
    """Python ortamını ve pip/python yollarını yönetir."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.python_path, self.pip_path = self._find_paths()
        self.logger.log("info", f"Python yolu: {self.python_path}")
        self.logger.log("info", f"Pip yolu: {self.pip_path}")

    def _find_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Mevcut Python ve pip yürütülebilir dosyalarının yollarını bulur."""
        python_path = Path(sys.executable)
        pip_path = python_path.parent / "pip.exe"
        if not pip_path.exists():
             pip_path = python_path.parent / "pip" # for non-windows
        
        if not python_path.exists():
            self.logger.log("error", "sys.executable yolu bulunamadı.")
            return None, None
        if not pip_path.exists():
            self.logger.log("warning", f"Pip yürütülebilir dosyası beklenen yolda bulunamadı: {pip_path}")
            # Alternatif bulma mekanizması eklenebilir
            return python_path, None
            
        return python_path, pip_path

class WheelCacheManager:
    """İndirilen Python wheel dosyalarını yönetir."""
    def __init__(self, cache_dir: Path, logger: AdvancedLogger, env_manager: EnvManager):
        self.cache_dir = cache_dir / "wheels"
        self.logger = logger
        self.env_manager = env_manager
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log("info", f"Wheel cache yöneticisi başlatıldı. Cache dizini: {self.cache_dir}")

    def find_wheel(self, package_spec: str) -> Optional[Path]:
        """Verilen bir paket için cache'de uygun bir wheel dosyası arar."""
        package_name = package_spec.split("==")[0].split("[")[0].replace("-", "_")
        for wheel_file in self.cache_dir.glob(f"{package_name}-*.whl"):
            # Burada daha karmaşık bir sürüm kontrolü yapılabilir, şimdilik ilk bulduğunu döndürür.
            self.logger.log("info", f"Cache'de '{package_spec}' için uygun wheel bulundu: {wheel_file.name}")
            return wheel_file
        self.logger.log("info", f"Cache'de '{package_spec}' için wheel bulunamadı.")
        return None

    def download_wheel(self, package_spec: str) -> Optional[Path]:
        """Bir paketi sadece wheel olarak indirir ve cache'e kaydeder."""
        self.logger.log("info", f"'{package_spec}' için wheel indirme işlemi başlatılıyor...")
        cmd = [
            str(self.env_manager.pip_path),
            "wheel",
            "--no-deps",  # Sadece ana paketi indir, bağımlılıkları değil
            "-w", str(self.cache_dir),
            package_spec
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            self.logger.log("debug", f"pip wheel çıktısı: {result.stdout}")
            # İndirilen dosyanın adını çıktıdan bulmak gerekir
            for line in result.stdout.splitlines():
                if "Successfully downloaded" in line:
                    # Bu çıktı formatı varsayımsaldır, pip versiyonuna göre değişebilir
                    downloaded_file_name = line.split(" ")[-1]
                    wheel_path = self.cache_dir / downloaded_file_name
                    if wheel_path.exists():
                        self.logger.log("info", f"'{package_spec}' başarıyla indirildi ve cache'e eklendi: {wheel_path}")
                        return wheel_path
            # Eğer yukarıdaki mantık çalışmazsa, indirilen dosyayı manuel bul
            return self.find_wheel(package_spec)

        except subprocess.CalledProcessError as e:
            self.logger.log("error", f"'{package_spec}' wheel indirilirken hata oluştu: {e.stderr}")
            return None
        except Exception as e:
            self.logger.log("error", f"'{package_spec}' wheel indirilirken beklenmedik hata: {e}")
            return None

    def clear_cache(self):
        """Tüm wheel cache'ini temizler."""
        self.logger.log("info", "Wheel cache temizleniyor...")
        for item in self.cache_dir.iterdir():
            item.unlink() if item.is_file() else shutil.rmtree(item)
        self.logger.log("info", "Wheel cache temizlendi.")

class KillSwitch:
    """Acil durdurma mekanizması."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.is_active = False

    def activate(self):
        self.is_active = True
        self.logger.log("warning", "KILL SWITCH AKTİF! Tüm operasyonlar durduruluyor.")

    def check(self):
        if self.is_active:
            raise InterruptedError("KillSwitch ile operasyon durduruldu.")

# --- YARDIMCI YÖNETİCİ SINIFLARI (Referans Mimariden) ---

class GracefulShutdownManager:
    """Uygulamanın düzgün bir şekilde sonlandırılmasını yönetir."""
    def __init__(self, logger):
        self.logger = logger
        self.tasks = []
        self.shutdown_requested = threading.Event()

    def register(self, task_name, stop_function):
        """Durdurulacak bir görevi kaydeder."""
        self.tasks.append({'name': task_name, 'stop_func': stop_function})
        self.logger.log("debug", f"Kapatma görevi kaydedildi: {task_name}")

    def shutdown(self):
        """Tüm kayıtlı görevleri sırayla durdurur."""
        if self.shutdown_requested.is_set():
            return
        self.logger.log("info", "Graceful shutdown başlatılıyor...")
        self.shutdown_requested.set()
        for task in reversed(self.tasks):
            try:
                self.logger.log("info", f"'{task['name']}' görevi durduruluyor...")
                task['stop_func']()
            except Exception as e:
                self.logger.log("error", f"'{task['name']}' görevi durdurulurken hata: {e}")
        self.logger.log("info", "Graceful shutdown tamamlandı.")

class DependencyOptimizer:
    """
    Kurulum sırasını optimize eder ve bağımlılık grafiğini analiz eder.
    'learned_dependencies.json' dosyasını kullanarak daha akıllı kararlar alabilir.
    """
    def __init__(self, logger: AdvancedLogger, dependency_registry: 'DependencyRegistry', learned_deps_path: Path):
        self.logger = logger
        self.dependency_registry = dependency_registry
        self.learned_deps_path = learned_deps_path
        self.learned_dependencies = self._load_learned_dependencies()
        self.graph = self._build_dependency_graph()

    def _load_learned_dependencies(self) -> Dict[str, List[str]]:
        """Öğrenilmiş bağımlılıkları dosyadan yükler."""
        try:
            with open(BASE_DIR / "learned_dependencies.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                self.logger.log("info", "Öğrenilmiş bağımlılıklar başarıyla yüklendi.")
                return data.get("dependencies", {})
        except FileNotFoundError:
            self.logger.log("info", "Öğrenilmiş bağımlılıklar dosyası bulunamadı.")
            return {}
        except Exception as e:
            self.logger.log("error", f"Öğrenilmiş bağımlılık dosyası okunurken hata: {e}")
            return {}

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Hem kayıtlı hem de öğrenilmiş bağımlılıklardan birleşik bir graf oluşturur.
        """
        graph = {}
        # Öğrenilmiş bağımlılıkları ekle
        for package, deps in self.learned_dependencies.items():
            graph[package] = list(set(deps))

        # Kayıtlı (kurulu) bağımlılıkları ekle/güncelle
        installed_packages = self.dependency_registry.registry.get("packages", {})
        for package, data in installed_packages.items():
            deps = data.get("dependencies", [])
            if package in graph:
                graph[package] = list(set(graph[package] + deps))
            else:
                graph[package] = list(set(deps))
        
        self.logger.log("debug", "Bağımlılık grafiği oluşturuldu.")
        return graph

    def get_optimized_order(self, packages_to_install: List[str]) -> List[str]:
        """
        Verilen paket listesi ve bilinen bağımlılıkları için topolojik sıralama kullanarak
        optimize edilmiş tam bir kurulum sırası döndürür.
        """
        self.logger.log("info", f"Kurulum sırası optimizasyonu başlatıldı for: {packages_to_install}")

        # 1. Adım: Kurulacak tüm paketleri ve bunların bilinen tüm bağımlılıklarını topla.
        all_packages_to_consider = set()
        q = deque()
        original_package_map = {}

        for pkg in packages_to_install:
            pkg_name_only = pkg.split("==")[0].split("[")[0]
            original_package_map[pkg_name_only] = pkg
            if pkg_name_only not in all_packages_to_consider:
                all_packages_to_consider.add(pkg_name_only)
                q.append(pkg_name_only)

        while q:
            package_name = q.popleft()
            dependencies = self.graph.get(package_name, [])
            for dep in dependencies:
                if dep not in all_packages_to_consider:
                    all_packages_to_consider.add(dep)
                    q.append(dep)

        # 2. Adım: Toplanan tüm paketler üzerinde topolojik sıralama yap.
        sorted_order = []
        visited = set()
        
        for pkg_name in list(all_packages_to_consider):
            if pkg_name not in visited:
                if not self._topological_sort_util(pkg_name, visited, set(), sorted_order):
                    self.logger.log("warning", "Bağımlılık grafiğinde döngü tespit edildi! Optimizasyon atlanıyor.")
                    return packages_to_install

        # 3. Adım: Son listeyi oluştur, orijinal versiyonları geri yükle.
        final_optimized_list = [original_package_map.get(p, p) for p in sorted_order]

        self.logger.log("info", f"Optimize edilmiş ve genişletilmiş kurulum sırası: {final_optimized_list}")
        return final_optimized_list


    def _topological_sort_util(self, package: str, visited: set, recursion_stack: set, sorted_order: List[str]):
        """Topolojik sıralama için yardımcı DFS fonksiyonu."""
        visited.add(package)
        recursion_stack.add(package)

        # Bu paketin bağımlılıklarını gez
        dependencies = self.graph.get(package, [])
        for dep in dependencies:
            if dep not in visited:
                if not self._topological_sort_util(dep, visited, recursion_stack, sorted_order):
                    return False # Döngü tespit edildi
            elif dep in recursion_stack:
                return False # Döngü tespit edildi

        # Tüm bağımlılıklar gezildikten sonra paketi listeye ekle
        if package not in sorted_order:
             sorted_order.append(package)
        recursion_stack.remove(package)
        return True

class ConflictManager:
    """Bağımlılık çakışmalarını yönetir."""
    def __init__(self, logger, dependency_registry):
        self.logger = logger
        self.dependency_registry = dependency_registry
        self.packaging_libs = get_packaging_libs()

    def check_conflicts(self, package_requirement: str) -> Optional[str]:
        """Bir paketin mevcut ortamla çakışıp çakışmadığını kontrol eder."""
        self.logger.log("info", f"'{package_requirement}' için çakışma kontrolü yapılıyor...")
        if not self.packaging_libs:
            self.logger.log("warning", "'packaging' kütüphanesi bulunamadığı için çakışma kontrolü atlanıyor.")
            return None
        
        try:
            Requirement = self.packaging_libs["Requirement"]
            req = Requirement(package_requirement)
            
            # Mevcut kurulu paketlerle karşılaştır (basit kontrol)
            # Gerçek bir implementasyon için 'pip' komutunun çıktısını analiz etmek gerekir.
            installed_packages = self.dependency_registry.registry.get("packages", {})
            for name, data in installed_packages.items():
                if name == req.name:
                    # Versiyon çakışması kontrolü
                    # Bu kısım daha da geliştirilebilir.
                    self.logger.log("warning", f"Potansiyel çakışma: '{req.name}' zaten kurulu (versiyon: {data.get('version', 'bilinmiyor')}).")

        except Exception as e:
            self.logger.log("error", f"Geçersiz gereksinim dizesi: '{package_requirement}'. Hata: {e}")
            return f"Geçersiz gereksinim: {package_requirement}"
            
        self.logger.log("info", f"'{package_requirement}' için önemli bir çakışma bulunamadı (basit kontrol).")
        return None

class CodeAnalyzer:
    """
    Kaynak kodunu statik olarak analiz ederek import ifadelerini ve potansiyel bağımlılıkları bulur.
    Önceki ModuleAnalyzer'ın yerini alır ve AST (Abstract Syntax Tree) kullanır.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def find_imports_from_source(self, file_path: Path) -> List[str]:
        """Bir Python kaynak dosyasını okur ve içindeki tüm importları bulur."""
        if not file_path.exists() or not file_path.is_file():
            self.logger.log("warning", f"Kod analizi için kaynak dosya bulunamadı: {file_path}")
            return []
        
        self.logger.log("info", f"Kaynak dosya analiz ediliyor: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=file_path.name)
            
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # 'import a.b.c' durumunda en üst seviye paket 'a'dır.
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    # 'from a.b import c' durumunda, 'a'yı al.
                    # 'from . import x' gibi göreceli importları şimdilik atla (level > 0)
                    if node.level == 0 and node.module:
                        imports.add(node.module.split('.')[0])
            
            self.logger.log("info", f"'{file_path.name}' içinde bulunan modüller: {list(imports)}")
            return list(imports)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.log("error", f"'{file_path.name}' dosyası analiz edilirken hata oluştu: {e}")
            return []
        except Exception as e:
            self.logger.log("error", f"'{file_path.name}' dosyası okunurken beklenmedik hata: {e}")
            return []


class TerminalLogAnalyzer:
    """Terminal log dosyasını analiz ederek hataları ve eksik modülleri bulur."""
    def __init__(self, logger):
        self.logger = logger
        # Örnek: "ImportError: No module named 'requests'" veya "ModuleNotFoundError: No module named 'numpy'"
        self.import_error_pattern = re.compile(
            r"(?:ImportError|ModuleNotFoundError): No module named '([^']*)'"
        )

    def analyze_line(self, line: str) -> Optional[str]:
        """Tek bir log satırını analiz eder ve eksik modül adını döndürür."""
        match = self.import_error_pattern.search(line)
        if match:
            missing_module = match.group(1)
            self.logger.log("info", f"Terminal logunda eksik modül tespit edildi: {missing_module}")
            return missing_module
        return None

class RealTimeLogMonitor:
    """Terminal log dosyasını gerçek zamanlı olarak izler ve kurulumları tetikler."""
    def __init__(self, log_file: Path, analyzer: TerminalLogAnalyzer, importer: 'AutoImporter', logger: AdvancedLogger):
        self.log_file = log_file
        self.analyzer = analyzer
        self.importer = importer
        self.logger = logger
        self.running = False
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        """İzleme thread'ini başlatır."""
        if not self.log_file.exists():
            self.logger.log("warning", f"İzlenecek log dosyası bulunamadı: {self.log_file}. İzleyici başlatılmıyor.")
            # Yine de dosyayı oluşturabiliriz.
            self.log_file.touch()
            
        self.running = True
        self.thread.start()
        self.logger.log("info", f"Gerçek zamanlı log izleyici '{self.log_file}' için başlatıldı.")

    def stop(self):
        """İzleme thread'ini durdurur."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.log("info", "Gerçek zamanlı log izleyici durduruldu.")

    def _monitor(self):
        """Log dosyasını izleyen ana döngü."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                f.seek(0, 2) # Dosyanın sonuna git
                while self.running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5) # Yeni satır yoksa bekle
                        continue
                    
                    missing_module = self.analyzer.analyze_line(line)
                    if missing_module:
                        self.logger.log("info", f"İzleyici, '{missing_module}' için otomatik kurulumu tetikliyor.")
                        # Doğrudan kurulum kuyruğuna ekle
                        self.importer.auto_install_package(missing_module)
        except FileNotFoundError:
            self.logger.log("warning", f"İzleme sırasında log dosyası kayboldu: {self.log_file}")
        except Exception as e:
            self.logger.log("error", f"Log izleme sırasında kritik hata: {e}")


# --- Ana AutoImporter Sınıfı ---
class AutoImporter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AutoImporter, cls).__new__(cls)
        return cls._instance

    def __init__(self, mode: str = 'NORMAL', monitor_terminal: bool = False, replay_args=None):
        if hasattr(self, 'initialized'):
            return

        # Modu ayarla
        try:
            op_mode = OperatingMode[mode.upper()]
        except KeyError:
            print(f"[PDS-X] UYARI: Geçersiz mod '{mode}'. 'NORMAL' moda geçiliyor.")
            op_mode = OperatingMode.NORMAL
            
        global mode_manager
        self.mode_manager = ModeManager(op_mode)
        mode_manager = self.mode_manager

        # Temel bileşenleri başlat
        self.logger = AdvancedLogger()
        self.shutdown_manager = GracefulShutdownManager(self.logger)
        self.env_manager = EnvManager(logger=self.logger)
        self.dependency_registry = DependencyRegistry(self.logger)
        self.pip_output_analyzer = PipOutputAnalyzer(self.logger)
        self.wheel_cache = WheelCacheManager(CACHE_DIR, self.logger, self.env_manager)
        self.kill_switch = KillSwitch(self.logger)
        self.summary_generator = ModuleSummaryGenerator(self.logger, self.mode_manager)
        
        # Gelişmiş ve analitik bileşenler
        self.dependency_optimizer = DependencyOptimizer(self.logger, self.dependency_registry, BASE_DIR / "learned_dependencies.json")
        self.conflict_manager = ConflictManager(self.logger, self.dependency_registry)
        self.code_analyzer = CodeAnalyzer(self.logger)
        self.resource_monitor = ResourceMonitor(self.logger)
        self.async_download_manager = AsyncDownloadManager(self)
        self.scientific_utils = ScientificUtils(self.logger)

        # Gerçek zamanlı izleme bileşenleri
        self.terminal_log_analyzer = TerminalLogAnalyzer(self.logger)
        self.log_monitor = RealTimeLogMonitor(TERMINAL_LOG_FILE, self.terminal_log_analyzer, self, self.logger)

        # Durum nitelikleri
        self.lock = threading.Lock()
        self.installation_queue = queue.Queue()
        self.active_install_threads = 0
        self.installation_complete_event = threading.Event()
        self.installation_complete_event.set() # Başlangıçta tamamlanmış durumda

        self.failed_packages = {}
        self.aliases = {}
        self.auto_install_enabled = True
        self.skip_modules = set(sys.builtin_module_names)
        
        self._prepare_aliases()
        
        # Kapatma yöneticisine görevleri kaydet
        self.shutdown_manager.register("ResourceMonitor", self.resource_monitor.stop)
        self.shutdown_manager.register("LogMonitor", self.log_monitor.stop)
        self.shutdown_manager.register("AsyncDownloadManager", self.async_download_manager.executor.shutdown)
        self.shutdown_manager.register("ScientificUtils", self.scientific_utils.process_executor.shutdown)

        # Tee nesnelerini kapatma görevini en sona ekle
        def close_tee_streams():
            if isinstance(sys.stdout, Tee):
                sys.stdout.close()
            if isinstance(sys.sauto_importer_fixed.pytderr, Tee):
                sys.stderr.close()
        self.shutdown_manager.register("TeeStreamCloser", close_tee_streams)


        if monitor_terminal:
            self.log_monitor.start()

        self.logger.log("info", f"AutoImporter v1.7.9.2 başlatıldı. Mod: {op_mode.name}, Terminal İzleme: {'Aktif' if monitor_terminal else 'Pasif'}")
        self.initialized = True

    def check_package_installed(self, package_name: str) -> bool:
        """Bir paketin mevcut ortamda kurulu olup olmadığını kontrol eder."""
        if not self.env_manager.pip_path:
            self.logger.log("warning", "Pip yolu bulunamadığı için paket kontrolü yapılamıyor.")
            return False
        try:
            cmd = [str(self.env_manager.pip_path), "show", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
            return result.returncode == 0
        except Exception as e:
            self.logger.log("error", f"'{package_name}' paketi kontrol edilirken hata: {e}")
            return False

    def auto_install_package(self, module_name: str, package_name: str = None, version: str = None):
        """
        Bir modül için kurulum sürecini başlatan ana giriş noktası.
        Kurulumu kuyruğa ekler ve işlemeyi tetikler.
        """
        with self.lock:
            if not self.auto_install_enabled:
                self.logger.log("info", "Otomatik kurulum devre dışı, işlem atlanıyor.")
                return

            pkg_to_install = self._resolve_package_name(module_name, package_name)
            if version:
                pkg_to_install += f"=={version}"

            # Zaten kuyrukta veya hatalı olarak işaretlenmiş mi kontrol et
            if pkg_to_install in list(self.installation_queue.queue) or pkg_to_install in self.failed_packages:
                self.logger.log("debug", f"Paket '{pkg_to_install}' zaten kuyrukta veya hatalı listesinde.")
                return

            self.logger.log("info", f"Otomatik kurulum için kuyruğa ekleniyor: {pkg_to_install}")
            self.installation_queue.put(pkg_to_install)
            self.installation_complete_event.clear() # Kurulum başladığında olayı temizle
        
        # Ayrı bir thread'de kuyruğu işle, ana thread'i bloklama
        # Eğer zaten bir işleyici çalışmıyorsa yenisini başlat
        with self.lock:
            if self.active_install_threads == 0:
                self.active_install_threads += 1
                threading.Thread(target=self._process_installation_queue).start()


    def install_package(self, package: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """
        Bir paketi kurar ve sonucu (başarı, hata, çıktı) bir sözlük olarak döndürür.
        Bu fonksiyon doğrudan _install_package_with_retry tarafından kullanılır.
        """
        start_time = time.time()
        self.logger.log("info", f"'{package}' kurulumu başlatılıyor... Argümanlar: {extra_args}")
        
        try:
            self.kill_switch.check()

            cmd = [str(self.env_manager.pip_path), "install"]
            if extra_args:
                cmd.extend(extra_args)
            cmd.append(package)

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            duration = time.time() - start_time

            output = result.stdout + "\\n" + result.stderr
            
            return {
                "returncode": result.returncode,
                "output": output,
                "duration": duration,
                "package": package
            }

        except Exception as e:
            duration = time.time() - start_time
            self.logger.log("error", f"'{package}' kurulumunda beklenmedik bir alt işlem hatası: {e}")
            return {
                "returncode": -1,
                "output": str(e),
                "duration": duration,
                "package": package,
                "exception": e
            }

    def _install_package_with_retry(self, package: str, max_retries: int = 3) -> List[str]:
        """
        Bir paketi akıllı deneme mekanizması ile kurar.
        Başarısız olursa ve yeni bağımlılıklar tespit ederse, bu bağımlılıkların bir listesini döndürür.
        """
        with self.lock:
            pkg_name_only = package.split("==")[0].split("[")[0]
            if self.check_package_installed(pkg_name_only):
                self.logger.log("info", f"Paket '{pkg_name_only}' zaten kurulu, kurulum atlanıyor.")
                self.summary_generator.add_module_status(pkg_name_only, "Atlandı", 0, reason="Zaten yüklü")
                return []

        retries = 0
        current_args = []
        new_dependencies_to_install = []
        package_to_install = package # Başlangıçta PyPI'dan alınacak paket adı

        # 1. Adım: Cache'i kontrol et
        cached_wheel_path = self.wheel_cache.find_wheel(package)
        if cached_wheel_path:
            self.logger.log("info", f"'{package}' için cache kullanılıyor: {cached_wheel_path}")
            package_to_install = str(cached_wheel_path)

        while retries < max_retries:
            self.logger.log("info", f"'{package_to_install}' kurulum denemesi {retries + 1}/{max_retries} (Argümanlar: {current_args})")
            
            install_result = self.install_package(package_to_install, extra_args=current_args)
            output = install_result["output"]
            duration = install_result["duration"]
            pkg_name_only = package.split("==")[0].split("[")[0]

            if install_result["returncode"] == 0:
                self.logger.log("info", f"'{package}' başarıyla kuruldu.")
                # Başarılı kurulumdan sonra, eğer cache'den kurulmadıysa, wheel'i cache'e ekle
                if not cached_wheel_path:
                    self.wheel_cache.download_wheel(package)

                self.dependency_registry.register_package(pkg_name_only, "latest", "Başarılı")
                reason = f"Düzeltildi ({current_args})" if retries > 0 else ""
                self.summary_generator.add_module_status(pkg_name_only, "Başarılı", duration, reason=reason)
                return new_dependencies_to_install

            # Kurulum başarısız, analiz et
            self.logger.log("warning", f"'{package}' kurulumu başarısız oldu. Hata analizi yapılıyor...")
            action_plan = self.pip_output_analyzer.analyze_output(output)
            suggestion = action_plan.get("suggestion", {})
            
            retries += 1

            if suggestion.get("action") == "retry":
                current_args.extend(suggestion.get("new_args", []))
                self.logger.log("info", f"Yeni deneme için argümanlar güncellendi: {current_args}")
            elif suggestion.get("action") == "install_new":
                new_deps = suggestion.get("packages", [])
                self.logger.log("info", f"Pip hatasından yeni bağımlılıklar bulundu: {new_deps}. Ana kuruluma devam etmeden önce bunlar denenecek.")
                new_dependencies_to_install.extend(new_deps)
                # Yeni bağımlılıkları hemen kuyruğa ekleyip bu paketi sonra tekrar denemek daha iyi olabilir.
                # Şimdilik listeyi döndürüyoruz.
                break # Bu paketin denemesini sonlandır, üst döngü yenileri eklesin.
            else: # "abort" veya bilinmeyen
                reason = suggestion.get("reason", "Bilinmeyen hata.")
                self.logger.log("error", f"'{package}' kurulumu kalıcı olarak başarısız oldu. Sebep: {reason}")
                self.summary_generator.add_module_status(pkg_name_only, "Başarısız", duration, error=output, reason=reason)
                raise PdsXInstallationError(f"'{package}' kurulumu başarısız oldu: {reason}", "E009", {"package": package, "pip_error": output})

        # Eğer döngü biterse ve hala başarılı olamadıysa
        self.logger.log("error", f"'{package}' maksimum deneme sayısına ulaştı ve kurulamadı.")
        self.summary_generator.add_module_status(pkg_name_only, "Başarısız", duration, error=output, reason="Maksimum deneme sayısına ulaşıldı")
        raise PdsXInstallationError(f"'{package}' maksimum deneme sayısına ulaştı ve kurulamadı.", "E011", {"package": package})


    def install_from_list(self, modules: List[str]):
        """Verilen bir listedeki tüm modülleri kurar."""
        self.logger.log("info", f"{len(modules)} modülden oluşan liste için toplu kurulum başlatılıyor.")
        
        # Kurulum sırasını optimize et
        optimized_modules = self.dependency_optimizer.get_optimized_order(modules)

        for module in optimized_modules:
            self.installation_queue.put(module)
        
        # Kurulum kuyruğunu işlemeyi tetikle
        self._process_installation_queue()
        self.logger.log("info", "Toplu kurulum listesi kuyruğa eklendi.")

    def install_from_file(self, file_path: str):
        """Verilen bir dosyadan (requirements.txt formatında) tüm modülleri kurar."""
        self.logger.log("info", f"'{file_path}' dosyasından toplu kurulum başlatılıyor.")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                modules = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            self.install_from_list(modules)
        except FileNotFoundError:
            self.logger.log("error", f"Kurulum dosyası bulunamadı: {file_path}")
            raise PdsXNotFoundError(f"Kurulum dosyası bulunamadı: {file_path}", "E012", {"path": file_path})
        except Exception as e:
            self.logger.log("error", f"Kurulum dosyası okunurken hata: {e}")
            raise PdsXException(f"Kurulum dosyası okunurken hata: {e}", "E013", {"path": file_path})

    def trigger_task(self, task: Dict[str, Any]):
        """PDS-X gibi harici sistemlerden gelen görevleri işlemek için programatik API."""
        task_name = task.get("task")
        payload = task.get("payload")
        self.logger.log("info", f"Programatik görev tetiklendi: {task_name}")

        if task_name == "install_list":
            if isinstance(payload, list):
                self.install_from_list(payload)
            else:
                self.logger.log("error", "install_list görevi için payload bir liste olmalıdır.")
        elif task_name == "install_file":
            if isinstance(payload, str):
                self.install_from_file(payload)
            else:
                self.logger.log("error", "install_file görevi için payload bir dosya yolu (string) olmalıdır.")
        else:
            self.logger.log("warning", f"Bilinmeyen görev adı: {task_name}")

    def import_and_install(self, module_name: str, package_name: str = None, version: str = None):
        """Modülü içe aktarır, yoksa kurar."""
        if not self.auto_install_enabled:
            return importlib.import_module(module_name)

        try:
            return importlib.import_module(module_name)
        except ImportError:
            self.logger.log("warning", f"Modül '{module_name}' bulunamadı. Otomatik kurulum denenecek.")
            self.auto_install_package(module_name, package_name, version)
            # Kurulumun bitmesini beklemek için basit bir mekanizma
            # Gelişmiş senaryolar için Future veya Event kullanılabilir
            # time.sleep(5) # Kurulumun bitmesi için bir süre bekle - KALDIRILDI
            self.wait_for_installs_to_complete()

            # Kuyruk işlendikten sonra tekrar dene
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pkg_to_install = self._resolve_package_name(module_name, package_name)
                self.logger.log("error", f"'{pkg_to_install}' kurulduktan sonra bile '{module_name}' modülü bulunamadı.")
                raise PdsXImportError(f"Modül '{module_name}' kurulumdan sonra bulunamadı.", "E005", {"module": module_name})

    def _process_installation_queue(self):
        """Kurulum kuyruğunu işler."""
        try:
            processed_in_this_run = set()
            while not self.installation_queue.empty():
                package = self.installation_queue.get()
                if package in processed_in_this_run:
                    self.installation_queue.task_done()
                    continue
                processed_in_this_run.add(package)

                try:
                    # install_package yerine _install_package_with_retry kullan
                    new_deps = self._install_package_with_retry(package)
                    if new_deps:
                        self.logger.log("info", f"'{package}' için yeni bağımlılıklar eklendi: {new_deps}")
                        for dep in new_deps:
                            if dep not in processed_in_this_run:
                                # Yeni bulunan bağımlılıkları kuyruğun başına ekle
                                # Bu, deque kullanılarak daha verimli hale getirilebilir
                                q_items = list(self.installation_queue.queue)
                                self.installation_queue = queue.Queue()
                                self.installation_queue.put(dep)
                                for item in q_items:
                                    self.installation_queue.put(item)

                except Exception as e:
                    self.logger.log("error", f"Kuyruk işlenirken '{package}' kurulamadı: {e}")
                    # Hatalı paketi tekrar denememek için işaretle
                    self.failed_packages[package.split("==")[0].split("[")[0]] = str(e)
                finally:
                    self.installation_queue.task_done()
        finally:
            # İşlem bittiğinde, olayı ayarla ve aktif thread sayısını düşür
            with self.lock:
                self.active_install_threads -= 1
                if self.installation_queue.empty():
                    self.installation_complete_event.set()


    def wait_for_installs_to_complete(self, timeout: Optional[float] = None):
        """Tüm kurulum işlemlerinin tamamlanmasını bekler."""
        self.logger.log("info", "Beklemedeki tüm kurulum görevlerinin tamamlanması bekleniyor...")
        self.installation_complete_event.wait(timeout)
        if not self.installation_complete_event.is_set():
            self.logger.log("warning", "Bekleme süresi aşıldı, ancak kurulumlar hala devam ediyor olabilir.")
        else:
            self.logger.log("info", "Tüm kurulum görevleri tamamlandı.")

    def _resolve_package_name(self, module_name: str, package_name: Optional[str] = None) -> str:
        """Modül adından paket adını çözer (alias'ları kullanarak)."""
        if package_name:
            return package_name
        # Alias kontrolü
        if module_name in self.aliases:
            return self.aliases[module_name]
        # Genel kural
        return module_name.replace('_', '-')

    def _prepare_aliases(self):
        """Kod içindeki ve harici dosyadan takma adları yükler."""
        # Kod içi varsayılanlar
        self.aliases = {
            "bs4": "beautifulsoup4",
            "cv2": "opencv-python",
            "skimage": "scikit-image",
            "sklearn": "scikit-learn",
            "yaml": "PyYAML",
        }
        # Harici dosyadan yükle
        if ALIAS_FILE.exists():
            try:
                with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and ":" in line:
                            alias, package = line.split(":", 1)
                            self.aliases[alias.strip()] = package.strip()
                self.logger.log("info", f"Harici alias dosyasından {len(self.aliases)} takma ad yüklendi.")
            except Exception as e:
                self.logger.log("error", f"Alias dosyası ({ALIAS_FILE}) okunurken hata: {e}")
        else:
            self.logger.log("info", "Harici alias dosyası bulunamadı, varsayılanlar kullanılıyor.")

    def shutdown(self):
        """Tüm arkaplan işlemlerini durdurur ve özeti yazdırır."""
        self.logger.log("info", "AutoImporter kapatılıyor...")
        self.shutdown_manager.shutdown()
        
        self.summary_generator.print_and_log_summary()
        
        # Tee nesnelerini kapatma işlemi artık shutdown_manager tarafından yönetiliyor.
        # if isinstance(sys.stdout, Tee):
        #     sys.stdout.close()
        # if isinstance(sys.stderr, Tee):
        #     sys.stderr.close()

# --- Ana Çalıştırma Bloğu ---
def main():
    parser = argparse.ArgumentParser(description="PDS-X Akıllı Modül Yükleyici")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=[e.name for e in OperatingMode], 
        default="NORMAL",
        help="Çalışma modunu ayarlar (NORMAL, SUPPRESSED, SILENT, TOTAL_SILENT)."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Terminal loglarını gerçek zamanlı olarak izleyerek otomatik kurulumu tetikler."
    )
    parser.add_argument(
        "--install-file",
        type=str,
        help="Belirtilen dosyadan (requirements.txt gibi) paketleri toplu olarak kurar."
    )
    parser.add_argument(
        "--install-list",
        nargs='+',
        help="Komut satırından verilen paket listesini toplu olarak kurar."
    )
    parser.add_argument(
        "module", 
        nargs='?', 
        help="İçe aktarılacak ve gerekirse kurulacak olan modülün adı."
    )

    args = parser.parse_args()

    # Tee nesnelerini başlatmadan önce log dosyasını temizle
    if TERMINAL_LOG_FILE.exists():
        try:
            # Log dosyasını yedeklemek yerine doğrudan temizleyebiliriz veya boyutuna göre karar verebiliriz.
            # Şimdilik basitlik adına içeriğini siliyoruz.
            open(TERMINAL_LOG_FILE, 'w').close()
        except Exception as e:
            print(f"[PDS-X] Uyarı: Terminal log dosyası temizlenemedi: {e}")

    # Tee nesnelerini başlat
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = Tee(original_stdout, TERMINAL_LOG_FILE)
    stderr_tee = Tee(original_stderr, TERMINAL_LOG_FILE)

    try:
        importer = AutoImporter(mode=args.mode, monitor_terminal=args.monitor)

        if args.install_file:
            importer.install_from_file(args.install_file)
        
        if args.install_list:
            importer.install_from_list(args.install_list)

        if args.module:
            importer.logger.log("info", f"'{args.module}' modülü için import ve kurulum süreci başlatılıyor...")
            try:
                module = importer.import_and_install(args.module)
                importer.logger.log("info", f"'{args.module}' başarıyla içe aktarıldı: {module}")
            except PdsXException as e:
                importer.logger.log("error", f"Modül işlenirken PDS-X hatası: {e.message}")

        # Eğer herhangi bir kurulum yapıldıysa, kuyruğun işlenmesini bekle
        # Bu basit bir bekleme mekanizmasıdır, daha gelişmiş bir yapı kurulabilir.
        if args.install_file or args.install_list or args.module:
            importer.wait_for_installs_to_complete()
            importer.logger.log("info", "Tüm görevler tamamlandı. Kapatılıyor...")

    except Exception as e:
        print(f"{colorama['Fore'].RED}Kritik bir hata oluştu: {e}{colorama['Style'].RESET_ALL}")
    finally:
        # AutoImporter'ın kapatma fonksiyonunu çağırarak kaynakları serbest bırak
        if 'importer' in locals() and importer:
            importer.shutdown()
        else:
            # Eğer importer hiç başlatılamadıysa, Tee'leri manuel kapat
            if isinstance(sys.stdout, Tee):
                sys.stdout.close()
            if isinstance(sys.stderr, Tee):
                sys.stderr.close()

if __name__ == "__main__":
    main()