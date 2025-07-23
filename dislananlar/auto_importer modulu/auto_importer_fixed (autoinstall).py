# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.2
# Date: June 22, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot)

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
import asyncio
from collections import deque

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
Elasticsearch = None
colorama = {}
Digraph = None

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

# Diğer lazy loader'lar
try:
    import winreg
except ImportError:
    winreg = None

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

try:
    from colorama import Fore, Style
    colorama = {'Fore': Fore, 'Style': Style}
except ImportError:
    # colorama yoksa, renk kodları olmadan çalışacak sahte nesneler oluştur
    class DummyColor:
        def __getattr__(self, name):
            return ''
    colorama = {'Fore': DummyColor(), 'Style': DummyColor()}

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None


# --- Sabitler ve Global Ayarlar ---
BASE_DIR = Path(__file__).resolve().parent
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
    """İndirilen wheel dosyalarını yönetir."""
    def __init__(self, cache_dir: Path, logger: AdvancedLogger):
        self.cache_dir = cache_dir
        self.logger = logger
        self.cache_dir.mkdir(exist_ok=True)

    def get_wheel_path(self, package_name: str, version: str) -> Optional[Path]:
        # Bu fonksiyon, belirli bir paket/versiyon için önbellekteki wheel dosyasını arar.
        # Gerçek implementasyon daha karmaşık olacaktır.
        return None

    def cache_installed_package(self, package_name: str):
        # Bu fonksiyon, yeni kurulan bir paketi wheel olarak önbelleğe alır.
        self.logger.log("info", f"'{package_name}' için önbelleğe alma (caching) özelliği henüz tam olarak implemente edilmedi.")
        pass

class ModuleSummaryGenerator:
    """Kurulum süreci sonunda bir özet oluşturur."""
    def __init__(self, logger: AdvancedLogger, mode_manager: ModeManager):
        self.logger = logger
        self.mode_manager = mode_manager
        self.summary: List[Dict[str, Any]] = []

    def add_module_status(self, name: str, status: str, duration: float, version: str = "N/A", source: str = "pip", reason: str = "", error: str = ""):
        self.summary.append({
            "module": name,
            "status": status,
            "duration_seconds": round(duration, 2),
            "version": version,
            "source": source,
            "reason": reason,
            "error": error
        })

    def print_and_log_summary(self):
        if not self.summary:
            return
        
        Fore = colorama['Fore']
        Style = colorama['Style']

        output = [f"\n{Fore.CYAN}--- PDS-X Kurulum Özeti ---{Style.RESET_ALL}"]
        for item in self.summary:
            color = Fore.GREEN if item['status'] == "Başarılı" else Fore.YELLOW if item['status'] == "Atlandı" else Fore.RED
            line = f"{color}[{item['status']:<10}] {item['module']:<25} (Süre: {item['duration_seconds']}s, Sürüm: {item['version']}) {Style.RESET_ALL}"
            if item['reason']:
                line += f" - {Fore.BLUE}{item['reason']}{Style.RESET_ALL}"
            if item['error']:
                 # Hata mesajını kısalt
                error_short = (item['error'][:100] + '...') if len(item['error']) > 100 else item['error']
                line += f"\n -> Hata: {Fore.RED}{error_short.strip()}{Style.RESET_ALL}"
            output.append(line)
        
        full_summary = "\n".join(output)
        self.logger.log("info", "Kurulum özeti oluşturuldu.")
        
        # ModeManager'a göre terminale yazdır
        if self.mode_manager.should_print_terminal():
            print(full_summary)
        
        # Log dosyasına her zaman (TOTAL_SILENT hariç) yaz
        if self.mode_manager.should_log_standard():
            # Renk kodlarını temizleyerek logla
            clean_summary = re.sub(r'\x1b\[[0-9;]*m', '', full_summary)
            self.logger.log("info", "\n--- Kurulum Özeti (Log) ---\n" + clean_summary)


# --- Bağımlılık Kayıt Sistemi ---
class DependencyRegistry:
    def __init__(self, logger: AdvancedLogger, registry_file: Path = CACHE_DIR / "dependencies.json"):
        self.logger = logger
        self.registry_file = registry_file
        self.registry = self.load_registry()

    def load_registry(self) -> Dict:
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {"packages": {}, "resolutions": {}}
        except Exception as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası yüklenemedi: {e}")
            return {"packages": {}, "resolutions": {}}

    def save_registry(self):
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası kaydedilemedi: {e}")

    def register_package(self, package: str, version: str, status: str, dependencies: List[str] = []):
        self.registry["packages"][package] = {
            "version": version,
            "status": status,
            "dependencies": dependencies,
            "timestamp": datetime.now().isoformat()
        }
        self.save_registry()
        self.logger.log("info", f"Paket '{package}=={version}' durumu '{status}' olarak kaydedildi.")

# --- Bağımlılık Çözümleyici ---
class DependencyResolver:
    def __init__(self, logger: AdvancedLogger, importer: 'AutoImporter', dependency_registry: DependencyRegistry):
        self.logger = logger
        self.importer = importer
        self.dependency_registry = dependency_registry

    def check_for_conflicts(self, package_requirement: str) -> Optional[str]:
        self.logger.log("info", f"'{package_requirement}' için çakışma kontrolü yapılıyor...")
        
        packaging_libs = get_packaging_libs()
        if not packaging_libs:
            self.logger.log("warning", "'packaging' kütüphanesi bulunamadığı için çakışma kontrolü atlanıyor.")
            return None

        try:
            Requirement = packaging_libs["Requirement"]
            Requirement(package_requirement) # Sadece geçerliliği kontrol et
        except Exception as e:
            self.logger.log("error", f"Geçersiz gereksinim dizesi: '{package_requirement}'. Hata: {e}")
            return f"Geçersiz gereksinim: {package_requirement}"

        # Bu fonksiyonun tam implementasyonu çok karmaşıktır.
        # Şimdilik temel bir kontrol yapıp geçiyoruz.
        # Gerçek bir çözümleyici `pip-tools` gibi kütüphaneler gerektirir.
        
        self.logger.log("info", f"'{package_requirement}' için önemli bir çakışma bulunamadı (basit kontrol).")
        return None

# --- Kaynak İzleme ve Analiz Araçları ---
class ResourceMonitor:
    def __init__(self, logger: AdvancedLogger, interval: int = 5):
        self.logger = logger
        self.running = False
        self.psutil = get_psutil()
        if self.psutil:
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
            self.logger.log("info", "Kaynak izleyici başlatıldı.")
        else:
            self.logger.log("warning", "psutil yüklenemediği için Kaynak İzleyici başlatılamadı.")

    def _monitor(self):
        while self.running:
            try:
                cpu = self.psutil.cpu_percent()
                mem = self.psutil.virtual_memory().percent
                load = self.psutil.disk_usage('/').percent
                if cpu > 90 or mem > 90 or load > 90:
                    self.logger.log("warning", f"Yüksek kaynak kullanımı: CPU {cpu}%, Bellek {mem}%, Disk {load}%")
            except Exception as e:
                self.logger.log("error", f"Kaynak izleme hatası: {e}")
            time.sleep(5)

    def stop(self):
        self.running = False

class PipOutputAnalyzer:
    """Pip komutunun çıktılarını analiz eder."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.dependency_pattern = re.compile(
            r"(?:Could not find a version that satisfies the requirement|No matching distribution found for) ([^\s(]+)"
        )

    def suggest_fix_args(self, error_output: str) -> List[str]:
        if "Microsoft Visual C++ 14.0 or greater is required" in error_output:
            self.logger.log("info", "Derleme araçları eksik. Wheel dosyası kullanımı öneriliyor.")
            return ["--only-binary=:all:"]
        if "failed with error code" in error_output:
            self.logger.log("info", "Derleme hatası. Önceden derlenmiş bir wheel kullanılması deneniyor.")
            return ["--only-binary=:all:"]
        return []

    def find_new_dependencies(self, output: str) -> List[str]:
        found_packages = []
        matches = self.dependency_pattern.findall(output)
        if not matches:
            return []

        self.logger.log("info", "Pip çıktısında potansiyel eksik bağımlılıklar bulundu.")
        for package_spec in matches:
            package_spec = package_spec.strip().replace(',', '')
            if package_spec and package_spec not in found_packages:
                self.logger.log("info", f"Yeni bağımlılık tespit edildi: {package_spec}")
                found_packages.append(package_spec)
        return found_packages

class AsyncDownloadManager:
    """Paketleri asenkron olarak indirmek ve kurmak için yönetici."""
    def __init__(self, auto_importer_instance, max_workers: int = 4):
        self.auto_importer = auto_importer_instance
        self.logger = auto_importer_instance.logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='PDSX_Downloader')
        self.loop = asyncio.get_event_loop()
        self.download_queue = asyncio.Queue()
        self.results = {}

    async def download_package(self, package_name: str):
        """Tek bir paketi indirmek için alt işlemi çalıştırır."""
        try:
            self.logger.log("info", f"Asenkron indirme başlatılıyor: {package_name}")
            # AutoImporter'daki install_package metodunu thread-safe bir şekilde çağır
            future = self.loop.run_in_executor(
                self.executor,
                self.auto_importer.install_package,
                package_name
            )
            result = await future
            self.results[package_name] = {"status": "Başarılı", "new_deps": result}
            self.logger.log("info", f"Asenkron indirme tamamlandı: {package_name}, Sonuç: {result}")
        except Exception as e:
            self.logger.log("error", f"Asenkron indirme sırasında hata: {package_name}, Hata: {e}")
            self.results[package_name] = {"status": "Hata", "error": str(e)}

    async def process_queue(self):
        """Sıradaki tüm paketleri indirir."""
        tasks = []
        while not self.download_queue.empty():
            package_name = await self.download_queue.get()
            tasks.append(self.download_package(package_name))
        await asyncio.gather(*tasks)
        return self.results

    def add_to_queue(self, package_name: str):
        self.download_queue.put_nowait(package_name)
        self.logger.log("info", f"Paket '{package_name}' indirme sırasına eklendi.")

class ScientificUtils:
    """Sistem metriklerini analiz etmek için bilimsel hesaplama araçları."""
    def __init__(self, logger):
        self.logger = logger
        self.process_executor = ProcessPoolExecutor(max_workers=2)

    def chaos_load_prediction(self) -> Dict:
        psutil = get_psutil()
        if not psutil:
            return {"status": "Bilinmiyor", "load_score": 0}
        
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        load_score = (cpu * 0.6) + (mem * 0.4)
        status = "Yüksek Yük" if load_score > 75 else "Orta Yük" if load_score > 50 else "Düşük Yük"
        return {"status": status, "load_score": load_score}

    def neural_load_balancer(self, system_metrics: Dict) -> str:
        """Sistem yüküne göre paralel veya sıralı indirme kararı verir."""
        load_score = system_metrics.get("load_score", 100)
        if load_score > 80:
            self.logger.log("info", "Nöral Yük Dengeleyici: Yüksek sistem yükü nedeniyle sıralı kurulum öneriliyor.")
            return "sequential"
        elif load_score > 50:
            self.logger.log("info", "Nöral Yük Dengeleyici: Orta sistem yükü nedeniyle sınırlı paralel kurulum öneriliyor.")
            return "limited_parallel"
        else:
            self.logger.log("info", "Nöral Yük Dengeleyici: Düşük sistem yükü, tam paralel kurulum uygun.")
            return "full_parallel"

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

class ModuleAnalyzer:
    """Modülleri ve bağımlılıklarını analiz eder."""
    def __init__(self, logger):
        self.logger = logger

    def get_module_dependencies(self, module_name: str) -> List[str]:
        """Bir modülün potansiyel bağımlılıklarını bulmaya çalışır (statik analiz)."""
        self.logger.log("info", f"'{module_name}' için bağımlılık analizi (henüz tam implemente edilmedi).")
        # Gelecekte 'importlib.metadata' veya AST analizi ile geliştirilebilir.
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
        self.wheel_cache = WheelCacheManager(CACHE_DIR, self.logger)
        self.kill_switch = KillSwitch(self.logger)
        self.summary_generator = ModuleSummaryGenerator(self.logger, self.mode_manager)
        
        # Gelişmiş ve analitik bileşenler
        self.conflict_manager = ConflictManager(self.logger, self.dependency_registry)
        self.module_analyzer = ModuleAnalyzer(self.logger)
        self.resource_monitor = ResourceMonitor(self.logger)
        self.async_download_manager = AsyncDownloadManager(self)
        self.scientific_utils = ScientificUtils(self.logger)

        # Gerçek zamanlı izleme bileşenleri
        self.terminal_log_analyzer = TerminalLogAnalyzer(self.logger)
        self.log_monitor = RealTimeLogMonitor(TERMINAL_LOG_FILE, self.terminal_log_analyzer, self, self.logger)

        # Durum nitelikleri
        self.lock = threading.Lock()
        self.installation_queue = queue.Queue()
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
            if pkg_to_install in self.installation_queue or pkg_to_install in self.failed_packages:
                self.logger.log("debug", f"Paket '{pkg_to_install}' zaten kuyrukta veya hatalı listesinde.")
                return

            self.logger.log("info", f"Otomatik kurulum için kuyruğa ekleniyor: {pkg_to_install}")
            self.installation_queue.put(pkg_to_install)
        
        # Ayrı bir thread'de kuyruğu işle, ana thread'i bloklama
        threading.Thread(target=self._process_installation_queue).start()


    def install_package(self, package: str) -> List[str]:
        """
        Bir paketi kurar. Başarısız olursa ve yeni bağımlılıklar tespit ederse,
        bu bağımlılıkların bir listesini döndürür. Başarılı olursa boş liste döndürür.
        """
        start_time = time.time()
        pkg_name = package.split("==")[0].split("[")[0]

        with self.lock:
            if self.check_package_installed(pkg_name):
                self.logger.log("info", f"Paket '{pkg_name}' zaten kurulu, kurulum atlanıyor.")
                self.summary_generator.add_module_status(pkg_name, "Atlandı", 0, reason="Zaten yüklü")
                return []

        try:
            self.logger.log("info", f"'{package}' kurulumu başlatılıyor.")
            self.kill_switch.check()

            conflict = self.conflict_manager.check_conflicts(package)
            if conflict:
                raise PdsXConflictError(f"'{package}' kurulumu çakışma nedeniyle durduruldu: {conflict}", "E008", {"package": package})

            cmd = [str(self.env_manager.pip_path), "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            duration = time.time() - start_time

            if result.returncode == 0:
                self.logger.log("info", f"'{package}' başarıyla kuruldu.")
                self.dependency_registry.register_package(pkg_name, "latest", "Başarılı")
                self.summary_generator.add_module_status(pkg_name, "Başarılı", duration)
                return []
            else:
                error_output = result.stderr or result.stdout
                self.logger.log("warning", f"'{package}' kurulumu başarısız oldu. Hata analizi yapılıyor...")
                
                new_deps = self.pip_output_analyzer.find_new_dependencies(error_output)
                if new_deps:
                    self.logger.log("info", f"Pip hatasından yeni bağımlılıklar bulundu: {new_deps}.")
                    return new_deps

                fix_args = self.pip_output_analyzer.suggest_fix_args(error_output)
                if fix_args:
                    self.logger.log("info", f"'{package}' için düzeltme argümanları ile yeniden denenecek: {fix_args}")
                    fix_cmd = [str(self.env_manager.pip_path), "install"] + fix_args + [package]
                    fix_result = subprocess.run(fix_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if fix_result.returncode == 0:
                        self.logger.log("info", f"'{package}' düzeltme sonrası başarıyla kuruldu.")
                        self.summary_generator.add_module_status(pkg_name, "Başarılı", duration, reason="Düzeltildi")
                        return []

                self.logger.log("error", f"'{package}' kurulumu kalıcı olarak başarısız oldu. Hata: {error_output}")
                self.summary_generator.add_module_status(pkg_name, "Başarısız", duration, error=error_output)
                raise PdsXInstallationError(f"'{package}' kurulumu başarısız oldu.", "E009", {"package": package, "pip_error": error_output})

        except (PdsXConflictError, PdsXInstallationError) as e:
            self.logger.log("error", f"'{package}' kurulumunda hata: {e.message}")
            self.summary_generator.add_module_status(pkg_name, "Başarısız", time.time() - start_time, error=e.message, reason=e.code)
            raise
        except Exception as e:
            self.logger.log("error", f"'{package}' kurulumunda beklenmedik hata: {e}")
            self.summary_generator.add_module_status(pkg_name, "Başarısız", time.time() - start_time, error=str(e))
            raise PdsXInstallationError(f"'{package}' kurulumunda beklenmedik hata: {e}", "E010", {"package": package}) from e

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
            time.sleep(5) # Kurulumun bitmesi için bir süre bekle

            # Kuyruk işlendikten sonra tekrar dene
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pkg_to_install = self._resolve_package_name(module_name, package_name)
                self.logger.log("error", f"'{pkg_to_install}' kurulduktan sonra bile '{module_name}' modülü bulunamadı.")
                raise PdsXImportError(f"Modül '{module_name}' kurulumdan sonra bulunamadı.", "E005", {"module": module_name})

    def _process_installation_queue(self):
        """Kurulum kuyruğunu işler."""
        processed_in_this_run = set()
        while not self.installation_queue.empty():
            package = self.installation_queue.get()
            if package in processed_in_this_run:
                continue
            processed_in_this_run.add(package)

            try:
                new_deps = self.install_package(package)
                if new_deps:
                    self.logger.log("info", f"'{package}' için yeni bağımlılıklar eklendi: {new_deps}")
                    for dep in new_deps:
                        if dep not in processed_in_this_run:
                            self.installation_queue.put(dep) # Önceliği yüksek
            except Exception as e:
                self.logger.log("error", f"Kuyruk işlenirken '{package}' kurulamadı: {e}")
                # Hatalı paketi tekrar denememek için işaretle
                self.failed_packages[package.split("==")[0].split("[")[0]] = str(e)


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
        
        # Tee nesnelerini kapat
        if isinstance(sys.stdout, Tee):
            sys.stdout.close()
        if isinstance(sys.stderr, Tee):
            sys.stderr.close()

# --- Ana Çalıştırma Bloğu ---
def main():
    parser = argparse.ArgumentParser(description="PDS-X Akıllı Modül Yükleyici")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=[e.name for e in OperatingMode], 
        default="NORMAL",
        help="Çıktı seviyesini belirler."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Terminal log dosyasını izleyerek otomatik kurulumu etkinleştirir."
    )
    parser.add_argument(
        "module",
        type=str,
        nargs="?",
        help="İçe aktarılacak ve gerekirse kurulacak modül."
    )
    args = parser.parse_args()

    importer = None
    # Tee nesnelerini en başta oluştur
    stdout_tee = Tee(sys.stdout, TERMINAL_LOG_FILE)
    stderr_tee = Tee(sys.stderr, TERMINAL_LOG_FILE)
    
    try:
        importer = AutoImporter(mode=args.mode, monitor_terminal=args.monitor)
        
        if args.module:
            importer.mode_manager.print(f"'{args.module}' modülü yükleniyor...", pdsx_prefix=True)
            module = importer.import_and_install(args.module)
            if module:
                importer.mode_manager.print(f"'{args.module}' başarıyla yüklendi ve içe aktarıldı.", pdsx_prefix=True)
                importer.mode_manager.print(f"Modül nesnesi: {module}", pdsx_prefix=True)
            else:
                importer.mode_manager.print(f"'{args.module}' yüklenemedi.", pdsx_prefix=True)
        else:
            importer.mode_manager.print("PDS-X Auto Importer başlatıldı. Test modülleri yükleniyor...", pdsx_prefix=True)
            importer.import_and_install("requests")
            importer.import_and_install("numpy")
            importer.import_and_install("non_existent_package_12345") # Hata senaryosu
            
            if args.monitor:
                importer.mode_manager.print("Terminal izleme modunda. Test için log dosyasına manuel olarak bir hata yazılıyor...", pdsx_prefix=True)
                try:
                    # Test: Log dosyasına manuel olarak bir hata yaz
                    with open(TERMINAL_LOG_FILE, "a", encoding="utf-8") as f:
                        f.write("\nTraceback (most recent call last):\n")
                        f.write("  File \"<stdin>\", line 1, in <module>\n")
                        f.write("ModuleNotFoundError: No module named 'matplotlib'\n")
                    importer.mode_manager.print("Test hatası ('matplotlib') log dosyasına yazıldı. Kurulumun başlaması bekleniyor...", pdsx_prefix=True)
                    importer.mode_manager.print("Çıkmak için 15 saniye bekleniyor...", pdsx_prefix=True)
                    time.sleep(15) # Kurulumun gerçekleşmesi için zaman tanıyın
                    importer.mode_manager.print("Test tamamlandı.", pdsx_prefix=True)
                except Exception as test_e:
                    importer.mode_manager.print(f"Test senaryosu sırasında bir hata oluştu: {test_e}", pdsx_prefix=True)


    except (KeyboardInterrupt, SystemExit):
        print("\nKullanıcı tarafından işlem kesildi. Kapatılıyor...")
    except Exception as e:
        print(f"Ana programda bir hata oluştu: {e}")
    finally:
        if importer:
            importer.shutdown()
        
        # Tee nesnelerini kapat
        stdout_tee.close()
        stderr_tee.close()


if __name__ == "__main__":
    main()