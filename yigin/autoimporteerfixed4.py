# -*- coding: utf-8 -*-
"""
PDS-X Auto-Importer: Akýllý, Otomatik ve Kendi Kendini Onaran Paket Yöneticisi
Versiyon: 1.7.9.8
Tarih: 22 Haziran 2025
Yazar: xAI (Gemini ile oluþturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot-Geminni)
"""

import sys
import io

# Windows'ta konsol karakter kodlamasý sorunlarýný çözmek için stdout ve stderr'i yeniden yapýlandýr.
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        # Bu satýr, print() fonksiyonunun doðrudan UTF-8 kullanmasýný saðlar.
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (TypeError, ValueError) as e:
        # Bu hata, betik bir IDE içinde veya zaten uyumlu bir terminalde çalýþtýrýldýðýnda oluþabilir.
        # Bu durumda, genellikle ek bir yapýlandýrmaya gerek yoktur.
        print(f"[PDS-X UYARI] Standart akýþlar yeniden yapýlandýrýlamadý: {e}", file=sys.__stderr__)

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import re
import ast
import time
import threading
import subprocess
import importlib
import queue
import argparse
import io
import shutil
import asyncio
import random
import logging
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from statistics import stdev
from scipy.stats import f_oneway

# --- PDS-X Özel Hata Sýnýflarý ---
# exception_manager3.py dosyasýndan alýnmýþ gibi davranýyoruz.
class PdsXException(Exception):
    """PDS-X için temel hata sýnýfý."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"[PDS-X Error Code: {code}] {message}")
        self.message = message
        self.code = code
        self.context = context or {}

class PdsXImportError(PdsXException, ImportError):
    """Modül import edilirken oluþan hatalar için."""
    pass

class PdsXNotFoundError(PdsXException, FileNotFoundError):
    """Dosya veya kaynak bulunamadýðýnda."""
    pass

class PdsXConflictError(PdsXException):
    """Baðýmlýlýk çakýþmalarý için özel hata sýnýfý."""
    pass

class PdsXInstallationError(PdsXException):
    """Paket kurulumu sýrasýnda genel hata."""
    pass

# --- Lazy Loading Mekanizmasý ---
# Bu mekanizma, modülleri sadece ihtiyaç duyulduðunda yükleyerek baþlangýç süresini kýsaltýr.
psutil = None
numpy = None
packaging = None
sklearn_components = {}
keyboard = None
winreg = None
elasticsearch_client = None # Deðiþtirildi
colorama = {}
graphviz_digraph = None # Deðiþtirildi

def get_psutil():
    """psutil'i lazy loading ile yükle"""
    global psutil
    if psutil is None:
        try:
            import psutil as ps
            psutil = ps
        except ImportError:
            print("[PDS-X] UYARI: 'psutil' yüklenemedi. Kaynak izleme devre dýþý.")
    return psutil

def get_numpy():
    """NumPy'ý lazy loading ile yükle"""
    global numpy
    if numpy is None:
        try:
            import numpy as np
            numpy = np
        except ImportError:
            print("[PDS-X] UYARI: 'numpy' yüklenemedi. Bilimsel hesaplama özellikleri devre dýþý.")
    return numpy

def get_sklearn_components():
    """Sklearn bileþenlerini lazy loading ile yükle"""
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
            print("[PDS-X] UYARI: 'scikit-learn' yüklenemedi. Anomali tespiti devre dýþý.")
    return sklearn_components

def get_packaging_libs():
    """'packaging' kütüphanesi bileþenleri için lazy loader."""
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
            print("[PDS-X] UYARI: 'packaging' kütüphanesi bulunamadý. Geliþmiþ sürüm kontrolü devre dýþý.")
            packaging = {} # Boþ dictionary ata, böylece tekrar denemez
    return packaging

def get_elasticsearch_client():
    """Elasticsearch istemcisini lazy loading ile yükle."""
    global elasticsearch_client
    if elasticsearch_client is None:
        try:
            from elasticsearch import Elasticsearch
            elasticsearch_client = Elasticsearch
        except ImportError:
            print("[PDS-X] UYARI: 'elasticsearch' kütüphanesi bulunamadý. Elasticsearch entegrasyonu devre dýþý.")
            elasticsearch_client = False # Tekrar denenmemesi için False olarak iþaretle
    return elasticsearch_client

def get_graphviz_digraph():
    """Graphviz Digraph'ý lazy loading ile yükle."""
    global graphviz_digraph
    if graphviz_digraph is None:
        try:
            from graphviz import Digraph
            graphviz_digraph = Digraph
        except ImportError:
            print("[PDS-X] UYARI: 'graphviz' kütüphanesi bulunamadý. Baðýmlýlýk grafiði oluþturma devre dýþý.")
            graphviz_digraph = False # Tekrar denenmemesi için False olarak iþaretle
    return graphviz_digraph

# Diðer lazy loader'lar
try:
    import winreg
except ImportError:
    winreg = None

try:
    from colorama import Fore, Style
    colorama = {'Fore': Fore, 'Style': Style}
except ImportError:
    # colorama yoksa, renk kodlarý olmadan çalýþacak sahte nesneler oluþtur
    class DummyColor:
        def __getattr__(self, name):
            return ''
    colorama = {'Fore': DummyColor(), 'Style': DummyColor()}


# --- Temel Yapýlandýrma ve Sabitler ---
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
VENV_DIR = BASE_DIR / ".pdsx_isolated_env"
CACHE_DIR = BASE_DIR / ".pdsx_cache"
ALIAS_FILE = BASE_DIR / "pdsx_aliases.txt"
TERMINAL_LOG_FILE = LOG_DIR / "pdsX_terminal.log"
JSONL_LOG_FILE = LOG_DIR / "pdsx_terminal.jsonl"

# --- Çalýþma Modlarý ---
class OperatingMode(Enum):
    NORMAL = "NORMAL"
    SUPPRESSED = "SUPPRESSED"
    SILENT = "SILENT"
    TOTAL_SILENT = "TOTAL_SILENT"

class ModeManager:
    """Çalýþma modunu yönetir ve çýktýlarý buna göre kontrol eder."""
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

# --- Geliþmiþ Loglama Sistemi ---
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
                print(f"[PDS-X] Uyarý: Terminal log yedeði alýnamadý: {e}")

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
        
        # ModeManager'a göre handler'larý yönet
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
            self.original_stream.write(f"[PDS-X] HATA: Terminal log dosyasý açýlamadý: {log_file_path}\n{e}\n")
        
        # sys.stdout/stderr'i deðiþtir
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

# --- EKSÝK YARDIMCI SINIFLARIN TANIMLANMASI ---

class DependencyRegistry:
    """Kurulu paketlerin ve baðýmlýlýklarýnýn kaydýný tutar."""
    def __init__(self, logger: AdvancedLogger, file_path: Path = BASE_DIR / "dependencies.json"):
        self.logger = logger
        self.file_path = file_path
        self.registry = self._load()

    def _normalize_name(self, package_name: str) -> str:
        """Paket adýný PEP 503 normallerine göre standartlaþtýrýr (küçük harf, _ -> -)."""
        if not package_name:
            return ""
        return package_name.lower().replace("_", "-")

    def _load(self) -> Dict[str, Any]:
        """Kayýt dosyasýný yükler."""
        if not self.file_path.exists():
            self.logger.log("info", "Baðýmlýlýk kayýt dosyasý bulunamadý, yeni bir tane oluþturuluyor.")
            return {"version": "1.0", "packages": {}}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.log("error", f"Baðýmlýlýk kayýt dosyasý yüklenirken hata: {e}. Yeni bir kayýt oluþturuluyor.")
            return {"version": "1.0", "packages": {}}

    def save(self):
        """Mevcut kayýt durumunu dosyaya yazar."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4)
            self.logger.log("debug", f"Baðýmlýlýk kaydý þuraya kaydedildi: {self.file_path}")
        except IOError as e:
            self.logger.log("error", f"Baðýmlýlýk kayýt dosyasý kaydedilirken hata: {e}")

    def add_package(self, package_name: str, version: str, dependencies: List[str]):
        """Kayýtlara yeni bir paket ekler veya mevcut olaný günceller."""
        normalized_name = self._normalize_name(package_name)
        if not normalized_name:
            return

        package_data = self.registry["packages"].get(normalized_name, {})
        
        # Versiyon geçmiþini tut
        version_history = package_data.get("versions", [])
        if version not in version_history:
            version_history.append(version)

        self.registry["packages"][normalized_name] = {
            "current_version": version,
            "versions": version_history,
            "dependencies": [self._normalize_name(dep) for dep in dependencies],
            "last_installed_at": datetime.now().isoformat()
        }
        self.save()

    def get_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Normalleþtirilmiþ bir paket adý için bilgi döndürür."""
        normalized_name = self._normalize_name(package_name)
        return self.registry["packages"].get(normalized_name)

class PipOutputAnalyzer:
    """'pip' komutunun çýktýlarýný analiz ederek hatalarý anlar, baþarýlarý kaydeder ve çözüm önerileri sunar."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        # Baþarý desenleri
        self.success_dep_pattern = re.compile(r"Collecting\s+([a-zA-Z0-9_.-]+)")
        self.success_version_pattern = re.compile(r"Successfully installed\s+.*\s+([a-zA-Z0-9_.-]+)-([0-9a-zA-Z._+-]+)")

        # Geniþletilmiþ Hata Desenleri
        self.error_patterns = {
            # Kategori: Paket Bulunamadý
            "no_matching_distribution": re.compile(r"ERROR: Could not find a version that satisfies the requirement ([^\s(]+)"),
            
            # Kategori: Derleme Hatalarý (Build Failures)
            "metadata_failed": re.compile(r"error: metadata-generation-failed|ERROR: Could not build wheels for ([^,]+?)(?:, which is required to install pyproject.toml-based projects)?$", re.MULTILINE),
            "missing_visual_cpp": re.compile(r"Microsoft Visual C\+\+ 14\.0 or greater is required", re.IGNORECASE),
            "cl_exe_failed": re.compile(r"error: command 'cl\.exe' failed", re.IGNORECASE),
            "missing_rust_compiler": re.compile(r"Could not find `rustc` in `PATH`", re.IGNORECASE),

            # Kategori: Að ve Baðlantý Hatalarý
            "network_error": re.compile(r"WARNING: Retrying .* after connection broken by .*SSLError|ProxyError|ConnectTimeoutError", re.IGNORECASE),
            "http_error": re.compile(r"HTTP error 404|Could not fetch URL", re.IGNORECASE),

            # Kategori: Ýzin ve Ortam Hatalarý
            "permission_error": re.compile(r"ERROR: Could not install packages due to an OSError: \[Errno 13\] Permission denied", re.IGNORECASE),
            
            # Kategori: Baðýmlýlýk Çakýþmalarý
            "dependency_conflict": re.compile(r"ERROR: Cannot install .* because these package versions have conflicting dependencies."),
        }

    def analyze_success(self, output: str) -> Dict[str, Any]:
        """Baþarýlý bir pip çýktýsýný analiz eder ve kurulan paketler ile baðýmlýlýklarý döndürür."""
        dependencies = self.success_dep_pattern.findall(output)
        
        installed_packages = {}
        for match in self.success_version_pattern.finditer(output):
            installed_packages[match.group(1)] = match.group(2)

        if not installed_packages:
            self.logger.log("warning", "Pip çýktýsýnda 'Successfully installed' deseni bulunamadý.")
            return {"main_package": None, "version": None, "dependencies": []}

        main_package_name = list(installed_packages.keys())[-1]
        main_package_version = installed_packages[main_package_name]

        cleaned_deps = [dep for dep in dependencies if dep.lower().replace('_', '-') != main_package_name.lower().replace('_', '-')]

        return {
            "main_package": main_package_name,
            "version": main_package_version,
            "dependencies": cleaned_deps
        }

    def analyze_failure(self, output: str) -> Dict[str, Any]:
        """
        Baþarýsýz bir pip çýktýsýný analiz eder ve bir eylem planý döndürür.
        Eylem planý: { "suggestion": { "action": "retry|abort|install_new", "new_args": [], "packages": [], "reason": "..." } }
        """
        self.logger.log("debug", f"Pip hata çýktýsý analizi baþlatýldý. Çýktý:\n{output[:500]}...")

        # Ýzin Hatasý (En öncelikli kontrol)
        if self.error_patterns["permission_error"].search(output):
            return {"suggestion": {"action": "abort", "reason": "Permission denied. Betiði yönetici olarak çalýþtýrmayý veya sanal bir ortam kullanmayý deneyin."}}

        # Að Hatalarý
        if self.error_patterns["network_error"].search(output):
            return {"suggestion": {"action": "retry", "reason": "Að baðlantý sorunu tespit edildi.", "new_args": ["--timeout=100"]}}
        if self.error_patterns["http_error"].search(output):
            return {"suggestion": {"action": "retry", "reason": "HTTP hatasý (örn. 404 Not Found) tespit edildi. Paket adý veya versiyonu yanlýþ olabilir.", "new_args": ["--timeout=100"]}}

        # Derleme Hatalarý
        if self.error_patterns["missing_visual_cpp"].search(output) or self.error_patterns["cl_exe_failed"].search(output):
            return {"suggestion": {"action": "abort", "reason": "Microsoft C++ Build Tools gerekli. https://visualstudio.microsoft.com/visual-cpp-build-tools/ adresinden yükleyin."}}
        
        if self.error_patterns["missing_rust_compiler"].search(output):
            return {"suggestion": {"action": "abort", "reason": "Bu paket için Rust derleyicisi gerekli. https://www.rust-lang.org/tools/install adresinden yükleyin."}}

        match = self.error_patterns["metadata_failed"].search(output)
        if match:
            package = match.group(1).strip() if match.group(1) else "bir paket"
            return {"suggestion": {"action": "install_new", "packages": ["setuptools", "wheel", "cython"], "reason": f"'{package}' için wheel derlenemedi. Gerekli derleme araçlarý (setuptools, wheel) kurulacak."}}

        # Paket Bulunamadý Hatasý
        match = self.error_patterns["no_matching_distribution"].search(output)
        if match:
            package = match.group(1)
            return {"suggestion": {"action": "abort", "reason": f"'{package}' için uyumlu bir daðýtým bulunamadý. Paket adý, versiyonu veya Python uyumluluðunu kontrol edin."}}
            
        # Baðýmlýlýk Çakýþmasý
        if self.error_patterns["dependency_conflict"].search(output):
             return {"suggestion": {"action": "abort", "reason": "Baðýmlýlýk çakýþmasý tespit edildi. Manuel müdahale gerekli."}}

        # Varsayýlan Durum (Bilinmeyen Hata)
        return {"suggestion": {"action": "abort", "reason": "Bilinmeyen bir pip hatasý oluþtu. Detaylar için loglarý kontrol edin."}}

class ModuleSummaryGenerator:
    """Kurulum iþlemleri hakkýnda özet ve istatistikler oluþturur.""" 
    def __init__(self, logger: AdvancedLogger, mode_manager: ModeManager):
        self.logger = logger
        self.mode_manager = mode_manager
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_module_status(self, package: str, status: str, duration: float, error: Optional[str] = None, reason: Optional[str] = None):
        """Bir modülün kurulum sonucunu listeye ekler."""
        self.results.append({
            "package": package,
            "status": status,
            "duration": duration,
            "error": error,
            "reason": reason
        })

    def print_and_log_summary(self):
        """Tamamlanan tüm kurulumlar için son bir özet oluþturur ve yazdýrýr."""
        if self.mode_manager.is_silent:
            return

        total_duration = time.time() - self.start_time
        successful_installs = [r for r in self.results if r["status"] in ["Baþarýlý", "Atlandý"]]
        failed_installs = [r for r in self.results if r["status"] == "Baþarýsýz"]

        # Renkleri al (colorama yüklü deðilse boþ string döndür)
        Fore = colorama.get('Fore') if colorama else type('d', (object,), {'__getattr__': lambda s, n: ''})()
        Style = colorama.get('Style') if colorama else type('d', (object,), {'__getattr__': lambda s, n: ''})()

        summary = [
            "\n" + "="*25 + " PDS-X Kurulum Özeti " + "="*25,
            f"Toplam {len(self.results)} paket iþlendi. Toplam Süre: {total_duration:.2f} saniye.",
            f"{Fore.GREEN}Baþarýlý Kurulumlar ({len(successful_installs)}):{Style.RESET_ALL}",
            "  " + (', '.join([r['package'] for r in successful_installs]) or "Yok"),
            f"{Fore.RED}Baþarýsýz Kurulumlar ({len(failed_installs)}):{Style.RESET_ALL}",
        ]

        if failed_installs:
            for r in failed_installs:
                reason_text = f" - Sebep: {r.get('reason', 'Bilinmiyor')}"
                summary.append(f"  - {Fore.YELLOW}{r['package']}{Style.RESET_ALL}{reason_text}")
        else:
            summary.append("  Yok")

        summary.append("="*70 + "\n")
        
        final_summary_text = "\n".join(summary)
        # Bu loglama hem dosyaya (JSON olmayan formatta) hem de konsola yazdýrýr.
        self.logger.log("info", final_summary_text)


class ResourceMonitor:
    """Sistem kaynaklarýný (CPU, Bellek) izler."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.psutil = get_psutil()
        self.running = False
        self.thread = None
        if not self.psutil:
            self.logger.log("warning", "psutil modülü bulunamadý. Kaynak izleme devre dýþý.")

    def start(self):
        if not self.psutil or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        self.logger.log("info", "Kaynak izleyici baþlatýldý.")

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
                self.logger.log("debug", f"Kaynak Kullanýmý: CPU: {cpu_usage}%, Bellek: {memory_info.percent}%")
            except Exception as e:
                self.logger.log("error", f"Kaynak izleme sýrasýnda hata: {e}")
                self.running = False # Hata durumunda döngüyü sonlandýr

class AsyncDownloadManager:
    """Asenkron iþlemler için (örn. wheel indirme) bir havuz yönetir."""
    def __init__(self, importer_instance: 'AutoImporter', max_workers: int = 4):
        self.importer = importer_instance
        self.logger = self.importer.logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger.log("info", f"Asenkron indirme yöneticisi {max_workers} iþçi ile baþlatýldý.")

    def submit_download(self, package_spec: str):
        """Bir wheel indirme görevini havuza gönderir."""
        self.logger.log("info", f"'{package_spec}' için asenkron wheel indirme görevi gönderildi.")
        return self.executor.submit(self.importer.wheel_cache.download_wheel, package_spec)

class ScientificUtils:
    """Bilimsel ve aðýr hesaplama gerektiren görevler için ayrý bir iþlem havuzu yönetir."""
    def __init__(self, logger: AdvancedLogger, max_workers: int = 2):
        self.logger = logger
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        self.logger.log("info", f"Bilimsel hesaplama iþlem havuzu {max_workers} iþçi ile baþlatýldý.")

    def submit_task(self, func, *args, **kwargs):
        """Aðýr bir görevi iþlem havuzuna gönderir."""
        self.logger.log("debug", f"'{func.__name__}' görevi bilimsel iþlem havuzuna gönderildi.")
        return self.process_executor.submit(func, *args, **kwargs)

class EnvManager:
    """Python ortamýný ve pip/python yollarýný yönetir."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.python_path, self.pip_path = self._find_paths()
        self.logger.log("info", f"Python yolu: {self.python_path}")
        self.logger.log("info", f"Pip yolu: {self.pip_path}")

    def _find_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Mevcut Python ve pip yürütülebilir dosyalarýnýn yollarýný bulur."""
        python_path = Path(sys.executable)
        pip_path = python_path.parent / "pip.exe"
        if not pip_path.exists():
             pip_path = python_path.parent / "pip" # for non-windows
        
        if not python_path.exists():
            self.logger.log("error", "sys.executable yolu bulunamadý.")
            return None, None
        if not pip_path.exists():
            self.logger.log("warning", f"Pip yürütülebilir dosyasý beklenen yolda bulunamadý: {pip_path}")
            # Alternatif bulma mekanizmasý eklenebilir
            return python_path, None
            
        return python_path, pip_path

class WheelCacheManager:
    """Ýndirilen Python wheel dosyalarýný yönetir."""
    def __init__(self, cache_dir: Path, logger: AdvancedLogger, env_manager: EnvManager):
        self.cache_dir = cache_dir / "wheels"
        self.logger = logger
        self.env_manager = env_manager
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log("info", f"Wheel cache yöneticisi baþlatýldý. Cache dizini: {self.cache_dir}")

    def find_wheel(self, package_spec: str) -> Optional[Path]:
        """Verilen bir paket için cache'de uygun bir wheel dosyasý arar."""
        package_name = package_spec.split("==")[0].split("[")[0].replace("-", "_")
        for wheel_file in self.cache_dir.glob(f"{package_name}-*.whl"):
            # Burada daha karmaþýk bir sürüm kontrolü yapýlabilir, þimdilik ilk bulduðunu döndürür.
            self.logger.log("info", f"Cache'de '{package_spec}' için uygun wheel bulundu: {wheel_file.name}")
            return wheel_file
        self.logger.log("info", f"Cache'de '{package_spec}' için wheel bulunamadý.")
        return None

    def download_wheel(self, package_spec: str) -> Optional[Path]:
        """Bir paketi sadece wheel olarak indirir ve cache'e kaydeder."""
        self.logger.log("info", f"'{package_spec}' için wheel indirme iþlemi baþlatýlýyor...")
        cmd = [
            str(self.env_manager.pip_path),
            "wheel",
            "--no-deps",  # Sadece ana paketi indir, baðýmlýlýklarý deðil
            "-w", str(self.cache_dir),
            package_spec
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            self.logger.log("debug", f"pip wheel çýktýsý: {result.stdout}")
            # Ýndirilen dosyanýn adýný çýktýdan bulmak gerekir
            for line in result.stdout.splitlines():
                if "Successfully downloaded" in line:
                    # Bu çýktý formatý varsayýmsaldýr, pip versiyonuna göre deðiþebilir
                    downloaded_file_name = line.split(" ")[-1]
                    wheel_path = self.cache_dir / downloaded_file_name
                    if wheel_path.exists():
                        self.logger.log("info", f"'{package_spec}' baþarýyla indirildi ve cache'e eklendi: {wheel_path}")
                        return wheel_path
            # Eðer yukarýdaki mantýk çalýþmazsa, indirilen dosyayý manuel bul
            return self.find_wheel(package_spec)

        except subprocess.CalledProcessError as e:
            self.logger.log("error", f"'{package_spec}' wheel indirilirken hata oluþtu: {e.stderr}")
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
    """Acil durdurma mekanizmasý."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.is_active = False

    def activate(self):
        self.is_active = True
        self.logger.log("warning", "KILL SWITCH AKTÝF! Tüm operasyonlar durduruluyor.")

    def check(self):
        if self.is_active:
            raise InterruptedError("KillSwitch ile operasyon durduruldu.")

# --- YENÝ AKILLI YÖNETÝCÝLER ---

class PipOptimizer:
    """
    Pip komutlarýný optimize eder ve en iyi kurulum pratiklerini uygular.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.base_args = [
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--prefer-binary",
            "--upgrade-strategy", "eager"
        ]
        self.logger.log("info", f"Pip Optimizer baþlatýldý. Varsayýlan argümanlar: {self.base_args}")

    def get_base_install_args(self) -> List[str]:
        """
        Kurulum için temel, optimize edilmiþ pip argümanlarýný döndürür.
        """
        # Gelecekte buraya að durumu, sistem yükü gibi durumlara göre
        # argümanlarý dinamik olarak deðiþtirme mantýðý eklenebilir.
        return self.base_args.copy()

class HeuristicManager:
    """
    Kurulum sonuçlarýndan öðrenerek gelecekteki kararlarý iyileþtiren sezgisel yönetici.
    'learned_dependencies.json' dosyasýný yönetir.
    """
    def __init__(self, logger: AdvancedLogger, learned_deps_path: Path):
        self.logger = logger
        self.learned_deps_path = learned_deps_path
        self.learned_dependencies = self._load()

    def _load(self) -> Dict[str, List[str]]:
        """Öðrenilmiþ baðýmlýlýklarý dosyadan yükler."""
        if not self.learned_deps_path.exists():
            self.logger.log("info", "Öðrenilmiþ baðýmlýlýklar dosyasý bulunamadý, yeni oluþturulacak.")
            return {}
        try:
            with open(self.learned_deps_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Basit bir doðrulama
                if "dependencies" in data and isinstance(data["dependencies"], dict):
                    self.logger.log("info", "Öðrenilmiþ baðýmlýlýklar baþarýyla yüklendi.")
                    return data["dependencies"]
                else:
                    self.logger.log("warning", "Öðrenilmiþ baðýmlýlýklar dosyasý geçersiz formatta. Yeni oluþturulacak.")
                    return {}
        except (json.JSONDecodeError, IOError) as e:
            self.logger.log("error", f"Öðrenilmiþ baðýmlýlýk dosyasý okunurken hata: {e}. Yeni oluþturulacak.")
            return {}

    def _save(self):
        """Öðrenilmiþ baðýmlýlýklarý dosyaya kaydeder."""
        try:
            with open(self.learned_deps_path, "w", encoding="utf-8") as f:
                # Dosyayý daha okunabilir bir formatta kaydet
                json.dump({"version": "1.1", "dependencies": self.learned_dependencies}, f, indent=4)
            self.logger.log("debug", f"Öðrenilmiþ baðýmlýlýklar þuraya kaydedildi: {self.learned_deps_path}")
        except IOError as e:
            self.logger.log("error", f"Öðrenilmiþ baðýmlýlýklar kaydedilirken hata: {e}")

    def learn_from_success(self, package_name: str, dependencies: List[str]):
        """
        Baþarýlý bir kurulumdan sonra baðýmlýlýklarý öðrenir ve kaydeder.
        """
        # PEP 503 normalizasyonu
        normalized_package = package_name.lower().replace("_", "-")
        normalized_deps = sorted(list(set([d.lower().replace("_", "-") for d in dependencies])))

        # Eðer yeni bilgi, eskisinden farklýysa güncelle
        if self.learned_dependencies.get(normalized_package) != normalized_deps:
            self.logger.log("info", f"Yeni baðýmlýlýk bilgisi öðrenildi: '{normalized_package}' -> {normalized_deps}")
            self.learned_dependencies[normalized_package] = normalized_deps
            self._save()

    def get_learned_dependencies(self) -> Dict[str, List[str]]:
        """Tüm öðrenilmiþ baðýmlýlýklarý döndürür."""
        return self.learned_dependencies.copy()


# --- YARDIMCI YÖNETÝCÝ SINIFLARI (Referans Mimariden) ---

class GracefulShutdownManager:
    """Uygulamanýn düzgün bir þekilde sonlandýrýlmasýný yönetir."""
    def __init__(self, logger):
        self.logger = logger
        self.tasks = []
        self.shutdown_requested = threading.Event()

    def register(self, task_name, stop_function):
        """Durdurulacak bir görevi kaydeder."""
        self.tasks.append({'name': task_name, 'stop_func': stop_function})
        self.logger.log("debug", f"Kapatma görevi kaydedildi: {task_name}")

    def shutdown(self):
        """Tüm kayýtlý görevleri sýrayla durdurur."""
        if self.shutdown_requested.is_set():
            return
        self.logger.log("info", "Graceful shutdown baþlatýlýyor...")
        self.shutdown_requested.set()
        for task in reversed(self.tasks):
            try:
                self.logger.log("info", f"'{task['name']}' görevi durduruluyor...")
                task['stop_func']()
            except Exception as e:
                # Hata nesnesini string'e çevirerek JSON serileþtirme sorununu çöz
                error_message = f"'{task['name']}' görevi durdurulurken hata: {str(e)}"
                self.logger.log("error", error_message)
        self.logger.log("info", "Graceful shutdown tamamlandý.")

class DependencyOptimizer:
    """
    Kurulum sýrasýný optimize eder ve baðýmlýlýk grafiðini analiz eder.
    'learned_dependencies.json' dosyasýný kullanarak daha akýllý kararlar alabilir.
    """
    def __init__(self, logger: AdvancedLogger, dependency_registry: 'DependencyRegistry', heuristic_manager: 'HeuristicManager'):
        self.logger = logger
        self.dependency_registry = dependency_registry
        self.heuristic_manager = heuristic_manager
        self.graph = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Hem kayýtlý hem de öðrenilmiþ baðýmlýlýklardan birleþik bir graf oluþturur.
        """
        graph = {}
        # Öðrenilmiþ baðýmlýlýklarý HeuristicManager'dan al
        learned_deps = self.heuristic_manager.get_learned_dependencies()
        for package, deps in learned_deps.items():
            graph[package] = list(set(deps))

        # Kayýtlý (kurulu) baðýmlýlýklarý ekle/güncelle
        installed_packages = self.dependency_registry.registry.get("packages", {})
        for package, data in installed_packages.items():
            deps = data.get("dependencies", [])
            if package in graph:
                graph[package] = list(set(graph[package] + deps))
            else:
                graph[package] = list(set(deps))
        
        self.logger.log("debug", "Baðýmlýlýk grafiði oluþturuldu.")
        return graph

    def get_optimized_order(self, packages_to_install: List[str]) -> List[str]:
        """
        Verilen paket listesi ve bilinen baðýmlýlýklarý için topolojik sýralama kullanarak
        optimize edilmiþ tam bir kurulum sýrasý döndürür.
        """
        self.logger.log("info", f"Kurulum sýrasý optimizasyonu baþlatýldý for: {packages_to_install}")

        # 1. Adým: Kurulacak tüm paketleri ve bunlarýn bilinen tüm alt baðýmlýlýklarýný topla.
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

        # 2. Adým: Toplanan tüm paketler üzerinde topolojik sýralama yap.
        sorted_order = []
        visited = set()
        
        for pkg_name in list(all_packages_to_consider):
            if pkg_name not in visited:
                if not self._topological_sort_util(pkg_name, visited, set(), sorted_order):
                    self.logger.log("warning", "Baðýmlýlýk grafiðinde döngü tespit edildi! Optimizasyon atlanýyor.")
                    return packages_to_install

        # 3. Adým: Son listeyi oluþtur, orijinal versiyonlarý geri yükle.
        final_optimized_list = [original_package_map.get(p, p) for p in sorted_order]

        self.logger.log("info", f"Optimize edilmiþ ve geniþletilmiþ kurulum sýrasý: {final_optimized_list}")
        return final_optimized_list


    def _topological_sort_util(self, package: str, visited: set, recursion_stack: set, sorted_order: List[str]):
        """Topolojik sýralama için yardýmcý DFS fonksiyonu."""
        visited.add(package)
        recursion_stack.add(package)

        # Bu paketin baðýmlýlýklarýný gez
        dependencies = self.graph.get(package, [])
        for dep in dependencies:
            if dep not in visited:
                if not self._topological_sort_util(dep, visited, recursion_stack, sorted_order):
                    return False # Döngü tespit edildi
            elif dep in recursion_stack:
                return False # Döngü tespit edildi

        # Tüm baðýmlýlýklar gezildikten sonra paketi listeye ekle
        if package not in sorted_order:
             sorted_order.append(package)
        recursion_stack.remove(package)
        return True

class ConflictManager:
    """Baðýmlýlýk çakýþmalarýný yönetir."""
    def __init__(self, logger, dependency_registry):
        self.logger = logger
        self.dependency_registry = dependency_registry
        self.packaging_libs = get_packaging_libs()

    def check_conflicts(self, package_requirement: str) -> Optional[str]:
        """Bir paketin mevcut ortamla çakýþýp çakýþmadýðýný kontrol eder."""
        self.logger.log("info", f"'{package_requirement}' için çakýþma kontrolü yapýlýyor...")
        if not self.packaging_libs:
            self.logger.log("warning", "'packaging' kütüphanesi bulunamadýðý için çakýþma kontrolü atlanýyor.")
            return None
        
        try:
            Requirement = self.packaging_libs["Requirement"]
            req = Requirement(package_requirement)
            
            # Mevcut kurulu paketlerle karþýlaþtýr (basit kontrol)
            # Gerçek bir implementasyon için 'pip' komutunun çýktýsýný analiz etmek gerekir.
            installed_packages = self.dependency_registry.registry.get("packages", {})
            for name, data in installed_packages.items():
                if name == req.name:
                    # Versiyon çakýþmasý kontrolü
                    # Bu kýsým daha da geliþtirilebilir.
                    self.logger.log("warning", f"Potansiyel çakýþma: '{req.name}' zaten kurulu (versiyon: {data.get('version', 'bilinmiyor')}).")

        except Exception as e:
            self.logger.log("error", f"Geçersiz gereksinim dizesi: '{package_requirement}'. Hata: {e}")
            return f"Geçersiz gereksinim: {package_requirement}"
            
        self.logger.log("info", f"'{package_requirement}' için önemli bir çakýþma bulunamadý (basit kontrol).")
        return None

class CodeAnalyzer:
    """
    Kaynak kodunu statik olarak analiz ederek import ifadelerini ve potansiyel baðýmlýlýklarý bulur.
    Önceki ModuleAnalyzer'ýn yerini alýr ve AST (Abstract Syntax Tree) kullanýr.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def find_imports_from_source(self, file_path: Path) -> List[str]:
        """Bir Python kaynak dosyasýný okur ve içindeki tüm importlarý bulur."""
        if not file_path.exists() or not file_path.is_file():
            self.logger.log("warning", f"Kod analizi için kaynak dosya bulunamadý: {file_path}")
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
                        # 'import a.b.c' durumunda en üst seviye paket 'a'dýr.
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    # 'from a.b import c' durumunda, 'a'yý al.
                    # 'from . import x' gibi göreceli importlarý þimdilik atla (level > 0)
                    if node.level == 0 and node.module:
                        imports.add(node.module.split('.')[0])
            
            self.logger.log("info", f"'{file_path.name}' içinde bulunan modüller: {list(imports)}")
            return list(imports)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.log("error", f"'{file_path.name}' dosyasý analiz edilirken hata oluþtu: {e}")
            return []
        except Exception as e:
            self.logger.log("error", f"'{file_path.name}' dosyasý okunurken beklenmedik hata: {e}")
            return []


class TerminalLogAnalyzer:
    """Terminal log dosyasýný analiz ederek hatalarý ve eksik modülleri bulur."""
    def __init__(self, logger):
        self.logger = logger
        # Örnek: "ImportError: No module named 'requests'" veya "ModuleNotFoundError: No module named 'numpy'"
        self.import_error_pattern = re.compile(
            r"(?:ImportError|ModuleNotFoundError): No module named '([^']*)'"
        )

    def analyze_line(self, line: str) -> Optional[str]:
        """Tek bir log satýrýný analiz eder ve eksik modül adýný döndürür."""
        match = self.import_error_pattern.search(line)
        if match:
            missing_module = match.group(1)
            self.logger.log("info", f"Terminal logunda eksik modül tespit edildi: {missing_module}")
            return missing_module
        return None

class RealTimeLogMonitor:
    """Terminal log dosyasýný gerçek zamanlý olarak izler ve kurulumlarý tetikler."""
    def __init__(self, log_file: Path, analyzer: TerminalLogAnalyzer, importer: 'AutoImporter', logger: AdvancedLogger):
        self.log_file = log_file
        self.analyzer = analyzer
        self.importer = importer
        self.logger = logger
        self.running = False
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        """Ýzleme thread'ini baþlatýr."""
        if not self.log_file.exists():
            self.logger.log("warning", f"Ýzlenecek log dosyasý bulunamadý: {self.log_file}. Ýzleyici baþlatýlmýyor.")
            # Yine de dosyayý oluþturabiliriz.
            self.log_file.touch()
            
        self.running = True
        self.thread.start()
        self.logger.log("info", f"Gerçek zamanlý log izleyici '{self.log_file}' için baþlatýldý.")

    def stop(self):
        """Ýzleme thread'ini durdurur."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.log("info", "Gerçek zamanlý log izleyici durduruldu.")

    def _monitor(self):
        """Log dosyasýný izleyen ana döngü."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                f.seek(0, 2) # Dosyanýn sonuna git
                while self.running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5) # Yeni satýr yoksa bekle
                        continue
                    
                    missing_module = self.analyzer.analyze_line(line)
                    if missing_module:
                        self.logger.log("info", f"Ýzleyici, '{missing_module}' için otomatik kurulumu tetikliyor.")
                        # Doðrudan kurulum kuyruðuna ekle
                        self.importer.auto_install_package(missing_module)
        except FileNotFoundError:
            self.logger.log("warning", f"Ýzleme sýrasýnda log dosyasý kayboldu: {self.log_file}")
        except Exception as e:
            self.logger.log("error", f"Log izleme sýrasýnda kritik hata: {e}")


# --- Akýllý Kurulum Yöneticisi ---
class SmartInstallManager:
    """
    Kurulum sürecini baþtan sona yöneten, diðer akýllý bileþenleri (Optimizer,
    ConflictManager vb.) kullanarak optimize edilmiþ ve güvenli bir kurulum akýþý saðlayan sýnýf.
    """
    def __init__(self, packages_to_install: List[str], installer: 'AutoImporter'):
        """
        Args:
            packages_to_install: Kurulmasý istenen paketlerin listesi (örn: ["requests", "pandas==1.5.3"]).
            installer: Ana AutoImporter örneði.
        """
        self.packages_to_install = packages_to_install
        self.installer = installer
        self.logger = self.installer.logger
        self.dependency_optimizer = self.installer.dependency_optimizer
        self.conflict_manager = self.installer.conflict_manager
        self.final_install_order: List[str] = []
        self.failed_packages: List[Dict[str, Any]] = []

    def _prepare_installation_plan(self):
        """Kurulum planýný hazýrlar: sýralamayý optimize eder ve çakýþmalarý kontrol eder."""
        self.logger.log("info", "Kurulum planý hazýrlanýyor...")

        # 1. Adým: DependencyOptimizer kullanarak en iyi kurulum sýrasýný al.
        # Bu metod, hem istenen paketleri hem de onlarýn bilinen tüm alt baðýmlýlýklarýný içeren
        # tam ve optimize edilmiþ bir liste döndürür.
        self.final_install_order = self.dependency_optimizer.get_optimized_order(self.packages_to_install)
        self.logger.log("info", f"Optimize edilmiþ kurulum sýrasý: {self.final_install_order}")

        # 2. Adým: Kuruluma baþlamadan önce potansiyel çakýþmalarý kontrol et.
        # Bu, bariz sürüm çakýþmalarýný önceden tespit etmeye yardýmcý olabilir.
        for package_spec in self.final_install_order:
            conflict = self.conflict_manager.check_conflicts(package_spec)
            if conflict:
                self.logger.log("warning", f"Potansiyel çakýþma uyarýsý: {conflict}. Kurulum denenecek ancak sorun yaþanabilir.")
                # Daha katý bir modda burada kurulumu durdurabiliriz.
                # self.installer.kill_switch.activate()
                # return False
        return True

    def _execute_installation(self):
        """Hazýrlanan plana göre kurulumu gerçekleþtirir."""
        self.logger.log("info", "Optimize edilmiþ plana göre kurulum yürütülüyor...")

        for package_spec in self.final_install_order:
            try:
                self.installer.kill_switch.check() # Her paketten önce kill switch'i kontrol et
                
                # Ana installer'daki akýllý deneme mekanizmasýný kullanarak paketi kur.
                # Bu fonksiyon baþarý durumunda True, baþarýsýzlýkta False döner.
                success = self.installer._install_package_with_retry(package_spec)
                
                if not success:
                    self.logger.log("error", f"'{package_spec}' paketi, tüm denemelere raðmen kurulamadý.")
                    self.failed_packages.append({"package": package_spec, "reason": "Kurulum denemeleri baþarýsýz oldu."})

            except InterruptedError as e:
                self.logger.log("critical", f"Kurulum acil durum anahtarý ile durduruldu: {e}")
                self.failed_packages.append({"package": package_spec, "reason": "KillSwitch aktif edildi."})
                break # Döngüyü sonlandýr
            except Exception as e:
                self.logger.log("critical", f"'{package_spec}' kurulumu sýrasýnda beklenmedik kritik hata: {e}", exc_info=True)
                self.failed_packages.append({"package": package_spec, "reason": str(e)})
                # Kritik hatada devam edip etmemek bir stratejiye baðlanabilir. Þimdilik devam ediyor.

    def _verify_environment(self):
        """Kurulum sonrasý ortamýn saðlýðýný kontrol eder."""
        self.logger.log("info", "Kurulum sonrasý ortam saðlýðý kontrol ediliyor...")
        self.conflict_manager.check_and_log_conflicts()


    def run(self):
        """Akýllý kurulum sürecini baþtan sona çalýþtýrýr."""
        self.logger.log("info", "?? Akýllý Kurulum Yöneticisi baþlatýlýyor...")
        
        # 1. Planý oluþtur
        if not self._prepare_installation_plan():
            self.logger.log("critical", "Kurulum planý oluþturulamadýðý için iþlem iptal edildi.")
            return

        # 2. Planý uygula
        self._execute_installation()

        # 3. Ortamý doðrula
        self._verify_environment()

        # 4. Sonuçlarý raporla
        if self.failed_packages:
            self.logger.log("warning", f"Kurulum süreci tamamlandý ancak {len(self.failed_packages)} paket kurulamadý.")
            for failed in self.failed_packages:
                self.logger.log("warning", f"  - Baþarýsýz: {failed['package']} (Neden: {failed['reason']})")
        else:
            self.logger.log("info", "? Akýllý Kurulum Yöneticisi tüm görevleri baþarýyla tamamladý.")


# --- Ana AutoImporter Sýnýfý ---
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

        # Temel bileþenleri baþlat
        self.logger = AdvancedLogger()
        self.shutdown_manager = GracefulShutdownManager(self.logger)
        self.env_manager = EnvManager(logger=self.logger)
        self.dependency_registry = DependencyRegistry(self.logger)
        self.pip_output_analyzer = PipOutputAnalyzer(self.logger)
        self.wheel_cache = WheelCacheManager(CACHE_DIR, self.logger, self.env_manager)
        self.kill_switch = KillSwitch(self.logger)
        self.summary_generator = ModuleSummaryGenerator(self.logger, self.mode_manager)
        
        # Yeni akýllý yöneticiler
        self.pip_optimizer = PipOptimizer(self.logger)
        self.heuristic_manager = HeuristicManager(self.logger, BASE_DIR / "learned_dependencies.json")

        # Geliþmiþ ve analitik bileþenler
        self.dependency_optimizer = DependencyOptimizer(self.logger, self.dependency_registry, self.heuristic_manager)
        self.conflict_manager = ConflictManager(self.logger, self.dependency_registry)
        self.code_analyzer = CodeAnalyzer(self.logger)
        self.resource_monitor = ResourceMonitor(self.logger)
        self.async_download_manager = AsyncDownloadManager(self)
        self.scientific_utils = ScientificUtils(self.logger)

        # Gerçek zamanlý izleme bileþenleri
        self.terminal_log_analyzer = TerminalLogAnalyzer(self.logger)
        self.log_monitor = RealTimeLogMonitor(TERMINAL_LOG_FILE, self.terminal_log_analyzer, self, self.logger)

        # Durum nitelikleri
        self.lock = threading.Lock()
        self.installation_queue = queue.Queue()
        self.active_install_threads = 0
        self.installation_complete_event = threading.Event()
        self.installation_complete_event.set() # Baþlangýçta tamamlanmýþ durumda

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
            if isinstance(sys.stderr, Tee):
                sys.stderr.close()
        self.shutdown_manager.register("TeeStreamCloser", close_tee_streams)


        if monitor_terminal:
            self.log_monitor.start()

        self.logger.log("info", f"AutoImporter v1.7.9.8 baþlatýldý. Mod: {op_mode.name}, Terminal Ýzleme: {'Aktif' if monitor_terminal else 'Pasif'}")
        self.initialized = True

    def check_package_installed(self, package_name: str) -> bool:
        """Bir paketin mevcut ortamda kurulu olup olmadýðýný kontrol eder."""
        if not self.env_manager.pip_path:
            self.logger.log("warning", "Pip yolu bulunamadýðý için paket kontrolü yapýlamýyor.")
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
        Bir modül için kurulum sürecini baþlatan ana giriþ noktasý.
        Kurulumu kuyruða ekler ve iþlemeyi tetikler.
        """
        with self.lock:
            if not self.auto_install_enabled:
                self.logger.log("info", "Otomatik kurulum devre dýþý, iþlem atlanýyor.")
                return

            pkg_to_install = self._resolve_package_name(module_name, package_name)
            if version:
                pkg_to_install += f"=={version}"

            # Zaten kuyrukta veya hatalý olarak iþaretlenmiþ mi kontrol et
            if pkg_to_install in list(self.installation_queue.queue) or pkg_to_install in self.failed_packages:
                self.logger.log("debug", f"Paket '{pkg_to_install}' zaten kuyrukta veya hatalý listesinde.")
                return

            self.logger.log("info", f"Otomatik kurulum için kuyruða ekleniyor: {pkg_to_install}")
            self.installation_queue.put(pkg_to_install)
            self.installation_complete_event.clear() # Kurulum baþladýðýnda olayý temizle
        
        # Ayrý bir thread'de kuyruðu iþle, ana thread'i bloklama
        # Eðer zaten bir iþleyici çalýþmýyorsa yenisini baþlat
        with self.lock:
            if self.active_install_threads == 0:
                self.active_install_threads += 1
                threading.Thread(target=self._process_installation_queue).start()


    def install_package(self, package: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """
        Bir paketi kurar ve sonucu (baþarý, hata, çýktý) bir sözlük olarak döndürür.
        Bu fonksiyon doðrudan _install_package_with_retry tarafýndan kullanýlýr.
        """
        start_time = time.time()
        self.logger.log("info", f"'{package}' kurulumu baþlatýlýyor... Argümanlar: {extra_args}")
        
        try:
            self.kill_switch.check()

            # PipOptimizer'dan temel argümanlarý al
            cmd = [str(self.env_manager.pip_path), "install"]
            cmd.extend(self.pip_optimizer.get_base_install_args())

            if extra_args:
                cmd.extend(extra_args)
            cmd.append(package)

            self.logger.log("debug", f"Çalýþtýrýlacak pip komutu: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            duration = time.time() - start_time

            output = result.stdout + "\n" + result.stderr
            
            return {
                "returncode": result.returncode,
                "output": output,
                "duration": duration,
                "package": package
            }

        except Exception as e:
            duration = time.time() - start_time
            self.logger.log("error", f"'{package}' kurulumunda beklenmedik bir alt iþlem hatasý: {e}")
            return {
                "returncode": -1,
                "output": str(e),
                "duration": duration,
                "package": package,
                "exception": e
            }

    def _install_package_with_retry(self, package: str, max_retries: int = 3) -> List[str]:
        """
        Bir paketi akýllý deneme mekanizmasý ile kurar.
        Baþarýsýz olursa ve yeni baðýmlýlýklar tespit ederse, bu baðýmlýlýklarýn bir listesini döndürür.
        """
        with self.lock:
            pkg_name_only = package.split("==")[0].split("[")[0]
            if self.check_package_installed(pkg_name_only):
                self.logger.log("info", f"Paket '{pkg_name_only}' zaten kurulu, kurulum atlanýyor.")
                self.summary_generator.add_module_status(pkg_name_only, "Atlandý", 0, reason="Zaten yüklü")
                return []

        retries = 0
        current_args = []
        new_dependencies_to_install = []
        package_to_install = package # Baþlangýçta PyPI'dan alýnacak paket adý
        duration = 0 # duration'ý döngü dýþýnda tanýmla

        # 1. Adým: Cache'i kontrol et
        cached_wheel_path = self.wheel_cache.find_wheel(package)
        if cached_wheel_path:
            self.logger.log("info", f"'{package}' için cache kullanýlýyor: {cached_wheel_path}")
            package_to_install = str(cached_wheel_path)

        while retries < max_retries:
            self.logger.log("info", f"'{package_to_install}' kurulum denemesi {retries + 1}/{max_retries} (Argümanlar: {current_args})")
            
            install_result = self.install_package(package_to_install, extra_args=current_args)
            output = install_result["output"]
            duration = install_result["duration"]
            pkg_name_only = package.split("==")[0].split("[")[0]

            if install_result["returncode"] == 0:
                self.logger.log("info", f"'{package}' baþarýyla kuruldu.")
                
                # Baþarýlý kurulumu analiz et ve kayýtlarý güncelle
                success_details = self.pip_output_analyzer.analyze_success(output)
                main_pkg = success_details.get("main_package") or pkg_name_only
                version = success_details.get("version")
                
                # Derin Baðýmlýlýk Çözümlemesi: pip show ile gerçek baðýmlýlýklarý al
                deep_deps = self._get_deep_dependencies(main_pkg)

                if main_pkg and version:
                    self.dependency_registry.add_package(main_pkg, version, deep_deps)
                    # Baþarýlý kurulumdan öðren
                    self.heuristic_manager.learn_from_success(main_pkg, deep_deps)
                else:
                    # Analiz baþarýsýz olsa bile ana paketi kaydet
                    self.dependency_registry.add_package(pkg_name_only, "latest", deep_deps)
                    self.heuristic_manager.learn_from_success(pkg_name_only, deep_deps) # Baðýmlýlýk olmasa da öðren

                # Baþarýlý kurulumdan sonra, eðer cache'den kurulmadýysa, wheel'i cache'e ekle
                if not cached_wheel_path:
                    self.wheel_cache.download_wheel(package)

                reason = f"Düzeltildi ({current_args})" if retries > 0 else ""
                self.summary_generator.add_module_status(pkg_name_only, "Baþarýlý", duration, reason=reason)
                return new_dependencies_to_install # Baþarýlý, yeni baðýmlýlýk yoksa boþ liste döner

            # Kurulum baþarýsýz, analiz et
            self.logger.log("warning", f"'{package}' kurulumu baþarýsýz oldu. Hata analizi yapýlýyor...")
            failure_analysis = self.pip_output_analyzer.analyze_failure(output)
            suggestion = failure_analysis.get("suggestion", {})
            
            retries += 1

            if suggestion.get("action") == "retry":
                current_args.extend(suggestion.get("new_args", []))
                self.logger.log("info", f"Yeni deneme için argümanlar güncellendi: {current_args}")
                # Döngüye devam et
            elif suggestion.get("action") == "install_new":
                new_deps = suggestion.get("packages", [])
                self.logger.log("info", f"Pip hatasýndan yeni baðýmlýlýklar bulundu: {new_deps}. Ana kuruluma devam etmeden önce bunlar denenecek.")
                new_dependencies_to_install.extend(new_deps)
                # Bu paketin denemesini sonlandýr, üst döngü yenileri eklesin.
                return new_dependencies_to_install
            else: # "abort" veya bilinmeyen
                reason = suggestion.get("reason", "Bilinmiyor hata.")
                self.logger.log("error", f"'{package}' kurulumu kalýcý olarak baþarýsýz oldu. Sebep: {reason}")
                self.summary_generator.add_module_status(pkg_name_only, "Baþarýsýz", duration, error=output, reason=reason)
                raise PdsXInstallationError(f"'{package}' kurulumu baþarýsýz oldu: {reason}", "E009", {"package": package, "pip_error": output})

        # Eðer döngü biterse ve hala baþarýlý olamadýysa
        self.logger.log("error", f"'{package}' maksimum deneme sayýsýna ulaþtý ve kurulamadý.")
        self.summary_generator.add_module_status(pkg_name_only, "Baþarýsýz", duration, error=output, reason="Maksimum deneme sayýsýna ulaþýldý")
        raise PdsXInstallationError(f"'{package}' maksimum deneme sayýsýna ulaþtý ve kurulamadý.", "E011", {"package": package})

    def _get_deep_dependencies(self, package_name: str) -> List[str]:
        """
        'pip show' komutunu kullanarak bir paketin gerçek baðýmlýlýklarýný alýr.
        """
        if not self.env_manager.pip_path:
            self.logger.log("warning", "Pip yolu bulunamadýðý için derin baðýmlýlýk kontrolü yapýlamýyor.")
            return []
        try:
            # Paket adýndan versiyon ve ekstralarý temizle
            clean_package_name = package_name.split("==")[0].split("[")[0]
            cmd = [str(self.env_manager.pip_path), "show", clean_package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            dependencies = []
            for line in result.stdout.splitlines():
                if line.startswith("Requires:"):
                    # "Requires: numpy, pandas" -> ["numpy", "pandas"]
                    deps_str = line.replace("Requires:", "").strip()
                    if deps_str:
                        dependencies = [dep.strip() for dep in deps_str.split(",")]
                    break # Requires satýrýný bulduktan sonra döngüden çýk
            
            self.logger.log("info", f"'{clean_package_name}' için 'pip show' ile bulunan baðýmlýlýklar: {dependencies}")
            return dependencies
        except subprocess.CalledProcessError:
            self.logger.log("warning", f"'{clean_package_name}' için 'pip show' çalýþtýrýlamadý. Paket bulunamýyor veya baþka bir hata var.")
            return []
        except Exception as e:
            self.logger.log("error", f"'{clean_package_name}' için derin baðýmlýlýklar alýnýrken hata: {e}")
            return []

    def install_from_list(self, packages: List[str]):
        """
        Verilen bir listedeki tüm paketleri toplu olarak kurar.
        Akýllý Kurulum Yöneticisi'ni kullanýr.
        """
        self.logger.log("info", f"Listedeki {len(packages)} paket için toplu kurulum baþlatýlýyor.")
        if not packages:
            return

        self.logger.log("info", "Akýllý Kurulum Yöneticisi çalýþtýrýlýyor...")
        # SmartInstallManager, bu toplu kurulum iþi için özel olarak tasarlanmýþtýr.
        smart_manager = SmartInstallManager(packages_to_install=packages, installer=self)
        try:
            # run() tüm akýllý kurulum adýmlarýný (planlama, yürütme, doðrulama) yönetir.
            smart_manager.run()
        except Exception as e:
            self.logger.log("error", f"Akýllý Kurulum Yöneticisi çalýþýrken bir hata oluþtu: {e}")
        finally:
            # Her toplu kurulumdan sonra bir özet raporu oluþtur.
            self.summary_generator.print_and_log_summary()

    def install_from_file(self, file_path: str):
        """
        Bir dosyadan (requirements.txt gibi) paket listesini okur ve kurar.
        """
        self.logger.log("info", f"'{file_path}' dosyasýndan paketler okunuyor...")
        try:
            p = Path(file_path)
            if not p.is_file():
                raise PdsXNotFoundError(f"Gereksinim dosyasý bulunamadý veya bir dizin: {file_path}", "E013")
            
            with open(p, "r", encoding="utf-8") as f:
                # Yorumlarý (# ile baþlayan) ve boþ satýrlarý atla
                packages = [
                    line.strip() for line in f 
                    if line.strip() and not line.strip().startswith("#")
                ]
            
            if packages:
                self.logger.log("info", f"'{file_path}' dosyasýndan {len(packages)} paket bulundu. Kurulum baþlatýlýyor.")
                self.install_from_list(packages)
            else:
                self.logger.log("info", "Gereksinim dosyasýnda kurulacak paket bulunamadý.")

        except Exception as e:
            self.logger.log("error", f"'{file_path}' dosyasýndan kurulum yapýlýrken hata: {e}")

    def import_and_install(self, module_name: str, package_name: str = None, version: str = None):
        """
        Bir modülü içe aktarmayý dener, baþarýsýz olursa otomatik olarak kurar.
        Bu, PDS-X ana döngüsünden çaðrýlacak ana fonksiyondur.
        """
        try:
            # 1. Adým: Modülü doðrudan içe aktarmayý dene
            self.logger.log("info", f"Modül içe aktarýlýyor: {module_name}")
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.log("warning", f"Modül '{module_name}' bulunamadý. Hata: {e}. Otomatik kurulum denenecek.")
            
            if not self.auto_install_enabled:
                self.logger.log("warning", "Otomatik kurulum devre dýþý olduðundan modül yüklenemedi.")
                raise # Kurulum kapalýysa hatayý tekrar yükselt

            # 2. Adým: Kurulumu tetikle
            try:
                # auto_install_package kurulumu kuyruða ekler ve arka planda baþlatýr.
                self.auto_install_package(module_name, package_name, version)
                # Kurulumun tamamlanmasýný bekle
                self.wait_for_installs_to_complete()

                # 3. Adým: Kurulumdan sonra tekrar içe aktarmayý dene
                self.logger.log("info", f"Kurulum sonrasý '{module_name}' tekrar içe aktarýlýyor.")
                return importlib.import_module(module_name)
            except Exception as install_error:
                pkg_to_install = self._resolve_package_name(module_name, package_name)
                self.logger.log("error", f"'{module_name}' modülü kurulduktan sonra bile içe aktarýlamadý: {install_error}")
                raise PdsXImportError(
                    f"Modül '{module_name}' (paket: {pkg_to_install}) kurulmaya çalýþýldý ancak sonrasýnda içe aktarýlamadý.",
                    "E012",
                    {"module": module_name, "package": pkg_to_install, "original_error": str(install_error)}
                ) from install_error

    def trigger_task(self, task: Dict[str, Any]):
        """PDS-X gibi harici sistemlerden gelen görevleri iþlemek için programatik API."""
        task_name = task.get("task")
        payload = task.get("payload")
        self.logger.log("info", f"Harici görev alýndý: {task_name}")

        if task_name == "install_list":
            if isinstance(payload, list):
                self.install_from_list(payload)
            else:
                self.logger.log("error", f"install_list görevi için payload liste olmalý, alýndý: {type(payload)}")
        elif task_name == "install_file":
            if isinstance(payload, str):
                self.install_from_file(payload)
            else:
                self.logger.log("error", f"install_file görevi için payload dosya yolu (str) olmalý, alýndý: {type(payload)}")
        else:
            self.logger.log("warning", f"Bilinmeyen görev alýndý: {task_name}")

    def _process_installation_queue(self):
        """Kurulum kuyruðunu iþler."""
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
                        self.logger.log("info", f"'{package}' için yeni baðýmlýlýklar eklendi: {new_deps}")
                        for dep in new_deps:
                            if dep not in processed_in_this_run:
                                # Yeni bulunan baðýmlýlýklarý kuyruðun baþýna ekle
                                # Bu, deque kullanýlarak daha verimli hale getirilebilir
                                q_items = list(self.installation_queue.queue)
                                self.installation_queue = queue.Queue()
                                self.installation_queue.put(dep)
                                for item in q_items:
                                    self.installation_queue.put(item)

                except Exception as e:
                    self.logger.log("error", f"Kuyruk iþlenirken '{package}' kurulamadý: {e}")
                    # Hatalý paketi tekrar denemek için iþaretle
                    self.failed_packages[package.split("==")[0].split("[")[0]] = str(e)
                finally:
                    self.installation_queue.task_done()
        finally:
            # Ýþlem bittiðinde, olayý ayarla ve aktif thread sayýsýný düþür
            with self.lock:
                self.active_install_threads -= 1
                if self.installation_queue.empty():
                    self.installation_complete_event.set()


    def wait_for_installs_to_complete(self, timeout: Optional[float] = None):
        """Tüm kurulum iþlemlerinin tamamlanmasýný bekler."""
        self.logger.log("info", "Beklemedeki tüm kurulum görevlerinin tamamlanmasý bekleniyor...")
        self.installation_complete_event.wait(timeout)
        if not self.installation_complete_event.is_set():
            self.logger.log("warning", "Bekleme süresi aþýldý, ancak kurulumlar hala devam ediyor olabilir.")
        else:
            self.logger.log("info", "Tüm kurulum görevleri tamamlandý.")

    def _resolve_package_name(self, module_name: str, package_name: Optional[str] = None) -> str:
        """Modül adýndan paket adýný çözer (alias'larý kullanarak)."""
        if package_name:
            return package_name
        # Alias kontrolü
        if module_name in self.aliases:
            return self.aliases[module_name]
        # Genel kural
        return module_name.replace('_', '-')

    def _prepare_aliases(self):
        """Kod içindeki ve harici dosyadan takma adlarý yükler."""
        # Kod içi varsayýlanlar
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
                self.logger.log("info", f"Harici alias dosyasýndan {len(self.aliases)} takma ad yüklendi.")
            except Exception as e:
                self.logger.log("error", f"Alias dosyasý ({ALIAS_FILE}) okunurken hata: {e}")
        else:
            self.logger.log("info", "Harici alias dosyasý bulunamadý, varsayýlanlar kullanýlýyor.")

    def shutdown(self):
        """Tüm arkaplan iþlemlerini durdurur ve özeti yazdýrýr."""
        self.logger.log("info", "AutoImporter kapatýlýyor...")
        self.shutdown_manager.shutdown()
        
        self.summary_generator.print_and_log_summary()
        
        # Tee nesnelerini kapatma iþlemi artýk shutdown_manager tarafýndan yönetiliyor.
        # if isinstance(sys.stdout, Tee):
        #     sys.stdout.close()
        # if isinstance(sys.stderr, Tee):
        #     sys.stderr.close()

# --- Ana Çalýþtýrma Bloðu ---
def main():
    parser = argparse.ArgumentParser(description="PDS-X Akýllý Modül Yükleyici")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=[e.name for e in OperatingMode], 
        default="NORMAL",
        help="Çalýþma modunu ayarlar (NORMAL, SUPPRESSED, SILENT, TOTAL_SILENT)."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Terminal loglarýný gerçek zamanlý olarak izleyerek otomatik kurulumu tetikler."
    )
    parser.add_argument(
        "--install-file",
        type=str,
        help="Belirtilen dosyadan (requirements.txt gibi) paketleri toplu olarak kurar."
    )
    parser.add_argument(
        "--install-list",
        nargs='+',
        help="Komut satýrýndan verilen paket listesini toplu olarak kurar."
    )
    parser.add_argument(
        "module", 
        nargs='?', 
        help="Ýçe aktarýlacak ve gerekirse kurulacak olan modülün adý."
    )
     
    args = parser.parse_args()

    # Tee nesnelerini baþlatmadan önce log dosyasýný temizle
    if TERMINAL_LOG_FILE.exists():
        try:
            TERMINAL_LOG_FILE.write_text('')  # Log dosyasýný temizle
        except Exception as e:
            print(f"[PDS-X] Uyarý: Terminal log dosyasý temizlenemedi: {e}")

    # Tee nesnelerini baþlat
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = Tee(original_stdout, TERMINAL_LOG_FILE)
    stderr_tee = Tee(original_stderr, TERMINAL_LOG_FILE)

    try:
        # AutoImporter örneðini baþlat
        importer = AutoImporter(mode=args.mode, monitor_terminal=args.monitor)
        
        # Komut satýrý seçeneklerini iþle
        if args.install_file:
            importer.install_from_file(args.install_file)
        if args.install_list:
            importer.install_from_list(args.install_list)
        if args.module:
            importer.import_and_install(args.module)
        
        # Kurulumlar tamamlanana kadar bekle ve ardýndan kapat
        importer.wait_for_installs_to_complete()
        importer.shutdown()
        
    except Exception as e:
        print(f"{colorama['Fore'].RED}Kritik bir hata oluþtu: {e}{colorama['Style'].RESET_ALL}")
    finally:
        # AutoImporter'ýn kapatma fonksiyonunu çaðýrarak kaynaklarý serbest býrak
        if 'importer' in locals() and importer:
            importer.shutdown()
        else:
            # Eðer importer hiç baþlatýlamadýysa, Tee'leri manuel kapat
            if isinstance(sys.stdout, Tee):
                sys.stdout.close()
            if isinstance(sys.stderr, Tee):
                sys.stderr.close()

# Ana blok
if __name__ == '__main__':
    main()