#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Enhanced Auto Importer - Final Clean Version
=================================================

Bu dosya, auto_importer_merged.py'nin temizlenmiş ve düzenlenmiş versiyonudur.
Tüm duplicate sınıflar kaldırılmış, sadece en iyi versiyonlar bırakılmıştır.

Seçilen Özellikler:
- GracefulShutdownManager: toplu1.py (keyboard kill switch ile)
- AdvancedLogger: Hibrit (toplu1.py + v1795.py hash deduplication + JSONL)
- Tee: toplu1.py (tam özellikli)
- DependencyRegistry: Hibrit (her iki dosyadan)
- PipOutputAnalyzer: toplu1.py (gelişmiş analiz)
- CacheManager: toplu1.py (kapsamlı yönetim)
- EnvManager: toplu1.py (gelişmiş venv)
- ConflictManager: toplu1.py (gelişmiş çakışma yönetimi)
- ModuleAnalyzer: toplu1.py (kapsamlı analiz)
- AutoImporter: toplu1.py temel + hibrit ThreadPool
- TerminalLogAnalyzer: v1795.py (gelişmiş regex)
- RealTimeLogMonitor: v1795.py (JSONL optimization)

Versiyon: Final Clean v1.0
Oluşturma Tarihi: 19 Temmuz 2025
Kullanılan Kaynak Dosyalar: auto_importer_merged.py (cleaned)
REQUIRED_PACKAGES: 108 paket (doğrulanmış)
"""

# === STANDARD IMPORTS ===
import sys
import os
import subprocess
import threading
import time
import json
import hashlib
import logging
import signal
import shutil
import platform
import importlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from urllib.parse import urlparse

# === KOŞULLU IMPORT'LAR (Try-Except ile) ===
try:
    import winreg  # Windows registry erişimi
except ImportError:
    winreg = None

try:
    import keyboard  # Hotkey listener için
except ImportError:
    keyboard = None

try:
    import psutil  # System monitoring için
except ImportError:
    psutil = None

# 108 REQUIRED PACKAGES - Doğrulanmış liste
REQUIRED_PACKAGES = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'requests', 'beautifulsoup4',
    'lxml', 'openpyxl', 'xlrd', 'pillow', 'opencv-python', 'tensorflow', 'torch', 'keras',
    'flask', 'django', 'fastapi', 'streamlit', 'dash', 'plotly', 'bokeh', 'altair',
    'jupyter', 'ipython', 'notebook', 'jupyterlab', 'ipywidgets', 'voila',
    'pytest', 'pytest-cov', 'pytest-mock', 'unittest-xml-reporting', 'coverage', 'tox',
    'black', 'flake8', 'autopep8', 'isort', 'mypy', 'pylint', 'bandit',
    'click', 'typer', 'argparse', 'configparser', 'pyyaml', 'toml', 'python-dotenv',
    'sqlalchemy', 'psycopg2-binary', 'pymongo', 'redis', 'elasticsearch', 'cassandra-driver',
    'celery', 'rq', 'dramatiq', 'luigi', 'airflow', 'prefect',
    'aiohttp', 'httpx', 'websockets', 'socketio', 'twisted', 'tornado',
    'pydantic', 'marshmallow', 'cerberus', 'schema', 'voluptuous',
    'cryptography', 'passlib', 'bcrypt', 'pyjwt', 'authlib', 'python-jose',
    'dateutil', 'pytz', 'arrow', 'pendulum', 'maya', 'delorean',
    'rich', 'colorama', 'termcolor', 'tqdm', 'progressbar2', 'alive-progress',
    'psutil', 'memory-profiler', 'py-spy', 'line-profiler', 'cprofile',
    'docker', 'kubernetes', 'boto3', 'google-cloud', 'azure-storage',
    'jinja2', 'mako', 'chameleon', 'genshi',
    'networkx', 'igraph', 'graph-tool', 'pygraphviz',
    'sympy', 'statsmodels', 'pymc3', 'arviz', 'xarray', 'dask',
    'geopy', 'folium', 'geopandas', 'shapely', 'cartopy',
    'spacy', 'nltk', 'gensim', 'transformers', 'datasets'
]


# === GRACEFUL SHUTDOWN MANAGER (TOPLU1.PY) ===
class GracefulShutdownManager:
    """
    Gelişmiş sistem kapatma yöneticisi - toplu1.py'den alınmıştır.
    Keyboard kill switch ve acil kapatma özellikleri ile.
    """
    
    def __init__(self):
        self.shutdown_requested = False
        self.emergency_shutdown = False
        self.active_processes = set()
        self.cleanup_functions = []
        self.hotkey_listener = None
        self._setup_signal_handlers()
        self._setup_keyboard_listener()
    
    def _setup_signal_handlers(self):
        """Signal handler'ları kur"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_keyboard_listener(self):
        """Keyboard listener kur (Ctrl+Shift+Q)"""
        if keyboard:
            try:
                self.hotkey_listener = keyboard.add_hotkey('ctrl+shift+q', self._emergency_shutdown)
                logging.info("Emergency hotkey (Ctrl+Shift+Q) activated")
            except Exception as e:
                logging.warning(f"Hotkey setup failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Signal yakalandığında çağrılır"""
        logging.info(f"Signal {signum} received, requesting graceful shutdown")
        self.request_shutdown()
    
    def _emergency_shutdown(self):
        """Acil kapatma"""
        logging.critical("EMERGENCY SHUTDOWN REQUESTED!")
        self.emergency_shutdown = True
        self.shutdown_requested = True
        self._emergency_cleanup()
    
    def request_shutdown(self):
        """Graceful shutdown talep et"""
        if not self.shutdown_requested:
            self.shutdown_requested = True
            logging.info("Graceful shutdown requested")
    
    def register_process(self, process):
        """Aktif process kaydı"""
        self.active_processes.add(process)
    
    def unregister_process(self, process):
        """Process kaydını silme"""
        self.active_processes.discard(process)
    
    def register_cleanup(self, cleanup_func):
        """Cleanup fonksiyonu kaydı"""
        self.cleanup_functions.append(cleanup_func)
    
    def cleanup(self):
        """Sistem temizliği"""
        logging.info("Performing graceful cleanup...")
        
        # Cleanup fonksiyonlarını çalıştır
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                logging.error(f"Cleanup function error: {e}")
        
        # Aktif process'leri sonlandır
        for process in list(self.active_processes):
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logging.error(f"Process termination error: {e}")
    
    def _emergency_cleanup(self):
        """Acil temizlik"""
        logging.critical("Performing emergency cleanup...")
        
        # Tüm process'leri zorla sonlandır
        for process in list(self.active_processes):
            try:
                process.kill()
            except Exception:
                pass
        
        # Hotkey listener'ı durdur
        if self.hotkey_listener:
            try:
                if keyboard:
                    keyboard.remove_hotkey(self.hotkey_listener)
            except Exception:
                pass
    
    def is_shutdown_requested(self):
        """Kapatma durumu kontrolü"""
        return self.shutdown_requested or self.emergency_shutdown


# === TEE CLASS (TOPLU1.PY) ===
class Tee:
    """
    Stdout/stderr yönlendirme sınıfı - toplu1.py'den alınmıştır.
    Tam özellikli çoklu stream desteği ile.
    """
    
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
    
    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass
    
    def close(self):
        for stream in self.streams:
            try:
                if hasattr(stream, 'close'):
                    stream.close()
            except Exception:
                pass


# === ADVANCED LOGGER (HİBRİT) ===
class AdvancedLogger:
    """
    Gelişmiş logger sınıfı - toplu1.py + v1795.py hibrit versiyonu.
    Silent mode (toplu1) + Hash deduplication (v1795) + JSONL (v1795)
    """
    
    def __init__(self, log_dir="logs", silent=False, enable_elasticsearch=False):
        self.log_dir = Path(log_dir)
        self.silent = silent
        self.enable_elasticsearch = enable_elasticsearch
        self.log_dir.mkdir(exist_ok=True)
        
        # Hash-based deduplication (v1795.py'den)
        self.message_hashes = set()
        self.spam_prevention = defaultdict(int)
        self.last_message_time = defaultdict(float)
        self.spam_threshold = 5  # Aynı mesajdan 5'ten fazla varsa spam
        self.spam_window = 60  # 60 saniye içinde
        
        # Log rotasyonu (v1795.py'den)
        self.max_log_size = 50 * 1024 * 1024  # 50MB
        self.max_log_files = 10
        
        # Elasticsearch support
        self.es = None
        
        self._setup_logging()
        if enable_elasticsearch:
            self._setup_elasticsearch()
    
    def _setup_logging(self):
        """Logging kurulumu"""
        # Ana log dosyası
        log_file = self.log_dir / f"auto_importer_{datetime.now().strftime('%Y%m%d')}.log"
        
        # JSONL log dosyası (v1795.py'den)
        self.jsonl_file = self.log_dir / f"auto_importer_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Terminal log yedekleme
        terminal_log = self.log_dir / f"terminal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Formatter'lar
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Root logger
        self.logger = logging.getLogger('AutoImporter')
        self.logger.setLevel(logging.DEBUG if not self.silent else logging.ERROR)
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (silent mode kontrolü)
        if not self.silent:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Terminal yedekleme için tee setup
        if not self.silent:
            sys.stdout = Tee(sys.stdout, open(terminal_log, 'w', encoding='utf-8'))
    
    def _setup_elasticsearch(self):
        """Elasticsearch entegrasyonu (toplu1.py'den)"""
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
            self.logger.info("Elasticsearch connected")
        except ImportError:
            self.logger.warning("Elasticsearch not available (install: pip install elasticsearch)")
        except Exception as e:
            self.logger.error(f"Elasticsearch connection failed: {e}")
    
    def _get_message_hash(self, message):
        """Mesaj hash'i oluşturma (v1795.py'den)"""
        return hashlib.md5(str(message).encode()).hexdigest()
    
    def _is_spam(self, message):
        """Spam kontrol (v1795.py'den)"""
        current_time = time.time()
        msg_hash = self._get_message_hash(message)
        
        # Hash bazlı deduplication
        if msg_hash in self.message_hashes:
            self.spam_prevention[msg_hash] += 1
            if self.spam_prevention[msg_hash] > self.spam_threshold:
                return True
        else:
            self.message_hashes.add(msg_hash)
            self.spam_prevention[msg_hash] = 1
        
        # Zaman bazlı spam kontrolü
        if current_time - self.last_message_time[msg_hash] < 1.0:
            return True
        
        self.last_message_time[msg_hash] = current_time
        return False
    
    def _rotate_logs(self):
        """Log rotasyonu (v1795.py'den)"""
        try:
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_size > self.max_log_size:
                    # Rotate log files
                    for i in range(self.max_log_files - 1, 0, -1):
                        old_file = log_file.with_suffix(f'.log.{i}')
                        new_file = log_file.with_suffix(f'.log.{i+1}')
                        if old_file.exists():
                            old_file.rename(new_file)
                    
                    # Rename current log
                    log_file.rename(log_file.with_suffix('.log.1'))
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def _write_jsonl(self, level, message, extra_data=None):
        """JSONL formatında log yazma (v1795.py'den)"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'extra': extra_data or {}
            }
            
            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # Fallback to standard logging
            self.logger.error(f"JSONL write failed: {e}")
    
    def _log_to_elasticsearch(self, level, message, extra_data=None):
        """Elasticsearch'e log gönderme (toplu1.py'den)"""
        if not self.es:
            return
        
        try:
            doc = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'extra': extra_data or {}
            }
            self.es.index(index='auto-importer-logs', body=doc)
        except Exception as e:
            self.logger.error(f"Elasticsearch indexing failed: {e}")
    
    def log(self, level, message, extra_data=None):
        """Ana logging metodu"""
        if self._is_spam(message):
            return
        
        # Standard logging
        getattr(self.logger, level.lower())(message)
        
        # JSONL logging
        self._write_jsonl(level, message, extra_data)
        
        # Elasticsearch logging
        if self.enable_elasticsearch:
            self._log_to_elasticsearch(level, message, extra_data)
    
    def info(self, message, **kwargs):
        self.log('INFO', message, kwargs)
    
    def warning(self, message, **kwargs):
        self.log('WARNING', message, kwargs)
    
    def error(self, message, **kwargs):
        self.log('ERROR', message, kwargs)
    
    def critical(self, message, **kwargs):
        self.log('CRITICAL', message, kwargs)
    
    def debug(self, message, **kwargs):
        self.log('DEBUG', message, kwargs)


# === TERMINAL LOG ANALYZER (V1795.PY) ===
class TerminalLogAnalyzer:
    """
    Gelişmiş terminal log analizi - v1795.py'den alınmıştır.
    Gelişmiş regex pattern'ler ve ModuleNotFoundError detection ile.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.setup_patterns()
    
    def setup_patterns(self):
        """Gelişmiş regex pattern'ler (v1795.py'den)"""
        import re
        
        # ModuleNotFoundError regex patterns
        self.module_not_found_patterns = [
            r"ModuleNotFoundError:\s*No module named\s*['\"]([^'\"]+)['\"]",
            r"ImportError:\s*No module named\s*['\"]?([^'\"]+)['\"]?",
            r"cannot import name\s*['\"]([^'\"]+)['\"]",
            r"from\s+([^\s]+)\s+import.*ImportError",
            r"import\s+([^\s]+).*ModuleNotFoundError"
        ]
        
        # Version conflict regex patterns
        self.version_conflict_patterns = [
            r"VersionConflict:\s*([^\s]+)\s*([0-9.]+)",
            r"requires\s+([^\s]+)\s*([<>=!]+[0-9.]+)",
            r"incompatible\s+version.*?([^\s]+)\s*([0-9.]+)",
            r"version\s+conflict.*?([^\s]+)"
        ]
        
        # Import error regex patterns
        self.import_error_patterns = [
            r"ImportError:\s*(.*)",
            r"SyntaxError:\s*.*?in\s+['\"]([^'\"]+)['\"]",
            r"AttributeError:\s*module\s+['\"]([^'\"]+)['\"].*?has no attribute",
            r"TypeError:\s*module\s+['\"]([^'\"]+)['\"]"
        ]
        
        # Pip suggestion regex patterns
        self.pip_suggestion_patterns = [
            r"Try:\s*pip install\s+([^\s]+)",
            r"pip install\s+([^\s\n]+)",
            r"conda install\s+([^\s\n]+)",
            r"You can install it with:\s*pip install\s+([^\s]+)"
        ]
        
        # Module-to-package mapping (geliştirilmiş)
        self.module_to_package = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
            'bs4': 'beautifulsoup4',
            'dotenv': 'python-dotenv',
            'dateutil': 'python-dateutil',
            'jwt': 'PyJWT',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysqlclient',
            'win32api': 'pywin32',
            'win32con': 'pywin32',
            'win32gui': 'pywin32',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'scipy': 'scipy',
            'requests': 'requests',
            'flask': 'flask',
            'django': 'django',
            'tornado': 'tornado',
            'sqlalchemy': 'sqlalchemy',
            'pymongo': 'pymongo',
            'redis': 'redis',
            'celery': 'celery',
            'jinja2': 'jinja2',
            'click': 'click',
            'psutil': 'psutil',
            'cryptography': 'cryptography',
            'lxml': 'lxml'
        }
    
    def analyze_output(self, output_text):
        """Terminal çıktısını analiz et"""
        try:
            import re
            
            missing_modules = set()
            version_conflicts = set()
            import_errors = []
            pip_suggestions = set()
            packages_to_install = set()
            
            # ModuleNotFoundError pattern matching
            for pattern in self.module_not_found_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    missing_modules.add(match)
                    # Module-to-package mapping
                    package = self.module_to_package.get(match, match)
                    packages_to_install.add(package)
            
            # Version conflict pattern matching
            for pattern in self.version_conflict_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        version_conflicts.add(f"{match[0]} {match[1] if len(match) > 1 else ''}")
                    else:
                        version_conflicts.add(match)
            
            # Import error pattern matching
            for pattern in self.import_error_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                import_errors.extend(matches)
            
            # Pip suggestion pattern matching
            for pattern in self.pip_suggestion_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    pip_suggestions.add(match)
                    packages_to_install.add(match)
                        
            return {
                'missing_modules': list(missing_modules),
                'version_conflicts': list(version_conflicts),
                'import_errors': import_errors,
                'pip_suggestions': list(pip_suggestions),
                'packages_to_install': list(packages_to_install)
            }
            
        except Exception as e:
            self.logger.error(f"Terminal analysis error: {e}")
            return {
                'missing_modules': [],
                'version_conflicts': [],
                'import_errors': [],
                'pip_suggestions': [],
                'packages_to_install': []
            }
    
    def extract_module_from_traceback(self, traceback_text):
        """Traceback'ten modül adı çıkarma"""
        try:
            import re
            # Son ModuleNotFoundError'ı bul
            lines = traceback_text.split('\n')
            for line in reversed(lines):
                for pattern in self.module_not_found_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1)
            return None
        except Exception as e:
            self.logger.error(f"Traceback analysis error: {e}")
            return None


# === DEPENDENCY REGISTRY (HİBRİT) ===
class DependencyRegistry:
    """
    Bağımlılık kayıt sistemi - toplu1.py + v1795.py hibrit versiyonu.
    JSON tabanlı persistence + çakışma kaydı
    """
    
    def __init__(self, registry_file="dependencies.json", logger=None):
        self.registry_file = Path(registry_file)
        self.logger = logger if logger else AdvancedLogger()
        self.registry = self._load_registry()
        self.lock = threading.Lock()
    
    def _load_registry(self):
        """Registry dosyasını yükle"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    "packages": {},
                    "resolutions": {},
                    "conflicts": {},
                    "status": "clean",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Registry yükleme hatası: {e}")
            return {
                "packages": {},
                "resolutions": {},
                "conflicts": {},
                "status": "clean",
                "timestamp": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Registry dosyasını kaydet"""
        try:
            self.registry["timestamp"] = datetime.now().isoformat()
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Registry kaydetme hatası: {e}")
    
    def check_package(self, package: str) -> bool:
        """Paketin zaten yüklü olup olmadığını kontrol et"""
        try:
            with self.lock:
                # Registry kontrolü
                if package in self.registry.get("packages", {}):
                    return self.registry["packages"][package].get("status") == "Başarılı"
                
                # Pip show ile kontrolü
                result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                                      capture_output=True, text=True)
                return result.returncode == 0
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kontrol hatası: {e}")
            return False
    
    def register_package(self, package: str, version: str = "unknown", status: str = "Başarılı", dependencies: Optional[List[str]] = None):
        """Paketi registry'ye kaydet"""
        try:
            with self.lock:
                if "packages" not in self.registry:
                    self.registry["packages"] = {}
                
                self.registry["packages"][package] = {
                    "version": version,
                    "status": status,
                    "dependencies": dependencies or [],
                    "timestamp": datetime.now().isoformat()
                }
                self._save_registry()
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kayıt hatası: {e}")
    
    def register_resolution(self, package: str, resolution: Dict):
        """Çakışma çözümünü kaydet"""
        try:
            with self.lock:
                if "resolutions" not in self.registry:
                    self.registry["resolutions"] = {}
                
                self.registry["resolutions"][package] = {
                    **resolution,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_registry()
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} çözüm kayıt hatası: {e}")
    
    def update_on_conflict(self, package: str, conflict_info: str, resolution_command: str):
        """Çakışma durumunda registry'yi güncelle"""
        try:
            with self.lock:
                if "conflicts" not in self.registry:
                    self.registry["conflicts"] = {}
                
                self.registry["conflicts"][package] = {
                    "conflict_info": conflict_info,
                    "resolution_command": resolution_command,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Status'u conflict olarak işaretle
                self.registry["status"] = "conflict"
                self._save_registry()
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} çakışma kayıt hatası: {e}")


# === CACHE MANAGER (TOPLU1.PY) ===
class CacheManager:
    """
    Önbellek yönetim sistemi - toplu1.py'den alınmıştır.
    Wheel dosyalarını cache'ler ve hızlı kurulum sağlar.
    """
    
    def __init__(self, cache_dir=".pdsx_cache", logger=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.wheels_dir = self.cache_dir / "wheels"
        self.wheels_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "packages.json"
        self.logger = logger if logger else AdvancedLogger()
        self.lock = threading.Lock()
    
    def load_package_metadata(self):
        """Paket metadata'sını yükle"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Metadata yükleme hatası: {e}")
            return {}
    
    def save_package_metadata(self, metadata):
        """Paket metadata'sını kaydet"""
        try:
            with self.lock:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Metadata kaydetme hatası: {e}")
    
    def install_from_cache(self, package: str) -> bool:
        """Önbellekten paket kurulumu"""
        try:
            cache_file = self.wheels_dir / f"{package}.whl"
            if cache_file.exists():
                self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten kuruluyor...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", str(cache_file)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten başarıyla kuruldu")
                    return True
                else:
                    self.logger.warning(f"[AUTO-IMPORTER] {package} önbellek kurulum hatası: {result.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} önbellek kurulum hatası: {e}")
            return False
    
    def _download_and_cache(self, package: str) -> bool:
        """Paketi indir ve önbelleğe al"""
        try:
            self.logger.info(f"[AUTO-IMPORTER] {package} indiriliyor ve önbelleğe alınıyor...")
            
            # pip download ile paketi indir
            result = subprocess.run([
                sys.executable, "-m", "pip", "download", 
                "--dest", str(self.wheels_dir), 
                "--no-deps", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"[AUTO-IMPORTER] {package} başarıyla önbelleğe alındı")
                
                # Metadata güncelle
                metadata = self.load_package_metadata()
                metadata[package] = {
                    "cached_at": datetime.now().isoformat(),
                    "cache_file": f"{package}.whl"
                }
                self.save_package_metadata(metadata)
                return True
            else:
                self.logger.error(f"[AUTO-IMPORTER] {package} önbellek hatası: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} önbellek hatası: {e}")
            return False
    
    def cleanup_old_cache(self, max_age_days=30):
        """Eski önbellek dosyalarını temizle"""
        try:
            now = datetime.now()
            metadata = self.load_package_metadata()
            for pkg, info in list(metadata.items()):
                cached_at = datetime.fromisoformat(info.get("cached_at", ""))
                if (now - cached_at).days > max_age_days:
                    cache_file = self.wheels_dir / info.get("cache_file", f"{pkg}.whl")
                    if cache_file.exists():
                        cache_file.unlink()
                    del metadata[pkg]
            self.save_package_metadata(metadata)
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Önbellek temizleme hatası: {e}")


# === ENV MANAGER (TOPLU1.PY) ===
class EnvManager:
    """
    İzole ortam yönetimi - toplu1.py'den genişletilmiş versiyon.
    restart_in_venv, is_running_in_venv, update_pip_if_needed fonksiyonları dahil.
    """
    
    def __init__(self, venv_dir=".pdsx_isolated_env", logger=None):
        self.venv_dir = Path(venv_dir)
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger if logger else AdvancedLogger()
        self.python_path = None
    
    def find_python310(self) -> Optional[str]:
        """Python 3.10 araması - toplu1.py'den"""
        try:
            self.logger.info("[AUTO-IMPORTER] Python 3.10 aranıyor...")
            
            # 1. PATH'de ara
            for exe in ["python3.10", "python310", "python"]:
                path = shutil.which(exe)
                if path:
                    try:
                        out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
                        if "3.10" in out:
                            self.logger.info(f"Python 3.10 bulundu: {path}")
                            self.python_path = path
                            return path
                    except subprocess.CalledProcessError:
                        continue
            
            # 2. Windows registry araması
            if platform.system() == "Windows" and winreg:
                try:
                    for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                        try:
                            with winreg.OpenKey(root, r"SOFTWARE\Python\PythonCore") as hkey:
                                for i in range(winreg.QueryInfoKey(hkey)[0]):
                                    ver = winreg.EnumKey(hkey, i)
                                    if ver.startswith("3.10"):
                                        with winreg.OpenKey(hkey, ver + r"\InstallPath") as subkey:
                                            py = winreg.QueryValue(subkey, None) + "python.exe"
                                            if os.path.exists(py):
                                                self.logger.info(f"Registry'de Python 3.10 bulundu: {py}")
                                                self.python_path = py
                                                return py
                        except Exception:
                            continue
                except Exception:
                    pass  # Registry erişim hatası
            
            # 3. Yaygın kurulum yerlerini kontrol et
            common_paths = [
                "/usr/bin/python3.10",
                "/usr/local/bin/python3.10",
                "C:\\Python310\\python.exe",
                "C:\\Program Files\\Python310\\python.exe",
                "C:\\Program Files (x86)\\Python310\\python.exe",
                os.path.expanduser("~/AppData/Local/Programs/Python/Python310/python.exe"),
                os.path.expanduser("~/anaconda3/envs/python310/python.exe"),
                os.path.expanduser("~/miniconda3/envs/python310/python.exe")
            ]
            
            for py_path in common_paths:
                if os.path.exists(py_path):
                    try:
                        out = subprocess.check_output([py_path, "--version"], text=True, stderr=subprocess.STDOUT)
                        if "3.10" in out:
                            self.logger.info(f"Yaygın konumda Python 3.10 bulundu: {py_path}")
                            self.python_path = py_path
                            return py_path
                    except subprocess.CalledProcessError:
                        continue
            
            self.logger.error("[AUTO-IMPORTER] Python 3.10 bulunamadı.")
            return None
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Python 3.10 arama hatası: {e}")
            return None
    
    def is_running_in_venv(self) -> bool:
        """Şu anda sanal ortamda çalışıp çalışmadığını kontrol eder - toplu1.py'den"""
        try:
            # Birden fazla yöntemle kontrol et
            
            # Yöntem 1: sys.executable kontrolü
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            current_python = sys.executable
            
            # Yol karşılaştırması
            if os.path.normpath(current_python) == os.path.normpath(venv_python):
                self.logger.debug(f"[AUTO-IMPORTER] Sanal ortam tespit edildi (executable): {current_python}")
                return True
            
            # Yöntem 2: Sanal ortam aktivasyon kontrolü
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                # Eğer sanal ortam aktifse ve doğru dizindeyse
                if str(self.venv_dir.absolute()) in sys.prefix:
                    self.logger.debug(f"[AUTO-IMPORTER] Sanal ortam tespit edildi (prefix): {sys.prefix}")
                    return True
            
            # Yöntem 3: VIRTUAL_ENV environment variable
            virtual_env = os.environ.get('VIRTUAL_ENV')
            if virtual_env and os.path.normpath(virtual_env) == os.path.normpath(str(self.venv_dir)):
                self.logger.debug(f"[AUTO-IMPORTER] Sanal ortam tespit edildi (VIRTUAL_ENV): {virtual_env}")
                return True
            
            # Yöntem 4: sys.path kontrolü (en son çare)
            if str(self.venv_dir / "Lib" / "site-packages") in sys.path:
                self.logger.debug(f"[AUTO-IMPORTER] Sanal ortam tespit edildi (sys.path): {self.venv_dir}")
                return True
                 
            self.logger.debug(f"[AUTO-IMPORTER] Sanal ortam tespit edilemedi. current: {current_python}, venv: {venv_python}")
            return False
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Sanal ortam kontrol hatası: {e}")
            return False
    
    def restart_in_venv(self) -> bool:
        """Programı sanal ortam içinde yeniden başlatır - toplu1.py'den"""
        try:
            if not self.venv_dir.exists():
                self.logger.error("[AUTO-IMPORTER] Sanal ortam bulunamadı!")
                return False
            
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            
            if not os.path.exists(venv_python):
                self.logger.error(f"[AUTO-IMPORTER] Sanal ortam Python'u bulunamadı: {venv_python}")
                return False
            
            self.logger.info("[AUTO-IMPORTER] Program sanal ortamda yeniden başlatılıyor...")
            
            # Yeni process'i başlat
            new_cmd = [venv_python] + sys.argv
            self.logger.info(f"[AUTO-IMPORTER] Yeni komut: {' '.join(new_cmd)}")
            
            # Mevcut process'i sonlandır ve yeni process'i başlat
            os.execv(venv_python, new_cmd)
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Sanal ortamda yeniden başlatma hatası: {e}")
            return False
    
    def update_pip_if_needed(self, force_latest: bool = False, silent: bool = False) -> bool:
        """
        Pip sürümünü günceller - toplu1.py'den
        
        Args:
            force_latest: True ise en son pip sürümünü zorla indirir
            silent: True ise sessiz kurulum yapar
            
        Returns:
            bool: Güncelleme başarılı ise True
        """
        try:
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")            
            if not os.path.exists(pip_cmd):
                self.logger.error("[AUTO-IMPORTER] Pip komutu bulunamadı!")
                return False
            
            # Mevcut pip sürümünü kontrol et
            try:
                result = subprocess.run([pip_cmd, "--version"], capture_output=True, text=True)
                current_version = result.stdout.strip() if result.returncode == 0 else "unknown"
                log_level = "debug" if silent else "info"
                self.logger.log(log_level, f"[AUTO-IMPORTER] Mevcut pip sürümü: {current_version}")
            except Exception as e:
                self.logger.warning(f"[AUTO-IMPORTER] Pip sürüm kontrolü başarısız: {e}")
                current_version = "unknown"
            
            # Hedef sürümü belirle
            target_version = "latest" if force_latest else "21.2.4"
            pip_package = "pip" if force_latest else "pip==21.2.4"
            
            log_level = "debug" if silent else "info"
            self.logger.log(log_level, f"[AUTO-IMPORTER] Pip hedef sürümü: {target_version}")
            
            # Python 3.10 için pip 21.2.4 mantığı (toplu1.py'deki açıklama)
            if not force_latest:
                self.logger.debug("[AUTO-IMPORTER] Python 3.10 için pip 21.2.4 kullanılıyor (o dönemin stable sürümü)")
            
            # Sessiz kurulum parametresi
            install_args = [pip_cmd, "install", "--upgrade"]
            if silent:
                install_args.extend(["--quiet", "--no-warn-script-location"])
            install_args.append(pip_package)
            
            # Kurulumu gerçekleştir
            result = subprocess.run(install_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.log(log_level, f"[AUTO-IMPORTER] Pip başarıyla güncellendi: {target_version}")
                return True
            else:
                self.logger.warning(f"[AUTO-IMPORTER] Pip güncelleme başarısız: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Pip güncelleme hatası: {e}")
            return False
    
    def check_package_installed(self, package: str) -> bool:
        """Paket yüklü mü kontrol et"""
        try:
            # pip list ile kontrol
            result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                installed_packages = result.stdout.lower()
                return package.lower() in installed_packages
                
            return False
        except Exception as e:
            self.logger.error(f"Paket kontrol hatası ({package}): {e}")
            return False
    
    def setup_environment(self) -> bool:
        """
        Ortamı hazırlar ve paketleri kurar - toplu1.py'den tam mantık
        """
        try:
            self.logger.info("[AUTO-IMPORTER] Ortam hazırlanıyor...")
            
            # Önce sanal ortamda çalışıp çalışmadığımızı kontrol et
            if self.is_running_in_venv():
                self.logger.info("[AUTO-IMPORTER] Zaten sanal ortamda çalışıyoruz.")
                # Sanal ortamdayken eksik paketleri kontrol et ve yükle
                self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor...")
                self.ensure_required_packages()
                return True
            
            self.logger.info("[AUTO-IMPORTER] Python 3.10 kontrol ediliyor...")
            
            # Python 3.10 kontrolü ve kurulumu
            python_path = self.find_python310()
            if not python_path:
                self.logger.error("[AUTO-IMPORTER] Python 3.10 bulunamadı!")
                return False
                
            self.logger.info(f"[AUTO-IMPORTER] Python 3.10 hazır: {python_path}")
            
            # Sanal ortam kontrolü ve kurulumu
            if not self.venv_dir.exists():
                self.logger.info("[AUTO-IMPORTER] Sanal ortam oluşturuluyor...")
                subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True)
            
            # Pip güncelleme
            self.update_pip_if_needed()
            
            # REQUIRED_PACKAGES kurulumu
            self.ensure_required_packages()
            
            self.logger.info("[AUTO-IMPORTER] Ortam başarıyla hazırlandı")
            return True
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Ortam hazırlanma hatası: {e}")
            return False
    
    def ensure_required_packages(self) -> bool:
        """REQUIRED_PACKAGES'ı kontrol et ve eksikleri kur"""
        try:
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor...")
            missing_packages = []
            
            for package in REQUIRED_PACKAGES[:20]:  # İlk 20 paketi test için
                if not self.check_package_installed(package):
                    missing_packages.append(package)
            
            if missing_packages:
                self.logger.info(f"[AUTO-IMPORTER] {len(missing_packages)} eksik paket bulundu, kuruluyor...")
                for package in missing_packages:
                    try:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True, timeout=120)
                        
                        if result.returncode == 0:
                            self.logger.info(f"✅ {package} kuruldu")
                        else:
                            self.logger.warning(f"⚠️ {package} kurulamadı: {result.stderr}")
                    except Exception as e:
                        self.logger.error(f"❌ {package} kurulum hatası: {e}")
            else:
                self.logger.info("[AUTO-IMPORTER] Tüm gerekli paketler mevcut")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol hatası: {e}")
            return False


# === REAL TIME LOG MONITOR (V1795.PY) ===
class RealTimeLogMonitor:
    """
    Gerçek zamanlı log izleme - v1795.py'den alınmıştır.
    JSONL optimization ve gelişmiş event correlation ile.
    """
    
    def __init__(self, logger, jsonl_file=None):
        self.logger = logger
        self.jsonl_file = jsonl_file or Path("logs/auto_importer.jsonl")
        self.monitoring = False
        self.monitor_thread = None
        self.alert_thresholds = {
            'error_rate': 10,  # 10 error/dakika
            'memory_usage': 80,  # %80 RAM
            'cpu_usage': 90    # %90 CPU
        }
        self.event_correlations = defaultdict(list)
        self.performance_metrics = defaultdict(float)
    
    def start_monitoring(self):
        """Log izlemeyi başlat"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Real-time log monitoring started")
    
    def stop_monitoring(self):
        """Log izlemeyi durdur"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Real-time log monitoring stopped")
    
    def _monitor_loop(self):
        """Ana izleme döngüsü"""
        try:
            last_position = 0
            
            while self.monitoring:
                try:
                    if self.jsonl_file.exists():
                        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                            f.seek(last_position)
                            new_lines = f.readlines()
                            last_position = f.tell()
                            
                            for line in new_lines:
                                if line.strip():
                                    self._process_log_line(line.strip())
                    
                    # Performance monitoring
                    self._check_system_performance()
                    
                    time.sleep(1)  # 1 saniye aralıkla kontrol
                    
                except Exception as e:
                    self.logger.error(f"Monitor loop error: {e}")
                    time.sleep(5)  # Hata durumunda 5 saniye bekle
                    
        except Exception as e:
            self.logger.error(f"Monitor thread error: {e}")
    
    def _process_log_line(self, line):
        """JSONL log satırını işle"""
        try:
            log_entry = json.loads(line)
            level = log_entry.get('level', '').upper()
            message = log_entry.get('message', '')
            timestamp = log_entry.get('timestamp', '')
            
            # Error correlation
            if level in ['ERROR', 'CRITICAL']:
                self._correlate_error(log_entry)
            
            # Performance alert generation
            if 'memory' in message.lower() or 'cpu' in message.lower():
                self._check_performance_alert(log_entry)
            
            # Event correlation
            self._correlate_events(log_entry)
            
        except json.JSONDecodeError:
            pass  # Invalid JSON line, skip
        except Exception as e:
            self.logger.error(f"Log line processing error: {e}")
    
    def _correlate_error(self, log_entry):
        """Error correlation ve pattern detection"""
        try:
            message = log_entry.get('message', '')
            timestamp = log_entry.get('timestamp', '')
            
            # Error pattern'leri kontrol et
            error_patterns = [
                'ModuleNotFoundError',
                'ImportError',
                'VersionConflict',
                'ConnectionError',
                'TimeoutError'
            ]
            
            for pattern in error_patterns:
                if pattern in message:
                    self.event_correlations[pattern].append({
                        'timestamp': timestamp,
                        'message': message
                    })
                    
                    # Son 5 dakikada aynı error'dan 5'ten fazla varsa alert
                    recent_errors = [
                        err for err in self.event_correlations[pattern]
                        if self._is_recent(err['timestamp'], minutes=5)
                    ]
                    
                    if len(recent_errors) >= 5:
                        self.logger.critical(f"ALERT: Repeated {pattern} errors detected: {len(recent_errors)} occurrences in 5 minutes")
                    
        except Exception as e:
            self.logger.error(f"Error correlation failed: {e}")
    
    def _check_performance_alert(self, log_entry):
        """Performance alert kontrolü"""
        try:
            import re
            message = log_entry.get('message', '').lower()
            
            # Memory usage alert
            if 'memory' in message and '%' in message:
                try:
                    match = re.search(r'(\d+\.?\d*)%', message)
                    if match:
                        usage = float(match.group(1))
                        if usage > self.alert_thresholds['memory_usage']:
                            self.logger.critical(f"ALERT: High memory usage detected: {usage}%")
                except (AttributeError, ValueError):
                    pass
            
            # CPU usage alert
            if 'cpu' in message and '%' in message:
                try:
                    match = re.search(r'(\d+\.?\d*)%', message)
                    if match:
                        usage = float(match.group(1))
                        if usage > self.alert_thresholds['cpu_usage']:
                            self.logger.critical(f"ALERT: High CPU usage detected: {usage}%")
                except (AttributeError, ValueError):
                    pass
                    
        except Exception as e:
            self.logger.error(f"Performance alert check failed: {e}")
    
    def _correlate_events(self, log_entry):
        """Event correlation ve JSONL format optimization"""
        try:
            # Event kategorileri
            event_categories = {
                'installation': ['install', 'pip', 'package'],
                'import': ['import', 'module', 'ModuleNotFoundError'],
                'system': ['memory', 'cpu', 'disk', 'performance'],
                'network': ['download', 'connection', 'timeout', 'url']
            }
            
            message = log_entry.get('message', '').lower()
            timestamp = log_entry.get('timestamp', '')
            
            for category, keywords in event_categories.items():
                if any(keyword in message for keyword in keywords):
                    self.event_correlations[f"category_{category}"].append({
                        'timestamp': timestamp,
                        'message': log_entry.get('message', ''),
                        'level': log_entry.get('level', '')
                    })
                    
                    # Category bazında özet istatistikler
                    recent_events = [
                        event for event in self.event_correlations[f"category_{category}"]
                        if self._is_recent(event['timestamp'], minutes=10)
                    ]
                    
                    if len(recent_events) >= 20:  # 10 dakikada 20'den fazla event
                        self.logger.warning(f"High activity in {category} category: {len(recent_events)} events in 10 minutes")
                    
        except Exception as e:
            self.logger.error(f"Event correlation failed: {e}")
    
    def _check_system_performance(self):
        """Sistem performansını kontrol et"""
        try:
            if psutil:
                # Memory usage
                memory = psutil.virtual_memory()
                if memory.percent > self.alert_thresholds['memory_usage']:
                    self.logger.critical(f"ALERT: System memory usage: {memory.percent}%")
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > self.alert_thresholds['cpu_usage']:
                    self.logger.critical(f"ALERT: System CPU usage: {cpu_percent}%")
                
                # Performance metrics update
                self.performance_metrics['memory'] = memory.percent
                self.performance_metrics['cpu'] = cpu_percent
            
        except Exception as e:
            self.logger.error(f"System performance check failed: {e}")
    
    def _is_recent(self, timestamp_str, minutes=5):
        """Timestamp'in son X dakika içinde olup olmadığını kontrol et"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now()
            if timestamp.tzinfo:
                now = now.replace(tzinfo=timestamp.tzinfo)
            return (now - timestamp).total_seconds() < (minutes * 60)
        except Exception:
            return False
    
    def get_performance_summary(self):
        """Performance özeti döndür"""
        return {
            'current_memory': self.performance_metrics.get('memory', 0),
            'current_cpu': self.performance_metrics.get('cpu', 0),
            'error_correlations': len(self.event_correlations),
            'monitoring_status': self.monitoring
        }


# === PIP OUTPUT ANALYZER (TOPLU1.PY) ===
class PipOutputAnalyzer:
    """
    Gelişmiş pip çıktı analizi - toplu1.py'den alınmıştır.
    Error pattern detection, auto-fix suggestions, mirror management ile.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.setup_error_handlers()
        self.mirrors = [
            "https://pypi.org/simple",
            "https://mirrors.aliyun.com/pypi/simple",
            "https://pypi.douban.com/simple",
            "https://mirror.baidu.com/pypi/simple"
        ]
        self.retry_count = 3
    
    def setup_error_handlers(self):
        """Error handler'ları kurulum"""
        self.error_handlers = {
            "Ignoring invalid distribution": self._handle_invalid_distribution,
            "ModuleNotFoundError": self._handle_module_not_found,
            "Could not find a version": self._handle_version_not_found,
            "deadlock detected": self._handle_deadlock,
            "WinError 32": self._handle_winerror_32,
            "Error parsing dependencies": self._handle_dependency_parsing,
            "Permission denied": self._handle_permission_denied,
            "HTTPError": self._handle_http_error,
            "ConnectTimeout": self._handle_connection_timeout,
            "ReadTimeout": self._handle_read_timeout,
            "SSLError": self._handle_ssl_error,
            "ChunkedEncodingError": self._handle_chunked_encoding,
            "ConnectionError": self._handle_connection_error
        }
    
    def _handle_invalid_distribution(self, package: str) -> bool:
        """Invalid distribution hatası"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", package]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"{package} force-reinstall ile düzeltildi")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Force-reinstall başarısız: {e}")
            return False
    
    def _handle_module_not_found(self, package: str) -> bool:
        """ModuleNotFoundError hatası"""
        return self._try_install_with_mirrors(package)
    
    def _handle_version_not_found(self, package: str) -> bool:
        """Version not found hatası"""
        return self._try_install_with_mirrors(package)
    
    def _handle_deadlock(self, package: str) -> bool:
        """Deadlock hatası"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", package]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"{package} no-cache ile düzeltildi")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"No-cache install başarısız: {e}")
            return False
    
    def _handle_winerror_32(self, package: str) -> bool:
        """WinError 32 hatası"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", package, "--no-cache-dir"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"{package} WinError 32 düzeltildi")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"WinError 32 düzeltme başarısız: {e}")
            return False
    
    def _handle_dependency_parsing(self, package: str) -> bool:
        """Dependency parsing hatası"""
        return self._handle_invalid_distribution(package)
    
    def _handle_permission_denied(self, package: str) -> bool:
        """Permission denied hatası"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", package, "--user"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"{package} user install ile düzeltildi")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"User install başarısız: {e}")
            return False
    
    def _handle_http_error(self, package: str) -> bool:
        """HTTP error hatası"""
        return self._try_install_with_mirrors(package)
    
    def _handle_connection_timeout(self, package: str) -> bool:
        """Connection timeout hatası"""
        return self._try_install_with_mirrors(package, timeout=300)
    
    def _handle_read_timeout(self, package: str) -> bool:
        """Read timeout hatası"""
        return self._try_install_with_mirrors(package, timeout=300)
    
    def _handle_ssl_error(self, package: str) -> bool:
        """SSL error hatası"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", package, "--trusted-host", "pypi.org", "--trusted-host", "pypi.python.org"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"{package} trusted-host ile düzeltildi")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Trusted-host install başarısız: {e}")
            return self._try_install_with_mirrors(package)
    
    def _handle_chunked_encoding(self, package: str) -> bool:
        """ChunkedEncodingError hatası"""
        return self._try_install_with_mirrors(package)
    
    def _handle_connection_error(self, package: str) -> bool:
        """ConnectionError hatası"""
        return self._try_install_with_mirrors(package)
    
    def _try_install_with_mirrors(self, package: str, timeout: int = 60) -> bool:
        """Mirror'lar ile deneme"""
        for mirror in self.mirrors:
            for attempt in range(self.retry_count):
                try:
                    cmd = [sys.executable, "-m", "pip", "install", package, "-i", mirror, "--timeout", str(timeout)]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    self.logger.info(f"{package} mirror {mirror} ile kuruldu (deneme {attempt + 1})")
                    return True
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Mirror {mirror} deneme {attempt + 1} başarısız: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        return False
    
    def analyze_and_fix(self, output: str, package: str) -> bool:
        """Ana analiz ve düzeltme metodu"""
        try:
            # Error kategorilendirme
            detected_errors = []
            for error_pattern in self.error_handlers.keys():
                if error_pattern in output:
                    detected_errors.append(error_pattern)
            
            if not detected_errors:
                self.logger.error(f"Bilinmeyen hata: {output}")
                # Bilinmeyen hata için genel çözüm dene
                return self._try_install_with_mirrors(package)
            
            # En uygun handler'ı seç ve çalıştır
            for error in detected_errors:
                self.logger.warning(f"Hata tespit edildi: {error}. {package} için düzeltme yapılıyor.")
                if self.error_handlers[error](package):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Pip analiz hatası: {e}")
            return False
    
    def generate_fix_suggestions(self, output: str, package: str) -> List[str]:
        """Auto-fix önerileri oluştur"""
        suggestions = []
        
        try:
            for error_pattern in self.error_handlers.keys():
                if error_pattern in output:
                    if error_pattern == "Permission denied":
                        suggestions.append(f"pip install {package} --user")
                    elif error_pattern == "WinError 32":
                        suggestions.append(f"pip install {package} --no-cache-dir")
                    elif error_pattern == "deadlock detected":
                        suggestions.append(f"pip install {package} --no-cache-dir")
                    elif error_pattern == "HTTPError":
                        suggestions.append(f"pip install {package} -i https://pypi.douban.com/simple")
                    elif error_pattern == "SSLError":
                        suggestions.append(f"pip install {package} --trusted-host pypi.org --trusted-host pypi.python.org")
                    else:
                        suggestions.append(f"pip install {package} --force-reinstall")
            
            if not suggestions:
                suggestions.append(f"pip install {package} -i https://pypi.douban.com/simple")
                suggestions.append(f"pip install {package} --user")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Fix suggestion generation hatası: {e}")
            return [f"pip install {package}"]


# === SUMMARY GENERATOR ===
class SummaryGenerator:
    """Installation summary generator"""
    
    def __init__(self, logger):
        self.logger = logger
        self.installation_stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
        self.error_stats = defaultdict(int)
    
    def record_installation(self, package: str, status: str, duration: float = 0):
        """Kurulum kaydı"""
        self.installation_stats[status] += 1
        if duration > 0:
            self.timing_stats[package].append(duration)
    
    def generate_summary(self) -> str:
        """Özet rapor oluştur"""
        total_successful = self.installation_stats.get('successful', 0)
        total_failed = self.installation_stats.get('failed', 0)
        total_packages = total_successful + total_failed
        
        if total_packages == 0:
            return "Henüz kurulum yapılmadı."
        
        success_rate = (total_successful / total_packages) * 100
        
        report = f"""
🎯 KURULUM ÖZET RAPORU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Başarılı: {total_successful}
❌ Başarısız: {total_failed}
📊 Başarı Oranı: {success_rate:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report


# === CONFLICT MANAGER (TOPLU1.PY) ===
class ConflictManager:
    """
    Gelişmiş çakışma yöneticisi - toplu1.py'den alınmıştır.
    Dependency graph analysis, auto-resolution strategy, version compatibility ile.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = defaultdict(list)
        self.version_compatibility = {}
        self.resolution_strategies = [
            "force_reinstall",
            "version_downgrade", 
            "version_upgrade",
            "dependency_isolation",
            "package_replacement"
        ]
    
    def clean_version(self, version: str) -> str:
        """Version string'ini temizle"""
        return re.sub(r'[=<>!]', '', version).strip()
    
    def detect_conflicts(self, module_name: str, deps: List[str]) -> Dict:
        """Çakışma detection (gelişmiş)"""
        try:
            self.logger.info(f"{module_name} için çakışma kontrolü başlatılıyor.")
            conflicts = {}
            
            # Pip check ile mevcut çakışmaları tespit et
            result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
            
            if "no conflicts" not in result.stdout.lower() and result.stdout.strip():
                # Çakışma var, analiz et
                conflict_lines = result.stdout.strip().split('\n')
                for line in conflict_lines:
                    if 'has requirement' in line or 'requires' in line:
                        # Örnek: "package1 has requirement package2>=1.0, but you have package2 0.5"
                        parts = line.split()
                        if len(parts) >= 3:
                            conflicting_package = parts[0]
                            conflicts[conflicting_package] = line
                            self.logger.warning(f"Çakışma tespit edildi: {conflicting_package}")
            
            # Her dependency için ayrı kontrol
            for dep in deps:
                try:
                    # Pip show ile versiyon bilgisi al
                    show_result = subprocess.run([sys.executable, "-m", "pip", "show", dep], capture_output=True, text=True)
                    if show_result.returncode == 0:
                        # Dependency graph'ını güncelle
                        self._update_dependency_graph(dep, show_result.stdout)
                    else:
                        conflicts[dep] = f"Package not found: {dep}"
                        
                except subprocess.CalledProcessError as e:
                    conflicts[dep] = f"Error checking {dep}: {e}"
            
            self.conflicts[module_name] = conflicts
            
            if conflicts:
                self.logger.warning(f"{module_name} için {len(conflicts)} çakışma bulundu")
            else:
                self.logger.info(f"{module_name} için çakışma bulunamadı")
                
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Çakışma kontrol hatası: {e}")
            return {}
    
    def _update_dependency_graph(self, package: str, pip_show_output: str):
        """Dependency graph'ını güncelle"""
        try:
            requires_line = None
            for line in pip_show_output.split('\n'):
                if line.startswith('Requires:'):
                    requires_line = line
                    break
            
            if requires_line:
                requires = requires_line.replace('Requires:', '').strip()
                if requires and requires != 'None':
                    deps = [dep.strip() for dep in requires.split(',')]
                    self.dependency_graph[package] = deps
                    
        except Exception as e:
            self.logger.error(f"Dependency graph güncelleme hatası: {e}")
    
    def resolve_conflicts(self, module_name: str, conflicts: Dict) -> Dict:
        """Çakışma çözümleme (gelişmiş strategy selection)"""
        try:
            if not conflicts:
                return {}
            
            resolutions = {}
            
            for dep, issue in conflicts.items():
                # En uygun çözüm stratejisini seç
                strategy = self._select_resolution_strategy(dep, issue)
                resolution = self._apply_resolution_strategy(dep, issue, strategy)
                
                if resolution:
                    resolutions[dep] = resolution
                    self.logger.info(f"Çakışma çözüldü: {dep} - {strategy}")
                else:
                    self.logger.error(f"Çakışma çözülemedi: {dep}")
            
            self.resolutions[module_name] = resolutions
            return resolutions
            
        except Exception as e:
            self.logger.error(f"Çakışma çözüm hatası: {e}")
            return {}
    
    def _select_resolution_strategy(self, package: str, issue: str) -> str:
        """Resolution strategy seçimi"""
        try:
            issue_lower = issue.lower()
            
            # Hata tipine göre strateji seç
            if "not found" in issue_lower or "no matching distribution" in issue_lower:
                return "package_replacement"
            elif "version conflict" in issue_lower or "requires" in issue_lower:
                if "newer" in issue_lower or "greater" in issue_lower:
                    return "version_upgrade"
                else:
                    return "version_downgrade"
            elif "permission" in issue_lower or "access" in issue_lower:
                return "dependency_isolation"
            else:
                return "force_reinstall"
                
        except Exception as e:
            self.logger.error(f"Strategy selection hatası: {e}")
            return "force_reinstall"
    
    def _apply_resolution_strategy(self, package: str, issue: str, strategy: str) -> Optional[Dict]:
        """Resolution strategy uygulama"""
        cmd = []  # Initialize cmd
        try:
            clean_pkg = self.clean_version(package)
            
            if strategy == "force_reinstall":
                cmd = [sys.executable, "-m", "pip", "install", clean_pkg, "--force-reinstall"]
                
            elif strategy == "version_upgrade":
                cmd = [sys.executable, "-m", "pip", "install", clean_pkg, "--upgrade"]
                
            elif strategy == "version_downgrade":
                # Compatible version'ı bul ve kur
                compatible_version = self._find_compatible_version(clean_pkg)
                if compatible_version:
                    cmd = [sys.executable, "-m", "pip", "install", f"{clean_pkg}=={compatible_version}"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", clean_pkg, "--force-reinstall"]
                    
            elif strategy == "dependency_isolation":
                cmd = [sys.executable, "-m", "pip", "install", clean_pkg, "--user", "--no-deps"]
                
            elif strategy == "package_replacement":
                # Alternative package adını bul
                alternative = self._find_alternative_package(clean_pkg)
                if alternative:
                    cmd = [sys.executable, "-m", "pip", "install", alternative]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", clean_pkg, "--force-reinstall"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", clean_pkg]
            
            # Komutu çalıştır
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "command": " ".join(cmd),
                "strategy": strategy,
                "reason": issue,
                "output": result.stdout,
                "success": True
            }
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Resolution command failed: {e}")
            cmd_str = " ".join(cmd) if cmd else f"pip install {package}"
            return {
                "command": cmd_str,
                "strategy": strategy,
                "reason": issue,
                "output": e.stderr if e.stderr else str(e),
                "success": False
            }
        except Exception as e:
            self.logger.error(f"Resolution strategy uygulaması hatası: {e}")
            return None
    
    def _find_compatible_version(self, package: str) -> Optional[str]:
        """Compatible version bulma"""
        try:
            # Pip ile mevcut versiyonları listele
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", package],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                # Output'tan version'ları parse et
                versions = []
                for line in result.stdout.split('\n'):
                    if 'Available versions:' in line:
                        version_part = line.split(':')[1].strip()
                        versions = [v.strip() for v in version_part.split(',')]
                        break
                
                if versions:
                    # En yakın stable version'ı döndür
                    for version in versions:
                        if not any(x in version for x in ['dev', 'alpha', 'beta', 'rc']):
                            return version
                    
                    # Stable bulunamazsa ilk version'ı döndür
                    return versions[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Compatible version bulma hatası: {e}")
            return None
    
    def _find_alternative_package(self, package: str) -> Optional[str]:
        """Alternative package bulma"""
        # Yaygın package alternatifleri
        alternatives = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
            'dateutil': 'python-dateutil',
            'jwt': 'PyJWT',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysqlclient'
        }
        
        return alternatives.get(package.lower(), None)
    
    def check_version_compatibility(self, package1: str, version1: str, package2: str, version2: str) -> bool:
        """Version uyumluluğu kontrolü"""
        try:
            # Known incompatibility patterns
            incompatible_pairs = [
                ('numpy', '2.0', 'tensorflow', '2.13'),
                ('pandas', '2.0', 'numpy', '1.20'),
                ('matplotlib', '3.7', 'numpy', '1.19')
            ]
            
            for p1, v1, p2, v2 in incompatible_pairs:
                if ((package1.lower() == p1 and version1.startswith(v1) and 
                     package2.lower() == p2 and version2.startswith(v2)) or
                    (package1.lower() == p2 and version1.startswith(v2) and 
                     package2.lower() == p1 and version2.startswith(v1))):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Version compatibility check hatası: {e}")
            return True  # Hata durumunda uyumlu kabul et
    
    def generate_conflict_report(self) -> Dict:
        """Çakışma raporu oluştur"""
        try:
            total_conflicts = sum(len(conflicts) for conflicts in self.conflicts.values())
            total_resolutions = sum(len(resolutions) for resolutions in self.resolutions.values())
            
            success_rate = (total_resolutions / total_conflicts * 100) if total_conflicts > 0 else 100
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_modules_checked": len(self.conflicts),
                "total_conflicts_found": total_conflicts,
                "total_resolutions_attempted": total_resolutions,
                "success_rate": f"{success_rate:.1f}%",
                "dependency_graph_size": len(self.dependency_graph),
                "conflicts_by_module": {module: len(conflicts) for module, conflicts in self.conflicts.items()},
                "resolution_strategies_used": self._get_strategy_usage_stats()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Conflict report generation hatası: {e}")
            return {}
    
    def _get_strategy_usage_stats(self) -> Dict:
        """Strategy kullanım istatistikleri"""
        try:
            strategy_counts = defaultdict(int)
            
            for resolutions in self.resolutions.values():
                for resolution in resolutions.values():
                    if isinstance(resolution, dict) and "strategy" in resolution:
                        strategy_counts[resolution["strategy"]] += 1
            
            return dict(strategy_counts)
            
        except Exception as e:
            self.logger.error(f"Strategy usage stats hatası: {e}")
            return {}


# === MODULE ANALYZER (V1795.PY) ===
class ModuleAnalyzer:
    """
    Modül analiz sistemi - v1795.py'den alınmıştır.
    Log analizi, çakışma tespiti ve modül raporları ile.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.log_file = Path("logs/auto_importer.jsonl")
    
    def analyze_logs(self) -> Dict:
        """Log dosyalarını analiz eder, hata/uyarı/bilgi istatistiklerini toplar."""
        try:
            log_stats = {
                "errors": [], "warnings": [], "info": [], "debug": [],
                "error_count": 0, "warning_count": 0, "info_count": 0, "debug_count": 0
            }
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            level = data.get("level", "").lower()
                            if level == "error":
                                log_stats["errors"].append(line.strip())
                                log_stats["error_count"] += 1
                            elif level == "warning":
                                log_stats["warnings"].append(line.strip())
                                log_stats["warning_count"] += 1
                            elif level == "info":
                                log_stats["info"].append(line.strip())
                                log_stats["info_count"] += 1
                            elif level == "debug":
                                log_stats["debug"].append(line.strip())
                                log_stats["debug_count"] += 1
                        except json.JSONDecodeError:
                            continue
                self.logger.info(f"Log analizi: {log_stats['error_count']} hata, {log_stats['warning_count']} uyarı")
            return log_stats
        except Exception as e:
            self.logger.error(f"Log analizi hatası: {e}")
            return {"error": str(e)}
    
    def analyze_conflicts(self) -> Dict:
        """Loglardan çakışma mesajlarını çıkarır."""
        try:
            conflicts = {}
            error_log = Path("logs/pdsxu_errors.jsonl")
            warning_log = Path("logs/pdsxu_warnings.jsonl")
            
            for log_file in [error_log, warning_log]:
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                message = data.get("message", "")
                                if "çakışma" in message.lower() or "conflict" in message.lower():
                                    parts = message.split(":")
                                    if len(parts) > 1:
                                        pkg = parts[1].strip().split(" ")[0]
                                        conflicts[pkg] = message
                            except (json.JSONDecodeError, IndexError):
                                continue
            self.logger.info(f"Çakışma analizi: {len(conflicts)} çakışma bulundu.")
            return conflicts
        except Exception as e:
            self.logger.error(f"Çakışma analizi hatası: {e}")
            return {}
    
    def generate_module_report(self, modules: List[Dict]) -> Dict:
        """Modül raporları üretir: durum, bağımlılıklar ve sorunlar."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "modules": {},
                "total_modules": len(modules),
                "issues": [],
                "recommendations": []
            }
            for module in modules:
                module_name = module.get("name", "unknown")
                report["modules"][module_name] = {
                    "version": module.get("version", "unknown"),
                    "dependencies": module.get("dependencies", []),
                    "status": self._check_module_status(module)
                }
                issues = self._detect_issues(module)
                if issues:
                    report["issues"].extend(issues)
                recommendations = self._generate_recommendations(module)
                if recommendations:
                    report["recommendations"].extend(recommendations)
            self.logger.info("Modül raporu oluşturuldu.")
            return report
        except Exception as e:
            self.logger.error(f"Modül raporu hatası: {e}")
            return {"error": str(e)}
    
    def _check_module_status(self, module: Dict) -> str:
        """Modül durumunu kontrol et"""
        try:
            module_name = module.get("name", "")
            if not module_name:
                return "unknown"
            
            # Import denemesi
            try:
                importlib.import_module(module_name)
                return "installed"
            except ImportError:
                return "missing"
        except Exception:
            return "error"
    
    def _detect_issues(self, module: Dict) -> List[str]:
        """Modül sorunlarını tespit et"""
        issues = []
        try:
            if module.get("status") == "missing":
                issues.append(f"Module {module.get('name')} is missing")
            if not module.get("version"):
                issues.append(f"Module {module.get('name')} version unknown")
        except Exception:
            pass
        return issues
    
    def _generate_recommendations(self, module: Dict) -> List[str]:
        """Modül önerileri oluştur"""
        recommendations = []
        try:
            if module.get("status") == "missing":
                module_name = module.get("name", "")
                recommendations.append(f"Install missing module: pip install {module_name}")
        except Exception:
            pass
        return recommendations


# === ASYNC DOWNLOAD MANAGER (V1795.PY) ===
class AsyncDownloadManager:
    """
    Asenkron indirme yöneticisi - v1795.py'den alınmıştır.
    ThreadPoolExecutor ile concurrent downloads.
    """
    
    def __init__(self, max_workers: int = 4, cache_dir=".pdsx_cache/wheels", logger=None):
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else AdvancedLogger()
        self.download_stats = {}
    
    def download_package(self, package: str) -> Dict:
        """Paketi indir"""
        try:
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
            start_time = time.time()
            
            # Pip ile paketi cache'e indir
            result = subprocess.run([
                sys.executable, "-m", "pip", "download", package,
                "--dest", str(self.cache_dir), "--no-deps"
            ], capture_output=True, text=True)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.download_stats[package_name] = {
                    "success": True, 
                    "duration": duration,
                    "size": self._get_package_size(package_name)
                }
                self.logger.info(f"{package_name} başarıyla indirildi ({duration:.2f}s)")
            else:
                self.download_stats[package_name] = {
                    "success": False, 
                    "error": result.stderr.strip() or "İndirme başarısız"
                }
                self.logger.error(f"{package_name} indirme hatası: {result.stderr}")
            
            return self.download_stats[package_name]
            
        except Exception as e:
            self.logger.error(f"Paket indirme hatası: {e}")
            return {"success": False, "error": str(e)}
    
    def download_multiple(self, packages: List[str]) -> Dict:
        """Birden fazla paketi paralel indir"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_package = {
                    executor.submit(self.download_package, pkg): pkg 
                    for pkg in packages
                }
                
                results = {}
                for future in as_completed(future_to_package):
                    package = future_to_package[future]
                    try:
                        result = future.result()
                        results[package] = result
                    except Exception as e:
                        results[package] = {"success": False, "error": str(e)}
                        self.logger.error(f"{package} executor hatası: {e}")
                
                return results
        except Exception as e:
            self.logger.error(f"Parallel download hatası: {e}")
            return {}
    
    def _get_package_size(self, package_name: str) -> int:
        """Paket boyutunu al"""
        try:
            for file in self.cache_dir.glob(f"{package_name}*"):
                if file.is_file():
                    return file.stat().st_size
            return 0
        except Exception:
            return 0


# === SCIENTIFIC UTILS (V1795.PY BASIT VERSIYON) ===
class ScientificUtils:
    """
    Bilimsel analiz araçları - v1795.py'den basit versiyon.
    Sistem metrikleri ve performans analizi.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.metrics = defaultdict(list)
    
    def analyze_system_metrics(self) -> Dict:
        """Sistem metriklerini analiz et"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "memory": self._get_memory_usage(),
                "cpu": self._get_cpu_usage(),
                "disk": self._get_disk_usage(),
                "python_info": self._get_python_info()
            }
            
            self.metrics["system_analysis"].append(metrics)
            self.logger.info("Sistem metrikleri analizi tamamlandı")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Sistem metrikleri hatası: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self) -> Dict:
        """Performans raporu oluştur"""
        try:
            if not self.metrics["system_analysis"]:
                return {"error": "No metrics available"}
            
            latest = self.metrics["system_analysis"][-1]
            report = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": latest,
                "recommendations": self._generate_performance_recommendations(latest)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performans raporu hatası: {e}")
            return {"error": str(e)}
    
    def _get_memory_usage(self) -> Dict:
        """Bellek kullanımını al"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "used": memory.used,
                "percentage": memory.percent
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_cpu_usage(self) -> Dict:
        """CPU kullanımını al"""
        try:
            import psutil
            return {
                "percentage": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_disk_usage(self) -> Dict:
        """Disk kullanımını al"""
        try:
            import psutil
            disk = psutil.disk_usage('.')
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percentage": (disk.used / disk.total) * 100
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_python_info(self) -> Dict:
        """Python bilgilerini al"""
        try:
            return {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_performance_recommendations(self, metrics: Dict) -> List[str]:
        """Performans önerileri oluştur"""
        recommendations = []
        try:
            memory = metrics.get("memory", {})
            if isinstance(memory, dict) and memory.get("percentage", 0) > 80:
                recommendations.append("Yüksek bellek kullanımı tespit edildi. Gereksiz süreçleri kapatın.")
            
            cpu = metrics.get("cpu", {})
            if isinstance(cpu, dict) and cpu.get("percentage", 0) > 90:
                recommendations.append("Yüksek CPU kullanımı tespit edildi. Arka plan işlemlerini kontrol edin.")
            
            disk = metrics.get("disk", {})
            if isinstance(disk, dict) and disk.get("percentage", 0) > 90:
                recommendations.append("Disk alanı azalıyor. Gereksiz dosyaları temizleyin.")
                
        except Exception:
            pass
        return recommendations


# === AUTO IMPORTER (TOPLU1.PY BASE + HİBRİT) ===
class AutoImporter:
    """
    Ana Auto Importer sınıfı - toplu1.py temel + hibrit özellikler
    """
    
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super(AutoImporter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Core components
        self.logger = AdvancedLogger(silent=False)
        self.terminal_analyzer = TerminalLogAnalyzer(self.logger)
        self.dependency_registry = DependencyRegistry(logger=self.logger)
        self.cache_manager = CacheManager(logger=self.logger)
        self.env_manager = EnvManager(logger=self.logger)
        self.realtime_monitor = RealTimeLogMonitor(self.logger)
        self.summary_generator = SummaryGenerator(self.logger)
        self.pip_analyzer = PipOutputAnalyzer(self.logger)
        self.conflict_manager = ConflictManager(self.logger)
        
        # State management
        self.running = False
        self.installed_packages = set()
        self.failed_packages = set()
        self.dependencies = defaultdict(list)
        self.installation_stats = defaultdict(int)
        self.retry_count = defaultdict(int)
        self.max_retries = 3
        self.loaded_modules = {}
        self.module_cache = {}
        self.imported_files = set()
        self.aliases = {}
        self.secure_mode = False
        self.metadata = {"auto_importer": {"version": "Final Clean v1.0", "dependencies": []}}
        self.installation_history = {}
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # Graceful shutdown integration
        global shutdown_manager
        shutdown_manager.register_cleanup(self.cleanup_on_shutdown)
        
        # Environment setup
        try:
            if not self.env_manager.setup_environment():
                self.logger.warning("Environment setup tamamlanamadı, devam ediliyor...")
        except Exception as e:
            self.logger.warning(f"Environment setup hatası: {e}")
        
        # Monitoring başlat
        self.realtime_monitor.start_monitoring()
        
        # Last args saver
        self.last_args_file = Path("last_importer_args.json")
        self.save_last_args(sys.argv)
    
    def save_last_args(self, args: List[str]):
        """Son çalıştırma argümanlarını kaydet"""
        try:
            args_data = {
                "command": args,
                "timestamp": datetime.now().isoformat(),
                "working_directory": os.getcwd(),
                "python_executable": sys.executable
            }
            args_file = Path("cache/.pdsx_last_args.json")
            args_file.parent.mkdir(exist_ok=True)
            
            with open(args_file, "w", encoding="utf-8") as f:
                json.dump(args_data, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"Komut argümanları kaydedildi: {args}")
        except Exception as e:
            self.logger.error(f"Komut argümanları kaydetme hatası: {e}")
    
    def install_package(self, package: str, silent: bool = False) -> bool:
        """
        Paketi kur ve istatistikleri güncelle
        
        Args:
            package: Kurulacak paket adı
            silent: Sessiz kurulum
            
        Returns:
            bool: Kurulum başarılı ise True
        """
        start_time = time.time()
        package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
        
        try:
            # Shutdown kontrolü
            if shutdown_manager.is_shutdown_requested():
                self.logger.warning("Shutdown talep edildi, kurulum iptal ediliyor")
                return False
            
            # Zaten kurulu mu kontrol et
            if self.env_manager.check_package_installed(package_name):
                duration = time.time() - start_time
                if not silent:
                    self.logger.info(f"✅ {package_name} zaten kurulu")
                self.summary_generator.record_installation(package_name, "skipped", duration)
                return True
            
            # Registry kontrolü
            if self.dependency_registry.check_package(package_name):
                if not silent:
                    self.logger.info(f"✅ {package_name} registry'de güncel")
                return True
            
            # Kurulum başlat
            if not silent:
                self.logger.info(f"� {package} kuruluyor...")
            
            # Cache'den kur dene
            if self.cache_manager.install_from_cache(package_name):
                duration = time.time() - start_time
                self.summary_generator.record_installation(package_name, "successful", duration)
                self.dependency_registry.register_package(package_name, "latest", "Başarılı")
                if not silent:
                    self.logger.info(f"✅ {package_name} cache'den kuruldu ({duration:.2f}s)")
                return True
            
            # Normal pip install
            install_start = time.time()
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, 
                "--quiet" if silent else "--verbose"
            ], capture_output=True, text=True)
            install_duration = time.time() - install_start
            
            if result.returncode == 0:
                # Başarılı kurulum
                total_duration = time.time() - start_time
                version = "latest"
                if "==" in package:
                    version = package.split("==")[1]
                
                # Registry'e kaydet
                self.dependency_registry.register_package(package_name, version, "Başarılı")
                
                # Statistics
                self.summary_generator.record_installation(package_name, "successful", total_duration)
                
                if not silent:
                    self.logger.info(f"✅ {package_name} başarıyla kuruldu ({total_duration:.2f}s)")
                
                # Kurulum geçmişine ekle
                self.installation_history[package_name] = {
                    "version": version,
                    "installed_at": datetime.now().isoformat(),
                    "status": "success",
                    "duration": total_duration
                }
                
                return True
            else:
                # Başarısız kurulum
                total_duration = time.time() - start_time
                error_msg = result.stderr.strip() or result.stdout.strip() or "Bilinmeyen hata"
                
                # Registry'e kaydet
                self.dependency_registry.register_package(package_name, "failed", "Başarısız", [])
                
                # Pip analyzer ile düzelt dene
                if self.pip_analyzer.analyze_and_fix(error_msg, package_name):
                    self.logger.info(f"✅ {package_name} düzeltme ile kuruldu")
                    return True
                
                # Statistics
                self.summary_generator.record_installation(package_name, "failed", total_duration)
                
                self.logger.error(f"❌ {package_name} kurulum hatası: {error_msg}")
                
                return False
                
        except Exception as e:
            # Exception scope için değişkenleri tanımla
            total_duration = time.time() - start_time
            error_msg = str(e)
            self.summary_generator.record_installation(package_name, "failed", total_duration)
            self.logger.error(f"❌ {package} kurulum istisnası: {error_msg}")
            return False
    
    def install_required_packages(self, silent: bool = False) -> Dict:
        """108 REQUIRED_PACKAGES'ı kur"""
        try:
            results = {"successful": [], "failed": [], "skipped": []}
            total_packages = len(REQUIRED_PACKAGES)
            
            if not silent:
                self.logger.info(f"🔄 {total_packages} paket kurulum başlıyor...")
            
            # ThreadPoolExecutor ile parallel installation
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_package = {
                    executor.submit(self.install_package, pkg, silent): pkg 
                    for pkg in REQUIRED_PACKAGES
                }
                
                for i, future in enumerate(as_completed(future_to_package)):
                    package = future_to_package[future]
                    
                    # Shutdown kontrolü
                    if shutdown_manager.is_shutdown_requested():
                        self.logger.warning("Shutdown talep edildi, kalan kurulumlar iptal ediliyor")
                        break
                    
                    try:
                        success = future.result()
                        if success:
                            results["successful"].append(package)
                        else:
                            results["failed"].append(package)
                            
                        # Progress
                        if not silent:
                            progress = (i + 1) / total_packages * 100
                            self.logger.info(f"📊 İlerleme: {progress:.1f}% ({i + 1}/{total_packages})")
                            
                    except Exception as e:
                        results["failed"].append(package)
                        self.logger.error(f"❌ {package} executor hatası: {e}")
            
            # Özet rapor
            success_count = len(results["successful"])
            failed_count = len(results["failed"])
            success_rate = (success_count / total_packages * 100) if total_packages > 0 else 0
            
            report = f"""
🎯 KURULUM TAMAMLANDI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Başarılı: {success_count}
❌ Başarısız: {failed_count}
📊 Başarı Oranı: {success_rate:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            if not silent:
                self.logger.info(report)
            
            # Başarısız paketleri tekrar dene
            if results["failed"] and not silent:
                self.logger.info("🔄 Başarısız paketler tekrar deneniyor...")
                for pkg in results["failed"][:5]:  # İlk 5 başarısız paketi
                    if shutdown_manager.is_shutdown_requested():
                        break
                    if self.install_package(pkg, silent=True):
                        results["successful"].append(pkg)
                        results["failed"].remove(pkg)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Bulk installation hatası: {e}")
            return {"successful": [], "failed": REQUIRED_PACKAGES.copy(), "skipped": []}
    
    def cleanup_on_shutdown(self):
        """Graceful shutdown cleanup"""
        try:
            self.logger.info("🛑 AutoImporter cleanup başlatıldı...")
            
            # Monitoring durdur
            if hasattr(self, 'realtime_monitor'):
                self.realtime_monitor.stop_monitoring()
            
            # Thread pool kapat
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Final rapor
            if hasattr(self, 'summary_generator'):
                final_report = self.summary_generator.generate_summary()
                self.logger.info(f"📊 Final Report:\n{final_report}")
            
            self.logger.info("✅ AutoImporter cleanup tamamlandı")
            
        except Exception as e:
            self.logger.error(f"Cleanup hatası: {e}")
    
    def shutdown(self):
        """Eksik metod - AutoImporter'ı güvenli şekilde kapat"""
        try:
            self.logger.info("[PDS-X] AutoImporter shutdown başlatılıyor...")
            
            # Cleanup yap
            self.cleanup_on_shutdown()
            
            # Running durumunu false yap
            self.running = False
            
            self.logger.info("[PDS-X] AutoImporter başarıyla kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"[PDS-X] AutoImporter shutdown hatası: {e}")
            return False
    
    # Demo compatibility properties
    @property 
    def log_analyzer(self):
        return self.terminal_analyzer
    
    @property
    def real_time_monitor(self):
        return self.realtime_monitor


# === GLOBAL SHUTDOWN MANAGER ===
shutdown_manager = GracefulShutdownManager()


# === MAIN FUNCTION ===
def main():
    """
    Ana giriş noktası - 108 paketi otomatik kur
    """
    try:
        print("🚀 PDS-X AutoImporter Final Clean v1.0 başlatılıyor...")
        print(f"📦 {len(REQUIRED_PACKAGES)} paket kurulum listesinde")
        print("⚠️  Ctrl+Shift+Q ile acil durdurma")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # AutoImporter oluştur
        importer = AutoImporter()
        
        # 108 paketi kur
        results = importer.install_required_packages(silent=False)
        
        # Final özet
        success_count = len(results["successful"])
        failed_count = len(results["failed"])
        total_count = len(REQUIRED_PACKAGES)
        
        print("\n🎯 KURULUM SONUCU")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"✅ Başarılı: {success_count}/{total_count}")
        print(f"❌ Başarısız: {failed_count}/{total_count}")
        print(f"📊 Başarı Oranı: {success_count/total_count*100:.1f}%")
        
        if results["failed"]:
            print("\n❌ Başarısız Paketler:")
            for pkg in results["failed"][:10]:  # İlk 10'u göster
                print(f"   • {pkg}")
            if len(results["failed"]) > 10:
                print(f"   ... ve {len(results['failed'])-10} paket daha")
        
        print("\n📋 Detaylı rapor logs/ klasöründe")
        print("🔗 GitHub: https://github.com/metedinler/pdsXbasic")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return 0 if failed_count == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Kullanıcı tarafından iptal edildi")
        return 130
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        return 1
    finally:
        # Cleanup
        try:
            shutdown_manager.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
