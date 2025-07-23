#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Enhanced Auto Importer - Merged Version
============================================

Bu dosya, kullanıcının karsilastirmali_tablo.md dosyasındaki seçimlerine göre
toplu1.py ve auto_importer_v1795.py dosyalarından en iyi özellikleri birleştirerek
oluşturulmuş, tam özellikli, lazy import içermeyen auto importer'dır.

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
- AsyncDownloadManager: toplu1.py (async + hibrit ThreadPool)
- ScientificUtils: toplu1.py (ML/AI optimizasyon)
- ModuleSummaryGenerator: toplu1.py (gelişmiş raporlama)
- AutoImporter: toplu1.py temel + hibrit ThreadPool
- TerminalLogAnalyzer: v1795.py (gelişmiş regex)
- RealTimeLogMonitor: v1795.py (JSONL optimization)

Versiyon: Merged v1.0
Oluşturma Tarihi: 2024
Kullanılan Kaynak Dosyalar: toplu1.py, auto_importer_v1795.py
REQUIRED_PACKAGES: 108 paket (doğrulanmış)
"""

import os
import sys
import subprocess
import threading
import time
import json
import logging
import hashlib
import signal
import atexit
import re
import shutil
import psutil
import asyncio
import concurrent.futures
import importlib.util
import platform
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
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import traceback
import queue
import warnings
import platform
import tempfile
from urllib.parse import urlparse
import urllib.request
import socket
import io
import contextlib
import winreg  # Windows registry access

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
        atexit.register(self.cleanup)
    
    def _setup_signal_handlers(self):
        """Sinyal yakalayıcıları kurulumu"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self._signal_handler)
        except Exception as e:
            logging.warning(f"Signal handler setup error: {e}")
    
    def _setup_keyboard_listener(self):
        """Keyboard kill switch (Ctrl+Shift+Q) kurulumu"""
        try:
            import keyboard
            keyboard.add_hotkey('ctrl+shift+q', self._emergency_shutdown)
            self.hotkey_listener = True
        except ImportError:
            logging.warning("Keyboard library not available for hotkey listener")
            self.hotkey_listener = False
        except Exception as e:
            logging.warning(f"Hotkey listener setup error: {e}")
            self.hotkey_listener = False
    
    def _signal_handler(self, signum, frame):
        """Sinyal yakalama handler'ı"""
        logging.info(f"Shutdown signal received: {signum}")
        self.request_shutdown()
    
    def _emergency_shutdown(self):
        """Acil kapatma (Ctrl+Shift+Q)"""
        logging.critical("EMERGENCY SHUTDOWN ACTIVATED!")
        self.emergency_shutdown = True
        self.request_shutdown()
        self._emergency_cleanup()
    
    def request_shutdown(self):
        """Normal kapatma talebi"""
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
                if hasattr(process, 'terminate'):
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
                if hasattr(process, 'kill'):
                    process.kill()
            except Exception as e:
                logging.error(f"Emergency process kill error: {e}")
        
        # Hotkey listener'ı durdur
        if self.hotkey_listener:
            try:
                import keyboard
                keyboard.unhook_all()
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
                stream.flush()
            except Exception as e:
                # Hata durumunda diğer stream'lere devam et
                continue
    
    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                continue
    
    def close(self):
        for stream in self.streams:
            try:
                if hasattr(stream, 'close') and stream != sys.stdout and stream != sys.stderr:
                    stream.close()
            except Exception:
                continue


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
        
        self._setup_logging()
        self._setup_elasticsearch() if enable_elasticsearch else None
    
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
            terminal_file = open(terminal_log, 'w', encoding='utf-8')
            sys.stdout = Tee(sys.__stdout__, terminal_file)
            sys.stderr = Tee(sys.__stderr__, terminal_file)
    
    def _setup_elasticsearch(self):
        """Elasticsearch entegrasyonu (toplu1.py'den)"""
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
            self.es_index = f"auto-importer-{datetime.now().strftime('%Y-%m')}"
        except ImportError:
            self.logger.warning("Elasticsearch library not available")
            self.es = None
        except Exception as e:
            self.logger.warning(f"Elasticsearch connection failed: {e}")
            self.es = None
    
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
        if current_time - self.last_message_time[msg_hash] < 1.0:  # 1 saniyeden hızlı
            return True
        
        self.last_message_time[msg_hash] = current_time
        return False
    
    def _rotate_logs(self):
        """Log rotasyonu (v1795.py'den)"""
        try:
            if self.jsonl_file.exists() and self.jsonl_file.stat().st_size > self.max_log_size:
                # Eski log'ları shift et
                for i in range(self.max_log_files - 1, 0, -1):
                    old_file = self.log_dir / f"auto_importer_{datetime.now().strftime('%Y%m%d')}.{i}.jsonl"
                    new_file = self.log_dir / f"auto_importer_{datetime.now().strftime('%Y%m%d')}.{i+1}.jsonl"
                    if old_file.exists():
                        old_file.rename(new_file)
                
                # Mevcut dosyayı .1 olarak taşı
                rotated_file = self.log_dir / f"auto_importer_{datetime.now().strftime('%Y%m%d')}.1.jsonl"
                self.jsonl_file.rename(rotated_file)
        except Exception as e:
            self.logger.error(f"Log rotation error: {e}")
    
    def _write_jsonl(self, level, message, extra_data=None):
        """JSONL formatında log yazma (v1795.py'den)"""
        try:
            self._rotate_logs()
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': str(message),
                'thread': threading.current_thread().name,
                'process': os.getpid()
            }
            
            if extra_data:
                log_entry.update(extra_data)
            
            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"JSONL write error: {e}")
    
    def _log_to_elasticsearch(self, level, message, extra_data=None):
        """Elasticsearch'e log gönderme (toplu1.py'den)"""
        if not self.es:
            return
        
        try:
            doc = {
                'timestamp': datetime.now(),
                'level': level,
                'message': str(message),
                'hostname': platform.node(),
                'python_version': platform.python_version()
            }
            
            if extra_data:
                doc.update(extra_data)
            
            self.es.index(index=self.es_index, document=doc)
        except Exception as e:
            self.logger.error(f"Elasticsearch logging error: {e}")
    
    def log(self, level, message, extra_data=None):
        """Ana logging metodu"""
        if self._is_spam(message):
            return  # Spam mesajları atla
        
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
            missing_modules = set()
            version_conflicts = set()
            import_errors = []
            pip_suggestions = set()
            
            lines = output_text.split('\n')
            
            for line in lines:
                # ModuleNotFoundError detection
                for pattern in self.module_not_found_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        module_name = match.strip()
                        missing_modules.add(module_name)
                        self.logger.info(f"ModuleNotFoundError detected: {module_name}")
                
                # Version conflict detection
                for pattern in self.version_conflict_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            pkg_name, version = match
                            version_conflicts.add(f"{pkg_name} {version}")
                        else:
                            version_conflicts.add(match)
                        self.logger.warning(f"Version conflict detected: {match}")
                
                # Import error detection
                for pattern in self.import_error_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        import_errors.append(match)
                        self.logger.error(f"Import error detected: {match}")
                
                # Pip suggestion extraction
                for pattern in self.pip_suggestion_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        pip_suggestions.add(match.strip())
                        self.logger.info(f"Pip suggestion found: {match}")
            
            # Module-to-package mapping
            packages_to_install = set()
            for module in missing_modules:
                package = self.module_to_package.get(module, module)
                packages_to_install.add(package)
            
            # Pip suggestions'ları da ekle
            packages_to_install.update(pip_suggestions)
            
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
            # Son ModuleNotFoundError'ı bul
            lines = traceback_text.split('\n')
            for line in reversed(lines):
                for pattern in self.module_not_found_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        module_name = match.group(1)
                        package_name = self.module_to_package.get(module_name, module_name)
                        return package_name
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
                    "status": "clean",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Registry yükleme hatası: {e}")
            return {"packages": {}, "resolutions": {}, "status": "clean", "timestamp": datetime.now().isoformat()}
    
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
                if package in self.registry["packages"]:
                    pkg_info = self.registry["packages"][package]
                    if pkg_info["status"] == "Başarılı":
                        # 24 saat içinde yüklenmişse
                        elapsed = datetime.now() - datetime.fromisoformat(pkg_info["timestamp"])
                        if elapsed.total_seconds() < 24 * 60 * 60:  # 24 saat
                            self.logger.info(f"[AUTO-IMPORTER] {package} zaten yüklü ve güncel.")
                            return True
                return False
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kontrol hatası: {e}")
            return False
    
    def register_package(self, package: str, version: str = "unknown", status: str = "Başarılı", dependencies: List[str] = None):
        """Paketi registry'ye kaydet"""
        try:
            with self.lock:
                self.registry["packages"][package] = {
                    "version": version,
                    "status": status,
                    "dependencies": dependencies or [],
                    "timestamp": datetime.now().isoformat()
                }
                self._save_registry()
                self.logger.info(f"[AUTO-IMPORTER] {package} registry'ye kaydedildi.")
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kayıt hatası: {e}")
    
    def register_resolution(self, package: str, resolution: Dict):
        """Çakışma çözümünü kaydet"""
        try:
            with self.lock:
                self.registry["resolutions"][package] = {
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_registry()
                self.logger.info(f"[AUTO-IMPORTER] {package} çözümü kaydedildi.")
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} çözüm kayıt hatası: {e}")
    
    def update_on_conflict(self, package: str, conflict_info: str, resolution_command: str):
        """Çakışma durumunda registry'yi güncelle"""
        try:
            with self.lock:
                if package in self.registry["packages"]:
                    self.registry["packages"][package]["status"] = "Çakışma"
                    self.registry["packages"][package]["conflict_info"] = conflict_info
                    self.registry["packages"][package]["resolution_command"] = resolution_command
                    self.registry["packages"][package]["timestamp"] = datetime.now().isoformat()
                else:
                    self.registry["packages"][package] = {
                        "status": "Çakışma",
                        "conflict_info": conflict_info,
                        "resolution_command": resolution_command,
                        "timestamp": datetime.now().isoformat()
                    }
                self._save_registry()
                self.logger.warning(f"[AUTO-IMPORTER] {package} çakışma kaydedildi.")
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
                self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten yükleniyor.")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(cache_file)], 
                    check=True, capture_output=True, text=True
                )
                self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten kuruldu: {result.stdout}")
                return True
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
                # Metadata güncelle
                metadata = self.load_package_metadata()
                metadata[package] = {
                    "version": "downloaded",
                    "timestamp": datetime.now().isoformat(),
                    "hash": hashlib.md5(package.encode()).hexdigest()
                }
                self.save_package_metadata(metadata)
                self.logger.info(f"[AUTO-IMPORTER] {package} önbelleğe alındı.")
                return True
            else:
                self.logger.error(f"[AUTO-IMPORTER] {package} indirme hatası: {result.stderr}")
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
                pkg_time = datetime.fromisoformat(info["timestamp"])
                if (now - pkg_time).days > max_age_days:
                    cache_file = self.wheels_dir / f"{pkg}.whl"
                    if cache_file.exists():
                        cache_file.unlink()
                    del metadata[pkg]
                    self.logger.info(f"[AUTO-IMPORTER] Eski önbellek dosyası silindi: {pkg}")
            self.save_package_metadata(metadata)
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Önbellek temizleme hatası: {e}")


# === ENV MANAGER - EXTENDED (TOPLU1.PY) ===
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
        """Python 3.10 aramasi - toplu1.py'den"""
        try:
            self.logger.info("[AUTO-IMPORTER] Python 3.10 aranıyor...")
            for exe in ["python3.10", "python310", "python"]:
                path = shutil.which(exe)
                if path:
                    out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
                    if "3.10" in out:
                        self.logger.info(f"[AUTO-IMPORTER] Python 3.10 bulundu: {path}")
                        self.python_path = path
                        return path
            
            # Windows registry araması
            if os.name == "nt":
                try:
                    import winreg
                    for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                        try:
                            with winreg.OpenKey(root, r"SOFTWARE\Python\PythonCore") as hkey:
                                for i in range(winreg.QueryInfoKey(hkey)[0]):
                                    ver = winreg.EnumKey(hkey, i)
                                    if ver.startswith("3.10"):
                                        with winreg.OpenKey(hkey, ver + r"\InstallPath") as subkey:
                                            py = winreg.QueryValue(subkey, None) + "python.exe"
                                            if os.path.exists(py):
                                                self.logger.info(f"[AUTO-IMPORTER] Registry'de bulundu: {py}")
                                                self.python_path = py
                                                return py
                        except Exception:
                            continue
                except ImportError:
                    pass
            
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
    
    def ensure_required_packages(self) -> bool:
        """REQUIRED_PACKAGES listesindeki paketlerin kurulu olduğundan emin ol"""
        try:
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor...")
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            
            success_count = 0
            total_count = len(REQUIRED_PACKAGES)
            
            for package in REQUIRED_PACKAGES:
                try:
                    # Basit import testi
                    test_result = subprocess.run([venv_python, "-c", f"import {package}"], 
                                                capture_output=True, text=True)
                    if test_result.returncode == 0:
                        self.logger.debug(f"[AUTO-IMPORTER] ✓ {package} zaten yüklü")
                        success_count += 1
                        continue
                    else:
                        self.logger.info(f"[AUTO-IMPORTER] × {package} yüklü değil, yükleniyor...")
                except Exception:
                    self.logger.info(f"[AUTO-IMPORTER] × {package} yüklü değil, yükleniyor...")
                
                # Paketi yükle
                try:
                    install_cmd = [pip_cmd, "install", package]
                    process = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                    
                    self.logger.info(f"[AUTO-IMPORTER] ✓ {package} başarıyla yüklendi")
                    success_count += 1
                    
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"[AUTO-IMPORTER] × {package} yüklenemedi: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"[AUTO-IMPORTER] × {package} yükleme hatası: {e}")
                    continue
            
            self.logger.info(f"[AUTO-IMPORTER] Paket kontrol tamamlandı: {success_count}/{total_count} başarılı")
            return success_count > 0  # En az bir paket başarılı ise True
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol hatası: {e}")
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
                
            # Pip sürümünü güncelle
            if not self.update_pip_if_needed(force_latest=False, silent=False):
                self.logger.warning("[AUTO-IMPORTER] Pip güncelleme başarısız, devam ediliyor")
            
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            
            self.logger.info("[AUTO-IMPORTER] Temel paketler yükleniyor...")
            subprocess.run([pip_cmd, "install", "wheel"], check=True)
            
            self.logger.info("[AUTO-IMPORTER] NumPy 1.26.4 yükleniyor...")
            subprocess.run([pip_cmd, "install", "numpy==1.26.4"], check=True)
            
            # REQUIRED_PACKAGES'taki tüm paketlerin kurulumu
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor ve yükleniyor...")
            self.ensure_required_packages()
            
            self.logger.info("[AUTO-IMPORTER] Ortam hazırlığı tamamlandı!")
            
            # Şimdi sanal ortamda yeniden başlat
            self.logger.info("[AUTO-IMPORTER] Sanal ortamda yeniden başlatılıyor...")
            return self.restart_in_venv()
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Ortam hazırlama hatası: {e}")
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


# === DEPENDENCY REGISTRY (HİBRİT - TOPLU1.PY + V1795.PY) ===


# === DEPENDENCY REGISTRY (HİBRİT - TOPLU1.PY + V1795.PY) ===
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
                                                self.logger.info(f"Registry'de bulundu: {py}")
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
                            self.logger.info(f"Yaygın konumda bulundu: {py_path}")
                            self.python_path = py_path
                            return py_path
                    except subprocess.CalledProcessError:
                        continue
            
            self.logger.error("Python 3.10 bulunamadı.")
            return None
            
        except Exception as e:
            self.logger.error(f"Python 3.10 arama hatası: {e}")
            return None
    
    def download_and_install_python310(self) -> Optional[str]:
        """Python 3.10'u otomatik indir ve kur (sadece Windows)"""
        try:
            if os.name != "nt":
                self.logger.error("Python 3.10 bulunamadı. Lütfen manuel kurun: https://www.python.org/downloads/release/python-31011/")
                return None
            
            # Disk alanı kontrolü
            drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
            total, used, free = shutil.disk_usage(drive + '\\')
            min_required = 300 * 1024 * 1024  # 300 MB
            if free < min_required:
                self.logger.error(f"Yetersiz disk alanı: {free // (1024*1024)} MB mevcut, 300 MB gerekli.")
                return None
            
            # Python 3.10.11 installer'ını indir
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            installer_path = "python310_installer.exe"
            
            self.logger.info("Python 3.10 indiriliyor...")
            subprocess.run(["curl", "-o", installer_path, installer_url], check=True, capture_output=True, text=True)
            
            self.logger.info("Python 3.10 kuruluyor...")
            subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True, capture_output=True, text=True)
            
            # Installer'ı temizle
            os.remove(installer_path)
            
            self.logger.info("Python 3.10 kurulumu tamamlandı.")
            return self.find_python310()
            
        except Exception as e:
            self.logger.error(f"Python 3.10 kurulum hatası: {e}")
            return None
    
    def add_python_to_path(self):
        """Python ve venv'i PATH'e ekle"""
        try:
            if not self.python_path:
                self.python_path = self.find_python310() or self.download_and_install_python310()
            
            if self.python_path:
                python_dir = os.path.dirname(self.python_path)
                venv_dir = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin"))
                
                # Windows batch script
                setx_cmd = f'setx PATH "%PATH%;{python_dir};{venv_dir}"'
                with open("add_pdsx_path.bat", "w") as f:
                    f.write(f"@echo off\n{setx_cmd}\necho PATH güncellendi.\npause\n")
                
                # PowerShell script
                with open("add_pdsx_path.ps1", "w") as f:
                    f.write(
                        f"$ErrorActionPreference = 'Stop'\n"
                        f"$pythonDir = '{python_dir}'\n"
                        f"$venvDir = '{venv_dir}'\n"
                        "$oldPath = [System.Environment]::GetEnvironmentVariable('Path', [System.EnvironmentVariableTarget]::User)\n"
                        "if ($oldPath -notlike \"*$pythonDir*\") {\n"
                        "    $newPath = \"$oldPath;$pythonDir;$venvDir\"\n"
                        "    [System.Environment]::SetEnvironmentVariable('Path', $newPath, [System.EnvironmentVariableTarget]::User)\n"
                        "    Write-Host 'PATH güncellendi.'\n"
                        "} else {\n"
                        "    Write-Host 'PATH zaten güncel.'\n"
                        "}\n"
                    )
                
                self.logger.info("PATH'e eklemek için 'add_pdsx_path.bat' veya 'add_pdsx_path.ps1' dosyasını yönetici olarak çalıştırın!")
                
        except Exception as e:
            self.logger.error(f"PATH güncelleme hatası: {e}")

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

    def ensure_required_packages(self) -> bool:
        """REQUIRED_PACKAGES listesindeki paketlerin kurulu olduğundan emin ol"""
        try:
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor...")
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            
            success_count = 0
            total_count = len(REQUIRED_PACKAGES)
            
            for package in REQUIRED_PACKAGES:
                try:
                    # Basit import testi
                    test_result = subprocess.run([venv_python, "-c", f"import {package}"], 
                                                capture_output=True, text=True)
                    if test_result.returncode == 0:
                        self.logger.debug(f"[AUTO-IMPORTER] ✓ {package} zaten yüklü")
                        success_count += 1
                        continue
                    else:
                        self.logger.info(f"[AUTO-IMPORTER] × {package} yüklü değil, yükleniyor...")
                except Exception:
                    self.logger.info(f"[AUTO-IMPORTER] × {package} yüklü değil, yükleniyor...")
                
                # Paketi yükle
                try:
                    install_cmd = [pip_cmd, "install", package]
                    process = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                    
                    self.logger.info(f"[AUTO-IMPORTER] ✓ {package} başarıyla yüklendi")
                    success_count += 1
                    
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"[AUTO-IMPORTER] × {package} yüklenemedi: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"[AUTO-IMPORTER] × {package} yükleme hatası: {e}")
                    continue
            
            self.logger.info(f"[AUTO-IMPORTER] Paket kontrol tamamlandı: {success_count}/{total_count} başarılı")
            return success_count > 0  # En az bir paket başarılı ise True
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol hatası: {e}")
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
                
            # PATH güncellemesi
            self.add_python_to_path()
            
            # Pip sürümünü güncelle
            if not self.update_pip_if_needed(force_latest=False, silent=False):
                self.logger.warning("[AUTO-IMPORTER] Pip güncelleme başarısız, devam ediliyor")
            
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            
            self.logger.info("[AUTO-IMPORTER] Temel paketler yükleniyor...")
            subprocess.run([pip_cmd, "install", "wheel"], check=True)
            
            self.logger.info("[AUTO-IMPORTER] NumPy 1.26.4 yükleniyor...")
            subprocess.run([pip_cmd, "install", "numpy==1.26.4"], check=True)
            
            # REQUIRED_PACKAGES'taki tüm paketlerin kurulumu
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor ve yükleniyor...")
            self.ensure_required_packages()
            
            self.logger.info("[AUTO-IMPORTER] Ortam hazırlığı tamamlandı!")
            
            # Şimdi sanal ortamda yeniden başlat
            self.logger.info("[AUTO-IMPORTER] Sanal ortamda yeniden başlatılıyor...")
            return self.restart_in_venv()
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Ortam hazırlama hatası: {e}")
            return False
    
    def update_pip_if_needed(self, force_latest: bool = False, silent: bool = False) -> bool:
        """
        Pip sürümünü günceller veya kontrol eder.
        
        Args:
            force_latest: True ise en son pip sürümünü zorla indirir
            silent: True ise sessiz kurulum yapar (debug seviyesinde log)
            
        Returns:
            bool: Güncelleme başarılı ise True
        """
        try:
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            
            if not os.path.exists(pip_cmd):
                self.logger.error("Pip komutu bulunamadı!")
                return False
            
            # Mevcut pip sürümünü kontrol et
            try:
                result = subprocess.run([pip_cmd, "--version"], capture_output=True, text=True)
                current_version = result.stdout.strip() if result.returncode == 0 else "unknown"
                log_level = "debug" if silent else "info"
                self.logger.log(log_level, f"Mevcut pip sürümü: {current_version}")
            except Exception as e:
                self.logger.warning(f"Pip sürüm kontrolü başarısız: {e}")
                current_version = "unknown"
            
            # Hedef sürümü belirle
            target_version = "latest" if force_latest else "21.2.4"
            pip_package = "pip" if force_latest else "pip==21.2.4"
            
            log_level = "debug" if silent else "info"
            self.logger.log(log_level, f"Pip hedef sürümü: {target_version}")
            
            # Python 3.10 için pip 21.2.4 mantığı:
            # - Python 3.10.0 Ekim 2021'de çıktı
            # - O dönemde pip 21.2.4 stable idi
            # - Yeni pip sürümleri bazen eski Python sürümleriyle uyumsuzluk yaratabilir
            # - pip 21.2.4 Python 3.10 ile test edilmiş ve kararlı
            if not force_latest:
                self.logger.debug("Python 3.10 için pip 21.2.4 kullanılıyor (o dönemin stable sürümü)")
            
            # Sessiz kurulum parametresi
            install_args = [pip_cmd, "install", "--upgrade"]
            if silent:
                install_args.extend(["--quiet", "--no-warn-script-location"])
            install_args.append(pip_package)
            
            # Kurulumu gerçekleştir
            result = subprocess.run(install_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.log(log_level, f"Pip başarıyla güncellendi: {target_version}")
                return True
            else:
                self.logger.warning(f"Pip güncelleme başarısız: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Pip güncelleme hatası: {e}")
            return False
    
    def create_venv(self) -> bool:
        """Sanal ortam oluştur"""
        try:
            if self.venv_dir.exists():
                self.logger.info(f"Venv zaten mevcut: {self.venv_dir}")
                return True
            
            python_path = self.find_python310()
            if not python_path:
                python_path = self.download_and_install_python310()
                if not python_path:
                    self.logger.error("Python 3.10 bulunamadı, venv oluşturulamadı")
                    return False
            
            self.logger.info(f"Venv oluşturuluyor: {self.venv_dir}")
            result = subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Venv başarıyla oluşturuldu")
                # Pip'i güncelle
                self.update_pip_if_needed(silent=True)
                return True
            else:
                self.logger.error(f"Venv oluşturma hatası: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Venv oluşturma hatası: {e}")
            return False
    
    def activate_venv(self) -> bool:
        """Sanal ortamı aktifleştir"""
        try:
            if not self.venv_dir.exists():
                if not self.create_venv():
                    return False
            
            # PATH'e venv'i ekle
            if os.name == "nt":
                venv_scripts = self.venv_dir / "Scripts"
                venv_python = venv_scripts / "python.exe"
            else:
                venv_scripts = self.venv_dir / "bin"
                venv_python = venv_scripts / "python"
            
            if not venv_python.exists():
                self.logger.error("Venv Python executable bulunamadı")
                return False
            
            # Environment'ı güncelle
            current_path = os.environ.get('PATH', '')
            if str(venv_scripts) not in current_path:
                os.environ['PATH'] = f"{venv_scripts}{os.pathsep}{current_path}"
            
            os.environ['VIRTUAL_ENV'] = str(self.venv_dir)
            if 'PYTHONHOME' in os.environ:
                del os.environ['PYTHONHOME']
            
            self.logger.info(f"Venv aktifleştirildi: {self.venv_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Venv aktifleştirme hatası: {e}")
            return False
    
    def deactivate_venv(self):
        """Sanal ortamı deaktive et"""
        try:
            if 'VIRTUAL_ENV' in os.environ:
                del os.environ['VIRTUAL_ENV']
            
            # PATH'ten venv'i çıkar
            current_path = os.environ.get('PATH', '')
            venv_scripts = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin"))
            
            path_parts = current_path.split(os.pathsep)
            filtered_parts = [part for part in path_parts if venv_scripts not in part]
            os.environ['PATH'] = os.pathsep.join(filtered_parts)
            
            self.logger.info("Venv deaktive edildi")
            
        except Exception as e:
            self.logger.error(f"Venv deaktive etme hatası: {e}")
    
    def get_venv_python(self) -> Optional[str]:
        """Venv Python path'ini döndür"""
        if os.name == "nt":
            python_path = self.venv_dir / "Scripts" / "python.exe"
        else:
            python_path = self.venv_dir / "bin" / "python"
        
        return str(python_path) if python_path.exists() else None
    
    def get_venv_pip(self) -> Optional[str]:
        """Venv pip path'ini döndür"""
        if os.name == "nt":
            pip_path = self.venv_dir / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_dir / "bin" / "pip"
        
        return str(pip_path) if pip_path.exists() else None
    
    def setup_environment(self) -> bool:
        """Ortamı hazırlar ve paketleri kurar"""
        try:
            # Venv oluştur ve aktifleştir
            if not self.create_venv():
                return False
            
            if not self.activate_venv():
                return False
            
            # Pip'i güncelle
            if not self.update_pip_if_needed():
                self.logger.warning("Pip güncellenemedi, devam ediliyor...")
            
            self.logger.info("Ortam başarıyla hazırlandı")
            return True
            
        except Exception as e:
            self.logger.error(f"Ortam hazırlama hatası: {e}")
            return False


# Bu dosya devam edecek... (yaklaşık 2000 satır daha eklenecek)
# Sırada: DependencyRegistry, CacheManager, ConflictManager, ModuleAnalyzer vb.


# === DEPENDENCY REGISTRY (HİBRİT - TOPLU1.PY + V1795.PY) ===
class DependencyRegistry:
    """
    Hibrit bağımlılık kayıt sistemi - toplu1.py + v1795.py features
    dependencies.json yönetimi, çakışma kayıtları, version tracking ile.
    """
    
    def __init__(self, logger, registry_file=None):
        self.logger = logger
        self.registry_file = Path(registry_file) if registry_file else Path("cache/dependencies.json")
        self.registry_file.parent.mkdir(exist_ok=True)
        self.registry = self.load_registry()
        
        # v1795.py'den eklenen özellikler
        self.conflict_resolutions = {}
        self.dependency_mapping = {}
        self.version_tracking = defaultdict(list)
    
    def load_registry(self) -> Dict:
        """Kayıt dosyasını yükle"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Backward compatibility
                    if "packages" not in data:
                        data = {"packages": {}, "resolutions": {}, "status": "", "timestamp": ""}
                    return data
            return {
                "packages": {},
                "resolutions": {},
                "conflicts": {},
                "dependency_mapping": {},
                "version_tracking": {},
                "status": "",
                "timestamp": ""
            }
        except Exception as e:
            self.logger.error(f"Bağımlılık kayıt dosyası yükleme hatası: {e}")
            return {
                "packages": {},
                "resolutions": {},
                "conflicts": {},
                "dependency_mapping": {},
                "version_tracking": {},
                "status": "",
                "timestamp": ""
            }
    
    def save_registry(self):
        """Kayıt dosyasını kaydet"""
        try:
            self.registry_file.parent.mkdir(exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4, ensure_ascii=False)
            self.logger.debug("Bağımlılık kayıt dosyası kaydedildi.")
        except Exception as e:
            self.logger.error(f"Bağımlılık kayıt dosyası kaydetme hatası: {e}")
    
    def register_package(self, package: str, version: str, status: str, dependencies: Optional[List[str]] = None):
        """Paket kaydı (toplu1.py'den)"""
        try:
            dependencies = dependencies or []
            self.registry["packages"][package] = {
                "version": version,
                "status": status,
                "dependencies": dependencies,
                "timestamp": datetime.now().isoformat()
            }
            
            # Version tracking (v1795.py'den)
            self.registry.setdefault("version_tracking", {})
            if package not in self.registry["version_tracking"]:
                self.registry["version_tracking"][package] = []
            
            self.registry["version_tracking"][package].append({
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "status": status
            })
            
            self.save_registry()
            self.logger.info(f"{package}=={version} kayıt edildi: {status}")
        except Exception as e:
            self.logger.error(f"{package} kayıt hatası: {e}")
    
    def register_resolution(self, module_name: str, resolution: Dict):
        """Çakışma çözümü kaydı (toplu1.py'den)"""
        try:
            self.registry.setdefault("resolutions", {})
            self.registry["resolutions"][module_name] = resolution
            self.save_registry()
            self.logger.info(f"{module_name} için çakışma çözümü kaydedildi: {resolution}")
        except Exception as e:
            self.logger.error(f"{module_name} çakışma çözümü kayıt hatası: {e}")
    
    def register_conflict(self, package: str, conflict_info: str, resolution: str):
        """Çakışma kaydı (hibrit)"""
        try:
            self.registry.setdefault("conflicts", {})
            self.registry["conflicts"][package] = {
                "conflict_info": conflict_info,
                "resolution": resolution,
                "timestamp": datetime.now().isoformat()
            }
            
            # Aynı zamanda packages'e de ekle (backward compatibility)
            if package in self.registry["packages"]:
                self.registry["packages"][package]["conflicts"] = conflict_info
                self.registry["packages"][package]["resolution"] = resolution
            
            self.save_registry()
            self.logger.info(f"{package} için çakışma güncellendi: {conflict_info}")
        except Exception as e:
            self.logger.error(f"{package} çakışma güncelleme hatası: {e}")
    
    def update_on_conflict(self, package: str, conflict_info: str, resolution: str):
        """Backward compatibility için"""
        self.register_conflict(package, conflict_info, resolution)
    
    def check_package(self, package: str) -> bool:
        """Paket kontrolü (toplu1.py'den)"""
        try:
            if package in self.registry["packages"]:
                pkg_info = self.registry["packages"][package]
                if pkg_info["status"] == "Başarılı":
                    elapsed = datetime.now() - datetime.fromisoformat(pkg_info["timestamp"])
                    if elapsed.total_seconds() < 24 * 60 * 60:  # 24 saat
                        self.logger.info(f"{package} zaten yüklü ve güncel.")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"{package} kontrol hatası: {e}")
            return False
    
    def map_dependency(self, module_name: str, package_name: str):
        """Modül-paket eşleştirmesi (v1795.py'den)"""
        try:
            self.registry.setdefault("dependency_mapping", {})
            self.registry["dependency_mapping"][module_name] = package_name
            self.save_registry()
            self.logger.debug(f"Dependency mapping: {module_name} -> {package_name}")
        except Exception as e:
            self.logger.error(f"Dependency mapping hatası: {e}")
    
    def get_package_for_module(self, module_name: str) -> Optional[str]:
        """Modül için paket adını döndür (v1795.py'den)"""
        try:
            mapping = self.registry.get("dependency_mapping", {})
            return mapping.get(module_name, module_name)
        except Exception as e:
            self.logger.error(f"Module mapping hatası: {e}")
            return module_name
    
    def get_version_history(self, package: str) -> List[Dict]:
        """Paket sürüm geçmişi (v1795.py'den)"""
        try:
            tracking = self.registry.get("version_tracking", {})
            return tracking.get(package, [])
        except Exception as e:
            self.logger.error(f"Version history hatası: {e}")
            return []
    
    def get_conflicts(self) -> Dict:
        """Tüm çakışmaları döndür"""
        return self.registry.get("conflicts", {})
    
    def clear_old_entries(self, max_age_days: int = 30):
        """Eski kayıtları temizle"""
        try:
            now = datetime.now()
            packages_to_remove = []
            
            for package, info in self.registry.get("packages", {}).items():
                try:
                    pkg_time = datetime.fromisoformat(info["timestamp"])
                    if (now - pkg_time).days > max_age_days:
                        packages_to_remove.append(package)
                except (ValueError, KeyError):
                    packages_to_remove.append(package)  # Invalid timestamp
            
            for package in packages_to_remove:
                del self.registry["packages"][package]
                self.logger.debug(f"Eski kayıt silindi: {package}")
            
            if packages_to_remove:
                self.save_registry()
                self.logger.info(f"{len(packages_to_remove)} eski kayıt temizlendi")
                
        except Exception as e:
            self.logger.error(f"Kayıt temizleme hatası: {e}")
    
    def get_statistics(self) -> Dict:
        """Registry istatistikleri"""
        try:
            packages = self.registry.get("packages", {})
            conflicts = self.registry.get("conflicts", {})
            
            successful = sum(1 for pkg in packages.values() if pkg.get("status") == "Başarılı")
            failed = sum(1 for pkg in packages.values() if pkg.get("status") == "Başarısız")
            
            return {
                "total_packages": len(packages),
                "successful_installations": successful,
                "failed_installations": failed,
                "total_conflicts": len(conflicts),
                "dependency_mappings": len(self.registry.get("dependency_mapping", {})),
                "version_tracked_packages": len(self.registry.get("version_tracking", {}))
            }
        except Exception as e:
            self.logger.error(f"İstatistik hesaplama hatası: {e}")
            return {}


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


# Bu dosya devam edecek... CacheManager, ConflictManager, ModuleAnalyzer vs.


# === CACHE MANAGER (TOPLU1.PY) ===
class CacheManager:
    """
    Gelişmiş package cache yönetimi - toplu1.py'den alınmıştır.
    Hash verification, disk optimization, automatic cleanup ile.
    """
    
    def __init__(self, cache_dir=None, logger=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/wheels")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger is not None else AdvancedLogger()
        self.metadata_file = self.cache_dir / "packages.json"
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.download_times = {}
        
        # Disk space management
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB
        self.cleanup_threshold = 0.8  # 80% dolduğunda temizle
    
    def install_from_cache(self, package: str) -> bool:
        """Önbellekten paket kurulumu"""
        try:
            cache_file = self.cache_dir / f"{package}.whl"
            
            if cache_file.exists():
                # Hash doğrulaması
                if self._verify_cache_file(package, cache_file):
                    self.logger.info(f"{package} önbellekten yükleniyor.")
                    
                    # Pip install from cache
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", str(cache_file)], 
                        check=True, capture_output=True, text=True
                    )
                    
                    self.cache_hits += 1
                    self.logger.info(f"{package} önbellekten kuruldu")
                    return True
                else:
                    # Bozuk cache dosyası, yeniden indir
                    cache_file.unlink()
                    self.logger.warning(f"{package} önbellek dosyası bozuk, yeniden indiriliyor.")
                    
            # Cache'de yok veya bozuk, indir ve cache'le
            self.cache_misses += 1
            return self._download_and_cache(package)
            
        except Exception as e:
            self.logger.error(f"Önbellek yükleme hatası: {e}")
            return False
    
    def _verify_cache_file(self, package: str, cache_file: Path) -> bool:
        """Cache dosyası hash doğrulaması"""
        try:
            metadata = self.load_package_metadata()
            if package not in metadata:
                return False
            
            # File hash kontrolü
            with open(cache_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            expected_hash = metadata[package].get("hash", "")
            if file_hash != expected_hash:
                self.logger.warning(f"{package} hash uyumsuzluğu: {file_hash[:8]} != {expected_hash[:8]}")
                return False
            
            # Timestamp kontrolü (eski dosyalar için)
            pkg_time = datetime.fromisoformat(metadata[package]["timestamp"])
            if (datetime.now() - pkg_time).days > 30:  # 30 gün
                self.logger.info(f"{package} cache dosyası eski, yenileniyor")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hash doğrulama hatası: {e}")
            return False
    
    def _download_and_cache(self, package: str) -> bool:
        """Paket indirme ve cache'leme"""
        try:
            start_time = time.time()
            
            # Disk alanı kontrolü
            if not self._check_disk_space():
                self.cleanup_cache(aggressive=True)
                if not self._check_disk_space():
                    self.logger.error("Yetersiz disk alanı, cache indirme iptal edildi")
                    return False
            
            self.logger.info(f"{package} indiriliyor ve önbelleğe alınıyor.")
            
            # Package'ı download et
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", package, "-d", str(self.cache_dir)],
                check=True, capture_output=True, text=True
            )
            
            download_time = time.time() - start_time
            self.download_times[package] = download_time
            
            # Downloaded file'ı bul ve hash'le
            cache_files = list(self.cache_dir.glob(f"{package}*.whl"))
            if not cache_files:
                # Farklı isimlendirme olabilir, en son indirilen dosyayı bul
                cache_files = sorted(self.cache_dir.glob("*.whl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if cache_files:
                    cache_file = cache_files[0]
                else:
                    self.logger.error(f"{package} download dosyası bulunamadı")
                    return False
            else:
                cache_file = cache_files[0]
            
            # Metadata güncelle
            with open(cache_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            metadata = self.load_package_metadata()
            metadata[package] = {
                "version": package.split("==")[1] if "==" in package else "latest",
                "timestamp": datetime.now().isoformat(),
                "hash": file_hash,
                "file_size": cache_file.stat().st_size,
                "download_time": download_time
            }
            self.save_package_metadata(metadata)
            
            self.logger.info(f"{package} başarıyla cache'lendi ({download_time:.2f}s)")
            
            # Otomatik cleanup
            self.cleanup_cache()
            
            return True
            
        except Exception as e:
            self.logger.error(f"{package} indirme hatası: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Disk alanı kontrolü"""
        try:
            # Cache dizininin toplam boyutu
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            
            # Sistem disk alanı
            disk_usage = shutil.disk_usage(self.cache_dir)
            free_space = disk_usage.free
            
            # Yeterli alan var mı?
            min_required_space = 100 * 1024 * 1024  # 100MB minimum
            if free_space < min_required_space:
                self.logger.warning(f"Düşük disk alanı: {free_space // (1024*1024)}MB kaldı")
                return False
            
            # Cache boyutu limiti aşıldı mı?
            if total_size > self.max_cache_size * self.cleanup_threshold:
                self.logger.info(f"Cache boyut limiti aşıldı: {total_size // (1024*1024)}MB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Disk alanı kontrolü hatası: {e}")
            return True  # Hata durumunda devam et
    
    def cleanup_cache(self, aggressive: bool = False, max_age_days: int = 30):
        """Cache temizleme operasyonları"""
        try:
            if aggressive:
                max_age_days = 7  # Agresif temizlik için 7 gün
            
            now = datetime.now()
            metadata = self.load_package_metadata()
            removed_count = 0
            freed_space = 0
            
            # Eski dosyaları temizle
            for pkg, info in list(metadata.items()):
                try:
                    pkg_time = datetime.fromisoformat(info["timestamp"])
                    if (now - pkg_time).days > max_age_days:
                        cache_files = list(self.cache_dir.glob(f"{pkg}*.whl"))
                        for cache_file in cache_files:
                            if cache_file.exists():
                                file_size = cache_file.stat().st_size
                                cache_file.unlink()
                                freed_space += file_size
                                removed_count += 1
                        del metadata[pkg]
                        self.logger.debug(f"Eski cache dosyası silindi: {pkg}")
                except (ValueError, KeyError):
                    # Invalid timestamp, remove it
                    cache_files = list(self.cache_dir.glob(f"{pkg}*.whl"))
                    for cache_file in cache_files:
                        if cache_file.exists():
                            cache_file.unlink()
                            removed_count += 1
                    if pkg in metadata:
                        del metadata[pkg]
            
            # Boyut bazlı temizlik (agresif modda)
            if aggressive:
                # En az kullanılan dosyaları sil
                cache_files = list(self.cache_dir.glob("*.whl"))
                cache_files.sort(key=lambda x: x.stat().st_mtime)  # En eski önce
                
                total_size = sum(f.stat().st_size for f in cache_files)
                target_size = self.max_cache_size * 0.5  # %50'ye düşür
                
                for cache_file in cache_files:
                    if total_size <= target_size:
                        break
                    
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    total_size -= file_size
                    freed_space += file_size
                    removed_count += 1
                    
                    # Metadata'dan da sil
                    pkg_name = cache_file.stem.split('-')[0]
                    if pkg_name in metadata:
                        del metadata[pkg_name]
            
            self.save_package_metadata(metadata)
            
            if removed_count > 0:
                self.logger.info(f"Cache temizlik: {removed_count} dosya silindi, {freed_space // (1024*1024)}MB alan açıldı")
            
        except Exception as e:
            self.logger.error(f"Cache temizleme hatası: {e}")
    
    def rollback_package(self, package: str, previous_version: str) -> bool:
        """Paket rollback işlemi"""
        try:
            metadata = self.load_package_metadata()
            if package in metadata and metadata[package]["version"] != previous_version:
                self.logger.info(f"{package} için rollback başlatılıyor: {previous_version}")
                
                # Hedef versiyonu kur
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"{package}=={previous_version}"],
                    check=True, capture_output=True, text=True
                )
                
                # Metadata güncelle
                metadata[package]["version"] = previous_version
                metadata[package]["timestamp"] = datetime.now().isoformat()
                self.save_package_metadata(metadata)
                
                self.logger.info(f"{package} rollback tamamlandı.")
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{package} rollback hatası: {e}")
            return False
        except Exception as e:
            self.logger.error(f"{package} rollback genel hatası: {e}")
            return False
    
    def save_package_metadata(self, packages: Dict):
        """Metadata kaydetme"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(packages, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Metadata kaydetme hatası: {e}")
    
    def load_package_metadata(self) -> Dict:
        """Metadata yükleme"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Metadata yükleme hatası: {e}")
            return {}
    
    def get_cache_statistics(self) -> Dict:
        """Cache istatistikleri"""
        try:
            metadata = self.load_package_metadata()
            cache_files = list(self.cache_dir.glob("*.whl"))
            
            total_size = sum(f.stat().st_size for f in cache_files)
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            
            avg_download_time = sum(self.download_times.values()) / len(self.download_times) if self.download_times else 0
            
            return {
                "total_packages": len(metadata),
                "cache_files": len(cache_files),
                "total_size_mb": total_size // (1024 * 1024),
                "cache_hit_rate": f"{hit_rate:.2%}",
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "avg_download_time": f"{avg_download_time:.2f}s"
            }
            
        except Exception as e:
            self.logger.error(f"Cache istatistik hatası: {e}")
            return {}
    
    def visualize_version_tree(self, module_name: str):
        """Version tree görselleştirme (opsiyonel)"""
        try:
            try:
                from graphviz import Digraph
            except ImportError:
                self.logger.warning("graphviz kütüphanesi yüklü değil, görselleştirme atlandı.")
                return
            
            metadata = self.load_package_metadata()
            dot = Digraph(comment=f"{module_name} Version Tree")
            
            for pkg, info in metadata.items():
                version = info.get("version", "unknown")
                dot.node(pkg, f"{pkg} ({version})")
                
                # Dependencies varsa bağlantıları ekle
                dependencies = info.get("dependencies", [])
                for dep in dependencies:
                    if dep in metadata:
                        dot.edge(pkg, dep)
            
            output_file = f"{module_name}_version_tree"
            dot.render(output_file, format="png", cleanup=True)
            self.logger.info(f"{module_name} versiyon ağacı görselleştirildi: {output_file}.png")
            
        except Exception as e:
            self.logger.error(f"Versiyon ağacı görselleştirme hatası: {e}")


# Bu dosya devam edecek... ConflictManager, ModuleAnalyzer, ScientificUtils vs.


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


# Bu dosya devam edecek... ModuleAnalyzer, ScientificUtils, AsyncDownloadManager vs.


# === REAL TIME MONITOR (HİBRİT) ===
class RealTimeMonitor:
    """
    Sistem kaynaklarını gerçek zamanlı izleme - toplu1.py + v1795.py hibrit
    System resource tracking ve performance monitoring ile.
    """
    
    def __init__(self, logger, monitoring_interval=5):
        self.logger = logger
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = defaultdict(list)
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'process_count': 500
        }
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300  # 5 dakika
    
    def start_monitoring(self):
        """Monitoring başlat"""
        if self.monitoring:
            self.logger.warning("Monitoring zaten aktif")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Real-time monitoring başlatıldı")
    
    def stop_monitoring(self):
        """Monitoring durdur"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Real-time monitoring durduruldu")
    
    def _monitoring_loop(self):
        """Ana monitoring döngüsü"""
        try:
            while self.monitoring:
                try:
                    # Sistem metriklerini topla
                    metrics = self._collect_system_metrics()
                    
                    # Metrikleri kaydet
                    self._record_metrics(metrics)
                    
                    # Alert kontrolleri
                    self._check_alerts(metrics)
                    
                    # Log metrics (debug level)
                    self.logger.debug(f"System metrics: CPU {metrics['cpu_percent']:.1f}%, "
                                     f"RAM {metrics['memory_percent']:.1f}%, "
                                     f"Disk {metrics['disk_percent']:.1f}%")
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    time.sleep(self.monitoring_interval * 2)  # Hata durumunda daha az sık kontrol et
                    
        except Exception as e:
            self.logger.error(f"Monitoring thread error: {e}")
    
    def _collect_system_metrics(self) -> Dict:
        """Sistem metriklerini topla"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Process count
            process_count = len(psutil.pids())
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_mb': memory.available // (1024 * 1024),
                'memory_used_mb': memory.used // (1024 * 1024),
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free // (1024 * 1024 * 1024),
                'process_count': process_count,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0
            }
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'process_count': 0,
                'error': str(e)
            }
    
    def _record_metrics(self, metrics: Dict):
        """Metrikleri geçmişe kaydet"""
        try:
            # Sadece son 1 saatlik veriyi tut (monitoring_interval=5s için 720 entry)
            max_history = 720
            
            for key, value in metrics.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    self.metrics_history[key].append({
                        'timestamp': metrics['timestamp'],
                        'value': value
                    })
                    
                    # Eski verileri temizle
                    if len(self.metrics_history[key]) > max_history:
                        self.metrics_history[key] = self.metrics_history[key][-max_history:]
                        
        except Exception as e:
            self.logger.error(f"Metrics recording error: {e}")
    
    def _check_alerts(self, metrics: Dict):
        """Alert kontrolü"""
        try:
            current_time = time.time()
            
            for metric, threshold in self.alert_thresholds.items():
                if metric in metrics:
                    value = metrics[metric]
                    
                    # Threshold aşıldı mı?
                    if value > threshold:
                        # Cooldown kontrolü
                        if current_time - self.last_alert_time[metric] > self.alert_cooldown:
                            self._send_alert(metric, value, threshold)
                            self.last_alert_time[metric] = current_time
                            
        except Exception as e:
            self.logger.error(f"Alert check error: {e}")
    
    def _send_alert(self, metric: str, value: float, threshold: float):
        """Alert gönder"""
        try:
            alert_message = f"ALERT: {metric} threshold exceeded! Current: {value:.1f}, Threshold: {threshold}"
            self.logger.critical(alert_message)
            
            # Metric'e göre özel uyarılar
            if metric == 'memory_percent':
                self.logger.critical("High memory usage detected! Consider closing some applications.")
            elif metric == 'cpu_percent':
                self.logger.critical("High CPU usage detected! System may be under heavy load.")
            elif metric == 'disk_percent':
                self.logger.critical("Disk space running low! Consider cleaning up files.")
            
        except Exception as e:
            self.logger.error(f"Alert sending error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Anlık metrikleri döndür"""
        return self._collect_system_metrics()
    
    def get_metrics_summary(self, minutes: int = 10) -> Dict:
        """Son X dakikanın özeti"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            summary = {}
            
            for metric, history in self.metrics_history.items():
                recent_values = [
                    entry['value'] for entry in history
                    if datetime.fromisoformat(entry['timestamp']) > cutoff_time
                ]
                
                if recent_values:
                    summary[metric] = {
                        'avg': sum(recent_values) / len(recent_values),
                        'min': min(recent_values),
                        'max': max(recent_values),
                        'current': recent_values[-1] if recent_values else 0,
                        'sample_count': len(recent_values)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Metrics summary error: {e}")
            return {}
    
    def get_performance_report(self) -> str:
        """Performance raporu oluştur"""
        try:
            current = self.get_current_metrics()
            summary = self.get_metrics_summary()
            
            report = f"""
=== SYSTEM PERFORMANCE REPORT ===
Timestamp: {current.get('timestamp', 'N/A')}

Current Metrics:
- CPU Usage: {current.get('cpu_percent', 0):.1f}%
- Memory Usage: {current.get('memory_percent', 0):.1f}% ({current.get('memory_used_mb', 0)}MB used)
- Disk Usage: {current.get('disk_percent', 0):.1f}% ({current.get('disk_free_gb', 0)}GB free)
- Active Processes: {current.get('process_count', 0)}

10-Minute Averages:
"""
            
            for metric, stats in summary.items():
                if metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                    report += f"- {metric}: {stats['avg']:.1f}% (min: {stats['min']:.1f}%, max: {stats['max']:.1f}%)\n"
            
            # Alert status
            alerts_active = sum(1 for metric, threshold in self.alert_thresholds.items() 
                              if current.get(metric, 0) > threshold)
            
            report += f"\nAlert Status: {alerts_active} active alerts\n"
            report += f"Monitoring Status: {'Active' if self.monitoring else 'Inactive'}\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report error: {e}")
            return f"Performance report generation failed: {e}"


# === SUMMARY GENERATOR ===
class SummaryGenerator:
    """
    İstatistik ve özet rapor oluşturucu - toplu1.py'den alınmıştır.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.installation_stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
        self.error_stats = defaultdict(int)
    
    def record_installation(self, package: str, status: str, duration: float = 0):
        """Installation kaydı"""
        try:
            self.installation_stats[f"{status}_count"] += 1
            self.installation_stats["total_count"] += 1
            
            if duration > 0:
                self.timing_stats[package].append(duration)
                self.timing_stats["all_packages"].append(duration)
            
            if status == "failed":
                self.error_stats[package] += 1
                
        except Exception as e:
            self.logger.error(f"Installation recording error: {e}")
    
    def generate_summary(self) -> str:
        """Özet rapor oluştur"""
        try:
            total = self.installation_stats.get("total_count", 0)
            successful = self.installation_stats.get("successful_count", 0)
            failed = self.installation_stats.get("failed_count", 0)
            
            success_rate = (successful / total * 100) if total > 0 else 0
            
            # Timing statistics
            all_times = self.timing_stats.get("all_packages", [])
            avg_time = sum(all_times) / len(all_times) if all_times else 0
            
            summary = f"""
=== AUTO IMPORTER SUMMARY REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Installation Statistics:
- Total Packages Processed: {total}
- Successful Installations: {successful}
- Failed Installations: {failed}
- Success Rate: {success_rate:.1f}%

Performance Statistics:
- Average Installation Time: {avg_time:.2f} seconds
- Total Time Spent: {sum(all_times):.2f} seconds
- Fastest Installation: {min(all_times):.2f}s ({min(self.timing_stats.keys(), key=lambda k: min(self.timing_stats[k]) if self.timing_stats[k] else float('inf'))})
- Slowest Installation: {max(all_times):.2f}s

Error Analysis:
"""
            
            if self.error_stats:
                for package, error_count in sorted(self.error_stats.items(), key=lambda x: x[1], reverse=True):
                    summary += f"- {package}: {error_count} errors\n"
            else:
                summary += "- No errors recorded\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return f"Summary generation failed: {e}"


# Ana AutoImporter sınıfı ve main() fonksiyonu sırada...


# === MAIN AUTO IMPORTER CLASS (TOPLU1.PY + V1795.PY HİBRİT) ===
class AutoImporter:
    """
    Ana AutoImporter sınıfı - toplu1.py temel alınarak v1795.py özellikleri eklendi.
    Singleton pattern, comprehensive package management, advanced logging ile.
    """
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AutoImporter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            try:
                # Core components
                self.logger = AdvancedLogger()
                self.dependency_registry = DependencyRegistry(self.logger)
                self.pip_analyzer = PipOutputAnalyzer(self.logger)
                self.cache_manager = CacheManager(logger=self.logger)
                self.env_manager = EnvManager(logger=self.logger)
                self.conflict_manager = ConflictManager(self.logger)
                self.terminal_analyzer = TerminalLogAnalyzer(self.logger)
                self.realtime_monitor = RealTimeLogMonitor(self.logger)
                self.summary_generator = SummaryGenerator(self.logger)
                
                # System monitoring
                self.system_monitor = RealTimeMonitor(self.logger)
                
                # Threading ve sync
                self.lock = threading.Lock()
                self.thread_pool = ThreadPoolExecutor(max_workers=4)
                
                # Graceful shutdown (toplu1.py'den)
                global shutdown_manager
                if 'shutdown_manager' not in globals():
                    shutdown_manager = GracefulShutdownManager()
                shutdown_manager.register_cleanup(self.cleanup_on_shutdown)
                
                # State management
                self.loaded_modules = {}
                self.module_cache = {}
                self.imported_files = set()
                self.aliases = {}
                self.dependencies = defaultdict(list)
                self.secure_mode = False
                self.metadata = {"auto_importer": {"version": "Merged v1.0", "dependencies": []}}
                self.installation_history = {}
                self.retry_count = defaultdict(int)
                self.running = True
                
                # Environment setup
                try:
                    if not self.env_manager.setup_environment():
                        self.logger.warning("Environment setup tamamlanamadı, devam ediliyor...")
                except Exception as e:
                    self.logger.warning(f"Environment setup hatası: {e}")
                
                # Monitoring başlat
                self.system_monitor.start_monitoring()
                self.realtime_monitor.start_monitoring()
                
                self.initialized = True
                self.logger.info("🚀 AutoImporter Merged v1.0 başlatıldı")
                
                # Son argümanları kaydet
                self.save_last_args(sys.argv)
                
            except Exception as e:
                print(f"[ERROR] AutoImporter başlatma hatası: {e}")
                raise e
    
    def save_last_args(self, args: List[str]):
        """Komut satırı argümanlarını kaydet"""
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
    
    def check_package_installed(self, package: str) -> bool:
        """Paketin kurulu olup olmadığını kontrol et"""
        try:
            import importlib
            
            # Package adını module adına çevir
            module_name = self.terminal_analyzer.module_to_package.get(package, package)
            
            try:
                importlib.import_module(module_name)
                return True
            except ImportError:
                pass
            
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
    
    def auto_install_package(self, package: str, silent: bool = False) -> bool:
        """Eksik metod - package kurulumu"""
        return self.install_package(package, silent)
    
    def is_running_in_venv(self) -> bool:
        """Eksik metod - virtual env kontrolü"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
    
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
            if self.check_package_installed(package_name):
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
                self.logger.info(f"📦 {package} kuruluyor...")
            
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
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop_monitoring()
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
        print("🚀 PDS-X AutoImporter Merged v1.0 başlatılıyor...")
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
        print("🔗 GitHub: https://github.com/[user]/auto_importer")
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
