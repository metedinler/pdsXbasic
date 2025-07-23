# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.5 - Merged Features: Terminal Log Analysis, Argparse Replay, Enhanced Dependencies, Real-time Monitoring
# Date: June 25, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot)

import os
import sys
import subprocess
import shutil
import importlib.util
import logging
import threading
import json
import time
import signal
import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import hashlib
import asyncio
import psutil
import re

# Lazy imports - sadece gerektiğinde yüklenecek
numpy = None
IsolationForest = None
StandardScaler = None
MLPClassifier = None
DecisionTreeClassifier = None
keyboard = None

def get_numpy():
    """NumPy'ı lazy loading ile yükle"""
    global numpy
    if numpy is None:
        try:
            import numpy as np
            numpy = np
        except ImportError:
            print("[PDS-X] NumPy yüklenemedi, sanal ortam kurulumu gerekli.")
            return None
    return numpy

def get_sklearn_components():
    """Sklearn bileşenlerini lazy loading ile yükle"""
    global IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier
    if IsolationForest is None:
        try:
            from sklearn.ensemble import IsolationForest as IF
            from sklearn.preprocessing import StandardScaler as SS
            from sklearn.neural_network import MLPClassifier as MLP
            from sklearn.tree import DecisionTreeClassifier as DTC
            IsolationForest = IF
            StandardScaler = SS
            MLPClassifier = MLP
            DecisionTreeClassifier = DTC
        except ImportError:
            print("[PDS-X] Scikit-learn yüklenemedi, sanal ortam kurulumu gerekli.")
            return None, None, None, None
    return IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier

def get_keyboard():
    """Keyboard modülünü lazy loading ile yükle"""
    global keyboard
    if keyboard is None:
        try:
            import keyboard as kb
            keyboard = kb
        except ImportError:
            print("[PDS-X] Keyboard modülü yüklenemedi.")
            return None
    return keyboard

# Graceful Shutdown Yönetimi
class GracefulShutdownManager:
    def __init__(self):
        self.shutdown_requested = False
        self.active_processes = []
        self.cleanup_functions = []
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Sinyal handler'larını kurar"""
        try:
            # SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self.signal_handler)
            # SIGTERM (normal termination)
            signal.signal(signal.SIGTERM, self.signal_handler)
            # Windows için SIGBREAK (Ctrl+Break)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self.signal_handler)
            # Program çıkışında temizlik
            atexit.register(self.cleanup)
            
            # Keyboard kill switch (Ctrl+Shift+Q) kurun
            self.setup_keyboard_kill_switch()
            
            print("[PDS-X] Graceful shutdown sistemi aktif (Ctrl+C veya Ctrl+Shift+Q ile güvenli çıkış)")
        except Exception as e:
            print(f"[PDS-X] Signal handler kurulum hatası: {e}")
    
    def setup_keyboard_kill_switch(self):
        """Ctrl+Shift+Q kombinasyonu için listener kur"""
        try:
            keyboard = get_keyboard()
            if keyboard:
                # Hotkey listener kurulumu
                keyboard.add_hotkey('ctrl+shift+q', self.emergency_shutdown, suppress=True)
                print("[PDS-X] Keyboard kill switch aktif: Ctrl+Shift+Q")
            else:
                print("[PDS-X] Keyboard modülü yüklenemedi, kill switch devre dışı")
        except Exception as e:
            print(f"[PDS-X] Keyboard kill switch kurulum hatası: {e}")
    
    def emergency_shutdown(self):
        """Acil kapatma işlemi (Ctrl+Shift+Q)"""
        try:
            print("\n[PDS-X] ACIL KAPATMA: Ctrl+Shift+Q kombinasyonu algılandı!")
            print("[PDS-X] Sistem güvenli şekilde kapatılıyor...")
            
            self.shutdown_requested = True
            
            # Hızlı temizlik işlemleri
            self.emergency_cleanup()
            
            # Sistem çıkışı
            print("[PDS-X] Acil kapatma tamamlandı.")
            os._exit(0)
            
        except Exception as e:
            print(f"[PDS-X] Acil kapatma hatası: {e}")
            os._exit(1)
    
    def emergency_cleanup(self):
        """Acil kapatma için hızlı temizlik"""
        try:
            print("[PDS-X] Acil temizlik başlatılıyor...")
            
            # Sadece kritik process'leri sonlandır
            for process in self.active_processes[-5:]:  # Son 5 process
                try:
                    if hasattr(process, 'poll') and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1)  # Daha kısa timeout
                        except subprocess.TimeoutExpired:
                            process.kill()
                except Exception:
                    pass  # Hızlı geçiş
            
            print("[PDS-X] Acil temizlik tamamlandı.")
            
        except Exception as e:
            print(f"[PDS-X] Acil temizlik hatası: {e}")
    
    def disable_keyboard_kill_switch(self):
        """Keyboard kill switch'i devre dışı bırak"""
        try:
            keyboard = get_keyboard()
            if keyboard:
                keyboard.remove_hotkey('ctrl+shift+q')
                print("[PDS-X] Keyboard kill switch devre dışı bırakıldı")
        except Exception as e:
            print(f"[PDS-X] Kill switch devre dışı bırakma hatası: {e}")
    
    def signal_handler(self, signum, frame):
        """Sinyal geldiğinde çalışır"""
        print(f"\n[PDS-X] Shutdown sinyali alındı (signal: {signum})")
        print("[PDS-X] Güvenli çıkış başlatılıyor...")
        self.shutdown_requested = True
        self.cleanup()
        
    def register_process(self, process):
        """Aktif process'i kaydet"""
        self.active_processes.append(process)
        
    def register_cleanup_function(self, func):
        """Temizlik fonksiyonu kaydet"""
        self.cleanup_functions.append(func)
        
    def cleanup(self):
        """Temizlik işlemlerini yapar"""
        try:
            print("[PDS-X] Temizlik işlemleri başlatılıyor...")
            
            # Aktif process'leri sonlandır
            for process in self.active_processes:
                try:
                    if hasattr(process, 'poll') and process.poll() is None:  # Hala çalışıyor
                        print(f"[PDS-X] Process sonlandırılıyor: {process.pid}")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            print(f"[PDS-X] Process zorla sonlandırılıyor: {process.pid}")
                            process.kill()
                except Exception as e:
                    print(f"[PDS-X] Process sonlandırma hatası: {e}")
            
            # Cleanup fonksiyonlarını çalıştır
            for cleanup_func in self.cleanup_functions:
                try:
                    cleanup_func()
                except Exception as e:
                    print(f"[PDS-X] Cleanup fonksiyonu hatası: {e}")
                    
            print("[PDS-X] Temizlik tamamlandı.")
            
        except Exception as e:
            print(f"[PDS-X] Temizlik hatası: {e}")
        finally:
            # Programı sonlandır
            if self.shutdown_requested:
                print("[PDS-X] Program güvenli şekilde sonlandırılıyor...")
                os._exit(0)

# Global shutdown manager
shutdown_manager = GracefulShutdownManager()

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
except ImportError:
    Fore = Style = type('Dummy', (), {'__getattr__': lambda self, name: ''})()

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

VENV_DIR = Path(".pdsx_isolated_env")
CACHE_DIR = Path(".pdsx_cache")
LOG_DIR = Path("logs")
TERMINAL_LOG = LOG_DIR / "pdsxu_terminal.jsonl"
INFO_LOG = LOG_DIR / "pdsxu_info.jsonl"
WARNING_LOG = LOG_DIR / "pdsxu_warnings.jsonl"
ERROR_LOG = LOG_DIR / "pdsxu_errors.jsonl"
PLAIN_TERMINAL_LOG = LOG_DIR / "pdsXu_terminal.log"
LAST_ARGS_FILE = Path(".pdsx_last_args.json")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BACKUPS = 5

# Terminal Log Analysis Regex Patterns
MODULE_NOT_FOUND_REGEX = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", re.IGNORECASE)
PIP_SUGGESTION_REGEX = re.compile(r"pip install ([^\s=]+)(?:==([^\s]+))?", re.IGNORECASE)
IMPORT_ERROR_REGEX = re.compile(r"ImportError.*?import ([^\s]+) from ([^\s]+)", re.IGNORECASE)

# Bağımlılık Listesi
REQUIRED_PACKAGES = [
    ("numpy==1.26.4", "numpy"), ("scipy==1.11.4", "scipy"), ("pandas==2.1.4", "pandas"),
    ("scikit-learn==1.3.2", "sklearn"), ("joblib==1.3.2", "joblib"), ("threadpoolctl==3.3.0", "threadpoolctl"),
    ("matplotlib==3.8.4", "matplotlib"), ("kiwisolver==1.4.5", "kiwisolver"), ("cycler==0.12.1", "cycler"),
    ("pyparsing==3.1.1", "pyparsing"), ("python-dateutil==2.8.2", "dateutil"), ("pillow==10.2.0", "PIL"),
    ("packaging==23.2", "packaging"), ("seaborn==0.13.2", "seaborn"), ("statsmodels==0.14.1", "statsmodels"),
    ("tornado==6.4", "tornado"), ("plotly==5.19.0", "plotly"), ("tenacity==8.2.3", "tenacity"),
    ("dash==2.15.0", "dash"), ("flask==3.0.2", "flask"), ("jinja2==3.1.3", "jinja2"),
    ("werkzeug==3.0.1", "werkzeug"), ("itsdangerous==2.1.2", "itsdangerous"), ("markupsafe==2.1.5", "markupsafe"),
    ("click==8.1.7", "click"), ("grpcio==1.62.0", "grpc"), ("protobuf==4.25.3", "google.protobuf"),
    ("aiohttp==3.9.3", "aiohttp"), ("async-timeout==4.0.3", "async_timeout"), ("yarl==1.9.4", "yarl"),
    ("multidict==6.0.5", "multidict"), ("attrs==23.2.0", "attr"), ("frozenlist==1.4.1", "frozenlist"),
    ("pyzmq==25.1.2", "zmq"), ("websocket-client==1.7.0", "websocket"), ("paho-mqtt==2.1.0", "paho.mqtt.client"),
    ("boto3==1.38.37", "boto3"), ("botocore==1.38.37", "botocore"), ("kafka-python==2.0.2", "kafka"),
    ("river==0.21.0", "river"), ("qiskit==1.0.1", "qiskit"), ("networkx==3.2.1", "networkx"),
    ("websockets==12.0", "websockets"), ("rich==13.7.0", "rich"), ("colorama==0.4.6", "colorama"),
    ("textblob==0.17.1", "textblob"), ("mysql-connector-python==8.3.0", "mysql.connector"),
    ("psutil==5.9.8", "psutil"), ("pyyaml==6.0.1", "yaml"), ("graphviz==0.20.1", "graphviz"),
    ("aiofiles==23.2.1", "aiofiles"), ("RestrictedPython>=6.2,<8.0", "RestrictedPython"),
    ("pdfplumber==0.10.3", "pdfplumber"), ("requests==2.32.4", "requests"), ("psycopg2-binary==2.9.9", "psycopg2"),
    ("elasticsearch==8.12.0", "elasticsearch"), ("elastic-transport==8.12.0", "elastic_transport"),
    ("nltk==3.9.1", "nltk"), ("tensorflow==2.15.0", "tensorflow"), ("torch==2.2.2", "torch"),
    ("torch-geometric==2.5.3", "torch_geometric"), ("spacy==3.5.3", "spacy"), ("transformers==4.52.4", "transformers"),
    ("pycryptodome==3.20.0", "Crypto"), ("jmespath==1.0.1", "jmespath"), ("pdfminer.six==20250327", "pdfminer"),
    ("pypdfium2>=4.18.0", "pypdfium2"), ("markdown-it-py>=2.2.0", "markdown_it_py"),
    ("pygments>=2.13.0,<3.0.0", "pygments"), ("catalogue<2.1.0,>=2.0.6", "catalogue"),
    ("cymem<2.1.0,>=2.0.2", "cymem"), ("langcodes<4.0.0,>=3.2.0", "langcodes"),
    ("murmurhash<1.1.0,>=0.28.0", "murmurhash"), ("preshed<3.1.0,>=3.0.2", "preshed"),
    ("pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4", "pydantic"), ("spacy-legacy<3.1.0,>=3.0.11", "spacy_legacy"),
    ("spacy-loggers<2.0.0,>=1.0.0", "spacy_loggers"), ("srsly<3.0.0,>=2.4.3", "srsly"),
    ("thinc<8.4.0,>=8.3.2", "thinc"), ("typer<1.0.0,>=0.3.0", "typer"), ("wasabi<1.2.0,>=0.9.1", "wasabi"),
    ("weasel<0.5.0,>=0.1.0", "weasel"), ("huggingface-hub<1.0,>=0.30.0", "huggingface_hub"),
    ("regex!=2019.12.17", "regex"), ("safetensors>=0.4.3", "safetensors"), ("tokenizers<0.22,>=0.21", "tokenizers"),
    ("cryptography==42.0.5", "cryptography"), ("cffi==1.16.0", "cffi"), ("pycparser==2.21", "pycparser"),
    ("six==1.16.0", "six"), ("pyasn1==0.5.1", "pyasn1"), ("pyasn1-modules==0.3.0", "pyasn1_modules"),
    ("idna==3.10", "idna"), ("charset_normalizer==3.4.2", "charset_normalizer"), ("urllib3==2.4.0", "urllib3"),
    ("certifi==2025.6.15", "certifi"), ("chardet==5.2.0", "chardet"), ("blis<0.8.0,>=0.7.8", "blis"),
    ("confection<1.0.0,>=0.0.1", "confection"), ("shellingham>=1.3.0", "shellingham"),
    ("smart-open<7.0.0,>=5.2.1", "smart_open"), ("cloudpathlib<1.0.0,>=0.7.0", "cloudpathlib"),
    ("annotated-types>=0.6.0", "annotated_types"), ("pydantic-core==2.33.2", "pydantic_core"),
    ("typing-inspection>=0.4.0", "typing_inspection"), ("pytest>=6.2.0", "pytest"), ("black>=21.5b2", "black"),
    ("pylint>=2.8.0", "pylint"), ("mypy>=0.910", "mypy"), ("coverage>=6.0", "coverage"),
    ("filelock>=3.18.0", "filelock"), ("fsspec>=2023.5.0", "fsspec"), ("typing-extensions>=3.7.4.3", "typing_extensions"),
    ("tqdm>=4.67.1", "tqdm"), ("exceptiongroup>=1", "exceptiongroup"), ("iniconfig>=1", "iniconfig"),
    ("pluggy<2,>=1.5", "pluggy"), ("tomli>=1", "tomli"), ("astroid<=3.4.0.dev0,>=3.3.8", "astroid"),
    ("dill>=0.2", "dill"), ("isort!=5.13,<7,>=4.2.5", "isort"), ("mccabe<0.8,>=0.6", "mccabe"),
    ("platformdirs>=2.2", "platformdirs"), ("tomlkit>=0.10.1", "tomlkit"), ("pycodestyle>=2.12.0", "pycodestyle"),
    ("autopep8>=2.3.2", "autopep8"), ("pathlib-abc==0.1.1", "pathlib_abc"), ("marisa-trie>=1.1.0", "marisa_trie"),
    ("setuptools>=80.9.0", "setuptools"), ("pytz>=2020.1", "pytz"), ("tzdata>=2022.1", "tzdata"),
    ("gensim==4.3.3", "gensim"), ("s3transfer<0.14.0,>=0.13.0", "s3transfer"), ("keyboard==0.13.5", "keyboard")
]

CORE_DEPENDENCIES = {
    "base": [pkg[0] for pkg in REQUIRED_PACKAGES],
    "optional": [],
}

MODULE_SPECIFIC_DEPS = {
    "core2-5.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
    "core2-6.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
    "libx_ml.py": ["torch==2.2.2", "transformers==4.52.4", "scikit-learn==1.3.2"],
    "libx_nlp.py": ["nltk==3.9.1", "spacy==3.5.3", "gensim==4.3.3"],
    "database_sql_isam.py": ["psycopg2-binary==2.9.9", "sqlite3"],
    "graph.py": ["networkx==3.2.1", "graphviz==0.20.1"],
}

def find_python310():
       logger = AdvancedLogger()
       env_manager = EnvManager(logger=logger)
       return env_manager.find_python310()

async def install_missing_packages():
       importer = AutoImporter()
       for pkg, _ in REQUIRED_PACKAGES:
           await importer.async_install_package(pkg)
       importer.summary_generator.print_summary()

# Gelişmiş Loglama Sistemi
class AdvancedLogger:
    def __init__(self, silent_mode: bool = False):
        # Temel attribute'ları başlat
        self.logged_messages = set()
        self.last_log_time = {}
        self.es = None
        self.logger = None
        self.handlers = {}
        self.terminal_handler = None
        self.stdout_handler = None
        self.silent_mode = silent_mode  # Yeni özellik: sessiz mod
        
        try:
            # logs dizini kontrol et ve oluştur
            LOG_DIR.mkdir(exist_ok=True)
            
            # pdsXu_terminal.log için yedekleme sistemi
            self.plain_terminal_log_file = PLAIN_TERMINAL_LOG
            if self.plain_terminal_log_file.exists():
                backup_file = self.plain_terminal_log_file.with_suffix(".bak")
                if backup_file.exists():
                    backup_file.unlink()  # Eski yedeği sil
                shutil.move(self.plain_terminal_log_file, backup_file)
                print(f"[PDS-X] Önceki terminal log yedeklendi: {backup_file}")
            
            # Terminal log handler - düz metin, zaman damgalı
            self.terminal_handler = logging.FileHandler(self.plain_terminal_log_file, encoding="utf-8")
            self.terminal_handler.setLevel(logging.INFO)
            self.terminal_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            
            # JSONL log handlers - seviye bazlı
            self.handlers = {
                "info": logging.FileHandler(INFO_LOG, encoding="utf-8"),
                "warning": logging.FileHandler(WARNING_LOG, encoding="utf-8"),
                "error": logging.FileHandler(ERROR_LOG, encoding="utf-8"),
                "terminal": logging.FileHandler(TERMINAL_LOG, encoding="utf-8"),
            }
            
            # JSONL formatı
            jsonl_formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
            
            # Handler'ları yapılandır
            for level, handler in self.handlers.items():
                if level == "terminal":
                    handler.setLevel(logging.INFO)  # terminal için INFO level
                else:
                    handler.setLevel(getattr(logging, level.upper()))
                handler.setFormatter(jsonl_formatter)
            
            # Ana logger'ı oluştur
            self.logger = logging.getLogger("autoimporter")
            self.logger.setLevel(logging.DEBUG)
            
            # Tüm handler'ları ekle
            self.logger.addHandler(self.terminal_handler)
            for handler in self.handlers.values():
                self.logger.addHandler(handler)
            
            # Terminal yönlendirme sistemi
            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
            sys.stdout = self.stdout_handler
            sys.stderr = Tee(sys.__stderr__, self.handlers["error"])
              # Elasticsearch bağlantısı (opsiyonel)
            if Elasticsearch:
                try:
                    self.es = Elasticsearch(["http://localhost:9200"])
                    self.logger.info("Elasticsearch sunucusuna bağlanıldı.")
                except Exception as e:
                    self.logger.warning(f"Elasticsearch sunucusu erişilemez, yerel loglamaya geçiliyor: {e}")
            
            self.logger.info("Loglama sistemi başlatıldı.")
            
        except Exception as e:
            print(f"[PDS-X] Loglama başlatma hatası: {e}")

    def log(self, level: str, message: str):
        """Ana loglama fonksiyonu - silent mode desteği ile"""
        try:
            # Spam koruması
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            current_time = datetime.now()
            
            if (message_hash not in self.logged_messages or 
                (current_time - self.last_log_time.get(message_hash, datetime.min)).total_seconds() > 1):
                
                # Silent mode kontrolü
                if not self.silent_mode:
                    # Info mesajlarını terminal'e de yazdır (silent mode değilse)
                    if level.lower() == "info":
                        print(f"[PDS-X] {message}")
                elif level.lower() in ["error", "warning"]:
                    # Silent mode'da bile error ve warning'leri göster
                    print(f"[PDS-X] {level.upper()}: {message}")
                
                # Logger varsa log'la
                if self.logger:
                    log_level = getattr(logging, level.upper(), logging.INFO)
                    self.logger.log(log_level, message)
                
                # Spam koruması için kaydet
                self.logged_messages.add(message_hash)
                self.last_log_time[message_hash] = current_time
                
                # Elasticsearch'e gönder (varsa)
                if self.es:
                    try:
                        self.es.index(index="pdsx_logs", body={
                            "timestamp": current_time.isoformat(),
                            "level": level.upper(),
                            "message": message,
                            "silent_mode": self.silent_mode
                        })
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Elasticsearch bağlantısı başarısız: {e}")
                        self.es = None
                
                # Log rotasyonu kontrol et
                self.rotate_logs()
                
        except Exception as e:
            print(f"[PDS-X] Loglama hatası: {e}")

    def set_silent_mode(self, silent: bool):
        """Silent mode'u etkinleştir/devre dışı bırak"""
        self.silent_mode = silent
        if silent:
            print("[PDS-X] Sessiz mod etkinleştirildi - sadece hatalar ve uyarılar gösterilecek")
        else:
            print("[PDS-X] Sessiz mod devre dışı bırakıldı - tüm loglar gösterilecek")

    def rotate_logs(self):
        """Log dosyalarını boyut kontrolü ile döndür"""
        try:
            log_files = [self.plain_terminal_log_file, INFO_LOG, WARNING_LOG, ERROR_LOG, TERMINAL_LOG]
            
            for log_file in log_files:
                if log_file.exists() and log_file.stat().st_size > MAX_LOG_SIZE:
                    # Hangi handler'ı kapatacağımızı belirle
                    if log_file == self.plain_terminal_log_file:
                        handler = self.terminal_handler
                    else:
                        handler = self.handlers.get(log_file.stem)
                    
                    if handler:
                        # Handler'ı kapat
                        handler.close()
                        self.logger.removeHandler(handler)
                        
                        # Dosyayı yedekle
                        backup_path = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                        shutil.move(log_file, backup_path)
                        
                        # Yeni handler oluştur
                        new_handler = logging.FileHandler(log_file, encoding="utf-8")
                        new_handler.setFormatter(handler.formatter)
                        new_handler.setLevel(handler.level)
                        
                        # Handler'ı güncelle ve ekle
                        if log_file == self.plain_terminal_log_file:
                            self.terminal_handler = new_handler
                            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
                            sys.stdout = self.stdout_handler
                        else:
                            self.handlers[log_file.stem] = new_handler
                        
                        self.logger.addHandler(new_handler)
                        
                        # Eski yedekleri temizle
                        self.cleanup_old_backups(log_file)
                        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Log döndürme hatası: {e}")
            else:
                print(f"[PDS-X] Log döndürme hatası: {e}")

    def cleanup_old_backups(self, log_file: Path):
        """Eski log yedeklerini temizle"""
        try:
            backup_pattern = f"{log_file.name}.*.bak"
            backups = sorted(
                log_file.parent.glob(backup_pattern), 
                key=lambda x: x.stat().st_mtime, 
                reverse=True
            )
              # MAX_BACKUPS'tan fazla yedek varsa eskilerini sil
            for old_backup in backups[MAX_BACKUPS:]:
                old_backup.unlink()
                if self.logger:
                    self.logger.info(f"Eski yedek silindi: {old_backup}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Yedek silme hatası: {e}")
            else:
                print(f"[PDS-X] Yedek silme hatası: {e}")
        try:
            backups = sorted(log_file.parent.glob(f"{log_file.name}.*.bak"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[MAX_BACKUPS:]:
                old_backup.unlink()
                self.logger.info(f"Eski yedek silindi: {old_backup}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Yedek silme hatası: {e}")
            else:
                print(f"[PDS-X] Yedek silme hatası: {e}")

# Terminal ve Log Yönlendirme
class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            try:
                output.write(text)
                output.flush()
            except Exception:
                pass

    def flush(self):
        for output in self.outputs:
            try:
                output.flush()
            except Exception:
                pass

# Modül Kurulum Özet Sistemi
class ModuleSummaryGenerator:
    """
    Paket kurulum işlemlerinin özetini oluşturur ve yazdırır.
    Kurulum başarıları, hataları ve istatistikleri gösterir.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.installation_stats = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "conflicts": [],
            "total_packages": 0,
            "start_time": None,
            "end_time": None
        }
        self.reset_stats()
    
    def reset_stats(self):
        """İstatistikleri sıfırla"""
        self.installation_stats = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "conflicts": [],
            "total_packages": 0,
            "start_time": datetime.now(),
            "end_time": None
        }
    
    def add_success(self, package_name: str, version: str = None):
        """Başarılı kurulum kaydı"""
        self.installation_stats["successful"].append({
            "package": package_name,
            "version": version or "latest",
            "timestamp": datetime.now().isoformat()
        })
    
    def add_failure(self, package_name: str, error: str):
        """Başarısız kurulum kaydı"""
        self.installation_stats["failed"].append({
            "package": package_name,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_skipped(self, package_name: str, reason: str):
        """Atlanan paket kaydı"""
        self.installation_stats["skipped"].append({
            "package": package_name,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_conflict(self, package_name: str, conflict_info: str):
        """Çakışma kaydı"""
        self.installation_stats["conflicts"].append({
            "package": package_name,
            "conflict": conflict_info,
            "timestamp": datetime.now().isoformat()
        })
    
    def finalize_stats(self):
        """İstatistikleri sonlandır"""
        self.installation_stats["end_time"] = datetime.now()
        self.installation_stats["total_packages"] = (
            len(self.installation_stats["successful"]) +
            len(self.installation_stats["failed"]) +
            len(self.installation_stats["skipped"])
        )
    
    def print_summary(self):
        """Kurulum özetini yazdır"""
        try:
            self.finalize_stats()
            
            print("\n" + "="*60)
            print("📦 PDS-X PAKET KURULUM ÖZETİ")
            print("="*60)
            
            # Zaman bilgisi
            start_time = self.installation_stats["start_time"]
            end_time = self.installation_stats["end_time"]
            if start_time and end_time:
                duration = end_time - start_time
                print(f"⏱️  Kurulum Süresi: {duration.total_seconds():.2f} saniye")
            
            # Genel istatistikler
            total = self.installation_stats["total_packages"]
            successful = len(self.installation_stats["successful"])
            failed = len(self.installation_stats["failed"])
            skipped = len(self.installation_stats["skipped"])
            conflicts = len(self.installation_stats["conflicts"])
            
            print(f"📊 Toplam İşlem: {total} paket")
            print(f"✅ Başarılı: {successful}")
            print(f"❌ Başarısız: {failed}")
            print(f"⏭️  Atlanan: {skipped}")
            if conflicts > 0:
                print(f"⚠️  Çakışma: {conflicts}")
            
            # Başarılı kurulumlar
            if successful > 0:
                print(f"\n✅ BAŞARILI KURULUMLAR ({successful} adet):")
                for pkg in self.installation_stats["successful"]:
                    version_info = f" v{pkg['version']}" if pkg['version'] != 'latest' else ""
                    print(f"   ✓ {pkg['package']}{version_info}")
            
            # Başarısız kurulumlar
            if failed > 0:
                print(f"\n❌ BAŞARISIZ KURULUMLAR ({failed} adet):")
                for pkg in self.installation_stats["failed"]:
                    print(f"   ✗ {pkg['package']}: {pkg['error']}")
            
            # Atlanan paketler
            if skipped > 0:
                print(f"\n⏭️  ATLANAN PAKETLER ({skipped} adet):")
                for pkg in self.installation_stats["skipped"]:
                    print(f"   - {pkg['package']}: {pkg['reason']}")
            
            # Çakışmalar
            if conflicts > 0:
                print(f"\n⚠️  ÇAKIŞMALAR ({conflicts} adet):")
                for conflict in self.installation_stats["conflicts"]:
                    print(f"   ! {conflict['package']}: {conflict['conflict']}")
            
            # Başarı oranı
            if total > 0:
                success_rate = (successful / total) * 100
                print(f"\n📈 Başarı Oranı: {success_rate:.1f}%")
                
                if success_rate >= 90:
                    print("🎉 Mükemmel! Kurulum başarıyla tamamlandı.")
                elif success_rate >= 70:
                    print("👍 İyi! Çoğu paket başarıyla kuruldu.")
                elif success_rate >= 50:
                    print("😐 Orta. Bazı paketlerde sorun var.")
                else:
                    print("😞 Dikkat! Birçok pakette sorun var.")
            
            print("="*60)
            print("📝 Detaylı log için pdsXu_errors.log dosyasını kontrol edin.")
            print("="*60 + "\n")
            
            # Log'a da kaydet
            self.logger.log("info", f"Kurulum özeti: {successful} başarılı, {failed} başarısız, {skipped} atlanan")
            
        except Exception as e:
            self.logger.log("error", f"Özet yazdırma hatası: {e}")
            print(f"❌ Özet yazdırırken hata: {e}")

# Ana Yükleyici
# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9
# Date: June 21, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot)

# [Diğer import'lar ve sınıflar burada tanımlı, yalnızca AutoImporter sınıfı güncelleniyor]

class AutoImporter:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AutoImporter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            try:
                self.logger = AdvancedLogger()
                self.dependency_registry = DependencyRegistry(self.logger)
                self.pip_analyzer = PipOutputAnalyzer(self.logger)
                self.cache_manager = CacheManager(logger=self.logger)
                self.env_manager = EnvManager(logger=self.logger)
                self.conflict_manager = ConflictManager(self.logger)
                self.module_analyzer = ModuleAnalyzer(self.logger)
                self.downloader = AsyncDownloadManager(logger=self.logger)
                self.summary_generator = ModuleSummaryGenerator(self.logger)
                
                # Yeni özellikler: Terminal log analizi ve real-time monitoring
                self.terminal_analyzer = TerminalLogAnalyzer(self.logger)
                self.realtime_monitor = RealTimeLogMonitor(self.logger, self.terminal_analyzer)
                
                self.lock = threading.Lock()
                
                # Graceful shutdown için cleanup fonksiyonu kaydet
                shutdown_manager.register_cleanup_function(self.cleanup_on_shutdown)
                  # Diğer attribute'lar
                self.loaded_modules = {}
                self.module_cache = {}
                self.imported_files = set()
                self.aliases = {}
                self.dependencies = defaultdict(list)
                self.secure_mode = False
                self.offline_cache = None
                self.metadata = {"auto_importer": {"version": "1.7.9.5", "dependencies": []}}
                self.installation_history = {}
                self.retry_count = {}
                self.running = True
                
                # Klavye kombinasyonu dinleyicisi (Ctrl+Shift+Q ile durdurma)
                try:
                    keyboard = get_keyboard()
                    if keyboard:
                        keyboard.on_press_key("q", self.check_stop_combination, suppress=True)
                        self.logger.log("info", "Klavye durdurma kombinasyonu (Ctrl+Shift+Q) etkinleştirildi")
                except Exception as e:
                    self.logger.log("warning", f"Klavye kombinasyonu kurulum hatası: {e}")
                
                # PATH güncellemesi
                try:
                    self.env_manager.add_python_to_path()
                except Exception as e:
                    self.logger.log("warning", f"PATH güncelleme hatası: {e}")
                  # Build tools kontrolü
                try:
                    self.check_build_tools()
                except Exception as e:
                    self.logger.log("warning", f"Build tools kontrol hatası: {e}")
                
                self.initialized = True
                self.logger.log("info", "AutoImporter başlatıldı")
                
                # Son argümanları kaydet
                self.save_last_args(sys.argv)
                
            except Exception as e:
                print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
                raise e

    def save_last_args(self, args: List[str]):
        """Komut satırı argümanlarını .pdsx_last_args.json'a kaydeder"""
        try:
            args_data = {
                "command": args,
                "timestamp": datetime.now().isoformat(),
                "working_directory": os.getcwd(),
                "python_executable": sys.executable
            }
            with open(LAST_ARGS_FILE, "w", encoding="utf-8") as f:
                json.dump(args_data, f, indent=4, ensure_ascii=False)
            self.logger.log("info", f"Komut argümanları kaydedildi: {args}")
        except Exception as e:
            self.logger.log("error", f"Komut argümanları kaydetme hatası: {e}")

    def load_last_args(self) -> Optional[Dict]:
        """Kaydedilmiş argümanları .pdsx_last_args.json'dan yükler"""
        try:
            if LAST_ARGS_FILE.exists():
                with open(LAST_ARGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.logger.log("info", f"Son argümanlar yüklendi: {data.get('command', [])}")
                    return data
            return None
        except Exception as e:
            self.logger.log("error", f"Komut argümanları yükleme hatası: {e}")
            return None

    def replay_last_command(self) -> bool:
        """Son kaydedilmiş komutu tekrar çalıştırır"""
        try:
            last_args_data = self.load_last_args()
            if not last_args_data:
                self.logger.log("warning", "Tekrar oynatılacak komut bulunamadı")
                return False
            
            last_command = last_args_data.get("command", [])
            last_working_dir = last_args_data.get("working_directory", os.getcwd())
            last_python = last_args_data.get("python_executable", sys.executable)
            
            self.logger.log("info", f"Komut tekrar oynatılıyor: {' '.join(last_command)}")
            self.logger.log("info", f"Çalışma dizini: {last_working_dir}")
            
            # Çalışma dizinini değiştir (eğer farklıysa)
            current_dir = os.getcwd()
            if os.path.exists(last_working_dir) and last_working_dir != current_dir:
                os.chdir(last_working_dir)
                self.logger.log("info", f"Çalışma dizini değiştirildi: {last_working_dir}")
              # Komutu çalıştır
            if len(last_command) > 1:
                # İlk eleman script adı, geri kalanı argümanlar
                result = subprocess.run([last_python] + last_command, capture_output=False, text=True)
                success = result.returncode == 0
                
                if success:
                    self.logger.log("info", "Komut tekrarı başarıyla tamamlandı")
                else:
                    self.logger.log("error", f"Komut tekrarı başarısız: return code {result.returncode}")
                
                return success
            else:
                self.logger.log("warning", "Tekrar oynatılacak yeterli argüman yok")
                return False
                
        except Exception as e:
            self.logger.log("error", f"Komut tekrarı hatası: {e}")
            return False
        finally:
            # Çalışma dizinini geri değiştir
            if 'current_dir' in locals() and current_dir != os.getcwd():
                os.chdir(current_dir)

    # Eksik metodlar - Demo'da kullanılan metodlar
    def check_package_installed(self, package: str) -> bool:
        """Paketin kurulu olup olmadığını kontrol eder"""
        try:
            # Önce importlib ile kontrol et
            import importlib
            # Paket adını module adına çevir
            module_name = self.terminal_analyzer.package_to_module.get(package, package)
            
            # Built-in modüller için
            if module_name in self.terminal_analyzer.builtin_modules:
                return True
                
            # Kurulu paketleri kontrol et
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
            self.logger.log("error", f"Paket kontrol hatası ({package}): {e}")
            return False
    
    def auto_install_package(self, package: str) -> bool:
        """Paketi otomatik olarak kurar"""
        try:
            if self.check_package_installed(package):
                self.logger.log("info", f"{package} zaten kurulu")
                return True
                
            self.logger.log("info", f"{package} kuruluyor...")
            return self.install_package(package, silent=True)
        except Exception as e:
            self.logger.log("error", f"Otomatik kurulum hatası ({package}): {e}")
            return False
    
    @property
    def log_analyzer(self):
        """Demo uyumluluğu için terminal_analyzer'a referans"""
        return self.terminal_analyzer
    
    @property
    def real_time_monitor(self):
        """Demo uyumluluğu için realtime_monitor'a referans"""
        return self.realtime_monitor
    
    def load_dependencies(self) -> Dict:
        """Dependency registry'yi yükler"""
        try:
            return self.dependency_registry.load_registry()
        except Exception as e:
            self.logger.log("error", f"Dependency loading hatası: {e}")
            return {}

    def install_package(self, package: str, silent: bool = False) -> bool:
        """
        Paketi kurar ve özet istatistiklerini günceller
        
        Args:
            package: Kurulacak paket adı (örn: "numpy==1.26.4" veya "pandas")
            silent: Sessiz kurulum (True ise sadece hata logları)
            
        Returns:
            bool: Kurulum başarılı ise True
        """
        try:
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
            
            # Zaten kurulu mu kontrol et
            if self.check_package_installed(package_name):
                if not silent:
                    self.logger.log("info", f"✅ {package_name} zaten kurulu")
                self.summary_generator.add_skipped(package_name, "Zaten kurulu")
                return True
            
            # Kurulum başlat
            if not silent:
                self.logger.log("info", f"📦 {package} kuruluyor...")
                
            # pip install komutu çalıştır
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, 
                "--quiet" if silent else "--verbose"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Başarılı kurulum
                version = "latest"
                if "==" in package:
                    version = package.split("==")[1]
                
                self.summary_generator.add_success(package_name, version)
                
                if not silent:
                    self.logger.log("info", f"✅ {package_name} başarıyla kuruldu")
                
                # Kurulum geçmişine ekle
                self.installation_history[package_name] = {
                    "version": version,
                    "installed_at": datetime.now().isoformat(),
                    "status": "success"
                }
                
                return True
            else:
                # Başarısız kurulum
                error_msg = result.stderr.strip() or result.stdout.strip() or "Bilinmeyen hata"
                self.summary_generator.add_failure(package_name, error_msg)
                
                self.logger.log("error", f"❌ {package_name} kurulum hatası: {error_msg}")
                
                # Kurulum geçmişine ekle
                self.installation_history[package_name] = {
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                    "status": "failed"
                }
                
                return False
                
        except Exception as e:
            error_msg = str(e)
            self.summary_generator.add_failure(package_name, error_msg)
            self.logger.log("error", f"❌ {package} kurulum istisnası: {error_msg}")
            return False

# Terminal Log Analizi ve Bağımlılık Öğrenme Sistemi
class TerminalLogAnalyzer:
    """
    Terminal loglarını analiz ederek eksik Python bağımlılıklarını tespit eder.
    ModuleNotFoundError, ImportError ve pip önerilerini yakalar.
    Gelişmiş özellikler: çakışma analizi, versiyon çıkarma, hash önbelleği.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.terminal_log_file = PLAIN_TERMINAL_LOG
        self.learned_dependencies_file = Path("learned_dependencies.json")
        self.learned_deps = self.load_learned_dependencies()
        
        # Hash önbelleği (tekrar analizi önlemek için)
        self.analyzed_hashes = set()
        
        # Gelişmiş hata pattern'leri
        self.error_patterns = {
            'module_not_found': [
                r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
                r"ImportError: No module named ['\"]([^'\"]+)['\"]",
                r"cannot import name ['\"]([^'\"]+)['\"]",
                r"ImportError: cannot import name ['\"]([^'\"]+)['\"]"
            ],
            'pip_suggestion': [
                r"pip install ([^\s=]+)(?:==([^\s]+))?",
                r"Try: pip install ([^\s=]+)(?:==([^\s]+))?",
                r"Run: pip install ([^\s=]+)(?:==([^\s]+))?",
                r"Install with: pip install ([^\s=]+)(?:==([^\s]+))?"
            ],
            'version_conflict': [
                r"requires ([^\s]+)==([^\s,]+)",
                r"requires ([^\s]+)>=([^\s,]+)",
                r"requires ([^\s]+)<=([^\s,]+)",
                r"incompatible with ([^\s]+) ([^\s]+)",
            ],
            'dependency_error': [
                r"ImportError: .*?requires '([^']+)'",
                r"ModuleNotFoundError: .*?requires '([^']+)'",
                r"Cannot find module '([^']+)'",
            ]
        }

    def analyze_log_file(self, file_path: str) -> List[str]:
        """Log dosyasını analiz et ve eksik bağımlılıkları tespit et"""
        try:
            if not os.path.exists(file_path):
                self.logger.log("error", f"Log dosyası bulunamadı: {file_path}")
                return []
            
            with open(file_path, "r", encoding="utf-8") as f:
                log_content = f.read()
                
            return self.analyze_log_content(log_content)
        except Exception as e:
            self.logger.log("error", f"Log dosyası analizi hatası: {e}")
            return []

    def analyze_log_content(self, log_content: str) -> List[str]:
        """Log içeriğini analiz et ve eksik bağımlılıkları tespit et"""
        found_packages = []
        version_conflicts = []
        dependency_errors = []
        
        try:
            # Daha önce analiz edildiyse hash kontrolü
            file_hash = hashlib.sha256(log_content.encode()).hexdigest()
            if file_hash in self.analyzed_hashes:
                self.logger.log("info", "Bu log içeriği daha önce analiz edildi, tekrar işleme alınıyor")
            else:
                # ModuleNotFoundError ve ImportError'ları çıkar
                missing_modules = self.extract_missing_imports(log_content)
                found_packages.extend(missing_modules)
                
                # Pip'in önerdiği paketleri çıkar
                pip_suggestions = self.parse_pip_suggestions(log_content)
                found_packages.extend(pip_suggestions)
                
                # Version conflict'leri çıkar
                version_conflicts = self.extract_version_conflicts(log_content)
                
                # Dependency error'ları çıkar
                dependency_errors = self.extract_dependency_errors(log_content)
                
                # Öğrenilmiş bağımlılıkları güncelle
                self.learned_deps["discovered_packages"] = list(set(self.learned_deps.get("discovered_packages", []) + found_packages))
                self.learned_deps["version_conflicts"] = list(set(self.learned_deps.get("version_conflicts", []) + version_conflicts))
                self.learned_deps["dependency_errors"] = list(set(self.learned_deps.get("dependency_errors", []) + dependency_errors))
                
                # Hash'i cache'e ekle
                self.analyzed_hashes.add(file_hash)
                
                # Kaydet
                self.save_learned_dependencies()
                
                self.logger.log("info", f"Yeni bağımlılıklar tespit edildi: {found_packages}")
            
            # Benzersiz paketleri döndür
            unique_packages = list(set(found_packages))
            self.logger.log("info", f"Terminal analizinde {len(unique_packages)} bağımlılık bulundu")
            
            return unique_packages
            
        except Exception as e:
            self.logger.log("error", f"Terminal log analizi hatası: {e}")
            return []
    
    def extract_missing_imports(self, log_content: str) -> List[str]:
        """ImportError ve ModuleNotFoundError'lardan eksik paketleri çıkar"""
        missing_modules = []
        
        try:
            for pattern in self.error_patterns['module_not_found']:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                for match in matches:
                    # Module adını temizle
                    module_name = match.strip()
                    
                    # Alt modülleri ana pakete çevir
                    if '.' in module_name:
                        module_name = module_name.split('.')[0]
                    
                    # Bilinen paket eşlemelerini uygula
                    mapped_package = self.map_module_to_package(module_name)
                    if mapped_package and mapped_package not in missing_modules:
                        missing_modules.append(mapped_package)
                        self.logger.log("info", f"Eksik modül tespit edildi: {module_name} -> {mapped_package}")
            
        except Exception as e:
            self.logger.log("error", f"Eksik import analizi hatası: {e}")
        
        return missing_modules
    
    def parse_pip_suggestions(self, log_content: str) -> List[str]:
        """Pip'in önerdiği paketleri parse et"""
        suggestions = []
        
        try:
            for pattern in self.error_patterns['pip_suggestion']:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                for match in matches:
                    package = match.strip()
                    if package and package not in suggestions:
                        suggestions.append(package)
                        self.logger.log("info", f"Pip önerisi tespit edildi: {package}")
            
        except Exception as e:
            self.logger.log("error", f"Pip öneri analizi hatası: {e}")
        
        return suggestions
    
    def extract_version_conflicts(self, log_content: str) -> List[Dict]:
        """Version conflict'leri tespit et"""
        conflicts = []
        
        try:
            for pattern in self.error_patterns['version_conflict']:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        conflict = {
                            "package": match[0],
                            "version": match[1],
                            "type": "requirement",
                            "timestamp": datetime.now().isoformat()
                        }
                        conflicts.append(conflict)
                        self.logger.log("warning", f"Version conflict: {match[0]} {match[1]}")
            
        except Exception as e:
            self.logger.log("error", f"Version conflict analizi hatası: {e}")
        
        return conflicts
    
    def map_module_to_package(self, module_name: str) -> Optional[str]:
        """Modül adını pip paket adına eşle"""
        # Bilinen modül -> paket eşlemeleri
        module_mappings = {
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'yaml': 'pyyaml',
            'sklearn': 'scikit-learn',
            'dateutil': 'python-dateutil',
            'serial': 'pyserial',
            'crypto': 'pycryptodome',
            'Crypto': 'pycryptodome',
            'jwt': 'pyjwt',
            'dns': 'dnspython',
            'docker': 'docker',
            'redis': 'redis',
            'pymongo': 'pymongo',
            'psycopg2': 'psycopg2-binary',
            'MySQLdb': 'mysql-connector-python',
            'win32api': 'pywin32',
            'tkinter': '',  # Built-in, paket gerekmez
            'sqlite3': '',  # Built-in
            'urllib': '',   # Built-in
            'json': '',     # Built-in
            'os': '',       # Built-in
            'sys': '',      # Built-in
            'time': '',     # Built-in
            'datetime': '', # Built-in
            're': '',       # Built-in
        }
        
        # Built-in modüller için None döndür
        if module_mappings.get(module_name) == '':
            return None
        
        # Eşleme varsa kullan, yoksa modül adını döndür
        return module_mappings.get(module_name, module_name)
    
    def generate_installation_plan(self, found_issues: List[str]) -> List[str]:
        """Tespit edilen sorunlar için kurulum planı oluştur"""
        try:
            installation_plan = []
            
            # Öğrenilmiş bağımlılıkları ekle
            for pkg in self.learned_deps.get("discovered_packages", []):
                if pkg not in installation_plan:
                    installation_plan.append(pkg)
            
            # Yeni bulunan sorunları ekle
            for issue in found_issues:
                if issue and issue not in installation_plan:
                    installation_plan.append(issue)
            
            # Öncelik sırasına göre sırala (temel paketler önce)
            priority_packages = ['numpy', 'scipy', 'pandas', 'matplotlib', 'requests', 'pillow']
            sorted_plan = []
            
            # Öncelikli paketleri önce ekle
            for priority_pkg in priority_packages:
                for pkg in installation_plan:
                    if priority_pkg in pkg.lower() and pkg not in sorted_plan:
                        sorted_plan.append(pkg)
              # Kalan paketleri ekle
            for pkg in installation_plan:
                if pkg not in sorted_plan:
                    sorted_plan.append(pkg)
            
            self.logger.log("info", f"Kurulum planı oluşturuldu: {len(sorted_plan)} paket")
            return sorted_plan
            
        except Exception as e:
            self.logger.log("error", f"Kurulum planı oluşturma hatası: {e}")
            return found_issues
    
    def get_learned_recommendations(self) -> List[str]:
        """Daha önce öğrenilmiş önerileri al"""
        try:
            return self.learned_deps.get("discovered_packages", [])
        except Exception as e:
            self.logger.log("error", f"Öğrenilmiş öneriler alınamadı: {e}")
            return []
    
    def clear_learned_dependencies(self):
        """Öğrenilmiş bağımlılıkları temizle - gelişmiş versiyon"""
        try:
            self.learned_deps = {
                "discovered_packages": [],
                "version_conflicts": [],
                "pip_suggestions": [],
                "dependency_errors": [],
                "package_metadata": {},
                "last_analysis": "",
                "analysis_count": 0,
                "auto_discovered": []
            }
            # Hash cache'i de temizle
            self.analyzed_hashes.clear()
            
            self.save_learned_dependencies()
            self.logger.log("info", "Öğrenilmiş bağımlılıklar ve cache temizlendi")
        except Exception as e:
            self.logger.log("error", f"Bağımlılık temizleme hatası: {e}")

    def get_learned_statistics(self) -> Dict:
        """Öğrenilmiş bağımlılıklarla ilgili istatistikleri döndürür"""
        try:
            stats = {
                "total_discovered": len(self.learned_deps.get("discovered_packages", [])),
                "version_conflicts": len(self.learned_deps.get("version_conflicts", [])),
                "dependency_errors": len(self.learned_deps.get("dependency_errors", [])),
                "pip_suggestions": len(self.learned_deps.get("pip_suggestions", [])),
                "analysis_count": self.learned_deps.get("analysis_count", 0),
                "last_analysis": self.learned_deps.get("last_analysis", ""),
                "auto_discovered_count": len(self.learned_deps.get("auto_discovered", []))
            }
            return stats
        except Exception as e:
            self.logger.log("error", f"İstatistik alma hatası: {e}")
            return {}

# Real-time Log İzleme Sistemi
class RealTimeLogMonitor:
    """
    Terminal çıktılarını real-time izler ve anlık hata yakalama yapar.
    ModuleNotFoundError'ları anlık olarak tespit eder ve gerekli paketleri belirler.
    """
    def __init__(self, logger: AdvancedLogger, terminal_analyzer: TerminalLogAnalyzer):
        self.logger = logger
        self.terminal_analyzer = terminal_analyzer
        self.monitoring = False
        self.monitor_thread = None
        self.detected_errors = []
        self.auto_install_queue = []
        
        # Real-time pattern'ler
        self.realtime_patterns = {
            'immediate_module_error': r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            'immediate_import_error': r"ImportError: No module named ['\"]([^'\"]+)['\"]",
            'immediate_pip_suggestion': r"pip install ([^\s]+)",
            'immediate_version_error': r"requires ([^\s]+)==([^\s,]+)"
        }
    
    def start_monitoring(self):
        """Real-time log monitoring başlat"""
        try:
            if self.monitoring:
                self.logger.log("warning", "Log monitoring zaten aktif")
                return
            
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.log("info", "Real-time log monitoring başlatıldı")
            
        except Exception as e:
            self.logger.log("error", f"Log monitoring başlatma hatası: {e}")
    
    def stop_monitoring(self):
        """Real-time log monitoring durdur"""
        try:
            if not self.monitoring:
                return
            
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            
            self.logger.log("info", "Real-time log monitoring durduruldu")
            
        except Exception as e:
            self.logger.log("error", f"Log monitoring durdurma hatası: {e}")
    
    def _monitor_loop(self):
        """Ana monitoring döngüsü"""
        try:
            terminal_log_file = PLAIN_TERMINAL_LOG
            
            # Log dosyasının son pozisyonunu takip et
            last_position = 0
            if terminal_log_file.exists():
                last_position = terminal_log_file.stat().st_size
            
            while self.monitoring:
                try:
                    if terminal_log_file.exists():
                        current_size = terminal_log_file.stat().st_size
                        
                        # Yeni veri varsa oku
                        if current_size > last_position:
                            with open(terminal_log_file, "r", encoding="utf-8") as f:
                                f.seek(last_position)
                                new_content = f.read()
                                last_position = current_size
                                
                                # Yeni içeriği analiz et
                                self._analyze_new_content(new_content)
                    
                    # Kısa bekle
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.log("error", f"Monitor loop hatası: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            self.logger.log("error", f"Monitor loop kritik hatası: {e}")
    
    def _analyze_new_content(self, content: str):
        """Yeni log içeriğini analiz et"""
        try:
            if not content.strip():
                return
            
            # Her satırı kontrol et
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    self.detect_import_errors_realtime(line.strip())
                    
        except Exception as e:
            self.logger.log("error", f"Real-time içerik analizi hatası: {e}")
    
    def detect_import_errors_realtime(self, line: str):
        """Gelen her log satırında import hatalarını kontrol et"""
        try:
            # ModuleNotFoundError kontrolü
            module_match = re.search(self.realtime_patterns['immediate_module_error'], line, re.IGNORECASE)
            if module_match:
                module_name = module_match.group(1)
                self._handle_missing_module(module_name, line)
                return
            
            # ImportError kontrolü
            import_match = re.search(self.realtime_patterns['immediate_import_error'], line, re.IGNORECASE)
            if import_match:
                module_name = import_match.group(1)
                self._handle_missing_module(module_name, line)
                return
            
            # Pip suggestion kontrolü
            pip_match = re.search(self.realtime_patterns['immediate_pip_suggestion'], line, re.IGNORECASE)
            if pip_match:
                package_name = pip_match.group(1)
                self._handle_pip_suggestion(package_name, line)
                return
            
            # Version error kontrolü
            version_match = re.search(self.realtime_patterns['immediate_version_error'], line, re.IGNORECASE)
            if version_match:
                package_name = version_match.group(1)
                version = version_match.group(2)
                self._handle_version_error(package_name, version, line)
                return
                 
        except Exception as e:
            self.logger.log("error", f"Real-time hata tespiti hatası: {e}")
    
    def _handle_missing_module(self, module_name: str, original_line: str):
        """Eksik modül tespit edildiğinde çalışır"""
        try:
            # Modülü pakete eşle
            package_name = self.terminal_analyzer.map_module_to_package(module_name)
            
            if package_name and package_name not in [item['package'] for item in self.detected_errors]:
                error_info = {
                    'type': 'missing_module',
                    'module': module_name,
                    'package': package_name,
                    'original_line': original_line,
                    'timestamp': datetime.now().isoformat(),
                    'auto_installable': True
                }
                
                self.detected_errors.append(error_info)
                self.auto_install_queue.append(package_name)
                
                self.logger.log("warning", f"Real-time tespit: Eksik modül {module_name} -> {package_name}")
                
                # Acil kurulum tetikle (opsiyonel)
                self._trigger_emergency_install(package_name)
                
        except Exception as e:
            self.logger.log("error", f"Eksik modül işleme hatası: {e}")
    
    def _handle_pip_suggestion(self, package_name: str, original_line: str):
        """Pip önerisi tespit edildiğinde çalışır"""
        try:
            if package_name not in [item['package'] for item in self.detected_errors]:
                error_info = {
                    'type': 'pip_suggestion',
                    'package': package_name,
                    'original_line': original_line,
                    'timestamp': datetime.now().isoformat(),
                    'auto_installable': True
                }
                
                self.detected_errors.append(error_info)
                self.auto_install_queue.append(package_name)
                
                self.logger.log("info", f"Real-time tespit: Pip önerisi {package_name}")
                
        except Exception as e:
            self.logger.log("error", f"Pip önerisi işleme hatası: {e}")
    
    def _handle_version_error(self, package_name: str, version: str, original_line: str):
        """Version hatası tespit edildiğinde çalışır"""
        try:
            error_info = {
                'type': 'version_conflict',
                'package': package_name,
                'required_version': version,
                'original_line': original_line,
                'timestamp': datetime.now().isoformat(),
                'auto_installable': True
            }
            
            self.detected_errors.append(error_info)
            self.auto_install_queue.append(f"{package_name}=={version}")
            
            self.logger.log("warning", f"Real-time tespit: Version conflict {package_name}=={version}")
            
        except Exception as e:
            self.logger.log("error", f"Version hatası işleme hatası: {e}")
    
    def _trigger_emergency_install(self, package_name: str):
        """Acil paket kurulumu tetikle (opsiyonel)"""
        try:
            # Bu fonksiyon isteğe bağlı olarak anlık kurulum yapabilir
            # Şimdilik sadece log'la, gerçek kurulum ana sistemde yapılacak
            self.logger.log("info", f"Acil kurulum gerekli: {package_name}")
            
        except Exception as e:
            self.logger.log("error", f"Acil kurulum tetikleme hatası: {e}")
    
    def is_monitoring(self) -> bool:
        """Monitoring aktif mi kontrol et"""
        return self.monitoring
    
    def get_detected_errors(self) -> List[Dict]:
        """Tespit edilen hataları döndür"""
        return self.detected_errors.copy()
    
    def get_auto_install_queue(self) -> List[str]:
        """Otomatik kurulum kuyruğunu döndür"""
        return self.auto_install_queue.copy()
    
    def clear_detected_errors(self):
        """Tespit edilen hataları temizle"""
        self.detected_errors.clear()
        self.auto_install_queue.clear()
        self.logger.log("info", "Real-time tespit edilen hatalar temizlendi")
    
    def get_monitoring_stats(self) -> Dict:
        """Monitoring istatistiklerini döndür"""
        return {
            'monitoring_active': self.monitoring,
            'detected_errors_count': len(self.detected_errors),
            'auto_install_queue_count': len(self.auto_install_queue),
            'thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False
        }


# Demo and testing functionality
if __name__ == "__main__":
    import argparse
    import time
    
    print("[PDS-X] AutoImporter başlatılıyor... Lütfen çayınızı kahvenizi alın ve bekleyin.")
    print("[PDS-X] Tüm işlemler otomatik olarak yapılıyor, lütfen bekleyin.")
    print("[PDS-X] Kütüphane yükleme işlemleri başlatıldı.")
    
    # Temel kurulum demo
    importer = AutoImporter()
    print("\n📦 Temel paket kurulumu yapılıyor...")
    importer.install_package("numpy==1.26.4")
    
    # Kurulum özeti yazdır
    print("\n" + "="*50)
    print("📊 KURULUM ÖZET RAPORU")
    print("="*50)
    importer.summary_generator.print_summary()


    def demo_auto_importer():
        """AutoImporter demo fonksiyonu"""
        print("="*60)
        print("PDS-X AutoImporter Demo")
        print("="*60)
        
        try:
            # AutoImporter'ı başlat
            auto_importer = AutoImporter()
              # Temel paketleri kontrol et ve kur
            print("\n1. Temel paketleri kontrol ediliyor...")
            required_packages = [
                "numpy", "pandas", "requests", "matplotlib", 
                "scipy", "scikit-learn", "pillow", "tqdm"
            ]
            
            # Kurulum özetini başlat
            auto_importer.summary_generator.reset_stats()
            
            for package in required_packages:
                if not auto_importer.check_package_installed(package):
                    print(f"   {package} eksik - kuruluyor...")
                    auto_importer.install_package(package, silent=False)
                else:
                    print(f"   ✓ {package} mevcut")
                    auto_importer.summary_generator.add_skipped(package, "Zaten kurulu")
            
            # Kurulum özeti yazdır
            print("\n📊 KURULUM ÖZETİ:")
            auto_importer.summary_generator.print_summary()
              # Log analizi demo
            print("\n2. Terminal log analizi demo...")
            terminal_analyzer = auto_importer.terminal_analyzer
            sample_log_content = """
            ImportError: No module named 'networkx'
            ModuleNotFoundError: No module named 'seaborn'
            Could not import 'plotly': pip install plotly
            """
            
            detected = terminal_analyzer.analyze_log_content(sample_log_content)
            print(f"   Tespit edilen eksik bağımlılıklar: {detected}")
            
            # Dependency management demo
            print("\n3. Bağımlılık yönetimi demo...")
            deps_before = auto_importer.load_dependencies()
            print(f"   Mevcut dependencies.json: {len(deps_before)} paket")
            
            # Real-time monitoring demo
            print("\n4. Real-time monitoring demo...")
            real_time_monitor = auto_importer.real_time_monitor
            print(f"   Monitoring durumu: {real_time_monitor.is_monitoring()}")
            
            # Test cleanup functionality
            print("\n5. Temizlik işlemi demo...")
            print("   Temizlik (cleanup) operasyonu:")
            print("   - Cache temizliği (.pdsx_cache/wheels)")
            print("   - Log rotasyonu (pdsXu_terminal.log.bak)")
            print("   - Geçici dosya temizliği")
            print("   Bu işlem sistem performansını artırır ve disk alanı serbest bırakır.")
            
            print("\n" + "="*60)
            print("Demo tamamlandı!")
            print("="*60)
            
        except Exception as e:
            print(f"Demo sırasında hata: {e}")
            import traceback
            traceback.print_exc()
    
    def test_exception_handling():
        """Exception handling test"""
        print("\n" + "="*60)
        print("Exception Handling Test")
        print("="*60)
        
        try:
            from exception_manager3 import PdsXException, PdsXSyntaxError, PdsXRuntimeError
            
            # Test basic exception
            try:
                raise PdsXException("Test exception message", "ERR_TEST", {"module": "auto_importer", "line_no": 123})
            except PdsXException as e:
                print(f"✓ PdsXException yakalandı: {e.get_formatted_error()}")
            
            # Test syntax error
            try:
                raise PdsXSyntaxError("Test syntax error", "ERR_SYNTAX_TEST", {"source": "DIM", "line_no": 456})
            except PdsXSyntaxError as e:
                print(f"✓ PdsXSyntaxError yakalandı: {e.get_formatted_error()}")
                print(f"  Öneri: {e.suggest_fix()}")
            
            # Test runtime error
            try:
                raise PdsXRuntimeError("Test runtime error", "ERR_RUNTIME_TEST", {"opcode": "TEST_OP", "line_no": 789})
            except PdsXRuntimeError as e:
                print(f"✓ PdsXRuntimeError yakalandı: {e.get_formatted_error()}")
            
            print("✓ Tüm exception testleri başarılı!")
            
        except ImportError as e:
            print(f"⚠ Exception manager import hatası: {e}")
            print("Fallback exception handling kullanılacak.")
    
    # Command line argument handling
    parser = argparse.ArgumentParser(description="PDS-X AutoImporter - Bağımlılık Yönetimi")
    parser.add_argument("--demo", action="store_true", help="AutoImporter demo çalıştır")
    parser.add_argument("--test-exceptions", action="store_true", help="Exception handling testini çalıştır")
    parser.add_argument("--install", nargs="+", help="Belirtilen paketleri kur")
    parser.add_argument("--check", nargs="+", help="Belirtilen paketleri kontrol et")
    parser.add_argument("--analyze-log", type=str, help="Belirtilen log dosyasını analiz et")
      args = parser.parse_args()
    
    if args.demo:
        demo_auto_importer()
    
    if args.test_exceptions:
        test_exception_handling()
    
    if args.install:
        auto_importer = AutoImporter()
        auto_importer.summary_generator.reset_stats()
        
        print(f"📦 {len(args.install)} paket kuruluyor...")
        for package in args.install:
            print(f"   → {package}")
            auto_importer.install_package(package, silent=False)
        
        # Kurulum özeti yazdır
        print("\n" + "="*50)
        print("📊 KURULUM ÖZET RAPORU")
        print("="*50)
        auto_importer.summary_generator.print_summary()
    
    if args.check:
        auto_importer = AutoImporter()
        for package in args.check:
            if auto_importer.check_package_installed(package):
                print(f"✓ {package} mevcut")
            else:
                print(f"✗ {package} eksik")
    
    if args.analyze_log:
        auto_importer = AutoImporter()
        if os.path.exists(args.analyze_log):
            detected = auto_importer.terminal_analyzer.analyze_log_file(args.analyze_log)
            print(f"Log analizi: {args.analyze_log}")
            print(f"Tespit edilen bağımlılıklar: {detected}")
        else:
            print(f"Log dosyası bulunamadı: {args.analyze_log}")
    
    # Eğer hiç argüman verilmemişse demo çalıştır
    if not any([args.demo, args.test_exceptions, args.install, args.check, args.analyze_log]):
        print("Argüman verilmedi, demo çalıştırılıyor...")
        demo_auto_importer()
        test_exception_handling()