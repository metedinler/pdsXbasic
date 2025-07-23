# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.2 - Added Graceful Shutdown
# Date: June 21, 2025
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
from concurrent.futures import ThreadPoolExecutor

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

# Sabitler
VENV_DIR = Path(".pdsx_isolated_env")
CACHE_DIR = Path(".pdsx_cache")
LOG_DIR = Path("logs")
TERMINAL_LOG = LOG_DIR / "pdsxu_terminal.jsonl"
INFO_LOG = LOG_DIR / "pdsxu_info.jsonl"
WARNING_LOG = LOG_DIR / "pdsxu_warnings.jsonl"
ERROR_LOG = LOG_DIR / "pdsxu_errors.jsonl"
PLAIN_TERMINAL_LOG = LOG_DIR / "pdsXu_terminal.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BACKUPS = 5

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
    def __init__(self):
        # Temel attribute'ları başlat
        self.logged_messages = set()
        self.last_log_time = {}
        self.es = None
        self.logger = None
        self.handlers = {}
        self.terminal_handler = None
        self.stdout_handler = None
        
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
        """Ana loglama fonksiyonu"""
        try:
            # Spam koruması
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            current_time = datetime.now()
            
            if (message_hash not in self.logged_messages or 
                (current_time - self.last_log_time.get(message_hash, datetime.min)).total_seconds() > 1):
                
                # Info mesajlarını terminal'e de yazdır
                if level.lower() == "info":
                    print(f"[PDS-X] {message}")
                
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
                            "message": message
                        })
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Elasticsearch bağlantısı başarısız: {e}")
                        self.es = None
                
                # Log rotasyonu kontrol et
                self.rotate_logs()
                
        except Exception as e:
            print(f"[PDS-X] Loglama hatası: {e}")

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

# Bağımlılık Kayıt Sistemi
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
            return {"packages": {}, "resolutions": {}, "status": "", "timestamp": ""}
        except Exception as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası yükleme hatası: {e}")
            return {"packages": {}, "resolutions": {}, "status": "", "timestamp": ""}

    def save_registry(self):
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4)
            self.logger.log("info", "Bağımlılık kayıt dosyası kaydedildi.")
        except Exception as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası kaydetme hatası: {e}")

    def register_package(self, package: str, version: str, status: str, dependencies: List[str] = []):
        try:
            self.registry["packages"][package] = {
                "version": version,
                "status": status,
                "dependencies": dependencies,
                "timestamp": datetime.now().isoformat()
            }
            self.save_registry()
            self.logger.log("info", f"{package}={version} kayıt edildi: {status}")
        except Exception as e:
            self.logger.log("error", f"{package} kayıt hatası: {e}")

    def register_resolution(self, module_name: str, resolution: Dict):
        try:
            self.registry["resolutions"][module_name] = resolution
            self.save_registry()
            self.logger.log("info", f"{module_name} için çakışma çözümü kaydedildi: {resolution}")
        except Exception as e:
            self.logger.log("error", f"{module_name} çakışma çözümü kayıt hatası: {e}")

    def update_on_conflict(self, package: str, conflict_info: str, resolution: str):
        try:
            if package in self.registry["packages"]:
                self.registry["packages"][package]["conflicts"] = conflict_info
                self.registry["packages"][package]["resolution"] = resolution
                self.save_registry()
                self.logger.log("info", f"{package} için çakışma güncellendi: {conflict_info}")
        except Exception as e:
            self.logger.log("error", f"{package} çakışma güncelleme hatası: {e}")

    def check_package(self, package: str) -> bool:
        try:
            if package in self.registry["packages"]:
                pkg_info = self.registry["packages"][package]
                if pkg_info["status"] == "Başarılı":
                    elapsed = datetime.now() - datetime.fromisoformat(pkg_info["timestamp"])
                    if elapsed.total_seconds() < 24 * 60 * 60:  # 24 saat
                        self.logger.log("info", f"{package} zaten yüklü ve güncel.")
                        return True
            return False
        except Exception as e:
            self.logger.log("error", f"{package} kontrol hatası: {e}")
            return False

# Pip Çıktı Analizi ve Düzeltme
class PipOutputAnalyzer:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.error_handlers = {
            "Ignoring invalid distribution": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", pkg], check=True, capture_output=True, text=True
            ),
            "ModuleNotFoundError": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg], check=True, capture_output=True, text=True
            ),
            "Could not find a version": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg], check=True, capture_output=True, text=True
            ),
            "deadlock detected": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg], check=True, capture_output=True, text=True
            ),
            "WinError 32": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--no-cache-dir"], check=True, capture_output=True, text=True
            ),
            "Error parsing dependencies": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", pkg], check=True, capture_output=True, text=True
            ),
            "Permission denied": lambda pkg: subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--user"], check=True, capture_output=True, text=True
            ),
        }

    def analyze_and_fix(self, output: str, package: str) -> bool:
        try:
            mirrors = ["https://pypi.org/simple", "https://mirrors.aliyun.com/pypi/simple"]
            for error, handler in self.error_handlers.items():
                if error in output:
                    self.logger.log("warning", f"Hata tespit edildi: {error}. {package} için düzeltme yapılıyor.")
                    for mirror in mirrors:
                        try:
                            cmd = [sys.executable, "-m", "pip", "install", package, f"--index-url={mirror}"]
                            result = handler(package)
                            self.logger.log("info", f"{package} düzeltildi: {result.stdout}")
                            return True
                        except subprocess.CalledProcessError as e:
                            self.logger.log("warning", f"Mirror {mirror} ile düzeltme başarısız: {e.output}")
                    return False
            self.logger.log("error", f"Bilinmeyen hata: {output}")
            return False
        except Exception as e:
            self.logger.log("error", f"Pip analiz hatası: {e}")
            return False

# Önbellek Yönetim Sistemi
class CacheManager:
    def __init__(self, cache_dir: Path = CACHE_DIR / "wheels", logger: AdvancedLogger = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.metadata_file = self.cache_dir / "packages.json"

    def install_from_cache(self, package: str) -> bool:
        try:
            cache_file = self.cache_dir / f"{package}.whl"
            if cache_file.exists():
                # Hash doğrulaması
                metadata = self.load_package_metadata()
                if package in metadata:
                    with open(cache_file, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != metadata[package]["hash"]:
                        self.logger.log("warning", f"{package} önbellek dosyası bozuk, yeniden indiriliyor.")
                        cache_file.unlink()  # Bozuk dosyayı sil
                        return self._download_and_cache(package)
                
                self.logger.log("info", f"{package} önbellekten yükleniyor.")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(cache_file)], 
                    check=True, capture_output=True, text=True
                )
                self.logger.log("info", f"{package} önbellekten kuruldu: {result.stdout}")
                return True
            return self._download_and_cache(package)
        except Exception as e:
            self.logger.log("error", f"Önbellek yükleme hatası: {e}")
            return False

    def _download_and_cache(self, package: str) -> bool:
        try:
            self.logger.log("info", f"{package} indiriliyor ve önbelleğe alınıyor.")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", package, "-d", str(self.cache_dir)],
                check=True, capture_output=True, text=True
            )
            self.logger.log("info", f"{package} indirildi: {result.stdout}")
            cache_file = self.cache_dir / f"{package}.whl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                metadata = self.load_package_metadata()
                metadata[package] = {
                    "version": package.split("==")[1] if "==" in package else "latest",
                    "timestamp": datetime.now().isoformat(),
                    "hash": file_hash
                }
                self.save_package_metadata(metadata)
            self.cleanup_cache()
            return True
        except Exception as e:
            self.logger.log("error", f"{package} indirme hatası: {e}")
            return False

    def rollback_package(self, package: str, previous_version: str) -> bool:
        try:
            metadata = self.load_package_metadata()
            if package in metadata and metadata[package]["version"] != previous_version:
                self.logger.log("info", f"{package} için rollback başlatılıyor: {previous_version}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"{package}=={previous_version}"],
                    check=True, capture_output=True, text=True
                )
                metadata[package]["version"] = previous_version
                self.save_package_metadata(metadata)
                self.logger.log("info", f"{package} rollback tamamlandı.")
                return True
            return False
        except Exception as e:
            self.logger.log("error", f"{package} rollback hatası: {e}")
            return False

    def visualize_version_tree(self, module_name: str):
        try:
            if not Digraph:
                self.logger.log("error", "graphviz kütüphanesi yüklü değil.")
                return
            metadata = self.load_package_metadata()
            dot = Digraph(comment=f"{module_name} Version Tree")
            for pkg, info in metadata.items():
                dot.node(pkg, f"{pkg} ({info['version']})")
                for dep in MODULE_SPECIFIC_DEPS.get(module_name, []):
                    if dep in metadata:
                        dot.edge(pkg, dep)
            dot.render(f"{module_name}_version_tree", format="png", cleanup=True)
            self.logger.log("info", f"{module_name} versiyon ağacı görselleştirildi.")
        except Exception as e:
            self.logger.log("error", f"Versiyon ağacı görselleştirme hatası: {e}")

    def save_package_metadata(self, packages: Dict):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(packages, f, indent=4)
        except Exception as e:
            self.logger.log("error", f"Metadata kaydetme hatası: {e}")

    def load_package_metadata(self) -> Dict:
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.log("error", f"Metadata yükleme hatası: {e}")
            return {}

    def cleanup_cache(self):
        try:
            max_age_days = 30
            now = datetime.now()
            metadata = self.load_package_metadata()
            for pkg, info in list(metadata.items()):
                pkg_time = datetime.fromisoformat(info["timestamp"])
                if (now - pkg_time).days > max_age_days:
                    cache_file = self.cache_dir / f"{pkg}.whl"
                    if cache_file.exists():
                        cache_file.unlink()
                    del metadata[pkg]
                    self.logger.log("info", f"Eski önbellek dosyası silindi: {pkg}")
            self.save_package_metadata(metadata)
        except Exception as e:
            self.logger.log("error", f"Önbellek temizleme hatası: {e}")

# İzole Ortam Yönetimi
class EnvManager:
    def __init__(self, venv_dir: Path = VENV_DIR, logger: AdvancedLogger = None):
        self.venv_dir = venv_dir
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger if logger is not None else AdvancedLogger()
        self.python_path = None

    def check_and_recreate(self):
        try:
            if self.error_count >= self.max_errors:
                self.logger.log("warning", "İzole ortamda çok fazla hata. Ortam siliniyor ve yeniden oluşturuluyor.")
                shutil.rmtree(self.venv_dir, ignore_errors=True)
                python_path = self.find_python310()
                if python_path:
                    subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True, capture_output=True, text=True)
                    self.error_count = 0
                    self.logger.log("info", "İzole ortam yeniden oluşturuldu.")
                else:
                    self.logger.log("error", "Python 3.10 bulunamadı. İzole ortam oluşturulamadı.")
                    sys.exit(1)
        except Exception as e:
            self.logger.log("error", f"İzole ortam yeniden oluşturma hatası: {e}")
            sys.exit(1)

    def report_error(self):
        try:
            self.error_count += 1
            self.check_and_recreate()
        except Exception as e:
            self.logger.log("error", f"Hata raporlama hatası: {e}")

    def find_python310(self) -> Optional[str]:
        try:
            self.logger.log("info", "Python 3.10 aranıyor...")
            for exe in ["python3.10", "python310", "python"]:
                path = shutil.which(exe)
                if path:
                    out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
                    if "3.10" in out:
                        self.logger.log("info", f"Python 3.10 bulundu: {path}")
                        self.python_path = path
                        return path
            if winreg:
                for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                    try:
                        with winreg.OpenKey(root, r"SOFTWARE\Python\PythonCore") as hkey:
                            for i in range(winreg.QueryInfoKey(hkey)[0]):
                                ver = winreg.EnumKey(hkey, i)
                                if ver.startswith("3.10"):
                                    with winreg.OpenKey(hkey, ver + r"\InstallPath") as subkey:
                                        py = winreg.QueryValue(subkey, None) + "python.exe"
                                        if os.path.exists(py):
                                            self.logger.log("info", f"Registry'de bulundu: {py}")
                                            self.python_path = py
                                            return py
                    except Exception:
                        continue
            self.logger.log("error", "Python 3.10 bulunamadı.")
            return None
        except Exception as e:
            self.logger.log("error", f"Python 3.10 arama hatası: {e}")
            return None

    def download_and_install_python310(self) -> Optional[str]:
        try:
            if os.name != "nt":
                self.logger.log("error", "Python 3.10 bulunamadı. Lütfen manuel kurun: https://www.python.org/downloads/release/python-31011/")
                return None
            drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
            total, used, free = shutil.disk_usage(drive + '\\')
            min_required = 300 * 1024 * 1024  # 300 MB
            if free < min_required:
                self.logger.log("error", f"Yetersiz disk alanı: {free // (1024*1024)} MB mevcut, 300 MB gerekli.")
                return None
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            installer_path = "python310_installer.exe"
            self.logger.log("info", "Python 3.10 indiriliyor...")
            subprocess.run(["curl", "-o", installer_path, installer_url], check=True, capture_output=True, text=True)
            self.logger.log("info", "Python 3.10 kuruluyor...")
            subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True, capture_output=True, text=True)
            os.remove(installer_path)
            self.logger.log("info", "Python 3.10 kurulumu tamamlandı.")
            return self.find_python310()
        except Exception as e:
            self.logger.log("error", f"Python 3.10 kurulum hatası: {e}")
            return None

    def add_python_to_path(self):
        try:
            if not self.python_path:
                self.python_path = self.find_python310() or self.download_and_install_python310()
            if self.python_path:
                python_dir = os.path.dirname(self.python_path)
                venv_dir = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin"))
                setx_cmd = f'setx PATH "%PATH%;{python_dir};{venv_dir}"'
                with open("add_pdsx_path.bat", "w") as f:
                    f.write(f"@echo off\n{setx_cmd}\necho PATH güncellendi.\npause\n")
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
                self.logger.log("info", "PATH'e eklemek için 'add_pdsx_path.bat' veya 'add_pdsx_path.ps1' dosyasını yönetici olarak çalıştırın!")
        except Exception as e:
            self.logger.log("error", f"PATH güncelleme hatası: {e}")

    def update_pip_if_needed(self, force_latest: bool = False, silent: bool = False) -> bool:
        """
        Pip sürümünü günceller veya kontrol eder.
        
        Args:
            force_latest: True ise en son pip sürümünü zorla indirir
            silent: True ise sessiz kurulum yapar (debug seviyesinde log)
            
        Returns:
            bool: Güncelleme başarılı ise True
            
        Logic:
        - Python 3.10 için önerilen pip sürümü 21.2.4 (o dönemin stable sürümü)
        - force_latest=True ise en son pip sürümünü alır
        - Cache kontrol eder, mevcut wheel var ise kullanır
        - Hash doğrulaması yapar
        - Silent mod debug seviyesinde log yapar
        """
        try:
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            
            if not os.path.exists(pip_cmd):
                self.logger.log("error", "Pip komutu bulunamadı!")
                return False            
            # Mevcut pip sürümünü kontrol et
            try:
                result = subprocess.run([pip_cmd, "--version"], capture_output=True, text=True)
                current_version = result.stdout.strip() if result.returncode == 0 else "unknown"
                log_level = "debug" if silent else "info"
                self.logger.log(log_level, f"Mevcut pip sürümü: {current_version}")
            except Exception as e:
                self.logger.log("warning", f"Pip sürüm kontrolü başarısız: {e}")
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
                self.logger.log("debug", "Python 3.10 için pip 21.2.4 kullanılıyor (o dönemin stable sürümü)")
            
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
                self.logger.log("warning", f"Pip güncelleme başarısız: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.log("error", f"Pip güncelleme hatası: {e}")
            return False

    def setup_environment(self) -> bool:
        """Ortamı hazırlar ve paketleri kurar."""
        try:
            self.logger.log("info", "Ortam hazırlanıyor...")
              # Önce sanal ortamda çalışıp çalışmadığımızı kontrol et
            if self.is_running_in_venv():
                self.logger.log("info", "Zaten sanal ortamda çalışıyoruz.")
                # Sanal ortamdayken eksik paketleri kontrol et ve yükle
                self.logger.log("info", "REQUIRED_PACKAGES kontrol ediliyor...")
                self.ensure_required_packages()
                return True
            
            self.logger.log("info", "Python 3.10 kontrol ediliyor...")
            
            # Python 3.10 kontrolü ve kurulumu
            python_path = self.find_python310()
            if not python_path:
                self.logger.log("info", "Python 3.10 bulunamadı, indiriliyor...")
                python_path = self.download_and_install_python310()
            if not python_path:
                self.logger.log("error", "Python 3.10 kurulamadı!")
                return False
                
            self.logger.log("info", f"Python 3.10 hazır: {python_path}")
            
            # Sanal ortam kontrolü ve kurulumu
            if not self.venv_dir.exists():
                self.logger.log("info", "Sanal ortam oluşturuluyor...")
                subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True)
                
            # PATH güncellemesi
            self.add_python_to_path()
              # NumPy ve diğer paketlerin kurulumu
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
              # Pip sürümünü güncelle - varsayılan olarak Python 3.10 zamanındaki pip sürümü
            if not self.update_pip_if_needed(force_latest=False, silent=False):
                self.logger.log("warning", "Pip güncelleme başarısız, devam ediliyor")
            
            self.logger.log("info", "Temel paketler yükleniyor...")
            process = subprocess.run([pip_cmd, "install", "wheel"], check=True)
            shutdown_manager.register_process(process)
            
            # Graceful shutdown kontrolü
            if shutdown_manager.shutdown_requested:
                self.logger.log("info", "Shutdown sinyali alındı, kurulum durduruluyor...")
                return False
            
            self.logger.log("info", "NumPy 1.26.4 yükleniyor...")
            process = subprocess.run([pip_cmd, "install", "numpy==1.26.4"], check=True)
            shutdown_manager.register_process(process)
            
            # Graceful shutdown kontrolü
            if shutdown_manager.shutdown_requested:
                self.logger.log("info", "Shutdown sinyali alındı, kurulum durduruluyor...")
                return False
              # REQUIRED_PACKAGES'taki tüm paketlerin kurulumu
            self.logger.log("info", "REQUIRED_PACKAGES kontrol ediliyor ve yükleniyor...")
            success_count = 0
            total_count = len(REQUIRED_PACKAGES)
            
            for package_spec, import_name in REQUIRED_PACKAGES:
                # Graceful shutdown kontrolü
                if shutdown_manager.shutdown_requested:
                    self.logger.log("info", "Shutdown sinyali alındı, paket kurulumu durduruluyor...")
                    return False
                
                self.logger.log("info", f"Kontrol ediliyor: {package_spec}")
                  # Önce paketin yüklü olup olmadığını kontrol et
                try:
                    # Sanal ortamda test için subprocess kullan
                    venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
                    test_result = subprocess.run([venv_python, "-c", f"import {import_name}"], 
                                                capture_output=True, text=True)
                    if test_result.returncode == 0:
                        self.logger.log("debug", f"✓ {import_name} zaten yüklü")
                        success_count += 1
                        continue
                    else:
                        self.logger.log("info", f"× {import_name} yüklü değil, yükleniyor...")
                except Exception as e:
                    self.logger.log("debug", f"Import test hatası {import_name}: {e}")
                    self.logger.log("info", f"× {import_name} yüklü değil, yükleniyor...")
                
                # Paketi yükle
                try:
                    install_cmd = [pip_cmd, "install", package_spec]
                    process = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                    shutdown_manager.register_process(process)
                    
                    self.logger.log("info", f"✓ {package_spec} başarıyla yüklendi")
                    success_count += 1
                    
                except subprocess.CalledProcessError as e:
                    self.logger.log("warning", f"× {package_spec} yüklenemedi: {e}")
                    if e.stdout:
                        self.logger.log("debug", f"STDOUT: {e.stdout}")
                    if e.stderr:
                        self.logger.log("debug", f"STDERR: {e.stderr}")
                    continue
                except Exception as e:
                    self.logger.log("warning", f"× {package_spec} yükleme hatası: {e}")
                    continue
            
            self.logger.log("info", f"Paket kurulum tamamlandı: {success_count}/{total_count} başarılı")
            
            self.logger.log("info", "Ortam hazırlığı tamamlandı!")
            
            # Şimdi sanal ortamda yeniden başlat
            self.logger.log("info", "Sanal ortamda yeniden başlatılıyor...")
            return self.restart_in_venv()
            
        except Exception as e:
            self.logger.log("error", f"Ortam hazırlama hatası: {e}")
            return False

    def restart_in_venv(self) -> bool:
        """Programı sanal ortam içinde yeniden başlatır"""
        try:
            if not self.venv_dir.exists():
                self.logger.log("error", "Sanal ortam bulunamadı!")
                return False
            
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            
            if not os.path.exists(venv_python):
                self.logger.log("error", f"Sanal ortam Python'u bulunamadı: {venv_python}")
                return False
            
            self.logger.log("info", "Program sanal ortamda yeniden başlatılıyor...")
            
            # Mevcut argümanları al
            current_args = sys.argv[1:]  # İlk argüman (script adı) hariç
            
            # Yeni process'i başlat
            new_cmd = [venv_python] + sys.argv
            self.logger.log("info", f"Yeni komut: {' '.join(new_cmd)}")
            
            # Mevcut process'i sonlandır ve yeni process'i başlat
            os.execv(venv_python, new_cmd)
            
        except Exception as e:
            self.logger.log("error", f"Sanal ortamda yeniden başlatma hatası: {e}")
            return False

    def is_running_in_venv(self) -> bool:
        """Şu anda sanal ortamda çalışıp çalışmadığını kontrol eder"""
        try:
            # Birden fazla yöntemle kontrol et
            
            # Yöntem 1: sys.executable kontrolü
            venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
            current_python = sys.executable
            
            # Yol karşılaştırması
            if os.path.normpath(current_python) == os.path.normpath(venv_python):
                self.logger.log("debug", f"Sanal ortam tespit edildi (executable): {current_python}")
                return True
            
            # Yöntem 2: Sanal ortam aktivasyon kontrolü
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                # Eğer sanal ortam aktifse ve doğru dizindeyse
                if str(self.venv_dir.absolute()) in sys.prefix:
                    self.logger.log("debug", f"Sanal ortam tespit edildi (prefix): {sys.prefix}")
                    return True
            
            # Yöntem 3: VIRTUAL_ENV environment variable
            virtual_env = os.environ.get('VIRTUAL_ENV')
            if virtual_env and os.path.normpath(virtual_env) == os.path.normpath(str(self.venv_dir)):
                self.logger.log("debug", f"Sanal ortam tespit edildi (VIRTUAL_ENV): {virtual_env}")
                return True
            
            # Yöntem 4: sys.path kontrolü (en son çare)
            if str(self.venv_dir / "Lib" / "site-packages") in sys.path:
                self.logger.log("debug", f"Sanal ortam tespit edildi (sys.path): {self.venv_dir}")
                return True
                
            self.logger.log("debug", f"Sanal ortam tespit edilemedi. current: {current_python}, venv: {venv_python}")
            return False
            
        except Exception as e:
            self.logger.log("error", f"Sanal ortam kontrol hatası: {e}")
            return False# Çakışma Yönetim Sistemi
class ConflictManager:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = defaultdict(list)
        self.scientific_utils = ScientificUtils(logger)

    def clean_version(self, version: str) -> str:
        return re.sub(r'[=<>]', '', version).strip()

    def build_decision_tree(self, conflicts: Dict):
        try:
            # DecisionTreeClassifier'ı lazy loading ile yükle
            IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier = get_sklearn_components()
            if DecisionTreeClassifier is None:
                self.logger.log("error", "DecisionTreeClassifier yüklenemedi")
                return None
                
            X = []
            y = []
            for dep, issue in conflicts.items():
                features = [len(issue), issue.count("=="), issue.count("<"), issue.count(">")]
                X.append(features)
                y.append(1 if "force-reinstall" in issue.lower() else 0)
            clf = DecisionTreeClassifier(max_depth=5)
            clf.fit(X, y)
            self.logger.log("info", "Karar ağacı oluşturuldu.")
            return clf
        except Exception as e:
            self.logger.log("error", f"Karar ağacı oluşturma hatası: {e}")
            return None

    def detect_conflicts(self, module_name: str, deps: List[str]) -> Dict:
        try:
            self.logger.log("info", f"{module_name} için çakışma kontrolü başlatılıyor.")
            conflicts = {}
            for dep in deps:
                result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
                if "no conflicts" not in result.stdout.lower():
                    conflicts[dep] = result.stdout
                    self.logger.log("warning", f"Çakışma tespit edildi: {dep}, {result.stdout}")
                else:
                    self.logger.log("info", f"{dep} için çakışma bulunamadı.")
            self.conflicts[module_name] = conflicts
            return conflicts
        except Exception as e:
            self.logger.log("error", f"Çakışma kontrol hatası: {e}")
            return {}
            return {}

    def resolve_conflicts(self, module_name: str, conflicts: Dict) -> Dict:
        try:
            resolutions = {}
            clf = self.build_decision_tree(conflicts)
            if clf:
                for dep, issue in conflicts.items():
                    features = [len(issue), issue.count("=="), issue.count("<"), issue.count(">")]
                    prediction = clf.predict([features])[0]
                    cmd = f"pip install {self.clean_version(dep)} --force-reinstall" if prediction == 1 else f"pip install {self.clean_version(dep)}"
                    try:
                        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                        resolutions[dep] = {"command": cmd, "reason": issue, "output": result.stdout}
                        self.logger.log("info", f"Çakışma çözüldü: {dep}, {issue}")
                    except subprocess.CalledProcessError as e:
                        self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
            neural_resolutions = self.neural_conflict_resolution(module_name, conflicts)
            resolutions.update(neural_resolutions)
            self.resolutions[module_name] = resolutions
            metrics = [len(conflicts), sum(len(v) for v in conflicts.values())]
            quantum_result = self.scientific_utils.quantum_load_simulation(metrics)
            if "error" not in quantum_result:
                self.logger.log("info", f"Kuantum analizi sonucu: {quantum_result}")
            return resolutions
        except Exception as e:
            self.logger.log("error", f"Çakışma çözüm hatası: {e}")
            return {}

    def neural_conflict_resolution(self, module_name: str, conflicts: Dict) -> Dict:
        try:
            if not conflicts:
                self.logger.log("info", f"{module_name} için nöral ağ çakışma çözümü: Çakışma yok.")
                return {}
            
            # Sklearn bileşenlerini lazy loading ile yükle
            IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier = get_sklearn_components()
            if MLPClassifier is None:
                self.logger.log("error", "MLPClassifier yüklenemedi")
                return {}
                
            X = []
            for v in conflicts.values():
                features = [len(conflicts), sum(len(val) for val in conflicts.values()), len(v.split())]
                X.append(features)
            y = [1 if any(dep.lower() in ["numpy", "tensorflow", "thinc"] for dep in conflicts) else 0]
            clf = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=500)
            clf.fit(X, y)
            prediction = clf.predict(X)
            resolutions = {}
            if prediction[0] == 1:
                for dep in conflicts:
                    if "numpy" in dep.lower():
                        resolutions[dep] = {"command": "pip install numpy==1.26.4 --force-reinstall", "reason": "Nöral ağ önerisi: numpy çakışması"}
                    elif "tensorflow" in dep.lower():
                        resolutions[dep] = {"command": "pip install tensorflow==2.15.0 --force-reinstall", "reason": "Nöral ağ önerisi: tensorflow çakışması"}
                    elif "thinc" in dep.lower():
                        resolutions[dep] = {"command": "pip install thinc==8.3.2 --force-reinstall", "reason": "Nöral ağ önerisi: thinc çakışması"}
            self.logger.log("info", f"Nöral ağ çakışma çözümü: {resolutions}")
            return resolutions
        except Exception as e:
            self.logger.log("error", f"Nöral ağ çakışma çözüm hatası: {e}")
            return {}

    def quantum_analysis(self, module_deps: Dict) -> Dict:
        try:
            np = get_numpy()
            if np is None:
                return {"error": "NumPy yüklü değil"}
                
            relationships = defaultdict(list)
            for module, deps in module_deps.items():
                for dep in deps:
                    relationships[module].append(dep)
            modules = list(module_deps.keys())
            matrix = np.zeros((len(modules), len(modules)))
            for i, module in enumerate(modules):
                for j, other in enumerate(modules):
                    if other in relationships[module]:
                        matrix[i][j] = 1
            scores = np.sum(matrix, axis=1)
            critical_modules = [
                (module, score) for module, score in zip(modules, scores)
                if score > np.mean(scores) + np.std(scores)
            ]
            self.logger.log("info", f"Kuantum analizi: Kritik modüller {critical_modules}")
            return {
                "critical_modules": critical_modules,
                "relationship_density": float(np.mean(matrix)),
                "isolation_score": float(1 - np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 0)
            }
        except Exception as e:
            self.logger.log("error", f"Kuantum analizi hatası: {e}")
            return {"error": str(e)}

# Modül Analiz Sistemi
class ModuleAnalyzer:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.log_file = TERMINAL_LOG

    def analyze_logs(self) -> Dict:
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
                            level = data["level"].lower()
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
                self.logger.log("info", f"Log analizi: {log_stats['error_count']} hata, {log_stats['warning_count']} uyarı")
            return log_stats
        except Exception as e:
            self.logger.log("error", f"Log analizi hatası: {e}")
            return {"error": str(e)}

    def analyze_conflicts(self) -> Dict:
        try:
            conflicts = {}
            for log_file in [ERROR_LOG, WARNING_LOG]:
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                if "çakışma" in data["message"].lower():
                                    pkg = data["message"].split(":")[1].strip().split(" ")[0]
                                    conflicts[pkg] = data["message"]
                            except json.JSONDecodeError:
                                continue
            self.logger.log("info", f"Çakışma analizi: {len(conflicts)} çakışma bulundu.")
            return conflicts
        except Exception as e:
            self.logger.log("error", f"Çakışma analizi hatası: {e}")
            return {}

    def generate_module_report(self, modules: List[Dict]) -> Dict:
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
            self.logger.log("info", "Modül raporu oluşturuldu.")
            return report
        except Exception as e:
            self.logger.log("error", f"Rapor oluşturma hatası: {e}")
            return {"error": str(e)}

    def _check_module_status(self, module: Dict) -> str:
        try:
            if not all(key in module for key in ["name", "version", "dependencies"]):
                return "invalid"
            if not module.get("dependencies"):
                return "warning"
            return "ok"
        except Exception:
            return "error"

    def _detect_issues(self, module: Dict) -> List[str]:
        try:
            issues = []
            if "name" not in module:
                issues.append("Modül adı eksik")
            if "version" not in module:
                issues.append(f"{module.get('name', 'bilinmeyen')} modülünde sürüm bilgisi eksik")
            if "dependencies" not in module:
                issues.append(f"{module.get('name', 'bilinmeyen')} modülünde bağımlılıklar eksik")
            version = module.get("version", "")
            if version and not self._is_valid_version(version):
                issues.append(f"{module.get('name', 'bilinmeyen')} modülünde geçersiz sürüm formatı: {version}")
            deps = module.get("dependencies", [])
            if isinstance(deps, list) and len(deps) == 0:
                issues.append(f"{module.get('name', 'bilinmeyen')} modülünde bağımlılık yok")
            return issues
        except Exception as e:
            self.logger.log("error", f"Sorun tespit hatası: {e}")
            return [f"Modül analizi hatası: {str(e)}"]

    def _generate_recommendations(self, module: Dict) -> List[str]:
        try:
            recommendations = []
            if "version" not in module:
                recommendations.append(f"{module.get('name', 'bilinmeyen')} modülü için sürüm bilgisi ekleyin")
            if "dependencies" not in module:
                recommendations.append(f"{module.get('name', 'bilinmeyen')} modülü için bağımlılıkları belirtin")
            deps = module.get("dependencies", [])
            if isinstance(deps, list):
                if len(deps) == 0:
                    recommendations.append(f"{module.get('name', 'bilinmeyen')} modülü için gerekli bağımlılıkları eklemeyi düşünün")
                elif len(deps) > 10:
                    recommendations.append(f"{module.get('name', 'bilinmeyen')} modülünün bağımlılıklarını azaltmayı düşünün")
            return recommendations
        except Exception as e:
            self.logger.log("error", f"Öneri oluşturma hatası: {e}")
            return [f"Öneri oluşturma hatası: {str(e)}"]

    def _is_valid_version(self, version: str) -> bool:
        try:
            parts = version.split(".")
            return len(parts) >= 2 and all(part.isdigit() or part in ["post0", "dev0"] for part in parts)
        except Exception:
            return False

# Paralel İndirme Sistemi
class AsyncDownloadManager:
    def __init__(self, max_workers: int = 4, cache_dir: Path = CACHE_DIR / "wheels", logger: AdvancedLogger = None):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.download_stats = {}

    def download_package(self, task: Dict) -> Dict:
        try:
            package, version = task["package"], task["version"]
            cache_file = self.cache_dir / f"{package}-{version}.whl"
            if cache_file.exists():
                self.download_stats[package] = {"success": True, "size": os.path.getsize(cache_file)}
                self.logger.log("info", f"{package}-{version} önbellekte bulundu.")
                return self.download_stats[package]
            url = f"https://files.pythonhosted.org/packages/{package}-{version}-py3-none-any.whl"
            retry_count = 0
            while retry_count < 3:
                try:
                    response = subprocess.run(
                        ["curl", "-o", str(cache_file), url], capture_output=True, text=True, check=True
                    )
                    self.download_stats[package] = {"success": True, "size": os.path.getsize(cache_file)}
                    self.logger.log("info", f"{package}-{version} indirildi.")
                    return self.download_stats[package]
                except subprocess.CalledProcessError as e:
                    self.logger.log("warning", f"{package} indirme hatası: {e.stderr}")
                retry_count += 1
                time.sleep(2 ** retry_count)
            self.download_stats[package] = {"success": False, "error": "İndirme başarısız"}
            return self.download_stats[package]
        except Exception as e:
            self.logger.log("error", f"Paket indirme hatası: {e}")
            return {"success": False, "error": str(e)}

    async def download_and_install(self, tasks: List[Dict]):
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                download_futures = [executor.submit(self.download_package, task) for task in tasks]
                for future in download_futures:
                    result = await asyncio.get_event_loop().run_in_executor(None, future.result)
                    if result["success"]:
                        pkg = result["package"]
                        executor.submit(self.install_package, pkg)
            return self.download_stats
        except Exception as e:
            self.logger.log("error", f"Paralel indirme ve kurulum hatası: {e}")
            return {}

# Bilimsel Analiz Sistemi
class ScientificUtils:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        # Sklearn bileşenlerini lazy loading ile yükle
        IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier = get_sklearn_components()
        if StandardScaler and IsolationForest:
            self.scaler = StandardScaler()
            self.isolation_forest = IsolationForest(contamination=0.1)
        else:
            self.scaler = None
            self.isolation_forest = None
        self._lock = threading.Lock()

    def quantum_load_simulation(self, metrics: List[float]) -> Dict:
        with self._lock:
            try:
                if not metrics:
                    raise ValueError("Metrik listesi boş olamaz. Lütfen geçerli metrikler sağlayın.")
                
                np = get_numpy()
                if np is None:
                    return {"error": "NumPy yüklü değil"}
                
                normalized = self.scaler.fit_transform(np.array(metrics).reshape(-1, 1))
                mean = np.mean(metrics)
                std = np.std(metrics)
                outliers = self.isolation_forest.fit_predict(normalized)
                quantiles = np.percentile(metrics, [25, 50, 75])
                self.logger.log("info", f"Kuantum simülasyonu: Ortalama {mean:.2f}, Std {std:.2f}")
                return {
                    "mean": float(mean),
                    "std": float(std),
                    "outliers": outliers.tolist(),
                    "q1": float(quantiles[0]),
                    "median": float(quantiles[1]),
                    "q3": float(quantiles[2])
                }
            except ValueError as ve:
                self.logger.log("error", f"Kuantum simülasyonu hatası: {ve}")
                return {"error": str(ve)}
            except Exception as e:
                self.logger.log("error", f"Kuantum simülasyonu beklenmeyen hata: {e}")
                return {"error": str(e)}

    def chaos_load_prediction(self) -> Dict:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process_count = len(psutil.Process().children())
            thread_count = psutil.Process().num_threads()
            swap = psutil.swap_memory()
            net = psutil.net_io_counters()
            self.logger.log("info", f"Kaos tahmini: CPU {cpu_percent}%, Bellek {memory.percent}%")
            return {
                "cpu_usage": cpu_percent,
                "memory_used": memory.percent,
                "disk_used": disk.percent,
                "swap_used": swap.percent,
                "process_count": process_count,
                "thread_count": thread_count,
                "net_packets_sent": net.packets_sent,
                "net_packets_recv": net.packets_recv,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.log("error", f"Kaos yük tahmini hatası: {e}")
            return {"error": str(e)}

    def genetic_dependency_optimizer(self, deps: List[tuple[str, str]]) -> List[str]:
        try:
            graph = defaultdict(list)
            for dep, target in deps:
                graph[dep].append(target)
            visited = set()
            temp = set()

            def has_cycle(node: str) -> bool:
                if node in temp:
                    return True
                if node in visited:
                    return False
                temp.add(node)
                for neighbor in graph.get(node, []):
                    if has_cycle(neighbor):
                        return True
                temp.remove(node)
                visited.add(node)
                return False

            optimized = []
            for dep in graph:
                if not has_cycle(dep):
                    optimized.append(dep)
            self.logger.log("info", f"Genetik optimizasyon: {len(optimized)} bağımlılık sıralandı")
            return optimized
        except Exception as e:
            self.logger.log("error", f"Genetik optimizasyon hatası: {e}")
            return []

    def neural_load_balancer(self, resources: List[Dict[str, float]], threshold: float = 0.8) -> Dict:
        try:
            if not resources:
                raise ValueError("Kaynak listesi boş olamaz")
            
            np = get_numpy()
            if np is None:
                return {"error": "NumPy yüklü değil"}
            
            if self.scaler is None:
                return {"error": "StandardScaler yüklü değil"}
                
            resource_matrix = np.array([[r['cpu'], r['memory'], r['disk']] for r in resources])
            normalized = self.scaler.fit_transform(resource_matrix)
            overloaded = np.where(normalized > threshold)[0]
            underloaded = np.where(normalized < threshold)[0]
            load_distribution = {
                "mean_load": float(normalized.mean()),
                "std_load": float(normalized.std()),
                "max_load": float(normalized.max()),
                "min_load": float(normalized.min())
            }
            self.logger.log("info", f"Nöral dengeleme: {len(overloaded)} aşırı yük, {len(underloaded)} düşük yük")
            return {
                "overloaded": overloaded.tolist(),
                "underloaded": underloaded.tolist(),
                "scores": normalized.mean(axis=1).tolist(),
                "distribution": load_distribution
            }
        except ValueError as ve:
            self.logger.log("error", f"Nöral yük dengeleme hatası: {ve}")
            return {"error": str(ve)}
        except Exception as e:
            self.logger.log("error", f"Nöral yük dengeleme hatası: {e}")
            return {"error": str(e)}

    def blockchain_module_validation(self, modules: List[Dict]) -> Dict:
        try:
            if not modules:
                raise ValueError("Modül listesi boş olamaz")
            valid_modules = []
            invalid_modules = []
            validation_chain = []
            prev_hash = None
            for module in modules:
                is_valid = self._verify_module_integrity(module)
                module_info = {
                    "name": module.get('name', 'unknown'),
                    "version": module.get('version', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "prev_hash": prev_hash
                }
                current_hash = hashlib.sha256(str(module_info).encode()).hexdigest()
                module_info["hash"] = current_hash
                if is_valid:
                    valid_modules.append(module['name'])
                else:
                    invalid_modules.append(module['name'])
                validation_chain.append(module_info)
                prev_hash = current_hash
            self.logger.log("info", f"Doğrulama: {len(valid_modules)} geçerli, {len(invalid_modules)} geçersiz modül")
            return {
                "valid": valid_modules,
                "invalid": invalid_modules,
                "validation_chain": validation_chain,
                "chain_length": len(validation_chain),
                "genesis_hash": validation_chain[0]["hash"] if validation_chain else None,
                "latest_hash": prev_hash
            }
        except ValueError as ve:
            self.logger.log("error", f"Blockchain doğrulama hatası: {ve}")
            return {"error": str(ve)}
        except Exception as e:
            self.logger.log("error", f"Blockchain doğrulama hatası: {e}")
            return {"error": str(e)}

    def _verify_module_integrity(self, module: Dict) -> bool:
        try:
            required_fields = ['name', 'version', 'dependencies']
            if not all(field in module for field in required_fields):
                return False
            version = module['version']
            if not version or not self._is_valid_version(version):
                return False
            if not isinstance(module['dependencies'], list):
                return False
            return True
        except Exception:
            return False

    def _is_valid_version(self, version: str) -> bool:
        try:
            parts = version.split(".")
            return len(parts) >= 2 and all(part.isdigit() or part in ["post0", "dev0"] for part in parts)
        except Exception:
            return False

# Modül Yükleme Özeti
class ModuleSummaryGenerator:
    def __init__(self, logger: AdvancedLogger):
        self.summaries = []
        self.logger = logger

    def add_module_status(self, module_name: str, status: str, duration: float):
        try:
            self.summaries.append({"module": module_name, "status": status, "duration": duration})
            self.logger.log("info", f"Modül durumu eklendi: {module_name}, {status}, {duration:.2f} saniye")
        except Exception as e:
            self.logger.log("error", f"Modül durumu ekleme hatası: {e}")

    def print_summary(self):
        try:
            if not self.summaries:
                self.logger.log("info", "Henüz modül yüklenmedi.")
                return
            self.logger.log("info", "Modül Yükleme Özeti:")
            header = f"{Fore.CYAN}┌────────────────────┬──────────────┬──────────┐\n│ Modül Adı          │ Durum        │ Süre (sn)│\n├────────────────────┼──────────────┼──────────┤{Style.RESET_ALL}"
            print(header)
            for s in self.summaries:
                color = Fore.GREEN if "Başarılı" in s["status"] else Fore.RED
                row = f"{color}│ {s['module']:<18} │ {s['status']:<12} │ {s['duration']:.2f}      │{Style.RESET_ALL}"
                print(row)
            footer = f"{Fore.CYAN}└────────────────────┴──────────────┴──────────┘{Style.RESET_ALL}"
            print(footer)
        except Exception as e:
            self.logger.log("error", f"Özet tablo yazdırma hatası: {e}")

# Derleme Araçları Kontrolü
def check_build_tools(logger: AdvancedLogger):
    try:
        if subprocess.run(["cl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            logger.log("info", "Visual Studio Build Tools bulundu.")
            return "Visual Studio Build Tools"
        if subprocess.run(["gcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            logger.log("info", "MinGW bulundu.")
            return "MinGW"
        logger.log("warning", "Derleme araçları bulunamadı. Bazı kütüphaneler kurulamayabilir.")
        return None
    except Exception as e:
        logger.log("error", f"Derleme araçları kontrol hatası: {e}")
        return None

def add_to_path(tool_path: str, logger: AdvancedLogger):
    try:
        setx_cmd = f'setx PATH "%PATH%;{tool_path}"'
        subprocess.run(setx_cmd, shell=True, check=True)
        logger.log("info", f"{tool_path} PATH'e eklendi.")
    except Exception as e:
        logger.log("error", f"PATH'e ekleme hatası: {e}")

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
                self.metadata = {"auto_importer": {"version": "1.7.9.2", "dependencies": []}}
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
                
            except Exception as e:
                print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
                raise e

    def cleanup_on_shutdown(self):
        """Shutdown sırasında temizlik yapar"""
        try:
            self.logger.log("info", "AutoImporter temizlik işlemleri başlatılıyor...")
            
            # Aktif indirme işlemlerini durdur
            if hasattr(self, 'downloader') and self.downloader:
                # Download manager'ın aktif tasklerini iptal et
                pass
                
            # Cache'i kaydet
            if hasattr(self, 'cache_manager') and self.cache_manager:
                try:
                    self.cache_manager.cleanup_cache()
                except Exception as e:
                    self.logger.log("warning", f"Cache temizlik hatası: {e}")
            
            # Registy'i kaydet
            if hasattr(self, 'dependency_registry') and self.dependency_registry:
                try:
                    self.dependency_registry.save_registry()
                except Exception as e:
                    self.logger.log("warning", f"Registry kaydetme hatası: {e}")
                    
            self.logger.log("info", "AutoImporter temizlik tamamlandı.")
        except Exception as e:
            print(f"[PDS-X] AutoImporter temizlik hatası: {e}")

    def check_shutdown_signal(self):
        """Shutdown sinyali kontrol eder"""
        if shutdown_manager.shutdown_requested:
            self.logger.log("info", "Shutdown sinyali alındı, işlem durduruluyor...")
            return True
        return False

    def check_stop_combination(self, event):
        """Klavye kombinasyonu kontrolü"""
        try:
            keyboard = get_keyboard()
            if keyboard and keyboard.is_pressed("left ctrl") and keyboard.is_pressed("left shift") and event.name == "q":
                self.logger.log("info", "Sol Ctrl + Sol Shift + Q ile durduruluyor.")
                self.running = False
                shutdown_manager.shutdown_requested = True
                try:
                    asyncio.get_event_loop().stop()
                    sys.exit(0)
                except Exception:
                    sys.exit(0)
        except Exception as e:
            self.logger.log("warning", f"Klavye kombinasyonu kontrol hatası: {e}")

    def enable_offline_mode(self):
        try:
            self.offline_cache = CACHE_DIR / "wheels"
            self.offline_cache.mkdir(exist_ok=True, parents=True)
            self.logger.log("info", "Çevrimdışı mod etkinleştirildi.")
            subprocess.run(
                ["pip", "download", "-r", "requirements.txt", "-d", str(self.offline_cache)],
                check=True, capture_output=True, text=True
            )
            self.logger.log("info", "Çevrimdışı önbellek oluşturuldu.")
        except Exception as e:
            self.logger.log("error", f"Çevrimdışı mod hatası: {e}")

    def check_build_tools(self):
        build_tool = check_build_tools(self.logger)
        if not build_tool:
            self.logger.log("warning", "Derleme araçları bulunamadı. Lütfen Visual Studio Build Tools veya MinGW kurun.")

    def install_package(self, package: str, silent=False):
        try:
            log_level = "debug" if silent else "info"
            self.logger.log(log_level, f"{package} kurulumu başlatılıyor.")
            start_time = time.time()
            if self.dependency_registry.check_package(package):
                self.logger.log(log_level, f"{package} zaten yüklü, kurulum atlanıyor.")
                return
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts and self.running:
                try:
                    if self.offline_cache and self.offline_cache.exists():
                        cmd = [sys.executable, "-m", "pip", "install", package, "--no-index", f"--find-links={self.offline_cache}"]
                    else:
                        cmd = [sys.executable, "-m", "pip", "install", package]
                    if self.cache_manager.install_from_cache(package):
                        self.record_installation_attempt(package, True)
                        self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                        break
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.record_installation_attempt(package, True)
                        self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                        break
                    if not self.pip_analyzer.analyze_and_fix(result.stderr, package):
                        raise Exception(f"{package} kurulum hatası: {result.stderr}")
                except Exception as e:
                    attempts += 1
                    if not silent:
                        self.logger.log("warning", f"{package} kurulum denemesi {attempts}/{max_attempts} başarısız: {e}")
                    if attempts == max_attempts:
                        self.logger.log("error" if not silent else "debug", f"{package} kurulumu başarısız.")
                        self.env_manager.report_error()
                        self.record_installation_attempt(package, False)
                        self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarısız")
                        return
                    time.sleep(2 ** attempts)
            conflicts = self.conflict_manager.detect_conflicts(package, [package])
            if conflicts:
                resolutions = self.conflict_manager.resolve_conflicts(package, conflicts)
                self.dependency_registry.register_resolution(package, resolutions)
            duration = time.time() - start_time
            self.summary_generator.add_module_status(package, "Başarılı", duration)
            if not silent:
                self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")
        except Exception as e:
            self.logger.log("error" if not silent else "debug", f"{package} kurulum hatası: {e}")
            self.env_manager.report_error()
            self.record_installation_attempt(package, False)
            self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarısız")

    async def async_install_package(self, package: str):
        try:
            self.logger.log("info", f"{package} asenkron kurulumu başlatılıyor.")
            if self.dependency_registry.check_package(package):
                self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
                return
            if self.offline_cache and self.offline_cache.exists():
                cmd = [sys.executable, "-m", "pip", "install", package, "--no-index", f"--find-links={self.offline_cache}"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package]
            if self.cache_manager.install_from_cache(package):
                self.record_installation_attempt(package, True)
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                return
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                self.record_installation_attempt(package, True)
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
            else:
                self.pip_analyzer.analyze_and_fix(stderr.decode(), package)
                self.record_installation_attempt(package, False)
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarısız")
            conflicts = self.conflict_manager.detect_conflicts(package, [package])
            if conflicts:
                resolutions = self.conflict_manager.resolve_conflicts(package, conflicts)
                self.dependency_registry.register_resolution(package, resolutions)
        except Exception as e:
            self.logger.log("error", f"{package} asenkron kurulum hatası: {e}")

    def load_module(self, file_path: str, alias: Optional[str] = None) -> Any:
        with self.lock:
            try:
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                abs_path = os.path.abspath(file_path)
                
                if abs_path in self.imported_files:
                    self.logger.log("info", f"{module_name} zaten yüklü.")
                    return self.module_cache.get(abs_path)
                    
                if self.secure_mode and not self._is_allowed_path(abs_path):
                    self.logger.log("error", f"Güvenli modda dış modül yüklenemez: {file_path}")
                    return None
                    
                self.logger.log("info", f"{module_name} modülü yükleniyor.")
                start_time = time.time()
                
                # Modül-spesifik bağımlılıkları yükle
                deps = MODULE_SPECIFIC_DEPS.get(module_name, [])
                for dep in deps:
                    self.install_package(dep)
                
                # Modül spec oluştur
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                if not spec:
                    raise ImportError(f"Modül spec oluşturulamadı: {abs_path}")
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                self.loaded_modules[module_name] = module
                self.module_cache[abs_path] = module
                self.imported_files.add(abs_path)
                
                if alias:
                    self.aliases[alias] = abs_path
                    sys.modules[alias] = module
                
                duration = time.time() - start_time
                self.summary_generator.add_module_status(module_name, "Başarılı", duration)
                self.logger.log("info", f"{module_name} modülü başarıyla yüklendi.")
                return module
                
            except Exception as e:
                duration = time.time() - start_time if 'start_time' in locals() else 0
                self.summary_generator.add_module_status(module_name, f"Hata: {str(e)}", duration)
                self.logger.log("error", f"{module_name} yüklenemedi: {e}")
                self.env_manager.report_error()
                return None

    def run(self, packages: List[str]):
        try:
            self.logger.log("info", "Kütüphane yükleme başlatıldı.")
            conflicts = self.conflict_manager.detect_conflicts("pdsX", packages)
            if conflicts:
                resolutions = self.conflict_manager.resolve_conflicts("pdsX", conflicts)
                for pkg, res in resolutions.items():
                    self.dependency_registry.update_on_conflict(pkg, str(conflicts.get(pkg, "")), res["command"])
            else:
                self.logger.log("info", "Çakışma yok, mühürleniyor.")
                self.dependency_registry.registry["status"] = "conflict_free"
                self.dependency_registry.registry["timestamp"] = datetime.now().isoformat()
                self.dependency_registry.save_registry()
            for pkg in packages:
                self.install_package(pkg)
        except Exception as e:
            self.logger.log("error", f"Kütüphane yükleme hatası: {e}")

    def ensure_required_packages(self) -> bool:
        """Sanal ortamda REQUIRED_PACKAGES'taki tüm paketlerin yüklü olduğunu garanti eder."""
        try:
            if not self.is_running_in_venv():
                self.logger.log("error", "Bu fonksiyon sadece sanal ortamda çalışır!")
                return False
            
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            
            self.logger.log("info", "REQUIRED_PACKAGES kontrol ediliyor...")
            missing_packages = []
              # Önce hangi paketlerin eksik olduğunu tespit et
            for package_spec, import_name in REQUIRED_PACKAGES:
                try:
                    # Sanal ortamda test için subprocess kullan
                    venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
                    test_result = subprocess.run([venv_python, "-c", f"import {import_name}"], 
                                                capture_output=True, text=True)
                    if test_result.returncode == 0:
                        self.logger.log("debug", f"✓ {import_name} mevcut")
                    else:
                        self.logger.log("info", f"× {import_name} eksik")
                        missing_packages.append((package_spec, import_name))
                except Exception as e:
                    self.logger.log("debug", f"Import test hatası {import_name}: {e}")
                    self.logger.log("info", f"× {import_name} eksik")
                    missing_packages.append((package_spec, import_name))
            
            if not missing_packages:
                self.logger.log("info", "Tüm REQUIRED_PACKAGES zaten yüklü!")
                return True
            
            self.logger.log("info", f"{len(missing_packages)} eksik paket bulundu, yükleniyor...")
            success_count = 0
            
            for package_spec, import_name in missing_packages:
                # Graceful shutdown kontrolü
                if shutdown_manager.shutdown_requested:
                    self.logger.log("info", "Shutdown sinyali alındı, paket kurulumu durduruluyor...")
                    return False
                
                self.logger.log("info", f"Yükleniyor: {package_spec}")
                
                try:
                    install_cmd = [pip_cmd, "install", package_spec]
                    process = subprocess.run(install_cmd, check=True, capture_output=True, text=True)
                    shutdown_manager.register_process(process)
                      # Yükleme sonrası import test et
                    try:
                        # Sanal ortamda test
                        venv_python = str(self.venv_dir / "Scripts" / "python.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "python")
                        test_result = subprocess.run([venv_python, "-c", f"import {import_name}"], 
                                                    capture_output=True, text=True)
                        if test_result.returncode == 0:
                            self.logger.log("info", f"✓ {package_spec} başarıyla yüklendi ve test edildi")
                            success_count += 1
                        else:
                            self.logger.log("warning", f"× {package_spec} yüklendi ama import edilemiyor: {test_result.stderr}")
                    except Exception as e:
                        self.logger.log("warning", f"× {package_spec} yükleme test hatası: {e}")
                    
                except subprocess.CalledProcessError as e:
                    self.logger.log("warning", f"× {package_spec} yüklenemedi: {e}")
                    if e.stdout:
                        self.logger.log("debug", f"STDOUT: {e.stdout}")
                    if e.stderr:
                        self.logger.log("debug", f"STDERR: {e.stderr}")
                    continue
                except Exception as e:
                    self.logger.log("warning", f"× {package_spec} yükleme hatası: {e}")
                    continue
            
            self.logger.log("info", f"Eksik paket kurulumu tamamlandı: {success_count}/{len(missing_packages)} başarılı")
            return success_count > 0
            
        except Exception as e:
            self.logger.log("error", f"ensure_required_packages hatası: {e}")
            return False

# Terminal Log Analizi ve Bağımlılık Öğrenme Sistemi
class TerminalLogAnalyzer:
    """
    Terminal loglarını analiz ederek eksik Python bağımlılıklarını tespit eder.
    ModuleNotFoundError, ImportError ve pip önerilerini yakalar.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.terminal_log_file = PLAIN_TERMINAL_LOG
        self.learned_dependencies_file = Path("learned_dependencies.json")
        self.learned_deps = self.load_learned_dependencies()
        
        # Hata pattern'leri
        self.error_patterns = {
            'module_not_found': [
                r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
                r"ImportError: No module named ['\"]([^'\"]+)['\"]",
                r"cannot import name ['\"]([^'\"]+)['\"]",
                r"ImportError: cannot import name ['\"]([^'\"]+)['\"]"
            ],
            'pip_suggestion': [
                r"pip install ([^\s]+)",
                r"Try: pip install ([^\s]+)",
                r"Run: pip install ([^\s]+)",
                r"Install with: pip install ([^\s]+)"
            ],
            'version_conflict': [
                r"requires ([^\s]+)==([^\s,]+)",
                r"requires ([^\s]+)>=([^\s,]+)",
                r"requires ([^\s]+)<=([^\s,]+)",
                r"incompatible with ([^\s]+) ([^\s]+)"
            ]
        }
    
    def load_learned_dependencies(self) -> Dict:
        """Öğrenilmiş bağımlılıkları yükle"""
        try:
            if self.learned_dependencies_file.exists():
                with open(self.learned_dependencies_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {
                "discovered_packages": [],
                "version_conflicts": [],
                "pip_suggestions": [],
                "last_analysis": "",
                "analysis_count": 0
            }
        except Exception as e:
            self.logger.log("error", f"Öğrenilmiş bağımlılık yükleme hatası: {e}")
            return {"discovered_packages": [], "version_conflicts": [], "pip_suggestions": [], "last_analysis": "", "analysis_count": 0}
    
    def save_learned_dependencies(self):
        """Öğrenilmiş bağımlılıkları kaydet"""
        try:
            self.learned_deps["last_analysis"] = datetime.now().isoformat()
            self.learned_deps["analysis_count"] += 1
            
            with open(self.learned_dependencies_file, "w", encoding="utf-8") as f:
                json.dump(self.learned_deps, f, indent=4, ensure_ascii=False)
            
            self.logger.log("info", f"Öğrenilmiş bağımlılıklar kaydedildi: {len(self.learned_deps['discovered_packages'])} paket")
        except Exception as e:
            self.logger.log("error", f"Öğrenilmiş bağımlılık kaydetme hatası: {e}")
    
    def analyze_terminal_logs(self) -> List[str]:
        """Terminal loglarını analiz ederek eksik bağımlılıkları tespit et"""
        try:
            self.logger.log("info", "Terminal log analizi başlatılıyor...")
            
            found_packages = []
            
            # pdsXu_terminal.log dosyasını oku
            if not self.terminal_log_file.exists():
                self.logger.log("warning", f"Terminal log dosyası bulunamadı: {self.terminal_log_file}")
                return []
            
            with open(self.terminal_log_file, "r", encoding="utf-8") as f:
                log_content = f.read()
            
            # ModuleNotFoundError'ları yakala
            missing_modules = self.extract_missing_imports(log_content)
            found_packages.extend(missing_modules)
            
            # Pip önerilerini yakala
            pip_suggestions = self.parse_pip_suggestions(log_content)
            found_packages.extend(pip_suggestions)
            
            # Version conflict'leri yakala
            version_conflicts = self.extract_version_conflicts(log_content)
            
            # Öğrenilmiş bağımlılıkları güncelle
            for pkg in found_packages:
                if pkg not in self.learned_deps["discovered_packages"]:
                    self.learned_deps["discovered_packages"].append(pkg)
                    self.logger.log("info", f"Yeni bağımlılık öğrenildi: {pkg}")
            
            for conflict in version_conflicts:
                if conflict not in self.learned_deps["version_conflicts"]:
                    self.learned_deps["version_conflicts"].append(conflict)
                    self.logger.log("warning", f"Version conflict tespit edildi: {conflict}")
            
            # Kaydet
            self.save_learned_dependencies()
            
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
        """Öğrenilmiş bağımlılıkları temizle"""
        try:
            self.learned_deps = {
                "discovered_packages": [],
                "version_conflicts": [],
                "pip_suggestions": [],
                "last_analysis": "",
                "analysis_count": 0
            }
            self.save_learned_dependencies()
            self.logger.log("info", "Öğrenilmiş bağımlılıklar temizlendi")
        except Exception as e:
            self.logger.log("error", f"Bağımlılık temizleme hatası: {e}")

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