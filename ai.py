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
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import hashlib
import asyncio
import psutil
from concurrent.futures import ThreadPoolExecutor

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
            logger = AdvancedLogger()
            logger.log("info", "NumPy eksik, yükleniyor...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"], check=True)
                import numpy as np
                numpy = np
            except Exception as e:
                logger.log("error", f"NumPy yükleme hatası: {e}")
                raise
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
            logger = AdvancedLogger()
            logger.log("info", "Scikit-learn eksik, yükleniyor...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2"], check=True)
                from sklearn.ensemble import IsolationForest as IF
                from sklearn.preprocessing import StandardScaler as SS
                from sklearn.neural_network import MLPClassifier as MLP
                from sklearn.tree import DecisionTreeClassifier as DTC
                IsolationForest = IF
                StandardScaler = SS
                MLPClassifier = MLP
                DecisionTreeClassifier = DTC
            except Exception as e:
                logger.log("error", f"Scikit-learn yükleme hatası: {e}")
                raise
    return IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier

def get_keyboard():
    """Keyboard modülünü lazy loading ile yükle"""
    global keyboard
    if keyboard is None:
        try:
            import keyboard as kb
            keyboard = kb
        except ImportError:
            logger = AdvancedLogger()
            logger.log("info", "Keyboard modülü eksik, yükleniyor...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "keyboard==0.13.5"], check=True)
                import keyboard as kb
                keyboard = kb
            except Exception as e:
                logger.log("error", f"Keyboard modülü yükleme hatası: {e}")
                raise
    return keyboard

class GracefulShutdownManager:
    def __init__(self):
        self.shutdown_requested = False
        self.active_processes = []
        self.cleanup_functions = []
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Sinyal handler'larını kurar. SIGINT, SIGTERM ve SIGBREAK sinyallerini yakalar."""
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self.signal_handler)
            atexit.register(self.cleanup)
            print("[PDS-X] Graceful shutdown sistemi aktif (Ctrl+C ile güvenli çıkış)")
        except Exception as e:
            print(f"[PDS-X] Signal handler kurulum hatası: {e}")
    
    def signal_handler(self, signum, frame):
        """Sinyal alındığında çalışır."""
        print(f"\n[PDS-X] Shutdown sinyali alındı (signal: {signum})")
        print("[PDS-X] Güvenli çıkış başlatılıyor...")
        self.shutdown_requested = True
        self.cleanup()
        
    def register_process(self, process):
        """Aktif süreci kaydeder."""
        if isinstance(process, subprocess.Popen):
            self.active_processes.append(process)
        
    def register_cleanup_function(self, func):
        """Temizlik fonksiyonunu kaydeder."""
        self.cleanup_functions.append(func)
        
    def cleanup(self):
        """
        Temizlik işlemleri: Aktif süreçleri sonlandırır, asenkron görevleri iptal eder ve kayıtlı temizlik fonksiyonlarını çalıştırır.
        """
        try:
            print("[PDS-X] Temizlik işlemleri başlatılıyor...")
            for process in self.active_processes:
                if hasattr(process, 'poll') and process.poll() is None:
                    print(f"[PDS-X] Process sonlandırılıyor: {process.pid}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"[PDS-X] Process zorla sonlandırılıyor: {process.pid}")
                        process.kill()
            loop = asyncio.get_event_loop()
            tasks = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            loop.run_until_complete(loop.shutdown_asyncgens())
            for cleanup_func in self.cleanup_functions:
                cleanup_func()
            print("[PDS-X] Temizlik tamamlandı.")
        except Exception as e:
            print(f"[PDS-X] Temizlik hatası: {e}")
        finally:
            if self.shutdown_requested:
                print("[PDS-X] Program güvenli şekilde sonlandırılıyor...")
                os._exit(0)

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
MAX_LOG_SIZE = 10 * 1024 * 1024
MAX_BACKUPS = 5

MODULE_NOT_FOUND_REGEX = re.compile(r"ModuleNotFoundError: No module named '(.*?)'")
PIP_SUGGESTION_REGEX = re.compile(r"pip install (.*?)(?:==([\d\.]+))?")
IMPORT_ERROR_REGEX = re.compile(r"ImportError: cannot import name '(.*?)' from '(.*?)'")

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

class TerminalLogAnalyzer:
    def __init__(self, logger: 'AdvancedLogger'):  # Forward reference için string kullanıyoruz
        self.logger = logger
        self.terminal_log = PLAIN_TERMINAL_LOG
        self.jsonl_log = TERMINAL_LOG

    def analyze_terminal_logs(self) -> Dict[str, Optional[str]]:
        """
        Terminal loglarını analiz eder, ModuleNotFoundError ve pip önerilerini çıkarır.
        Returns: {paket_adı: versiyon} sözlüğü.
        """
        try:
            dependencies = {}
            if self.terminal_log.exists():
                with open(self.terminal_log, "r", encoding="utf-8") as f:
                    for line in f:
                        match = MODULE_NOT_FOUND_REGEX.search(line)
                        if match:
                            package = match.group(1)
                            dependencies[package] = None
                            self.logger.log("info", f"ModuleNotFoundError tespit edildi: {package}")
                        match = PIP_SUGGESTION_REGEX.search(line)
                        if match:
                            package = match.group(1).strip()
                            version = match.group(2) if match.group(2) else None
                            dependencies[package] = version
                            self.logger.log("info", f"Pip önerisi tespit edildi: {package}=={version if version else 'latest'}")
                        match = IMPORT_ERROR_REGEX.search(line)
                        if match:
                            module = match.group(1)
                            package = match.group(2)
                            dependencies[package] = None
                            self.logger.log("info", f"ImportError tespit edildi: {module} from {package}")
            if self.jsonl_log.exists():
                with open(self.jsonl_log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            message = data.get("message", "")
                            match = MODULE_NOT_FOUND_REGEX.search(message)
                            if match:
                                package = match.group(1)
                                dependencies[package] = None
                                self.logger.log("info", f"JSONL: ModuleNotFoundError tespit edildi: {package}")
                            match = PIP_SUGGESTION_REGEX.search(message)
                            if match:
                                package = match.group(1).strip()
                                version = match.group(2) if match.group(2) else None
                                dependencies[package] = version
                                self.logger.log("info", f"JSONL: Pip önerisi tespit edildi: {package}=={version if version else 'latest'}")
                            match = IMPORT_ERROR_REGEX.search(message)
                            if match:
                                module = match.group(1)
                                package = match.group(2)
                                dependencies[package] = None
                                self.logger.log("info", f"JSONL: ImportError tespit edildi: {module} from {package}")
                        except json.JSONDecodeError:
                            continue
            return dependencies
        except Exception as e:
            self.logger.log("error", f"Terminal log analizi hatası: {e}")
            return {}

class AdvancedLogger:
    def __init__(self, dual_output: bool = True):
        self.logged_messages = set()
        self.last_log_time = {}
        self.es = None
        self.logger = None
        self.handlers = {}
        self.terminal_handler = None
        self.stdout_handler = None
        self.dual_output = dual_output
        
        try:
            LOG_DIR.mkdir(exist_ok=True)
            self.plain_terminal_log_file = PLAIN_TERMINAL_LOG
            if self.plain_terminal_log_file.exists():
                backup_file = self.plain_terminal_log_file.with_suffix(".bak")
                if backup_file.exists():
                    backup_file.unlink()
                shutil.move(self.plain_terminal_log_file, backup_file)
                print(f"[PDS-X] Önceki terminal log yedeklendi: {backup_file}")
            
            self.terminal_handler = logging.FileHandler(self.plain_terminal_log_file, encoding="utf-8")
            self.terminal_handler.setLevel(logging.INFO)
            self.terminal_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            
            self.handlers = {
                "info": logging.FileHandler(INFO_LOG, encoding="utf-8"),
                "warning": logging.FileHandler(WARNING_LOG, encoding="utf-8"),
                "error": logging.FileHandler(ERROR_LOG, encoding="utf-8"),
                "terminal": logging.FileHandler(TERMINAL_LOG, encoding="utf-8"),
            }
            
            jsonl_formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
            for level, handler in self.handlers.items():
                if level == "terminal":
                    handler.setLevel(logging.INFO)
                else:
                    handler.setLevel(getattr(logging, level.upper()))
                handler.setFormatter(jsonl_formatter)
            
            self.logger = logging.getLogger("autoimporter")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(self.terminal_handler)
            for handler in self.handlers.values():
                self.logger.addHandler(handler)
            
            if self.dual_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(logging.Formatter("[PDS-X] %(message)s"))
                self.logger.addHandler(console_handler)
            
            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
            sys.stdout = self.stdout_handler
            sys.stderr = Tee(sys.__stderr__, self.handlers["error"])
            
            if Elasticsearch:
                try:
                    self.es = Elasticsearch(["http://localhost:9200"])
                    if self.es.ping():
                        self.logger.log("info", "Elasticsearch sunucusuna bağlanıldı.")
                    else:
                        self.logger.log("warning", "Elasticsearch sunucusu erişilemez, yerel loglamaya geçiliyor.")
                        self.es = None
                except Exception as e:
                    self.logger.log("warning", f"Elasticsearch bağlantı hatası: {e}")
                    self.es = None
            
            self.logger.log("info", "Loglama sistemi başlatıldı.")
            
        except Exception as e:
            print(f"[PDS-X] Loglama başlatma hatası: {e}")

    def log(self, level: str, message: str):
        try:
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            current_time = datetime.now()
            if (message_hash not in self.logged_messages or 
                (current_time - self.last_log_time.get(message_hash, datetime.min)).total_seconds() > 1):
                if self.dual_output and level.lower() == "info":
                    print(f"[PDS-X] {message}")
                if self.logger:
                    log_level = getattr(logging, level.upper(), logging.INFO)
                    self.logger.log(log_level, message)
                self.logged_messages.add(message_hash)
                self.last_log_time[message_hash] = current_time
                if self.es:
                    try:
                        self.es.index(index="pdsx_logs", body={
                            "timestamp": current_time.isoformat(),
                            "level": level.upper(),
                            "message": message
                        })
                    except Exception as e:
                        if self.logger:
                            self.logger.log("warning", f"Elasticsearch bağlantısı başarısız: {e}")
                        self.es = None
                self.rotate_logs()
                
        except Exception as e:
            print(f"[PDS-X] Loglama hatası: {e}")

    def rotate_logs(self):
        try:
            log_files = [self.plain_terminal_log_file, INFO_LOG, WARNING_LOG, ERROR_LOG, TERMINAL_LOG]
            for log_file in log_files:
                if log_file.exists() and log_file.stat().st_size > MAX_LOG_SIZE:
                    handler = self.terminal_handler if log_file == self.plain_terminal_log_file else self.handlers.get(log_file.stem)
                    if handler:
                        handler.close()
                        self.logger.removeHandler(handler)
                        backup_path = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                        shutil.move(log_file, backup_path)
                        new_handler = logging.FileHandler(log_file, encoding="utf-8")
                        new_handler.setFormatter(handler.formatter)
                        new_handler.setLevel(handler.level)
                        if log_file == self.plain_terminal_log_file:
                            self.terminal_handler = new_handler
                            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
                            sys.stdout = self.stdout_handler
                        else:
                            self.handlers[log_file.stem] = new_handler
                        self.logger.addHandler(new_handler)
                        self.cleanup_old_backups(log_file)
                        
        except Exception as e:
            if self.logger:
                self.logger.log("error", f"Log döndürme hatası: {e}")
            else:
                print(f"[PDS-X] Log döndürme hatası: {e}")

    def cleanup_old_backups(self, log_file: Path):
        try:
            backups = sorted(log_file.parent.glob(f"{log_file.name}.*.bak"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[MAX_BACKUPS:]:
                old_backup.unlink()
                if self.logger:
                    self.logger.log("info", f"Eski yedek silindi: {old_backup}")
        except Exception as e:
            if self.logger:
                self.logger.log("error", f"Yedek silme hatası: {e}")
            else:
                print(f"[PDS-X] Yedek silme hatası: {e}")

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
            return {"packages": {}, "resolutions": {}, "auto_discovered": {}, "status": "", "timestamp": ""}
        except Exception as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası yükleme hatası: {e}")
            return {"packages": {}, "resolutions": {}, "auto_discovered": {}, "status": "", "timestamp": ""}

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
                    if elapsed.total_seconds() < 24 * 60 * 60:
                        self.logger.log("info", f"{package} zaten yüklü ve güncel.")
                        return True
            return False
        except Exception as e:
            self.logger.log("error", f"{package} kontrol hatası: {e}")
            return False

    def auto_update_from_terminal(self, dependencies: Dict[str, Optional[str]]):
        """Terminal loglarından öğrenilen bağımlılıkları auto_discovered altına ekler."""
        try:
            for package, version in dependencies.items():
                self.registry["auto_discovered"][package] = {
                    "version": version if version else "latest",
                    "timestamp": datetime.now().isoformat(),
                    "source": "terminal_log"
                }
            self.save_registry()
            self.logger.log("info", f"Terminalden öğrenilen {len(dependencies)} bağımlılık dependencies.json'a eklendi.")
        except Exception as e:
            self.logger.log("error", f"Terminal bağımlılık güncelleme hatası: {e}")

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
                            process = handler(package)
                            shutdown_manager.register_process(process)
                            self.logger.log("info", f"{package} düzeltildi: {process.stdout}")
                            return True
                        except subprocess.CalledProcessError as e:
                            self.logger.log("warning", f"Mirror {mirror} ile düzeltme başarısız: {e.output}")
                    return False
            self.logger.log("error", f"Bilinmeyen hata: {output}")
            return False
        except Exception as e:
            self.logger.log("error", f"Pip analiz hatası: {e}")
            return False

class CacheManager:
    def __init__(self, cache_dir: Path = CACHE_DIR / "wheels", logger: 'AdvancedLogger' = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.metadata_file = self.cache_dir / "packages.json"

    def install_from_cache(self, package: str) -> bool:
        """Önbellekten paket kurar. Hash doğrulaması yapar, bozuksa yeniden indirir."""
        try:
            cache_file = self.cache_dir / f"{package}.whl"
            if cache_file.exists():
                metadata = self.load_package_metadata()
                if package in metadata:
                    with open(cache_file, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != metadata[package]["hash"]:
                        self.logger.log("warning", f"{package} önbellek dosyası bozuk, yeniden indiriliyor.")
                        cache_file.unlink()
                        if not self._download_and_cache(package):
                            self.logger.log("error", f"{package} yeniden indirme başarısız.")
                            return False
                self.logger.log("info", f"{package} önbellekten yükleniyor.")
                process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(cache_file)], 
                    check=True, capture_output=True, text=True
                )
                shutdown_manager.register_process(process)
                self.logger.log("info", f"{package} önbellekten kuruldu: {process.stdout}")
                return True
            return self._download_and_cache(package)
        except Exception as e:
            self.logger.log("error", f"Önbellek yükleme hatası: {e}")
            return False

    def _download_and_cache(self, package: str) -> bool:
        try:
            self.logger.log("info", f"{package} indiriliyor ve önbelleğe alınıyor.")
            process = subprocess.run(
                [sys.executable, "-m", "pip", "download", package, "-d", str(self.cache_dir)],
                check=True, capture_output=True, text=True
            )
            shutdown_manager.register_process(process)
            self.logger.log("info", f"{package} indirildi: {process.stdout}")
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
                process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"{package}=={previous_version}"],
                    check=True, capture_output=True, text=True
                )
                shutdown_manager.register_process(process)
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
        """Önbellek temizleme: 30 günden eski .whl dosyalarını ve metadata’yı siler."""
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
                        self.logger.log("info", f"Eski önbellek dosyası silindi: {cache_file}")
                    del metadata[pkg]
                    self.logger.log("info", f"Metadata’dan kaldırıldı: {pkg}")
            self.save_package_metadata(metadata)
        except Exception as e:
            self.logger.log("error", f"Önbellek temizleme hatası: {e}")

class EnvManager:
    def __init__(self, venv_dir: Path = VENV_DIR, logger: 'AdvancedLogger' = None):
        self.venv_dir = venv_dir
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger if logger is not None else AdvancedLogger()
        self.python_path = None
        self.terminal_analyzer = TerminalLogAnalyzer(self.logger)

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
            min_required = 300 * 1024 * 1024
            if free < min_required:
                self.logger.log("error", f"Yetersiz disk alanı: {free // (1024*1024)} MB mevcut, 300 MB gerekli.")
                return None
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            installer_path = "python310_installer.exe"
            self.logger.log("info", "Python 3.10 indiriliyor...")
            process = subprocess.run(["curl", "-o", installer_path, installer_url], check=True, capture_output=True, text=True)
            shutdown_manager.register_process(process)
            self.logger.log("info", "Python 3.10 kuruluyor...")
            process = subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True, capture_output=True, text=True)
            shutdown_manager.register_process(process)
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