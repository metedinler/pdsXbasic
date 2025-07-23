# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.5
# Date: June 23, 2025
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

# Lazy imports
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
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BACKUPS = 5

MODULE_NOT_FOUND_REGEX = re.compile(r"ModuleNotFoundError: No module named '(.*?)'")
PIP_SUGGESTION_REGEX = re.compile(r"pip install (.*?)(?:==([\d\.]+))?")
IMPORT_ERROR_REGEX = re.compile(r"ImportError: cannot import name '(.*?)' from '(.*?)'")
VERSION_CONFLICT_REGEX = re.compile(r"(.+?) has requirement (.+?), but you have (.+?)\.")

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
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.terminal_log = PLAIN_TERMINAL_LOG
        self.jsonl_log = TERMINAL_LOG
        self.last_analyzed_hash = None

    def map_module_to_package(self, module: str) -> str:
        """Modül adını paket adına eşleştirir."""
        mapping = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "sklearn": "scikit-learn",
            "dateutil": "python-dateutil",
        }
        return mapping.get(module, module)

    def extract_version_conflicts(self, message: str) -> Dict[str, str]:
        """Versiyon çakışmalarını ayrıştırır."""
        conflicts = {}
        match = VERSION_CONFLICT_REGEX.search(message)
        if match:
            pkg, required, installed = match.groups()
            conflicts[pkg] = f"Required: {required}, Installed: {installed}"
        return conflicts

    def analyze_terminal_logs(self) -> Dict[str, Optional[str]]:
        """
        Terminal loglarını analiz eder, ModuleNotFoundError, pip önerileri ve versiyon çakışmalarını çıkarır.
        Returns: {paket_adı: versiyon} sözlüğü.
        """
        try:
            dependencies = {}
            current_hash = hashlib.sha256()
            if self.terminal_log.exists():
                with open(self.terminal_log, "r", encoding="utf-8") as f:
                    content = f.read()
                    current_hash.update(content.encode())
                    if self.last_analyzed_hash == current_hash.hexdigest():
                        self.logger.log("debug", "Loglar daha önce analiz edildi, tekrar işlenmedi.")
                        return dependencies
                    for line in content.splitlines():
                        match = MODULE_NOT_FOUND_REGEX.search(line)
                        if match:
                            package = self.map_module_to_package(match.group(1))
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
                            package = self.map_module_to_package(match.group(2))
                            dependencies[package] = None
                            self.logger.log("info", f"ImportError tespit edildi: {module} from {package}")
                        conflicts = self.extract_version_conflicts(line)
                        for pkg, conflict in conflicts.items():
                            dependencies[pkg] = None
                            self.logger.log("info", f"Versiyon çakışması tespit edildi: {pkg} - {conflict}")
            if self.jsonl_log.exists():
                with open(self.jsonl_log, "r", encoding="utf-8") as f:
                    content = f.read()
                    current_hash.update(content.encode())
                    if self.last_analyzed_hash == current_hash.hexdigest():
                        return dependencies
                    for line in content.splitlines():
                        try:
                            data = json.loads(line)
                            message = data.get("message", "")
                            match = MODULE_NOT_FOUND_REGEX.search(message)
                            if match:
                                package = self.map_module_to_package(match.group(1))
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
                                package = self.map_module_to_package(match.group(2))
                                dependencies[package] = None
                                self.logger.log("info", f"JSONL: ImportError tespit edildi: {module} from {package}")
                            conflicts = self.extract_version_conflicts(message)
                            for pkg, conflict in conflicts.items():
                                dependencies[pkg] = None
                                self.logger.log("info", f"JSONL: Versiyon çakışması tespit edildi: {pkg} - {conflict}")
                        except json.JSONDecodeError:
                            continue
            self.last_analyzed_hash = current_hash.hexdigest()
            return dependencies
        except Exception as e:
            self.logger.log("error", f"Terminal log analizi hatası: {e}")
            return {}

class RealTimeLogMonitor:
    def __init__(self, logger: AdvancedLogger, log_file: Path = PLAIN_TERMINAL_LOG):
        self.logger = logger
        self.log_file = log_file
        self.realtime_patterns = [
            MODULE_NOT_FOUND_REGEX,
            PIP_SUGGESTION_REGEX,
            IMPORT_ERROR_REGEX,
            VERSION_CONFLICT_REGEX,
        ]
        self.running = True
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def monitor(self):
        """Log dosyasını gerçek zamanlı olarak izler."""
        try:
            while self.running and not shutdown_manager.shutdown_requested:
                if self.log_file.exists():
                    with open(self.log_file, "r", encoding="utf-8") as f:
                        f.seek(0, os.SEEK_END)
                        while self.running:
                            line = f.readline()
                            if not line:
                                time.sleep(0.1)
                                continue
                            self._process_line(line.strip())
                else:
                    time.sleep(1)
        except Exception as e:
            self.logger.log("error", f"Gerçek zamanlı log izleme hatası: {e}")

    def _process_line(self, line: str):
        """Log satırını analiz eder ve çakışmaları işler."""
        for pattern in self.realtime_patterns:
            match = pattern.search(line)
            if match:
                if pattern == MODULE_NOT_FOUND_REGEX:
                    self._handle_missing_module(match.group(1))
                elif pattern == VERSION_CONFLICT_REGEX:
                    self.logger.log("warning", f"Versiyon çakışması: {match.group(0)}")
                else:
                    self.logger.log("info", f"Gerçek zamanlı tespit: {match.group(0)}")

    def _handle_missing_module(self, module: str):
        """Eksik modülü ConflictManager’a bildirir."""
        package = TerminalLogAnalyzer(self.logger).map_module_to_package(module)
        self.logger.log("info", f"Eksik modül tespit edildi: {module} -> {package}")
        conflict_manager = ConflictManager(self.logger)
        conflict_manager.detect_conflicts(module, [package])

    def get_auto_install_queue(self) -> List[str]:
        """Otomatik kurulum için kuyruk döndürür."""
        return [pkg for pkg in self._process_line.cache if pkg]

    def stop(self):
        self.running = False
        self.thread.join()

class AdvancedLogger:
    def __init__(self, dual_output: bool = True, silent_mode: bool = False):
        self.logged_messages = set()
        self.last_log_time = {}
        self.es = None
        self.logger = None
        self.handlers = {}
        self.terminal_handler = None
        self.stdout_handler = None
        self.dual_output = dual_output
        self.silent_mode = silent_mode
        
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
            
            if self.dual_output and not self.silent_mode:
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

    async def async_log(self, level: str, message: str):
        """Asenkron log yazımı"""
        import aiofiles
        try:
            async with aiofiles.open(self.plain_terminal_log_file, "a", encoding="utf-8") as f:
                await f.write(f"{datetime.now()} - {level.upper()} - {message}\n")
        except Exception as e:
            print(f"[PDS-X] Asenkron log hatası: {e}")

    def log(self, level: str, message: str):
        try:
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            current_time = datetime.now()
            if (message_hash not in self.logged_messages or 
                (current_time - self.last_log_time.get(message_hash, datetime.min)).total_seconds() > 1):
                if self.dual_output and level.lower() == "info" and not self.silent_mode:
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
                asyncio.run(self.async_log(level, message))
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

    def prioritize_dependencies(self, dependencies: Dict[str, Optional[str]]) -> List[str]:
        """Bağımlılıkları önceliklendirir."""
        return sorted(dependencies.keys(), key=lambda x: len(dependencies[x]) if dependencies[x] else 0, reverse=True)

    def migrate_learned_deps(self, learned_file: Path):
        """learned_dependencies.json’dan verileri taşır."""
        try:
            if learned_file.exists():
                with open(learned_file, "r", encoding="utf-8") as f:
                    learned = json.load(f)
                for pkg, info in learned.items():
                    self.registry["auto_discovered"][pkg] = info
                self.save_registry()
                self.logger.log("info", f"Learned dependencies taşındı: {len(learned)} kayıt")
        except Exception as e:
            self.logger.log("error", f"Learned dependencies taşıma hatası: {e}")

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
    def __init__(self, cache_dir: Path = CACHE_DIR / "wheels", logger: AdvancedLogger = None):
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
                    check=True, capture_output=True, text=True, timeout=300
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
                check=True, capture_output=True, text=True, timeout=300
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
                    check=True, capture_output=True, text=True, timeout=300
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
    def __init__(self, venv_dir: Path = VENV_DIR, logger: AdvancedLogger = None):
        self.venv_dir = venv_dir
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger if logger is not None else AdvancedLogger()
        self.python_path = None
        self.terminal_analyzer = TerminalLogAnalyzer(self.logger)
        self.error_history = []

    def check_and_recreate(self):
        try:
            if self.error_count >= self.max_errors:
                self.logger.log("warning", "İzole ortamda çok fazla hata. Ortam siliniyor ve yeniden oluşturuluyor.")
                shutil.rmtree(self.venv_dir, ignore_errors=True)
                python_path = self.find_python310()
                if python_path:
                    subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True, capture_output=True, text=True, timeout=300)
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
            self.error_history.append({"timestamp": datetime.now().isoformat(), "error": "Environment error"})
            self.check_and_recreate()
        except Exception as e:
            self.logger.log("error", f"Hata raporlama hatası: {e}")

    def find_python310(self) -> Optional[str]:
        try:
            self.logger.log("info", "Python 3.10 aranıyor...")
            for exe in ["python3.10", "python310", "python"]:
                path = shutil.which(exe)
                if path:
                    out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT, timeout=300)
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
            process = subprocess.run(["curl", "-o", installer_path, installer_url], check=True, capture_output=True, text=True, timeout=300)
            shutdown_manager.register_process(process)
            self.logger.log("info", "Python 3.10 kuruluyor...")
            process = subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True, capture_output=True, text=True, timeout=300)
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

    def update_pip_if_needed(self, force_latest: bool = False, silent: bool = False) -> bool:
        try:
            pip_cmd = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip")
            if not os.path.exists(pip_cmd):
                self.logger.log("error", "Pip komutu bulunamadı!")
                return False
            try:
                result = subprocess.run([pip_cmd, "--version"], capture_output=True, text=True, timeout=300)
                current_version = result.stdout.strip() if result.returncode == 0 else "unknown"
                log_level = "debug" if silent else "info"
                self.logger.log(log_level, f"Mevcut pip sürümü: {current_version}")
            except Exception as e:
                self.logger.log("warning", f"Pip sürüm kontrolü başarısız: {e}")
                current_version = "unknown"
            target_version = "latest" if force_latest else "21.2.4"
            pip_package = "pip" if force_latest else "pip==21.2.4"
            log_level = "debug" if silent else "info"
            self.logger.log(log_level, f"Pip hedef sürümü: {target_version}")
            if not force_latest:
                self.logger.log("debug", "Python 3.10 için pip 21.2.4 kullanılıyor (o dönemin stable sürümü)")
            install_args = [pip_cmd, "install", "--upgrade"]
            if silent:
                install_args.extend(["--quiet", "--no-warn-script-location"])
            install_args.append(pip_package)
            process = subprocess.run(install_args, capture_output=True, text=True, timeout=300)
            shutdown_manager.register_process(process)
            if shutdown_manager.shutdown_requested:
                self.logger.log("warning", "Shutdown sinyali alındı, pip güncellemesi iptal edildi")
                return False
            if process.returncode == 0:
                if self.test_pip_compatibility():
                    self.logger.log(log_level, f"Pip başarıyla güncellendi: {target_version}")
                    return True
                else:
                    self.logger.log("warning", "Pip sürümü uyumsuz, 21.2.4'e dönülüyor...")
                    process = subprocess.run([pip_cmd, "install", "--force-reinstall", "pip==21.2.4"], check=True, capture_output=True, text=True, timeout=300)
                    shutdown_manager.register_process(process)
                    return True
            else:
                self.logger.log("warning", f"Pip güncelleme başarısız: {process.stderr}")
                return False
        except Exception as e:
            self.logger.log("error", f"Pip güncelleme hatası: {e}")
            return False

    def test_pip_compatibility(self):
        try:
            pip_cmd = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip")
            process = subprocess.run([pip_cmd, "check"], capture_output=True, text=True, timeout=300)
            shutdown_manager.register_process(process)
            if "no conflicts" not in process.stdout.lower():
                self.logger.log("warning", "Pip sürümü uyumsuz, önceki sürüme dönülüyor...")
                process = subprocess.run([pip_cmd, "install", "--force-reinstall", "pip==21.2.4"], check=True, capture_output=True, text=True, timeout=300)
                shutdown_manager.register_process(process)
                return False
            return True
        except Exception as e:
            self.logger.log("error", f"Pip uyumluluk testi hatası: {e}")
            return False

    def setup_environment(self) -> bool:
        """
        Sanal ortamı hazırlar, REQUIRED_PACKAGES’ı kurar, terminal loglarından öğrenilen bağımlılıkları yükler.
        Hata alınmayana kadar döngü devam eder.
        """
        try:
            self.logger.log("info", "Ortam hazırlanıyor...")
            if self.is_running_in_venv():
                self.logger.log("info", "Zaten sanal ortamda çalışıyoruz.")
                return self.ensure_required_packages()
            python_path = self.find_python310()
            if not python_path:
                self.logger.log("info", "Python 3.10 bulunamadı, indiriliyor...")
                python_path = self.download_and_install_python310()
            if not python_path:
                self.logger.log("error", "Python 3.10 kurulamadı!")
                return False
            self.logger.log("info", f"Python 3.10 hazır: {python_path}")
            if not self.venv_dir.exists():
                self.logger.log("info", "Sanal ortam oluşturuluyor...")
                process = subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True, timeout=300)
                shutdown_manager.register_process(process)
            self.add_python_to_path()
            pip_cmd = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip")
            if not self.update_pip_if_needed(force_latest=False, silent=False):
                self.logger.log("warning", "Pip güncelleme başarısız, devam ediliyor")
            process = subprocess.run([pip_cmd, "install", "wheel"], check=True, timeout=300)
            shutdown_manager.register_process(process)
            if shutdown_manager.shutdown_requested:
                self.logger.log("info", "Shutdown sinyali alındı, kurulum durduruluyor...")
                return False
            process = subprocess.run([pip_cmd, "install", "numpy==1.26.4"], check=True, timeout=300)
            shutdown_manager.register_process(process)
            if shutdown_manager.shutdown_requested:
                self.logger.log("info", "Shutdown sinyali alındı, kurulum durduruluyor...")
                return False
            return self.ensure_required_packages()
        except Exception as e:
            self.logger.log("error", f"Ortam hazırlama hatası: {e}")
            return False

    def restart_in_venv(self, args: Optional[List[str]] = None) -> bool:
        try:
            if not self.venv_dir.exists():
                self.logger.log("error", "Sanal ortam bulunamadı!")
                return False
            venv_python = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python")
            if not os.path.exists(venv_python):
                self.logger.log("error", f"Sanal ortam Python'u bulunamadı: {venv_python}")
                return False
            self.logger.log("info", "Program sanal ortamda yeniden başlatılıyor...")
            current_args = args if args else sys.argv
            new_cmd = [venv_python] + current_args
            self.logger.log("info", f"Yeni komut: {' '.join(new_cmd)}")
            os.execv(venv_python, new_cmd)
        except Exception as e:
            self.logger.log("error", f"Sanal ortamda yeniden başlatma hatası: {e}")
            return False

    def is_running_in_venv(self) -> bool:
        try:
            venv_python = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python")
            current_python = sys.executable
            if os.path.normpath(current_python) == os.path.normpath(venv_python):
                self.logger.log("debug", f"Sanal ortam tespit edildi (executable): {current_python}")
                return True
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                if str(self.venv_dir.absolute()) in sys.prefix:
                    self.logger.log("debug", f"Sanal ortam tespit edildi (prefix): {sys.prefix}")
                    return True
            virtual_env = os.environ.get('VIRTUAL_ENV')
            if virtual_env and os.path.normpath(virtual_env) == os.path.normpath(str(self.venv_dir)):
                self.logger.log("debug", f"Sanal ortam tespit edildi (VIRTUAL_ENV): {virtual_env}")
                return True
            if str(self.venv_dir / "Lib" / "site-packages") in sys.path:
                self.logger.log("debug", f"Sanal ortam tespit edildi (sys.path): {self.venv_dir}")
                return True
            self.logger.log("debug", f"Sanal ortam tespit edilemedi. current: {current_python}, venv: {venv_python}")
            return False
        except Exception as e:
            self.logger.log("error", f"Sanal ortam kontrol hatası: {e}")
            return False

    def check_multiple_packages(self, packages):
        try:
            venv_python = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python")
            imports = ";".join(f"import {import_name}" for _, import_name in packages)
            process = subprocess.run([venv_python, "-c", imports], capture_output=True, text=True, timeout=300)
            shutdown_manager.register_process(process)
            if process.returncode != 0:
                self.logger.log("warning", f"Toplu import testi başarısız: {process.stderr}")
            return process.returncode == 0
        except Exception as e:
            self.logger.log("debug", f"Toplu import testi hatası: {e}")
            return False

    def ensure_required_packages(self) -> bool:
        """
        REQUIRED_PACKAGES ve dependencies.json’daki paketleri kurar, terminal loglarından öğrenilen bağımlılıkları yükler.
        Hata alınmayana kadar döngü devam eder.
        """
        try:
            if not self.is_running_in_venv():
                self.logger.log("error", "Bu fonksiyon sadece sanal ortamda çalışır!")
                return False
            pip_cmd = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip")
            dependency_registry = DependencyRegistry(self.logger)
            self.logger.log("info", "REQUIRED_PACKAGES kontrol ediliyor...")
            max_attempts = 5
            attempt = 0
            monitor = RealTimeLogMonitor(self.logger)
            while attempt < max_attempts:
                if shutdown_manager.shutdown_requested:
                    self.logger.log("info", "Shutdown sinyali alındı, paket kurulumu durduruluyor...")
                    monitor.stop()
                    return False
                missing_packages = []
                if self.check_multiple_packages(REQUIRED_PACKAGES):
                    self.logger.log("info", "Tüm REQUIRED_PACKAGES zaten yüklü!")
                else:
                    for package_spec, import_name in REQUIRED_PACKAGES:
                        try:
                            venv_python = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python")
                            process = subprocess.run([venv_python, "-c", f"import {import_name}"], capture_output=True, text=True, timeout=300)
                            shutdown_manager.register_process(process)
                            if process.returncode == 0:
                                self.logger.log("debug", f"✓ {import_name} mevcut")
                            else:
                                self.logger.log("info", f"× {import_name} eksik")
                                missing_packages.append((package_spec, import_name))
                        except Exception as e:
                            self.logger.log("debug", f"Import test hatası {import_name}: {e}")
                            missing_packages.append((package_spec, import_name))
                success_count = 0
                for package_spec, import_name in missing_packages:
                    self.logger.log("info", f"Yükleniyor: {package_spec}")
                    try:
                        install_cmd = [pip_cmd, "install", package_spec]
                        process = subprocess.run(install_cmd, check=True, capture_output=True, text=True, timeout=300)
                        shutdown_manager.register_process(process)
                        try:
                            process = subprocess.run([venv_python, "-c", f"import {import_name}"], capture_output=True, text=True, timeout=300)
                            shutdown_manager.register_process(process)
                            if process.returncode == 0:
                                self.logger.log("info", f"✓ {package_spec} başarıyla yüklendi ve test edildi")
                                dependency_registry.register_package(package_spec, package_spec.split("==")[1] if "==" in package_spec else "latest", "Başarılı")
                                success_count += 1
                            else:
                                self.logger.log("warning", f"× {package_spec} yüklendi ama import edilemiyor: {process.stderr}")
                        except Exception as e:
                            self.logger.log("warning", f"× {package_spec} yükleme test hatası: {e}")
                    except subprocess.CalledProcessError as e:
                        self.logger.log("warning", f"× {package_spec} yüklenemedi: {e}")
                        if e.stdout:
                            self.logger.log("debug", f"STDOUT: {e.stdout}")
                        if e.stderr:
                            self.logger.log("debug", f"STDERR: {e.stderr}")
                        continue
                self.logger.log("info", f"REQUIRED_PACKAGES kurulumu: {success_count}/{len(missing_packages)} başarılı")
                learned_deps = self.terminal_analyzer.analyze_terminal_logs()
                if learned_deps:
                    self.logger.log("info", f"Terminalden öğrenilen {len(learned_deps)} bağımlılık bulundu.")
                    dependency_registry.auto_update_from_terminal(learned_deps)
                    for package, version in learned_deps.items():
                        if shutdown_manager.shutdown_requested:
                            self.logger.log("info", "Shutdown sinyali alındı, paket kurulumu durduruluyor...")
                            monitor.stop()
                            return False
                        package_spec = f"{package}=={version}" if version else package
                        self.logger.log("info", f"Öğrenilen bağımlılık yükleniyor: {package_spec}")
                        try:
                            install_cmd = [pip_cmd, "install", package_spec]
                            process = subprocess.run(install_cmd, check=True, capture_output=True, text=True, timeout=300)
                            shutdown_manager.register_process(process)
                            self.logger.log("info", f"✓ {package_spec} başarıyla yüklendi")
                            dependency_registry.register_package(package_spec, version if version else "latest", "Başarılı")
                            success_count += 1
                        except subprocess.CalledProcessError as e:
                            self.logger.log("warning", f"× {package_spec} yüklenemedi: {e}")
                            if e.stdout:
                                self.logger.log("debug", f"STDOUT: {e.stdout}")
                            if e.stderr:
                                self.logger.log("debug", f"STDERR: {e.stderr}")
                            continue
                auto_discovered = dependency_registry.registry.get("auto_discovered", {})
                for package, info in auto_discovered.items():
                    if shutdown_manager.shutdown_requested:
                        self.logger.log("info", "Shutdown sinyali alındı, paket kurulumu durduruluyor...")
                        monitor.stop()
                        return False
                    package_spec = f"{package}=={info['version']}" if info['version'] != "latest" else package
                    if not dependency_registry.check_package(package_spec):
                        self.logger.log("info", f"Auto-discovered bağımlılık yükleniyor: {package_spec}")
                        try:
                            install_cmd = [pip_cmd, "install", package_spec]
                            process = subprocess.run(install_cmd, check=True, capture_output=True, text=True, timeout=300)
                            shutdown_manager.register_process(process)
                            self.logger.log("info", f"✓ {package_spec} başarıyla yüklendi")
                            dependency_registry.register_package(package_spec, info['version'], "Başarılı")
                            success_count += 1
                        except subprocess.CalledProcessError as e:
                            self.logger.log("warning", f"× {package_spec} yüklenemedi: {e}")
                            if e.stdout:
                                self.logger.log("debug", f"STDOUT: {e.stdout}")
                            if e.stderr:
                                self.logger.log("debug", f"STDERR: {e.stderr}")
                            continue
                if success_count == len(missing_packages) + len(learned_deps) + len(auto_discovered):
                    self.logger.log("info", "Tüm bağımlılıklar başarıyla yüklendi.")
                    monitor.stop()
                    return True
                attempt += 1
                self.logger.log("info", f"Deneme {attempt}/{max_attempts}: Hatalar tespit edildi, yeniden deneniyor...")
            self.logger.log("error", f"Maksimum deneme sayısına ulaşıldı, bazı bağımlılıklar yüklenemedi.")
            monitor.stop()
            return False
        except Exception as e:
            self.logger.log("error", f"ensure_required_packages hatası: {e}")
            monitor.stop()
            return False

class ConflictManager:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = defaultdict(list)
        self.scientific_utils = ScientificUtils(logger)
        self.terminal_analyzer = TerminalLogAnalyzer(logger)

    def clean_version(self, version: str) -> str:
        return re.sub(r'[=<>]', '', version).strip()

    def build_decision_tree(self, conflicts: Dict):
        try:
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
                process = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=300)
                shutdown_manager.register_process(process)
                if "no conflicts" not in process.stdout.lower():
                    conflicts[dep] = process.stdout
                    self.logger.log("warning", f"Çakışma tespit edildi: {dep}, {process.stdout}")
                else:
                    self.logger.log("info", f"{dep} için çakışma bulunamadı.")
            self.conflicts[module_name] = conflicts
            return conflicts
        except Exception as e:
            self.logger.log("error", f"Çakışma kontrol hatası: {e}")
            return {}

    def analyze_and_learn_from_terminal(self) -> Dict[str, Optional[str]]:
        """Terminal loglarından çakışma ve bağımlılık bilgisi öğrenir."""
        return self.terminal_analyzer.analyze_terminal_logs()

    def update_dependencies_json(self, dependency_registry: DependencyRegistry, learned_deps: Dict[str, Optional[str]]):
        """Öğrenilen bağımlılıkları dependencies.json’a ekler."""
        try:
            dependency_registry.auto_update_from_terminal(learned_deps)
        except Exception as e:
            self.logger.log("error", f"Dependencies.json güncelleme hatası: {e}")

    def resolve_conflicts(self, module_name: str, conflicts: Dict, dependency_registry: DependencyRegistry) -> Dict:
        try:
            resolutions = {}
            clf = self.build_decision_tree(conflicts)
            learned_deps = self.analyze_and_learn_from_terminal()
            prioritized_deps = dependency_registry.prioritize_dependencies(learned_deps)
            if clf:
                for dep in prioritized_deps:
                    if dep in conflicts:
                        issue = conflicts[dep]
                        features = [len(issue), issue.count("=="), issue.count("<"), issue.count(">")]
                        prediction = clf.predict([features])[0]
                        cmd = f"pip install {self.clean_version(dep)} --force-reinstall" if prediction == 1 else f"pip install {self.clean_version(dep)}"
                        try:
                            process = subprocess.run(cmd.split(), capture_output=True, text=True, check=True, timeout=300)
                            shutdown_manager.register_process(process)
                            resolutions[dep] = {"command": cmd, "reason": issue, "output": process.stdout}
                            self.logger.log("info", f"Çakışma çözüldü: {dep}, {issue}")
                        except subprocess.CalledProcessError as e:
                            self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
                            self.logger.log("info", "Karmaşık çakışma tespit edildi, manuel müdahale önerilir.")
            neural_resolutions = self.neural_conflict_resolution(module_name, conflicts)
            resolutions.update(neural_resolutions)
            self.resolutions[module_name] = resolutions
            if learned_deps:
                self.update_dependencies_json(dependency_registry, learned_deps)
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
        """Modül bağımlılıklarını analiz ederek kritik modülleri ve ilişki yoğunluğunu belirler."""
        try:
            np = get_numpy()
            if np is None:
                self.logger.log("error", "NumPy yüklü değil")
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

class ModuleAnalyzer:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.log_file = TERMINAL_LOG

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
        """Loglardan çakışma mesajlarını çıkarır."""
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

class AsyncDownloadManager:
    def __init__(self, max_workers: int = 4, cache_dir: Path = CACHE_DIR / "wheels", logger: AdvancedLogger = None):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.download_stats = {}

    def download_package(self, task: Dict) -> Dict:
        """Belirtilen paketi indirir ve önbelleğe kaydeder."""
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
                    process = subprocess.run(
                        ["curl", "-o", str(cache_file), url], capture_output=True, text=True, check=True, timeout=300
                    )
                    shutdown_manager.register_process(process)
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
                results = await asyncio.gather(*[asyncio.get_event_loop().run_in_executor(None, future.result) for future in download_futures])
                for result in results:
                    if result["success"]:
                        pkg = result["package"]
                        executor.submit(self.install_package, pkg)
            return self.download_stats
        except Exception as e:
            self.logger.log("error", f"Paralel indirme ve kurulum hatası: {e}")
            return {}

    def install_package(self, package: str):
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package], check=True, capture_output=True, text=True, timeout=300
            )
            shutdown_manager.register_process(process)
            self.logger.log("info", f"{package} kuruldu.")
        except Exception as e:
            self.logger.log("error", f"{package} kurulum hatası: {e}")

class ScientificUtils:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        IsolationForest, StandardScaler, MLPClassifier, DecisionTreeClassifier = get_sklearn_components()
        if StandardScaler and IsolationForest:
            self.scaler = StandardScaler()
            self.isolation_forest = IsolationForest(contamination=0.1)
        else:
            self.scaler = None
            self.isolation_forest = None
        self._lock = threading.Lock()

    def quantum_load_simulation(self, metrics: List[float]) -> Dict:
        """Metrikleri analiz ederek istatistiksel özet üretir (ortalama, standart sapma, aykırı değerler)."""
        with self._lock:
            try:
                if not metrics:
                    raise ValueError("Metrik listesi boş olamaz. Lütfen geçerli metrikler sağlayın.")
                np = get_numpy()
                if np is None:
                    self.logger.log("error", "NumPy yüklü değil")
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
        """Sistem kaynaklarını (CPU, bellek, disk) analiz eder ve kaos yük tahmini üretir."""
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
        """Bağımlılık grafiğini analiz eder ve döngüleri önleyerek optimize edilmiş sıralama üretir."""
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
        """Sistem kaynaklarını analiz eder ve yük dengesizliklerini tespit eder."""
        try:
            if not resources:
                raise ValueError("Kaynak listesi boş olamaz")
            np = get_numpy()
            if np is None:
                self.logger.log("error", "NumPy yüklü değil")
                return {"error": "NumPy yüklü değil"}
            if self.scaler is None:
                self.logger.log("error", "StandardScaler yüklü değil")
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
        """Modülleri doğrular ve bir blockchain benzeri doğrulama zinciri oluşturur."""
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

class ModuleSummaryGenerator:
    def __init__(self, logger: AdvancedLogger):
        self.summaries = []
        self.logger = logger

    def add_module_status(self, module_name: str, status: str, duration: float):
        """Modül kurulum durumunu kaydeder ve özet tablosuna ekler."""
        try:
            self.summaries.append({"module": module_name, "status": status, "duration": duration})
            self.logger.log("info", f"Modül durumu eklendi: {module_name}, {status}, {duration:.2f} saniye")
        except Exception as e:
            self.logger.log("error", f"Modül durumu ekleme hatası: {e}")

    def print_summary(self):
        """Kurulum özet tablosunu konsola ve loga yazdırır."""
        try:
            if not self.summaries:
                self.logger.log("info", "Henüz modül yüklenmedi.")
                return
            self.logger.log("info", "Modül Yükleme Özeti:")
            header = f"{Fore.CYAN}┌────────────────────┬──────────────┬──────────┐\n│ Modül Adı          │ Durum        │ Süre (sn)│\n├────────────────────┼──────────────┼──────────┤{Style.RESET_ALL}"
            print(header)
            for s in self.summaries:
                color = Fore.GREEN if "Başarılı" in s["status"] or "Atlandı" in s["status"] else Fore.RED
                row = f"{color}│ {s['module']:<18} │ {s['status']:<12} │ {s['duration']:.2f}      │{Style.RESET_ALL}"
                print(row)
            footer = f"{Fore.CYAN}└────────────────────┴──────────────┴──────────┘{Style.RESET_ALL}"
            print(footer)
        except Exception as e:
            self.logger.log("error", f"Özet tablo yazdırma hatası: {e}")

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
                shutdown_manager.register_cleanup_function(self.cleanup_on_shutdown)
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
                self.monitor = RealTimeLogMonitor(self.logger)
                try:
                    keyboard = get_keyboard()
                    if keyboard:
                        keyboard.on_press_key("q", lambda _: self.check_stop_combination(), suppress=False)
                except Exception as e:
                    self.logger.log("error", f"Klavye kill switch kurulum hatası: {e}")
                self.initialized = True
            except Exception as e:
                self.logger.log("error", f"AutoImporter başlatma hatası: {e}")

    def cleanup_on_shutdown(self):
        """Shutdown sırasında temizlik yapar ve özet sunar."""
        try:
            self.logger.log("info", "Temizlik başlatılıyor...")
            deleted_files = []
            for dir_path in [CACHE_DIR, LOG_DIR]:
                if dir_path.exists():
                    for file in dir_path.glob("*"):
                        if file.is_file() and file.stat().st_mtime < time.time() - 30 * 24 * 3600:
                            file.unlink()
                            deleted_files.append(str(file))
                            self.logger.log("info", f"Silindi: {file}")
            self.monitor.stop()
            self.print_cleanup_summary(deleted_files)
        except Exception as e:
            self.logger.log("error", f"Temizlik hatası: {e}")

    def print_cleanup_summary(self, deleted_files: List[str]):
        """Temizlik özetini kullanıcıya sunar."""
        try:
            self.logger.log("info", f"Temizlik Özeti: {len(deleted_files)} dosya silindi.")
            if deleted_files:
                print(f"[PDS-X] Temizlik Özeti: {len(deleted_files)} dosya silindi.")
                for file in deleted_files[:5]:
                    print(f"[PDS-X] Silindi: {file}")
                if len(deleted_files) > 5:
                    print("[PDS-X] ve daha fazlası...")
        except Exception as e:
            self.logger.log("error", f"Temizlik özeti hatası: {e}")

    def check_shutdown_signal(self) -> bool:
        return shutdown_manager.shutdown_requested

    def check_stop_combination(self):
        """Ctrl+Shift+Q ile programı durdurur."""
        try:
            keyboard = get_keyboard()
            if keyboard and keyboard.is_pressed("ctrl+shift+q"):
                self.logger.log("info", "Ctrl+Shift+Q ile durdurma sinyali alındı.")
                shutdown_manager.shutdown_requested = True
                self.cleanup_on_shutdown()
                sys.exit(0)
        except Exception as e:
            self.logger.log("error", f"Durdurma kombinasyonu kontrol hatası: {e}")

    def save_last_args(self, args: List[str]):
        """Komut argümanlarını kaydeder."""
        try:
            with self.lock:
                with open(LAST_ARGS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"args": args, "timestamp": datetime.now().isoformat()}, f)
                self.logger.log("info", "Son argümanlar kaydedildi.")
        except Exception as e:
            self.logger.log("error", f"Argüman kaydetme hatası: {e}")

    def load_last_args(self) -> List[str]:
        """Komut argümanlarını yükler."""
        try:
            with self.lock:
                if LAST_ARGS_FILE.exists():
                    with open(LAST_ARGS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, dict) or "args" not in data:
                            self.logger.log("warning", "Geçersiz argüman dosyası formatı.")
                            return sys.argv
                        return data["args"]
                return sys.argv
        except Exception as e:
            self.logger.log("error", f"Argüman yükleme hatası: {e}")
            return sys.argv

    def enable_offline_mode(self):
        """Çevrimdışı modu etkinleştirir."""
        try:
            self.offline_cache = self.cache_manager.load_package_metadata()
            self.secure_mode = True
            self.logger.log("info", "Çevrimdışı mod etkinleştirildi.")
        except Exception as e:
            self.logger.log("error", f"Çevrimdışı mod etkinleştirme hatası: {e}")

    def check_build_tools(self) -> bool:
        """Derleme araçlarını kontrol eder."""
        try:
            for tool in ["gcc", "make"]:
                if not shutil.which(tool):
                    self.logger.log("warning", f"{tool} bulunamadı, bazı paketler derlenemeyebilir.")
                    return False
            self.logger.log("info", "Derleme araçları mevcut.")
            return True
        except Exception as e:
            self.logger.log("error", f"Derleme araçları kontrol hatası: {e}")
            return False

    def install_package(self, package: str, max_retries: int = 3) -> bool:
        """Paketi kurar."""
        try:
            start_time = time.time()
            if self.offline_cache and package in self.offline_cache:
                success = self.cache_manager.install_from_cache(package)
            else:
                process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package], check=True, capture_output=True, text=True, timeout=300
                )
                shutdown_manager.register_process(process)
                success = True
                self.retry_count[package] = self.retry_count.get(package, 0)
            if success:
                duration = time.time() - start_time
                self.summary_generator.add_module_status(package, "Başarılı", duration)
                self.installation_history[package] = {"status": "Başarılı", "timestamp": datetime.now().isoformat()}
                self.logger.log("info", f"{package} başarıyla yüklendi.")
                return True
            else:
                raise subprocess.CalledProcessError(1, "pip install")
        except subprocess.CalledProcessError as e:
            self.retry_count[package] = self.retry_count.get(package, 0) + 1
            if self.retry_count[package] <= max_retries:
                self.logger.log("warning", f"{package} yükleme hatası, yeniden deneniyor ({self.retry_count[package]}/{max_retries}): {e.output}")
                return self.install_package(package, max_retries)
            duration = time.time() - start_time
            self.summary_generator.add_module_status(package, "Başarısız", duration)
            self.installation_history[package] = {"status": "Başarısız", "timestamp": datetime.now().isoformat()}
            self.logger.log("error", f"{package} yüklenemedi: {e.output}")
            return False

    async def async_install_package(self, package: str) -> bool:
        """Paketi asenkron kurar."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.install_package(package))
            return result
        except Exception as e:
            self.logger.log("error", f"Asenkron kurulum hatası: {e}")
            return False

    def load_module(self, module_name: str) -> Any:
        """Python modülünü yükler."""
        try:
            if module_name in self.loaded_modules:
                return self.loaded_modules[module_name]
            module = importlib.import_module(module_name)
            self.loaded_modules[module_name] = module
            self.logger.log("info", f"{module_name} modülü yüklendi.")
            return module
        except Exception as e:
            self.logger.log("error", f"{module_name} modül yükleme hatası: {e}")
            return None

    def run(self, args: List[str]):
        """Paketleri kurar ve özet üretir."""
        try:
            parser = argparse.ArgumentParser(description="PDS-X Auto Importer")
            parser.add_argument("--silent", action="store_true", help="Sessiz modda çalışır")
            parser.add_argument("--replay", action="store_true", help="Son argümanları tekrarlar")
            parsed_args = parser.parse_args(args[1:])
            self.logger = AdvancedLogger(silent_mode=parsed_args.silent)
            self.save_last_args(args)
            if parsed_args.replay:
                args = self.load_last_args()
                self.logger.log("info", f"Replay ile argümanlar yüklendi: {args}")
            if not self.env_manager.setup_environment():
                self.logger.log("error", "Ortam hazırlama başarısız.")
                return
            self.env_manager.restart_in_venv(args)
            tasks = [{"package": pkg, "version": pkg.split("==")[1] if "==" in pkg else "latest"} for pkg, _ in REQUIRED_PACKAGES]
            asyncio.run(self.downloader.download_and_install(tasks))
            self.summary_generator.print_summary()
        except Exception as e:
            self.logger.log("error", f"Çalıştırma hatası: {e}")

if __name__ == "__main__":
    importer = AutoImporter()
    importer.run(sys.argv)