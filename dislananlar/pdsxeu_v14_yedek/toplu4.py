# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9
# Date: June 21, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot)

import sys
import os
# import re # KULLANILMIYOR: Bu import artık kodun hiçbir yerinde kullanılmıyor.
import json
import shutil
import logging
import hashlib
import subprocess
import time
import importlib.util
import threading
# import numpy as np # Lazy loading için kaldırıldı
# Lazy-loaded modüller için importları kaldır
# import numpy as np 
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import asyncio
import aiohttp
# import psutil # ResourceMonitor içinde lazy-loading ile yüklenecek
# Lazy-loaded modüller için importları kaldır
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
# import keyboard

from concurrent.futures import ThreadPoolExecutor

# Lazy imports - sadece gerektiğinde yüklenecek
IsolationForest = None
StandardScaler = None
MLPClassifier = None
DecisionTreeClassifier = None
keyboard = None
numpy = None # NumPy için lazy loading
psutil = None # psutil için lazy loading


def get_psutil():
    """psutil'i lazy loading ile yükle"""
    global psutil
    if psutil is None:
        try:
            import psutil as ps
            psutil = ps
        except ImportError:
            print("[PDS-X] psutil yüklenemedi, kurulum gerekli.")
            return None
    return psutil


def get_numpy():
    """NumPy'ı lazy loading ile yükle"""
    global numpy
    if numpy is None:
        try:
            import numpy as np
            numpy = np
        except ImportError:
            # Kurulumu tetiklemek için bir mekanizma eklenebilir
            # Şimdilik sadece logluyoruz.
            # TODO: Logger burada mevcut değil, global bir log fonksiyonu düşünülmeli
            print("[PDS-X] NumPy yüklenemedi, kurulum gerekli.")
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
            print("[PDS-X] Scikit-learn yüklenemedi, kurulum gerekli.")
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

# KULLANILMIYOR: Bu yapılar daha dinamik bir bağımlılık çözümleme mekanizması için
# tasarlanmıştı ancak mevcut implementasyonda aktif olarak kullanılmıyor.
# Gelecekteki geliştirmeler için kaldırılıyor.
# CORE_DEPENDENCIES = {
#     "base": [pkg[0] for pkg in REQUIRED_PACKAGES],
#     "optional": [],
# }

# MODULE_SPECIFIC_DEPS = {
#     "core2-5.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
#     "core2-6.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
#     "libx_ml.py": ["torch==2.2.2", "transformers==4.52.4", "scikit-learn==1.3.2"],
#     "libx_nlp.py": ["nltk==3.9.1", "spacy==3.5.3", "gensim==4.3.3"],
#     "database_sql_isam.py": ["psycopg2-binary==2.9.9", "sqlite3"],
#     "graph.py": ["networkx==3.2.1", "graphviz==0.20.1"],
# }

# KULLANILMIYOR: Bu fonksiyon EnvManager sınıfı içindeki metodlarla yedeklendi.
# def find_python310():
#        logger = AdvancedLogger()
#        env_manager = EnvManager(logger=logger)
#        return env_manager.find_python310()

# KULLANILMIYOR: Bu fonksiyonun işlevselliği AutoImporter sınıfına taşındı.
# async def install_missing_packages():
#        importer = AutoImporter()
#        for pkg, _ in REQUIRED_PACKAGES:
#            await importer.async_install_package(pkg)
#        importer.summary_generator.print_summary()

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

# KULLANILMIYOR: Bu yapılar daha dinamik bir bağımlılık çözümleme mekanizması için
# tasarlanmıştı ancak mevcut implementasyonda aktif olarak kullanılmıyor.
# Gelecekteki geliştirmeler için kaldırılıyor.
# CORE_DEPENDENCIES = {
#     "base": [pkg[0] for pkg in REQUIRED_PACKAGES],
#     "optional": [],
# }

# MODULE_SPECIFIC_DEPS = {
#     "core2-5.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
#     "core2-6.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
#     "libx_ml.py": ["torch==2.2.2", "transformers==4.52.4", "scikit-learn==1.3.2"],
#     "libx_nlp.py": ["nltk==3.9.1", "spacy==3.5.3", "gensim==4.3.3"],
#     "database_sql_isam.py": ["psycopg2-binary==2.9.9", "sqlite3"],
#     "graph.py": ["networkx==3.2.1", "graphviz==0.20.1"],
# }

# KULLANILMIYOR: Bu fonksiyon EnvManager sınıfı içindeki metodlarla yedeklendi.
# def find_python310():
#        logger = AdvancedLogger()
#        env_manager = EnvManager(logger=logger)
#        return env_manager.find_python310()

# KULLANILMIYOR: Bu fonksiyonun işlevselliği AutoImporter sınıfına taşındı.
# async def install_missing_packages():
#        importer = AutoImporter()
#        for pkg, _ in REQUIRED_PACKAGES:
#            await importer.async_install_package(pkg)
#        importer.summary_generator.print_summary()

# Gelişmiş Loglama Sistemi
class AdvancedLogger:
    def __init__(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            # Düz metin terminal logları için
            self.plain_terminal_log_file = PLAIN_TERMINAL_LOG
            if self.plain_terminal_log_file.exists():
                backup_file = self.plain_terminal_log_file.with_suffix(".bak")
                if backup_file.exists():
                    backup_file.unlink()
                shutil.move(self.plain_terminal_log_file, backup_file)
            self.terminal_handler = logging.FileHandler(self.plain_terminal_log_file, encoding="utf-8")
            self.terminal_handler.setLevel(logging.INFO)
            self.terminal_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            # JSONL logları için
            self.handlers = {
                "terminal": logging.FileHandler(TERMINAL_LOG, encoding="utf-8"),
                "error": logging.FileHandler(ERROR_LOG, encoding="utf-8"),
                "warning": logging.FileHandler(WARNING_LOG, encoding="utf-8"),
                "info": logging.FileHandler(INFO_LOG, encoding="utf-8"),
            }
            jsonl_formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
            for level, handler in self.handlers.items():
                handler.setLevel(getattr(logging, level.upper()))
                handler.setFormatter(jsonl_formatter)
            self.logger = logging.getLogger("autoimporter")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(self.terminal_handler)
            for handler in self.handlers.values():
                self.logger.addHandler(handler)
            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
            sys.stdout = self.stdout_handler
            sys.stderr = Tee(sys.__stderr__, self.handlers["error"])
            self.es = None
            if Elasticsearch:
                try:
                    self.es = Elasticsearch(["http://localhost:9200"])
                    self.logger.info("Elasticsearch sunucusuna bağlanıldı.")
                except Exception as e:
                    self.logger.warning(f"Elasticsearch sunucusu erişilemez, yerel loglamaya geçiliyor: {e}")
            self.logged_messages = set()
            self.last_log_time = {}
            self.logger.info("Loglama sistemi başlatıldı.")
        except Exception as e:
            print(f"[PDS-X] Loglama başlatma hatası: {e}")

    def log(self, level: str, message: str):
        try:
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            if message_hash not in self.logged_messages or (datetime.now() - self.last_log_time.get(message_hash, datetime.min)).total_seconds() > 1:
                if level.lower() == "info":
                    print(f"[PDS-X] {message}")
                self.logger.log(getattr(logging, level.upper()), message)
                self.logged_messages.add(message_hash)
                self.last_log_time[message_hash] = datetime.now()
                if self.es:
                    try:
                        self.es.index(index="pdsx_logs", body={
                            "timestamp": datetime.now().isoformat(),
                            "level": level.upper(),
                            "message": message
                        })
                    except Exception as e:
                        self.logger.warning(f"Elasticsearch bağlantısı başarısız, yerel loglamaya devam ediliyor: {e}")
                        self.es = None
            self.rotate_logs()
        except Exception as e:
            print(f"[PDS-X] Loglama hatası: {e}")

    def rotate_logs(self):
        try:
            for log_file in [self.plain_terminal_log_file, TERMINAL_LOG, INFO_LOG, WARNING_LOG, ERROR_LOG]:
                if log_file.exists() and log_file.stat().st_size > MAX_LOG_SIZE:
                    handler = self.terminal_handler if log_file == self.plain_terminal_log_file else self.handlers[log_file.stem]
                    handler.close()
                    backup_path = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                    shutil.move(log_file, backup_path)
                    new_handler = logging.FileHandler(log_file, encoding="utf-8")
                    new_handler.setFormatter(handler.formatter)
                    new_handler.setLevel(handler.level)
                    self.logger.addHandler(new_handler)
                    if log_file == self.plain_terminal_log_file:
                        self.terminal_handler = new_handler
                        self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
                        sys.stdout = self.stdout_handler
                    else:
                        self.handlers[log_file.stem] = new_handler
                    self.cleanup_old_backups(log_file)
        except Exception as e:
            self.logger.error(f"Log döndürme hatası: {e}")

    def cleanup_old_backups(self, log_file: Path):
        try:
            backups = sorted(log_file.parent.glob(f"{log_file.name}.*.bak"), key=lambda x: x.stat().st_mtime, reverse=True)
            for old_backup in backups[MAX_BACKUPS:]:
                old_backup.unlink()
                self.logger.info(f"Eski yedek silindi: {old_backup}")
        except Exception as e:
            self.logger.error(f"Yedek silme hatası: {e}")

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

# Kaynak İzleme
class ResourceMonitor:
    def __init__(self, logger: AdvancedLogger, interval: int = 5):
        self.logger = logger
        self.interval = interval
        self.running = True
        self.psutil = get_psutil() # psutil'i lazy load et
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        if self.psutil:
            self.thread.start()
            self.logger.log("info", "Kaynak izleyici başlatıldı.")
        else:
            self.logger.log("warning", "psutil yüklenemediği için Kaynak İzleyici başlatılamadı.")

    def _monitor(self):
        while self.running:
            try:
                cpu_usage = self.psutil.cpu_percent()
                memory_info = self.psutil.virtual_memory()
                if cpu_usage > 90 or memory_info.percent > 90:
                    self.logger.log("warning", f"Yüksek kaynak kullanımı: CPU {cpu_usage}%, Bellek {memory_info.percent}%")
            except Exception as e:
                self.logger.log("error", f"Kaynak izleme hatası: {e}")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.logger.log("info", "Kaynak izleyici durduruldu.")

# Sezgisel Yönetim
class HeuristicManager:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.rules = {
            "retry_on_network_error": True,
            "use_binary_on_compile_fail": True,
        }
        self.logger.log("info", "Sezgisel yönetim başlatıldı.")

    def get_heuristic(self, rule_name: str) -> bool:
        return self.rules.get(rule_name, False)

# Anomali Tespiti
class AnomalyDetector:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.np = get_numpy()
        if self.np:
            _, self.scaler, _, _ = get_sklearn_components()
            self.logger.log("info", "Anomali tespit sistemi başlatıldı.")
        else:
            self.logger.log("warning", "NumPy yüklenemediği için AnomalyDetector başlatılamadı.")


    def fit(self, data: list):
        if not self.np or not self.scaler:
            self.logger.log("warning", "Anomali modeli eğitilemiyor: NumPy veya StandardScaler eksik.")
            return
        try:
            IsolationForest, _, _, _ = get_sklearn_components()
            if not IsolationForest:
                self.logger.log("warning", "IsolationForest yüklenemedi.")
                return
            self.model = IsolationForest(contamination=0.1)
            scaled_data = self.scaler.fit_transform(self.np.array(data).reshape(-1, 1))
            self.model.fit(scaled_data)
            self.is_fitted = True
            self.logger.log("info", "Anomali tespit modeli eğitildi.")
        except Exception as e:
            self.logger.log("error", f"Anomali modeli eğitme hatası: {e}")

    def predict(self, value: float) -> Optional[int]:
        if not self.is_fitted or not self.model:
            self.logger.log("warning", "Anomali tahmini yapılamıyor: Model eğitilmemiş.")
            return None
        try:
            scaled_value = self.scaler.transform(self.np.array([[value]]))
            prediction = self.model.predict(scaled_value)
            return int(prediction[0])
        except Exception as e:
            self.logger.log("error", f"Anomali tahmin hatası: {e}")
            return None
# --- YENİ EKLENEN YARDIMCI SINIFLAR ---
# Bu sınıflar, AutoImporter'ın ana mantığını daha modüler ve yönetilebilir hale getirmek için
# önceki versiyondan ilham alınarak veya yeniden yazılarak eklenmiştir.
# Tanımlamaların AutoImporter'dan önce yapılması, "Undefined name" hatalarını önler.

class KillSwitch:
    """
    Acil durdurma mekanizması. Belirli bir koşulda (örneğin, klavye kısayolu)
    tüm işlemleri güvenli bir şekilde durdurmak için kullanılabilir.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.is_active = False
        self.logger.log("info", "KillSwitch başlatıldı.")

    def activate(self):
        """Kill switch'i aktif eder."""
        self.is_active = True
        self.logger.log("warning", "KILL SWITCH AKTİF! Tüm operasyonlar durduruluyor.")

    def check(self):
        """Kill switch aktif ise bir hata fırlatarak işlemi keser."""
        if self.is_active:
            raise InterruptedError("KillSwitch ile operasyon durduruldu.")

class PipOutputAnalyzer:
    """
    Pip komutunun çıktılarını analiz ederek olası hataları tespit eder ve
    çözüm için ek komut argümanları önerir.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger

    def suggest_fix_args(self, error_output: str) -> List[str]:
        """Pip hata çıktısını analiz eder ve çözüm için argümanlar önerir."""
        if "Microsoft Visual C++ 14.0 or greater is required" in error_output:
            self.logger.log("info", "Derleme araçları eksik. Wheel dosyası kullanımı öneriliyor.")
            return ["--only-binary=:all:"]
        if "failed with error code" in error_output:
            self.logger.log("info", "Derleme hatası. Önceden derlenmiş bir wheel kullanılması deneniyor.")
            return ["--only-binary=:all:"]
        # Gelecekte daha fazla kural eklenebilir.
        return []

class WheelCacheManager:
    """
    İndirilen tekerlek (.whl) dosyalarını yönetir. Bu, aynı paketlerin
    tekrar tekrar indirilmesini önleyerek kurulum sürecini hızlandırır.
    """
    def __init__(self, cache_dir: Path, logger: AdvancedLogger):
        self.cache_dir = cache_dir / "wheels"
        self.logger = logger
        self.metadata_file = self.cache_dir / "metadata.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log("info", f"Wheel önbelleği başlatıldı: {self.cache_dir}")

    def get_wheel_path(self, package_name: str, package_version: str) -> Optional[Path]:
        """Önbellekte uyumlu bir .whl dosyası arar."""
        # Not: Bu basitleştirilmiş bir aramadır. Gerçek bir uygulama Python sürümü,
        # ABI ve platform etiketlerini de kontrol etmelidir.
        try:
            # Paket adlarındaki '-' karakterini '_' ile değiştirmek standartlara daha uygun.
            search_pattern = f"{package_name.replace('-', '_')}-{package_version}-*.whl"
            for f in self.cache_dir.glob(search_pattern):
                self.logger.log("info", f"Önbellekte wheel bulundu: {f}")
                return f
        except Exception as e:
            self.logger.log("error", f"{package_name}=={package_version} için wheel arama hatası: {e}")
        return None

    def cache_wheel(self, wheel_path: Path):
        """Bir .whl dosyasını önbelleğe kopyalar."""
        if not wheel_path.exists():
            self.logger.log("warning", f"Önbelleğe alınacak dosya bulunamadı: {wheel_path}")
            return
        try:
            shutil.copy(wheel_path, self.cache_dir)
            self.logger.log("info", f"Wheel önbelleğe alındı: {wheel_path.name}")
        except Exception as e:
            self.logger.log("error", f"{wheel_path.name} önbelleğe alınamadı: {e}")

class EnvManager:
    """
    İzole Python sanal ortamını (venv) yönetir. Ortamın oluşturulması,
    kontrol edilmesi ve içindeki Python/pip yollarının bulunmasından sorumludur.
    """
    def __init__(self, logger: AdvancedLogger, venv_dir: Path = VENV_DIR):
        self.logger = logger
        self.venv_dir = venv_dir
        self.python_path = self._find_executable("python")
        self.pip_path = self._find_executable("pip")

    def _find_executable(self, name: str) -> Optional[Path]:
        """Sanal ortam içinde belirtilen çalıştırılabilir dosyayı bulur."""
        try:
            if self.venv_dir.exists():
                exe_name = f"{name}.exe" if os.name == "nt" else name
                path = self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / exe_name
                if path.exists():
                    return path
        except Exception as e:
            self.logger.log("error", f"{name} bulunurken hata oluştu: {e}")
        return None

    def setup_environment(self):
        """Sanal ortam mevcut değilse oluşturur."""
        if self.venv_dir.exists():
            self.logger.log("info", f"Sanal ortam zaten mevcut: {self.venv_dir}")
            return

        self.logger.log("info", f"Sanal ortam oluşturuluyor: {self.venv_dir}...")
        try:
            system_python = sys.executable
            if not system_python:
                self.logger.log("error", "Sistem Python yorumlayıcısı bulunamadı.")
                return

            subprocess.run([system_python, "-m", "venv", str(self.venv_dir)], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            self.logger.log("info", "Sanal ortam başarıyla oluşturuldu.")
            # Yolları yeniden al
            self.python_path = self._find_executable("python")
            self.pip_path = self._find_executable("pip")
        except subprocess.CalledProcessError as e:
            self.logger.log("error", f"Sanal ortam oluşturulamadı: {e.stderr}")
        except Exception as e:
            self.logger.log("error", f"Sanal ortam oluşturulurken beklenmedik hata: {e}")

    def report_error(self):
        """Hata raporlama için placeholder fonksiyon."""
        self.logger.log("error", "EnvManager'da bir hata raporlandı.")

class ModuleSummaryGenerator:
    """
    Modül yükleme işlemlerinin bir özetini oluşturur, renkli bir tablo olarak
    konsola basar ve detaylı bir JSON dosyası olarak dışa aktarır.
    """
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.summary_data = []
        self.start_time = time.time()

    def add_module_status(self, module_name: str, status: str, duration: float, source: str = "pip", version: str = "N/A", error: Optional[str] = None, reason: Optional[str] = None):
        """Bir modülün kurulum durumunu özete ekler."""
        entry = {
            "module": module_name,
            "status": status,
            "duration_seconds": round(duration, 2),
            "source": source,
            "version": version,
            "error": error,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self.summary_data.append(entry)

    def print_summary(self):
        """Renkli özet tablosunu konsola yazdırır."""
        if not self.summary_data:
            print(f"{Fore.YELLOW}Bu oturumda hiçbir modül işlenmedi.{Style.RESET_ALL}")
            return

        total_duration = time.time() - self.start_time
        success_count = sum(1 for s in self.summary_data if s['status'] == 'Başarılı')
        failed_count = sum(1 for s in self.summary_data if s['status'] == 'Başarısız')
        # KULLANILMIYOR: skipped_count değişkeni özet tablosunda gösterilmiyor ve başka bir yerde kullanılmıyor.
        # skipped_count = sum(1 for s in self.summary_data if s['status'] == 'Atlandı')

        print(f"\n{Fore.CYAN}{'='*25} PDS-X Kurulum Özeti {'='*25}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Zaman Damgası: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Toplam Süre: {total_duration:.2f} saniye{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Başarılı: {success_count}{Style.RESET_ALL} | {Fore.RED}Başarısız: {failed_count}{Style.RESET_ALL} | {Fore.YELLOW}Atlandı: {skipped_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*85}{Style.RESET_ALL}")

        print(f"{Fore.WHITE + Style.BRIGHT}{'Modül':<25}{'Sürüm':<15}{'Durum':<20}{'Kaynak':<10}{'Süre (s)':<15}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*85}{Style.RESET_ALL}")

        for s in self.summary_data:
            color = Fore.GREEN if s['status'] == 'Başarılı' else Fore.RED if s['status'] == 'Başarısız' else Fore.YELLOW
            status_text = s['status']
            if s['reason']:
                status_text = f"{s['status']} ({s['reason']})"
            
            print(f"{color}{s['module']:<25}{s['version']:<15}{status_text:<20}{s['source']:<10}{s['duration_seconds']:<15.2f}{Style.RESET_ALL}")
            if s['error']:
                # Hata mesajını daha okunaklı yap
                error_msg = str(s['error']).replace('\n', ' ').strip()
                print(f"{Fore.RED}  └─ Hata: {error_msg[:100]}...{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'='*85}{Style.RESET_ALL}\n")
        self.export_summary_to_json()

    def export_summary_to_json(self, filename: str = "pdsx_install_summary.json"):
        """Özet verilerini bir JSON dosyasına aktarır."""
        try:
            LOG_DIR.mkdir(exist_ok=True)
            summary_output = {
                "summary_metadata": {
                    "creation_timestamp": datetime.now().isoformat(),
                    "total_duration_seconds": round(time.time() - self.start_time, 2),
                    "total_modules_processed": len(self.summary_data),
                    "counts": {
                        "successful": sum(1 for s in self.summary_data if s['status'] == 'Başarılı'),
                        "failed": sum(1 for s in self.summary_data if s['status'] == 'Başarısız'),
                        "skipped": sum(1 for s in self.summary_data if s['status'] == 'Atlandı'),
                    }
                },
                "modules": self.summary_data
            }
            filepath = LOG_DIR / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary_output, f, indent=4, ensure_ascii=False)
            self.logger.log("info", f"Kurulum özeti dışa aktarıldı: {filepath}")
        except Exception as e:
            self.logger.log("error", f"JSON özeti dışa aktarılamadı: {e}")

# --- BİTİŞ ---

# Ana AutoImporter Sınıfı
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
                self.env_manager = EnvManager(logger=self.logger)
                self.dependency_registry = DependencyRegistry(self.logger)
                self.resource_monitor = ResourceMonitor(self.logger)
                self.heuristic_manager = HeuristicManager(self.logger)
                self.anomaly_detector = AnomalyDetector(self.logger)
                self.pip_output_analyzer = PipOutputAnalyzer(self.logger)
                self.wheel_cache = WheelCacheManager(CACHE_DIR, self.logger) # WheelCacheManager eklendi
                self.kill_switch = KillSwitch(self.logger)
                self.summary_generator = ModuleSummaryGenerator(self.logger)
                self.dependency_graph = {}

                # --- YENİ EKLENEN NİTELİKLER ---
                self.lock = threading.Lock()
                self.running = True
                self.installation_history = {}
                self.retry_count = {}
                self.imported_files = set()
                self.module_cache = {}
                self.aliases = {}
                self.loaded_modules = {}
                self.secure_mode = False
                # --- BİTİŞ ---

                self.initialized = True
                self.logger.log("info", "AutoImporter başlatıldı.")
            except Exception as e:
                print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
                sys.exit(1)

    def check_package_installed(self, package_name: str) -> bool:
        """Bir paketin mevcut sanal ortamda kurulu olup olmadığını kontrol eder."""
        if not self.env_manager.pip_path:
            self.logger.log("warning", "Pip yolu bulunamadığı için paket kontrolü yapılamıyor.")
            return False
        try:
            # `pip show` komutu, paket kuruluysa 0, değilse 1 döndürür.
            cmd = [str(self.env_manager.pip_path), "show", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
            is_installed = result.returncode == 0
            if is_installed:
                self.logger.log("info", f"Paket '{package_name}' zaten kurulu.")
            return is_installed
        except FileNotFoundError:
            self.logger.log("error", "pip komutu bulunamadı. Sanal ortam düzgün kurulmamış olabilir.")
            return False
        except Exception as e:
            self.logger.log("error", f"'{package_name}' paketi kontrol edilirken beklenmedik bir hata oluştu: {e}")
            return False

    def auto_install_package(self, package: str) -> bool:
        """Bir paketi, yalnızca kurulu değilse otomatik olarak kurar."""
        self.logger.log("info", f"Otomatik kurulum süreci başlatıldı: '{package}'")
        try:
            if self.check_package_installed(package.split('==')[0]):
                self.summary_generator.add_module_status(package, "Atlandı", 0, reason="Zaten yüklü")
                return True
            
            self.logger.log("info", f"'{package}' paketi kuruluyor...")
            self.install_package(package) # install_package zaten detaylı loglama ve özetleme yapıyor.
            # Kurulumun başarılı olup olmadığını tekrar kontrol et
            if self.check_package_installed(package.split('==')[0]):
                self.logger.log("info", f"'{package}' başarıyla kuruldu ve doğrulandı.")
                return True
            else:
                self.logger.log("error", f"'{package}' kurulumu yapıldı ancak doğrulanamadı.")
                return False
        except Exception as e:
            self.logger.log("error", f"'{package}' için otomatik kurulum sürecinde hata: {e}")
            return False

    def install_package(self, package: str):
        try:
            self.logger.log("info", f"{package} kurulumu başlatılıyor.")
            start_time = time.time()

            # --- WheelCacheManager entegrasyonu ---
            pkg_name, pkg_version = (package.split("==") + [None])[:2]
            wheel_path = None
            if pkg_version:
                wheel_path = self.wheel_cache.get_wheel_path(pkg_name, pkg_version)
            if wheel_path and wheel_path.exists():
                self.logger.log("info", f"Önbellekten .whl kuruluyor: {wheel_path}")
                pip_cmd = str(self.env_manager.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.env_manager.venv_dir / "bin" / "pip")
                cmd = [pip_cmd, "install", str(wheel_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                duration = time.time() - start_time
                if result.returncode == 0:
                    self.dependency_registry.register_package(
                        package,
                        pkg_version or "latest",
                        "Başarılı"
                    )
                    self.summary_generator.add_module_status(pkg_name, "Başarılı", duration, source="cache", version=pkg_version or "N/A")
                    self.logger.log("info", f"{package} önbellekten başarıyla kuruldu.")
                    return
                else:
                    self.logger.log("warning", f"Önbellekten kurulum başarısız, pip ile devam ediliyor: {result.stderr}")
            # --- Bitiş ---

            if self.dependency_registry.check_package(package):
                self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
                self.summary_generator.add_module_status(pkg_name, "Atlandı", 0, reason="Zaten yüklü", version=pkg_version or "N/A")
                return

            attempts = 0
            max_attempts = 3

            while attempts < max_attempts:
                try:
                    if not self.env_manager.venv_dir.exists():
                        self.env_manager.setup_environment()

                    pip_cmd = str(self.env_manager.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.env_manager.venv_dir / "bin" / "pip")
                    cmd = [pip_cmd, "install", package]

                    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

                    if result.returncode == 0:
                        self.dependency_registry.register_package(
                            package,
                            package.split("==")[1] if "==" in package else "latest",
                            "Başarılı"
                        )
                        self.logger.log("info", f"{package} başarıyla kuruldu.")
                        break
                    else:
                        self.logger.log("warning", f"İlk kurulum denemesi başarısız: {package}. Çıktı analiz ediliyor...")
                        error_output = result.stderr
                        fix_args = self.pip_output_analyzer.suggest_fix_args(error_output) # DÜZELTME: pip_analyzer -> pip_output_analyzer

                        if fix_args:
                            self.logger.log("info", f"{package} için düzeltme deneniyor: {fix_args}")
                            fix_cmd = [pip_cmd, "install"] + fix_args + [package]
                            fix_result = subprocess.run(fix_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

                            if fix_result.returncode == 0:
                                self.dependency_registry.register_package(
                                    package,
                                    package.split("==")[1] if "==" in package else "latest",
                                    "Başarılı"
                                )
                                self.logger.log("info", f"{package} düzeltme sonrası başarıyla kuruldu.")
                                break
                            else:
                                raise Exception(f"Düzeltme denemesi başarısız: {package}. Hata: {fix_result.stderr}")
                        else:
                            raise Exception(f"{package} kurulum hatası: {error_output}")

                except Exception as e:
                    attempts += 1
                    self.logger.log("warning", f"{package} kurulum denemesi {attempts}/{max_attempts} başarısız: {e}")

                    if attempts == max_attempts:
                        self.logger.log("error", f"{package} kurulumu tüm denemelere rağmen başarısız oldu.")
                        self.env_manager.report_error()
                        self.dependency_registry.register_package(
                            package,
                            package.split("==")[1] if "==" in package else "latest",
                            "Başarısız"
                        )
                        duration = time.time() - start_time
                        self.summary_generator.add_module_status(pkg_name, "Başarısız", duration, version=pkg_version or "N/A", error=str(e))
                        return

                    time.sleep(2 ** attempts)

            duration = time.time() - start_time
            self.summary_generator.add_module_status(pkg_name, "Başarılı", duration, version=pkg_version or "N/A")
            self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")

        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.log("error", f"{package} kurulumunda beklenmedik hata: {e}")
            self.env_manager.report_error()
            self.dependency_registry.register_package(
                package,
                package.split("==")[1] if "==" in package else "latest",
                "Başarısız"
            )
            self.summary_generator.add_module_status(pkg_name, "Başarısız", duration, version=pkg_version or "N/A", error=str(e))
