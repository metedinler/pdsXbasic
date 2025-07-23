# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9
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
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import hashlib
import asyncio
import aiohttp
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import keyboard
import re
from concurrent.futures import ThreadPoolExecutor

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

# İzole Ortam Yönetimi
class EnvManager:
    def __init__(self, venv_dir: Path = VENV_DIR, logger: AdvancedLogger = None):
        self.venv_dir = venv_dir
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger
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

    def setup_environment(self):
        """İzole ortam kurar ve gerekli kurulumları yapar"""
        try:
            self.logger.log("info", "İzole ortam kurulumu başlatılıyor...")
            
            # Python 3.10 kontrolü
            python_path = self.find_python310()
            if not python_path:
                python_path = self.download_and_install_python310()
                if not python_path:
                    self.logger.log("error", "Python 3.10 kurulumu başarısız.")
                    return False
            
            # Venv oluştur
            if not self.venv_dir.exists():
                self.logger.log("info", "Sanal ortam oluşturuluyor...")
                subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True)
                self.logger.log("info", "Sanal ortam oluşturuldu.")
                
            # Pip güncelle
            pip_cmd = str(self.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.venv_dir / "bin" / "pip")
            self.logger.log("info", "Pip güncelleniyor...")
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
            self.logger.log("info", "Pip güncellendi.")
            
            self.logger.log("info", "İzole ortam kurulumu tamamlandı.")
            return True
            
        except Exception as e:
            self.logger.log("error", f"İzole ortam kurulum hatası: {e}")
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
                self.dependency_registry = DependencyRegistry(self.logger)
                self.env_manager = EnvManager(logger=self.logger)
                self.summary_generator = ModuleSummaryGenerator(self.logger)
                self.lock = threading.Lock()
                self.loaded_modules = {}
                self.module_cache = {}
                self.imported_files = set()
                self.aliases = {}
                self.dependencies = defaultdict(list)
                self.secure_mode = False
                self.offline_cache = None
                self.metadata = {"auto_importer": {"version": "1.7.9", "dependencies": []}}
                self.installation_history = {}
                self.retry_count = {}
                self.running = True
                
                # Keyboard interrupt handling
                try:
                    keyboard.on_press_key("q", self.check_stop_combination, suppress=True)
                except Exception as e:
                    self.logger.log("warning", f"Keyboard kütüphanesi hatası: {e}")
                
                self.env_manager.add_python_to_path()
                self.check_build_tools()
                self.logger.log("info", "AutoImporter başlatıldı.")
                self.initialized = True
            except Exception as e:
                print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
                sys.exit(1)

    def check_stop_combination(self, event):
        try:
            if keyboard.is_pressed("left ctrl") and keyboard.is_pressed("left shift") and event.name == "q":
                self.logger.log("info", "Sol Ctrl + Sol Shift + Q ile durduruluyor.")
                self.running = False
                sys.exit(0)
        except Exception as e:
            self.logger.log("error", f"Durdurma hatası: {e}")

    def check_build_tools(self):
        build_tool = check_build_tools(self.logger)
        if not build_tool:
            self.logger.log("warning", "Derleme araçları bulunamadı. Lütfen Visual Studio Build Tools veya MinGW kurun.")

    def install_package(self, package: str):
        try:
            self.logger.log("info", f"{package} kurulumu başlatılıyor.")
            start_time = time.time()
            
            if self.dependency_registry.check_package(package):
                self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
                return
            
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts and self.running:
                try:
                    # Sanal ortam kurulmuş mu kontrol et
                    if not self.env_manager.venv_dir.exists():
                        self.env_manager.setup_environment()
                    
                    # Pip komutu oluştur
                    pip_cmd = str(self.env_manager.venv_dir / "Scripts" / "pip.exe") if os.name == "nt" else str(self.env_manager.venv_dir / "bin" / "pip")
                    cmd = [pip_cmd, "install", package]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.record_installation_attempt(package, True)
                        self.dependency_registry.register_package(
                            package, 
                            package.split("==")[1] if "==" in package else "latest", 
                            "Başarılı"
                        )
                        self.logger.log("info", f"{package} başarıyla kuruldu.")
                        break
                    else:
                        raise Exception(f"{package} kurulum hatası: {result.stderr}")
                        
                except Exception as e:
                    attempts += 1
                    self.logger.log("warning", f"{package} kurulum denemesi {attempts}/{max_attempts} başarısız: {e}")
                    
                    if attempts == max_attempts:
                        self.logger.log("error", f"{package} kurulumu başarısız.")
                        self.env_manager.report_error()
                        self.record_installation_attempt(package, False)
                        self.dependency_registry.register_package(
                            package, 
                            package.split("==")[1] if "==" in package else "latest", 
                            "Başarısız"
                        )
                        return
                    
                    time.sleep(2 ** attempts)
            
            duration = time.time() - start_time
            self.summary_generator.add_module_status(package, "Başarılı", duration)
            self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")
            
        except Exception as e:
            self.logger.log("error", f"{package} kurulum hatası: {e}")
            self.env_manager.report_error()
            self.record_installation_attempt(package, False)
            self.dependency_registry.register_package(
                package, 
                package.split("==")[1] if "==" in package else "latest", 
                "Başarısız"
            )

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
            
            # İzole ortam kurulumunu kontrol et
            if not self.env_manager.setup_environment():
                self.logger.log("error", "İzole ortam kurulamadı.")
                return
            
            for pkg in packages:
                if self.running:
                    self.install_package(pkg)
                    
        except Exception as e:
            self.logger.log("error", f"Kütüphane yükleme hatası: {e}")

    def _is_allowed_path(self, path: str) -> bool:
        try:
            allowed_dirs = [os.path.abspath("."), os.path.abspath("libs")]
            return any(path.startswith(d) for d in allowed_dirs)
        except Exception as e:
            self.logger.log("error", f"Yol kontrol hatası: {e}")
            return False

    def record_installation_attempt(self, module_name: str, success: bool):
        try:
            if success:
                self.installation_history[module_name] = datetime.now().isoformat()
                self.retry_count[module_name] = 0
                self.logger.log("info", f"{module_name} kurulumu başarılı, tarih kaydedildi.")
            else:
                if module_name not in self.retry_count:
                    self.retry_count[module_name] = 0
                self.retry_count[module_name] += 1
                self.logger.log("warning", f"{module_name} kurulumu başarısız, deneme sayısı: {self.retry_count[module_name]}")
        except Exception as e:
            self.logger.log("error", f"Yükleme kaydı hatası: {e}")

if __name__ == "__main__":
    importer = AutoImporter()
    importer.install_package("numpy==1.26.4")
    importer.summary_generator.print_summary()
    print("[PDS-X] AutoImporter başlatılıyor... Lutfen cayinizi kahvenizi alın ve bekleyin.")
    print("[PDS-X] Tum islemler otomatik olarak yapılıyor, lütfen bekleyin. ")
    print("[PDS-X] Kütüphane yükleme işlemleri başlatıldı.")
