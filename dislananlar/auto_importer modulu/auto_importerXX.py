# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.6.0
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import asyncio
import aiohttp
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import winreg
except ImportError:
    winreg = None

# Sabitler
VENV_DIR = Path(".pdsx_isolated_env")
CACHE_DIR = Path(".pdsx_cache")
LOG_DIR = Path("logs")
TERMINAL_LOG = LOG_DIR / "pdsxu_terminal.jsonl"
ERROR_LOG = LOG_DIR / "pdsxu_errors.jsonl"
WARNING_LOG = LOG_DIR / "pdsxu_warnings.jsonl"
INFO_LOG = LOG_DIR / "pdsxu_info.jsonl"
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
    ("thinc<8.4.0,>=8.3.4", "thinc"), ("typer<1.0.0,>=0.3.0", "typer"), ("wasabi<1.2.0,>=0.9.1", "wasabi"),
    ("weasel<0.5.0,>=0.1.0", "weasel"), ("huggingface-hub<1.0,>=0.30.0", "huggingface_hub"),
    ("regex!=2019.12.17", "regex"), ("safetensors>=0.4.3", "safetensors"), ("tokenizers<0.22,>=0.21", "tokenizers"),
    ("cryptography==42.0.5", "cryptography"), ("cffi==1.16.0", "cffi"), ("pycparser==2.21", "pycparser"),
    ("six==1.16.0", "six"), ("pyasn1==0.5.1", "pyasn1"), ("pyasn1-modules==0.3.0", "pyasn1_modules"),
    ("idna==3.10", "idna"), ("charset_normalizer==3.4.2", "charset_normalizer"), ("urllib3==2.4.0", "urllib3"),
    ("certifi==2025.6.15", "certifi"), ("chardet==5.2.0", "chardet"), ("blis<0.8.0,>=0.7.8", "blis"),
    ("confection<1.0.0,>=0.0.1", "confection"), ("shellingham>=1.3.0", "shellingham"),
    ("smart-open<7.0.0,>=5.2.1", "smart_open"), ("cloudpathlib<1.0.0,>=0.7.0", "cloudpathlib"),
    ("annotated-types>=0.6.0", "annotated_types"), ("pydantic-core==2.33.2", "pydantic_core"),
    ("typing-inspection>=0.4.0", "typing_inspect"), ("pytest>=6.2.0", "pytest"), ("black>=21.5b2", "black"),
    ("pylint>=2.8.0", "pylint"), ("mypy>=0.910", "mypy"), ("coverage>=6.2", "coverage"),
    ("filelock>=3.18.0", "filelock"), ("fsspec>=2023.5.0", "fsspec"), ("typing-extensions>=3.7.4.3", "typing_extensions"),
    ("tqdm>=4.67.1", "tqdm"), ("exceptiongroup>=1", "exceptiongroup"), ("iniconfig>=1", "iniconfig"),
    ("pluggy<2,>=1.5", "pluggy"), ("tomli>=1", "tomli"), ("astroid<=3.4.0.dev0,>=3.3.8", "astroid"),
    ("dill>=0.2", "dill"), ("isort!=5.13,<7,>=4.2.5", "isort"), ("mccabe<0.8,>=0.6", "mccabe"),
    ("platformdirs>=2.2", "platformdirs"), ("tomlkit>=0.10.1", "tomlkit"), ("pycodestyle>=2.12.0", "pycodestyle"),
    ("autopep8>=2.3.2", "autopep8"), ("pathlib-abc==0.1.1", "pathlib_abc"), ("marisa-trie>=1.1.0", "marisa_trie"),
    ("setuptools>=80.9.0", "setuptools"), ("pytz>=2020.1", "pytz"), ("tzdata>=2022.1", "tzdata"),
    ("gensim==4.3.3", "gensim"), ("s3transfer<0.14.0,>=0.13.0", "s3transfer")
]

CORE_DEPENDENCIES = {
    "base": [pkg[0] for pkg in REQUIRED_PACKAGES[:50]],
    "optional": [pkg[0] for pkg in REQUIRED_PACKAGES[50:60]],
}

MODULE_SPECIFIC_DEPS = {
    "core2-5.py": ["tensorflow==2.15.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
    "libx_ml.py": ["torch==2.2.2", "transformers==4.52.4", "scikit-learn==1.3.2"],
    "libx_nlp.py": ["nltk==3.9.1", "spacy==3.5.3", "gensim==4.3.3"],
    "database_sql_isam.py": ["psycopg2-binary==2.9.9", "sqlite3"],
    "graph.py": ["networkx==3.2.1", "graphviz==0.20.1"],
}

# Gelişmiş Loglama Sistemi
class AdvancedLogger:
    def __init__(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            self.handlers = {
                "terminal": logging.FileHandler(TERMINAL_LOG, encoding="utf-8"),
                "error": logging.FileHandler(ERROR_LOG, encoding="utf-8"),
                "warning": logging.FileHandler(WARNING_LOG, encoding="utf-8"),
                "info": logging.FileHandler(INFO_LOG, encoding="utf-8"),
            }
            self.logger = logging.getLogger("autoimporter")
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
            for handler in self.handlers.values():
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.stdout_handler = Tee(sys.__stdout__, self.handlers["terminal"])
            sys.stdout = self.stdout_handler
            sys.stderr = Tee(sys.__stderr__, self.handlers["error"])
            self.logger.info("Loglama sistemi başlatıldı.")
        except Exception as e:
            print(f"[PDS-X] Loglama başlatma hatası: {e}")

    def log(self, level: str, message: str):
        try:
            print(f"[PDS-X] {message}")
            getattr(self.logger, level.lower())(message)
            self.rotate_logs()
        except Exception as e:
            print(f"[PDS-X] Loglama hatası: {e}")

    def rotate_logs(self):
        try:
            for log_file in self.handlers:
                file_path = Path(self.handlers[log_file].baseFilename)
                if file_path.exists() and file_path.stat().st_size > MAX_LOG_SIZE:
                    self.handlers[log_file].close()
                    backup_path = file_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                    shutil.move(file_path, backup_path)
                    self.handlers[log_file] = logging.FileHandler(file_path, encoding="utf-8")
                    self.handlers[log_file].setFormatter(
                        logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
                    )
                    self.logger.addHandler(self.handlers[log_file])
                    self.cleanup_old_backups(file_path)
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
        }

    def analyze_and_fix(self, output: str, package: str) -> bool:
        try:
            for error, handler in self.error_handlers.items():
                if error in output:
                    self.logger.log("warning", f"Hata tespit edildi: {error}. {package} için düzeltme yapılıyor.")
                    try:
                        result = handler(package)
                        self.logger.log("info", f"{package} düzeltildi: {result.stdout}")
                        return True
                    except subprocess.CalledProcessError as e:
                        self.logger.log("error", f"{package} düzeltme başarısız: {e.output}")
                        return False
            self.logger.log("error", f"Bilinmeyen hata: {output}")
            return False
        except Exception as e:
            self.logger.log("error", f"Pip analiz hatası: {e}")
            return False

# Önbellek Yönetim Sistemi
class CacheManager:
    def __init__(self, cache_dir: Path = CACHE_DIR, logger: AdvancedLogger = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.metadata_file = self.cache_dir / "packages.json"

    def install_from_cache(self, package: str) -> bool:
        try:
            cache_file = self.cache_dir / f"{package}.whl"
            if cache_file.exists():
                self.logger.log("info", f"{package} önbellekten yükleniyor.")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(cache_file)], check=True, capture_output=True, text=True
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
            metadata = self.load_package_metadata()
            metadata[package] = {"version": package.split("==")[1] if "==" in package else "latest", "timestamp": datetime.now().isoformat()}
            self.save_package_metadata(metadata)
            return True
        except Exception as e:
            self.logger.log("error", f"{package} indirme hatası: {e}")
            return False

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

# Çakışma Yönetim Sistemi
class ConflictManager:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = defaultdict(list)

    def detect_conflicts(self, module_name: str, deps: List[str]) -> Dict:
        try:
            self.logger.log("info", f"{module_name} için çakışma kontrolü başlatılıyor.")
            conflicts = {}
            for dep in deps:
                result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
                if "no conflicts" not in result.stdout.lower():
                    conflicts[dep] = result.stdout
                    self.logger.log("warning", f"Çakışma tespit edildi: {dep}, {result.stdout}")
            self.conflicts[module_name] = conflicts
            return conflicts
        except Exception as e:
            self.logger.log("error", f"Çakışma kontrol hatası: {e}")
            return {}

    def resolve_conflicts(self, module_name: str, conflicts: Dict) -> Dict:
        try:
            resolutions = {}
            for dep, issue in conflicts.items():
                if "tensorflow" in dep.lower() and "numpy" in issue.lower():
                    cmd = "pip install numpy==1.26.4 --force-reinstall"
                    reason = "TensorFlow ile uyumluluk için numpy sürümü sınırlandırıldı"
                elif "thinc" in dep.lower() and "numpy" in issue.lower():
                    cmd = "pip install numpy==1.26.4 --force-reinstall"
                    reason = "thinc ile uyumluluk için numpy sürümü sınırlandırıldı"
                else:
                    cmd = f"pip install {dep} --force-reinstall"
                    reason = "Genel çakışma çözümü"
                try:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                    resolutions[dep] = {"command": cmd, "reason": reason, "output": result.stdout}
                    self.logger.log("info", f"Çakışma çözüldü: {dep}, {reason}")
                except subprocess.CalledProcessError as e:
                    self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
            self.resolutions[module_name] = resolutions
            return resolutions
        except Exception as e:
            self.logger.log("error", f"Çakışma çözüm hatası: {e}")
            return {}

    def quantum_analysis(self, module_deps: Dict) -> Dict:
        try:
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
                "isolation_score": float(1 - np.std(scores) / np.mean(scores))
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
                self.logger.log("info", f"Log analizi: {log_stats['error_count']} hata, {log_stats['warning_count']} uyarı")
            return log_stats
        except Exception as e:
            self.logger.log("error", f"Log analizi hatası: {e}")
            return {"error": str(e)}

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
            return len(parts) >= 2 and all(part.isdigit() for part in parts)
        except Exception:
            return False

# Paralel İndirme Sistemi
class AsyncDownloadManager:
    def __init__(self, max_workers: int = 4, cache_dir: Path = CACHE_DIR, logger: AdvancedLogger = None):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.download_stats = {}

    async def download_package(self, task: Dict) -> Dict:
        try:
            package, version = task["package"], task["version"]
            url = f"https://files.pythonhosted.org/packages/{package}-{version}.whl"
            retry_count = 0
            while retry_count < 3:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.read()
                                cache_path = self.cache_dir / f"{package}-{version}.whl"
                                cache_path.write_bytes(content)
                                self.download_stats[package] = {"success": True, "size": len(content)}
                                self.logger.log("info", f"{package}-{version} indirildi.")
                                return self.download_stats[package]
                            else:
                                self.logger.log("warning", f"{package} indirme hatası: HTTP {response.status}")
                except Exception as e:
                    self.logger.log("error", f"{package} indirme hatası: {e}")
                retry_count += 1
                await asyncio.sleep(2 ** retry_count)
            self.download_stats[package] = {"success": False, "error": "İndirme başarısız"}
            return self.download_stats[package]
        except Exception as e:
            self.logger.log("error", f"Paket indirme hatası: {e}")
            return {"success": False, "error": str(e)}

    async def download_all(self, tasks: List[Dict]) -> Dict:
        try:
            async with asyncio.TaskGroup() as tg:
                download_tasks = [tg.create_task(self.download_package(task)) for task in tasks]
            return self.download_stats
        except Exception as e:
            self.logger.log("error", f"Paralel indirme hatası: {e}")
            return {}

# Bilimsel Analiz Sistemi
class ScientificUtils:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1)
        self._lock = threading.Lock()

    def quantum_load_simulation(self, metrics: List[float]) -> Dict:
        with self._lock:
            try:
                if not metrics:
                    raise ValueError("Metrik listesi boş olamaz")
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
            except Exception as e:
                self.logger.log("error", f"Kuantum simülasyonu hatası: {e}")
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
        except Exception as e:
            self.logger.log("error", f"Blockchain doğrulama hatası: {e}")
            return {"error": str(e)}

    def _verify_module_integrity(self, module: Dict) -> bool:
        try:
            required_fields = ['name', 'version', 'dependencies']
            if not all(field in module for field in required_fields):
                return False
            version_parts = module['version'].split('.')
            if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
                return False
            if not isinstance(module['dependencies'], list):
                return False
            return True
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
            header = "┌────────────────────┬──────────────┬──────────┐\n│ Modül Adı          │ Durum        │ Süre (sn)│\n├────────────────────┼──────────────┼──────────┤"
            print(header)
            for s in self.summaries:
                row = f"│ {s['module']:<18} │ {s['status']:<12} │ {s['duration']:.2f}      │"
                print(row)
            footer = "└────────────────────┴──────────────┴──────────┘"
            print(footer)
        except Exception as e:
            self.logger.log("error", f"Özet tablo yazdırma hatası: {e}")

# Ana Yükleyici
class AutoImporter:
    def __init__(self):
        try:
            self.logger = AdvancedLogger()
            self.pip_analyzer = PipOutputAnalyzer(self.logger)
            self.cache_manager = CacheManager(logger=self.logger)
            self.env_manager = EnvManager(logger=self.logger)
            self.conflict_manager = ConflictManager(self.logger)
            self.module_analyzer = ModuleAnalyzer(self.logger)
            self.downloader = AsyncDownloadManager(logger=self.logger)
            self.summary_generator = ModuleSummaryGenerator(self.logger)
            self.lock = threading.Lock()
            self.loaded_modules = {}
            self.module_cache = {}
            self.imported_files = set()
            self.aliases = {}
            self.dependencies = defaultdict(list)
            self.secure_mode = False
            self.metadata = {"auto_importer": {"version": "1.6.0", "dependencies": []}}
            self.installation_history = {}
            self.retry_count = {}
            self.env_manager.add_python_to_path()
            self.logger.log("info", "AutoImporter başlatıldı.")
        except Exception as e:
            print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
            sys.exit(1)

    def install_package(self, package: str):
        try:
            self.logger.log("info", f"{package} kurulumu başlatılıyor.")
            start_time = time.time()
            if package not in self.installation_history or self.should_retry_install(package):
                if self.cache_manager.install_from_cache(package):
                    self.logger.log("info", f"{package} başarıyla kuruldu.")
                    self.record_installation_attempt(package, True)
                    return
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package], capture_output=True, text=True
                )
                if result.returncode != 0:
                    if not self.pip_analyzer.analyze_and_fix(result.stderr, package):
                        self.logger.log("error", f"{package} kurulum hatası: {result.stderr}")
                        self.env_manager.report_error()
                        self.record_installation_attempt(package, False)
                        return
                self.logger.log("info", f"{package} başarıyla kuruldu.")
                self.record_installation_attempt(package, True)
            conflicts = self.conflict_manager.detect_conflicts(package, [package])
            if conflicts:
                self.conflict_manager.resolve_conflicts(package, conflicts)
            duration = time.time() - start_time
            self.summary_generator.add_module_status(package, "Başarılı", duration)
            self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")
        except Exception as e:
            self.logger.log("error", f"{package} kurulum hatası: {e}")
            self.env_manager.report_error()
            self.record_installation_attempt(package, False)

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
                duration = time.time() - start_time
                self.summary_generator.add_module_status(module_name, f"Hata: {str(e)}", duration)
                self.logger.log("error", f"{module_name} yüklenemedi: {e}")
                self.env_manager.report_error()
                return None

    def _is_allowed_path(self, path: str) -> bool:
        try:
            allowed_dirs = [os.path.abspath("."), os.path.abspath("libs")]
            return any(path.startswith(d) for d in allowed_dirs)
        except Exception as e:
            self.logger.log("error", f"Yol kontrol hatası: {e}")
            return False

    def is_recently_installed(self, module_name: str, timeout_minutes: int = 30) -> bool:
        try:
            if module_name not in self.installation_history:
                return False
            last_install = self.installation_history[module_name]
            elapsed = datetime.now() - datetime.fromisoformat(last_install)
            return elapsed.total_seconds() < timeout_minutes * 60
        except Exception as e:
            self.logger.log("error", f"Yükleme zaman kontrol hatası: {e}")
            return False

    def should_retry_install(self, module_name: str, max_retries: int = 3) -> bool:
        try:
            if module_name not in self.retry_count:
                self.retry_count[module_name] = 0
                return True
            return self.retry_count[module_name] < max_retries
        except Exception as e:
            self.logger.log("error", f"Yeniden deneme kontrol hatası: {e}")
            return False

    def record_installation_attempt(self, module_name: str, success: bool):
        try:
            if success:
                self.installation_history[module_name] = datetime.now().isoformat()
                self.retry_count[module_name] = 0
            else:
                if module_name not in self.retry_count:
                    self.retry_count[module_name] = 0
                self.retry_count[module_name] += 1
        except Exception as e:
            self.logger.log("error", f"Yükleme kaydı hatası: {e}")

    async def run_in_parallel(self, packages: List[str]):
        try:
            tasks = [{"package": pkg, "version": pkg.split("==")[1] if "==" in pkg else "latest"} for pkg in packages]
            await self.downloader.download_all(tasks)
            for package in packages:
                self.install_package(package)
        except Exception as e:
            self.logger.log("error", f"Paralel kurulum hatası: {e}")

    def validate_environment(self) -> Dict[str, bool]:
        try:
            status = {
                "python_version": False,
                "pip_available": False,
                "venv_available": False
            }
            python_version = sys.version_info
            status["python_version"] = python_version.major == 3 and python_version.minor == 10
            subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, check=True)
            status["pip_available"] = True
            subprocess.run([sys.executable, "-m", "venv", "--help"], capture_output=True, check=True)
            status["venv_available"] = True
            self.logger.log("info", f"Ortam kontrolü: {status}")
            return status
        except Exception as e:
            self.logger.log("error", f"Ortam kontrol hatası: {e}")
            return {"python_version": False, "pip_available": False, "venv_available": False}

def find_python310():
    try:
        return AutoImporter().env_manager.find_python310()
    except Exception as e:
        print(f"[PDS-X] Python 3.10 bulma hatası: {e}")
        return None

def install_missing_packages():
    try:
        importer = AutoImporter()
        for pkg in CORE_DEPENDENCIES["base"]:
            importer.install_package(pkg)
    except Exception as e:
        print(f"[PDS-X] Bağımlılık yükleme hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        importer = AutoImporter()
        status = importer.validate_environment()
        if not all(status.values()):
            importer.logger.log("error", "Sistem gereksinimleri karşılanmıyor.")
            sys.exit(1)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(importer.run_in_parallel(["numpy==1.26.4", "dash==2.15.0", "tensorflow==2.15.0"]))
        module = importer.load_module("core2-6.py")
        importer.module_analyzer.analyze_logs()
        importer.summary_generator.print_summary()
        importer.logger.log("info", "PDS-X işlemi tamamlandı.")
    except Exception as e:
        print(f"[PDS-X] Ana çalışma hatası: {e}")
        sys.exit(1)