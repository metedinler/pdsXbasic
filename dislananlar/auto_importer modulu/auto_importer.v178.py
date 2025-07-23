# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.8
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
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import hashlib
import asyncio
import aiohttp
import psutil
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

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
WHEELS_DIR = CACHE_DIR / "wheels"
LOG_DIR = Path("logs")
TERMINAL_LOG = LOG_DIR / "pdsX_terminal.log"
INFO_LOG = LOG_DIR / "pdsx_info.jsonl"
WARNING_LOG = LOG_DIR / "pdsx_warnings.jsonl"
ERROR_LOG = LOG_DIR / "pdsx_errors.jsonl"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BACKUPS = 5

# Bağımlılık Listesi (minimuma indirilmiş)
REQUIRED_PACKAGES = [
    ("numpy==1.26.4", "numpy"),
    ("requests==2.32.4", "requests"),
    ("scikit-learn==1.3.2", "sklearn"),
]

CORE_DEPENDENCIES = {
    "base": [pkg[0] for pkg in REQUIRED_PACKAGES],
    "optional": [],
}

MODULE_SPECIFIC_DEPS = {
    "core2-5.py": ["numpy==1.26.4"],
    "core2-6.py": ["numpy==1.26.4"],
}

# Gelişmiş Loglama Sistemi
class AdvancedLogger:
    def __init__(self):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            # Terminal logları için yedekleme ve handler
            if TERMINAL_LOG.exists():
                backup_file = TERMINAL_LOG.with_suffix(".bak")
                if backup_file.exists():
                    backup_file.unlink()
                shutil.move(TERMINAL_LOG, backup_file)
            self.terminal_handler = logging.FileHandler(TERMINAL_LOG, encoding="utf-8")
            self.terminal_handler.setLevel(logging.INFO)
            self.terminal_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            # JSONL logları için handler'lar
            self.handlers = {
                "info": logging.FileHandler(INFO_LOG, encoding="utf-8"),
                "warning": logging.FileHandler(WARNING_LOG, encoding="utf-8"),
                "error": logging.FileHandler(ERROR_LOG, encoding="utf-8"),
            }
            jsonl_formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
            for level, handler in self.handlers.items():
                handler.setLevel(getattr(logging, level.upper()))
                handler.setFormatter(jsonl_formatter)
            # Logger ayarları
            self.logger = logging.getLogger("autoimporter")
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(self.terminal_handler)
            for handler in self.handlers.values():
                self.logger.addHandler(handler)
            # Terminal yönlendirme
            self.stdout_handler = Tee(sys.__stdout__, self.terminal_handler)
            sys.stdout = self.stdout_handler
            sys.stderr = Tee(sys.__stderr__, self.handlers["error"])
            # Elasticsearch (opsiyonel)
            self.es = None
            if Elasticsearch:
                try:
                    self.es = Elasticsearch(["http://localhost:9200"])
                    self.logger.info("Elasticsearch sunucusuna bağlanıldı.")
                except Exception as e:
                    self.logger.warning(f"Elasticsearch sunucusu erişilemez, yerel loglamaya geçiliyor: {e}")
            self.logged_messages = set()
            self.logger.info("Loglama sistemi başlatıldı.")
        except Exception as e:
            print(f"[PDS-X] Loglama başlatma hatası: {e}")

    def log(self, level: str, message: str):
        try:
            message_hash = hashlib.sha256(message.encode()).hexdigest()
            if message_hash not in self.logged_messages:
                if level.lower() == "info":
                    print(f"[PDS-X] {message}")
                self.logger.log(logging.getLevelName(level.upper()), message)
                self.logged_messages.add(message_hash)
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
            for log_file in [TERMINAL_LOG, INFO_LOG, WARNING_LOG, ERROR_LOG]:
                if log_file.exists() and log_file.stat().st_size > MAX_LOG_SIZE:
                    handler = self.terminal_handler if log_file == TERMINAL_LOG else self.handlers[log_file.stem]
                    handler.close()
                    backup_path = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                    shutil.move(log_file, backup_path)
                    new_handler = logging.FileHandler(log_file, encoding="utf-8")
                    new_handler.setFormatter(handler.formatter)
                    new_handler.setLevel(handler.level)
                    self.logger.addHandler(new_handler)
                    if log_file == TERMINAL_LOG:
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
    def __init__(self, logger: AdvancedLogger, registry_file: Path = CACHE_DIR / "package_registry.json"):
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
    def __init__(self, cache_dir: Path = WHEELS_DIR, logger: AdvancedLogger = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.metadata_file = CACHE_DIR / "packages.json"

    def install_from_cache(self, package: str) -> bool:
        try:
            package_name, version = self._parse_package(package)
            cache_file = self.cache_dir / f"{package_name}-{version}.whl"
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
            package_name, version = self._parse_package(package)
            cache_file = self.cache_dir / f"{package_name}-{version}.whl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                metadata = self.load_package_metadata()
                metadata[package] = {
                    "version": version,
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

    def _parse_package(self, package: str) -> tuple:
        match = re.match(r"([a-zA-Z0-9\-_.]+)([><=]=?[\d.]+)?", package)
        if match:
            name = match.group(1)
            version = match.group(2).strip(">=<===") if match.group(2) else "latest"
            return name, version
        return package, "latest"

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

    def add_python_to_path(self):
        try:
            if not self.python_path:
                self.python_path = self.find_python310()
            if self.python_path:
                python_dir = os.path.dirname(self.python_path)
                venv_dir = str(self.venv_dir / ("Scripts" if os.name == "nt" else "bin"))
                setx_cmd = f'setx PATH "%PATH%;{python_dir};{venv_dir}"'
                with open("add_pdsx_path.bat", "w") as f:
                    f.write(f"@echo off\n{setx_cmd}\necho PATH güncellendi.\npause\n")
                self.logger.log("info", "PATH'e eklemek için 'add_pdsx_path.bat' dosyasını yönetici olarak çalıştırın!")
        except Exception as e:
            self.logger.log("error", f"PATH güncelleme hatası: {e}")

# Çakışma Yönetim Sistemi
class ConflictManager:
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = defaultdict(list)
        self.decision_tree = DecisionTreeClassifier()
        self.label_encoder = LabelEncoder()

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

    def resolve_conflicts(self, module_name: str, conflicts: Dict) -> Dict:
        try:
            resolutions = {}
            # Loglardan çakışma ve kurulum hatalarını analiz et
            log_stats = self.analyze_logs()
            conflict_details = log_stats.get("warnings", []) + log_stats.get("errors", [])
            # Karar ağacı için özellikler hazırla
            X = []
            y = []
            for dep in conflicts:
                features = [len(conflicts), len(conflict_details), 1 if "numpy" in dep.lower() else 0]
                X.append(features)
                y.append(self._get_resolution_label(dep))
            if X and y:
                self.label_encoder.fit(y)
                y_encoded = self.label_encoder.transform(y)
                self.decision_tree.fit(X, y_encoded)
                predictions = self.decision_tree.predict(X)
                predicted_labels = self.label_encoder.inverse_transform(predictions)
                for dep, pred_label in zip(conflicts.keys(), predicted_labels):
                    cmd = f"pip install {pred_label} --force-reinstall"
                    try:
                        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                        resolutions[dep] = {"command": cmd, "reason": f"Karar ağacı önerisi: {pred_label}", "output": result.stdout}
                        self.logger.log("info", f"Çakışma çözüldü: {dep}, {pred_label}")
                    except subprocess.CalledProcessError as e:
                        self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
            return resolutions
        except Exception as e:
            self.logger.log("error", f"Çakışma çözüm hatası: {e}")
            return {}

    def _get_resolution_label(self, dep: str) -> str:
        if "numpy" in dep.lower():
            return "numpy==1.26.4"
        return dep

    def analyze_logs(self) -> Dict:
        try:
            log_stats = {
                "errors": [], "warnings": [], "info": [], "debug": [],
                "error_count": 0, "warning_count": 0, "info_count": 0, "debug_count": 0
            }
            for log_file in [INFO_LOG, WARNING_LOG, ERROR_LOG]:
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
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
            self.logger.log("info", f"Log analizi: {log_stats['error_count']} hata, {log_stats['warning_count']} uyarı")
            return log_stats
        except Exception as e:
            self.logger.log("error", f"Log analizi hatası: {e}")
            return {"error": str(e)}

# Paralel İndirme Sistemi
class AsyncDownloadManager:
    def __init__(self, max_workers: int = 4, cache_dir: Path = WHEELS_DIR, logger: AdvancedLogger = None):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.download_stats = {}

    async def download_package(self, task: Dict) -> Dict:
        try:
            package, version = task["package"], task["version"]
            package_name, _ = self._parse_package(package)
            cache_file = self.cache_dir / f"{package_name}-{version}.whl"
            if cache_file.exists():
                self.logger.log("info", f"{package} zaten önbellekte mevcut.")
                return {"success": True, "size": cache_file.stat().st_size}
            url = f"https://files.pythonhosted.org/packages/{package_name}-{version}-py3-none-any.whl"
            retry_count = 0
            while retry_count < 3:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.read()
                                cache_path = self.cache_dir / f"{package_name}-{version}.whl"
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

    def _parse_package(self, package: str) -> tuple:
        match = re.match(r"([a-zA-Z0-9\-_.]+)([><=]=?[\d.]+)?", package)
        if match:
            name = match.group(1)
            version = match.group(2).strip(">=<===") if match.group(2) else "latest"
            return name, version
        return package, "latest"

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
                self.downloader = AsyncDownloadManager(logger=self.logger)
                self.summary_generator = ModuleSummaryGenerator(self.logger)
                self.lock = threading.Lock()
                self.loaded_modules = {}
                self.module_cache = {}
                self.imported_files = set()
                self.running = True
                self._register_stop_combination()
                self.env_manager.add_python_to_path()
                self.check_build_tools()
                self.logger.log("info", "AutoImporter başlatıldı.")
                self.initialized = True
            except Exception as e:
                print(f"[PDS-X] AutoImporter başlatma hatası: {e}")
                sys.exit(1)

    def _register_stop_combination(self):
        try:
            import keyboard
            keyboard.on_press_key("q", self.check_stop_combination, suppress=True)
        except ImportError:
            self.logger.log("warning", "keyboard kütüphanesi yüklü değil, manuel durdurma devre dışı.")

    def check_stop_combination(self, event):
        try:
            import keyboard
            if keyboard.is_pressed("left ctrl") and keyboard.is_pressed("left shift") and event.name == "q":
                self.logger.log("info", "Sol Ctrl + Sol Shift + Q ile durduruluyor.")
                self.running = False
                sys.exit(0)
        except Exception as e:
            self.logger.log("error", f"Durdurma kombinasyonu hatası: {e}")

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
            if self.cache_manager.install_from_cache(package):
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                duration = time.time() - start_time
                self.summary_generator.add_module_status(package, "Başarılı", duration)
                self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")
                return
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts and self.running:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                        break
                    if not self.pip_analyzer.analyze_and_fix(result.stderr, package):
                        raise Exception(f"{package} kurulum hatası: {result.stderr}")
                except Exception as e:
                    attempts += 1
                    self.logger.log("warning", f"{package} kurulum denemesi {attempts}/{max_attempts} başarısız: {e}")
                    if attempts == max_attempts:
                        self.logger.log("error", f"{package} kurulumu başarısız.")
                        self.env_manager.report_error()
                        self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarısız")
                        return
                    time.sleep(2 ** attempts)
            conflicts = self.conflict_manager.detect_conflicts(package, [package])
            if conflicts:
                resolutions = self.conflict_manager.resolve_conflicts(package, conflicts)
                self.dependency_registry.register_resolution(package, resolutions)
            duration = time.time() - start_time
            self.summary_generator.add_module_status(package, "Başarılı", duration)
            self.logger.log("info", f"{package} kurulum süresi: {duration:.2f} saniye")
        except Exception as e:
            self.logger.log("error", f"{package} kurulum hatası: {e}")
            self.env_manager.report_error()

    async def async_install_package(self, package: str):
        try:
            self.logger.log("info", f"{package} asenkron kurulumu başlatılıyor.")
            if self.dependency_registry.check_package(package):
                self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
                return
            if self.cache_manager.install_from_cache(package):
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
                return
            cmd = [sys.executable, "-m", "pip", "install", package]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarılı")
            else:
                self.pip_analyzer.analyze_and_fix(stderr.decode(), package)
                self.dependency_registry.register_package(package, package.split("==")[1] if "==" in package else "latest", "Başarısız")
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
                self.logger.log("info", f"{module_name} modülü başarıyla yüklendi: {duration:.2f} saniye")
                return module
            except Exception as e:
                duration = time.time() - start_time
                self.summary_generator.add_module_status(module_name, f"Hata: {str(e)}", duration)
                self.logger.log("error", f"{module_name} yüklenemedi: {e}")
                return None

    def run(self, packages: List[str]):
        try:
            self.logger.log("info", "Kütüphane yükleme başlatıldı.")
            conflicts = self.conflict_manager.detect_conflicts("pdsX", packages)
            if conflicts:
                resolutions = self.conflict_manager.resolve_conflicts("pdsX", conflicts)
                for pkg, res in resolutions.items():
                    self.dependency_registry.register_resolution(pkg, res)
            else:
                self.logger.log("info", "Çakışma yok, mühürleniyor.")
                self.dependency_registry.registry["status"] = "conflict_free"
                self.dependency_registry.registry["timestamp"] = datetime.now().isoformat()
                self.dependency_registry.save_registry()
            for pkg in packages:
                self.install_package(pkg)
        except Exception as e:
            self.logger.log("error", f"Kütüphane yükleme hatası: {e}")

    async def run_in_parallel(self, packages: List[str]):
        try:
            start_time = time.time()
            tasks = [{"package": pkg, "version": pkg.split("==")[1] if "==" in pkg else "latest"} for pkg in packages]
            await self.downloader.download_all(tasks)
            async with asyncio.TaskGroup() as tg:
                install_tasks = [tg.create_task(self.async_install_package(pkg)) for pkg in packages]
            duration = time.time() - start_time
            self.logger.log("info", f"Paralel kurulum tamamlandı: {duration:.2f} saniye")
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

async def install_missing_packages():
    try:
        importer = AutoImporter()
        if importer.dependency_registry.registry.get("status") == "conflict_free":
            print("AutoImporter çakışmasız başladı, interpreter çalıştırılıyor.")
            return
        await importer.run_in_parallel(CORE_DEPENDENCIES["base"])
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
        loop.run_until_complete(install_missing_packages())
        module = importer.load_module("core2-6.py")
        importer.summary_generator.print_summary()
        importer.logger.log("info", "PDS-X işlemi tamamlandı.")
    except Exception as e:
        print(f"[PDS-X] Ana çalışma hatası: {e}")
        sys.exit(1)