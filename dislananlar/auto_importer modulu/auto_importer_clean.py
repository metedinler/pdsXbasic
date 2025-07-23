# auto_importer.py - PDS-X Akıllı Modül Yükleyici
# Version: 1.7.9.5 - Enhanced PDS-X Integration
# Date: July 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, düzenleyen GitHub Copilot)

# PDS-X İhraç Tanımı
__pdsX_exports__ = {
    "AutoImporter": "Otomatik modül yükleme ve bağımlılık yönetimi ana sınıfı",
    "PdsXException": "PDS-X özel istisna sınıfı",
    "DependencyRegistry": "Bağımlılık kayıt defteri sınıfı",
    "EnvManager": "Sanal ortam yönetimi sınıfı",
    "RealTimeMonitor": "Gerçek zamanlı sistem izleme sınıfı",
    "AdvancedLogger": "Gelişmiş log yönetimi sınıfı",
    "SummaryGenerator": "Özet rapor üretimi sınıfı",
    "TerminalAnalyzer": "Terminal çıktı analizi sınıfı",
    "GracefulShutdownManager": "Güvenli kapatma yönetimi sınıfı",
    "get_numpy": "NumPy lazy loading fonksiyonu",
    "get_sklearn_components": "Sklearn bileşenleri lazy loading fonksiyonu", 
    "get_keyboard": "Keyboard modülü lazy loading fonksiyonu",
    "install_missing_packages": "Eksik paketleri kur",
    "find_python310": "Python 3.10 kurulumunu bul",
    "setup_pdsX_environment": "PDS-X çalışma ortamını hazırla",
}

# PDS-X Özel İstisna Sınıfı
class PdsXException(Exception):
    def __init__(self, message: str, code: str = "ERR_UNDEFINED"):
        self.code = code
        super().__init__(f"[Error {code}] {message}")

# Standart Kütüphaneler
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
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict
import hashlib
import asyncio
import psutil
import re

# 108 Gerekli Paket Listesi
REQUIRED_PACKAGES = [
    ("numpy", "1.24.3"), ("pandas", "2.0.3"), ("matplotlib", "3.7.1"), ("seaborn", "0.12.2"),
    ("scikit-learn", "1.3.0"), ("scipy", "1.11.1"), ("tensorflow", "2.13.0"), ("torch", "2.0.1"),
    ("keras", "2.13.1"), ("opencv-python", "4.8.0.74"), ("pillow", "10.0.0"), ("requests", "2.31.0"),
    ("beautifulsoup4", "4.12.2"), ("selenium", "4.11.2"), ("flask", "2.3.2"), ("django", "4.2.3"),
    ("fastapi", "0.101.0"), ("streamlit", "1.25.0"), ("dash", "2.11.1"), ("plotly", "5.15.0"),
    ("bokeh", "3.2.1"), ("altair", "5.0.1"), ("jupyter", "1.0.0"), ("ipython", "8.14.0"),
    ("notebook", "6.5.4"), ("jupyterlab", "4.0.3"), ("spyder", "5.4.3"), ("psutil", "5.9.5"),
    ("pymongo", "4.4.1"), ("sqlalchemy", "2.0.19"), ("redis", "4.6.0"), ("elasticsearch", "8.8.2"),
    ("kafka-python", "2.0.2"), ("celery", "5.3.1"), ("pytest", "7.4.0"), ("pytest-cov", "4.1.0"),
    ("black", "23.7.0"), ("flake8", "6.0.0"), ("mypy", "1.4.1"), ("bandit", "1.7.5"),
    ("click", "8.1.6"), ("typer", "0.9.0"), ("rich", "13.4.2"), ("colorama", "0.4.6"),
    ("tqdm", "4.65.0"), ("alive-progress", "3.1.4"), ("python-dotenv", "1.0.0"), ("pyyaml", "6.0.1"),
    ("toml", "0.10.2"), ("configparser", "5.3.0"), ("argparse", "1.4.0"), ("logging", "0.4.9.6"),
    ("datetime", "5.2"), ("pathlib", "1.0.1"), ("collections", "0.1.1"), ("itertools", "0.1.0"),
    ("functools", "0.5"), ("operator", "0.0.1"), ("math", "0.0.1"), ("statistics", "0.1.0"),
    ("random", "0.1"), ("string", "0.1.0"), ("re", "0.1.0"), ("json", "0.1.0"),
    ("pickle", "0.1.0"), ("csv", "0.1.0"), ("xml", "0.1.0"), ("html", "0.1.0"),
    ("urllib", "0.1.0"), ("http", "0.1.0"), ("socket", "0.1.0"), ("threading", "0.1.0"),
    ("multiprocessing", "0.1.0"), ("asyncio", "0.1.0"), ("concurrent", "0.1.0"), ("queue", "0.1.0"),
    ("time", "0.1.0"), ("calendar", "0.1.0"), ("locale", "0.1.0"), ("gettext", "0.1.0"),
    ("unicodedata", "0.1.0"), ("codecs", "0.1.0"), ("base64", "0.1.0"), ("binascii", "0.1.0"),
    ("hashlib", "0.1.0"), ("hmac", "0.1.0"), ("secrets", "0.1.0"), ("ssl", "0.1.0"),
    ("cryptography", "41.0.3"), ("bcrypt", "4.0.1"), ("passlib", "1.7.4"), ("jwt", "2.8.0"),
    ("oauthlib", "3.2.2"), ("authlib", "1.2.1"), ("ldap3", "2.9.1"), ("paramiko", "3.2.0"),
    ("fabric", "3.1.0"), ("invoke", "2.2.0"), ("pexpect", "4.8.0"), ("ptyprocess", "0.7.0"),
    ("keyboard", "0.13.5"), ("mouse", "0.7.1"), ("pyautogui", "0.9.54"), ("pynput", "1.7.6"),
    ("tkinter", "0.1.0"), ("pygame", "2.5.0"), ("pyglet", "2.0.8"), ("kivy", "2.2.0"),
    ("wxpython", "4.2.1"), ("pyside6", "6.5.1"), ("pyqt6", "6.5.2"), ("flet", "0.9.0")
]

# Lazy imports - sadece gerektiğinde yüklenecek
numpy = None
IsolationForest = None
StandardScaler = None
MLPClassifier = None
DecisionTreeClassifier = None
keyboard = None

class AdvancedLogger:
    """Gelişmiş loglama sınıfı"""
    def __init__(self, name: str = 'AutoImporter'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name.lower()}.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"[PDS-X] Log dosyası oluşturulamadı: {e}")
            
    def log(self, level: str, message: str) -> None:
        """Mesaj logla"""
        if hasattr(self.logger, level.lower()):
            getattr(self.logger, level.lower())(message)
        else:
            self.logger.info(f"[{level.upper()}] {message}")
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        self.logger.critical(message)
    
    @property
    def name(self) -> str:
        return self.logger.name

class RealTimeMonitor:
    """Gerçek zamanlı sistem izleme"""
    def __init__(self):
        self._running = False
        self._monitor_thread = None
        self.resource_usage = {
            'cpu': [],
            'memory': [],
            'start_time': datetime.now()
        }
        
    def start(self):
        """İzlemeyi başlat"""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        
    def stop(self):
        """İzlemeyi durdur"""
        if self._monitor_thread and self._running:
            self._running = False
            self._monitor_thread.join(timeout=1.0)
            if self._monitor_thread.is_alive():
                print("[PDS-X] Monitor thread güvenli şekilde durdurulamadı!")
            
    def _monitor_loop(self):
        """Ana izleme döngüsü"""
        while self._running:
            try:
                self.update()
                time.sleep(1)  # Her saniye güncelle
            except Exception as e:
                print(f"[PDS-X] Monitor hatası: {e}")
                time.sleep(5)  # Hata durumunda 5 saniye bekle
        
    def update(self):
        """Sistem metriklerini güncelle"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            self.resource_usage['cpu'].append(cpu_percent)
            self.resource_usage['memory'].append(memory.percent)
            
            # Son 1 saatlik veriyi tut
            if len(self.resource_usage['cpu']) > 3600:
                self.resource_usage['cpu'] = self.resource_usage['cpu'][-3600:]
                self.resource_usage['memory'] = self.resource_usage['memory'][-3600:]
                
            # Yüksek kullanım uyarısı
            if cpu_percent > 90 or memory.percent > 90:
                print(f"[PDS-X] UYARI: Yüksek kaynak kullanımı! (CPU: {cpu_percent}%, Bellek: {memory.percent}%)")
        except Exception as e:
            print(f"[PDS-X] Metrik güncelleme hatası: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        if not self.resource_usage['cpu']:
            return {
                'cpu_avg': 0,
                'memory_avg': 0,
                'uptime': 0,
                'samples': 0
            }
            
        return {
            'cpu_avg': sum(self.resource_usage['cpu']) / len(self.resource_usage['cpu']),
            'memory_avg': sum(self.resource_usage['memory']) / len(self.resource_usage['memory']),
            'uptime': (datetime.now() - self.resource_usage['start_time']).total_seconds(),
            'samples': len(self.resource_usage['cpu'])
        }

class SummaryGenerator:
    """Özet rapor üretimi"""
    def __init__(self):
        self.events = []
        self.successes = []
        self.skipped = {}
        self.failures = {}
        
    def add_event(self, event):
        self.events.append({
            'timestamp': datetime.now().isoformat(),
            'event': event
        })
        
    def add_success(self, package):
        self.successes.append(package)
        self.add_event(f"Paket kuruldu: {package}")
        
    def add_skipped(self, package, reason):
        self.skipped[package] = reason
        self.add_event(f"Paket atlandı: {package} ({reason})")
        
    def add_failure(self, package, error):
        self.failures[package] = error
        self.add_event(f"Paket kurulum hatası: {package} ({error})")

    def print_summary(self):
        """Özeti ekrana yazdır"""
        print("\n=== PDS-X AutoImporter Özet ===")
        print(f"Başarılı kurulumlar: {len(self.successes)}")
        if self.successes:
            print("- " + "\n- ".join(self.successes))
            
        print(f"\nAtlanan paketler: {len(self.skipped)}")
        if self.skipped:
            for pkg, reason in self.skipped.items():
                print(f"- {pkg}: {reason}")
                
        print(f"\nBaşarısız kurulumlar: {len(self.failures)}")
        if self.failures:
            for pkg, error in self.failures.items():
                print(f"- {pkg}: {error}")

class TerminalAnalyzer:
    """Terminal çıktı analizi"""
    KNOWN_PACKAGES = {
        'numpy': ['numpy'],
        'pandas': ['pandas'],
        'sklearn': ['scikit-learn'],
        'tensorflow': ['tensorflow'],
        'torch': ['torch'],
        'keras': ['keras'],
        'matplotlib': ['matplotlib'],
        'seaborn': ['seaborn'],
        'opencv': ['opencv-python'],
        'nltk': ['nltk'],
        'spacy': ['spacy']
    }

    def __init__(self):
        self.logger = AdvancedLogger("TerminalAnalyzer")
        
    def map_module_to_package(self, module_name: str, reverse: bool = False) -> str:
        """Modül adını paket adına dönüştür"""
        if reverse:
            for pkg_name, modules in self.KNOWN_PACKAGES.items():
                if module_name in modules:
                    return pkg_name
            return module_name
            
        if module_name in self.KNOWN_PACKAGES:
            return self.KNOWN_PACKAGES[module_name][0]
        return module_name

    def parse_pip_output(self, output: str) -> Dict[str, str]:
        """Pip çıktısını ayrıştır"""
        results = {}
        for line in output.splitlines():
            match = re.match(r"([a-zA-Z0-9\-_]+)==([0-9\.]+)", line)
            if match:
                package, version = match.groups()
                results[package] = version
        return results

    def detect_import_error(self, error_msg: str) -> Optional[str]:
        """Import hata mesajından modül adını bul"""
        match = re.search(r"No module named '([^']+)'", error_msg)
        if match:
            return self.map_module_to_package(match.group(1))
        return None
        
    def parse_conflict_output(self, output: str) -> List[str]:
        """Çakışma çıktısını analiz et"""
        conflicts = []
        lines = output.splitlines()
        
        for line in lines:
            # Çakışma kalıplarını ara
            if "conflict" in line.lower() or "incompatible" in line.lower():
                # Paket adını çıkar
                match = re.search(r"([a-zA-Z0-9\-_]+)", line)
                if match:
                    conflicts.append(match.group(1))
                    
            # Pip hata mesajları
            elif "ERROR:" in line and "requires" in line:
                match = re.search(r"([a-zA-Z0-9\-_]+) requires", line)
                if match:
                    conflicts.append(match.group(1))
                    
        return list(set(conflicts))  # Tekrarları kaldır

class EnvManager:
    """PDS-X Sanal ortam yönetimi"""
    def __init__(self):
        self.logger = AdvancedLogger("EnvManager")
        self._pip_path = None
        self._venv_python = None
        self._venv_dir = None
        self.target_python_version = "3.10"
        self.pdsX_venv_name = "pdsX_isolated_env"
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def check_python_version(self) -> bool:
        """Python sürümünü kontrol et"""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.logger.info(f"Mevcut Python sürümü: {current_version}")
        
        if current_version != self.target_python_version:
            self.logger.warning(f"[PDS-X] Python {current_version} tespit edildi. PDS-X için Python {self.target_python_version} gerekli!")
            return False
        return True
        
    def find_python310(self) -> Optional[Path]:
        """Python 3.10 kurulumunu bul"""
        possible_paths = [
            Path(r"C:\Python310\python.exe"),
            Path(r"C:\Program Files\Python310\python.exe"),
            Path(r"C:\Program Files (x86)\Python310\python.exe"),
            Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "python.exe",
        ]
        
        # PATH'de python3.10 ara
        try:
            result = subprocess.run(["python3.10", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0 and "3.10" in result.stdout:
                python310_path = shutil.which("python3.10")
                if python310_path:
                    return Path(python310_path)
        except:
            pass
            
        # Sabit yolları kontrol et
        for path in possible_paths:
            if path.exists():
                try:
                    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False)
                    if result.returncode == 0 and "3.10" in result.stdout:
                        self.logger.info(f"Python 3.10 bulundu: {path}")
                        return path
                except:
                    continue
                    
        self.logger.error("Python 3.10 bulunamadı!")
        return None
        
    def install_python310(self) -> bool:
        """Python 3.10'u otomatik indir ve kur"""
        self.logger.info("Python 3.10 indiriliyor ve kuruluyor...")
        
        try:
            import urllib.request
            import tempfile
            
            # Python 3.10 installer URL
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            
            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp_file:
                self.logger.info("Python 3.10 installer indiriliyor...")
                urllib.request.urlretrieve(installer_url, tmp_file.name)
                
                # Sessiz kurulum
                install_cmd = [
                    tmp_file.name,
                    "/quiet",
                    "InstallAllUsers=0",
                    "PrependPath=1",
                    "Include_test=0"
                ]
                
                self.logger.info("Python 3.10 kuruluyor...")
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                
                # Geçici dosyayı sil
                os.unlink(tmp_file.name)
                
                if result.returncode == 0:
                    self.logger.info("Python 3.10 başarıyla kuruldu!")
                    return True
                else:
                    self.logger.error(f"Python 3.10 kurulum hatası: {result.stderr}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Python 3.10 kurulum hatası: {e}")
            return False
            
    def setup_pdsX_venv(self) -> bool:
        """PDS-X sanal ortamını kur"""
        venv_path = Path.cwd() / self.pdsX_venv_name
        
        if venv_path.exists():
            self.logger.info(f"PDS-X sanal ortamı mevcut: {venv_path}")
            self._venv_dir = venv_path
            return True
            
        # Python 3.10'u bul
        python310 = self.find_python310()
        if not python310:
            if not self.install_python310():
                return False
            python310 = self.find_python310()
            if not python310:
                return False
                
        try:
            self.logger.info(f"PDS-X sanal ortamı oluşturuluyor: {venv_path}")
            result = subprocess.run([
                str(python310), "-m", "venv", str(venv_path)
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.logger.info("PDS-X sanal ortamı başarıyla oluşturuldu!")
                self._venv_dir = venv_path
                return True
            else:
                self.logger.error(f"Sanal ortam oluşturma hatası: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Sanal ortam kurulum hatası: {e}")
            return False

    def activate_pdsX_venv(self) -> bool:
        """PDS-X sanal ortamını aktive et"""
        if not self._venv_dir:
            if not self.setup_pdsX_venv():
                return False
                
        if not self._venv_dir:
            return False
                
        try:
            venv_path = Path(str(self._venv_dir))
            
            # Windows için activate script
            if os.name == "nt":
                activate_script = venv_path / "Scripts" / "activate.bat"
                python_exe = venv_path / "Scripts" / "python.exe"
            else:
                activate_script = venv_path / "bin" / "activate"
                python_exe = venv_path / "bin" / "python"
                
            if python_exe.exists():
                self._venv_python = python_exe
                self.logger.info(f"PDS-X sanal ortamı aktive edildi: {python_exe}")
                return True
            else:
                self.logger.error(f"Sanal ortam Python bulunamadı: {python_exe}")
                return False
                
        except Exception as e:
            self.logger.error(f"Sanal ortam aktivasyon hatası: {e}")
            return False

    def get_pip_path(self) -> Optional[Path]:
        """Pip yolunu bul"""
        if self._pip_path:
            return self._pip_path
            
        try:
            # Önce sanal ortam pip'ini dene
            if self._venv_dir:
                if os.name == "nt":
                    venv_pip = self._venv_dir / "Scripts" / "pip.exe"
                else:
                    venv_pip = self._venv_dir / "bin" / "pip"
                    
                if venv_pip.exists():
                    self._pip_path = venv_pip
                    return self._pip_path
            
            # Global pip'i dene
            pip_path = shutil.which("pip")
            if pip_path:
                self._pip_path = Path(pip_path)
                return self._pip_path
                
            # Python modülü olarak pip'i dene
            python_path = self._venv_python or sys.executable
            result = subprocess.run(
                [str(python_path), "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                self._pip_path = Path(python_path)
                return self._pip_path
                
            self.logger.error("Pip bulunamadı!")
            return None
        except Exception as e:
            self.logger.error(f"Pip yolu alınırken hata: {e}")
            return None

    def get_venv_python_path(self) -> Optional[Path]:
        """Sanal ortam Python yolunu bul"""
        if self._venv_python:
            return self._venv_python
            
        # PDS-X sanal ortamını aktive et
        if self.activate_pdsX_venv():
            return self._venv_python
            
        # Fallback: global Python
        self._venv_python = Path(sys.executable)
        return self._venv_python
            
    def _find_python_in_venv(self, venv_dir: Path) -> Optional[Path]:
        """Sanal ortamda Python yürütülebilir dosyasını bul"""
        try:
            if os.name == "nt":  # Windows
                python_path = venv_dir / "Scripts" / "python.exe"
            else:  # Linux/Mac
                python_path = venv_dir / "bin" / "python"
                
            if python_path.exists():
                return python_path
            return None
        except Exception as e:
            self.logger.error(f"Sanal ortamda Python aranırken hata: {e}")
            return None

    def is_running_in_venv(self) -> bool:
        """Sanal ortamda çalışıp çalışmadığını kontrol et"""
        # PDS-X sanal ortamı kontrolü
        pdsX_venv_path = Path.cwd() / self.pdsX_venv_name
        if pdsX_venv_path.exists():
            return True
            
        # Genel sanal ortam kontrolü
        return "VIRTUAL_ENV" in os.environ or Path(".venv").exists()
        
    def setup_environment(self, **kwargs) -> bool:
        """PDS-X için tam ortam kurulumu"""
        self.logger.info("PDS-X ortam kurulumu başlatılıyor...")
        
        # 1. Python sürüm kontrolü
        if not self.check_python_version():
            self.logger.info("Python 3.10 kurulumu gerekli...")
            if not self.install_python310():
                return False
                
        # 2. PDS-X sanal ortamı
        if not self.setup_pdsX_venv():
            return False
            
        # 3. Sanal ortamı aktive et
        if not self.activate_pdsX_venv():
            return False
            
        self.logger.info("PDS-X ortam kurulumu tamamlandı!")
        return True

class DependencyRegistry:
    """Bağımlılık kayıt defteri"""
    def __init__(self):
        self.logger = AdvancedLogger("DependencyRegistry")
        self.registry_file = Path("dependencies.json")
        self.learned_file = Path("learned_dependencies.json")
        self.registry = {}
        self.learned = defaultdict(list)
        
    def register(self, package: str, version: str = "latest"):
        """Bağımlılığı kaydet"""
        try:
            self.registry[package] = {
                "version": version,
                "installed_at": datetime.now().isoformat(),
                "installer": "auto_importer"
            }
            self.save_registry()
        except Exception as e:
            self.logger.error(f"Bağımlılık kaydedilirken hata: {e}")
            
    def load_registry(self) -> Dict[str, Any]:
        """Kayıt defterini yükle"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            if self.learned_file.exists():
                with open(self.learned_file, "r", encoding="utf-8") as f:
                    self.learned = defaultdict(list, json.load(f))
            return self.registry
        except Exception as e:
            self.logger.error(f"Kayıt defteri yüklenirken hata: {e}")
            return {}
            
    def save_registry(self):
        """Kayıt defterini kaydet"""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            with open(self.learned_file, "w", encoding="utf-8") as f:
                json.dump(dict(self.learned), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Kayıt defteri kaydedilirken hata: {e}")
            
    def learn_dependency(self, module: str, package: str):
        """Modül-paket ilişkisini öğren"""
        if package not in self.learned[module]:
            self.learned[module].append(package)
            self.save_registry()

class AutoImporter:
    """PDS-X Akıllı Modül Yükleyici Ana Sınıfı"""
    def __init__(self, entry_point_script=None, mode="NORMAL", log_to_file=True, log_to_terminal=True, silent_install=False, pdsX_args=None):
        # Ana parametreler
        self.entry_point_script = entry_point_script or "pdsXuv14.py"
        self.mode = mode
        self.log_to_file = log_to_file
        self.log_to_terminal = log_to_terminal
        self.silent_install = silent_install
        self.pdsX_args = pdsX_args or []
        
        # PDS-X özel dosyalar
        self.last_args_file = Path(".pdsx_last_args.json")
        self.dependencies_file = Path("dependencies.json")
        self.learned_deps_file = Path("learned_dependencies.json")
        self.backup_suffix = ".bak"
        
        # Log yönetimi
        self.setup_log_rotation()
        
        # Yardımcı bileşenler
        self.env_manager = EnvManager()
        self.dependency_registry = DependencyRegistry()
        self.logger = AdvancedLogger("PDS-X-AutoImporter") 
        self.real_time_monitor = RealTimeMonitor()
        self.summary_generator = SummaryGenerator()
        self.terminal_analyzer = TerminalAnalyzer()
        
        # Güvenli kapatma
        self.setup_signal_handlers()
        
        # Bileşenleri başlat
        self.initialize_components()
        
    def setup_log_rotation(self):
        """Log dosyalarını yönet"""
        log_files = ["pdsxu_errors.log", "pdsxu_info.log", "pdsxu_terminal.log"]
        max_size = 5 * 1024 * 1024  # 5MB
        
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists() and log_path.stat().st_size > max_size:
                backup_path = log_path.with_suffix(f"{log_path.suffix}{self.backup_suffix}")
                if backup_path.exists():
                    backup_path.unlink()  # Önceki backup'ı sil
                log_path.rename(backup_path)  # Mevcut dosyayı backup yap
                log_path.touch()  # Yeni boş dosya oluştur
                
    def setup_signal_handlers(self):
        """Güvenli kapatma için sinyal işleyicileri"""
        def signal_handler(signum, frame):
            self.logger.info(f"Sinyal alındı: {signum}")
            self.emergency_shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
            
        # Çıkışta temizlik
        atexit.register(self.cleanup)

    def emergency_shutdown(self):
        """Acil güvenli kapatma"""
        try:
            self.logger.warning("Acil kapatma başlatılıyor...")
            
            # Son durumu kaydet
            self.save_last_state()
            
            # Servisleri durdur
            self.real_time_monitor.stop()
            
            # Dosyaları kaydet
            self.dependency_registry.save_registry()
            
            self.logger.info("Acil kapatma tamamlandı")
            sys.exit(0)
            
        except Exception as e:
            self.logger.error(f"Acil kapatma hatası: {e}")
            os._exit(1)
            
    def save_last_state(self):
        """Son durumu kaydet"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "args": self.pdsX_args,
                "entry_script": self.entry_point_script,
                "mode": self.mode,
                "last_operation": "shutdown"
            }
            
            with open(self.last_args_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
            self.logger.info("Son durum kaydedildi")
            
        except Exception as e:
            self.logger.error(f"Son durum kaydetme hatası: {e}")

    def restart_pdsX_with_correct_env(self) -> bool:
        """PDS-X'i doğru ortamda yeniden başlat"""
        try:
            self.logger.info("PDS-X doğru ortamda yeniden başlatılıyor...")
            
            # Sanal ortam Python yolu
            venv_python = self.env_manager.get_venv_python_path()
            if not venv_python:
                self.logger.error("Sanal ortam Python bulunamadı!")
                return False
                
            # PDS-X komutunu oluştur
            cmd = [str(venv_python), self.entry_point_script] + self.pdsX_args
            
            self.logger.info(f"PDS-X komutu: {' '.join(cmd)}")
            
            # Mevcut AutoImporter'ı temizle
            self.cleanup()
            
            # PDS-X'i başlat
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                self.logger.info("PDS-X başarıyla tamamlandı")
                return True
            else:
                self.logger.error(f"PDS-X hata kodu: {result.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"PDS-X yeniden başlatma hatası: {e}")
            return False
            
    def handle_package_conflicts(self, conflict_info: str) -> bool:
        """Paket çakışmalarını çöz"""
        self.logger.info("Paket çakışması tespit edildi, çözüm aranıyor...")
        
        try:
            # Çakışan paketleri analiz et
            conflicts = self.terminal_analyzer.parse_conflict_output(conflict_info)
            
            for conflict in conflicts:
                self.logger.info(f"Çakışma çözülüyor: {conflict}")
                
                # Kural tabanlı çözüm
                if self.resolve_conflict_rule_based(conflict):
                    continue
                    
                # Genetik algoritma ile çözüm
                if self.resolve_conflict_genetic(conflict):
                    continue
                    
                self.logger.warning(f"Çözülemedi: {conflict}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Çakışma çözme hatası: {e}")
            return False
            
    def resolve_conflict_rule_based(self, conflict: str) -> bool:
        """Kural tabanlı çakışma çözme"""
        # Basit kural tabanlı çözümler
        rules = {
            "tensorflow": "pip install tensorflow==2.13.0 --no-deps",
            "numpy": "pip install numpy==1.24.3 --force-reinstall",
            "pillow": "pip install pillow --upgrade",
        }
        
        for package, solution in rules.items():
            if package.lower() in conflict.lower():
                self.logger.info(f"Kural tabanlı çözüm uygulanıyor: {solution}")
                try:
                    result = subprocess.run(solution.split(), capture_output=True, text=True)
                    return result.returncode == 0
                except:
                    return False
                    
        return False
        
    def resolve_conflict_genetic(self, conflict: str) -> bool:
        """Genetik algoritma ile çakışma çözme"""
        # Basitleştirilmiş genetik algoritma yaklaşımı
        self.logger.info("Genetik algoritma ile çözüm deneniyor...")
        
        # Bu kısım gerçek bir genetik algoritma implementasyonu gerektirir
        # Şimdilik basit bir deneme-yanılma yöntemi kullanıyoruz
        
        strategies = [
            "--force-reinstall",
            "--no-deps", 
            "--upgrade",
            "--no-cache-dir"
        ]
        
        for strategy in strategies:
            try:
                pip_cmd = self.env_manager.get_pip_path()
                if pip_cmd:
                    cmd = [str(pip_cmd), "install", conflict, strategy]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.logger.info(f"Genetik çözüm başarılı: {strategy}")
                        return True
            except:
                continue
                
        return False
        
    def _safe_get_pip_path(self):
        """Güvenli pip yolu alma"""
        try:
            pip_path = self.env_manager.get_pip_path()
            if pip_path:
                return str(pip_path)
            return None
        except Exception as e:
            self.log(f"Pip yolu alınamadı: {e}", "error")
            return None
            
    def log(self, message: str, level: str = "info") -> None:
        """Mesajı logla"""
        if hasattr(self.logger, 'log'):
            self.logger.log(level, message)
        else:
            print(f"[PDS-X] {level.upper()}: {message}")
            
    def _handle_keyboard_event(self, event):
        """Klavye olaylarını yönet"""
        try:
            if hasattr(event, 'event_type') and event.event_type == 'down':
                if hasattr(event, 'name') and event.name == 'q':
                    if hasattr(event, 'modifiers') and event.modifiers == {'ctrl', 'shift'}:
                        self._emergency_shutdown()
        except Exception as e:
            self.log(f"Klavye olayı işleme hatası: {e}", "error")

    def _emergency_shutdown(self):
        """Acil kapatma işlemi"""
        try:
            self.log("Acil kapatma başlatılıyor...", "warning")
            self.real_time_monitor.stop()
            self.cleanup()
            sys.exit(1)
        except Exception as e:
            self.log(f"Acil kapatma hatası: {e}", "error")
            os._exit(1)

    def initialize_components(self):
        """Bileşenleri başlatır"""
        try:
            # PDS-X ortam kontrolü
            if not self.setup_pdsX_environment():
                self.logger.warning("PDS-X ortam kurulumu eksik, devam ediliyor...")
            
            self.real_time_monitor.start()
            self.dependency_registry.load_registry()
            self.start_background_services()
            self.log("Bileşenler başarıyla başlatıldı", "info")
            return True
        except Exception as e:
            self.log(f"Bileşen başlatma hatası: {e}", "error")
            return False
            
    def setup_pdsX_environment(self) -> bool:
        """PDS-X için tam ortam kurulumu"""
        self.logger.info("PDS-X ortam kontrolü başlatılıyor...")
        
        try:
            # 1. Python sürüm kontrolü
            if not self.env_manager.check_python_version():
                self.logger.info("Uyumsuz Python sürümü, PDS-X yeniden başlatılacak...")
                if self.env_manager.setup_environment():
                    return self.restart_pdsX_with_correct_env()
                return False
                
            # 2. Sanal ortam kontrolü
            if not self.env_manager.is_running_in_venv():
                self.logger.info("Sanal ortam bulunamadı, oluşturuluyor...")
                if self.env_manager.setup_environment():
                    return self.restart_pdsX_with_correct_env()
                return False
                
            # 3. Cache dizini oluştur
            self.env_manager.cache_dir.mkdir(exist_ok=True)
            
            self.logger.info("PDS-X ortam kontrolü tamamlandı")
            return True
            
        except Exception as e:
            self.logger.error(f"PDS-X ortam kurulum hatası: {e}")
            return False
            
    def install_from_cache(self, package: str) -> bool:
        """Cache'den paket kur"""
        try:
            cache_file = self.env_manager.cache_dir / f"{package}.whl"
            if cache_file.exists():
                self.logger.info(f"Cache'den kuruluyor: {package}")
                
                pip_cmd = self.env_manager.get_pip_path()
                if pip_cmd:
                    result = subprocess.run([
                        str(pip_cmd), "install", str(cache_file)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.logger.info(f"Cache'den başarıyla kuruldu: {package}")
                        return True
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Cache kurulum hatası: {e}")
            return False
            
    def download_to_cache(self, package: str) -> bool:
        """Paketi cache'e indir"""
        try:
            pip_cmd = self.env_manager.get_pip_path()
            if not pip_cmd:
                return False
                
            result = subprocess.run([
                str(pip_cmd), "download", package, 
                "--dest", str(self.env_manager.cache_dir)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Cache'e indirildi: {package}")
                return True
            else:
                self.logger.warning(f"Cache indirme başarısız: {package}")
                return False
                
        except Exception as e:
            self.logger.error(f"Cache indirme hatası: {e}")
            return False
            
    def start_background_services(self):
        """Arka plan servislerini başlatır"""
        try:
            if not hasattr(self, 'real_time_monitor') or not self.real_time_monitor:
                self.real_time_monitor = RealTimeMonitor()
                self.real_time_monitor.start()
            
            # Diğer arka plan servisleri burada başlatılabilir
            self.log("Arka plan servisleri başlatıldı", "info")
            return True
        except Exception as e:
            self.log(f"Arka plan servisleri başlatılamadı: {e}", "error")
            return False

    def auto_install_package(self, package: str) -> bool:
        """PDS-X için gelişmiş paket kurulumu"""
        try:
            if self.check_package_installed(package):
                self.log(f"{package} zaten kurulu.")
                self.summary_generator.add_skipped(package, "Zaten kurulu")
                return True

            self.log(f"🔄 {package} kuruluyor...")
            
            # 1. Cache'den deneme
            if self.install_from_cache(package):
                self.summary_generator.add_success(package)
                self.dependency_registry.register(package.split("==")[0], package.split("==")[1] if "==" in package else "latest")
                return True
            
            # 2. Normal kurulum
            pip_cmd = self.env_manager.get_pip_path()
            if not pip_cmd:
                self.log("Pip bulunamadı!", "error")
                return False

            # Kurulum komutunu oluştur
            install_command = [str(pip_cmd), "install", package]
            if self.silent_install:
                install_command.append("--quiet")
            
            # Cache'e de indir
            self.download_to_cache(package)
                
            result = subprocess.run(install_command, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.log(f"✅ {package} başarıyla kuruldu.")
                self.summary_generator.add_success(package)
                self.dependency_registry.register(package.split("==")[0], package.split("==")[1] if "==" in package else "latest")
                return True
            else:
                error_message = result.stderr or result.stdout
                self.log(f"❌ {package} kurulum hatası: {error_message}", "error")
                
                # Çakışma var mı kontrol et
                if "conflict" in error_message.lower() or "incompatible" in error_message.lower():
                    self.log("Çakışma tespit edildi, çözüm deneniyor...", "warning")
                    if self.handle_package_conflicts(error_message):
                        # Çakışma çözüldü, tekrar dene
                        return self.auto_install_package(package)
                
                self.summary_generator.add_failure(package, error_message)
                return False
                
        except Exception as e:
            error_str = str(e)
            self.log(f"❌ {package} kurulurken hata: {error_str}", "error")
            self.summary_generator.add_failure(package, error_str)
            return False
            
    async def async_install_package(self, package: str) -> bool:
        """Paketi asenkron olarak kur"""
        return self.auto_install_package(package)
        
    def load_dependencies(self) -> Dict[str, str]:
        """Bağımlılık kaydını yükle"""
        try:
            deps = self.dependency_registry.load_registry()
            if not deps or not isinstance(deps, dict):
                return {}
            return deps
        except Exception as e:
            self.log(f"Bağımlılıklar yüklenemedi: {e}", "error")
            return {}

    def install_required_packages(self):
        """Gerekli paketleri kur"""
        success_count = 0
        for pkg, _ in REQUIRED_PACKAGES:
            if self.auto_install_package(pkg):
                success_count += 1
            
        self.log(f"Gerekli paketler kuruldu: {success_count}/{len(REQUIRED_PACKAGES)}")
        self.summary_generator.print_summary()
            
    def cleanup(self):
        """Temizlik işlemleri"""
        self.log("AutoImporter temizlik işlemleri başlatılıyor...", "info")
        self.real_time_monitor.stop()
        self.summary_generator.print_summary()
        self.log("AutoImporter temizlik tamamlandı.", "info")

    def check_package_installed(self, package: str) -> bool:
        """Paketin kurulu olup olmadığını kontrol et"""
        try:
            if not self.env_manager.is_running_in_venv():
                return False

            venv_python = self.env_manager.get_venv_python_path()
            if not venv_python:
                return False

            import_name = package.split("==")[0].split(">")[0].split("<")[0]
            import_name = self.terminal_analyzer.map_module_to_package(import_name, reverse=True)

            result = subprocess.run(
                [str(venv_python), "-c", f"import {import_name}"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            self.log(f"{package} kontrol hatası: {e}", "debug")
            return False
            
    def save_last_args(self, args: List[str]):
        """Son argümanları kaydet"""
        try:
            with open(self.last_args_file, "w", encoding="utf-8") as f:
                json.dump({"args": args, "timestamp": datetime.now().isoformat()}, f)
            self.log(f"Son argümanlar kaydedildi: {args}")
        except Exception as e:
            self.log(f"Son argümanlar kaydedilemedi: {e}", "error")

    def load_last_args(self) -> Optional[Dict]:
        """Son argümanları yükle"""
        try:
            if self.last_args_file.exists():
                with open(self.last_args_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.log(f"Son argümanlar yüklenemedi: {e}", "error")
            return None

    def replay_last_command(self) -> bool:
        """Son komutu tekrar çalıştır"""
        try:
            last = self.load_last_args()
            if not last or "args" not in last:
                self.log("Önceki komut bulunamadı.", "warning")
                return False
            args = last["args"]
            self.log(f"Önceki komut tekrar çalıştırılıyor: {args}")
            subprocess.run([sys.executable] + args)
            return True
        except Exception as e:
            self.log(f"Komut tekrar çalıştırılamadı: {e}", "error")
            return False

# =====================================================
# PDS-X AutoImporter Ana Entegrasyon Fonksiyonları
# =====================================================

def setup_pdsX_environment(**kwargs) -> bool:
    """PDS-X için tam ortam kurulumu - dışarıdan çağrılabilir"""
    try:
        print("🚀 PDS-X AutoImporter başlatılıyor...")
        
        # AutoImporter oluştur - TAMAMEN OTOMATİK
        ai = AutoImporter(
            entry_point_script=kwargs.get('entry_script', 'pdsXuv14.py'),
            mode=kwargs.get('mode', 'SILENT_AUTO'),
            silent_install=True,
            log_to_terminal=False,
            pdsX_args=kwargs.get('args', [])
        )
        
        print("✅ PDS-X ortam kurulumu tamamlandı!")
        return True
        
    except Exception as e:
        print(f"❌ PDS-X ortam kurulum hatası: {e}")
        return False

def silent_auto_setup():
    """Sessiz otomatik kurulum - REPL için"""
    try:
        from auto_importer_silent import SilentAutoImporter
        installer = SilentAutoImporter(mode="REPL_PREP")
        return installer.get_stats()
    except Exception as e:
        print(f"⚠️  Sessiz kurulum hatası: {e}")
        return {"error": str(e)}

def find_python310() -> Optional[str]:
    """Python 3.10 kurulumunu bul - dışarıdan çağrılabilir"""
    try:
        env_manager = EnvManager()
        python310_path = env_manager.find_python310()
        return str(python310_path) if python310_path else None
    except Exception as e:
        print(f"Python 3.10 arama hatası: {e}")
        return None

def install_missing_packages(packages: List[str], silent: bool = True) -> int:
    """Eksik paketleri kur - dışarıdan çağrılabilir"""
    try:
        ai = AutoImporter(silent_install=silent)
        success_count = 0
        
        for package in packages:
            if ai.auto_install_package(package):
                success_count += 1
                
        ai.cleanup()
        return success_count
        
    except Exception as e:
        print(f"Paket kurulum hatası: {e}")
        return 0

# =====================================================
# PDS-X AutoImporter İnteraktif Test Modu
# =====================================================

def interactive_demo():
    """DEVRE DIŞI - Bu fonksiyon import sırasında çalışmasın diye devre dışı bırakıldı"""
    print("📦 PDS-X AutoImporter (modül olarak yüklendi - interaktif demo devre dışı)")
    return  # Hızlıca çık, interaktif menüyü başlatma
    """Sessiz otomatik kurulum - interaktif değil"""
    print("� PDS-X AutoImporter sessiz otomatik kurulum başlatılıyor...")
    
    try:
        # Tamamen otomatik AutoImporter
        ai = AutoImporter(
            mode="SILENT_AUTO", 
            silent_install=True,
            log_to_terminal=False
        )
        
        print("✅ PDS-X AutoImporter sessiz kurulum tamamlandı!")
        
    except Exception as e:
        print(f"❌ Sessiz kurulum hatası: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # AutoImporter oluştur
        print("📦 PDS-X AutoImporter başlatılıyor...")
        ai = AutoImporter(mode="PDS-X_DEMO", silent_install=True)
        
        print(f"✅ AutoImporter başlatıldı! Mode: {ai.mode}")
        
        # Sistem durumu göster
        print("\n📊 PDS-X Sistem Durumu:")
        print(f"   🐍 Python Yolu: {sys.executable}")
        print(f"   📁 Çalışma Dizini: {Path.cwd()}")
        print(f"   🔧 PDS-X Sanal Ortam: {'✅ Aktif' if ai.env_manager.is_running_in_venv() else '❌ Kurulacak'}")
        
        # Python sürüm kontrolü
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"   🐍 Python Sürümü: {current_version} {'✅' if current_version == '3.10' else '⚠️  (3.10 öneriliyor)'}")
        
        # Cache durumu
        cache_size = len(list(ai.env_manager.cache_dir.glob("*.whl"))) if ai.env_manager.cache_dir.exists() else 0
        print(f"   📦 Cache: {cache_size} paket")
        
        # Monitor istatistikleri
        time.sleep(2)  # Biraz veri toplanması için bekle
        stats = ai.real_time_monitor.get_stats()
        print(f"   💻 CPU Ortalama: {stats['cpu_avg']:.1f}%")
        print(f"   🧠 Bellek Ortalama: {stats['memory_avg']:.1f}%")
        print(f"   ⏱️  Çalışma Süresi: {stats['uptime']:.1f} saniye")
        
        # İnteraktif menü
        while True:
            print("\n🎯 PDS-X AutoImporter - Ne yapmak istiyorsunuz?")
            print("1. 📦 Tek paket kur (cache ile)")
            print("2. 🔍 Paket kontrolü yap")
            print("3. 📋 Bağımlılık kayıt defterini göster")
            print("4. 📊 Sistem istatistiklerini göster")
            print("5. 🔄 PDS-X gerekli paketleri kur (ilk 5)")
            print("6. 🧪 Lazy import test")
            print("7. �️  PDS-X ortam durumunu kontrol et")
            print("8. 🚀 PDS-X'i yeniden başlat (simülasyon)")
            print("9. �🛑 Çıkış")
            
            choice = input("\n👉 Seçiminiz (1-9): ").strip()
            
            if choice == "1":
                package = input("📦 Kurmak istediğiniz paket adı: ").strip()
                if package:
                    print(f"\n🔄 {package} kuruluyor (cache kontrollü)...")
                    success = ai.auto_install_package(package)
                    if success:
                        print(f"✅ {package} başarıyla kuruldu!")
                    else:
                        print(f"❌ {package} kurulumu başarısız!")
                        
            elif choice == "2":
                package = input("🔍 Kontrol edilecek paket adı: ").strip()
                if package:
                    installed = ai.check_package_installed(package)
                    print(f"📦 {package}: {'✅ Kurulu' if installed else '❌ Kurulu değil'}")
                    
            elif choice == "3":
                deps = ai.load_dependencies()
                print(f"\n📋 PDS-X Bağımlılık Kayıtları ({len(deps)} adet):")
                if deps:
                    for pkg, info in deps.items():
                        version = info.get('version', 'unknown') if isinstance(info, dict) else info
                        installed_at = info.get('installed_at', 'unknown') if isinstance(info, dict) else 'unknown'
                        print(f"   📦 {pkg}: {version} ({installed_at})")
                else:
                    print("   📭 Henüz kayıtlı bağımlılık yok")
                    
            elif choice == "4":
                stats = ai.real_time_monitor.get_stats()
                print(f"\n📊 Güncel PDS-X Sistem İstatistikleri:")
                print(f"   💻 CPU Ortalama: {stats['cpu_avg']:.1f}%")
                print(f"   🧠 Bellek Ortalama: {stats['memory_avg']:.1f}%") 
                print(f"   ⏱️  Çalışma Süresi: {stats['uptime']:.1f} saniye")
                print(f"   📈 Veri Noktası: {stats['samples']} adet")
                
                # Cache istatistikleri
                cache_files = list(ai.env_manager.cache_dir.glob("*.whl")) if ai.env_manager.cache_dir.exists() else []
                print(f"   📦 Cache: {len(cache_files)} paket")
                
            elif choice == "5":
                print("\n🔄 PDS-X için ilk 5 gerekli paket kuruluyor...")
                success_count = 0
                for i, (pkg, ver) in enumerate(REQUIRED_PACKAGES[:5]):
                    print(f"📦 {i+1}/5: {pkg}=={ver} kuruluyor...")
                    if ai.auto_install_package(f"{pkg}=={ver}"):
                        success_count += 1
                        print(f"   ✅ {pkg} kuruldu")
                    else:
                        print(f"   ❌ {pkg} başarısız")
                print(f"\n📊 PDS-X Sonuç: {success_count}/5 paket kuruldu")
                
            elif choice == "6":
                print("\n🧪 PDS-X Lazy Import Test:")
                try:
                    global numpy
                    if numpy is None:
                        print("   📦 NumPy yükleniyor...")
                        import numpy as np
                        numpy = np
                        print("   ✅ NumPy başarıyla yüklendi!")
                    else:
                        print("   ✅ NumPy zaten yüklü!")
                    
                    # Basit test
                    arr = numpy.array([1, 2, 3, 4, 5])
                    print(f"   🔢 Test array: {arr}")
                    print(f"   📈 Array sum: {numpy.sum(arr)}")
                    
                except Exception as e:
                    print(f"   ❌ NumPy test hatası: {e}")
                    
            elif choice == "7":
                print("\n�️  PDS-X Ortam Durumu:")
                
                # Python sürüm kontrolü
                version_ok = ai.env_manager.check_python_version()
                print(f"   🐍 Python 3.10: {'✅' if version_ok else '❌'}")
                
                # Sanal ortam kontrolü
                venv_ok = ai.env_manager.is_running_in_venv()
                print(f"   🔧 PDS-X Sanal Ortam: {'✅' if venv_ok else '❌'}")
                
                # Python 3.10 arama
                python310 = ai.env_manager.find_python310()
                print(f"   🔍 Python 3.10 Yolu: {python310 if python310 else '❌ Bulunamadı'}")
                
                # Pip kontrolü
                pip_path = ai.env_manager.get_pip_path()
                print(f"   📦 Pip: {'✅' if pip_path else '❌'} {pip_path if pip_path else ''}")
                
            elif choice == "8":
                print("\n🚀 PDS-X Yeniden Başlatma Simülasyonu:")
                print("   📝 Son durum kaydediliyor...")
                ai.save_last_state()
                print("   ✅ Durum kaydedildi")
                print("   🔄 PDS-X doğru ortamda yeniden başlatılacak...")
                print("   (Gerçek durumda PDS-X burada yeniden başlatılır)")
                
            elif choice == "9":
                print("\n👋 PDS-X AutoImporter Demo kapatılıyor...")
                break
                
            else:
                print("❌ Geçersiz seçim! Lütfen 1-9 arası bir sayı girin.")
        
        # Temizlik
        print("\n🧹 PDS-X temizlik işlemleri...")
        ai.cleanup()
        print("✅ PDS-X AutoImporter Demo tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo kullanıcı tarafından durduruldu!")
        print("   💾 Güvenli kapatma işlemi...")
    except Exception as e:
        print(f"\n❌ Demo hatası: {e}")
        import traceback
        traceback.print_exc()

# Ana çalıştırma bloğu
if __name__ == "__main__":
    # Eğer bu dosya doğrudan çalıştırılırsa sessiz otomatik mod
    print("🔧 PDS-X AutoImporter otomatik mod başlatılıyor...")
    
    try:
        # Otomatik AutoImporter
        ai = AutoImporter(
            entry_point_script='pdsXuv14.py',
            mode="AUTO_SILENT",
            silent_install=True,
            log_to_terminal=False
        )
        
        print("✅ PDS-X AutoImporter otomatik tamamlandı!")
        
    except Exception as e:
        print(f"❌ AutoImporter hatası: {e}")
# MODÜL OLARAK İMPORT EDİLİRSE HİÇBİR ŞEY YAPMAYACAK
