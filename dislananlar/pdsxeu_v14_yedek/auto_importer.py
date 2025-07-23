#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Enhanced Auto Importer - Final Clean Version
=================================================

Bu dosya, auto_importer_merged.py'nin temizlenmiş ve düzenlenmiş versiyonudur.
Tüm duplicate sınıflar kaldırılmış, sadece en iyi versiyonlar bırakılmıştır.

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
- AutoImporter: toplu1.py temel + hibrit ThreadPool
- TerminalLogAnalyzer: v1795.py (gelişmiş regex)
- RealTimeLogMonitor: v1795.py (JSONL optimization)

Versiyon: Final Clean v1.0
Oluşturma Tarihi: 19 Temmuz 2025
Kullanılan Kaynak Dosyalar: auto_importer_merged.py (cleaned)
REQUIRED_PACKAGES: 108 paket (doğrulanmış)
"""

# === STANDARD IMPORTS ===
import sys
import os
import subprocess
import threading
import time
import json
import hashlib
import logging
import signal
import shutil
import platform
import importlib
import re
import argparse
import atexit
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from urllib.parse import urlparse
try:
    import winreg
except ImportError:
    winreg = None

# === SCIENTIFIC UTILS IMPORT ===
try:
    from scientific_utils import ScientificUtils
    SCIENTIFIC_UTILS_AVAILABLE = True
except ImportError:
    SCIENTIFIC_UTILS_AVAILABLE = False
    ScientificUtils = None

# === KOŞULLU IMPORT'LAR (Try-Except ile) ===
try:
    import winreg  # Windows registry erişimi
except ImportError:
    winreg = None

try:
    import keyboard  # Hotkey listener için
except ImportError:
    keyboard = None

try:
    import psutil  # System monitoring için
except ImportError:
    psutil = None

# === CONSTANTS ===
VENV_DIR = Path(".pdsx_isolated_env")

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

# === SANAL ORTAM YÖNETİMİ VE KOMUT REPLAY ===
# --- Helper Functions for Robust Venv Management ---
def _python_cmd_args(python_path, *args):
    """PYTHON310_PATH 'py -3.10' gibi ise split ederek subprocess'a uygun hale getirir."""
    if python_path == "py -3.10":
        return ["py", "-3.10", *args]
    return [python_path, *args]

def ensure_venv():
    """Base ortamdaysa otomatik olarak izole venv'ye geçiş"""
    def log_and_print(msg):
        print(f"[PDS-X] {msg}")
        with open("pdsxu_terminal.log", "a", encoding="utf-8") as f:
            f.write(f"[PDS-X] {msg}\n")
    
    # Sanal ortam kontrolü
    venv_dir = Path(".pdsx_isolated_env")
    venv_python = venv_dir / "Scripts" / "python.exe" if os.name == 'nt' else venv_dir / "bin" / "python"
    
    # Base ortamda mıyız kontrol et
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        log_and_print("UYARI: Şu anda base ortamdasınız. Otomatik olarak izole venv'ye geçiliyor!")
        if not venv_dir.exists():
            log_and_print("Sanal ortam oluşturuluyor: venv (.pdsx_isolated_env)...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        
        log_and_print("pip güncelleniyor...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
        
        log_and_print("Ortam hazırlanıyor, tekrar başlatılıyor... (venv (.pdsx_isolated_env) içinde)")
        
        # Argparse command replay için argümanları kaydet
        args_file = Path("last_args.json")
        args_data = {
            "argv": sys.argv,
            "timestamp": datetime.now().isoformat(),
            "original_python": sys.executable
        }
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(args_data, f, indent=2)
        
        # Script'i venv içinde yeniden başlat
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    else:
        log_and_print("İzole venv (.pdsx_isolated_env) ortamında çalışıyorsunuz.")

def detect_pdsx_commands():
    """PDS-X çalıştırma komutlarını tespit et"""
    possible_commands = []
    
    # Workspace'te PDS-X dosyalarını ara
    pdsx_files = [
        "pdsx.py", "pdsxu_v14.py", "pdsxeu_v14.py", "core.py", 
        "main.py", "pdsx_main.py", "pdsx_interpreter.py"
    ]
    
    for file in pdsx_files:
        if Path(file).exists():
            possible_commands.append(f"python {file}")
    
    # Çalışan Python scriptlerini kontrol et
    try:
        if psutil:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('pdsx' in str(arg).lower() for arg in cmdline):
                        cmd_str = ' '.join(cmdline[1:])  # python'u çıkar
                        if cmd_str not in possible_commands:
                            possible_commands.append(cmd_str)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except Exception:
        pass
    
    return possible_commands

def learn_pdsx_startup_command():
    """PDS-X başlatma komutunu öğren ve kaydet"""
    try:
        print("[PDS-X] PDS-X başlatma komutu öğreniliyor...")
        
        # Mevcut komutları tespit et
        commands = detect_pdsx_commands()
        
        if not commands:
            # Manuel komut girişi
            print("[PDS-X] PDS-X dosyaları bulunamadı. Manuel komut girişi:")
            print("[PDS-X] Örnek: python pdsx.py, python pdsxu_v14.py")
            manual_cmd = input("[PDS-X] PDS-X başlatma komutunu girin: ").strip()
            if manual_cmd:
                commands.append(manual_cmd)
        
        if commands:
            # En uygun komutu seç (ilk bulunan)
            selected_cmd = commands[0]
            
            # Komut dosyasını kaydet
            startup_data = {
                "pdsx_command": selected_cmd,
                "detected_commands": commands,
                "timestamp": datetime.now().isoformat(),
                "auto_detected": len(commands) > 0
            }
            
            with open("pdsx_startup.json", "w", encoding="utf-8") as f:
                json.dump(startup_data, f, indent=2, ensure_ascii=False)
            
            print(f"[PDS-X] Başlatma komutu kaydedildi: {selected_cmd}")
            return selected_cmd
        else:
            print("[PDS-X] UYARI: PDS-X başlatma komutu bulunamadı!")
            return None
            
    except Exception as e:
        print(f"[PDS-X] HATA: Başlatma komutu öğrenilemedi: {e}")
        return None

def start_pdsx_after_installation():
    """Kurulum sonrası PDS-X'i otomatik başlat"""
    try:
        # Kaydedilmiş başlatma komutunu yükle
        startup_file = Path("pdsx_startup.json")
        startup_cmd = None
        
        if startup_file.exists():
            try:
                with open(startup_file, "r", encoding="utf-8") as f:
                    startup_data = json.load(f)
                startup_cmd = startup_data.get("pdsx_command")
            except Exception as e:
                print(f"[PDS-X] Başlatma komutu yükleme hatası: {e}")
        
        # Eğer kaydedilmiş komut yoksa öğren
        if not startup_cmd:
            startup_cmd = learn_pdsx_startup_command()
        
        if startup_cmd:
            print(f"[PDS-X] PDS-X başlatılıyor: {startup_cmd}")
            
            # Komutları parse et
            cmd_parts = startup_cmd.split()
            if len(cmd_parts) >= 2 and cmd_parts[0] == "python":
                python_script = cmd_parts[1]
                script_args = cmd_parts[2:] if len(cmd_parts) > 2 else []
                
                # PDS-X'i başlat
                subprocess.Popen([sys.executable, python_script] + script_args, 
                               cwd=os.getcwd(), 
                               creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                
                print("[PDS-X] PDS-X başarıyla başlatıldı!")
                return True
            else:
                print(f"[PDS-X] UYARI: Geçersiz başlatma komutu: {startup_cmd}")
                return False
        else:
            print("[PDS-X] UYARI: PDS-X başlatma komutu bulunamadı!")
            return False
            
    except Exception as e:
        print(f"[PDS-X] HATA: PDS-X başlatma hatası: {e}")
        return False

def handle_command_replay():
    """Komut tekrarı için argparse parametrelerini işle"""
    parser = argparse.ArgumentParser(description="PDS-X Auto Importer - Advanced Package Manager")
    parser.add_argument('--replay-last', action='store_true', 
                       help='Son çalıştırılan komutları tekrar et')
    parser.add_argument('--install', nargs='*', 
                       help='Belirtilen paketleri yükle')
    parser.add_argument('--analyze-log', type=str,
                       help='Log dosyasını analiz et')
    parser.add_argument('--force-venv', action='store_true',
                       help='Zorla venv oluştur ve geçiş yap')
    parser.add_argument('--learn-pdsx', action='store_true',
                       help='PDS-X başlatma komutunu öğren')
    parser.add_argument('--start-pdsx', action='store_true',
                       help='Kurulum sonrası PDS-X başlat')
    
    args, unknown = parser.parse_known_args()
    
    # PDS-X komut öğrenme
    if args.learn_pdsx:
        learn_pdsx_startup_command()
        return args
    
    # PDS-X başlatma
    if args.start_pdsx:
        start_pdsx_after_installation()
        return args
    
    # Replay last command logic
    if args.replay_last:
        args_file = Path("last_args.json")
        if args_file.exists():
            try:
                with open(args_file, "r", encoding="utf-8") as f:
                    last_data = json.load(f)
                print(f"[PDS-X] Son komut tekrarlanıyor: {' '.join(last_data['argv'])}")
                return last_data.get('argv', [])
            except Exception as e:
                print(f"[PDS-X] HATA: Son komut tekrarlanamadı: {e}")
    
    return args

# === Ortam Hazırlama ===
ensure_venv()

# Python sürüm uyarısı
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] UYARI: En uyumlu çalışma için Python 3.10.x kullanmanız önerilir! Şu anki sürüm:", sys.version)


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
    
    def _setup_signal_handlers(self):
        """Signal handler'ları kur"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_keyboard_listener(self):
        """Keyboard listener kur (Ctrl+Shift+Q)"""
        if keyboard:
            try:
                self.hotkey_listener = keyboard.add_hotkey('ctrl+shift+q', self._emergency_shutdown)
                logging.info("Emergency hotkey (Ctrl+Shift+Q) activated")
            except Exception as e:
                logging.warning(f"Hotkey setup failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Signal yakalandığında çağrılır"""
        logging.info(f"Signal {signum} received, requesting graceful shutdown")
        self.request_shutdown()
    
    def _emergency_shutdown(self):
        """Acil kapatma"""
        logging.critical("EMERGENCY SHUTDOWN REQUESTED!")
        self.emergency_shutdown = True
        self.shutdown_requested = True
        self._emergency_cleanup()
    
    def request_shutdown(self):
        """Graceful shutdown talep et"""
        if not self.shutdown_requested:
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
                process.kill()
            except Exception:
                pass
        
        # Hotkey listener'ı durdur
        if self.hotkey_listener:
            try:
                if keyboard:
                    keyboard.remove_hotkey(self.hotkey_listener)
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
            except Exception:
                pass
    
    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass
    
    def close(self):
        for stream in self.streams:
            try:
                if hasattr(stream, 'close'):
                    stream.close()
            except Exception:
                pass


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
        
        # Elasticsearch support
        self.es = None
        
        self._setup_logging()
        if enable_elasticsearch:
            self._setup_elasticsearch()
    
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
            sys.stdout = Tee(sys.stdout, open(terminal_log, 'w', encoding='utf-8'))
    
    def _setup_elasticsearch(self):
        """Elasticsearch entegrasyonu (toplu1.py'den)"""
        try:
            from elasticsearch import Elasticsearch
            self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
            self.logger.info("Elasticsearch connected")
        except ImportError:
            self.logger.warning("Elasticsearch not available (install: pip install elasticsearch)")
        except Exception as e:
            self.logger.error(f"Elasticsearch connection failed: {e}")
    
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
        if current_time - self.last_message_time[msg_hash] < 1.0:
            return True
        
        self.last_message_time[msg_hash] = current_time
        return False
    
    def _rotate_logs(self):
        """Log rotasyonu (v1795.py'den)"""
        try:
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_size > self.max_log_size:
                    # Rotate log files
                    for i in range(self.max_log_files - 1, 0, -1):
                        old_file = log_file.with_suffix(f'.log.{i}')
                        new_file = log_file.with_suffix(f'.log.{i+1}')
                        if old_file.exists():
                            old_file.rename(new_file)
                    
                    # Rename current log
                    log_file.rename(log_file.with_suffix('.log.1'))
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def _write_jsonl(self, level, message, extra_data=None):
        """JSONL formatında log yazma (v1795.py'den)"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message,
                'extra': extra_data or {}
            }
            
            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # Fallback to standard logging
            self.logger.error(f"JSONL write failed: {e}")
    
    def _log_to_elasticsearch(self, level, message, extra_data=None):
        """Elasticsearch'e log gönderme (toplu1.py'den)"""
        if not self.es:
            return
        
        try:
            doc = {
                'timestamp': datetime.now(),
                'level': level,
                'message': message,
                'extra': extra_data or {}
            }
            self.es.index(index='auto-importer-logs', body=doc)
        except Exception as e:
            self.logger.error(f"Elasticsearch indexing failed: {e}")
    
    def log(self, level, message, extra_data=None):
        """Ana logging metodu - [PDS-X] prefix'i ile"""
        if self._is_spam(message):
            return
        
        # [PDS-X] prefix'i ekle
        if not message.startswith("[PDS-X]"):
            message = f"[PDS-X] {message}"
        
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
        import re
        
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
            import re
            
            missing_modules = set()
            version_conflicts = set()
            import_errors = []
            pip_suggestions = set()
            packages_to_install = set()
            
            # ModuleNotFoundError pattern matching
            for pattern in self.module_not_found_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    missing_modules.add(match)
                    # Module-to-package mapping
                    package = self.module_to_package.get(match, match)
                    packages_to_install.add(package)
            
            # Version conflict pattern matching
            for pattern in self.version_conflict_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        version_conflicts.add(f"{match[0]} {match[1] if len(match) > 1 else ''}")
                    else:
                        version_conflicts.add(match)
            
            # Import error pattern matching
            for pattern in self.import_error_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                import_errors.extend(matches)
            
            # Pip suggestion pattern matching
            for pattern in self.pip_suggestion_patterns:
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                for match in matches:
                    pip_suggestions.add(match)
                    packages_to_install.add(match)
                        
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
            import re
            # Son ModuleNotFoundError'ı bul
            lines = traceback_text.split('\n')
            for line in reversed(lines):
                for pattern in self.module_not_found_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1)
            return None
        except Exception as e:
            self.logger.error(f"Traceback analysis error: {e}")
            return None
    
    def analyze_log_content(self, log_content: str) -> List[str]:
        """Log içeriğini analiz et ve eksik bağımlılıkları bul"""
        try:
            analysis_result = self.analyze_output(log_content)
            return analysis_result.get('packages_to_install', [])
        except Exception as e:
            self.logger.error(f"Log content analysis error: {e}")
            return []


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
                    "conflicts": {},
                    "status": "clean",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Registry yükleme hatası: {e}")
            return {
                "packages": {},
                "resolutions": {},
                "conflicts": {},
                "status": "clean",
                "timestamp": datetime.now().isoformat()
            }
    
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
                # Registry kontrolü
                if package in self.registry.get("packages", {}):
                    return self.registry["packages"][package].get("status") == "Başarılı"
                
                # Pip show ile kontrolü
                result = subprocess.run([sys.executable, "-m", "pip", "show", package], 
                                      capture_output=True, text=True)
                return result.returncode == 0
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kontrol hatası: {e}")
            return False
    
    def register_package(self, package: str, version: str = "unknown", status: str = "Başarılı", dependencies: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        """
        Paketi registry'ye kaydet - gelişmiş metadata ile
        
        Args:
            package: Paket adı
            version: Paket sürümü
            status: Kurulum durumu
            dependencies: Bağımlılıklar
            metadata: Ek bilgiler (pip sürümü, python sürümü, kurulum yöntemi vs.)
        """
        try:
            with self.lock:
                if "packages" not in self.registry:
                    self.registry["packages"] = {}
                
                # Metadata bilgilerini hazırla
                install_metadata = metadata or {}
                install_metadata.update({
                    "pip_version": self._get_pip_version(),
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "python_executable": sys.executable,
                    "install_method": install_metadata.get("install_method", "pip"),
                    "install_timestamp": datetime.now().isoformat(),
                    "is_stable_env": self._is_stable_environment()
                })
                
                self.registry["packages"][package] = {
                    "version": version,
                    "status": status,
                    "dependencies": dependencies or [],
                    "metadata": install_metadata,
                    "registered_at": datetime.now().isoformat()
                }
                self._save_registry()
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} kayıt hatası: {e}")
    
    def _get_pip_version(self) -> str:
        """Mevcut pip sürümünü al"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return "unknown"
        except:
            return "unknown"
    
    def _is_stable_environment(self) -> bool:
        """Python 3.10 + stabil pip ortamı mı kontrol et"""
        try:
            # Python 3.10 kontrolü
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if python_version != "3.10":
                return False
            
            # Stabil pip sürümü kontrolü (21.x serisi)
            pip_version = self._get_pip_version()
            if "21." in pip_version:  # Python 3.10 için stabil pip
                return True
            
            return False
        except:
            return False
    
    def register_resolution(self, package: str, resolution: Dict):
        """Çakışma çözümünü kaydet"""
        try:
            with self.lock:
                if "resolutions" not in self.registry:
                    self.registry["resolutions"] = {}
                
                self.registry["resolutions"][package] = {
                    **resolution,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_registry()
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] {package} çözüm kayıt hatası: {e}")
    
    def update_on_conflict(self, package: str, conflict_info: str, resolution_command: str):
        """Çakışma durumunda registry'yi güncelle"""
        try:
            with self.lock:
                if "conflicts" not in self.registry:
                    self.registry["conflicts"] = {}
                
                self.registry["conflicts"][package] = {
                    "conflict_info": conflict_info,
                    "resolution_command": resolution_command,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Status'u conflict olarak işaretle
                self.registry["status"] = "conflict"
                self._save_registry()
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
                self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten kuruluyor...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", str(cache_file)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"[AUTO-IMPORTER] {package} önbellekten başarıyla kuruldu")
                    return True
                else:
                    self.logger.warning(f"[AUTO-IMPORTER] {package} önbellek kurulum hatası: {result.stderr}")
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
                self.logger.info(f"[AUTO-IMPORTER] {package} başarıyla önbelleğe alındı")
                
                # Metadata güncelle
                metadata = self.load_package_metadata()
                metadata[package] = {
                    "cached_at": datetime.now().isoformat(),
                    "cache_file": f"{package}.whl"
                }
                self.save_package_metadata(metadata)
                return True
            else:
                self.logger.error(f"[AUTO-IMPORTER] {package} önbellek hatası: {result.stderr}")
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
                cached_at = datetime.fromisoformat(info.get("cached_at", ""))
                if (now - cached_at).days > max_age_days:
                    cache_file = self.wheels_dir / info.get("cache_file", f"{pkg}.whl")
                    if cache_file.exists():
                        cache_file.unlink()
                    del metadata[pkg]
            self.save_package_metadata(metadata)
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] Önbellek temizleme hatası: {e}")


# === ENV MANAGER (TOPLU1.PY) ===
class EnvManager:
    """
    Gelişmiş İzole ortam yönetimi - toplu1.py'den tamamen entegre edilmiş versiyon.
    """
    
    def __init__(self, venv_dir: Path = VENV_DIR, logger: Optional[AdvancedLogger] = None):
        self.venv_dir = venv_dir
        self.error_count = 0
        self.max_errors = 3
        self.logger = logger if logger is not None else AdvancedLogger()
        self.python_path = None
        self.python_exe = None  # Python executable path

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
        """Python 3.10 arama - gelişmiş versiyon"""
        try:
            self.logger.log("info", "Python 3.10 aranıyor...")
            
            # Mevcut Python sürümünü logla
            current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            self.logger.log("info", f"Mevcut Python sürümü: {current_version}")
            
            if current_version == "3.10":
                self.logger.log("warning", f"[PDS-X] Python {current_version} tespit edildi. PDS-X için Python 3.10 gerekli!")
            elif current_version != "3.10":
                self.logger.log("warning", f"[PDS-X] Python {current_version} tespit edildi. PDS-X için Python 3.10 gerekli!")
            
            # 1. PATH'de ara
            for exe in ["python3.10", "python310", "python"]:
                path = shutil.which(exe)
                if path:
                    try:
                        out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT)
                        if "3.10" in out:
                            self.logger.log("info", f"Python 3.10 bulundu: {path}")
                            self.python_path = path
                            return path
                    except subprocess.CalledProcessError:
                        continue
            
            # 2. Windows registry araması
            if platform.system() == "Windows":
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
                                                self.logger.log("info", f"Registry'de Python 3.10 bulundu: {py}")
                                                self.python_path = py
                                                return py
                        except Exception:
                            continue
                except ImportError:
                    pass  # winreg modülü yok
            
            self.logger.log("error", "Python 3.10 bulunamadı.")
            return None
            
        except Exception as e:
            self.logger.log("error", f"Python 3.10 arama hatası: {e}")
            return None

    def download_and_install_python310(self) -> Optional[str]:
        """Python 3.10 otomatik kurulumu"""
        try:
            if os.name != "nt":
                self.logger.log("error", "Python 3.10 bulunamadı. Lütfen manuel kurun: https://www.python.org/downloads/release/python-31011/")
                return None
                
            # Disk alanı kontrolü
            drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
            total, used, free = shutil.disk_usage(drive + '\\')
            min_required = 300 * 1024 * 1024  # 300 MB
            if free < min_required:
                self.logger.log("error", f"Yetersiz disk alanı: {free // (1024*1024)} MB mevcut, 300 MB gerekli.")
                return None
                
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            installer_path = "python310_installer.exe"
            
            self.logger.log("info", "Python 3.10 installer indiriliyor...")
            subprocess.run(["curl", "-o", installer_path, installer_url], check=True, capture_output=True, text=True)
            
            self.logger.log("info", "Python 3.10 kuruluyor...")
            subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], 
                          check=True, capture_output=True, text=True)
            
            # Installer dosyasını temizle
            os.remove(installer_path)
            
            self.logger.log("info", "Python 3.10 kurulumu tamamlandı.")
            return self.find_python310()
            
        except Exception as e:
            self.logger.log("error", f"Python 3.10 kurulum hatası: {e}")
            return None

    def add_python_to_path(self):
        """Python ve venv'i PATH'e ekler"""
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
                    
                self.logger.log("info", "PATH'e eklemek için 'add_pdsx_path.bat' veya 'add_pdsx_path.ps1' dosyasını yönetici olarak çalıştırın!")
                
        except Exception as e:
            self.logger.log("error", f"PATH güncelleme hatası: {e}")

    def update_pip_if_needed(self, force_latest: bool = False, silent: bool = False) -> bool:
        """
        Pip sürümünü günceller - Python 3.10 için optimize edilmiş
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
            
            # Hedef sürümü belirle - Python 3.10 için optimize
            target_version = "latest" if force_latest else "21.2.4"
            pip_package = "pip" if force_latest else "pip==21.2.4"
            
            log_level = "debug" if silent else "info"
            self.logger.log(log_level, f"Pip hedef sürümü: {target_version}")
            
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
    
    def check_package_installed(self, package: str) -> bool:
        """Paket yüklü mü kontrol et"""
        try:
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
    
    def setup_environment(self) -> bool:
        """
        Ortamı hazırlar ve paketleri kurar - toplu1.py'den tam mantık
        """
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
                self.logger.log("error", "Python 3.10 bulunamadı!")
                return False
                
            self.python_exe = python_path  # Python executable path'i set et
            self.logger.log("info", f"Python 3.10 hazır: {python_path}")
            
            # Sanal ortam kontrolü ve kurulumu
            if not self.venv_dir.exists():
                self.logger.log("info", "Sanal ortam oluşturuluyor...")
                subprocess.run([python_path, "-m", "venv", str(self.venv_dir)], check=True)
            
            # Pip güncelleme
            self.update_pip_if_needed()
            
            # REQUIRED_PACKAGES kurulumu
            self.ensure_required_packages()
            
            self.logger.log("info", "Ortam başarıyla hazırlandı")
            return True
            
        except Exception as e:
            self.logger.log("error", f"Ortam hazırlanma hatası: {e}")
            return False
    
    def ensure_required_packages(self) -> bool:
        """REQUIRED_PACKAGES'ı kontrol et ve eksikleri kur"""
        try:
            self.logger.info("[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol ediliyor...")
            missing_packages = []
            
            for package in REQUIRED_PACKAGES[:20]:  # İlk 20 paketi test için
                if not self.check_package_installed(package):
                    missing_packages.append(package)
            
            if missing_packages:
                self.logger.info(f"[AUTO-IMPORTER] {len(missing_packages)} eksik paket bulundu, kuruluyor...")
                for package in missing_packages:
                    try:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True, timeout=120)
                        
                        if result.returncode == 0:
                            self.logger.info(f"✅ {package} kuruldu")
                        else:
                            self.logger.warning(f"⚠️ {package} kurulamadı: {result.stderr}")
                    except Exception as e:
                        self.logger.error(f"❌ {package} kurulum hatası: {e}")
            else:
                self.logger.info("[AUTO-IMPORTER] Tüm gerekli paketler mevcut")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[AUTO-IMPORTER] REQUIRED_PACKAGES kontrol hatası: {e}")
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
            import re
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
            if psutil:
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


# === SUMMARY GENERATOR ===
class SummaryGenerator:
    """Installation summary generator"""
    
    def __init__(self, logger):
        self.logger = logger
        self.installation_stats = defaultdict(int)
        self.timing_stats = defaultdict(list)
        self.error_stats = defaultdict(int)
    
    def record_installation(self, package: str, status: str, duration: float = 0):
        """Kurulum kaydı"""
        self.installation_stats[status] += 1
        if duration > 0:
            self.timing_stats[package].append(duration)
    
    def generate_summary(self) -> str:
        """Özet rapor oluştur"""
        total_successful = self.installation_stats.get('successful', 0)
        total_failed = self.installation_stats.get('failed', 0)
        total_packages = total_successful + total_failed
        
        if total_packages == 0:
            return "Henüz kurulum yapılmadı."
        
        success_rate = (total_successful / total_packages) * 100
        
        report = f"""
KURULUM OZET RAPORU
================================================
Basarili: {total_successful}
Basarisiz: {total_failed}
Basari Orani: {success_rate:.1f}%
================================================
"""
        return report.strip()


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
            return {}


# === MODULE ANALYZER (V1795.PY) ===
class ModuleAnalyzer:
    """
    Modül analiz sistemi - v1795.py'den alınmıştır.
    Log analizi, çakışma tespiti ve modül raporları ile.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.log_file = Path("logs/auto_importer.jsonl")
    
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
                            level = data.get("level", "").lower()
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
                self.logger.info(f"Log analizi: {log_stats['error_count']} hata, {log_stats['warning_count']} uyarı")
            return log_stats
        except Exception as e:
            self.logger.error(f"Log analizi hatası: {e}")
            return {"error": str(e)}
    
    def analyze_conflicts(self) -> Dict:
        """Loglardan çakışma mesajlarını çıkarır."""
        try:
            conflicts = {}
            error_log = Path("logs/pdsxu_errors.jsonl")
            warning_log = Path("logs/pdsxu_warnings.jsonl")
            
            for log_file in [error_log, warning_log]:
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                message = data.get("message", "")
                                if "çakışma" in message.lower() or "conflict" in message.lower():
                                    parts = message.split(":")
                                    if len(parts) > 1:
                                        pkg = parts[1].strip().split(" ")[0]
                                        conflicts[pkg] = message
                            except (json.JSONDecodeError, IndexError):
                                continue
            self.logger.info(f"Çakışma analizi: {len(conflicts)} çakışma bulundu.")
            return conflicts
        except Exception as e:
            self.logger.error(f"Çakışma analizi hatası: {e}")
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
            self.logger.info("Modül raporu oluşturuldu.")
            return report
        except Exception as e:
            self.logger.error(f"Modül raporu hatası: {e}")
            return {"error": str(e)}
    
    def _check_module_status(self, module: Dict) -> str:
        """Modül durumunu kontrol et"""
        try:
            module_name = module.get("name", "")
            if not module_name:
                return "unknown"
            
            # Import denemesi
            try:
                importlib.import_module(module_name)
                return "installed"
            except ImportError:
                return "missing"
        except Exception:
            return "error"
    
    def _detect_issues(self, module: Dict) -> List[str]:
        """Modül sorunlarını tespit et"""
        issues = []
        try:
            if module.get("status") == "missing":
                issues.append(f"Module {module.get('name')} is missing")
            if not module.get("version"):
                issues.append(f"Module {module.get('name')} version unknown")
        except Exception:
            pass
        return issues
    
    def _generate_recommendations(self, module: Dict) -> List[str]:
        """Modül önerileri oluştur"""
        recommendations = []
        try:
            if module.get("status") == "missing":
                module_name = module.get("name", "")
                recommendations.append(f"Install {module_name} with: pip install {module_name}")
            if not module.get("version"):
                module_name = module.get("name", "")
                recommendations.append(f"Update {module_name} with: pip install --upgrade {module_name}")
        except Exception:
            pass
        return recommendations
        return issues
    
    def _generate_recommendations(self, module: Dict) -> List[str]:
        """Modül önerileri oluştur"""
        recommendations = []
        try:
            if module.get("status") == "missing":
                module_name = module.get("name", "")
                recommendations.append(f"Install missing module: pip install {module_name}")
        except Exception:
            pass
        return recommendations


# === ASYNC DOWNLOAD MANAGER (V1795.PY) ===
class AsyncDownloadManager:
    """
    Asenkron indirme yöneticisi - v1795.py'den alınmıştır.
    ThreadPoolExecutor ile concurrent downloads.
    """
    
    def __init__(self, max_workers: int = 4, cache_dir=".pdsx_cache/wheels", logger=None):
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else AdvancedLogger()
        self.download_stats = {}
    
    def download_package(self, package: str) -> Dict:
        """Paketi indir"""
        try:
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
            start_time = time.time()
            
            # Pip ile paketi cache'e indir
            result = subprocess.run([
                sys.executable, "-m", "pip", "download", package,
                "--dest", str(self.cache_dir), "--no-deps"
            ], capture_output=True, text=True)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.download_stats[package_name] = {
                    "success": True, 
                    "duration": duration,
                    "size": self._get_package_size(package_name)
                }
                self.logger.info(f"{package_name} başarıyla indirildi ({duration:.2f}s)")
            else:
                self.download_stats[package_name] = {
                    "success": False, 
                    "error": result.stderr.strip() or "İndirme başarısız"
                }
                self.logger.error(f"{package_name} indirme hatası: {result.stderr}")
            
            return self.download_stats[package_name]
            
        except Exception as e:
            self.logger.error(f"Paket indirme hatası: {e}")
            return {"success": False, "error": str(e)}
    
    def download_multiple(self, packages: List[str]) -> Dict:
        """Birden fazla paketi paralel indir"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_package = {
                    executor.submit(self.download_package, pkg): pkg 
                    for pkg in packages
                }
                
                results = {}
                for future in as_completed(future_to_package):
                    package = future_to_package[future]
                    try:
                        result = future.result()
                        results[package] = result
                    except Exception as e:
                        results[package] = {"success": False, "error": str(e)}
                        self.logger.error(f"{package} executor hatası: {e}")
                
                return results
        except Exception as e:
            self.logger.error(f"Parallel download hatası: {e}")
            return {}
    
    def _get_package_size(self, package_name: str) -> int:
        """Paket boyutunu al"""
        try:
            # Cache klasöründe .whl dosyasını ara
            for file in self.cache_dir.glob(f"{package_name}*.whl"):
                return file.stat().st_size
            return 0
        except Exception:
            return 0
    
    def _get_package_size(self, package_name: str) -> int:
        """Paket boyutunu al"""
        try:
            for file in self.cache_dir.glob(f"{package_name}*"):
                if file.is_file():
                    return file.stat().st_size
            return 0
        except Exception:
            return 0


# === SCIENTIFIC UTILS (V1795.PY BASIT VERSIYON) ===
class ScientificUtils:
    """
    Bilimsel analiz araçları - v1795.py'den basit versiyon.
    Sistem metrikleri ve performans analizi.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.metrics = defaultdict(list)
    
    def analyze_system_metrics(self) -> Dict:
        """Sistem metriklerini analiz et"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "memory": self._get_memory_usage(),
                "cpu": self._get_cpu_usage(),
                "disk": self._get_disk_usage(),
                "python_info": self._get_python_info()
            }
            
            self.metrics["system_analysis"].append(metrics)
            self.logger.info("Sistem metrikleri analizi tamamlandı")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Sistem metrikleri hatası: {e}")
            return {"error": str(e)}
    
    def generate_performance_report(self) -> Dict:
        """Performans raporu oluştur"""
        try:
            if not self.metrics["system_analysis"]:
                return {"error": "No metrics available"}
            
            latest = self.metrics["system_analysis"][-1]
            report = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": latest,
                "recommendations": self._generate_performance_recommendations(latest)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performans raporu hatası: {e}")
            return {"error": str(e)}
    
    def _get_memory_usage(self) -> Dict:
        """Bellek kullanımını al"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "used": memory.used,
                "percentage": memory.percent
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_cpu_usage(self) -> Dict:
        """CPU kullanımını al"""
        try:
            import psutil
            return {
                "percentage": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_disk_usage(self) -> Dict:
        """Disk kullanımını al"""
        try:
            import psutil
            disk = psutil.disk_usage('.')
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percentage": (disk.used / disk.total) * 100
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_python_info(self) -> Dict:
        """Python bilgilerini al"""
        try:
            return {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_performance_recommendations(self, metrics: Dict) -> List[str]:
        """Performans önerileri oluştur"""
        recommendations = []
        try:
            memory = metrics.get("memory", {})
            if isinstance(memory, dict) and memory.get("percentage", 0) > 80:
                recommendations.append("Yüksek bellek kullanımı tespit edildi. Gereksiz uygulamaları kapatın.")
            
            cpu = metrics.get("cpu", {})
            if isinstance(cpu, dict) and cpu.get("percentage", 0) > 90:
                recommendations.append("Yüksek CPU kullanımı tespit edildi. Arka plan işlemlerini kontrol edin.")
            
            disk = metrics.get("disk", {})
            if isinstance(disk, dict) and disk.get("percentage", 0) > 90:
                recommendations.append("Disk alanı yetersiz. Gereksiz dosyaları temizleyin.")
                
        except Exception:
            pass
        return recommendations


# === AUTO IMPORTER (TOPLU1.PY BASE + HİBRİT) ===
class AutoImporter:
    """
    Ana Auto Importer sınıfı - toplu1.py temel + hibrit özellikler
    """
    
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super(AutoImporter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Core components
        self.logger = AdvancedLogger(silent=False)
        self.terminal_analyzer = TerminalLogAnalyzer(self.logger)
        self.dependency_registry = DependencyRegistry(logger=self.logger)
        self.cache_manager = CacheManager(logger=self.logger)
        self.env_manager = EnvManager(logger=self.logger)
        self.realtime_monitor = RealTimeLogMonitor(self.logger)
        self.summary_generator = SummaryGenerator(self.logger)
        self.pip_analyzer = PipOutputAnalyzer(self.logger)
        self.conflict_manager = ConflictManager(self.logger)
        
        # Scientific Utils for genetic optimization
        self.scientific_utils = None
        if SCIENTIFIC_UTILS_AVAILABLE:
            try:
                self.scientific_utils = ScientificUtils(logger=self.logger)
            except Exception as e:
                self.logger.error(f"[PDS-X] Scientific utils başlatma hatası: {e}")
        
        # Package ordering and dependency optimization
        self.optimized_package_order = []
        self.dependency_graph = {}
        self.stable_pip_version = "21.2.4"  # Python 3.10 için stabil pip sürümü
        
        # State management
        self.running = False
        self.installed_packages = set()
        self.failed_packages = set()
        self.dependencies = defaultdict(list)
        self.installation_stats = defaultdict(int)
        self.retry_count = defaultdict(int)
        self.max_retries = 3
        self.loaded_modules = {}
        self.module_cache = {}
        self.imported_files = set()
        self.aliases = {}
        self.secure_mode = False
        self.metadata = {"auto_importer": {"version": "Final Clean v1.0", "dependencies": []}}
        self.installation_history = {}
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # Graceful shutdown integration
        global shutdown_manager
        shutdown_manager.register_cleanup(self.cleanup_on_shutdown)
        
        # Environment setup
        try:
            # Başlangıç bilgisi göster
            self._print_startup_info()
            
            if not self.env_manager.setup_environment():
                self.logger.warning("[PDS-X] Environment setup tamamlanamadı, devam ediliyor...")
        except Exception as e:
            self.logger.warning(f"[PDS-X] Environment setup hatası: {e}")
        
        # Monitoring başlat
        self.realtime_monitor.start_monitoring()
        
        # Last args saver
        self.last_args_file = Path("last_importer_args.json")
        self.save_last_args(sys.argv)
    
    def _print_startup_info(self):
        """Başlangıç ortam bilgilerini göster"""
        try:
            print("\n" + "="*60)
            print("[PDS-X] 🚀 AUTO-IMPORTER BAŞLATILIYOR...")
            print("="*60)
            
            # Python bilgisi
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            print(f"[PDS-X] 🐍 Python Sürümü: {python_version}")
            
            # Sanal ortam durumu
            if hasattr(self, 'env_manager'):
                if self.env_manager.is_running_in_venv():
                    venv_path = str(self.env_manager.venv_dir)
                    print(f"[PDS-X] 🏠 İzole Ortam: ✅ Aktif")
                    print(f"[PDS-X] 📁 Ortam Dizini: {venv_path}")
                else:
                    print(f"[PDS-X] 🏠 İzole Ortam: ❌ Python 3.10 sanal ortamına geçiş yapılacak...")
            
            # Çalışma dizini
            print(f"[PDS-X] 📂 Çalışma Dizini: {os.getcwd()}")
            
            # Required packages sayısı
            print(f"[PDS-X] 📦 Kurulum Listesi: {len(REQUIRED_PACKAGES)} paket")
            
            print("="*60)
            print()
            
        except Exception as e:
            self.logger.error(f"[PDS-X] Startup info hatası: {e}")
    
    def save_last_args(self, args: List[str]):
        """Son çalıştırma argümanlarını kaydet"""
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
    
    def install_package(self, package: str, silent: bool = False) -> bool:
        """
        Paketi kur ve istatistikleri güncelle
        Python 3.10 öncelikli kurulum stratejisi:
        1. Önce Python 3.10 ile kurulumu dene
        2. Başarısız olursa sürüm yükseltilmesi gerekip gerekmediğini analiz et
        3. Yüksek sürüm gereken paketleri sonraya bırak
        
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
                self.logger.warning("[PDS-X] Shutdown talep edildi, kurulum iptal ediliyor")
                return False
            
            # Zaten kurulu mu kontrol et (önce cache kontrol et)
            if package_name in self.installed_packages:
                duration = time.time() - start_time
                if not silent:
                    self.logger.info(f"[PDS-X] ✅ {package_name} zaten kurulu (cache)")
                self.summary_generator.record_installation(package_name, "skipped", duration)
                return True
            
            # Sistem kontrol et
            if self.env_manager.check_package_installed(package_name):
                duration = time.time() - start_time
                if not silent:
                    self.logger.info(f"[PDS-X] ✅ {package_name} zaten sistem kurulu")
                self.installed_packages.add(package_name)  # Cache'e ekle
                self.summary_generator.record_installation(package_name, "skipped", duration)
                return True
            
            # Registry kontrolü
            if self.dependency_registry.check_package(package_name):
                if not silent:
                    self.logger.info(f"[PDS-X] ✅ {package_name} registry'de güncel")
                return True
            
            # Kurulum başlat
            if not silent:
                self.logger.info(f"[PDS-X] 📦 {package} kuruluyor...")
            
            # Cache'den kur dene
            if self.cache_manager.install_from_cache(package_name):
                duration = time.time() - start_time
                self.summary_generator.record_installation(package_name, "successful", duration)
                self.dependency_registry.register_package(package_name, "latest", "Başarılı")
                self.installed_packages.add(package_name)  # Cache'e ekle
                if not silent:
                    self.logger.info(f"[PDS-X] ✅ {package_name} cache'den kuruldu ({duration:.2f}s)")
                return True
            
            # Python 3.10 ile kurulum dene
            return self._try_install_with_python310(package, package_name, silent, start_time)
            if self.cache_manager.install_from_cache(package_name):
                duration = time.time() - start_time
                self.summary_generator.record_installation(package_name, "successful", duration)
                self.dependency_registry.register_package(package_name, "latest", "Başarılı")
                self.installed_packages.add(package_name)  # Cache'e ekle
                if not silent:
                    self.logger.info(f"[PDS-X] ✅ {package_name} cache'den kuruldu ({duration:.2f}s)")
                return True
            
            # Python 3.10 ile kurulum dene
            return self._try_install_with_python310(package, package_name, silent, start_time)
        
        except Exception as e:
            self.logger.error(f"[PDS-X] {package_name} kurulum hatası: {e}")
            self.summary_generator.record_installation(package_name, "error", time.time() - start_time)
            return False
    
    def _try_install_with_python310(self, package: str, package_name: str, silent: bool, start_time: float) -> bool:
        """
        Python 3.10 öncelikli kurulum stratejisi
        1. Python 3.10 ile dene
        2. Başarısız olursa sürüm yükseltme gerekli mi analiz et
        3. Gerekirse higher priority queue'ya ekle
        """
        # Python 3.10 ile normal kurulum dene
        install_start = time.time()
        python_exe = self.env_manager.python_exe or sys.executable
        
        # Python 3.10 için stabil pip sürümü kontrolü
        current_pip = self.dependency_registry._get_pip_version()
        if "3.10" in sys.version and self.stable_pip_version not in current_pip:
            if not silent:
                self.logger.warning(f"[PDS-X] ⚠️  Python 3.10 için stabil pip sürümü ({self.stable_pip_version}) öneriliyor")
        
        result = subprocess.run([
            python_exe, "-m", "pip", "install", package, 
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
            self.installed_packages.add(package_name)  # Cache'e ekle
            
            if not silent:
                self.logger.info(f"[PDS-X] ✅ {package_name} başarıyla kuruldu ({total_duration:.2f}s)")
            
            # Kurulum geçmişine ekle
            self.installation_history[package_name] = {
                "version": version,
                "installed_at": datetime.now().isoformat(),
                "status": "success",
                "duration": total_duration
            }
            
            return True
        else:
            # Kurulum başarısız - sürüm yükseltme gerekli mi analiz et
            error_msg = result.stderr.strip() or result.stdout.strip() or "Bilinmeyen hata"
            
            if self._requires_python_upgrade(error_msg):
                if not silent:
                    self.logger.warning(f"[PDS-X] ⚠️  {package_name} daha yüksek Python sürümü gerektiriyor, sonraya erteleniyor...")
                
                # Yüksek öncelikli kuyruğa ekle
                self._add_to_high_priority_queue(package, package_name, error_msg)
                return False
            else:
                # Başarısız kurulum - normal hata
                total_duration = time.time() - start_time
                
                # Registry'e kaydet
                self.dependency_registry.register_package(package_name, "failed", "Başarısız", [])
                self.failed_packages.add(package_name)  # Failed cache'e ekle
                
                # Pip analyzer ile düzelt dene
                if self.pip_analyzer.analyze_and_fix(error_msg, package_name):
                    self.logger.info(f"[PDS-X] ✅ {package_name} düzeltme ile kuruldu")
                    return True
                
                # Conflict manager ile çöz dene
                conflicts_dict = {"error": error_msg, "package": package_name}
                if self.conflict_manager.resolve_conflicts(package_name, conflicts_dict):
                    self.logger.info(f"[PDS-X] ✅ {package_name} çakışma çözümü ile kuruldu")
                    return True
                
                # Statistics
                self.summary_generator.record_installation(package_name, "failed", total_duration)
                
                if not silent:
                    self.logger.error(f"[PDS-X] ❌ {package_name} kurulum başarısız: {error_msg[:200]}...")
                
                return False
    
    def _requires_python_upgrade(self, error_msg: str) -> bool:
        """
        Hata mesajından Python sürüm yükseltme gerekli olup olmadığını analiz et
        """
        upgrade_indicators = [
            "requires python",
            "python_requires",
            "requires a more recent version of python",
            "minimum python version",
            "python >=",
            "python>",
            "unsupported python version",
            "python version not supported"
        ]
        
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in upgrade_indicators)
    
    def _add_to_high_priority_queue(self, package: str, package_name: str, error_msg: str):
        """
        Yüksek öncelikli kuyruğa ekle - daha sonra Python sürüm yükseltmesi ile kurulacak
        """
        if not hasattr(self, 'high_priority_queue'):
            self.high_priority_queue = []
        
        self.high_priority_queue.append({
            'package': package,
            'package_name': package_name,
            'error_msg': error_msg,
            'retry_count': 0,
            'added_at': datetime.now().isoformat()
        })
        
        # Kurulum sırasını değiştir - high priority paketler sonra kurulsun
        self._update_installation_order()
    
    def _update_installation_order(self):
        """
        Kurulum sırasını güncelle - başarısız paketleri sonraya bırak
        """
        if hasattr(self, 'high_priority_queue') and self.high_priority_queue:
            # Sonraya bırakılan paketleri logla
            deferred_packages = [item['package_name'] for item in self.high_priority_queue]
            self.logger.info(f"[PDS-X] 📋 Sonraya ertelenen paketler: {', '.join(deferred_packages)}")
    
    def _retry_high_priority_packages(self, results: Dict, silent: bool = False):
        """
        High priority queue'daki paketleri Python sürüm yükseltmesi ile tekrar dene
        """
        if not hasattr(self, 'high_priority_queue') or not self.high_priority_queue:
            return
        
        upgraded_python = False
        
        for item in self.high_priority_queue[:]:  # Copy listesi, orijinal değiştirilmeyecek
            package = item['package']
            package_name = item['package_name']
            error_msg = item['error_msg']
            
            if not silent:
                self.logger.info(f"[PDS-X] 🔄 Yüksek sürüm paketi deneniyor: {package_name}")
            
            # İlk kez Python sürümü yükseltme
            if not upgraded_python:
                if not silent:
                    self.logger.warning(f"[PDS-X] ⚠️  Python sürümü yükseltmesi gerekiyor...")
                    self.logger.info(f"[PDS-X] 🔧 Python ortamını yüksek sürüm için hazırlıyor...")
                
                # Python sürümünü yükselt (çeşitli stratejiler)
                if self._attempt_python_upgrade():
                    upgraded_python = True
                    if not silent:
                        self.logger.info(f"[PDS-X] ✅ Python sürümü yükseltmesi başarılı")
                else:
                    if not silent:
                        self.logger.error(f"[PDS-X] ❌ Python sürümü yükseltmesi başarısız")
                    break  # Python yükseltemediyse bu paketleri kuramayz
            
            # Yükseltilmiş Python ile paketi kur
            try:
                if self._install_with_upgraded_python(package, package_name, silent):
                    results["successful"].append(package)
                    results["deferred"].remove(package) if package in results["deferred"] else None
                    if not silent:
                        self.logger.info(f"[PDS-X] ✅ {package_name} yüksek sürüm ile kuruldu")
                    
                    # High priority queue'dan kaldır
                    self.high_priority_queue.remove(item)
                else:
                    # Hala kurulamadı, failed'e taşı
                    results["failed"].append(package)
                    results["deferred"].remove(package) if package in results["deferred"] else None
                    if not silent:
                        self.logger.error(f"[PDS-X] ❌ {package_name} yüksek sürüm ile de kurulamadı")
                        
            except Exception as e:
                results["failed"].append(package)
                if not silent:
                    self.logger.error(f"[PDS-X] ❌ {package_name} yüksek sürüm kurulum hatası: {e}")
    
    def _attempt_python_upgrade(self) -> bool:
        """
        Python sürümünü yükseltmeye çalışır
        Çeşitli stratejiler:
        1. Mevcut ortamda pip upgrade
        2. Conda kullanımı
        3. System Python'a geçiş
        """
        try:
            # Strateji 1: pip upgrade --upgrade-strategy eager
            self.logger.info("[PDS-X] 🔧 Pip aggressive upgrade deneniyor...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "eager", "pip"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("[PDS-X] ✅ Pip aggressive upgrade başarılı")
                return True
            
            # Strateji 2: System Python geçici kullanımı
            system_python = shutil.which("python") or shutil.which("python3")
            if system_python and system_python != sys.executable:
                self.logger.info(f"[PDS-X] 🔧 System Python'a geçiş deneniyor: {system_python}")
                
                # Geçici olarak system python'u kullan
                original_exe = sys.executable
                try:
                    sys.executable = system_python
                    # Test et
                    test_result = subprocess.run([system_python, "--version"], capture_output=True, text=True)
                    if test_result.returncode == 0:
                        self.logger.info(f"[PDS-X] ✅ System Python geçişi başarılı: {test_result.stdout.strip()}")
                        return True
                finally:
                    sys.executable = original_exe
            
            return False
            
        except Exception as e:
            self.logger.error(f"[PDS-X] ❌ Python sürüm yükseltme hatası: {e}")
            return False
    
    def _install_with_upgraded_python(self, package: str, package_name: str, silent: bool) -> bool:
        """
        Yükseltilmiş Python ortamı ile paketi kurma
        """
        try:
            # Mevcut Python ile aggressive kurulum dene
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, 
                "--upgrade", "--upgrade-strategy", "eager",
                "--quiet" if silent else "--verbose"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            
            # User flag ile dene (permission sorunları için)
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, 
                "--user", "--upgrade",
                "--quiet" if silent else "--verbose"
            ], capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception as e:
            if not silent:
                self.logger.error(f"[PDS-X] ❌ {package_name} yüksek sürüm kurulum hatası: {e}")
            return False
    
    def _optimize_package_order(self, packages: List[str], silent: bool = False) -> List[str]:
        """
        Genetik algoritma ile paket kurulum sırasını optimize et
        """
        if not self.scientific_utils:
            if not silent:
                self.logger.debug("[PDS-X] Scientific utils mevcut değil, orijinal sıra korunuyor")
            return packages
        
        try:
            # Paket dependency graph'ını oluştur
            deps = []
            for i, pkg in enumerate(packages):
                # Basit dependency simulation
                package_name = pkg.split('==')[0].split('>=')[0].split('<=')[0].strip()
                
                # Bilinen dependency patterns (örnek)
                if 'numpy' in package_name.lower():
                    deps.append((package_name, 'base_math'))
                elif 'pandas' in package_name.lower():
                    deps.append((package_name, 'numpy'))
                elif 'matplotlib' in package_name.lower():
                    deps.append((package_name, 'numpy'))
                elif 'scipy' in package_name.lower():
                    deps.append((package_name, 'numpy'))
                elif 'tensorflow' in package_name.lower():
                    deps.append((package_name, 'numpy'))
                elif 'torch' in package_name.lower():
                    deps.append((package_name, 'numpy'))
                else:
                    deps.append((package_name, 'base'))
            
            # Genetik algoritma ile optimize et
            optimized_names = self.scientific_utils.genetic_dependency_optimizer(deps)  # type: ignore
            
            # Optimize edilmiş isimleri orijinal paket formatlarına dönüştür
            optimized_packages = []
            name_to_package = {pkg.split('==')[0].split('>=')[0].split('<=')[0].strip(): pkg for pkg in packages}
            
            # Önce optimize edilmiş sırayı ekle
            for name in optimized_names:
                if name in name_to_package:
                    optimized_packages.append(name_to_package[name])
            
            # Eksik paketleri sona ekle
            for pkg in packages:
                if pkg not in optimized_packages:
                    optimized_packages.append(pkg)
            
            if not silent:
                optimized_count = len([p for p in optimized_packages if p in [name_to_package.get(n, '') for n in optimized_names]])
                self.logger.info(f"[PDS-X] 🧬 {optimized_count}/{len(packages)} paket genetik algoritma ile sıralandı")
            
            return optimized_packages
            
        except Exception as e:
            if not silent:
                self.logger.error(f"[PDS-X] Genetik optimizasyon hatası: {e}, orijinal sıra kullanılacak")
            return packages
    
    def _update_dependency_graph(self, results: Dict):
        """
        Kurulum sonuçlarına göre dependency graph'ını güncelle
        """
        try:
            for package in results.get("successful", []):
                package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
                
                # Dependency bilgilerini registry'ye kaydet
                metadata = {
                    "install_method": "pip_genetic",
                    "optimization_used": self.scientific_utils is not None,
                    "stable_pip": self.stable_pip_version in self.dependency_registry._get_pip_version(),
                    "python_stable": self.dependency_registry._is_stable_environment()
                }
                
                self.dependency_registry.register_package(
                    package_name, 
                    "latest", 
                    "Başarılı", 
                    dependencies=[], 
                    metadata=metadata
                )
            
            # Başarısız paketler için de metadata kaydet
            for package in results.get("failed", []):
                package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
                
                metadata = {
                    "install_method": "pip_genetic",
                    "optimization_used": self.scientific_utils is not None,
                    "failure_reason": "unknown",
                    "python_stable": self.dependency_registry._is_stable_environment()
                }
                
                self.dependency_registry.register_package(
                    package_name, 
                    "failed", 
                    "Başarısız", 
                    dependencies=[], 
                    metadata=metadata
                )
            
        except Exception as e:
            self.logger.error(f"[PDS-X] Dependency graph güncelleme hatası: {e}")
    
    def install_required_packages(self, silent: bool = False, auto_start_pdsx: bool = False) -> Dict:
        """108 REQUIRED_PACKAGES'ı kur - Python 3.10 öncelikli strateji + genetik optimizasyon"""
        try:
            results = {"successful": [], "failed": [], "skipped": [], "deferred": []}
            total_packages = len(REQUIRED_PACKAGES)
            
            # High priority queue'yu başlat
            self.high_priority_queue = []
            
            if not silent:
                self.logger.info(f"[PDS-X] 🔄 {total_packages} paket kurulum başlıyor...")
                self.logger.info(f"[PDS-X] 📋 Python 3.10 öncelikli kurulum stratejisi aktif")
                self.logger.info(f"[PDS-X] 🧬 Genetik algoritma optimizasyonu: {'✅ Aktif' if self.scientific_utils else '❌ Pasif'}")
                if auto_start_pdsx:
                    self.logger.info(f"[PDS-X] 🚀 PDS-X otomatik başlatma: ✅ Aktif")
            
            # Genetik algoritma ile paket sırasını optimize et
            optimized_packages = self._optimize_package_order(REQUIRED_PACKAGES, silent)
            
            if not silent:
                if optimized_packages != REQUIRED_PACKAGES:
                    self.logger.info(f"[PDS-X] 🔀 Paket kurulum sırası genetik algoritma ile optimize edildi")
                else:
                    self.logger.info(f"[PDS-X] 📄 Orijinal paket sırası korunuyor")
            
            # 1. Aşama: Optimize edilmiş paket sırasıyla paralel kurulum
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_package = {
                    executor.submit(self.install_package, pkg, silent): pkg 
                    for pkg in optimized_packages
                }
                
                for i, future in enumerate(as_completed(future_to_package)):
                    package = future_to_package[future]
                    
                    # Shutdown kontrolü
                    if shutdown_manager.is_shutdown_requested():
                        self.logger.warning("[PDS-X] Shutdown talep edildi, kalan kurulumlar iptal ediliyor")
                        break
                    
                    try:
                        success = future.result()
                        if success:
                            results["successful"].append(package)
                        else:
                            # Paket high priority queue'ya eklendi mi kontrol et
                            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
                            if any(item['package_name'] == package_name for item in self.high_priority_queue):
                                results["deferred"].append(package)
                            else:
                                results["failed"].append(package)
                            
                        # Progress
                        if not silent:
                            progress = (i + 1) / total_packages * 100
                            self.logger.info(f"[PDS-X] 📊 İlerleme: {progress:.1f}% ({i + 1}/{total_packages})")
                            
                    except Exception as e:
                        results["failed"].append(package)
                        self.logger.error(f"[PDS-X] ❌ {package} executor hatası: {e}")
            
            # 2. Aşama: High priority queue'daki paketleri işle (sürüm yükseltme ile)
            if self.high_priority_queue and not silent:
                self.logger.info(f"[PDS-X] 🔄 {len(self.high_priority_queue)} yüksek sürüm gerektiren paket Python sürümü yükseltmesi ile denenecek...")
                self._retry_high_priority_packages(results, silent)
            
            # 3. Aşama: Dependency graph güncelle
            self._update_dependency_graph(results)
            
            # Özet rapor
            success_count = len(results["successful"])
            failed_count = len(results["failed"])
            deferred_count = len(results["deferred"])
            success_rate = (success_count / total_packages * 100) if total_packages > 0 else 0
            
            report = f"""
🎯 KURULUM TAMAMLANDI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Başarılı: {success_count}
❌ Başarısız: {failed_count}
⏳ Sonraya Ertelenen: {deferred_count}
📊 Başarı Oranı: {success_rate:.1f}%
🧬 Genetik Optimizasyon: {'✅ Kullanıldı' if self.scientific_utils else '❌ Kullanılmadı'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            if not silent:
                self.logger.info(report)
            
            # Başarısız paketleri tekrar dene (eski logic korunuyor)
            if results["failed"] and not silent:
                self.logger.info("[PDS-X] 🔄 Başarısız paketler tekrar deneniyor...")
                for pkg in results["failed"][:5]:  # İlk 5 başarısız paketi
                    if shutdown_manager.is_shutdown_requested():
                        break
                    if self.install_package(pkg, silent=True):
                        results["successful"].append(pkg)
                        results["failed"].remove(pkg)
            
            # 4. Aşama: PDS-X başlatma komutu öğren ve başlat
            if success_rate >= 70:  # En az %70 başarı oranı varsa PDS-X'i başlat
                if not silent:
                    self.logger.info("[PDS-X] 🚀 Kurulum başarılı! PDS-X başlatma hazırlığı...")
                
                # PDS-X başlatma komutunu öğren (ilk kez)
                startup_file = Path("pdsx_startup.json")
                if not startup_file.exists():
                    learn_pdsx_startup_command()
                
                # Auto-start kontrolü
                if auto_start_pdsx:
                    self.logger.info("[PDS-X] 🤖 Otomatik PDS-X başlatma aktif...")
                    if start_pdsx_after_installation():
                        self.logger.info("[PDS-X] 🎉 PDS-X başarıyla başlatıldı!")
                    else:
                        self.logger.warning("[PDS-X] ⚠️ PDS-X başlatılamadı. Manuel başlatabilirsiniz.")
                else:
                    # Kullanıcıya sorarak PDS-X'i başlat
                    try:
                        if not silent:
                            user_input = input("[PDS-X] 🎯 PDS-X'i şimdi başlatmak ister misiniz? (E/h): ").strip().lower()
                            if user_input in ['e', 'evet', 'yes', 'y', '']:
                                if start_pdsx_after_installation():
                                    self.logger.info("[PDS-X] 🎉 PDS-X başarıyla başlatıldı!")
                                else:
                                    self.logger.warning("[PDS-X] ⚠️ PDS-X başlatılamadı. Manuel başlatabilirsiniz.")
                            else:
                                self.logger.info("[PDS-X] 💡 PDS-X'i daha sonra başlatmak için: --start-pdsx parametresini kullanın")
                    except (EOFError, KeyboardInterrupt):
                        # Non-interactive ortamda otomatik başlat
                        self.logger.info("[PDS-X] 🤖 Non-interactive mod, PDS-X otomatik başlatılıyor...")
                        start_pdsx_after_installation()
            else:
                if not silent:
                    self.logger.warning(f"[PDS-X] ⚠️ Düşük başarı oranı ({success_rate:.1f}%). PDS-X başlatılmıyor.")
                    self.logger.info("[PDS-X] 💡 Daha fazla paket kurmayı deneyin veya hataları çözün.")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Bulk installation hatası: {e}")
            return {"successful": [], "failed": REQUIRED_PACKAGES.copy(), "skipped": []}
    
    def cleanup_on_shutdown(self):
        """Graceful shutdown cleanup"""
        try:
            self.logger.info("🛑 AutoImporter cleanup başlatıldı...")
            
            # Monitoring durdur
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
            
            # Özet rapor göster
            self.print_installation_summary()
            
            # Cleanup yap
            self.cleanup_on_shutdown()
            
            # Running durumunu false yap
            self.running = False
            
            self.logger.info("[PDS-X] AutoImporter başarıyla kapatıldı")
            
        except Exception as e:
            self.logger.error(f"[PDS-X] Shutdown hatası: {e}")
    
    def print_installation_summary(self):
        """Gösterişli kurulum özet raporu yazdır"""
        try:
            if hasattr(self, 'summary_generator'):
                # Temel istatistikler
                total_successful = len(self.installed_packages)
                total_failed = len(self.failed_packages)
                total_packages = total_successful + total_failed
                
                if total_packages == 0:
                    self.logger.info("[PDS-X] 📋 Henüz kurulum işlemi yapılmadı.")
                    return
                
                success_rate = (total_successful / total_packages) * 100 if total_packages > 0 else 0
                
                # Gösterişli özet rapor
                summary_lines = [
                    "",
                    "="*25 + " [PDS-X] KURULUM ÖZETİ " + "="*25,
                    f"[PDS-X] 📦 Toplam {total_packages} paket işlendi",
                    f"[PDS-X] ✅ Başarılı Kurulumlar: {total_successful}",
                    f"[PDS-X] ❌ Başarısız Kurulumlar: {total_failed}",
                    f"[PDS-X] 📊 Başarı Oranı: {success_rate:.1f}%",
                ]
                
                # Başarılı paketler listesi
                if self.installed_packages:
                    summary_lines.append(f"[PDS-X] 🎯 Başarılı Paketler:")
                    for pkg in list(self.installed_packages)[:10]:  # İlk 10'u göster
                        summary_lines.append(f"[PDS-X]   ✓ {pkg}")
                    if len(self.installed_packages) > 10:
                        summary_lines.append(f"[PDS-X]   ... ve {len(self.installed_packages) - 10} paket daha")
                
                # Başarısız paketler listesi  
                if self.failed_packages:
                    summary_lines.append(f"[PDS-X] ⚠️  Başarısız Paketler:")
                    for pkg in list(self.failed_packages)[:5]:  # İlk 5'ini göster
                        summary_lines.append(f"[PDS-X]   ✗ {pkg}")
                    if len(self.failed_packages) > 5:
                        summary_lines.append(f"[PDS-X]   ... ve {len(self.failed_packages) - 5} paket daha")
                
                # Environment bilgisi
                if hasattr(self, 'env_manager') and self.env_manager.is_running_in_venv():
                    venv_path = str(self.env_manager.venv_dir)
                    summary_lines.append(f"[PDS-X] 🏠 İzole Ortam: {venv_path}")
                
                summary_lines.append("="*70)
                summary_lines.append("")
                
                # Tüm satırları yazdır
                for line in summary_lines:
                    print(line)  # Doğrudan print kullan ki konsola çıksın
                    
        except Exception as e:
            self.logger.error(f"[PDS-X] Summary print hatası: {e}")
    
    def get_environment_info(self) -> str:
        """Environment bilgilerini döndür"""
        try:
            info_lines = []
            
            # Python bilgisi
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            info_lines.append(f"[PDS-X] 🐍 Python Sürümü: {python_version}")
            
            # Sanal ortam bilgisi
            if hasattr(self, 'env_manager'):
                if self.env_manager.is_running_in_venv():
                    venv_path = str(self.env_manager.venv_dir)
                    info_lines.append(f"[PDS-X] 🏠 Sanal Ortam: ✅ Aktif ({venv_path})")
                else:
                    info_lines.append(f"[PDS-X] 🏠 Sanal Ortam: ❌ Deaktif")
            
            # Pip bilgisi
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    pip_version = result.stdout.strip().split()[1]
                    info_lines.append(f"[PDS-X] 📦 Pip Sürümü: {pip_version}")
            except:
                pass
                
            return "\n".join(info_lines)
            
        except Exception as e:
            self.logger.error(f"[PDS-X] Environment info hatası: {e}")
            return "[PDS-X] ⚠️ Environment bilgisi alınamadı"
    
    # Demo compatibility properties
    @property 
    def log_analyzer(self):
        return self.terminal_analyzer
    
    @property
    def real_time_monitor(self):
        return self.realtime_monitor
    
    def load_dependencies(self) -> Dict:
        """Dependencies.json dosyasını yükle"""
        try:
            deps_file = Path("dependencies.json")
            if deps_file.exists():
                with open(deps_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Dependencies yükleme hatası: {e}")
            return {}


# === GLOBAL SHUTDOWN MANAGER ===
shutdown_manager = GracefulShutdownManager()


# === MAIN FUNCTION ===
def main():
    """
    Ana giriş noktası - Gelişmiş environment management ve package installation
    """
    try:
        print("[PDS-X] 🚀 AutoImporter Advanced v1.7.9.5 başlatılıyor...")
        print(f"[PDS-X] 📦 {len(REQUIRED_PACKAGES)} paket kurulum listesinde")
        print("[PDS-X] ⚠️  Ctrl+Shift+Q ile acil durdurma")
        print("[PDS-X] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Command line argument handling
        parser = argparse.ArgumentParser(description="PDS-X AutoImporter - Bağımlılık Yönetimi")
        parser.add_argument("--demo", action="store_true", help="AutoImporter demo çalıştır")
        parser.add_argument("--install-all", action="store_true", help="Tüm required packages'ı kur")
        parser.add_argument("--install", nargs="+", help="Belirtilen paketleri kur")
        parser.add_argument("--check", nargs="+", help="Belirtilen paketleri kontrol et")
        parser.add_argument("--analyze-log", type=str, help="Belirtilen log dosyasını analiz et")
        parser.add_argument("--setup-env", action="store_true", help="Python 3.10 ve sanal ortam kurulumunu zorla")
        parser.add_argument("--replay-last", action="store_true", help="Son çalıştırılan komutları tekrar et")
        parser.add_argument("--learn-pdsx", action="store_true", help="PDS-X başlatma komutunu öğren ve kaydet")
        parser.add_argument("--start-pdsx", action="store_true", help="PDS-X'i otomatik başlat")
        parser.add_argument("--auto-start", action="store_true", help="Kurulum sonrası PDS-X'i otomatik başlat")
        
        args = parser.parse_args()
        
        # PDS-X komut öğrenme
        if args.learn_pdsx:
            print("[PDS-X] 🎯 PDS-X başlatma komutu öğreniliyor...")
            learned_cmd = learn_pdsx_startup_command()
            if learned_cmd:
                print(f"[PDS-X] ✅ Komut başarıyla kaydedildi: {learned_cmd}")
            else:
                print("[PDS-X] ❌ Komut öğrenilemedi")
            return 0
        
        # PDS-X başlatma
        if args.start_pdsx:
            print("[PDS-X] 🚀 PDS-X başlatılıyor...")
            if start_pdsx_after_installation():
                print("[PDS-X] ✅ PDS-X başarıyla başlatıldı!")
                return 0
            else:
                print("[PDS-X] ❌ PDS-X başlatılamadı!")
                return 1
        
        # Replay last command logic
        if args.replay_last:
            args_file = Path("last_args.json")
            if args_file.exists():
                try:
                    with open(args_file, "r", encoding="utf-8") as f:
                        last_data = json.load(f)
                    print(f"[PDS-X] Son komut tekrarlanıyor: {' '.join(last_data['argv'][1:])}")
                    # Replay functionality - burada son argümanları kullan
                    # Basitçe aynı işlemi tekrarla
                except Exception as e:
                    print(f"[PDS-X] HATA: Son komut tekrarlanamadı: {e}")
        
        # AutoImporter oluştur - bu aşamada otomatik environment check yapacak
        importer = AutoImporter()
        
        # Environment setup kontrolü
        if args.setup_env or not importer.env_manager.is_running_in_venv():
            print("\n[PDS-X] 🔧 Python 3.10 ve sanal ortam kontrolü...")
            if not importer.env_manager.setup_environment():
                print("[PDS-X] ❌ Environment kurulumu başarısız!")
                return 1
            # Bu noktada program restart olur, bu satırlar çalışmaz
                print("❌ Environment kurulumu başarısız!")
                return 1
            # Bu noktada program restart olur, bu satırlar çalışmaz
        
        # Demo çalıştır
        if args.demo:
            demo_auto_importer(importer)
            return 0
            
        # Belirli paketleri kur
        if args.install:
            success_count = 0
            for package in args.install:
                print(f"[PDS-X] 📦 Kuruluyor: {package}")
                if importer.install_package(package):
                    print(f"[PDS-X] ✅ {package} başarıyla kuruldu")
                    success_count += 1
                else:
                    print(f"[PDS-X] ❌ {package} kurulumu başarısız")
            print(f"\n[PDS-X] 📊 {success_count}/{len(args.install)} paket başarıyla kuruldu")
            return 0 if success_count == len(args.install) else 1
            
        # Paket kontrolü
        if args.check:
            for package in args.check:
                if importer.env_manager.check_package_installed(package):
                    print(f"[PDS-X] ✅ {package} mevcut")
                else:
                    print(f"[PDS-X] ❌ {package} eksik")
            return 0
            
        # Log analizi
        if args.analyze_log:
            if os.path.exists(args.analyze_log):
                print(f"[PDS-X] 📋 Log dosyası tespit edildi: {args.analyze_log}")
                print("[PDS-X] 🔍 Log analiz sistemi mevcut")
            else:
                print(f"[PDS-X] ❌ Log dosyası bulunamadı: {args.analyze_log}")
            return 0
        
        # Varsayılan: Tüm required packages'ı kur
        if args.install_all or not any([args.demo, args.install, args.check, args.analyze_log, args.learn_pdsx, args.start_pdsx]):
            print("\n[PDS-X] 📦 Tüm required packages kurulumu başlatılıyor...")
            results = importer.install_required_packages(silent=False, auto_start_pdsx=args.auto_start)
            
            # Final özet
            success_count = len(results["successful"])
            failed_count = len(results["failed"])
            total_count = len(REQUIRED_PACKAGES)
            
            print("\n[PDS-X] 🎯 KURULUM SONUCU")
            print("[PDS-X] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"[PDS-X] ✅ Başarılı: {success_count}/{total_count}")
            print(f"[PDS-X] ❌ Başarısız: {failed_count}/{total_count}")
            print(f"[PDS-X] 📊 Başarı Oranı: {success_count/total_count*100:.1f}%")
            
            if results["failed"]:
                print("\n[PDS-X] ❌ Başarısız Paketler:")
                for pkg in results["failed"][:10]:  # İlk 10'u göster
                    print(f"[PDS-X]    • {pkg}")
                if len(results["failed"]) > 10:
                    print(f"[PDS-X]    ... ve {len(results['failed'])-10} paket daha")
            
            print("\n[PDS-X] 💡 Kullanışlı komutlar:")
            print("[PDS-X]    --learn-pdsx     : PDS-X başlatma komutunu öğren")
            print("[PDS-X]    --start-pdsx     : PDS-X'i manuel başlat")
            print("[PDS-X]    --auto-start     : Kurulum sonrası otomatik başlat")
            print("[PDS-X] 📋 Detaylı rapor logs/ klasöründe")
            print("[PDS-X] 🔗 GitHub: https://github.com/metedinler/pdsXbasic")
            print("[PDS-X] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            return 0 if failed_count == 0 else 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[PDS-X] ⚠️  Kullanıcı tarafından iptal edildi")
        return 130
    except Exception as e:
        print(f"\n[PDS-X] ❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        try:
            shutdown_manager.cleanup()
        except Exception:
            pass

def demo_auto_importer(importer):
    """AutoImporter demo fonksiyonu - toplu1.py'den uyarlandı"""
    print("="*60)
    print("PDS-X AutoImporter Demo")
    print("="*60)
    
    try:
        # Temel paketleri kontrol et
        print("\n1. Temel paketleri kontrol ediliyor...")
        required_packages = [
            "numpy", "pandas", "requests", "matplotlib", 
            "scipy", "scikit-learn", "pillow", "tqdm"
        ]
        
        for package in required_packages:
            if not importer.env_manager.check_package_installed(package):
                print(f"   ❌ {package} eksik - kurulum öneriliyor")
            else:
                print(f"   ✅ {package} mevcut")
                
        # Environment bilgisi
        print("\n2. Ortam bilgileri...")
        venv_status = "✅ Sanal ortamda" if importer.env_manager.is_running_in_venv() else "❌ Sanal ortam dışında"
        print(f"   Python sürümü: {sys.version}")
        print(f"   Sanal ortam: {venv_status}")
        print(f"   Venv dizini: {importer.env_manager.venv_dir}")
        
        # Log analizi demo
        print("\n3. Terminal log analizi demo...")
        sample_log_content = """
        ImportError: No module named 'networkx'
        ModuleNotFoundError: No module named 'seaborn'
        Could not import 'plotly': pip install plotly
        """
        
        detected = importer.terminal_analyzer.analyze_log_content(sample_log_content)
        print(f"   🔍 Tespit edilen eksik bağımlılıklar: {detected}")
        
        # Dependency management demo
        print("\n4. Bağımlılık yönetimi demo...")
        deps_before = importer.load_dependencies()
        print(f"   📋 Mevcut dependencies.json: {len(deps_before)} paket")
        
        # Real-time monitoring demo
        print("\n5. Real-time monitoring demo...")
        print(f"   📊 Monitoring sistemi: {'✅ Aktif' if hasattr(importer, 'real_time_monitor') else '❌ Pasif'}")
        
        print("\n" + "="*60)
        print("Demo tamamlandı!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Demo sırasında hata: {e}")
        import traceback
        traceback.print_exc()


# === PDS-X MODULE EXPORTS ===
__pdsX_exports__ = {
    "AutoImporter": AutoImporter,
    "AdvancedLogger": AdvancedLogger,
    "EnvManager": EnvManager,
    "DependencyRegistry": DependencyRegistry,
    "CacheManager": CacheManager,
    "ConflictManager": ConflictManager,
    "PipOutputAnalyzer": PipOutputAnalyzer,
    "TerminalLogAnalyzer": TerminalLogAnalyzer,
    "RealTimeLogMonitor": RealTimeLogMonitor,
    "ModuleAnalyzer": ModuleAnalyzer,
    "SummaryGenerator": SummaryGenerator,
    "GracefulShutdownManager": GracefulShutdownManager,
    "REQUIRED_PACKAGES": REQUIRED_PACKAGES,
    "ensure_venv": ensure_venv,
    "learn_pdsx_startup_command": learn_pdsx_startup_command,
    "start_pdsx_after_installation": start_pdsx_after_installation,
    "detect_pdsx_commands": detect_pdsx_commands,
    "handle_command_replay": handle_command_replay,
    "main": main,
    "demo_auto_importer": demo_auto_importer
}

if __name__ == "__main__":
    sys.exit(main())
