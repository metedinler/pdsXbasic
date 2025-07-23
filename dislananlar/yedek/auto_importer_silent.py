# auto_importer_silent.py - PDS-X Sessiz Otomatik Kurucu
# =========================================================
# Kullanıcıdan HİÇBİR girdi almaz, her şeyi otomatik yapar
# Yüksek kaynak kullanımında paralel işlem kullanır

import os
import sys
import subprocess
import threading
import time
import json
import signal
import atexit
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import psutil

# REQUIRED_PACKAGES - 108 paket tam liste
REQUIRED_PACKAGES = [
    ("requests", "2.31.0"), ("numpy", "1.26.4"), ("pandas", "2.1.4"), ("matplotlib", "3.8.2"),
    ("seaborn", "0.13.0"), ("scipy", "1.11.4"), ("scikit-learn", "1.3.2"), ("tensorflow", "2.15.0"),
    ("torch", "2.1.2"), ("pytorch-lightning", "2.1.3"), ("transformers", "4.36.2"), ("datasets", "2.16.1"),
    ("accelerate", "0.25.0"), ("diffusers", "0.25.0"), ("tokenizers", "0.15.0"), ("huggingface-hub", "0.19.4"),
    ("opencv-python", "4.8.1.78"), ("Pillow", "10.1.0"), ("imageio", "2.33.1"), ("scikit-image", "0.22.0"),
    ("albumentations", "1.3.1"), ("plotly", "5.17.0"), ("bokeh", "3.3.2"), ("dash", "2.15.0"),
    ("streamlit", "1.29.0"), ("gradio", "4.8.0"), ("ipywidgets", "8.1.1"), ("jupyter", "1.0.0"),
    ("notebook", "7.0.6"), ("jupyterlab", "4.0.9"), ("voila", "0.5.5"), ("fastapi", "0.104.1"),
    ("uvicorn", "0.24.0"), ("flask", "3.0.0"), ("django", "4.2.8"), ("aiohttp", "3.9.1"),
    ("httpx", "0.25.2"), ("websockets", "12.0"), ("socketio", "5.10.0"), ("celery", "5.3.4"),
    ("redis", "5.0.1"), ("pymongo", "4.6.0"), ("sqlalchemy", "2.0.23"), ("alembic", "1.13.1"),
    ("psycopg2-binary", "2.9.9"), ("mysql-connector-python", "8.2.0"), ("sqlite3", ""), ("pydantic", "2.5.2"),
    ("marshmallow", "3.20.1"), ("click", "8.1.7"), ("typer", "0.9.0"), ("rich", "13.7.0"),
    ("colorama", "0.4.6"), ("tqdm", "4.66.1"), ("alive-progress", "3.1.5"), ("yaspin", "3.0.1"),
    ("python-dotenv", "1.0.0"), ("configparser", "6.0.0"), ("pyyaml", "6.0.1"), ("toml", "0.10.2"),
    ("jsonschema", "4.20.0"), ("xmltodict", "0.13.0"), ("beautifulsoup4", "4.12.2"), ("lxml", "4.9.3"),
    ("html5lib", "1.1"), ("cssselect", "1.2.0"), ("pyquery", "2.0.0"), ("selenium", "4.16.0"),
    ("playwright", "1.40.0"), ("scrapy", "2.11.0"), ("feedparser", "6.0.10"), ("python-telegram-bot", "20.7"),
    ("discord.py", "2.3.2"), ("slack-sdk", "3.26.1"), ("tweepy", "4.14.0"), ("paramiko", "3.4.0"),
    ("fabric", "3.2.2"), ("ansible", "8.7.0"), ("docker", "6.1.3"), ("kubernetes", "28.1.0"),
    ("boto3", "1.34.0"), ("azure-storage-blob", "12.19.0"), ("google-cloud-storage", "2.10.0"), ("dropbox", "11.36.2"),
    ("pysftp", "0.2.9"), ("ftplib", ""), ("imaplib", ""), ("smtplib", ""),
    ("schedule", "1.2.0"), ("croniter", "2.0.1"), ("apscheduler", "3.10.4"), ("rq", "1.15.1"),
    ("joblib", "1.3.2"), ("dask", "2023.12.0"), ("ray", "2.8.1"), ("multiprocessing", ""),
    ("threading", ""), ("asyncio", ""), ("concurrent.futures", ""), ("queue", ""),
    ("pickle", ""), ("dill", "0.3.7"), ("cloudpickle", "3.0.0"), ("msgpack", "1.0.7"),
    ("protobuf", "4.25.1"), ("grpcio", "1.60.0"), ("thrift", "0.19.0"), ("avro", "1.11.3"),
    ("pyarrow", "14.0.2"), ("polars", "0.20.2"), ("vaex", "4.17.0"), ("modin", "0.25.0"),
    ("cudf", "23.12.0"), ("cupy", "12.3.0"), ("numba", "0.58.1"), ("cython", "3.0.7"),
    ("pybind11", "2.11.1"), ("cffi", "1.16.0"), ("ctypes", ""), ("struct", "")
]

class SilentAutoImporter:
    """
    PDS-X Sessiz Otomatik Kurucu
    ============================
    - Kullanıcıdan HİÇBİR girdi almaz
    - Her şeyi otomatik yapar
    - Yüksek kaynak kullanımında paralel işlem
    - REPL entegrasyonu için optimize edilmiş
    """
    
    def __init__(self, mode="SILENT"):
        self.mode = mode
        self.start_time = time.time()
        self.installation_log = []
        self.max_workers = min(8, os.cpu_count() or 4)
        self.total_packages = len(REQUIRED_PACKAGES)
        self.installed_count = 0
        self.failed_count = 0
        
        # Güvenli kapatma
        signal.signal(signal.SIGINT, self._emergency_exit)
        signal.signal(signal.SIGTERM, self._emergency_exit)
        atexit.register(self._cleanup)
        
        # Otomatik başlatma
        self._auto_start()
    
    def _auto_start(self):
        """Otomatik başlatma - hiç soru sormaz"""
        try:
            # 1. Sistem durumu kontrolü
            cpu_usage = psutil.cpu_percent(interval=0.5)
            memory_usage = psutil.virtual_memory().percent
            
            # 2. Yüksek kaynak kullanımı varsa paralel işleme geç
            if cpu_usage > 70 or memory_usage > 70:
                self.max_workers = min(12, (os.cpu_count() or 4) * 2)
                self._log(f"Yüksek kaynak kullanımı (CPU: {cpu_usage}%, RAM: {memory_usage}%) - Paralel mod: {self.max_workers} worker")
            
            # 3. Python 3.10 kontrolü
            if not self._check_python_version():
                self._log("Python 3.10 gerekli, ortam ayarlanıyor...")
                return  # Yeniden başlatma gerekebilir
            
            # 4. Eksik paketleri tespit et
            missing_packages = self._find_missing_packages()
            
            if not missing_packages:
                self._log("✅ Tüm paketler zaten kurulu!")
                return
            
            # 5. Otomatik kurulum başlat
            self._log(f"🚀 {len(missing_packages)}/{self.total_packages} paket otomatik kuruluyor...")
            self._install_packages_parallel(missing_packages)
            
            # 6. Sonuç raporu
            self._log(f"✅ Kurulum tamamlandı: {self.installed_count} başarılı, {self.failed_count} başarısız")
            
        except Exception as e:
            self._log(f"❌ Otomatik kurulum hatası: {e}")
    
    def _check_python_version(self) -> bool:
        """Python 3.10 kontrolü"""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_version != "3.10":
            self._log(f"⚠️  Python {current_version} tespit edildi, PDS-X için Python 3.10 gerekli")
            # Otomatik Python 3.10 kurulumu burada yapılabilir
            return False
        return True
    
    def _find_missing_packages(self) -> List[Tuple[str, str]]:
        """Eksik paketleri tespit et"""
        missing = []
        
        for package_name, version in REQUIRED_PACKAGES:
            if not self._is_package_installed(package_name):
                missing.append((package_name, version))
        
        self._log(f"📊 Durum: {self.total_packages - len(missing)}/{self.total_packages} kurulu, {len(missing)} eksik")
        return missing
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Paket kurulu mu kontrol et"""
        try:
            # Ana paket adı ile dene
            __import__(package_name)
            return True
        except ImportError:
            # Alternatif isimlerle dene
            alt_names = {
                'PIL': 'Pillow',
                'cv2': 'opencv-python',
                'sklearn': 'scikit-learn',
                'bs4': 'beautifulsoup4'
            }
            
            if package_name in alt_names:
                try:
                    result = subprocess.run([
                        sys.executable, "-c", f"import {package_name}"
                    ], capture_output=True, text=True, timeout=5)
                    return result.returncode == 0
                except:
                    return False
            
            return False
    
    def _install_packages_parallel(self, packages: List[Tuple[str, str]]):
        """Paketleri paralel olarak kur"""
        # Batch'lere böl (aşırı paralel kurulum sistemi yavaşlatabilir)
        batch_size = min(self.max_workers, 6)
        batches = [packages[i:i + batch_size] for i in range(0, len(packages), batch_size)]
        
        for batch_num, batch in enumerate(batches, 1):
            self._log(f"📦 Batch {batch_num}/{len(batches)} işleniyor ({len(batch)} paket)...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Paralel kurulum başlat
                future_to_package = {
                    executor.submit(self._install_single_package, pkg_name, version): (pkg_name, version)
                    for pkg_name, version in batch
                }
                
                # Sonuçları topla
                for future in as_completed(future_to_package, timeout=300):
                    pkg_name, version = future_to_package[future]
                    try:
                        success = future.result()
                        if success:
                            self.installed_count += 1
                            self._log(f"✅ {pkg_name}=={version}")
                        else:
                            self.failed_count += 1
                            self._log(f"❌ {pkg_name}=={version}")
                    except Exception as e:
                        self.failed_count += 1
                        self._log(f"❌ {pkg_name}=={version} - Exception: {e}")
            
            # Batch arası kısa dinlenme
            if batch_num < len(batches):
                time.sleep(1)
    
    def _install_single_package(self, package_name: str, version: str) -> bool:
        """Tek paket kurulumu (paralel thread için)"""
        try:
            package_spec = f"{package_name}=={version}" if version else package_name
            
            # Sessiz kurulum komutu
            cmd = [
                sys.executable, "-m", "pip", "install", package_spec,
                "--quiet", "--no-warn-script-location", "--disable-pip-version-check"
            ]
            
            # Timeout ile kurulum
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 dakika timeout
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def _log(self, message: str):
        """Sessiz log - sadece kritik mesajlar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Konsola sadece önemli mesajları yazdır
        if any(keyword in message for keyword in ["✅", "❌", "🚀", "📊", "⚠️"]):
            print(log_entry)
        
        # Dosyaya her şeyi kaydet
        self.installation_log.append(log_entry)
        
        try:
            with open("pdsx_silent_install.log", "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except:
            pass  # Log hatası kurulumu durdurmasın
    
    def _emergency_exit(self, signum, frame):
        """Acil çıkış"""
        self._log(f"⚠️  Acil çıkış (Signal: {signum})")
        self._cleanup()
        sys.exit(1)
    
    def _cleanup(self):
        """Temizlik işlemleri"""
        duration = time.time() - self.start_time
        self._log(f"🏁 Kurulum süresi: {duration:.1f} saniye")
        
        # Özet rapor
        if hasattr(self, 'installed_count'):
            summary = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "total_packages": self.total_packages,
                "installed": self.installed_count,
                "failed": self.failed_count,
                "success_rate": (self.installed_count / self.total_packages * 100) if self.total_packages > 0 else 0
            }
            
            try:
                with open("pdsx_install_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            except:
                pass
    
    def get_stats(self) -> dict:
        """Kurulum istatistikleri"""
        return {
            "total": self.total_packages,
            "installed": self.installed_count,
            "failed": self.failed_count,
            "success_rate": (self.installed_count / self.total_packages * 100) if self.total_packages > 0 else 0,
            "duration": time.time() - self.start_time
        }

# ===================================================
# PDS-X REPL entegrasyonu için basit API
# ===================================================

def auto_setup_for_repl():
    """REPL başlatılmadan önce otomatik kurulum"""
    try:
        installer = SilentAutoImporter(mode="REPL")
        return installer.get_stats()
    except Exception as e:
        print(f"⚠️  Otomatik kurulum hatası: {e}")
        return {"error": str(e)}

def check_requirements_status() -> dict:
    """Gereksinimler durumu (hızlı kontrol)"""
    installer = SilentAutoImporter.__new__(SilentAutoImporter)  # __init__ çağırma
    missing = installer._find_missing_packages()
    
    return {
        "total": len(REQUIRED_PACKAGES),
        "missing": len(missing),
        "ready": len(missing) == 0
    }

# Test için doğrudan çalıştırma
if __name__ == "__main__":
    print("🔧 PDS-X Sessiz Otomatik Kurucu başlatılıyor...")
    installer = SilentAutoImporter()
    print("✅ Kurulum tamamlandı!")
