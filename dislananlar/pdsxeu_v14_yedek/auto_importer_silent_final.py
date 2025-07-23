#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X SilentAutoImporter v2.0 - Tamamen Otomatik, Sessiz Kurulum
================================================================

KULLANICIDAN HİÇBİR GİRDİ ALMAZ - Tamamen otomatik çalışır!

Özellikler:
- ❌ Kullanıcı girdisi yok
- ✅ Paralel kurulum 
- ✅ Akıllı çakışma çözümü
- ✅ Gerçek zamanlı monitoring
- ✅ Güvenli ortam yönetimi
- ✅ REPL entegrasyonu
"""

import sys
import os
import subprocess
import json
import time
import asyncio
import logging
import signal
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Sık kullanılan modüller için lazy import
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil bulunamadı, paralel kurulum sınırlı olacak")

# =====================================================
# PDS-X İçin Gerekli Paketler - 108 Tam Liste
# =====================================================

CRITICAL_PACKAGES = [
    "numpy==1.24.3", "pandas==2.0.3", "matplotlib==3.7.2",
    "scipy==1.11.1", "scikit-learn==1.3.0", "requests==2.31.0",
    "beautifulsoup4==4.12.2", "lxml==4.9.3", "pillow==10.0.0",
    "opencv-python==4.8.0.74"
]

AI_ML_PACKAGES = [
    "torch==2.0.1", "tensorflow==2.13.0", "transformers==4.33.2",
    "openai==0.27.8", "langchain==0.0.267", "chromadb==0.4.6",
    "sentence-transformers==2.2.2", "faiss-cpu==1.7.4",
    "spacy==3.6.1", "nltk==3.8.1"
]

DATA_PACKAGES = [
    "jupyter==1.0.0", "ipykernel==6.25.1", "ipywidgets==8.0.7",
    "plotly==5.16.1", "seaborn==0.12.2", "bokeh==3.2.2",
    "streamlit==1.26.0", "dash==2.14.1", "fastapi==0.103.1",
    "uvicorn==0.23.2"
]

DEV_PACKAGES = [
    "pytest==7.4.2", "black==23.7.0", "flake8==6.0.0", 
    "mypy==1.5.1", "pre-commit==3.4.0", "tqdm==4.66.1",
    "click==8.1.7", "rich==13.5.2", "typer==0.9.0",
    "pydantic==2.3.0"
]

WEB_PACKAGES = [
    "selenium==4.12.0", "scrapy==2.11.0", "flask==2.3.3",
    "django==4.2.5", "aiohttp==3.8.5", "httpx==0.24.1",
    "websockets==11.0.3", "socketio==5.8.0"
]

CRYPTO_PACKAGES = [
    "cryptography==41.0.4", "pycryptodome==3.18.0", 
    "pyotp==2.9.0", "qrcode==7.4.2"
]

SYSTEM_PACKAGES = [
    "psutil==5.9.5", "watchdog==3.0.0", "schedule==1.2.0",
    "apscheduler==3.10.4", "redis==4.6.0", "pymongo==4.5.0",
    "sqlalchemy==2.0.21", "alembic==1.12.0"
]

MISC_PACKAGES = [
    "python-dotenv==1.0.0", "pyyaml==6.0.1", "toml==0.10.2",
    "configparser==6.0.0", "argparse==1.4.0", "logging==0.4.9.6",
    "datetime==5.2", "pathlib==1.0.1", "collections==0.1.1",
    "itertools==10.1.0", "functools==1.0.0", "operator==1.0.0"
]

# Tüm paketleri birleştir
ALL_PACKAGES = (CRITICAL_PACKAGES + AI_ML_PACKAGES + DATA_PACKAGES + 
                DEV_PACKAGES + WEB_PACKAGES + CRYPTO_PACKAGES + 
                SYSTEM_PACKAGES + MISC_PACKAGES)

# =====================================================
# Yardımcı Sınıflar
# =====================================================

class SilentLogger:
    """Sessiz loglama sistemi"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Log dosyası
        self.log_file = self.log_dir / f"silent_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Logger kurulumu
        self.logger = logging.getLogger('SilentAutoImporter')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)

class ResourceMonitor:
    """Sistem kaynak monitörü"""
    
    def __init__(self):
        self.is_monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.start_time = time.time()
        
    def start(self):
        """Monitoring başlat"""
        if not HAS_PSUTIL:
            return False
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True
        
    def stop(self):
        """Monitoring durdur"""
        self.is_monitoring = False
        
    def _monitor_loop(self):
        """Monitoring döngüsü"""
        while self.is_monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)
                
                # Son 60 sample tut
                if len(self.cpu_samples) > 60:
                    self.cpu_samples = self.cpu_samples[-60:]
                    self.memory_samples = self.memory_samples[-60:]
                    
                time.sleep(5)  # 5 saniye ara
                
            except Exception:
                break
                
    def get_stats(self) -> Dict:
        """İstatistikleri al"""
        if not self.cpu_samples:
            return {
                'cpu_avg': 0.0,
                'memory_avg': 0.0,
                'uptime': time.time() - self.start_time,
                'samples': 0,
                'can_parallel': False
            }
            
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples)
        memory_avg = sum(self.memory_samples) / len(self.memory_samples)
        
        return {
            'cpu_avg': cpu_avg,
            'memory_avg': memory_avg,
            'uptime': time.time() - self.start_time,
            'samples': len(self.cpu_samples),
            'can_parallel': cpu_avg < 70 and memory_avg < 80  # Paralel kurulum için uygun mu?
        }

class EnvManager:
    """Python ortam yöneticisi"""
    
    def __init__(self, venv_name: str = "pdsX_env"):
        self.venv_name = venv_name
        self.venv_path = Path.cwd() / self.venv_name
        self.cache_dir = Path.cwd() / "pip_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def is_venv_exists(self) -> bool:
        """Sanal ortam var mı?"""
        return (self.venv_path / "Scripts" / "python.exe").exists()
        
    def create_venv(self) -> bool:
        """Sanal ortam oluştur"""
        try:
            if self.is_venv_exists():
                return True
                
            result = subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], capture_output=True, text=True, check=True)
            
            return self.is_venv_exists()
            
        except Exception:
            return False
            
    def get_venv_python(self) -> Optional[Path]:
        """Sanal ortam Python yolu"""
        if self.is_venv_exists():
            return self.venv_path / "Scripts" / "python.exe"
        return None
        
    def get_venv_pip(self) -> Optional[Path]:
        """Sanal ortam Pip yolu"""
        if self.is_venv_exists():
            return self.venv_path / "Scripts" / "pip.exe"
        return None
        
    def is_running_in_venv(self) -> bool:
        """Şu anda sanal ortamda mı?"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

class ConflictResolver:
    """Paket çakışma çözücü"""
    
    def __init__(self, logger: SilentLogger):
        self.logger = logger
        self.known_conflicts = {
            'tensorflow': ['torch'],
            'torch': ['tensorflow'],
            'opencv-python': ['opencv-contrib-python'],
            'pillow': ['PIL']
        }
        
    def detect_conflicts(self, package: str, error_msg: str) -> List[str]:
        """Çakışan paketleri tespit et"""
        conflicts = []
        
        # Bilinen çakışmalar
        pkg_name = package.split('==')[0].lower()
        if pkg_name in self.known_conflicts:
            conflicts.extend(self.known_conflicts[pkg_name])
            
        # Hata mesajından çıkar
        if 'conflict' in error_msg.lower():
            # Basit regex ile paket ismi yakala
            import re
            conflict_match = re.search(r'conflicts with (\w+)', error_msg)
            if conflict_match:
                conflicts.append(conflict_match.group(1))
                
        return list(set(conflicts))
        
    def resolve_conflicts(self, package: str, conflicts: List[str]) -> bool:
        """Çakışmaları çöz"""
        try:
            self.logger.info(f"Çakışma çözülüyor: {package} <-> {conflicts}")
            
            # Çakışan paketleri kaldır
            for conflict_pkg in conflicts:
                self._uninstall_package(conflict_pkg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Çakışma çözüm hatası: {e}")
            return False
            
    def _uninstall_package(self, package: str) -> bool:
        """Paketi kaldır"""
        try:
            pip_path = EnvManager().get_venv_pip()
            if not pip_path:
                return False
                
            result = subprocess.run([
                str(pip_path), "uninstall", package, "-y"
            ], capture_output=True, text=True, check=False)
            
            success = result.returncode == 0
            if success:
                self.logger.info(f"Paket kaldırıldı: {package}")
            else:
                self.logger.warning(f"Paket kaldırma başarısız: {package}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Paket kaldırma hatası {package}: {e}")
            return False

# =====================================================
# Ana SilentAutoImporter Sınıfı
# =====================================================

class SilentAutoImporter:
    """Tamamen otomatik, sessiz paket kurucusu"""
    
    def __init__(self, mode: str = "SILENT_AUTO"):
        self.mode = mode
        self.start_time = time.time()
        
        # Bileşenler
        self.logger = SilentLogger()
        self.env_manager = EnvManager()
        self.monitor = ResourceMonitor()
        self.conflict_resolver = ConflictResolver(self.logger)
        
        # Durum takibi
        self.installed_packages = set()
        self.failed_packages = set()
        self.skipped_packages = set()
        self.stats = {
            'total': len(ALL_PACKAGES),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(),
            'conflicts_resolved': 0
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._emergency_stop)
        signal.signal(signal.SIGTERM, self._emergency_stop)
        
        # Başlangıç
        self.logger.info(f"SilentAutoImporter başlatıldı - Mode: {mode}")
        self._initialize()
        
    def _initialize(self):
        """Kurulumu başlat"""
        try:
            print("🔧 PDS-X SilentAutoImporter hazırlanıyor...")
            
            # Sanal ortam kontrolü/oluşturma
            if not self.env_manager.is_venv_exists():
                print("📦 Python sanal ortamı oluşturuluyor...")
                if not self.env_manager.create_venv():
                    raise Exception("Sanal ortam oluşturulamadı!")
                    
            # Monitoring başlat
            if self.monitor.start():
                self.logger.info("Kaynak monitoring başlatıldı")
                
            print("✅ SilentAutoImporter hazır!")
            
        except Exception as e:
            self.logger.error(f"Başlatma hatası: {e}")
            raise
            
    def _emergency_stop(self, signum, frame):
        """Acil durdurma"""
        print("\n🛑 Acil durdurma sinyali alındı!")
        self.logger.warning("Acil durdurma - istatistikler kaydediliyor")
        self._save_progress()
        sys.exit(1)
        
    def _save_progress(self):
        """İlerlemeyi kaydet"""
        try:
            progress_file = Path.cwd() / "auto_import_progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'stats': {**self.stats, 'start_time': self.stats['start_time'].isoformat()},
                    'installed': list(self.installed_packages),
                    'failed': list(self.failed_packages),
                    'skipped': list(self.skipped_packages),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"İlerleme kaydedildi: {progress_file}")
            
        except Exception as e:
            self.logger.error(f"İlerleme kaydetme hatası: {e}")
            
    def check_package_installed(self, package: str) -> bool:
        """Paket kurulu mu kontrol et"""
        try:
            python_path = self.env_manager.get_venv_python()
            if not python_path:
                return False
                
            pkg_name = package.split('==')[0].replace('-', '_')
            
            result = subprocess.run([
                str(python_path), "-c", f"import {pkg_name}"
            ], capture_output=True, text=True, check=False)
            
            return result.returncode == 0
            
        except Exception:
            return False
            
    def install_single_package(self, package: str) -> bool:
        """Tek paket kur"""
        try:
            # Zaten kurulu mu?
            if self.check_package_installed(package):
                self.skipped_packages.add(package)
                self.stats['skipped'] += 1
                self.logger.info(f"Zaten kurulu: {package}")
                return True
                
            pip_path = self.env_manager.get_venv_pip()
            if not pip_path:
                raise Exception("Pip bulunamadı!")
                
            # Kurulum komutu
            cmd = [
                str(pip_path), "install", package,
                "--quiet", "--no-warn-script-location",
                "--cache-dir", str(self.env_manager.cache_dir)
            ]
            
            self.logger.info(f"Kuruluyor: {package}")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=300, check=False  # 5 dakika timeout
            )
            
            if result.returncode == 0:
                self.installed_packages.add(package)
                self.stats['success'] += 1
                self.logger.info(f"✅ Kuruldu: {package}")
                return True
            else:
                # Çakışma var mı?
                error_msg = result.stderr + result.stdout
                conflicts = self.conflict_resolver.detect_conflicts(package, error_msg)
                
                if conflicts:
                    self.logger.warning(f"Çakışma tespit edildi: {package} <-> {conflicts}")
                    
                    if self.conflict_resolver.resolve_conflicts(package, conflicts):
                        self.stats['conflicts_resolved'] += 1
                        # Tekrar dene
                        return self.install_single_package(package)
                
                self.failed_packages.add(package)
                self.stats['failed'] += 1
                self.logger.error(f"❌ Başarısız: {package} - {error_msg[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.failed_packages.add(package)
            self.stats['failed'] += 1
            self.logger.error(f"⏱️ Timeout: {package}")
            return False
            
        except Exception as e:
            self.failed_packages.add(package)
            self.stats['failed'] += 1
            self.logger.error(f"❌ Hata: {package} - {e}")
            return False
            
    def parallel_install(self, packages: List[str], max_workers: int = 3) -> List[bool]:
        """Paralel kurulum"""
        self.logger.info(f"Paralel kurulum başlıyor: {len(packages)} paket, {max_workers} worker")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Görevleri başlat
            future_to_package = {
                executor.submit(self.install_single_package, pkg): pkg 
                for pkg in packages
            }
            
            # Sonuçları topla
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress göster
                    progress = (self.stats['success'] + self.stats['failed'] + self.stats['skipped'])
                    percent = (progress / self.stats['total']) * 100
                    print(f"\r📊 İlerleme: {progress}/{self.stats['total']} (%{percent:.1f})", end='', flush=True)
                    
                except Exception as e:
                    self.logger.error(f"Future hatası {package}: {e}")
                    results.append(False)
                    
        print()  # Yeni satır
        return results
        
    def sequential_install(self, packages: List[str]) -> List[bool]:
        """Sıralı kurulum"""
        self.logger.info(f"Sıralı kurulum başlıyor: {len(packages)} paket")
        
        results = []
        for i, package in enumerate(packages, 1):
            result = self.install_single_package(package)
            results.append(result)
            
            # Progress göster
            percent = (i / len(packages)) * 100
            print(f"\r📊 İlerleme: {i}/{len(packages)} (%{percent:.1f})", end='', flush=True)
            
        print()  # Yeni satır
        return results
        
    def run_full_install(self, force_sequential: bool = False) -> Dict:
        """Tam kurulum çalıştır"""
        try:
            print("🚀 PDS-X tam paket kurulumu başlıyor...")
            self.logger.info("Tam kurulum başlatıldı")
            
            # Kaynak durumu kontrol et
            stats = self.monitor.get_stats()
            can_parallel = stats['can_parallel'] and HAS_PSUTIL and not force_sequential
            
            if can_parallel:
                print(f"⚡ Paralel kurulum modu (CPU: {stats['cpu_avg']:.1f}%, RAM: {stats['memory_avg']:.1f}%)")
                
                # Kritik paketleri önce kur
                print("🔥 Kritik paketler kuruluyor...")
                self.sequential_install(CRITICAL_PACKAGES)
                
                # Kalan paketleri paralel kur
                remaining = [pkg for pkg in ALL_PACKAGES if pkg not in CRITICAL_PACKAGES]
                if remaining:
                    print("⚡ Paralel kurulum başlıyor...")
                    
                    # Optimal worker sayısı
                    max_workers = min(4, max(2, int(psutil.cpu_count() * 0.5)))
                    self.parallel_install(remaining, max_workers)
                    
            else:
                print("🔄 Sıralı kurulum modu")
                self.sequential_install(ALL_PACKAGES)
                
            # Sonuçları hesapla
            total_time = time.time() - self.start_time
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            
            # Özet
            print(f"\n📊 PDS-X Kurulum Tamamlandı!")
            print(f"   ✅ Başarılı: {self.stats['success']}")
            print(f"   ❌ Başarısız: {self.stats['failed']}")
            print(f"   ⏭️  Atlandı: {self.stats['skipped']}")
            print(f"   🔧 Çakışma çözüldü: {self.stats['conflicts_resolved']}")
            print(f"   📈 Başarı oranı: %{success_rate:.1f}")
            print(f"   ⏱️  Süre: {total_time:.1f} saniye")
            
            self.logger.info(f"Kurulum tamamlandı - Başarı: {success_rate:.1f}%")
            
            # İlerlemeyi kaydet
            self._save_progress()
            
            return {
                'success': self.stats['success'],
                'failed': self.stats['failed'],
                'skipped': self.stats['skipped'],
                'total': self.stats['total'],
                'success_rate': success_rate,
                'total_time': total_time,
                'conflicts_resolved': self.stats['conflicts_resolved']
            }
            
        except Exception as e:
            self.logger.error(f"Tam kurulum hatası: {e}")
            print(f"❌ Kurulum hatası: {e}")
            return {'error': str(e)}
            
        finally:
            self.monitor.stop()
            
    def quick_check(self) -> Dict:
        """Hızlı durum kontrolü"""
        try:
            installed_count = 0
            missing_packages = []
            
            print("🔍 Hızlı paket kontrolü...")
            
            for i, package in enumerate(ALL_PACKAGES, 1):
                if self.check_package_installed(package):
                    installed_count += 1
                else:
                    missing_packages.append(package)
                    
                # Progress
                if i % 10 == 0:
                    print(f"\r📊 Kontrol: {i}/{len(ALL_PACKAGES)}", end='', flush=True)
                    
            print()
            
            coverage = (installed_count / len(ALL_PACKAGES)) * 100
            
            result = {
                'installed': installed_count,
                'missing': len(missing_packages),
                'total': len(ALL_PACKAGES),
                'coverage': coverage,
                'missing_packages': missing_packages[:10]  # İlk 10 eksik paket
            }
            
            print(f"📊 Paket Durumu: {installed_count}/{len(ALL_PACKAGES)} (%{coverage:.1f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hızlı kontrol hatası: {e}")
            return {'error': str(e)}
            
    def cleanup(self):
        """Temizlik"""
        try:
            self.monitor.stop()
            self._save_progress()
            self.logger.info("SilentAutoImporter temizlendi")
            
        except Exception as e:
            self.logger.error(f"Temizlik hatası: {e}")

# =====================================================
# REPL Entegrasyon Fonksiyonları
# =====================================================

def auto_setup_for_repl() -> Dict:
    """REPL için otomatik kurulum"""
    try:
        print("🔧 REPL için PDS-X paketleri kontrol ediliyor...")
        
        installer = SilentAutoImporter(mode="REPL_PREP")
        
        # Hızlı kontrol
        status = installer.quick_check()
        
        # Eksik paketler varsa kur
        if status.get('coverage', 0) < 90:  # %90'dan az kuruluysa
            print("📦 Eksik paketler kuruluyor...")
            result = installer.run_full_install()
            installer.cleanup()
            return result
        else:
            print("✅ Tüm paketler hazır!")
            installer.cleanup()
            return status
            
    except Exception as e:
        print(f"❌ REPL kurulum hatası: {e}")
        return {'error': str(e)}

def check_requirements_status() -> Dict:
    """Gereklilikler durumunu kontrol et"""
    try:
        installer = SilentAutoImporter(mode="STATUS_CHECK")
        result = installer.quick_check()
        installer.cleanup()
        return result
        
    except Exception as e:
        return {'error': str(e)}

def install_missing_only(max_missing: int = 20) -> Dict:
    """Sadece eksik paketleri kur"""
    try:
        installer = SilentAutoImporter(mode="MISSING_ONLY")
        
        # Kontrol et
        status = installer.quick_check()
        missing = status.get('missing_packages', [])
        
        if not missing:
            print("✅ Tüm paketler zaten kurulu!")
            installer.cleanup()
            return status
            
        # Sınırlı sayıda kur
        to_install = missing[:max_missing]
        print(f"📦 {len(to_install)} eksik paket kuruluyor...")
        
        results = installer.sequential_install(to_install)
        installer.cleanup()
        
        return {
            'installed_now': sum(results),
            'attempted': len(to_install),
            'remaining_missing': max(0, len(missing) - len(to_install))
        }
        
    except Exception as e:
        return {'error': str(e)}

# =====================================================
# Ana Çalıştırma
# =====================================================

def main():
    """Ana fonksiyon"""
    try:
        print("🚀 PDS-X SilentAutoImporter v2.0")
        print("=" * 50)
        
        # Args kontrol
        if len(sys.argv) > 1:
            mode = sys.argv[1].upper()
        else:
            mode = "FULL_AUTO"
            
        if mode == "CHECK":
            result = check_requirements_status()
            print(f"📊 Sonuç: {result}")
            
        elif mode == "MISSING":
            result = install_missing_only()
            print(f"📊 Sonuç: {result}")
            
        elif mode == "REPL":
            result = auto_setup_for_repl()
            print(f"📊 Sonuç: {result}")
            
        else:
            # Tam kurulum
            installer = SilentAutoImporter(mode="FULL_AUTO")
            result = installer.run_full_install()
            installer.cleanup()
            print(f"📊 Sonuç: {result}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu!")
        
    except Exception as e:
        print(f"❌ Ana hata: {e}")
        
if __name__ == "__main__":
    main()
