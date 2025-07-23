# auto_importer_clean.py - PDS-X Minimal Working Version
# GEÇICI ÇÖZÜM: Interaktif demo problemini çözmek için minimal wrapper

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# PDS-X İhraç Tanımı
__pdsX_exports__ = {
    "AutoImporter": "Otomatik modül yükleme ana sınıfı",
    "EnvManager": "Sanal ortam yönetimi sınıfı", 
    "setup_pdsX_environment": "PDS-X çalışma ortamını hazırla",
}

class PdsXException(Exception):
    def __init__(self, message: str, code: str = "ERR_UNDEFINED"):
        self.code = code
        super().__init__(f"[Error {code}] {message}")

class EnvManager:
    """Minimal Sanal Ortam Yöneticisi"""
    def __init__(self):
        self.cache_dir = Path(".pdsx_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def is_running_in_venv(self) -> bool:
        """Virtual environment kontrolü"""
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
    
    def check_python_version(self) -> bool:
        """Python sürüm kontrolü"""
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return version == "3.10"
    
    def setup_environment(self) -> bool:
        """Ortam kurulumu"""
        try:
            print(f"[EnvManager] Python {sys.version_info.major}.{sys.version_info.minor} ortamı hazır")
            return True
        except Exception as e:
            print(f"[EnvManager] Ortam kurulum hatası: {e}")
            return False
    
    def get_pip_path(self) -> Optional[str]:
        """Pip yolunu bul"""
        return sys.executable

class AutoImporter:
    """Minimal AutoImporter - İnteraktif Demo Olmadan"""
    def __init__(self, entry_point_script=None, mode="NORMAL", log_to_file=True, 
                 log_to_terminal=True, silent_install=False, pdsX_args=None):
        self.entry_point_script = entry_point_script or "pdsXuv14.py"
        self.mode = mode
        self.silent_install = silent_install
        self.env_manager = EnvManager()
        
        print(f"[AutoImporter] Minimal mode başlatıldı: {mode}")
        
        # Bileşenleri başlat
        self.initialize_components()
        
    def initialize_components(self):
        """Bileşenleri başlat"""
        try:
            if not self.env_manager.is_running_in_venv():
                print("[AutoImporter] UYARI: Sanal ortam aktif değil")
            print("[AutoImporter] Minimal bileşenler başlatıldı")
            return True
        except Exception as e:
            print(f"[AutoImporter] Bileşen başlatma hatası: {e}")
            return False
            
    def setup_pdsX_environment(self) -> bool:
        """PDS-X ortam kurulumu"""
        return self.env_manager.setup_environment()
            
    def start_background_services(self):
        """Arka plan servislerini başlat"""
        print("[AutoImporter] Minimal arka plan servisleri başlatıldı")
        
    def shutdown(self):
        """Servisleri kapat"""
        print("[AutoImporter] Minimal servisler kapatıldı")
        
    def install_requirements(self):
        """Bağımlılıkları kur"""
        print("[AutoImporter] Minimal requirements - atlandı")
        return True
        
    def auto_install_package(self, package: str) -> bool:
        """Minimal paket kurulumu"""
        try:
            print(f"[AutoImporter] {package} kuruluyor...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"[AutoImporter] ✅ {package} başarıyla kuruldu")
                return True
            else:
                print(f"[AutoImporter] ❌ {package} kurulumu başarısız")
                return False
        except Exception as e:
            print(f"[AutoImporter] {package} kurulum hatası: {e}")
            return False
    
    def check_package_installed(self, package: str) -> bool:
        """Paket kurulu mu kontrolü"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "show", package
            ], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
        
    def cleanup(self):
        """Temizlik"""
        print("[AutoImporter] Minimal cleanup tamamlandı")

def setup_pdsX_environment(**kwargs) -> bool:
    """PDS-X ortam kurulumu - global fonksiyon"""
    try:
        env_manager = EnvManager()
        return env_manager.setup_environment()
    except Exception as e:
        print(f"[setup_pdsX_environment] Hata: {e}")
        return False

# Modül olarak import edilirse sadece sınıfları export et
if __name__ == "__main__":
    print("🔧 PDS-X AutoImporter (Minimal) - Doğrudan çalıştırıldı")
    ai = AutoImporter(mode="STANDALONE")
    print("✅ Minimal AutoImporter test tamamlandı!")
else:
    print("📦 PDS-X AutoImporter (Minimal) modülü yüklendi")
