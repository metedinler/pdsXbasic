#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Minimal AutoImporter
==========================

Sadece temel paketleri yükleyen minimal AutoImporter.
108 paket yerine sadece gerekli olanlar.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Minimal paket listesi - sadece gerekliler
MINIMAL_PACKAGES = [
    'numpy',        # Temel matematik
    'pandas',       # Veri analizi
    'matplotlib',   # Grafik
    'requests',     # HTTP
    'beautifulsoup4', # Web scraping
    'pillow',       # Resim işleme
    'flask',        # Web framework
    'click',        # CLI
    'pyyaml',       # YAML
    'rich',         # Terminal
    'psutil'        # Sistem
]

class MinimalAutoImporter:
    """Minimal AutoImporter sınıfı"""
    
    def __init__(self):
        self.cache_dir = Path(".pdsx_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.wheels_dir = self.cache_dir / "wheels"
        self.wheels_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "packages.json"
        self.installed_packages = set()
        print("[AutoImporter] Minimal AutoImporter başlatıldı")
        
    def load_metadata(self):
        """Metadata yükle"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"[AutoImporter] Metadata hata: {e}")
            return {}
            
    def save_metadata(self, metadata):
        """Metadata kaydet"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AutoImporter] Metadata kaydetme hatası: {e}")
            
    def is_installed(self, package):
        """Paket kurulu mu kontrol et"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
            
    def install_from_cache(self, package):
        """Cache'den kur"""
        cache_file = self.wheels_dir / f"{package}.whl"
        if cache_file.exists():
            print(f"[AutoImporter] {package} cache'den kuruluyor...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", str(cache_file)
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[AutoImporter] ✅ {package} cache'den kuruldu")
                return True
            else:
                print(f"[AutoImporter] ❌ Cache kurulum hatası: {result.stderr}")
        return False
        
    def download_and_cache(self, package):
        """Paketi indir ve cache'e al"""
        print(f"[AutoImporter] {package} indiriliyor ve cache'e alınıyor...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "download", 
            "--dest", str(self.wheels_dir), 
            "--no-deps", package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[AutoImporter] ✅ {package} cache'e alındı")
            
            # Metadata güncelle
            metadata = self.load_metadata()
            metadata[package] = {
                "cached_at": str(Path().cwd()),
                "status": "cached"
            }
            self.save_metadata(metadata)
            return True
        else:
            print(f"[AutoImporter] ❌ Cache alma hatası: {result.stderr}")
            return False
            
    def install_package(self, package):
        """Paket kur (cache mantığı ile)"""
        # Zaten kurulu mu?
        if self.is_installed(package):
            print(f"[AutoImporter] ✅ {package} zaten kurulu")
            return True
            
        # Cache'den dene
        if self.install_from_cache(package):
            self.installed_packages.add(package)
            return True
            
        # Normal kurulum
        print(f"[AutoImporter] {package} normal kurulum...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[AutoImporter] ✅ {package} kuruldu")
            self.installed_packages.add(package)
            
            # Cache'e al
            self.download_and_cache(package)
            return True
        else:
            print(f"[AutoImporter] ❌ {package} kurulum hatası: {result.stderr}")
            return False
            
    def install_minimal_packages(self):
        """Minimal paketleri kur"""
        print(f"[AutoImporter] {len(MINIMAL_PACKAGES)} minimal paket kuruluyor...")
        
        success_count = 0
        for package in MINIMAL_PACKAGES:
            if self.install_package(package):
                success_count += 1
                
        print(f"[AutoImporter] ✅ {success_count}/{len(MINIMAL_PACKAGES)} paket kuruldu")
        return success_count == len(MINIMAL_PACKAGES)
        
    def get_status(self):
        """Durum bilgisi"""
        metadata = self.load_metadata()
        return {
            "installed_packages": len(self.installed_packages),
            "cached_packages": len(metadata),
            "cache_dir": str(self.cache_dir),
            "minimal_packages": len(MINIMAL_PACKAGES)
        }

def main():
    """Test fonksiyonu"""
    ai = MinimalAutoImporter()
    print("Minimal AutoImporter Test")
    print("Status:", ai.get_status())
    
    # Bir paket test et
    if not ai.is_installed('requests'):
        print("requests kurulacak...")
        ai.install_package('requests')
    else:
        print("requests zaten kurulu")

if __name__ == "__main__":
    main()
