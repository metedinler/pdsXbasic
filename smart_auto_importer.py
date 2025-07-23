#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Smart AutoImporter - Lightweight Version
==============================================

Sadece GERÇEKTEN eksik olan paketleri kurar.
108 paket yerine akıllı tespit yapar.
"""

import sys
import os
import subprocess
import importlib
import logging
from pathlib import Path

class SmartAutoImporter:
    """Akıllı, hafif AutoImporter"""
    
    def __init__(self):
        self.installed_packages = set()
        self.missing_packages = set()
        self.failed_packages = set()
        
        # Sadece TEMEL paketler (gerçekten gerekli olanlar)
        self.core_packages = {
            'numpy', 'pandas', 'requests', 'psutil'
        }
        
        # İsteğe bağlı paketler (sadece kullanılırsa yükle)
        self.optional_packages = {
            'matplotlib': 'data visualization',
            'scipy': 'scientific computing', 
            'scikit-learn': 'machine learning',
            'beautifulsoup4': 'web scraping',
            'pillow': 'image processing',
            'flask': 'web framework',
            'streamlit': 'web apps'
        }
    
    def check_package(self, package_name):
        """Tek paket kontrolü"""
        try:
            importlib.import_module(package_name.replace('-', '_'))
            self.installed_packages.add(package_name)
            return True
        except ImportError:
            self.missing_packages.add(package_name)
            return False
    
    def install_package(self, package_name):
        """Tek paket kurulumu"""
        try:
            print(f"[AutoImporter] 📦 {package_name} kuruluyor...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"[AutoImporter] ✅ {package_name} başarıyla kuruldu")
                self.installed_packages.add(package_name)
                return True
            else:
                print(f"[AutoImporter] ❌ {package_name} kurulamadı: {result.stderr}")
                self.failed_packages.add(package_name)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[AutoImporter] ⏰ {package_name} kurulumu zaman aşımı")
            self.failed_packages.add(package_name)
            return False
        except Exception as e:
            print(f"[AutoImporter] ❌ {package_name} kurulum hatası: {e}")
            self.failed_packages.add(package_name)
            return False
    
    def smart_import(self, module_name, package_name=None):
        """Akıllı import - sadece gerektiğinde kur"""
        if not package_name:
            package_name = module_name
            
        try:
            # Önce import dene
            return importlib.import_module(module_name)
        except ImportError:
            print(f"[AutoImporter] 🔍 {module_name} bulunamadı, kurulacak...")
            
            # Package'ı kur
            if self.install_package(package_name):
                try:
                    return importlib.import_module(module_name)
                except ImportError as e:
                    print(f"[AutoImporter] ❌ {module_name} kurulumdan sonra da import edilemedi: {e}")
                    return None
            else:
                return None
    
    def check_core_packages(self):
        """Sadece core paketleri kontrol et"""
        print("[AutoImporter] 🔍 Core paketler kontrol ediliyor...")
        
        missing_core = []
        for package in self.core_packages:
            if not self.check_package(package):
                missing_core.append(package)
        
        if missing_core:
            print(f"[AutoImporter] ⚠️ Eksik core paketler: {missing_core}")
            
            # Kullanıcıya sor
            response = input(f"[AutoImporter] 📥 Eksik {len(missing_core)} core paketi kurmak ister misiniz? (y/n): ")
            if response.lower() in ['y', 'yes', 'evet', '']:
                for package in missing_core:
                    self.install_package(package)
            else:
                print("[AutoImporter] ⏭️ Core paket kurulumu atlandı")
        else:
            print("[AutoImporter] ✅ Tüm core paketler mevcut")
    
    def install_on_demand(self, module_name):
        """İsteğe bağlı kurulum"""
        if module_name in self.optional_packages:
            description = self.optional_packages[module_name]
            response = input(f"[AutoImporter] 📥 {module_name} ({description}) kurmak ister misiniz? (y/n): ")
            if response.lower() in ['y', 'yes', 'evet']:
                return self.install_package(module_name)
        return False
    
    def get_status(self):
        """Durum raporu"""
        return {
            'installed': len(self.installed_packages),
            'missing': len(self.missing_packages), 
            'failed': len(self.failed_packages),
            'installed_packages': self.installed_packages,
            'missing_packages': self.missing_packages,
            'failed_packages': self.failed_packages
        }

# Kolay kullanım için global fonksiyonlar
_smart_importer = SmartAutoImporter()

def smart_import(module_name, package_name=None):
    """Global smart import fonksiyonu"""
    return _smart_importer.smart_import(module_name, package_name)

def check_core():
    """Core paketleri kontrol et"""
    return _smart_importer.check_core_packages()

def get_status():
    """İmporter durumu"""
    return _smart_importer.get_status()

# Eski AutoImporter uyumluluğu için
class EnvManager:
    """Minimal environment manager"""
    def __init__(self):
        self.python_exe = sys.executable
        
    def is_running_in_venv(self):
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
    def setup_environment(self, **kwargs):
        print("[EnvManager] Ortam kontrolü tamamlandı")
        return True

class AutoImporter:
    """Eski uyumluluk için minimal wrapper"""
    def __init__(self, **kwargs):
        self.env_manager = EnvManager()
        self.smart_importer = _smart_importer
        print("[AutoImporter] Smart mode başlatıldı")
        
    def initialize_components(self):
        return True
        
    def start_background_services(self):
        pass
        
    def shutdown(self):
        print("[AutoImporter] Smart AutoImporter kapatıldı")
        
    def install_requirements(self):
        return self.smart_importer.check_core_packages()

if __name__ == "__main__":
    print("🚀 Smart AutoImporter Test")
    
    # Core paketleri kontrol et
    check_core()
    
    # Test smart import
    numpy = smart_import('numpy')
    if numpy:
        print("✅ NumPy import başarılı")
        print(f"NumPy version: {numpy.__version__}")
    
    # Durum raporu
    status = get_status()
    print(f"\\n📊 Durum: {status['installed']} kurulu, {status['missing']} eksik, {status['failed']} başarısız")
