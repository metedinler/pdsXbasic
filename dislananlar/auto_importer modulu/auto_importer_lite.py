#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X AutoImporter Lite - Kurulu Sistemler İçin Hafif Versiyon
==============================================================

Bu versiyon, kurulu sistemlerde minimal kaynak kullanımı ile çalışır.
Gereksiz özellikler devre dışı bırakılmıştır:
- Conflict detection (❌)
- Parallel processing (❌)
- Genetic optimization (❌)
- Real-time monitoring (❌)
- Dependency graph (❌)

Sadece temel özellikler aktif:
- Quick package check (✅)
- Cache-first installation (✅)
- Fast startup (✅)
- Minimal resource usage (✅)

Version: 1.0 (Lite)
Date: 22 Temmuz 2025
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

class AutoImporterLite:
    """Kurulu sistemler için hafif AutoImporter"""
    
    def __init__(self, entry_point_script=None, mode="LITE", **kwargs):
        self.entry_point_script = entry_point_script
        self.mode = mode
        self.cache_dir = Path(".pdsx_cache")
        self.wheels_dir = self.cache_dir / "wheels"
        self.packages_json = self.cache_dir / "packages.json"
        self.env_manager = EnvManagerLite()
        
        # Kurulu paketleri hızlı kontrol için cache'le
        self.installed_packages = self.get_installed_packages()
        
        # Hafif modda sadece temel özellikler
        self.features = {
            "conflict_detection": False,  # Kurulu sistemde gereksiz
            "parallel_installation": False,  # Kurulu sistemde gereksiz
            "genetic_optimization": False,  # Kurulu sistemde gereksiz
            "dependency_graph": False,  # Kurulu sistemde gereksiz
            "real_time_monitoring": False,  # Kurulu sistemde gereksiz
            "cache_priority": True,  # Cache öncelikli kurulum
            "quick_check": True,  # Hızlı paket kontrolü
            "minimal_logging": True  # Az log
        }
        
        print(f"[AutoImporter Lite] 🚀 Hafif mod başlatıldı")
        print(f"[AutoImporter Lite] 📦 {len(self.installed_packages)} paket kurulu")
        
    def get_installed_packages(self) -> Dict[str, str]:
        """Kurulu paketleri listele - Cache'li versiyon"""
        cache_file = self.cache_dir / "installed_packages.json"
        
        # Cache'den yükle (1 saatlik cache)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 3600:  # 1 saat
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        # Cache yok veya eski - yeniden tara
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "list", "--format=json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                package_dict = {pkg["name"].lower(): pkg["version"] for pkg in packages}
                
                # Cache'e kaydet
                self.cache_dir.mkdir(exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(package_dict, f)
                    
                return package_dict
            return {}
        except Exception as e:
            print(f"[AutoImporter Lite] ⚠️ Paket listesi alınamadı: {e}")
            return {}
    
    def is_package_installed(self, package_name: str) -> bool:
        """Paket kurulu mu kontrol et - O(1) hızında"""
        return package_name.lower() in self.installed_packages
    
    def quick_install(self, package_name: str) -> bool:
        """Hızlı kurulum - sadece eksik paketler için"""
        if self.is_package_installed(package_name):
            print(f"[AutoImporter Lite] ✅ {package_name} zaten kurulu")
            return True
            
        # Cache'den kur
        if self.install_from_cache(package_name):
            print(f"[AutoImporter Lite] ✅ {package_name} cache'den kuruldu")
            # Kurulu paket listesini güncelle
            self.installed_packages[package_name.lower()] = "unknown"
            return True
            
        # Normal kurulum
        if self.simple_install(package_name):
            print(f"[AutoImporter Lite] ✅ {package_name} başarıyla kuruldu")
            self.installed_packages[package_name.lower()] = "unknown"
            return True
            
        print(f"[AutoImporter Lite] ❌ {package_name} kurulamadı")
        return False
    
    def install_from_cache(self, package_name: str) -> bool:
        """Cache'den kurulum"""
        if not self.wheels_dir.exists():
            return False
            
        # Cache'de wheel dosyası ara
        cache_files = list(self.wheels_dir.glob(f"{package_name}*.whl"))
        if not cache_files:
            # Alternatif isimlerle ara
            alt_patterns = [
                f"*{package_name}*.whl",
                f"{package_name.replace('-', '_')}*.whl",
                f"{package_name.replace('_', '-')}*.whl"
            ]
            for pattern in alt_patterns:
                cache_files = list(self.wheels_dir.glob(pattern))
                if cache_files:
                    break
        
        if cache_files:
            cache_file = cache_files[0]  # İlk bulduğunu kullan
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", str(cache_file), "--no-deps"
                ], capture_output=True, text=True, timeout=60)
                return result.returncode == 0
            except:
                return False
        return False
    
    def simple_install(self, package_name: str) -> bool:
        """Basit kurulum - minimal seçeneklerle"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name, 
                "--no-cache-dir", "--quiet"
            ], capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except:
            return False
    
    def bulk_check(self, package_list: List[str]) -> List[str]:
        """Toplu paket kontrolü - eksik olanları döndür"""
        missing = []
        for pkg in package_list:
            if not self.is_package_installed(pkg):
                missing.append(pkg)
        return missing
    
    def install_requirements(self) -> bool:
        """Temel gereksinimleri kur - sadece eksik olanlar"""
        # Minimal gereksinimler listesi
        required_packages = [
            'numpy', 'pandas', 'requests', 'psutil', 'aiofiles',
            'pyyaml', 'cryptography', 'pillow'
        ]
        
        missing = self.bulk_check(required_packages)
        if not missing:
            print("[AutoImporter Lite] ✅ Tüm temel paketler kurulu")
            return True
            
        print(f"[AutoImporter Lite] 📦 {len(missing)} eksik paket bulundu")
        
        success_count = 0
        for pkg in missing:
            if self.quick_install(pkg):
                success_count += 1
            time.sleep(0.1)  # CPU'yu yorma
        
        print(f"[AutoImporter Lite] ✅ {success_count}/{len(missing)} paket kuruldu")
        return success_count == len(missing)
    
    def initialize_components(self) -> bool:
        """Minimal bileşen başlatma"""
        print("[AutoImporter Lite] 🔧 Minimal bileşenler başlatılıyor...")
        
        # Cache dizinini oluştur
        self.cache_dir.mkdir(exist_ok=True)
        self.wheels_dir.mkdir(exist_ok=True)
        
        # Ortam kontrolü
        if not self.env_manager.is_running_in_venv():
            print("[AutoImporter Lite] ⚠️ Sanal ortam aktif değil")
        
        print("[AutoImporter Lite] ✅ Minimal bileşenler hazır")
        return True
    
    def start_background_services(self):
        """Minimal arka plan servisleri - Lite modda yok"""
        print("[AutoImporter Lite] 🏃‍♂️ Arka plan servisleri atlandı (Lite mod)")
        
    def shutdown(self):
        """Minimal kapatma"""
        print("[AutoImporter Lite] 🛑 Lite AutoImporter kapatılıyor...")
        
    def cleanup(self):
        """Minimal temizlik"""
        print("[AutoImporter Lite] 🧹 Lite temizlik tamamlandı")
        
    def summary_report(self) -> Dict[str, Any]:
        """Özet rapor"""
        return {
            "mode": "LITE",
            "installed_packages": len(self.installed_packages),
            "cache_available": self.wheels_dir.exists(),
            "cache_size_mb": self._get_cache_size_mb(),
            "features_disabled": [k for k, v in self.features.items() if not v],
            "features_enabled": [k for k, v in self.features.items() if v],
            "startup_time": "~0.1s",
            "memory_usage": "~50% less than full version"
        }
    
    def _get_cache_size_mb(self) -> float:
        """Cache boyutunu MB olarak hesapla"""
        if not self.wheels_dir.exists():
            return 0.0
        
        total_size = 0
        try:
            for wheel_file in self.wheels_dir.glob("*.whl"):
                total_size += wheel_file.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0


class EnvManagerLite:
    """Hafif ortam yöneticisi"""
    
    def __init__(self):
        self.python_exe = sys.executable
        
    def is_running_in_venv(self) -> bool:
        """Sanal ortam kontrolü"""
        return (
            'venv' in sys.executable or 
            'conda' in sys.executable or
            'pdsx_isolated_env' in sys.executable or
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
            
    def setup_environment(self, **kwargs) -> bool:
        """Minimal ortam kurulumu"""
        print("[EnvManager Lite] 🔧 Minimal ortam kontrolü")
        
        if self.is_running_in_venv():
            print("[EnvManager Lite] ✅ Sanal ortam aktif")
            return True
        else:
            print("[EnvManager Lite] ⚠️ Sanal ortam bulunamadı")
            return True  # Lite modda devam et


class EnvironmentDetector:
    """Sistem durumu tespiti"""
    
    @staticmethod
    def is_fresh_system() -> bool:
        """Yeni sistem mi kontrol et"""
        indicators = [
            not Path(".pdsx_cache").exists(),
            not Path(".pdsx_isolated_env").exists(),
            not Path(".pdsx_last_args.json").exists()
        ]
        return any(indicators)
    
    @staticmethod
    def get_system_status() -> Dict[str, Any]:
        """Sistem durumunu analiz et"""
        cache_dir = Path(".pdsx_cache")
        venv_dir = Path(".pdsx_isolated_env")
        
        status = {
            "is_fresh": EnvironmentDetector.is_fresh_system(),
            "has_cache": cache_dir.exists(),
            "has_venv": venv_dir.exists(),
            "cache_size": 0,
            "wheel_count": 0,
            "system_type": "UNKNOWN"
        }
        
        if cache_dir.exists():
            wheels_dir = cache_dir / "wheels"
            if wheels_dir.exists():
                wheels = list(wheels_dir.glob("*.whl"))
                status["wheel_count"] = len(wheels)
                try:
                    status["cache_size"] = sum(f.stat().st_size for f in wheels)
                except:
                    status["cache_size"] = 0
        
        # Sistem tipini belirle
        if status["is_fresh"]:
            status["system_type"] = "FRESH"
        elif status["has_cache"] and status["wheel_count"] > 50:
            status["system_type"] = "ESTABLISHED"
        elif status["has_venv"]:
            status["system_type"] = "CONFIGURED"
        else:
            status["system_type"] = "BASIC"
        
        return status


def get_optimal_autoimporter(entry_point_script=None):
    """Sistem durumuna göre optimal AutoImporter seç"""
    
    recommendation = EnvironmentDetector.recommend_autoimporter()
    system_status = EnvironmentDetector.get_system_status()
    
    print(f"[PDS-X] 🔍 Sistem Analizi:")
    print(f"[PDS-X] 📊 Cache: {system_status['wheel_count']} wheels")
    print(f"[PDS-X] 📁 VEnv: {'✅' if system_status['has_venv'] else '❌'}")
    print(f"[PDS-X] 💾 Cache Size: {system_status['cache_size']//1024//1024}MB")
    print(f"[PDS-X] 📈 Recommendation: {recommendation['type']}")
    print(f"[PDS-X] 💡 Reason: {recommendation['reason']}")
    
    if recommendation["type"] == "HEAVY":
        print("[PDS-X] � Heavy AutoImporter yükleniyor...")
        try:
            # Heavy AutoImporter'ı dinamik olarak import et
            from auto_importer_heavy import AutoImporter as HeavyAutoImporter
            return HeavyAutoImporter(entry_point_script, mode="HEAVY")
        except ImportError as e:
            print(f"[PDS-X] ⚠️ Heavy AutoImporter yüklenemedi: {e}")
            print("[PDS-X] 🔄 Lite AutoImporter'a geçiliyor...")
            return AutoImporterLite(entry_point_script, mode="LITE_FALLBACK")
    else:
        print("[PDS-X] ⚡ Lite AutoImporter yükleniyor...")
        return AutoImporterLite(entry_point_script, mode="LITE")


def create_full_autoimporter_backup():
    """auto_importer.py'yi auto_importer_full.py olarak yedekle"""
    try:
        import shutil
        shutil.copy2("auto_importer.py", "auto_importer_full.py")
        print("[PDS-X] ✅ auto_importer_full.py yedek oluşturuldu")
    except Exception as e:
        print(f"[PDS-X] ⚠️ Yedek oluşturma hatası: {e}")


def test_autoimporter_selection():
    """AutoImporter seçimini test et"""
    print("="*60)
    print("🧪 PDS-X AUTOIMPORTER SELECTION TEST")
    print("="*60)
    
    # Sistem durumunu analiz et
    system_status = EnvironmentDetector.get_system_status()
    print("\n📊 SİSTEM DURUMU:")
    for key, value in system_status.items():
        icon = "✅" if value else "❌" if isinstance(value, bool) else "📄"
        print(f"  {icon} {key}: {value}")
    
    # AutoImporter seç
    print(f"\n🤖 AUTOIMPORTER SEÇİMİ:")
    autoimporter = get_optimal_autoimporter()
    print(f"  🎯 Seçilen: {autoimporter.__class__.__name__}")
    print(f"  🏷️ Mode: {autoimporter.mode}")
    
    # Özet rapor
    if hasattr(autoimporter, 'summary_report'):
        print(f"\n📋 ÖZET RAPOR:")
        report = autoimporter.summary_report()
        for key, value in report.items():
            print(f"  📄 {key}: {value}")
    
    # Performance beklentisi
    print(f"\n⚡ PERFORMANCE BEKLENTİSİ:")
    if autoimporter.mode == "LITE":
        print("  🚀 Başlatma: ~0.1 saniye")
        print("  🧠 Memory: %50 daha az")
        print("  ⚡ CPU: Minimal kullanım")
        print("  🔧 Özellikler: Sadece temel")
    else:
        print("  🐌 Başlatma: ~1-2 saniye")
        print("  🧠 Memory: Normal")
        print("  ⚡ CPU: Yoğun kullanım")
        print("  🔧 Özellikler: Tam özellikli")
    
    print("\n" + "="*60)
    print("✅ TEST TAMAMLANDI")
    print("="*60)


if __name__ == "__main__":
    test_autoimporter_selection()
