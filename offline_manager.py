"""
PDS-X Çevrimdışı Mod Yöneticisi
"""
from pathlib import Path
import json
import shutil
import hashlib
import time
from typing import Dict, List, Optional

class OfflineModeManager:
    """Çevrimdışı mod yönetimi sınıfı."""
    
    def __init__(self, cache_dir: str = ".pdsx_cache"):
        self.cache_dir = Path(cache_dir)
        self.packages_dir = self.cache_dir / "packages"
        self.metadata_file = self.cache_dir / "offline_metadata.json"
        self.offline_mode = False
        
        # Dizinleri oluştur
        self._ensure_dirs()
        self._load_metadata()
        
    def _ensure_dirs(self):
        """Gerekli dizinleri oluşturur."""
        self.cache_dir.mkdir(exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
        
    def _load_metadata(self):
        """Meta veriyi yükler."""
        if self.metadata_file.exists():
            try:
                self.metadata = json.loads(self.metadata_file.read_text())
            except json.JSONDecodeError:
                self.metadata = {"packages": {}, "last_sync": 0}
        else:
            self.metadata = {"packages": {}, "last_sync": 0}
            
    def _save_metadata(self):
        """Meta veriyi kaydeder."""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))
        
    def enable_offline_mode(self):
        """Çevrimdışı modu etkinleştirir."""
        self.offline_mode = True
        
    def disable_offline_mode(self):
        """Çevrimdışı modu devre dışı bırakır."""
        self.offline_mode = False
        
    def is_offline(self) -> bool:
        """Çevrimdışı modun durumunu döndürür."""
        return self.offline_mode
        
    def cache_package(self, package_name: str, version: str, content: bytes):
        """Paketi önbelleğe alır."""
        pkg_hash = self._calculate_hash(content)
        cache_path = self.packages_dir / f"{package_name}-{version}.whl"
        
        # Paketi kaydet
        cache_path.write_bytes(content)
        
        # Meta veriyi güncelle
        self.metadata["packages"][f"{package_name}-{version}"] = {
            "cached_at": time.time(),
            "hash": pkg_hash,
            "size": len(content)
        }
        self._save_metadata()
        
    def get_cached_package(self, package_name: str, version: str) -> Optional[bytes]:
        """Önbellekteki paketi döndürür."""
        cache_path = self.packages_dir / f"{package_name}-{version}.whl"
        if cache_path.exists():
            return cache_path.read_bytes()
        return None
        
    def is_package_cached(self, package_name: str, version: str) -> bool:
        """Paketin önbellekte olup olmadığını kontrol eder."""
        return (self.packages_dir / f"{package_name}-{version}.whl").exists()
        
    def get_cached_packages(self) -> List[Dict]:
        """Önbellekteki tüm paketlerin listesini döndürür."""
        packages = []
        for pkg_file in self.packages_dir.glob("*.whl"):
            pkg_name = pkg_file.stem
            if pkg_name in self.metadata["packages"]:
                info = self.metadata["packages"][pkg_name]
                packages.append({
                    "name": pkg_name,
                    "cached_at": info["cached_at"],
                    "size": info["size"]
                })
        return packages
        
    def clear_cache(self, older_than_days: int = None):
        """Önbelleği temizler."""
        if older_than_days:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            for pkg_name, info in list(self.metadata["packages"].items()):
                if info["cached_at"] < cutoff_time:
                    pkg_path = self.packages_dir / f"{pkg_name}.whl"
                    if pkg_path.exists():
                        pkg_path.unlink()
                    del self.metadata["packages"][pkg_name]
        else:
            shutil.rmtree(self.packages_dir)
            self.packages_dir.mkdir()
            self.metadata["packages"] = {}
            
        self._save_metadata()
        
    def _calculate_hash(self, content: bytes) -> str:
        """Paket içeriği için hash değeri hesaplar."""
        return hashlib.sha256(content).hexdigest()
        
    def verify_package(self, package_name: str, version: str) -> bool:
        """Paket bütünlüğünü doğrular."""
        pkg_key = f"{package_name}-{version}"
        if pkg_key not in self.metadata["packages"]:
            return False
            
        cache_path = self.packages_dir / f"{pkg_key}.whl"
        if not cache_path.exists():
            return False
            
        content = cache_path.read_bytes()
        current_hash = self._calculate_hash(content)
        return current_hash == self.metadata["packages"][pkg_key]["hash"]
