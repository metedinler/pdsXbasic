"""
base_module_manager.py - PDS-X BASIC v14u Temel Modül Yönetim Sınıfı
Version: 1.0.0
Date: June 9, 2025
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
import shutil

class BaseModuleManager:
    """Temel modül yönetim sınıfı. AutoImporter ve AutoInstaller için temel sınıf."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.dependency_file = self.workspace_path / "dependencies.json"
        self.env_path = self.workspace_path / ".pdsX_isolated_env"
        self._load_dependencies()
        
    def _load_dependencies(self) -> None:
        """Bağımlılık dosyasını yükler veya oluşturur."""
        try:
            if self.dependency_file.exists():
                with open(self.dependency_file, 'r', encoding='utf-8') as f:
                    self.dependencies = json.load(f)
            else:
                self.dependencies = {
                    "modules": {},
                    "installed_packages": {}
                }
                self._save_dependencies()
        except Exception as e:
            logging.error(f"Bağımlılık dosyası yükleme hatası: {str(e)}")
            self.dependencies = {"modules": {}, "installed_packages": {}}
            
    def _save_dependencies(self) -> None:
        """Bağımlılıkları dosyaya kaydeder."""
        try:
            with open(self.dependency_file, 'w', encoding='utf-8') as f:
                json.dump(self.dependencies, f, indent=4)
        except Exception as e:
            logging.error(f"Bağımlılık dosyası kaydetme hatası: {str(e)}")
            
    def get_module_dependencies(self, module_name: str) -> Dict:
        """Bir modülün bağımlılıklarını döndürür."""
        return self.dependencies["modules"].get(module_name, {})
        
    def update_module_dependencies(self, module_name: str, deps: Dict) -> None:
        """Bir modülün bağımlılıklarını günceller."""
        if module_name not in self.dependencies["modules"]:
            self.dependencies["modules"][module_name] = {}
        self.dependencies["modules"][module_name].update(deps)
        self._save_dependencies()
        
    def is_package_installed(self, package: str, version: Optional[str] = None) -> bool:
        """Bir paketin kurulu olup olmadığını kontrol eder."""
        installed_version = self.dependencies["installed_packages"].get(package)
        if not installed_version:
            return False
        if version and version != "latest":
            from pkg_resources import parse_version
            return parse_version(installed_version) >= parse_version(version)
        return True
        
    def install_package(self, package: str, version: Optional[str] = None) -> bool:
        """Güvenli paket kurulumu yapar."""
        try:
            if version and version != "latest":
                package_spec = f"{package}=={version}"
            else:
                package_spec = package
                
            python_path = str(self.env_path / "Scripts" / "python.exe") if os.name == "nt" else str(self.env_path / "bin" / "python")
            
            result = subprocess.run(
                [python_path, "-m", "pip", "install", package_spec],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.dependencies["installed_packages"][package] = version or "latest"
                self._save_dependencies()
                logging.info(f"Paket kuruldu: {package_spec}")
                return True
            else:
                logging.error(f"Paket kurulum hatası: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Paket kurulum hatası ({package}): {str(e)}")
            return False
            
    def check_environment(self) -> bool:
        """Python sürümü ve ortam kontrolü yapar."""
        python_version = sys.version_info
        if not (python_version.major == 3 and python_version.minor == 10):
            logging.error("Python 3.10 gerekli!")
            return False
            
        if not self.env_path.exists():
            try:
                import venv
                venv.create(self.env_path, with_pip=True)
                logging.info(f"Yeni ortam oluşturuldu: {self.env_path}")
            except Exception as e:
                logging.error(f"Ortam oluşturma hatası: {str(e)}")
                return False
                
        return True
