"""
autoinstaller.py - PDS-X BASIC v14u Otomatik Paket Yükleyici
Version: 1.1.0
Date: June 9, 2025
"""

import os
import sys
import ast
import json
import venv
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional

from base_module_manager import BaseModuleManager
from pdsx_exception import PdsXException

# PDS-X Temel Kütüphane Listesi
CORE_PACKAGES = {
    "required": [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "tensorflow",
        "graphviz",
        "requests",
        "aiohttp",
        "websockets",
        "psycopg2-binary",
        "pyyaml"
    ],
    "optional": [
        "transformers",
        "nltk",
        "spacy",
        "gensim"
    ]
}

# Modül-Kütüphane Eşleştirmeleri
MODULE_PACKAGES = {
    "libx_ml.py": ["torch", "transformers", "scikit-learn"],
    "libx_nlp.py": ["nltk", "spacy", "gensim"],
    "database_sql_isam.py": ["psycopg2-binary", "sqlite3"],
    "graph.py": ["networkx", "graphviz"],
}

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_installer.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("autoinstaller")

class ModuleAnalyzer:
    """Modül analiz sınıfı."""
    
    def __init__(self):
        self.import_cache = {}
        
    def analyze_imports(self, module_path: str) -> Set[str]:
        """Modüldeki importları analiz eder."""
        if module_path in self.import_cache:
            return self.import_cache[module_path]
            
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            self.import_cache[module_path] = imports
            return imports
            
        except Exception as e:
            logging.error(f"Modül analiz hatası ({module_path}): {str(e)}")
            return set()

class DependencyManager:
    def __init__(self, workspace_path):
        self.workspace_path = Path(workspace_path)
        self.dependency_file = self.workspace_path / "dependencies.json"
        self.venv_path = self.workspace_path / ".pdsX_isolated_env"  # Doğru venv yolu
        self.module_analyzer = ModuleAnalyzer()
        self.installed_packages = {}  # {package_name: version}
        self.module_dependencies = {}  # {module_name: set(package_names)}
        self._load_dependencies()
        self._setup_venv()

    def _load_dependencies(self) -> None:
        """Kayıtlı bağımlılıkları yükler."""
        try:
            if self.dependency_file.exists():
                with open(self.dependency_file, 'r') as f:
                    data = json.load(f)
                    self.installed_packages = data.get("installed_packages", {})
                    self.module_dependencies = {
                        k: set(v) for k, v in data.get("module_dependencies", {}).items()
                    }
        except Exception as e:
            log.error(f"Bağımlılık dosyası yükleme hatası: {str(e)}")

    def _save_dependencies(self) -> None:
        """Bağımlılıkları kaydeder."""
        try:
            data = {
                "installed_packages": self.installed_packages,
                "module_dependencies": {
                    k: list(v) for k, v in self.module_dependencies.items()
                }
            }
            with open(self.dependency_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            log.error(f"Bağımlılık dosyası kaydetme hatası: {str(e)}")

    def _setup_venv(self) -> None:
        """Sanal ortam oluşturur veya var olanı kullanır."""
        if not self.venv_path.exists():
            try:
                venv.create(self.venv_path, with_pip=True)
                log.info(f"Sanal ortam oluşturuldu: {self.venv_path}")
            except Exception as e:
                log.error(f"Sanal ortam oluşturma hatası: {str(e)}")
                raise PdsXException(f"Sanal ortam oluşturma hatası: {str(e)}")

    def _get_venv_python(self) -> str:
        """Sanal ortamdaki Python yorumlayıcısının yolunu döndürür."""
        if sys.platform == "win32":
            return str(self.venv_path / "Scripts" / "python.exe")
        return str(self.venv_path / "bin" / "python")

    def check_conflicts(self, packages: List[str]) -> List[str]:
        """Paket çakışmalarını kontrol eder."""
        conflicts = []
        for package in packages:
            name = package.split("==")[0]
            if name in self.installed_packages:
                for module, deps in self.module_dependencies.items():
                    if name in deps:
                        conflicts.append(f"{package} paketi {module} modülü tarafından kullanılıyor")
        return conflicts

    def install_packages(self, module_name: str, packages: List[str]) -> bool:
        """Modül için gerekli paketleri yükler."""
        try:
            # Çakışmaları kontrol et
            conflicts = self.check_conflicts(packages)
            if conflicts:
                log.error(f"Paket çakışmaları: {conflicts}")
                raise PdsXException(f"Paket çakışmaları: {', '.join(conflicts)}")

            # Paketleri yükle
            python_exe = self._get_venv_python()
            for package in packages:
                if package not in self.installed_packages:
                    cmd = [python_exe, "-m", "pip", "install", package]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise PdsXException(f"Paket yükleme hatası: {result.stderr}")
                    name = package.split("==")[0]
                    self.installed_packages[name] = package

            # Modül bağımlılıklarını kaydet
            package_names = {p.split("==")[0] for p in packages}
            self.module_dependencies[module_name] = package_names
            self._save_dependencies()
            
            log.info(f"{module_name} için {len(packages)} paket yüklendi")
            return True
        except Exception as e:
            log.error(f"Paket yükleme hatası ({module_name}): {str(e)}")
            return False

    def uninstall_module_packages(self, module_name: str) -> bool:
        """Modüle ait paketleri kaldırır."""
        try:
            if (module_name not in self.module_dependencies):
                return True

            packages = self.module_dependencies[module_name]
            python_exe = self._get_venv_python()

            # Paketleri başka modüller kullanıyor mu kontrol et
            for package in packages:
                other_users = [
                    m for m, deps in self.module_dependencies.items()
                    if m != module_name and package in deps
                ]
                if not other_users:
                    cmd = [python_exe, "-m", "pip", "uninstall", "-y", package]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise PdsXException(f"Paket kaldırma hatası: {result.stderr}")
                    self.installed_packages.pop(package, None)

            # Modül kayıtlarını temizle
            del self.module_dependencies[module_name]
            self._save_dependencies()
            
            log.info(f"{module_name} için paketler kaldırıldı")
            return True
        except Exception as e:
            log.error(f"Paket kaldırma hatası ({module_name}): {str(e)}")
            return False

    def get_module_environment(self, module_name: str) -> Dict[str, str]:
        """Modül için ortam değişkenlerini döndürür."""
        if module_name not in self.module_dependencies:
            return {}

        return {
            "PYTHONPATH": str(self.venv_path / "Lib" / "site-packages"),
            "VIRTUAL_ENV": str(self.venv_path),
            "PATH": os.environ["PATH"] + os.pathsep + str(self.venv_path / "Scripts")
        }

    def analyze_new_module(self, module_path):
        """Yeni bir modülü analiz eder ve bağımlılıklarını belirler."""
        log.info(f"Yeni modül analiz ediliyor: {module_path}")
        
        # İmportları analiz et
        imports = self.module_analyzer.analyze_imports(module_path)
        
        # Temel paketleri kontrol et
        required_packages = set()
        for imp in imports:
            if imp in CORE_PACKAGES["required"]:
                required_packages.add(imp)
            elif imp in CORE_PACKAGES["optional"]:
                required_packages.add(imp)
                
        # Özel modül bağımlılıklarını kontrol et
        module_name = Path(module_path).name
        if module_name in MODULE_PACKAGES:
            required_packages.update(MODULE_PACKAGES[module_name])
            
        return required_packages

    def install_dependencies(self, packages):
        """Belirlenen bağımlılıkları güvenli şekilde kurar."""
        success = True
        for package in packages:
            if not self._install_package(package):
                success = False
                log.error(f"Paket kurulumu başarısız: {package}")
                
        return success
        
    def _install_package(self, package):
        """Tek bir paketi güvenli şekilde kurar."""
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.installed_packages[package] = "installed"
                log.info(f"Paket kuruldu: {package}")
                return True
            else:
                log.error(f"Paket kurulum hatası: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Paket kurulum hatası ({package}): {str(e)}")
            return False

    def update_exports(self, module_path):
        """Modülün exports tanımını günceller."""
        try:
            from add_exports import add_exports_to_file
            add_exports_to_file(module_path)
            log.info(f"Exports tanımı güncellendi: {module_path}")
            return True
        except Exception as e:
            log.error(f"Exports güncelleme hatası: {str(e)}")
            return False

class PackageManager:
    """Paket kurulum ve yönetiminden sorumlu sınıf."""
    
    def __init__(self, venv_path: Path):
        self.venv_path = venv_path
        self.pip_path = self._get_pip_path()
        self.installed_cache: Dict[str, str] = {}  # {package: version}
        self._load_installed_packages()
        
    def _get_pip_path(self) -> str:
        """Virtual env içindeki pip yolunu bulur."""
        if os.name == 'nt':
            return str(self.venv_path / "Scripts" / "pip.exe")
        return str(self.venv_path / "bin" / "pip")
        
    def _load_installed_packages(self) -> None:
        """Kurulu paketleri ve versiyonlarını yükler."""
        try:
            result = subprocess.run(
                [self.pip_path, "list", "--format=json"],
                capture_output=True,
                text=True
            )
            packages = json.loads(result.stdout)
            self.installed_cache = {
                p["name"]: p["version"] for p in packages
            }
        except Exception as e:
            log.error(f"Paket listesi yükleme hatası: {str(e)}")
            
    def install_safe(self, package: str) -> bool:
        """Güvenli paket kurulumu yapar."""
        try:
            if package in self.installed_cache:
                log.info(f"Paket zaten kurulu: {package}")
                return True
                
            log.info(f"Paket kuruluyor: {package}")
            result = subprocess.run(
                [self.pip_path, "install", package],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.installed_cache[package] = "unknown"
                log.info(f"Paket kuruldu: {package}")
                return True
            else:
                log.error(f"Paket kurulum hatası: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Paket kurulum hatası ({package}): {str(e)}")
            return False
            
    def check_conflicts(self, packages: List[str]) -> List[str]:
        """Paket çakışmalarını kontrol eder."""
        conflicts = []
        for pkg in packages:
            try:
                result = subprocess.run(
                    [self.pip_path, "check"],
                    capture_output=True,
                    text=True
                )
                if pkg in result.stdout and "conflict" in result.stdout.lower():
                    conflicts.append(pkg)
            except:
                pass
        return conflicts

class ModuleAnalyzer:
    """Yeni modüllerin bağımlılıklarını analiz eder."""
    
    def __init__(self):
        self.import_cache: Dict[str, Set[str]] = {}
        
    def analyze_imports(self, module_path: str) -> Set[str]:
        """Modül dosyasındaki import ifadelerini analiz eder."""
        if module_path in self.import_cache:
            return self.import_cache[module_path]
            
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            self.import_cache[module_path] = imports
            return imports
            
        except Exception as e:
            log.error(f"Modül analiz hatası ({module_path}): {str(e)}")
            return set()
            
    def detect_dependencies(self, imports: Set[str]) -> Set[str]:
        """Import edilen modüllerin hangi pip paketlerine ait olduğunu belirler."""
        required_packages = set()
        
        for imp in imports:
            # Standart kütüphane kontrolü
            if self._is_stdlib_module(imp):
                continue
                
            # Bilinen paket eşleştirmeleri
            if imp in self._get_package_mapping():
                required_packages.add(self._get_package_mapping()[imp])
                
        return required_packages
        
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Bir modülün Python standart kütüphanesine ait olup olmadığını kontrol eder."""
        try:
            module_info = sys.modules.get(module_name)
            if module_info:
                module_path = getattr(module_info, '__file__', '')
                return 'site-packages' not in str(module_path)
            return False
        except:
            return False
            
    def _get_package_mapping(self) -> Dict[str, str]:
        """Modül-Paket eşleştirmelerini döndürür."""
        return {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'graphviz': 'graphviz',
            'requests': 'requests',
            'aiohttp': 'aiohttp',
            'websockets': 'websockets',
            'psycopg2': 'psycopg2-binary',
            'yaml': 'pyyaml',
            'transformers': 'transformers',
            'nltk': 'nltk',
            'spacy': 'spacy',
            'gensim': 'gensim',
            'networkx': 'networkx'
        }

class AutoInstaller(BaseModuleManager):
    """Yeni modüllerin analizi ve bağımlılık yönetimi."""
    
    def __init__(self, workspace_path: str):
        super().__init__(workspace_path)
        self.analyzer = ModuleAnalyzer()
        
    def analyze_new_module(self, module_path: str) -> Dict:
        """Yeni modülü analiz eder ve bağımlılıklarını tespit eder."""
        module_name = Path(module_path).name
        logging.info(f"Yeni modül analiz ediliyor: {module_name}")
        
        # İmportları analiz et
        imports = self.analyzer.analyze_imports(module_path)
        
        # Bağımlılıkları belirle
        dependencies = self._detect_package_dependencies(imports)
        
        # Modül bilgilerini güncelle
        self.update_module_dependencies(module_name, {
            "imports": list(imports),
            "dependencies": dependencies,
            "last_analyzed": "2025-06-09"  # TODO: Gerçek tarih ekle
        })
        
        return dependencies
        
    def install_module_dependencies(self, module_path: str) -> bool:
        """Modül için gerekli bağımlılıkları kurar."""
        try:
            # Önce modülü analiz et
            dependencies = self.analyze_new_module(module_path)
            
            # Eksik bağımlılıkları kur
            for package, version in dependencies.items():
                if not self.is_package_installed(package, version):
                    if not self.install_package(package, version):
                        logging.error(f"Paket kurulumu başarısız: {package}")
                        return False
                        
            return True
            
        except Exception as e:
            logging.error(f"Bağımlılık kurulum hatası: {str(e)}")
            return False
            
    def _detect_package_dependencies(self, imports: Set[str]) -> Dict[str, str]:
        """Import edilen modüllerin pip paket karşılıklarını bulur."""
        dependencies = {}
        
        # Bilinen paket eşleştirmeleri
        PACKAGE_MAP = {
            "numpy": ("numpy", "1.26.4"),
            "pandas": ("pandas", "2.1.4"),
            "sklearn": ("scikit-learn", "1.3.2"),
            "torch": ("torch", "2.2.2"),
            "tensorflow": ("tensorflow", "2.15.0"),
            "transformers": ("transformers", "latest"),
            "nltk": ("nltk", "latest"),
            "spacy": ("spacy", "latest"),
            "gensim": ("gensim", "latest"),
            "networkx": ("networkx", "latest"),
            "graphviz": ("graphviz", "latest"),
            # ... diğer eşleştirmeler ...
        }
        
        for imp in imports:
            if imp in PACKAGE_MAP:
                package, version = PACKAGE_MAP[imp]
                dependencies[package] = version
                
        return dependencies

# Ana program başlangıcı
if __name__ == "__main__":
    print("autoinstaller.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")
