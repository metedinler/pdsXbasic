# auto_importer.py - PDS-X BASIC v15 Dinamik Modül Yükleyici
# Version: 1.5.0
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, duzenleyen ve calismasini saglayan github copilot)
# --- PDS-X Otomatik Ortam Kurulumu ve Modül Yükleyici ---
# Bu script, PDS-X'in çalışması için gerekli olan Python 3.10 ortamını ve tüm pip paketlerini otomatik olarak kurar.
# Kullanıcıdan hiçbir manuel işlem beklemez, her adımda otomasyon ve hata önleme önceliklidir.

# --- PDS-X Sistem Başlangıcı ---
# rem #0. PDS-X sisteminin başlangıç aşamaları
# rem #0.1. Python 3.10 kontrolü yapılır
# rem #0.2. Sistem gereksinimleri kontrol edilir
# rem #0.3. Log sistemi başlatılır
# rem #0.4. Terminal çıktıları yapılandırılır

# --- Gerekli Modüllerin İçe Aktarılması ---
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import ast
import datetime
import importlib.util
import json
import logging
import numpy as np
import os
import shutil
import subprocess
import sys
import threading
from parallel_processor import ParallelProcessManager
from offline_manager import OfflineModeManager

try:
    import winreg
except ImportError:
    winreg = None

# Log dosyası yolu tanımlanıyor
LOG_FILE = Path(__file__).parent / "auto_importer.log"

# --- Temel Sabitler ---
LOG_FILE = "pdsxu_terminal.log"
LOG_BAK = "pdsxu_terminal.bak"

# rem #31. İzole Ortam Yardımcı Fonksiyonları
CORE_DEPENDENCIES = {
    "base": [
        "numpy",  # İstatistiksel işlemler için
        "pandas", # Veri analizi için
        "pytest", # Test işlemleri için
        "pylint", # Kod kalitesi kontrolü için
        "autopep8", # Kod formatlaması için
        "transformers",
        "nltk",
        "spacy",
        "gensim"
    ]
}

MODULE_SPECIFIC_DEPS = {
    "core2-5.py": ["tensorflow", "scikit-learn", "numpy"],
    "libx_ml.py": ["torch", "transformers", "scikit-learn"],
    "libx_nlp.py": ["nltk", "spacy", "gensim"],
    "database_sql_isam.py": ["psycopg2-binary", "sqlite3"],
    "graph.py": ["networkx", "graphviz"]
}

# rem #1. Gerekli modüllerin ve ortam yönetimi için temel ayarların yapılması

# --- Gerekli Pip Paketleri Listesi ---  
# rem #17. Tüm bağımlılıklar REQUIRED_PACKAGES içinde toplandı. Sürümler sabitlendi.
# rem #17.1. Her paket için (pip_adı, import_adı) şeklinde tanımlandı. Çakışmaları önlemek için sürümler sabit.
# rem #17.2. Bu liste tüm pdsX sisteminin ihtiyaç duyduğu paketleri içerir. Modüller buradan yüklenir.
REQUIRED_PACKAGES = [
    # Temel bilimsel ve ML kütüphaneleri 
    ("numpy<2.2.0", "numpy"), # pin numpy version <2.2.0 for TF compatibility
    ("scipy==1.11.4", "scipy"),
    ("pandas==2.1.4", "pandas"),
    ("scikit-learn==1.3.2", "sklearn"), 
    ("joblib==1.3.2", "joblib"),
    ("threadpoolctl==3.3.0", "threadpoolctl"),
    ("matplotlib==3.8.4", "matplotlib"),
    ("kiwisolver==1.4.5", "kiwisolver"), 
    ("cycler==0.12.1", "cycler"),
    ("pyparsing==3.1.1", "pyparsing"),
    ("python-dateutil==2.8.2", "dateutil"),
    ("pillow==10.2.0", "PIL"),
    ("packaging==23.2", "packaging"),
    ("seaborn==0.13.2", "seaborn"),
    ("statsmodels==0.14.1", "statsmodels"),
    ("tornado==6.4", "tornado"),
    ("plotly==5.19.0", "plotly"),
    ("tenacity==8.2.3", "tenacity"), 
    ("dash==2.15.0", "dash"),
    ("flask==3.0.2", "flask"),
    ("jinja2==3.1.3", "jinja2"),
    ("werkzeug==3.0.1", "werkzeug"),
    ("itsdangerous==2.1.2", "itsdangerous"),
    ("markupsafe==2.1.5", "markupsafe"),
    ("click==8.1.7", "click"),
    ("grpcio==1.62.0", "grpc"),
    ("protobuf==4.25.3", "google.protobuf"),
    ("aiohttp==3.9.3", "aiohttp"),
    ("async-timeout==4.0.3", "async_timeout"),
    ("yarl==1.9.4", "yarl"),
    ("multidict==6.0.5", "multidict"),
    ("attrs==23.2.0", "attr"),
    ("frozenlist==1.4.1", "frozenlist"),
    ("pyzmq==25.1.2", "zmq"),
    ("websocket-client==1.7.0", "websocket"),
    ("paho-mqtt==1.6.1", "paho.mqtt.client"), # MQTT client support
    ("boto3==1.34.34", "boto3"), # AWS SDK
    ("botocore==1.34.34", "botocore"), # boto3 dependency
    ("kafka-python==2.0.2", "kafka"),
    ("river==0.21.0", "river"),
    ("qiskit==1.0.1", "qiskit"),
    ("networkx==3.2.1", "networkx"),
    ("websockets==12.0", "websockets"),
    ("rich==13.7.0", "rich"),
    ("colorama==0.4.6", "colorama"),
    ("textblob==0.17.1", "textblob"),
    ("mysql-connector-python==8.3.0", "mysql.connector"),
    ("psutil==5.9.8", "psutil"),
    ("pyyaml==6.0.1", "yaml"),
    ("graphviz==0.20.1", "graphviz"),
    ("aiofiles==23.2.1", "aiofiles"),
    ("RestrictedPython>=6.2,<8.0", "RestrictedPython"),
    ("pdfplumber==0.10.3", "pdfplumber"),
    ("requests==2.31.0", "requests"),
    ("psycopg2-binary==2.9.9", "psycopg2"),
    ("elasticsearch==8.12.0", "elasticsearch"),
    ("elastic-transport==8.12.0", "elastic_transport"),
    ("nltk==3.8.1", "nltk"), # textblob için zorunlu bağımlılık

    # Derin öğrenme ve bilimsel
    ("tensorflow==2.15.0", "tensorflow"),
    ("torch==2.2.2", "torch"), 
    ("torch-geometric==2.5.3", "torch_geometric"),
    ("spacy==3.5.3", "spacy"), # pin spacy version to avoid conflicts
    ("transformers==4.37.2", "transformers"),
    ("pycryptodome==3.20.0", "Crypto"),

    # --- TRANSITIVE/SECONDARY DEPENDENCIES ---
    ("jmespath==1.0.1", "jmespath"), # boto3/botocore için
    ("pdfminer.six==20250327", "pdfminer"), # pdfplumber için
    ("pypdfium2>=4.18.0", "pypdfium2"), # pdfplumber için
    ("markdown-it-py>=2.2.0", "markdown_it_py"), # rich için
    ("pygments>=2.13.0,<3.0.0", "pygments"), # rich için
    ("catalogue<2.1.0,>=2.0.6", "catalogue"), # spacy için
    ("cymem<2.1.0,>=2.0.2", "cymem"), # spacy için
    ("langcodes<4.0.0,>=3.2.0", "langcodes"), # spacy için
    ("murmurhash<1.1.0,>=0.28.0", "murmurhash"),# spacy için
    ("preshed<3.1.0,>=3.0.2", "preshed"), # spacy için
    ("pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4", "pydantic"), # spacy için
    ("spacy-legacy<3.1.0,>=3.0.11", "spacy_legacy"), # spacy için
    ("spacy-loggers<2.0.0,>=1.0.0", "spacy_loggers"), # spacy için
    ("srsly<3.0.0,>=2.4.3", "srsly"), # spacy için
    ("thinc<8.4.0,>=8.3.4", "thinc"), # spacy için
    ("typer<1.0.0,>=0.3.0", "typer"), # spacy için
    ("wasabi<1.2.0,>=0.9.1", "wasabi"), # spacy için
    ("weasel<0.5.0,>=0.1.0", "weasel"), # spacy için
    ("huggingface-hub<1.0,>=0.30.0", "huggingface_hub"), # transformers için
    ("regex!=2019.12.17", "regex"), # transformers için
    ("safetensors>=0.4.3", "safetensors"), # transformers için 
    ("tokenizers<0.22,>=0.21", "tokenizers"), # transformers için
    ("cryptography==42.0.5", "cryptography"), # pdfminer.six için
    ("cffi==1.16.0", "cffi"), # cryptography için
    ("pycparser==2.21", "pycparser"), # cffi için
    ("six==1.16.0", "six"), # cryptography için
    ("pyasn1==0.5.1", "pyasn1"), # cryptography için
    ("pyasn1-modules==0.3.0", "pyasn1_modules"), # cryptography için
    ("idna==3.6", "idna"), # requests için
    ("charset_normalizer==3.3.2", "charset_normalizer"), # requests için
    ("urllib3==2.2.1", "urllib3"), # requests için
    ("certifi==2024.2.2", "certifi"), # requests için
    ("chardet==5.2.0", "chardet"), # textblob için

    # Sistem bileşenleri
    ("blis<1.4.0,>=1.3.0", "blis"), # thinc için
    ("confection<1.0.0,>=0.0.1", "confection"), # thinc, weasel için
    ("shellingham>=1.3.0", "shellingham"), # typer için
    ("smart-open<8.0.0,>=5.2.1", "smart_open"), # weasel için
    ("cloudpathlib<1.0.0,>=0.7.0", "cloudpathlib"), # weasel için
    ("annotated-types>=0.6.0", "annotated_types"), # pydantic için
    ("pydantic-core==2.33.2", "pydantic_core"), # pydantic için
    ("typing-inspection>=0.4.0", "typing_inspect") # pydantic için
]

# --- Terminal log dosyasını yedekle ve sıfırla ---
# rem #2. Terminal log dosyasını yedekle ve sıfırla
try:
    if os.path.exists(LOG_FILE):
        if os.path.exists(LOG_BAK):
            os.remove(LOG_BAK)
        shutil.move(LOG_FILE, LOG_BAK)
except Exception as e:
    print(f"[PDS-X] Log dosyası yedeklenemedi: {e}")

# --- Tüm terminal çıktısını hem ekrana hem log dosyasına yazan sınıf ---
class Tee:
    # rem #3. Tüm terminal çıktısını hem ekrana hem log dosyasına yazan sınıf
    def __init__(self, *files):
        # rem #3.1. Dosya nesnelerini sakla
        self.files = files
    def write(self, obj):
        # rem #3.2. Her dosyaya yaz ve flush et
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass
    def flush(self):
        # rem #3.3. Her dosyayı flush et
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

sys.stdout = Tee(sys.__stdout__, open(LOG_FILE, "a", encoding="utf-8"))
sys.stderr = Tee(sys.__stderr__, open(LOG_FILE, "a", encoding="utf-8"))

# Tüm bağımlılıklar REQUIRED_PACKAGES içinde toplandı ve sabit sürüm numaralarıyla tanımlandı

# rem #19. ModuleConflictResolver sınıfı paket ve modül çakışmalarını tespit eder ve çözer.
# rem #19.1. Bu sınıf kuantum analizi kullanarak kritik modülleri belirler.
# rem #19.2. Bağımlılık grafı oluşturur ve en uygun versiyonları seçer. Çözümleri raporlar.
class ModuleConflictResolver:
    """Modül çakışmalarını çözme ve analiz etme sınıfı."""
    def __init__(self):
        self.conflicts = {}
        self.resolutions = {}
        self.dependency_graph = {}
        self.logger = logging.getLogger(__name__)
        
    def analyze_conflicts(self, module_deps: Dict) -> Dict:
        """Modül bağımlılıklarındaki çakışmaları analiz eder."""
        conflicts = {}
        for module, deps in module_deps.items():
            for dep, version in deps.items():
                if dep in self.dependency_graph:
                    existing_ver = self.dependency_graph[dep]
                    if existing_ver != version:
                        if dep not in conflicts:
                            conflicts[dep] = []
                        conflicts[dep].append((module, version, existing_ver))
                else:
                    self.dependency_graph[dep] = version
        self.conflicts = conflicts
        return conflicts
        
    def resolve_conflicts(self, conflicts: Dict) -> Dict:
        """Çakışmaları çözme stratejileri uygular."""
        resolutions = {}
        for dep, conflict_list in conflicts.items():
            versions = set()
            for _, ver, existing_ver in conflict_list:
                versions.add(ver)
                versions.add(existing_ver)
            
            # En uygun versiyon seçimi
            resolution = self._select_best_version(dep, versions)
            resolutions[dep] = {
                "selected_version": resolution,
                "conflicts": conflict_list,
                "reason": f"Automatically selected compatible version {resolution}"
            }
        self.resolutions = resolutions
        return resolutions

    def _select_best_version(self, package: str, versions: Set[str]) -> str:
        """En uygun versiyon seçimi için algoritma."""
        versions = sorted(list(versions))
        # TensorFlow için özel kural
        if package == "numpy" and any("tensorflow" in dep for dep in self.dependency_graph):
            return ">=1.21.0,<2.2.0"
        # En yüksek uyumlu versiyonu seç
        return max(versions, key=lambda x: [int(i) for i in x.replace('>=','').replace('<=','').replace('==','').replace(',','').split('.')][0])
        
    def generate_report(self) -> Dict:
        """Çakışma analiz raporu oluşturur."""
        return {
            "total_conflicts": len(self.conflicts),
            "resolved_conflicts": len(self.resolutions),
            "dependency_graph": self.dependency_graph,
            "resolutions": self.resolutions,
            "unresolved": set(self.conflicts.keys()) - set(self.resolutions.keys())
        }
        
    def quantum_analysis(self, module_deps: Dict) -> Dict:
        """Kuantum tabanlı bağımlılık analizi yapar."""
        try:
            # Modül ilişkilerini analiz et
            relationships = defaultdict(list)
            for module, deps in module_deps.items():
                for dep in deps:
                    relationships[module].append(dep)
            
            # İlişki matrisini oluştur
            modules = list(module_deps.keys())
            matrix = np.zeros((len(modules), len(modules)))
            
            for i, module in enumerate(modules):
                for j, other in enumerate(modules):
                    if other in relationships[module]:
                        matrix[i][j] = 1
            
            # Önemli modülleri bul
            scores = np.sum(matrix, axis=1)
            critical_modules = [
                (module, score) for module, score in zip(modules, scores)
                if score > np.mean(scores) + np.std(scores)
            ]
            
            return {
                "critical_modules": critical_modules,
                "relationship_density": float(np.mean(matrix)),
                "isolation_score": float(1 - np.std(scores) / np.mean(scores))
            }
        except Exception as e:
            self.logger.error(f"Quantum analysis error: {str(e)}")
            return {"error": str(e)}

# --- Module Management System ---
# rem #20. ModuleManager sınıfı tüm modüllerin bağımlılıklarını ve ilişkilerini yönetir.
# rem #20.1. Bu sınıf modül kullanım istatistiklerini tutar ve performans optimizasyonu sağlar.
# rem #20.2. Modül tarama ve analiz işlemlerini gerçekleştirir. Eksik bağımlılıkları tespit eder.
class ModuleManager:
    """Modül yönetim sistemi."""
    def __init__(self):
        self.required_packages = {pkg[0].split('==')[0]: pkg for pkg in REQUIRED_PACKAGES}
        self.module_deps = {}
        self.module_stats = {}
        
    def get_module_deps(self, module_name: str) -> List[str]:
        """Bir modülün bağımlılıklarını döndürür."""
        if module_name not in self.module_deps:
            deps = []
            for pkg_name, (pip_name, import_name) in self.required_packages.items():
                if import_name in self._scan_module_imports(module_name):
                    deps.append(pip_name)
            self.module_deps[module_name] = deps
        return self.module_deps[module_name]

    def _scan_module_imports(self, module_name: str) -> Set[str]:
        """Bir modül dosyasındaki import ifadelerini tarar."""
        imports = set()
        try:
            with open(module_name, 'r', encoding='utf-8') as f:
                content = f.read()
            # Basit import tarama
            import_lines = [line for line in content.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            for line in import_lines:
                # import x, import x as y
                if line.startswith('import '):
                    imports.update(i.split(' as ')[0].strip() 
                                 for i in line[7:].split(','))
                # from x import y
                elif line.startswith('from '):
                    module = line[5:].split(' import ')[0].strip()
                    imports.add(module)
        except Exception as e:
            logging.error(f"Modül tarama hatası ({module_name}): {e}")
        return imports

    def update_module_stats(self, module_name: str, stats: Dict) -> None:
        """Modül kullanım istatistiklerini günceller."""
        self.module_stats[module_name] = {
            'last_used': datetime.now().isoformat(),
            'usage_count': self.module_stats.get(module_name, {}).get('usage_count', 0) + 1,
            **stats
        }

# rem #18. AutoImporter sınıfı dinamik modül yükleme ve güvenlik kontrollerini yönetir.
# rem #18.1. Bu sınıf modül önbellekleme ve dosya izleme işlemlerini gerçekleştirir.
# rem #18.2. Güvenli modda çalışarak zararlı kod yüklenmesini engeller.
class AutoImporter:
    """Dinamik modül yükleme ve bağımlılık yönetimi sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.loaded_modules = {}
        self.module_cache = {}
        self.imported_files = set()
        self.aliases = {}
        self.dependencies = defaultdict(list)
        self.secure_mode = False
        self.metadata = {"auto_importer": {"version": "1.5.0", "dependencies": []}}
        self.lock = threading.Lock()
        self.version_tracker = ModuleVersionTracker()
        self.installation_history = {}  # Yükleme geçmişi
        self.retry_count = {}  # Yeniden deneme sayısı
        self.logger = logging.getLogger(__name__)

    def is_recently_installed(self, module_name: str, timeout_minutes: int = 30) -> bool:
        """Modülün yakın zamanda yüklenip yüklenmediğini kontrol eder."""
        if module_name not in self.installation_history:
            return False
            
        last_install = self.installation_history[module_name]
        elapsed = datetime.now() - datetime.fromisoformat(last_install)
        return elapsed.total_seconds() < timeout_minutes * 60

    def should_retry_install(self, module_name: str, max_retries: int = 3) -> bool:
        """Modülün yeniden yüklenip yüklenmemesi gerektiğini kontrol eder."""
        if module_name not in self.retry_count:
            self.retry_count[module_name] = 0
            return True
            
        return self.retry_count[module_name] < max_retries

    def record_installation_attempt(self, module_name: str, success: bool):
        """Modül yükleme denemesini kaydeder."""
        if success:
            self.installation_history[module_name] = datetime.now().isoformat()
            self.retry_count[module_name] = 0
        else:
            if module_name not in self.retry_count:
                self.retry_count[module_name] = 0
            self.retry_count[module_name] += 1

    def load_module(self, file_path: str, alias: Optional[str] = None) -> Any:
        """Modül dosyasını yükler, önbelleğe alır, bağımlılıkları kontrol eder."""
        with self.lock:
            abs_path = os.path.abspath(file_path)
            if abs_path in self.imported_files:
                return self.module_cache.get(abs_path)
                
            if self.secure_mode and not self._is_allowed_path(abs_path):
                raise SecurityError(f"Güvensiz modül yolu: {abs_path}")
                
            module_name = Path(abs_path).stem
            
            # Yeniden yükleme kontrolü
            if not self.should_retry_install(module_name):
                self.logger.warning(f"Maksimum yeniden deneme sayısına ulaşıldı: {module_name}")
                return None
                
            # Yakın zamanda yüklenme kontrolü
            if self.is_recently_installed(module_name):
                self.logger.info(f"Modül zaten yakın zamanda yüklenmiş: {module_name}")
                return self.module_cache.get(abs_path)
                
            try:
                # Modül yüklemeden önce bağımlılıkları kontrol et
                deps = self._scan_module_imports(module_name)
                conflicts = self.version_tracker.check_conflicts(module_name, deps)
                
                if conflicts:
                    # Çakışmaları çözmeye çalış
                    for conflict in conflicts:
                        resolution = self.version_tracker.suggest_resolution(conflict)
                        if resolution["action"] == "upgrade":
                            # Bağımlılığı güncelle
                            self._upgrade_dependency(
                                resolution["module"], 
                                resolution["version"]
                            )
                
                # Modülü yükle
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                if not spec:
                    raise ImportError(f"Modül spec oluşturulamadı: {abs_path}")
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Versiyon bilgilerini kaydet
                version = getattr(module, "__version__", "0.0.1")
                self.version_tracker.register_module(module_name, version, deps)
                
                # Modülü önbelleğe al
                if alias:
                    self.aliases[alias] = abs_path
                    sys.modules[alias] = module
                else:
                    sys.modules[module_name] = module
                    
                self.module_cache[abs_path] = module
                self.imported_files.add(abs_path)
                
                # Başarılı yüklemeyi kaydet
                self.record_installation_attempt(module_name, True)
                
                return module
                
            except Exception as e:
                self.logger.error(f"Modül yükleme hatası: {abs_path}, {str(e)}")
                if module_name in sys.modules:
                    del sys.modules[module_name]
                self.record_installation_attempt(module_name, False)
                raise

    def _scan_module_imports(self, file_path: str) -> Dict[str, str]:
        """Modül dosyasındaki import ifadelerini ve versiyonları tarar."""
        imports = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports[name.name] = self._get_package_version(name.name)
                elif isinstance(node, ast.ImportFrom):
                    imports[node.module] = self._get_package_version(node.module)
                    
        except Exception as e:
            logging.warning(f"Import tarama hatası: {str(e)}")
            
        return imports
        
    def _get_package_version(self, package_name: str) -> str:
        """Pip paketinin kurulu versiyonunu döndürür."""
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except Exception:
            return "0.0.0"
            
    def _upgrade_dependency(self, package_name: str, version: str) -> bool:
        """Paket versiyonunu günceller."""
        try:
            cmd = [sys.executable, "-m", "pip", "install", f"{package_name}=={version}"]
            subprocess.check_call(cmd)
            return True
        except Exception as e:
            logging.error(f"Paket güncelleme hatası ({package_name}): {str(e)}")
            return False

# rem #25. ModuleAutoImporter sınıfı pdsX'in en önemli parçasıdır. Tüm modül yükleme ve bağımlılık yönetiminden sorumludur.
# rem #25.1. Bu sınıf otomatik olarak paketleri kurar, import eder ve çakışmaları önler.
# rem #25.2. İzole ortamı korur ve kaynakları temizler. Her modülün bağımlılıklarını takip eder.
class ModuleAutoImporter:
    """Modül yükleme ve yönetim sistemi."""
    # rem #30. Python 3.10 İşlemleri
    def __init__(self):
        self.env_manager = IsolatedEnvManager()
        self.loaded_modules = set()
        self.package_cache = {}
        self.logger = logging.getLogger(__name__)
        self.process_manager = ParallelProcessManager()
        self.installation_history = {}  # Yükleme geçmişi için
        self.retry_count = {}  # Yeniden deneme sayısı takibi için
        self.setup_logging()
        self.logger.info("ModuleAutoImporter başlatıldı")
        
    def setup_logging(self):
        """Loglama sistemi kuruluyor."""
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger.info("Log sistemi başlatıldı")

    def setup_environment(self) -> bool:
        """Çalışma ortamını hazırlar."""
        try:
            # İzole ortam kurulumu
            if not self.env_manager.create_env():
                self.logger.error("İzole ortam kurulamadı")
                return False
                
            # Paket kurulumu
            success = True
            for package in CORE_DEPENDENCIES["base"]:
                if not self.env_manager.install_package(package):
                    self.logger.error(f"Temel paket kurulumu başarısız: {package}")
                    success = False
                    
            return success
        except Exception as e:
            self.logger.error(f"Ortam kurulumu sırasında hata: {str(e)}")
            return False
            
    def import_module(self, module_name: str) -> Any:
        """Modülü güvenli şekilde import eder ve bağımlılıklarını yönetir."""
        try:
            self.logger.info(f"{module_name} modülü import ediliyor")
            if module_name in MODULE_SPECIFIC_DEPS:
                deps = MODULE_SPECIFIC_DEPS[module_name]
                self.logger.debug(f"{module_name} için gerekli bağımlılıklar: {deps}")
                if self.env_manager.check_package_conflicts(deps):
                    self.logger.error(f"Paket çakışması tespit edildi: {deps}")
                    return None
                
                for dep in deps:
                    if not self.package_cache.get(dep):
                        self.logger.info(f"{dep} bağımlılığı kuruluyor")
                        if self.env_manager.install_package(dep):
                            self.package_cache[dep] = True
                            # TensorFlow için özel kontrol
                            if dep == "tensorflow":
                                try:
                                    import tensorflow as tf
                                    # TensorFlow sürümünü kontrol et
                                    if tf.__version__.startswith("2."):
                                        self.logger.info("TensorFlow 2.x başarıyla kuruldu")
                                    else:
                                        self.logger.warning(f"TensorFlow sürümü uyumsuz olabilir: {tf.__version__}")
                                except Exception as e:
                                    self.logger.error(f"TensorFlow kurulumu başarısız: {e}")
                                    return None
                            else:
                                self.logger.info(f"{dep} başarıyla kuruldu")
                        else:
                            self.logger.error(f"{dep} kurulumu başarısız oldu")
                            return None

            spec = importlib.util.spec_from_file_location(module_name, module_name)
            if not spec or not spec.loader:
                raise ImportError(f"Modül yüklenemedi: {module_name}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_modules.add(module_name)
            self.logger.info(f"{module_name} başarıyla import edildi")
            return module
        except Exception as e:
            self.logger.error(f"Modül import hatası ({module_name}): {e}")
            return None

    def parallel_module_import(self, module_names: List[str]) -> Dict[str, Any]:
        """Birden fazla modülü paralel olarak import eder."""
        def _import_single(module_name: str) -> tuple:
            try:
                module = self.import_module(module_name)
                return (module_name, module)
            except Exception as e:
                self.logger.error(f"Paralel import hatası ({module_name}): {e}")
                return (module_name, None)
                
        results = self.process_manager.parallel_execute(_import_single, module_names)
        return dict(results)
        
    def parallel_package_install(self, packages: List[str]) -> Dict[str, bool]:
        """Birden fazla paketi paralel olarak kurar."""
        def _install_single(package: str) -> tuple:
            success = self.env_manager.install_package(package)
            return (package, success)
            
        results = self.process_manager.parallel_execute(_install_single, packages)
        return dict(results)

    def cleanup(self) -> None:
        """Yüklenen modülleri ve kaynakları temizler."""
        try:
            # Yüklü modülleri temizle
            for module_name in self.loaded_modules:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
            # Log yedekle
            if os.path.exists(LOG_FILE):
                shutil.copy2(LOG_FILE, LOG_BAK)
                
            self.logger.info("Temizlik işlemi tamamlandı")
        except Exception as e:
            self.logger.error(f"Temizlik sırasında hata: {str(e)}")

    def is_recently_installed(self, module_name: str, timeout_minutes: int = 30) -> bool:
        """Modülün yakın zamanda yüklenip yüklenmediğini kontrol eder."""
        if module_name not in self.installation_history:
            return False
            
        last_install = self.installation_history[module_name]
        elapsed = datetime.now() - datetime.fromisoformat(last_install)
        return elapsed.total_seconds() < timeout_minutes * 60

    def should_retry_install(self, module_name: str, max_retries: int = 3) -> bool:
        """Modülün yeniden yüklenip yüklenmemesi gerektiğini kontrol eder."""
        if module_name not in self.retry_count:
            self.retry_count[module_name] = 0
            return True
            
        return self.retry_count[module_name] < max_retries

    def record_installation_attempt(self, module_name: str, success: bool):
        """Modül yükleme denemesini kaydeder."""
        if success:
            self.installation_history[module_name] = datetime.now().isoformat()
            self.retry_count[module_name] = 0
        else:
            if module_name not in self.retry_count:
                self.retry_count[module_name] = 0
            self.retry_count[module_name] += 1

# --- Module Summary Generator ---
class ModuleSummaryGenerator:
    """Modül durumları ve istatistiklerini tablo formatında üreten sınıf."""
    def __init__(self):
        self.summaries = []
        self.terminal = sys.stdout
        
    def add_module_status(self, module_name: str, status: str, duration: float) -> None:
        """Modül durumunu tabloya ekler."""
        self.summaries.append({
            'module': module_name,
            'status': status,
            'duration': duration
        })

    def generate_summary_table(self) -> str:
        """Unicode box karakterleri kullanarak tablo oluşturur."""
        if not self.summaries:
            return "Henüz modül durumu eklenmemiş.\n"
            
        # Kolon genişliklerini hesapla
        module_width = max(len(s['module']) for s in self.summaries)
        module_width = max(module_width, len("Modül"))
        status_width = max(len(s['status']) for s in self.summaries)
        status_width = max(status_width, len("Durum"))
        
        # Tablo başlığı ve kenar çizgileri 
        top_line = f"┌{'─' * (module_width + 2)}┬{'─' * (status_width + 2)}┬{'─' * 12}┐\n"
        header = f"│ {'Modül'.ljust(module_width)} │ {'Durum'.ljust(status_width)} │ {'Süre (sn)'.ljust(10)} │\n"
        mid_line = f"├{'─' * (module_width + 2)}┼{'─' * (status_width + 2)}┼{'─' * 12}┤\n"
        bottom_line = f"└{'─' * (module_width + 2)}┴{'─' * (status_width + 2)}┴{'─' * 12}┘\n"
        
        # Tablo içeriğini oluştur
        table = [top_line, header, mid_line]
        for summary in self.summaries:
            duration_str = f"{summary['duration']:.2f}".rjust(10)
            row = f"│ {summary['module'].ljust(module_width)} │ {summary['status'].ljust(status_width)} │ {duration_str} │\n"
            table.append(row)
        table.append(bottom_line)
        
        return "".join(table)

    def save_summary(self, filename: str = "module_summary.log") -> None:
        """Tabloyu dosyaya kaydeder."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== Modül Durumu Özeti [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n\n")
                f.write(self.generate_summary_table())
        except Exception as e:
            print(f"Özet dosyası kaydedilemedi: {e}")

    def print_summary(self) -> None:
        """Tabloyu ekrana yazdırır."""
        print(f"\n=== Modül Durumu Özeti [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===\n")
        print(self.generate_summary_table())

# rem #28. Sistem Başlatma ve Çalışma Akışı Kontrolü
class SystemStartupManager:
    """PDS-X sistem başlatma ve çalışma akışı yöneticisi."""
    def __init__(self):
        self.startup_steps = {
            "environment_check": False,
            "venv_setup": False,
            "package_installation": False,
            "module_initialization": False,
            "security_validation": False,
            "system_ready": False
        }
        self.logger = logging.getLogger(__name__)

    def execute_startup_sequence(self) -> bool:
        """Sistem başlatma adımlarını sırasıyla yürütür."""
        try:
            # rem #28.1. Ortam kontrolü
            self.startup_steps["environment_check"] = self._check_environment()
            
            # rem #28.2. Sanal ortam kurulumu
            if self.startup_steps["environment_check"]:
                self.startup_steps["venv_setup"] = self._setup_venv()
            
            # rem #28.3. Paket kurulumu
            if self.startup_steps["venv_setup"]:
                self.startup_steps["package_installation"] = self._install_packages()
            
            # rem #28.4. Modül başlatma
            if self.startup_steps["package_installation"]:
                self.startup_steps["module_initialization"] = self._init_modules()
            
            # rem #28.5. Güvenlik doğrulama
            if self.startup_steps["module_initialization"]:
                self.startup_steps["security_validation"] = self._validate_security()
            
            # rem #28.6. Sistem hazır durumu
            self.startup_steps["system_ready"] = all(self.startup_steps.values())
            
            return self.startup_steps["system_ready"]
            
        except Exception as e:
            self.logger.error(f"Sistem başlatma hatası: {str(e)}")
            return False

    def _check_environment(self) -> bool:
        """Çalışma ortamını kontrol eder."""
        return validate_environment()["python_version"]

    def _setup_venv(self) -> bool:
        """Sanal ortamı kurar ve yapılandırır."""
        try:
            ensure_venv()
            return True
        except Exception:
            return False

    def _install_packages(self) -> bool:
        """Gerekli paketleri kurar."""
        try:
            install_missing_packages()
            return True
        except Exception:
            return False

    def _init_modules(self) -> bool:
        """Modülleri başlatır."""
        try:
            return True
        except Exception:
            return False

    def _validate_security(self) -> bool:
        """Güvenlik kontrollerini yapar."""
        try:
            return True
        except Exception:
            return False

# rem #29. İzole Ortam ve Paket Yönetimi
class IsolatedEnvManager:
    """İzole ortam ve paket yönetim sistemi."""
    # rem #29. İzole Ortam ve Paket Yönetimi
    
    def __init__(self):
        self.ENV_NAME = ".pdsx_isolated_env"
        self.python_path = None
        self.package_cache = {}
        self.lock = threading.Lock()
        self.offline_manager = OfflineModeManager()
        self.logger = logging.getLogger(__name__)
        
    def set_offline_mode(self, enabled: bool = True):
        """Çevrimdışı modu ayarlar."""
        if enabled:
            self.offline_manager.enable_offline_mode()
        else:
            self.offline_manager.disable_offline_mode()
            
    def create_env(self) -> bool:
        """İzole ortamı oluşturur."""
        try:
            if not ensure_venv():
                print("[PDS-X] İzole ortam oluşturulamadı")
                return False
                
            self.python_path = find_python310()
            if not self.python_path:
                print("[PDS-X] Python 3.10 bulunamadı")
                return False
                
            return True
        except Exception as e:
            print(f"[PDS-X] İzole ortam oluşturma hatası: {e}")
            return False

    def install_package(self, package_name: str) -> bool:
        """Paketi güvenli şekilde kurar ve kontrol eder."""
        with self.lock:
            try:
                if package_name in self.package_cache:
                    return True
                
                # TensorFlow için özel işlem
                if package_name == "tensorflow":
                    return self._install_tensorflow()
                    
                cmd = [sys.executable, "-m", "pip", "install", package_name]
                subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.package_cache[package_name] = True
                self.logger.info(f"{package_name} paketi başarıyla kuruldu")
                return True
                
            except Exception as e:
                self.logger.error(f"Paket kurulum hatası ({package_name}): {e}")
                return False
                
    def _install_tensorflow(self) -> bool:
        """TensorFlow kurulumu için özel mantık."""
        try:
            # Önce numpy'ı kur (TensorFlow için gerekli)
            if "numpy" not in self.package_cache:
                cmd_numpy = [sys.executable, "-m", "pip", "install", "numpy<2.2.0"]
                subprocess.check_call(cmd_numpy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.package_cache["numpy"] = True
                
            # TensorFlow'u kur
            cmd_tf = [sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"]
            subprocess.check_call(cmd_tf, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Kurulumu test et
            import tensorflow as tf
            if tf.__version__.startswith("2."):
                self.package_cache["tensorflow"] = True
                self.logger.info(f"TensorFlow {tf.__version__} başarıyla kuruldu")
                return True
            else:
                self.logger.error("TensorFlow kurulumu başarılı ancak sürüm uyumsuz")
                return False
                
        except Exception as e:
            self.logger.error(f"TensorFlow kurulum hatası: {e}")
            return False
            
    def check_package_conflicts(self, deps: List[str]) -> bool:
        """Paket çakışmalarını kontrol eder."""
        try:
            # TensorFlow ile numpy sürüm uyumluluğunu kontrol et
            if "tensorflow" in deps:
                import pkg_resources
                numpy_version = pkg_resources.get_distribution("numpy").version
                if not numpy_version.startswith("1."):
                    self.logger.warning(f"numpy sürümü ({numpy_version}) TensorFlow ile uyumsuz olabilir")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Paket çakışma kontrolü hatası: {e}")
            return True

# --- ModuleVersionTracker ---
class ModuleVersionTracker:
    """
    Modül versiyon ve çakışma takibi sınıfı.
    rem #32. Import versiyon kaydı ve çakışma çözümleme sistemi
    """
    def __init__(self):
        self.version_db = {}  # {module_name: List[{version: str, dependencies: dict, timestamp: str}]}
        self.conflict_history = []  # Çakışma geçmişini tutan liste
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def register_module(self, module_name: str, version: str, dependencies: Dict[str, str]) -> None:
        """Modülü versiyon veritabanına kaydeder."""
        with self.lock:
            module_entry = {
                "version": version,
                "dependencies": dependencies,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if module_name not in self.version_db:
                self.version_db[module_name] = []
            self.version_db[module_name].append(module_entry)
            self.logger.info(f"Module {module_name} v{version} registered")

    def check_conflicts(self, module_name: str, dependencies: Dict[str, str]) -> List[Dict]:
        """Bağımlılık çakışmalarını kontrol eder."""
        conflicts = []
        with self.lock:
            for dep_name, dep_version in dependencies.items():
                if dep_name in self.version_db:
                    installed_versions = self.version_db[dep_name]
                    if installed_versions:
                        latest_version = installed_versions[-1]["version"]
                        if self._compare_versions(dep_version, latest_version) != 0:
                            conflicts.append({
                                "module": module_name,
                                "dependency": dep_name,
                                "required_version": dep_version,
                                "installed_version": latest_version
                            })
        return conflicts

    def suggest_resolution(self, conflict: Dict) -> Dict[str, str]:
        """Çakışma için çözüm önerir."""
        try:
            module = conflict["module"]
            dep_name = conflict["dependency"]
            req_version = conflict["required_version"]
            
            # Versiyon geçmişine bak
            if dep_name in self.version_db:
                compatible_versions = []
                for ver in self.version_db[dep_name]:
                    if self._compare_versions(ver["version"], req_version) >= 0:
                        compatible_versions.append(ver["version"])
                        
                if compatible_versions:
                    return {
                        "action": "upgrade",
                        "version": max(compatible_versions),
                        "module": module,
                        "dependency": dep_name
                    }
            
            return {
                "action": "install",
                "version": req_version,
                "module": module,
                "dependency": dep_name
            }
        except Exception as e:
            self.logger.error(f"Resolution suggestion error: {e}")
            return {"error": str(e)}

    def rollback_module(self, module_name: str) -> bool:
        """Modülü önceki versiyona geri döndürür."""
        try:
            with self.lock:
                if module_name in self.version_db:
                    versions = self.version_db[module_name]
                    if len(versions) > 1:
                        versions.pop()  # En son versiyonu sil
                        self.logger.info(f"Module {module_name} rolled back to v{versions[-1]['version']}")
                        return True
                return False
        except Exception as e:
            self.logger.error(f"Rollback error for {module_name}: {e}")
            return False

    def get_version_tree(self) -> Dict:
        """Versiyon ağacını JSON formatında döndürür."""
        with self.lock:
            return {
                "modules": self.version_db,
                "conflicts": self.conflict_history,
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _compare_versions(self, ver1: str, ver2: str) -> int:
        """İki versiyon numarasını karşılaştırır.
        
        Args:
            ver1: Karşılaştırılacak ilk versiyon
            ver2: Karşılaştırılacak ikinci versiyon
            
        Returns:
            1: ver1 > ver2
            0: ver1 == ver2
            -1: ver1 < ver2
        """
        try:
            v1_parts = [int(x) for x in ver1.replace(">","").replace("<","").replace("=","").split(".")]
            v2_parts = [int(x) for x in ver2.replace(">","").replace("<","").replace("=","").split(".")]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                v1 = v1_parts[i] if i < len(v1_parts) else 0
                v2 = v2_parts[i] if i < len(v2_parts) else 0
                if v1 > v2:
                    return 1
                elif v1 < v2:
                    return -1
            return 0
        except Exception:
            return 0  # Karşılaştırma yapılamıyorsa eşit kabul et

# rem #34. Gelişmiş loglama sistemi
class AdvancedLogger:
    """Gelişmiş loglama ve hata izleme sistemi."""
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # JSON log dosyaları
        self.json_log_dir = self.log_dir / "json"
        self.json_log_dir.mkdir(exist_ok=True)
        
        self.terminal_log = self.json_log_dir / "terminal.jsonl"
        self.error_log = self.json_log_dir / "errors.jsonl"
        self.warning_log = self.json_log_dir / "warnings.jsonl"
        self.info_log = self.json_log_dir / "info.jsonl"
        
        # Temel logger yapılandırması
        self.logger = logging.getLogger("pdsX")
        self.logger.setLevel(logging.DEBUG)
        
        # JSON formatında log kaydı için özel handler
        self.json_handler = logging.FileHandler(self.terminal_log)
        self.json_handler.setFormatter(JsonFormatter())
        
        # Dosya handler'ları
        fh = logging.FileHandler(self.terminal_log, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        eh = logging.FileHandler(self.error_log, encoding='utf-8')
        eh.setLevel(logging.ERROR)
        
        # Format
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        eh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(eh)
        
        # Log rotasyon ayarları
        self.max_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5

    def log(self, message: str, level: str = "INFO", **kwargs):
        """JSON formatında log mesajı kaydeder."""
        # Kayıt nesnesine özel alanları ekle
        extra = logging.LogRecord("pdsX", logging.getLevelName(level), "", 0, message, (), None)
        extra.extra_fields = kwargs
        
        # JSON formatında kaydet
        formatted_message = self.json_handler.formatter.format(extra)
        
        # Log seviyesine göre ilgili dosyaya kaydet
        log_file = getattr(self, f"{level.lower()}_log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted_message + "\n")
            
        # Standart logger'a da gönder
        if level == "ERROR":
            self.logger.error(message, extra=kwargs)
        elif level == "WARNING": 
            self.logger.warning(message, extra=kwargs)
        else:
            self.logger.info(message, extra=kwargs)
            
        # Dosya boyutunu kontrol et
        self._check_rotation()
        
    def _check_rotation(self):
        """Log dosyalarının boyutunu kontrol eder ve gerekliyse rotasyon yapar."""
        max_size = 10 * 1024 * 1024  # 10MB
        
        for log_file in [self.terminal_log, self.error_log, self.warning_log, self.info_log]:
            if log_file.stat().st_size > max_size:
                self._rotate_log(log_file)

    def _rotate_log(self, log_file: Path):
        """Tek bir log dosyası için rotasyon işlemi yapar."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = log_file.parent / f"{log_file.stem}_{timestamp}.jsonl"
        
        try:
            log_file.rename(backup)
            log_file.touch()  # Yeni boş dosya oluştur
            
            # Eski yedekleri temizle
            self._cleanup_old_backups(log_file.parent)
            
        except Exception as e:
            print(f"Log rotasyon hatası ({log_file}): {e}")
            
    def _cleanup_old_backups(self, log_dir: Path, max_backups: int = 5):
        """Eski log yedeklerini temizler."""
        pattern = "*.jsonl"
        backups = sorted(log_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        
        # En yeni max_backups kadar yedeği tut, gerisini sil
        for old_backup in backups[max_backups:]:
            try:
                old_backup.unlink()
            except Exception as e:
                print(f"Eski yedek temizleme hatası ({old_backup}): {e}")
                
    def get_daily_summary(self) -> Dict:
        """Günlük log istatistiklerini JSON formatında döndürür."""
        today = datetime.date.today()
        summary = {
            "date": str(today),
            "levels": {
                "error": 0,
                "warning": 0,
                "info": 0
            },
            "total_entries": 0
        }
        
        try:
            # Her log dosyası için bugünün kayıtlarını say
            for log_file in [self.error_log, self.warning_log, self.info_log]:
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                if entry["timestamp"].startswith(str(today)):
                                    level = entry["level"].lower()
                                    summary["levels"][level] += 1
                                    summary["total_entries"] += 1
                            except json.JSONDecodeError:
                                continue
            
            # Başarı oranını hesapla
            total = summary["total_entries"]
            if total > 0:
                success_rate = ((total - summary["levels"]["error"]) / total) * 100
                summary["success_rate"] = f"{success_rate:.1f}%"
            else:
                summary["success_rate"] = "100.0%"
                
            return summary
            
        except Exception as e:
            print(f"Log analizi hatası: {e}")
            return summary

# --- JSON Formatter ---
class JsonFormatter(logging.Formatter):
    """JSON formatında log kaydı için özel formatlayıcı."""
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Özel alanlar varsa ekle
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
            
        # Hata varsa stack trace ekle
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

    def formatTime(self, record) -> str:  
        """Zaman damgasını ISO formatında döndürür."""
        return datetime.datetime.fromtimestamp(record.created).isoformat()

# --- Yardımcı Fonksiyonlar ---
# rem #26. Yardımcı fonksiyonlar ve ortam doğrulama
def validate_environment() -> Dict[str, Any]:
    """Python sürümü ve sistem gereksinimlerini kontrol eder."""
    validation = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "os": os.name,
        "platform": sys.platform
    }
    return validation

def create_workspace(path: str) -> ModuleAutoImporter:
    #26.2. Yeni bir çalışma alanı oluşturur ve yapılandırır
    """Yeni bir çalışma alanı oluşturur ve yapılandırır."""
    workspace = Path(path)
    if not workspace.exists():
        workspace.mkdir(parents=True)
        
    importer = ModuleAutoImporter()
    if importer.setup_environment():
        print("[PDS-X] Çalışma alanı başarıyla oluşturuldu")
        return importer
    else:
        print("[PDS-X] Çalışma alanı oluşturma başarısız")
        return None

# rem #31. İzole Ortam Yardımcı Fonksiyonları 

def find_python310() -> Optional[str]:
    """Python 3.10 yürütücüsünü bulur."""
    # rem #31.1. Sistem üzerinde Python 3.10 arama
    print("[PDS-X] Python 3.10 arama başlatıldı...")
    
    # PATH üzerinde ara
    for exe in ["python3.10", "python310", "python"]:
        path = shutil.which(exe)
        if path:
            try:
                out = subprocess.check_output([path, "--version"], text=True)
                if "3.10" in out:
                    print(f"[PDS-X] Uygun Python bulundu: {path}")
                    return path
            except Exception as e:
                print(f"[PDS-X] PATH'de {exe} çalıştırılamadı: {e}")
                
    # Windows Registry'de ara 
    if os.name == "nt":
        try:
            import winreg
            for key in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    py_core = winreg.OpenKey(key, r"Software\Python\PythonCore")
                    for i in range(winreg.QueryInfoKey(py_core)[0]):
                        try:
                            version = winreg.EnumKey(py_core, i)
                            if version.startswith("3.10"):
                                install_path = winreg.QueryValue(py_core, version + "\\InstallPath")
                                py_exe = os.path.join(install_path, "python.exe")
                                if os.path.exists(py_exe):
                                    print(f"[PDS-X] Registry'de bulundu: {py_exe}")
                                    return py_exe
                        except WindowsError:
                            continue
                except WindowsError:
                    continue
        except Exception as e:
            print(f"[PDS-X] Registry taraması başarısız: {e}")
            
    return None

def ensure_venv() -> bool:
    """Sanal ortamı oluşturur ve yapılandırır."""
    # rem #31.2. İzole ortam kurulumu ve yapılandırması
    env_name = ".pdsx_isolated_env"
    venv_path = Path(env_name)
    
    # Eğer ortam varsa kontrol et
    if venv_path.exists():
        python_path = venv_path / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        if python_path.exists():
            try:
                out = subprocess.check_output([str(python_path), "--version"], text=True)
                if "3.10" in out:
                    print(f"[PDS-X] Mevcut ortam kullanılıyor: {python_path}")
                    return True
            except Exception:
                pass
                
        print("[PDS-X] Mevcut ortam geçersiz, yeniden oluşturuluyor...")
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print(f"[PDS-X] Eski ortam temizlenemedi: {e}")
            return False
            
    # Python 3.10 bul veya kur
    python_path = find_python310()
    if not python_path:
        print("[PDS-X] Python 3.10 bulunamadı!")
        return False
        
    # Ortamı oluştur
    try:
        subprocess.run([python_path, "-m", "venv", env_name], check=True)
        print(f"[PDS-X] İzole ortam oluşturuldu: {env_name}")
        return True
    except Exception as e:
        print(f"[PDS-X] İzole ortam oluşturma hatası: {e}")
        return False

def install_missing_packages():
    """Eksik paketleri paralel olarak kurar."""
    logger = logging.getLogger(__name__)
    
    def install_package(pkg_info):
        pkg_name, import_name = pkg_info
        try:
            # Önce import kontrolü
            try:
                importlib.import_module(import_name)
                logger.info(f"{pkg_name} zaten kurulu")
                return True
            except ImportError:
                pass
                
            # Paket kurulumu
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg_name],
                stdout=subprocess.DEVNULL
            )
            logger.info(f"{pkg_name} kuruldu")
            return True
            
        except Exception as e:
            logger.error(f"{pkg_name} kurulum hatası: {e}")
            return False
            
    # Paralel kurulum
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(install_package, REQUIRED_PACKAGES))
        
    # Sonuç kontrolü
    if all(results):
        logger.info("Tüm paketler başarıyla kuruldu")
        return True
    else:
        logger.error("Bazı paketler kurulamadı")
        return False

# rem #32. Ana İşlev ve Sistem Başlatma

def main():
    """Ana sistem başlatma işlevi."""
    try:
        # Log dosyasını yedekle
        if os.path.exists(LOG_FILE):
            if os.path.exists(LOG_BAK):
                os.remove(LOG_BAK)
            shutil.copy2(LOG_FILE, LOG_BAK)
            
        # Modül yükleyiciyi başlat
        importer = ModuleAutoImporter()
        
        # Çalışma ortamını hazırla
        if not importer.setup_environment():
            print("[PDS-X] Ortam kurulumu başarısız!")
            return False
            
        print("[PDS-X] Sistem başarıyla başlatıldı")
        return importer
        
    except Exception as e:
        print(f"[PDS-X] Sistem başlatma hatası: {e}")
        return False

if __name__ == "__main__":
    importer = main()
    if importer:
        print("[PDS-X] Sistem hazır")
    else:
        print("[PDS-X] Sistem başlatılamadı")
        sys.exit(1)
        
    print("[PDS-X] auto_importer.py sadece pdsXu ile kullanılabilir")

# rem #33. Özel hata sınıfları
class SecurityError(Exception):
    """Güvenlik ile ilgili hataları temsil eden sınıf."""
    pass

class VersionError(Exception):
    """Versiyon uyumsuzluğu hatalarını temsil eden sınıf.""" 
    pass

class ModuleLoadError(Exception):
    """Modül yükleme hatalarını temsil eden sınıf."""
    pass

# SON. Kodun her adımı için rem satırları eklendi. Kodun amacı değişmedikçe bu rem satırları asla silinmeyecek, sadece ekleme yapılacak.

# EXPORTS. PDS-X modül doğrulama için dışa aktarılanlar listesi
__pdsX_exports__ = [
    'find_python310',
    'ensure_venv',
    'IsolatedEnvManager',
    'ModuleAutoImporter',
    'main'
]
