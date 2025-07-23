# -*- coding: utf-8 -*-
"""
PDS-X Auto-Importer: Akıllı, Otomatik ve Kendi Kendini Onaran Paket Yöneticisi
Versiyon: 1.8.0.0
Tarih: 23 Haziran 2025
Yazar: xAI (Fikir: Mete Dinler, Geliştirme: GitHub Copilot & Gemini)
"""
from __future__ import annotations
import sys
import subprocess
import threading
import time
import json
import re
import logging
import argparse
import asyncio
import os
import traceback
import signal
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from collections import deque, defaultdict
from statistics import stdev
from datetime import datetime

# --- Gerekli Kütüphaneleri Yüklemeyi Dene ---
try:
    import colorama
    import psutil
    import networkx as nx
    from scipy.stats import levene
except ImportError as e:
    print(f"[PDS-X Auto-Importer] Başlangıç için gerekli bir kütüphane eksik: {e.name}. Lütfen manuel olarak kurun: pip install {e.name}")
    sys.exit(1)

# --- Sabitler ve Temel Yapılandırma ---
BASE_DIR = Path(__file__).parent.resolve()
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
TERMINAL_LOG_FILE = LOGS_DIR / "terminal_output.log"
ALIAS_FILE = BASE_DIR / "pdsx_aliases.txt"
DEPENDENCY_FILE = BASE_DIR / "pdsx_dependencies.json"
LEARNED_DEPS_FILE = BASE_DIR / "learned_dependencies.json"
WHEEL_CACHE_DIR = BASE_DIR / "wheel_cache"

# --- Özel Hata Sınıfları ---
class PdsXError(Exception):
    """AutoImporter için temel hata sınıfı."""
    def __init__(self, message: str, code: str, details: Dict[str, Any] = None):
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code
        self.details = details or {}

class PdsXInstallationError(PdsXError):
    """Paket kurulumu sırasında oluşan hatalar için."""
    pass

class PdsXNotFoundError(PdsXError):
    """Dosya veya paket bulunamadığında kullanılır."""
    pass

class PdsXImportError(PdsXError):
    """Kurulum sonrası içe aktarma başarısız olduğunda kullanılır."""
    pass

# --- Gelişmiş Loglama Sınıfı ---
class AdvancedLogger:
    """Hem konsola hem de dosyalara JSON formatında loglama yapan gelişmiş logger."""
    def __init__(self, name: str, log_dir: Path = LOGS_DIR):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.shutdown_callback = None

    def setup_logging(self, level: str = 'INFO', json_log_file: str = 'auto_importer_events.jsonl'):
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Mevcut handler'ları temizle
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Konsol Handler (Renkli ve basit format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(f'%(asctime)s - %(name)s - {self._get_level_color()}%(levelname)s{colorama.Style.RESET_ALL} - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # JSONL Dosya Handler (Yapısal loglama için)
        json_handler = logging.FileHandler(self.log_dir / json_log_file, mode='w', encoding='utf-8')
        json_handler.setLevel(logging.DEBUG) # Dosyaya her şeyi yaz
        json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}')
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)

    def _get_level_color(self) -> str:
        # Bu metodun doğrudan formatter içinde kullanılması zor, bu yüzden log metodunda renk ekleyeceğiz.
        # Şimdilik temel bir format bırakalım.
        return ''

    def log(self, level: str, message: str, *args, **kwargs):
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        
        # Mesajı JSON uyumlu hale getir
        if isinstance(message, dict) or isinstance(message, list):
            message = json.dumps(message, ensure_ascii=False, default=str)
        else:
            message = str(message).replace('"', '\\"') # Basit kaçış

        log_func(message, *args, **kwargs)

    def register_shutdown_callback(self, callback: Callable):
        self.shutdown_callback = callback

    def shutdown(self):
        if self.shutdown_callback:
            self.shutdown_callback()
        logging.shutdown()

# --- Tee (Loglama için Yardımcı Sınıf) ---
class Tee:
    """Hem dosyaya hem de orijinal stdout/stderr'e yazan bir nesne."""
    def __init__(self, original_stream, log_file_path: Path):
        self.original_stream = original_stream
        self.log_file = None
        try:
            self.log_file = open(log_file_path, 'w', encoding='utf-8', buffering=1)
        except IOError as e:
            self.original_stream.write(f"[PDS-X] HATA: Terminal log dosyası açılamadı: {log_file_path}\\n{e}\\n")

    def write(self, text):
        self.original_stream.write(text)
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        if self.log_file:
            self.log_file.close()

# --- Bağımlılık ve Sistem Analizi Araçları ---
class PipOutputAnalyzer:
    """Pip'in çıktılarını analiz ederek başarı, hata ve bağımlılık bilgilerini çıkarır."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.success_pattern = re.compile(r"Successfully installed (.+)", re.IGNORECASE)
        self.dependency_pattern = re.compile(r"Collecting ([a-zA-Z0-9\-_.]+)", re.IGNORECASE)
        self.error_patterns = {
            "permission_error": re.compile(r"Permission denied|permission error", re.IGNORECASE),
            "network_error": re.compile(r"Could not connect|timed out|network is unreachable", re.IGNORECASE),
            "http_error": re.compile(r"HTTP error 404|403", re.IGNORECASE),
            "missing_visual_cpp": re.compile(r"Microsoft Visual C\+\+ .* is required", re.IGNORECASE),
            "cl_exe_failed": re.compile(r"error: command 'cl.exe' failed", re.IGNORECASE),
            "missing_rust_compiler": re.compile(r"Could not find `cargo`", re.IGNORECASE),
            "metadata_failed": re.compile(r"error: metadata-generation-failed", re.IGNORECASE),
            "no_matching_distribution": re.compile(r"Could not find a version that satisfies the requirement (.+)", re.IGNORECASE),
            "dependency_conflict": re.compile(r"conflicting requirements", re.IGNORECASE),
        }

    def analyze_success(self, output: str, main_package_name: str) -> Dict[str, Any]:
        self.logger.log("debug", f"Başarı analizi başlatıldı. Ana paket: '{main_package_name}'.")
        version = None
        success_matches = self.success_pattern.findall(output)
        main_package_line = ""
        for match in success_matches:
            installed_packages = match.split()
            for pkg_with_ver in installed_packages:
                if pkg_with_ver.lower().startswith(main_package_name.lower() + '-'):
                    main_package_line = pkg_with_ver
                    break
            if main_package_line:
                break

        if main_package_line:
            parts = main_package_line.rsplit('-', 1)
            if len(parts) == 2 and parts[1][0].isdigit():
                version = parts[1]
                self.logger.log("info", f"Başarıyla analiz edildi: Paket '{main_package_name}', Sürüm '{version}'")
            else:
                self.logger.log("warning", f"'{main_package_name}' için sürüm numarası ayrıştırılamadı. Satır: '{main_package_line}'")
        else:
            self.logger.log("warning", f"'{main_package_name}' için 'Successfully installed' satırı bulunamadı.")

        dependencies = self.dependency_pattern.findall(output)
        cleaned_deps = [dep for dep in dependencies if dep.lower() != main_package_name.lower()]
        return {"main_package": main_package_name, "version": version, "dependencies": list(set(cleaned_deps))}

    def analyze_failure(self, output: str) -> Dict[str, Any]:
        self.logger.log("debug", f"Pip hata çıktısı analizi başlatıldı.")
        for key, pattern in self.error_patterns.items():
            if pattern.search(output):
                self.logger.log("warning", f"Hata deseni bulundu: {key}")
                # Bu kısım daha da geliştirilebilir, şimdilik sadece deseni döndürelim
                return {"error_type": key, "details": pattern.search(output).group(0)}
        return {"error_type": "unknown", "details": "Bilinmeyen bir pip hatası."}

class DependencyRegistry:
    """Paketlerin sürümlerini, bağımlılıklarını ve kurulum geçmişini yönetir."""
    def __init__(self, logger: AdvancedLogger, file_path: Path):
        self.logger = logger
        self.file_path = file_path
        self.registry = self._load()
        self.graph = nx.DiGraph()
        self.build_graph_from_registry()

    def _normalize_name(self, package_name: str) -> str:
        return package_name.lower().replace("_", "-") if package_name else ""

    def _load(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            self.logger.log("info", "Bağımlılık kayıt dosyası bulunamadı, yeni bir tane oluşturuluyor.")
            return {"version": "1.0", "packages": {}}
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası yüklenirken hata: {e}. Yeni bir kayıt oluşturuluyor.")
            return {"version": "1.0", "packages": {}}

    def save(self):
        try:
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=4)
        except IOError as e:
            self.logger.log("error", f"Bağımlılık kayıt dosyası kaydedilirken hata: {e}")

    def add_package(self, package_name: str, version: str, dependencies: List[str]):
        normalized_name = self._normalize_name(package_name)
        if not normalized_name: return

        package_data = self.registry["packages"].get(normalized_name, {})
        version_history = package_data.get("versions", [])
        if version not in version_history:
            version_history.append(version)

        self.registry["packages"][normalized_name] = {
            "current_version": version,
            "versions": version_history,
            "dependencies": sorted([self._normalize_name(dep) for dep in dependencies]),
            "last_installed_at": datetime.now().isoformat()
        }
        self.update_graph(normalized_name, dependencies)
        self.save()

    def get_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        return self.registry["packages"].get(self._normalize_name(package_name))

    def build_graph_from_registry(self):
        self.graph.clear()
        packages = self.registry.get("packages", {})
        for pkg_name, pkg_info in packages.items():
            self.graph.add_node(pkg_name)
            for dep in pkg_info.get("dependencies", []):
                self.graph.add_node(dep)
                self.graph.add_edge(dep, pkg_name) # Bağımlılık -> Paket
        self.logger.log("info", f"Bağımlılık grafiği kayıtlardan oluşturuldu. Düğümler: {len(self.graph.nodes())}, Kenarlar: {len(self.graph.edges())}")

    def update_graph(self, package_name: str, dependencies: List[str]):
        normalized_name = self._normalize_name(package_name)
        self.graph.add_node(normalized_name)
        for dep in dependencies:
            normalized_dep = self._normalize_name(dep)
            self.graph.add_node(normalized_dep)
            self.graph.add_edge(normalized_dep, normalized_name)
        self.logger.log("debug", f"Grafik güncellendi: {package_name}")

class HeuristicManager:
    """Paketlerin önemini ve türünü belirlemek için sezgisel yöntemler kullanır."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.foundational_keywords = {"core", "base", "lib", "common", "utils"}
        self.data_science_keywords = {"numpy", "scipy", "pandas", "matplotlib", "seaborn", "scikit-learn", "tensorflow", "torch", "keras"}

    def is_foundational(self, package_name: str) -> bool:
        return any(keyword in package_name.lower() for keyword in self.foundational_keywords)

class DependencyOptimizer:
    """Bağımlılık grafiğini kullanarak en verimli kurulum sırasını belirler."""
    def __init__(self, logger: AdvancedLogger, dependency_registry: DependencyRegistry):
        self.logger = logger
        self.dependency_registry = dependency_registry

    def get_optimized_order(self, modules: List[str]) -> List[str]:
        self.logger.log("info", "Bağımlılık optimizasyonu başlatılıyor...")
        # Sadece istenen modülleri ve onların bağımlılıklarını içeren bir alt grafik oluştur
        subgraph = self.dependency_registry.graph.subgraph(nx.ancestors(self.dependency_registry.graph, modules) | set(modules))
        
        try:
            # Topolojik sıralama, bağımlılıkların önce kurulmasını sağlar
            optimized_order = list(nx.topological_sort(subgraph))
            # Sadece başlangıçta istenen modülleri içeren bir sıra döndür, ancak sıralama tüm bağımlılıkları dikkate alsın
            final_order = [m for m in optimized_order if m in modules]
            self.logger.log("info", f"Optimize edilmiş kurulum sırası: {final_order}")
            return final_order
        except nx.NetworkXUnfeasible as e:
            self.logger.log("warning", f"Topolojik sıralama mümkün değil (muhtemelen döngüsel bağımlılık): {e}. Orijinal sıra kullanılacak.")
            return modules # Döngü varsa, en azından denemeye devam et
        except Exception as e:
            self.logger.log("error", f"Optimizasyon sırasında beklenmedik hata: {e}", exc_info=True)
            return modules

class GracefulShutdownManager:
    """Uygulamanın düzgün bir şekilde kapanmasını yönetir."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.tasks: List[Dict[str, Any]] = []
        self.shutdown_started = False

    def register_task(self, name: str, stop_func: Callable, *args):
        self.tasks.append({'name': name, 'stop_func': stop_func, 'args': args})
        self.logger.log("debug", f"Kapanış görevi kaydedildi: {name}")

    def shutdown(self):
        if self.shutdown_started: return
        self.shutdown_started = True
        self.logger.log("info", "Graceful shutdown başlatılıyor...")
        for task in reversed(self.tasks):
            try:
                task_name = task['name']
                self.logger.log("info", f"'{task_name}' görevi durduruluyor...")
                task['stop_func'](*task['args'])
            except Exception as e:
                task_name_for_error = task.get('name', 'Bilinmeyen Görev')
                self.logger.log("error", f"'{task_name_for_error}' görevi durdurulurken hata oluştu: {e}", exc_info=True)
        self.logger.log("info", "Graceful shutdown tamamlandı.")

class SummaryGenerator:
    """Kurulum süreci hakkında özet bir rapor oluşturur."""
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_module_status(self, package: str, status: str, duration: float, reason: Optional[str] = None):
        self.results.append({"package": package, "status": status, "duration": duration, "reason": reason})

    def print_summary(self):
        total_duration = time.time() - self.start_time
        successful = [r for r in self.results if r["status"] == "Başarılı"]
        failed = [r for r in self.results if r["status"] == "Başarısız"]
        skipped = [r for r in self.results if r["status"] == "Atlandı"]

        report = [
            f"\n--- Kurulum Özeti ---",
            f"Toplam Süre: {total_duration:.2f}s",
            f"Başarılı: {len(successful)}, Başarısız: {len(failed)}, Atlandı: {len(skipped)}",
        ]
        if successful:
            report.append(f"{colorama.Fore.GREEN}Başarılı Paketler:{colorama.Style.RESET_ALL} " + ", ".join(r['package'] for r in successful))
        if failed:
            report.append(f"{colorama.Fore.RED}Başarısız Paketler:{colorama.Style.RESET_ALL}")
            for r in failed:
                report.append(f"  - {r['package']}: {r.get('reason', 'Bilinmiyor')}")
        
        self.logger.log("info", "\n".join(report))

class ConflictResolver:
    """Bağımlılık çakışmalarını tespit eder ve çözüm önerileri sunar."""
    def __init__(self, installer: AutoImporter, logger: AdvancedLogger):
        self.installer = installer
        self.logger = logger

    def suggest_resolution(self, package: str, failure_details: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log("debug", f"Çözüm önerisi analizi: {package}, Hata: {failure_details}")
        error_type = failure_details.get("error_type")
        if error_type == "dependency_conflict":
            # Basit strateji: --use-feature=2020-resolver ile tekrar dene
            return {"action": "retry_with_args", "new_args": ["--use-feature=2020-resolver"], "reason": "Yeni bağımlılık çözücü deneniyor."}
        if error_type == "metadata_failed":
            return {"action": "install_before", "packages": ["wheel", "setuptools"], "reason": "Metadata hatası için build bağımlılıkları kuruluyor."}
        
        return {"action": "abort", "reason": f"Çözümlenemeyen hata: {error_type}"}

# --- Akıllı Kurulum Yöneticisi ---
class SmartInstallManager:
    """Kurulum sürecini sistem durumuna, bağımlılıklara ve sezgisel bilgilere göre dinamik olarak yöneten akıllı bir sistem."""
    def __init__(self, modules: List[str], installer: AutoImporter):
        self.modules = modules
        self.installer = installer
        self.logger = installer.logger
        self.chaos_report = {}

    def run_all(self):
        self.logger.log("info", f"Akıllı Kurulum Yöneticisi '{len(self.modules)}' modül için başlatıldı.")
        self.analyze_chaos()
        prioritized_queue = self.prioritize_adaptive()
        optimized_order = self.installer.dependency_optimizer.get_optimized_order(prioritized_queue)
        
        final_queue = optimized_order or prioritized_queue
        self.logger.log("info", f"Nihai Kurulum Kuyruğu: {final_queue}")
        
        try:
            asyncio.run(self.schedule_installation(final_queue))
        except Exception as e:
            self.logger.log("critical", f"Akıllı kurulum zamanlamasında kritik hata: {e}", exc_info=True)

    def analyze_chaos(self, duration: int = 5):
        self.logger.log("info", "Chaos analizi başlatılıyor...")
        cpu, mem = [], []
        try:
            for _ in range(duration):
                cpu.append(psutil.cpu_percent(interval=0.1))
                mem.append(psutil.virtual_memory().percent)
        except Exception as e:
            self.logger.log("error", f"Kaynak kullanımı ölçülürken hata: {e}")
            cpu, mem = [0], [0]

        cpu_std = round(stdev(cpu), 2) if len(cpu) > 1 else 0.0
        mem_std = round(stdev(mem), 2) if len(mem) > 1 else 0.0
        chaos_index = round((cpu_std + mem_std) / 2, 2)
        self.chaos_report = {"chaos_index": chaos_index}
        self.logger.log("info", f"📊 Chaos Raporu: {json.dumps(self.chaos_report)}")

    def prioritize_adaptive(self) -> List[str]:
        chaos_index = self.chaos_report.get("chaos_index", 0)
        def weight(mod: str) -> int:
            w = 10
            if self.installer.heuristic_manager.is_foundational(mod): w -= 5
            if self.installer.dependency_registry.get_package_info(mod): w -= 2
            w += int(chaos_index) # Kaos arttıkça ağırlık artar
            return w
        
        sorted_modules = sorted(self.modules, key=weight)
        self.logger.log("info", f"📦 Adaptive Sıralama: {sorted_modules}")
        return sorted_modules

    async def schedule_installation(self, queue: List[str]):
        tasks = []
        for mod in queue:
            delay = self.get_delay()
            task = asyncio.create_task(self.install_with_delay(mod, delay))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        new_dependencies = []
        for res in results:
            if isinstance(res, list):
                new_dependencies.extend(res)
            elif isinstance(res, Exception):
                self.logger.log("error", f"Asenkron kurulumda bir görevde hata oluştu: {res}", exc_info=True)

        if new_dependencies:
            unique_new_deps = list(set(new_dependencies))
            self.logger.log("info", f"Yeni bağımlılıklar bulundu: {unique_new_deps}. Tekrar kuruluyor...")
            # Bu yeni bağımlılıkları doğrudan ana kurulum listesine ekle
            self.installer.install_packages(unique_new_deps, use_smart_installer=True)

    async def install_with_delay(self, mod: str, delay: float) -> List[str]:
        if delay > 0:
            await asyncio.sleep(delay)
        self.logger.log("info", f"'{mod}' için asenkron kurulum başlatılıyor (gecikme: {delay:.2f}s).")
        try:
            loop = asyncio.get_running_loop()
            # _install_package_with_retry senkron bir fonksiyon, executor'da çalıştır
            new_deps = await loop.run_in_executor(
                None, self.installer._install_package_with_retry, mod
            )
            return new_deps
        except Exception as e:
            self.logger.log("error", f"'{mod}' asenkron kurulumu sırasında hata: {e}", exc_info=True)
            return []

    def get_delay(self) -> float:
        chaos = self.chaos_report.get("chaos_index", 0)
        return min(max(0.1, chaos / 10), 2.0) # 0.1s ile 2s arasında bir gecikme

# --- Ana Sınıf ---
class AutoImporter:
    """PDS-X için Python kütüphanelerini otomatik olarak yöneten, kuran ve onaran ana sınıf."""
    def __init__(self, mode: str = 'auto', log_level: str = 'INFO', monitor_terminal: bool = False):
        self.logger = AdvancedLogger("AutoImporter")
        self.logger.setup_logging(level=log_level)
        
        self.shutdown_manager = GracefulShutdownManager(self.logger)
        self.shutdown_manager.register_task("LoggerShutdown", self.logger.shutdown)

        self.pip_output_analyzer = PipOutputAnalyzer(self.logger)
        self.dependency_registry = DependencyRegistry(self.logger, DEPENDENCY_FILE)
        self.heuristic_manager = HeuristicManager(self.logger)
        self.dependency_optimizer = DependencyOptimizer(self.logger, self.dependency_registry)
        self.conflict_resolver = ConflictResolver(self, self.logger)
        self.summary_generator = SummaryGenerator(self.logger)
        
        self.mode = mode
        self.installation_pool = ThreadPoolExecutor(max_workers=self._get_max_workers())
        self.installing_packages = set()
        self.failed_packages = {}
        self.lock = threading.Lock()

        self._setup_signal_handlers()
        self.shutdown_manager.register_task("ThreadPoolExecutor", self.installation_pool.shutdown, True)
        
        self._prepare_aliases()

    def _get_max_workers(self) -> int:
        return min(32, (os.cpu_count() or 1) + 4)

    def _setup_signal_handlers(self):
        def handle_signal(signum, frame):
            self.logger.log("warning", f"Sinyal {signum} alındı, uygulama kapatılıyor...")
            self.shutdown()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _prepare_aliases(self):
        self.aliases = {"bs4": "beautifulsoup4", "cv2": "opencv-python", "sklearn": "scikit-learn"}
        if ALIAS_FILE.exists():
            try:
                with ALIAS_FILE.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and not line.startswith("#") and ":" in line:
                            alias, package = line.split(":", 1)
                            self.aliases[alias.strip()] = package.strip()
            except Exception as e:
                self.logger.log("error", f"Alias dosyası ({ALIAS_FILE}) okunurken hata: {e}")

    def _resolve_package_name(self, module_name: str) -> str:
        return self.aliases.get(module_name, module_name.replace('_', '-'))

    def _run_pip_command(self, command: List[str], package: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            self.logger.log("debug", f"Pip komutu çalıştırılıyor: {' '.join(command)}")
            result = subprocess.run(
                command, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False
            )
            duration = time.time() - start_time
            return {"package": package, "returncode": result.returncode, "output": result.stdout + "\n" + result.stderr, "duration": duration}
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log("error", f"'{package}' kurulumunda beklenmedik bir hata: {e}", exc_info=True)
            return {"package": package, "returncode": -1, "output": str(e), "duration": duration}

    def _install_package_with_retry(self, package: str, max_retries: int = 3) -> List[str]:
        retries = 0
        current_args: List[str] = []
        
        while retries <= max_retries:
            pkg_name_only = package.split("==")[0].split("[")[0]
            command = [sys.executable, "-m", "pip", "install"] + current_args + [package]
            install_result = self._run_pip_command(command, package)
            output = install_result["output"]
            duration = install_result["duration"]

            if install_result["returncode"] == 0:
                self.logger.log("info", f"'{package}' başarıyla kuruldu.")
                success_details = self.pip_output_analyzer.analyze_success(output, pkg_name_only)
                version = success_details.get("version") or "latest"
                deps = success_details.get("dependencies", [])
                self.dependency_registry.add_package(pkg_name_only, version, deps)
                self.summary_generator.add_module_status(pkg_name_only, "Başarılı", duration, reason=f"Denenen argümanlar: {current_args}" if current_args else "Standart kurulum")
                return deps

            self.logger.log("warning", f"'{package}' kurulumu deneme #{retries + 1} başarısız oldu.")
            failure_details = self.pip_output_analyzer.analyze_failure(output)
            
            resolution = self.conflict_resolver.suggest_resolution(package, failure_details)
            action = resolution.get("action")

            if action == "retry_with_args":
                new_args = resolution.get("new_args", [])
                current_args.extend(new_args)
                self.logger.log("info", f"Yeni deneme için argümanlar güncellendi: {current_args}")
            elif action == "install_before":
                deps_to_install = resolution.get("packages", [])
                self.logger.log("info", f"Ön bağımlılıklar kuruluyor: {deps_to_install}")
                self.install_packages(deps_to_install, use_smart_installer=False) # Bunları basitçe kur
            else: # Abort
                self.logger.log("error", f"'{package}' için çözüm bulunamadı. Sebep: {resolution.get('reason')}")
                self.failed_packages[pkg_name_only] = resolution.get('reason')
                self.summary_generator.add_module_status(pkg_name_only, "Başarısız", duration, reason=resolution.get('reason'))
                return [] # Başarısız oldu

            retries += 1

        self.logger.log("error", f"'{package}' maksimum deneme sayısına ulaştı ve kurulamadı.")
        self.failed_packages[package] = "Maksimum deneme sayısına ulaşıldı."
        self.summary_generator.add_module_status(package, "Başarısız", 0, reason="Maksimum deneme")
        return []

    def check_package_installed(self, package_name: str) -> bool:
        try:
            cmd = [sys.executable, "-m", "pip", "show", package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
            return result.returncode == 0
        except Exception as e:
            self.logger.log("error", f"'{package_name}' paketi kontrol edilirken hata: {e}")
            return False

    def install_packages(self, packages: List[str], use_smart_installer: bool = True):
        # Kurulumdan önce kurulması gerekenleri filtrele
        packages_to_install = []
        for pkg in packages:
            pkg_name_only = pkg.split("==")[0].split("[")[0]
            if self.mode == 'force' or not self.check_package_installed(pkg_name_only):
                packages_to_install.append(pkg)
            else:
                self.logger.log("info", f"Paket '{pkg_name_only}' zaten kurulu, atlanıyor.")
                self.summary_generator.add_module_status(pkg_name_only, "Atlandı", 0, reason="Zaten yüklü")
        
        if not packages_to_install:
            self.logger.log("info", "Kurulacak yeni paket bulunamadı.")
            return

        if use_smart_installer and len(packages_to_install) > 1:
            self.logger.log("info", "Akıllı Kurulum Yöneticisi kullanılıyor...")
            smart_manager = SmartInstallManager(modules=packages_to_install, installer=self)
            smart_manager.run_all()
        else:
            self.logger.log("info", "Standart (sıralı) kurulum yöneticisi kullanılıyor...")
            for pkg in packages_to_install:
                self._install_package_with_retry(pkg)

    def wait_for_installs_to_complete(self, timeout: Optional[float] = None):
        self.logger.log("info", "Tüm kurulum görevlerinin tamamlanması bekleniyor...")
        # ThreadPoolExecutor'ın kapanmasını bekle
        self.installation_pool.shutdown(wait=True)
        self.logger.log("info", "Tüm kurulum görevleri tamamlandı.")

    def shutdown(self):
        self.logger.log("info", "AutoImporter kapatılıyor...")
        self.shutdown_manager.shutdown()
        self.summary_generator.print_summary()

# --- Ana Çalıştırma Bloğu ---
def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Tee loglamasını başlat
    tee_stdout, tee_stderr = None, None
    try:
        colorama.init(autoreset=True)
        
        parser = argparse.ArgumentParser(description="PDS-X Auto-Importer: Python paketlerini akıllıca kurun ve yönetin.")
        parser.add_argument('packages', nargs='*', help="Kurulacak paketlerin listesi")
        parser.add_argument('--file', '-f', help="Paket listesini içeren bir requirements.txt dosyası.")
        parser.add_argument('--mode', choices=['auto', 'force', 'dry-run'], default='auto', help="Kurulum modu.")
        parser.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="Log seviyesi.")
        parser.add_argument('--monitor', action='store_true', help="Terminal loglarını gerçek zamanlı izle (henüz tam olarak entegre değil).")
        args = parser.parse_args()

        # Tee'yi başlatmadan önce importer'ı başlat ki logger'ı olsun
        importer = AutoImporter(mode=args.mode, log_level=args.level, monitor_terminal=args.monitor)
        
        # Tee'yi şimdi başlat
        tee_stdout = Tee(sys.stdout, TERMINAL_LOG_FILE)
        tee_stderr = Tee(sys.stderr, TERMINAL_LOG_FILE)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        # Kapanışta Tee'yi kapatmayı kaydet
        importer.shutdown_manager.register_task("Tee Stdout", tee_stdout.close)
        importer.shutdown_manager.register_task("Tee Stderr", tee_stderr.close)

        packages_to_install = args.packages
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    packages_to_install.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
            except FileNotFoundError:
                importer.logger.log("error", f"Requirements dosyası bulunamadı: {args.file}")
                importer.shutdown()
                return

        if not packages_to_install:
            importer.logger.log("warning", "Kurulacak paket belirtilmedi. Varsayılan dosyalar kontrol ediliyor.")
            for file in ['pdsx_requirements.txt', 'requirements.txt']:
                if os.path.exists(file):
                    importer.logger.log("info", f"'{file}' bulundu ve okunuyor.")
                    with open(file, 'r') as f:
                        packages_to_install.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
        
        if packages_to_install:
            unique_packages = sorted(list(set(packages_to_install)))
            importer.logger.log("info", f"Kurulum için hazırlanan paketler: {unique_packages}")
            importer.install_packages(unique_packages)
        else:
            importer.logger.log("info", "Kurulacak yeni paket bulunamadı.")

        importer.shutdown()

    except Exception as e:
        print(f"{colorama.Fore.RED}KRİTİK HATA: {e}{colorama.Style.RESET_ALL}")
        traceback.print_exc()
    finally:
        # Her durumda orijinal stdout/stderr'i geri yükle
        if isinstance(sys.stdout, Tee):
            sys.stdout.original_stream.flush()
            sys.stdout = sys.stdout.original_stream
        if isinstance(sys.stderr, Tee):
            sys.stderr.original_stream.flush()
            sys.stderr = sys.stderr.original_stream

if __name__ == '__main__':
    main()