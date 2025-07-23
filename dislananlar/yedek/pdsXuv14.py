# Bu ana program, PDS-X BASIC v14u yorumlayıcısını başlatır.
# bu ana program, diger tum pdsX ana modullerinin birlestirilecegi ana programdir
# pylint: skip-file
#  flake8: noqa
import os, sys, subprocess, json, atexit
from pathlib import Path
import logging

# Log handler'ları temizle ve yeni ayarlar
for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)

# Temel log ayarları
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Terminal handler - UTF-8 encoding
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.root.addHandler(console_handler)

# Dosya handler'ları - UTF-8 encoding
def setup_file_handler(filename):
    handler = logging.FileHandler(filename, mode='a', encoding='utf-8', errors='ignore')
    handler.setFormatter(log_formatter)
    return handler

# Her dosya için ayrı handler
error_handler = setup_file_handler("pdsxu_errors.log")
info_handler = setup_file_handler("pdsxu_info.log")
terminal_handler = setup_file_handler("pdsxu_terminal.log")

# Handler'ları ana logger'a ekle
logging.root.addHandler(error_handler)
logging.root.addHandler(info_handler)
logging.root.addHandler(terminal_handler)

# Log seviyesi ayarı
logging.root.setLevel(logging.DEBUG)

# Çıkışta handler'ları temizle
def cleanup_handlers():
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
atexit.register(cleanup_handlers)


# AutoImporter ve EnvManager - PDS-X entegrasyon
print("[PDS-X] AutoImporter modülü import ediliyor...")

# auto_importer.py ana modülü
try:
    from auto_importer import EnvManager, AutoImporter
    print("[PDS-X] ✅ PDS-X AutoImporter ana modül başarıyla yüklendi!")
    IMPORT_SUCCESS = True
    
except ImportError as e:
    print(f"[PDS-X] ⚠️  auto_importer import hatası: {e}")
    IMPORT_SUCCESS = False
    
    # Last resort: Minimal AutoImporter sınıfı
    class EnvManager:
        def __init__(self):
            self.python_exe = sys.executable
            
        def is_running_in_venv(self):
            return 'venv' in sys.executable or 'conda' in sys.executable
                
            def setup_environment(self, **kwargs):
                print("[EnvManager] Minimal setup - ortam hazır sayılıyor")
                return True
                
        class AutoImporter:
            """Minimal AutoImporter fallback class"""
            def __init__(self, entry_point_script=None, mode="NORMAL",
                         log_to_file=True, log_to_terminal=True, **kwargs):
                self.entry_point_script = entry_point_script
                self.mode = mode
                self.log_to_file = log_to_file
                self.log_to_terminal = log_to_terminal
                self.env_manager = EnvManager()
                print(f"[AutoImporter] Minimal mode başlatıldı: {mode}")
                
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
                
            def cleanup(self):
                """Temizlik"""
                print("[AutoImporter] Minimal cleanup tamamlandı")
        
        # Minimal setup fonksiyonu
        def setup_pdsX_environment(**kwargs):
            print("[PDS-X] Minimal ortam kurulumu")
            return True
            
        print("[PDS-X] ✅ Minimal AutoImporter oluşturuldu")


# Exception management - global import
try:
    from exception_manager3 import PdsXException, PdsXSyntaxError, PdsXRuntimeError
except ImportError:
    # Fallback if exception_manager3 is not available
    class PdsXException(Exception):
        def __init__(self, message, code="ERR_UNKNOWN", context=None):
            super().__init__(message)
            self.code = code
            self.context = context or {}
    
    class PdsXSyntaxError(PdsXException):
        pass
    
    class PdsXRuntimeError(PdsXException):
        pass


print("[PDS-X] -----------------------------------------------------------")   
print(f"[PDS-X] Python Executable: {sys.executable}")
print(f"[PDS-X] Python Version: {sys.version}")
print("[PDS-X] -----------------------------------------------------------")
print("[PDS-X] P R O G R A M M E R   D E V E L O P M E N T   S Y S T E M .")
print("[PDS-X]       PDS-X BASIC v14u Çok-Paradigmalı Yorumlayıcı         ")
print("[PDS-X] -----------------------------------------------------------")

# Argparse komutlarının hatırlanması için dosya
ARGS_REPLAY_FILE = Path(".pdsx_last_args.json")

def save_args_for_replay(args_list):
    """Argparse komutlarını daha sonra tekrarlamak için kaydet"""
    try:
        with open(ARGS_REPLAY_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "args": args_list,
                "timestamp": str(os.path.getctime(__file__) if os.path.exists(__file__) else "unknown"),
                "executable": sys.executable
            }, f, indent=2)
        print(f"[PDS-X] Komut argümanları kaydedildi: {args_list}")
    except Exception as e:
        print(f"[PDS-X] UYARI: Argüman kaydetme hatası: {e}")

def load_and_replay_args():
    """Kaydedilmiş argparse komutlarını yükle ve tekrarla"""
    try:
        if ARGS_REPLAY_FILE.exists():
            with open(ARGS_REPLAY_FILE, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            saved_args = saved_data.get("args", [])
            if saved_args:
                print(f"[PDS-X] Kaydedilmiş komut tekrarlanıyor: {saved_args}")
                # Dosyayı sil ki infinite loop olmasın
                ARGS_REPLAY_FILE.unlink()
                return saved_args
            else:
                print("[PDS-X] Kaydedilmiş komut boş, REPL moduna geçiliyor")
                ARGS_REPLAY_FILE.unlink()
        return None
    except Exception as e:
        print(f"[PDS-X] UYARI: Argüman tekrar yükleme hatası: {e}")
        return None

# Başlangıçta sistem argümanlarını kaydet (dependency yüklemesinden sonra replay için)
original_args = sys.argv[1:] if len(sys.argv) > 1 else []

# Sanal ortamı hazırla - sadece gerektiğinde
env_manager = EnvManager()

# AutoImporter'ı başlat ve yapılandır
try:
    auto_importer_instance = AutoImporter(
        entry_point_script=__file__,
        mode="NORMAL",
        log_to_file=True,
        log_to_terminal=True
    )
    auto_importer_instance.initialize_components()
    auto_importer_instance.start_background_services()
    
    # Graceful shutdown için kaydet
    import atexit
    atexit.register(auto_importer_instance.shutdown)
    
except Exception as e:
    print(f"[PDS-X] KRİTİK HATA: AutoImporter başlatılamadı: {e}", file=sys.stderr)
    # AutoImporter olmadan devam etmeyi deneyebilir veya çıkabiliriz.
    # Şimdilik devam etmesine izin verelim.
    auto_importer_instance = None

# --- Sanal Ortam ve Bağımlılık Yönetimi (AutoImporter tarafından yönetilir) ---
# AutoImporter örneği başarıyla oluşturulduysa, ortam kurulumunu ona devret.
if auto_importer_instance and auto_importer_instance.env_manager:
    env_manager = auto_importer_instance.env_manager
    
    # Sanal ortamın kurulu ve aktif olup olmadığını kontrol et.
    # setup_environment metodu, zaten kuruluysa yeniden kurmaz, sadece aktif eder.
    if not env_manager.is_running_in_venv():
        print("[PDS-X] Sanal ortam kurulumu AutoImporter ile başlatılıyor...")
        
        # Orijinal komut argümanlarını tekrar çalıştırmak için kaydet
        if original_args:
            save_args_for_replay(original_args)
        
        # Kurulumu ve yeniden başlatmayı tetikle
        if not env_manager.setup_environment(restart_on_success=True):
            print("[PDS-X] HATA: AutoImporter ile ortam hazırlığı başarısız oldu.", file=sys.stderr)
            sys.exit(1)
        # setup_environment(restart=True) çağrıldığında betik yeniden başlayacağı için
        # normalde buraya kodun ulaşmaması gerekir. Ulaşırsa bir sorun vardır.
        print("[PDS-X] UYARI: Betik yeniden başlatılamadı. Manuel yeniden başlatma gerekebilir.")
        sys.exit(0)

    else:
        print("[PDS-X] Sanal ortam zaten aktif, devam ediliyor...")
        # Ortam zaten aktifse, kaydedilmiş bir komut olup olmadığını kontrol et ve çalıştır.
        replay_args = load_and_replay_args()
        if replay_args:
            sys.argv = [sys.argv[0]] + replay_args
            print(f"[PDS-X] Tekrarlanan komut ile yeniden çalıştırılıyor: {sys.argv}")

else:
    print("[PDS-X] UYARI: AutoImporter veya EnvManager başlatılamadığı için ortam kontrolü atlandı.", file=sys.stderr)


print(f"[PDS-X] Python Executable: {sys.executable}")
print(f"[PDS-X] Python Version: {sys.version}/n")
print("[PDS-X] -----------------------------------------------------------")
print("[PDS-X] P R O G R A M M E R   D E V E L O P M E N T   S Y S T E M .")
print("[PDS-X]       PDS-X BASIC v14u Çok-Paradigmalı Yorumlayıcı         ")
print("[PDS-X] -----------------------------------------------------------")
print("[PDS-X]       Zuhtu Mete Dinler tarafından geliştirildi.           ")
print("[PDS-X] -----------------------------------------------------------")

if sys.version_info.major != 3 or sys.version_info.minor < 10:
    print("[PDS-X] UYARI: Python 3.10 veya üzeri önerilir. Devam ediliyor.")

# pdsXu.py - PDS-X BASIC v14u Çok-Paradigmalı Yorumlayıcı
# Version: 14u
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir: Mete Dinler duzeltme: github copilot)

import os
import re
import time
import json
import logging
import argparse
import ast
import pickle
import threading
import asyncio
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
from collections import deque, defaultdict
from pathlib import Path
import decimal
import glob

import importlib
import importlib.util
# Dynamically load core2-6 module due to hyphen in filename
spec = importlib.util.spec_from_file_location("core2_6", os.path.join(os.path.dirname(__file__), "core2-6.py"))
core2_6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core2_6)
CoreManager = core2_6.CoreManager

# Dynamically load bytecode_engine module due to parentheses in filename
spec_bytecode = importlib.util.spec_from_file_location("bytecode_engine_core2duo", os.path.join(os.path.dirname(__file__), "bytecode_engine(core2duo).py"))
bytecode_engine_module = importlib.util.module_from_spec(spec_bytecode)
spec_bytecode.loader.exec_module(bytecode_engine_module)
BytecodeEngine = bytecode_engine_module.BytecodeEngine

from typing import Callable, Optional, Any

# Core Imports 
from module_manager import ModuleManager
from bytecode_compiler import BytecodeCompiler
from bytecode_manager import BytecodeManager

# LibX Imports
from libx_jit import LibXJIT
from libx_data import LibXData
from libx_logic import LibXLogic
from libx_gui import LibXGui
from libx_concurrency import LibXConcurrency, AsyncManager
from libx_nlp import LibXNLP
from libx_network import LibXNetwork
from lib_db import LibDB
from sqlite import SQLiteManager

from lowlevel import LowLevelManager
from memory_manager import MemoryManager
from pipe3 import PipeManager

# Python 3.10 kontrollü modüller - hata varsa bypass et
try:
    from tree3 import TreeManager  # Güncellendi: tree -> tree3
except SystemExit:
    print("[PDS-X] UYARI: tree3 modülü Python sürüm kontrolü nedeniyle atlandı")
    TreeManager = None
except ImportError as e:
    print(f"[PDS-X] UYARI: tree3 import hatası: {e}")
    TreeManager = None

from graph2 import GraphManager  # Güncellendi: graph -> graph2
from functional2 import FunctionalManager  # Güncellendi: functional -> functional2
from save import SaveManager
from f11_backtrace_logger import BacktraceLogger
from f12_timer_manager import TimerManager
from reply_extension import ReplyExtension
from data_structures import DataStructures
from oop_and_class2 import OOPManager
from save_load_system2 import SaveLoadSystem
from multithreading_process import MultithreadingProcessManager
from database_sql_isam import DatabaseManager
from pipe_monitor_gui import PipeMonitorGUIManager
from export_report_doc import ExportReportDocManager

from command_executor import execute_command

__version__ = "14u"

# Yardımcı Desenler
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
STR_RE = re.compile(r'^".*?"$|^\'.*?\'')

def _to_num(s: str):
    return float(s) if '.' in s else int()

# Yüksek hassasiyetli kayan nokta tipleri
class Float128(decimal.Decimal):
    def __new__(cls, value=0):
        context = decimal.Context(prec=34)
        return decimal.Decimal.__new__(cls, str(value), context)

class Float256(decimal.Decimal):
    def __new__(cls, value=0):
        context = decimal.Context(prec=68)
        return decimal.Decimal.__new__(cls, str(value), context)

class Float512(decimal.Decimal):
    def __new__(cls, value=0):
        context = decimal.Context(prec=136)
        return decimal.Decimal.__new__(cls, str(value), context)

class PluginManager:
    """Dinamik plugin yönetimi (eklenti yükleme/çıkarma/listeleme)."""
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {}
        Path(plugin_dir).mkdir(exist_ok=True)

    def load_plugin(self, plugin_name):
        try:
            sys.path.insert(0, self.plugin_dir)
            module = importlib.import_module(plugin_name)
            self.plugins[plugin_name] = module
            logging.info(f"Plugin yüklendi: {plugin_name}")
            return module
        except Exception as e:
            logging.error(f"Plugin yükleme hatası: {plugin_name}: {e}")
            raise PdsXException(f"Plugin yükleme hatası: {plugin_name}: {e}")

    def unload_plugin(self, plugin_name):
        if plugin_name in self.plugins:
            del sys.modules[plugin_name]
            del self.plugins[plugin_name]
            logging.info(f"Plugin çıkarıldı: {plugin_name}")
        else:
            raise PdsXException(f"Plugin bulunamadı: {plugin_name}")

    def list_plugins(self):
        return list(self.plugins.keys())

    def discover_plugins(self):
        return [Path(f).stem for f in glob.glob(f"{self.plugin_dir}/*.py")]

class ExceptionManager:
    """Gelişmiş hata yönetimi ve loglama.""" 
    def __init__(self, interpreter):
        self.interpreter = interpreter
    async def handle_error(self, exc):
        logging.error(f"Exception: {exc}")
        print(f"[HATA] {exc}")

class PDSXIntegrator:
    def __init__(self):
        self.core = None
        self.memory_manager = None
        self.lowlevel = None
        self.exception_handler = None
        self.logger = None
        self.timer = None
        self.data_structures = {}
        self.libx_modules = {}
        
    def init_core_components(self):
        """Çekirdek bileşenleri başlat"""
        from core import Core
        from memory_manager import MemoryManager
        from lowlevel import LowLevel
        
        self.core = Core()
        self.memory_manager = MemoryManager()
        self.lowlevel = LowLevel()        
        # Bağımlılıkları ayarla
        self.core.set_memory_manager(self.memory_manager)
        self.memory_manager.set_lowlevel(self.lowlevel)
        
    def init_infrastructure(self):
        """Temel altyapıyı başlat"""
        from exception_manager3 import PdsXException, PdsXSyntaxError, PdsXRuntimeError
        from f11_backtrace_logger import BacktraceLogger
        from f12_timer_manager import TimerManager
        
        # Exception classes'ı global olarak kaydet
        self.PdsXException = PdsXException
        self.PdsXSyntaxError = PdsXSyntaxError
        self.PdsXRuntimeError = PdsXRuntimeError
        
        self.logger = BacktraceLogger()
        self.timer = TimerManager()
        
    def init_data_structures(self):
        """Veri yapılarını başlat"""
        from tree3 import TreeManager as Tree # tree -> tree3 olarak güncellendi
        from graph2 import GraphManager as Graph # graph -> graph2 olarak güncellendi
        from data_structures import DataStructures
        
        self.data_structures = {
            'tree': Tree(),
            'graph': Graph(),
            'general': DataStructures()
        }
        
    def init_libx_modules(self):
        """LibX modüllerini başlat"""
        from libx_data import LibXData
        from libx_concurrency import LibXConcurrency
        from libx_logic import LibXLogic
        
        self.libx_modules = {
            'data': LibXData(),
            'concurrency': LibXConcurrency(),
            'logic': LibXLogic()
        }
        
    def initialize_all(self):
        """Tüm bileşenleri sırayla başlat"""
        print("[PDS-X] Modül entegrasyonu başlatılıyor...")
        
        try:
            self.init_core_components()
            print("[PDS-X] Çekirdek bileşenler hazır.")
            
            self.init_infrastructure()
            print("[PDS-X] Altyapı sistemleri hazır.")
            
            self.init_data_structures() 
            print("[PDS-X] Veri yapıları hazır.")
            
            self.init_libx_modules()
            print("[PDS-X] LibX modülleri hazır.")
            
            print("[PDS-X] Modül entegrasyonu tamamlandı.")
            return True
            
        except Exception as e:
            print(f"[PDS-X] HATA: Modül entegrasyonu başarısız: {str(e)}")
            return False

def initialize_bytecode_managers():
    """Bytecode yönetim sistemini başlatır."""
    global bytecode_compiler, bytecode_manager
    
    # Bytecode yöneticilerini başlat 
    bytecode_compiler = BytecodeCompiler()
    bytecode_manager = BytecodeManager()

    # Asenkron döngüyü başlat
    bytecode_manager.start_async_loop()

    logging.info("[PDS-X] Bytecode sistemi başlatıldı")
    return True

# --- Çekirdek modül fonksiyon ve veri yapılarının otomatik entegrasyonu ---
CORE_MODULES_LIST = [
    ('PIPE3', PipeManager),
    ('TREE3', TreeManager),
    ('GRAPH2', GraphManager),
    ('FUNCTIONAL2', FunctionalManager),
    ('CORE', CoreManager),
    ('LIBX_JIT', LibXJIT),
    ('LIBX_DATA', LibXData),
    ('LIBX_LOGIC', LibXLogic),
    ('LIBX_GUI', LibXGui),
    ('LIBX_CONCURRENCY', LibXConcurrency),
    ('LIBX_NLP', LibXNLP),
    ('LIBX_NETWORK', LibXNetwork),
    ('LIB_DB', LibDB),
    ('SQLITE', SQLiteManager),
    ('LOWLEVEL', LowLevelManager),
    ('MEMORY_MANAGER', MemoryManager),
    ('SAVE', SaveManager),
    ('F11_BACKTRACE_LOGGER', BacktraceLogger),
    ('F12_TIMER_MANAGER', TimerManager),
    ('REPLY_EXTENSION', ReplyExtension),
    ('DATA_STRUCTURES', DataStructures),
    ('OOP_AND_CLASS2', OOPManager),
    ('SAVE_LOAD_SYSTEM2', SaveLoadSystem),
    ('MULTITHREADING_PROCESS', MultithreadingProcessManager),
    ('DATABASE_SQL_ISAM', DatabaseManager),
    ('PIPE_MONITOR_GUI', PipeMonitorGUIManager),
    ('EXPORT_REPORT_DOC', ExportReportDocManager)
]

def _add_core_module_exports_to_tables(function_table, type_table):
    for mod_name, mod_obj in CORE_MODULES_LIST:
        # Fonksiyonlar
        for attr in dir(mod_obj):
            if not attr.startswith('_'):
                meth = getattr(mod_obj, attr)
                if callable(meth):
                    key = f"{mod_name}_{attr}".upper()
                    if key not in function_table:
                        function_table[key] = meth
        # Sınıflar ve veri yapıları
        if hasattr(mod_obj, '__name__'):
            type_key = mod_name.upper()
            if type_key not in type_table:
                type_table[type_key] = mod_obj

class PdsXv14uInterpreter:
    def __init__(self):
        # Değişkenler ve Kapsamlar
        self.global_vars = {}
        self.shared_vars = defaultdict(list)
        self.local_scopes = [{}]
        self.types = {}
        self.classes = {}
        self.interfaces = {}
        self.functions = {}
        self.subs = {}
        self.labels = {}
        
        # Program ve Yürütme
        self.program = []
        self.program_counter = 0
        self.call_stack = []
        self.running = False
        self.bytecode = []
        
        # Yöneticiler
        self.core = CoreManager(self)  # YENİ
        self.jit_manager = LibXJIT(self)
        self.data_manager = LibXData(self)
        self.logic_manager = LibXLogic(self)
        self.gui_manager = LibXGui(self)
        self.concurrency_manager = LibXConcurrency(self)
        self.async_manager = AsyncManager()
        self.nlp_manager = LibXNLP(self)
        self.network_manager = LibXNetwork(self)
        self.db_manager = LibDB(self)
        self.sqlite_manager = SQLiteManager(self)
        self.bytecode_compiler = BytecodeCompiler()
        self.bytecode_manager = BytecodeManager(self)
        self.lowlevel_manager = LowLevelManager(self)
        self.memory_manager = MemoryManager(self)
        self.pipe_manager = PipeManager(self)
        self.tree_manager = TreeManager(self)
        self.graph_manager = GraphManager(self)
        self.functional_manager = FunctionalManager(self)
        self.save_manager = SaveManager(self)
        self.backtrace_logger = BacktraceLogger(self)
        self.timer_manager = TimerManager(self.gui_manager)
        self.repl_extensions = ReplyExtension(self)
        self.data_structures = DataStructures(self)
        self.oop_manager = OOPManager(self)
        self.save_load_system = SaveLoadSystem(self)
        self.multithreading_manager = MultithreadingProcessManager(self)
        self.module_manager = ModuleManager(os.path.dirname(os.path.abspath(__file__)))
        self.database_isam = DatabaseManager(self)
        self.pipe_monitor_gui = PipeMonitorGUIManager(self)
        self.report_exporter = ExportReportDocManager(self)
        self.plugin_manager = PluginManager()
        self.exception_manager = ExceptionManager(self)
        
        # Bytecode ve performans özellikleri ekleniyor
        self.bytecode_compiler = BytecodeCompiler() 
        self.bytecode_manager = BytecodeManager(self)
        self.bytecode_engine = BytecodeEngine(self) # Dinamik olarak yüklenen sınıfı kullan
        
        # Asenkron döngüyü başlat
        self.bytecode_manager.start_async_loop()
        
        # Performans izleme
        self.performance_metrics = {
            "start_time": time.time(),
            "memory_usage": 0,
            "cpu_usage": 0,
            "bytecode_stats": {
                "compiled": 0,
                "optimized": 0,
                "executed": 0
            }
        }
        
        # Yeni bytecode operasyonları
        self.bytecode_opcodes = {}
        
        # Diğer Ayarlar
        self.db_connections = {}
        self.file_handles = {}
        self.error_handler = None
        self.gosub_handler = None
        self.error_sub = None
        self.debug_mode = False
        self.trace_mode = False
        self.loop_stack = []
        self.select_stack = []
        self.if_stack = []
        self.data_list = []
        self.data_pointer = 0
        self.transaction_active = {}
        self.modules = {"core": {"functions": {}, "classes": {}, "program": []}}
        self.current_module = "main"
        self.repl_mode = False
        self.language = "en"
        self.translations = self.load_translations("lang.json")
        self.async_tasks = []
        self.performance_metrics = {"start_time": time.time(), "memory_usage": 0, "cpu_usage": 0}
        self.supported_encodings = [
            "utf-8", "cp1254", "iso-8859-9", "ascii", "utf-16", "utf-32",
            "cp1252", "iso-8859-1", "windows-1250", "latin-9",
            "cp932", "gb2312", "gbk", "euc-kr", "cp1251", "iso-8859-5",
            "cp1256", "iso-8859-6", "cp874", "iso-8859-7", "cp1257", "iso-8859-8",
            "utf-8-sig", "utf-8-bom-less"
        ]
        self.expr_cache = {}
        
        # Tip Tablosu
        self.type_table = {
            "STRING": str, "INTEGER": int, "LONG": int, "SINGLE": float, "DOUBLE": float,
            "BYTE": int, "SHORT": int, "UNSIGNED INTEGER": int, "CHAR": str,
            "LIST": list, "DICT": dict, "SET": set, "TUPLE": tuple,
            "ARRAY": np.array, "DATAFRAME": pd.DataFrame, "POINTER": None,
            "STRUCT": dict, "UNION": None, "ENUM": dict, "VOID": None, "BITFIELD": int,
            "FLOAT128": Float128, "FLOAT256": Float256, "FLOAT512": Float512, "STRING8": str, "STRING16": str,
            "BOOLEAN": bool, "NULL": type(None), "NAN": float
        }
        
        # Fonksiyon Tablosu (CoreManager ve diğer modüllerden)
        self.function_table = {
            "MID$": lambda s, start, length: s[start-1:start-1+length],
            "ABS": abs, "INT": int,
            "LEFT$": lambda s, n: s[:n], "RIGHT$": lambda s, n: s[-n:],
            "LTRIM$": lambda s: s.lstrip(), "RTRIM$": lambda s: s.rstrip(),
            "STRING$": lambda n, c: c * n, "SPACE$": lambda n: " " * n,
            "INSTR": lambda start, s, sub: s.find(sub, start-1) + 1,
            "UCASE$": lambda s: s.upper(), "LCASE$": lambda s: s.lower(),
            "STR$": lambda n: str(n), "SQR": np.sqrt, "SIN": np.sin,
            "COS": np.cos, "TAN": np.tan, "LOG": np.log, "EXP": np.exp,
            "ATN": np.arctan, "FIX": lambda x: int(x), "ROUND": lambda x, n=0: round(x, n),
            "SGN": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
            "MOD": lambda x, y: x % y, "MIN": lambda *args: min(args),
            "MAX": lambda *args: max(args), "TIMER": lambda: time.time(),
            "DATE$": lambda: time.strftime("%m-%d-%Y"),
            "TIME$": lambda: time.strftime("%H:%M:%S"),
            "INKEY$": lambda: input()[:1], "ENVIRON$": lambda var: os.environ.get(var, ""),
            "COMMAND$": lambda: " ".join(sys.argv[1:]),
            "CSRLIN": lambda: 1, "POS": lambda x: 1, "VAL": lambda s: float(s) if s.replace(".", "").isdigit() else 0,
            "ASC": lambda c: ord(c[0]),
            "MEAN": np.mean, "MEDIAN": np.median, "MODE": lambda x: stats.mode(x)[0][0],
            "STD": np.std, "VAR": np.var, "SUM": np.sum, "PROD": np.prod,
            "PERCENTILE": np.percentile, "QUANTILE": np.quantile,
            "CORR": lambda x, y: np.corrcoef(x, y)[0, 1], "COV": np.cov,
            "DESCRIBE": lambda df: df.describe(), "GROUPBY": lambda df, col: df.groupby(col),
            "FILTER": lambda df, cond: df.query(cond), "SORT": lambda df, col: df.sort_values(col),
            "HEAD": lambda df, n=5: df.head(n), "TAIL": lambda df, n=5: df.tail(n),
            "MERGE": lambda df1, df2, on: pd.merge(df1, df2, on=on),
            "TTEST": lambda sample1, sample2: stats.ttest_ind(sample1, sample2),
            "CHISQUARE": lambda observed: stats.chisquare(observed),
            "ANOVA": lambda *groups: stats.f_oneway(*groups),
            "REGRESS": lambda x, y: stats.linregress(x, y),
            # CoreManager Fonksiyonları
            "PDF_READ_TEXT": self.core.pdf_read_text,
            "PDF_EXTRACT_TABLES": self.core.pdf_extract_tables,
            "WEB_GET": self.core.web_get,
            "SYSTEM": self.core.system,
            "EACH": self.core.each,
            "SELECT": self.core.select,
            "INSERT": self.core.insert,
            "REMOVE": self.core.remove,
            "POP": self.core.pop,
            "CLEAR": self.core.clear,
            "SLICE": self.core.slice,
            "KEYS": self.core.keys,
            "TIME_NOW": self.core.time_now,
            "DATE_NOW": self.core.date_now,
            "TIMER": self.core.timer,
            "RANDOM_INT": self.core.random_int,
            "ASSERT": self.core.assert_,
            "LOG": self.core.log,
            "IFTHEN": self.core.ifthen,
            "EXISTS": self.core.exists,
            "MKDIR": self.core.mkdir,
            "GETENV": self.core.getenv,
            "EXIT": self.core.exit,
            "JOIN_PATH": self.core.join_path,
            "COPY_FILE": self.core.copy_file,
            "MOVE_FILE": self.core.move_file,
            "DELETE_FILE": self.core.delete_file,
            "FLOOR": self.core.floor,
            "CEIL": self.core.ceil,
            "SPLIT": self.core.split,
            "JOIN": self.core.join,
            "READ_LINES": self.core.read_lines,
            "WRITE_JSON": self.core.write_json,
            "READ_JSON": self.core.read_json,
            "LIST_DIR": self.core.list_dir,
            "PING": self.core.ping,
            "SUM": self.core.sum,
            "MEAN": self.core.mean,
            "MIN": self.core.min,
            "MAX": self.core.max,
            "ROUND": self.core.round,
            "TRIM": self.core.trim,
            "REPLACE": self.core.replace,
            "FORMAT": self.core.format,
            "TRACE": self.core.trace,
            "TRY_CATCH": self.core.try_catch,
            "SLEEP": self.core.sleep,
            "DATE_DIFF": self.core.date_diff,
            "WAIT": self.concurrency_manager.wait,
            "MERGE": self.core.merge,
            "SORT": self.core.sort,
            "MEMORY_USAGE": self.core.memory_usage,
            "CPU_COUNT": self.core.cpu_count,
            "TYPE_OF": self.core.type_of,
            "IS_EMPTY": self.core.is_empty,
            "LEN": self.core.len,
            "VAL": self.core.val,
            "STR": self.core.str,
            "LISTFILE": self.core.listfile,
            "STACK": self.core.stack,
            "PUSH": self.core.push,
            "POP": self.core.pop,
            "QUEUE": self.core.queue,
            "ENQUEUE": self.core.enqueue,
            "DEQUEUE": self.core.dequeue,
            "RUNASYNC_CORE": self.core.execute_async,
            "RUNASYNC_MT": self.multithreading_manager.execute_async,
            "RUNASYNC_BYTECODE": self.bytecode_manager.execute_async,
            "RUNASYNC_PIPE": self.pipe_manager.execute_async,
            # Düşük Seviye
            "BITSET": self.lowlevel_manager.bitset,
            "BITGET": self.lowlevel_manager.bitget,
            "MEMCPY": self.lowlevel_manager.memcpy,
            "MEMSET": self.lowlevel_manager.memset,
            # Fonksiyonel
            "MAP": self.functional_manager.map,
            "FILTER": self.functional_manager.filter,
            "REDUCE": self.functional_manager.reduce,
            "OMEGA": self.functional_manager.omega,
            # Diğer
            "NEW": self.memory_manager.allocate,
            "DELETE": self.memory_manager.release,
            "SIZEOF": self.memory_manager.sizeof,
            "ASYNC_WAIT": self.core.async_wait,
            "THREAD_COUNT": threading.active_count,
            "CURRENT_THREAD": threading.get_ident
        }
        
        # Çekirdek modül fonksiyon ve veri yapılarının otomatik entegrasyonu
        _add_core_module_exports_to_tables(self.function_table, self.type_table)

        # Operatör Tablosu
        self.operator_table = {
            '++': lambda x: x + 1,
            '--': lambda x: x - 1,
            '<<': lambda x, y: x << y,
            '>>': lambda x, y: x >> y,
            '&': lambda x, y: x & y,
            '|': lambda x, y: x | y,
            '^': lambda x, y: x ^ y,
            '~': lambda x: ~x,
            'AND': lambda x, y: x and y,
            'OR': lambda x, y: x or y,
            'XOR': lambda x, y: bool(x) != bool(y),
            'NOT': lambda x: not x,
            'IMP': lambda x, y: not x or y,          # Implication (ise)
            'EQV': lambda x, y: x == y,              # Equivalence (çift yönlü ise / XNOR)
            'XNOR': lambda x, y: x == y,             # XNOR
            'NAND': lambda x, y: not (x and y),      # NAND
            'NOR': lambda x, y: not (x or y),        # NOR
            'P': lambda x, y: not (x or y),          # Peirce Oku (↓)
            'S': lambda x, y: not (x and y),         # Sheffer Çubuğu (|)
            '+=': lambda x, y: x + y,
            '-=': lambda x, y: x - y,
            '*=': lambda x, y: x * y,
            '/=': lambda x, y: x / y,
            '%=': lambda x, y: x % y,
            '&=': lambda x, y: x & y,
            '|=': lambda x, y: x | y,
            '^=': lambda x, y: x ^ y,
            '<<=': lambda x, y: x << y,
            '>>=': lambda x, y: x >> y,
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            '<': lambda x, y: x < y,
            '>': lambda x, y: x > y,
            '<=': lambda x, y: x <= y,
            '>=': lambda x, y: x >= y,
            '<>': lambda x, y: x != y,
            '&&': lambda x, y: x and y,
            '||': lambda x, y: x or y,
            '%': lambda x, y: x % y,
            '**': lambda x, y: x ** y,
            '//': lambda x, y: x // y
        }
        self.execute_command = execute_command  # execute_command metodunu ekle

    def load_translations(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Dil dosyası bulunamadı. Varsayılan İngilizce kullanılacak.")
            return {
                "en": {"PRINT": "Print", "ERROR": "Error", "LET": "Let", "DIM": "Dim"},
                "tr": {"PRINT": "Yaz", "ERROR": "Hata", "LET": "Atama", "DIM": "Tanımla"}
            }

    def translate(self, key):
        return self.translations.get(self.language, {}).get(key, key)

    def current_scope(self):
        return self.local_scopes[-1]

    def evaluate_expression(self, expr, scope_name=None):
        cache_key = (expr, scope_name)
        if cache_key not in self.expr_cache:
            try:
                tree = ast.parse(expr, mode='eval')
                self.expr_cache[cache_key] = compile(tree, '<string>', 'eval')
            except SyntaxError:
                raise PdsXException(f"Geçersiz ifade: {expr}")
        namespace = {}
        namespace.update(self.global_vars)
        namespace.update(self.current_scope())
        namespace.update(self.function_table)
        try:
            return eval(self.expr_cache[cache_key], namespace)
        except Exception as e:
            raise PdsXException(f"İfade değerlendirme hatası: {expr}, {str(e)}")

    async def load_config(self, config_file):
        try:
            import aiofiles
            async with aiofiles.open(config_file, "r", encoding="utf-8") as f:
                self.config = json.loads(await f.read())
            logging.info(f"Yapılandırma yüklendi: {config_file}")
        except Exception as e:
            await self.exception_manager.handle_error(f"Config yükleme hatası: {e}")
            self.config = {}

    async def load_program(self, file_path):
        try:
            import aiofiles
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                code = await f.read()
            self.parse_program(code)
            logging.info(f"Program yüklendi: {file_path}")
        except Exception as e:
            await self.exception_manager.handle_error(f"Program yükleme hatası: {e}")

    async def execute_command_async(self, command, scope_name=None):
        try:
            if isinstance(command, str) and command.strip().upper().startswith("REPLY"):
                self.repl_extensions.parse_reply_command(command)
                return None
            result = self.execute_command(command, scope_name)
            return result
        except Exception as e:
            await self.exception_manager.handle_error(e)
            return None

    async def run_async(self):
        self.running = True
        self.program_counter = 0
        while self.running and self.program_counter < len(self.program):
            command = self.program[self.program_counter]
            if self.debug_mode:
                print(f"DEBUG: Satır {self.program_counter + 1}: {command}")
            next_pc = await self.execute_command_async(command)
            if next_pc is not None:
                self.program_counter = next_pc
            else:
                self.program_counter += 1
        self.running = False

    async def interactive_shell(self):
        self.repl_mode = True
        print("pdsXu REPL (Async) - Çıkmak için EXIT yazın")
        while self.repl_mode:
            try:
                command = input("[pdsX-Basic]>>> ")
                if command.strip().upper() == "EXIT":
                    self.repl_mode = False
                    break
                await self.execute_command_async(command)
            except Exception as e:
                await self.exception_manager.handle_error(e)
        self.repl_mode = False

    # Alias run method to run_async for backward compatibility
    run = run_async

# --- Otomatik modül entegrasyonu ve komut parser güncellemesi (güncellenmiş) ---
KULLANILACAK_MODULLER = [
    "module_manager", "bytecode_compiler", "bytecode_manager",
    "core2_6", "libx_jit", "libx_data", "libx_logic", "libx_gui",
    "libx_concurrency", "libx_nlp", "libx_network", "lib_db", "sqlite",
    "lowlevel", "memory_manager", "pipe3", "tree3", "graph2", "functional2",
    "save", "f11_backtrace_logger", "f12_timer_manager", "reply_extension",
    "data_structures", "oop_and_class2", "save_load_system2", "multithreading_process",
    "database_sql_isam", "pipe_monitor_gui", "export_report_doc"
]

MODULE_COMMAND_MAP = {}
MODULE_OBJECTS = {}
for module_name in sorted(KULLANILACAK_MODULLER, reverse=True):
    try:
        mod = importlib.import_module(module_name)
        MODULE_OBJECTS[module_name] = mod
        if hasattr(mod, "__pdsX_exports__"):
            exports = mod.__pdsX_exports__
            # functions anahtarı altındaki fonksiyonlar
            for func_name, func in exports.get("functions", {}).items():
                if callable(func):
                    cmd = f"{module_name.upper()}.{func_name.upper()}"
                    MODULE_COMMAND_MAP[cmd] = func
            # classes anahtarı altındaki staticmethod/fonksiyonlar
            for class_name, class_obj in exports.get("classes", {}).items():
                for attr in dir(class_obj):
                    if not attr.startswith("_"):
                        meth = getattr(class_obj, attr)
                        if callable(meth):
                            cmd = f"{module_name.upper()}.{class_name.upper()}_{attr.upper()}"
                            MODULE_COMMAND_MAP[cmd] = meth
    except Exception as e:
        print(f"[PDS-X] Modül otomatik entegrasyon hatası: {module_name}: {e}")

# --- Gelişmiş PDSx komut parser ---
def parse_pdsx_command(command: str, *args, **kwargs):
    """Komutun başındaki modül ve fonksiyona göre yönlendirme yapar."""
    try:
        parts = command.strip().split()
        if not parts:
            raise ValueError("Boş komut")
        cmd_key = parts[0].upper()
        if cmd_key in MODULE_COMMAND_MAP:
            func = MODULE_COMMAND_MAP[cmd_key]
            return func(*parts[1:], *args, **kwargs)
        else:
            raise ValueError(f"Bilinmeyen komut: {cmd_key}")
    except Exception as e:
        print(f"[PDS-X] Komut parse hatası: {e}")
        return None

CORE_MODULES = [
    "pdsx_exception", "pdsx_exception2", "module_manager", "core2-6", "memory_manager",
    "save_load_system2", "event", "libxcore", "auto_importer", "autoinstaller",
    "base_module_manager", "oop_and_class2", "clazz", "save", "sqlite", "reply_extension",
    "lowlevel", "data_structures", "f11_backtrace_logger", "f12_timer_manager",
    "multithreading_process", "pipe_monitor_gui", "export_report_doc", "bytecode_compiler",
    "bytecode_manager", "command_executor"
]

LOADABLE_MODULES = [
    "core2-5", "database_sql_isam", "bytecode_engine", "functional2", "graph2", "libx_ml",
    "libx_nlp", "libx_network", "pipe3", "tree3", "tree2", "tpl", "tpl3", "oop_and_class",
    "eventx", "lib_db"
]

# --- Otomatik modül entegrasyonu ve komut parser güncellemesi ---
MODULE_COMMAND_MAP = {}
MODULE_OBJECTS = {}
for module_name in sorted(LOADABLE_MODULES, reverse=True):
    try:
        mod = importlib.import_module(module_name)
        MODULE_OBJECTS[module_name] = mod
        if hasattr(mod, "__pdsX_exports__"):
            exports = mod.__pdsX_exports__
            for func_name, func in exports.get("functions", {}).items():
                cmd = f"{module_name.upper()}.{func_name.upper()}"
                MODULE_COMMAND_MAP[cmd] = func
    except Exception as e:
        print(f"[PDS-X] Modül otomatik entegrasyon hatası: {module_name}: {e}")

# --- Gelişmiş PDSx komut parser ---
def parse_pdsx_command(command: str, *args, **kwargs):
    """Komutun başındaki modül ve fonksiyona göre yönlendirme yapar."""
    try:
        parts = command.strip().split()
        if not parts:
            raise ValueError("Boş komut")
        cmd_key = parts[0].upper()
        if cmd_key in MODULE_COMMAND_MAP:
            func = MODULE_COMMAND_MAP[cmd_key]
            return func(*parts[1:], *args, **kwargs)
        else:
            raise ValueError(f"Bilinmeyen komut: {cmd_key}")
    except Exception as e:
        print(f"[PDS-X] Komut parse hatası: {e}")
        return None

def validate_modules():
    """Modülleri doğrula ve son versiyonları kontrol et.""" 
    from module_validator import validate_all_modules, ModuleVersionValidator
    
    validator = ModuleVersionValidator()
    core_results = validate_all_modules(CORE_MODULES)
    
    # Temel modülleri kontrol et
    if not all(core_results.values()):
        invalid_modules = [m for m, v in core_results.items() if not v]
        raise PdsXException(f"Temel modüllerde hata: {invalid_modules}")
        
    # Son versiyonları kontrol et
    version_results = {}
    for module_name in CORE_MODULES + LOADABLE_MODULES:
        try:
            module = importlib.import_module(module_name)
            version_results[module_name] = validator.check_module_version(module_name, module)
            if not version_results[module_name]:
                logging.warning(f"Modül versiyon uyumsuzluğu: {module_name}")
        except ImportError:
            logging.warning(f"Modül yüklenemedi: {module_name}")
            version_results[module_name] = False
            
    return version_results

class CommandParser:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.command_registry = {}
        self.function_registry = {}
        self.variable_registry = {}
        self.data_structure_registry = {}
        self.load_from_json("dependencies.json")

    def load_from_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Komutlar
            for module, commands in data.get("commands", {}).items():
                for cmd_name, cmd_info in commands.items():
                    self.register_command(cmd_name, cmd_info["function"], module, cmd_info.get("alias"), cmd_info["description"])
            # Fonksiyonlar
            for module, functions in data.get("functions", {}).items():
                for func_name, func_info in functions.items():
                    self.register_function(func_name, func_info["function"], module, func_info.get("alias"), func_info["description"])
            # Veri Yapıları
            for module, structures in data.get("data_structures", {}).items():
                for struct_name, struct_info in structures.items():
                    self.register_data_structure(struct_name, struct_info["type"], module, struct_info.get("alias"), struct_info["description"])
            logging.debug(f"Kayıtlar yüklendi: {len(self.command_registry)} komut, {len(self.function_registry)} fonksiyon")
        except Exception as e:
            logging.error(f"Kayıt yükleme hatası: {str(e)}")
            raise PdsXException(f"Kayıt yükleme hatası: {str(e)}")

    def register_command(self, command_name: str, function: Callable, module: str, alias: Optional[str] = None, description: str = ""):
        cmd_upper = command_name.upper()
        if cmd_upper in self.command_registry:
            alias = alias or f"{module}_{command_name.replace(' ', '_').upper()}"
            print(f"Çakışma: {command_name}. Alias atandı: {alias}")
        self.command_registry[cmd_upper] = (function, module, alias, description)
        if alias:
            self.command_registry[alias.upper()] = (function, module, alias, description)

    def register_function(self, func_name: str, function: Callable, module: str, alias: Optional[str] = None, description: str = ""):
        func_upper = func_name.upper()
        if func_upper in self.function_registry:
            alias = alias or f"{module}_{func_name.upper()}"
            print(f"Çakışma: {func_name}. Alias atandı: {alias}")
        self.function_registry[func_upper] = (function, module, alias, description)
        if alias:
            self.function_registry[alias.upper()] = (function, module, alias, description)

    def register_data_structure(self, struct_name: str, type_ref: Any, module: str, alias: Optional[str] = None, description: str = ""):
        struct_upper = struct_name.upper()
        if struct_upper in self.data_structure_registry:
            alias = alias or f"{module}_{struct_name.upper()}"
            print(f"Çakışma: {struct_name}. Alias atandı: {alias}")
        self.data_structure_registry[struct_upper] = (type_ref, module, alias, description)
        if alias:
            self.data_structure_registry[alias.upper()] = (type_ref, module, alias, description)

    def alias(self, original: str, alias_name: str, registry_type: str):
        original_upper = original.upper()
        if registry_type == "command":
            registry = self.command_registry
        elif registry_type == "function":
            registry = self.function_registry
        elif registry_type == "data_structure":
            registry = self.data_structure_registry
        else:
            raise PdsXException(f"Geçersiz kayıt tipi: {registry_type}")
        if original_upper in registry:
            func, module, _, desc = registry[original_upper]
            registry[alias_name.upper()] = (func, module, alias_name, desc)
            logging.debug(f"Alias atandı: {original} -> {alias_name}")
        else:
            raise PdsXException(f"Kayıt bulunamadı: {original}")

    def parse_command(self, command: str, scope_name: Optional[str] = None) -> Any:
        command = command.strip()
        if not command:
            return None
        command_upper = command.upper()

        try:
            for cmd_name, (func, module, alias, desc) in self.command_registry.items():
                if command_upper.startswith(cmd_name):
                    args_match = re.match(rf"{cmd_name}\s*(.+)?", command, re.IGNORECASE)
                    args = []
                    if args_match and args_match.group(1):
                        args_str = args_match.group(1)
                        args = [self.interpreter.evaluate_expression(a.strip(), scope_name) 
                                for a in args_str.split(",") if a.strip()]
                    result = func(*args)
                    logging.debug(f"Komut yürütüldü: {cmd_name}, args={args}")
                    return result
            
            # Dinamik modül komutları
            if command_upper.startswith("MODULE."):
                parts = command.split(".")
                if len(parts) >= 3:
                    module_name = parts[1].lower()
                    cmd_name = parts[2].split("(")[0].upper()
                    if module_name in self.interpreter.modules:
                        module = self.interpreter.modules[module_name]
                        if hasattr(module, cmd_name):
                            args_match = re.search(r"\((.*?)\)", command)
                            args = []
                            if args_match:
                                args = [self.interpreter.evaluate_expression(a.strip(), scope_name) 
                                        for a in args_match.group(1).split(",")]
                            return getattr(module, cmd_name)(*args)
            
            raise PdsXException(f"Bilinmeyen komut: {command}")
        
        except Exception as e:
            logging.error(f"Komut ayrıştırma hatası: {str(e)}")
            raise PdsXException(f"Komut ayrıştırma hatası: {str(e)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDS-X BASIC v15 Yorumlayıcısı")
    parser.add_argument("--version", action="version", version="PDS-X BASIC v15.0")
    parser.add_argument("--file", type=str, help=".basX dosyasını çalıştırır")
    parser.add_argument("--debug", action="store_true", help="Hata ayıklama modu")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli REPL modu")
    parser.add_argument("--repl", action="store_true", help="Gelişmiş REPL modu")
    parser.add_argument("--auto-import", action="store_true", help="AutoImporter demo modu")
    parser.add_argument("--setup-env", action="store_true", help="PDS-X ortam kurulumu")
    parser.add_argument("--output", type=str, help="Çıktı dosyası (CSV/JSON/YAML)")
    parser.add_argument("--config", type=str, help="Yapılandırma dosyası (JSON/TXT/YAML)")
    parser.add_argument("--silent", action="store_true", help="Konsol çıktısını kapatır")
    parser.add_argument("--profile", action="store_true", help="Performans profili")
    parser.add_argument("--trace", action="store_true", help="Komut izleme")
    parser.add_argument("--help", action="store_true", help="Çift dilli yardım (tr/en)")
    parser.add_argument("--plugin", type=str, help="Eklenti yükler")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log seviyesi")
    parser.add_argument("--lang", type=str, default="tr", help="Dil seçimi (tr/en)")
    parser.add_argument("--theme", type=str, default="dark", help="Tema (dark/light)")
    parser.add_argument("--test", type=str, help="Test çalıştırır")
    parser.add_argument("--bytecode", type=str, help="Bytecode derler/çalıştırır")
    parser.add_argument("--secure", action="store_true", help="Güvenli mod")
    parser.add_argument("--monitor", action="store_true", help="Kaynak izleme")
    return parser.parse_args()

async def main():
    args = parse_args()

    # AutoImporter kontrolleri önce yapılır
    if args.auto_import:
        print("🚀 PDS-X AutoImporter demo modu başlatılıyor...")
        try:
            from auto_importer import interactive_demo
            interactive_demo()
            return
        except ImportError as e:
            print(f"❌ AutoImporter demo başlatılamadı: {e}")
            return
    
    if args.setup_env:
        print("🔧 PDS-X ortam kurulumu başlatılıyor...")
        try:
            success = setup_pdsX_environment(
                mode="SETUP",
                args=sys.argv
            )
            if success:
                print("✅ PDS-X ortam kurulumu tamamlandı!")
            else:
                print("❌ PDS-X ortam kurulumu başarısız!")
            return
        except Exception as e:
            print(f"❌ Ortam kurulumu hatası: {e}")
            return

    # İnteraktif REPL modu kontrolü
    if args.interactive or args.repl:
        print("🚀 PDS-X İnteraktif REPL modu başlatılıyor...")
        try:
            from pdsx_repl import start_pdsx_repl
            
            # AutoImporter ile entegre REPL
            ai_instance = None
            if 'auto_importer_instance' in globals():
                ai_instance = auto_importer_instance
            
            start_pdsx_repl(ai_instance)
            return
            
        except ImportError as e:
            print(f"❌ REPL modülü bulunamadı: {e}")
            print("🔧 Basit REPL moduna geçiliyor...")
            
            # Fallback: Basit REPL
            print("""
🔧 PDS-X Basit REPL Modu
========================
Bu modda temel PDS-X komutlarını test edebilirsiniz.
Çıkmak için 'exit' yazın.
""")
            
            while True:
                try:
                    user_input = input("PDS-X> ")
                    if user_input.strip().lower() in ('exit', 'quit', '.exit'):
                        print("👋 PDS-X REPL kapatılıyor...")
                        break
                    elif user_input.strip():
                        print(f"📤 Komut alındı: {user_input}")
                        # Basit komut işleme burada yapılabilir
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 PDS-X REPL kapatılıyor...")
                    break
            return

    # Normal PDS-X yorumlayıcı
    interpreter = PdsXv14uInterpreter()

    try:
        event_manager = None
    except Exception as e:
        print(f"Event sistemi başlatılamadı: {e}")

    if args.config:
        await interpreter.load_config(args.config)

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.DEBUG))

    if args.file:
        await interpreter.load_program(args.file)
        await interpreter.run()
    elif args.test:
        import pytest
        sys.exit(pytest.main([args.test]))
    elif args.bytecode:
        try:
            from bytecode_engine_wd658160_ import BytecodeEngine
        except ImportError:
            try:
                from bytecode_engine import BytecodeEngine
            except ImportError:
                print("❌ BytecodeEngine bulunamadı")
                return
        engine = BytecodeEngine(interpreter)
        await engine.execute_bytecode(args.bytecode)
    else:
        # Argüman verilmemişse otomatik REPL
        print("📋 Dosya belirtilmedi, REPL moduna geçiliyor...")
        try:
            from pdsx_repl import start_pdsx_repl
            ai_instance = None
            if 'auto_importer_instance' in globals():
                ai_instance = auto_importer_instance
            start_pdsx_repl(ai_instance)
        except ImportError:
            print("🔧 Basit komut satırı modu aktif")
            await interpreter.interactive_shell()

if __name__ == "__main__":
    valid_modules = validate_modules()
    asyncio.run(main())
