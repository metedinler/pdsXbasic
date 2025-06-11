import sys
print(f"[PDS-X] Python Executable: {sys.executable}")
print(f"[PDS-X] Python Version: {sys.version}")
print("[PDS-X] P R O G R A M M E R   D E V E L O P M E N T   S Y S T E M .")
print("[PDS-X]       PDS-X BASIC v14u Çok-Paradigmalı Yorumlayıcı         ")

from auto_importer import install_missing_packages
install_missing_packages()

if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] HATA: Bu modül sadece Python 3.10 ortamında çalışır! Lütfen pdsX'in ana başlatıcısını kullanın.")
    sys.exit(1)

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
import importlib
import glob

# Core Imports 
from module_manager import ModuleManager
from bytecode_compiler import BytecodeCompiler
from bytecode_manager import BytecodeManager
from core2_5 import Core25Features

# LibX Imports
from libxcore import LibXCore
from libx_jit import LibXJIT
from libx_data import LibXData
from libx_logic import LibXLogic
from libx_gui import LibXGui
from libx_concurrency import LibXConcurrency, AsyncManager
from libx_nlp import LibXNLP
from libx_network import LibXNetwork
from lib_db import LibDB
from sqlite import SQLiteManager

# Modül İçe Aktarmaları (aynı dizinde)
from module_manager import ModuleManager
from libxcore import LibXCore
from libx_jit import LibXJIT
from libx_data import LibXData
from libx_logic import LibXLogic
from libx_gui import LibXGui
from libx_concurrency import LibXConcurrency, AsyncManager
from libx_nlp import LibXNLP
from libx_network import LibXNetwork
from lib_db import LibDB
from sqlite import SQLiteManager
from bytecode_compiler import BytecodeCompiler
from bytecode_manager import BytecodeManager

from lowlevel import LowLevelManager
from memory_manager import MemoryManager
from pipe import PipeManager
from event import EventManager
from tree import TreeManager
from graph import GraphManager
from functional import FunctionalManager
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

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdsXu")

from command_executor import execute_command

# Yardımcı Desenler
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
STR_RE = re.compile(r'^".*?"$|^\'.*?\'')

def _to_num(s: str):
    return float(s) if '.' in s else int()

class PdsXException(Exception):
    pass

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
            log.info(f"Plugin yüklendi: {plugin_name}")
            return module
        except Exception as e:
            log.error(f"Plugin yükleme hatası: {plugin_name}: {e}")
            raise PdsXException(f"Plugin yükleme hatası: {plugin_name}: {e}")

    def unload_plugin(self, plugin_name):
        if plugin_name in self.plugins:
            del sys.modules[plugin_name]
            del self.plugins[plugin_name]
            log.info(f"Plugin çıkarıldı: {plugin_name}")
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
        log.error(f"Exception: {exc}")
        print(f"[HATA] {exc}")

class PDSXIntegrator:
    def __init__(self):
        self.core = None
        self.memory_manager = None
        self.lowlevel = None
        self.event_system = None
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
        from event import EventSystem
        from pdsx_exception import PdsXException
        from f11_backtrace_logger import BacktraceLogger
        from f12_timer_manager import TimerManager
        
        self.event_system = EventSystem()
        self.exception_handler = PdsXException()
        self.logger = BacktraceLogger()
        self.timer = TimerManager()
        
        # Bağımlılıkları ayarla
        self.exception_handler.set_logger(self.logger)
        self.event_system.set_timer(self.timer)
        
    def init_data_structures(self):
        """Veri yapılarını başlat"""
        from tree import Tree
        from graph import Graph
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
    global bytecode_compiler, bytecode_manager, core25_features
    
    # Bytecode yöneticilerini başlat
    bytecode_compiler = BytecodeCompiler()
    bytecode_manager = BytecodeManager()
    core25_features = Core25Features()

    # Core2duo özelliklerini bytecode_manager'a bağla
    bytecode_manager.register_core_features(core25_features)
    
    # Asenkron döngüyü başlat
    bytecode_manager.start_async_loop()

    log.info("[PDS-X] Bytecode sistemi başlatıldı")
    return True

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
        self.core = LibXCore(self)
        self.jit_manager = LibXJIT(self)
        self.data_manager = LibXData(self)
        self.logic_manager = LibXLogic(self)
        self.event_manager = EventManager()
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
        self.timer_manager = TimerManager(self.event_manager)
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
        self.core25_features = Core25Features()
        
        # Core2duo özelliklerini bytecode_manager'a bağla
        self.bytecode_manager.register_core_features(self.core25_features)
        
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
        self.bytecode_opcodes = {
            "SIMD": self.core25_features.simd_ops,
            "NEURAL": self.core25_features.neural_ops,  
            "QUANTUM": self.core25_features.quantum_ops,
            "GENETIC": self.core25_features.genetic_ops,
            "BLOCKCHAIN": self.core25_features.blockchain_ops
        }
        
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
        
        # Fonksiyon Tablosu (LibXCore ve diğer modüllerden)
        self.function_table = {
            "MID$": lambda s, start, length: s[start-1:start-1+length],
            "LEN": len, "RND": random.random, "ABS": abs, "INT": int,
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
            # LibXCore Fonksiyonları
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
            # "RUN_ASYNC": self.core.execute_async,  # Bu satır kaldırıldı, çünkü LibXCore'da execute_async yok.
            # "WAIT": self.core.wait,  # Bu satır kaldırıldı, çünkü LibXCore'da wait yok.
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
            '+=': lambda x, y: x + y,
            '-=': lambda x, y: x - y,
            '*=': lambda x, y: x * y,
            '/=': lambda x, y: x / y,
            '%=': lambda x, y: x % y,
            '&=': lambda x, y: x & y,
            '|=': lambda x, y: x | y,
            '^=': lambda x, y: x ^ y,
            '<<=': lambda x, y: x << y,
            '>>=': lambda x, y: x >> y
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
            log.info(f"Yapılandırma yüklendi: {config_file}")
        except Exception as e:
            await self.exception_manager.handle_error(f"Config yükleme hatası: {e}")
            self.config = {}

    async def load_program(self, file_path):
        try:
            import aiofiles
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                code = await f.read()
            self.parse_program(code)
            log.info(f"Program yüklendi: {file_path}")
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

CORE_MODULES = [
    "pdsx_exception",
    "pdsx_exception2",
    "module_manager",
    "core",
    "memory_manager",
    "save_load_system2",
    "event",
    "libxcore",
    "auto_importer",
    "autoinstaller",
    "base_module_manager",
    "oop_and_class2",
    "clazz",
    "save",
    "sqlite",
    "reply_extension",
    "lowlevel",
    "data_structures",
    "f11_backtrace_logger",
    "f12_timer_manager",
    "multithreading_process",
    "pipe_monitor_gui",
    "export_report_doc"
]

LOADABLE_MODULES = [
    "core2-5",
    "database_sql_isam",
    "bytecode_engine",
    "functional",
    "graph",
    "libx_ml", 
    "libx_nlp",
    "libx_network",
    "pipe",
    "tree",
    "tree2",
    "tree3",
    "tpl",
    "tpl3",
    "oop_and_class",
    "eventx",
    "functional2",
    "graph2",
    "lib_db",
    "bytecode_compiler",
    "bytecode_manager"
]

def validate_modules():
    from module_validator import validate_all_modules
    core_results = validate_all_modules(CORE_MODULES)
    if not all(core_results.values()):
        invalid_modules = [m for m, v in core_results.items() if not v]
        raise PdsXException(f"Temel modüllerde hata: {invalid_modules}")
    loadable_results = validate_all_modules(LOADABLE_MODULES)
    for module, is_valid in loadable_results.items():
        if not is_valid:
            log.warning(f"Yüklenebilir modül hazır değil: {module}")
    return loadable_results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDS-X BASIC v15 Yorumlayıcısı")
    parser.add_argument("--version", action="version", version="PDS-X BASIC v15.0")
    parser.add_argument("--file", type=str, help=".basX dosyasını çalıştırır")
    parser.add_argument("--debug", action="store_true", help="Hata ayıklama modu")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli kabuk")
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
    required_libs = ["numpy", "pandas", "scipy", "aiohttp", "aiofiles", "pyyaml", "pdfplumber"]
    for lib in required_libs:
        install_library(lib)

    interpreter = PDSXInterpreter()

    if args.config:
        await interpreter.load_config(args.config)

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.DEBUG))

    if args.file:
        await interpreter.load_program(args.file)
        await interpreter.run()
    elif args.interactive:
        await interpreter.interactive_shell()
    elif args.test:
        import pytest
        sys.exit(pytest.main([args.test]))
    elif args.bytecode:
        from bytecode_engine import BytecodeEngine
        engine = BytecodeEngine(interpreter)
        await engine.execute_bytecode(args.bytecode)

if __name__ == "__main__":
    asyncio.run(main())

if __name__ == "__main__":
    valid_modules = validate_modules()
    asyncio.run(main())