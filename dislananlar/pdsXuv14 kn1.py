# kontrol noktasi 2 de verilen pdsXeuv14.py dosyasını tamamlayarak, 
# PDS-X BASIC v15 yorumlayıcısının temel işlevselliğini ve modüler 
# yapısını oluşturuyoruz. Bu yorumlayıcı, Python'un RestrictedPython 
# kütüphanesini kullanarak güvenli bir şekilde kod derleme ve yürütme 
# işlemleri gerçekleştiriyor. Ayrıca, çeşitli kütüphanelerle entegrasyon 
# sağlayarak genişletilebilir bir yapı sunuyor.

# pdsXe_uv14.py - PDS-X Enhanced Interpreter
# Version: 1.0.0
# Date: June 14, 2025
import re, os, sys, json, time, random, math, struct, logging, asyncio, threading, psutil, ast, traceback
import numpy as np, pandas as pd, scipy.stats as stats, pdfplumber, sqlite3, tkinter as tk, ctypes, subprocess
import yaml, xml.etree.ElementTree as ET, watchdog.observers, watchdog.events, numba, pynvml, sympy as sp
import paho.mqtt.client, kafka, websocket, grpc, zmq.asyncio, torch, torch_geometric, river.anomaly
import qiskit, tensorflow_federated as tff, networkx, matplotlib.pyplot, seaborn, plotly.express, dash
import dask.distributed, prometheus_client, graphviz, boto3, elasticsearch, mysql.connector, psycopg2
import aiofiles, aiohttp, decimal, nltk, spacy, transformers, textblob, websockets, requests
import tensorflow as tf, cryptography.fernet as fernet, cryptography.hazmat.primitives as crypto_primitives
from RestrictedPython import compile_restricted, safe_globals, utility_builtins
from argparse import ArgumentParser
from collections import defaultdict, namedtuple, deque
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import importlib.metadata

# Çekirdek modüller
from command_executor import CommandExecutor
from libx_core import LibXCore
from libx_logic import LibXLogic
from libx_jit import LibXJIT
from libx_task import TaskScheduler
from libx_gui import LibXGui
from libx_stream import StreamManager
from libx_security import SecurityManager
from libx_web import WebManager
from bus3 import BusManager
from event3 import EventManager
from exception_manager3 import ExceptionManager
from libx_concurrency import LibXConcurrency
from lib_db import LibDB
from f12_timer_manager import TimerManager
from libx_data import LibXData
from f11_backtrace_logger import BacktraceLogger
from graph2 import GraphManager
from export_report_doc import ExportReportDocManager
from bytecode_engine_core2duo2 import BytecodeEngine
from memory_manager import MemoryManager, StructInstance, UnionInstance, Pointer, EnumInstance
from dll_manager import DllManager
from api_manager import ApiManager
from inline_asm import InlineASM
from inline_c import InlineC
from unsafe_memory import UnsafeMemoryManager
from sys_call import SysCallWrapper
from plugin_manager import PluginManager
from repl_utils import ReplUtils
from async_manager import AsyncManager
from data_types import Scalar, Vector, Matrix, Tensor, Float128, Float256, Float512
from libx_network import LibXNetwork
from lowlevel import LowLevelManager
from libx_nlp import LibXNLP
from module_validator import ModuleVersionValidator
from module_manager import ModuleManager
from multithreading_process import MultithreadingProcessManager
from database_sql_isam import DatabaseManager
from core2_5 import CoreManager, QuantumState, HoloData, ChaosField, NeuralTensor, BlockchainLedger, SecureData, Signature, IotMessage
from pdsx_exception import PdsXException

logging.basicConfig(filename='pdsxe_errors.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModulePluginManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.modules = {}
        self.plugins = {}
        self.lock = threading.Lock()

    def load_module(self, file_name, module_name=None):
        """Modül yükler ve __pdsX_exports__ yapısını işler."""
        module = self.interpreter.module_manager.load_module(file_name, module_name)
        exports = getattr(module, '__pdsX_exports__', None)
        if exports:
            self.add_exports(module_name or file_name, exports)
        return module

    def add_exports(self, module_name: str, exports: Dict):
        """Çalışma zamanında __pdsX_exports__ yapısını ekler."""
        with self.lock:
            self.modules[module_name] = exports
            for class_name, cls in exports.get("classes", {}).items():
                self.interpreter.type_table[class_name] = cls
            for func_name, func in exports.get("functions", {}).items():
                self.interpreter.function_table[func_name.upper()] = func
            for var_name, var in exports.get("variables", {}).items():
                self.interpreter.global_vars[var_name] = var
            log.debug(f"Modül exports eklendi: {module_name}")

class PdsXe_uv14:
    def __init__(self):
        self.global_vars = {}
        self.shared_vars = defaultdict(list)
        self.local_scopes = [{}]
        self.types = {}
        self.classes = {}
        self.functions = {}
        self.subs = {}
        self.labels = {}
        self.program = []
        self.program_counter = 0
        self.call_stack = []
        self.running = False
        self.db_connections = {}
        self.file_handles = {}
        self.error_handler = None
        self.gosub_handler = None
        self.debug_mode = False
        self.trace_mode = False
        self.loop_stack = []
        self.select_stack = []
        self.if_stack = []
        self.data_list = []
        self.data_pointer = 0
        self.transaction_active = {}
        self.modules = {"core": {"functions": {}, "classes": {}, "program": [], "variables": {}}}
        self.current_module = "main"
        self.repl_mode = False
        self.language = "en"
        self.translations = self.load_translations("lang.json")
        self.memory_pool = {}
        self.next_address = 1000
        self.expr_cache = {}
        self.variable_cache = {}
        self.bytecode = []
        self.async_tasks = []
        self.performance_metrics = {"start_time": time.time(), "memory_usage": 0}
        self.supported_encodings = [
            "utf-8", "cp1254", "iso-8859-9", "ascii", "utf-16", "utf-32",
            "cp1252", "iso-8859-1", "windows-1250", "latin-9",
            "cp932", "gb2312", "gbk", "euc-kr", "cp1251", "iso-8859-5",
            "cp1256", "iso-8859-6", "cp874", "iso-8859-7", "cp1257", "iso-8859-8"
        ]
        self.fact_list = []
        self.rule_list = []
        self.query_list = []
        self.asm_blocks = []
        self.event_handlers = {}
        self.opcodes = ["SIMD", "NEURAL", "QUANTUM"]
        self.object_counter = defaultdict(int)
        self.object_registry = {}
        self.restricted_scopes = set()
        self.restricted_vars = set()
        self.command_executor = CommandExecutor(self)
        self.core = LibXCore(self)
        self.logic = LibXLogic(self)
        self.jit = LibXJIT(self)
        self.task_scheduler = TaskScheduler(self)
        self.gui = LibXGui(self)
        self.stream_manager = StreamManager()
        self.security_manager = SecurityManager()
        self.web_manager = WebManager()
        self.bus_manager = BusManager(self)
        self.event_manager = EventManager(self)
        self.exception_manager = ExceptionManager(self)
        self.concurrency = LibXConcurrency(self)
        self.db = LibDB(self)
        self.timer = TimerManager(self)
        self.data = LibXData(self)
        self.backtrace = BacktraceLogger(self)
        self.graph = GraphManager(self)
        self.export_report = ExportReportDocManager(self)
        self.bytecode_engine = BytecodeEngine(self)
        self.module_plugin_manager = ModulePluginManager(self)
        self.memory_manager = MemoryManager(self)
        self.dll_manager = DllManager()
        self.api_manager = ApiManager()
        self.inline_asm = InlineASM()
        self.inline_c = InlineC()
        self.unsafe_memory = UnsafeMemoryManager()
        self.syscall_wrapper = SysCallWrapper()
        self.repl_utils = ReplUtils()
        self.async_manager = AsyncManager()
        self.network = LibXNetwork(self)
        self.lowlevel = LowLevelManager(self)
        self.nlp = LibXNLP(self)
        self.module_validator = ModuleVersionValidator()
        self.module_manager = ModuleManager(base_path=".")
        self.multithreading = MultithreadingProcessManager(self)
        self.database = DatabaseManager(self)
        self.core_manager = CoreManager(self)
        self.ml = None  # Yer tutucu, libx_ml modülü yüklendiğinde atanacak
        self.libx_modules = {
            "core": self.core,
            "logic": self.logic,
            "jit": self.jit,
            "network": self.network,
            "lowlevel": self.lowlevel,
            "nlp": self.nlp,
            "gui": self.gui,
            "concurrency": self.concurrency,
            "data": self.data,
            "db": self.db,
            "graph": self.graph,
            "timer": self.timer,
            "backtrace": self.backtrace,
            "export_report": self.export_report
        }
        self.bytecode_opcodes = {
            "SIMD": {"ADD": self.bytecode_engine.op_simd_add},
            "NEURAL": {},
            "QUANTUM": {},
            "GENETIC": {},
            "BLOCKCHAIN": {}
        }
        self.type_table = {
            "STRING": str, "INTEGER": int, "LONG": int, "SINGLE": float, "DOUBLE": float,
            "BYTE": int, "SHORT": int, "UNSIGNED INTEGER": int, "CHAR": str,
            "LIST": list, "DICT": dict, "SET": set, "TUPLE": tuple,
            "ARRAY": np.ndarray, "DATAFRAME": pd.DataFrame, "POINTER": Pointer,
            "STRUCT": StructInstance, "UNION": UnionInstance, "ENUM": EnumInstance,
            "VOID": type(None), "BITFIELD": int, "FLOAT128": Float128, "FLOAT256": Float256,
            "FLOAT512": Float512, "STRING8": str, "STRING16": str, "BOOLEAN": bool,
            "NULL": type(None), "NAN": float, "YAPI": dict, "BUS_DATA": dict,
            "SCALAR": Scalar, "VECTOR": Vector, "MATRIX": Matrix, "TENSOR": Tensor,
            "QUANTUM_STATE": QuantumState, "HOLO_DATA": HoloData, "CHAOS_FIELD": ChaosField,
            "NEURAL_TENSOR": NeuralTensor, "BLOCKCHAIN_LEDGER": BlockchainLedger,
            "SECURE_DATA": SecureData, "SIGNATURE": Signature, "IOT_MESSAGE": IotMessage
        }
        self.function_table = {
            "MID$": lambda s, start, length: s[start-1:start-1+length],
            "LEN": len,
            "RND": random.random,
            "ABS": abs,
            "INT": int,
            "LEFT$": lambda s, n: s[:n],
            "RIGHT$": lambda s, n: s[-n:],
            "LTRIM$": lambda s: s.lstrip(),
            "RTRIM$": lambda s: s.rstrip(),
            "STRING$": lambda n, c: c * n,
            "SPACE$": lambda n: " " * n,
            "INSTR": lambda start, s, sub: s.find(sub, start-1) + 1,
            "UCASE$": lambda s: s.upper(),
            "LCASE$": lambda s: s.lower(),
            "STR$": lambda n: str(n),
            "SQR": np.sqrt,
            "SIN": np.sin,
            "COS": np.cos,
            "TAN": np.tan,
            "LOG": np.log,
            "EXP": np.exp,
            "ATN": np.arctan,
            "FIX": lambda x: int(x),
            "ROUND": lambda x, n=0: round(x, n),
            "SGN": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
            "MOD": lambda x, y: x % y,
            "MIN": min,
            "MAX": max,
            "TIMER": time.time,
            "DATE$": lambda: time.strftime("%m-%d-%Y"),
            "TIME$": lambda: time.strftime("%H:%M:%S"),
            "INKEY$": lambda: input()[:1],
            "ENVIRON$": lambda var: os.environ.get(var, ""),
            "COMMAND$": lambda: " ".join(sys.argv[1:]),
            "CSRLIN": lambda: 1,
            "POS": lambda x: 1,
            "VAL": lambda s: float(s) if s.replace(".", "").isdigit() else 0,
            "ASC": lambda c: ord(c[0]),
            "MEAN": self.data.mean,
            "MEDIAN": self.data.median,
            "MODE": self.data.mode,
            "STD": self.data.std,
            "VAR": self.data.var,
            "SUM": self.data.sum,
            "PROD": self.data.prod,
            "PERCENTILE": self.data.percentile,
            "QUANTILE": self.data.quantile,
            "CORR": self.data.corr,
            "COV": self.data.cov,
            "DESCRIBE": self.data.describe,
            "GROUPBY": self.data.groupby,
            "TTEST": self.data.ttest,
            "CHISQUARE": self.data.chisquare,
            "ANOVA": self.data.anova,
            "REGRESS": self.data.regress,
            "PDF_READ_TEXT": self.core_manager.pdf_read_text,
            "PDF_EXTRACT_TABLES": self.core_manager.pdf_extract_tables,
            "WEB_GET": self.network.web_get,
            "SYSTEM": self.core_manager.system,
            "QUANTUM_CORR": self.core_manager.quantum_correlation_analysis,
            "CHAOS_DETECT": self.core_manager.chaos_pattern_detection,
            "NEURAL_PROCESS": self.core_manager.neural_data_processing,
            "GENETIC_OPT": self.core_manager.genetic_optimization_engine,
            "BLOCKCHAIN_CHECK": self.core_manager.blockchain_integrity_check,
            "HASH_DATA": self.core_manager.hash_data,
            "CHECK_AUTH": self.core_manager.check_auth,
            "PROLOG_QUERY": self.logic.prolog_query,
            "PROLOG_ASSERT": self.logic.prolog_assert,
            "COMPILE_ASM": self.jit.compile_asm,
            "COMPILE_C": self.jit.compile_c,
            "COMPILE_JIT": self.jit.compile_jit,
            "WEB_POST": self.network.web_post,
            "LOAD_API": self.network.load_api,
            "OAUTH_REQUEST": self.network.oauth_request,
            "REPLY": self.network.reply,
            "BITSET": self.lowlevel.bitset,
            "BITGET": self.lowlevel.bitget,
            "MEMCPY": self.lowlevel.memcpy,
            "MEMSET": self.lowlevel.memset,
            "PTR": self.lowlevel.ptr,
            "DEREF": self.lowlevel.deref,
            "READ_MEMORY": self.lowlevel.read_memory,
            "WRITE_MEMORY": self.lowlevel.write_memory,
            "GET_MEMORY_INFO": self.lowlevel.get_memory_info,
            "ANALYZE_TEXT": self.nlp.analyze_text,
            "TOKENIZE": self.nlp.tokenize,
            "SENTIMENT_ANALYSIS": self.nlp.sentiment_analysis,
            "SUMMARIZE_TEXT": self.nlp.summarize_text,
            "NAMED_ENTITY_RECOGNITION": self.nlp.named_entity_recognition,
            "POS_TAGGING": self.nlp.pos_tagging,
            "DEPENDENCY_PARSING": self.nlp.dependency_parsing,
            "TEXT_CLASSIFICATION": self.nlp.text_classification,
            "GENERATE_TEXT": self.nlp.generate_text,
            "VALIDATE_EXPORTS": self.module_validator.validate_exports,
            "CHECK_SYNTAX": self.module_validator.check_syntax,
            "CHECK_DEPENDENCIES": self.module_validator.check_dependencies,
            "VALIDATE_ALL_MODULES": self.module_validator.validate_all_modules,
            "ALLOCATE": self.memory_manager.allocate,
            "RELEASE": self.memory_manager.release,
            "SIZEOF": self.memory_manager.sizeof,
            "LOAD_MODULE": self.module_manager.load_module,
            "UNLOAD_MODULE": self.module_manager.unload_module,
            "LIST_MODULES": self.module_manager.list_modules,
            "SAVE_MODULE": self.module_manager.save_module,
            "EDIT_MODULE": self.module_manager.edit_module,
            "CREATE_THREAD": self.multithreading.create_thread,
            "CREATE_PROCESS": self.multithreading.create_process,
            "PARALLEL_MAP": self.multithreading.parallel_map,
            "EXECUTE_ASYNC": self.multithreading.execute_async,
            "DB_CONNECT": self.database.connect,
            "CREATE_ISAM_TABLE": self.database.create_isam_table,
            "EXECUTE_ASYNC_QUERY": self.database.execute_async_query,
            "BUS_DEFINE": self.bus_manager.define,
            "BUS_PUBLISH": self.bus_manager.publish,
            "BUS_SUBSCRIBE": self.bus_manager.subscribe,
            "BUS_GET_STATUS": self.bus_manager.get_status
        }
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
        self.install_missing_libraries()

    def current_scope(self):
        """Geçerli kapsamı döndürür."""
        return self.local_scopes[-1] if self.local_scopes else self.global_vars

    def evaluate_expression(self, expr: str, scope_name: Optional[str] = None) -> Any:
        """İfadeyi değerlendirir."""
        if not expr:
            return None
        try:
            # Kapsam belirleme
            scope = self.modules[scope_name]["variables"] if scope_name and scope_name in self.modules else self.current_scope()
            # Güvenli değerlendirme için RestrictedPython kullanımı
            code = compile_restricted(expr, '<inline>', 'eval')
            globals_dict = safe_globals.copy()
            globals_dict.update(self.function_table)
            globals_dict.update(scope)
            result = eval(code, globals_dict)
            self.object_counter["EVAL"] += 1
            self.object_registry[id(result)] = {
                "type": "EVAL",
                "name": expr[:50],
                "atom": str(result)[:100]
            }
            return result
        except Exception as e:
            raise PdsXException(f"İfade değerlendirme hatası: {str(e)}", context={"expr": expr, "scope": scope_name})

    def install_missing_libraries(self):
        """Eksik kütüphaneleri yükler."""
        required = {
            'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'psutil': 'psutil',
            'pdfplumber': 'pdfplumber', 'beautifulsoup4': 'beautifulsoup4', 'requests': 'requests',
            'packaging': 'packaging', 'pyyaml': 'pyyaml', 'watchdog': 'watchdog',
            'pynvml': 'pynvml', 'numba': 'numba', 'sympy': 'sympy', 'tkinter': 'tkinter',
            'pyqt5': 'PyQt5', 'wxpython': 'wxPython', 'graphql-core': 'graphql-core',
            'paramiko': 'paramiko', 'paho-mqtt': 'paho-mqtt', 'kafka-python': 'kafka-python',
            'websocket-client': 'websocket-client', 'grpcio': 'grpcio', 'pyzmq': 'pyzmq',
            'torch': 'torch', 'torch-geometric': 'torch-geometric', 'river': 'river',
            'qiskit': 'qiskit', 'tensorflow-federated': 'tensorflow-federated',
            'networkx': 'networkx', 'matplotlib': 'matplotlib', 'seaborn': 'seaborn',
            'plotly': 'plotly', 'dash': 'dash', 'dask': 'dask', 'prometheus-client': 'prometheus-client',
            'graphviz': 'graphviz', 'boto3': 'boto3', 'elasticsearch': 'elasticsearch',
            'mysql-connector-python': 'mysql-connector-python', 'psycopg2': 'psycopg2',
            'scikit-learn': 'scikit-learn', 'restrictedpython': 'RestrictedPython',
            'cython': 'cython', 'aiofiles': 'aiofiles', 'aiohttp': 'aiohttp',
            'nltk': 'nltk', 'spacy': 'spacy', 'transformers': 'transformers',
            'textblob': 'textblob', 'websockets': 'websockets', 'tensorflow': 'tensorflow',
            'cryptography': 'cryptography', 'pycryptodome': 'pycryptodome'
        }
        installed = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}
        missing = [lib for lib, pkg in required.items() if lib not in installed]
        if missing:
            print(f"Eksik kütüphaneler yükleniyor: {missing}")
            for lib in missing:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', required[lib]])
                except subprocess.CalledProcessError:
                    print(f"Kütüphane yüklenemedi: {lib}")

    def load_translations(self, file_path: str) -> Dict:
        """Dil çevirilerini yükler."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"en": {"PRINT": "Print", "ERROR": "Error"}, "tr": {"PRINT": "Yazdır", "ERROR": "Hata"}}

    def parse_program(self, code: str, module_name: str = "main", lightweight: bool = False, as_library: bool = False) -> None:
        """Programı ayrıştırır ve yürütmeye hazırlar."""
        lines = code.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("'") and not line.startswith("REM"):
                if ":" in line and not line.startswith("DATA"):
                    label_match = re.match(r"(\w+):", line)
                    if label_match:
                        label = label_match.group(1)
                        self.labels[label] = i
                        line = line[len(label) + 1:].strip()
                if line:
                    self.program.append((line, i))
        if not as_library:
            self.modules[module_name] = {
                "functions": self.functions.copy(),
                "classes": self.classes.copy(),
                "program": self.program,
                "variables": {}
            }
        log.debug(f"Program ayrıştırıldı: {module_name}, Satır sayısı: {len(self.program)}")

    async def run_async(self, code: str) -> None:
        """Kodu asenkron çalıştırır."""
        self.parse_program(code)
        self.running = True
        while self.running and self.program_counter < len(self.program):
            command, _ = self.program[self.program_counter]
            try:
                await self.command_executor.execute_async(command)
                self.program_counter += 1
            except PdsXException as e:
                await self.exception_manager.handle_error(e)
                break
        self.running = False

    def execute_command(self, command: str, scope_name: Optional[str] = None) -> Optional[int]:
        """Komutu çalıştırır."""
        return self.command_executor.execute(command, scope_name)

    def repl(self) -> None:
        """Etkileşimli kabuğu başlatır."""
        print(f"PDS-X BASIC v15.0 - Etkileşimli Kabuk (Dil: {self.language})")
        self.repl_mode = True
        while self.repl_mode:
            try:
                command = input("PDS-X> ")
                if command.lower() in ["exit", "quit"]:
                    break
                result = self.execute_command(command)
                if result is not None:
                    print(result)
            except PdsXException as e:
                print(f"Hata: {str(e)}")
            except KeyboardInterrupt:
                print("\nÇıkış için 'exit' veya 'quit' yazın.")
        self.repl_mode = False

    def system_monitor(self) -> Dict:
        """Sistem kaynaklarını izler."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_threads": threading.active_count(),
            "async_tasks": len(self.async_tasks)
        }

def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = ArgumentParser(description="PDS-X BASIC v15 Yorumlayıcısı")
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

def main():
    """Ana giriş noktası."""
    args = parse_args()
    interpreter = PdsXe_uv14()
    if args.debug:
        interpreter.debug_mode = True
    if args.trace:
        interpreter.trace_mode = True
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
        if args.bytecode:
            interpreter.bytecode = interpreter.bytecode_engine.compile_to_bytecode(code)
            interpreter.bytecode_engine.execute_bytecode()
        else:
            asyncio.run(interpreter.run_async(code))
    if args.interactive:
        interpreter.repl()

if __name__ == "__main__":
    main()