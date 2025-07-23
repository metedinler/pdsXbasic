
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
from bus3 import BusManager
from core2_5 import CoreManager, QuantumState, HoloData, ChaosField, NeuralTensor, BlockchainLedger, SecureData, Signature, IotMessage
from core2_6 import Core2_6Manager
from functional import Functional
from save_load_system2 import SaveLoadSystem
from tree import TreeManager
from event3 import EventManager
from exception_manager3 import ExceptionManager
from libx_concurrency import LibXConcurrency
from lib_db import LibDB
from f12_timer_manager import TimerManager
from libx_data import LibXData
from f11_backtrace_logger import BacktraceLogger
from graph2 import GraphManager
from export_report_doc import ExportReportDocManager
from libx_logic import LibXLogic
from libx_jit import LibXJIT
from libx_gui import LibXGui
from bytecode_engine_core2duo2 import BytecodeEngine
from libx_ml import LibXML
from libx_network import LibXNetwork
from lowlevel import LowLevelManager
from libx_nlp import LibXNLP
from module_validator import ModuleVersionValidator
from memory_manager import MemoryManager, StructInstance, UnionInstance, Pointer, EnumInstance
from module_manager import ModuleManager
from multithreading_process import MultithreadingProcessManager
from pdsx_exception import PdsXException
from pdsx_exception2 import PdsXException2
from database_sql_isam import DatabaseManager
from pipe import PipeManager
from autoinstaller import AutoInstaller
from libx_stream import StreamManager
from libx_security import SecurityManager
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
from command_executor import CommandExecutor

logging.basicConfig(filename='pdsxe_errors.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("pdsXe_uv14")

class ModulePluginManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.modules = {}
        self.plugins = {}
        self.lock = threading.Lock()

    def load_module(self, file_name: str, module_name: Optional[str] = None) -> Any:
        """Modül yükler ve __pdsX_exports__ yapısını işler."""
        module = self.interpreter.module_manager.load_module(file_name, module_name)
        exports = getattr(module, '__pdsX_exports__', None)
        if exports:
            self.add_exports(module_name or file_name, exports)
        return module

    def add_exports(self, module_name: str, exports: Dict) -> None:
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
        self.global_vars: Dict[str, Any] = {}
        self.shared_vars: Dict[str, List[Any]] = defaultdict(list)
        self.local_scopes: List[Dict[str, Any]] = [{}]
        self.types: Dict[str, Dict[str, str]] = {}
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.subs: Dict[str, Dict[str, Any]] = {}
        self.labels: Dict[str, int] = {}
        self.program: List[tuple[str, int]] = []
        self.program_counter: int = 0
        self.call_stack: List[int] = []
        self.running: bool = False
        self.db_connections: Dict[str, Any] = {}
        self.file_handles: Dict[str, Any] = {}
        self.error_handler: Optional[int] = None
        self.gosub_handler: Optional[int] = None
        self.debug_mode: bool = False
        self.trace_mode: bool = False
        self.loop_stack: List[Dict[str, Any]] = []
        self.select_stack: List[Dict[str, Any]] = []
        self.if_stack: List[bool] = []
        self.data_list: List[str] = []
        self.data_pointer: int = 0
        self.transaction_active: Dict[str, bool] = {}
        self.modules: Dict[str, Dict[str, Any]] = {"core": {"functions": {}, "classes": {}, "program": [], "variables": {}}}
        self.current_module: str = "main"
        self.repl_mode: bool = False
        self.language: str = "en"
        self.translations: Dict[str, Dict[str, str]] = self.load_translations("lang.json")
        self.memory_pool: Dict[int, Any] = {}
        self.next_address: int = 1000
        self.expr_cache: Dict[str, Any] = {}
        self.variable_cache: Dict[str, Any] = {}
        self.bytecode: List[Any] = []
        self.async_tasks: List[Any] = []
        self.performance_metrics: Dict[str, float] = {"start_time": time.time(), "memory_usage": 0}
        self.supported_encodings: List[str] = [
            "utf-8", "cp1254", "iso-8859-9", "ascii", "utf-16", "utf-32",
            "cp1252", "iso-8859-1", "windows-1250", "latin-9",
            "cp932", "gb2312", "gbk", "euc-kr", "cp1251", "iso-8859-5",
            "cp1256", "iso-8859-6", "cp874", "iso-8859-7", "cp1257", "iso-8859-8"
        ]
        self.fact_list: List[str] = []
        self.rule_list: List[str] = []
        self.query_list: List[str] = []
        self.asm_blocks: List[str] = []
        self.event_handlers: Dict[str, str] = {}
        self.opcodes: List[str] = ["SIMD", "NEURAL", "QUANTUM"]
        self.object_counter: Dict[str, int] = defaultdict(int)
        self.object_registry: Dict[int, Dict[str, Any]] = {}
        self.restricted_scopes: set = set()
        self.restricted_vars: set = set()
        self.command_executor = CommandExecutor(self)
        self.core_manager = CoreManager(self)
        self.core2_6_manager = Core2_6Manager(self)
        self.functional = Functional(self)
        self.save_load_system = SaveLoadSystem(self)
        self.tree_manager = TreeManager(self)
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
        self.logic = LibXLogic(self)
        self.jit = LibXJIT(self)
        self.gui = LibXGui(self)
        self.bytecode_engine = BytecodeEngine(self)
        self.ml = LibXML(self)
        self.network = LibXNetwork(self)
        self.lowlevel = LowLevelManager(self)
        self.nlp = LibXNLP(self)
        self.module_validator = ModuleVersionValidator()
        self.memory_manager = MemoryManager(self)
        self.module_manager = ModuleManager(base_path=".")
        self.multithreading = MultithreadingProcessManager(self)
        self.database = DatabaseManager(self)
        self.pipe = PipeManager(self)
        self.autoinstaller = AutoInstaller(self)
        self.stream_manager = StreamManager()
        self.security_manager = SecurityManager()
        self.dll_manager = DllManager()
        self.api_manager = ApiManager()
        self.inline_asm = InlineASM()
        self.inline_c = InlineC()
        self.unsafe_memory = UnsafeMemoryManager()
        self.syscall_wrapper = SysCallWrapper()
        self.plugin_manager = PluginManager(self)
        self.repl_utils = ReplUtils()
        self.async_manager = AsyncManager()
        self.module_plugin_manager = ModulePluginManager(self)
        self.libx_modules: Dict[str, Any] = {
            "core": self.core_manager,
            "core2_6": self.core2_6_manager,
            "logic": self.logic,
            "jit": self.jit,
            "network": self.network,
            "ml": self.ml,
            "lowlevel": self.lowlevel,
            "nlp": self.nlp,
            "gui": self.gui,
            "concurrency": self.concurrency,
            "data": self.data,
            "db": self.db,
            "graph": self.graph,
            "timer": self.timer,
            "backtrace": self.backtrace,
            "export_report": self.export_report,
            "functional": self.functional,
            "save_load": self.save_load_system,
            "tree": self.tree_manager
        }
        self.bytecode_opcodes: Dict[str, Dict[str, Callable]] = {
            "SIMD": {"ADD": self.bytecode_engine.op_simd_add},
            "NEURAL": {},
            "QUANTUM": {},
            "GENETIC": {},
            "BLOCKCHAIN": {}
        }
        self.type_table: Dict[str, type] = {
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
        self.function_table: Dict[str, Callable] = {
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
            "COMMAND$": lambda: " ".






    def load_static_modules(self) -> None:
        """Statik modülleri auto_importer ile yükler."""
        try:
            static_modules = [
                "core2-5", "core2-6", "bus3", "functional", "save_load_system2", "tree",
                "event3", "exception_manager3", "libx_concurrency", "lib_db",
                "f12_timer_manager", "f11_backtrace_logger", "graph2", "export_report_doc",
                "libx_logic", "libx_jit", "libx_gui", "bytecode_engine_core2duo2",
                "libx_ml", "libx_network", "lowlevel", "libx_nlp", "module_validator",
                "memory_manager", "module_manager", "multithreading_process",
                "database_sql_isam", "pipe3", "libx_stream", "libx_security",
                "dll_manager", "api_manager", "inline_asm", "inline_c", "unsafe_memory",
                "sys_call", "plugin_manager", "repl_utils", "async_manager",
                "reply_extension"
            ]
            for module_name in static_modules:
                module_path = f"{module_name}.py"
                if os.path.exists(module_path):
                    self.module_plugin_manager.load_module(module_path, module_name)
                    add_exports_to_file(module_path)
                else:
                    log.warning(f"Statik modül bulunamadı: {module_name}")
            log.info("Statik modüller yüklendi")
        except Exception as e:
            log.error(f"Statik modül yükleme hatası: {str(e)}")
            raise PdsXException(f"Statik modül yükleme hatası: {str(e)}")

    def install_missing_libraries(self) -> None:
        """Eksik kütüphaneleri yükler."""
        try:
            with open("dependencies.json", "r", encoding="utf-8") as f:
                deps = json.load(f)
            required = {pkg["name"]: pkg["version"] for pkg in deps["installed_packages"]}
            installed = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}
            missing = [lib for lib in required if lib not in installed]
            if missing:
                log.info(f"Eksik kütüphaneler yükleniyor: {missing}")
                python_exe = str(Path(self.autoinstaller.venv_path) / ("Scripts" if sys.platform == "win32" else "bin") / "python")
                for lib in missing:
                    subprocess.check_call([python_exe, '-m', 'pip', 'install', required[lib]])
                    log.debug(f"Paket yüklendi: {required[lib]}")
        except FileNotFoundError:
            log.error("dependencies.json bulunamadı, autoinstaller ile oluşturuluyor")
            self.autoinstaller.initialize_dependencies()
        except subprocess.CalledProcessError as e:
            log.error(f"Kütüphane yüklenemedi: {lib}, Hata: {str(e)}")

    def alias(self, command: str, alias_name: str, scope_name: Optional[str] = None):
        """Komut, fonksiyon veya veri yapısına alias atar."""
        command_upper = command.upper()
        scope = self.modules[scope_name]["variables"] if scope_name else self.current_scope()
        with self.lock:
            if command_upper in self.command_executor.command_handlers:
                self.command_executor.command_handlers[alias_name.upper()] = self.command_executor.command_handlers[command_upper]
                self.object_counter["ALIAS"] += 1
                self.object_registry[id(alias_name)] = {
                    "type": "ALIAS",
                    "name": alias_name,
                    "atom": command
                }
                log.debug(f"Alias atandı: {command} -> {alias_name}")
            elif command_upper in self.function_table:
                self.function_table[alias_name.upper()] = self.function_table[command_upper]
                self.object_counter["ALIAS"] += 1
                self.object_registry[id(alias_name)] = {
                    "type": "ALIAS",
                    "name": alias_name,
                    "atom": command
                }
                log.debug(f"Fonksiyon alias atandı: {command} -> {alias_name}")
            elif command_upper in self.type_table:
                self.type_table[alias_name.upper()] = self.type_table[command_upper]
                self.object_counter["ALIAS"] += 1
                self.object_registry[id(alias_name)] = {
                    "type": "ALIAS",
                    "name": alias_name,
                    "atom": command
                }
                log.debug(f"Veri yapısı alias atandı: {command} -> {alias_name}")
            else:
                raise PdsXException(f"Komut, fonksiyon veya veri yapısı bulunamadı: {command}", code="ALIAS001")

    def current_scope(self) -> Dict[str, Any]:
        """Geçerli kapsamı döndürür."""
        return self.local_scopes[-1] if self.local_scopes else self.global_vars

    def evaluate_expression(self, expr: str, scope_name: Optional[str] = None) -> Any:
        """İfadeyi değerlendirir."""
        if not expr:
            return None
        try:
            scope = self.modules[scope_name]["variables"] if scope_name and scope_name in self.modules else self.current_scope()
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

    def load_translations(self, file_path: str) -> Dict[str, Dict[str, str]]:
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

    def system_monitor(self) -> Dict[str, float]:
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
    interpreter = PdsXe_uv14_2()
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