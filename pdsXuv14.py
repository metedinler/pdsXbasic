# pdsXu.py - PDS-X BASIC v14u Çok-Paradigmalı Yorumlayıcı
# Version: 14u
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import sys
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
from collections import deque, defaultdict
from pathlib import Path

# Modül İçe Aktarmaları (aynı dizinde)
from libxcore import LibXCore
from libx_jit import LibXJIT
from libx_data import LibXData
from libx_logic import LibXLogic
from libx_gui import LibXGui
from libx_concurrency import LibXConcurrency
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
from f13_repl_extensions import REPLExtensions
from f14_data_structures import DataStructures
from f15_oop_and_class import OOPManager
from f16_save_load_system import SaveLoadSystem
from f17_multithreading_process import MultithreadingManager
from f18_database_sql_isam import DatabaseISAM
from f19_pipe_monitor_gui import PipeMonitorGUI
from f20_export_report_doc import ReportExporter

__version__ = "14u"

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdsXu")

# Yardımcı Desenler
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
STR_RE = re.compile(r'^".*?"$|^\'.*?\'$')

def _to_num(s: str):
    return float(s) if '.' in s else int(s)

class PdsXException(Exception):
    pass

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
        self.gui_manager = LibXGui(self)
        self.concurrency_manager = LibXConcurrency(self)
        self.nlp_manager = LibXNLP(self)
        self.network_manager = LibXNetwork(self)
        self.db_manager = LibDB(self)
        self.sqlite_manager = SQLiteManager(self)
        self.bytecode_compiler = BytecodeCompiler()
        self.bytecode_manager = BytecodeManager()
        self.lowlevel_manager = LowLevelManager(self)
        self.memory_manager = MemoryManager(self)
        self.pipe_manager = PipeManager(self)
        self.event_manager = EventManager()
        self.tree_manager = TreeManager()
        self.graph_manager = GraphManager()
        self.functional_manager = FunctionalManager()
        self.save_manager = SaveManager(self)
        self.backtrace_logger = BacktraceLogger()
        self.timer_manager = TimerManager(self.event_manager)
        self.repl_extensions = REPLExtensions()
        self.data_structures = DataStructures()
        self.oop_manager = OOPManager()
        self.save_load_system = SaveLoadSystem()
        self.multithreading_manager = MultithreadingManager()
        self.database_isam = DatabaseISAM()
        self.pipe_monitor_gui = PipeMonitorGUI()
        self.report_exporter = ReportExporter()
        
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
            "FLOAT128": np.float128, "FLOAT256": np.float256, "STRING8": str, "STRING16": str,
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
            "RUN_ASYNC": self.core.run_async,
            "WAIT": self.core.wait,
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

    def parse_program(self, code, module_name="main", lightweight=False, as_library=False):
        self.current_module = module_name
        self.modules[module_name] = {
            "program": [],
            "functions": {},
            "subs": {},
            "classes": {},
            "interfaces": {},
            "types": {},
            "labels": {}
        }
        current_sub = None
        current_function = None
        current_type = None
        current_class = None
        current_interface = None
        type_fields = {}
        class_info = {}
        interface_info = {}
        enum_values = {}
        lines = code.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            line_upper = line.upper()
            if line_upper.startswith("'") or line_upper.startswith("REM "):
                i += 1
                continue
            if ":" in line and not line_upper.startswith("PIPE("):
                parts = [p.strip() for p in line.split(":")]
                for part in parts:
                    self.program.append((part, None))
                i += 1
                continue
            if "/" in line and line_upper.startswith(("FOR ", "IF ")):
                parts = [p.strip() for p in line.split("/")]
                for part in parts:
                    self.program.append((part, None))
                i += 1
                continue
            if line_upper.startswith("COMPILE "):
                lang, rest = re.match(r"COMPILE\s+(\w+)\s+(.+)?", line, re.IGNORECASE).groups()
                j = i + 1
                block = []
                while j < len(lines) and not lines[j].strip().upper().startswith(f"END {lang}"):
                    block.append(lines[j])
                    j += 1
                output_name = re.match(r"AS\s+\"([^\"]+)\"", lines[j].strip(), re.IGNORECASE).group(1)
                self.program.append({"type": "compile", "language": lang, "code": "\n".join(block), "output_name": output_name})
                i = j + 1
                continue
            if line_upper.startswith("SAVE BYTECODE"):
                path, compress = re.match(r"SAVE BYTECODE\s+\"([^\"]+)\"\s*(COMPRESS)?", line, re.IGNORECASE).groups()
                self.program.append({"type": "save_bytecode", "path": path, "compress": bool(compress)})
                i += 1
                continue
            if line_upper.startswith("LOAD BYTECODE"):
                path = re.match(r"LOAD BYTECODE\s+\"([^\"]+)\"", line, re.IGNORECASE).group(1)
                self.program.append({"type": "load_bytecode", "path": path})
                i += 1
                continue
            if line_upper.startswith("FUNC "):
                expr = line[5:].strip()
                self.function_table["_func"] = lambda *args: eval(expr, dict(zip(['x','y','z'], args)))
                i += 1
                continue
            if line_upper.startswith("GAMMA "):
                expr = line[6:].strip()
                self.function_table["_gamma"] = self.core.omega('x', 'y', expr)
                i += 1
                continue
            if line_upper.startswith("FACT "):
                self.program.append({"type": "fact", "fact": line[5:].strip()})
                i += 1
                continue
            if line_upper.startswith("RULE "):
                match = re.match(r"RULE\s+(\w+)\s*:-\s*(.+)", line, re.IGNORECASE)
                if match:
                    head, body = match.groups()
                    self.program.append({"type": "rule", "head": head, "body": body.strip()})
                i += 1
                continue
            if line_upper.startswith("QUERY "):
                self.program.append({"type": "query", "goal": line[6:].strip()})
                i += 1
                continue
            if line_upper.startswith("INTERFACE "):
                match = re.match(r"INTERFACE\s+(\w+)", line, re.IGNORECASE)
                if match:
                    name = match.group(1)
                    current_interface = name
                    interface_info[name] = {'methods': {}}
                    self.modules[module_name]["interfaces"][name] = interface_info[name]
                    i += 1
                    continue
            if line_upper == "END INTERFACE":
                if current_interface:
                    self.interfaces[current_interface] = interface_info[current_interface]
                    current_interface = None
                    i += 1
                    continue
            if line_upper.startswith("ABSTRACT CLASS "):
                match = re.match(r"ABSTRACT CLASS\s+(\w+)(?:\s+EXTENDS\s+(.+))?", line, re.IGNORECASE)
                if match:
                    class_name, parent_names = match.groups()
                    parent_list = [p.strip() for p in parent_names.split(",")] if parent_names else []
                    current_class = class_name
                    class_info[class_name] = {
                        'methods': {},
                        'private_methods': {},
                        'static_vars': {},
                        'parent': parent_names,
                        'abstract': True
                    }
                    i += 1
                    continue
            if line_upper.startswith("CLASS "):
                match = re.match(r"CLASS\s+(\w+)(?:\s+EXTENDS\s+(.+))?", line, re.IGNORECASE)
                if match:
                    class_name, parent_names = match.groups()
                    parent_list = [p.strip() for p in parent_names.split(",")] if parent_names else []
                    current_class = class_name
                    class_info[class_name] = {
                        'methods': {},
                        'private_methods': {},
                        'static_vars': {},
                        'parent': parent_names,
                        'abstract': False
                    }
                    i += 1
                    continue
            if line_upper.startswith("END CLASS"):
                if current_class:
                    self.classes[current_class] = self.oop_manager.build_class(class_info[current_class])
                    self.modules[module_name]["classes"][current_class] = self.classes[current_class]
                    current_class = None
                    i += 1
                    continue
            if current_class:
                if line_upper.startswith("SUB ") or line_upper.startswith("PRIVATE SUB ") or \
                   line_upper.startswith("FUNCTION ") or line_upper.startswith("PRIVATE FUNCTION "):
                    is_private = line_upper.startswith(("PRIVATE SUB ", "PRIVATE FUNCTION "))
                    prefix = "PRIVATE " if is_private else ""
                    method_type = "SUB" if line_upper.startswith((prefix + "SUB ")) else "FUNCTION"
                    match = re.match(rf"{prefix}{method_type}\s+(\w+)(?:\(.*\))?", line, re.IGNORECASE)
                    if match:
                        method_name = match.group(1)
                        method_body = []
                        j = i + 1
                        while j < len(lines) and lines[j].strip().upper() != f"END {method_type}":
                            method_body.append(lines[j].strip())
                            j += 1
                        params = re.search(r"\((.*?)\)", line, re.IGNORECASE)
                        params = params.group(1).split(",") if params else []
                        params = [p.strip() for p in params]
                        method_lambda = lambda self, *args, **kwargs: self.execute_method(method_name, method_body, params, args, scope_name=current_class)
                        if is_private:
                            class_info[current_class]['private_methods'][method_name] = method_lambda
                        else:
                            class_info[current_class]['methods'][method_name] = method_lambda
                        i = j + 1
                        continue
                if line_upper.startswith("STATIC "):
                    match = re.match(r"STATIC\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                    if match:
                        var_name, var_type = match.groups()
                        class_info[current_class]['static_vars'][var_name] = self.type_table.get(var_type, None)()
                        i += 1
                        continue
            if line_upper.startswith("TYPE "):
                type_name = line[5:].strip()
                current_type = type_name
                type_fields[type_name] = []
                i += 1
                while i < len(lines) and not lines[i].strip().upper().startswith("END TYPE"):
                    field_line = lines[i].strip()
                    if field_line:
                        match = re.match(r"FIELD\s+(\w+)\s+AS\s+(\w+)(?:\s*,\s*([\d,]+))?", field_line, re.IGNORECASE)
                        if match:
                            fname, ftype, dimstr = match.groups()
                            dims = [int(x) for x in dimstr.split(',')] if dimstr else []
                            type_fields[type_name].append((fname, ftype, dims))
                        i += 1
                self.program.append({"type": "type", "name": type_name, "fields": type_fields[type_name]})
                i += 1
                continue
            if line_upper.startswith("ENUM "):
                enum_name = line[5:].strip()
                current_type = enum_name
                enum_values[enum_name] = {}
                value_index = 0
                i += 1
                while i < len(lines) and not lines[i].strip().upper().startswith("END ENUM"):
                    value_name = lines[i].strip()
                    if value_name:
                        enum_values[enum_name][value_name] = value_index
                        value_index += 1
                    i += 1
                self.program.append({"type": "enum", "name": enum_name, "values": enum_values[enum_name]})
                i += 1
                continue
            self.program.append((line, None))
            i += 1

    def execute_command(self, command, scope_name=None):
        if isinstance(command, dict):
            cmd_type = command["type"]
            if cmd_type == "compile":
                self.jit_manager.compile(command["language"], command["code"], command["output_name"])
                return None
            if cmd_type == "save_bytecode":
                self.bytecode_manager.save_bytecode(self.bytecode, command["path"], command["compress"])
                return None
            if cmd_type == "load_bytecode":
                self.bytecode = self.bytecode_manager.load_bytecode(command["path"])
                for cmd_type, line, tree in self.bytecode:
                    if cmd_type == "EXEC":
                        exec(pickle.loads(tree), self.current_scope())
                    else:
                        self.execute_command(line)
                return None
            if cmd_type == "fact":
                self.logic_manager.parse_prolog_command(f"FACT {command['fact']}")
                return None
            if cmd_type == "rule":
                self.logic_manager.parse_prolog_command(f"RULE {command['head']} :- {command['body']}")
                return None
            if cmd_type == "query":
                self.logic_manager.parse_prolog_command(f"QUERY {command['goal']}")
                return None
            if cmd_type == "type":
                self.memory_manager.type_manager.define_type(command["name"], command["fields"])
                return None
            if cmd_type == "enum":
                self.memory_manager.enum_manager.define(command["name"], list(command["values"].keys()))
                return None

        command = command.strip()
        if not command:
            return None
        command_upper = command.upper()

        if self.trace_mode:
            self.backtrace_logger.log(f"TRACE: Satır {self.program_counter + 1}: {command}")

        try:
            if command_upper == "CLS":
                os.system('cls' if os.name == 'nt' else 'clear')
                return None
            if command_upper.startswith("BEEP"):
                print("\a")
                return None
            if command_upper.startswith("PRINT"):
                match = re.match(r"PRINT\s*(.+)?", command, re.IGNORECASE)
                if match:
                    expr = match.group(1)
                    if expr:
                        parts = [part.strip() for part in expr.split(";")]
                        output = []
                        for part in parts:
                            if part:
                                result = self.evaluate_expression(part, scope_name)
                                output.append(str(result))
                            else:
                                output.append(" ")
                        print("".join(output), end="")
                    else:
                        print()
                    return None
                raise PdsXException("PRINT komutunda sözdizimi hatası")
            if command_upper.startswith("LET"):
                match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
                if match:
                    var_name, expr = match.groups()
                    value = self.evaluate_expression(expr, scope_name)
                    self.current_scope()[var_name] = value
                    return None
                raise PdsXException("LET komutunda sözdizimi hatası")
            if command_upper.startswith("DIM"):
                match = re.match(r"DIM\s+(\w+)\s*(?:\((.+)\))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name, dims, var_type = match.groups()
                    if dims:
                        dims = [self.evaluate_expression(d.strip(), scope_name) for d in dims.split(",")]
                        if var_type.upper() in self.type_table:
                            self.current_scope()[var_name] = np.zeros(dims, dtype=self.type_table[var_type.upper()])
                        elif var_type.upper() in self.types:
                            type_info = self.types[var_type.upper()]
                            if type_info["kind"] == "STRUCT":
                                self.current_scope()[var_name] = self.memory_manager.create_struct_instance(type_info["fields"])
                            elif type_info["kind"] == "UNION":
                                self.current_scope()[var_name] = self.memory_manager.create_union_instance(type_info["fields"])
                            elif type_info["kind"] == "ENUM":
                                self.current_scope()[var_name] = self.memory_manager.create_enum_instance(type_info["values"])
                        else:
                            raise PdsXException(f"Geçersiz veri tipi: {var_type}")
                    else:
                        if var_type.upper() in self.type_table:
                            self.current_scope()[var_name] = self.type_table[var_type.upper()]()
                        elif var_type.upper() in self.types:
                            type_info = self.types[var_type.upper()]
                            if type_info["kind"] == "STRUCT":
                                self.current_scope()[var_name] = self.memory_manager.create_struct_instance(type_info["fields"])
                            elif type_info["kind"] == "UNION":
                                self.current_scope()[var_name] = self.memory_manager.create_union_instance(type_info["fields"])
                            elif type_info["kind"] == "ENUM":
                                self.current_scope()[var_name] = self.memory_manager.create_enum_instance(type_info["values"])
                        else:
                            raise PdsXException(f"Geçersiz veri tipi: {var_type}")
                    return None
                raise PdsXException("DIM komutunda sözdizimi hatası")
            if command_upper.startswith("IF"):
                match = re.match(r"IF\s+(.+)\s+THEN\s*(.+)?", command, re.IGNORECASE)
                if match:
                    condition, then_clause = match.groups()
                    condition_result = self.evaluate_expression(condition, scope_name)
                    self.if_stack.append({"condition": condition_result, "else": False})
                    if condition_result:
                        if then_clause:
                            return self.execute_command(then_clause, scope_name)
                    else:
                        return self.find_else_or_endif()
                    return None
                raise PdsXException("IF komutunda sözdizimi hatası")
            if command_upper == "ELSE":
                if self.if_stack and not self.if_stack[-1]["else"]:
                    self.if_stack[-1]["else"] = True
                    if self.if_stack[-1]["condition"]:
                        return self.find_endif()
                    return None
                raise PdsXException("ELSE komutu eşleşen IF olmadan kullanıldı")
            if command_upper == "ENDIF":
                if self.if_stack:
                    self.if_stack.pop()
                    return None
                raise PdsXException("ENDIF komutu eşleşen IF olmadan kullanıldı")
            if command_upper.startswith("FOR"):
                match = re.match(r"FOR\s+(\w+)\s*=\s*(.+)\s+TO\s+(.+)(?:\s+STEP\s+(.+))?", command, re.IGNORECASE)
                if match:
                    var_name, start, end, step = match.groups()
                    start_val = self.evaluate_expression(start, scope_name)
                    end_val = self.evaluate_expression(end, scope_name)
                    step_val = self.evaluate_expression(step, scope_name) if step else 1
                    self.current_scope()[var_name] = start_val
                    self.loop_stack.append({
                        "type": "FOR",
                        "var": var_name,
                        "end": end_val,
                        "step": step_val,
                        "start_line": self.program_counter
                    })
                    if step_val > 0 and start_val > end_val or step_val < 0 and start_val < end_val:
                        return self.find_next()
                    return None
                raise PdsXException("FOR komutunda sözdizimi hatası")
            if command_upper == "NEXT":
                if self.loop_stack and self.loop_stack[-1]["type"] == "FOR":
                    loop = self.loop_stack[-1]
                    var_name = loop["var"]
                    current_val = self.current_scope()[var_name]
                    step = loop["step"]
                    end = loop["end"]
                    current_val += step
                    self.current_scope()[var_name] = current_val
                    if step > 0 and current_val <= end or step < 0 and current_val >= end:
                        return loop["start_line"]
                    else:
                        self.loop_stack.pop()
                    return None
                raise PdsXException("NEXT komutu eşleşen FOR olmadan kullanıldı")
            if command_upper.startswith("OPEN DATABASE"):
                match = re.match(r"OPEN DATABASE\s+(.+)", command, re.IGNORECASE)
                if match:
                    db_name = match.group(1).strip()
                    self.db_manager.open_database(db_name)
                    return None
                raise PdsXException("OPEN DATABASE komutunda sözdizimi hatası")
            if command_upper.startswith("SQLITE CONNECT"):
                match = re.match(r"SQLITE CONNECT\s+\"([^\"]+)\"\s+AS\s+#(\d+)", command, re.IGNORECASE)
                if match:
                    db_path, db_number = match.groups()
                    self.sqlite_manager.sqlite_connect(db_path, int(db_number))
                    return None
                raise PdsXException("SQLITE CONNECT komutunda sözdizimi hatası")
            if command_upper.startswith("SQL RESULT TO ARRAY"):
                match = re.match(r"SQL RESULT TO ARRAY\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    self.current_scope()[var_name] = self.db_manager.sql_result_to_array()
                    return None
                raise PdsXException("SQL RESULT TO ARRAY komutunda sözdizimi hatası")
            if command_upper.startswith("TRY"):
                match = re.match(r"TRY\s+(.+)\s+CATCH\s+(.+)(?:\s+FINALLY\s+(.+))?", command, re.IGNORECASE)
                if match:
                    try_block, catch_block, finally_block = match.groups()
                    try:
                        self.execute_command(try_block, scope_name)
                    except Exception as e:
                        self.current_scope()['_error'] = str(e)
                        self.execute_command(catch_block, scope_name)
                    finally:
                        if finally_block:
                            self.execute_command(finally_block, scope_name)
                    return None
                raise PdsXException("TRY CATCH FINALLY komutunda sözdizimi hatası")
            if command_upper.startswith("WINDOW"):
                match = re.match(r"WINDOW\s+(\w+)\s+(.*)", command, re.IGNORECASE)
                if match:
                    name, rest = match.groups()
                    params = self.gui_manager._parse_params(rest)
                    self.program.append({"type": "window", "name": name, "width": params["WIDTH"], "height": params["HEIGHT"], "title": params.get("TITLE", "")})
                    return None
                raise PdsXException("WINDOW komutunda sözdizimi hatası")
            if command_upper.startswith("BITSET"):
                match = re.match(r"BITSET\s+(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)", command, re.IGNORECASE)
                if match:
                    ptr, field, value, bits = map(int, match.groups())
                    self.lowlevel_manager.bitset(ptr, field, value, bits)
                    return None
                raise PdsXException("BITSET komutunda sözdizimi hatası")
            if command_upper.startswith("THREAD"):
                match = re.match(r"THREAD\s+(\w+)\s*,\s*(.+)", command, re.IGNORECASE)
                if match:
                    thread_name, sub_name = match.groups()
                    if sub_name in self.subs:
                        self.concurrency_manager.add_thread(sub_name, [])
                        self.current_scope()[thread_name] = threading.get_ident()
                    else:
                        raise PdsXException(f"Alt program bulunamadı: {sub_name}")
                    return None
                raise PdsXException("THREAD komutunda sözdizimi hatası")
            if command_upper.startswith("PIPE"):
                self.pipe_manager.parse_pipe_command(command)
                return None
            if command_upper.startswith("NLP"):
                self.nlp_manager.execute_nlp_command(command[4:].strip())
                return None
            if command_upper.startswith("REGISTER PROLOG HANDLER"):
                self.logic_manager.register_prolog_handler(command[22:].strip())
                return None
            if command_upper.startswith("SAVE PIPE"):
                match = re.match(r"SAVE PIPE\s+(\w+)\s*,\s*\"([^\"]+)\"\s*(COMPRESS)?", command, re.IGNORECASE)
                if match:
                    pid, path, compress = match.groups()
                    self.save_manager.save_pipe(pid, self.current_scope().get(pid), path, bool(compress))
                    return None
                raise PdsXException("SAVE PIPE komutunda sözdizimi hatası")
            if command_upper.startswith("ON INTERRUPT"):
                match = re.match(r"ON\s+INTERRUPT\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    sig, evt = match.groups()
                    self.event_manager.map_signal(sig, evt)
                    return None
                raise PdsXException("ON INTERRUPT komutunda sözdizimi hatası")
            raise PdsXException(f"Bilinmeyen komut: {command}")

        except Exception as e:
            self.backtrace_logger.log(f"Error: {str(e)}")
            if self.error_sub:
                return self.execute_command(f"CALL {self.error_sub}", scope_name)
            elif self.error_handler:
                self.current_scope()["ERR"] = str(e)
                return self.error_handler
            raise PdsXException(f"Hata: {str(e)}")

    def run(self, code):
        self.parse_program(code)
        self.running = True
        self.program_counter = 0
        while self.running and self.program_counter < len(self.program):
            command = self.program[self.program_counter]
            if self.debug_mode:
                print(f"DEBUG: Satır {self.program_counter + 1}: {command}")
            next_pc = self.execute_command(command)
            if next_pc is not None:
                self.program_counter = next_pc
            else:
                self.program_counter += 1
        self.running = False

    def repl(self):
        self.repl_mode = True
        print("pdsXu REPL - Çıkmak için EXIT yazın")
        while self.repl_mode:
            try:
                command = input(">>> ")
                if command.strip().upper() == "EXIT":
                    self.repl_mode = False
                    break
                self.execute_command(command)
            except PdsXException as e:
                print(f"Hata: {e}")
            except Exception as e:
                print(f"Beklenmeyen hata: {e}")
        self.repl_mode = False

def main():
    parser = argparse.ArgumentParser(description="pdsXu Interpreter v14u")
    parser.add_argument("file", nargs="?", help="Çalıştırılacak BASIC dosyası")
    parser.add_argument("--repl", action="store_true", help="REPL modunda çalıştır")
    parser.add_argument("--log", default="INFO", help="Loglama seviyesi")
    args = parser.parse_args()

    lvl = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(lvl)

    interpreter = PdsXv14uInterpreter()
    if args.repl:
        interpreter.repl()
    elif args.file:
        src = Path(args.file).read_text(encoding="utf-8", errors="ignore")
        interpreter.run(src)
    else:
        print("Lütfen bir dosya belirtin veya --repl ile REPL modunu kullanın")

if __name__ == "__main__":
    main()