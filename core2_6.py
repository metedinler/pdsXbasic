# core2.py - PDS-X BASIC v15 Çekirdek İşlemler Kütüphanesi
# Version: 1.5.0
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import os
import sys
import re
import time
import json
import logging
import asyncio
try:
    import aiofiles
except ImportError:
    aiofiles = None  # aiofiles will be None if not installed
import random
import threading
import ctypes
import requests
import pdfplumber
import aiohttp
import decimal
import shutil
import platform
import subprocess
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache, reduce
import numpy as np
import pandas as pd
from scipy import stats, linalg, fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from hashlib import sha256
import psutil
import graphviz
from pdsx_unified_exception import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError, PdsXTypeError,
    PdsXValueError, PdsXIOError, PdsXNetworkError
)
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import paho.mqtt.client as mqtt
import datetime

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("core2")

# Özel Yüksek Hassasiyetli Tipler
class Float128(decimal.Decimal):
    def __init__(self, value):
        decimal.getcontext().prec = 34
        super().__init__(value)

class Float256(decimal.Decimal):
    def __init__(self, value):
        decimal.getcontext().prec = 68
        super().__init__(value)

class Float512(decimal.Decimal):
    def __init__(self, value):
        decimal.getcontext().prec = 136
        super().__init__(value)

class Skaler(float):
    def __new__(cls, value):
        return float.__new__(cls, value)

class Vector:
    def __init__(self, data: List[float]):
        self.data = [float(x) for x in data]
    
    def to_numpy(self):
        return np.array(self.data)

class Matrix:
    def __init__(self, data: List[List[float]]):
        self.data = [[float(x) for x in row] for row in data]
    
    def to_numpy(self):
        return np.array(self.data)

class Tensor:
    def __init__(self, data: List):
        self.data = data
    
    def to_numpy(self):
        return np.array(self.data)

class QuantumState:
    def __init__(self, amplitudes: List[complex]):
        self.amplitudes = [complex(x) for x in amplitudes]
        norm = sum(abs(a) ** 2 for a in self.amplitudes) ** 0.5
        self.amplitudes = [a / norm for a in self.amplitudes]
    
    def measure(self):
        probs = [abs(a) ** 2 for a in self.amplitudes]
        return np.random.choice(len(self.amplitudes), p=probs)
    
    def to_numpy(self):
        return np.array(self.amplitudes, dtype=np.complex128)

class HoloData:
    def __init__(self, data: bytes):
        self.data = data
        self.compressed = self._compress()
    
    def _compress(self):
        data_array = np.frombuffer(self.data, dtype=np.uint8)
        fft_data = fft.fft(data_array)
        return fft_data.tobytes()

class ChaosField:
    def __init__(self, data: List[float]):
        self.data = data
    
    def simulate(self, steps: int = 1000):
        def lorenz(t, state, sigma=10, rho=28, beta=8/3):
            x, y, z = state
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        from scipy.integrate import solve_ivp
        t_span = (0, steps / 10)
        y0 = self.data[:3] if len(self.data) >= 3 else [1.0, 1.0, 1.0]
        sol = solve_ivp(lorenz, t_span, y0, t_eval=np.linspace(0, steps / 10, steps))
        return sol.y.tolist()

class NeuralTensor:
    def __init__(self, data: List):
        self.data = data
    
    def to_numpy(self):
        return np.array(self.data)

class BlockchainLedger:
    def __init__(self, ledger: Dict, interpreter):
        self.ledger = ledger
        self.interpreter = interpreter
    
    def add_block(self, data: Dict):
        prev_hash = self.interpreter.current_scope().get("_PREV_BLOCK_HASH", "")
        block = {"data": data, "prev_hash": prev_hash}
        block_str = json.dumps(block, sort_keys=True)
        block_hash = sha256(block_str.encode("utf-8")).hexdigest()
        self.ledger[block_hash] = block
        self.interpreter.current_scope()["_PREV_BLOCK_HASH"] = block_hash

class SecureData:
    def __init__(self, data: bytes, key: bytes):
        self.data = self._encrypt(data, key)
    
    def _encrypt(self, data: bytes, key: bytes) -> bytes:
        f = Fernet(key)
        return f.encrypt(data)
    
    def decrypt(self, key: bytes) -> bytes:
        f = Fernet(key)
        return f.decrypt(self.data)

class Signature:
    def __init__(self, data: bytes, private_key: bytes):
        self.signature = self._sign(data, private_key)
    
    def _sign(self, data: bytes, private_key: bytes) -> bytes:
        priv = serialization.load_pem_private_key(private_key, password=None)
        return priv.sign(
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
    
    def verify(self, data: bytes, public_key: bytes) -> bool:
        pub = serialization.load_pem_public_key(public_key)
        try:
            pub.verify(
                self.signature,
                data,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except:
            return False

class IotMessage:
    def __init__(self, topic: str, payload: bytes, timestamp: float):
        self.topic = topic
        self.payload = payload
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "payload": self.payload.decode(), "timestamp": self.timestamp}

class AsyncCore:
    def __init__(self):
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

class Core2:
    def __init__(self):
        self.async_core = AsyncCore()

def hash_data(data: str) -> str:
    try:
        data_bytes = str(data).encode()
        hash_value = sha256(data_bytes).hexdigest()
        return hash_value
    except Exception as e:
        raise PdsXRuntimeError(f"Hash hatası: {str(e)}")

class CoreManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.metadata: Dict = {
            "core2": {
                "version": "1.5.0",
                "dependencies": [
                    "numpy", "pandas", "scipy", "tensorflow", "pdfplumber",
                    "requests", "aiohttp", "graphviz", "aiofiles", "psutil",
                    "decimal", "platform", "functools"
                ]
            }
        }
        self.data_types: Dict[str, type] = {
            "BYTE": int, "SHORT": int, "INTEGER": int, "LONG": int,
            "SINGLE": float, "DOUBLE": float, "FLOAT": float,
            "FLOAT128": Float128, "FLOAT256": Float256, "FLOAT512": Float512,
            "CHAR": str, "STRING": str, "VARCHAR": str, "TEXT": str,
            "BOOLEAN": bool, "NULL": type(None), "NAN": float,
            "BITFIELD": int, "POINTER": None, "ATOM": str, "VOID": None,
            "STREAM": bytes, "LIST": list, "DICT": dict, "SET": set, "TUPLE": tuple,
            "ARRAY": np.ndarray, "DATAFRAME": pd.DataFrame,
            "SKALER": Skaler, "VECTOR": Vector, "MATRIX": Matrix, "TENSOR": Tensor,
            "STRUCT": dict, "UNION": dict, "ENUM": dict, "CLASS": dict, "CLAZZ": dict, "YAPI": dict,
            "STACK": deque, "QUEUE": deque, "TREE": dict, "GRAPH": dict,
            "QUANTUM_STATE": QuantumState, "HOLO_DATA": HoloData,
            "CHAOS_FIELD": ChaosField, "NEURAL_TENSOR": NeuralTensor, "BLOCKCHAIN_LEDGER": BlockchainLedger,
            "SECURE_DATA": SecureData, "SIGNATURE": Signature, "IOT_MESSAGE": IotMessage
        }
        self.command_handlers: Dict[str, Callable] = {
                        "LET": self.handle_let,
            "IF": self.handle_if,
            "FOR": self.handle_for,
            "FOREACH": self.handle_foreach,
            "WHILE": self.handle_while,
            "DIM": self.handle_dim,
            "END": self.handle_end,
            "CLASS": self.handle_class,
            "YAPI": self.handle_yapi,
            "PRINT": self.handle_print,
            "GOTO": self.handle_goto,
            "GOSUB": self.handle_gosub,
            "RETURN": self.handle_return,
            "SELECT CASE": self.handle_select_case,
            "DATA": self.handle_data,
            "READ": self.handle_read,
            "RESTORE": self.handle_restore,
            "CHAIN": self.handle_chain,
            "CONT": self.handle_cont,
            "STOP": self.handle_stop,
            "TRON": self.handle_tron,
            "TROFF": self.handle_troff,
            "COMMON": self.handle_common,
            "DECLARE": self.handle_declare,
            "DEF": self.handle_def,
            "EXIT": self.handle_exit,
            "UNDIM": self.handle_undim,
            "SETFIELD": self.handle_setfield,
            "GETFIELD": self.handle_getfield,
            "ADDFIELD": self.handle_addfield,
            "REMOVEFIELD": self.handle_removefield,
            "NEWOBJ": self.handle_newobj,
            "COUNTOBJ": self.handle_countobj,
            "INSPOBJ": self.handle_inspobj,
            "CALLAPI": self.handle_callapi,
            "CALLDLL": self.handle_calldll,
            "SART": self.handle_sart,
            "ALIAS": self.handle_alias,
            "RESTRICT": self.handle_restrict,
            "CLEAR BASIC": self.handle_clear_basic,
            "LISTFILES": self.handle_listfiles,
            "LISTPROG": self.handle_listprog,
            "CHECKFILE": self.handle_checkfile,
            "SCREEN": self.handle_screen,
            "SOUND": self.handle_sound,
            "SEC VAR": self.handle_sec_var,
            "MON VAR": self.handle_mon_var,
            "CONVERT": self.handle_convert,
            "CAST": self.handle_cast,
            "ALTER TABLE": self.handle_alter_table,
            "CREATE VIEW": self.handle_create_view,
            "ENCRYPT": self.handle_encrypt,
            "DECRYPT": self.handle_decrypt,
            "SIGN": self.handle_sign,
            "VERIFY": self.handle_verify,
            "SECURE_VAR": self.handle_secure_var,
            "CONNECT_IOT": self.handle_connect_iot,
            "PUBLISH_IOT": self.handle_publish_iot,
            "SUBSCRIBE_IOT": self.handle_subscribe_iot,
            # Yeni eklenen komutlar
            "SYSINFO": self.handle_sysinfo,
            "CPUINFO": self.handle_cpuinfo,
            "DISKINFO": self.handle_diskinfo,
            "MONITOR": self.handle_monitor,
            "ASSERT": self.handle_assert,
            "MERGE": self.handle_merge,
            "SORT": self.handle_sort,
            "MAP": self.handle_map,
            "FILTER": self.handle_filter,
            "REDUCE": self.handle_reduce,
            "TRY": self.handle_try,
            "TRACE": self.handle_trace,
            "DATEDIFF": self.handle_datediff,
            "WAIT": self.handle_wait,
            "PUBLISH_SYSINFO": self.handle_publish_sysinfo,
            "SIMPLE_MODE": self.handle_simple_mode,
            "SYSTEM": self.handle_system
        }
        self.function_table: Dict[str, Callable] = {
            # Standart PDS-X BASIC fonksiyonları
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
            "SQR": lambda x: float(np.sqrt(x)),
            "SIN": lambda x: float(np.sin(x)),
            "COS": lambda x: float(np.cos(x)),
            "TAN": lambda x: float(np.tan(x)),
            "LOG": lambda x: float(np.log(x)),
            "EXP": lambda x: float(np.exp(x)),
            "ATN": lambda x: float(np.arctan(x)),
            "FIX": lambda x: int(x),
            "ROUND": lambda x, n=0: round(x, n),
            "SGN": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
            "MOD": lambda x, y: x % y,
            "MIN": lambda *args: min(args),
            "MAX": lambda *args: max(args),
            "TIMER": lambda: time.time(),
            "DATE$": lambda: time.strftime("%m-%d-%Y"),
            "TIME$": lambda: time.strftime("%H:%M:%S"),
            "INKEY$": lambda: input()[:1],
            "ENVIRON$": lambda var: os.environ.get(var, ""),
            "COMMAND$": lambda: " ".join(sys.argv[1:]),
            "CSRLIN": lambda: 1,
            "POS": lambda x: 1,
            "VAL": lambda s: float(s) if s.replace(".", "").isdigit() else 0,
            "ASC": lambda c: ord(c[0]),
            # Bilimsel ve veri analizi fonksiyonları
            "MEAN": lambda x: float(np.mean(x)),
            "MEDIAN": lambda x: float(np.median(x)),
            "MODE": lambda x: float(stats.mode(x)[0][0]),
            "STD": lambda x: float(np.std(x)),
            "VAR": lambda x: float(np.var(x)),
            "SUM": lambda x: float(np.sum(x)),
            "PROD": lambda x: float(np.prod(x)),
            "PERCENTILE": lambda x, p: float(np.percentile(x, p)),
            "QUANTILE": lambda x, q: float(np.quantile(x, q)),
            "CORR": lambda x, y: float(np.corrcoef(x, y)[0, 1]),
            "COV": lambda x, y: float(np.cov(x, y)),
            "DESCRIBE": lambda df: df.describe(),
            "GROUPBY": lambda df, col: df.groupby(col),
            "FILTER": lambda df, cond: df.query(cond),
            "SORT": lambda df, col: df.sort_values(col),
            "HEAD": lambda df, n=5: df.head(n),
            "TAIL": lambda df, n=5: df.tail(n),
            "MERGE": lambda df1, df2, on: pd.merge(df1, df2, on=on),
            "TTEST": lambda sample1, sample2: stats.ttest_ind(sample1, sample2),
            "CHISQUARE": lambda observed: stats.chisquare(observed),
            "ANOVA": lambda *groups: stats.f_oneway(*groups),
            "REGRESS": lambda x, y: stats.linregress(x, y),
            "Np": lambda func, *args, **kwargs: getattr(np, func)(*args, **kwargs),
            "pd": lambda func, *args, **kwargs: getattr(pd, func)(*args, **kwargs),
            "sc": lambda func, *args, **kwargs: getattr(stats, func)(*args, **kwargs),
            # Özel fonksiyonlar
            "PDF_READ_TEXT": self.pdf_read_text,
            "PDF_EXTRACT_TABLES": self.pdf_extract_tables,
            "WEB_GET": self.web_get,
            "SYSTEM": self.system,
            "QUANTUM_CORR": self.quantum_correlation_analysis,
            "CHAOS_DETECT": self.chaos_pattern_detection,
            "NEURAL_PROCESS": self.neural_data_processing,
            "GENETIC_OPT": self.genetic_optimization_engine,
            "BLOCKCHAIN_CHECK": self.blockchain_integrity_check,
            "HASH": self.hash_data,
            "CHECK_AUTH": self.check_auth,
            "CHECK_IOT_STATUS": self.check_iot_status,
            "GET_IOT_MESSAGE": self.get_iot_message,
            # Yeni eklenen fonksiyonlar
            "SYSTEM_INFO": self.system_info,
            "CPU_INFO": self.cpu_info,
            "DISK_INFO": self.disk_info,
            "MONITOR_RESOURCES": self.monitor_resources,
            "ASSERT": self.assert_,
            "MERGE": self.merge,
            "SORT": self.sort,
            "MAP": self.map_,
            "FILTER": self.filter_,
            "REDUCE": self.reduce_,
            "TRY_CATCH": self.try_catch,
            "TRACE": self.trace,
            "DATE_DIFF": self.date_diff,
            "ASYNC_WAIT": self.async_wait
        }
        self._init_neural_model()

    def _init_neural_model(self) -> None:
        """Nöral ağ modelini başlatır."""
        try:
            self.neural_model = Sequential([
                LSTM(64, input_shape=(10, 1), return_sequences=True),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            self.neural_model.compile(optimizer="adam", loss="mse")
            log.debug("Nöral model başlatıldı")
        except Exception as e:
            log.error(f"Nöral model başlatma hatası: {str(e)}")
            raise PdsXRuntimeError(f"Nöral model başlatma hatası: {str(e)}", context={"source": "_init_neural_model"})

    async def handle_let(self, command: str) -> None:
        """Değişkene değer atama komutunu işler."""
        match = re.match(r"LET\s+(.+?)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LET komutunda sözdizimi hatası", context={"source": "handle_let", "line_no": self.interpreter.program_counter})
        var_names, expr = match.groups()
        try:
            values = self.interpreter.evaluate_expression(expr)
            var_names = [v.strip() for v in var_names.split(",")]
            with self.lock:
                if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                    if len(var_names) != len(values):
                        raise PdsXValueError(f"Değişken sayısı ({len(var_names)}) ile değer sayısı ({len(values)}) uyuşmuyor", code="LET001")
                    for var, val in zip(var_names, values):
                        self.interpreter.current_scope()[var] = val
                        self.interpreter.object_counter["VARIABLE"] += 1
                        self.interpreter.object_registry[id(val)] = {"type": "VARIABLE", "name": var, "atom": str(val)}
                elif isinstance(values, dict):
                    for var in var_names:
                        self.interpreter.current_scope()[var] = values.get(var, None)
                        self.interpreter.object_counter["VARIABLE"] += 1
                        self.interpreter.object_registry[id(values)] = {"type": "VARIABLE", "name": var, "atom": json.dumps(values)}
                elif isinstance(values, (pd.DataFrame, self.data_types["MATRIX"], self.data_types["TENSOR"], self.data_types["QUANTUM_STATE"], self.data_types["HOLO_DATA"], self.data_types["CHAOS_FIELD"], self.data_types["NEURAL_TENSOR"], self.data_types["BLOCKCHAIN_LEDGER"])):
                    for var in var_names:
                        self.interpreter.current_scope()[var] = values
                        type_name = next(t for t, v in self.data_types.items() if isinstance(values, v))
                        self.interpreter.object_counter[type_name] += 1
                        self.interpreter.object_registry[id(values)] = {"type": type_name, "name": var, "atom": str(values)}
                else:
                    for var in var_names:
                        self.interpreter.current_scope()[var] = values
                        self.interpreter.object_counter["VARIABLE"] += 1
                        self.interpreter.object_registry[id(values)] = {"type": "VARIABLE", "name": var, "atom": str(values)}
                self.interpreter.object_counter["LET"] += 1
            log.debug(f"Değişkenler atandı: {var_names} = {values}")
        except Exception as e:
            raise PdsXRuntimeError(f"Atama hatası: {str(e)}", code="LET002")

    async def handle_if(self, command: str) -> Optional[int]:
        """Koşullu yürütme komutunu işler."""
        match = re.match(r"IF\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+))?$", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("IF komutunda sözdizimi hatası", context={"source": "handle_if", "line_no": self.interpreter.program_counter})
        condition, then_block, else_block = match.groups()
        try:
            if self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(then_block.strip())
            elif else_block:
                await self.interpreter.execute_command(else_block.strip())
            with self.lock:
                self.interpreter.object_counter["IF"] += 1
        except Exception as e:
            raise PdsXRuntimeError(f"IF koşul değerlendirme hatası: {str(e)}", code="IF001")
        return None

    async def handle_for(self, command: str) -> Optional[int]:
        """Sayısal döngü komutunu işler."""
        match = re.match(r"FOR\s+(\w+)\s*=\s*(\S+)\s+TO\s+(\S+)(?:\s+STEP\s+(\S+))?\s+DO\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("FOR komutunda sözdizimi hatası", context={"source": "handle_for", "line_no": self.program_counter})
        var_name, start, end, step, body = match.groups()
        start, end = float(start), float(end)
        step = float(step) if step else 1.0
        if step == 0:
            raise PdsXValueError("Adım sıfır olamaz", code="FOR001")
        with self.lock:
            self.interpreter.current_scope()[var_name] = start
            self.loop_stack.append({
                "type": "FOR",
                "var": var_name,
                "start": start,
                "end": end,
                "step": step,
                "index": 0,
                "history": [start],
                "start_pc": self.program_counter
            })
            self.interpreter.object_counter["FOR"] += 1
        try:
            await self.interpreter.execute_command(body)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR döngü hatası: {str(e)}", code="FOR002")
        return None

    async def handle_foreach(self, command: str) -> Optional[int]:
        """Koleksiyon döngüsü kurar."""
        match = re.match(r"FOREACH\s+(\w+)\s+IN\s+(\w+)\s+DO\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("FOREACH komutunda sözdizimi hatası", context={"source": "handle_foreach", "line_no": self.program_counter})
        var_name, collection_name, body = match.groups()
        collection = self.interpreter.current_scope().get(collection_name)
        if not isinstance(collection, (list, dict, set, tuple, pd.DataFrame, np.ndarray)):
            raise PdsXTypeError(f"Geçersiz koleksiyon: {collection_name}", code="FOREACH001")
        items = list(collection.items() if isinstance(collection, dict) else collection)
        if isinstance(collection, pd.DataFrame):
            items = [row.to_dict() for _, row in collection.iterrows()]
        elif isinstance(collection, np.ndarray):
            items = collection.tolist()
        with self.lock:
            self.loop_stack.append({
                "type": "FOREACH",
                "var": var_name,
                "collection": collection,
                "items": items,
                "index": 0,
                "history": [],
                "start_pc": self.program_counter
            })
            self.interpreter.object_counter["FOREACH"] += 1
        try:
            if items:
                self.interpreter.current_scope()[var_name] = items[0] if not isinstance(collection, dict) else items[0][1]
                await self.interpreter.execute_command(body)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR EACH döngü hatası: {str(e)}", code="FOREACH002")
        return None

    async def handle_while(self, command: str) -> Optional[int]:
        """Koşullu döngü komutunu işler."""
        match = re.match(r"WHILE\s+(.+?)\s+DO\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("WHILE komutunda sözdizimi hatası", context={"source": "handle_while", "line_no": self.program_counter})
        condition, body = match.groups()
        with self.lock:
            self.loop_stack.append({"type": "WHILE", "start_pc": self.program_counter})
            self.interpreter.object_counter["WHILE"] += 1
        try:
            if self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(body)
                return self.loop_stack[-1]["start_pc"]
            else:
                self.loop_stack.pop()
        except Exception as e:
            raise PdsXRuntimeError(f"WHILE döngü hatası: {str(e)}", code="WHILE001")
        return None

    async def handle_dim(self, command: str) -> None:
        """Değişken tanımlama komutunu işler."""
        match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)(?:\s*=\s*(.+?))?(?:\s*\[\s*(\d+)\s*\])?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DIM komutunda sözdizimi hatası", context={"source": "handle_dim", "line_no": self.program_counter})
        var_name, type_name, initial_value, size = match.groups()
        if type_name not in self.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", code="DIM001")
        value = None
        try:
            if initial_value:
                value = self.interpreter.evaluate_expression(initial_value)
            elif type_name in ("LIST", "DICT", "SET", "STACK", "QUEUE"):
                value = [] if type_name == "LIST" else {} if type_name == "DICT" else set() if type_name == "SET" else deque()
            elif type_name == "ARRAY":
                size = int(size) if size else 16
                value = np.zeros(size, dtype=np.float32)
            elif type_name == "DATAFRAME":
                value = pd.DataFrame()
            elif type_name == "SKALER":
                value = self.data_types["SKALER"](0.0)
            elif type_name == "VECTOR":
                size = int(size) if size else 16
                value = self.data_types["VECTOR"]([0.0] * size)
            elif type_name == "MATRIX":
                size = int(size) if size else 16
                value = self.data_types["MATRIX"]([[0.0] * size for _ in range(size)])
            elif type_name == "TENSOR":
                size = int(size) if size else 16
                value = self.data_types["TENSOR"]([[[0.0] * size for _ in range(size)] for _ in range(size)])
            elif type_name == "QUANTUM_STATE":
                size = int(size) if size else 2
                value = self.data_types["QUANTUM_STATE"]([1.0 / (size ** 0.5)] * size)
            elif type_name == "HOLO_DATA":
                value = self.data_types["HOLO_DATA"](b"")
            elif type_name == "CHAOS_FIELD":
                value = self.data_types["CHAOS_FIELD"]([1.0, 1.0, 1.0])
            elif type_name == "NEURAL_TENSOR":
                value = self.data_types["NEURAL_TENSOR"]([])
            elif type_name == "BLOCKCHAIN_LEDGER":
                value = self.data_types["BLOCKCHAIN_LEDGER"]({}, self.interpreter)
            elif type_name == "ENUM":
                value = {}
            elif type_name in ("STRUCT", "UNION", "CLASS", "CLAZZ", "YAPI"):
                value = {}
            elif type_name in ("BYTE", "SHORT", "INTEGER", "LONG"):
                value = 0
            elif type_name in ("FLOAT128", "FLOAT256", "FLOAT512"):
                value = self.data_types[type_name]("0.0")
            with self.lock:
                self.interpreter.current_scope()[var_name] = value
                self.interpreter.object_counter[type_name] += 1
                self.interpreter.object_registry[id(value)] = {
                    "type": type_name,
                    "name": var_name,
                    "atom": str(value) if value is not None else "null"
                }
                self.interpreter.object_counter["DIM"] += 1
            log.debug(f"Değişken tanımlandı: {var_name} AS {type_name} = {value}")
        except Exception as e:
            raise PdsXRuntimeError(f"DIM tanımlama hatası: {str(e)}", code="DIM002")

    async def handle_end(self, command: str) -> None:
        """Blok veya program sonlandırma komutunu işler."""
        match = re.match(r"END\s*(\w+)?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("END komutunda sözdizimi hatası", context={"source": "handle_end", "line_no": self.program_counter})
        block_type = match.group(1)
        with self.lock:
            if block_type:
                if block_type not in self.data_types and block_type not in ("FOR", "WHILE", "IF", "CLASS", "YAPI", "SELECT"):
                    raise PdsXSyntaxError(f"Geçersiz END tipi: {block_type}", code="END001")
                if block_type in ("FOR", "WHILE"):
                    if not self.loop_stack or self.loop_stack[-1]["type"] != block_type:
                        raise PdsXRuntimeError(f"Kapatılacak {block_type} bloğu yok", code="END002")
                    self.loop_stack.pop()
                elif block_type == "IF":
                    if not self.interpreter.if_stack:
                        raise PdsXRuntimeError("Kapatılacak IF bloğu yok", code="END003")
                    self.interpreter.if_stack.pop()
            else:
                self.interpreter.running = False
            self.interpreter.object_counter["END"] += 1
        log.debug(f"Blok/Program sonlandırıldı: {block_type or 'PROGRAM'}")

    async def handle_class(self, command: str) -> None:
        """Sınıf tanımlama komutunu işler."""
        match = re.match(r"CLASS\s+(\w+)\s+DO\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CLASS komutunda sözdizimi hatası", context={"source": "handle_class", "line_no": self.program_counter})
        class_name, body = match.groups()
        class_def = {"methods": {}, "properties": {}, "subs": {}, "functions": {}}
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("SUB "):
                sub_match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)\s*(.+?)\s+END\s+SUB", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if sub_match:
                    sub_name, params, sub_body = sub_match.groups()
                    class_def["subs"][sub_name] = {"params": params, "body": sub_body}
                    self.interpreter.object_counter["SUB"] += 1
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}", code="CLASS001")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="CLASS002")
                    class_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}", code="CLASS003")
            elif line.upper().startswith("PROP "):
                prop_match = re.match(r"PROP\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if prop_match:
                    prop_name, prop_type = prop_match.groups()
                    if prop_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}", code="CLASS004")
                    class_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"PROP tanımı hatalı: {line}", code="CLASS005")
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[class_name] = class_def
            self.interpreter.object_counter["CLASS"] += 1
        log.debug(f"Sınıf tanımlandı: {class_name}")

    async def handle_yapi(self, command: str) -> None:
        """Nesne tabanlı sınıf oluşturma komutunu işler."""
        match = re.match(r"YAPI\s+(\w+)\s+DO\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("YAPI komutunda sözdizimi hatası", context={"source": "handle_yapi", "line_no": self.program_counter})
        yapi_name, body = match.groups()
        yapi_def = {"methods": {}, "properties": {}, "subs": {}, "functions": {}, "gamma": {}, "omega": {}}
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("SUB "):
                sub_match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)\s*(.+?)\s+END\s+SUB", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if sub_match:
                    sub_name, params, sub_body = sub_match.groups()
                    yapi_def["subs"][sub_name] = {"params": params, "body": sub_body}
                    self.interpreter.object_counter["SUB"] += 1
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}", code="YAPI001")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="YAPI002")
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}", code="YAPI003")
            elif line.upper().startswith("GAMMA "):
                gamma_match = re.match(r"GAMMA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+GAMMA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if gamma_match:
                    gamma_name, params, return_type, gamma_body = gamma_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="YAPI004")
                    yapi_def["gamma"][gamma_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": gamma_body,
                        "partial": lambda *args: lambda *rest: self.interpreter.evaluate_expression(f"{gamma_body}({','.join(map(str, args + rest))})")
                    }
                    self.interpreter.object_counter["GAMMA"] += 1
                    i += gamma_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"GAMMA tanımı hatalı: {line}", code="YAPI005")
            elif line.upper().startswith("OMEGA "):
                omega_match = re.match(r"OMEGA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+OMEGA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if omega_match:
                    omega_name, params, return_type, omega_body = omega_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="YAPI006")
                    yapi_def["omega"][omega_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": omega_body,
                        "self_apply": lambda x: self.interpreter.evaluate_expression(f"{omega_body}({x})")
                    }
                    self.interpreter.object_counter["OMEGA"] += 1
                    i += omega_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"OMEGA tanımı hatalı: {line}", code="YAPI007")
            elif line.upper().startswith("DIM "):
                dim_match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if dim_match:
                    prop_name, prop_type = dim_match.groups()
                    if prop_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}", code="YAPI008")
                    yapi_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"DIM tanımı hatalı: {line}", code="YAPI009")
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[yapi_name] = yapi_def
            self.interpreter.object_counter["YAPI"] += 1
        log.debug(f"YAPI tanımlandı: {yapi_name}")

    async def handle_print(self, command: str) -> None:
        """Ekrana yazdırma komutunu işler."""
        match = re.match(r"PRINT\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("PRINT komutunda sözdizimi hatası", context={"source": "handle_print", "line_no": self.interpreter.program_counter})
        expr = match.group(1)
        try:
            value = self.interpreter.evaluate_expression(expr)
            print(value)
            with self.lock:
                self.interpreter.object_counter["PRINT"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "PRINT",
                    "name": "PRINT",
                    "atom": str(value)[:100]
                }
            log.debug(f"PRINT yürütüldü: {value}")
        except Exception as e:
            raise PdsXRuntimeError(f"PRINT değerlendirme hatası: {str(e)}", code="PRINT001")

    async def handle_goto(self, command: str) -> Optional[int]:
        """Etikete atlama komutunu işler."""
        match = re.match(r"GOTO\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GOTO komutunda sözdizimi hatası", context={"source": "handle_goto", "line_no": self.interpreter.program_counter})
        label = match.group(1)
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOTO001")
        with self.lock:
            self.interpreter.object_counter["GOTO"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "GOTO",
                "name": label,
                "atom": command
            }
        log.debug(f"GOTO yürütüldü: {label}")
        return self.interpreter.labels[label]

    async def handle_gosub(self, command: str) -> Optional[int]:
        """Alt yordama atlama komutunu işler."""
        match = re.match(r"GOSUB\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GOSUB komutunda sözdizimi hatası", context={"source": "handle_gosub", "line_no": self.interpreter.program_counter})
        label = match.group(1)
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOSUB001")
        with self.lock:
            self.interpreter.call_stack.append({"return_pc": self.interpreter.program_counter + 1})
            self.interpreter.object_counter["GOSUB"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "GOSUB",
                "name": label,
                "atom": command
            }
        log.debug(f"GOSUB yürütüldü: {label}")
        return self.interpreter.labels[label]

    async def handle_return(self, command: str) -> Optional[int]:
        """Alt yordamdan dönüş komutunu işler."""
        if not self.interpreter.call_stack:
            raise PdsXRuntimeError("Geri dönülecek yordam yok", code="RETURN001")
        with self.lock:
            return_pc = self.interpreter.call_stack.pop()["return_pc"]
            self.interpreter.object_counter["RETURN"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "RETURN",
                "name": "RETURN",
                "atom": command
            }
        log.debug("RETURN yürütüldü")
        return return_pc

    async def handle_select_case(self, command: str) -> None:
        """Çoklu koşullu yapı komutunu işler."""
        match = re.match(r"SELECT\s+CASE\s+(.+?)\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SELECT CASE komutunda sözdizimi hatası", context={"source": "handle_select_case", "line_no": self.program_counter})
        expr, cases_str = match.groups()
        value = self.interpreter.evaluate_expression(expr)
        case_matched = False
        cases = re.findall(r"CASE\s+(.+?)\s+DO\s+(.+?)(?=CASE|END\s+SELECT|$)", cases_str, re.IGNORECASE | re.DOTALL)
        for case_value, case_body in cases:
            if case_value.strip().upper() == "ELSE" or self.interpreter.evaluate_expression(case_value.strip()) == value:
                await self.interpreter.execute_command(case_body.strip())
                case_matched = True
                break
        with self.lock:
            self.interpreter.object_counter["SELECT_CASE"] += 1
        log.debug(f"SELECT CASE yürütüldü: {expr}")

    async def handle_data(self, command: str) -> None:
        """Veri tanımlama komutunu işler."""
        match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DATA komutunda sözdizimi hatası", context={"source": "handle_data", "line_no": self.program_counter})
        values = match.group(1).split(",")
        with self.lock:
            self.interpreter.data_list.extend([v.strip() for v in values])
            self.interpreter.object_counter["DATA"] += len(values)
        log.debug(f"Veri tanımlandı: {values}")

    async def handle_read(self, command: str) -> None:
        """Veri okuma komutunu işler."""
        match = re.match(r"READ\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("READ komutunda sözdizimi hatası", context={"source": "handle_read", "line_no": self.program_counter})
        var_names = [v.strip() for v in match.group(1).split(",")]
        with self.lock:
            for var in var_names:
                if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                    raise PdsXRuntimeError("Veri listesi sonu", code="READ001")
                value = self.interpreter.data_list[self.interpreter.data_pointer]
                self.interpreter.current_scope()[var] = value
                self.interpreter.data_pointer += 1
                self.interpreter.object_counter["VARIABLE"] += 1
        log.debug(f"Veri okundu: {var_names}")

    async def handle_restore(self, command: str) -> None:
        """Veri işaretçisini sıfırlama komutunu işler."""
        with self.lock:
            self.interpreter.data_pointer = 0
            self.interpreter.object_counter["RESTORE"] += 1
        log.debug("Veri işaretçisi sıfırlandı")

    async def handle_chain(self, command: str) -> None:
        """Yeni programı zincirleme komutunu işler."""
        match = re.match(r"CHAIN\s+\"([^\"]+)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CHAIN komutunda sözdizimi hatası", context={"source": "handle_chain", "line_no": self.program_counter})
        program_file = match.group(1)
        try:
            async with aiofiles.open(program_file, "r", encoding="utf-8") as f:
                program_text = await f.read()
            self.interpreter.load_program(program_text)
            with self.lock:
                self.interpreter.object_counter["CHAIN"] += 1
        except Exception as e:
            raise PdsXIOException(f"Program yükleme hatası: {str(e)}", code="CHAIN001")

    async def handle_cont(self, command: str) -> None:
        """Yürütmeye devam etme komutunu işler."""
        with self.lock:
            self.interpreter.paused = False
            self.interpreter.object_counter["CONT"] += 1
        log.debug("Yürütme devam ediyor")

    async def handle_stop(self, command: str) -> None:
        """Yürütmeyi durdurma komutunu işler."""
        with self.lock:
            self.interpreter.paused = True
            self.interpreter.object_counter["STOP"] += 1
        log.debug("Yürütme durduruldu")

    async def handle_tron(self, command: str) -> None:
        """İzleme modunu açma komutunu işler."""
        with self.lock:
            self.interpreter.trace_mode = True
            self.interpreter.object_counter["TRON"] += 1
        log.debug("İzleme modu açıldı")

    async def handle_troff(self, command: str) -> None:
        """İzleme modunu kapatma komutunu işler."""
        with self.lock:
            self.interpreter.trace_mode = False
            self.interpreter.object_counter["TROFF"] += 1
        log.debug("İzleme modu kapatıldı")

    async def handle_common(self, command: str) -> None:
        """Değişkenleri paylaşma komutunu işler."""
        match = re.match(r"COMMON\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("COMMON komutunda sözdizimi hatası", context={"source": "handle_common", "line_no": self.program_counter})
        var_names = [v.strip() for v in match.group(1).split(",")]
        with self.lock:
            for var in var_names:
                if var in self.interpreter.current_scope():
                    self.interpreter.shared_vars[var].append(self.interpreter.current_scope()[var])
                    self.interpreter.object_counter["COMMON"] += 1
                else:
                    raise PdsXRuntimeError(f"Değişken bulunamadı: {var}", code="COMMON001")
        log.debug(f"Paylaşılan değişkenler: {var_names}")

    async def handle_declare(self, command: str) -> None:
        """Fonksiyon/yordam tanımlama komutunu işler."""
        match = re.match(r"DECLARE\s+(SUB|FUNCTION)\s+(\w+)\s*\((.*?)\)(?:\s+AS\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DECLARE komutunda sözdizimi hatası", context={"source": "handle_declare", "line_no": self.program_counter})
        decl_type, name, params, return_type = match.groups()
        with self.lock:
            if decl_type.upper() == "SUB":
                self.interpreter.subs[name] = {"params": params}
                self.interpreter.object_counter["SUB"] += 1
            else:
                if return_type not in self.data_types:
                    raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="DECLARE001")
                self.interpreter.functions[name] = {"params": params, "return_type": return_type}
                self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"{decl_type} tanımlandı: {name}")

    async def handle_def(self, command: str) -> None:
        """Fonksiyon tanımlama komutunu işler."""
        match = re.match(r"DEF\s+(\w+)\s*\((.*?)\)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DEF komutunda sözdizimi hatası", context={"source": "handle_def", "line_no": self.program_counter})
        func_name, params, expr = match.groups()
        with self.lock:
            self.interpreter.functions[func_name] = {
                "params": params,
                "body": lambda *args: self.interpreter.evaluate_expression(f"{expr}({','.join(map(str, args))})")
            }
            self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"Fonksiyon tanımlandı: {func_name}")

    async def handle_exit(self, command: str) -> None:
        """Döngü/yordamdan çıkma komutunu işler."""
        match = re.match(r"EXIT\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("EXIT komutunda sözdizimi hatası", context={"source": "handle_exit", "line_no": self.program_counter})
        exit_type = match.group(1).upper()
        with self.lock:
            if exit_type in ("FOR", "WHILE"):
                if not self.loop_stack or self.loop_stack[-1]["type"] != exit_type:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} döngüsü yok", code="EXIT001")
                self.loop_stack.pop()
            else:
                if not self.call_stack:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} yordamı yok", code="EXIT002")
                self.call_stack.pop()
            self.interpreter.object_counter["EXIT"] += 1
        log.debug(f"{exit_type}’den çıkıldı")

    async def handle_undim(self, command: str) -> None:
        """Değişken kaldırma komutunu işler."""
        match = re.match(r"UNDIM\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("UNDIM komutunda sözdizimi hatası", context={"source": "handle_undim", "line_no": self.program_counter})
        var_name = match.group(1)
        with self.lock:
            if var_name in self.interpreter.current_scope():
                value = self.interpreter.current_scope()[var_name]
                del self.interpreter.current_scope()[var_name]
                type_name = next((t for t, v in self.data_types.items() if isinstance(value, v)), "VARIABLE")
                self.interpreter.object_counter[type_name] -= 1
                self.interpreter.object_counter["UNDIM"] += 1
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="UNDIM001")
        log.debug(f"Değişken kaldırıldı: {var_name}")

    async def handle_setfield(self, command: str) -> None:
        """Yapı alanını güncelleme komutunu işler."""
        match = re.match(r"SETFIELD\s+(\w+)\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SETFIELD komutunda sözdizimi hatası", context={"source": "handle_setfield", "line_no": self.program_counter})
        var_name, field, value_expr = match.groups()
        value = self.interpreter.evaluate_expression(value_expr)
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="SETFIELD001")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", code="SETFIELD002")
            struct[field] = value
            self.interpreter.object_counter["SETFIELD"] += 1
        log.debug(f"Yapı alanı güncellendi: {var_name}.{field} = {value}")

    async def handle_getfield(self, command: str) -> None:
        """Yapı alanını alma komutunu işler."""
        match = re.match(r"GETFIELD\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GETFIELD komutunda sözdizimi hatası", context={"source": "handle_getfield", "line_no": self.program_counter})
        var_name, field, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="GETFIELD001")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", code="GETFIELD002")
            value = struct.get(field)
            self.interpreter.current_scope()[new_var] = value
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_counter["GETFIELD"] += 1
        log.debug(f"Yapı alanı alındı: {new_var} = {var_name}.{field}")

    async def handle_addfield(self, command: str) -> None:
        """Yapıya dinamik alan ekleme komutunu işler."""
        match = re.match(r"ADDFIELD\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ADDFIELD komutunda sözdizimi hatası", context={"source": "handle_addfield", "line_no": self.program_counter})
        var_name, field, type_name = match.groups()
        if type_name not in self.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", code="ADDFIELD001")
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="ADDFIELD002")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", code="ADDFIELD003")
            struct[field] = None
            self.interpreter.object_counter[type_name] += 1
            self.interpreter.object_counter["ADDFIELD"] += 1
        log.debug(f"Yapıya alan eklendi: {var_name}.{field} AS {type_name}")

    async def handle_removefield(self, command: str) -> None:
        """Yapıdan alan kaldırma komutunu işler."""
        match = re.match(r"REMOVEFIELD\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("REMOVEFIELD komutunda sözdizimi hatası", context={"source": "handle_removefield", "line_no": self.program_counter})
        var_name, field = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="REMOVEFIELD001")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", code="REMOVEFIELD002")
            if field in struct:
                struct.pop(field)
                self.interpreter.object_counter["REMOVEFIELD"] += 1
            else:
                raise PdsXRuntimeError(f"Alan bulunamadı: {field}", code="REMOVEFIELD003")
        log.debug(f"Yapıdan alan kaldırıldı: {var_name}.{field}")

    async def handle_newobj(self, command: str) -> None:
        """Nesne oluşturma komutunu işler."""
        match = re.match(r"NEWOBJ\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("NEWOBJ komutunda sözdizimi hatası", context={"source": "handle_newobj", "line_no": self.program_counter})
        class_name, params, var_name = match.groups()
        if class_name not in self.interpreter.classes:
            raise PdsXRuntimeError(f"Sınıf bulunamadı: {class_name}", code="NEWOBJ001")
        class_def = self.interpreter.classes[class_name]
        instance = {
            "class": class_name,
            "properties": {prop: None for prop, prop_type in class_def["properties"].items()},
            "subs": class_def["subs"],
            "functions": class_def["functions"],
            "gamma": class_def.get("gamma", {}),
            "omega": class_def.get("omega", {})
        }
        param_values = [self.interpreter.evaluate_expression(p.strip()) for p in params.split(",") if p.strip()]
        if "Init" in class_def["subs"]:
            init_params = class_def["subs"]["Init"]["params"].split(",")
            if len(param_values) != len([p for p in init_params if p.strip()]):
                raise PdsXValueError(f"Geçersiz parametre sayısı: {class_name}.Init", code="NEWOBJ002")
            with self.lock:
                self.interpreter.current_scope().update({f"_{i}": v for i, v in enumerate(param_values)})
                await self.interpreter.execute_command(class_def["subs"]["Init"]["body"])
                for i in range(len(param_values)):
                    self.interpreter.current_scope().pop(f"_{i}", None)
        with self.lock:
            self.interpreter.current_scope()[var_name] = instance
            self.interpreter.object_counter["OBJECT"] += 1
            self.interpreter.object_counter["NEWOBJ"] += 1
        log.debug(f"Nesne oluşturuldu: {var_name} AS {class_name}")

    async def handle_countobj(self, command: str) -> None:
        """Nesne sayısını sayma komutunu işler."""
        match = re.match(r"COUNTOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("COUNTOBJ komutunda sözdizimi hatası", context={"source": "handle_countobj", "line_no": self.program_counter})
        type_name, var_name = match.groups()
        with self.lock:
            count = self.interpreter.object_counter.get(type_name, 0)
            self.interpreter.current_scope()[var_name] = count
            self.interpreter.object_counter["COUNTOBJ"] += 1
        log.debug(f"Nesne sayıldı: {type_name} = {count}")

    async def handle_inspobj(self, command: str) -> None:
        """Nesne detaylarını alma komutunu işler."""
        match = re.match(r"INSPOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("INSPOBJ komutunda sözdizimi hatası", context={"source": "handle_inspobj", "line_no": self.program_counter})
        obj_id, var_name = match.groups()
        with self.lock:
            obj = self.interpreter.object_registry.get(int(obj_id), {})
            if not obj:
                raise PdsXRuntimeError(f"Nesne bulunamadı: {obj_id}", code="INSPOBJ001")
            self.interpreter.current_scope()[var_name] = obj
            self.interpreter.object_counter["INSPOBJ"] += 1
        log.debug(f"Nesne incelendi: {obj_id}")

    async def handle_callapi(self, command: str) -> None:
        """HTTP API isteği yapma komutunu işler."""
        match = re.match(r"CALLAPI\s+\"([^\"]+)\"\s+(\w+)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLAPI komutunda sözdizimi hatası", context={"source": "handle_callapi", "line_no": self.program_counter})
        url, method, headers, data, var_name = match.groups()
        try:
            headers = json.loads(headers)
            data = json.loads(data)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for attempt in range(3):
                    try:
                        if method.upper() == "GET":
                            async with session.get(url, headers=headers, params=data) as resp:
                                if resp.status != 200:
                                    raise PdsXNetworkError(f"HTTP hatası: {resp.status}", code="CALLAPI001")
                                response = await resp.json()
                        elif method.upper() == "POST":
                            async with session.post(url, headers=headers, json=data) as resp:
                                if resp.status != 200:
                                    raise PdsXNetworkError(f"HTTP hatası: {resp.status}", code="CALLAPI002")
                                response = await resp.json()
                        else:
                            raise PdsXValueError(f"Desteklenmeyen metod: {method}", code="CALLAPI003")
                        break
                    except aiohttp.ClientError as e:
                        if attempt == 2:
                            raise PdsXNetworkError(f"API çağrısı hatası: {str(e)}", code="CALLAPI004")
                        await asyncio.sleep(1)
            with self.lock:
                self.interpreter.current_scope()[var_name] = response
                self.interpreter.object_counter["CALLAPI"] += 1
        except Exception as e:
            raise PdsXNetworkError(f"API çağrısı hatası: {str(e)}", code="CALLAPI005")

    async def handle_calldll(self, command: str) -> None:
        """DLL fonksiyonu çağırma komutunu işler."""
        match = re.match(r"CALLDLL\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*\((.*?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLDLL komutunda sözdizimi hatası", context={"source": "handle_calldll", "line_no": self.program_counter})
        dll_name, func_name, params, var_name = match.groups()
        try:
            dll = ctypes.WinDLL(dll_name)
            func = getattr(dll, func_name)
            param_values = [self.interpreter.evaluate_expression(p.strip()) for p in params.split(",") if p.strip()]
            result = func(*param_values)
            with self.lock:
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["CALLDLL"] += 1
        except Exception as e:
            raise PdsXRuntimeError(f"DLL çağrısı hatası: {str(e)}", code="CALLDLL001")

    async def handle_sart(self, command: str) -> Optional[int]:
        """Koşullu boru hattı atlama komutunu işler."""
        match = re.match(r"SART\s+(.+?)\s+IN\s+(\w+)\s+TO\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SART komutunda sözdizimi hatası", context={"source": "handle_sart", "line_no": self.program_counter})
        condition, pipe_id, label = match.groups()
        try:
            if self.interpreter.evaluate_expression(condition):
                if pipe_id not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", code="SART001")
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="SART002")
                with self.lock:
                    self.interpreter.object_counter["SART"] += 1
                log.debug(f"SART atlandı: {pipe_id} -> {label}")
                return self.interpreter.labels[label]
        except Exception as e:
            raise PdsXRuntimeError(f"SART değerlendirme hatası: {str(e)}", code="SART003")
        return None

    async def handle_alias(self, command: str) -> None:
        """İsim değiştirme komutunu işler."""
        match = re.match(r"ALIAS\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ALIAS komutunda sözdizimi hatası", context={"source": "handle_alias", "line_no": self.program_counter})
        old_name, new_name = match.groups()
        with self.lock:
            if old_name in self.interpreter.current_scope():
                self.interpreter.current_scope()[new_name] = self.interpreter.current_scope()[old_name]
                self.interpreter.object_counter["ALIAS"] += 1
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {old_name}", code="ALIAS001")
        log.debug(f"İsim değiştirildi: {old_name} AS {new_name}")

    async def handle_restrict(self, command: str) -> None:
        """Erişim sınırlama komutunu işler."""
        match = re.match(r"RESTRICT\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("RESTRICT komutunda sözdizimi hatası", context={"source": "handle_restrict", "line_no": self.program_counter})
        scope = match.group(1).upper()
        with self.lock:
            if scope not in ("GLOBAL", "SHARED", "LOCAL"):
                raise PdsXValueError(f"Geçersiz kapsam: {scope}", code="RESTRICT001")
            self.interpreter.restricted_scopes.add(scope)
            self.interpreter.object_counter["RESTRICT"] += 1
        log.debug(f"Kapsam sınırlandırıldı: {scope}")

    async def handle_clear_basic(self, command: str) -> None:
        """Değişkenleri sıfırlama komutunu işler."""
        match = re.match(r"CLEAR\s+BASIC\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CLEAR BASIC komutunda sözdizimi hatası", context={"source": "handle_clear_basic", "line_no": self.program_counter})
        scope = match.group(1).upper()
        with self.lock:
            if scope == "GLOBAL":
                self.interpreter.global_vars.clear()
            elif scope == "SHARED":
                self.interpreter.shared_vars.clear()
            elif scope == "LOCAL":
                self.interpreter.current_scope().clear()
            else:
                raise PdsXValueError(f"Geçersiz kapsam: {scope}", code="CLEAR_BASIC001")
            self.interpreter.object_counter["CLEAR_BASIC"] += 1
        log.debug(f"{scope} değişkenler sıfırlandı")

    async def handle_listfiles(self, command: str) -> None:
        """Dosya numaralarını listeleme komutunu işler."""
        match = re.match(r"LISTFILES\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LISTFILES komutunda sözdizimi hatası", context={"source": "handle_listfiles", "line_no": self.program_counter})
        var_name = match.group(1)
        files = list(self.interpreter.file_handles.keys())
        with self.lock:
            self.interpreter.current_scope()[var_name] = files
            self.interpreter.object_counter["LISTFILES"] += 1
        log.debug(f"Dosyalar listelendi: {files}")

    async def handle_listprog(self, command: str) -> None:
        """Program satırlarını listeleme komutunu işler."""
        match = re.match(r"LISTPROG\s+(\d+)\s+TO\s+(\d+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LISTPROG komutunda sözdizimi hatası", context={"source": "handle_listprog", "line_no": self.program_counter})
        start, end = map(int, match.groups())
        lines = [self.interpreter.program[i][0] for i in range(start-1, min(end, len(self.interpreter.program)))]
        with self.lock:
            self.interpreter.current_scope()["_PROGRAM_LIST"] = lines
            self.interpreter.object_counter["LISTPROG"] += 1
        for line in lines:
            print(line)
        log.debug(f"Program listelendi: {start} TO {end}")

    async def handle_checkfile(self, command: str) -> None:
        """Dosya durumunu kontrol etme komutunu işler."""
        match = re.match(r"CHECKFILE\s+#(\d+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CHECKFILE komutunda sözdizimi hatası", context={"source": "handle_checkfile", "line_no": self.program_counter})
        file_num, var_name = match.groups()
        file_num = int(file_num)
        status = {"exists": file_num in self.interpreter.file_handles, "open": False}
        if status["exists"]:
            status["open"] = not self.interpreter.file_handles[file_num].closed
        with self.lock:
            self.interpreter.current_scope()[var_name] = status
            self.interpreter.object_counter["CHECKFILE"] += 1
        log.debug(f"Dosya durumu kontrol edildi: #{file_num}, {status}")

    async def handle_screen(self, command: str) -> None:
        """Grafik ekran modunu ayarlama komutunu işler."""
        match = re.match(r"SCREEN\s+(\d+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SCREEN komutunda sözdizimi hatası", context={"source": "handle_screen", "line_no": self.program_counter})
        mode = int(match.group(1))
        with self.lock:
            self.interpreter.current_scope()["_SCREEN_MODE"] = mode
            self.interpreter.object_counter["SCREEN"] += 1
        log.debug(f"Ekran modu ayarlandı: {mode}")

    async def handle_sound(self, command: str) -> None:
        """Ses üretme komutunu işler."""
        match = re.match(r"SOUND\s+(\d+)\s*,\s*(\d+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SOUND komutunda sözdizimi hatası", context={"source": "handle_sound", "line_no": self.program_counter})
        freq, duration = map(int, match.groups())
        with self.lock:
            self.interpreter.object_counter["SOUND"] += 1
        log.debug(f"Ses üretildi: Frekans={freq}, Süre={duration}")

    async def handle_sec_var(self, command: str) -> None:
        """Değişken erişimini kısıtlama komutunu işler."""
        match = re.match(r"SEC\s+VAR\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SEC VAR komutunda sözdizimi hatası", context={"source": "handle_sec_var", "line_no": self.program_counter})
        var_name = match.group(1)
        with self.lock:
            self.interpreter.restricted_vars.add(var_name)
            self.interpreter.object_counter["SEC_VAR"] += 1
        log.debug(f"Değişken kısıtlandı: {var_name}")

    async def handle_mon_var(self, command: str) -> None:
        """Değişken istatistiklerini toplama komutunu işler."""
        match = re.match(r"MON\s+VAR\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("MON VAR komutunda sözdizimi hatası", context={"source": "handle_mon_var", "line_no": self.program_counter})
        var_name, stat_var = match.groups()
        stats = {"access_count": 0, "last_access": time.time()}
        if var_name in self.interpreter.current_scope():
            stats["value"] = self.interpreter.current_scope()[var_name]
        with self.lock:
            self.interpreter.current_scope()[stat_var] = stats
            self.interpreter.object_counter["MON_VAR"] += 1
        log.debug(f"Değişken izlendi: {var_name}, İstatistikler: {stats}")

    async def handle_convert(self, command: str) -> None:
        """Tip dönüşüm komutunu işler."""
        match = re.match(r"CONVERT\s+(\w+)\s+FROM\s+(\w+)\s+TO\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CONVERT komutunda sözdizimi hatası", context={"source": "handle_convert", "line_no": self.program_counter})
        var_name, source_type, target_type, new_var = match.groups()
        if source_type not in self.data_types or target_type not in self.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {source_type} veya {target_type}", code="CONVERT001")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="CONVERT002")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CONVERT"] += 1
            log.debug(f"Tip dönüşümü yapıldı: {var_name} ({source_type}) -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", code="CONVERT003")

    async def handle_cast(self, command: str) -> None:
        """Hızlı tip dönüşüm komutunu işler."""
        match = re.match(r"CAST\s+(\w+)\s+TO\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CAST komutunda sözdizimi hatası", context={"source": "handle_cast", "line_no": self.program_counter})
        var_name, target_type, new_var = match.groups()
        if target_type not in self.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {target_type}", code="CAST001")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="CAST002")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CAST"] += 1
            log.debug(f"Hızlı tip dönüşümü yapıldı: {var_name} -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", code="CAST003")

    async def handle_alter_table(self, command: str) -> None:
        """Veritabanı tablosunu değiştirme komutunu işler."""
        match = re.match(r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ALTER TABLE komutunda sözdizimi hatası", context={"source": "handle_alter_table", "line_no": self.program_counter})
        table_name, column_name, column_type = match.groups()
        with self.lock:
            self.interpreter.object_counter["ALTER_TABLE"] += 1
        log.debug(f"Tablo değiştirildi: {table_name}, Yeni sütun: {column_name} AS {column_type}")

    async def handle_create_view(self, command: str) -> None:
        """Veritabanı görünümü oluşturma komutunu işler."""
        match = re.match(r"CREATE\s+VIEW\s+(\w+)\s+AS\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CREATE VIEW komutunda sözdizimi hatası", context={"source": "handle_create_view", "line_no": self.interpreter.program_counter})
        view_name, query = match.groups()
        with self.lock:
            self.interpreter.object_counter["CREATE_VIEW"] += 1
        log.debug(f"Görünüm oluşturuldu: {view_name}, Sorgu: {query}")

    async def handle_encrypt(self, command: str) -> None:
        """Veriyi şifreleme komutunu işler."""
        match = re.match(r"ENCRYPT\s+(\w+)\s+WITH\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ENCRYPT komutunda sözdizimi hatası", context={"source": "handle_encrypt", "line_no": self.interpreter.program_counter})
        var_name, key, new_var = match.groups()
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_encrypt", "line_no": self.interpreter.program_counter})
        data = str(self.interpreter.current_scope()[var_name]).encode()
        try:
            encrypted = SecureData(data, key.encode()).data
            with self.lock:
                self.interpreter.current_scope()[new_var] = encrypted
                self.interpreter.object_counter["ENCRYPT"] += 1
                self.interpreter.object_registry[id(encrypted)] = {
                    "type": "SECURE_DATA",
                    "name": new_var,
                    "atom": encrypted.hex()[:100]
                }
            log.debug(f"Veri şifrelendi: {var_name} -> {new_var}")
        except Exception as e:
            raise PdsXRuntimeError(f"Şifreleme hatası: {str(e)}", context={"source": "handle_encrypt"})

    async def handle_decrypt(self, command: str) -> None:
        """Şifrelenmiş veriyi çözme komutunu işler."""
        match = re.match(r"DECRYPT\s+(\w+)\s+WITH\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DECRYPT komutunda sözdizimi hatası", context={"source": "handle_decrypt", "line_no": self.interpreter.program_counter})
        var_name, key, new_var = match.groups()
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_decrypt", "line_no": self.interpreter.program_counter})
        encrypted_data = self.interpreter.current_scope()[var_name]
        if not isinstance(encrypted_data, bytes):
            raise PdsXTypeError(f"Şifrelenmiş veri değil: {var_name}", context={"source": "handle_decrypt"})
        try:
            secure_data = SecureData(b"", key.encode())
            decrypted = secure_data.decrypt(encrypted_data)
            with self.lock:
                self.interpreter.current_scope()[new_var] = decrypted.decode()
                self.interpreter.object_counter["DECRYPT"] += 1
                self.interpreter.object_registry[id(decrypted)] = {
                    "type": "STRING",
                    "name": new_var,
                    "atom": decrypted.decode()[:100]
                }
            log.debug(f"Veri çözüldü: {var_name} -> {new_var}")
        except Exception as e:
            raise PdsXRuntimeError(f"Şifre çözme hatası: {str(e)}", context={"source": "handle_decrypt"})

    async def handle_sign(self, command: str) -> None:
        """Veriyi imzalama komutunu işler."""
        match = re.match(r"SIGN\s+(\w+)\s+WITH\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SIGN komutunda sözdizimi hatası", context={"source": "handle_sign", "line_no": self.interpreter.program_counter})
        var_name, private_key, new_var = match.groups()
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_sign", "line_no": self.interpreter.program_counter})
        data = str(self.interpreter.current_scope()[var_name]).encode()
        try:
            signature = Signature(data, private_key.encode()).signature
            with self.lock:
                self.interpreter.current_scope()[new_var] = signature
                self.interpreter.object_counter["SIGN"] += 1
                self.interpreter.object_registry[id(signature)] = {
                    "type": "SIGNATURE",
                    "name": new_var,
                    "atom": signature.hex()[:100]
                }
            log.debug(f"Veri imzalandı: {var_name} -> {new_var}")
        except Exception as e:
            raise PdsXRuntimeError(f"İmzalama hatası: {str(e)}", context={"source": "handle_sign"})

    async def handle_verify(self, command: str) -> None:
        """İmzayı doğrulama komutunu işler."""
        match = re.match(r"VERIFY\s+(\w+)\s+WITH\s+(\w+)\s+AND\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("VERIFY komutunda sözdizimi hatası", context={"source": "handle_verify", "line_no": self.interpreter.program_counter})
        data_var, signature_var, public_key, result_var = match.groups()
        if data_var not in self.interpreter.current_scope() or signature_var not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {data_var} veya {signature_var}", context={"source": "handle_verify", "line_no": self.interpreter.program_counter})
        data = str(self.interpreter.current_scope()[data_var]).encode()
        signature = self.interpreter.current_scope()[signature_var]
        try:
            signature_obj = Signature(b"", b"")
            is_valid = signature_obj.verify(data, signature, public_key.encode())
            with self.lock:
                self.interpreter.current_scope()[result_var] = is_valid
                self.interpreter.object_counter["VERIFY"] += 1
                self.interpreter.object_registry[id(is_valid)] = {
                    "type": "BOOLEAN",
                    "name": result_var,
                    "atom": str(is_valid)
                }
            log.debug(f"İmza doğrulandı: {data_var}, Sonuç: {is_valid}")
        except Exception as e:
            raise PdsXRuntimeError(f"Doğrulama hatası: {str(e)}", context={"source": "handle_verify"})

    async def handle_secure_var(self, command: str) -> None:
        """Güvenli değişken tanımlama komutunu işler."""
        match = re.match(r"SECURE_VAR\s+(\w+)\s+AS\s+(\w+)(?:\s*=\s*(.+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SECURE_VAR komutunda sözdizimi hatası", context={"source": "handle_secure_var", "line_no": self.interpreter.program_counter})
        var_name, type_name, initial_value = match.groups()
        if type_name not in self.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", context={"source": "handle_secure_var"})
        value = None
        if initial_value:
            value = self.interpreter.evaluate_expression(initial_value)
        try:
            with self.lock:
                self.interpreter.current_scope()[var_name] = value
                self.interpreter.restricted_vars.add(var_name)
                self.interpreter.object_counter[type_name] += 1
                self.interpreter.object_counter["SECURE_VAR"] += 1
                self.interpreter.object_registry[id(value)] = {
                    "type": type_name,
                    "name": var_name,
                    "atom": str(value)[:100] if value is not None else "null"
                }
            log.debug(f"Güvenli değişken tanımlandı: {var_name} AS {type_name}")
        except Exception as e:
            raise PdsXRuntimeError(f"Güvenli değişken tanımlama hatası: {str(e)}", context={"source": "handle_secure_var"})

    async def handle_connect_iot(self, command: str) -> None:
        """IoT cihazına bağlanma komutunu işler."""
        match = re.match(r"CONNECT_IOT\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CONNECT_IOT komutunda sözdizimi hatası", context={"source": "handle_connect_iot", "line_no": self.interpreter.program_counter})
        broker, client_id, var_name = match.groups()
        try:
            client = mqtt.Client(client_id)
            client.connect(broker, 1883, 60)
            client.loop_start()
            with self.lock:
                self.interpreter.current_scope()[var_name] = client
                self.interpreter.object_counter["IOT_CONNECTION"] += 1
                self.interpreter.object_registry[id(client)] = {
                    "type": "IOT_CLIENT",
                    "name": var_name,
                    "atom": f"{broker}:{client_id}"
                }
            log.debug(f"IoT bağlantısı kuruldu: {broker}, {client_id}")
        except Exception as e:
            raise PdsXNetworkError(f"IoT bağlantı hatası: {str(e)}", context={"source": "handle_connect_iot"})

    async def handle_publish_iot(self, command: str) -> None:
        """IoT cihazına veri yayınlama komutunu işler."""
        match = re.match(r"PUBLISH_IOT\s+(\w+)\s+TO\s+\"([^\"]+)\"\s+WITH\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("PUBLISH_IOT komutunda sözdizimi hatası", context={"source": "handle_publish_iot", "line_no": self.interpreter.program_counter})
        client_var, topic, data_var = match.groups()
        if client_var not in self.interpreter.current_scope() or data_var not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var} veya {data_var}", context={"source": "handle_publish_iot", "line_no": self.interpreter.program_counter})
        client = self.interpreter.current_scope()[client_var]
        data = str(self.interpreter.current_scope()[data_var]).encode()
        try:
            client.publish(topic, data)
            with self.lock:
                self.interpreter.object_counter["PUBLISH_IOT"] += 1
                self.interpreter.object_registry[id(data)] = {
                    "type": "IOT_MESSAGE",
                    "name": topic,
                    "atom": data.decode()[:100]
                }
            log.debug(f"IoT verisi yayınlandı: {topic}")
        except Exception as e:
            raise PdsXNetworkError(f"IoT yayın hatası: {str(e)}", context={"source": "handle_publish_iot"})

    async def handle_subscribe_iot(self, command: str) -> None:
        """IoT cihazından veri aboneliği komutunu işler."""
        match = re.match(r"SUBSCRIBE_IOT\s+(\w+)\s+TO\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SUBSCRIBE_IOT komutunda sözdizimi hatası", context={"source": "handle_subscribe_iot", "line_no": self.interpreter.program_counter})
        client_var, topic, var_name = match.groups()
        if client_var not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var}", context={"source": "handle_subscribe_iot", "line_no": self.interpreter.program_counter})
        client = self.interpreter.current_scope()[client_var]
        try:
            def on_message(client, userdata, msg):
                with self.lock:
                    self.interpreter.current_scope()[var_name] = IotMessage(msg.topic, msg.payload, time.time()).to_dict()
                    self.interpreter.object_counter["IOT_MESSAGE"] += 1
            client.on_message = on_message
            client.subscribe(topic)
            with self.lock:
                self.interpreter.object_counter["SUBSCRIBE_IOT"] += 1
            log.debug(f"IoT aboneliği yapıldı: {topic}")
        except Exception as e:
            raise PdsXNetworkError(f"IoT abonelik hatası: {str(e)}", context={"source": "handle_subscribe_iot"})

    async def handle_sysinfo(self, command: str) -> None:
        """Sistem bilgilerini alır."""
        match = re.match(r"SYSINFO\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SYSINFO komutunda sözdizimi hatası", context={"source": "handle_sysinfo", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        result = self.system_info()
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SYSINFO"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "SYSINFO",
                "name": var_name,
                "atom": json.dumps(result)
            }
        log.debug(f"SYSINFO atandı: {var_name}")

    async def handle_cpuinfo(self, command: str) -> None:
        """CPU bilgilerini alır."""
        match = re.match(r"CPUINFO\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CPUINFO komutunda sözdizimi hatası", context={"source": "handle_cpuinfo", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        result = self.cpu_info()
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["CPUINFO"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "CPUINFO",
                "name": var_name,
                "atom": json.dumps(result)
            }
        log.debug(f"CPUINFO atandı: {var_name}")

    async def handle_diskinfo(self, command: str) -> None:
        """Disk bilgilerini alır."""
        match = re.match(r"DISKINFO\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DISKINFO komutunda sözdizimi hatası", context={"source": "handle_diskinfo", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        result = self.disk_info()
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["DISKINFO"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "DISKINFO",
                "name": var_name,
                "atom": json.dumps(result)
            }
        log.debug(f"DISKINFO atandı: {var_name}")

    async def handle_monitor(self, command: str) -> None:
        """Sistem kaynaklarını izler."""
        match = re.match(r"MONITOR\s+INTERVAL\s+(\d+\.?\d*)\s+DURATION\s+(\d+\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("MONITOR komutunda sözdizimi hatası", context={"source": "handle_monitor", "line_no": self.interpreter.program_counter})
        interval, duration, var_name = map(float, match.groups()[:2]) + [match.group(3)]
        result = await self.monitor_resources(interval, duration)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["MONITOR"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "MONITOR",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"MONITOR atandı: {var_name}")

    async def handle_assert(self, command: str) -> None:
        """Koşul kontrolü yapar."""
        match = re.match(r"ASSERT\s+(.+?)(?:\s*,\s*(.+))?$", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ASSERT komutunda sözdizimi hatası", context={"source": "handle_assert", "line_no": self.interpreter.program_counter})
        condition, message = match.groups()
        message = message or "Koşul sağlanmadı"
        condition_result = self.interpreter.evaluate_expression(condition)
        self.assert_(condition_result, message)
        with self.lock:
            self.interpreter.object_counter["ASSERT"] += 1
        log.debug(f"ASSERT kontrolü: {condition}")

    async def handle_merge(self, command: str) -> None:
        """Koleksiyonları birleştirir."""
        match = re.match(r"MERGE\s+(.+?)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("MERGE komutunda sözdizimi hatası", context={"source": "handle_merge", "line_no": self.interpreter.program_counter})
        collections_str, var_name = match.groups()
        collections = [self.interpreter.evaluate_expression(c.strip()) for c in collections_str.split(",")]
        result = self.merge(*collections)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["MERGE"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "MERGE",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"MERGE atandı: {var_name}")

    async def handle_sort(self, command: str) -> None:
        """Koleksiyonu sıralar."""
        match = re.match(r"SORT\s+(\w+)(?:\s+KEY\s+(\w+))?(?:\s+REVERSE)?\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SORT komutunda sözdizimi hatası", context={"source": "handle_sort", "line_no": self.interpreter.program_counter})
        collection_name, key, var_name = match.groups()
        reverse = "REVERSE" in command.upper()
        collection = self.interpreter.current_scope().get(collection_name)
        result = self.sort(collection, key, reverse)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SORT"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "SORT",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"SORT atandı: {var_name}")

    async def handle_map(self, command: str) -> None:
        """Koleksiyondaki elemanları dönüştürür."""
        match = re.match(r"MAP\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("MAP komutunda sözdizimi hatası", context={"source": "handle_map", "line_no": self.interpreter.program_counter})
        func, collection_name, var_name = match.groups()
        collection = self.interpreter.current_scope().get(collection_name)
        result = self.map_(func, collection)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["MAP"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "MAP",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"MAP atandı: {var_name}")

    async def handle_filter(self, command: str) -> None:
        """Koleksiyondan elemanları filtreler."""
        match = re.match(r"FILTER\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("FILTER komutunda sözdizimi hatası", context={"source": "handle_filter", "line_no": self.interpreter.program_counter})
        func, collection_name, var_name = match.groups()
        collection = self.interpreter.current_scope().get(collection_name)
        result = self.filter_(func, collection)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["FILTER"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "FILTER",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"FILTER atandı: {var_name}")

    async def handle_reduce(self, command: str) -> None:
        """Koleksiyonu birleştirir."""
        match = re.match(r"REDUCE\s+(\w+)\s+(\w+)(?:\s+INITIAL\s+(.+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("REDUCE komutunda sözdizimi hatası", context={"source": "handle_reduce", "line_no": self.interpreter.program_counter})
        func, collection_name, initial, var_name = match.groups()
        collection = self.interpreter.current_scope().get(collection_name)
        initial_value = self.interpreter.evaluate_expression(initial) if initial else None
        result = self.reduce_(func, collection, initial_value)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["REDUCE"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "REDUCE",
                "name": var_name,
                "atom": str(result)
            }
        log.debug(f"REDUCE atandı: {var_name}")

    async def handle_try(self, command: str) -> None:
        """Güvenli komut yürütme."""
        match = re.match(r"TRY\s+(.+?)\s+CATCH\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("TRY komutunda sözdizimi hatası", context={"source": "handle_try", "line_no": self.interpreter.program_counter})
        cmd, var_name = match.groups()
        result = await self.try_catch(cmd, var_name)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["TRY"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "TRY",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"TRY atandı: {var_name}")

    async def handle_trace(self, command: str) -> None:
        """Çağrı yığınını alır."""
        match = re.match(r"TRACE\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("TRACE komutunda sözdizimi hatası", context={"source": "handle_trace", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        result = self.trace()
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["TRACE"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "TRACE",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"TRACE atandı: {var_name}")

    async def handle_datediff(self, command: str) -> None:
        """Tarih farkını hesaplar."""
        match = re.match(r"DATEDIFF\s+\"([^\"]+)\",\s*\"([^\"]+)\"\s*(?:UNIT\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DATEDIFF komutunda sözdizimi hatası", context={"source": "handle_datediff", "line_no": self.interpreter.program_counter})
        date1, date2, unit, var_name = match.groups()
        unit = unit or "days"
        result = self.date_diff(date1, date2, unit)
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["DATEDIFF"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "DATEDIFF",
                "name": var_name,
                "atom": str(result)
            }
        log.debug(f"DATEDIFF atandı: {var_name}")

    async def handle_wait(self, command: str) -> None:
        """Asenkron bekleme yapar."""
        match = re.match(r"WAIT\s+(\d+\.?\d*)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("WAIT komutunda sözdizimi hatası", context={"source": "handle_wait", "line_no": self.interpreter.program_counter})
        seconds = float(match.group(1))
        await self.async_wait(seconds)
        with self.lock:
            self.interpreter.object_counter["WAIT"] += 1
        log.debug(f"WAIT tamamlandı: {seconds} saniye")

    async def handle_publish_sysinfo(self, command: str) -> None:
        """Sistem bilgilerini IoT cihazına yayınlar."""
        match = re.match(r"PUBLISH_SYSINFO\s+(\w+)\s+TO\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("PUBLISH_SYSINFO komutunda sözdizimi hatası", context={"source": "handle_publish_sysinfo", "line_no": self.interpreter.program_counter})
        client_var, topic = match.groups()
        with self.lock:
            if client_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var}", context={"source": "handle_publish_sysinfo", "line_no": self.interpreter.program_counter})
            client = self.interpreter.current_scope()[client_var]
            data = json.dumps(self.system_info()).encode()
            try:
                client.publish(topic, data)
                self.interpreter.object_counter["PUBLISH_SYSINFO"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "PUBLISH_SYSINFO",
                    "name": topic,
                    "atom": command
                }
            except Exception as e:
                raise PdsXRuntimeError(f"Yayın hatası: {str(e)}", context={"source": "handle_publish_sysinfo"})
        log.debug(f"Sistem bilgileri yayınlandı: {topic}")

    async def handle_simple_mode(self, command: str) -> None:
        """v14 uyumlu basit mod etkinleştirir."""
        match = re.match(r"SIMPLE_MODE\s+(ON|OFF)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SIMPLE_MODE komutunda sözdizimi hatası", context={"source": "handle_simple_mode", "line_no": self.interpreter.program_counter})
        mode = match.group(1).upper()
        with self.lock:
            self.interpreter.simple_mode = (mode == "ON")
            self.interpreter.object_counter["SIMPLE_MODE"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "SIMPLE_MODE",
                "name": mode,
                "atom": command
            }
        log.debug(f"Basit mod: {mode}")

    async def handle_system(self, command: str) -> None:
        """v14 uyumlu sistem bilgisi komutu."""
        match = re.match(r"SYSTEM\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SYSTEM komutunda sözdizimi hatası", context={"source": "handle_system", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        result = self.system_info()
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SYSTEM"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "SYSTEM",
                "name": var_name,
                "atom": json.dumps(result)
            }
        log.debug(f"SYSTEM atandı: {var_name}")

    def system_info(self) -> Dict[str, Any]:
        """CPU, bellek ve işletim sistemi bilgilerini döndürür."""
        try:
            info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB cinsinden
                "memory_available": psutil.virtual_memory().available / (1024 ** 3),
                "boot_time": datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
            }
            with self.lock:
                self.interpreter.object_counter["SYSTEM_INFO"] += 1
                self.interpreter.object_registry[id(info)] = {
                    "type": "SYSTEM_INFO",
                    "name": "system_info",
                    "atom": json.dumps(info)
                }
            log.debug(f"Sistem bilgileri alındı: {info}")
            return info
        except Exception as e:
            raise PdsXRuntimeError(f"Sistem bilgisi hatası: {str(e)}", context={"source": "system_info"})

    def cpu_info(self) -> Dict[str, Any]:
        """CPU kullanım ve bilgilerini döndürür."""
        try:
            info = {
                "usage_percent": psutil.cpu_percent(interval=0.1),
                "cores_physical": psutil.cpu_count(logical=False),
                "cores_logical": psutil.cpu_count(logical=True),
                "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "freq_min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None
            }
            with self.lock:
                self.interpreter.object_counter["CPU_INFO"] += 1
                self.interpreter.object_registry[id(info)] = {
                    "type": "CPU_INFO",
                    "name": "cpu_info",
                    "atom": json.dumps(info)
                }
            log.debug(f"CPU bilgileri alındı: {info}")
            return info
        except Exception as e:
            raise PdsXRuntimeError(f"CPU bilgisi hatası: {str(e)}", context={"source": "cpu_info"})

    def disk_info(self) -> Dict[str, Any]:
        """Disk kullanım bilgilerini döndürür."""
        try:
            disk = psutil.disk_usage("/")
            info = {
                "total": disk.total / (1024 ** 3),  # GB cinsinden
                "used": disk.used / (1024 ** 3),
                "free": disk.free / (1024 ** 3),
                "percent": disk.percent
            }
            with self.lock:
                self.interpreter.object_counter["DISK_INFO"] += 1
                self.interpreter.object_registry[id(info)] = {
                    "type": "DISK_INFO",
                    "name": "disk_info",
                    "atom": json.dumps(info)
                }
            log.debug(f"Disk bilgileri alındı: {info}")
            return info
        except Exception as e:
            raise PdsXRuntimeError(f"Disk bilgisi hatası: {str(e)}", context={"source": "disk_info"})

    async def monitor_resources(self, interval: float = 1.0, duration: float = 10.0) -> List[Dict[str, Any]]:
        """Sistem kaynaklarını belirli bir süre izler."""
        try:
            results = []
            start_time = time.time()
            while time.time() - start_time < duration:
                info = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent
                }
                results.append(info)
                await asyncio.sleep(interval)
            with self.lock:
                self.interpreter.object_counter["MONITOR_RESOURCES"] += 1
                self.interpreter.object_registry[id(results)] = {
                    "type": "MONITOR_RESOURCES",
                    "name": "monitor_resources",
                    "atom": json.dumps(results)[:100]
                }
            log.debug(f"Kaynak izleme tamamlandı: {len(results)} kayıt")
            return results
        except Exception as e:
            raise PdsXRuntimeError(f"Kaynak izleme hatası: {str(e)}", context={"source": "monitor_resources"})

    def assert_(self, condition: Any, message: str = "Koşul sağlanmadı") -> None:
        """Koşul kontrolü yapar, başarısız olursa hata fırlatır."""
        try:
            if not condition:
                raise PdsXValueError(message, context={"source": "assert_"})
            with self.lock:
                self.interpreter.object_counter["ASSERT"] += 1
                self.interpreter.object_registry[id(condition)] = {
                    "type": "ASSERT",
                    "name": "assert",
                    "atom": str(condition)
                }
            log.debug(f"Koşul kontrolü başarılı: {condition}")
        except Exception as e:
            raise PdsXValueError(f"Koşul hatası: {str(e)}", context={"source": "assert_"})

    def merge(self, *collections: Any) -> List[Any]:
        """Birden fazla koleksiyonu birleştirir."""
        try:
            result = []
            for collection in collections:
                if isinstance(collection, (list, tuple, set, np.ndarray, Vector, Matrix, Tensor)):
                    result.extend(list(collection))
                elif isinstance(collection, dict):
                    result.extend(list(collection.values()))
                elif isinstance(collection, pd.DataFrame):
                    result.extend(collection.to_dict("records"))
                else:
                    result.append(collection)
            with self.lock:
                self.interpreter.object_counter["MERGE"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "MERGE",
                    "name": "merge",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Koleksiyonlar birleştirildi: {len(result)} eleman")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Birleştirme hatası: {str(e)}", context={"source": "merge"})

    def sort(self, collection: Any, key: Optional[str] = None, reverse: bool = False) -> List[Any]:
        """Koleksiyonu sıralar."""
        try:
            if isinstance(collection, pd.DataFrame):
                result = collection.sort_values(by=key, ascending=not reverse) if key else collection.sort_index(ascending=not reverse)
            elif isinstance(collection, (np.ndarray, Vector, Matrix, Tensor)):
                result = np.sort(collection.to_numpy() if hasattr(collection, "to_numpy") else collection).tolist()
                if reverse:
                    result = result[::-1]
            else:
                result = sorted(collection, key=lambda x: x[key] if key and isinstance(x, dict) else x, reverse=reverse)
            with self.lock:
                self.interpreter.object_counter["SORT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "SORT",
                    "name": "sort",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Koleksiyon sıralandı: {len(result)} eleman")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Sıralama hatası: {str(e)}", context={"source": "sort"})

    def map_(self, func: str, collection: Any) -> List[Any]:
        """Koleksiyondaki elemanları dönüştürür."""
        try:
            if func not in self.function_table:
                raise PdsXRuntimeError(f"Fonksiyon bulunamadı: {func}", context={"source": "map_"})
            result = [self.function_table[func](item) for item in collection]
            with self.lock:
                self.interpreter.object_counter["MAP"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "MAP",
                    "name": "map",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Map işlemi yapıldı: {func}, {len(result)} eleman")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Map hatası: {str(e)}", context={"source": "map_"})

    def filter_(self, func: str, collection: Any) -> List[Any]:
        """Koleksiyondan elemanları filtreler."""
        try:
            if func not in self.function_table:
                raise PdsXRuntimeError(f"Fonksiyon bulunamadı: {func}", context={"source": "filter_"})
            result = [item for item in collection if self.function_table[func](item)]
            with self.lock:
                self.interpreter.object_counter["FILTER"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "FILTER",
                    "name": "filter",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Filtreleme yapıldı: {func}, {len(result)} eleman")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Filtreleme hatası: {str(e)}", context={"source": "filter_"})

    def reduce_(self, func: str, collection: Any, initial: Any = None) -> Any:
        """Koleksiyonu birleştirir."""
        try:
            if func not in self.function_table:
                raise PdsXRuntimeError(f"Fonksiyon bulunamadı: {func}", context={"source": "reduce_"})
            result = reduce(self.function_table[func], collection, initial) if initial is not None else reduce(self.function_table[func], collection)
            with self.lock:
                self.interpreter.object_counter["REDUCE"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "REDUCE",
                    "name": "reduce",
                    "atom": str(result)
                }
            log.debug(f"Reduce işlemi yapıldı: {func}, Sonuç: {result}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Reduce hatası: {str(e)}", context={"source": "reduce_"})

    async def try_catch(self, command: str, var_name: str) -> Dict[str, Any]:
        """Güvenli fonksiyon çağrısı yapar, hataları yakalar."""
        try:
            await self.interpreter.execute_command(command)
            result = {"success": True, "result": self.interpreter.current_scope().get(var_name, None), "error": None}
        except Exception as e:
            result = {"success": False, "result": None, "error": str(e)}
        with self.lock:
            self.interpreter.object_counter["TRY_CATCH"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "TRY_CATCH",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"TRY_CATCH sonucu: {result}")
        return result

    def trace(self) -> List[str]:
        """Çağrı yığınını döndürür."""
        try:
            stack = traceback.format_stack()[:-1]  # Son çağrıyı hariç tut
            result = [line.strip() for line in stack]
            with self.lock:
                self.interpreter.object_counter["TRACE"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "TRACE",
                    "name": "trace",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Çağrı yığını alındı: {len(result)} satır")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Trace hatası: {str(e)}", context={"source": "trace"})

    def date_diff(self, date1: str, date2: str, unit: str = "days") -> float:
        """İki tarih arası farkı hesaplar."""
        try:
            d1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
            d2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
            delta = d2 - d1
            if unit.lower() == "days":
                result = delta.total_seconds() / (24 * 3600)
            elif unit.lower() == "hours":
                result = delta.total_seconds() / 3600
            elif unit.lower() == "minutes":
                result = delta.total_seconds() / 60
            elif unit.lower() == "seconds":
                result = delta.total_seconds()
            else:
                raise PdsXValueError(f"Geçersiz birim: {unit}", context={"source": "date_diff"})
            with self.lock:
                self.interpreter.object_counter["DATE_DIFF"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "DATE_DIFF",
                    "name": "date_diff",
                    "atom": str(result)
                }
            log.debug(f"Tarih farkı hesaplandı: {result} {unit}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Tarih farkı hatası: {str(e)}", context={"source": "date_diff"})

    async def async_wait(self, seconds: float) -> None:
        """Asenkron olarak belirtilen süre bekler."""
        try:
            await asyncio.sleep(seconds)
            with self.lock:
                self.interpreter.object_counter["ASYNC_WAIT"] += 1
                self.interpreter.object_registry[id(seconds)] = {
                    "type": "ASYNC_WAIT",
                    "name": "async_wait",
                    "atom": str(seconds)
                }
            log.debug(f"Asenkron bekleme: {seconds} saniye")
        except Exception as e:
            raise PdsXRuntimeError(f"Asenkron bekleme hatası: {str(e)}", context={"source": "async_wait"})

    def check_auth(self, user: str, role: str) -> bool:
        """Kullanıcı yetkisini kontrol eder."""
        try:
            auth_users = self.interpreter.current_scope().get("_AUTH_USERS", {})
            auth = user in auth_users and role in auth_users.get(user, [])
            with self.lock:
                self.interpreter.object_counter["CHECK_AUTH"] += 1
                self.interpreter.object_registry[id(auth)] = {
                    "type": "CHECK_AUTH",
                    "name": f"{user}_{role}",
                    "atom": str(auth)
                }
            log.debug(f"Yetki kontrolü: {user}, {role} -> {auth}")
            return auth
        except Exception as e:
            raise PdsXRuntimeError(f"Yetki kontrol hatası: {str(e)}", context={"source": "check_auth"})

    def neural_data_processing(self, data: Union[List[float], NeuralTensor]) -> NeuralTensor:
        """Nöral ağlarla veri işleme (LSTM modeli)."""
        try:
            d = data.data if isinstance(data, NeuralTensor) else np.array(data, dtype=np.float64)
            if d.size < 10:
                raise PdsXValueError("Veri boyutu en az 10 olmalı", context={"source": "neural_data_processing"})
            d = d.reshape(-1, 10, 1)
            processed = self.neural_model.predict(d, verbose=0)
            result = NeuralTensor(processed.tolist())
            with self.lock:
                self.interpreter.object_counter["NEURAL_PROCESS"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "NEURAL_PROCESS",
                    "name": "neural_output",
                    "atom": str(result)
                }
            log.debug(f"Nöral veri işleme: {processed.shape}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Nöral işleme hatası: {str(e)}", context={"source": "neural_data_processing"})

    def quantum_correlation_analysis(self, data1: List[float], data2: List[float]) -> float:
        """Kuantum korelasyon analizi yapar."""
        try:
            d1 = np.array(data1)
            d2 = np.array(data2)
            if len(d1) != len(d2):
                raise PdsXValueError("Veri uzunlukları eşit olmalı", context={"source": "quantum_correlation_analysis"})
            corr = np.corrcoef(d1, d2)[0, 1]
            with self.lock:
                self.interpreter.object_counter["QUANTUM_CORR"] += 1
                self.interpreter.object_registry[id(corr)] = {
                    "type": "QUANTUM_CORR",
                    "name": "quantum_corr",
                    "atom": str(corr)
                }
            log.debug(f"Kuantum korelasyon: {corr}")
            return float(corr)
        except Exception as e:
            raise PdsXRuntimeError(f"Kuantum korelasyon hatası: {str(e)}", context={"source": "quantum_correlation_analysis"})

    def chaos_pattern_detection(self, data: List[float]) -> Dict[str, Any]:
        """Kaos örüntülerini tespit eder."""
        try:
            d = np.array(data)
            fft_result = fft.fft(d)
            power_spectrum = np.abs(fft_result) ** 2
            result = {
                "dominant_freq": float(np.argmax(power_spectrum)),
                "power_spectrum": power_spectrum.tolist()
            }
            with self.lock:
                self.interpreter.object_counter["CHAOS_DETECT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "CHAOS_DETECT",
                    "name": "chaos_pattern",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Kaos örüntüsü tespit edildi: {result['dominant_freq']}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Kaos tespit hatası: {str(e)}", context={"source": "chaos_pattern_detection"})

    def genetic_optimization_engine(self, fitness_func: str, population_size: int = 100, generations: int = 50) -> Dict[str, Any]:
        """Genetik algoritma ile optimizasyon yapar."""
        try:
            if fitness_func not in self.function_table:
                raise PdsXRuntimeError(f"Fonksiyon bulunamadı: {fitness_func}", context={"source": "genetic_optimization_engine"})
            population = np.random.rand(population_size, 10)
            for _ in range(generations):
                fitness = np.array([self.function_table[fitness_func](ind) for ind in population])
                best_idx = np.argsort(fitness)[-int(population_size * 0.2):]
                parents = population[best_idx]
                offspring = parents[np.random.choice(len(parents), size=population_size)]
                population = offspring + np.random.normal(0, 0.1, offspring.shape)
            best_individual = population[np.argmax(fitness)]
            result = {"best_solution": best_individual.tolist(), "best_fitness": float(np.max(fitness))}
            with self.lock:
                self.interpreter.object_counter["GENETIC_OPT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "GENETIC_OPT",
                    "name": "genetic_opt",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Genetik optimizasyon tamamlandı: {result['best_fitness']}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Genetik optimizasyon hatası: {str(e)}", context={"source": "genetic_optimization_engine"})

    def blockchain_integrity_check(self, ledger: BlockchainLedger) -> bool:
        """Blok zinciri bütünlüğünü kontrol eder."""
        try:
            prev_hash = ""
            for block_hash, block in ledger.ledger.items():
                if block["prev_hash"] != prev_hash:
                    return False
                block_str = json.dumps(block["data"], sort_keys=True)
                if sha256(block_str.encode()).hexdigest() != block_hash:
                    return False
                prev_hash = block_hash
            with self.lock:
                self.interpreter.object_counter["BLOCKCHAIN_CHECK"] += 1
                self.interpreter.object_registry[id(True)] = {
                    "type": "BLOCKCHAIN_CHECK",
                    "name": "blockchain_check",
                    "atom": "True"
                }
            log.debug("Blok zinciri bütünlüğü doğrulandı")
            return True
        except Exception as e:
            raise PdsXRuntimeError(f"Blok zinciri kontrol hatası: {str(e)}", context={"source": "blockchain_integrity_check"})

    def hash_data(self, data: str) -> str:
        """Veriyi hash'ler."""
        try:
            data_bytes = str(data).encode()
            hash_value = sha256(data_bytes).hexdigest()
            with self.lock:
                self.interpreter.object_counter["HASH"] += 1
                self.interpreter.object_registry[id(hash_value)] = {
                    "type": "HASH",
                    "name": "hash",
                    "atom": hash_value[:100]
                }
            log.debug(f"Veri hash'lendi: {hash_value}")
            return hash_value
        except Exception as e:
            raise PdsXRuntimeError(f"Hash hatası: {str(e)}", context={"source": "hash_data"})

    def check_iot_status(self, client_var: str) -> Dict[str, Any]:
        """IoT cihazının durumunu kontrol eder."""
        try:
            if client_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var}", context={"source": "check_iot_status"})
            client = self.interpreter.current_scope()[client_var]
            status = {
                "connected": client.is_connected(),
                "last_message_time": time.time()
            }
            with self.lock:
                self.interpreter.object_counter["CHECK_IOT_STATUS"] += 1
                self.interpreter.object_registry[id(status)] = {
                    "type": "CHECK_IOT_STATUS",
                    "name": "iot_status",
                    "atom": json.dumps(status)
                }
            log.debug(f"IoT durumu kontrol edildi: {status}")
            return status
        except Exception as e:
            raise PdsXRuntimeError(f"IoT durum kontrol hatası: {str(e)}", context={"source": "check_iot_status"})

    def get_iot_message(self, topic: str) -> Optional[IotMessage]:
        """IoT mesajını alır."""
        try:
            msg = self.interpreter.current_scope().get(f"_IOT_MSG_{topic}")
            if msg:
                result = IotMessage(msg["topic"], msg["payload"].encode(), msg["timestamp"])
                with self.lock:
                    self.interpreter.object_counter["GET_IOT_MESSAGE"] += 1
                    self.interpreter.object_registry[id(result)] = {
                        "type": "IOT_MESSAGE",
                        "name": "iot_message",
                        "atom": json.dumps(result.to_dict())[:100]
                    }
                log.debug(f"IoT mesajı alındı: {topic}")
                return result
            return None
        except Exception as e:
            raise PdsXRuntimeError(f"IoT mesaj alma hatası: {str(e)}", context={"source": "get_iot_message"})

    def pdf_read_text(self, file_path: str) -> str:
        """PDF dosyasından metin çıkarır."""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            with self.lock:
                self.interpreter.object_counter["PDF_READ_TEXT"] += 1
                self.interpreter.object_registry[id(text)] = {
                    "type": "PDF_READ_TEXT",
                    "name": "pdf_text",
                    "atom": text[:100]
                }
            log.debug(f"PDF metni çıkarıldı: {file_path}")
            return text
        except Exception as e:
            raise PdsXIOException(f"PDF okuma hatası: {str(e)}", context={"source": "pdf_read_text"})

    def pdf_extract_tables(self, file_path: str) -> List[List[Any]]:
        """PDF dosyasından tabloları çıkarır."""
        try:
            with pdfplumber.open(file_path) as pdf:
                tables = []
                for page in pdf.pages:
                    extracted = page.extract_tables()
                    if extracted:
                        tables.extend(extracted)
            with self.lock:
                self.interpreter.object_counter["PDF_EXTRACT_TABLES"] += 1
                self.interpreter.object_registry[id(tables)] = {
                    "type": "PDF_EXTRACT_TABLES",
                    "name": "pdf_tables",
                    "atom": json.dumps(tables)[:100]
                }
            log.debug(f"PDF tabloları çıkarıldı: {file_path}")
            return tables
        except Exception as e:
            raise PdsXIOException(f"PDF tablo çıkarma hatası: {str(e)}", context={"source": "pdf_extract_tables"})

    def web_get(self, url: str) -> Dict[str, Any]:
        """Web sitesinden veri çeker."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = {
                "status": response.status_code,
                "content": response.text,
                "headers": dict(response.headers)
            }
            with self.lock:
                self.interpreter.object_counter["WEB_GET"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "WEB_GET",
                    "name": "web_data",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Web verisi alındı: {url}")
            return result
        except Exception as e:
            raise PdsXNetworkError(f"Web veri alma hatası: {str(e)}", context={"source": "web_get"})

    def system(self, command: str) -> str:
        """Sistem komutu çalıştırır."""
        try:
            result = subprocess.check_output(command, shell=True, text=True)
            with self.lock:
                self.interpreter.object_counter["SYSTEM"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "SYSTEM",
                    "name": "system_output",
                    "atom": result[:100]
                }
            log.debug(f"Sistem komutu çalıştırıldı: {command}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Sistem komut hatası: {str(e)}", context={"source": "system"})

# Modül ihracı
__pdsX_exports__ = {
    "classes": {
        "Float128": Float128,
        "Float256": Float256,
        "Float512": Float512,
        "Skaler": Skaler,
        "Vector": Vector,
        "Matrix": Matrix,
        "Tensor": Tensor,
        "QuantumState": QuantumState,
        "HoloData": HoloData,
        "ChaosField": ChaosField,
        "NeuralTensor": NeuralTensor,
        "BlockchainLedger": BlockchainLedger,
        "SecureData": SecureData,
        "Signature": Signature,
        "IotMessage": IotMessage,
        "AsyncCore": AsyncCore,
        "Core2": Core2,
        "CoreManager": CoreManager
    },
    "functions": {
        "hash_data": hash_data,
        "check_auth": CoreManager.check_auth,
        "system_info": CoreManager.system_info,
        "cpu_info": CoreManager.cpu_info,
        "disk_info": CoreManager.disk_info,
        "monitor_resources": CoreManager.monitor_resources,
        "assert_": CoreManager.assert_,
        "merge": CoreManager.merge,
        "sort": CoreManager.sort,
        "map_": CoreManager.map_,
        "filter_": CoreManager.filter_,
        "reduce_": CoreManager.reduce_,
        "try_catch": CoreManager.try_catch,
        "trace": CoreManager.trace,
        "date_diff": CoreManager.date_diff,
        "async_wait": CoreManager.async_wait,
        "quantum_correlation_analysis": CoreManager.quantum_correlation_analysis,
        "chaos_pattern_detection": CoreManager.chaos_pattern_detection,
        "neural_data_processing": CoreManager.neural_data_processing,
        "genetic_optimization_engine": CoreManager.genetic_optimization_engine,
        "blockchain_integrity_check": CoreManager.blockchain_integrity_check,
        "pdf_read_text": CoreManager.pdf_read_text,
        "pdf_extract_tables": CoreManager.pdf_extract_tables,
        "web_get": CoreManager.web_get,
        "system": CoreManager.system,
        "check_iot_status": CoreManager.check_iot_status,
        "get_iot_message": CoreManager.get_iot_message
    },
    "variables": {
        "core_version": "1.5.0",
        "supported_features": ["async", "blockchain", "ai", "mqtt", "crypto"]
    }
}