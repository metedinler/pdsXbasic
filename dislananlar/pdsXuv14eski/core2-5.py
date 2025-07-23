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
import aiofiles
import random
import threading
import ctypes
import requests
import pdfplumber
import aiohttp
import decimal
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import stats, linalg, fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from hashlib import sha256
import psutil
import graphviz
from pdsx_exception2 import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError, PdsXTypeError,
    PdsXValueError, PdsXIOException, PdsXNetworkError
)
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import paho.mqtt.client as mqtt # MQTT desteği için
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
        super().__init__(value, prec=34)

class Float256(decimal.Decimal):
    def __init__(self, value):
        super().__init__(value, prec=68)

class Float512(decimal.Decimal):
    def __init__(self, value):
        super().__init__(value, prec=136)

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

# Deneysel Veri Yapıları
class QuantumState:
    def __init__(self, amplitudes: List[complex]):
        self.amplitudes = [complex(x) for x in amplitudes]
        norm = sum(abs(a) ** 2 for a in self.amplitudes) ** 0.5
        self.amplitudes = [a / norm for a in self.amplitudes]
    
    def measure(self):
        """Kuantum ölçüm simülasyonu."""
        probs = [abs(a) ** 2 for a in self.amplitudes]
        return np.random.choice(len(self.amplitudes), p=probs)
    
    def to_numpy(self):
        return np.array(self.amplitudes, dtype=np.complex128)

class HoloData:
    def __init__(self, data: bytes):
        self.data = data
        self.compressed = self._compress()
    
    def _compress(self):
        """FFT tabanlı holografik sıkıştırma."""
        data_array = np.frombuffer(self.data, dtype=np.uint8)
        fft_data = fft.fft(data_array)
        return fft_data.tobytes()

class ChaosField:
    def __init__(self, data: List[float]):
        self.data = data
    
    def simulate(self, steps: int = 1000):
        """Lorenz sistemi simülasyonu."""
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
    def __init__(self, ledger: Dict):
        self.ledger = ledger
    
    def add_block(self, data: Dict):
        """Yeni blok ekler."""
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
        from cryptography.fernet import Fernet
        f = Fernet(key)
        return f.encrypt(data)
    
    def decrypt(self, key: bytes) -> bytes:
        from cryptography.fernet import Fernet
        f = Fernet(key)
        return f.decrypt(self.data)

class Signature:
    def __init__(self, data: bytes, private_key: bytes):
        self.signature = self._sign(data, private_key)
    
    def _sign(self, data: bytes, private_key: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        priv = serialization.load_pem_private_key(private_key, password=None)
        return priv.sign(data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    
    def verify(self, data: bytes, public_key: bytes) -> bool:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        pub = serialization.load_pem_public_key(public_key)
        try:
            pub.verify(self.signature, data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
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

class CoreManager:
    """PDS-X BASIC v15 çekirdek işlemler sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.metadata: Dict = {
            "core2": {
                "version": "1.5.0",
                "dependencies": [
                    "numpy", "pandas", "scipy", "tensorflow", "pdfplumber",
                    "requests", "aiohttp", "graphviz", "aiofiles", "psutil", "decimal"
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
            "SUBSCRIBE_IOT": self.handle_subscribe_iot
        }
        self.function_table: Dict[str, Callable] = {
            "MID$": lambda s, start, length: s[start-1:start-1+length],
            "LEN": len, "RND": random.random, "ABS": abs, "INT": int,
            "LEFT$": lambda s, n: s[:n], "RIGHT$": lambda s, n: s[-n:],
            "LTRIM$": lambda s: s.lstrip(), "RTRIM$": lambda s: s.rstrip(),
            "STRING$": lambda n, c: c * n, "SPACE$": lambda n: " " * n,
            "INSTR": lambda start, s, sub: s.find(sub, start-1) + 1,
            "UCASE$": lambda s: s.upper(), "LCASE$": lambda s: s.lower(),
            "STR$": lambda n: str(n), "SQR": lambda x: float(np.sqrt(x)),
            "SIN": lambda x: float(np.sin(x)), "COS": lambda x: float(np.cos(x)),
            "TAN": lambda x: float(np.tan(x)), "LOG": lambda x: float(np.log(x)),
            "EXP": lambda x: float(np.exp(x)), "ATN": lambda x: float(np.arctan(x)),
            "FIX": lambda x: int(x), "ROUND": lambda x, n=0: round(x, n),
            "SGN": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
            "MOD": lambda x, y: x % y, "MIN": lambda *args: min(args),
            "MAX": lambda *args: max(args), "TIMER": lambda: time.time(),
            "DATE$": lambda: time.strftime("%m-%d-%Y"),
            "TIME$": lambda: time.strftime("%H:%M:%S"),
            "INKEY$": lambda: input()[:1], "ENVIRON$": lambda var: os.environ.get(var, ""),
            "COMMAND$": lambda: " ".join(sys.argv[1:]),
            "CSRLIN": lambda: 1, "POS": lambda x: 1, "VAL": lambda s: float(s) if s.replace(".", "").isdigit() else 0,
            "ASC": lambda c: ord(c[0]),
            "MEAN": lambda x: float(np.mean(x)), "MEDIAN": lambda x: float(np.median(x)),
            "MODE": lambda x: float(stats.mode(x)[0][0]),
            "STD": lambda x: float(np.std(x)), "VAR": lambda x: float(np.var(x)),
            "SUM": lambda x: float(np.sum(x)), "PROD": lambda x: float(np.prod(x)),
            "PERCENTILE": lambda x, p: float(np.percentile(x, p)),
            "QUANTILE": lambda x, q: float(np.quantile(x, q)),
            "CORR": lambda x, y: float(np.corrcoef(x, y)[0, 1]), "COV": lambda x, y: float(np.cov(x, y)),
            "DESCRIBE": lambda df: df.describe(), "GROUPBY": lambda df, col: df.groupby(col),
            "FILTER": lambda df, cond: df.query(cond), "SORT": lambda df, col: df.sort_values(col),
            "HEAD": lambda df, n=5: df.head(n), "TAIL": lambda df, n=5: df.tail(n),
            "MERGE": lambda df1, df2, on: pd.merge(df1, df2, on=on),
            "TTEST": lambda sample1, sample2: stats.ttest_ind(sample1, sample2),
            "CHISQUARE": lambda observed: stats.chisquare(observed),
            "ANOVA": lambda *groups: stats.f_oneway(*groups),
            "REGRESS": lambda x, y: stats.linregress(x, y),
            "Np": lambda func, *args, **kwargs: getattr(np, func)(*args, **kwargs),
            "pd": lambda func, *args, **kwargs: getattr(pd, func)(*args, **kwargs),
            "sc": lambda func, *args, **kwargs: getattr(stats, func)(*args, **kwargs),
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
            "GET_IOT_MESSAGE": self.get_iot_message
        }
        self._init_neural_model()

    def _init_neural_model(self) -> None:
        """Nöral veri işleme için model başlatır."""
        self.neural_model = Sequential([
            LSTM(128, input_shape=(10, 1), return_sequences=True),
            LSTM(64),
            Dense(1, activation="sigmoid")
        ])
        self.neural_model.compile(optimizer="adam", loss="mse")

    async def handle_let(self, command: str) -> None:
        """Değişkene değer atar (güçlendirilmiş)."""
        match = re.match(r"(?:LET\s+)?((?:\w+\s*(?:,\s*\w+)*))\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LET/Atama komutunda sözdizimi hatası", context={"source": "handle_let", "line_no": self.interpreter.program_counter})
        var_names, expr = match.groups()
        try:
            values = self.interpreter.evaluate_expression(expr)
            var_names = [v.strip() for v in var_names.split(",")]
            with self.lock:
                if isinstance(values, (list, tuple, np.ndarray, pd.Series, Vector)):
                    if len(var_names) != len(values):
                        raise PdsXValueError(f"Değişken sayısı ({len(var_names)}) ile değer sayısı ({len(values)}) uyuşmuyor", context={"source": "handle_let"})
                    for var, val in zip(var_names, values):
                        self.interpreter.current_scope()[var] = val
                        self.interpreter.object_counter["VARIABLE"] += 1
                        self.interpreter.object_registry[id(val)] = {"type": "VARIABLE", "name": var, "atom": str(val)}
                elif isinstance(values, dict):
                    for var in var_names:
                        self.interpreter.current_scope()[var] = values.get(var, None)
                        self.interpreter.object_counter["VARIABLE"] += 1
                        self.interpreter.object_registry[id(values)] = {"type": "VARIABLE", "name": var, "atom": json.dumps(values)}
                elif isinstance(values, (pd.DataFrame, Matrix, Tensor, QuantumState, HoloData, ChaosField, NeuralTensor, BlockchainLedger)):
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
                self.interpreter.object_registry[id(command)] = {"type": "LET", "name": ", ".join(var_names), "atom": command}
            log.debug(f"Değişkenler atandı: {var_names} = {values}")
        except Exception as e:
            raise PdsXRuntimeError(f"Atama hatası: {str(e)}", context={"source": "handle_let", "line_no": self.interpreter.program_counter})

    async def handle_if(self, command: str) -> Optional[int]:
        """Koşullu yürütme."""
        match = re.match(r"IF\s+(.+)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+?))?(?:\s+END\s+IF)?$", command, re.IGNORECASE | re.DOTALL)
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
                self.interpreter.object_registry[id(command)] = {"type": "IF", "name": "IF", "atom": command}
        except Exception as e:
            raise PdsXRuntimeError(f"IF koşul değerlendirme hatası: {str(e)}", context={"source": "handle_if", "line_no": self.interpreter.program_counter})
        return None

    async def handle_for(self, command: str) -> Optional[int]:
        """Sayısal döngü (FOR ... NEXT ve FOR ... END FOR)."""
        end_pattern = r"END\s+FOR|NEXT"
        match = re.match(r"FOR\s+(\w+)\s*=\s*([-]?\d+\.?\d*)\s+TO\s+([-]?\d+\.?\d*)(?:\s+STEP\s+([-]?\d+\.?\d*))?\s+(.+?)\s+({})".format(end_pattern), command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("FOR komutunda sözdizimi hatası", context={"source": "handle_for", "line_no": self.interpreter.program_counter})
        var_name, start, end, step, body, end_token = match.groups()
        start, end = float(start), float(end)
        step = float(step) if step else 1.0
        if step == 0:
            raise PdsXValueError("Adım sıfır olamaz", context={"source": "handle_for", "line_no": self.interpreter.program_counter})
        with self.lock:
            self.interpreter.current_scope()[var_name] = start
            self.interpreter.loop_stack.append({"type": "FOR", "var": var_name, "index": 0, "history": [start]})
            self.interpreter.object_counter["FOR"] += 1
            self.interpreter.object_registry[id(command)] = {"type": "FOR", "name": var_name, "atom": command}
        try:
            current = start
            index = 0
            while ((step > 0 and current <= end) or (step < 0 and current >= end)):
                self.interpreter.current_scope()[var_name] = current
                await self.interpreter.execute_command(body)
                if "EXIT FOR" in body.upper():
                    break
                if "PREV" in body.upper():
                    index = max(0, index - 1)
                    current = self.interpreter.loop_stack[-1]["history"][index]
                    continue
                current += step
                index += 1
                with self.lock:
                    self.interpreter.loop_stack[-1]["index"] = index
                    self.interpreter.loop_stack[-1]["history"].append(current)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR döngü hatası: {str(e)}", context={"source": "handle_for", "line_no": self.interpreter.program_counter})
        finally:
            with self.lock:
                self.interpreter.loop_stack.pop()
        return None

    async def handle_foreach(self, command: str) -> Optional[int]:
        """Koleksiyon döngüsü (FOR EACH ... NEXT/END FOR)."""
        end_pattern = r"END\s+FOR|NEXT"
        match = re.match(r"FOR\s+EACH\s+(\w+)\s+IN\s+(\w+)\s+(.+?)\s+({})".format(end_pattern), command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("FOR EACH komutunda sözdizimi hatası", context={"source": "handle_foreach", "line_no": self.interpreter.program_counter})
        var_name, collection_name, body, end_token = match.groups()
        collection = self.interpreter.current_scope().get(collection_name)
        if not isinstance(collection, (list, dict, set, tuple, pd.DataFrame, np.ndarray, Vector, Matrix, Tensor)):
            raise PdsXTypeError(f"Geçersiz koleksiyon: {collection_name}", context={"source": "handle_foreach", "line_no": self.interpreter.program_counter})
        with self.lock:
            self.interpreter.loop_stack.append({"type": "FOREACH", "var": var_name, "index": 0, "history": []})
            self.interpreter.object_counter["FOREACH"] += 1
            self.interpreter.object_registry[id(command)] = {"type": "FOREACH", "name": var_name, "atom": command}
        try:
            items = list(collection.items() if isinstance(collection, dict) else collection)
            if isinstance(collection, pd.DataFrame):
                items = [row.to_dict() for _, row in collection.iterrows()]
            elif isinstance(collection, (np.ndarray, Vector, Matrix, Tensor)):
                items = collection.tolist() if isinstance(collection, np.ndarray) else collection.data
            index = 0
            while index < len(items):
                item = items[index]
                self.interpreter.current_scope()[var_name] = item if not isinstance(collection, dict) else item[1]
                await self.interpreter.execute_command(body)
                if "EXIT FOR" in body.upper():
                    break
                if "PREV" in body.upper():
                    index = max(0, index - 1)
                    continue
                index += 1
                with self.lock:
                    self.interpreter.loop_stack[-1]["index"] = index
                    self.interpreter.loop_stack[-1]["history"].append(item)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR EACH döngü hatası: {str(e)}", context={"source": "handle_foreach", "line_no": self.interpreter.program_counter})
        finally:
            with self.lock:
                self.interpreter.loop_stack.pop()
        return None

    async def handle_while(self, command: str) -> Optional[int]:
        """Koşullu döngü."""
        match = re.match(r"WHILE\s+(.+)\s+(.+?)\s+END\s+WHILE", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("WHILE komutunda sözdizimi hatası", context={"source": "handle_while", "line_no": self.interpreter.program_counter})
        condition, body = match.groups()
        with self.lock:
            self.interpreter.loop_stack.append({"type": "WHILE"})
            self.interpreter.object_counter["WHILE"] += 1
            self.interpreter.object_registry[id(command)] = {"type": "WHILE", "name": "WHILE", "atom": command}
        try:
            while self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(body)
                if "EXIT WHILE" in body.upper():
                    break
        except Exception as e:
            raise PdsXRuntimeError(f"WHILE döngü hatası: {str(e)}", context={"source": "handle_while", "line_no": self.interpreter.program_counter})
        finally:
            with self.lock:
                self.interpreter.loop_stack.pop()
        return None

    async def handle_dim(self, command: str) -> None:
        """Değişken tanımlar (tek satırlık ve kapanışlı yazım)."""
        match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)(?:\s+(.+?))?(?:\s*\*\s*(\d+))?$", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DIM komutunda sözdizimi hatası", context={"source": "handle_dim", "line_no": self.interpreter.program_counter})
        var_name, type_name, initial_value, size = match.groups()
        if type_name not in self.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", context={"source": "handle_dim", "line_no": self.interpreter.program_counter})
        value = None
        try:
            if initial_value:
                value = self.interpreter.evaluate_expression(initial_value)
            elif type_name in ("LIST", "DICT", "SET", "STACK", "QUEUE"):
                value = [] if type_name == "LIST" else {} if type_name == "DICT" else set() if type_name == "SET" else deque()
            elif type_name == "ARRAY":
                size = int(size) if size else 10
                value = np.zeros(size, dtype=np.float64)
            elif type_name == "DATAFRAME":
                value = pd.DataFrame()
            elif type_name == "SKALER":
                value = Skaler(0.0)
            elif type_name == "VECTOR":
                size = int(size) if size else 10
                value = Vector([0.0] * size)
            elif type_name == "MATRIX":
                size = int(size) if size else 10
                value = Matrix([[0.0] * size for _ in range(size)])
            elif type_name == "TENSOR":
                size = int(size) if size else 10
                value = Tensor([[[0.0] * size for _ in range(size)] for _ in range(size)])
            elif type_name == "QUANTUM_STATE":
                size = int(size) if size else 2
                value = QuantumState([1.0 / (size ** 0.5)] * size)
            elif type_name == "HOLO_DATA":
                value = HoloData(b"")
            elif type_name == "CHAOS_FIELD":
                value = ChaosField([1.0, 1.0, 1.0])
            elif type_name == "NEURAL_TENSOR":
                value = NeuralTensor([])
            elif type_name == "BLOCKCHAIN_LEDGER":
                value = BlockchainLedger({})
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
                self.interpreter.object_registry[id(command)] = {
                    "type": "DIM",
                    "name": var_name,
                    "atom": command
                }
            log.debug(f"Değişken tanımlandı: {var_name} AS {type_name} = {value}")
        except Exception as e:
            raise PdsXRuntimeError(f"DIM tanımlama hatası: {str(e)}", context={"source": "handle_dim", "line_no": self.interpreter.program_counter})

    async def handle_end(self, command: str) -> None:
        """Blok, veri yapısı veya program sonlandırır."""
        match = re.match(r"END(?:\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("END komutunda sözdizimi hatası", context={"source": "handle_end", "line_no": self.interpreter.program_counter})
        block_type = match.group(1).upper() if match.group(1) else None
        with self.lock:
            if block_type:
                if block_type not in self.data_types and block_type not in ("FOR", "WHILE", "IF", "CLASS", "YAPI", "SELECT"):
                    raise PdsXSyntaxError(f"Geçersiz END tipi: {block_type}", context={"source": "handle_end", "line_no": self.interpreter.program_counter})
                if block_type in ("FOR", "WHILE"):
                    if not self.interpreter.loop_stack or self.interpreter.loop_stack[-1]["type"] != block_type:
                        raise PdsXRuntimeError(f"Kapatılacak {block_type} bloğu yok", context={"source": "handle_end", "line_no": self.interpreter.program_counter})
                    self.interpreter.loop_stack.pop()
                elif block_type in self.data_types:
                    # Veri yapısı kapanışı
                    pass  # Değer atama DIM ile yapıldı, END sadece kapanış
            else:
                # Program sonu
                self.interpreter.running = False
            self.interpreter.object_counter["END"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "END",
                "name": block_type or "PROGRAM",
                "atom": command
            }
        log.debug(f"Blok/Program sonlandırıldı: {block_type or 'PROGRAM'}")

    async def handle_class(self, command: str) -> None:
        """Sınıf tanımlar."""
        match = re.match(r"CLASS\s+(\w+)\s+(.+?)\s+END\s+CLASS", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("CLASS komutunda sözdizimi hatası", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
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
                    self.interpreter.object_registry[id(sub_body)] = {
                        "type": "SUB",
                        "name": sub_name,
                        "atom": sub_body
                    }
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
                    class_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("PROP "):
                prop_match = re.match(r"PROP\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if prop_match:
                    prop_name, prop_type = prop_match.groups()
                    if prop_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
                    class_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"PROP tanımı hatalı: {line}", context={"source": "handle_class", "line_no": self.interpreter.program_counter})
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[class_name] = class_def
            self.interpreter.object_counter["CLASS"] += 1
            self.interpreter.object_registry[id(class_def)] = {
                "type": "CLASS",
                "name": class_name,
                "atom": class_name
            }
        log.debug(f"Sınıf tanımlandı: {class_name}")

    async def handle_yapi(self, command: str) -> None:
        """Nesne tabanlı sınıf oluşturucu (YAPI)."""
        match = re.match(r"YAPI\s+(\w+)\s+(.+?)\s+END\s+YAPI", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("YAPI komutunda sözdizimi hatası", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
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
                    self.interpreter.object_registry[id(sub_body)] = {
                        "type": "SUB",
                        "name": sub_name,
                        "atom": sub_body
                    }
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("FUNC "):
                func_match = re.match(r"FUNC\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNC", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNC tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("GAMMA "):
                gamma_match = re.match(r"GAMMA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+GAMMA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if gamma_match:
                    gamma_name, params, return_type, gamma_body = gamma_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                    yapi_def["gamma"][gamma_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": gamma_body,
                        "partial": lambda *args: lambda *rest: self.interpreter.evaluate_expression(f"{gamma_body}({','.join(map(str, args + rest))})")
                    }
                    self.interpreter.object_counter["GAMMA"] += 1
                    self.interpreter.object_registry[id(gamma_body)] = {
                        "type": "GAMMA",
                        "name": gamma_name,
                        "atom": gamma_body
                    }
                    i += gamma_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"GAMMA tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("OMEGA "):
                omega_match = re.match(r"OMEGA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+OMEGA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if omega_match:
                    omega_name, params, return_type, omega_body = omega_match.groups()
                    if return_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                    yapi_def["omega"][omega_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": omega_body,
                        "self_apply": lambda x: self.interpreter.evaluate_expression(f"{omega_body}({x})")
                    }
                    self.interpreter.object_counter["OMEGA"] += 1
                    self.interpreter.object_registry[id(omega_body)] = {
                        "type": "OMEGA",
                        "name": omega_name,
                        "atom": omega_body
                    }
                    i += omega_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"OMEGA tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
            elif line.upper().startswith("DIM "):
                dim_match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if dim_match:
                    prop_name, prop_type = dim_match.groups()
                    if prop_type not in self.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                    yapi_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"DIM tanımı hatalı: {line}", context={"source": "handle_yapi", "line_no": self.interpreter.program_counter})
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[yapi_name] = yapi_def
            self.interpreter.object_counter["YAPI"] += 1
            self.interpreter.object_registry[id(yapi_def)] = {
                "type": "YAPI",
                "name": yapi_name,
                "atom": yapi_name
            }
        log.debug(f"YAPI tanımlandı: {yapi_name}")

    async def handle_select_case(self, command: str) -> None:
        """Çoklu koşullu yapı."""
        match = re.match(r"SELECT CASE\s+(.+?)\s+(.+?)\s+END\s+SELECT", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXSyntaxError("SELECT CASE komutunda sözdizimi hatası", context={"source": "handle_select_case", "line_no": self.interpreter.program_counter})
        expr, body = match.groups()
        value = self.interpreter.evaluate_expression(expr)
        lines = body.split("\n")
        case_matched = False
        for line in lines:
            line = line.strip()
            if line.upper().startswith("CASE "):
                case_match = re.match(r"CASE\s+(.+?)\s*:\s*(.+)", line, re.IGNORECASE)
                if case_match:
                    case_value, case_body = case_match.groups()
                    if self.interpreter.evaluate_expression(case_value) == value:
                        await self.interpreter.execute_command(case_body)
                        case_matched = True
                        break
            elif line.upper().startswith("CASE ELSE"):
                if not case_matched:
                    case_else_match = re.match(r"CASE ELSE\s*:\s*(.+)", line, re.IGNORECASE)
                    if case_else_match:
                        case_body = case_else_match.group(1)
                        await self.interpreter.execute_command(case_body)
                        break
        with self.lock:
            self.interpreter.object_counter["SELECT CASE"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "SELECT CASE",
                "name": "SELECT CASE",
                "atom": command
            }
        log.debug(f"SELECT CASE yürütüldü: {expr}")

    async def handle_data(self, command: str) -> None:
        """Veri tanımlar."""
        match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DATA komutunda sözdizimi hatası", context={"source": "handle_data", "line_no": self.interpreter.program_counter})
        values = [v.strip() for v in match.group(1).split(",")]
        with self.lock:
            self.interpreter.data_list.extend(values)
            self.interpreter.object_counter["DATA"] += len(values)
            for v in values:
                self.interpreter.object_registry[id(v)] = {
                    "type": "DATA",
                    "name": "DATA",
                    "atom": v
                }
        log.debug(f"Veri tanımlandı: {values}")

    async def handle_read(self, command: str) -> None:
        """Veri okur."""
        match = re.match(r"READ\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("READ komutunda sözdizimi hatası", context={"source": "handle_read", "line_no": self.interpreter.program_counter})
        var_names = [v.strip() for v in match.group(1).split(",")]
        with self.lock:
            for var in var_names:
                if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                    raise PdsXRuntimeError("Veri listesi sonu", context={"source": "handle_read", "line_no": self.interpreter.program_counter})
                value = self.interpreter.data_list[self.interpreter.data_pointer]
                self.interpreter.current_scope()[var] = value
                self.interpreter.data_pointer += 1
                self.interpreter.object_counter["VARIABLE"] += 1
                self.interpreter.object_registry[id(value)] = {
                    "type": "VARIABLE",
                    "name": var,
                    "atom": value
                }
        log.debug(f"Veri okundu: {var_names}")

    async def handle_restore(self, command: str) -> None:
        """Veri işaretçisini sıfırlar."""
        with self.lock:
            self.interpreter.data_pointer = 0
            self.interpreter.object_counter["RESTORE"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "RESTORE",
                "name": "RESTORE",
                "atom": command
            }
        log.debug("Veri işaretçisi sıfırlandı")

    async def handle_chain(self, command: str) -> None:
        """Yeni programı zincirler."""
        match = re.match(r"CHAIN\s+\"([^\"]+)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CHAIN komutunda sözdizimi hatası", context={"source": "handle_chain", "line_no": self.interpreter.program_counter})
        program_file = match.group(1)
        try:
            async with aiofiles.open(program_file, "r", encoding="utf-8") as f:
                program_text = await f.read()
            self.interpreter.load_program(program_text)
            with self.lock:
                self.interpreter.object_counter["CHAIN"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "CHAIN",
                    "name": program_file,
                    "atom": command
                }
            log.debug(f"Program zincirlendi: {program_file}")
        except Exception as e:
            raise PdsXIOException(f"Program yükleme hatası: {str(e)}", context={"source": "handle_chain", "line_no": self.interpreter.program_counter})

    async def handle_cont(self, command: str) -> None:
        """Yürütmeye devam eder."""
        with self.lock:
            self.interpreter.paused = False
            self.interpreter.object_counter["CONT"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "CONT",
                "name": "CONT",
                "atom": command
            }
        log.debug("Yürütme devam ediyor")

    async def handle_stop(self, command: str) -> None:
        """Yürütmeyi durdurur."""
        with self.lock:
            self.interpreter.paused = True
            self.interpreter.object_counter["STOP"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "STOP",
                "name": "STOP",
                "atom": command
            }
        log.debug("Yürütme durduruldu")

    async def handle_tron(self, command: str) -> None:
        """İzleme modunu açar."""
        with self.lock:
            self.interpreter.trace_mode = True
            self.interpreter.object_counter["TRON"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "TRON",
                "name": "TRON",
                "atom": command
            }
        log.debug("İzleme modu açıldı")

    async def handle_troff(self, command: str) -> None:
        """İzleme modunu kapatır."""
        with self.lock:
            self.interpreter.trace_mode = False
            self.interpreter.object_counter["TROFF"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "TROFF",
                "name": "TROFF",
                "atom": command
            }
        log.debug("İzleme modu kapatıldı")

    async def handle_common(self, command: str) -> None:
        """Değişkenleri paylaşır."""
        match = re.match(r"COMMON\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("COMMON komutunda sözdizimi hatası", context={"source": "handle_common", "line_no": self.interpreter.program_counter})
        var_names = [v.strip() for v in match.group(1).split(",")]
        with self.lock:
            for var in var_names:
                if var in self.interpreter.current_scope():
                    self.interpreter.shared_vars[var].append(self.interpreter.current_scope()[var])
                    self.interpreter.object_counter["COMMON"] += 1
                    self.interpreter.object_registry[id(var)] = {
                        "type": "COMMON",
                        "name": var,
                        "atom": var
                    }
        log.debug(f"Paylaşılan değişkenler: {var_names}")

    async def handle_declare(self, command: str) -> None:
        """Fonksiyon/yordam tanımlar."""
        match = re.match(r"DECLARE\s+(SUB|FUNCTION)\s+(\w+)\s*\((.*?)\)(?:\s+AS\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DECLARE komutunda sözdizimi hatası", context={"source": "handle_declare", "line_no": self.interpreter.program_counter})
        decl_type, name, params, return_type = match.groups()
        with self.lock:
            if decl_type.upper() == "SUB":
                self.interpreter.subs[name] = {"params": params}
                self.interpreter.object_counter["SUB"] += 1
                self.interpreter.object_registry[id(name)] = {
                    "type": "SUB",
                    "name": name,
                    "atom": name
                }
            else:
                if return_type not in self.data_types:
                    raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", context={"source": "handle_declare", "line_no": self.interpreter.program_counter})
                self.interpreter.functions[name] = {"params": params, "return_type": return_type}
                self.interpreter.object_counter["FUNCTION"] += 1
                self.interpreter.object_registry[id(name)] = {
                    "type": "FUNCTION",
                    "name": name,
                    "atom": name
                }
        log.debug(f"{decl_type} tanımlandı: {name}")

    async def handle_def(self, command: str) -> None:
        """Fonksiyon tanımlar."""
        match = re.match(r"DEF\s+(\w+)\s*\((.*?)\)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DEF komutunda sözdizimi hatası", context={"source": "handle_def", "line_no": self.interpreter.program_counter})
        func_name, params, expr = match.groups()
        with self.lock:
            self.interpreter.functions[func_name] = {
                "params": params,
                "body": lambda *args: self.interpreter.evaluate_expression(f"{expr}({','.join(map(str, args))})")
            }
            self.interpreter.object_counter["FUNCTION"] += 1
            self.interpreter.object_registry[id(expr)] = {
                "type": "FUNCTION",
                "name": func_name,
                "atom": expr
            }
        log.debug(f"Fonksiyon tanımlandı: {func_name}")

    async def handle_exit(self, command: str) -> None:
        """Döngü/yordamdan çıkar."""
        match = re.match(r"EXIT\s+(FOR|WHILE|SUB|FUNCTION)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("EXIT komutunda sözdizimi hatası", context={"source": "handle_exit", "line_no": self.interpreter.program_counter})
        exit_type = match.group(1).upper()
        with self.lock:
            if exit_type in ("FOR", "WHILE"):
                if not self.interpreter.loop_stack or self.interpreter.loop_stack[-1]["type"] != exit_type:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} döngüsü yok", context={"source": "handle_exit", "line_no": self.interpreter.program_counter})
                self.interpreter.loop_stack.pop()
            else:
                if not self.interpreter.call_stack:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} yordamı yok", context={"source": "handle_exit", "line_no": self.interpreter.program_counter})
                self.interpreter.call_stack.pop()
            self.interpreter.object_counter["EXIT"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "EXIT",
                "name": exit_type,
                "atom": command
            }
        log.debug(f"{exit_type}’den çıkıldı")

    async def handle_undim(self, command: str) -> None:
        """Değişkeni kaldırır."""
        match = re.match(r"UNDIM\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("UNDIM komutunda sözdizimi hatası", context={"source": "handle_undim", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        with self.lock:
            if var_name in self.interpreter.current_scope():
                value = self.interpreter.current_scope()[var_name]
                del self.interpreter.current_scope()[var_name]
                type_name = next((t for t, v in self.data_types.items() if isinstance(value, v)), "VARIABLE")
                self.interpreter.object_counter[type_name] -= 1
                self.interpreter.object_registry.pop(id(value), None)
                self.interpreter.object_counter["UNDIM"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "UNDIM",
                    "name": var_name,
                    "atom": command
                }
        log.debug(f"Değişken kaldırıldı: {var_name}")

    async def handle_setfield(self, command: str) -> None:
        """Yapı alanını günceller."""
        match = re.match(r"SETFIELD\s+(\w+)\.(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SETFIELD komutunda sözdizimi hatası", context={"source": "handle_setfield", "line_no": self.interpreter.program_counter})
        var_name, field, value_expr = match.groups()
        value = self.interpreter.evaluate_expression(value_expr)
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_setfield", "line_no": self.interpreter.program_counter})
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", context={"source": "handle_setfield", "line_no": self.interpreter.program_counter})
            struct[field] = value
            self.interpreter.object_counter["SETFIELD"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "SETFIELD",
                "name": f"{var_name}.{field}",
                "atom": command
            }
        log.debug(f"Yapı alanı güncellendi: {var_name}.{field} = {value}")

    async def handle_getfield(self, command: str) -> None:
        """Yapı alanını alır."""
        match = re.match(r"GETFIELD\s+(\w+)\.(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GETFIELD komutunda sözdizimi hatası", context={"source": "handle_getfield", "line_no": self.interpreter.program_counter})
        var_name, field, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_getfield", "line_no": self.interpreter.program_counter})
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", context={"source": "handle_getfield", "line_no": self.interpreter.program_counter})
            value = struct.get(field)
            self.interpreter.current_scope()[new_var] = value
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_counter["GETFIELD"] += 1
            self.interpreter.object_registry[id(value)] = {
                "type": "VARIABLE",
                "name": new_var,
                "atom": str(value) if value is not None else "null"
            }
            self.interpreter.object_registry[id(command)] = {
                "type": "GETFIELD",
                "name": f"{var_name}.{field}",
                "atom": command
            }
        log.debug(f"Yapı alanı alındı: {new_var} = {var_name}.{field}")

    async def handle_addfield(self, command: str) -> None:
        """Yapıya dinamik alan ekler."""
        match = re.match(r"ADDFIELD\s+(\w+)\.(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ADDFIELD komutunda sözdizimi hatası", context={"source": "handle_addfield", "line_no": self.interpreter.program_counter})
        var_name, field, type_name = match.groups()
        if type_name not in self.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", context={"source": "handle_addfield", "line_no": self.interpreter.program_counter})
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_addfield", "line_no": self.interpreter.program_counter})
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", context={"source": "handle_addfield", "line_no": self.interpreter.program_counter})
            struct[field] = None
            self.interpreter.object_counter[type_name] += 1
            self.interpreter.object_counter["ADDFIELD"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "ADDFIELD",
                "name": f"{var_name}.{field}",
                "atom": command
            }
        log.debug(f"Yapıya alan eklendi: {var_name}.{field} AS {type_name}")

    async def handle_removefield(self, command: str) -> None:
        """Yapıdan alan kaldırır."""
        match = re.match(r"REMOVEFIELD\s+(\w+)\.(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("REMOVEFIELD komutunda sözdizimi hatası", context={"source": "handle_removefield", "line_no": self.interpreter.program_counter})
        var_name, field = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_removefield", "line_no": self.interpreter.program_counter})
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}", context={"source": "handle_removefield", "line_no": self.interpreter.program_counter})
            if field in struct:
                value = struct.pop(field)
                type_name = next((t for t, v in self.data_types.items() if isinstance(value, v)), "VARIABLE")
                self.interpreter.object_counter[type_name] -= 1
                self.interpreter.object_counter["REMOVEFIELD"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "REMOVEFIELD",
                    "name": f"{var_name}.{field}",
                    "atom": command
                }
        log.debug(f"Yapıdan alan kaldırıldı: {var_name}.{field}")

    async def handle_newobj(self, command: str) -> None:
        """Nesne oluşturur."""
        match = re.match(r"NEWOBJ\s+(\w+)\((.*?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("NEWOBJ komutunda sözdizimi hatası", context={"source": "handle_newobj", "line_no": self.interpreter.program_counter})
        class_name, params, var_name = match.groups()
        if class_name not in self.interpreter.classes:
            raise PdsXRuntimeError(f"Sınıf bulunamadı: {class_name}", context={"source": "handle_newobj", "line_no": self.interpreter.program_counter})
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
                raise PdsXValueError(f"Geçersiz parametre sayısı: {class_name}.Init", context={"source": "handle_newobj", "line_no": self.interpreter.program_counter})
            with self.lock:
                self.interpreter.current_scope().update({f"_{i}": v for i, v in enumerate(param_values)})
                await self.interpreter.execute_command(class_def["subs"]["Init"]["body"])
                for i in range(len(param_values)):
                    self.interpreter.current_scope().pop(f"_{i}", None)
        with self.lock:
            self.interpreter.current_scope()[var_name] = instance
            self.interpreter.object_counter["OBJECT"] += 1
            self.interpreter.object_registry[id(instance)] = {
                "type": "OBJECT",
                "name": var_name,
                "class": class_name,
                "atom": f"obj_{class_name}_{var_name}"
            }
            self.interpreter.object_counter["NEWOBJ"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "NEWOBJ",
                "name": var_name,
                "atom": command
            }
        log.debug(f"Nesne oluşturuldu: {var_name} AS {class_name}")

    async def handle_countobj(self, command: str) -> None:
        """Nesne sayısını döndürür."""
        match = re.match(r"COUNTOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("COUNTOBJ komutunda sözdizimi hatası", context={"source": "handle_countobj", "line_no": self.interpreter.program_counter})
        type_name, var_name = match.groups()
        with self.lock:
            count = self.interpreter.object_counter.get(type_name, 0)
            self.interpreter.current_scope()[var_name] = count
            self.interpreter.object_counter["COUNTOBJ"] += 1
            self.interpreter.object_registry[id(count)] = {
                "type": "COUNTOBJ",
                "name": var_name,
                "atom": str(count)
            }
        log.debug(f"Nesne sayıldı: {type_name} = {count}")

    async def handle_inspobj(self, command: str) -> None:
        """Nesne detaylarını alır."""
        match = re.match(r"INSPOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("INSPOBJ komutunda sözdizimi hatası", context={"source": "handle_inspobj", "line_no": self.interpreter.program_counter})
        obj_id, var_name = match.groups()
        with self.lock:
            obj = self.interpreter.object_registry.get(int(obj_id), {})
            if not obj:
                raise PdsXRuntimeError(f"Nesne bulunamadı: {obj_id}", context={"source": "handle_inspobj", "line_no": self.interpreter.program_counter})
            self.interpreter.current_scope()[var_name] = obj
            self.interpreter.object_counter["INSPOBJ"] += 1
            self.interpreter.object_registry[id(obj)] = {
                "type": "INSPOBJ",
                "name": var_name,
                "atom": json.dumps(obj)[:100]
            }
        log.debug(f"Nesne incelendi: {obj_id}")

    async def handle_callapi(self, command: str) -> None:
        """HTTP API isteği yapar."""
        match = re.match(r"CALLAPI\s+\"([^\"]+)\"\s*,\s*METHOD=\"(\w+)\"\s*,\s*HEADERS=\{(.+?)\}\s*,\s*DATA=\{(.+?)\}\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLAPI komutunda sözdizimi hatası", context={"source": "handle_callapi", "line_no": self.interpreter.program_counter})
        url, method, headers, data, var_name = match.groups()
        try:
            headers = json.loads(f"{{{headers}}}")
            data = json.loads(f"{{{data}}}")
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=data) as resp:
                        response = await resp.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as resp:
                        response = await resp.json()
                else:
                    raise PdsXValueError(f"Desteklenmeyen metod: {method}", context={"source": "handle_callapi", "line_no": self.interpreter.program_counter})
            with self.lock:
                self.interpreter.current_scope()[var_name] = response
                self.interpreter.object_counter["CALLAPI"] += 1
                self.interpreter.object_registry[id(response)] = {
                    "type": "CALLAPI",
                    "name": var_name,
                    "atom": json.dumps(response)[:100]
                }
            log.debug(f"API çağrısı başarılı: {url}, Yanıt: {response}")
        except Exception as e:
            raise PdsXNetworkError(f"API çağrısı hatası: {str(e)}", context={"source": "handle_callapi", "line_no": self.interpreter.program_counter})

    async def handle_calldll(self, command: str) -> None:
        """DLL fonksiyonu çağırır."""
        match = re.match(r"CALLDLL\s+\"([^\"]+)\"\s*,\s*\"(\w+)\"\s*,\s*PARAMS=\((.+?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLDLL komutunda sözdizimi hatası", context={"source": "handle_calldll", "line_no": self.interpreter.program_counter})
        dll_name, func_name, params, var_name = match.groups()
        try:
            dll = ctypes.WinDLL(dll_name)
            func = getattr(dll, func_name)
            param_values = [self.interpreter.evaluate_expression(p.strip()) for p in params.split(",") if p.strip()]
            result = func(*param_values)
            with self.lock:
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["CALLDLL"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "CALLDLL",
                    "name": var_name,
                    "atom": str(result)
                }
            log.debug(f"DLL çağrısı başarılı: {dll_name}.{func_name}, Sonuç: {result}")
        except Exception as e:
            raise PdsXRuntimeError(f"DLL çağrısı hatası: {str(e)}", context={"source": "handle_calldll", "line_no": self.interpreter.program_counter})

        async def handle_sart(self, command: str) -> Optional[int]:
        """Koşullu boru hattı atlaması."""
        match = re.match(r"SART\s+(.+)\s+ATLA\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SART komutunda sözdizimi hatası", context={"source": "handle_sart", "line_no": self.interpreter.program_counter})
        condition, pipe_id, label = match.groups()
        try:
            if self.interpreter.evaluate_expression(condition):
                if pipe_id not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", context={"source": "handle_sart", "line_no": self.interpreter.program_counter})
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", context={"source": "handle_sart", "line_no": self.interpreter.program_counter})
                with self.lock:
                    self.interpreter.object_counter["SART"] += 1
                    self.interpreter.object_registry[id(command)] = {
                        "type": "SART",
                        "name": label,
                        "atom": command
                    }
                return self.interpreter.labels[label]
        except Exception as e:
            raise PdsXRuntimeError(f"SART değerlendirme hatası: {str(e)}", context={"source": "handle_sart", "line_no": self.interpreter.program_counter})
        return None

    async def handle_alias(self, command: str) -> None:
        """İsim değiştirme."""
        match = re.match(r"ALIAS\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ALIAS komutunda sözdizimi hatası", context={"source": "handle_alias", "line_no": self.interpreter.program_counter})
        old_name, new_name = match.groups()
        with self.lock:
            if old_name in self.interpreter.current_scope():
                self.interpreter.current_scope()[new_name] = self.interpreter.current_scope()[old_name]
                self.interpreter.object_counter["ALIAS"] += 1
                self.interpreter.object_registry[id(new_name)] = {
                    "type": "ALIAS",
                    "name": new_name,
                    "atom": old_name
                }
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {old_name}", context={"source": "handle_alias", "line_no": self.interpreter.program_counter})
        log.debug(f"İsim değiştirildi: {old_name} AS {new_name}")

    async def handle_restrict(self, command: str) -> None:
        """Erişim sınırlandırır."""
        match = re.match(r"RESTRICT\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("RESTRICT komutunda sözdizimi hatası", context={"source": "handle_restrict", "line_no": self.interpreter.program_counter})
        scope = match.group(1).strip().upper()
        with self.lock:
            if scope not in ("GLOBAL", "SHARED", "LOCAL"):
                raise PdsXValueError(f"Geçersiz kapsam: {scope}", context={"source": "handle_restrict", "line_no": self.interpreter.program_counter})
            self.interpreter.restricted_scopes.add(scope)
            self.interpreter.object_counter["RESTRICT"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "RESTRICT",
                "name": scope,
                "atom": command
            }
        log.debug(f"Kapsam sınırlandırıldı: {scope}")

    async def handle_clear_basic(self, command: str) -> None:
        """Değişkenleri sıfırlar."""
        match = re.match(r"CLEAR BASIC\s+(GLOBAL|SHARED|LOCAL)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CLEAR BASIC komutunda sözdizimi hatası", context={"source": "handle_clear_basic", "line_no": self.interpreter.program_counter})
        scope = match.group(1).upper()
        with self.lock:
            if scope == "GLOBAL":
                self.interpreter.global_vars.clear()
            elif scope == "SHARED":
                self.interpreter.shared_vars.clear()
            elif scope == "LOCAL":
                self.interpreter.current_scope().clear()
            self.interpreter.object_counter["CLEAR BASIC"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "CLEAR BASIC",
                "name": scope,
                "atom": command
            }
        log.debug(f"{scope} değişkenler sıfırlandı")

    async def handle_listfiles(self, command: str) -> None:
        """Dosya numaralarını listeler."""
        match = re.match(r"LISTFILES\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LISTFILES komutunda sözdizimi hatası", context={"source": "handle_listfiles", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        files = list(self.interpreter.file_handles.keys())
        with self.lock:
            self.interpreter.current_scope()[var_name] = files
            self.interpreter.object_counter["LISTFILES"] += 1
            self.interpreter.object_registry[id(files)] = {
                "type": "LISTFILES",
                "name": var_name,
                "atom": json.dumps(files)
            }
        log.debug(f"Dosyalar listelendi: {files}")

    async def handle_listprog(self, command: str) -> None:
        """Program satırlarını listeler."""
        match = re.match(r"LISTPROG\s*(?:(\d+)\s+TO\s+(\d+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LISTPROG komutunda sözdizimi hatası", context={"source": "handle_listprog", "line_no": self.interpreter.program_counter})
        start, end = match.groups()
        start = int(start) if start else 1
        end = int(end) if end else len(self.interpreter.program)
        lines = [self.interpreter.program[i][0] for i in range(start-1, min(end, len(self.interpreter.program)))]
        with self.lock:
            self.interpreter.current_scope()["_PROGRAM_LIST"] = lines
            self.interpreter.object_counter["LISTPROG"] += 1
            self.interpreter.object_registry[id(lines)] = {
                "type": "LISTPROG",
                "name": "_PROGRAM_LIST",
                "atom": json.dumps(lines)
            }
        for line in lines:
            print(line)
        log.debug(f"Program listelendi: {start} TO {end}")

    async def handle_checkfile(self, command: str) -> None:
        """Dosya durumunu kontrol eder."""
        match = re.match(r"CHECKFILE\s+#(\d+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CHECKFILE komutunda sözdizimi hatası", context={"source": "handle_checkfile", "line_no": self.interpreter.program_counter})
        file_num, var_name = match.groups()
        file_num = int(file_num)
        status = {"exists": file_num in self.interpreter.file_handles, "open": False}
        if status["exists"]:
            status["open"] = not self.interpreter.file_handles[file_num].closed
        with self.lock:
            self.interpreter.current_scope()[var_name] = status
            self.interpreter.object_counter["CHECKFILE"] += 1
            self.interpreter.object_registry[id(status)] = {
                "type": "CHECKFILE",
                "name": var_name,
                "atom": json.dumps(status)
            }
        log.debug(f"Dosya durumu kontrol edildi: #{file_num}, {status}")

    async def handle_screen(self, command: str) -> None:
        """Grafik ekran modunu ayarlar."""
        match = re.match(r"SCREEN\s+(\d+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SCREEN komutunda sözdizimi hatası", context={"source": "handle_screen", "line_no": self.interpreter.program_counter})
        mode = int(match.group(1))
        with self.lock:
            self.interpreter.current_scope()["_SCREEN_MODE"] = mode
            self.interpreter.object_counter["SCREEN"] += 1
            self.interpreter.object_registry[id(mode)] = {
                "type": "SCREEN",
                "name": "_SCREEN_MODE",
                "atom": str(mode)
            }
        log.debug(f"Ekran modu ayarlandı: {mode}")

    async def handle_sound(self, command: str) -> None:
        """Ses üretir."""
        match = re.match(r"SOUND\s+(\d+),\s+(\d+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SOUND komutunda sözdizimi hatası", context={"source": "handle_sound", "line_no": self.interpreter.program_counter})
        freq, duration = map(int, match.groups())
        with self.lock:
            self.interpreter.object_counter["SOUND"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "SOUND",
                "name": "SOUND",
                "atom": command
            }
        log.debug(f"Ses üretildi: Frekans={freq}, Süre={duration}")

    async def handle_sec_var(self, command: str) -> None:
        """Değişken erişimini kısıtlar."""
        match = re.match(r"SEC VAR\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SEC VAR komutunda sözdizimi hatası", context={"source": "handle_sec_var", "line_no": self.interpreter.program_counter})
        var_name = match.group(1)
        with self.lock:
            self.interpreter.restricted_vars.add(var_name)
            self.interpreter.object_counter["SEC VAR"] += 1
            self.interpreter.object_registry[id(var_name)] = {
                "type": "SEC VAR",
                "name": var_name,
                "atom": var_name
            }
        log.debug(f"Değişken kısıtlandı: {var_name}")

    async def handle_mon_var(self, command: str) -> None:
        """Değişken istatistiklerini toplar."""
        match = re.match(r"MON VAR\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("MON VAR komutunda sözdizimi hatası", context={"source": "handle_mon_var", "line_no": self.interpreter.program_counter})
        var_name, stat_var = match.groups()
        stats = {"access_count": 0, "last_access": time.time()}
        if var_name in self.interpreter.current_scope():
            stats["value"] = self.interpreter.current_scope()[var_name]
        with self.lock:
            self.interpreter.current_scope()[stat_var] = stats
            self.interpreter.object_counter["MON VAR"] += 1
            self.interpreter.object_registry[id(stats)] = {
                "type": "MON VAR",
                "name": stat_var,
                "atom": json.dumps(stats)
            }
        log.debug(f"Değişken izlendi: {var_name}, İstatistikler: {stats}")

    async def handle_convert(self, command: str) -> None:
        """Tip dönüşümü yapar."""
        match = re.match(r"CONVERT\s+(\w+)\s+FROM\s+(\w+)\s+TO\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CONVERT komutunda sözdizimi hatası", context={"source": "handle_convert", "line_no": self.interpreter.program_counter})
        var_name, source_type, target_type, new_var = match.groups()
        if source_type not in self.data_types or target_type not in self.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {source_type} veya {target_type}", context={"source": "handle_convert", "line_no": self.interpreter.program_counter})
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_convert", "line_no": self.interpreter.program_counter})
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_registry[id(converted)] = {
                    "type": target_type,
                    "name": new_var,
                    "atom": str(converted)
                }
                self.interpreter.object_counter["CONVERT"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "CONVERT",
                    "name": new_var,
                    "atom": command
                }
            log.debug(f"Tip dönüşümü yapıldı: {var_name} ({source_type}) -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", context={"source": "handle_convert", "line_no": self.interpreter.program_counter})

    async def handle_cast(self, command: str) -> None:
        """Hızlı tip dönüşümü yapar."""
        match = re.match(r"CAST\s+(\w+)\s+AS\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CAST komutunda sözdizimi hatası", context={"source": "handle_cast", "line_no": self.interpreter.program_counter})
        var_name, target_type, new_var = match.groups()
        if target_type not in self.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {target_type}", context={"source": "handle_cast", "line_no": self.interpreter.program_counter})
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", context={"source": "handle_cast", "line_no": self.interpreter.program_counter})
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_registry[id(converted)] = {
                    "type": target_type,
                    "name": new_var,
                    "atom": str(converted)
                }
                self.interpreter.object_counter["CAST"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "CAST",
                    "name": new_var,
                    "atom": command
                }
            log.debug(f"Hızlı tip dönüşümü yapıldı: {var_name} -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", context={"source": "handle_cast", "line_no": self.interpreter.program_counter})

    async def handle_alter_table(self, command: str) -> None:
        """Veritabanı tablosunu değiştirir."""
        match = re.match(r"ALTER TABLE\s+(\w+)\s+ADD\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ALTER TABLE komutunda sözdizimi hatası", context={"source": "handle_alter_table", "line_no": self.interpreter.program_counter})
        table_name, column_name, column_type = match.groups()
        with self.lock:
            self.interpreter.object_counter["ALTER TABLE"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "ALTER TABLE",
                "name": table_name,
                "atom": command
            }
        log.debug(f"Tablo değiştirildi: {table_name}, Yeni sütun: {column_name} AS {column_type}")

    async def handle_create_view(self, command: str) -> None:
        """Veritabanı görünümü oluşturur."""
        match = re.match(r"CREATE VIEW\s+(\w+)\s+AS\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CREATE VIEW komutunda sözdizimi hatası", context={"source": "handle_create_view", "line_no": self.interpreter.program_counter})
        view_name, query = match.groups()
        with self.lock:
            self.interpreter.object_counter["CREATE VIEW"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "CREATE VIEW",
                "name": view_name,
                "atom": command
            }
        log.debug(f"Görünüm oluşturuldu: {view_name}, Sorgu: {query}")

    async def handle_encrypt(self, command: str) -> None:
        """Veriyi şifreler."""
        match = re.match(r"ENCRYPT\s+(\w+)\s+WITH\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ENCRYPT komutunda sözdizimi hatası", context={"source": "handle_encrypt", "line_no": self.interpreter.program_counter})
        var_name, key_var, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope() or key_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name} veya {key_var}", context={"source": "handle_encrypt"})
            data = str(self.interpreter.current_scope()[var_name]).encode()
            key = str(self.interpreter.current_scope()[key_var]).encode()
            secure_data = SecureData(data, key)
            self.interpreter.current_scope()[new_var] = secure_data
            self.interpreter.object_counter["SECURE_DATA"] += 1
            self.interpreter.object_registry[id(secure_data)] = {"type": "SECURE_DATA", "name": new_var, "atom": str(secure_data)}
            self.interpreter.object_counter["ENCRYPT"] += 1
        log.debug(f"Veri şifrelendi: {new_var}")

    async def handle_decrypt(self, command: str) -> None:
        """Şifrelenmiş veriyi çözer."""
        match = re.match(r"DECRYPT\s+(\w+)\s+WITH\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DECRYPT komutunda sözdizimi hatası", context={"source": "handle_decrypt", "line_no": self.interpreter.program_counter})
        var_name, key_var, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope() or key_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name} veya {key_var}", context={"source": "handle_decrypt"})
            secure_data = self.interpreter.current_scope()[var_name]
            if not isinstance(secure_data, SecureData):
                raise PdsXTypeError(f"Geçersiz veri tipi: {var_name}", context={"source": "handle_decrypt"})
            key = str(self.interpreter.current_scope()[key_var]).encode()
            try:
                decrypted = secure_data.decrypt(key).decode()
                self.interpreter.current_scope()[new_var] = decrypted
                self.interpreter.object_counter["VARIABLE"] += 1
                self.interpreter.object_counter["DECRYPT"] += 1
            except Exception as e:
                raise PdsXRuntimeError(f"Şifre çözme hatası: {str(e)}", context={"source": "handle_decrypt"})
        log.debug(f"Veri çözüldü: {new_var}")

    async def handle_sign(self, command: str) -> None:
        """Veriye dijital imza ekler."""
        match = re.match(r"SIGN\s+(\w+)\s+WITH\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SIGN komutunda sözdizimi hatası", context={"source": "handle_sign", "line_no": self.interpreter.program_counter})
        var_name, key_var, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope() or key_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name} veya {key_var}", context={"source": "handle_sign"})
            data = str(self.interpreter.current_scope()[var_name]).encode()
            private_key = str(self.interpreter.current_scope()[key_var]).encode()
            signature = Signature(data, private_key)
            self.interpreter.current_scope()[new_var] = signature
            self.interpreter.object_counter["SIGNATURE"] += 1
            self.interpreter.object_registry[id(signature)] = {"type": "SIGNATURE", "name": new_var, "atom": str(signature)}
            self.interpreter.object_counter["SIGN"] += 1
        log.debug(f"Veri imzalandı: {new_var}")

    async def handle_verify(self, command: str) -> None:
        """Dijital imzayı doğrular."""
        match = re.match(r"VERIFY\s+(\w+)\s+WITH\s+(\w+)\s+AND\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("VERIFY komutunda sözdizimi hatası", context={"source": "handle_verify", "line_no": self.interpreter.program_counter})
        var_name, signature_var, key_var, new_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope() or signature_var not in self.interpreter.current_scope() or key_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}, {signature_var} veya {key_var}", context={"source": "handle_verify"})
            data = str(self.interpreter.current_scope()[var_name]).encode()
            signature = self.interpreter.current_scope()[signature_var]
            public_key = str(self.interpreter.current_scope()[key_var]).encode()
            if not isinstance(signature, Signature):
                raise PdsXTypeError(f"Geçersiz imza: {signature_var}", context={"source": "handle_verify"})
            is_valid = signature.verify(data, public_key)
            self.interpreter.current_scope()[new_var] = is_valid
            self.interpreter.object_counter["VERIFY"] += 1
        log.debug(f"İmza doğrulandı: {new_var} = {is_valid}")

    async def handle_secure_var(self, command: str) -> None:
        """Değişkeni şifrelenmiş olarak depolar."""
        match = re.match(r"SECURE_VAR\s+(\w+)\s+WITH\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SECURE_VAR komutunda sözdizimi hatası", context={"source": "handle_secure_var", "line_no": self.interpreter.program_counter})
        var_name, key_var = match.groups()
        with self.lock:
            if var_name not in self.interpreter.current_scope() or key_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name} veya {key_var}", context={"source": "handle_secure_var"})
            data = str(self.interpreter.current_scope()[var_name]).encode()
            key = str(self.interpreter.current_scope()[key_var]).encode()
            secure_data = SecureData(data, key)
            self.interpreter.current_scope()[var_name] = secure_data
            self.interpreter.restricted_vars.add(var_name)
            self.interpreter.object_counter["SECURE_VAR"] += 1
        log.debug(f"Değişken şifrelendi: {var_name}")

    async def handle_connect_iot(self, command: str) -> None:
        """MQTT broker'ına bağlanır."""
        match = re.match(r"CONNECT_IOT\s+BROKER\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CONNECT_IOT komutunda sözdizimi hatası", context={"source": "handle_connect_iot", "line_no": self.interpreter.program_counter})
        broker_url, var_name = match.groups()
        with self.lock:
            try:
                client = mqtt.Client()
                client.connect(broker_url)
                self.interpreter.current_scope()[var_name] = client
                self.interpreter.object_counter["IOT_CONNECTION"] += 1
                self.interpreter.object_registry[id(client)] = {"type": "IOT_CONNECTION", "name": var_name, "atom": broker_url}
                self.interpreter.object_counter["CONNECT_IOT"] += 1
            except Exception as e:
                raise PdsXRuntimeError(f"IoT bağlantı hatası: {str(e)}", context={"source": "handle_connect_iot"})
        log.debug(f"IoT broker'ına bağlanıldı: {broker_url}")

    async def handle_publish_iot(self, command: str) -> None:
        """IoT cihazına veri yayınlar."""
        match = re.match(r"PUBLISH_IOT\s+(\w+)\s+TO\s+(\w+)\s+ON\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("PUBLISH_IOT komutunda sözdizimi hatası", context={"source": "handle_publish_iot", "line_no": self.interpreter.program_counter})
        client_var, topic, data_var = match.groups()
        with self.lock:
            if client_var not in self.interpreter.current_scope() or data_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var} veya {data_var}", context={"source": "handle_publish_iot"})
            client = self.interpreter.current_scope()[client_var]
            data = str(self.interpreter.current_scope()[data_var]).encode()
            try:
                client.publish(topic, data)
                self.interpreter.object_counter["PUBLISH_IOT"] += 1
            except Exception as e:
                raise PdsXRuntimeError(f"Yayın hatası: {str(e)}", context={"source": "handle_publish_iot"})
        log.debug(f"IoT mesajı yayınlandı: {topic}")

    async def handle_subscribe_iot(self, command: str) -> None:
        """IoT cihazından veri alır."""
        match = re.match(r"SUBSCRIBE_IOT\s+(\w+)\s+FROM\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SUBSCRIBE_IOT komutunda sözdizimi hatası", context={"source": "handle_subscribe_iot", "line_no": self.interpreter.program_counter})
        client_var, topic, var_name = match.groups()
        with self.lock:
            if client_var not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {client_var}", context={"source": "handle_subscribe_iot"})
            client = self.interpreter.current_scope()[client_var]
            try:
                def on_message(client, userdata, msg):
                    iot_msg = IotMessage(msg.topic, msg.payload, time.time())
                    self.interpreter.current_scope()[var_name] = iot_msg
                    self.interpreter.object_counter["IOT_MESSAGE"] += 1
                    self.interpreter.object_registry[id(iot_msg)] = {"type": "IOT_MESSAGE", "name": var_name, "atom": str(iot_msg)}
                client.on_message = on_message
                client.subscribe(topic)
                client.loop_start()
                self.interpreter.object_counter["SUBSCRIBE_IOT"] += 1
            except Exception as e:
                raise PdsXRuntimeError(f"Abonelik hatası: {str(e)}", context={"source": "handle_subscribe_iot"})
        log.debug(f"IoT aboneliği başlatıldı: {topic}")

    async def pdf_read_text(self, file_path: str) -> str:
        """PDF dosyasından metin okur."""
        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            with self.lock:
                self.interpreter.object_counter["PDF_READ_TEXT"] += 1
                self.interpreter.object_registry[id(text)] = {
                    "type": "PDF_READ_TEXT",
                    "name": file_path,
                    "atom": text[:100]
                }
            log.debug(f"PDF metni okundu: {file_path}, Uzunluk: {len(text)}")
            return text
        except Exception as e:
            raise PdsXIOException(f"PDF okuma hatası: {str(e)}", context={"source": "pdf_read_text", "line_no": self.interpreter.program_counter})

    async def pdf_extract_tables(self, file_path: str) -> List[pd.DataFrame]:
        """PDF dosyasından tablolar çıkarır."""
        try:
            with pdfplumber.open(file_path) as pdf:
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
            with self.lock:
                self.interpreter.object_counter["PDF_EXTRACT_TABLES"] += 1
                self.interpreter.object_registry[id(tables)] = {
                    "type": "PDF_EXTRACT_TABLES",
                    "name": file_path,
                    "atom": str(len(tables))
                }
            log.debug(f"PDF tabloları çıkarıldı: {file_path}, Tablo sayısı: {len(tables)}")
            return tables
        except Exception as e:
            raise PdsXIOException(f"PDF tablo çıkarma hatası: {str(e)}", context={"source": "pdf_extract_tables", "line_no": self.interpreter.program_counter})

    async def web_get(self, url: str) -> Dict:
        """HTTP GET isteği yapar."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise PdsXNetworkError(f"HTTP hatası: {resp.status}", context={"source": "web_get", "line_no": self.interpreter.program_counter})
                    response = await resp.json()
            with self.lock:
                self.interpreter.object_counter["WEB_GET"] += 1
                self.interpreter.object_registry[id(response)] = {
                    "type": "WEB_GET",
                    "name": url,
                    "atom": json.dumps(response)[:100]
                }
            log.debug(f"Web isteği başarılı: {url}, Yanıt: {response}")
            return response
        except Exception as e:
            raise PdsXNetworkError(f"Web isteği hatası: {str(e)}", context={"source": "web_get", "line_no": self.interpreter.program_counter})

    async def system(self, cmd: str) -> int:
        """Sistem komutu yürütür."""
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise PdsXRuntimeError(f"Sistem komut hatası: {stderr.decode()}", context={"source": "system", "line_no": self.interpreter.program_counter})
            with self.lock:
                self.interpreter.object_counter["SYSTEM"] += 1
                self.interpreter.object_registry[id(cmd)] = {
                    "type": "SYSTEM",
                    "name": cmd,
                    "atom": cmd
                }
            log.debug(f"Sistem komutu yürütüldü: {cmd}, Çıkış kodu: {process.returncode}")
            return process.returncode
        except Exception as e:
            raise PdsXRuntimeError(f"Sistem komut hatası: {str(e)}", context={"source": "system", "line_no": self.interpreter.program_counter})

    # Deneysel/Bilimsel İşlevler
    def quantum_correlation_analysis(self, state1: Union[List[complex], QuantumState], state2: Union[List[complex], QuantumState]) -> Dict:
        """Kuantum korelasyon analizi yapar (Bell eşitsizlikleri)."""
        try:
            s1 = state1.amplitudes if isinstance(state1, QuantumState) else np.array(state1, dtype=np.complex128)
            s2 = state2.amplitudes if isinstance(state2, QuantumState) else np.array(state2, dtype=np.complex128)
            if s1.shape != s2.shape:
                raise PdsXValueError("Kuantum durum boyutları uyuşmuyor", context={"source": "quantum_correlation_analysis"})
            corr_matrix = np.outer(s1.conj(), s2).real
            bell_violation = np.abs(corr_matrix).sum() > 2 * np.sqrt(2)
            result = {
                "correlation_matrix": corr_matrix.tolist(),
                "bell_violation": bell_violation,
                "entanglement_score": float(np.linalg.norm(corr_matrix))
            }
            with self.lock:
                self.interpreter.object_counter["QUANTUM_CORR"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "QUANTUM_CORR",
                    "name": "quantum_analysis",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Kuantum korelasyon analizi: {result}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Kuantum korelasyon hatası: {str(e)}", context={"source": "quantum_correlation_analysis"})

    def chaos_pattern_detection(self, data: Union[List[float], ChaosField]) -> Dict:
        """Kaotik sistem analizi yapar (Lyapunov üssü, Runge-Kutta)."""
        try:
            d = data.data if isinstance(data, ChaosField) else np.array(data, dtype=np.float64)
            def lorenz(t, state, sigma=10, rho=28, beta=8/3):
                x, y, z = state
                return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
            from scipy.integrate import solve_ivp
            t_span = (0, 100)
            y0 = d[:3] if len(d) >= 3 else [1.0, 1.0, 1.0]
            sol = solve_ivp(lorenz, t_span, y0, t_eval=np.linspace(0, 100, 1000))
            lyapunov = np.log(np.abs(np.diff(sol.y[0])).mean() + 1e-6)
            result = {
                "lyapunov_exponent": float(lyapunov),
                "chaos_level": min(max(lyapunov / 2, 0), 1),
                "trajectory": sol.y.tolist()
            }
            with self.lock:
                self.interpreter.object_counter["CHAOS_DETECT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "CHAOS_DETECT",
                    "name": "chaos_analysis",
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"Kaotik desen analizi: {result}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Kaotik desen hatası: {str(e)}", context={"source": "chaos_pattern_detection"})

    def neural_data_processing(self, data: Union[List[float], NeuralTensor]) -> NeuralTensor:
        """Nöral ağlarla veri işleme (LSTM modeli)."""
        try:
            d = data.data if isinstance(data, NeuralTensor) else np.array(data, dtype=np.float64).reshape(-1, 10, 1)
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

    def genetic_optimization_engine(self, task: str, params: Dict) -> Dict:
        """Genetik algoritmalarla optimizasyon."""
        try:
            pop_size = params.get("pop_size", 100)
            generations = params.get("generations", 50)
            mutation_rate = params.get("mutation_rate", 0.1)
            population = [random.uniform(-10, 10) for _ in range(pop_size)]
            def fitness(x):
                return np.sin(x) + np.cos(x)  # Mock fitness fonksiyonu
            for _ in range(generations):
                fitness_scores = [fitness(x) for x in population]
                parents = [population[i] for i in np.argsort(fitness_scores)[-pop_size//2:]]
                offspring = []
                for _ in range(pop_size - len(parents)):
                    p1, p2 = random.sample(parents, 2)
                    child = (p1 + p2) / 2
                    if random.random() < mutation_rate:
                        child += random.gauss(0, 1)
                    offspring.append(child)
                population = parents + offspring
            fitness_scores = [fitness(x) for x in population]
            best_idx = np.argmax(fitness_scores)
            result = {
                "optimal_value": population[best_idx],
                "fitness": fitness_scores[best_idx]
            }
            with self.lock:
                self.interpreter.object_counter["GENETIC_OPT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "GENETIC_OPT",
                    "name": "genetic_output",
                    "atom": json.dumps(result)
                }
            log.debug(f"Genetik optimizasyon: {result}")
            return result
        except Exception as e:
            raise PdsXRuntimeError(f"Genetik optimizasyon hatası: {str(e)}", context={"source": "genetic_optimization_engine"})

    def blockchain_integrity_check(self, data: Union[Dict, BlockchainLedger]) -> str:
        """Blockchain tabanlı veri doğrulama."""
        try:
            d = data.ledger if isinstance(data, BlockchainLedger) else data
            data_str = json.dumps(d, sort_keys=True)
            data_hash = sha256(data_str.encode("utf-8")).hexdigest()
            prev_hash = self.interpreter.current_scope().get("_PREV_BLOCK_HASH", "")
            integrity = prev_hash == "" or sha256(prev_hash.encode("utf-8")).hexdigest() == data_hash
            with self.lock:
                self.interpreter.current_scope()["_PREV_BLOCK_HASH"] = data_hash
                self.interpreter.object_counter["BLOCKCHAIN_CHECK"] += 1
                self.interpreter.object_registry[id(data_hash)] = {
                    "type": "BLOCKCHAIN_CHECK",
                    "name": "blockchain_hash",
                    "atom": data_hash
                }
            log.debug(f"Blockchain doğrulama: {data_hash}, Bütünlük: {integrity}")
            return data_hash
        except Exception as e:
            raise PdsXRuntimeError(f"Blockchain doğrulama hatası: {str(e)}", context={"source": "blockchain_integrity_check"})
    def hash_data(self, data: str) -> str:
        """Verinin SHA-256 hash'ini üretir."""
        try:
            data_bytes = str(data).encode()
            hash_value = sha256(data_bytes).hexdigest()
            with self.lock:
                self.interpreter.object_counter["HASH"] += 1
                self.interpreter.object_registry[id(hash_value)] = {"type": "HASH", "name": "hash", "atom": hash_value}
            log.debug(f"Hash üretildi: {hash_value}")
            return hash_value
        except Exception as e:
            raise PdsXRuntimeError(f"Hash hatası: {str(e)}", context={"source": "hash_data"})

    def check_auth(self, user: str, role: str) -> bool:
        """Kullanıcı yetkisini kontrol eder."""
        try:
            # Mock yetkilendirme (gerçek RBAC sistemi entegre edilecek)
            auth = (user in self.interpreter.current_scope().get("_AUTH_USERS", {})) and (role in self.interpreter.current_scope().get("_AUTH_USERS", {}).get(user, []))
            with self.lock:
                self.interpreter.object_counter["CHECK_AUTH"] += 1
                self.interpreter.object_registry[id(auth)] = {"type": "CHECK_AUTH", "name": f"{user}_{role}", "atom": str(auth)}
            log.debug(f"Yetki kontrolü: {user}, {role} -> {auth}")
            return auth
        except Exception as e:
            raise PdsXRuntimeError(f"Yetki kontrol hatası: {str(e)}", context={"source": "check_auth"})
        
    async def parse_core_command(self, command: str) -> Optional[int]:
        """Çekirdek komutları ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        for cmd_name, handler in self.command_handlers.items():
            if command_upper.startswith(cmd_name):
                try:
                    return await handler(command)
                except PdsXException as e:
                    await self.interpreter.exception_manager.handle_error(e)
                    raise
                except Exception as e:
                    await self.interpreter.exception_manager.handle_error(e)
                    raise PdsXRuntimeError(f"Komut yürütme hatası: {str(e)}", context={"source": "parse_core_command", "line_no": self.interpreter.program_counter})
        raise PdsXSyntaxError(f"Bilinmeyen çekirdek komut: {command}", context={"source": "parse_core_command", "line_no": self.interpreter.program_counter})

if __name__ == "__main__":
    print("core2.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")