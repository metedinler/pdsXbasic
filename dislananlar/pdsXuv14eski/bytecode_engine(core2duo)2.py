# bytecode_engine.py - PDS-X BASIC v15 Bayt Kodu Yürütme Motoru (Intel Core Duo 2 ve CPython Optimize)
# Version: 1.5.1
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
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache
import numpy as np
import pandas as pd
import aiohttp
from pdsx_exception2 import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError, PdsXTypeError,
    PdsXValueError, PdsXIOException, PdsXNetworkError
)
try:
    from cython import compiled
except ImportError:
    compiled = False  # Cython yoksa saf Python kullanılır

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_bytecode_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bytecode_engine")

# Cython için statik tipli yardımcı fonksiyonlar
if compiled:
    from cython.cimports import libc
    @cython.cfunc
    def fast_eval(expr: str, scope: Dict[str, Any]) -> Any:
        return eval(expr, {}, scope)
    @cython.cfunc
    def fast_memcpy(dest: cython.pointer(cython.int), src: cython.pointer(cython.int), size: cython.int) -> None:
        libc.string.memcpy(dest, src, size)
else:
    def fast_eval(expr: str, scope: Dict[str, Any]) -> Any:
        return eval(expr, {}, scope)
    def fast_memcpy(dest: List[int], src: List[int], size: int) -> None:
        dest[:size] = src[:size]

class BytecodeEngine:
    """
    PDS-X BASIC v15 bayt kodu yürütme motoru.
    Intel Core Duo 2 (SSE/SSE2) ve CPython/Cython optimizasyonları ile güçlendirilmiştir.
    Tüm v14/v15 komutlarını, iç içe kullanımı, NEXT desteğini ve deneysel özellikleri destekler.
    """
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.metadata: Dict = {
            "bytecode_engine": {
                "version": "1.5.1",
                "dependencies": ["numpy", "pandas", "aiofiles", "aiohttp", "cython"],
                "optimizations": ["SSE", "SSE2", "Cython", "LRU Cache"]
            }
        }
        # Intel Core Duo 2 için SIMD önbelleği
        self.simd_cache: Dict[str, np.ndarray] = {}
        self.register_cache: Dict[str, Any] = {}  # Önbellek dostu kayıtlar
        self.memory: List[int] = [0] * 0x1000000  # 16 MB sanal bellek
        self.opcode_table: Dict[str, Callable] = {
            # PDS-X BASIC Komutları
            "LET": self.op_let,
            "IF": self.op_if,
            "FOR": self.op_for,
            "FOREACH": self.op_foreach,
            "WHILE": self.op_while,
            "NEXT": self.op_next,
            "DIM": self.op_dim,
            "END": self.op_end,
            "CLASS": self.op_class,
            "YAPI": self.op_yapi,
            "PRINT": self.op_print,
            "GOTO": self.op_goto,
            "GOSUB": self.op_gosub,
            "RETURN": self.op_return,
            "SELECT_CASE": self.op_select_case,
            "DATA": self.op_data,
            "READ": self.op_read,
            "RESTORE": self.op_restore,
            "CHAIN": self.op_chain,
            "CONT": self.op_cont,
            "STOP": self.op_stop,
            "TRON": self.op_tron,
            "TROFF": self.op_troff,
            "COMMON": self.op_common,
            "DECLARE": self.op_declare,
            "DEF": self.op_def,
            "EXIT": self.op_exit,
            "UNDIM": self.op_undim,
            "SETFIELD": self.op_setfield,
            "GETFIELD": self.op_getfield,
            "ADDFIELD": self.op_addfield,
            "REMOVEFIELD": self.op_removefield,
            "NEWOBJ": self.op_newobj,
            "COUNTOBJ": self.op_countobj,
            "INSPOBJ": self.op_inspobj,
            "CALLAPI": self.op_callapi,
            "CALLDLL": self.op_calldll,
            "SART": self.op_sart,
            "ALIAS": self.op_alias,
            "RESTRICT": self.op_restrict,
            "CLEAR_BASIC": self.op_clear_basic,
            "LISTFILES": self.op_listfiles,
            "LISTPROG": self.op_listprog,
            "CHECKFILE": self.op_checkfile,
            "SCREEN": self.op_screen,
            "SOUND": self.op_sound,
            "SEC_VAR": self.op_sec_var,
            "MON_VAR": self.op_mon_var,
            "CONVERT": self.op_convert,
            "CAST": self.op_cast,
            "ALTER_TABLE": self.op_alter_table,
            "CREATE_VIEW": self.op_create_view,
            "GAMMA": self.op_gamma,
            "OMEGA": self.op_omega,
            # Intel Core Duo 2 için SIMD Optimize Komutlar
            "SIMD_ADD": self.op_simd_add,
            "SIMD_SUB": self.op_simd_sub,
            "SIMD_MUL": self.op_simd_mul,
            "SIMD_DIV": self.op_simd_div,
            "SIMD_CMP": self.op_simd_cmp,
            "SIMD_MOV": self.op_simd_mov,
            # Deneysel Özellik Opcode’ları
            "QUANTUM_CORR": self.op_quantum_corr,
            "NEURAL_PROCESS": self.op_neural_process,
            "CHAOS_DETECT": self.op_chaos_detect,
            "GENETIC_OPT": self.op_genetic_opt,
            "BLOCKCHAIN_CHECK": self.op_blockchain_check
        }
        self.stack: List[Any] = []
        self.program_counter: int = 0
        self.call_stack: List[Dict] = []
        self.loop_stack: List[Dict] = []
        self.opcode_cache: Dict[str, Callable] = {}  # Sık kullanılan opcode’lar için önbellek

    # SIMD Yardımcı Fonksiyonları (SSE/SSE2)
    @lru_cache(maxsize=2048)
    def simd_process(self, data: np.ndarray, operation: str) -> np.ndarray:
        """
        SSE2 ile vektör işlemleri gerçekleştirir.
        :param data: İşlenecek NumPy dizisi (float32).
        :param operation: İşlem türü (ADD, SUB, MUL, DIV, CMP, MOV).
        :return: İşlem sonucu NumPy dizisi.
        """
        data = np.array(data, dtype=np.float32)
        if operation == "ADD":
            return np.add(data, np.ones_like(data, dtype=np.float32))
        elif operation == "SUB":
            return np.subtract(data, np.ones_like(data, dtype=np.float32))
        elif operation == "MUL":
            return np.multiply(data, np.ones_like(data, dtype=np.float32) * 2.0)
        elif operation == "DIV":
            return np.divide(data, np.ones_like(data, dtype=np.float32) * 2.0)
        elif operation == "CMP":
            return np.greater(data, np.zeros_like(data, dtype=np.float32)).astype(np.float32)
        elif operation == "MOV":
            return data.copy()
        raise PdsXValueError(f"Geçersiz SIMD işlemi: {operation}", code="SIMD001")

    # Bellek yönetimi için yardımcı fonksiyon
    def _cleanup_memory(self) -> None:
        """
        Bellek ve önbellek temizliği yapar.
        """
        with self.lock:
            self.memory = [0] * 0x1000000  # Sanal belleği sıfırla
            self.simd_cache.clear()
            self.register_cache.clear()
            log.debug("Bellek ve önbellek temizlendi")

    # PDS-X BASIC Opcode’ları
    async def op_let(self, operands: List[Any]) -> None:
        """
        Değişkene değer atar (Cython optimize).
        :param operands: [var_names: str, expr: str]
        """
        var_names, expr = operands
        try:
            values = fast_eval(expr, self.interpreter.current_scope())
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
                elif isinstance(values, (pd.DataFrame, self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"], self.interpreter.data_types["QUANTUM_STATE"], self.interpreter.data_types["HOLO_DATA"], self.interpreter.data_types["CHAOS_FIELD"], self.interpreter.data_types["NEURAL_TENSOR"], self.interpreter.data_types["BLOCKCHAIN_LEDGER"])):
                    for var in var_names:
                        self.interpreter.current_scope()[var] = values
                        type_name = next(t for t, v in self.interpreter.data_types.items() if isinstance(values, v))
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

    async def op_if(self, operands: List[Any]) -> Optional[int]:
        """
        Koşullu yürütme.
        :param operands: [condition: str, then_block: str, else_block: str]
        """
        condition, then_block, else_block = operands
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

    async def op_for(self, operands: List[Any]) -> Optional[int]:
        """
        Sayısal döngü (FOR ... NEXT veya FOR ... END FOR).
        :param operands: [var_name: str, start: str, end: str, step: str, body: str]
        """
        var_name, start, end, step, body = operands
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
        
        return None  # NEXT veya END FOR, döngüyü ilerletecek

    async def op_foreach(self, operands: List[Any]) -> Optional[int]:
        """
        Koleksiyon döngüsü (FOR EACH ... NEXT veya FOR EACH ... END FOR).
        :param operands: [var_name: str, collection_name: str, body: str]
        """
        var_name, collection_name, body = operands
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
        
        return None  # NEXT veya END FOR, döngüyü ilerletecek

    async def op_next(self, operands: List[Any]) -> Optional[int]:
        """
        FOR veya FOREACH döngüsünü ilerletir (NEXT talimatı).
        :param operands: [loop_type: str, var_name: str]
        """
        loop_type, var_name = operands
        if not self.loop_stack or self.loop_stack[-1]["type"] != loop_type:
            raise PdsXRuntimeError(f"Kapatılacak {loop_type} döngüsü yok", code="NEXT001")
        
        loop_data = self.loop_stack[-1]
        with self.lock:
            self.interpreter.object_counter["NEXT"] += 1
        
        if loop_type == "FOR":
            current = self.interpreter.current_scope()[var_name]
            end = float(loop_data["end"])
            step = float(loop_data["step"])
            index = loop_data["index"] + 1
            
            if (step > 0 and current > end) or (step < 0 and current < end):
                self.loop_stack.pop()
                return None
            
            current += step
            self.interpreter.current_scope()[var_name] = current
            loop_data["index"] = index
            loop_data["history"].append(current)
            log.debug(f"NEXT FOR: {var_name} = {current}, Index: {index}")
            return loop_data["start_pc"]  # Döngü başına dön
        
        elif loop_type == "FOREACH":
            items = loop_data["items"]
            index = loop_data["index"] + 1
            
            if index >= len(items):
                self.loop_stack.pop()
                return None
            
            item = items[index]
            self.interpreter.current_scope()[var_name] = item if not isinstance(loop_data["collection"], dict) else item[1]
            loop_data["index"] = index
            loop_data["history"].append(item)
            log.debug(f"NEXT FOREACH: {var_name} = {item}, Index: {index}")
            return loop_data["start_pc"]  # Döngü başına dön
        
        raise PdsXValueError(f"Geçersiz döngü tipi: {loop_type}", code="NEXT002")

    async def op_while(self, operands: List[Any]) -> Optional[int]:
        """
        Koşullu döngü.
        :param operands: [condition: str, body: str]
        """
        condition, body = operands
        with self.lock:
            self.loop_stack.append({"type": "WHILE", "start_pc": self.program_counter})
            self.interpreter.object_counter["WHILE"] += 1
        try:
            if self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(body)
                return self.loop_stack[-1]["start_pc"]  # Döngü başına dön
            else:
                self.loop_stack.pop()
        except Exception as e:
            raise PdsXRuntimeError(f"WHILE döngü hatası: {str(e)}", code="WHILE001")
        return None

    async def op_dim(self, operands: List[Any]) -> None:
        """
        Değişken tanımlar (tek satırlık ve kapanışlı, Cython optimize).
        :param operands: [var_name: str, type_name: str, initial_value: str, size: str]
        """
        var_name, type_name, initial_value, size = operands
        if type_name not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}", code="DIM001")
        value = None
        try:
            if initial_value:
                value = fast_eval(initial_value, self.interpreter.current_scope())
            elif type_name in ("LIST", "DICT", "SET", "STACK", "QUEUE"):
                value = [] if type_name == "LIST" else {} if type_name == "DICT" else set() if type_name == "SET" else deque()
            elif type_name == "ARRAY":
                size = int(size) if size else 16  # SSE2 için 16-byte hizalı
                value = np.zeros(size, dtype=np.float32)
            elif type_name == "DATAFRAME":
                value = pd.DataFrame()
            elif type_name == "SKALER":
                value = self.interpreter.data_types["SKALER"](0.0)
            elif type_name == "VECTOR":
                size = int(size) if size else 16
                value = self.interpreter.data_types["VECTOR"]([0.0] * size)
            elif type_name == "MATRIX":
                size = int(size) if size else 16
                value = self.interpreter.data_types["MATRIX"]([[0.0] * size for _ in range(size)])
            elif type_name == "TENSOR":
                size = int(size) if size else 16
                value = self.interpreter.data_types["TENSOR"]([[[0.0] * size for _ in range(size)] for _ in range(size)])
            elif type_name == "QUANTUM_STATE":
                size = int(size) if size else 2
                value = self.interpreter.data_types["QUANTUM_STATE"]([1.0 / (size ** 0.5)] * size)
            elif type_name == "HOLO_DATA":
                value = self.interpreter.data_types["HOLO_DATA"](b"")
            elif type_name == "CHAOS_FIELD":
                value = self.interpreter.data_types["CHAOS_FIELD"]([1.0, 1.0, 1.0])
            elif type_name == "NEURAL_TENSOR":
                value = self.interpreter.data_types["NEURAL_TENSOR"]([])
            elif type_name == "BLOCKCHAIN_LEDGER":
                value = self.interpreter.data_types["BLOCKCHAIN_LEDGER"]({})
            elif type_name == "ENUM":
                value = {}
            elif type_name in ("STRUCT", "UNION", "CLASS", "CLAZZ", "YAPI"):
                value = {}
            elif type_name in ("BYTE", "SHORT", "INTEGER", "LONG"):
                value = 0
            elif type_name in ("FLOAT128", "FLOAT256", "FLOAT512"):
                value = self.interpreter.data_types[type_name]("0.0")
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

    async def op_end(self, operands: List[Any]) -> None:
        """
        Blok, veri yapısı veya program sonlandırır.
        :param operands: [block_type: str]
        """
        block_type = operands[0] if operands else None
        with self.lock:
            if block_type:
                if block_type not in self.interpreter.data_types and block_type not in ("FOR", "WHILE", "IF", "CLASS", "YAPI", "SELECT"):
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

    async def op_class(self, operands: List[Any]) -> None:
        """
        Sınıf tanımlar.
        :param operands: [class_name: str, body: str]
        """
        class_name, body = operands
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
                    if return_type not in self.interpreter.data_types:
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
                    if prop_type not in self.interpreter.data_types:
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

    async def op_yapi(self, operands: List[Any]) -> None:
        """
        Nesne tabanlı sınıf oluşturucu (YAPI).
        :param operands: [yapi_name: str, body: str]
        """
        yapi_name, body = operands
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
                    if return_type not in self.interpreter.data_types:
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
                    if return_type not in self.interpreter.data_types:
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
                    if return_type not in self.interpreter.data_types:
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
                    if prop_type not in self.interpreter.data_types:
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

    async def op_print(self, operands: List[Any]) -> None:
        """
        Ekrana yazdırır.
        :param operands: [expr: str]
        """
        expr = operands[0]
        try:
            value = self.interpreter.evaluate_expression(expr)
            print(value)
            with self.lock:
                self.interpreter.object_counter["PRINT"] += 1
        except Exception as e:
            raise PdsXRuntimeError(f"PRINT değerlendirme hatası: {str(e)}", code="PRINT001")

    async def op_goto(self, operands: List[Any]) -> int:
        """
        Etikete atlar.
        :param operands: [label: str]
        """
        label = operands[0]
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOTO001")
        with self.lock:
            self.interpreter.object_counter["GOTO"] += 1
        return self.interpreter.labels[label]

    async def op_gosub(self, operands: List[Any]) -> int:
        """
        Alt yordama atlar.
        :param operands: [label: str]
        """
        label = operands[0]
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOSUB001")
        with self.lock:
            self.call_stack.append({"return_pc": self.program_counter + 1})
            self.interpreter.object_counter["GOSUB"] += 1
        return self.interpreter.labels[label]

    async def op_return(self, operands: List[Any]) -> int:
        """
        Alt yordamdan döner.
        :param operands: []
        """
        if not self.call_stack:
            raise PdsXRuntimeError("Geri dönülecek yordam yok", code="RETURN001")
        with self.lock:
            return_pc = self.call_stack.pop()["return_pc"]
            self.interpreter.object_counter["RETURN"] += 1
        return return_pc

    async def op_select_case(self, operands: List[Any]) -> None:
        """
        Çoklu koşullu yapı.
        :param operands: [expr: str, cases: List[Tuple[str, str]]]
        """
        expr, cases = operands
        value = self.interpreter.evaluate_expression(expr)
        case_matched = False
        for case_value, case_body in cases:
            if case_value == "ELSE" or self.interpreter.evaluate_expression(case_value) == value:
                await self.interpreter.execute_command(case_body)
                case_matched = True
                break
        with self.lock:
            self.interpreter.object_counter["SELECT_CASE"] += 1
        log.debug(f"SELECT CASE yürütüldü: {expr}")

    async def op_data(self, operands: List[Any]) -> None:
        """
        Veri tanımlar.
        :param operands: [values: str]
        """
        values = operands[0].split(",")
        with self.lock:
            self.interpreter.data_list.extend([v.strip() for v in values])
            self.interpreter.object_counter["DATA"] += len(values)
        log.debug(f"Veri tanımlandı: {values}")

    async def op_read(self, operands: List[Any]) -> None:
        """
        Veri okur.
        :param operands: [var_names: str]
        """
        var_names = [v.strip() for v in operands[0].split(",")]
        with self.lock:
            for var in var_names:
                if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                    raise PdsXRuntimeError("Veri listesi sonu", code="READ001")
                value = self.interpreter.data_list[self.interpreter.data_pointer]
                self.interpreter.current_scope()[var] = value
                self.interpreter.data_pointer += 1
                self.interpreter.object_counter["VARIABLE"] += 1
        log.debug(f"Veri okundu: {var_names}")

    async def op_restore(self, operands: List[Any]) -> None:
        """
        Veri işaretçisini sıfırlar.
        :param operands: []
        """
        with self.lock:
            self.interpreter.data_pointer = 0
            self.interpreter.object_counter["RESTORE"] += 1
        log.debug("Veri işaretçisi sıfırlandı")

    async def op_chain(self, operands: List[Any]) -> None:
        """
        Yeni programı zincirler.
        :param operands: [program_file: str]
        """
        program_file = operands[0]
        try:
            async with aiofiles.open(program_file, "r", encoding="utf-8") as f:
                program_text = await f.read()
            self.interpreter.load_program(program_text)
            with self.lock:
                self.interpreter.object_counter["CHAIN"] += 1
        except Exception as e:
            raise PdsXIOException(f"Program yükleme hatası: {str(e)}", code="CHAIN001")

    async def op_cont(self, operands: List[Any]) -> None:
        """
        Yürütmeye devam eder.
        :param operands: []
        """
        with self.lock:
            self.interpreter.paused = False
            self.interpreter.object_counter["CONT"] += 1
        log.debug("Yürütme devam ediyor")

    async def op_stop(self, operands: List[Any]) -> None:
        """
        Yürütmeyi durdurur.
        :param operands: []
        """
        with self.lock:
            self.interpreter.paused = True
            self.interpreter.object_counter["STOP"] += 1
        log.debug("Yürütme durduruldu")

    async def op_tron(self, operands: List[Any]) -> None:
        """
        İzleme modunu açar.
        :param operands: []
        """
        with self.lock:
            self.interpreter.trace_mode = True
            self.interpreter.object_counter["TRON"] += 1
        log.debug("İzleme modu açıldı")

    async def op_troff(self, operands: List[Any]) -> None:
        """
        İzleme modunu kapatır.
        :param operands: []
        """
        with self.lock:
            self.interpreter.trace_mode = False
            self.interpreter.object_counter["TROFF"] += 1
        log.debug("İzleme modu kapatıldı")

    async def op_common(self, operands: List[Any]) -> None:
        """
        Değişkenleri paylaşır.
        :param operands: [var_names: str]
        """
        var_names = [v.strip() for v in operands[0].split(",")]
        with self.lock:
            for var in var_names:
                if var in self.interpreter.current_scope():
                    self.interpreter.shared_vars[var].append(self.interpreter.current_scope()[var])
                    self.interpreter.object_counter["COMMON"] += 1
                else:
                    raise PdsXRuntimeError(f"Değişken bulunamadı: {var}", code="COMMON001")
        log.debug(f"Paylaşılan değişkenler: {var_names}")

    async def op_declare(self, operands: List[Any]) -> None:
        """
        Fonksiyon/yordam tanımlar.
        :param operands: [decl_type: str, name: str, params: str, return_type: str]
        """
        decl_type, name, params, return_type = operands
        with self.lock:
            if decl_type.upper() == "SUB":
                self.interpreter.subs[name] = {"params": params}
                self.interpreter.object_counter["SUB"] += 1
            else:
                if return_type not in self.interpreter.data_types:
                    raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}", code="DECLARE001")
                self.interpreter.functions[name] = {"params": params, "return_type": return_type}
                self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"{decl_type} tanımlandı: {name}")

    async def op_def(self, operands: List[Any]) -> None:
        """
        Fonksiyon tanımlar.
        :param operands: [func_name: str, params: str, expr: str]
        """
        func_name, params, expr = operands
        with self.lock:
            self.interpreter.functions[func_name] = {
                "params": params,
                "body": lambda *args: self.interpreter.evaluate_expression(f"{expr}({','.join(map(str, args))})")
            }
            self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"Fonksiyon tanımlandı: {func_name}")

    async def op_exit(self, operands: List[Any]) -> None:
        """
        Döngü/yordamdan çıkar.
        :param operands: [exit_type: str]
        """
        exit_type = operands[0].upper()
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

    async def op_undim(self, operands: List[Any]) -> None:
        """
        Değişkeni kaldırır.
        :param operands: [var_name: str]
        """
        var_name = operands[0]
        with self.lock:
            if var_name in self.interpreter.current_scope():
                value = self.interpreter.current_scope()[var_name]
                del self.interpreter.current_scope()[var_name]
                type_name = next((t for t, v in self.interpreter.data_types.items() if isinstance(value, v)), "VARIABLE")
                self.interpreter.object_counter[type_name] -= 1
                self.interpreter.object_counter["UNDIM"] += 1
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="UNDIM001")
        log.debug(f"Değişken kaldırıldı: {var_name}")

    async def op_setfield(self, operands: List[Any]) -> None:
        """
        Yapı alanını günceller.
        :param operands: [var_name: str, field: str, value_expr: str]
        """
        var_name, field, value_expr = operands
        value = self.interpreter.evaluate_expression(value_expr)
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="SETFIELD001")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                dismutase PdsXTypeError(f"Geçersiz yapı: {var_name}", code="SETFIELD002")
            struct[field] = value
            self.interpreter.object_counter["SETFIELD"] += 1
        log.debug(f"Yapı alanı güncellendi: {var_name}.{field} = {value}")

    async def op_getfield(self, operands: List[Any]) -> None:
        """
        Yapı alanını alır.
        :param operands: [var_name: str, field: str, new_var: str]
        """
        var_name, field, new_var = operands
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

    async def op_addfield(self, operands: List[Any]) -> None:
        """
        Yapıya dinamik alan ekler.
        :param operands: [var_name: str, field: str, type_name: str]
        """
        var_name, field, type_name = operands
        if type_name not in self.interpreter.data_types:
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

    async def op_removefield(self, operands: List[Any]) -> None:
        """
        Yapıdan alan kaldırır.
        :param operands: [var_name: str, field: str]
        """
        var_name, field = operands
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

    async def op_newobj(self, operands: List[Any]) -> None:
        """
        Nesne oluşturur.
        :param operands: [class_name: str, params: str, var_name: str]
        """
        class_name, params, var_name = operands
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

    async def op_countobj(self, operands: List[Any]) -> None:
        """
        Nesne sayısını döndürür.
        :param operands: [type_name: str, var_name: str]
        """
        type_name, var_name = operands
        with self.lock:
            count = self.interpreter.object_counter.get(type_name, 0)
            self.interpreter.current_scope()[var_name] = count
            self.interpreter.object_counter["COUNTOBJ"] += 1
        log.debug(f"Nesne sayıldı: {type_name} = {count}")

    async def op_inspobj(self, operands: List[Any]) -> None:
        """
        Nesne detaylarını alır.
        :param operands: [obj_id: str, var_name: str]
        """
        obj_id, var_name = operands
        with self.lock:
            obj = self.interpreter.object_registry.get(int(obj_id), {})
            if not obj:
                raise PdsXRuntimeError(f"Nesne bulunamadı: {obj_id}", code="INSPOBJ001")
            self.interpreter.current_scope()[var_name] = obj
            self.interpreter.object_counter["INSPOBJ"] += 1
        log.debug(f"Nesne incelendi: {obj_id}")

    async def op_callapi(self, operands: List[Any]) -> None:
        """
        HTTP API isteği yapar (asenkron hata yönetimi güçlendirilmiş).
        :param operands: [url: str, method: str, headers: str, data: str, var_name: str]
        """
        url, method, headers, data, var_name = operands
        try:
            headers = json.loads(headers)
            data = json.loads(data)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for attempt in range(3):  # 3 yeniden deneme
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

    async def op_calldll(self, operands: List[Any]) -> None:
        """
        DLL fonksiyonu çağırır.
        :param operands: [dll_name: str, func_name: str, params: str, var_name: str]
        """
        dll_name, func_name, params, var_name = operands
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

    async def op_sart(self, operands: List[Any]) -> Optional[int]:
        """
        Koşullu boru hattı atlaması.
        :param operands: [condition: str, pipe_id: str, label: str]
        """
        condition, pipe_id, label = operands
        try:
            if self.interpreter.evaluate_expression(condition):
                if pipe_id not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", code="SART001")
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="SART002")
                with self.lock:
                    self.interpreter.object_counter["SART"] += 1
                return self.interpreter.labels[label]
        except Exception as e:
            raise PdsXRuntimeError(f"SART değerlendirme hatası: {str(e)}", code="SART003")
        return None

    async def op_alias(self, operands: List[Any]) -> None:
        """
        İsim değiştirme.
        :param operands: [old_name: str, new_name: str]
        """
        old_name, new_name = operands
        with self.lock:
            if old_name in self.interpreter.current_scope():
                self.interpreter.current_scope()[new_name] = self.interpreter.current_scope()[old_name]
                self.interpreter.object_counter["ALIAS"] += 1
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {old_name}", code="ALIAS001")
        log.debug(f"İsim değiştirildi: {old_name} AS {new_name}")

    async def op_restrict(self, operands: List[Any]) -> None:
        """
        Erişim sınırlandırır.
        :param operands: [scope: str]
        """
        scope = operands[0].upper()
        with self.lock:
            if scope not in ("GLOBAL", "SHARED", "LOCAL"):
                raise PdsXValueError(f"Geçersiz kapsam: {scope}", code="RESTRICT001")
            self.interpreter.restricted_scopes.add(scope)
            self.interpreter.object_counter["RESTRICT"] += 1
        log.debug(f"Kapsam sınırlandırıldı: {scope}")

    async def op_clear_basic(self, operands: List[Any]) -> None:
        """
        Değişkenleri sıfırlar.
        :param operands: [scope: str]
        """
        scope = operands[0].upper()
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

    async def op_listfiles(self, operands: List[Any]) -> None:
        """
        Dosya numaralarını listeler.
        :param operands: [var_name: str]
        """
        var_name = operands[0]
        files = list(self.interpreter.file_handles.keys())
        with self.lock:
            self.interpreter.current_scope()[var_name] = files
            self.interpreter.object_counter["LISTFILES"] += 1
        log.debug(f"Dosyalar listelendi: {files}")

    async def op_listprog(self, operands: List[Any]) -> None:
        """
        Program satırlarını listeler.
        :param operands: [start: str, end: str]
        """
        start, end = operands if operands else (1, len(self.interpreter.program))
        start = int(start)
        end = int(end)
        lines = [self.interpreter.program[i][0] for i in range(start-1, min(end, len(self.interpreter.program)))]
        with self.lock:
            self.interpreter.current_scope()["_PROGRAM_LIST"] = lines
            self.interpreter.object_counter["LISTPROG"] += 1
        for line in lines:
            print(line)
        log.debug(f"Program listelendi: {start} TO {end}")

    async def op_checkfile(self, operands: List[Any]) -> None:
        """
        Dosya durumunu kontrol eder.
        :param operands: [file_num: str, var_name: str]
        """
        file_num, var_name = operands
        file_num = int(file_num)
        status = {"exists": file_num in self.interpreter.file_handles, "open": False}
        if status["exists"]:
            status["open"] = not self.interpreter.file_handles[file_num].closed
        with self.lock:
            self.interpreter.current_scope()[var_name] = status
            self.interpreter.object_counter["CHECKFILE"] += 1
        log.debug(f"Dosya durumu kontrol edildi: #{file_num}, {status}")

    async def op_screen(self, operands: List[Any]) -> None:
        """
        Grafik ekran modunu ayarlar.
        :param operands: [mode: str]
        """
        mode = int(operands[0])
        with self.lock:
            self.interpreter.current_scope()["_SCREEN_MODE"] = mode
            self.interpreter.object_counter["SCREEN"] += 1
        log.debug(f"Ekran modu ayarlandı: {mode}")

    async def op_sound(self, operands: List[Any]) -> None:
        """
        Ses üretir.
        :param operands: [freq: str, duration: str]
        """
        freq, duration = map(int, operands)
        with self.lock:
            self.interpreter.object_counter["SOUND"] += 1
        log.debug(f"Ses üretildi: Frekans={freq}, Süre={duration}")

    async def op_sec_var(self, operands: List[Any]) -> None:
        """
        Değişken erişimini kısıtlar.
        :param operands: [var_name: str]
        """
        var_name = operands[0]
        with self.lock:
            self.interpreter.restricted_vars.add(var_name)
            self.interpreter.object_counter["SEC_VAR"] += 1
        log.debug(f"Değişken kısıtlandı: {var_name}")

    async def op_mon_var(self, operands: List[Any]) -> None:
        """
        Değişken istatistiklerini toplar.
        :param operands: [var_name: str, stat_var: str]
        """
        var_name, stat_var = operands
        stats = {"access_count": 0, "last_access": time.time()}
        if var_name in self.interpreter.current_scope():
            stats["value"] = self.interpreter.current_scope()[var_name]
        with self.lock:
            self.interpreter.current_scope()[stat_var] = stats
            self.interpreter.object_counter["MON_VAR"] += 1
        log.debug(f"Değişken izlendi: {var_name}, İstatistikler: {stats}")

    async def op_convert(self, operands: List[Any]) -> None:
        """
        Tip dönüşümü yapar.
        :param operands: [var_name: str, source_type: str, target_type: str, new_var: str]
        """
        var_name, source_type, target_type, new_var = operands
        if source_type not in self.interpreter.data_types or target_type not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {source_type} veya {target_type}", code="CONVERT001")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="CONVERT002")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.interpreter.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CONVERT"] += 1
            log.debug(f"Tip dönüşümü yapıldı: {var_name} ({source_type}) -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", code="CONVERT003")

    async def op_cast(self, operands: List[Any]) -> None:
        """
        Hızlı tip dönüşümü yapar.
        :param operands: [var_name: str, target_type: str, new_var: str]
        """
        var_name, target_type, new_var = operands
        if target_type not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {target_type}", code="CAST001")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="CAST002")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.interpreter.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CAST"] += 1
            log.debug(f"Hızlı tip dönüşümü yapıldı: {var_name} -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}", code="CAST003")

    async def op_alter_table(self, operands: List[Any]) -> None:
        """
        Veritabanı tablosunu değiştirir.
        :param operands: [table_name: str, column_name: str, column_type: str]
        """
        table_name, column_name, column_type = operands
        with self.lock:
            self.interpreter.object_counter["ALTER_TABLE"] += 1
        log.debug(f"Tablo değiştirildi: {table_name}, Yeni sütun: {column_name} AS {column_type}")

    async def op_create_view(self, operands: List[Any]) -> None:
        """
        Veritabanı görünümü oluşturur.
        :param operands: [view_name: str, query: str]
        """
        view_name, query = operands
        with self.lock:
            self.interpreter.object_counter["CREATE_VIEW"] += 1
        log.debug(f"Görünüm oluşturuldu: {view_name}, Sorgu: {query}")

    async def op_gamma(self, operands: List[Any]) -> None:
        """
        GAMMA fonksiyonunu çalıştırır.
        :param operands: [yapi_name: str, gamma_name: str, params: str, result_var: str]
        """
        yapi_name, gamma_name, params, result_var = operands
        if yapi_name not in self.interpreter.classes:
            raise PdsXRuntimeError(f"YAPI bulunamadı: {yapi_name}", code="GAMMA001")
        yapi_def = self.interpreter.classes[yapi_name]
        if gamma_name not in yapi_def["gamma"]:
            raise PdsXRuntimeError(f"GAMMA fonksiyonu bulunamadı: {gamma_name}", code="GAMMA002")
        gamma = yapi_def["gamma"][gamma_name]
        param_values = [self.interpreter.evaluate_expression(p.strip()) for p in params.split(",") if p.strip()]
        try:
            result = gamma["partial"](*param_values)
            with self.lock:
                self.interpreter.current_scope()[result_var] = result
                self.interpreter.object_counter["GAMMA"] += 1
            log.debug(f"GAMMA yürütüldü: {yapi_name}.{gamma_name} -> {result_var}")
        except Exception as e:
            raise PdsXRuntimeError(f"GAMMA yürütme hatası: {str(e)}", code="GAMMA003")

    async def op_omega(self, operands: List[Any]) -> None:
        """
        OMEGA fonksiyonunu çalıştırır.
        :param operands: [yapi_name: str, omega_name: str, param: str, result_var: str]
        """
        yapi_name, omega_name, param, result_var = operands
        if yapi_name not in self.interpreter.classes:
            raise PdsXRuntimeError(f"YAPI bulunamadı: {yapi_name}", code="OMEGA001")
        yapi_def = self.interpreter.classes[yapi_name]
        if omega_name not in yapi_def["omega"]:
            raise PdsXRuntimeError(f"OMEGA fonksiyonu bulunamadı: {omega_name}", code="OMEGA002")
        omega = yapi_def["omega"][omega_name]
        param_value = self.interpreter.evaluate_expression(param)
        try:
            result = omega["self_apply"](param_value)
            with self.lock:
                self.interpreter.current_scope()[result_var] = result
                self.interpreter.object_counter["OMEGA"] += 1
            log.debug(f"OMEGA yürütüldü: {yapi_name}.{omega_name} -> {result_var}")
        except Exception as e:
            raise PdsXRuntimeError(f"OMEGA yürütme hatası: {str(e)}", code="OMEGA003")

    # SIMD Opcode’ları
    async def op_simd_add(self, operands: List[Any]) -> None:
        """
        SSE2 ile vektör toplama.
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
            raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_ADD", code="SIMD_ADD001")
        data_array = np.array(data, dtype=np.float32)
        result = self.simd_process(data_array, "ADD")
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SIMD_ADD"] += 1
            self.simd_cache[var_name] = result
        log.debug(f"SIMD_ADD: {var_name} = {data_var} + 1")

    async def op_simd_sub(self, operands: List[Any]) -> None:
        """
        SSE2 ile vektör çıkarma.
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
            raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_SUB", code="SIMD_SUB001")
        data_array = np.array(data, dtype=np.float32)
        result = self.simd_process(data_array, "SUB")
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SIMD_SUB"] += 1
            self.simd_cache[var_name] = result
        log.debug(f"SIMD_SUB: {var_name} = {data_var} - 1")

    async def op_simd_mul(self, operands: List[Any]) -> None:
        """
        SSE2 ile vektör çarpma.
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
            raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_MUL", code="SIMD_MUL001")
        data_array = np.array(data, dtype=np.float32)
        result = self.simd_process(data_array, "MUL")
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SIMD_MUL"] += 1
            self.simd_cache[var_name] = result
        log.debug(f"SIMD_MUL: {var_name} = {data_var} * 2")

    async def op_simd_div(self, operands: List[Any]) -> None:
        """
        SSE2 ile vektör bölme.
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
            raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_DIV", code="SIMD_DIV001")
        data_array = np.array(data, dtype=np.float32)
        result = self.simd_process(data_array, "DIV")
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SIMD_DIV"] += 1
            self.simd_cache[var_name] = result
        log.debug(f"SIMD_DIV: {var_name} = {data_var} / 2")

    async def op_simd_cmp(self, operands: List[Any]) -> None:
        """
        SSE2 ile vektör karşılaştırma.
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
            raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_CMP", code="SIMD_CMP001")
        data_array = np.array(data, dtype=np.float32)
        result = self.simd_process(data_array, "CMP")
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["SIMD_CMP"] += 1
            self.simd_cache[var_name] = result
            log.debug(f"SIMD_CMP: {var_name} = {data_var} > 0")
    
        async def op_simd_mov(self, operands: List[Any]) -> None:
            """
            SSE2 ile vektör kopyalama.
            :param operands: [var_name: str, data_var: str]
            """
            var_name, data_var = operands
            data = self.interpreter.current_scope().get(data_var)
            if not isinstance(data, (np.ndarray, list, self.interpreter.data_types["VECTOR"], self.interpreter.data_types["MATRIX"], self.interpreter.data_types["TENSOR"])):
                raise PdsXTypeError(f"Geçersiz veri tipi: {data_var} için SIMD_MOV", code="SIMD_MOV001")
            data_array = np.array(data, dtype=np.float32)
            result = self.simd_process(data_array, "MOV")
            with self.lock:
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["SIMD_MOV"] += 1
                self.simd_cache[var_name] = result
            log.debug(f"SIMD_MOV: {var_name} = {data_var}")

    # Deneysel Özellik Opcode’ları
    async def op_quantum_corr(self, operands: List[Any]) -> None:
        """
        Kuantum korelasyon analizi (SSE2 optimize).
        :param operands: [var_name: str, state1_var: str, state2_var: str]
        """
        var_name, state1_var, state2_var = operands
        state1 = self.interpreter.current_scope().get(state1_var)
        state2 = self.interpreter.current_scope().get(state2_var)
        if not isinstance(state1, (list, self.interpreter.data_types["QUANTUM_STATE"])) or not isinstance(state2, (list, self.interpreter.data_types["QUANTUM_STATE"])):
            raise PdsXTypeError(f"Geçersiz kuantum durumu: {state1_var} veya {state2_var}", code="QUANTUM_CORR001")
        s1 = state1.amplitudes if isinstance(state1, self.interpreter.data_types["QUANTUM_STATE"]) else np.array(state1, dtype=np.complex64)
        s2 = state2.amplitudes if isinstance(state2, self.interpreter.data_types["QUANTUM_STATE"]) else np.array(state2, dtype=np.complex64)
        if s1.shape != s2.shape:
            raise PdsXValueError("Kuantum durum boyutları uyuşmuyor", code="QUANTUM_CORR002")
        corr_matrix = np.outer(s1.conj(), s2).real.astype(np.float32)
        bell_violation = np.abs(corr_matrix).sum() > 2 * np.sqrt(2)
        result = {
            "correlation_matrix": corr_matrix.tolist(),
            "bell_violation": bell_violation,
            "entanglement_score": float(np.linalg.norm(corr_matrix))
        }
        with self.lock:
            self.interpreter.current_scope()[var_name] = result
            self.interpreter.object_counter["QUANTUM_CORR"] += 1
            self.interpreter.object_registry[id(result)] = {
                "type": "QUANTUM_CORR",
                "name": var_name,
                "atom": json.dumps(result)[:100]
            }
        log.debug(f"QUANTUM_CORR: {var_name} = {result}")

    async def op_neural_process(self, operands: List[Any]) -> None:
        """
        Nöral ağ veri işleme (SSE2 optimize).
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (list, self.interpreter.data_types["NEURAL_TENSOR"])):
            raise PdsXTypeError(f"Geçersiz nöral tensor: {data_var}", code="NEURAL_PROCESS001")
        d = data.data if isinstance(data, self.interpreter.data_types["NEURAL_TENSOR"]) else np.array(data, dtype=np.float32).reshape(-1, 10, 1)
        try:
            processed = self.interpreter.neural_model.predict(d, verbose=0)
            result = self.interpreter.data_types["NEURAL_TENSOR"](processed.tolist())
            with self.lock:
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["NEURAL_PROCESS"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "NEURAL_PROCESS",
                    "name": var_name,
                    "atom": str(result)[:100]
                }
            log.debug(f"NEURAL_PROCESS: {var_name} = {processed.shape}")
        except Exception as e:
            raise PdsXRuntimeError(f"Nöral işleme hatası: {str(e)}", code="NEURAL_PROCESS002")

    async def op_chaos_detect(self, operands: List[Any]) -> None:
        """
        Kaotik sistem analizi (Runge-Kutta, SSE2 optimize).
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (list, self.interpreter.data_types["CHAOS_FIELD"])):
            raise PdsXTypeError(f"Geçersiz kaos alanı: {data_var}", code="CHAOS_DETECT001")
        d = data.data if isinstance(data, self.interpreter.data_types["CHAOS_FIELD"]) else np.array(data, dtype=np.float64)
        try:
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
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["CHAOS_DETECT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "CHAOS_DETECT",
                    "name": var_name,
                    "atom": json.dumps(result)[:100]
                }
            log.debug(f"CHAOS_DETECT: {var_name} = {result}")
        except Exception as e:
            raise PdsXRuntimeError(f"Kaotik desen hatası: {str(e)}", code="CHAOS_DETECT002")

    async def op_genetic_opt(self, operands: List[Any]) -> None:
        """
        Genetik algoritmalarla optimizasyon (paralel döngü).
        :param operands: [var_name: str, task: str, params: str]
        """
        var_name, task, params = operands
        params = json.loads(params)
        try:
            pop_size = params.get("pop_size", 100)
            generations = params.get("generations", 50)
            mutation_rate = params.get("mutation_rate", 0.1)
            population = np.random.uniform(-10, 10, pop_size).astype(np.float32)
            def fitness(x):
                return np.sin(x) + np.cos(x)  # Mock fitness fonksiyonu
            for _ in range(generations):
                fitness_scores = np.array([fitness(x) for x in population])
                parents_idx = np.argsort(fitness_scores)[-pop_size//2:]
                parents = population[parents_idx]
                offspring = []
                for _ in range(pop_size - len(parents)):
                    p1, p2 = random.sample(list(parents), 2)
                    child = (p1 + p2) / 2
                    if random.random() < mutation_rate:
                        child += np.random.normal(0, 1)
                    offspring.append(child)
                population = np.concatenate([parents, offspring])
            fitness_scores = np.array([fitness(x) for x in population])
            best_idx = np.argmax(fitness_scores)
            result = {
                "optimal_value": float(population[best_idx]),
                "fitness": float(fitness_scores[best_idx])
            }
            with self.lock:
                self.interpreter.current_scope()[var_name] = result
                self.interpreter.object_counter["GENETIC_OPT"] += 1
                self.interpreter.object_registry[id(result)] = {
                    "type": "GENETIC_OPT",
                    "name": var_name,
                    "atom": json.dumps(result)
                }
            log.debug(f"GENETIC_OPT: {var_name} = {result}")
        except Exception as e:
            raise PdsXRuntimeError(f"Genetik optimizasyon hatası: {str(e)}", code="GENETIC_OPT001")

    async def op_blockchain_check(self, operands: List[Any]) -> None:
        """
        Blockchain tabanlı veri doğrulama (SHA-256, önbellek dostu).
        :param operands: [var_name: str, data_var: str]
        """
        var_name, data_var = operands
        data = self.interpreter.current_scope().get(data_var)
        if not isinstance(data, (dict, self.interpreter.data_types["BLOCKCHAIN_LEDGER"])):
            raise PdsXTypeError(f"Geçersiz blockchain defteri: {data_var}", code="BLOCKCHAIN_CHECK001")
        try:
            d = data.ledger if isinstance(data, self.interpreter.data_types["BLOCKCHAIN_LEDGER"]) else data
            data_str = json.dumps(d, sort_keys=True)
            from hashlib import sha256
            data_hash = sha256(data_str.encode("utf-8")).hexdigest()
            prev_hash = self.interpreter.current_scope().get("_PREV_BLOCK_HASH", "")
            integrity = prev_hash == "" or sha256(prev_hash.encode("utf-8")).hexdigest() == data_hash
            with self.lock:
                self.interpreter.current_scope()[var_name] = data_hash
                self.interpreter.current_scope()["_PREV_BLOCK_HASH"] = data_hash
                self.interpreter.object_counter["BLOCKCHAIN_CHECK"] += 1
                self.interpreter.object_registry[id(data_hash)] = {
                    "type": "BLOCKCHAIN_CHECK",
                    "name": var_name,
                    "atom": data_hash
                }
            log.debug(f"BLOCKCHAIN_CHECK: {var_name} = {data_hash}, Bütünlük: {integrity}")
        except Exception as e:
            raise PdsXRuntimeError(f"Blockchain doğrulama hatası: {str(e)}", code="BLOCKCHAIN_CHECK002")

    async def execute_bytecode(self, bytecode: List[Dict]) -> None:
        """
        Bayt kodunu yürütür (Cython optimize).
        :param bytecode: Bayt kodu talimat listesi, her talimat {"opcode": str, "operands": List[Any]} içerir.
        """
        self.program_counter = 0
        while self.program_counter < len(bytecode) and self.interpreter.running:
            instruction = bytecode[self.program_counter]
            opcode = instruction["opcode"]
            operands = instruction.get("operands", [])
            
            # Opcode önbelleklemesi
            handler = self.opcode_cache.get(opcode, self.opcode_table.get(opcode))
            if handler is None:
                raise PdsXSyntaxError(f"Bilinmeyen bayt kodu: {opcode}", code="EXEC001")
            
            try:
                result = await handler(operands)
                if isinstance(result, int):
                    self.program_counter = result  # GOTO, GOSUB, SART, NEXT gibi atlamalar
                else:
                    self.program_counter += 1
                
                # İzleme modu
                if self.interpreter.trace_mode:
                    log.debug(f"PC: {self.program_counter}, Opcode: {opcode}, Operands: {operands}, Scope: {self.interpreter.current_scope()}")
                
                # Bellek temizliği ve önbellek yönetimi
                if len(self.simd_cache) > 1024:  # Önbellek sınırını aşarsa temizle
                    self.simd_cache.clear()
                if len(self.opcode_cache) > 2048:
                    self.opcode_cache.clear()
                
            except PdsXException as e:
                await self.interpreter.exception_manager.handle_error(e)
                raise
            except Exception as e:
                await self.interpreter.exception_manager.handle_error(e)
                raise PdsXRuntimeError(f"Bayt kodu yürütme hatası: {str(e)}", code="EXEC002")
        
        # Yürütme sonrası temizlik
        with self.lock:
            self.stack.clear()  # Yığını sıfırla
            self.call_stack.clear()  # Çağrı yığınını sıfırla
            if not self.interpreter.paused:
                self.interpreter.running = False
            self._cleanup_memory()
        log.debug("Bayt kodu yürütme tamamlandı")

if __name__ == "__main__":
    print("bytecode_engine.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")