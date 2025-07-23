# bytecode_engine.py - PDS-X BASIC v15 Bayt Kodu Yürütme Motoru (WD65816 Destekli)
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
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache
import numpy as np
import pandas as pd
from pdsx_exception2 import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError, PdsXTypeError,
    PdsXValueError, PdsXIOException, PdsXNetworkError
)

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_bytecode_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bytecode_engine")

class BytecodeEngine:
    """PDS-X BASIC v15 bayt kodu yürütme motoru, WD65816 desteği ile."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.metadata: Dict = {
            "bytecode_engine": {
                "version": "1.5.0",
                "dependencies": ["numpy", "pandas", "aiofiles"]
            }
        }
        # WD65816 Kayıtları
        self.registers = {
            "A": 0,  # 16-bit akümülatör
            "X": 0,  # X indeksi
            "Y": 0,  # Y indeksi
            "S": 0x1FF,  # Yığın işaretçisi
            "D": 0,  # Doğrudan sayfa
            "DBR": 0,  # Veri bankası
            "P": 0x30  # Durum bayrakları (N, V, M, X, D, I, Z, C)
        }
        self.memory = [0] * 0x1000000  # 16 MB adres alanı
        self.opcode_table: Dict[str, Callable] = {
            # PDS-X BASIC Komutları
            "LET": self.op_let,
            "IF": self.op_if,
            "FOR": self.op_for,
            "FOREACH": self.op_foreach,
            "WHILE": self.op_while,
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
            # WD65816 Komutları
            "LDA_ABS": self.op_lda_abs,
            "LDA_LONG": self.op_lda_long,
            "LDA_IND": self.op_lda_ind,
            "STA_ABS": self.op_sta_abs,
            "STA_LONG": self.op_sta_long,
            "STA_IND": self.op_sta_ind,
            "ADC_ABS": self.op_adc_abs,
            "SBC_ABS": self.op_sbc_abs,
            "CMP_ABS": self.op_cmp_abs,
            "JMP_ABS": self.op_jmp_abs,
            "JSR_ABS": self.op_jsr_abs,
            "RTS": self.op_rts,
            "BRK": self.op_brk,
            "MVN": self.op_mvn,
            "MVP": self.op_mvp,
            "PEA": self.op_pea,
            "PEI": self.op_pei,
            "PER": self.op_per,
            "TCD": self.op_tcd,
            "TDC": self.op_tdc,
            "BRA": self.op_bra,
            "BRL": self.op_brl,
            "SEP": self.op_sep,
            "REP": self.op_rep
        }
        self.stack: List[Any] = []
        self.program_counter: int = 0
        self.call_stack: List[Dict] = []
        self.loop_stack: List[Dict] = []

    # WD65816 Yardımcı Fonksiyonları
    def get_address(self, mode: str, operand: Any) -> int:
        """Adresleme moduna göre adres hesaplar."""
        if mode == "ABS":
            return operand & 0xFFFF | (self.registers["DBR"] << 16)
        elif mode == "LONG":
            return operand & 0xFFFFFF
        elif mode == "IND":
            addr = operand & 0xFFFF | (self.registers["DBR"] << 16)
            return (self.memory[addr] | (self.memory[addr + 1] << 8) | (self.registers["DBR"] << 16)) & 0xFFFFFF
        elif mode == "IND_X":
            addr = (operand + self.registers["X"]) & 0xFFFF | (self.registers["DBR"] << 16)
            return (self.memory[addr] | (self.memory[addr + 1] << 8) | (self.registers["DBR"] << 16)) & 0xFFFFFF
        elif mode == "IND_Y":
            addr = operand & 0xFFFF | (self.registers["DBR"] << 16)
            base = (self.memory[addr] | (self.memory[addr + 1] << 8) | (self.registers["DBR"] << 16)) & 0xFFFFFF
            return (base + self.registers["Y"]) & 0xFFFFFF
        elif mode == "STACK_REL":
            return (self.registers["S"] + operand) & 0xFFFF
        else:
            raise PdsXValueError(f"Geçersiz adresleme modu: {mode}")

    def update_flags(self, value: int) -> None:
        """Durum bayraklarını günceller."""
        self.registers["P"] &= ~0x82  # Z ve N bayraklarını sıfırla
        if value == 0:
            self.registers["P"] |= 0x02  # Z bayrağı
        if value & 0x8000:
            self.registers["P"] |= 0x80  # N bayrağı

    # PDS-X BASIC Opcode’ları
    async def op_let(self, operands: List[Any]) -> None:
        """Değişkene değer atar."""
        var_names, expr = operands
        try:
            values = self.interpreter.evaluate_expression(expr)
            var_names = [v.strip() for v in var_names.split(",")]
            with self.lock:
                if isinstance(values, (list, tuple, np.ndarray, pd.Series)):
                    if len(var_names) != len(values):
                        raise PdsXValueError(f"Değişken sayısı ({len(var_names)}) ile değer sayısı ({len(values)}) uyuşmuyor")
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
            raise PdsXRuntimeError(f"Atama hatası: {str(e)}")

    async def op_if(self, operands: List[Any]) -> Optional[int]:
        """Koşullu yürütme."""
        condition, then_block, else_block = operands
        try:
            if self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(then_block.strip())
            elif else_block:
                await self.interpreter.execute_command(else_block.strip())
            with self.lock:
                self.interpreter.object_counter["IF"] += 1
        except Exception as e:
            raise PdsXRuntimeError(f"IF koşul değerlendirme hatası: {str(e)}")
        return None

    async def op_for(self, operands: List[Any]) -> Optional[int]:
        """Sayısal döngü."""
        var_name, start, end, step, body = operands
        start, end = float(start), float(end)
        step = float(step) if step else 1.0
        if step == 0:
            raise PdsXValueError("Adım sıfır olamaz")
        with self.lock:
            self.interpreter.current_scope()[var_name] = start
            self.loop_stack.append({"type": "FOR", "var": var_name, "index": 0, "history": [start]})
            self.interpreter.object_counter["FOR"] += 1
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
                    current = self.loop_stack[-1]["history"][index]
                    continue
                current += step
                index += 1
                with self.lock:
                    self.loop_stack[-1]["index"] = index
                    self.loop_stack[-1]["history"].append(current)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR döngü hatası: {str(e)}")
        finally:
            with self.lock:
                self.loop_stack.pop()
        return None

    async def op_foreach(self, operands: List[Any]) -> Optional[int]:
        """Koleksiyon döngüsü."""
        var_name, collection_name, body = operands
        collection = self.interpreter.current_scope().get(collection_name)
        if not isinstance(collection, (list, dict, set, tuple, pd.DataFrame, np.ndarray)):
            raise PdsXTypeError(f"Geçersiz koleksiyon: {collection_name}")
        with self.lock:
            self.loop_stack.append({"type": "FOREACH", "var": var_name, "index": 0, "history": []})
            self.interpreter.object_counter["FOREACH"] += 1
        try:
            items = list(collection.items() if isinstance(collection, dict) else collection)
            if isinstance(collection, pd.DataFrame):
                items = [row.to_dict() for _, row in collection.iterrows()]
            elif isinstance(collection, np.ndarray):
                items = collection.tolist()
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
                    self.loop_stack[-1]["index"] = index
                    self.loop_stack[-1]["history"].append(item)
        except Exception as e:
            raise PdsXRuntimeError(f"FOR EACH döngü hatası: {str(e)}")
        finally:
            with self.lock:
                self.loop_stack.pop()
        return None

    async def op_while(self, operands: List[Any]) -> Optional[int]:
        """Koşullu döngü."""
        condition, body = operands
        with self.lock:
            self.loop_stack.append({"type": "WHILE"})
            self.interpreter.object_counter["WHILE"] += 1
        try:
            while self.interpreter.evaluate_expression(condition):
                await self.interpreter.execute_command(body)
                if "EXIT WHILE" in body.upper():
                    break
        except Exception as e:
            raise PdsXRuntimeError(f"WHILE döngü hatası: {str(e)}")
        finally:
            with self.lock:
                self.loop_stack.pop()
        return None

    async def op_dim(self, operands: List[Any]) -> None:
        """Değişken tanımlar (tek satırlık ve kapanışlı)."""
        var_name, type_name, initial_value, size = operands
        if type_name not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}")
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
                value = self.interpreter.data_types["SKALER"](0.0)
            elif type_name == "VECTOR":
                size = int(size) if size else 10
                value = self.interpreter.data_types["VECTOR"]([0.0] * size)
            elif type_name == "MATRIX":
                size = int(size) if size else 10
                value = self.interpreter.data_types["MATRIX"]([[0.0] * size for _ in range(size)])
            elif type_name == "TENSOR":
                size = int(size) if size else 10
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
            raise PdsXRuntimeError(f"DIM tanımlama hatası: {str(e)}")

    async def op_end(self, operands: List[Any]) -> None:
        """Blok, veri yapısı veya program sonlandırır."""
        block_type = operands[0] if operands else None
        with self.lock:
            if block_type:
                if block_type not in self.interpreter.data_types and block_type not in ("FOR", "WHILE", "IF", "CLASS", "YAPI", "SELECT"):
                    raise PdsXSyntaxError(f"Geçersiz END tipi: {block_type}")
                if block_type in ("FOR", "WHILE"):
                    if not self.loop_stack or self.loop_stack[-1]["type"] != block_type:
                        raise PdsXRuntimeError(f"Kapatılacak {block_type} bloğu yok")
                    self.loop_stack.pop()
            else:
                self.interpreter.running = False
            self.interpreter.object_counter["END"] += 1
        log.debug(f"Blok/Program sonlandırıldı: {block_type or 'PROGRAM'}")

    async def op_class(self, operands: List[Any]) -> None:
        """Sınıf tanımlar."""
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
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}")
                    class_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}")
            elif line.upper().startswith("PROP "):
                prop_match = re.match(r"PROP\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if prop_match:
                    prop_name, prop_type = prop_match.groups()
                    if prop_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}")
                    class_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"PROP tanımı hatalı: {line}")
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[class_name] = class_def
            self.interpreter.object_counter["CLASS"] += 1
        log.debug(f"Sınıf tanımlandı: {class_name}")

    async def op_yapi(self, operands: List[Any]) -> None:
        """Nesne tabanlı sınıf oluşturucu (YAPI)."""
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
                    raise PdsXSyntaxError(f"SUB tanımı hatalı: {line}")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}")
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"FUNCTION tanımı hatalı: {line}")
            elif line.upper().startswith("GAMMA "):
                gamma_match = re.match(r"GAMMA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+GAMMA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if gamma_match:
                    gamma_name, params, return_type, gamma_body = gamma_match.groups()
                    if return_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}")
                    yapi_def["gamma"][gamma_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": gamma_body,
                        "partial": lambda *args: lambda *rest: self.interpreter.evaluate_expression(f"{gamma_body}({','.join(map(str, args + rest))})")
                    }
                    self.interpreter.object_counter["GAMMA"] += 1
                    i += gamma_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"GAMMA tanımı hatalı: {line}")
            elif line.upper().startswith("OMEGA "):
                omega_match = re.match(r"OMEGA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+OMEGA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if omega_match:
                    omega_name, params, return_type, omega_body = omega_match.groups()
                    if return_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}")
                    yapi_def["omega"][omega_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": omega_body,
                        "self_apply": lambda x: self.interpreter.evaluate_expression(f"{omega_body}({x})")
                    }
                    self.interpreter.object_counter["OMEGA"] += 1
                    i += omega_body.count("\n") + 2
                else:
                    raise PdsXSyntaxError(f"OMEGA tanımı hatalı: {line}")
            elif line.upper().startswith("DIM "):
                dim_match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if dim_match:
                    prop_name, prop_type = dim_match.groups()
                    if prop_type not in self.interpreter.data_types:
                        raise PdsXTypeError(f"Geçersiz özellik tipi: {prop_type}")
                    yapi_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type] += 1
                else:
                    raise PdsXSyntaxError(f"DIM tanımı hatalı: {line}")
                i += 1
            else:
                i += 1
        with self.lock:
            self.interpreter.classes[yapi_name] = yapi_def
            self.interpreter.object_counter["YAPI"] += 1
        log.debug(f"YAPI tanımlandı: {yapi_name}")

    async def op_print(self, operands: List[Any]) -> None:
        """Ekrana yazdırır."""
        expr = operands[0]
        try:
            value = self.interpreter.evaluate_expression(expr)
            print(value)
            with self.lock:
                self.interpreter.object_counter["PRINT"] += 1
        except Exception as e:
            raise PdsXRuntimeError(f"PRINT değerlendirme hatası: {str(e)}")

    async def op_goto(self, operands: List[Any]) -> int:
        """Etikete atlar."""
        label = operands[0]
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}")
        with self.lock:
            self.interpreter.object_counter["GOTO"] += 1
        return self.interpreter.labels[label]

    async def op_gosub(self, operands: List[Any]) -> int:
        """Alt yordama atlar."""
        label = operands[0]
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}")
        with self.lock:
            self.call_stack.append({"return_pc": self.program_counter + 1})
            self.interpreter.object_counter["GOSUB"] += 1
        return self.interpreter.labels[label]

    async def op_return(self, operands: List[Any]) -> int:
        """Alt yordamdan döner."""
        if not self.call_stack:
            raise PdsXRuntimeError("Geri dönülecek yordam yok")
        with self.lock:
            return_pc = self.call_stack.pop()["return_pc"]
            self.interpreter.object_counter["RETURN"] += 1
        return return_pc

    async def op_select_case(self, operands: List[Any]) -> None:
        """Çoklu koşullu yapı."""
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
        """Veri tanımlar."""
        values = operands[0].split(",")
        with self.lock:
            self.interpreter.data_list.extend([v.strip() for v in values])
            self.interpreter.object_counter["DATA"] += len(values)
        log.debug(f"Veri tanımlandı: {values}")

    async def op_read(self, operands: List[Any]) -> None:
        """Veri okur."""
        var_names = [v.strip() for v in operands[0].split(",")]
        with self.lock:
            for var in var_names:
                if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                    raise PdsXRuntimeError("Veri listesi sonu")
                value = self.interpreter.data_list[self.interpreter.data_pointer]
                self.interpreter.current_scope()[var] = value
                self.interpreter.data_pointer += 1
                self.interpreter.object_counter["VARIABLE"] += 1
        log.debug(f"Veri okundu: {var_names}")

    async def op_restore(self, operands: List[Any]) -> None:
        """Veri işaretçisini sıfırlar."""
        with self.lock:
            self.interpreter.data_pointer = 0
            self.interpreter.object_counter["RESTORE"] += 1
        log.debug("Veri işaretçisi sıfırlandı")

    async def op_chain(self, operands: List[Any]) -> None:
        """Yeni programı zincirler."""
        program_file = operands[0]
        try:
            async with aiofiles.open(program_file, "r", encoding="utf-8") as f:
                program_text = await f.read()
            self.interpreter.load_program(program_text)
            with self.lock:
                self.interpreter.object_counter["CHAIN"] += 1
        except Exception as e:
            raise PdsXIOException(f"Program yükleme hatası: {str(e)}")

    async def op_cont(self, operands: List[Any]) -> None:
        """Yürütmeye devam eder."""
        with self.lock:
            self.interpreter.paused = False
            self.interpreter.object_counter["CONT"] += 1
        log.debug("Yürütme devam ediyor")

    async def op_stop(self, operands: List[Any]) -> None:
        """Yürütmeyi durdurur."""
        with self.lock:
            self.interpreter.paused = True
            self.interpreter.object_counter["STOP"] += 1
        log.debug("Yürütme durduruldu")

    async def op_tron(self, operands: List[Any]) -> None:
        """İzleme modunu açar."""
        with self.lock:
            self.interpreter.trace_mode = True
            self.interpreter.object_counter["TRON"] += 1
        log.debug("İzleme modu açıldı")

    async def op_troff(self, operands: List[Any]) -> None:
        """İzleme modunu kapatır."""
        with self.lock:
            self.interpreter.trace_mode = False
            self.interpreter.object_counter["TROFF"] += 1
        log.debug("İzleme modu kapatıldı")

    async def op_common(self, operands: List[Any]) -> None:
        """Değişkenleri paylaşır."""
        var_names = [v.strip() for v in operands[0].split(",")]
        with self.lock:
            for var in var_names:
                if var in self.interpreter.current_scope():
                    self.interpreter.shared_vars[var].append(self.interpreter.current_scope()[var])
                    self.interpreter.object_counter["COMMON"] += 1
        log.debug(f"Paylaşılan değişkenler: {var_names}")

    async def op_declare(self, operands: List[Any]) -> None:
        """Fonksiyon/yordam tanımlar."""
        decl_type, name, params, return_type = operands
        with self.lock:
            if decl_type.upper() == "SUB":
                self.interpreter.subs[name] = {"params": params}
                self.interpreter.object_counter["SUB"] += 1
            else:
                if return_type not in self.interpreter.data_types:
                    raise PdsXTypeError(f"Geçersiz dönüş tipi: {return_type}")
                self.interpreter.functions[name] = {"params": params, "return_type": return_type}
                self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"{decl_type} tanımlandı: {name}")

    async def op_def(self, operands: List[Any]) -> None:
        """Fonksiyon tanımlar."""
        func_name, params, expr = operands
        with self.lock:
            self.interpreter.functions[func_name] = {
                "params": params,
                "body": lambda *args: self.interpreter.evaluate_expression(f"{expr}({','.join(map(str, args))})")
            }
            self.interpreter.object_counter["FUNCTION"] += 1
        log.debug(f"Fonksiyon tanımlandı: {func_name}")

    async def op_exit(self, operands: List[Any]) -> None:
        """Döngü/yordamdan çıkar."""
        exit_type = operands[0].upper()
        with self.lock:
            if exit_type in ("FOR", "WHILE"):
                if not self.loop_stack or self.loop_stack[-1]["type"] != exit_type:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} döngüsü yok")
                self.loop_stack.pop()
            else:
                if not self.call_stack:
                    raise PdsXRuntimeError(f"Çıkılacak {exit_type} yordamı yok")
                self.call_stack.pop()
            self.interpreter.object_counter["EXIT"] += 1
        log.debug(f"{exit_type}’den çıkıldı")

    async def op_undim(self, operands: List[Any]) -> None:
        """Değişkeni kaldırır."""
        var_name = operands[0]
        with self.lock:
            if var_name in self.interpreter.current_scope():
                value = self.interpreter.current_scope()[var_name]
                del self.interpreter.current_scope()[var_name]
                type_name = next((t for t, v in self.interpreter.data_types.items() if isinstance(value, v)), "VARIABLE")
                self.interpreter.object_counter[type_name] -= 1
                self.interpreter.object_counter["UNDIM"] += 1
        log.debug(f"Değişken kaldırıldı: {var_name}")

    async def op_setfield(self, operands: List[Any]) -> None:
        """Yapı alanını günceller."""
        var_name, field, value_expr = operands
        value = self.interpreter.evaluate_expression(value_expr)
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}")
            struct[field] = value
            self.interpreter.object_counter["SETFIELD"] += 1
        log.debug(f"Yapı alanı güncellendi: {var_name}.{field} = {value}")

    async def op_getfield(self, operands: List[Any]) -> None:
        """Yapı alanını alır."""
        var_name, field, new_var = operands
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}")
            value = struct.get(field)
            self.interpreter.current_scope()[new_var] = value
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_counter["GETFIELD"] += 1
        log.debug(f"Yapı alanı alındı: {new_var} = {var_name}.{field}")

    async def op_addfield(self, operands: List[Any]) -> None:
        """Yapıya dinamik alan ekler."""
        var_name, field, type_name = operands
        if type_name not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz veri tipi: {type_name}")
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}")
            struct[field] = None
            self.interpreter.object_counter[type_name] += 1
            self.interpreter.object_counter["ADDFIELD"] += 1
        log.debug(f"Yapıya alan eklendi: {var_name}.{field} AS {type_name}")

    async def op_removefield(self, operands: List[Any]) -> None:
        """Yapıdan alan kaldırır."""
        var_name, field = operands
        with self.lock:
            if var_name not in self.interpreter.current_scope():
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
            struct = self.interpreter.current_scope()[var_name]
            if not isinstance(struct, dict):
                raise PdsXTypeError(f"Geçersiz yapı: {var_name}")
            if field in struct:
                struct.pop(field)
                self.interpreter.object_counter["REMOVEFIELD"] += 1
        log.debug(f"Yapıdan alan kaldırıldı: {var_name}.{field}")

    async def op_newobj(self, operands: List[Any]) -> None:
        """Nesne oluşturur."""
        class_name, params, var_name = operands
        if class_name not in self.interpreter.classes:
            raise PdsXRuntimeError(f"Sınıf bulunamadı: {class_name}")
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
                raise PdsXValueError(f"Geçersiz parametre sayısı: {class_name}.Init")
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
        """Nesne sayısını döndürür."""
        type_name, var_name = operands
        with self.lock:
            count = self.interpreter.object_counter.get(type_name, 0)
            self.interpreter.current_scope()[var_name] = count
            self.interpreter.object_counter["COUNTOBJ"] += 1
        log.debug(f"Nesne sayıldı: {type_name} = {count}")

    async def op_inspobj(self, operands: List[Any]) -> None:
        """Nesne detaylarını alır."""
        obj_id, var_name = operands
        with self.lock:
            obj = self.interpreter.object_registry.get(int(obj_id), {})
            if not obj:
                raise PdsXRuntimeError(f"Nesne bulunamadı: {obj_id}")
            self.interpreter.current_scope()[var_name] = obj
            self.interpreter.object_counter["INSPOBJ"] += 1
        log.debug(f"Nesne incelendi: {obj_id}")

    async def op_callapi(self, operands: List[Any]) -> None:
        """HTTP API isteği yapar."""
        url, method, headers, data, var_name = operands
        try:
            headers = json.loads(headers)
            data = json.loads(data)
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=data) as resp:
                        response = await resp.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as resp:
                        response = await resp.json()
                else:
                    raise PdsXValueError(f"Desteklenmeyen metod: {method}")
            with self.lock:
                self.interpreter.current_scope()[var_name] = response
                self.interpreter.object_counter["CALLAPI"] += 1
        except Exception as e:
            raise PdsXNetworkError(f"API çağrısı hatası: {str(e)}")

    async def op_calldll(self, operands: List[Any]) -> None:
        """DLL fonksiyonu çağırır."""
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
            raise PdsXRuntimeError(f"DLL çağrısı hatası: {str(e)}")

    async def op_sart(self, operands: List[Any]) -> Optional[int]:
        """Koşullu boru hattı atlaması."""
        condition, pipe_id, label = operands
        try:
            if self.interpreter.evaluate_expression(condition):
                if pipe_id not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}")
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {label}")
                with self.lock:
                    self.interpreter.object_counter["SART"] += 1
                return self.interpreter.labels[label]
        except Exception as e:
            raise PdsXRuntimeError(f"SART değerlendirme hatası: {str(e)}")
        return None

    async def op_alias(self, operands: List[Any]) -> None:
        """İsim değiştirme."""
        old_name, new_name = operands
        with self.lock:
            if old_name in self.interpreter.current_scope():
                self.interpreter.current_scope()[new_name] = self.interpreter.current_scope()[old_name]
                self.interpreter.object_counter["ALIAS"] += 1
            else:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {old_name}")
        log.debug(f"İsim değiştirildi: {old_name} AS {new_name}")

    async def op_restrict(self, operands: List[Any]) -> None:
        """Erişim sınırlandırır."""
        scope = operands[0].upper()
        with self.lock:
            if scope not in ("GLOBAL", "SHARED", "LOCAL"):
                raise PdsXValueError(f"Geçersiz kapsam: {scope}")
            self.interpreter.restricted_scopes.add(scope)
            self.interpreter.object_counter["RESTRICT"] += 1
        log.debug(f"Kapsam sınırlandırıldı: {scope}")

    async def op_clear_basic(self, operands: List[Any]) -> None:
        """Değişkenleri sıfırlar."""
        scope = operands[0].upper()
        with self.lock:
            if scope == "GLOBAL":
                self.interpreter.global_vars.clear()
            elif scope == "SHARED":
                self.interpreter.shared_vars.clear()
            elif scope == "LOCAL":
                self.interpreter.current_scope().clear()
            self.interpreter.object_counter["CLEAR_BASIC"] += 1
        log.debug(f"{scope} değişkenler sıfırlandı")

    async def op_listfiles(self, operands: List[Any]) -> None:
        """Dosya numaralarını listeler."""
        var_name = operands[0]
        files = list(self.interpreter.file_handles.keys())
        with self.lock:
            self.interpreter.current_scope()[var_name] = files
            self.interpreter.object_counter["LISTFILES"] += 1
        log.debug(f"Dosyalar listelendi: {files}")

    async def op_listprog(self, operands: List[Any]) -> None:
        """Program satırlarını listeler."""
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
        """Dosya durumunu kontrol eder."""
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
        """Grafik ekran modunu ayarlar."""
        mode = int(operands[0])
        with self.lock:
            self.interpreter.current_scope()["_SCREEN_MODE"] = mode
            self.interpreter.object_counter["SCREEN"] += 1
        log.debug(f"Ekran modu ayarlandı: {mode}")

    async def op_sound(self, operands: List[Any]) -> None:
        """Ses üretir."""
        freq, duration = map(int, operands)
        with self.lock:
            self.interpreter.object_counter["SOUND"] += 1
        log.debug(f"Ses üretildi: Frekans={freq}, Süre={duration}")

    async def op_sec_var(self, operands: List[Any]) -> None:
        """Değişken erişimini kısıtlar."""
        var_name = operands[0]
        with self.lock:
            self.interpreter.restricted_vars.add(var_name)
            self.interpreter.object_counter["SEC_VAR"] += 1
        log.debug(f"Değişken kısıtlandı: {var_name}")

    async def op_mon_var(self, operands: List[Any]) -> None:
        """Değişken istatistiklerini toplar."""
        var_name, stat_var = operands
        stats = {"access_count": 0, "last_access": time.time()}
        if var_name in self.interpreter.current_scope():
            stats["value"] = self.interpreter.current_scope()[var_name]
        with self.lock:
            self.interpreter.current_scope()[stat_var] = stats
            self.interpreter.object_counter["MON_VAR"] += 1
        log.debug(f"Değişken izlendi: {var_name}, İstatistikler: {stats}")

    async def op_convert(self, operands: List[Any]) -> None:
        """Tip dönüşümü yapar."""
        var_name, source_type, target_type, new_var = operands
        if source_type not in self.interpreter.data_types or target_type not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {source_type} veya {target_type}")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.interpreter.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CONVERT"] += 1
            log.debug(f"Tip dönüşümü yapıldı: {var_name} ({source_type}) -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}")

    async def op_cast(self, operands: List[Any]) -> None:
        """Hızlı tip dönüşümü yapar."""
        var_name, target_type, new_var = operands
        if target_type not in self.interpreter.data_types:
            raise PdsXTypeError(f"Geçersiz tip: {target_type}")
        if var_name not in self.interpreter.current_scope():
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}")
        value = self.interpreter.current_scope()[var_name]
        try:
            converted = self.interpreter.data_types[target_type](value)
            with self.lock:
                self.interpreter.current_scope()[new_var] = converted
                self.interpreter.object_counter[target_type] += 1
                self.interpreter.object_counter["CAST"] += 1
            log.debug(f"Hızlı tip dönüşümü yapıldı: {var_name} -> {new_var} ({target_type})")
        except Exception as e:
            raise PdsXRuntimeError(f"Tip dönüşüm hatası: {str(e)}")

    async def op_alter_table(self, operands: List[Any]) -> None:
        """Veritabanı tablosunu değiştirir."""
        table_name, column_name, column_type = operands
        with self.lock:
            self.interpreter.object_counter["ALTER_TABLE"] += 1
        log.debug(f"Tablo değiştirildi: {table_name}, Yeni sütun: {column_name} AS {column_type}")

    async def op_create_view(self, operands: List[Any]) -> None:
        """Veritabanı görünümü oluşturur."""
        view_name, query = operands
        with self.lock:
            self.interpreter.object_counter["CREATE_VIEW"] += 1
        log.debug(f"Görünüm oluşturuldu: {view_name}, Sorgu: {query}")

    # WD65816 Opcode’ları
    async def op_lda_abs(self, operands: List[Any]) -> None:
        """Mutlak adresle A’ya yükle."""
        addr = self.get_address("ABS", operands[0])
        self.registers["A"] = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        self.update_flags(self.registers["A"])
        log.debug(f"LDA_ABS: A = {self.registers['A']} from ${addr:06X}")

    async def op_lda_long(self, operands: List[Any]) -> None:
        """Uzun adresle A’ya yükle."""
        addr = self.get_address("LONG", operands[0])
        self.registers["A"] = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        self.update_flags(self.registers["A"])
        log.debug(f"LDA_LONG: A = {self.registers['A']} from ${addr:06X}")

    async def op_lda_ind(self, operands: List[Any]) -> None:
        """Dolaylı adresle A’ya yükle."""
        addr = self.get_address("IND", operands[0])
        self.registers["A"] = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        self.update_flags(self.registers["A"])
        log.debug(f"LDA_IND: A = {self.registers['A']} from ${addr:06X}")

    async def op_sta_abs(self, operands: List[Any]) -> None:
        """A’yı mutlak adrese sakla."""
        addr = self.get_address("ABS", operands[0])
        self.memory[addr] = self.registers["A"] & 0xFF
        self.memory[addr + 1] = (self.registers["A"] >> 8) & 0xFF
        log.debug(f"STA_ABS: ${addr:06X} = {self.registers['A']}")

    async def op_sta_long(self, operands: List[Any]) -> None:
        """A’yı uzun adrese sakla."""
        addr = self.get_address("LONG", operands[0])
        self.memory[addr] = self.registers["A"] & 0xFF
        self.memory[addr + 1] = (self.registers["A"] >> 8) & 0xFF
        log.debug(f"STA_LONG: ${addr:06X} = {self.registers['A']}")

    async def op_sta_ind(self, operands: List[Any]) -> None:
        """A’yı dolaylı adrese sakla."""
        addr = self.get_address("IND", operands[0])
        self.memory[addr] = self.registers["A"] & 0xFF
        self.memory[addr + 1] = (self.registers["A"] >> 8) & 0xFF
        log.debug(f"STA_IND: ${addr:06X} = {self.registers['A']}")

    async def op_adc_abs(self, operands: List[Any]) -> None:
        """Mutlak adresten toplama."""
        addr = self.get_address("ABS", operands[0])
        value = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        carry = (self.registers["P"] & 0x01)
        result = (self.registers["A"] + value + carry) & 0xFFFF
        self.registers["P"] &= ~0x41  # C ve V bayraklarını sıfırla
        if result > 0xFFFF:
            self.registers["P"] |= 0x01  # C bayrağı
        if ((self.registers["A"] ^ result) & (value ^ result) & 0x8000):
            self.registers["P"] |= 0x40  # V bayrağı
        self.registers["A"] = result
        self.update_flags(self.registers["A"])
        log.debug(f"ADC_ABS: A = {self.registers['A']} from ${addr:06X}")

    async def op_sbc_abs(self, operands: List[Any]) -> None:
        """Mutlak adresten çıkarma."""
        addr = self.get_address("ABS", operands[0])
        value = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        carry = (self.registers["P"] & 0x01)
        result = (self.registers["A"] - value - (1 - carry)) & 0xFFFF
        self.registers["P"] &= ~0x41
        if result <= 0xFFFF:
            self.registers["P"] |= 0x01
        if ((self.registers["A"] ^ value) & (self.registers["A"] ^ result) & 0x8000):
            self.registers["P"] |= 0x40
        self.registers["A"] = result
        self.update_flags(self.registers["A"])
        log.debug(f"SBC_ABS: A = {self.registers['A']} from ${addr:06X}")

    async def op_cmp_abs(self, operands: List[Any]) -> None:
        """Mutlak adresteki değeri A ile karşılaştır."""
        addr = self.get_address("ABS", operands[0])
        value = (self.memory[addr] | (self.memory[addr + 1] << 8)) & 0xFFFF
        result = (self.registers["A"] - value) & 0xFFFF
        self.registers["P"] &= ~0x03
        if result == 0:
            self.registers["P"] |= 0x02
        if self.registers["A"] >= value:
            self.registers["P"] |= 0x01
        self.update_flags(result)
        log.debug(f"CMP_ABS: A = {self.registers['A']} vs ${addr:06X}")

    async def op_jmp_abs(self, operands: List[Any]) -> int:
        """Mutlak adrese atla."""
        addr = self.get_address("ABS", operands[0])
        self.program_counter = addr
        return self.program_counter

    async def op_jsr_abs(self, operands: List[Any]) -> int:
        """Mutlak adrese alt yordam çağır."""
        addr = self.get_address("ABS", operands[0])
        with self.lock:
            self.call_stack.append({"return_pc": self.program_counter + 1})
        self.program_counter = addr
        return self.program_counter

    async def op_rts(self, operands: List[Any]) -> int:
        """Alt yordamdan döner."""
        if not self.call_stack:
            raise PdsXRuntimeError("Geri dönülecek yordam yok")
        with self.lock:
            return_pc = self.call_stack.pop()["return_pc"]
        return return_pc

    async def op_brk(self, operands: List[Any]) -> None:
        """Kesme tetikler."""
        with self.lock:
            self.interpreter.paused = True
            self.interpreter.object_counter["BRK"] += 1
        log.debug("Kesme tetiklendi")

    async def op_mvn(self, operands: List[Any]) -> None:
        """Blok taşıma (negatif)."""
        src_bank, dst_bank = operands
        count = self.registers["A"]
        src_addr = (self.registers["X"] | (src_bank << 16)) & 0xFFFFFF
        dst_addr = (self.registers["Y"] | (dst_bank << 16)) & 0xFFFFFF
        for i in range(count):
            self.memory[dst_addr + i] = self.memory[src_addr + i]
        self.registers["X"] += count
        self.registers["Y"] += count
        self.registers["A"] -= count
        log.debug(f"MVN: {count} bayt taşındı, ${src_addr:06X} -> ${dst_addr:06X}")

    async def op_mvp(self, operands: List[Any]) -> None:
        """Blok taşıma (pozitif)."""
        src_bank, dst_bank = operands
        count = self.registers["A"]
        src_addr = ((self.registers["X"] + count - 1) | (src_bank << 16)) & 0xFFFFFF
        dst_addr = ((self.registers["Y"] + count - 1) | (dst_bank << 16)) & 0xFFFFFF
        for i in range(count):
            self.memory[dst_addr - i] = self.memory[src_addr - i]
        self.registers["X"] -= count
        self.registers["Y"] -= count
        self.registers["A"] -= count
        log.debug(f"MVP: {count} bayt taşındı, ${src_addr:06X} -> ${dst_addr:06X}")

    async def op_pea(self, operands: List[Any]) -> None:
        """Etkin adresi yığına it."""
        addr = operands[0]
        with self.lock:
            self.registers["S"] -= 2
            self.memory[self.registers["S"]] = addr & 0xFF
            self.memory[self.registers["S"] + 1] = (addr >> 8) & 0xFF
        log.debug(f"PEA: ${addr:04X} yığına itildi")

    async def op_pei(self, operands: List[Any]) -> None:
        """Dolaylı adresi yığına it."""
        addr = self.get_address("IND", operands[0])
        with self.lock:
            self.registers["S"] -= 2
            self.memory[self.registers["S"]] = addr & 0xFF
            self.memory[self.registers["S"] + 1] = (addr >> 8) & 0xFF
        log.debug(f"PEI: ${addr:04X} yığına itildi")

    async def op_per(self, operands: List[Any]) -> None:
        """Göreli adresi yığına it."""
        offset = operands[0]
        addr = (self.program_counter + offset) & 0xFFFF
        with self.lock:
            self.registers["S"] -= 2
            self.memory[self.registers["S"]] = addr & 0xFF
            self.memory[self.registers["S"] + 1] = (addr >> 8) & 0xFF
        log.debug(f"PER: ${addr:04X} yığına itildi")

    async def op_tcd(self, operands: List[Any]) -> None:
        """C’yi D’ye aktar."""
        self.registers["D"] = self.registers["A"]
        self.update_flags(self.registers["D"])
        log.debug(f"TCD: D = {self.registers['D']}")

    async def op_tdc(self, operands: List[Any]) -> None:
        """D’yi C’ye aktar."""
        self.registers["A"] = self.registers["D"]
        self.update_flags(self.registers["A"])
        log.debug(f"TDC: A = {self.registers['A']}")

    async def op_bra(self, operands: List[Any]) -> int:
        """Her zaman dallan."""
        offset = operands[0]
        self.program_counter += offset
        return self.program_counter

    async def op_brl(self, operands: List[Any]) -> int:
        """Uzun dallan."""
        offset = operands[0]
        self.program_counter += offset
        return self.program_counter

    async def op_sep(self, operands: List[Any]) -> None:
        """Bayrakları sıfırla."""
        flags = operands[0]
        self.registers["P"] |= flags
        log.debug(f"SEP: P = {self.registers['P']:02X}")

    async def op_rep(self, operands: List[Any]) -> None:
        """Bayrakları ayarla."""
        flags = operands[0]
        self.registers["P"] &= ~flags
        log.debug(f"REP: P = {self.registers['P']:02X}")

    async def execute_bytecode(self, bytecode: List[Dict]) -> None:
        """Bayt kodunu yürütür."""
        self.program_counter = 0
        while self.program_counter < len(bytecode) and self.interpreter.running:
            instruction = bytecode[self.program_counter]
            opcode = instruction["opcode"]
            operands = instruction.get("operands", [])
            if opcode in self.opcode_table:
                try:
                    result = await self.opcode_table[opcode](operands)
                    if isinstance(result, int):
                        self.program_counter = result
                    else:
                        self.program_counter += 1
                except PdsXException as e:
                    await self.interpreter.exception_manager.handle_error(e)
                    raise
                except Exception as e:
                    await self.interpreter.exception_manager.handle_error(e)
                    raise PdsXRuntimeError(f"Bayt kodu yürütme hatası: {str(e)}")
            else:
                raise PdsXSyntaxError(f"Bilinmeyen bayt kodu: {opcode}")
        log.debug("Bayt kodu yürütme tamamlandı")

if __name__ == "__main__":
    print("bytecode_engine.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")