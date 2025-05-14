# memory_manager.py - PDS-X BASIC v14u Bellek ve Veri Tipi Yönetim Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading
import weakref
import gc
import sys
import numpy as np
import pandas as pd

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("memory_manager")

class PdsXException(Exception):
    pass

class TypeManager:
    """Veri tipi yönetimi sınıfı."""
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.types = {}  # Tanımlı veri tipleri: {name: {kind, fields/values}}
        self.instances = defaultdict(list)  # Örnekler: {type_name: [instance_id, ...]}
        self.lock = threading.Lock()

    def define_type(self, name: str, fields: List[Tuple[str, str, List[int]]]) -> None:
        """Yeni bir veri tipi tanımlar."""
        with self.lock:
            if name.upper() in self.types:
                raise PdsXException(f"Veri tipi zaten tanımlı: {name}")
            try:
                self.types[name.upper()] = {
                    "kind": "STRUCT",
                    "fields": [(fname, ftype.upper(), dims) for fname, ftype, dims in fields]
                }
                log.debug(f"Veri tipi tanımlandı: {name}, fields={fields}")
            except Exception as e:
                log.error(f"Veri tipi tanımlama hatası: {str(e)}")
                raise PdsXException(f"Veri tipi tanımlama hatası: {str(e)}")

    def define_enum(self, name: str, values: List[str]) -> None:
        """Yeni bir enum tanımlar."""
        with self.lock:
            if name.upper() in self.types:
                raise PdsXException(f"Enum zaten tanımlı: {name}")
            try:
                self.types[name.upper()] = {
                    "kind": "ENUM",
                    "values": {v: i for i, v in enumerate(values)}
                }
                log.debug(f"Enum tanımlandı: {name}, values={values}")
            except Exception as e:
                log.error(f"Enum tanımlama hatası: {str(e)}")
                raise PdsXException(f"Enum tanımlama hatası: {str(e)}")

    def define_class(self, name: str, methods: Dict, private_methods: Dict, static_vars: Dict, parent: Optional[str], abstract: bool) -> None:
        """Yeni bir sınıf tanımlar."""
        with self.lock:
            if name.upper() in self.types:
                raise PdsXException(f"Sınıf zaten tanımlı: {name}")
            try:
                self.types[name.upper()] = {
                    "kind": "CLASS",
                    "methods": methods,
                    "private_methods": private_methods,
                    "static_vars": static_vars,
                    "parent": parent.upper() if parent else None,
                    "abstract": abstract
                }
                log.debug(f"Sınıf tanımlandı: {name}, abstract={abstract}")
            except Exception as e:
                log.error(f"Sınıf tanımlama hatası: {str(e)}")
                raise PdsXException(f"Sınıf tanımlama hatası: {str(e)}")

    def define_interface(self, name: str, methods: Dict) -> None:
        """Yeni bir arabirim tanımlar."""
        with self.lock:
            if name.upper() in self.types:
                raise PdsXException(f"Arabirim zaten tanımlı: {name}")
            try:
                self.types[name.upper()] = {
                    "kind": "INTERFACE",
                    "methods": methods
                }
                log.debug(f"Arabirim tanımlandı: {name}")
            except Exception as e:
                log.error(f"Arabirim tanımlama hatası: {str(e)}")
                raise PdsXException(f"Arabirim tanımlama hatası: {str(e)}")

    def create_struct_instance(self, type_name: str) -> int:
        """Yapı örneği oluşturur."""
        with self.lock:
            type_info = self.types.get(type_name.upper())
            if not type_info or type_info["kind"] != "STRUCT":
                raise PdsXException(f"Geçersiz yapı tipi: {type_name}")
            try:
                instance_id = id({})
                instance = {}
                for fname, ftype, dims in type_info["fields"]:
                    if dims:
                        instance[fname] = np.zeros(dims, dtype=self.memory_manager.type_table.get(ftype, object))
                    else:
                        instance[fname] = self.memory_manager.type_table.get(ftype, object)()
                self.instances[type_name.upper()].append(instance_id)
                self.memory_manager.heap[instance_id] = instance
                self.memory_manager.ref_counts[instance_id] = 1
                log.debug(f"Yapı örneği oluşturuldu: type={type_name}, id={instance_id}")
                return instance_id
            except Exception as e:
                log.error(f"Yapı örneği oluşturma hatası: {str(e)}")
                raise PdsXException(f"Yapı örneği oluşturma hatası: {str(e)}")

    def create_enum_instance(self, type_name: str, value: str) -> int:
        """Enum örneği oluşturur."""
        with self.lock:
            type_info = self.types.get(type_name.upper())
            if not type_info or type_info["kind"] != "ENUM":
                raise PdsXException(f"Geçersiz enum tipi: {type_name}")
            if value not in type_info["values"]:
                raise PdsXException(f"Geçersiz enum değeri: {value}")
            try:
                instance_id = id(value)
                self.instances[type_name.upper()].append(instance_id)
                self.memory_manager.heap[instance_id] = type_info["values"][value]
                self.memory_manager.ref_counts[instance_id] = 1
                log.debug(f"Enum örneği oluşturuldu: type={type_name}, value={value}, id={instance_id}")
                return instance_id
            except Exception as e:
                log.error(f"Enum örneği oluşturma hatası: {str(e)}")
                raise PdsXException(f"Enum örneği oluşturma hatası: {str(e)}")

    def create_class_instance(self, type_name: str) -> int:
        """Sınıf örneği oluşturur."""
        with self.lock:
            type_info = self.types.get(type_name.upper())
            if not type_info or type_info["kind"] != "CLASS":
                raise PdsXException(f"Geçersiz sınıf tipi: {type_name}")
            if type_info["abstract"]:
                raise PdsXException(f"Abstract sınıf örneği oluşturulamaz: {type_name}")
            try:
                instance_id = id({})
                instance = {
                    "methods": type_info["methods"],
                    "private_methods": type_info["private_methods"],
                    "vars": type_info["static_vars"].copy(),
                    "type": type_name.upper()
                }
                # Ebeveyn sınıf varsa miras al
                if type_info["parent"]:
                    parent_info = self.types.get(type_info["parent"])
                    if parent_info and parent_info["kind"] == "CLASS":
                        instance["methods"].update(parent_info["methods"])
                        instance["private_methods"].update(parent_info["private_methods"])
                        instance["vars"].update(parent_info["static_vars"])
                self.instances[type_name.upper()].append(instance_id)
                self.memory_manager.heap[instance_id] = instance
                self.memory_manager.ref_counts[instance_id] = 1
                log.debug(f"Sınıf örneği oluşturuldu: type={type_name}, id={instance_id}")
                return instance_id
            except Exception as e:
                log.error(f"Sınıf örneği oluşturma hatası: {str(e)}")
                raise PdsXException(f"Sınıf örneği oluşturma hatası: {str(e)}")

    def get_instance(self, instance_id: int) -> Any:
        """Örnek verisini döndürür."""
        with self.lock:
            if instance_id not in self.memory_manager.heap:
                raise PdsXException(f"Geçersiz örnek ID: {instance_id}")
            return self.memory_manager.heap[instance_id]

    def delete_instance(self, instance_id: int) -> None:
        """Örneği siler."""
        with self.lock:
            if instance_id not in self.memory_manager.heap:
                raise PdsXException(f"Geçersiz örnek ID: {instance_id}")
            try:
                type_name = next((t for t, instances in self.instances.items() if instance_id in instances), None)
                if type_name:
                    self.instances[type_name].remove(instance_id)
                self.memory_manager.release(instance_id)
                log.debug(f"Örnek silindi: id={instance_id}")
            except Exception as e:
                log.error(f"Örnek silme hatası: {str(e)}")
                raise PdsXException(f"Örnek silme hatası: {str(e)}")

class MemoryManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.heap = {}  # Bellek bloğu: {instance_id: data}
        self.ref_counts = {}  # Referans sayacı: {instance_id: int}
        self.type_manager = TypeManager(self)
        self.lock = threading.Lock()
        self.metadata = {"memory_manager": {"version": "1.0.0", "dependencies": ["numpy", "pandas"]}}
        
        # Veri tipi tablosu
        self.type_table = {
            "STRING": str, "INTEGER": int, "LONG": int, "SINGLE": float, "DOUBLE": float,
            "BYTE": int, "SHORT": int, "UNSIGNED INTEGER": int, "CHAR": str,
            "LIST": list, "DICT": dict, "SET": set, "TUPLE": tuple,
            "ARRAY": np.ndarray, "DATAFRAME": pd.DataFrame, "POINTER": None,
            "STRUCT": dict, "UNION": None, "ENUM": int, "VOID": None, "BITFIELD": int,
            "FLOAT128": np.float128, "FLOAT256": np.float256, "STRING8": str, "STRING16": str,
            "BOOLEAN": bool, "NULL": type(None), "NAN": float
        }

    def allocate(self, size: int) -> int:
        """Bellek bloğu tahsis eder."""
        with self.lock:
            try:
                instance_id = id(bytearray(size))
                self.heap[instance_id] = bytearray(size)
                self.ref_counts[instance_id] = 1
                log.debug(f"Bellek tahsis edildi: id={instance_id}, size={size}")
                return instance_id
            except Exception as e:
                log.error(f"Bellek tahsis hatası: {str(e)}")
                raise PdsXException(f"Bellek tahsis hatası: {str(e)}")

    def release(self, instance_id: int) -> None:
        """ Bellek bloğunu serbest bırakır."""
        with self.lock:
            if instance_id not in self.heap:
                raise PdsXException(f"Geçersiz örnek ID: {instance_id}")
            try:
                self.ref_counts[instance_id] -= 1
                if self.ref_counts[instance_id] <= 0:
                    del self.heap[instance_id]
                    del self.ref_counts[instance_id]
                    # Çöp toplama için zayıf referansları temizle
                    gc.collect()
                log.debug(f"Bellek serbest bırakıldı: id={instance_id}, ref_count={self.ref_counts.get(instance_id, 0)}")
            except Exception as e:
                log.error(f"Bellek serbest bırakma hatası: {str(e)}")
                raise PdsXException(f"Bellek serbest bırakma hatası: {str(e)}")

    def sizeof(self, instance_id: int) -> int:
        """ Örneğin bellek boyutunu döndürür."""
        with self.lock:
            if instance_id not in self.heap:
                raise PdsXException(f"Geçersiz örnek ID: {instance_id}")
            try:
                size = sys.getsizeof(self.heap[instance_id])
                log.debug(f"Boyut alındı: id={instance_id}, size={size}")
                return size
            except Exception as e:
                log.error(f"Boyut alma hatası: {str(e)}")
                raise PdsXException(f"Boyut alma hatası: {str(e)}")

    def parse_memory_command(self, command: str) -> None:
        """ Bellek yönetimi komutlarını ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("TYPE "):
                match = re.match(r"TYPE\s+(\w+)\s+(.+?)\s+END TYPE", command, re.IGNORECASE | re.DOTALL)
                if match:
                    name, fields_str = match.groups()
                    fields = []
                    for field_line in fields_str.split("\n"):
                        field_match = re.match(r"FIELD\s+(\w+)\s+AS\s+(\w+)(?:\s*,\s*([\d,]+))?", field_line, re.IGNORECASE)
                        if field_match:
                            fname, ftype, dimstr = field_match.groups()
                            dims = [int(x) for x in dimstr.split(",")] if dimstr else []
                            fields.append((fname, ftype, dims))
                    self.type_manager.define_type(name, fields)
                else:
                    raise PdsXException("TYPE komutunda sözdizimi hatası")
            elif command_upper.startswith("ENUM "):
                match = re.match(r"ENUM\s+(\w+)\s+(.+?)\s+END ENUM", command, re.IGNORECASE | re.DOTALL)
                if match:
                    name, values_str = match.groups()
                    values = [v.strip() for v in values_str.split("\n") if v.strip()]
                    self.type_manager.define_enum(name, values)
                else:
                    raise PdsXException("ENUM komutunda sözdizimi hatası")
            elif command_upper.startswith("NEW "):
                match = re.match(r"NEW\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    type_name, var_name = match.groups()
                    type_info = self.type_manager.types.get(type_name.upper())
                    if not type_info:
                        raise PdsXException(f"Geçersiz veri tipi: {type_name}")
                    if type_info["kind"] == "STRUCT":
                        instance_id = self.type_manager.create_struct_instance(type_name)
                    elif type_info["kind"] == "ENUM":
                        # Varsayılan olarak ilk enum değeri
                        instance_id = self.type_manager.create_enum_instance(type_name, list(type_info["values"].keys())[0])
                    elif type_info["kind"] == "CLASS":
                        instance_id = self.type_manager.create_class_instance(type_name)
                    else:
                        raise PdsXException(f"Desteklenmeyen veri tipi: {type_name}")
                    self.interpreter.current_scope()[var_name] = instance_id
                else:
                    raise PdsXException("NEW komutunda sözdizimi hatası")
            elif command_upper.startswith("DELETE "):
                match = re.match(r"DELETE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    instance_id = self.interpreter.current_scope().get(var_name)
                    if instance_id:
                        self.type_manager.delete_instance(instance_id)
                        del self.interpreter.current_scope()[var_name]
                    else:
                        raise PdsXException(f"Geçersiz örnek değişkeni: {var_name}")
                else:
                    raise PdsXException("DELETE komutunda sözdizimi hatası")
            elif command_upper.startswith("SIZEOF "):
                match = re.match(r"SIZEOF\s+(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    var_name, result_var = match.groups()
                    instance_id = self.interpreter.current_scope().get(var_name)
                    if instance_id:
                        size = self.sizeof(instance_id)
                        self.interpreter.current_scope()[result_var] = size
                    else:
                        raise PdsXException(f"Geçersiz örnek değişkeni: {var_name}")
                else:
                    raise PdsXException("SIZEOF komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen bellek komutu: {command}")
        except Exception as e:
            log.error(f"Bellek komut hatası: {str(e)}")
            raise PdsXException(f"Bellek komut hatası: {str(e)}")

if __name__ == "__main__":
    print("memory_manager.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")