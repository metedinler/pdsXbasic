# lowlevel.py - PDS-X BASIC v14u Düşük Seviyeli Bellek ve İşlem Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import ctypes
import platform
import logging
import re
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import struct
import threading

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("lowlevel")

class PdsXException(Exception):
    pass

class LowLevelManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.heap = {}  # Bellek bloğu: {ptr: bytearray}
        self.ref_counts = {}  # Referans sayacı: {ptr: int}
        self.bitfields = defaultdict(dict)  # Bit alanları: {ptr: {field: (value, bits)}}
        self.pointers = {}  # İşaretçiler: {ptr_id: (target_ptr, offset)}
        self.lock = threading.Lock()
        self.metadata = {"lowlevel": {"version": "1.0.0", "dependencies": ["ctypes"]}}
        
        # Platform bağımlı yapılandırma
        self.is_windows = platform.system() == "Windows"
        self.ptr_size = ctypes.sizeof(ctypes.c_void_p)

    def allocate(self, size: int) -> int:
        """Bellek bloğu tahsis eder."""
        with self.lock:
            try:
                ptr = id(bytearray(size))
                self.heap[ptr] = bytearray(size)
                self.ref_counts[ptr] = 1
                log.debug(f"Bellek tahsis edildi: ptr={ptr}, size={size}")
                return ptr
            except Exception as e:
                log.error(f"Bellek tahsis hatası: {str(e)}")
                raise PdsXException(f"Bellek tahsis hatası: {str(e)}")

    def release(self, ptr: int) -> None:
        """Bellek bloğunu serbest bırakır."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            try:
                self.ref_counts[ptr] -= 1
                if self.ref_counts[ptr] <= 0:
                    del self.heap[ptr]
                    del self.ref_counts[ptr]
                    self.bitfields.pop(ptr, None)
                    # İşaretçi referanslarını temizle
                    for ptr_id, (target_ptr, _) in list(self.pointers.items()):
                        if target_ptr == ptr:
                            del self.pointers[ptr_id]
                log.debug(f"Bellek serbest bırakıldı: ptr={ptr}, ref_count={self.ref_counts.get(ptr, 0)}")
            except Exception as e:
                log.error(f"Bellek serbest bırakma hatası: {str(e)}")
                raise PdsXException(f"Bellek serbest bırakma hatası: {str(e)}")

    def bitset(self, ptr: int, field: str, value: int, bits: int) -> None:
        """Bit alanına değer atar."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            if bits <= 0 or bits > 64:
                raise PdsXException(f"Geçersiz bit sayısı: {bits}")
            try:
                self.bitfields[ptr][field] = (value & ((1 << bits) - 1), bits)
                log.debug(f"Bitset tamamlandı: ptr={ptr}, field={field}, value={value}, bits={bits}")
            except Exception as e:
                log.error(f"Bitset hatası: {str(e)}")
                raise PdsXException(f"Bitset hatası: {str(e)}")

    def bitget(self, ptr: int, field: str) -> int:
        """Bit alanından değer okur."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            try:
                value, _ = self.bitfields[ptr].get(field, (0, 0))
                log.debug(f"Bitget tamamlandı: ptr={ptr}, field={field}, value={value}")
                return value
            except Exception as e:
                log.error(f"Bitget hatası: {str(e)}")
                raise PdsXException(f"Bitget hatası: {str(e)}")

    def memcpy(self, dst_ptr: int, src_ptr: int, size: int) -> None:
        """Bellek bloğunu kopyalar."""
        with self.lock:
            if dst_ptr not in self.heap or src_ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: dst={dst_ptr}, src={src_ptr}")
            if size < 0 or size > len(self.heap[src_ptr]) or size > len(self.heap[dst_ptr]):
                raise PdsXException(f"Geçersiz kopyalama boyutu: {size}")
            try:
                self.heap[dst_ptr][:size] = self.heap[src_ptr][:size]
                log.debug(f"Memcpy tamamlandı: dst={dst_ptr}, src={src_ptr}, size={size}")
            except Exception as e:
                log.error(f"Memcpy hatası: {str(e)}")
                raise PdsXException(f"Memcpy hatası: {str(e)}")

    def memset(self, ptr: int, value: int, size: int) -> None:
        """Bellek bloğunu belirli bir değerle doldurur."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            if size < 0 or size > len(self.heap[ptr]):
                raise PdsXException(f"Geçersiz doldurma boyutu: {size}")
            try:
                self.heap[ptr][:size] = bytearray([value & 0xFF] * size)
                log.debug(f"Memset tamamlandı: ptr={ptr}, value={value}, size={size}")
            except Exception as e:
                log.error(f"Memset hatası: {str(e)}")
                raise PdsXException(f"Memset hatası: {str(e)}")

    def ptr(self, target_ptr: int, offset: int = 0) -> int:
        """ İşaretçi oluşturur."""
        with self.lock:
            if target_ptr not in self.heap:
                raise PdsXException(f"Geçersiz hedef işaretçi: {target_ptr}")
            try:
                ptr_id = id((target_ptr, offset))
                self.pointers[ptr_id] = (target_ptr, offset)
                self.ref_counts[target_ptr] += 1
                log.debug(f"İşaretçi oluşturuldu: ptr_id={ptr_id}, target={target_ptr}, offset={offset}")
                return ptr_id
            except Exception as e:
                log.error(f"İşaretçi oluşturma hatası: {str(e)}")
                raise PdsXException(f"İşaretçi oluşturma hatası: {str(e)}")

    def deref(self, ptr_id: int, size: int = 1) -> Union[int, bytes]:
        """ İşaretçiyi çözer ve veri döndürür."""
        with self.lock:
            if ptr_id not in self.pointers:
                raise PdsXException(f"Geçersiz işaretçi ID: {ptr_id}")
            target_ptr, offset = self.pointers[ptr_id]
            if target_ptr not in self.heap:
                raise PdsXException(f"Geçersiz hedef işaretçi: {target_ptr}")
            if offset < 0 or offset + size > len(self.heap[target_ptr]):
                raise PdsXException(f"Geçersiz işaretçi ofseti veya boyutu: offset={offset}, size={size}")
            try:
                data = self.heap[target_ptr][offset,offset + size]
                if size == 1:
                    return data[0]
                return bytes(data)
                log.debug(f"Deref tamamlandı: ptr_id={ptr_id}, target={target_ptr}, offset={offset}, size={size}")
            except Exception as e:
                log.error(f"Deref hatası: {str(e)}")
                raise PdsXException(f"Deref hatası: {str(e)}")

    def read_memory(self, ptr: int, size: int) -> bytes:
        """Bellek bloğundan veri okur."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            if size < 0 or size > len(self.heap[ptr]):
                raise PdsXException(f"Geçersiz okuma boyutu: {size}")
            try:
                data = bytes(self.heap[ptr][:size])
                log.debug(f"Bellek okundu: ptr={ptr}, size={size}")
                return data
            except Exception as e:
                log.error(f"Bellek okuma hatası: {str(e)}")
                raise PdsXException(f"Bellek okuma hatası: {str(e)}")

    def write_memory(self, ptr: int, data: bytes) -> None:
        """ Bellek bloğuna veri yazar."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            if len(data) > len(self.heap[ptr]):
                raise PdsXException(f"Geçersiz yazma boyutu: {len(data)}")
            try:
                self.heap[ptr][:len(data)] = data
                log.debug(f"Bellek yazıldı: ptr={ptr}, size={len(data)}")
            except Exception as e:
                log.error(f"Bellek yazma hatası: {str(e)}")
                raise PdsXException(f"Bellek yazma hatası: {str(e)}")

    def get_memory_info(self, ptr: int) -> Dict:
        """ Bellek bloğu hakkında bilgi döndürür."""
        with self.lock:
            if ptr not in self.heap:
                raise PdsXException(f"Geçersiz işaretçi: {ptr}")
            try:
                info = {
                    "size": len(self.heap[ptr]),
                    "ref_count": self.ref_counts.get(ptr, 0),
                    "bitfields": list(self.bitfields[ptr].keys()),
                    "pointers": [ptr_id for ptr_id, (target, _) in self.pointers.items() if target == ptr]
                }
                log.debug(f"Bellek bilgisi alındı: ptr={ptr}, {info}")
                return info
            except Exception as e:
                log.error(f"Bellek bilgi alma hatası: {str(e)}")
                raise PdsXException(f"Bellek bilgi alma hatası: {str(e)}")

    def parse_lowlevel_command(self, command: str) -> None:
        """ Düşük seviyeli komutları ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("BITSET "):
                match = re.match(r"BITSET\s+(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)", command, re.IGNORECASE)
                if match:
                    ptr, field, value, bits = match.groups()
                    self.bitset(int(ptr), field, int(value), int(bits))
                else:
                    raise PdsXException("BITSET komutunda sözdizimi hatası")
            elif command_upper.startswith("BITGET "):
                match = re.match(r"BITGET\s+(\d+)\s*,\s*(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    ptr, field, var_name = match.groups()
                    result = self.bitget(int(ptr), field)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("BITGET komutunda sözdizimi hatası")
            elif command_upper.startswith("MEMCPY "):
                match = re.match(r"MEMCPY\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", command, re.IGNORECASE)
                if match:
                    dst_ptr, src_ptr, size = map(int, match.groups())
                    self.memcpy(dst_ptr, src_ptr, size)
                else:
                    raise PdsXException("MEMCPY komutunda sözdizimi hatası")
            elif command_upper.startswith("MEMSET "):
                match = re.match(r"MEMSET\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", command, re.IGNORECASE)
                if match:
                    ptr, value, size = map(int, match.groups())
                    self.memset(ptr, value, size)
                else:
                    raise PdsXException("MEMSET komutunda sözdizimi hatası")
            elif command_upper.startswith("PTR "):
                match = re.match(r"PTR\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    target_ptr, offset, var_name = match.groups()
                    ptr_id = self.ptr(int(target_ptr), int(offset))
                    self.interpreter.current_scope()[var_name] = ptr_id
                else:
                    raise PdsXException("PTR komutunda sözdizimi hatası")
            elif command_upper.startswith("DEREF "):
                match = re.match(r"DEREF\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    ptr_id, size, var_name = match.groups()
                    result = self.deref(int(ptr_id), int(size))
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DEREF komutunda sözdizimi hatası")
            elif command_upper.startswith("READ MEMORY "):
                match = re.match(r"READ MEMORY\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    ptr, size, var_name = match.groups()
                    result = self.read_memory(int(ptr), int(size))
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("READ MEMORY komutunda sözdizimi hatası")
            elif command_upper.startswith("WRITE MEMORY "):
                match = re.match(r"WRITE MEMORY\s+(\d+)\s*,\s*\"([^\"]+)\"\s*", command, re.IGNORECASE)
                if match:
                    ptr, data = match.groups()
                    self.write_memory(int(ptr), data.encode())
                else:
                    raise PdsXException("WRITE MEMORY komutunda sözdizimi hatası")
            elif command_upper.startswith("MEMORY INFO "):
                match = re.match(r"MEMORY INFO\s+(\d+)\s*,\s*(\w+)", command, re.IGNORECASE)
                if match:
                    ptr, var_name = match.groups()
                    result = self.get_memory_info(int(ptr))
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("MEMORY INFO komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen düşük seviyeli komut: {command}")
        except Exception as e:
            log.error(f"Düşük seviyeli komut hatası: {str(e)}")
            raise PdsXException(f"Düşük seviyeli komut hatası: {str(e)}")

if __name__ == "__main__":
    print("lowlevel.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")