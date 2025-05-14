# libxcore.py - PDS-X BASIC v14u Çekirdek Kütüphane
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import os
import sys
import json
import time
import random
import math
import shutil
import glob
import socket
import psutil
import multiprocessing
import threading
import subprocess
import datetime
import numpy as np
import pandas as pd
import pdfplumber
import requests
from pathlib import Path
from typing import Any, Callable, List, Dict, Optional
from collections import deque
import logging
import traceback

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libxcore")

class PdsXException(Exception):
    pass

class LibXCore:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.default_encoding = "utf-8"
        self.supported_encodings = [
            "utf-8", "cp1254", "iso-8859-9", "ascii", "utf-16", "utf-32",
            "cp1252", "iso-8859-1", "windows-1250", "latin-9",
            "cp932", "gb2312", "gbk", "euc-kr", "cp1251", "iso-8859-5",
            "cp1256", "iso-8859-6", "cp874", "iso-8859-7", "cp1257", "iso-8859-8",
            "utf-8-sig", "utf-8-bom-less"
        ]
        self.metadata = {"libx_core": {"version": "1.0.0", "dependencies": []}}
        self.stacks = {}
        self.queues = {}
        self.active_pipes = 0
        self.active_threads = 0

    # Koleksiyon İşlemleri
    def each(self, func: Callable, iterable: Any) -> None:
        """Her bir öğe için fonksiyon uygular."""
        for item in iterable:
            func(item)

    def select(self, func: Callable, iterable: Any) -> List:
        """Fonksiyona uyan öğeleri filtreler."""
        return [item for item in iterable if func(item)]

    def insert(self, collection: Any, value: Any, index: Optional[int] = None, key: Optional[Any] = None) -> None:
        """Koleksiyona öğe ekler."""
        if isinstance(collection, list):
            if index is None:
                collection.append(value)
            else:
                collection.insert(index, value)
        elif isinstance(collection, dict):
            if key is None:
                raise PdsXException("DICT için anahtar gerekli")
            collection[key] = value
        else:
            raise PdsXException("Geçersiz veri tipi")

    def remove(self, collection: Any, index: Optional[int] = None, key: Optional[Any] = None) -> None:
        """Koleksiyondan öğe siler."""
        if isinstance(collection, list):
            if index is None:
                raise PdsXException("Liste için indeks gerekli")
            collection.pop(index)
        elif isinstance(collection, dict):
            if key is None:
                raise PdsXException("DICT için anahtar gerekli")
            collection.pop(key, None)
        else:
            raise PdsXException("Geçersiz veri tipi")

    def pop(self, collection: Any) -> Any:
        """Listeden son öğeyi çıkarır."""
        if isinstance(collection, list):
            return collection.pop()
        raise PdsXException("Yalnızca liste için geçerli")

    def clear(self, collection: Any) -> None:
        """Koleksiyonu temizler."""
        if isinstance(collection, (list, dict)):
            collection.clear()
        else:
            raise PdsXException("Geçersiz veri tipi")

    def slice(self, iterable: Any, start: int, end: Optional[int] = None) -> Any:
        """Iterable'ı dilimler."""
        return iterable[start:end]

    def keys(self, obj: Dict) -> List:
        """Sözlüğün anahtarlarını döndürür."""
        if isinstance(obj, dict):
            return list(obj.keys())
        raise PdsXException("Yalnızca DICT için geçerli")

    # Veri Yapıları
    def stack(self) -> int:
        """Yeni bir yığın oluşturur."""
        stack_id = id([])
        self.stacks[stack_id] = deque()
        return stack_id

    def push(self, stack_id: int, item: Any) -> None:
        """Yığına öğe ekler."""
        if stack_id in self.stacks:
            self.stacks[stack_id].append(item)
        else:
            raise PdsXException("Geçersiz yığın")

    def pop(self, stack_id: int) -> Any:
        """Yığından öğe çıkarır."""
        if stack_id in self.stacks and self.stacks[stack_id]:
            return self.stacks[stack_id].pop()
        raise PdsXException("Yığın boş veya geçersiz")

    def queue(self) -> int:
        """Yeni bir kuyruk oluşturur."""
        queue_id = id([])
        self.queues[queue_id] = deque()
        return queue_id

    def enqueue(self, queue_id: int, item: Any) -> None:
        """Kuyruğa öğe ekler."""
        if queue_id in self.queues:
            self.queues[queue_id].append(item)
        else:
            raise PdsXException("Geçersiz kuyruk")

    def dequeue(self, queue_id: int) -> Any:
        """Kuyruktan öğe çıkarır."""
        if queue_id in self.queues and self.queues[queue_id]:
            return self.queues[queue_id].popleft()
        raise PdsXException("Kuyruk boş veya geçersiz")

    # Dosya İşlemleri
    def open(self, file_path: str, mode: str, encoding: str = "utf-8") -> Any:
        """Dosya açar."""
        mode_map = {
            "INPUT": "r", "OUTPUT": "w", "APPEND": "a", "BINARY": "rb", "RANDOM": "r+b"
        }
        if mode.upper() not in mode_map:
            raise PdsXException(f"Geçersiz dosya modu: {mode}")
        return open(file_path, mode_map[mode.upper()], encoding=encoding if mode.upper() != "BINARY" else None)

    def read_lines(self, path: str) -> List[str]:
        """Dosyadan satırları okur."""
        with open(path, "r", encoding=self.default_encoding) as f:
            return f.readlines()

    def write_json(self, obj: Any, path: str) -> None:
        """JSON dosyasına yazar."""
        with open(path, "w", encoding=self.default_encoding) as f:
            json.dump(obj, f)

    def read_json(self, path: str) -> Any:
        """JSON dosyasından okur."""
        with open(path, "r", encoding=self.default_encoding) as f:
            return json.load(f)

    def list_dir(self, path: str) -> List[str]:
        """Dizindeki dosyaları listeler."""
        return os.listdir(path)

    def listfile(self, path: str, pattern: str = "*") -> List[Dict]:
        """Dizindeki dosyaları metadata ile listeler."""
        files = glob.glob(os.path.join(path, pattern))
        return [{"name": f, "metadata": {"compressed": f.endswith(".hz")}} for f in files]

    def exists(self, path: str) -> bool:
        """Dosya/dizin varlığını kontrol eder."""
        return os.path.exists(path)

    def mkdir(self, path: str) -> None:
        """Dizin oluşturur."""
        os.makedirs(path, exist_ok=True)

    def kill(self, path: str) -> None:
        """Dosyayı siler."""
        if os.path.exists(path):
            os.remove(path)
        else:
            raise PdsXException(f"Dosya bulunamadı: {path}")

    def name(self, old_path: str, new_path: str) -> None:
        """Dosyayı yeniden adlandırır."""
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        else:
            raise PdsXException(f"Dosya bulunamadı: {old_path}")

    def rmdir(self, path: str) -> None:
        """Dizini siler."""
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            raise PdsXException(f"Dizin bulunamadı: {path}")

    def chdir(self, path: str) -> None:
        """Çalışma dizinini değiştirir."""
        if os.path.exists(path):
            os.chdir(path)
        else:
            raise PdsXException(f"Dizin bulunamadı: {path}")

    def copy_file(self, src: str, dst: str) -> None:
        """Dosyayı kopyalar."""
        shutil.copy(src, dst)

    def move_file(self, src: str, dst: str) -> None:
        """Dosyayı taşır."""
        shutil.move(src, dst)

    def delete_file(self, path: str) -> None:
        """Dosyayı siler (kill ile aynı)."""
        self.kill(path)

    def join_path(self, *parts: str) -> str:
        """Dosya yollarını birleştirir."""
        return os.path.join(*parts)

    def load_hz(self, path: str) -> str:
        """HZ dosyasını okur."""
        with open(path, "r", encoding=self.default_encoding) as f:
            return f.read()

    # Sistem İşlemleri
    def system(self, resource: str) -> Any:
        """Sistem kaynaklarını izler."""
        process = psutil.Process()
        if resource.lower() == "ram":
            return psutil.virtual_memory().available / 1024 / 1024
        elif resource.lower() == "cpu":
            return {"cores": multiprocessing.cpu_count(), "usage": psutil.cpu_percent()}
        elif resource.lower() == "gpu":
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info.append({
                        "memory_total": mem_info.total / 1024 / 1024,
                        "memory_used": mem_info.used / 1024 / 1024,
                        "utilization": util.gpu
                    })
                return gpu_info
            except ImportError:
                return "GPU izleme için pynvml gerekli"
        elif resource.lower() == "process":
            return len(psutil.pids())
        elif resource.lower() == "thread":
            return threading.active_count()
        elif resource.lower() == "pipe":
            return self.active_pipes
        else:
            raise PdsXException(f"Geçersiz kaynak: {resource}")

    def memory_usage(self) -> float:
        """Bellek kullanımını MB cinsinden döndürür."""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def cpu_count(self) -> int:
        """CPU çekirdek sayısını döndürür."""
        return multiprocessing.cpu_count()

    def getenv(self, name: str) -> str:
        """Çevresel değişkeni döndürür."""
        return os.getenv(name)

    def exit(self, code: int) -> None:
        """Programı belirtilen çıkış koduyla sonlandırır."""
        sys.exit(code)

    def ping(self, host: str) -> bool:
        """Host'a ping atar."""
        try:
            socket.gethostbyname(host)
            return True
        except socket.error:
            return False

    # Zaman ve Tarih
    def time_now(self) -> str:
        """Geçerli zamanı döndürür."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def date_now(self) -> str:
        """Geçerli tarihi döndürür."""
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def timer(self) -> float:
        """Geçerli zaman damgasını döndürür."""
        return time.time()

    def date_diff(self, date1: str, date2: str, unit: str = "days") -> int:
        """İki tarih arasındaki farkı hesaplar."""
        d1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
        delta = d2 - d1
        if unit.lower() == "days":
            return delta.days
        elif unit.lower() == "seconds":
            return int(delta.total_seconds())
        raise PdsXException("Geçersiz birim")

    def sleep(self, seconds: float) -> None:
        """Belirtilen süre kadar bekler."""
        time.sleep(seconds)

    async def async_wait(self, seconds: float) -> None:
        """Asenkron bekleme."""
        await asyncio.sleep(seconds)

    # Matematik ve Metin
    def random_int(self, min_val: int, max_val: int) -> int:
        """Rastgele tamsayı üretir."""
        return random.randint(min_val, max_val)

    def floor(self, x: float) -> int:
        """Aşağı yuvarlar."""
        return math.floor(x)

    def ceil(self, x: float) -> int:
        """Yukarı yuvarlar."""
        return math.ceil(x)

    def round(self, x: float, digits: int = 0) -> float:
        """Yuvarlar."""
        return round(x, digits)

    def sum(self, iterable: List) -> float:
        """Toplam hesaplar."""
        return sum(iterable)

    def mean(self, iterable: List) -> float:
        """Ortalama hesaplar."""
        return sum(iterable) / len(iterable) if iterable else 0

    def min(self, iterable: List) -> Any:
        """Minimum değeri bulur."""
        return min(iterable) if iterable else None

    def max(self, iterable: List) -> Any:
        """Maksimum değeri bulur."""
        return max(iterable) if iterable else None

    def split(self, s: str, sep: str) -> List[str]:
        """Metni böler."""
        return s.split(sep)

    def join(self, iterable: List, sep: str) -> str:
        """Metinleri birleştirir."""
        return sep.join(iterable)

    def trim(self, s: str) -> str:
        """Metni kırpar."""
        return s.strip()

    def replace(self, s: str, old: str, new: str) -> str:
        """Metinde değiştirme yapar."""
        return s.replace(old, new)

    def format(self, s: str, *args: Any) -> str:
        """Metni formatlar."""
        return s.format(*args)

    # PDF İşlemleri
    def pdf_read_text(self, file_path: str) -> str:
        """PDF'den metin çıkarır."""
        if not os.path.exists(file_path):
            return "PDF bulunamadı"
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text

    def pdf_extract_tables(self, file_path: str) -> List[List]:
        """PDF'den tablolar çıkarır."""
        if not os.path.exists(file_path):
            return []
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                tables.extend(page_tables)
        return tables

    # Ağ İşlemleri
    def web_get(self, url: str) -> str:
        """URL'den veri çeker."""
        try:
            response = requests.get(url)
            return response.text
        except Exception as e:
            return f"Hata: {e}"

    # Hata ve Kontrol
    def assert_(self, condition: bool, message: str) -> None:
        """Koşulu kontrol eder."""
        if not condition:
            raise PdsXException(f"Assert hatası: {message}")

    def log(self, message: str, level: str = "INFO", target: Optional[str] = None) -> None:
        """Log mesajı kaydeder."""
        log_message = f"[{level}] {message}"
        if target:
            with open(target, "a", encoding=self.default_encoding) as f:
                f.write(log_message + "\n")
        else:
            print(log_message)
        getattr(log, level.lower())(message)

    def ifthen(self, condition: bool, value1: Any, value2: Any) -> Any:
        """Koşullu değer döndürür."""
        return value1 if condition else value2

    def try_catch(self, block: Callable, handler: Callable) -> Any:
        """Hata yakalar."""
        try:
            return block()
        except Exception as e:
            return handler(str(e))

    def trace(self) -> List[str]:
        """Çağrı yığınını döndürür."""
        return traceback.format_stack()

    # Tür ve Kontrol
    def type_of(self, value: Any) -> str:
        """Değerin türünü döndürür."""
        if value is None:
            return "NULL"
        if isinstance(value, float) and math.isnan(value):
            return "NAN"
        if isinstance(value, int):
            return "INTEGER"
        elif isinstance(value, float):
            return "FLOAT"
        elif isinstance(value, str):
            return "STRING"
        elif isinstance(value, list):
            return "LIST"
        elif isinstance(value, dict):
            return "DICT"
        elif isinstance(value, bool):
            return "BOOLEAN"
        elif isinstance(value, np.ndarray):
            return "ARRAY"
        elif isinstance(value, pd.DataFrame):
            return "DATAFRAME"
        return "UNKNOWN"

    def is_empty(self, collection: Any) -> bool:
        """Koleksiyonun boş olup olmadığını kontrol eder."""
        return len(collection) == 0

    def len(self, obj: Any) -> int:
        """Objenin uzunluğunu döndürür."""
        return len(obj)

    def val(self, s: str) -> float:
        """Metni sayıya çevirir."""
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                raise PdsXException(f"Geçersiz değer: {s}")

    def str(self, value: Any) -> str:
        """Değeri metne çevirir."""
        return str(value)

    # Fonksiyonel Programlama
    def omega(self, *args: Any) -> Callable:
        """Parametreli lambda fonksiyonu oluşturur."""
        params = args[:-1]
        expr = args[-1]
        return lambda *values: eval(expr, {p: v for p, v in zip(params, values)})

    def map(self, func: Callable, iterable: Any) -> List:
        """Her öğeye fonksiyon uygular."""
        return [func(x) for x in iterable]

    def filter(self, func: Callable, iterable: Any) -> List:
        """Fonksiyona uyan öğeleri filtreler."""
        return [x for x in iterable if func(x)]

    def reduce(self, func: Callable, iterable: Any, initial: Any) -> Any:
        """Iterable'ı azaltır."""
        result = initial
        for x in iterable:
            result = func(result, x)
        return result

    # Kütüphane Yönetimi
    def list_lib(self, lib_name: str) -> Dict:
        """Kütüphane içeriğini listeler."""
        module = self.interpreter.modules.get(lib_name, {})
        return {"functions": list(module.get("functions", {}).keys()), "classes": list(module.get("classes", {}).keys())}

    def version(self, lib_name: str) -> str:
        """Kütüphane sürümünü döndürür."""
        return self.metadata.get(lib_name, {}).get("version", "unknown")

    def require_version(self, lib_name: str, required_version: str) -> None:
        """Kütüphane sürümünü kontrol eder."""
        current = self.version(lib_name)
        if not self._check_version(current, required_version):
            raise PdsXException(f"Versiyon uyumsuzluğu: {lib_name} {required_version} gerekli, {current} bulundu")

    def _check_version(self, current: str, required: str) -> bool:
        from packaging import version
        return version.parse(current) >= version.parse(required)

    # Kodlama
    def set_encoding(self, encoding: str) -> None:
        """Varsayılan kodlamayı ayarlar."""
        if encoding.lower() in self.supported_encodings:
            self.default_encoding = encoding.lower()
        else:
            raise PdsXException(f"Desteklenmeyen kodlama: {encoding}")

if __name__ == "__main__":
    print("libxcore.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")