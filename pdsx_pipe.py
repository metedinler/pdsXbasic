# pdsx_pipe.py - PDS-X BASIC v15 Veri İletişim ve Boru Yönetimi
# Version: 1.0.0
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import os
import sys
import time
import json
import queue
import select
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class PdsXPipe:
    """PDS-X Veri İletişim Borusu"""
    name: str
    mode: str = "rw"  # r: read, w: write, rw: read/write
    buffer_size: int = 8192  # Varsayılan tampon boyutu
    timeout: float = 0.1  # Varsayılan zaman aşımı
    encoding: str = "utf-8"  # Varsayılan karakter kodlaması
    max_size: int = 1024 * 1024  # Maksimum veri boyutu (1MB)
    _buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: Lock = field(default_factory=Lock)
    _subscribers: List[Callable] = field(default_factory=list)
    _stats: Dict = field(default_factory=lambda: {
        "bytes_read": 0,
        "bytes_written": 0,
        "messages_read": 0,
        "messages_written": 0,
        "last_activity": time.time()
    })
    
    def __post_init__(self):
        """Boru başlatma sonrası yapılandırma"""
        self._event_loop = asyncio.new_event_loop()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._closed = False
        self._error_handler = None
        
    def write(self, data: Any) -> bool:
        """Boruya veri yazar"""
        if "w" not in self.mode:
            raise PdsXPipeError("Boru yazma modunda değil", error_code="ERR_PIPE_MODE")
            
        if self._closed:
            raise PdsXPipeError("Boru kapalı", error_code="ERR_PIPE_CLOSED")
            
        try:
            with self._lock:
                if isinstance(data, (str, bytes, int, float, bool)):
                    self._buffer.append(data)
                else:
                    self._buffer.append(json.dumps(data))
                
                self._stats["bytes_written"] += sys.getsizeof(data)
                self._stats["messages_written"] += 1
                self._stats["last_activity"] = time.time()
                
                # Aboneleri bilgilendir
                for subscriber in self._subscribers:
                    try:
                        subscriber(data)
                    except Exception as e:
                        if self._error_handler:
                            self._error_handler(e)
                            
                return True
        except Exception as e:
            if self._error_handler:
                self._error_handler(e)
            raise PdsXPipeError(f"Veri yazma hatası: {str(e)}", error_code="ERR_PIPE_WRITE")
            
    def read(self, timeout: float = None) -> Any:
        """Borudan veri okur"""
        if "r" not in self.mode:
            raise PdsXPipeError("Boru okuma modunda değil", error_code="ERR_PIPE_MODE")
            
        if self._closed:
            raise PdsXPipeError("Boru kapalı", error_code="ERR_PIPE_CLOSED")
            
        timeout = timeout or self.timeout
        start_time = time.time()
        
        while True:
            with self._lock:
                if len(self._buffer) > 0:
                    data = self._buffer.popleft()
                    self._stats["bytes_read"] += sys.getsizeof(data)
                    self._stats["messages_read"] += 1
                    self._stats["last_activity"] = time.time()
                    return data
                    
            if time.time() - start_time > timeout:
                raise PdsXPipeError("Okuma zaman aşımı", error_code="ERR_PIPE_TIMEOUT")
                
            time.sleep(0.001)  # CPU yükünü azalt
            
    async def write_async(self, data: Any) -> bool:
        """Boruya asenkron veri yazar"""
        return await self._event_loop.run_in_executor(
            self._thread_pool, self.write, data
        )
        
    async def read_async(self, timeout: float = None) -> Any:
        """Borudan asenkron veri okur"""
        return await self._event_loop.run_in_executor(
            self._thread_pool, self.read, timeout
        )
        
    def subscribe(self, callback: Callable) -> None:
        """Veri değişikliği için abone ekler"""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
                
    def unsubscribe(self, callback: Callable) -> None:
        """Veri değişikliği aboneliğini kaldırır"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
                
    def set_error_handler(self, handler: Callable) -> None:
        """Hata işleyici ayarlar"""
        self._error_handler = handler
        
    def clear(self) -> None:
        """Boru tamponunu temizler"""
        with self._lock:
            self._buffer.clear()
            
    def close(self) -> None:
        """Boruyu kapatır"""
        self._closed = True
        self._thread_pool.shutdown(wait=True)
        
    def get_stats(self) -> Dict:
        """Boru istatistiklerini döndürür"""
        return dict(self._stats)
        
    def is_empty(self) -> bool:
        """Boru boş mu kontrol eder"""
        return len(self._buffer) == 0
        
    def is_closed(self) -> bool:
        """Boru kapalı mı kontrol eder"""
        return self._closed
        
    def __enter__(self):
        """Context manager için giriş"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager için çıkış"""
        self.close()
        
    def __len__(self) -> int:
        """Boru tampon uzunluğunu döndürür"""
        return len(self._buffer)
        
    def __str__(self) -> str:
        """Boru string gösterimi"""
        return f"PdsXPipe(name='{self.name}', mode='{self.mode}', size={len(self)})"
        
class PdsXPipeManager:
    """PDS-X Boru Yöneticisi"""
    def __init__(self):
        self._pipes: Dict[str, PdsXPipe] = {}
        self._lock = Lock()
        self._stats = {
            "total_pipes": 0,
            "active_pipes": 0,
            "total_data_transfer": 0
        }
        
    def create_pipe(self, name: str, **kwargs) -> PdsXPipe:
        """Yeni boru oluşturur"""
        with self._lock:
            if name in self._pipes:
                raise PdsXPipeError(f"'{name}' isimli boru zaten var", error_code="ERR_PIPE_EXISTS")
                
            pipe = PdsXPipe(name, **kwargs)
            self._pipes[name] = pipe
            self._stats["total_pipes"] += 1
            self._stats["active_pipes"] += 1
            return pipe
            
    def get_pipe(self, name: str) -> PdsXPipe:
        """Var olan boruyu döndürür"""
        with self._lock:
            if name not in self._pipes:
                raise PdsXPipeError(f"'{name}' isimli boru bulunamadı", error_code="ERR_PIPE_NOT_FOUND")
            return self._pipes[name]
            
    def delete_pipe(self, name: str) -> None:
        """Boruyu siler"""
        with self._lock:
            if name in self._pipes:
                self._pipes[name].close()
                del self._pipes[name]
                self._stats["active_pipes"] -= 1
                
    def list_pipes(self) -> List[str]:
        """Tüm boru isimlerini listeler"""
        return list(self._pipes.keys())
        
    def get_stats(self) -> Dict:
        """Boru yöneticisi istatistiklerini döndürür"""
        total_transfer = sum(
            pipe._stats["bytes_read"] + pipe._stats["bytes_written"]
            for pipe in self._pipes.values()
        )
        self._stats["total_data_transfer"] = total_transfer
        return dict(self._stats)
        
    def cleanup(self) -> None:
        """Tüm boruları temizler"""
        with self._lock:
            for pipe in self._pipes.values():
                pipe.close()
            self._pipes.clear()
            self._stats["active_pipes"] = 0
            
    def __enter__(self):
        """Context manager için giriş"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager için çıkış"""
        self.cleanup()
        
# Singleton pipe manager instance
pipe_manager = PdsXPipeManager()

from pdsx_unified_exception import PdsXPipeError

# Test ve örnek kullanım
if __name__ == "__main__":
    # Test kodu
    def error_handler(e):
        print(f"Hata: {str(e)}")
        
    def data_callback(data):
        print(f"Yeni veri: {data}")
        
    try:
        # Pipe manager örneği
        with pipe_manager as pm:
            # Test pipe oluştur
            test_pipe = pm.create_pipe("test", mode="rw", buffer_size=1024)
            test_pipe.set_error_handler(error_handler)
            test_pipe.subscribe(data_callback)
            
            # Senkron veri yazma/okuma testi
            test_pipe.write("Test mesajı 1")
            test_pipe.write({"key": "value"})
            print(test_pipe.read())
            print(test_pipe.read())
            
            # Asenkron test
            async def async_test():
                await test_pipe.write_async("Async test mesajı")
                result = await test_pipe.read_async()
                print(f"Async sonuç: {result}")
                
            asyncio.run(async_test())
            
            # İstatistikler
            print(f"Pipe stats: {test_pipe.get_stats()}")
            print(f"Manager stats: {pm.get_stats()}")
            
    except PdsXPipeError as e:
        print(f"Pipe hatası: {str(e)} (Kod: {e.error_code})")
    except Exception as e:
        print(f"Genel hata: {str(e)}")
else:
    print("pdsx_pipe.py modülü başarıyla içe aktarıldı.")

# Modül dışa aktarma
__pdsX_exports__ = {
    "classes": {
        "PdsXPipe": PdsXPipe,
        "PdsXPipeManager": PdsXPipeManager,
        "PdsXPipeError": PdsXPipeError
    },
    "variables": {
        "pipe_manager": pipe_manager,
        "dependencies": [
            "numpy",
            "pandas",
            "torch"
        ]
    }
}
