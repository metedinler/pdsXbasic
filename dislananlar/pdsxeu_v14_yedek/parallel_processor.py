"""
PDS-X Paralel İşlem Yöneticisi
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import functools
import time
import json
from typing import List, Dict, Any, Callable
from pathlib import Path

class ParallelProcessManager:
    """Çoklu işlemci yönetim sistemi."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.stats = {}
        self.active_processes = {}
        
    def parallel_execute(self, func: Callable, items: List[Any], **kwargs) -> List[Any]:
        """Fonksiyonu parallel olarak çalıştırır."""
        start_time = time.time()
        
        # ProcessPoolExecutor ile paralel çalıştırma
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            if kwargs:
                func = functools.partial(func, **kwargs)
            results = list(executor.map(func, items))
            
        # İstatistikleri güncelle
        duration = time.time() - start_time
        self._update_stats(func.__name__, len(items), duration)
        
        return results
        
    def _update_stats(self, func_name: str, item_count: int, duration: float):
        """İşlem istatistiklerini günceller."""
        if func_name not in self.stats:
            self.stats[func_name] = {
                "total_runs": 0,
                "total_items": 0,
                "total_duration": 0,
                "avg_duration": 0
            }
            
        stats = self.stats[func_name]
        stats["total_runs"] += 1
        stats["total_items"] += item_count
        stats["total_duration"] += duration
        stats["avg_duration"] = stats["total_duration"] / stats["total_runs"]
        
    def start_background_process(self, func: Callable, *args, **kwargs) -> str:
        """Arka planda sürekli çalışacak bir işlem başlatır."""
        process = mp.Process(target=func, args=args, kwargs=kwargs)
        process.daemon = True
        process.start()
        
        process_id = str(process.pid)
        self.active_processes[process_id] = {
            "process": process,
            "function": func.__name__,
            "start_time": time.time()
        }
        
        return process_id
        
    def stop_process(self, process_id: str) -> bool:
        """Çalışan bir işlemi durdurur."""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]["process"]
            process.terminate()
            process.join()
            del self.active_processes[process_id]
            return True
        return False
        
    def get_process_info(self, process_id: str = None) -> Dict:
        """Çalışan işlemler hakkında bilgi verir."""
        if process_id:
            if process_id in self.active_processes:
                info = self.active_processes[process_id]
                return {
                    "function": info["function"],
                    "running_time": time.time() - info["start_time"],
                    "is_alive": info["process"].is_alive()
                }
            return {}
            
        return {pid: {
            "function": info["function"],
            "running_time": time.time() - info["start_time"],
            "is_alive": info["process"].is_alive()
        } for pid, info in self.active_processes.items()}
        
    def get_stats(self) -> Dict:
        """İşlem istatistiklerini döndürür."""
        return self.stats.copy()
        
    def clear_stats(self):
        """İstatistikleri sıfırlar."""
        self.stats.clear()
        
    def cleanup(self):
        """Tüm aktif işlemleri temizler."""
        for process_id in list(self.active_processes.keys()):
            self.stop_process(process_id)
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
