# libx_concurrency.py - PDS-X BASIC v14u Eşzamanlılık ve Asenkron İşlem Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import threading
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import queue
import logging
from pathlib import Path
import uuid
import time
import psutil

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_concurrency")

class PdsXException(Exception):
    pass

class AsyncManager:
    def __init__(self):
        self.tasks = []
        self.task_ids = {}
        self.loop = asyncio.new_event_loop()
        self.running = False
        self.lock = threading.Lock()

    def add_task(self, task: Callable, task_id: str, *args, **kwargs) -> None:
        """Asenkron görev ekler."""
        with self.lock:
            coro = self._wrap_task(task, *args, **kwargs)
            task_obj = asyncio.run_coroutine_threadsafe(coro, self.loop)
            self.tasks.append(task_obj)
            self.task_ids[task_id] = task_obj
            log.debug(f"Asenkron görev eklendi: {task_id}")

    async def _wrap_task(self, task: Callable, *args, **kwargs) -> Any:
        """Görev yürütme için sarmalayıcı."""
        try:
            return await asyncio.to_thread(task, *args, **kwargs)
        except Exception as e:
            log.error(f"Asenkron görev hatası: {str(e)}")
            raise PdsXException(f"Asenkron görev hatası: {str(e)}")

    def cancel_task(self, task_id: str) -> None:
        """Belirtilen görevi iptal eder."""
        with self.lock:
            task = self.task_ids.get(task_id)
            if task:
                task.cancel()
                self.tasks.remove(task)
                del self.task_ids[task_id]
                log.debug(f"Asenkron görev iptal edildi: {task_id}")
            else:
                raise PdsXException(f"Görev bulunamadı: {task_id}")

    def run_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        if not self.running:
            self.running = True
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            except KeyboardInterrupt:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()
                self.running = False
                log.info("Asenkron döngü durduruldu")

    def stop_loop(self) -> None:
        """Asenkron döngüyü durdurur."""
        if self.running:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.running = False
            log.info("Asenkron döngü durduruluyor")

class ConcurrencyManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.async_manager = AsyncManager()
        self.thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.threads = {}
        self.processes = {}
        self.task_queue = queue.Queue()
        self.active_threads = 0
        self.active_processes = 0
        self.metadata = {"libx_concurrency": {"version": "1.0.0", "dependencies": []}}
        self.lock = threading.Lock()
        self.async_thread = None

    def start_async_loop(self) -> None:
        """Asenkron döngüyü ayrı bir iş parçacığında başlatır."""
        if not self.async_thread or not self.async_thread.is_alive():
            self.async_thread = threading.Thread(target=self.async_manager.run_loop, daemon=True)
            self.async_thread.start()
            log.debug("Asenkron döngü iş parçacığı başlatıldı")

    def add_thread(self, sub_name: str, params: List[Any], thread_id: Optional[str] = None) -> str:
        """Yeni bir iş parçacığı başlatır."""
        with self.lock:
            thread_id = thread_id or str(uuid.uuid4())
            if thread_id in self.threads:
                raise PdsXException(f"İş parçacığı ID zaten kullanımda: {thread_id}")
            
            def target():
                try:
                    self.interpreter.execute_sub(sub_name, params)
                except Exception as e:
                    log.error(f"İş parçacığı hatası: {sub_name}, {str(e)}")
                    self.interpreter.current_scope()["_ERROR"] = str(e)
                finally:
                    with self.lock:
                        self.active_threads -= 1
                        del self.threads[thread_id]
            
            thread = threading.Thread(target=target, daemon=True)
            self.threads[thread_id] = thread
            self.active_threads += 1
            thread.start()
            log.debug(f"İş parçacığı başlatıldı: {thread_id}, {sub_name}")
            return thread_id

    def add_process(self, sub_name: str, params: List[Any], process_id: Optional[str] = None) -> str:
        """Yeni bir süreç başlatır."""
        with self.lock:
            process_id = process_id or str(uuid.uuid4())
            if process_id in self.processes:
                raise PdsXException(f"Süreç ID zaten kullanımda: {process_id}")
            
            def target():
                try:
                    self.interpreter.execute_sub(sub_name, params)
                except Exception as e:
                    log.error(f"Süreç hatası: {sub_name}, {str(e)}")
                finally:
                    with self.lock:
                        self.active_processes -= 1
                        del self.processes[process_id]
            
            process = multiprocessing.Process(target=target, daemon=True)
            self.processes[process_id] = process
            self.active_processes += 1
            process.start()
            log.debug(f"Süreç başlatıldı: {process_id}, {sub_name}")
            return process_id

    def add_async_task(self, sub_name: str, params: List[Any], task_id: Optional[str] = None) -> str:
        """Yeni bir asenkron görev başlatır."""
        task_id = task_id or str(uuid.uuid4())
        self.start_async_loop()
        
        async def async_target():
            return await asyncio.to_thread(self.interpreter.execute_sub, sub_name, params)
        
        self.async_manager.add_task(async_target, task_id)
        log.debug(f"Asenkron görev başlatıldı: {task_id}, {sub_name}")
        return task_id

    def wait(self, task_id: str, timeout: Optional[float] = None) -> None:
        """Belirtilen görevin tamamlanmasını bekler."""
        with self.lock:
            if task_id in self.threads:
                thread = self.threads[task_id]
                thread.join(timeout)
                if thread.is_alive():
                    raise PdsXException(f"İş parçacığı zaman aşımına uğradı: {task_id}")
                log.debug(f"İş parçacığı tamamlandı: {task_id}")
            elif task_id in self.processes:
                process = self.processes[task_id]
                process.join(timeout)
                if process.is_alive():
                    raise PdsXException(f"Süreç zaman aşımına uğradı: {task_id}")
                log.debug(f"Süreç tamamlandı: {task_id}")
            elif task_id in self.async_manager.task_ids:
                task = self.async_manager.task_ids[task_id]
                try:
                    asyncio.run_coroutine_threadsafe(task, self.async_manager.loop).result(timeout)
                    log.debug(f"Asenkron görev tamamlandı: {task_id}")
                except asyncio.TimeoutError:
                    raise PdsXException(f"Asenkron görev zaman aşımına uğradı: {task_id}")
            else:
                raise PdsXException(f"Görev bulunamadı: {task_id}")

    def cancel(self, task_id: str) -> None:
        """Belirtilen görevi iptal eder."""
        with self.lock:
            if task_id in self.threads:
                thread = self.threads[task_id]
                # Python'da iş parçacığı iptali sınırlıdır, sadece loglama yapılır
                log.warning(f"İş parçacığı iptali desteklenmiyor: {task_id}")
            elif task_id in self.processes:
                process = self.processes[task_id]
                process.terminate()
                self.active_processes -= 1
                del self.processes[task_id]
                log.debug(f"Süreç iptal edildi: {task_id}")
            elif task_id in self.async_manager.task_ids:
                self.async_manager.cancel_task(task_id)
                log.debug(f"Asenkron görev iptal edildi: {task_id}")
            else:
                raise PdsXException(f"Görev bulunamadı: {task_id}")

    def submit_thread(self, func: Callable, *args, **kwargs) -> str:
        """İş parçacığı havuzuna bir fonksiyon gönderir."""
        thread_id = str(uuid.uuid4())
        future = self.thread_pool.submit(func, *args, **kwargs)
        self.threads[thread_id] = future
        self.active_threads += 1
        log.debug(f"İş parçacığı havuzuna gönderildi: {thread_id}")
        return thread_id

    def submit_process(self, func: Callable, *args, **kwargs) -> str:
        """Süreç havuzuna bir fonksiyon gönderir."""
        process_id = str(uuid.uuid4())
        future = self.process_pool.submit(func, *args, **kwargs)
        self.processes[process_id] = future
        self.active_processes += 1
        log.debug(f"Süreç havuzuna gönderildi: {process_id}")
        return process_id

    def get_task_status(self, task_id: str) -> Dict:
        """Görev durumunu döndürür."""
        with self.lock:
            if task_id in self.threads:
                thread = self.threads[task_id]
                return {"type": "thread", "running": thread.is_alive() if isinstance(thread, threading.Thread) else not thread.done()}
            elif task_id in self.processes:
                process = self.processes[task_id]
                return {"type": "process", "running": process.is_alive() if isinstance(process, multiprocessing.Process) else not process.done()}
            elif task_id in self.async_manager.task_ids:
                task = self.async_manager.task_ids[task_id]
                return {"type": "async", "running": not task.done()}
            else:
                raise PdsXException(f"Görev bulunamadı: {task_id}")

    def monitor_resources(self) -> Dict:
        """Eşzamanlılık kaynaklarını izler."""
        return {
            "active_threads": self.active_threads,
            "active_processes": self.active_processes,
            "active_async_tasks": len(self.async_manager.tasks),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024
        }

    def parse_concurrency_command(self, command: str) -> None:
        """Eşzamanlılık komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("THREAD "):
                match = re.match(r"THREAD\s+(\w+)\s*,\s*(.+)", command, re.IGNORECASE)
                if match:
                    thread_id, sub_name = match.groups()
                    self.add_thread(sub_name, [], thread_id)
                    self.interpreter.current_scope()[thread_id] = thread_id
                else:
                    raise PdsXException("THREAD komutunda sözdizimi hatası")
            elif command_upper.startswith("ASYNC "):
                match = re.match(r"ASYNC\s+(\w+)\s*,\s*(.+)", command, re.IGNORECASE)
                if match:
                    task_id, sub_name = match.groups()
                    self.add_async_task(sub_name, [], task_id)
                    self.interpreter.current_scope()[task_id] = task_id
                else:
                    raise PdsXException("ASYNC komutunda sözdizimi hatası")
            elif command_upper.startswith("WAIT "):
                match = re.match(r"WAIT\s+(\w+)(?:\s+(\d+))?", command, re.IGNORECASE)
                if match:
                    task_id, timeout = match.groups()
                    timeout = float(timeout) if timeout else None
                    self.wait(task_id, timeout)
                else:
                    raise PdsXException("WAIT komutunda sözdizimi hatası")
            elif command_upper.startswith("CANCEL "):
                match = re.match(r"CANCEL\s+(\w+)", command, re.IGNORECASE)
                if match:
                    task_id = match.group(1)
                    self.cancel(task_id)
                else:
                    raise PdsXException("CANCEL komutunda sözdizimi hatası")
            elif command_upper == "MONITOR":
                status = self.monitor_resources()
                self.interpreter.current_scope()["_MONITOR"] = status
            else:
                raise PdsXException(f"Bilinmeyen eşzamanlılık komutu: {command}")
        except Exception as e:
            log.error(f"Eşzamanlılık komut hatası: {str(e)}")
            raise PdsXException(f"Eşzamanlılık komut hatası: {str(e)}")

    def shutdown(self) -> None:
        """Eşzamanlılık havuzlarını kapatır."""
        with self.lock:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            self.async_manager.stop_loop()
            self.active_threads = 0
            self.active_processes = 0
            self.threads.clear()
            self.processes.clear()
            self.async_manager.tasks.clear()
            self.async_manager.task_ids.clear()
            log.info("Eşzamanlılık havuzları kapatıldı")

if __name__ == "__main__":
    print("libx_concurrency.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")