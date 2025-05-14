# pipe.py - PDS-X BASIC v14u Boru Hattı İşlem Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from collections import defaultdict
import pickle
import gzip
import uuid
import time
import psutil
import numpy as np

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pipe")

# class PdsXException(Exception):
#     pass

class PipelineInstance:
    """Boru hattı örneği sınıfı."""
    def __init__(self, interpreter, pipe_id: str, commands: List[Dict], alias: Optional[str] = None, 
                 return_var: Optional[str] = None, onerror: Optional[str] = None, priority: str = "NORMAL"):
        self.interpreter = interpreter
        self.pipe_id = pipe_id
        self.commands = commands
        self.alias = alias or pipe_id
        self.return_var = return_var
        self.onerror = onerror
        self.priority = priority.upper()
        self.current_index = 0
        self.data = []
        self.active = False
        self.status = {"executed": [], "pending": commands.copy(), "errors": []}
        self.labels = {}
        self.lock = threading.Lock()
        self.start_time = None
        self.execution_time = 0.0

    def add_command(self, command: Dict, step_no: Optional[int] = None, position: Optional[str] = None) -> None:
        """Boru hattına yeni bir komut ekler."""
        with self.lock:
            try:
                if step_no is not None:
                    self.commands.insert(int(step_no), command)
                    self.status["pending"].insert(int(step_no), command)
                elif position == "START":
                    self.commands.insert(0, command)
                    self.status["pending"].insert(0, command)
                elif position == "END":
                    self.commands.append(command)
                    self.status["pending"].append(command)
                else:
                    self.commands.append(command)
                    self.status["pending"].append(command)
                log.debug(f"Boru hattına komut eklendi: pipe_id={self.pipe_id}, command={command}")
            except Exception as e:
                log.error(f"Komut ekleme hatası: {str(e)}")
                raise PdsXException(f"Komut ekleme hatası: {str(e)}")

    def remove_command(self, step_no: int) -> None:
        """Boru hattından bir komutu siler."""
        with self.lock:
            if 0 <= step_no < len(self.commands):
                try:
                    self.status["pending"].remove(self.commands[step_no])
                    self.commands.pop(step_no)
                    log.debug(f"Boru hattından komut silindi: pipe_id={self.pipe_id}, step_no={step_no}")
                except Exception as e:
                    log.error(f"Komut silme hatası: {str(e)}")
                    raise PdsXException(f"Komut silme hatası: {str(e)}")
            else:
                raise PdsXException(f"Geçersiz adım numarası: {step_no}")

    def execute(self, parallel: bool = False, executor_type: str = "thread") -> Optional[Any]:
        """Boru hattını yürütür."""
        self.active = True
        self.start_time = time.time()
        try:
            if parallel:
                if executor_type.lower() == "thread":
                    with ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=True)) as executor:
                        futures = [executor.submit(self.interpreter.execute_command, cmd) 
                                   for cmd in self.commands[self.current_index:]]
                        for future, cmd in zip(futures, self.commands[self.current_index:]):
                            try:
                                future.result()
                                self.current_index += 1
                                with self.lock:
                                    self.status["executed"].append(cmd)
                                    self.status["pending"].remove(cmd)
                            except Exception as e:
                                self.status["errors"].append(str(e))
                                if self.onerror:
                                    self.interpreter.execute_command(self.onerror)
                                raise PdsXException(f"Paralel yürütme hatası: {str(e)}")
                elif executor_type.lower() == "process":
                    with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
                        futures = [executor.submit(self.interpreter.execute_command, cmd) 
                                   for cmd in self.commands[self.current_index:]]
                        for future, cmd in zip(futures, self.commands[self.current_index:]):
                            try:
                                future.result()
                                self.current_index += 1
                                with self.lock:
                                    self.status["executed"].append(cmd)
                                    self.status["pending"].remove(cmd)
                            except Exception as e:
                                self.status["errors"].append(str(e))
                                if self.onerror:
                                    self.interpreter.execute_command(self.onerror)
                                raise PdsXException(f"Paralel yürütme hatası: {str(e)}")
                else:
                    raise PdsXException(f"Geçersiz yürütücü tipi: {executor_type}")
            else:
                for cmd in self.commands[self.current_index:]:
                    try:
                        self.interpreter.execute_command(cmd)
                        self.current_index += 1
                        with self.lock:
                            self.status["executed"].append(cmd)
                            self.status["pending"].remove(cmd)
                    except Exception as e:
                        self.status["errors"].append(str(e))
                        if self.onerror:
                            self.interpreter.execute_command(self.onerror)
                        raise PdsXException(f"Yürütme hatası: {str(e)}")
            
            self.execution_time = time.time() - self.start_time
            log.debug(f"Boru hattı yürütüldü: pipe_id={self.pipe_id}, parallel={parallel}, time={self.execution_time}")
            
            if self.return_var:
                return self.interpreter.current_scope().get(self.return_var, None)
            return None
        except Exception as e:
            log.error(f"Boru hattı yürütme hatası: {str(e)}")
            raise PdsXException(f"Boru hattı yürütme hatası: {str(e)}")
        finally:
            self.active = False

    async def execute_async(self) -> Optional[Any]:
        """Boru hattını asenkron olarak yürütür."""
        self.active = True
        self.start_time = time.time()
        try:
            for cmd in self.commands[self.current_index:]:
                try:
                    await asyncio.to_thread(self.interpreter.execute_command, cmd)
                    self.current_index += 1
                    with self.lock:
                        self.status["executed"].append(cmd)
                        self.status["pending"].remove(cmd)
                except Exception as e:
                    self.status["errors"].append(str(e))
                    if self.onerror:
                        await asyncio.to_thread(self.interpreter.execute_command, self.onerror)
                    raise PdsXException(f"Asenkron yürütme hatası: {str(e)}")
            
            self.execution_time = time.time() - self.start_time
            log.debug(f"Asenkron boru hattı yürütüldü: pipe_id={self.pipe_id}, time={self.execution_time}")
            
            if self.return_var:
                return self.interpreter.current_scope().get(self.return_var, None)
            return None
        except Exception as e:
            log.error(f"Asenkron boru hattı yürütme hatası: {str(e)}")
            raise PdsXException(f"Asenkron boru hattı yürütme hatası: {str(e)}")
        finally:
            self.active = False

    def next(self) -> None:
        """Bir sonraki komutu yürütür."""
        with self.lock:
            if self.current_index < len(self.commands):
                try:
                    self.interpreter.execute_command(self.commands[self.current_index])
                    self.status["executed"].append(self.commands[self.current_index])
                    self.status["pending"].remove(self.commands[self.current_index])
                    self.current_index += 1
                    log.debug(f"Boru hattı bir sonraki adım yürütüldü: pipe_id={self.pipe_id}, step={self.current_index}")
                except Exception as e:
                    self.status["errors"].append(str(e))
                    if self.onerror:
                        self.interpreter.execute_command(self.onerror)
                    raise PdsXException(f"Adım yürütme hatası: {str(e)}")
            else:
                raise PdsXException(f"Boru hattında yürütülecek komut kalmadı: pipe_id={self.pipe_id}")

    def set_label(self, label: Union[str, int], step_no: int) -> None:
        """Boru hattına etiket ekler."""
        with self.lock:
            try:
                if 0 <= step_no < len(self.commands):
                    self.labels[str(label)] = step_no
                    log.debug(f"Etiket eklendi: pipe_id={self.pipe_id}, label={label}, step_no={step_no}")
                else:
                    raise PdsXException(f"Geçersiz adım numarası: {step_no}")
            except Exception as e:
                log.error(f"Etiket ekleme hatası: {str(e)}")
                raise PdsXException(f"Etiket ekleme hatası: {str(e)}")

    def get_label(self, label: str) -> int:
        """Etiketin adım numarasını döndürür."""
        with self.lock:
            step_no = self.labels.get(str(label))
            if step_no is None:
                raise PdsXException(f"Etiket bulunamadı: {label}")
            return step_no

    def get_status(self) -> Dict:
        """Boru hattı durumunu döndürür."""
        with self.lock:
            return {
                "pipe_id": self.pipe_id,
                "alias": self.alias,
                "active": self.active,
                "executed": len(self.status["executed"]),
                "pending": len(self.status["pending"]),
                "errors": self.status["errors"],
                "execution_time": self.execution_time,
                "priority": self.priority
            }

    def save(self, path: str, compress: bool = False) -> None:
        """Boru hattını dosyaya kaydeder."""
        with self.lock:
            try:
                data = {
                    "pipe_id": self.pipe_id,
                    "commands": self.commands,
                    "alias": self.alias,
                    "return_var": self.return_var,
                    "onerror": self.onerror,
                    "priority": self.priority,
                    "status": self.status,
                    "labels": self.labels
                }
                serialized = pickle.dumps(data)
                if compress:
                    with gzip.open(path, "wb") as f:
                        f.write(serialized)
                else:
                    with open(path, "wb") as f:
                        f.write(serialized)
                log.debug(f"Boru hattı kaydedildi: pipe_id={self.pipe_id}, path={path}, compress={compress}")
            except Exception as e:
                log.error(f"Boru hattı kaydetme hatası: {str(e)}")
                raise PdsXException(f"Boru hattı kaydetme hatası: {str(e)}")

    def load(self, path: str) -> None:
        """Boru hattını dosyadan yükler."""
        with self.lock:
            try:
                if path.endswith(".gz"):
                    with gzip.open(path, "rb") as f:
                        data = pickle.loads(f.read())
                else:
                    with open(path, "rb") as f:
                        data = pickle.loads(f.read())
                
                self.pipe_id = data["pipe_id"]
                self.commands = data["commands"]
                self.alias = data["alias"]
                self.return_var = data["return_var"]
                self.onerror = data["onerror"]
                self.priority = data["priority"]
                self.status = data["status"]
                self.labels = data["labels"]
                self.current_index = len(self.status["executed"])
                log.debug(f"Boru hattı yüklendi: pipe_id={self.pipe_id}, path={path}")
            except Exception as e:
                log.error(f"Boru hattı yükleme hatası: {str(e)}")
                raise PdsXException(f"Boru hattı yükleme hatası: {str(e)}")

class PipeManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.pipelines = {}
        self.active_pipes = 0
        self.executor = ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=True) * 2)
        self.async_loop = asyncio.new_event_loop()
        self.metadata = {"pipe": {"version": "1.0.0", "dependencies": ["psutil", "numpy"]}}
        self.lock = threading.Lock()

    def start_async_loop(self) -> None:
        """Asenkron döngüyü ayrı bir iş parçacığında başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not hasattr(self, "async_thread") or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron boru hattı döngüsü başlatıldı")

    def parse_pipe_command(self, command: str) -> None:
        """Boru hattı komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("PIPE "):
                match = re.match(r"PIPE\s+(\w+)(?:\s+RETURN\s+(\w+))?(?:\s+PARALLEL\s+(\w+))?(?:\s+ONERROR\s+(\w+))?(?:\s+PRIORITY\s+(\w+))?", 
                               command, re.IGNORECASE)
                if not match:
                    raise PdsXException("PIPE komutunda sözdizimi hatası")
                
                pipe_id, return_var, executor_type, onerror, priority = match.groups()
                executor_type = executor_type or "thread"
                priority = priority or "NORMAL"
                commands = []
                
                # Boru hattı komutlarını ayrıştır
                lines = command.split("\n")
                i = 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line.upper().startswith("END PIPE"):
                        break
                    if line.upper().startswith("STEP "):
                        cmd = {"type": "step", "command": line[5:].strip()}
                        commands.append(cmd)
                    elif line.upper().startswith("LABEL "):
                        match = re.match(r"LABEL\s+(\w+)\s+(\d+)", line, re.IGNORECASE)
                        if match:
                            label, step_no = match.groups()
                            commands.append({"type": "label", "label": label, "step_no": int(step_no)})
                        else:
                            raise PdsXException("LABEL komutunda sözdizimi hatası")
                    elif line.upper().startswith("DATA "):
                        match = re.match(r"DATA\s+(.+)", line, re.IGNORECASE)
                        if match:
                            data = self.interpreter.evaluate_expression(match.group(1))
                            commands.append({"type": "data", "value": data})
                        else:
                            raise PdsXException("DATA komutunda sözdizimi hatası")
                    i += 1
                
                # Boru hattı örneği oluştur
                pipeline = PipelineInstance(self.interpreter, pipe_id, commands, pipe_id, return_var, onerror, priority)
                with self.lock:
                    self.pipelines[pipe_id] = pipeline
                    self.active_pipes += 1
                
                # Boru hattını yürüt
                if "PARALLEL" in command_upper:
                    self.executor.submit(pipeline.execute, parallel=True, executor_type=executor_type)
                elif "ASYNC" in command_upper:
                    self.start_async_loop()
                    asyncio.run_coroutine_threadsafe(pipeline.execute_async(), self.async_loop)
                else:
                    pipeline.execute(parallel=False)
                
                log.debug(f"Boru hattı başlatıldı: pipe_id={pipe_id}, parallel={'PARALLEL' in command_upper}, async={'ASYNC' in command_upper}")
            
            elif command_upper.startswith("PIPE ADD "):
                match = re.match(r"PIPE ADD\s+(\w+)\s+(.+?)(?:\s+AT\s+(\d+|\w+))?", command, re.IGNORECASE | re.DOTALL)
                if match:
                    pipe_id, cmd_str, position = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    cmd = {"type": "step", "command": cmd_str.strip()}
                    if position:
                        if position.isdigit():
                            pipeline.add_command(cmd, step_no=int(position))
                        else:
                            pipeline.add_command(cmd, position=position.upper())
                    else:
                        pipeline.add_command(cmd)
                else:
                    raise PdsXException("PIPE ADD komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE REMOVE "):
                match = re.match(r"PIPE REMOVE\s+(\w+)\s+(\d+)", command, re.IGNORECASE)
                if match:
                    pipe_id, step_no = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    pipeline.remove_command(int(step_no))
                else:
                    raise PdsXException("PIPE REMOVE komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE EXECUTE "):
                match = re.match(r"PIPE EXECUTE\s+(\w+)(?:\s+PARALLEL\s+(\w+))?", command, re.IGNORECASE)
                if match:
                    pipe_id, executor_type = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    if executor_type:
                        self.executor.submit(pipeline.execute, parallel=True, executor_type=executor_type)
                    else:
                        pipeline.execute(parallel=False)
                else:
                    raise PdsXException("PIPE EXECUTE komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE EXECUTE ASYNC "):
                match = re.match(r"PIPE EXECUTE ASYNC\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id = match.group(1)
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    self.start_async_loop()
                    asyncio.run_coroutine_threadsafe(pipeline.execute_async(), self.async_loop)
                else:
                    raise PdsXException("PIPE EXECUTE ASYNC komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE NEXT "):
                match = re.match(r"PIPE NEXT\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id = match.group(1)
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    pipeline.next()
                else:
                    raise PdsXException("PIPE NEXT komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE SET LABEL "):
                match = re.match(r"PIPE SET LABEL\s+(\w+)\s+(\w+)\s+(\d+)", command, re.IGNORECASE)
                if match:
                    pipe_id, label, step_no = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    pipeline.set_label(label, int(step_no))
                else:
                    raise PdsXException("PIPE SET LABEL komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE GET LABEL "):
                match = re.match(r"PIPE GET LABEL\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, label, var_name = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    step_no = pipeline.get_label(label)
                    self.interpreter.current_scope()[var_name] = step_no
                else:
                    raise PdsXException("PIPE GET LABEL komutunda sözdizimi hatası")
            
            elif command_upper.startswith("PIPE STATUS "):
                match = re.match(r"PIPE STATUS\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, var_name = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    status = pipeline.get_status()
                    self.interpreter.current_scope()[var_name] = status
                else:
                    raise PdsXException("PIPE STATUS komutunda sözdizimi hatası")
            
            elif command_upper.startswith("SAVE PIPE "):
                match = re.match(r"SAVE PIPE\s+(\w+)\s*,\s*\"([^\"]+)\"\s*(COMPRESS)?", command, re.IGNORECASE)
                if match:
                    pipe_id, path, compress = match.groups()
                    pipeline = self.pipelines.get(pipe_id)
                    if not pipeline:
                        raise PdsXException(f"Boru hattı bulunamadı: {pipe_id}")
                    pipeline.save(path, bool(compress))
                else:
                    raise PdsXException("SAVE PIPE komutunda sözdizimi hatası")
            
            elif command_upper.startswith("LOAD PIPE "):
                match = re.match(r"LOAD PIPE\s+(\w+)\s*,\s*\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    pipe_id, path = match.groups()
                    pipeline = PipelineInstance(self.interpreter, pipe_id, [])
                    pipeline.load(path)
                    with self.lock:
                        self.pipelines[pipe_id] = pipeline
                        self.active_pipes += 1
                else:
                    raise PdsXException("LOAD PIPE komutunda sözdizimi hatası")
            
            else:
                raise PdsXException(f"Bilinmeyen boru hattı komutu: {command}")
        except Exception as e:
            log.error(f"Boru hattı komut hatası: {str(e)}")
            raise PdsXException(f"Boru hattı komut hatası: {str(e)}")

    def shutdown(self) -> None:
        """Boru hattı yürütücülerini kapatır."""
        with self.lock:
            try:
                self.executor.shutdown(wait=True)
                if hasattr(self, "async_thread") and self.async_thread.is_alive():
                    self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                    self.async_loop.run_until_complete(self.async_loop.shutdown_asyncgens())
                    self.async_loop.close()
                self.pipelines.clear()
                self.active_pipes = 0
                log.info("Boru hattı yürütücüleri kapatıldı")
            except Exception as e:
                log.error(f"Boru hattı kapatma hatası: {str(e)}")
                raise PdsXException(f"Boru hattı kapatma hatası: {str(e)}")

if __name__ == "__main__":
    print("pipe.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")