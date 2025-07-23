# pipe3.py - PDS-X BASIC v15 Ultra Güçlü Boru Hattı İşleme Kütüphanesi
# Version: 3.0.0
# Date: June 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import asyncio
import threading
import time
import json
import uuid
import logging
import pandas as pd
import numpy as np
import re
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union, Callable
from pdsx_exception2 import PdsXPipeError
# ZMQ (isteğe bağlı)
try:
    import zmq.asyncio
    ZMQ_ENABLED = True
except ImportError:
    ZMQ_ENABLED = False
    print("[PDS-X] ZMQ bulunamadı, ZMQ iletişimi devre dışı.")
    # Dummy ZMQ sınıfları
    class DummyZMQSocket:
        def bind(self, *args): pass
        async def send_json(self, *args): pass
        async def recv_json(self, *args): return {}
        def close(self): pass
    
    class DummyZMQContext:
        def socket(self, *args):
            return DummyZMQSocket()
        def term(self): pass
    
    class zmq:
        class asyncio:
            Context = DummyZMQContext
            PUB = "PUB"
            SUB = "SUB"

# Prometheus Client (isteğe bağlı)
try:
    from prometheus_client import Counter, Gauge, start_http_server
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
    print("[PDS-X] Prometheus Client bulunamadı, istatistik raporlama devre dışı.")
    # Dummy Counter ve Gauge sınıfları
    class Counter:
        def __init__(self, *args, **kwargs): 
            self.name = kwargs.get('name', '')
            self.labelnames = kwargs.get('labelnames', [])
        def labels(self, **kwargs): return self
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
    class Gauge:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', '')
            self.labelnames = kwargs.get('labelnames', [])
        def labels(self, **kwargs): return self
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
    def start_http_server(port): pass

# Qiskit (isteğe bağlı)
try:
    import qiskit
    from qiskit import QuantumCircuit, execute, Aer
    QISKIT_ENABLED = True
except ImportError:
    QISKIT_ENABLED = False
    print("[PDS-X] Qiskit bulunamadı, kuantum işleme devre dışı.")
    # Dummy Qiskit sınıfları
    class DummyQuantumCircuit:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'dummy_circuit')
        def h(self, *args): return self
        def cx(self, *args): return self
        def measure(self, *args): return self
        def draw(self, *args, **kwargs): return "Dummy Quantum Circuit"
    
    class DummyAer:
        @staticmethod
        def get_backend(*args, **kwargs):
            return DummyBackend()
    
    class DummyBackend:
        def __init__(self):
            self.name = "dummy_backend"
        def run(self, *args, **kwargs):
            return DummyJob()
    
    class DummyJob:
        def result(self):
            return DummyResult()
    
    class DummyResult:
        def get_counts(self, *args):
            return {"00": 500, "11": 500}  # Dummy sonuçlar
    
    QuantumCircuit = DummyQuantumCircuit
    Aer = DummyAer
    execute = lambda circuit, backend: DummyJob()

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_pipe.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pipe3")

# PIPE_DATA Veri Yapısı
PIPE_DATA = {
    "id": str,           # Pipe kimliği
    "status": str,       # Durum (queued, running, vb.)
    "mode": str,         # async/sync
    "config": dict,      # Yapılandırma seçenekleri
    "stages": list,      # Pipe aşamaları
    "flags": dict,       # Bayraklar (RUNNING, SECURED, vb.)
    "priority": float,   # Öncelik seviyesi
    "timestamp": float,  # Oluşturma zamanı
    "input_data": Any,   # Giriş verisi
    "subroutines": dict, # Alt programlar
    "commands": list     # Pipe içindeki komutlar
}

class FlagManager:
    """Dinamik bayrak yönetimi sınıfı."""
    def __init__(self):
        self.flags: Dict[str, Dict[str, bool]] = defaultdict(lambda: {
            "QUEUED": False, "RUNNING": False, "COMPLETED": False, "FAILED": False,
            "PAUSED": False, "SECURED": False, "MONITORED": False, "RECURSING": False,
            "PRIORITY_UPDATED": False, "LOCKED": False, "UNLOCKED": True
        })
        self.bus_manager = None
        self._lock = threading.RLock()

    def set_flag(self, pipe_id: str, flag: str, value: bool = True) -> None:
        with self._lock:
            self.flags[pipe_id][flag] = value
            if self.bus_manager:
                asyncio.create_task(self.bus_manager.publish(
                    topic="pipe_flag",
                    data={"pipe_id": pipe_id, "flag": flag, "value": value}
                ))
            log.debug(f"Bayrak ayarlandı: {pipe_id}, {flag} = {value}")

    def get_flag(self, pipe_id: str, flag: str) -> bool:
        with self._lock:
            return self.flags[pipe_id].get(flag, False)

    def clear_flag(self, pipe_id: str, flag: str) -> None:
        with self._lock:
            self.flags[pipe_id].pop(flag, None)
            if self.bus_manager:
                asyncio.create_task(self.bus_manager.publish(
                    topic="pipe_flag",
                    data={"pipe_id": pipe_id, "flag": flag, "action": "cleared"}
                ))
            log.debug(f"Bayrak temizlendi: {pipe_id}, {flag}")

class PipeInstanceManager:
    """Pipe örnek yönetimi sınıfı."""
    def __init__(self, max_instances: int = 65536):
        self.instances: Dict[str, List[Dict]] = defaultdict(list)
        self.max_instances = max_instances
        self.lock = asyncio.Lock()

    async def create_instance(self, pipe_id: str, instance_id: str, priority: float) -> bool:
        async with self.lock:
            if len(self.instances[pipe_id]) >= self.max_instances:
                raise PdsXPipeError(f"Maksimum örnek sınırına ulaşıldı: {pipe_id} (PIPE701)", context={"source": "create_instance"})
            self.instances[pipe_id].append({"id": instance_id, "priority": priority, "start_time": time.time()})
            log.debug(f"Örnek oluşturuldu: {pipe_id}, Örnek: {instance_id}")
            return True

    async def destroy_instance(self, pipe_id: str, instance_id: str) -> None:
        async with self.lock:
            self.instances[pipe_id] = [inst for inst in self.instances[pipe_id] if inst["id"] != instance_id]
            log.debug(f"Örnek yok edildi: {pipe_id}, Örnek: {instance_id}")

class PipeManager:
    """PDS-X BASIC v15 Ultra Güçlü Boru Hattı İşleme sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.RLock()
        self.async_lock = asyncio.Lock()
        self.pipes: Dict[str, Dict] = {}
        self.buffers: Dict[str, Any] = {}
        self.event_queue = asyncio.PriorityQueue()
        self.instance_manager = PipeInstanceManager()
        self.flag_manager = FlagManager()
        self.event_loop = asyncio.get_event_loop()
        self.bus_manager = None  # Bus Manager referansı
        
        # ZMQ kurulumu (opsiyonel)
        if ZMQ_ENABLED:
            self.zmq_ctx = zmq.asyncio.Context()
            self.zmq_sock = self.zmq_ctx.socket(zmq.asyncio.PUB)
            self.zmq_sock.bind("tcp://*:5557")
        else:
            self.zmq_ctx = None
            self.zmq_sock = None
            
        # Prometheus metrikleri (opsiyonel)
        if PROMETHEUS_ENABLED:
            self.prom_counter = Counter("pipex_pipes_total", "Toplam işlenen pipe sayısı", ["pipe_id"])
            self.prom_status = Gauge("pipex_status", "Pipe durumu", ["pipe_id", "status"])
            start_http_server(8002)
        else:
            self.prom_counter = Counter("pipex_pipes_total", "Toplam işlenen pipe sayısı", ["pipe_id"])
            self.prom_status = Gauge("pipex_status", "Pipe durumu", ["pipe_id", "status"])
            
        self._init_pipe()

    def _init_pipe(self) -> None:
        try:
            with self.lock:
                self.interpreter.object_counter["PIPE_INIT"] = self.interpreter.object_counter.get("PIPE_INIT", 0) + 1
            log.debug("Pipe yöneticisi başlatıldı")
            asyncio.create_task(self._process_queue())
        except Exception as e:
            raise PdsXPipeError(f"Pipe başlatma hatası: {str(e)} (PIPE001)", context={"source": "_init_pipe"})

    async def _process_queue(self) -> None:
        while True:
            try:
                priority, pipe_data = await self.event_queue.get()
                pipe_id = pipe_data["pipe_id"]
                instance_id = pipe_data.get("instance_id")
                await self._execute_pipe(pipe_id, instance_id)
                self.event_queue.task_done()
                self.prom_counter.labels(pipe_id=pipe_id).inc()
            except Exception as e:
                log.error(f"Kuyruk işleme hatası: {str(e)}")

    @lru_cache(maxsize=1024)
    def _parse_config(self, config: str) -> Dict[str, Any]:
        try:
            config_dict = {}
            if config:
                pairs = config.split()
                for pair in pairs:
                    if ":" in pair:
                        key, value = pair.split(":", 1)
                        if key.lower() in ("stages", "max_instances", "max_recursion"):
                            value = int(value)
                        elif key.lower() in ("priority"):
                            value = float(value)
                        elif key.lower() in ("encrypt", "live"):
                            value = value.lower() == "true"
                        config_dict[key.lower()] = value
            return config_dict
        except Exception as e:
            raise PdsXPipeError(f"Konfigürasyon ayrıştırma hatası: {str(e)} (PIPE002)", context={"source": "_parse_config"})

    async def parse_pipe_block(self, block: str, interpreter):
        """Yapısal PIPE ... END PIPE veya kompakt PIPE (...) bloğunu ayrıştırır."""
        block = block.strip()
        if block.upper().startswith("PIPE ") and "END PIPE" in block.upper():
            # Geleneksel yapı
            lines = block.split('\n')
            header = lines[0].strip()
            match = re.match(r"PIPE\s+(\w+)(?:\s+<(.+?)>)?(?:\s+(\d+))?(?:\s*CONFIG\s+\"(.+?)\")?(?:\s*ALIAS\s+(\w+))?", header, re.IGNORECASE)
            if not match:
                raise PdsXPipeError("Geçersiz PIPE başlığı (PIPE902)", context={"source": "parse_pipe_block"})

            pipe_id, input_data, pipe_num, config, alias = match.groups()
            config_dict = self._parse_config(config or "")
            input_data = eval(input_data) if input_data else None
            pipe_id = pipe_id or f"pipe_{pipe_num or uuid.uuid4().hex[:8]}"

            await self.define(pipe_id, config, input_data, alias)
            commands = []
            subroutines = {}
            current_sub = None
            for line in lines[1:-1]:
                line = line.strip()
                if line.upper().startswith("SUB "):
                    match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)", line, re.IGNORECASE)
                    if match:
                        sub_name, params = match.groups()
                        current_sub = sub_name
                        subroutines[sub_name] = {"params": params, "commands": []}
                    continue
                elif line.upper() == "END SUB":
                    current_sub = None
                    continue
                if current_sub:
                    subroutines[current_sub]["commands"].append(line)
                else:
                    commands.append(line)
            self.pipes[pipe_id]["commands"] = commands
            self.pipes[pipe_id]["subroutines"] = subroutines
            log.debug(f"Pipe bloğu kaydedildi: {pipe_id}")
        elif block.upper().startswith("PIPE ") and block.endswith(")"):
            # Kompakt yapı
            match = re.match(r"PIPE\s+(\w+)\s*(?:<(.+?)>)?\s*\((.*?)\)", block, re.IGNORECASE)
            if not match:
                raise PdsXPipeError("Geçersiz kompakt PIPE sözdizimi (PIPE903)", context={"source": "parse_pipe_block"})
            pipe_id, input_data, content = match.groups()
            pipe_id = pipe_id or f"pipe_{uuid.uuid4().hex[:8]}"
            input_data = eval(input_data) if input_data else None
            parts = content.split("|")
            commands = parts[0].split(":") if parts[0] else []
            config = ""
            subroutines = {}
            if len(parts) > 1:
                for part in parts[1].split(":"):
                    if part.strip():
                        if part.startswith("CONFIG "):
                            config = part.replace("CONFIG ", "")
                        elif part.startswith("SUB "):
                            sub_match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)", part, re.IGNORECASE)
                            if sub_match:
                                sub_name, params = sub_match.groups()
                                subroutines[sub_name] = {"params": params, "commands": []}
                        else:
                            if sub_name in subroutines:
                                subroutines[sub_name]["commands"].append(part)
                            else:
                                commands.append(part)
            await self.define(pipe_id, config, input_data)
            self.pipes[pipe_id]["commands"] = commands
            self.pipes[pipe_id]["subroutines"] = subroutines
            log.debug(f"Kompakt pipe kaydedildi: {pipe_id}")

    async def define(self, pipe_id: str, config: str, input_data: Any = None, alias: Optional[str] = None) -> str:
        """PIPE DEFINE <id> CONFIG "<options>" AS <var>"""
        async with self.async_lock:
            try:
                if pipe_id in self.pipes:
                    raise PdsXPipeError(f"Pipe zaten mevcut: {pipe_id} (PIPE003)", context={"source": "define"})
                config_dict = self._parse_config(config)
                pipe_id = pipe_id or f"pipe_{uuid.uuid4().hex[:8]}"
                self.pipes[pipe_id] = {
                    "id": pipe_id,
                    "status": "ready",
                    "mode": config_dict.get("mode", "async"),
                    "config": config_dict,
                    "stages": [],
                    "commands": [],
                    "subroutines": {},
                    "instances": [],
                    "max_instances": config_dict.get("max_instances", 65536),
                    "priority": config_dict.get("priority", 1.0),
                    "timestamp": time.time(),
                    "alias": alias,
                    "input_data": input_data
                }
                self.buffers[pipe_id] = input_data
                self.flag_manager.set_flag(pipe_id, "QUEUED")
                with self.lock:
                    self.interpreter.object_counter["PIPE_DEFINE"] = self.interpreter.object_counter.get("PIPE_DEFINE", 0) + 1
                    self.interpreter.object_registry[pipe_id] = {
                        "type": "PIPE_DATA",
                        "name": pipe_id,
                        "atom": f"pipe:{pipe_id}"
                    }
                log.debug(f"Pipe tanımlandı: {pipe_id}, Konfigürasyon: {config_dict}")
                return pipe_id
            except Exception as e:
                raise PdsXPipeError(f"Pipe tanımlama hatası: {str(e)} (PIPE004)", context={"source": "define"})

    async def start(self, pipe_id: str, mode: str = "async") -> None:
        """START PIPE <id> MODE "<async/sync>" """
        async with self.async_lock:
            try:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE005)", context={"source": "start"})
                self.pipes[pipe_id]["mode"] = mode
                self.flag_manager.set_flag(pipe_id, "RUNNING")
                instance_id = f"instance_{uuid.uuid4().hex[:8]}"
                await self.instance_manager.create_instance(pipe_id, instance_id, self.pipes[pipe_id]["priority"])
                await self.event_queue.put((
                    self.pipes[pipe_id]["priority"],
                    {"pipe_id": pipe_id, "instance_id": instance_id}
                ))
                self.prom_status.labels(pipe_id=pipe_id, status="running").set(1)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_status",
                        data={"pipe_id": pipe_id, "status": "running"}
                    )
                log.debug(f"Pipe başlatıldı: {pipe_id}, Mod: {mode}")
            except Exception as e:
                raise PdsXPipeError(f"Pipe başlatma hatası: {str(e)} (PIPE006)", context={"source": "start"})

    async def _execute_pipe(self, pipe_id: str, instance_id: str) -> None:
        try:
            async with self.async_lock:
                pipe = self.pipes[pipe_id]
                current_sub = None
                for cmd in pipe["commands"]:
                    cmd = cmd.strip()
                    if cmd.upper().startswith("SUB "):
                        match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)", cmd, re.IGNORECASE)
                        if match:
                            current_sub = match.group(1)
                        continue
                    elif cmd.upper() == "END SUB":
                        current_sub = None
                        continue
                    if not current_sub:
                        if cmd.upper().startswith("CALL SUB "):
                            match = re.match(r"CALL SUB\s+(\w+)\s*\((.*?)\)", cmd, re.IGNORECASE)
                            if match:
                                sub_name, args = match.groups()
                                if sub_name in pipe["subroutines"]:
                                    for sub_cmd in pipe["subroutines"][sub_name]["commands"]:
                                        await self.interpreter.execute_command(sub_cmd)
                        else:
                            await self.interpreter.execute_command(cmd)
                self.flag_manager.set_flag(pipe_id, "COMPLETED")
                await self.instance_manager.destroy_instance(pipe_id, instance_id)
                self.prom_status.labels(pipe_id=pipe_id, status="completed").set(1)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_status",
                        data={"pipe_id": pipe_id, "status": "completed"}
                    )
                log.debug(f"Pipe yürütüldü: {pipe_id}, Örnek: {instance_id}")
        except Exception as e:
            self.flag_manager.set_flag(pipe_id, "FAILED")
            self.prom_status.labels(pipe_id=pipe_id, status="failed").set(1)
            raise PdsXPipeError(f"Pipe yürütme hatası: {str(e)} (PIPE007)", context={"source": "_execute_pipe"})

    async def stop(self, pipe_id: str, config: str = "") -> None:
        """STOP PIPE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE008)", context={"source": "stop"})
                self.pipes[pipe_id]["status"] = "stopped"
                self.flag_manager.set_flag(pipe_id, "COMPLETED")
                self.prom_status.labels(pipe_id=pipe_id, status="stopped").set(0)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_status",
                        data={"pipe_id": pipe_id, "status": "stopped"}
                    )
                log.debug(f"Pipe durduruldu: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Pipe durdurma hatası: {str(e)} (PIPE009)", context={"source": "stop"})

    async def pause(self, pipe_id: str, config: str = "") -> None:
        """PAUSE PIPE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE010)", context={"source": "pause"})
                self.pipes[pipe_id]["status"] = "paused"
                self.flag_manager.set_flag(pipe_id, "PAUSED")
                self.prom_status.labels(pipe_id=pipe_id, status="paused").set(1)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_status",
                        data={"pipe_id": pipe_id, "status": "paused"}
                    )
                log.debug(f"Pipe duraklatıldı: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Pipe duraklatma hatası: {str(e)} (PIPE011)", context={"source": "pause"})

    async def resume(self, pipe_id: str, config: str = "") -> None:
        """RESUME PIPE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE012)", context={"source": "resume"})
                self.pipes[pipe_id]["status"] = "running"
                self.flag_manager.clear_flag(pipe_id, "PAUSED")
                self.flag_manager.set_flag(pipe_id, "RUNNING")
                self.prom_status.labels(pipe_id=pipe_id, status="running").set(1)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_status",
                        data={"pipe_id": pipe_id, "status": "running"}
                    )
                log.debug(f"Pipe devam ettirildi: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Pipe devam ettirme hatası: {str(e)} (PIPE013)", context={"source": "resume"})

    async def branch(self, pipe_id: str, target_pipe: str, condition: str, config: str) -> None:
        """PIPE BRANCH <id> TO <target_pipe> IF "<condition>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes or target_pipe not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} veya {target_pipe} (PIPE014)", context={"source": "branch"})
                if await self.interpreter.evaluate_condition(condition):
                    await self.redirect(pipe_id, target_pipe, config)
                log.debug(f"Pipe dallanma kontrolü: {pipe_id} -> {target_pipe}")
        except Exception as e:
            raise PdsXPipeError(f"Dallanma hatası: {str(e)} (PIPE015)", context={"source": "branch"})

    async def redirect(self, pipe_id: str, target_pipe: str, config: str) -> None:
        """PIPE REDIRECT <id> TO <target_pipe> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes or target_pipe not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} veya {target_pipe} (PIPE016)", context={"source": "redirect"})
                config_dict = self._parse_config(config)
                self.buffers[target_pipe] = self.buffers.get(pipe_id)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_redirect",
                        data={"pipe_id": pipe_id, "target_pipe": target_pipe}
                    )
                log.debug(f"Pipe yönlendirildi: {pipe_id} -> {target_pipe}")
        except Exception as e:
            raise PdsXPipeError(f"Yönlendirme hatası: {str(e)} (PIPE017)", context={"source": "redirect"})

    async def jump(self, pipe_id: str, stage: str, config: str) -> None:
        """PIPE JUMP <id> TO <stage> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE018)", context={"source": "jump"})
                config_dict = self._parse_config(config)
                self.pipes[pipe_id]["current_stage"] = stage
                log.debug(f"Pipe atlama: {pipe_id}, Aşama: {stage}")
        except Exception as e:
            raise PdsXPipeError(f"Atlama hatası: {str(e)} (PIPE019)", context={"source": "jump"})

    async def iot_connect(self, pipe_id: str, device_id: str, config: str) -> None:
        """PIPE IOT CONNECT <id> DEVICE "<device_id>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE020)", context={"source": "iot_connect"})
                iot_manager = self.interpreter.get_manager("iot")
                await iot_manager.connect(device_id, config)
                self.buffers[pipe_id] = iot_manager.get_data(device_id)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_iot",
                        data={"pipe_id": pipe_id, "device_id": device_id}
                    )
                log.debug(f"IoT bağlantısı: {pipe_id}, Cihaz: {device_id}")
        except Exception as e:
            raise PdsXPipeError(f"IoT bağlantı hatası: {str(e)} (PIPE021)", context={"source": "iot_connect"})

    async def parallel(self, pipe_id: str, cpu_id: str, config: str) -> None:
        """PIPE PARALLEL <id> CPU "<cpu_id>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE022)", context={"source": "parallel"})
                concurrency_manager = self.interpreter.get_manager("concurrency")
                await concurrency_manager.assign_cpu(pipe_id, cpu_id)
                log.debug(f"Paralel yürütme: {pipe_id}, CPU: {cpu_id}")
        except Exception as e:
            raise PdsXPipeError(f"Paralel yürütme hatası: {str(e)} (PIPE023)", context={"source": "parallel"})

    async def recurse(self, pipe_id: str, depth: int, config: str) -> None:
        """PIPE RECURSE <id> DEPTH <depth> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE024)", context={"source": "recurse"})
                config_dict = self._parse_config(config)
                self.flag_manager.set_flag(pipe_id, "RECURSING")
                for _ in range(depth):
                    await self._execute_pipe(pipe_id, f"recurse_{uuid.uuid4().hex[:8]}")
                self.flag_manager.clear_flag(pipe_id, "RECURSING")
                log.debug(f"Rekürsif yürütme: {pipe_id}, Derinlik: {depth}")
        except Exception as e:
            raise PdsXPipeError(f"Rekürsif yürütme hatası: {str(e)} (PIPE025)", context={"source": "recurse"})

    async def monitor_connect(self, pipe_id: str, monitor_id: str, config: str) -> None:
        """PIPE MONITOR CONNECT <id> TO "<monitor_id>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE026)", context={"source": "monitor_connect"})
                monitor_manager = self.interpreter.get_manager("monitor")
                await monitor_manager.connect(pipe_id, monitor_id, config)
                self.flag_manager.set_flag(pipe_id, "MONITORED")
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="monitor",
                        data={"pipe_id": pipe_id, "monitor_id": monitor_id}
                    )
                log.debug(f"Monitör bağlantısı: {pipe_id}, Monitör: {monitor_id}")
        except Exception as e:
            raise PdsXPipeError(f"Monitör bağlantı hatası: {str(e)} (PIPE027)", context={"source": "monitor_connect"})

    async def execute(self, pipe_id: str, command: str, config: str) -> None:
        """PIPE EXECUTE <id> COMMAND "<pdsx_cmd>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE028)", context={"source": "execute"})
                await self.interpreter.execute_command(command)
                self.pipes[pipe_id]["commands"].append(command)
                log.debug(f"Komut çalıştırıldı: {pipe_id}, Komut: {command}")
        except Exception as e:
            raise PdsXPipeError(f"Komut çalıştırma hatası: {str(e)} (PIPE029)", context={"source": "execute"})

    async def function_call(self, pipe_id: str, func_name: str, config: str) -> Any:
        """PIPE FUNCTION CALL <id> FUNCTION "<func_name>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE030)", context={"source": "function_call"})
                if func_name in self.interpreter.function_table:
                    result = await self.interpreter.function_table[func_name]()
                    self.buffers[pipe_id] = result
                    log.debug(f"Fonksiyon çağrıldı: {pipe_id}, Fonksiyon: {func_name}")
                    return result
                raise PdsXPipeError(f"Fonksiyon bulunamadı: {func_name} (PIPE031)", context={"source": "function_call"})
        except Exception as e:
            raise PdsXPipeError(f"Fonksiyon çağırma hatası: {str(e)} (PIPE032)", context={"source": "function_call"})

    async def scope_create(self, pipe_id: str, config: str) -> None:
        """PIPE SCOPE CREATE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE033)", context={"source": "scope_create"})
                scope_manager = self.interpreter.get_manager("scope")
                await scope_manager.create_scope(pipe_id)
                log.debug(f"Kapsam oluşturuldu: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Kapsam oluşturma hatası: {str(e)} (PIPE034)", context={"source": "scope_create"})

    async def scope_destroy(self, pipe_id: str, config: str) -> None:
        """PIPE SCOPE DESTROY <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE035)", context={"source": "scope_destroy"})
                scope_manager = self.interpreter.get_manager("scope")
                await scope_manager.destroy_scope(pipe_id)
                log.debug(f"Kapsam yok edildi: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Kapsam yok etme hatası: {str(e)} (PIPE036)", context={"source": "scope_destroy"})

    async def data_send(self, pipe_id: str, target: str, data: Any, config: str) -> None:
        """PIPE DATA SEND <id> TO "<target>" DATA "<data>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE037)", context={"source": "data_send"})
                config_dict = self._parse_config(config)
                if target.startswith("pipe:"):
                    target_pipe = target.replace("pipe:", "")
                    if target_pipe in self.pipes:
                        self.buffers[target_pipe] = data
                elif self.bus_manager:
                    await self.bus_manager.publish(
                        topic=target,
                        data={"pipe_id": pipe_id, "data": data}
                    )
                log.debug(f"Veri gönderildi: {pipe_id}, Hedef: {target}")
        except Exception as e:
            raise PdsXPipeError(f"Veri gönderme hatası: {str(e)} (PIPE038)", context={"source": "data_send"})

    async def data_receive(self, pipe_id: str, source: str, config: str) -> None:
        """PIPE DATA RECEIVE <id> FROM "<source>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE039)", context={"source": "data_receive"})
                config_dict = self._parse_config(config)
                if source.startswith("pipe:"):
                    source_pipe = source.replace("pipe:", "")
                    if source_pipe in self.pipes:
                        self.buffers[pipe_id] = self.buffers.get(source_pipe)
                elif self.bus_manager:
                    data = await self.bus_manager.subscribe(source)
                    self.buffers[pipe_id] = data["data"]
                log.debug(f"Veri alındı: {pipe_id}, Kaynak: {source}")
        except Exception as e:
            raise PdsXPipeError(f"Veri alma hatası: {str(e)} (PIPE040)", context={"source": "data_receive"})

    async def interrupt(self, pipe_id: str, interrupt_type: str, config: str) -> None:
        """PIPE INTERRUPT <id> TYPE "<type>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE041)", context={"source": "interrupt"})
                config_dict = self._parse_config(config)
                event_manager = self.interpreter.get_manager("event")
                await event_manager.trigger(f"interrupt_{pipe_id}", interrupt_type=interrupt_type)
                self.flag_manager.set_flag(pipe_id, "INTERRUPTED")
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_interrupt",
                        data={"pipe_id": pipe_id, "type": interrupt_type}
                    )
                log.debug(f"Kesme tetiklendi: {pipe_id}, Tür: {interrupt_type}")
        except Exception as e:
            raise PdsXPipeError(f"Kesme hatası: {str(e)} (PIPE042)", context={"source": "interrupt"})

    async def cloud_sync(self, pipe_id: str, cloud_service: str, config: str) -> None:
        """PIPE CLOUD SYNC <id> TO "<cloud_service>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE043)", context={"source": "cloud_sync"})
                config_dict = self._parse_config(config)
                network_manager = self.interpreter.get_manager("network")
                await network_manager.sync_to_cloud(self.buffers.get(pipe_id), cloud_service, config)
                log.debug(f"Bulut senkronizasyonu: {pipe_id}, Servis: {cloud_service}")
        except Exception as e:
            raise PdsXPipeError(f"Bulut senkronizasyon hatası: {str(e)} (PIPE044)", context={"source": "cloud_sync"})

    async def network_connect(self, pipe_id: str, network: str, config: str) -> None:
        """PIPE NETWORK CONNECT <id> TO "<network>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE045)", context={"source": "network_connect"})
                config_dict = self._parse_config(config)
                network_manager = self.interpreter.get_manager("network")
                await network_manager.connect(pipe_id, network, config)
                log.debug(f"Ağ bağlantısı: {pipe_id}, Ağ: {network}")
        except Exception as e:
            raise PdsXPipeError(f"Ağ bağlantı hatası: {str(e)} (PIPE046)", context={"source": "network_connect"})

    async def experiment(self, pipe_id: str, method: str, config: str) -> None:
        """PIPE EXPERIMENT <id> METHOD "<method>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE047)", context={"source": "experiment"})
                config_dict = self._parse_config(config)
                experiment_manager = self.interpreter.get_manager("experiment")
                await experiment_manager.run(pipe_id, method, config)
                log.debug(f"Deneysel yöntem: {pipe_id}, Yöntem: {method}")
        except Exception as e:
            raise PdsXPipeError(f"Deneysel yöntem hatası: {str(e)} (PIPE048)", context={"source": "experiment"})

    async def paradigm_set(self, pipe_id: str, paradigm: str, config: str) -> None:
        """PIPE PARADIGM SET <id> TO "<paradigm>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE049)", context={"source": "paradigm_set"})
                config_dict = self._parse_config(config)
                self.pipes[pipe_id]["paradigm"] = paradigm.lower()
                log.debug(f"Paradigma ayarlandı: {pipe_id}, Paradigma: {paradigm}")
        except Exception as e:
            raise PdsXPipeError(f"Paradigma ayarlama hatası: {str(e)} (PIPE050)", context={"source": "paradigm_set"})

    async def data_structure_use(self, pipe_id: str, data_type: str, config: str) -> None:
        """PIPE DATA STRUCTURE USE <id> TYPE "<type>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE051)", context={"source": "data_structure_use"})
                config_dict = self._parse_config(config)
                data_manager = self.interpreter.get_manager("data")
                self.buffers[pipe_id] = await data_manager.create_structure(data_type, config)
                log.debug(f"Veri yapısı kullanıldı: {pipe_id}, Tür: {data_type}")
        except Exception as e:
            raise PdsXPipeError(f"Veri yapısı kullanma hatası: {str(e)} (PIPE052)", context={"source": "data_structure_use"})

    async def flow_control(self, pipe_id: str, action: str, config: str) -> None:
        """PIPE FLOW CONTROL <id> ACTION "<action>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE053)", context={"source": "flow_control"})
                config_dict = self._parse_config(config)
                if action.lower() == "hold":
                    await self.pause(pipe_id)
                elif action.lower() == "release":
                    await self.resume(pipe_id)
                log.debug(f"Akış kontrolü: {pipe_id}, Eylem: {action}")
        except Exception as e:
            raise PdsXPipeError(f"Akış kontrol hatası: {str(e)} (PIPE054)", context={"source": "flow_control"})

    async def resource_allocate(self, pipe_id: str, resource: str, config: str) -> None:
        """PIPE RESOURCE ALLOCATE <id> RESOURCE "<resource>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE055)", context={"source": "resource_allocate"})
                config_dict = self._parse_config(config)
                resource_manager = self.interpreter.get_manager("resource")
                await resource_manager.allocate(pipe_id, resource, config)
                log.debug(f"Kaynak ayrıldı: {pipe_id}, Kaynak: {resource}")
        except Exception as e:
            raise PdsXPipeError(f"Kaynak ayırma hatası: {str(e)} (PIPE056)", context={"source": "resource_allocate"})

    async def resource_free(self, pipe_id: str, resource: str, config: str) -> None:
        """PIPE RESOURCE FREE <id> RESOURCE "<resource>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE057)", context={"source": "resource_free"})
                config_dict = self._parse_config(config)
                resource_manager = self.interpreter.get_manager("resource")
                await resource_manager.free(pipe_id, resource, config)
                log.debug(f"Kaynak serbest bırakıldı: {pipe_id}, Kaynak: {resource}")
        except Exception as e:
            raise PdsXPipeError(f"Kaynak serbest bırakma hatası: {str(e)} (PIPE058)", context={"source": "resource_free"})

    async def timeout_set(self, pipe_id: str, time: str, config: str) -> None:
        """PIPE TIMEOUT SET <id> TIME "<time>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE059)", context={"source": "timeout_set"})
                config_dict = self._parse_config(config)
                timeout_manager = self.interpreter.get_manager("timeout")
                await timeout_manager.set_timeout(pipe_id, time, config)
                log.debug(f"Zaman aşımı ayarlandı: {pipe_id}, Süre: {time}")
        except Exception as e:
            raise PdsXPipeError(f"Zaman aşımı ayarlama hatası: {str(e)} (PIPE060)", context={"source": "timeout_set"})

    async def retry(self, pipe_id: str, attempts: int, config: str) -> None:
        """PIPE RETRY <id> ATTEMPTS <attempts> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE061)", context={"source": "retry"})
                config_dict = self._parse_config(config)
                for _ in range(attempts):
                    try:
                        await self._execute_pipe(pipe_id, f"retry_{uuid.uuid4().hex[:8]}")
                        break
                    except:
                        continue
                log.debug(f"Yeniden deneme: {pipe_id}, Deneme sayısı: {attempts}")
        except Exception as e:
            raise PdsXPipeError(f"Yeniden deneme hatası: {str(e)} (PIPE062)", context={"source": "retry"})

    async def alert(self, pipe_id: str, condition: str, config: str) -> None:
        """PIPE ALERT <id> CONDITION "<condition>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE063)", context={"source": "alert"})
                config_dict = self._parse_config(config)
                if await self.interpreter.evaluate_condition(condition):
                    event_manager = self.interpreter.get_manager("event")
                    await event_manager.trigger(f"alert_{pipe_id}")
                log.debug(f"Uyarı tetiklendi: {pipe_id}, Koşul: {condition}")
        except Exception as e:
            raise PdsXPipeError(f"Uyarı hatası: {str(e)} (PIPE064)", context={"source": "alert"})

    async def auto_scale(self, pipe_id: str, config: str) -> None:
        """PIPE AUTO SCALE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE065)", context={"source": "auto_scale"})
                config_dict = self._parse_config(config)
                scale_manager = self.interpreter.get_manager("scale")
                await scale_manager.auto_scale(pipe_id, config)
                log.debug(f"Otomatik ölçeklendirme: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Otomatik ölçeklendirme hatası: {str(e)} (PIPE066)", context={"source": "auto_scale"})

    async def distribute(self, pipe_id: str, nodes: str, config: str) -> None:
        """PIPE DISTRIBUTE <id> TO "<nodes>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE067)", context={"source": "distribute"})
                config_dict = self._parse_config(config)
                network_manager = self.interpreter.get_manager("network")
                await network_manager.distribute(pipe_id, nodes.split(","), config)
                log.debug(f"Dağıtık yürütme: {pipe_id}, Düğümler: {nodes}")
        except Exception as e:
            raise PdsXPipeError(f"Dağıtık yürütme hatası: {str(e)} (PIPE068)", context={"source": "distribute"})

    async def sync(self, pipe_id: str, other_pipe: str, config: str) -> None:
        """PIPE SYNC <id> WITH "<other_pipe>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes or other_pipe not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} veya {other_pipe} (PIPE069)", context={"source": "sync"})
                config_dict = self._parse_config(config)
                await self.start(other_pipe, self.pipes[other_pipe]["mode"])
                log.debug(f"Pipe senkronizasyonu: {pipe_id}, Diğer: {other_pipe}")
        except Exception as e:
            raise PdsXPipeError(f"Senkronizasyon hatası: {str(e)} (PIPE070)", context={"source": "sync"})

    async def async_mode(self, pipe_id: str, config: str) -> None:
        """PIPE ASYNC <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE071)", context={"source": "async_mode"})
                config_dict = self._parse_config(config)
                self.pipes[pipe_id]["mode"] = "async"
                log.debug(f"Asenkron mod: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Asenkron mod hatası: {str(e)} (PIPE072)", context={"source": "async_mode"})

    async def log(self, pipe_id: str, message: str, config: str) -> None:
        """PIPE LOG <id> MESSAGE "<message>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE073)", context={"source": "log"})
                config_dict = self._parse_config(config)
                log.info(f"Pipe log: {pipe_id}, Mesaj: {message}")
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_log",
                        data={"pipe_id": pipe_id, "message": message}
                    )
                log.debug(f"Log kaydedildi: {pipe_id}, Mesaj: {message}")
        except Exception as e:
            raise PdsXPipeError(f"Log hatası: {str(e)} (PIPE074)", context={"source": "log"})

    async def event_trigger(self, pipe_id: str, event: str, config: str) -> None:
        """PIPE EVENT TRIGGER <id> <event> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE075)", context={"source": "event_trigger"})
                config_dict = self._parse_config(config)
                event_manager = self.interpreter.get_manager("event")
                await event_manager.trigger(event)
                log.debug(f"Olay tetiklendi: {pipe_id}, Olay: {event}")
        except Exception as e:
            raise PdsXPipeError(f"Olay tetikleme hatası: {str(e)} (PIPE076)", context={"source": "event_trigger"})

    async def error_handle(self, pipe_id: str, handler: str, config: str) -> None:
        """PIPE ERROR HANDLE <id> HANDLER "<handler>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE077)", context={"source": "error_handle"})
                config_dict = self._parse_config(config)
                self.pipes[pipe_id]["error_handler"] = handler
                log.debug(f"Hata işleyici ayarlandı: {pipe_id}, İşleyici: {handler}")
        except Exception as e:
            raise PdsXPipeError(f"Hata işleyici ayarlama hatası: {str(e)} (PIPE078)", context={"source": "error_handle"})

    async def secure(self, pipe_id: str, config: str) -> None:
        """PIPE SECURE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE079)", context={"source": "secure"})
                config_dict = self._parse_config(config)
                security_manager = self.interpreter.get_manager("security")
                await security_manager.encrypt_pipe(pipe_id, config)
                self.flag_manager.set_flag(pipe_id, "SECURED")
                log.debug(f"Pipe güvenli hale getirildi: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Güvenlik hatası: {str(e)} (PIPE080)", context={"source": "secure"})

    async def unsecure(self, pipe_id: str, config: str) -> None:
        """PIPE UNSECURE <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE081)", context={"source": "unsecure"})
                config_dict = self._parse_config(config)
                security_manager = self.interpreter.get_manager("security")
                await security_manager.decrypt_pipe(pipe_id, config)
                self.flag_manager.clear_flag(pipe_id, "SECURED")
                log.debug(f"Pipe güvensiz hale getirildi: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Güvensiz hale getirme hatası: {str(e)} (PIPE082)", context={"source": "unsecure"})

    async def group(self, group_id: str, pipe_ids: List[str], config: str) -> None:
        """PIPE GROUP <group_id> <pipe_id1> <pipe_id2> ... CONFIG "<options>" """
        try:
            async with self.async_lock:
                config_dict = self._parse_config(config)
                for pipe_id in pipe_ids:
                    if pipe_id not in self.pipes:
                        raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE083)", context={"source": "group"})
                self.pipes[group_id] = {"group": pipe_ids, "status": "ready", "config": config_dict}
                self.flag_manager.set_flag(group_id, "GROUPED")
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_group",
                        data={"group_id": group_id, "pipe_ids": pipe_ids}
                    )
                log.debug(f"Pipe grubu oluşturuldu: {group_id}, Pipe’lar: {pipe_ids}")
        except Exception as e:
            raise PdsXPipeError(f"Grup oluşturma hatası: {str(e)} (PIPE084)", context={"source": "group"})

    async def ungroup(self, group_id: str, config: str) -> None:
        """PIPE UNGROUP <group_id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if group_id not in self.pipes:
                    raise PdsXPipeError(f"Grup bulunamadı: {group_id} (PIPE085)", context={"source": "ungroup"})
                config_dict = self._parse_config(config)
                del self.pipes[group_id]
                self.flag_manager.set_flag(group_id, "UNGROUPED")
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_group",
                        data={"group_id": group_id, "action": "ungrouped"}
                    )
                log.debug(f"Pipe grubu çözüldü: {group_id}")
        except Exception as e:
            raise PdsXPipeError(f"Grup çözme hatası: {str(e)} (PIPE086)", context={"source": "ungroup"})

    async def broadcast(self, pipe_id: str, config: str) -> None:
        """PIPE BROADCAST <id> CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE087)", context={"source": "broadcast"})
                config_dict = self._parse_config(config)
                if self.bus_manager:
                    await self.bus_manager.publish(
                        topic="pipe_broadcast",
                        data={"pipe_id": pipe_id, "data": self.buffers.get(pipe_id)}
                    )
                self.flag_manager.set_flag(pipe_id, "BROADCASTED")
                log.debug(f"Pipe yayınlandı: {pipe_id}")
        except Exception as e:
            raise PdsXPipeError(f"Yayın hatası: {str(e)} (PIPE088)", context={"source": "broadcast"})

    async def quantum_execute(self, pipe_id: str, circuit: str, config: str) -> None:
        """PIPE QUANTUM EXECUTE <id> CIRCUIT "<circuit>" CONFIG "<options>" """
        try:
            async with self.async_lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE089)", context={"source": "quantum_execute"})
                config_dict = self._parse_config(config)
                circuit = QuantumCircuit(2, 2)
                for gate in circuit.split("-"):
                    if gate.startswith("H"):
                        qubit = int(gate.replace("H-Qubit", ""))
                        circuit.h(qubit)
                    elif gate.startswith("CX"):
                        qubits = [int(q) for q in gate.replace("CX", "").split(",")]
                        circuit.cx(qubits[0], qubits[1])
                backend = Aer.get_backend("qasm_simulator")
                job = execute(circuit, backend, shots=1024)
                result = job.result()
                self.buffers[pipe_id] = result.get_counts()
                log.debug(f"Kuantum devresi çalıştırıldı: {pipe_id}, Devre: {circuit}")
        except Exception as e:
            raise PdsXPipeError(f"Kuantum çalıştırma hatası: {str(e)} (PIPE090)", context={"source": "quantum_execute"})

    def get_status(self, pipe_id: str) -> str:
        """GET_PIPE_STATUS(id)"""
        try:
            with self.lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE101)", context={"source": "get_status"})
                status = self.pipes[pipe_id]["status"]
                log.debug(f"Pipe durumu alındı: {pipe_id}, Durum: {status}")
                return status
        except Exception as e:
            raise PdsXPipeError(f"Durum alma hatası: {str(e)} (PIPE102)", context={"source": "get_status"})

    def get_stats(self, pipe_id: str) -> Dict[str, Any]:
        """GET_PIPE_STATS(id)"""
        try:
            with self.lock:
                if pipe_id not in self.pipes:
                    raise PdsXPipeError(f"Pipe bulunamadı: {pipe_id} (PIPE103)", context={"source": "get_stats"})
                stats = {
                    "runtime": time.time() - self.pipes[pipe_id]["timestamp"],
                    "instances": len(self.instance_manager.instances[pipe_id]),
                    "stages_completed": len(self.pipes[pipe_id]["stages"])
                }
                log.debug(f"Pipe istatistikleri alındı: {pipe_id}, İstatistikler: {stats}")
                return stats
        except Exception as e:
            raise PdsXPipeError(f"İstatistik alma hatası: {str(e)} (PIPE104)", context={"source": "get_stats"})

    async def _publish_bus_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Bus manager üzerinden olay yayınla (opsiyonel)"""
        if hasattr(self, 'bus_manager') and self.bus_manager is not None:
            try:
                await self.bus_manager.publish(topic=topic, data=data)
            except Exception as e:
                logging.warning(f"Bus manager publish hatası ({topic}): {e}")

    async def _subscribe_bus_event(self, source: str) -> Optional[Dict[str, Any]]:
        """Bus manager'dan olay al (opsiyonel)"""
        if hasattr(self, 'bus_manager') and self.bus_manager is not None:
            try:
                return await self.bus_manager.subscribe(source)
            except Exception as e:
                logging.warning(f"Bus manager subscribe hatası ({source}): {e}")
        return None

    def parse_pipe_command(self, command: str, interpreter):
        """Pipe komutlarını ayrıştırır ve çalıştırır."""
        command_upper = command.strip().upper()
        if command_upper.startswith("PIPE ") and "END PIPE" in command_upper:
            asyncio.run(self.parse_pipe_block(command, interpreter))
            return
        try:
            match = re.match(r"PIPE DEFINE\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?(?:\s+AS\s+(\w+))?", command, re.IGNORECASE)
            if match:
                pipe_id, config, var = match.groups()
                pipe_id = asyncio.run(self.define(pipe_id, config or ""))
                if var:
                    self.interpreter.current_scope()[var] = pipe_id
                return
            # Diğer komutlar için benzer regex ve çağrılar...
            log.warning(f"Bilinmeyen komut: {command}")
        except Exception as e:
            raise PdsXPipeError(f"Komut işleme hatası: {str(e)} (PIPE900)", context={"source": "parse_pipe_command"})

    def _process_subroutine(self, command: str, subroutines: Dict[str, Dict[str, List[str]]], current_sub: Optional[str] = None) -> Optional[str]:
        """Alt program komutlarını işle ve aktif alt program adını döndür"""
        parts = command.split()
        if len(parts) >= 2:
            if parts[0].upper() == "SUBROUTINE":
                current_sub = parts[1]
                if current_sub not in subroutines:
                    subroutines[current_sub] = {"commands": []}
            elif current_sub is not None and current_sub in subroutines:
                subroutines[current_sub]["commands"].append(command)
        return current_sub

    def _process_pipe_command(self, command: str, pipe_id: str) -> None:
        """Pipe komutunu işle ve gerekli bus eventlerini yayınla"""
        parts = command.split()
        cmd_type = parts[0].upper() if parts else ""
        
        status_map = {
            "START": "running",
            "STOP": "stopped",
            "PAUSE": "paused",
            "RESUME": "running"
        }
        
        if cmd_type in status_map:
            asyncio.create_task(self._publish_bus_event(
                topic="pipe_status",
                data={"pipe_id": pipe_id, "status": status_map[cmd_type]}
            ))

if __name__ == "__main__":
    print("pipe3.py bağımsız çalıştırılamaz. PDSxU ile kullanın.")