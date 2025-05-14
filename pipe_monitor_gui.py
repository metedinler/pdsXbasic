# pipe_monitor_gui.py - PDS-X BASIC v14u Kanal, İzleme ve GUI Yönetim Kütüphanesi
# Version: 1.0.0
# Date: May 13, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import threading
import asyncio
import time
import json
import pickle
import base64
import gzip
import zlib
import uuid
import hashlib
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
import websockets
import os
import multiprocessing
from pdsx_exception import PdsXException
import functools
from save_load_system import format_registry, supported_encodings, compression_methods, decompression_methods

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pipe_monitor_gui")

# Dekoratör
def synchronized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with args[0].lock:
            return fn(*args, **kwargs)
    return wrapped

class PipeConnection:
    """Kanal bağlantı sınıfı."""
    def __init__(self, pipe_id: str, pipe_type: str, params: Dict, encoding: str = "utf-8"):
        self.pipe_id = pipe_id
        self.pipe_type = pipe_type.lower()
        self.params = params
        self.encoding = encoding
        self.connection = None
        self.buffer = deque()
        self.lock = threading.Lock()
        self.connect()

    @synchronized
    def connect(self) -> None:
        """Kanala bağlanır."""
        try:
            if self.pipe_type == "websocket":
                self.connection = None  # WebSocket asenkron bağlanır
            elif self.pipe_type == "file":
                self.connection = open(self.params.get("path"), "a+b", encoding=self.encoding)
            elif self.pipe_type == "named_pipe":
                if os.name == "nt":
                    self.connection = open(self.params.get("path"), "r+b")
                else:
                    self.connection = open(self.params.get("path"), "r+")
            else:
                raise PdsXException(f"Desteklenmeyen kanal tipi: {self.pipe_type}")
            log.debug(f"Kanal bağlantısı kuruldu: pipe_id={self.pipe_id}, type={self.pipe_type}")
        except Exception as e:
            log.error(f"Kanal bağlantı hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Kanal bağlantı hatası: {str(e)}")

    @synchronized
    def write(self, data: Any, compress: Optional[str] = None) -> None:
        """Kanala veri yazar."""
        try:
            serialized = format_registry["json"]["serialize"](data)
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            serialized = compression_methods[compress](serialized)
            
            if self.pipe_type == "websocket":
                raise PdsXException("WebSocket yazma asenkron olmalı")
            else:
                self.connection.write(serialized)
                self.connection.flush()
            log.debug(f"Kanal yazma: pipe_id={self.pipe_id}, compress={compress}")
        except Exception as e:
            log.error(f"Kanal yazma hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Kanal yazma hatası: {str(e)}")

    @synchronized
    def read(self, decompress: Optional[str] = None) -> Any:
        """Kanaldan veri okur."""
        try:
            decompress = decompress or "none"
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            if self.pipe_type == "websocket":
                raise PdsXException("WebSocket okuma asenkron olmalı")
            else:
                serialized = self.connection.readline()
                if not serialized:
                    return None
                serialized = decompression_methods[decompress](serialized)
                return format_registry["json"]["deserialize"](serialized)
            log.debug(f"Kanal okuma: pipe_id={self.pipe_id}, decompress={decompress}")
        except Exception as e:
            log.error(f"Kanal okuma hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Kanal okuma hatası: {str(e)}")

    async def async_write(self, data: Any, compress: Optional[str] = None) -> None:
        """Kanala asenkron veri yazar."""
        try:
            serialized = format_registry["json"]["serialize"](data)
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            serialized = compression_methods[compress](serialized)
            
            if self.pipe_type == "websocket":
                async with websockets.connect(self.params.get("url")) as ws:
                    await ws.send(serialized)
            else:
                self.connection.write(serialized)
                self.connection.flush()
            log.debug(f"Asenkron kanal yazma: pipe_id={self.pipe_id}, compress={compress}")
        except Exception as e:
            log.error(f"Asenkron kanal yazma hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Asenkron kanal yazma hatası: {str(e)}")

    async def async_read(self, decompress: Optional[str] = None) -> Any:
        """Kanaldan asenkron veri okur."""
        try:
            decompress = decompress or "none"
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            if self.pipe_type == "websocket":
                async with websockets.connect(self.params.get("url")) as ws:
                    serialized = await ws.recv()
                    serialized = decompression_methods[decompress](serialized)
                    return format_registry["json"]["deserialize"](serialized)
            else:
                serialized = self.connection.readline()
                if not serialized:
                    return None
                serialized = decompression_methods[decompress](serialized)
                return format_registry["json"]["deserialize"](serialized)
            log.debug(f"Asenkron kanal okuma: pipe_id={self.pipe_id}, decompress={decompress}")
        except Exception as e:
            log.error(f"Asenkron kanal okuma hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Asenkron kanal okuma hatası: {str(e)}")

    @synchronized
    def close(self) -> None:
        """Kanalı kapatır."""
        try:
            if self.connection and self.pipe_type != "websocket":
                self.connection.close()
            log.debug(f"Kanal kapatıldı: pipe_id={self.pipe_id}")
        except Exception as e:
            log.error(f"Kanal kapatma hatası: pipe_id={self.pipe_id}, hata={str(e)}")
            raise PdsXException(f"Kanal kapatma hatası: {str(e)}")

class MonitorSession:
    """İzleme oturumu sınıfı."""
    def __init__(self, session_id: str, pipe_id: str, interval: float = 1.0):
        self.session_id = session_id
        self.pipe_id = pipe_id
        self.interval = interval
        self.metrics = {
            "data_rate": 0.0,
            "error_count": 0,
            "last_update": time.time(),
            "data_count": 0
        }
        self.running = False
        self.lock = threading.Lock()

    @synchronized
    def start(self, pipe: 'PipeConnection') -> None:
        """İzleme oturumunu başlatır."""
        try:
            self.running = True
            threading.Thread(target=self._monitor, args=(pipe,), daemon=True).start()
            log.debug(f"İzleme oturumu başlatıldı: session_id={self.session_id}, pipe_id={self.pipe_id}")
        except Exception as e:
            log.error(f"İzleme başlatma hatası: session_id={self.session_id}, hata={str(e)}")
            raise PdsXException(f"İzleme başlatma hatası: {str(e)}")

    def _monitor(self, pipe: 'PipeConnection') -> None:
        """İzleme döngüsü."""
        while self.running:
            try:
                start_time = time.time()
                data = pipe.read()
                if data:
                    self.metrics["data_count"] += 1
                    self.metrics["data_rate"] = self.metrics["data_count"] / (time.time() - self.metrics["last_update"])
                    self.metrics["last_update"] = time.time()
                time.sleep(self.interval)
            except Exception as e:
                self.metrics["error_count"] += 1
                log.warning(f"İzleme hatası: session_id={self.session_id}, hata={str(e)}")

    @synchronized
    def stop(self) -> None:
        """İzleme oturumunu durdurur."""
        try:
            self.running = False
            log.debug(f"İzleme oturumu durduruldu: session_id={self.session_id}")
        except Exception as e:
            log.error(f"İzleme durdurma hatası: session_id={self.session_id}, hata={str(e)}")
            raise PdsXException(f"İzleme durdurma hatası: {str(e)}")

class GUIWindow:
    """GUI pencere sınıfı."""
    def __init__(self, window_id: str, title: str, geometry: str = "800x600"):
        self.window_id = window_id
        self.title = title
        self.geometry = geometry
        self.root = tk.Tk()
        self.widgets = {}
        self.lock = threading.Lock()
        self._initialize()

    @synchronized
    def _initialize(self) -> None:
        """Pencereyi başlatır."""
        try:
            self.root.title(self.title)
            self.root.geometry(self.geometry)
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            log.debug(f"GUI penceresi başlatıldı: window_id={self.window_id}, title={self.title}")
        except Exception as e:
            log.error(f"GUI başlatma hatası: window_id={self.window_id}, hata={str(e)}")
            raise PdsXException(f"GUI başlatma hatası: {str(e)}")

    @synchronized
    def add_widget(self, widget_type: str, name: str, params: Dict) -> None:
        """Widget ekler."""
        try:
            if widget_type == "label":
                widget = ttk.Label(self.root, text=params.get("text", ""))
                widget.pack()
            elif widget_type == "button":
                widget = ttk.Button(self.root, text=params.get("text", ""), command=params.get("command", lambda: None))
                widget.pack()
            elif widget_type == "entry":
                widget = ttk.Entry(self.root)
                widget.pack()
            else:
                raise PdsXException(f"Desteklenmeyen widget tipi: {widget_type}")
            self.widgets[name] = widget
            log.debug(f"Widget eklendi: window_id={self.window_id}, name={name}, type={widget_type}")
        except Exception as e:
            log.error(f"Widget ekleme hatası: window_id={self.window_id}, hata={str(e)}")
            raise PdsXException(f"Widget ekleme hatası: {str(e)}")

    @synchronized
    def show(self) -> None:
        """Pencereyi gösterir."""
        try:
            self.root.mainloop()
            log.debug(f"GUI penceresi gösterildi: window_id={self.window_id}")
        except Exception as e:
            log.error(f"GUI gösterme hatası: window_id={self.window_id}, hata={str(e)}")
            raise PdsXException(f"GUI gösterme hatası: {str(e)}")

    @synchronized
    def close(self) -> None:
        """Pencereyi kapatır."""
        try:
            self.root.destroy()
            log.debug(f"GUI penceresi kapatıldı: window_id={self.window_id}")
        except Exception as e:
            log.error(f"GUI kapatma hatası: window_id={self.window_id}, hata={str(e)}")
            raise PdsXException(f"GUI kapatma hatası: {str(e)}")

class QuantumPipeCorrelator:
    """Kuantum tabanlı kanal korelasyon sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (pipe_id1, pipe_id2, score)}

    def correlate(self, pipe1: PipeConnection, pipe2: PipeConnection) -> str:
        """İki kanalı kuantum simülasyonuyla ilişkilendirir."""
        try:
            set1 = set(str(pipe1.buffer))
            set2 = set(str(pipe2.buffer))
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (pipe1.pipe_id, pipe2.pipe_id, score)
            log.debug(f"Kuantum korelasyon: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumPipeCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumPipeCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[str, str, float]]:
        """Korelasyonu döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumPipeCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumPipeCorrelator get_correlation hatası: {str(e)}")

class HoloPipeCompressor:
    """Holografik kanal veri sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [serialized_data]}

    def compress(self, pipe: PipeConnection) -> str:
        """Kanal verisini holografik olarak sıkıştırır."""
        try:
            serialized = pickle.dumps(list(pipe.buffer))
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            log.debug(f"Holografik veri sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloPipeCompressor compress hatası: {str(e)}")
            raise PdsXException(f"HoloPipeCompressor compress hatası: {str(e)}")

    def decompress(self, pattern: str) -> Optional[List]:
        """Veriyi geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                serialized = self.storage[pattern][-1]
                return pickle.loads(serialized)
            return None
        except Exception as e:
            log.error(f"HoloPipeCompressor decompress hatası: {str(e)}")
            raise PdsXException(f"HoloPipeCompressor decompress hatası: {str(e)}")

class SmartPipeOptimizer:
    """AI tabanlı kanal optimizasyon sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(data_size, transfer_time, timestamp)]

    def optimize(self, data_size: int, transfer_time: float) -> str:
        """Kanalı optimize bir şekilde planlar."""
        try:
            features = np.array([[data_size, transfer_time, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:
                    strategy = "PARALLEL"
                    log.warning(f"Kanal optimize edildi: strategy={strategy}, score={anomaly_score}")
                    return strategy
            return "SEQUENTIAL"
        except Exception as e:
            log.error(f"SmartPipeOptimizer optimize hatası: {str(e)}")
            raise PdsXException(f"SmartPipeOptimizer optimize hatası: {str(e)}")

class TemporalPipeGraph:
    """Zaman temelli kanal ilişkileri grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {pipe_id: timestamp}
        self.edges = defaultdict(list)  # {pipe_id: [(related_pipe_id, weight)]}

    def add_pipe(self, pipe_id: str, timestamp: float) -> None:
        """Kanalı grafiğe ekler."""
        try:
            self.vertices[pipe_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: pipe_id={pipe_id}")
        except Exception as e:
            log.error(f"TemporalPipeGraph add_pipe hatası: {str(e)}")
            raise PdsXException(f"TemporalPipeGraph add_pipe hatası: {str(e)}")

    def add_relation(self, pipe_id1: str, pipe_id2: str, weight: float) -> None:
        """Kanallar arasında ilişki kurar."""
        try:
            self.edges[pipe_id1].append((pipe_id2, weight))
            self.edges[pipe_id2].append((pipe_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {pipe_id1} <-> {pipe_id2}")
        except Exception as e:
            log.error(f"TemporalPipeGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalPipeGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Kanal grafiğini analiz eder."""
        try:
            clusters = defaultdict(list)
            visited = set()
            
            def dfs(vid: str, cluster_id: str):
                visited.add(vid)
                clusters[cluster_id].append(vid)
                for neighbor_id, _ in self.edges[vid]:
                    if neighbor_id not in visited:
                        dfs(neighbor_id, cluster_id)
            
            for vid in self.vertices:
                if vid not in visited:
                    dfs(vid, str(uuid.uuid4()))
            
            log.debug(f"Temporal graph analiz edildi: clusters={len(clusters)}")
            return clusters
        except Exception as e:
            log.error(f"TemporalPipeGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalPipeGraph analyze hatası: {str(e)}")

class PipeShield:
    """Tahmini kanal hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(data_size, transfer_time, timestamp)]

    def train(self, data_size: int, transfer_time: float) -> None:
        """Kanal verileriyle modeli eğitir."""
        try:
            features = np.array([data_size, transfer_time, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("PipeShield modeli eğitildi")
        except Exception as e:
            log.error(f"PipeShield train hatası: {str(e)}")
            raise PdsXException(f"PipeShield train hatası: {str(e)}")

    def predict(self, data_size: int, transfer_time: float) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[data_size, transfer_time, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: data_size={data_size}")
            return is_anomaly
        except Exception as e:
            log.error(f"PipeShield predict hatası: {str(e)}")
            raise PdsXException(f"PipeShield predict hatası: {str(e)}")

class PipeMonitorGUIManager:
    """Kanal, izleme ve GUI yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.pipes = {}  # {pipe_id: PipeConnection}
        self.monitors = {}  # {session_id: MonitorSession}
        self.windows = {}  # {window_id: GUIWindow}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumPipeCorrelator()
        self.holo_compressor = HoloPipeCompressor()
        self.smart_optimizer = SmartPipeOptimizer()
        self.temporal_graph = TemporalPipeGraph()
        self.pipe_shield = PipeShield()
        self.lock = threading.Lock()
        self.metadata = {
            "pipe_monitor_gui": {
                "version": "1.0.0",
                "dependencies": [
                    "tkinter", "matplotlib", "numpy", "scikit-learn", "graphviz",
                    "websockets", "pdsx_exception", "save_load_system"
                ]
            }
        }
        self.max_pipes = 100

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron kanal döngüsü başlatıldı")

    @synchronized
    def create_pipe(self, pipe_type: str, params: Dict, encoding: str = "utf-8") -> str:
        """Yeni bir kanal oluşturur."""
        try:
            pipe_id = str(uuid.uuid4())
            pipe = PipeConnection(pipe_id, pipe_type, params, encoding)
            self.pipes[pipe_id] = pipe
            self.temporal_graph.add_pipe(pipe_id, time.time())
            log.debug(f"Kanal oluşturuldu: pipe_id={pipe_id}, type={pipe_type}")
            return pipe_id
        except Exception as e:
            log.error(f"Kanal oluşturma hatası: {str(e)}")
            raise PdsXException(f"Kanal oluşturma hatası: {str(e)}")

    @synchronized
    def start_monitor(self, pipe_id: str, interval: float = 1.0) -> str:
        """Yeni bir izleme oturumu başlatır."""
        try:
            if pipe_id not in self.pipes:
                raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
            session_id = str(uuid.uuid4())
            monitor = MonitorSession(session_id, pipe_id, interval)
            self.monitors[session_id] = monitor
            monitor.start(self.pipes[pipe_id])
            log.debug(f"İzleme başlatıldı: session_id={session_id}, pipe_id={pipe_id}")
            return session_id
        except Exception as e:
            log.error(f"İzleme başlatma hatası: {str(e)}")
            raise PdsXException(f"İzleme başlatma hatası: {str(e)}")

    @synchronized
    def create_gui(self, title: str, geometry: str = "800x600") -> str:
        """Yeni bir GUI penceresi oluşturur."""
        try:
            window_id = str(uuid.uuid4())
            window = GUIWindow(window_id, title, geometry)
            self.windows[window_id] = window
            log.debug(f"GUI penceresi oluşturuldu: window_id={window_id}, title={title}")
            return window_id
        except Exception as e:
            log.error(f"GUI oluşturma hatası: {str(e)}")
            raise PdsXException(f"GUI oluşturma hatası: {str(e)}")

    def parse_pipe_monitor_gui_command(self, command: str) -> None:
        """Kanal, izleme ve GUI komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            # PIPE CREATE
            if command_upper.startswith("PIPE CREATE "):
                match = re.match(r"PIPE CREATE\s+(\w+)\s+\[(.+?)\]\s*(?:ENCODING\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_type, params_str, encoding, var_name = match.groups()
                    params = eval(params_str, self.interpreter.current_scope())
                    encoding = encoding or "utf-8"
                    pipe_id = self.create_pipe(pipe_type, params, encoding)
                    self.interpreter.current_scope()[var_name] = pipe_id
                else:
                    raise PdsXException("PIPE CREATE komutunda sözdizimi hatası")

            # PIPE WRITE
            elif command_upper.startswith("PIPE WRITE "):
                match = re.match(r"PIPE WRITE\s+(\w+)\s+(.+?)\s*(?:COMPRESS\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, data_str, compress, var_name = match.groups()
                    if pipe_id not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
                    data = self.interpreter.evaluate_expression(data_str)
                    self.pipes[pipe_id].write(data, compress)
                    self.pipe_shield.train(len(str(data)), 0.1)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("PIPE WRITE komutunda sözdizimi hatası")

            # PIPE READ
            elif command_upper.startswith("PIPE READ "):
                match = re.match(r"PIPE READ\s+(\w+)\s*(?:DECOMPRESS\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, decompress, var_name = match.groups()
                    if pipe_id not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
                    data = self.pipes[pipe_id].read(decompress)
                    self.pipe_shield.train(len(str(data)), 0.1)
                    self.interpreter.current_scope()[var_name] = data
                else:
                    raise PdsXException("PIPE READ komutunda sözdizimi hatası")

            # PIPE ASYNC WRITE
            elif command_upper.startswith("PIPE ASYNC WRITE "):
                match = re.match(r"PIPE ASYNC WRITE\s+(\w+)\s+(.+?)\s*(?:COMPRESS\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, data_str, compress, var_name = match.groups()
                    if pipe_id not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
                    data = self.interpreter.evaluate_expression(data_str)
                    asyncio.run(self.pipes[pipe_id].async_write(data, compress))
                    self.pipe_shield.train(len(str(data)), 0.1)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("PIPE ASYNC WRITE komutunda sözdizimi hatası")

            # PIPE ASYNC READ
            elif command_upper.startswith("PIPE ASYNC READ "):
                match = re.match(r"PIPE ASYNC READ\s+(\w+)\s*(?:DECOMPRESS\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, decompress, var_name = match.groups()
                    if pipe_id not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
                    data = asyncio.run(self.pipes[pipe_id].async_read(decompress))
                    self.pipe_shield.train(len(str(data)), 0.1)
                    self.interpreter.current_scope()[var_name] = data
                else:
                    raise PdsXException("PIPE ASYNC READ komutunda sözdizimi hatası")

            # MONITOR START
            elif command_upper.startswith("MONITOR START "):
                match = re.match(r"MONITOR START\s+(\w+)\s*(?:INTERVAL\s+(\d*\.?\d*))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, interval, var_name = match.groups()
                    interval = float(interval) if interval else 1.0
                    session_id = self.start_monitor(pipe_id, interval)
                    self.interpreter.current_scope()[var_name] = session_id
                else:
                    raise PdsXException("MONITOR START komutunda sözdizimi hatası")

            # MONITOR STOP
            elif command_upper.startswith("MONITOR STOP "):
                match = re.match(r"MONITOR STOP\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    session_id, var_name = match.groups()
                    if session_id not in self.monitors:
                        raise PdsXException(f"İzleme oturumu bulunamadı: {session_id}")
                    self.monitors[session_id].stop()
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("MONITOR STOP komutunda sözdizimi hatası")

            # GUI CREATE
            elif command_upper.startswith("GUI CREATE "):
                match = re.match(r"GUI CREATE\s+\"([^\"]+)\"\s*(?:GEOMETRY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    title, geometry, var_name = match.groups()
                    geometry = geometry or "800x600"
                    window_id = self.create_gui(title, geometry)
                    self.interpreter.current_scope()[var_name] = window_id
                else:
                    raise PdsXException("GUI CREATE komutunda sözdizimi hatası")

            # GUI ADD WIDGET
            elif command_upper.startswith("GUI ADD WIDGET "):
                match = re.match(r"GUI ADD WIDGET\s+(\w+)\s+(\w+)\s+\"([^\"]+)\"\s+\[(.+?)\]\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    window_id, widget_type, name, params_str, var_name = match.groups()
                    if window_id not in self.windows:
                        raise PdsXException(f"Pencere bulunamadı: {window_id}")
                    params = eval(params_str, self.interpreter.current_scope())
                    self.windows[window_id].add_widget(widget_type, name, params)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("GUI ADD WIDGET komutunda sözdizimi hatası")

            # GUI SHOW
            elif command_upper.startswith("GUI SHOW "):
                match = re.match(r"GUI SHOW\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    window_id, var_name = match.groups()
                    if window_id not in self.windows:
                        raise PdsXException(f"Pencere bulunamadı: {window_id}")
                    threading.Thread(target=self.windows[window_id].show, daemon=True).start()
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("GUI SHOW komutunda sözdizimi hatası")

            # GUI PLOT
            elif command_upper.startswith("GUI PLOT "):
                match = re.match(r"GUI PLOT\s+(\w+)\s+\[(.+?)\]\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    session_id, data_str, output_path, var_name = match.groups()
                    if session_id not in self.monitors:
                        raise PdsXException(f"İzleme oturumu bulunamadı: {session_id}")
                    data = eval(data_str, self.interpreter.current_scope())
                    plt.figure()
                    plt.plot(data)
                    plt.savefig(output_path)
                    plt.close()
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("GUI PLOT komutunda sözdizimi hatası")

            # PIPE ANALYZE
            elif command_upper.startswith("PIPE ANALYZE "):
                match = re.match(r"PIPE ANALYZE\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_pipes": len(self.pipes),
                        "total_monitors": len(self.monitors),
                        "total_windows": len(self.windows),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [pid for pid, p in self.pipes.items() if self.pipe_shield.predict(len(str(p.buffer)), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("PIPE ANALYZE komutunda sözdizimi hatası")

            # PIPE VISUALIZE
            elif command_upper.startswith("PIPE VISUALIZE "):
                match = re.match(r"PIPE VISUALIZE\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    output_path, format_type, var_name = match.groups()
                    format_type = format_type or "png"
                    dot = graphviz.Digraph(format=format_type)
                    for pid, pipe in self.pipes.items():
                        node_label = f"ID: {pid}\nType: {pipe.pipe_type}\nEncoding: {pipe.encoding}"
                        dot.node(pid, node_label, color="blue")
                    for sid, monitor in self.monitors.items():
                        node_label = f"ID: {sid}\nType: Monitor\nPipe: {monitor.pipe_id}"
                        dot.node(sid, node_label, color="green")
                    for wid, window in self.windows.items():
                        node_label = f"ID: {wid}\nType: GUI\nTitle: {window.title}"
                        dot.node(wid, node_label, color="red")
                    for pid1 in self.temporal_graph.edges:
                        for pid2, weight in self.temporal_graph.edges[pid1]:
                            dot.edge(pid1, pid2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    self.interpreter.current_scope()[var_name] = True
                    log.debug(f"Kanal görselleştirildi: path={output_path}.{format_type}")
                else:
                    raise PdsXException("PIPE VISUALIZE komutunda sözdizimi hatası")

            # PIPE QUANTUM
            elif command_upper.startswith("PIPE QUANTUM "):
                match = re.match(r"PIPE QUANTUM\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id1, pipe_id2, var_name = match.groups()
                    if pipe_id1 not in self.pipes or pipe_id2 not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id1} veya {pipe_id2}")
                    correlation_id = self.quantum_correlator.correlate(self.pipes[pipe_id1], self.pipes[pipe_id2])
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("PIPE QUANTUM komutunda sözdizimi hatası")

            # PIPE HOLO
            elif command_upper.startswith("PIPE HOLO "):
                match = re.match(r"PIPE HOLO\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id, var_name = match.groups()
                    if pipe_id not in self.pipes:
                        raise PdsXException(f"Kanal bulunamadı: {pipe_id}")
                    pattern = self.holo_compressor.compress(self.pipes[pipe_id])
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("PIPE HOLO komutunda sözdizimi hatası")

            # PIPE SMART
            elif command_upper.startswith("PIPE SMART "):
                match = re.match(r"PIPE SMART\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, transfer_time, var_name = match.groups()
                    data_size = int(data_size)
                    transfer_time = float(transfer_time)
                    strategy = self.smart_optimizer.optimize(data_size, transfer_time)
                    self.interpreter.current_scope()[var_name] = strategy
                else:
                    raise PdsXException("PIPE SMART komutunda sözdizimi hatası")

            # PIPE TEMPORAL
            elif command_upper.startswith("PIPE TEMPORAL "):
                match = re.match(r"PIPE TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    pipe_id1, pipe_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(pipe_id1, pipe_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("PIPE TEMPORAL komutunda sözdizimi hatası")

            # PIPE PREDICT
            elif command_upper.startswith("PIPE PREDICT "):
                match = re.match(r"PIPE PREDICT\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, transfer_time, var_name = match.groups()
                    data_size = int(data_size)
                    transfer_time = float(transfer_time)
                    is_anomaly = self.pipe_shield.predict(data_size, transfer_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("PIPE PREDICT komutunda sözdizimi hatası")

            else:
                raise PdsXException(f"Bilinmeyen kanal komutu: {command}")
        except Exception as e:
            log.error(f"Kanal komut hatası: {str(e)}")
            raise PdsXException(f"Kanal komut hatası: {str(e)}")

if __name__ == "__main__":
    print("pipe_monitor_gui.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")