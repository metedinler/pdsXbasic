# data_structures.py - PDS-X BASIC v14u Veri Yapıları Kütüphanesi
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
import yaml
import base64
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import heapq
import uuid
import hashlib
import graphviz
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sklearn.ensemble import IsolationForest  # AI tabanlı anomali algılama
from pdsx_exception import PdsXException  # Hata yönetimi için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("data_structures")

class ImmutableList:
    """Değiştirilemez liste sınıfı."""
    def __init__(self, items: List[Any]):
        self.items = tuple(items)

    def append(self, item: Any) -> 'ImmutableList':
        return ImmutableList(list(self.items) + [item])

    def __getitem__(self, index: int) -> Any:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return f"ImmutableList({list(self.items)})"

class ImmutableSet:
    """Değiştirilemez küme sınıfı."""
    def __init__(self, items: List[Any]):
        self.items = frozenset(items)

    def add(self, item: Any) -> 'ImmutableSet':
        return ImmutableSet(list(self.items) + [item])

    def __contains__(self, item: Any) -> bool:
        return item in self.items

    def __str__(self) -> str:
        return f"ImmutableSet({set(self.items)})"

class ImmutableDict:
    """Değiştirilemez sözlük sınıfı."""
    def __init__(self, items: Dict[Any, Any]):
        self.items = frozenset(items.items())

    def set(self, key: Any, value: Any) -> 'ImmutableDict':
        new_items = dict(self.items)
        new_items[key] = value
        return ImmutableDict(new_items)

    def __getitem__(self, key: Any) -> Any:
        return dict(self.items)[key]

    def __str__(self) -> str:
        return f"ImmutableDict({dict(self.items)})"

class PriorityQueue:
    """Öncelik kuyruğu sınıfı."""
    def __init__(self):
        self.heap = []
        self.lock = threading.Lock()

    def enqueue(self, item: Any, priority: float) -> None:
        with self.lock:
            heapq.heappush(self.heap, (priority, item))
            log.debug(f"Öncelik kuyruğuna eklendi: item={item}, priority={priority}")

    def dequeue(self) -> Any:
        with self.lock:
            if not self.heap:
                raise PdsXException("Öncelik kuyruğu boş")
            _, item = heapq.heappop(self.heap)
            log.debug(f"Öncelik kuyruğundan çıkarıldı: item={item}")
            return item

    def __len__(self) -> int:
        return len(self.heap)

class QuantumLattice:
    """Kuantum tabanlı veri örgüsü sınıfı."""
    def __init__(self):
        self.lattice = {}  # {node_id: (value, state)}

    def add_node(self, value: Any) -> str:
        """Veriyi kuantum simülasyonuyla ekler."""
        try:
            node_id = str(uuid.uuid4())
            state = np.random.choice([0, 1], p=[0.5, 0.5])  # Süperpozisyon simülasyonu
            self.lattice[node_id] = (value, state)
            log.debug(f"Kuantum örgüsüne düğüm eklendi: node_id={node_id}, state={state}")
            return node_id
        except Exception as e:
            log.error(f"QuantumLattice add_node hatası: {str(e)}")
            raise PdsXException(f"QuantumLattice add_node hatası: {str(e)}")

    def get_node(self, node_id: str) -> Optional[Any]:
        """Veriyi ölçüm simülasyonuyla alır."""
        try:
            if node_id in self.lattice:
                value, state = self.lattice[node_id]
                if state == 1:  # Aktif durum
                    log.debug(f"Kuantum örgüsünden veri alındı: node_id={node_id}")
                    return value
                return None
            raise PdsXException(f"Düğüm bulunamadı: {node_id}")
        except Exception as e:
            log.error(f"QuantumLattice get_node hatası: {str(e)}")
            raise PdsXException(f"QuantumLattice get_node hatası: {str(e)}")

class HoloCompressor:
    """Holografik veri sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [data]}

    def compress(self, data: Any) -> str:
        """Veriyi holografik olarak sıkıştırır."""
        try:
            serialized = pickle.dumps(data)
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            log.debug(f"Holografik veri sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloCompressor compress hatası: {str(e)}")
            raise PdsXException(f"HoloCompressor compress hatası: {str(e)}")

    def decompress(self, pattern: str) -> Optional[Any]:
        """Veriyi geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                serialized = self.storage[pattern][-1]
                return pickle.loads(serialized)
            return None
        except Exception as e:
            log.error(f"HoloCompressor decompress hatası: {str(e)}")
            raise PdsXException(f"HoloCompressor decompress hatası: {str(e)}")

class SmartFabric:
    """AI tabanlı otomatik veri yapısı optimizasyon sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(size, access_time, timestamp)]

    def optimize(self, size: int, access_time: float) -> str:
        """Veri yapısını optimize bir şekilde seçer."""
        try:
            features = np.array([[size, access_time, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:  # Anomali tespit edildi
                    structure = "GRAPH"  # Daha karmaşık yapı öner
                    log.warning(f"Veri yapısı optimize edildi: structure={structure}, score={anomaly_score}")
                    return structure
            return "LIST"  # Varsayılan yapı
        except Exception as e:
            log.error(f"SmartFabric optimize hatası: {str(e)}")
            raise PdsXException(f"SmartFabric optimize hatası: {str(e)}")

class TemporalDataGraph:
    """Zaman temelli veri yapısı grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {struct_id: timestamp}
        self.edges = defaultdict(list)  # {struct_id: [(related_struct_id, weight)]}

    def add_structure(self, struct_id: str, timestamp: float) -> None:
        """Veri yapısını grafiğe ekler."""
        try:
            self.vertices[struct_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: struct_id={struct_id}")
        except Exception as e:
            log.error(f"TemporalDataGraph add_structure hatası: {str(e)}")
            raise PdsXException(f"TemporalDataGraph add_structure hatası: {str(e)}")

    def add_relation(self, struct_id1: str, struct_id2: str, weight: float) -> None:
        """Veri yapıları arasında ilişki kurar."""
        try:
            self.edges[struct_id1].append((struct_id2, weight))
            self.edges[struct_id2].append((struct_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {struct_id1} <-> {struct_id2}")
        except Exception as e:
            log.error(f"TemporalDataGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalDataGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Veri yapısı grafiğini analiz eder."""
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
            log.error(f"TemporalDataGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalDataGraph analyze hatası: {str(e)}")

class DataShield:
    """Tahmini veri yapısı hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(size, access_time, timestamp)]

    def train(self, size: int, access_time: float) -> None:
        """Veri yapısı verileriyle modeli eğitir."""
        try:
            features = np.array([size, access_time, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("DataShield modeli eğitildi")
        except Exception as e:
            log.error(f"DataShield train hatası: {str(e)}")
            raise PdsXException(f"DataShield train hatası: {str(e)}")

    def predict(self, size: int, access_time: float) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[size, access_time, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: size={size}")
            return is_anomaly
        except Exception as e:
            log.error(f"DataShield predict hatası: {str(e)}")
            raise PdsXException(f"DataShield predict hatası: {str(e)}")

class DataStructuresManager:
    """Veri yapıları yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.structures = {}  # {struct_id: (type, instance)}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_lattice = QuantumLattice()
        self.holo_compressor = HoloCompressor()
        self.smart_fabric = SmartFabric()
        self.temporal_graph = TemporalDataGraph()
        self.data_shield = DataShield()
        self.lock = threading.Lock()
        self.metadata = {
            "data_structures": {
                "version": "1.0.0",
                "dependencies": ["graphviz", "numpy", "scikit-learn", "pycryptodome", "pyyaml", "pdsx_exception"]
            }
        }
        self.max_structures = 10000

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron veri yapısı döngüsü başlatıldı")

    def create_structure(self, struct_type: str, items: Optional[List[Any]] = None) -> str:
        """Veri yapısı oluşturur."""
        with self.lock:
            try:
                struct_id = str(uuid.uuid4())
                struct_type = struct_type.lower()
                timestamp = time.time()
                
                if struct_type == "list":
                    instance = ImmutableList(items or [])
                elif struct_type == "set":
                    instance = ImmutableSet(items or [])
                elif struct_type == "dict":
                    instance = ImmutableDict(items or {})
                elif struct_type == "queue":
                    instance = deque(items or [])
                elif struct_type == "stack":
                    instance = list(items or [])
                elif struct_type == "priority_queue":
                    instance = PriorityQueue()
                else:
                    raise PdsXException(f"Desteklenmeyen veri yapısı: {struct_type}")
                
                self.structures[struct_id] = (struct_type, instance)
                self.temporal_graph.add_structure(struct_id, timestamp)
                self.data_shield.train(len(items or []), 0.1)
                log.debug(f"Veri yapısı oluşturuldu: struct_id={struct_id}, type={struct_type}")
                return struct_id
            except Exception as e:
                log.error(f"Create structure hatası: {str(e)}")
                raise PdsXException(f"Create structure hatası: {str(e)}")

    def serialize_structure(self, struct_id: str, format_type: str = "json") -> bytes:
        """Veri yapısını serileştirir."""
        try:
            struct_type, instance = self.structures.get(struct_id, (None, None))
            if not instance:
                raise PdsXException(f"Veri yapısı bulunamadı: {struct_id}")
            
            data = instance.items if struct_type in ("list", "set", "dict") else list(instance)
            format_type = format_type.lower()
            if format_type == "json":
                return json.dumps(data).encode('utf-8')
            elif format_type == "pickle":
                return pickle.dumps(data)
            elif format_type == "yaml":
                return yaml.dump(data).encode('utf-8')
            elif format_type == "pdsx":
                pdsx_data = {"data": data, "meta": {"type": struct_type, "timestamp": time.time()}}
                return json.dumps(pdsx_data).encode('utf-8')
            else:
                raise PdsXException(f"Desteklenmeyen serileştirme formatı: {format_type}")
        except Exception as e:
            log.error(f"Serialize structure hatası: {str(e)}")
            raise PdsXException(f"Serialize structure hatası: {str(e)}")

    def encrypt_structure(self, struct_id: str, key: bytes, method: str = "aes") -> bytes:
        """Veri yapısını şifreler."""
        try:
            serialized = self.serialize_structure(struct_id)
            method = method.lower()
            if method == "aes":
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                return cipher.nonce + tag + ciphertext
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Encrypt structure hatası: {str(e)}")
            raise PdsXException(f"Encrypt structure hatası: {str(e)}")

    def parse_ds_command(self, command: str) -> None:
        """Veri yapısı komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("DS LIST "):
                match = re.match(r"DS LIST\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    items_str, var_name = match.groups()
                    items = eval(items_str, self.interpreter.current_scope())
                    struct_id = self.create_structure("list", items)
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS LIST komutunda sözdizimi hatası")
            elif command_upper.startswith("DS SET "):
                match = re.match(r"DS SET\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    items_str, var_name = match.groups()
                    items = eval(items_str, self.interpreter.current_scope())
                    struct_id = self.create_structure("set", items)
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS SET komutunda sözdizimi hatası")
            elif command_upper.startswith("DS DICT "):
                match = re.match(r"DS DICT\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    items_str, var_name = match.groups()
                    items = eval(items_str, self.interpreter.current_scope())
                    struct_id = self.create_structure("dict", items)
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS DICT komutunda sözdizimi hatası")
            elif command_upper.startswith("DS QUEUE "):
                match = re.match(r"DS QUEUE\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    items_str, var_name = match.groups()
                    items = eval(items_str, self.interpreter.current_scope())
                    struct_id = self.create_structure("queue", items)
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS QUEUE komutunda sözdizimi hatası")
            elif command_upper.startswith("DS STACK "):
                match = re.match(r"DS STACK\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    items_str, var_name = match.groups()
                    items = eval(items_str, self.interpreter.current_scope())
                    struct_id = self.create_structure("stack", items)
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS STACK komutunda sözdizimi hatası")
            elif command_upper.startswith("DS PRIORITY_QUEUE "):
                match = re.match(r"DS PRIORITY_QUEUE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    struct_id = self.create_structure("priority_queue")
                    self.interpreter.current_scope()[var_name] = struct_id
                else:
                    raise PdsXException("DS PRIORITY_QUEUE komutunda sözdizimi hatası")
            elif command_upper.startswith("DS PUSH "):
                match = re.match(r"DS PUSH\s+(\w+)\s+(.+?)\s*(\d*\.?\d*)?", command, re.IGNORECASE)
                if match:
                    struct_id, item_str, priority = match.groups()
                    item = self.interpreter.evaluate_expression(item_str)
                    if struct_id not in self.structures:
                        raise PdsXException(f"Veri yapısı bulunamadı: {struct_id}")
                    struct_type, instance = self.structures[struct_id]
                    if struct_type == "stack":
                        instance.append(item)
                    elif struct_type == "queue":
                        instance.append(item)
                    elif struct_type == "priority_queue":
                        priority = float(priority) if priority else 0.0
                        instance.enqueue(item, priority)
                    else:
                        raise PdsXException(f"Desteklenmeyen yapı için push: {struct_type}")
                else:
                    raise PdsXException("DS PUSH komutunda sözdizimi hatası")
            elif command_upper.startswith("DS POP "):
                match = re.match(r"DS POP\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    struct_id, var_name = match.groups()
                    if struct_id not in self.structures:
                        raise PdsXException(f"Veri yapısı bulunamadı: {struct_id}")
                    struct_type, instance = self.structures[struct_id]
                    if struct_type == "stack":
                        if not instance:
                            raise PdsXException("Yığın boş")
                        item = instance.pop()
                    elif struct_type == "queue":
                        if not instance:
                            raise PdsXException("Kuyruk boş")
                        item = instance.popleft()
                    elif struct_type == "priority_queue":
                        item = instance.dequeue()
                    else:
                        raise PdsXException(f"Desteklenmeyen yapı için pop: {struct_type}")
                    self.interpreter.current_scope()[var_name] = item
                else:
                    raise PdsXException("DS POP komutunda sözdizimi hatası")
            elif command_upper.startswith("DS SERIALIZE "):
                match = re.match(r"DS SERIALIZE\s+(\w+)\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    struct_id, format_type, var_name = match.groups()
                    format_type = format_type or "json"
                    serialized = self.serialize_structure(struct_id, format_type)
                    self.interpreter.current_scope()[var_name] = serialized
                else:
                    raise PdsXException("DS SERIALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("DS ENCRYPT "):
                match = re.match(r"DS ENCRYPT\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    struct_id, key_str, method, var_name = match.groups()
                    key = base64.b64decode(key_str)
                    encrypted = self.encrypt_structure(struct_id, key, method)
                    self.interpreter.current_scope()[var_name] = encrypted
                else:
                    raise PdsXException("DS ENCRYPT komutunda sözdizimi hatası")
            elif command_upper.startswith("DS ANALYZE "):
                match = re.match(r"DS ANALYZE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_structures": len(self.structures),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [sid for sid, (st, si) in self.structures.items() if self.data_shield.predict(len(str(si)), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DS ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("DS VISUALIZE "):
                match = re.match(r"DS VISUALIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    output_path, format = match.groups()
                    format = format or "png"
                    dot = graphviz.Digraph(format=format)
                    for sid, (st, si) in self.structures.items():
                        node_label = f"ID: {sid}\nType: {st}\nSize: {len(str(si))}"
                        dot.node(sid, node_label)
                    for sid1 in self.temporal_graph.edges:
                        for sid2, weight in self.temporal_graph.edges[sid1]:
                            dot.edge(sid1, sid2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    log.debug(f"Veri yapıları görselleştirildi: path={output_path}.{format}")
                else:
                    raise PdsXException("DS VISUALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("DS QUANTUM "):
                match = re.match(r"DS QUANTUM\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    value_str, var_name = match.groups()
                    value = self.interpreter.evaluate_expression(value_str)
                    node_id = self.quantum_lattice.add_node(value)
                    self.interpreter.current_scope()[var_name] = node_id
                else:
                    raise PdsXException("DS QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("DS HOLO "):
                match = re.match(r"DS HOLO\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    struct_id, var_name = match.groups()
                    if struct_id not in self.structures:
                        raise PdsXException(f"Veri yapısı bulunamadı: {struct_id}")
                    _, instance = self.structures[struct_id]
                    pattern = self.holo_compressor.compress(instance)
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("DS HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("DS SMART "):
                match = re.match(r"DS SMART\s+(\d+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    size, access_time, var_name = match.groups()
                    size = int(size)
                    access_time = float(access_time)
                    structure = self.smart_fabric.optimize(size, access_time)
                    self.interpreter.current_scope()[var_name] = structure
                else:
                    raise PdsXException("DS SMART komutunda sözdizimi hatası")
            elif command_upper.startswith("DS TEMPORAL "):
                match = re.match(r"DS TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    struct_id1, struct_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(struct_id1, struct_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("DS TEMPORAL komutunda sözdizimi hatası")
            elif command_upper.startswith("DS PREDICT "):
                match = re.match(r"DS PREDICT\s+(\d+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    size, access_time, var_name = match.groups()
                    size = int(size)
                    access_time = float(access_time)
                    is_anomaly = self.data_shield.predict(size, access_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("DS PREDICT komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen veri yapısı komutu: {command}")
        except Exception as e:
            log.error(f"Veri yapısı komut hatası: {str(e)}")
            raise PdsXException(f"Veri yapısı komut hatası: {str(e)}")

if __name__ == "__main__":
    print("data_structures.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")