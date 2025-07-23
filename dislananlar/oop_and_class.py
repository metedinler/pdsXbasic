# oop_and_class.py - PDS-X BASIC v14u Nesne Yönelimli Programlama ve Sınıf Kütüphanesi
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
from collections import defaultdict
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
log = logging.getLogger("oop_and_class")

class ClassDefinition:
    """Sınıf tanımı sınıfı."""
    def __init__(self, class_id: str, name: str, methods: Dict[str, Callable] = None, properties: Dict[str, Any] = None, parent_id: Optional[str] = None):
        self.class_id = class_id
        self.name = name
        self.methods = methods or {}
        self.properties = properties or {}
        self.parent_id = parent_id
        self.metadata = {"created_at": time.time(), "instance_count": 0}
        self.lock = threading.Lock()

    def add_method(self, method_name: str, method: Callable) -> None:
        """Sınıfa metod ekler."""
        with self.lock:
            self.methods[method_name] = method
            log.debug(f"Metod eklendi: class_id={self.class_id}, method={method_name}")

    def add_property(self, prop_name: str, value: Any) -> None:
        """Sınıfa özellik ekler."""
        with self.lock:
            self.properties[prop_name] = value
            log.debug(f"Özellik eklendi: class_id={self.class_id}, property={prop_name}")

    def instantiate(self) -> 'ClassInstance':
        """Sınıf örneği oluşturur."""
        with self.lock:
            instance_id = str(uuid.uuid4())
            instance = ClassInstance(instance_id, self)
            self.metadata["instance_count"] += 1
            log.debug(f"Sınıf örneği oluşturuldu: class_id={self.class_id}, instance_id={instance_id}")
            return instance

class ClassInstance:
    """Sınıf örneği sınıfı."""
    def __init__(self, instance_id: str, class_def: ClassDefinition):
        self.instance_id = instance_id
        self.class_def = class_def
        self.state = {}  # Örnek özel durum
        self.lock = threading.Lock()

    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """Metodu çağırır."""
        with self.lock:
            method = self.class_def.methods.get(method_name)
            if not method:
                raise PdsXException(f"Metod bulunamadı: {method_name}")
            try:
                return method(*args, **kwargs)
            except Exception as e:
                log.error(f"Metod çağrı hatası: {str(e)}")
                raise PdsXException(f"Metod çağrı hatası: {str(e)}")

    def set_property(self, prop_name: str, value: Any) -> None:
        """Özelliği ayarlar."""
        with self.lock:
            self.state[prop_name] = value
            log.debug(f"Özellik ayarlandı: instance_id={self.instance_id}, property={prop_name}")

    def get_property(self, prop_name: str) -> Any:
        """Özelliği alır."""
        with self.lock:
            return self.state.get(prop_name, self.class_def.properties.get(prop_name))

class QuantumClassCorrelator:
    """Kuantum tabanlı sınıf bağıntı analizi sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (class_id1, class_id2, score)}

    def correlate(self, class1: ClassDefinition, class2: ClassDefinition) -> str:
        """İki sınıfı kuantum simülasyonuyla ilişkilendirir."""
        try:
            set1 = set(list(class1.methods.keys()) + list(class1.properties.keys()))
            set2 = set(list(class2.methods.keys()) + list(class2.properties.keys()))
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (class1.class_id, class2.class_id, score)
            log.debug(f"Kuantum bağıntı: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumClassCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumClassCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[str, str, float]]:
        """Bağıntıyı döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumClassCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumClassCorrelator get_correlation hatası: {str(e)}")

class HoloClassCompressor:
    """Holografik sınıf sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [class_data]}

    def compress(self, class_def: ClassDefinition) -> str:
        """Sınıfı holografik olarak sıkıştırır."""
        try:
            serialized = pickle.dumps({
                "name": class_def.name,
                "methods": {k: str(v) for k, v in class_def.methods.items()},
                "properties": class_def.properties,
                "parent_id": class_def.parent_id
            })
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            log.debug(f"Holografik sınıf sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloClassCompressor compress hatası: {str(e)}")
            raise PdsXException(f"HoloClassCompressor compress hatası: {str(e)}")

    def decompress(self, pattern: str) -> Optional[Dict]:
        """Sınıfı geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                serialized = self.storage[pattern][-1]
                return pickle.loads(serialized)
            return None
        except Exception as e:
            log.error(f"HoloClassCompressor decompress hatası: {str(e)}")
            raise PdsXException(f"HoloClassCompressor decompress hatası: {str(e)}")

class SmartClassFabric:
    """AI tabanlı otomatik sınıf optimizasyon sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(method_count, prop_count, timestamp)]

    def optimize(self, method_count: int, prop_count: int) -> str:
        """Sınıf yapısını optimize bir şekilde seçer."""
        try:
            features = np.array([[method_count, prop_count, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:  # Anomali tespit edildi
                    structure = "ABSTRACT"  # Daha soyut yapı öner
                    log.warning(f"Sınıf optimize edildi: structure={structure}, score={anomaly_score}")
                    return structure
            return "CONCRETE"  # Varsayılan yapı
        except Exception as e:
            log.error(f"SmartClassFabric optimize hatası: {str(e)}")
            raise PdsXException(f"SmartClassFabric optimize hatası: {str(e)}")

class TemporalClassGraph:
    """Zaman temelli sınıf ilişkileri grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {class_id: timestamp}
        self.edges = defaultdict(list)  # {class_id: [(related_class_id, weight)]}

    def add_class(self, class_id: str, timestamp: float) -> None:
        """Sınıfı grafiğe ekler."""
        try:
            self.vertices[class_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: class_id={class_id}")
        except Exception as e:
            log.error(f"TemporalClassGraph add_class hatası: {str(e)}")
            raise PdsXException(f"TemporalClassGraph add_class hatası: {str(e)}")

    def add_relation(self, class_id1: str, class_id2: str, weight: float) -> None:
        """Sınıflar arasında ilişki kurar."""
        try:
            self.edges[class_id1].append((class_id2, weight))
            self.edges[class_id2].append((class_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {class_id1} <-> {class_id2}")
        except Exception as e:
            log.error(f"TemporalClassGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalClassGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Sınıf grafiğini analiz eder."""
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
            log.error(f"TemporalClassGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalClassGraph analyze hatası: {str(e)}")

class ClassShield:
    """Tahmini sınıf hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(method_count, prop_count, timestamp)]

    def train(self, method_count: int, prop_count: int) -> None:
        """Sınıf verileriyle modeli eğitir."""
        try:
            features = np.array([method_count, prop_count, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("ClassShield modeli eğitildi")
        except Exception as e:
            log.error(f"ClassShield train hatası: {str(e)}")
            raise PdsXException(f"ClassShield train hatası: {str(e)}")

    def predict(self, method_count: int, prop_count: int) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[method_count, prop_count, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: method_count={method_count}")
            return is_anomaly
        except Exception as e:
            log.error(f"ClassShield predict hatası: {str(e)}")
            raise PdsXException(f"ClassShield predict hatası: {str(e)}")

class OOPManager:
    """Nesne yönelimli programlama yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.classes = {}  # {class_id: ClassDefinition}
        self.instances = {}  # {instance_id: ClassInstance}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumClassCorrelator()
        self.holo_compressor = HoloClassCompressor()
        self.smart_fabric = SmartClassFabric()
        self.temporal_graph = TemporalClassGraph()
        self.class_shield = ClassShield()
        self.lock = threading.Lock()
        self.metadata = {
            "oop_and_class": {
                "version": "1.0.0",
                "dependencies": ["graphviz", "numpy", "scikit-learn", "pycryptodome", "pyyaml", "pdsx_exception"]
            }
        }
        self.max_classes = 1000

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron sınıf döngüsü başlatıldı")

    def define_class(self, name: str, parent_id: Optional[str] = None) -> str:
        """Sınıf tanımlar."""
        with self.lock:
            try:
                class_id = str(uuid.uuid4())
                class_def = ClassDefinition(class_id, name, parent_id=parent_id)
                self.classes[class_id] = class_def
                self.temporal_graph.add_class(class_id, time.time())
                self.class_shield.train(len(class_def.methods), len(class_def.properties))
                log.debug(f"Sınıf tanımlandı: class_id={class_id}, name={name}")
                return class_id
            except Exception as e:
                log.error(f"Define class hatası: {str(e)}")
                raise PdsXException(f"Define class hatası: {str(e)}")

    async def define_async_class(self, name: str, parent_id: Optional[str] = None) -> str:
        """Asenkron sınıf tanımlar."""
        try:
            class_id = str(uuid.uuid4())
            class_def = ClassDefinition(class_id, name, parent_id=parent_id)
            self.classes[class_id] = class_def
            self.temporal_graph.add_class(class_id, time.time())
            self.class_shield.train(len(class_def.methods), len(class_def.properties))
            self.start_async_loop()
            log.debug(f"Asenkron sınıf tanımlandı: class_id={class_id}, name={name}")
            return class_id
        except Exception as e:
            log.error(f"Define async class hatası: {str(e)}")
            raise PdsXException(f"Define async class hatası: {str(e)}")

    def serialize_class(self, class_id: str, format_type: str = "json") -> bytes:
        """Sınıfı serileştirir."""
        try:
            class_def = self.classes.get(class_id)
            if not class_def:
                raise PdsXException(f"Sınıf bulunamadı: {class_id}")
            
            data = {
                "name": class_def.name,
                "methods": {k: str(v) for k, v in class_def.methods.items()},
                "properties": class_def.properties,
                "parent_id": class_def.parent_id,
                "metadata": class_def.metadata
            }
            format_type = format_type.lower()
            if format_type == "json":
                return json.dumps(data).encode('utf-8')
            elif format_type == "pickle":
                return pickle.dumps(data)
            elif format_type == "yaml":
                return yaml.dump(data).encode('utf-8')
            elif format_type == "pdsx":
                pdsx_data = {"data": data, "meta": {"type": "class", "timestamp": time.time()}}
                return json.dumps(pdsx_data).encode('utf-8')
            else:
                raise PdsXException(f"Desteklenmeyen serileştirme formatı: {format_type}")
        except Exception as e:
            log.error(f"Serialize class hatası: {str(e)}")
            raise PdsXException(f"Serialize class hatası: {str(e)}")

    def encrypt_class(self, class_id: str, key: bytes, method: str = "aes") -> bytes:
        """Sınıfı şifreler."""
        try:
            serialized = self.serialize_class(class_id)
            method = method.lower()
            if method == "aes":
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                return cipher.nonce + tag + ciphertext
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Encrypt class hatası: {str(e)}")
            raise PdsXException(f"Encrypt class hatası: {str(e)}")

    def parse_oop_command(self, command: str) -> None:
        """OOP komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("CLASS DEFINE "):
                match = re.match(r"CLASS DEFINE\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    name, parent_id, var_name = match.groups()
                    class_id = self.define_class(name, parent_id)
                    self.interpreter.current_scope()[var_name] = class_id
                else:
                    raise PdsXException("CLASS DEFINE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS ASYNC DEFINE "):
                match = re.match(r"CLASS ASYNC DEFINE\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    name, parent_id, var_name = match.groups()
                    class_id = asyncio.run(self.define_async_class(name, parent_id))
                    self.interpreter.current_scope()[var_name] = class_id
                else:
                    raise PdsXException("CLASS ASYNC DEFINE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS METHOD "):
                match = re.match(r"CLASS METHOD\s+(\w+)\s+\"([^\"]+)\"\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    class_id, method_name, expr = match.groups()
                    if class_id not in self.classes:
                        raise PdsXException(f"Sınıf bulunamadı: {class_id}")
                    def method(*args, **kwargs):
                        local_scope = self.interpreter.current_scope().copy()
                        for i, arg in enumerate(args):
                            local_scope[f"arg{i}"] = arg
                        local_scope.update(kwargs)
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(expr)
                    self.classes[class_id].add_method(method_name, method)
                else:
                    raise PdsXException("CLASS METHOD komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS PROPERTY "):
                match = re.match(r"CLASS PROPERTY\s+(\w+)\s+\"([^\"]+)\"\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id, prop_name, value_str, var_name = match.groups()
                    if class_id not in self.classes:
                        raise PdsXException(f"Sınıf bulunamadı: {class_id}")
                    value = self.interpreter.evaluate_expression(value_str)
                    self.classes[class_id].add_property(prop_name, value)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("CLASS PROPERTY komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS INSTANTIATE "):
                match = re.match(r"CLASS INSTANTIATE\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id, var_name = match.groups()
                    if class_id not in self.classes:
                        raise PdsXException(f"Sınıf bulunamadı: {class_id}")
                    instance = self.classes[class_id].instantiate()
                    self.instances[instance.instance_id] = instance
                    self.interpreter.current_scope()[var_name] = instance.instance_id
                else:
                    raise PdsXException("CLASS INSTANTIATE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS CALL "):
                match = re.match(r"CLASS CALL\s+(\w+)\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    instance_id, method_name, args_str, var_name = match.groups()
                    if instance_id not in self.instances:
                        raise PdsXException(f"Örnek bulunamadı: {instance_id}")
                    args = eval(args_str, self.interpreter.current_scope())
                    args = args if isinstance(args, list) else [args]
                    result = self.instances[instance_id].call_method(method_name, *args)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("CLASS CALL komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS SET PROPERTY "):
                match = re.match(r"CLASS SET PROPERTY\s+(\w+)\s+\"([^\"]+)\"\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    instance_id, prop_name, value_str, var_name = match.groups()
                    if instance_id not in self.instances:
                        raise PdsXException(f"Örnek bulunamadı: {instance_id}")
                    value = self.interpreter.evaluate_expression(value_str)
                    self.instances[instance_id].set_property(prop_name, value)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("CLASS SET PROPERTY komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS GET PROPERTY "):
                match = re.match(r"CLASS GET PROPERTY\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    instance_id, prop_name, var_name = match.groups()
                    if instance_id not in self.instances:
                        raise PdsXException(f"Örnek bulunamadı: {instance_id}")
                    value = self.instances[instance_id].get_property(prop_name)
                    self.interpreter.current_scope()[var_name] = value
                else:
                    raise PdsXException("CLASS GET PROPERTY komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS SERIALIZE "):
                match = re.match(r"CLASS SERIALIZE\s+(\w+)\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id, format_type, var_name = match.groups()
                    format_type = format_type or "json"
                    serialized = self.serialize_class(class_id, format_type)
                    self.interpreter.current_scope()[var_name] = serialized
                else:
                    raise PdsXException("CLASS SERIALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS ENCRYPT "):
                match = re.match(r"CLASS ENCRYPT\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id, key_str, method, var_name = match.groups()
                    key = base64.b64decode(key_str)
                    encrypted = self.encrypt_class(class_id, key, method)
                    self.interpreter.current_scope()[var_name] = encrypted
                else:
                    raise PdsXException("CLASS ENCRYPT komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS ANALYZE "):
                match = re.match(r"CLASS ANALYZE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_classes": len(self.classes),
                        "total_instances": len(self.instances),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [cid for cid, c in self.classes.items() if self.class_shield.predict(len(c.methods), len(c.properties))]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("CLASS ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS VISUALIZE "):
                match = re.match(r"CLASS VISUALIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    output_path, format = match.groups()
                    format = format or "png"
                    dot = graphviz.Digraph(format=format)
                    for cid, class_def in self.classes.items():
                        node_label = f"ID: {cid}\nName: {class_def.name}\nMethods: {len(class_def.methods)}\nProps: {len(class_def.properties)}"
                        dot.node(cid, node_label)
                        if class_def.parent_id:
                            dot.edge(class_def.parent_id, cid, label="inherits")
                    for cid1 in self.temporal_graph.edges:
                        for cid2, weight in self.temporal_graph.edges[cid1]:
                            dot.edge(cid1, cid2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    log.debug(f"Sınıflar görselleştirildi: path={output_path}.{format}")
                else:
                    raise PdsXException("CLASS VISUALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS QUANTUM "):
                match = re.match(r"CLASS QUANTUM\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id1, class_id2, var_name = match.groups()
                    if class_id1 not in self.classes or class_id2 not in self.classes:
                        raise PdsXException(f"Sınıf bulunamadı: {class_id1} veya {class_id2}")
                    correlation_id = self.quantum_correlator.correlate(self.classes[class_id1], self.classes[class_id2])
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("CLASS QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS HOLO "):
                match = re.match(r"CLASS HOLO\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id, var_name = match.groups()
                    if class_id not in self.classes:
                        raise PdsXException(f"Sınıf bulunamadı: {class_id}")
                    pattern = self.holo_compressor.compress(self.classes[class_id])
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("CLASS HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS SMART "):
                match = re.match(r"CLASS SMART\s+(\d+)\s+(\d+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    method_count, prop_count, var_name = match.groups()
                    method_count = int(method_count)
                    prop_count = int(prop_count)
                    structure = self.smart_fabric.optimize(method_count, prop_count)
                    self.interpreter.current_scope()[var_name] = structure
                else:
                    raise PdsXException("CLASS SMART komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS TEMPORAL "):
                match = re.match(r"CLASS TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    class_id1, class_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(class_id1, class_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("CLASS TEMPORAL komutunda sözdizimi hatası")
            elif command_upper.startswith("CLASS PREDICT "):
                match = re.match(r"CLASS PREDICT\s+(\d+)\s+(\d+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    method_count, prop_count, var_name = match.groups()
                    method_count = int(method_count)
                    prop_count = int(prop_count)
                    is_anomaly = self.class_shield.predict(method_count, prop_count)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("CLASS PREDICT komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen OOP komutu: {command}")
        except Exception as e:
            log.error(f"OOP komut hatası: {str(e)}")
            raise PdsXException(f"OOP komut hatası: {str(e)}")

if __name__ == "__main__":
    print("oop_and_class.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")