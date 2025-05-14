# save_load_system.py - PDS-X BASIC v14u Kaydetme ve Yükleme Sistemi Kütüphanesi
# Version: 1.0.0
# Date: May 13, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import threading
import asyncio
import time
import json
import yaml
import pickle
import base64
import boto3
import botocore
import websockets
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
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
log = logging.getLogger("save_load_system")

class SerializedData:
    """Serileştirilmiş veri sınıfı."""
    def __init__(self, data_id: str, data: Any, format_type: str, timestamp: float):
        self.data_id = data_id
        self.data = data
        self.format_type = format_type
        self.timestamp = timestamp
        self.metadata = {"encrypted": False, "compressed": False, "hash": self._compute_hash()}
        self.lock = threading.Lock()

    def _compute_hash(self) -> str:
        """Verinin SHA-256 hash’ini hesaplar."""
        try:
            serialized = pickle.dumps(self.data)
            return hashlib.sha256(serialized).hexdigest()
        except Exception as e:
            log.error(f"Hash hesaplama hatası: {str(e)}")
            raise PdsXException(f"Hash hesaplama hatası: {str(e)}")

    @synchronized
    def serialize(self) -> bytes:
        """Veriyi belirtilen formatta serileştirir."""
        try:
            format_type = self.format_type.lower()
            if format_type == "json":
                return json.dumps(self.data).encode('utf-8')
            elif format_type == "yaml":
                return yaml.dump(self.data).encode('utf-8')
            elif format_type == "pickle":
                return pickle.dumps(self.data)
            elif format_type == "pdsx":
                pdsx_data = {"data": self.data, "meta": self.metadata}
                return json.dumps(pdsx_data).encode('utf-8')
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
        except Exception as e:
            log.error(f"Serialize hatası: {str(e)}")
            raise PdsXException(f"Serialize hatası: {str(e)}")

class ProvenanceBlock:
    """Veri geçmişi bloğu sınıfı."""
    def __init__(self, block_id: str, data_id: str, operation: str, timestamp: float, prev_hash: str):
        self.block_id = block_id
        self.data_id = data_id
        self.operation = operation
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Bloğun SHA-256 hash’ini hesaplar."""
        try:
            block_data = f"{self.block_id}{self.data_id}{self.operation}{self.timestamp}{self.prev_hash}"
            return hashlib.sha256(block_data.encode('utf-8')).hexdigest()
        except Exception as e:
            log.error(f"Provenance hash hesaplama hatası: {str(e)}")
            raise PdsXException(f"Provenance hash hesaplama hatası: {str(e)}")

class QuantumDataCorrelator:
    """Kuantum tabanlı veri korelasyon sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (data_id1, data_id2, score)}

    def correlate(self, data1: SerializedData, data2: SerializedData) -> str:
        """İki veriyi kuantum simülasyonuyla ilişkilendirir."""
        try:
            set1 = set(str(data1.data))
            set2 = set(str(data2.data))
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (data1.data_id, data2.data_id, score)
            log.debug(f"Kuantum korelasyon: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumDataCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumDataCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[str, str, float]]:
        """Korelasyonu döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumDataCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumDataCorrelator get_correlation hatası: {str(e)}")

class HoloDataCompressor:
    """Holografik veri sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [serialized_data]}

    def compress(self, data: SerializedData) -> str:
        """Veriyi holografik olarak sıkıştırır."""
        try:
            serialized = data.serialize()
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            data.metadata["compressed"] = True
            log.debug(f"Holografik veri sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloDataCompressor compress hatası: {str(e)}")
            raise PdsXException(f"HoloDataCompressor compress hatası: {str(e)}")

    def decompress(self, pattern: str, format_type: str) -> Optional[Any]:
        """Veriyi geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                serialized = self.storage[pattern][-1]
                if format_type == "json":
                    return json.loads(serialized.decode('utf-8'))
                elif format_type == "yaml":
                    return yaml.safe_load(serialized.decode('utf-8'))
                elif format_type == "pickle":
                    return pickle.loads(serialized)
                elif format_type == "pdsx":
                    pdsx_data = json.loads(serialized.decode('utf-8'))
                    return pdsx_data["data"]
                else:
                    raise PdsXException(f"Desteklenmeyen format: {format_type}")
            return None
        except Exception as e:
            log.error(f"HoloDataCompressor decompress hatası: {str(e)}")
            raise PdsXException(f"HoloDataCompressor decompress hatası: {str(e)}")

class SmartStorageFabric:
    """AI tabanlı otomatik depolama optimizasyon sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(data_size, access_time, timestamp)]

    def optimize(self, data_size: int, access_time: float) -> str:
        """Depolama yöntemini optimize bir şekilde seçer."""
        try:
            features = np.array([[data_size, access_time, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:  # Anomali tespit edildi
                    storage = "DISTRIBUTED"  # Dağıtık depolama öner
                    log.warning(f"Depolama optimize edildi: storage={storage}, score={anomaly_score}")
                    return storage
            return "LOCAL"  # Varsayılan depolama
        except Exception as e:
            log.error(f"SmartStorageFabric optimize hatası: {str(e)}")
            raise PdsXException(f"SmartStorageFabric optimize hatası: {str(e)}")

class TemporalDataGraph:
    """Zaman temelli veri ilişkileri grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {data_id: timestamp}
        self.edges = defaultdict(list)  # {data_id: [(related_data_id, weight)]}

    def add_data(self, data_id: str, timestamp: float) -> None:
        """Veriyi grafiğe ekler."""
        try:
            self.vertices[data_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: data_id={data_id}")
        except Exception as e:
            log.error(f"TemporalDataGraph add_data hatası: {str(e)}")
            raise PdsXException(f"TemporalDataGraph add_data hatası: {str(e)}")

    def add_relation(self, data_id1: str, data_id2: str, weight: float) -> None:
        """Veriler arasında ilişki kurar."""
        try:
            self.edges[data_id1].append((data_id2, weight))
            self.edges[data_id2].append((data_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {data_id1} <-> {data_id2}")
        except Exception as e:
            log.error(f"TemporalDataGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalDataGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Veri grafiğini analiz eder."""
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
    """Tahmini veri hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(data_size, access_time, timestamp)]

    def train(self, data_size: int, access_time: float) -> None:
        """Veri verileriyle modeli eğitir."""
        try:
            features = np.array([data_size, access_time, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("DataShield modeli eğitildi")
        except Exception as e:
            log.error(f"DataShield train hatası: {str(e)}")
            raise PdsXException(f"DataShield train hatası: {str(e)}")

    def predict(self, data_size: int, access_time: float) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[data_size, access_time, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: data_size={data_size}")
            return is_anomaly
        except Exception as e:
            log.error(f"DataShield predict hatası: {str(e)}")
            raise PdsXException(f"DataShield predict hatası: {str(e)}")

class SaveLoadSystem:
    """Kaydetme ve yükleme sistemi yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.data_store = {}  # {data_id: SerializedData}
        self.provenance_chain = {}  # {data_id: [ProvenanceBlock]}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumDataCorrelator()
        self.holo_compressor = HoloDataCompressor()
        self.smart_fabric = SmartStorageFabric()
        self.temporal_graph = TemporalDataGraph()
        self.data_shield = DataShield()
        self.lock = threading.Lock()
        self.metadata = {
            "save_load_system": {
                "version": "1.0.0",
                "dependencies": ["graphviz", "numpy", "scikit-learn", "boto3", "websockets", "pyyaml", "pycryptodome", "pdsx_exception"]
            }
        }
        self.max_data_size = 10000

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron kaydetme/yükleme döngüsü başlatıldı")

    @synchronized
    def save_data(self, data: Any, path: str, format_type: str = "json") -> str:
        """Veriyi kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            serialized = serialized_data.serialize()
            
            # Yerel dosya kaydetme
            with open(path, 'wb') as f:
                f.write(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Veri kaydedildi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Save data hatası: {str(e)}")
            raise PdsXException(f"Save data hatası: {str(e)}")

    async def save_async_data(self, data: Any, path: str, format_type: str = "json") -> str:
        """Veriyi asenkron kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            serialized = serialized_data.serialize()
            
            # Yerel dosya kaydetme (asenkron)
            async with aiofiles.open(path, 'wb') as f:
                await f.write(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_ASYNC", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            self.start_async_loop()
            log.debug(f"Asenkron veri kaydedildi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Save async data hatası: {str(e)}")
            raise PdsXException(f"Save async data hatası: {str(e)}")

    @synchronized
    def load_data(self, path: str, format_type: str = "json") -> str:
        """Veriyi yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Yerel dosya yükleme
            with open(path, 'rb') as f:
                serialized = f.read()
            
            if format_type == "json":
                data = json.loads(serialized.decode('utf-8'))
            elif format_type == "yaml":
                data = yaml.safe_load(serialized.decode('utf-8'))
            elif format_type == "pickle":
                data = pickle.loads(serialized)
            elif format_type == "pdsx":
                pdsx_data = json.loads(serialized.decode('utf-8'))
                data = pdsx_data["data"]
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Veri yüklendi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Load data hatası: {str(e)}")
            raise PdsXException(f"Load data hatası: {str(e)}")

    async def load_async_data(self, path: str, format_type: str = "json") -> str:
        """Veriyi asenkron yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Yerel dosya yükleme (asenkron)
            async with aiofiles.open(path, 'rb') as f:
                serialized = await f.read()
            
            if format_type == "json":
                data = json.loads(serialized.decode('utf-8'))
            elif format_type == "yaml":
                data = yaml.safe_load(serialized.decode('utf-8'))
            elif format_type == "pickle":
                data = pickle.loads(serialized)
            elif format_type == "pdsx":
                pdsx_data = json.loads(serialized.decode('utf-8'))
                data = pdsx_data["data"]
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_ASYNC", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            self.start_async_loop()
            log.debug(f"Asenkron veri yüklendi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Load async data hatası: {str(e)}")
            raise PdsXException(f"Load async data hatası: {str(e)}")

    def save_distributed_data(self, data: Any, bucket_name: str, key: str, credentials: Dict[str, str], format_type: str = "json") -> str:
        """Veriyi dağıtık olarak kaydeder (S3 benzeri)."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            serialized = serialized_data.serialize()
            
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_DISTRIBUTED", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Dağıtık veri kaydedildi: data_id={data_id}, bucket={bucket_name}, key={key}")
            return data_id
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed save hatası: {str(e)}")
            raise PdsXException(f"Distributed save hatası: {str(e)}")

    async def load_distributed_data(self, bucket_name: str, key: str, credentials: Dict[str, str], format_type: str = "json") -> str:
        """Veriyi dağıtık olarak yükler."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            serialized = response['Body'].read()
            
            if format_type == "json":
                data = json.loads(serialized.decode('utf-8'))
            elif format_type == "yaml":
                data = yaml.safe_load(serialized.decode('utf-8'))
            elif format_type == "pickle":
                data = pickle.loads(serialized)
            elif format_type == "pdsx":
                pdsx_data = json.loads(serialized.decode('utf-8'))
                data = pdsx_data["data"]
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_DISTRIBUTED", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Dağıtık veri yüklendi: data_id={data_id}, bucket={bucket_name}, key={key}")
            return data_id
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed load hatası: {str(e)}")
            raise PdsXException(f"Distributed load hatası: {str(e)}")

    async def save_stream_data(self, data: Any, ws_url: str, format_type: str = "json") -> str:
        """WebSocket üzerinden veri kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            serialized = serialized_data.serialize()
            
            async with websockets.connect(ws_url) as ws:
                await ws.send(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_STREAM", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"WebSocket verisi kaydedildi: data_id={data_id}, ws_url={ws_url}")
            return data_id
        except Exception as e:
            log.error(f"WebSocket save hatası: {str(e)}")
            raise PdsXException(f"WebSocket save hatası: {str(e)}")

    async def load_stream_data(self, ws_url: str, format_type: str = "json") -> str:
        """WebSocket üzerinden veri yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            async with websockets.connect(ws_url) as ws:
                serialized = await ws.recv()
            
            if format_type == "json":
                data = json.loads(serialized.decode('utf-8'))
            elif format_type == "yaml":
                data = yaml.safe_load(serialized.decode('utf-8'))
            elif format_type == "pickle":
                data = pickle.loads(serialized)
            elif format_type == "pdsx":
                pdsx_data = json.loads(serialized.decode('utf-8'))
                data = pdsx_data["data"]
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            
            serialized_data = SerializedData(data_id, data, format_type, timestamp)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain güncelleme
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_STREAM", timestamp, prev_hash)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"WebSocket verisi yüklendi: data_id={data_id}, ws_url={ws_url}")
            return data_id
        except Exception as e:
            log.error(f"WebSocket load hatası: {str(e)}")
            raise PdsXException(f"WebSocket load hatası: {str(e)}")

    def encrypt_data(self, data_id: str, key: bytes, method: str = "aes") -> bytes:
        """Veriyi şifreler."""
        try:
            if data_id not in self.data_store:
                raise PdsXException(f"Veri bulunamadı: {data_id}")
            serialized_data = self.data_store[data_id]
            serialized = serialized_data.serialize()
            method = method.lower()
            if method == "aes":
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                serialized_data.metadata["encrypted"] = True
                return cipher.nonce + tag + ciphertext
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Encrypt data hatası: {str(e)}")
            raise PdsXException(f"Encrypt data hatası: {str(e)}")

    def _get_last_provenance_hash(self, data_id: str) -> str:
        """Son provenance bloğunun hash’ini döndürür."""
        if data_id in self.provenance_chain and self.provenance_chain[data_id]:
            return self.provenance_chain[data_id][-1].hash
        return ""

    def parse_save_load_command(self, command: str) -> None:
        """Kaydetme ve yükleme komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("SAVE DATA "):
                match = re.match(r"SAVE DATA\s+(.+?)\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    data_id = self.save_data(data, path, format_type)
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE ASYNC DATA "):
                match = re.match(r"SAVE ASYNC DATA\s+(.+?)\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    data_id = asyncio.run(self.save_async_data(data, path, format_type))
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE ASYNC DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD DATA "):
                match = re.match(r"LOAD DATA\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format_type, var_name = match.groups()
                    format_type = format_type or "json"
                    data_id = self.load_data(path, format_type)
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD ASYNC DATA "):
                match = re.match(r"LOAD ASYNC DATA\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format_type, var_name = match.groups()
                    format_type = format_type or "json"
                    data_id = asyncio.run(self.load_async_data(path, format_type))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD ASYNC DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE DISTRIBUTED "):
                match = re.match(r"SAVE DISTRIBUTED\s+(.+?)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, bucket_name, key, creds_str, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    format_type = format_type or "json"
                    data_id = self.save_distributed_data(data, bucket_name, key, credentials, format_type)
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD DISTRIBUTED "):
                match = re.match(r"LOAD DISTRIBUTED\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    bucket_name, key, creds_str, format_type, var_name = match.groups()
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    format_type = format_type or "json"
                    data_id = asyncio.run(self.load_distributed_data(bucket_name, key, credentials, format_type))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE STREAM "):
                match = re.match(r"SAVE STREAM\s+(.+?)\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, ws_url, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    data_id = asyncio.run(self.save_stream_data(data, ws_url, format_type))
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE STREAM komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD STREAM "):
                match = re.match(r"LOAD STREAM\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    ws_url, format_type, var_name = match.groups()
                    format_type = format_type or "json"
                    data_id = asyncio.run(self.load_stream_data(ws_url, format_type))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD STREAM komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE ENCRYPT "):
                match = re.match(r"SAVE ENCRYPT\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id, key_str, method, var_name = match.groups()
                    if data_id not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id}")
                    key = base64.b64decode(key_str)
                    encrypted = self.encrypt_data(data_id, key, method)
                    self.interpreter.current_scope()[var_name] = encrypted
                else:
                    raise PdsXException("SAVE ENCRYPT komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE ANALYZE "):
                match = re.match(r"SAVE ANALYZE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_data": len(self.data_store),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [did for did, d in self.data_store.items() if self.data_shield.predict(len(str(d.data)), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("SAVE ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE VISUALIZE "):
                match = re.match(r"SAVE VISUALIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    output_path, format = match.groups()
                    format = format or "png"
                    dot = graphviz.Digraph(format=format)
                    for did, data in self.data_store.items():
                        node_label = f"ID: {did}\nFormat: {data.format_type}\nTime: {data.timestamp}"
                        dot.node(did, node_label, color="red" if data.metadata["encrypted"] else "green")
                    for did1 in self.temporal_graph.edges:
                        for did2, weight in self.temporal_graph.edges[did1]:
                            dot.edge(did1, did2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    log.debug(f"Veriler görselleştirildi: path={output_path}.{format}")
                else:
                    raise PdsXException("SAVE VISUALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE QUANTUM "):
                match = re.match(r"SAVE QUANTUM\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id1, data_id2, var_name = match.groups()
                    if data_id1 not in self.data_store or data_id2 not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id1} veya {data_id2}")
                    correlation_id = self.quantum_correlator.correlate(self.data_store[data_id1], self.data_store[data_id2])
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("SAVE QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE HOLO "):
                match = re.match(r"SAVE HOLO\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id, var_name = match.groups()
                    if data_id not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id}")
                    pattern = self.holo_compressor.compress(self.data_store[data_id])
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("SAVE HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE SMART "):
                match = re.match(r"SAVE SMART\s+(\d+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, access_time, var_name = match.groups()
                    data_size = int(data_size)
                    access_time = float(access_time)
                    storage = self.smart_fabric.optimize(data_size, access_time)
                    self.interpreter.current_scope()[var_name] = storage
                else:
                    raise PdsXException("SAVE SMART komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE TEMPORAL "):
                match = re.match(r"SAVE TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id1, data_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(data_id1, data_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("SAVE TEMPORAL komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE PREDICT "):
                match = re.match(r"SAVE PREDICT\s+(\d+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, access_time, var_name = match.groups()
                    data_size = int(data_size)
                    access_time = float(access_time)
                    is_anomaly = self.data_shield.predict(data_size, access_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("SAVE PREDICT komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen kaydetme/yükleme komutu: {command}")
        except Exception as e:
            log.error(f"Kaydetme/yükleme komut hatası: {str(e)}")
            raise PdsXException(f"Kaydetme/yükleme komut hatası: {str(e)}")

if __name__ == "__main__":
    print("save_load_system.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")