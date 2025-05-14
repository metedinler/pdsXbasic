# save_load_system.py - PDS-X BASIC v14u Kaydetme ve Yükleme Sistemi Kütüphanesi
# Version: 1.1.0
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
import gzip
import zlib
import boto3
import botocore
import websockets
import aiofiles
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import uuid
import hashlib
import graphviz
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sklearn.ensemble import IsolationForest
from pdsx_exception import PdsXException

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("save_load_system")

# Format Kayıt Sistemi
format_registry = {
    "basx": {
        "serialize": lambda data: str(data).encode('utf-8'),
        "deserialize": lambda serialized: serialized.decode('utf-8')
    },
    "libx": {
        "serialize": lambda data: json.dumps(data).encode('utf-8'),
        "deserialize": lambda serialized: json.loads(serialized.decode('utf-8'))
    },
    "hz": {
        "serialize": lambda data: str(data).encode('utf-8'),
        "deserialize": lambda serialized: serialized.decode('utf-8')
    },
    "hx": {
        "serialize": lambda data: str(data).encode('utf-8'),
        "deserialize": lambda serialized: serialized.decode('utf-8')
    },
    "mx": {
        "serialize": lambda data: str(data).encode('utf-8'),
        "deserialize": lambda serialized: serialized.decode('utf-8')
    },
    "lx": {
        "serialize": lambda data: str(data).encode('utf-8'),
        "deserialize": lambda serialized: serialized.decode('utf-8')
    },
    "bcx": {
        "serialize": lambda data: pickle.dumps(data),
        "deserialize": lambda serialized: pickle.loads(serialized)
    },
    "bcd": {
        "serialize": lambda data: pickle.dumps(data),
        "deserialize": lambda serialized: pickle.loads(serialized)
    },
    "json": {
        "serialize": lambda data: json.dumps(data).encode('utf-8'),
        "deserialize": lambda serialized: json.loads(serialized.decode('utf-8'))
    },
    "yaml": {
        "serialize": lambda data: yaml.dump(data).encode('utf-8'),
        "deserialize": lambda serialized: yaml.safe_load(serialized.decode('utf-8'))
    },
    "pickle": {
        "serialize": lambda data: pickle.dumps(data),
        "deserialize": lambda serialized: pickle.loads(serialized)
    },
    "pdsx": {
        "serialize": lambda data: json.dumps({"data": data, "meta": {"type": "pdsx"}}).encode('utf-8'),
        "deserialize": lambda serialized: json.loads(serialized.decode('utf-8'))["data"]
    }
}

# Encoding Desteği
supported_encodings = [
    "utf-8", "cp1254", "iso-8859-9", "ascii", "utf-16", "utf-32",
    "cp1252", "iso-8859-1", "windows-1250", "latin-9",
    "cp932", "gb2312", "gbk", "euc-kr", "cp1251", "iso-8859-5",
    "cp1256", "iso-8859-6", "cp874", "iso-8859-7", "cp1257", "iso-8859-8"
]

# Sıkıştırma Yöntemleri
compression_methods = {
    "gzip": lambda data: gzip.compress(data),
    "zlib": lambda data: zlib.compress(data),
    "none": lambda data: data
}

decompression_methods = {
    "gzip": lambda data: gzip.decompress(data),
    "zlib": lambda data: zlib.decompress(data),
    "none": lambda data: data
}

# Dekoratör
def synchronized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with args[0].lock:
            return fn(*args, **kwargs)
    return wrapped

class SerializedData:
    """Serileştirilmiş veri sınıfı."""
    def __init__(self, data_id: str, data: Any, format_type: str, encoding: str, timestamp: float):
        self.data_id = data_id
        self.data = data
        self.format_type = format_type.lower()
        self.encoding = encoding.lower()
        self.timestamp = timestamp
        self.metadata = {
            "encrypted": False,
            "compressed": "none",
            "encoding": encoding,
            "hash": self._compute_hash(),
            "format": format_type
        }
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
            if self.format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {self.format_type}")
            if self.encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {self.encoding}")
            return format_registry[self.format_type]["serialize"](self.data)
        except Exception as e:
            log.error(f"Serialize hatası: {str(e)}")
            raise PdsXException(f"Serialize hatası: {str(e)}")

class ProvenanceBlock:
    """Veri geçmişi bloğu sınıfı."""
    def __init__(self, block_id: str, data_id: str, operation: str, timestamp: float, prev_hash: str, metadata: Dict):
        self.block_id = block_id
        self.data_id = data_id
        self.operation = operation
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.metadata = metadata
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Bloğun SHA-256 hash’ini hesaplar."""
        try:
            block_data = f"{self.block_id}{self.data_id}{self.operation}{self.timestamp}{self.prev_hash}{json.dumps(self.metadata)}"
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
            data.metadata["compressed"] = "holo"
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
                if format_type not in format_registry:
                    raise PdsXException(f"Desteklenmeyen format: {format_type}")
                return format_registry[format_type]["deserialize"](serialized)
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
                if anomaly_score < -0.5:
                    storage = "DISTRIBUTED"
                    log.warning(f"Depolama optimize edildi: storage={storage}, score={anomaly_score}")
                    return storage
            return "LOCAL"
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
                "version": "1.1.0",
                "dependencies": [
                    "graphviz", "numpy", "scikit-learn", "boto3", "websockets",
                    "pyyaml", "pycryptodome", "aiofiles", "pdsx_exception"
                ]
            }
        }
        self.max_data_size = 10000
        self.data_list = []  # DATA komutları için
        self.data_pointer = 0  # READ için

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
    def save_data(self, data: Any, path: str, format_type: str = "json", encoding: str = "utf-8",
                  compress: Optional[str] = None, encrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Veriyi kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized = serialized_data.serialize()
            
            # Sıkıştırma
            serialized = compression_methods[compress](serialized)
            serialized_data.metadata["compressed"] = compress
            
            # Şifreleme
            if encrypt == "aes" and key:
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                serialized = cipher.nonce + tag + ciphertext
                serialized_data.metadata["encrypted"] = True
            
            # Metadata ile kaydetme
            metadata = serialized_data.metadata
            with open(path + ".meta", 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
            with open(path, 'wb') as f:
                f.write(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Veri kaydedildi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Save data hatası: {str(e)}")
            raise PdsXException(f"Save data hatası: {str(e)}")

    async def save_async_data(self, data: Any, path: str, format_type: str = "json", encoding: str = "utf-8",
                             compress: Optional[str] = None, encrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Veriyi asenkron kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized = serialized_data.serialize()
            
            # Sıkıştırma
            serialized = compression_methods[compress](serialized)
            serialized_data.metadata["compressed"] = compress
            
            # Şifreleme
            if encrypt == "aes" and key:
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                serialized = cipher.nonce + tag + ciphertext
                serialized_data.metadata["encrypted"] = True
            
            # Metadata ile kaydetme
            metadata = serialized_data.metadata
            async with aiofiles.open(path + ".meta", 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata))
            async with aiofiles.open(path, 'wb') as f:
                await f.write(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_ASYNC", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            self.start_async_loop()
            log.debug(f"Asenkron veri kaydedildi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Save async data hatası: {str(e)}")
            raise PdsXException(f"Save async data hatası: {str(e)}")

    @synchronized
    def load_data(self, path: str, format_type: Optional[str] = None, encoding: Optional[str] = None,
                  decompress: Optional[str] = None, decrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Veriyi yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Metadata okuma
            metadata = {}
            meta_path = path + ".meta"
            if Path(meta_path).exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Otomatik format/encoding tespiti
            format_type = format_type or metadata.get("format", "json")
            encoding = encoding or metadata.get("encoding", "utf-8")
            decompress = decompress or metadata.get("compressed", "none")
            decrypt = decrypt if decrypt is not None else ("aes" if metadata.get("encrypted", False) else None)
            
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            # Veri yükleme
            with open(path, 'rb') as f:
                serialized = f.read()
            
            # Şifre çözme
            if decrypt == "aes" and key:
                nonce, tag, ciphertext = serialized[:16], serialized[16:32], serialized[32:]
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                serialized = cipher.decrypt_and_verify(ciphertext, tag)
            
            # Sıkıştırmayı çözme
            serialized = decompression_methods[decompress](serialized)
            
            # Deserializasyon
            data = format_registry[format_type]["deserialize"](serialized)
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized_data.metadata.update(metadata)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Veri yüklendi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Load data hatası: {str(e)}")
            raise PdsXException(f"Load data hatası: {str(e)}")

    async def load_async_data(self, path: str, format_type: Optional[str] = None, encoding: Optional[str] = None,
                             decompress: Optional[str] = None, decrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Veriyi asenkron yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Metadata okuma
            metadata = {}
            meta_path = path + ".meta"
            if Path(meta_path).exists():
                async with aiofiles.open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())
            
            # Otomatik format/encoding tespiti
            format_type = format_type or metadata.get("format", "json")
            encoding = encoding or metadata.get("encoding", "utf-8")
            decompress = decompress or metadata.get("compressed", "none")
            decrypt = decrypt if decrypt is not None else ("aes" if metadata.get("encrypted", False) else None)
            
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            # Veri yükleme
            async with aiofiles.open(path, 'rb') as f:
                serialized = await f.read()
            
            # Şifre çözme
            if decrypt == "aes" and key:
                nonce, tag, ciphertext = serialized[:16], serialized[16:32], serialized[32:]
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                serialized = cipher.decrypt_and_verify(ciphertext, tag)
            
            # Sıkıştırmayı çözme
            serialized = decompression_methods[decompress](serialized)
            
            # Deserializasyon
            data = format_registry[format_type]["deserialize"](serialized)
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized_data.metadata.update(metadata)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_ASYNC", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            self.start_async_loop()
            log.debug(f"Asenkron veri yüklendi: data_id={data_id}, path={path}, format={format_type}")
            return data_id
        except Exception as e:
            log.error(f"Load async data hatası: {str(e)}")
            raise PdsXException(f"Load async data hatası: {str(e)}")

    def save_distributed_data(self, data: Any, bucket_name: str, key: str, credentials: Dict[str, str],
                             format_type: str = "json", encoding: str = "utf-8", compress: Optional[str] = None,
                             encrypt: Optional[str] = None, key_bytes: Optional[bytes] = None) -> str:
        """Veriyi dağıtık olarak kaydeder (S3 benzeri)."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized = serialized_data.serialize()
            
            # Sıkıştırma
            serialized = compression_methods[compress](serialized)
            serialized_data.metadata["compressed"] = compress
            
            # Şifreleme
            if encrypt == "aes" and key_bytes:
                cipher = AES.new(key_bytes, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                serialized = cipher.nonce + tag + ciphertext
                serialized_data.metadata["encrypted"] = True
            
            # S3’e kaydetme
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=serialized)
            s3_client.put_object(Bucket=bucket_name, Key=f"{key}.meta", Body=json.dumps(serialized_data.metadata).encode('utf-8'))
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_DISTRIBUTED", timestamp, prev_hash, serialized_data.metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Dağıtık veri kaydedildi: data_id={data_id}, bucket={bucket_name}, key={key}")
            return data_id
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed save hatası: {str(e)}")
            raise PdsXException(f"Distributed save hatası: {str(e)}")

    async def load_distributed_data(self, bucket_name: str, key: str, credentials: Dict[str, str],
                                   format_type: Optional[str] = None, encoding: Optional[str] = None,
                                   decompress: Optional[str] = None, decrypt: Optional[str] = None, key_bytes: Optional[bytes] = None) -> str:
        """Veriyi dağıtık olarak yükler."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Metadata okuma
            metadata = {}
            try:
                meta_response = s3_client.get_object(Bucket=bucket_name, Key=f"{key}.meta")
                metadata = json.loads(meta_response['Body'].read().decode('utf-8'))
            except botocore.exceptions.ClientError:
                pass
            
            # Otomatik format/encoding tespiti
            format_type = format_type or metadata.get("format", "json")
            encoding = encoding or metadata.get("encoding", "utf-8")
            decompress = decompress or metadata.get("compressed", "none")
            decrypt = decrypt if decrypt is not None else ("aes" if metadata.get("encrypted", False) else None)
            
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            # Veri yükleme
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            serialized = response['Body'].read()
            
            # Şifre çözme
            if decrypt == "aes" and key_bytes:
                nonce, tag, ciphertext = serialized[:16], serialized[16:32], serialized[32:]
                cipher = AES.new(key_bytes, AES.MODE_EAX, nonce=nonce)
                serialized = cipher.decrypt_and_verify(ciphertext, tag)
            
            # Sıkıştırmayı çözme
            serialized = decompression_methods[decompress](serialized)
            
            # Deserializasyon
            data = format_registry[format_type]["deserialize"](serialized)
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized_data.metadata.update(metadata)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_DISTRIBUTED", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"Dağıtık veri yüklendi: data_id={data_id}, bucket={bucket_name}, key={key}")
            return data_id
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed load hatası: {str(e)}")
            raise PdsXException(f"Distributed load hatası: {str(e)}")

    async def save_stream_data(self, data: Any, ws_url: str, format_type: str = "json", encoding: str = "utf-8",
                              compress: Optional[str] = None, encrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """WebSocket üzerinden veri kaydeder."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            compress = compress or "none"
            if compress not in compression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {compress}")
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized = serialized_data.serialize()
            
            # Sıkıştırma
            serialized = compression_methods[compress](serialized)
            serialized_data.metadata["compressed"] = compress
            
            # Şifreleme
            if encrypt == "aes" and key:
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                serialized = cipher.nonce + tag + ciphertext
                serialized_data.metadata["encrypted"] = True
            
            # WebSocket ile gönderme
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps(serialized_data.metadata).encode('utf-8'))
                await ws.send(serialized)
            
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "SAVE_STREAM", timestamp, prev_hash, serialized_data.metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"WebSocket verisi kaydedildi: data_id={data_id}, ws_url={ws_url}")
            return data_id
        except Exception as e:
            log.error(f"WebSocket save hatası: {str(e)}")
            raise PdsXException(f"WebSocket save hatası: {str(e)}")

    async def load_stream_data(self, ws_url: str, format_type: Optional[str] = None, encoding: Optional[str] = None,
                              decompress: Optional[str] = None, decrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """WebSocket üzerinden veri yükler."""
        try:
            data_id = str(uuid.uuid4())
            timestamp = time.time()
            
            async with websockets.connect(ws_url) as ws:
                metadata_str = await ws.recv()
                metadata = json.loads(metadata_str.decode('utf-8'))
                serialized = await ws.recv()
            
            # Otomatik format/encoding tespiti
            format_type = format_type or metadata.get("format", "json")
            encoding = encoding or metadata.get("encoding", "utf-8")
            decompress = decompress or metadata.get("compressed", "none")
            decrypt = decrypt if decrypt is not None else ("aes" if metadata.get("encrypted", False) else None)
            
            if format_type not in format_registry:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
            if encoding not in supported_encodings:
                raise PdsXException(f"Desteklenmeyen encoding: {encoding}")
            if decompress not in decompression_methods:
                raise PdsXException(f"Desteklenmeyen sıkıştırma: {decompress}")
            
            # Şifre çözme
            if decrypt == "aes" and key:
                nonce, tag, ciphertext = serialized[:16], serialized[16:32], serialized[32:]
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                serialized = cipher.decrypt_and_verify(ciphertext, tag)
            
            # Sıkıştırmayı çözme
            serialized = decompression_methods[decompress](serialized)
            
            # Deserializasyon
            data = format_registry[format_type]["deserialize"](serialized)
            
            serialized_data = SerializedData(data_id, data, format_type, encoding, timestamp)
            serialized_data.metadata.update(metadata)
            self.data_store[data_id] = serialized_data
            self.temporal_graph.add_data(data_id, timestamp)
            self.data_shield.train(len(serialized), 0.1)
            
            # Provenance chain
            prev_hash = self._get_last_provenance_hash(data_id)
            block = ProvenanceBlock(str(uuid.uuid4()), data_id, "LOAD_STREAM", timestamp, prev_hash, metadata)
            self.provenance_chain.setdefault(data_id, []).append(block)
            
            log.debug(f"WebSocket verisi yüklendi: data_id={data_id}, ws_url={ws_url}")
            return data_id
        except Exception as e:
            log.error(f"WebSocket load hatası: {str(e)}")
            raise PdsXException(f"WebSocket load hatası: {str(e)}")

    @synchronized
    def read_data(self, path: str, format_type: Optional[str] = None, encoding: Optional[str] = None,
                  decompress: Optional[str] = None, decrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Dosyadan veri okur."""
        try:
            data_id = self.load_data(path, format_type, encoding, decompress, decrypt, key)
            serialized_data = self.data_store[data_id]
            self.data_list.extend([serialized_data.data] if not isinstance(serialized_data.data, list) else serialized_data.data)
            self.data_pointer = 0
            log.debug(f"Veri okundu ve data listesine eklendi: data_id={data_id}")
            return data_id
        except Exception as e:
            log.error(f"Read data hatası: {str(e)}")
            raise PdsXException(f"Read data hatası: {str(e)}")

    @synchronized
    def write_data(self, data: Any, path: str, format_type: str = "json", encoding: str = "utf-8",
                   compress: Optional[str] = None, encrypt: Optional[str] = None, key: Optional[bytes] = None) -> str:
        """Veriyi dosyaya yazar."""
        return self.save_data(data, path, format_type, encoding, compress, encrypt, key)

    def add_data(self, data: List[Any]) -> None:
        """DATA komutu için veri ekler."""
        try:
            self.data_list.extend(data)
            log.debug(f"DATA eklendi: {data}")
        except Exception as e:
            log.error(f"Add data hatası: {str(e)}")
            raise PdsXException(f"Add data hatası: {str(e)}")

    def read_data_list(self, var_names: List[str]) -> None:
        """READ komutu için veri listesinden okur."""
        try:
            for var_name in var_names:
                if self.data_pointer >= len(self.data_list):
                    raise PdsXException("Veri listesi sonuna ulaşıldı")
                self.interpreter.current_scope()[var_name] = self.data_list[self.data_pointer]
                self.data_pointer += 1
            log.debug(f"READ tamamlandı: {var_names}")
        except Exception as e:
            log.error(f"Read data list hatası: {str(e)}")
            raise PdsXException(f"Read data list hatası: {str(e)}")

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
            # SAVE DATA
            if command_upper.startswith("SAVE DATA "):
                match = re.match(r"SAVE DATA\s+(.+?)\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:COMPRESS\s+(\w+))?\s*(?:ENCRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format_type, encoding, compress, encrypt, key_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    compress = compress or "none"
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = self.save_data(data, path, format_type, encoding, compress, encrypt, key)
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE DATA komutunda sözdizimi hatası")
            
            # SAVE ASYNC DATA
            elif command_upper.startswith("SAVE ASYNC DATA "):
                match = re.match(r"SAVE ASYNC DATA\s+(.+?)\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:COMPRESS\s+(\w+))?\s*(?:ENCRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format_type, encoding, compress, encrypt, key_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    compress = compress or "none"
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = asyncio.run(self.save_async_data(data, path, format_type, encoding, compress, encrypt, key))
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE ASYNC DATA komutunda sözdizimi hatası")
            
            # LOAD DATA
            elif command_upper.startswith("LOAD DATA "):
                match = re.match(r"LOAD DATA\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:DECOMPRESS\s+(\w+))?\s*(?:DECRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format_type, encoding, decompress, decrypt, key_str, var_name = match.groups()
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = self.load_data(path, format_type, encoding, decompress, decrypt, key)
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD DATA komutunda sözdizimi hatası")
            
            # LOAD ASYNC DATA
            elif command_upper.startswith("LOAD ASYNC DATA "):
                match = re.match(r"LOAD ASYNC DATA\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:DECOMPRESS\s+(\w+))?\s*(?:DECRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format_type, encoding, decompress, decrypt, key_str, var_name = match.groups()
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = asyncio.run(self.load_async_data(path, format_type, encoding, decompress, decrypt, key))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD ASYNC DATA komutunda sözdizimi hatası")
            
            # SAVE DISTRIBUTED
            elif command_upper.startswith("SAVE DISTRIBUTED "):
                match = re.match(r"SAVE DISTRIBUTED\s+(.+?)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:COMPRESS\s+(\w+))?\s*(?:ENCRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, bucket_name, key, creds_str, format_type, encoding, compress, encrypt, key_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    compress = compress or "none"
                    key_bytes = base64.b64decode(key_str) if key_str else None
                    data_id = self.save_distributed_data(data, bucket_name, key, credentials, format_type, encoding, compress, encrypt, key_bytes)
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE DISTRIBUTED komutunda sözdizimi hatası")
            
            # LOAD DISTRIBUTED
            elif command_upper.startswith("LOAD DISTRIBUTED "):
                match = re.match(r"LOAD DISTRIBUTED\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:DECOMPRESS\s+(\w+))?\s*(?:DECRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    bucket_name, key, creds_str, format_type, encoding, decompress, decrypt, key_str, var_name = match.groups()
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    decompress = decompress or "none"
                    key_bytes = base64.b64decode(key_str) if key_str else None
                    data_id = asyncio.run(self.load_distributed_data(bucket_name, key, credentials, format_type, encoding, decompress, decrypt, key_bytes))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD DISTRIBUTED komutunda sözdizimi hatası")
            
            # SAVE STREAM
            elif command_upper.startswith("SAVE STREAM "):
                match = re.match(r"SAVE STREAM\s+(.+?)\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:COMPRESS\s+(\w+))?\s*(?:ENCRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, ws_url, format_type, encoding, compress, encrypt, key_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    compress = compress or "none"
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = asyncio.run(self.save_stream_data(data, ws_url, format_type, encoding, compress, encrypt, key))
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("SAVE STREAM komutunda sözdizimi hatası")
            
            # LOAD STREAM
            elif command_upper.startswith("LOAD STREAM "):
                match = re.match(r"LOAD STREAM\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:DECOMPRESS\s+(\w+))?\s*(?:DECRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    ws_url, format_type, encoding, decompress, decrypt, key_str, var_name = match.groups()
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    decompress = decompress or "none"
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = asyncio.run(self.load_stream_data(ws_url, format_type, encoding, decompress, decrypt, key))
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("LOAD STREAM komutunda sözdizimi hatası")
            
            # SAVE ENCRYPT
            elif command_upper.startswith("SAVE ENCRYPT "):
                match = re.match(r"SAVE ENCRYPT\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id, key_str, method, var_name = match.groups()
                    if data_id not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id}")
                    key = base64.b64decode(key_str)
                    encrypted = self.encrypt_data(data_id, key, method)
                    self.interpreter.current_scope()[var_name] = encrypted
                else:
                    raise PdsXException("SAVE ENCRYPT komutunda sözdizimi hatası")
            
            # READ DATA
            elif command_upper.startswith("READ DATA "):
                match = re.match(r"READ DATA\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:DECOMPRESS\s+(\w+))?\s*(?:DECRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format_type, encoding, decompress, decrypt, key_str, var_name = match.groups()
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = self.read_data(path, format_type, encoding, decompress, decrypt, key)
                    self.interpreter.current_scope()[var_name] = self.data_store[data_id].data
                else:
                    raise PdsXException("READ DATA komutunda sözdizimi hatası")
            
            # WRITE DATA
            elif command_upper.startswith("WRITE DATA "):
                match = re.match(r"WRITE DATA\s+(.+?)\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s*(?:ENCODING\s+(\w+))?\s*(?:COMPRESS\s+(\w+))?\s*(?:ENCRYPT\s+(\w+)\s+KEY\s+\"([^\"]+)\")?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format_type, encoding, compress, encrypt, key_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    encoding = encoding or "utf-8"
                    compress = compress or "none"
                    key = base64.b64decode(key_str) if key_str else None
                    data_id = self.write_data(data, path, format_type, encoding, compress, encrypt, key)
                    self.interpreter.current_scope()[var_name] = data_id
                else:
                    raise PdsXException("WRITE DATA komutunda sözdizimi hatası")
            
            # DATA
            elif command_upper.startswith("DATA "):
                match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
                if match:
                    data_str = match.group(1)
                    data_items = [self.interpreter.evaluate_expression(item.strip()) for item in data_str.split(",")]
                    self.add_data(data_items)
                else:
                    raise PdsXException("DATA komutunda sözdizimi hatası")
            
            # READ
            elif command_upper.startswith("READ "):
                match = re.match(r"READ\s+(.+)", command, re.IGNORECASE)
                if match:
                    var_names = [name.strip() for name in match.group(1).split(",")]
                    self.read_data_list(var_names)
                else:
                    raise PdsXException("READ komutunda sözdizimi hatası")
            
            # ANALYZE DATA
            elif command_upper.startswith("ANALYZE DATA "):
                match = re.match(r"ANALYZE DATA\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_data": len(self.data_store),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [did for did, d in self.data_store.items() if self.data_shield.predict(len(str(d.data)), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("ANALYZE DATA komutunda sözdizimi hatası")
            
            # VISUALIZE DATA
            elif command_upper.startswith("VISUALIZE DATA "):
                match = re.match(r"VISUALIZE DATA\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    output_path, format_type, var_name = match.groups()
                    format_type = format_type or "png"
                    dot = graphviz.Digraph(format=format_type)
                    for did, data in self.data_store.items():
                        node_label = f"ID: {did}\nFormat: {data.format_type}\nEncoding: {data.encoding}\nTime: {data.timestamp}"
                        dot.node(did, node_label, color="red" if data.metadata["encrypted"] else "green")
                    for did1 in self.temporal_graph.edges:
                        for did2, weight in self.temporal_graph.edges[did1]:
                            dot.edge(did1, did2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    self.interpreter.current_scope()[var_name] = True
                    log.debug(f"Veriler görselleştirildi: path={output_path}.{format_type}")
                else:
                    raise PdsXException("VISUALIZE DATA komutunda sözdizimi hatası")
            
            # QUANTUM DATA
            elif command_upper.startswith("QUANTUM DATA "):
                match = re.match(r"QUANTUM DATA\s+(\w+)\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id1, data_id2, var_name = match.groups()
                    if data_id1 not in self.data_store or data_id2 not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id1} veya {data_id2}")
                    correlation_id = self.quantum_correlator.correlate(self.data_store[data_id1], self.data_store[data_id2])
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("QUANTUM DATA komutunda sözdizimi hatası")
            
            # HOLO DATA
            elif command_upper.startswith("HOLO DATA "):
                match = re.match(r"HOLO DATA\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id, var_name = match.groups()
                    if data_id not in self.data_store:
                        raise PdsXException(f"Veri bulunamadı: {data_id}")
                    pattern = self.holo_compressor.compress(self.data_store[data_id])
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("HOLO DATA komutunda sözdizimi hatası")
            
            # SMART DATA
            elif command_upper.startswith("SMART DATA "):
                match = re.match(r"SMART DATA\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, access_time, var_name = match.groups()
                    data_size = int(data_size)
                    access_time = float(access_time)
                    storage = self.smart_fabric.optimize(data_size, access_time)
                    self.interpreter.current_scope()[var_name] = storage
                else:
                    raise PdsXException("SMART DATA komutunda sözdizimi hatası")
            
            # TEMPORAL DATA
            elif command_upper.startswith("TEMPORAL DATA "):
                match = re.match(r"TEMPORAL DATA\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id1, data_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(data_id1, data_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("TEMPORAL DATA komutunda sözdizimi hatası")
            
            # PREDICT DATA
            elif command_upper.startswith("PREDICT DATA "):
                match = re.match(r"PREDICT DATA\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_size, access_time, var_name = match.groups()
                    data_size = int(data_size)
                    access_time = float(access_time)
                    is_anomaly = self.data_shield.predict(data_size, access_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("PREDICT DATA komutunda sözdizimi hatası")
            
            else:
                raise PdsXException(f"Bilinmeyen kaydetme/yükleme komutu: {command}")
        except Exception as e:
            log.error(f"Kaydetme/yükleme komut hatası: {str(e)}")
            raise PdsXException(f"Kaydetme/yükleme komut hatası: {str(e)}")

if __name__ == "__main__":
    print("save_load_system.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")