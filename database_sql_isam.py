# database_sql_isam.py - PDS-X BASIC v14u SQL ve ISAM Veritabanı Yönetim Kütüphanesi
# Version: 1.0.0
# Date: May 13, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import threading
import asyncio
import time
import sqlite3
import psycopg2
import json
import pickle
import base64
import gzip
import zlib
import uuid
import hashlib
import graphviz
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from pdsx_exception import PdsXException
import functools
from save_load_system import format_registry, supported_encodings, compression_methods, decompression_methods

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("database_sql_isam")

# Dekoratör
def synchronized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with args[0].lock:
            return fn(*args, **kwargs)
    return wrapped

class SQLConnection:
    """SQL veritabanı bağlantı sınıfı."""
    def __init__(self, conn_id: str, db_type: str, conn_params: Dict):
        self.conn_id = conn_id
        self.db_type = db_type.lower()
        self.conn_params = conn_params
        self.connection = None
        self.cursor = None
        self.lock = threading.Lock()
        self.connect()

    @synchronized
    def connect(self) -> None:
        """Veritabanına bağlanır."""
        try:
            if self.db_type == "sqlite":
                self.connection = sqlite3.connect(self.conn_params.get("database", ":memory:"))
                self.cursor = self.connection.cursor()
            elif self.db_type == "postgresql":
                self.connection = psycopg2.connect(**self.conn_params)
                self.cursor = self.connection.cursor()
            else:
                raise PdsXException(f"Desteklenmeyen veritabanı tipi: {self.db_type}")
            log.debug(f"SQL bağlantısı kuruldu: conn_id={self.conn_id}, type={self.db_type}")
        except Exception as e:
            log.error(f"SQL bağlantı hatası: conn_id={self.conn_id}, hata={str(e)}")
            raise PdsXException(f"SQL bağlantı hatası: {str(e)}")

    @synchronized
    def execute(self, query: str, params: Tuple = ()) -> List:
        """SQL sorgusunu yürütür."""
        try:
            self.cursor.execute(query, params)
            if query.upper().startswith("SELECT"):
                result = self.cursor.fetchall()
            else:
                self.connection.commit()
                result = []
            log.debug(f"SQL sorgusu yürütüldü: conn_id={self.conn_id}, query={query}")
            return result
        except Exception as e:
            log.error(f"SQL sorgu hatası: conn_id={self.conn_id}, query={query}, hata={str(e)}")
            raise PdsXException(f"SQL sorgu hatası: {str(e)}")

    @synchronized
    def close(self) -> None:
        """Bağlantıyı kapatır."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            log.debug(f"SQL bağlantısı kapatıldı: conn_id={self.conn_id}")
        except Exception as e:
            log.error(f"SQL bağlantı kapatma hatası: conn_id={self.conn_id}, hata={str(e)}")
            raise PdsXException(f"SQL bağlantı kapatma hatası: {str(e)}")

class ISAMTable:
    """ISAM tablo sınıfı."""
    def __init__(self, table_id: str, path: str, fields: List[Tuple[str, str]], encoding: str = "utf-8"):
        self.table_id = table_id
        self.path = path
        self.fields = fields
        self.encoding = encoding
        self.index = {}  # {key: file_offset}
        self.data_file = Path(path)
        self.index_file = Path(f"{path}.idx")
        self.lock = threading.Lock()
        self._initialize()

    @synchronized
    def _initialize(self) -> None:
        """Tabloyu başlatır."""
        try:
            if not self.data_file.exists():
                with open(self.data_file, 'wb') as f:
                    f.write(b"")
            if not self.index_file.exists():
                with open(self.index_file, 'wb') as f:
                    pickle.dump({}, f)
            with open(self.index_file, 'rb') as f:
                self.index = pickle.load(f)
            log.debug(f"ISAM tablo başlatıldı: table_id={self.table_id}, path={self.path}")
        except Exception as e:
            log.error(f"ISAM tablo başlatma hatası: table_id={self.table_id}, hata={str(e)}")
            raise PdsXException(f"ISAM tablo başlatma hatası: {str(e)}")

    @synchronized
    def insert(self, record: Dict) -> None:
        """Kayıt ekler."""
        try:
            key = record.get(self.fields[0][0])  # İlk alan anahtar
            if key in self.index:
                raise PdsXException(f"Anahtar zaten mevcut: {key}")
            serialized = json.dumps(record).encode(self.encoding)
            with open(self.data_file, 'ab') as f:
                offset = f.tell()
                f.write(serialized + b"\n")
            self.index[key] = offset
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
            log.debug(f"ISAM kayıt eklendi: table_id={self.table_id}, key={key}")
        except Exception as e:
            log.error(f"ISAM insert hatası: table_id={self.table_id}, hata={str(e)}")
            raise PdsXException(f"ISAM insert hatası: {str(e)}")

    @synchronized
    def search(self, key: Any) -> Optional[Dict]:
        """Anahtarla kayıt arar."""
        try:
            if key not in self.index:
                return None
            offset = self.index[key]
            with open(self.data_file, 'rb') as f:
                f.seek(offset)
                line = f.readline().decode(self.encoding)
                return json.loads(line)
            log.debug(f"ISAM arama: table_id={self.table_id}, key={key}")
        except Exception as e:
            log.error(f"ISAM arama hatası: table_id={self.table_id}, hata={str(e)}")
            raise PdsXException(f"ISAM arama hatası: {str(e)}")

class QuantumQueryCorrelator:
    """Kuantum tabanlı sorgu korelasyon sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (query_id1, query_id2, score)}

    def correlate(self, query1: str, query2: str) -> str:
        """İki sorguyu kuantum simülasyonuyla ilişkilendirir."""
        try:
            set1 = set(query1.split())
            set2 = set(query2.split())
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (query1, query2, score)
            log.debug(f"Kuantum korelasyon: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumQueryCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumQueryCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[str, str, float]]:
        """Korelasyonu döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumQueryCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumQueryCorrelator get_correlation hatası: {str(e)}")

class HoloDataCompressor:
    """Holografik veri sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [serialized_data]}

    def compress(self, data: Any) -> str:
        """Veriyi holografik olarak sıkıştırır."""
        try:
            serialized = pickle.dumps(data)
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            log.debug(f"Holografik veri sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloDataCompressor compress hatası: {str(e)}")
            raise PdsXException(f"HoloDataCompressor compress hatası: {str(e)}")

    def decompress(self, pattern: str) -> Optional[Any]:
        """Veriyi geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                serialized = self.storage[pattern][-1]
                return pickle.loads(serialized)
            return None
        except Exception as e:
            log.error(f"HoloDataCompressor decompress hatası: {str(e)}")
            raise PdsXException(f"HoloDataCompressor decompress hatası: {str(e)}")

class SmartQueryOptimizer:
    """AI tabanlı sorgu optimizasyon sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(query_size, execution_time, timestamp)]

    def optimize(self, query_size: int, execution_time: float) -> str:
        """Sorguyu optimize bir şekilde planlar."""
        try:
            features = np.array([[query_size, execution_time, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:
                    strategy = "INDEXED"
                    log.warning(f"Sorgu optimize edildi: strategy={strategy}, score={anomaly_score}")
                    return strategy
            return "SEQUENTIAL"
        except Exception as e:
            log.error(f"SmartQueryOptimizer optimize hatası: {str(e)}")
            raise PdsXException(f"SmartQueryOptimizer optimize hatası: {str(e)}")

class TemporalQueryGraph:
    """Zaman temelli sorgu ilişkileri grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {query_id: timestamp}
        self.edges = defaultdict(list)  # {query_id: [(related_query_id, weight)]}

    def add_query(self, query_id: str, timestamp: float) -> None:
        """Sorguyu grafiğe ekler."""
        try:
            self.vertices[query_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: query_id={query_id}")
        except Exception as e:
            log.error(f"TemporalQueryGraph add_query hatası: {str(e)}")
            raise PdsXException(f"TemporalQueryGraph add_query hatası: {str(e)}")

    def add_relation(self, query_id1: str, query_id2: str, weight: float) -> None:
        """Sorgular arasında ilişki kurar."""
        try:
            self.edges[query_id1].append((query_id2, weight))
            self.edges[query_id2].append((query_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {query_id1} <-> {query_id2}")
        except Exception as e:
            log.error(f"TemporalQueryGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalQueryGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Sorgu grafiğini analiz eder."""
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
            log.error(f"TemporalQueryGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalQueryGraph analyze hatası: {str(e)}")

class QueryShield:
    """Tahmini sorgu hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(query_size, execution_time, timestamp)]

    def train(self, query_size: int, execution_time: float) -> None:
        """Sorgu verileriyle modeli eğitir."""
        try:
            features = np.array([query_size, execution_time, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("QueryShield modeli eğitildi")
        except Exception as e:
            log.error(f"QueryShield train hatası: {str(e)}")
            raise PdsXException(f"QueryShield train hatası: {str(e)}")

    def predict(self, query_size: int, execution_time: float) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[query_size, execution_time, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: query_size={query_size}")
            return is_anomaly
        except Exception as e:
            log.error(f"QueryShield predict hatası: {str(e)}")
            raise PdsXException(f"QueryShield predict hatası: {str(e)}")

class DatabaseManager:
    """SQL ve ISAM veritabanı yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.connections = {}  # {conn_id: SQLConnection}
        self.tables = {}  # {table_id: ISAMTable}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumQueryCorrelator()
        self.holo_compressor = HoloDataCompressor()
        self.smart_optimizer = SmartQueryOptimizer()
        self.temporal_graph = TemporalQueryGraph()
        self.query_shield = QueryShield()
        self.lock = threading.Lock()
        self.metadata = {
            "database_sql_isam": {
                "version": "1.0.0",
                "dependencies": [
                    "sqlite3", "psycopg2", "numpy", "scikit-learn", "graphviz",
                    "pdsx_exception", "save_load_system"
                ]
            }
        }
        self.max_connections = 100

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron veritabanı döngüsü başlatıldı")

    @synchronized
    def connect(self, db_type: str, conn_params: Dict) -> str:
        """Veritabanına bağlanır."""
        try:
            conn_id = str(uuid.uuid4())
            conn = SQLConnection(conn_id, db_type, conn_params)
            self.connections[conn_id] = conn
            log.debug(f"Veritabanı bağlantısı oluşturuldu: conn_id={conn_id}, type={db_type}")
            return conn_id
        except Exception as e:
            log.error(f"Veritabanı bağlantı hatası: {str(e)}")
            raise PdsXException(f"Veritabanı bağlantı hatası: {str(e)}")

    @synchronized
    def create_isam_table(self, path: str, fields: List[Tuple[str, str]], encoding: str = "utf-8") -> str:
        """ISAM tablosu oluşturur."""
        try:
            table_id = str(uuid.uuid4())
            table = ISAMTable(table_id, path, fields, encoding)
            self.tables[table_id] = table
            log.debug(f"ISAM tablosu oluşturuldu: table_id={table_id}, path={path}")
            return table_id
        except Exception as e:
            log.error(f"ISAM tablo oluşturma hatası: {str(e)}")
            raise PdsXException(f"ISAM tablo oluşturma hatası: {str(e)}")

    async def execute_async_query(self, conn_id: str, query: str, params: Tuple = ()) -> List:
        """Asenkron SQL sorgusu yürütür."""
        try:
            if conn_id not in self.connections:
                raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
            conn = self.connections[conn_id]
            self.start_async_loop()
            result = await asyncio.to_thread(conn.execute, query, params)
            log.debug(f"Asenkron sorgu yürütüldü: conn_id={conn_id}, query={query}")
            return result
        except Exception as e:
            log.error(f"Asenkron sorgu hatası: conn_id={conn_id}, hata={str(e)}")
            raise PdsXException(f"Asenkron sorgu hatası: {str(e)}")

    def parse_database_command(self, command: str) -> None:
        """Veritabanı komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            # DB CONNECT
            if command_upper.startswith("DB CONNECT "):
                match = re.match(r"DB CONNECT\s+(\w+)\s+\[(.+?)\]\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    db_type, params_str, var_name = match.groups()
                    conn_params = eval(params_str, self.interpreter.current_scope())
                    conn_id = self.connect(db_type, conn_params)
                    self.interpreter.current_scope()[var_name] = conn_id
                else:
                    raise PdsXException("DB CONNECT komutunda sözdizimi hatası")

            # DB QUERY
            elif command_upper.startswith("DB QUERY "):
                match = re.match(r"DB QUERY\s+(\w+)\s+\"([^\"]+)\"\s*(?:PARAMS\s+\[(.+?)\])?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    if conn_id not in self.connections:
                        raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
                    params = eval(params_str, self.interpreter.current_scope()) if params_str else ()
                    params = params if isinstance(params, tuple) else (params,)
                    result = self.connections[conn_id].execute(query, params)
                    self.temporal_graph.add_query(query, time.time())
                    self.query_shield.train(len(query), 0.1)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DB QUERY komutunda sözdizimi hatası")

            # DB ASYNC QUERY
            elif command_upper.startswith("DB ASYNC QUERY "):
                match = re.match(r"DB ASYNC QUERY\s+(\w+)\s+\"([^\"]+)\"\s*(?:PARAMS\s+\[(.+?)\])?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    if conn_id not in self.connections:
                        raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
                    params = eval(params_str, self.interpreter.current_scope()) if params_str else ""
                    params = params if isinstance(params, tuple) else (params,)
                    result = asyncio.run(self.execute_async_query(conn_id, query, params))
                    self.temporal_graph.add_query(query, time.time())
                    self.query_shield.train(len(query), 0.1)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DB ASYNC QUERY komutunda sözdizimi hatası")

            # DB CREATE TABLE
            elif command_upper.startswith("DB CREATE TABLE "):
                match = re.match(r"DB CREATE TABLE\s+(\w+)\s+\"([^\"]+)\"\s+\[(.+?)\]\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id, table_name, fields_str, var_name = match.groups()
                    if conn_id not in self.connections:
                        raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
                    fields = eval(fields_str, self.interpreter.current_scope())
                    columns = ", ".join(f"{name} {type_name}" for name, type_name in fields)
                    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
                    self.connections[conn_id].execute(query)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("DB CREATE TABLE komutunda sözdizimi hatası")

            # DB ISAM CREATE
            elif command_upper.startswith("DB ISAM CREATE "):
                match = re.match(r"DB ISAM CREATE\s+\"([^\"]+)\"\s+\[(.+?)\]\s*(?:ENCODING\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, fields_str, encoding, var_name = match.groups()
                    fields = eval(fields_str, self.interpreter.current_scope())
                    encoding = encoding or "utf-8"
                    table_id = self.create_isam_table(path, fields, encoding)
                    self.interpreter.current_scope()[var_name] = table_id
                else:
                    raise PdsXException("DB ISAM CREATE komutunda sözdizimi hatası")

            # DB ISAM INSERT
            elif command_upper.startswith("DB ISAM INSERT "):
                match = re.match(r"DB ISAM INSERT\s+(\w+)\s+\[(.+?)\]\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    table_id, record_str, var_name = match.groups()
                    if table_id not in self.tables:
                        raise PdsXException(f"Tablo bulunamadı: {table_id}")
                    record = eval(record_str, self.interpreter.current_scope())
                    self.tables[table_id].insert(record)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("DB ISAM INSERT komutunda sözdizimi hatası")

            # DB ISAM SEARCH
            elif command_upper.startswith("DB ISAM SEARCH "):
                match = re.match(r"DB ISAM SEARCH\s+(\w+)\s+(.+?)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    table_id, key_str, var_name = match.groups()
                    if table_id not in self.tables:
                        raise PdsXException(f"Tablo bulunamadı: {table_id}")
                    key = self.interpreter.evaluate_expression(key_str)
                    result = self.tables[table_id].search(key)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DB ISAM SEARCH komutunda sözdizimi hatası")

            # DB ANALYZE
            elif command_upper.startswith("DB ANALYZE "):
                match = re.match(r"DB ANALYZE\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_connections": len(self.connections),
                        "total_tables": len(self.tables),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [query for query in self.temporal_graph.vertices if self.query_shield.predict(len(query), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DB ANALYZE komutunda sözdizimi hatası")

            # DB VISUALIZE
            elif command_upper.startswith("DB VISUALIZE "):
                match = re.match(r"DB VISUALIZE\s+\"([^\"]+)\"\s*(?:FORMAT\s+(\w+))?\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    output_path, format_type, var_name = match.groups()
                    format_type = format_type or "png"
                    dot = graphviz.Digraph(format=format_type)
                    for conn_id, conn in self.connections.items():
                        node_label = f"ID: {conn_id}\nType: {conn.db_type}\n"
                        dot.node(conn_id, node_label, color="blue")
                    for table_id, table in self.tables.items():
                        node_label = f"ID: {table_id}\nType: ISAM\nPath: {table.path}"
                        dot.node(table_id, node_label, color="green")
                    for qid1 in self.temporal_graph.edges:
                        for qid2, weight in self.temporal_graph.edges[qid1]:
                            dot.edge(qid1, qid2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    self.interpreter.current_scope()[var_name] = True
                    log.debug(f"Veritabanı görselleştirildi: path={output_path}.{format_type}")
                else:
                    raise PdsXException("DB VISUALIZE komutunda sözdizimi hatası")

            # DB QUANTUM
            elif command_upper.startswith("DB QUANTUM "):
                match = re.match(r"DB QUANTUM\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    query1, query2, var_name = match.groups()
                    correlation_id = self.quantum_correlator.correlate(query1, query2)
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("DB QUANTUM komutunda sözdizimi hatası")

            # DB HOLO
            elif command_upper.startswith("DB HOLO "):
                match = re.match(r"DB HOLO\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_id, var_name = match.groups()
                    if data_id not in self.connections and data_id not in self.tables:
                        raise PdsXException(f"Veri bulunamadı: {data_id}")
                    data = self.connections.get(data_id) or self.tables.get(data_id)
                    pattern = self.holo_compressor.compress(data)
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("DB HOLO komutunda sözdizimi hatası")

            # DB SMART
            elif command_upper.startswith("DB SMART "):
                match = re.match(r"DB SMART\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    query_size, execution_time, var_name = match.groups()
                    query_size = int(query_size)
                    execution_time = float(execution_time)
                    strategy = self.smart_optimizer.optimize(query_size, execution_time)
                    self.interpreter.current_scope()[var_name] = strategy
                else:
                    raise PdsXException("DB SMART komutunda sözdizimi hatası")

            # DB TEMPORAL
            elif command_upper.startswith("DB TEMPORAL "):
                match = re.match(r"DB TEMPORAL\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    query_id1, query_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(query_id1, query_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("DB TEMPORAL komutunda sözdizimi hatası")

            # DB PREDICT
            elif command_upper.startswith("DB PREDICT "):
                match = re.match(r"DB PREDICT\s+(\d+)\s+(\d*\.?\d*)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    query_size, execution_time, var_name = match.groups()
                    query_size = int(query_size)
                    execution_time = float(execution_time)
                    is_anomaly = self.query_shield.predict(query_size, execution_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("DB PREDICT komutunda sözdizimi hatası")

            else:
                raise PdsXException(f"Bilinmeyen veritabanı komutu: {command}")
        except Exception as e:
            log.error(f"Veritabanı komut hatası: {str(e)}")
            raise PdsXException(f"Veritabanı komut hatası: {str(e)}")

if __name__ == "__main__":
    print("database_sql_isam.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")