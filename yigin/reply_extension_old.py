# reply_extension.py - PDS-X BASIC v14u Yanıt Uzantısı Kütüphanesi
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
import xml.etree.ElementTree as ET
import base64
import sys
import subprocess
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Program manager import
from program_manager import MultiLineProgramManager
from pathlib import Path
from collections import defaultdict
import uuid
import hashlib
import graphviz
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sklearn.ensemble import IsolationForest  # AI tabanlı anomali algılama
import boto3
import botocore
import websockets
from pdsx_exception import PdsXException  # Hata yönetimi için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("reply_extension")

class Response:
    """Temel yanıt sınıfı."""
    def __init__(self, response_id: str, data: Any, timestamp: float):
        self.response_id = response_id
        self.data = data
        self.timestamp = timestamp
        self.metadata = {"format": "json", "encrypted": False, "compressed": False}
        self.execution_time = 0.0

    def format(self, format_type: str) -> bytes:
        """Yanıtı belirtilen formatta serileştirir."""
        try:
            format_type = format_type.lower()
            self.metadata["format"] = format_type
            if format_type == "json":
                return json.dumps(self.data).encode('utf-8')
            elif format_type == "yaml":
                return yaml.dump(self.data).encode('utf-8')
            elif format_type == "xml":
                root = ET.Element("response")
                root.text = str(self.data)
                return ET.tostring(root)
            elif format_type == "pdsx":
                pdsx_data = {"data": self.data, "meta": self.metadata}
                return json.dumps(pdsx_data).encode('utf-8')
            else:
                raise PdsXException(f"Desteklenmeyen format: {format_type}")
        except Exception as e:
            log.error(f"Response format hatası: {str(e)}")
            raise PdsXException(f"Response format hatası: {str(e)}")

class QuantumResponseCorrelator:
    """Kuantum tabanlı yanıt korelasyon sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (response1_id, response2_id, score)}

    def correlate(self, response1: Response, response2: Response) -> str:
        """İki yanıtı kuantum simülasyonuyla ilişkilendirir."""
        try:
            # Basit simülasyon: yanıt verilerinin benzerliği (Jaccard)
            set1 = set(str(response1.data))
            set2 = set(str(response2.data))
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (response1.response_id, response2.response_id, score)
            log.debug(f"Kuantum korelasyon: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumResponseCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumResponseCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[str, str, float]]:
        """Korelasyonu döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumResponseCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumResponseCorrelator get_correlation hatası: {str(e)}")

class HoloResponse:
    """Holografik yanıt sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [response_data]}

    def compress(self, response: Response) -> str:
        """Yanıtı holografik olarak sıkıştırır."""
        try:
            serialized = response.format(response.metadata["format"])
            pattern = hashlib.sha256(serialized).hexdigest()[:16]
            self.storage[pattern].append(serialized)
            log.debug(f"Holografik yanıt sıkıştırıldı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloResponse compress hatası: {str(e)}")
            raise PdsXException(f"HoloResponse compress hatası: {str(e)}")

    def decompress(self, pattern: str) -> Optional[bytes]:
        """Yanıtı geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                return self.storage[pattern][-1]
            return None
        except Exception as e:
            log.error(f"HoloResponse decompress hatası: {str(e)}")
            raise PdsXException(f"HoloResponse decompress hatası: {str(e)}")

class SmartRouter:
    """AI tabanlı otomatik yanıt yönlendirme sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(response_size, exec_time, timestamp)]

    def route(self, response: Response, exec_time: float) -> str:
        """Yanıtı optimize bir şekilde yönlendirir."""
        try:
            response_size = len(str(response.data))
            features = np.array([[response_size, exec_time, time.time()]])
            self.history.append(features[0])
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                anomaly_score = self.model.score_samples(features)[0]
                if anomaly_score < -0.5:  # Anomali tespit edildi
                    route = "ALTERNATE"  # Alternatif rota
                    log.warning(f"Yanıt yönlendirme optimize edildi: route={route}, score={anomaly_score}")
                    return route
            return "DEFAULT"
        except Exception as e:
            log.error(f"SmartRouter route hatası: {str(e)}")
            raise PdsXException(f"SmartRouter route hatası: {str(e)}")

class TemporalResponseGraph:
    """Zaman temelli yanıt grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {response_id: timestamp}
        self.edges = defaultdict(list)  # {response_id: [(related_response_id, weight)]}

    def add_response(self, response_id: str, timestamp: float) -> None:
        """Yanıtı grafiğe ekler."""
        try:
            self.vertices[response_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: response_id={response_id}")
        except Exception as e:
            log.error(f"TemporalResponseGraph add_response hatası: {str(e)}")
            raise PdsXException(f"TemporalResponseGraph add_response hatası: {str(e)}")

    def add_relation(self, response_id1: str, response_id2: str, weight: float) -> None:
        """Yanıtlar arasında ilişki kurar."""
        try:
            self.edges[response_id1].append((response_id2, weight))
            self.edges[response_id2].append((response_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {response_id1} <-> {response_id2}")
        except Exception as e:
            log.error(f"TemporalResponseGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalResponseGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Yanıt grafiğini analiz eder."""
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
            log.error(f"TemporalResponseGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalResponseGraph analyze hatası: {str(e)}")

class ResponseShield:
    """Tahmini yanıt hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []  # [(response_size, exec_time, timestamp)]

    def train(self, response_size: int, exec_time: float) -> None:
        """Yanıt verileriyle modeli eğitir."""
        try:
            features = np.array([response_size, exec_time, time.time()])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("ResponseShield modeli eğitildi")
        except Exception as e:
            log.error(f"ResponseShield train hatası: {str(e)}")
            raise PdsXException(f"ResponseShield train hatası: {str(e)}")

    def predict(self, response_size: int, exec_time: float) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[response_size, exec_time, time.time()]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: response_size={response_size}")
            return is_anomaly
        except Exception as e:
            log.error(f"ResponseShield predict hatası: {str(e)}")
            raise PdsXException(f"ResponseShield predict hatası: {str(e)}")

class ReplyExtension:
    """Yanıt uzantısı yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.responses = {}  # {response_id: Response}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumResponseCorrelator()
        self.holo_response = HoloResponse()
        self.smart_router = SmartRouter()
        self.temporal_graph = TemporalResponseGraph()
        self.response_shield = ResponseShield()
        self.lock = threading.Lock()
        
        # Çok satırlı program sistemi
        self.program_manager = MultiLineProgramManager()
        
        self.metadata = {
            "reply_extension": {
                "version": "1.0.0",
                "dependencies": ["graphviz", "numpy", "scikit-learn", "boto3", "websockets", "pyyaml", "pycryptodome", "pdsx_exception"]
            }
        }
        self.max_response_size = 10000

    def start_async_loop(self) -> None:
        """Asenkron döngüyü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron yanıt döngüsü başlatıldı")

    def send_reply(self, data: Any, format_type: str = "json") -> str:
        """Yanıt gönderir."""
        with self.lock:
            try:
                response_id = str(uuid.uuid4())
                timestamp = time.time()
                response = Response(response_id, data, timestamp)
                serialized = response.format(format_type)
                self.responses[response_id] = response
                exec_time = 0.1  # Varsayılan yürütme süresi tahmini
                self.response_shield.train(len(serialized), exec_time)
                self.temporal_graph.add_response(response_id, timestamp)
                log.debug(f"Yanıt gönderildi: response_id={response_id}, format={format_type}")
                return response_id
            except Exception as e:
                log.error(f"Send reply hatası: {str(e)}")
                raise PdsXException(f"Send reply hatası: {str(e)}")

    async def send_async_reply(self, data: Any, format_type: str = "json") -> str:
        """Asenkron yanıt gönderir."""
        try:
            response_id = str(uuid.uuid4())
            timestamp = time.time()
            response = Response(response_id, data, timestamp)
            serialized = response.format(format_type)
            self.responses[response_id] = response
            exec_time = 0.1
            self.response_shield.train(len(serialized), exec_time)
            self.temporal_graph.add_response(response_id, timestamp)
            self.start_async_loop()
            log.debug(f"Asenkron yanıt gönderildi: response_id={response_id}, format={format_type}")
            return response_id
        except Exception as e:
            log.error(f"Send async reply hatası: {str(e)}")
            raise PdsXException(f"Send async reply hatası: {str(e)}")

    def send_distributed_reply(self, data: Any, queue_name: str, credentials: Dict[str, str]) -> str:
        """Dağıtık yanıt gönderir (SQS benzeri)."""
        try:
            sqs_client = boto3.client(
                'sqs',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            response_id = str(uuid.uuid4())
            timestamp = time.time()
            response = Response(response_id, data, timestamp)
            serialized = response.format("json")
            sqs_client.send_message(QueueUrl=queue_name, MessageBody=base64.b64encode(serialized).decode('utf-8'))
            self.responses[response_id] = response
            self.temporal_graph.add_response(response_id, timestamp)
            log.debug(f"Dağıtık yanıt gönderildi: response_id={response_id}, queue={queue_name}")
            return response_id
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed reply hatası: {str(e)}")
            raise PdsXException(f"Distributed reply hatası: {str(e)}")

    async def send_websocket_reply(self, data: Any, ws_url: str) -> str:
        """WebSocket üzerinden yanıt gönderir."""
        try:
            response_id = str(uuid.uuid4())
            timestamp = time.time()
            response = Response(response_id, data, timestamp)
            serialized = response.format("json")
            async with websockets.connect(ws_url) as ws:
                await ws.send(serialized)
            self.responses[response_id] = response
            self.temporal_graph.add_response(response_id, timestamp)
            log.debug(f"WebSocket yanıtı gönderildi: response_id={response_id}, ws_url={ws_url}")
            return response_id
        except Exception as e:
            log.error(f"WebSocket reply hatası: {str(e)}")
            raise PdsXException(f"WebSocket reply hatası: {str(e)}")

    def encrypt_reply(self, response: Response, key: bytes, method: str = "aes") -> bytes:
        """Yanıtı şifreler."""
        try:
            method = method.lower()
            serialized = response.format(response.metadata["format"])
            if method == "aes":
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(serialized)
                response.metadata["encrypted"] = True
                return cipher.nonce + tag + ciphertext
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Encrypt reply hatası: {str(e)}")
            raise PdsXException(f"Encrypt reply hatası: {str(e)}")

    def parse_reply_command(self, command: str) -> None:
        """Yanıt uzantısı komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("REPLY SEND "):
                match = re.match(r"REPLY SEND\s+(.+?)\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    response_id = self.send_reply(data, format_type)
                    self.interpreter.current_scope()[var_name] = response_id
                else:
                    raise PdsXException("REPLY SEND komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY ASYNC "):
                match = re.match(r"REPLY ASYNC\s+(.+?)\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, format_type, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format_type = format_type or "json"
                    response_id = asyncio.run(self.send_async_reply(data, format_type))
                    self.interpreter.current_scope()[var_name] = response_id
                else:
                    raise PdsXException("REPLY ASYNC komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY DISTRIBUTED "):
                match = re.match(r"REPLY DISTRIBUTED\s+(.+?)\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, queue_name, creds_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    response_id = self.send_distributed_reply(data, queue_name, credentials)
                    self.interpreter.current_scope()[var_name] = response_id
                else:
                    raise PdsXException("REPLY DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY WEBSOCKET "):
                match = re.match(r"REPLY WEBSOCKET\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, ws_url, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    response_id = asyncio.run(self.send_websocket_reply(data, ws_url))
                    self.interpreter.current_scope()[var_name] = response_id
                else:
                    raise PdsXException("REPLY WEBSOCKET komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY ENCRYPT "):
                match = re.match(r"REPLY ENCRYPT\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_id, key_str, method, var_name = match.groups()
                    if response_id not in self.responses:
                        raise PdsXException(f"Yanıt bulunamadı: {response_id}")
                    key = base64.b64decode(key_str)
                    encrypted = self.encrypt_reply(self.responses[response_id], key, method)
                    self.interpreter.current_scope()[var_name] = encrypted
                else:
                    raise PdsXException("REPLY ENCRYPT komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY ANALYZE "):
                match = re.match(r"REPLY ANALYZE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = {
                        "total_responses": len(self.responses),
                        "clusters": self.temporal_graph.analyze(),
                        "anomalies": [rid for rid, r in self.responses.items() if self.response_shield.predict(len(str(r.data)), 0.1)]
                    }
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("REPLY ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY VISUALIZE "):
                match = re.match(r"REPLY VISUALIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    output_path, format = match.groups()
                    format = format or "png"
                    dot = graphviz.Digraph(format=format)
                    for rid, response in self.responses.items():
                        node_label = f"ID: {rid}\nTime: {response.timestamp}\nFormat: {response.metadata['format']}"
                        dot.node(rid, node_label, color="red" if response.metadata["encrypted"] else "green")
                    for rid1 in self.temporal_graph.edges:
                        for rid2, weight in self.temporal_graph.edges[rid1]:
                            dot.edge(rid1, rid2, label=str(weight))
                    dot.render(output_path, cleanup=True)
                    log.debug(f"Yanıtlar görselleştirildi: path={output_path}.{format}")
                else:
                    raise PdsXException("REPLY VISUALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY QUANTUM "):
                match = re.match(r"REPLY QUANTUM\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_id1, response_id2, var_name = match.groups()
                    if response_id1 not in self.responses or response_id2 not in self.responses:
                        raise PdsXException(f"Yanıt bulunamadı: {response_id1} veya {response_id2}")
                    correlation_id = self.quantum_correlator.correlate(self.responses[response_id1], self.responses[response_id2])
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("REPLY QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY HOLO "):
                match = re.match(r"REPLY HOLO\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_id, var_name = match.groups()
                    if response_id not in self.responses:
                        raise PdsXException(f"Yanıt bulunamadı: {response_id}")
                    pattern = self.holo_response.compress(self.responses[response_id])
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("REPLY HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY SMART "):
                match = re.match(r"REPLY SMART\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_id, var_name = match.groups()
                    if response_id not in self.responses:
                        raise PdsXException(f"Yanıt bulunamadı: {response_id}")
                    route = self.smart_router.route(self.responses[response_id], 0.1)
                    self.interpreter.current_scope()[var_name] = route
                else:
                    raise PdsXException("REPLY SMART komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY TEMPORAL "):
                match = re.match(r"REPLY TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_id1, response_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(response_id1, response_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("REPLY TEMPORAL komutunda sözdizimi hatası")
            elif command_upper.startswith("REPLY PREDICT "):
                match = re.match(r"REPLY PREDICT\s+(\d+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    response_size, exec_time, var_name = match.groups()
                    response_size = int(response_size)
                    exec_time = float(exec_time)
                    is_anomaly = self.response_shield.predict(response_size, exec_time)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("REPLY PREDICT komutunda sözdizimi hatası")
            # ÇOK SATIRLI PROGRAM KOMUTLARI
            elif command_upper.startswith("PROGRAM "):
                # Program yazma modunu başlat
                if self.program_manager.start_program(command):
                    return  # Program moduna geç
                else:
                    raise PdsXException("Program başlatma hatası")
            elif command_upper == "END PROGRAM":
                # Program yazma modunu bitir
                if self.program_manager.end_program():
                    return
                else:
                    raise PdsXException("Program bitirme hatası")
            elif command_upper.startswith("RUN PROGRAM "):
                # Program çalıştır
                match = re.match(r"RUN PROGRAM\s+([a-zA-Z_][a-zA-Z0-9_]*)", command, re.IGNORECASE)
                if match:
                    program_name = match.group(1)
                    if self.program_manager.run_program(program_name):
                        print(f"[PDS-X] ✅ Program çalıştırıldı: {program_name}")
                    else:
                        raise PdsXException(f"Program çalıştırma hatası: {program_name}")
                else:
                    raise PdsXException("RUN PROGRAM komutunda sözdizimi hatası")
            elif command_upper == "LIST":
                # Tüm programları listele
                self.program_manager.list_programs()
            elif command_upper.startswith("LIST "):
                # Belirli program listele/göster
                match = re.match(r"LIST\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\.[a-zA-Z]+)?", command, re.IGNORECASE)
                if match:
                    program_name = match.group(1)
                    extension = match.group(2)
                    if extension:
                        # Program içeriğini göster
                        self.program_manager.show_program(program_name, extension)
                    else:
                        # Program içeriğini göster (uzantısız)
                        self.program_manager.show_program(program_name)
                else:
                    # Uzantıya göre filtrele
                    ext_match = re.match(r"LIST\s+(\.[a-zA-Z]+)", command, re.IGNORECASE)
                    if ext_match:
                        extension = ext_match.group(1)
                        self.program_manager.list_programs(extension)
                    else:
                        raise PdsXException("LIST komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen yanıt komutu: {command}")
        except Exception as e:
            log.error(f"Yanıt komut hatası: {str(e)}")
            raise PdsXException(f"Yanıt komut hatası: {str(e)}")
    
    def handle_program_input(self, line: str) -> bool:
        """
        Program yazma modundayken girdiyi işle
        Returns: True if still in program mode, False if exited
        """
        try:
            if self.program_manager.in_program_mode:
                return self.program_manager.add_program_line(line)
            return False
        except Exception as e:
            print(f"[PDS-X] ❌ Program girdi hatası: {e}")
            return False


# === ÇOK SATIRLI PROGRAM SİSTEMİ ===
class MultiLineProgramManager:
    """
    Çok satırlı program yazma ve yönetme sistemi
    
    Desteklenen format:
    PROGRAM <program_name>.<extension>
    <kod satırları>
    END PROGRAM
    
    Desteklenen uzantılar:
    - .basX  : PDS-X BASIC (varsayılan)
    - .libx  : LibX library pseudocode  
    - .pdsx  : PDS-X komutları ve makrolar
    - .py    : Python
    - .js    : JavaScript
    - .sql   : SQL queries
    """
    
    def __init__(self, interpreter=None):
        self.interpreter = interpreter
        self.programs = {}  # Program ismi -> {'code': str, 'extension': str, 'metadata': dict}
        self.programs_dir = Path("programs")
        self.programs_dir.mkdir(exist_ok=True)
        
        # Desteklenen uzantılar ve özellikleri
        self.supported_extensions = {
            '.basx': {'name': 'PDS-X BASIC', 'executable': True, 'encrypt': True, 'compress': True},
            '.libx': {'name': 'LibX Library', 'executable': True, 'encrypt': True, 'compress': True},
            '.pdsx': {'name': 'PDS-X Commands', 'executable': True, 'encrypt': True, 'compress': True},
            '.py': {'name': 'Python', 'executable': True, 'encrypt': False, 'compress': False},
            '.js': {'name': 'JavaScript', 'executable': False, 'encrypt': False, 'compress': False},
            '.sql': {'name': 'SQL Queries', 'executable': False, 'encrypt': False, 'compress': False},
            '.txt': {'name': 'Plain Text', 'executable': False, 'encrypt': False, 'compress': False}
        }
        
        self.current_program = None
        self.current_code_lines = []
        self.in_program_mode = False
        
        # Şifreleme ve sıkıştırma için
        self.compression_enabled = True
        self.encryption_enabled = True
        self.encryption_key = None
        
    def add_line(self, line: str) -> bool:
        """Program modunda satır ekle"""
        if not self.in_program_mode:
            return False
        self.current_code_lines.append(line)
        return True
        
    def end_program(self) -> bool:
        """Program yazma modunu sonlandır ve kaydet"""
        if not self.in_program_mode or self.current_program is None:
            return False
        
        # Program kodunu birleştir
        code = '\n'.join(self.current_code_lines)
        
        # Program bilgilerini kaydet
        program_key = self.current_program['name']
        if program_key not in self.programs:
            self.programs[program_key] = {}
        
        self.programs[program_key][self.current_program['extension']] = {
            'code': code,
            'metadata': {
                'created': time.time(),
                'lines': len(self.current_code_lines),
                'size': len(code)
            }
        }
        
        # Dosyaya kaydet
        try:
            self._save_program_to_file(program_key, self.current_program['extension'], code)
        except Exception as e:
            print(f"[PDS-X] ⚠️ Dosya kaydetme hatası: {e}")
        
        print(f"[PDS-X] ✅ Program kaydedildi: {self.current_program['full_name']}")
        print(f"[PDS-X] 📊 {len(self.current_code_lines)} satır, {len(code)} karakter")
        
        # Modu sıfırla
        self.in_program_mode = False
        self.current_program = None
        self.current_code_lines = []
        
        return True
        
    def _get_all_programs(self) -> dict:
        """Tüm programları döndür"""
        return self.programs.copy()
        
    def _get_program_content(self, name: str, extension: str) -> str:
        """Program içeriğini döndür"""
        if name in self.programs and extension in self.programs[name]:
            return self.programs[name][extension]['code']
        return ""
        
    def _compress_content(self, content: str) -> bytes:
        """İçeriği sıkıştır"""
        import gzip
        return gzip.compress(content.encode('utf-8'))
        
    def _decompress_content(self, compressed: bytes) -> str:
        """Sıkıştırılmış içeriği aç"""
        import gzip
        return gzip.decompress(compressed).decode('utf-8')
        
    def _encrypt_content(self, content: str, password: str) -> bytes:
        """İçeriği şifrele (basit XOR)"""
        content_bytes = content.encode('utf-8')
        password_bytes = password.encode('utf-8')
        result = bytearray()
        for i, byte in enumerate(content_bytes):
            result.append(byte ^ password_bytes[i % len(password_bytes)])
        return bytes(result)
        
    def _decrypt_content(self, encrypted: bytes, password: str) -> str:
        """Şifrelenmiş içeriği çöz"""
        password_bytes = password.encode('utf-8')
        result = bytearray()
        for i, byte in enumerate(encrypted):
            result.append(byte ^ password_bytes[i % len(password_bytes)])
        return result.decode('utf-8')
        
    def _save_program_to_file(self, name: str, extension: str, code: str):
        """Programı dosyaya kaydet"""
        filename = f"{name}{extension}"
        filepath = self.programs_dir / filename
        
        # Uzantı özelliklerini al
        ext_props = self.supported_extensions.get(extension, {})
        
        content = code
        
        # Sıkıştırma uygula
        if ext_props.get('compress', False) and self.compression_enabled:
            content = self._compress_content(content)
            filepath = filepath.with_suffix(filepath.suffix + '.gz')
        
        # Şifreleme uygula (henüz basit versiyonu)
        if ext_props.get('encrypt', False) and self.encryption_enabled:
            if isinstance(content, str):
                content = content.encode('utf-8')
            # Basit şifreleme: dosya adının hash'ini anahtar olarak kullan
            import hashlib
            key = hashlib.md5(name.encode()).hexdigest()[:16]
            content = self._encrypt_content(content.decode('utf-8') if isinstance(content, bytes) else content, key)
            filepath = filepath.with_suffix(filepath.suffix + '.enc')
        
        # Dosyaya yaz
        mode = 'wb' if isinstance(content, bytes) else 'w'
        encoding = None if isinstance(content, bytes) else 'utf-8'
        
        with open(filepath, mode, encoding=encoding) as f:
            f.write(content)
        
    def start_program(self, program_declaration: str) -> bool:
        """
        Program yazma modunu başlat
        
        Format: PROGRAM <name>.<extension>
        Örnek: PROGRAM hello.basX
               PROGRAM mylib.libx
               PROGRAM scripts.pdsx
        """
        try:
            # Program deklarasyonunu parse et
            match = re.match(r'PROGRAM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z]+)?)', program_declaration.strip())
            if not match:
                print("[PDS-X] ❌ Geçersiz program deklarasyonu. Format: PROGRAM <name>.<extension>")
                return False
            
            program_name_with_ext = match.group(1)
            
            # Uzantı kontrolü
            if '.' in program_name_with_ext:
                program_name, extension = program_name_with_ext.rsplit('.', 1)
                extension = '.' + extension.lower()
            else:
                program_name = program_name_with_ext
                extension = '.basx'  # Varsayılan uzantı
            
            # Desteklenen uzantı kontrolü
            if extension not in self.supported_extensions:
                print(f"[PDS-X] ⚠️ Desteklenmeyen uzantı: {extension}")
                print(f"[PDS-X] 📋 Desteklenen uzantılar: {', '.join(self.supported_extensions.keys())}")
                return False
            
            # Program modu başlat
            self.current_program = {
                'name': program_name,
                'extension': extension,
                'full_name': f"{program_name}{extension}"
            }
            self.current_code_lines = []
            self.in_program_mode = True
            
            ext_info = self.supported_extensions[extension]
            print(f"[PDS-X] 📝 Program yazma modu başlatıldı: {self.current_program['full_name']}")
            print(f"[PDS-X] 🔤 Dil: {ext_info['name']}")
            print(f"[PDS-X] 💾 Şifreleme: {'✅' if ext_info['encrypt'] else '❌'}")
            print(f"[PDS-X] 🗜️ Sıkıştırma: {'✅' if ext_info['compress'] else '❌'}")
            print(f"[PDS-X] ⚡ Çalıştırılabilir: {'✅' if ext_info['executable'] else '❌'}")
            print(f"[PDS-X] 📋 'END PROGRAM' yazarak tamamlayın")
            
            return True
            
        except Exception as e:
            print(f"[PDS-X] ❌ Program başlatma hatası: {e}")
            return False
    
    def add_program_line(self, line: str) -> bool:
        """Program modundayken kod satırı ekle"""
        if not self.in_program_mode:
            return False
        
        # END PROGRAM kontrolü
        if line.strip().upper() == 'END PROGRAM':
            return self.end_program()
        
        # Normal kod satırını ekle
        self.current_code_lines.append(line)
        print(f"[{len(self.current_code_lines):3d}] {line}")
        return True
        try:
            if not self.encryption_key:
                self.encryption_key = get_random_bytes(32)  # AES-256 key
            
            cipher = AES.new(self.encryption_key, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(code.encode('utf-8'))
            
            # Nonce + tag + ciphertext'i base64'e çevir
            encrypted_data = cipher.nonce + tag + ciphertext
            return base64.b64encode(encrypted_data).decode('ascii')
        except Exception:
            return code
    
    def _decrypt_code(self, encrypted_code: str) -> str:
        """Şifreli kodu çöz"""
        try:
            if not self.encryption_key:
                return encrypted_code
            
            encrypted_data = base64.b64decode(encrypted_code.encode('ascii'))
            nonce = encrypted_data[:16]  # AES-GCM nonce
            tag = encrypted_data[16:32]   # AES-GCM tag
            ciphertext = encrypted_data[32:]
            
            cipher = AES.new(self.encryption_key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext.decode('utf-8')
        except Exception:
            return encrypted_code
    
    def _decompress_code(self, compressed_code: str) -> str:
        """Sıkıştırılmış kodu aç"""
        try:
            import gzip
            compressed_data = base64.b64decode(compressed_code.encode('ascii'))
            decompressed = gzip.decompress(compressed_data)
            return decompressed.decode('utf-8')
        except Exception:
            return compressed_code
    
    def _save_program_to_file(self, program_name: str, file_path: Path):
        """Programı dosyaya kaydet"""
        try:
            program_data = self.programs[program_name]
            
            # JSON formatında kaydet
            save_data = {
                'name': program_name,
                'code': program_data['code'],
                'extension': program_data['extension'],
                'metadata': program_data['metadata'],
                'pdsx_program_format': '1.0'
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[PDS-X] ⚠️ Dosya kaydetme hatası: {e}")
    
    def run_program(self, program_name: str) -> bool:
        """Programı çalıştır"""
        try:
            if program_name not in self.programs:
                # Dosyadan yüklemeyi dene
                if not self._load_program_from_file(program_name):
                    print(f"[PDS-X] ❌ Program bulunamadı: {program_name}")
                    return False
            
            program_data = self.programs[program_name]
            
            if not program_data['metadata']['executable']:
                print(f"[PDS-X] ⚠️ Program çalıştırılabilir değil: {program_name}")
                return False
            
            # Kodu çöz ve aç
            code = program_data['code']
            metadata = program_data['metadata']
            
            if metadata.get('encrypted', False):
                code = self._decrypt_code(code)
                print("[PDS-X] 🔓 Program şifre çözüldü")
            
            if metadata.get('compressed', False):
                code = self._decompress_code(code)
                print("[PDS-X] 📦 Program açıldı")
            
            print(f"[PDS-X] ⚡ Program çalıştırılıyor: {program_name}{program_data['extension']}")
            print(f"[PDS-X] 📊 {metadata['line_count']} satır")
            
            # Uzantıya göre çalıştır
            extension = program_data['extension']
            
            if extension in ['.basx', '.pdsx']:
                # PDS-X komutları olarak çalıştır
                return self._execute_pdsx_code(code)
            elif extension == '.libx':
                # LibX library olarak çalıştır
                return self._execute_libx_code(code)
            elif extension == '.py':
                # Python olarak çalıştır
                return self._execute_python_code(code)
            else:
                print(f"[PDS-X] ⚠️ Çalıştırma desteklenmiyor: {extension}")
                return False
                
        except Exception as e:
            print(f"[PDS-X] ❌ Program çalıştırma hatası: {e}")
            return False
    
    def _execute_pdsx_code(self, code: str) -> bool:
        """PDS-X kodunu çalıştır"""
        try:
            if not self.interpreter:
                print("[PDS-X] ⚠️ Interpreter mevcut değil")
                return False
            
            lines = code.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        # PDS-X komutunu çalıştır
                        self.interpreter.execute_command(line)
                    except Exception as e:
                        print(f"[PDS-X] ❌ Satır {line_num} hatası: {e}")
                        return False
            
            print("[PDS-X] ✅ PDS-X program tamamlandı")
            return True
            
        except Exception as e:
            print(f"[PDS-X] ❌ PDS-X çalıştırma hatası: {e}")
            return False
    
    def _execute_libx_code(self, code: str) -> bool:
        """LibX kodunu çalıştır"""
        print("[PDS-X] 🔧 LibX code execution - şu anda desteklenmiyor")
        return False
    
    def _execute_python_code(self, code: str) -> bool:
        """Python kodunu çalıştır"""
        try:
            print("[PDS-X] 🐍 Python kodu çalıştırılıyor...")
            exec(code, {'__name__': '__pdsx_program__'})
            print("[PDS-X] ✅ Python program tamamlandı")
            return True
        except Exception as e:
            print(f"[PDS-X] ❌ Python çalıştırma hatası: {e}")
            return False
    
    def list_programs(self, filter_ext: Optional[str] = None) -> None:
        """Programları listele"""
        try:
            # Dosyadan yükle
            self._load_all_programs()
            
            if not self.programs:
                print("[PDS-X] 📋 Kayıtlı program yok")
                return
            
            print(f"[PDS-X] 📋 Kayıtlı Programlar ({len(self.programs)} adet):")
            print("[PDS-X] " + "="*60)
            
            for name, data in self.programs.items():
                metadata = data['metadata']
                extension = data['extension']
                
                # Filtre uygula
                if filter_ext and extension != filter_ext:
                    continue
                
                full_name = f"{name}{extension}"
                lang = metadata['language']
                lines = metadata['line_count']
                executable = "⚡" if metadata['executable'] else "📄"
                encrypted = "🔒" if metadata.get('encrypted', False) else ""
                compressed = "🗜️" if metadata.get('compressed', False) else ""
                
                print(f"[PDS-X] {executable} {full_name:<20} | {lang:<15} | {lines:3d} satır {encrypted}{compressed}")
            
            print("[PDS-X] " + "="*60)
            print("[PDS-X] 💡 Kullanım: RUN PROGRAM <name> veya LIST <name>")
            
        except Exception as e:
            print(f"[PDS-X] ❌ Program listeleme hatası: {e}")
    
    def show_program(self, program_name: str, extension: Optional[str] = None) -> None:
        """Program içeriğini göster"""
        try:
            # Program adını normalize et
            if extension:
                full_key = program_name
                if extension not in program_name:
                    full_key = f"{program_name}{extension}"
            else:
                full_key = program_name
            
            # Programı bul
            found_program = None
            found_key = None
            
            for key, data in self.programs.items():
                if key == program_name or key == full_key:
                    found_program = data
                    found_key = key
                    break
                elif f"{key}{data['extension']}" == full_key:
                    found_program = data
                    found_key = key
                    break
            
            if not found_program:
                # Dosyadan yüklemeyi dene
                if not self._load_program_from_file(program_name):
                    print(f"[PDS-X] ❌ Program bulunamadı: {program_name}")
                    return
                found_program = self.programs[program_name]
                found_key = program_name
            
            # Kodu çöz
            if 'original_code' in found_program:
                code = found_program['original_code']
            else:
                code = found_program['code']
                metadata = found_program['metadata']
                
                if metadata.get('encrypted', False):
                    code = self._decrypt_code(code)
                
                if metadata.get('compressed', False):
                    code = self._decompress_code(code)
            
            # İçeriği göster
            metadata = found_program['metadata']
            full_name = f"{found_key}{found_program['extension']}"
            
            print(f"[PDS-X] 📄 Program: {full_name}")
            print(f"[PDS-X] 🔤 Dil: {metadata['language']}")
            print(f"[PDS-X] 📊 {metadata['line_count']} satır")
            print("[PDS-X] " + "="*60)
            
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                print(f"[{i:3d}] {line}")
            
            print("[PDS-X] " + "="*60)
            
        except Exception as e:
            print(f"[PDS-X] ❌ Program gösterme hatası: {e}")
    
    def _load_program_from_file(self, program_name: str) -> bool:
        """Programı dosyadan yükle"""
        try:
            # Olası dosya adlarını dene
            possible_files = []
            for ext in self.supported_extensions.keys():
                possible_files.append(self.programs_dir / f"{program_name}{ext}")
            
            for file_path in possible_files:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.programs[program_name] = {
                        'code': data['code'],
                        'extension': data['extension'],
                        'metadata': data['metadata']
                    }
                    return True
            
            return False
            
        except Exception as e:
            print(f"[PDS-X] ⚠️ Program yükleme hatası: {e}")
            return False
    
    def _load_all_programs(self):
        """Tüm programları dosyadan yükle"""
        try:
            for file_path in self.programs_dir.glob("*"):
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    program_name = file_path.stem
                    if program_name not in self.programs:
                        self._load_program_from_file(program_name)
        except Exception:
            pass


if __name__ == "__main__":
    print("reply_extension.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")