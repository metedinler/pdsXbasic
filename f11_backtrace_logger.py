# f11_backtrace_logger.py - PDS-X BASIC v14u Hata İzleme ve Loglama Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import traceback
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import threading
import json
import csv
import xml.etree.ElementTree as ET
import graphviz
import uuid
import numpy as np
from sklearn.ensemble import IsolationForest  # AI tabanlı anomali algılama
from pdsx_exception import PdsXException  # Hata yönetimi için
import elasticsearch
from elasticsearch import Elasticsearch

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("backtrace_logger")

class HoloTrace:
    """Holografik yığın izi sıkıştırma sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [trace]}

    def encode(self, trace: List[str]) -> str:
        """Yığın izini holografik olarak sıkıştırır."""
        try:
            trace_str = json.dumps(trace)
            pattern = hashlib.sha256(trace_str.encode()).hexdigest()[:16]
            self.storage[pattern].append(trace)
            log.debug(f"Holografik iz kodlandı: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloTrace encode hatası: {str(e)}")
            raise PdsXException(f"HoloTrace encode hatası: {str(e)}")

    def decode(self, pattern: str) -> Optional[List[str]]:
        """Holografik izi geri yükler."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                return self.storage[pattern][-1]
            return None
        except Exception as e:
            log.error(f"HoloTrace decode hatası: {str(e)}")
            raise PdsXException(f"HoloTrace decode hatası: {str(e)}")

class QuantumCorrelator:
    """Kuantum tabanlı hata korelasyon simülasyonu sınıfı."""
    def __init__(self):
        self.correlations = {}  # {correlation_id: (trace1, trace2, score)}

    def correlate(self, trace1: List[str], trace2: List[str]) -> str:
        """İki yığın izini kuantum simülasyonuyla ilişkilendirir."""
        try:
            # Basit simülasyon: iz benzerliği (Jaccard)
            set1, set2 = set(trace1), set(trace2)
            score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            correlation_id = str(uuid.uuid4())
            self.correlations[correlation_id] = (trace1, trace2, score)
            log.debug(f"Kuantum korelasyon: id={correlation_id}, score={score}")
            return correlation_id
        except Exception as e:
            log.error(f"QuantumCorrelator correlate hatası: {str(e)}")
            raise PdsXException(f"QuantumCorrelator correlate hatası: {str(e)}")

    def get_correlation(self, correlation_id: str) -> Optional[Tuple[List[str], List[str], float]]:
        """Korelasyonu döndürür."""
        try:
            return self.correlations.get(correlation_id)
        except Exception as e:
            log.error(f"QuantumCorrelator get_correlation hatası: {str(e)}")
            raise PdsXException(f"QuantumCorrelator get_correlation hatası: {str(e)}")

class EvolvingLogger:
    """Kendi kendine evrilen loglama sistemi sınıfı."""
    def __init__(self):
        self.rules = {}  # {pattern: action}
        self.model = IsolationForest(contamination=0.1)  # Anomali algılama
        self.training_data = []

    def log(self, trace: List[str], context: Dict[str, Any]) -> None:
        """Yığın izini loglar ve modeli eğitir."""
        try:
            features = np.array([len(trace), context.get("error_count", 0), time.time()])
            self.training_data.append(features)
            if len(self.training_data) > 100:
                self.model.fit(np.array(self.training_data))
                anomaly_score = self.model.score_samples([features])[0]
                if anomaly_score < -0.5:  # Anomali tespit edildi
                    pattern = hashlib.sha256(json.dumps(trace).encode()).hexdigest()[:8]
                    self.rules[pattern] = "ALERT"
                    log.warning(f"Anomali tespit edildi: pattern={pattern}, score={anomaly_score}")
            log.debug("Evrilen loglama yapıldı")
        except Exception as e:
            log.error(f"EvolvingLogger log hatası: {str(e)}")
            raise PdsXException(f"EvolvingLogger log hatası: {str(e)}")

    def get_action(self, trace: List[str]) -> Optional[str]:
        """İlgili aksiyonu döndürür."""
        try:
            pattern = hashlib.sha256(json.dumps(trace).encode()).hexdigest()[:8]
            return self.rules.get(pattern)
        except Exception as e:
            log.error(f"EvolvingLogger get_action hatası: {str(e)}")
            raise PdsXException(f"EvolvingLogger get_action hatası: {str(e)}")

class TemporalGraph:
    """Zaman temelli hata grafiği sınıfı."""
    def __init__(self):
        self.vertices = {}  # {trace_id: timestamp}
        self.edges = defaultdict(list)  # {trace_id: [(related_trace_id, weight)]}

    def add_trace(self, trace_id: str, timestamp: float) -> None:
        """Hata izini grafiğe ekler."""
        try:
            self.vertices[trace_id] = timestamp
            log.debug(f"Temporal graph düğümü eklendi: trace_id={trace_id}")
        except Exception as e:
            log.error(f"TemporalGraph add_trace hatası: {str(e)}")
            raise PdsXException(f"TemporalGraph add_trace hatası: {str(e)}")

    def add_relation(self, trace_id1: str, trace_id2: str, weight: float) -> None:
        """Hata izleri arasında ilişki kurar."""
        try:
            self.edges[trace_id1].append((trace_id2, weight))
            self.edges[trace_id2].append((trace_id1, weight))
            log.debug(f"Temporal graph kenarı eklendi: {trace_id1} <-> {trace_id2}")
        except Exception as e:
            log.error(f"TemporalGraph add_relation hatası: {str(e)}")
            raise PdsXException(f"TemporalGraph add_relation hatası: {str(e)}")

    def analyze(self) -> Dict[str, List[str]]:
        """Hata grafiğini analiz eder."""
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
            log.error(f"TemporalGraph analyze hatası: {str(e)}")
            raise PdsXException(f"TemporalGraph analyze hatası: {str(e)}")

class ErrorShield:
    """Tahmini hata kalkanı sınıfı."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.history = []

    def train(self, trace: List[str], context: Dict[str, Any]) -> None:
        """Hata verileriyle modeli eğitir."""
        try:
            features = np.array([len(trace), context.get("error_count", 0), context.get("timestamp", time.time())])
            self.history.append(features)
            if len(self.history) > 50:
                self.model.fit(np.array(self.history))
                log.debug("ErrorShield modeli eğitildi")
        except Exception as e:
            log.error(f"ErrorShield train hatası: {str(e)}")
            raise PdsXException(f"ErrorShield train hatası: {str(e)}")

    def predict(self, trace: List[str], context: Dict[str, Any]) -> bool:
        """Potansiyel hatayı tahmin eder."""
        try:
            features = np.array([[len(trace), context.get("error_count", 0), context.get("timestamp", time.time())]])
            if len(self.history) < 50:
                return False
            prediction = self.model.predict(features)[0]
            is_anomaly = prediction == -1
            if is_anomaly:
                log.warning(f"Potansiyel hata tahmin edildi: trace_len={len(trace)}")
            return is_anomaly
        except Exception as e:
            log.error(f"ErrorShield predict hatası: {str(e)}")
            raise PdsXException(f"ErrorShield predict hatası: {str(e)}")

class BacktraceLogger:
    """Hata izleme ve loglama yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.logs = []  # [(timestamp, trace, context)]
        self.holo_trace = HoloTrace()
        self.quantum_correlator = QuantumCorrelator()
        self.evolving_logger = EvolvingLogger()
        self.temporal_graph = TemporalGraph()
        self.error_shield = ErrorShield()
        self.lock = threading.Lock()
        self.metadata = {"backtrace_logger": {"version": "1.0.0", "dependencies": ["graphviz", "numpy", "scikit-learn", "elasticsearch", "pdsx_exception"]}}
        self.max_log_size = 10000
        self.es_client = None  # Elasticsearch istemcisi

    def connect_elasticsearch(self, host: str, port: int) -> None:
        """Elasticsearch’e bağlanır."""
        try:
            self.es_client = Elasticsearch([{'host': host, 'port': port}])
            if not self.es_client.ping():
                raise PdsXException("Elasticsearch bağlantısı başarısız")
            log.debug(f"Elasticsearch bağlantısı kuruldu: {host}:{port}")
        except Exception as e:
            log.error(f"Elasticsearch bağlantı hatası: {str(e)}")
            raise PdsXException(f"Elasticsearch bağlantı hatası: {str(e)}")

    def log_trace(self, error: Optional[PdsXException] = None, context: Optional[Dict] = None) -> str:
        """Hata izini loglar."""
        with self.lock:
            try:
                trace = traceback.format_stack()[:-1] if not error else error.stack_trace
                context = context or {}
                context["error_count"] = len(self.logs)
                context["timestamp"] = time.time()
                trace_id = str(uuid.uuid4())
                
                # Standart loglama
                self.logs.append((context["timestamp"], trace, context))
                if len(self.logs) > self.max_log_size:
                    self.logs.pop(0)
                
                # Evolving logger ile anomali tespiti
                self.evolving_logger.log(trace, context)
                
                # Temporal graph’e ekle
                self.temporal_graph.add_trace(trace_id, context["timestamp"])
                
                # Error shield ile tahmin
                if self.error_shield.predict(trace, context):
                    context["predicted_anomaly"] = True
                
                # Elasticsearch’e loglama
                if self.es_client:
                    self.es_client.index(index="pdsx_logs", id=trace_id, body={
                        "timestamp": context["timestamp"],
                        "trace": trace,
                        "context": context
                    })
                
                log.debug(f"Hata izi loglandı: trace_id={trace_id}")
                return trace_id
            except Exception as e:
                log.error(f"Log trace hatası: {str(e)}")
                raise PdsXException(f"Log trace hatası: {str(e)}")

    def analyze_logs(self) -> Dict:
        """Logları analiz eder."""
        try:
            clusters = self.temporal_graph.analyze()
            result = {
                "total_logs": len(self.logs),
                "clusters": clusters,
                "anomalies": [log for log in self.logs if log[2].get("predicted_anomaly", False)]
            }
            log.debug(f"Log analizi tamamlandı: clusters={len(clusters)}")
            return result
        except Exception as e:
            log.error(f"Analyze logs hatası: {str(e)}")
            raise PdsXException(f"Analyze logs hatası: {str(e)}")

    def visualize_logs(self, output_path: str, format: str = "png") -> None:
        """Logları zaman çizelgesi olarak görselleştirir."""
        try:
            dot = graphviz.Digraph(format=format)
            for timestamp, trace, context in self.logs[:100]:  # İlk 100 log
                trace_id = str(uuid.uuid4())
                node_label = f"Time: {timestamp}\nError: {context.get('error', 'None')}"
                dot.node(trace_id, node_label, color="red" if context.get("predicted_anomaly", False) else "blue")
            
            dot.render(output_path, cleanup=True)
            log.debug(f"Loglar görselleştirildi: path={output_path}.{format}")
        except Exception as e:
            log.error(f"Visualize logs hatası: {str(e)}")
            raise PdsXException(f"Visualize logs hatası: {str(e)}")

    def parse_backtrace_command(self, command: str) -> None:
        """Hata izleme komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("LOG TRACE "):
                match = re.match(r"LOG TRACE\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    context_str, var_name = match.groups()
                    context = eval(context_str, self.interpreter.current_scope())
                    trace_id = self.log_trace(context=context)
                    self.interpreter.current_scope()[var_name] = trace_id
                else:
                    raise PdsXException("LOG TRACE komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG BACKTRACE "):
                match = re.match(r"LOG BACKTRACE\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    error_var, var_name = match.groups()
                    error = self.interpreter.current_scope().get(error_var)
                    if not isinstance(error, PdsXException):
                        raise PdsXException("Geçerli bir hata nesnesi gerekli")
                    trace_id = self.log_trace(error=error)
                    self.interpreter.current_scope()[var_name] = trace_id
                else:
                    raise PdsXException("LOG BACKTRACE komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG ANALYZE "):
                match = re.match(r"LOG ANALYZE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    result = self.analyze_logs()
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("LOG ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG VISUALIZE "):
                match = re.match(r"LOG VISUALIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    output_path, format = match.groups()
                    format = format or "png"
                    self.visualize_logs(output_path, format)
                else:
                    raise PdsXException("LOG VISUALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG DISTRIBUTED "):
                match = re.match(r"LOG DISTRIBUTED\s+\"([^\"]+)\"\s+(\d+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    host, port, context_str, var_name = match.groups()
                    port = int(port)
                    context = eval(context_str, self.interpreter.current_scope())
                    self.connect_elasticsearch(host, port)
                    trace_id = self.log_trace(context=context)
                    self.interpreter.current_scope()[var_name] = trace_id
                else:
                    raise PdsXException("LOG DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG HOLO "):
                match = re.match(r"LOG HOLO\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    context_str, var_name = match.groups()
                    context = eval(context_str, self.interpreter.current_scope())
                    trace = traceback.format_stack()[:-1]
                    pattern = self.holo_trace.encode(trace)
                    self.interpreter.current_scope()[var_name] = pattern
                else:
                    raise PdsXException("LOG HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG QUANTUM "):
                match = re.match(r"LOG QUANTUM\s+(.+?)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    trace1_str, trace2_str, var_name = match.groups()
                    trace1 = eval(trace1_str, self.interpreter.current_scope())
                    trace2 = eval(trace2_str, self.interpreter.current_scope())
                    correlation_id = self.quantum_correlator.correlate(trace1, trace2)
                    self.interpreter.current_scope()[var_name] = correlation_id
                else:
                    raise PdsXException("LOG QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG EVOLVE "):
                match = re.match(r"LOG EVOLVE\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    context_str, var_name = match.groups()
                    context = eval(context_str, self.interpreter.current_scope())
                    trace = traceback.format_stack()[:-1]
                    self.evolving_logger.log(trace, context)
                    action = self.evolving_logger.get_action(trace)
                    self.interpreter.current_scope()[var_name] = action or "NONE"
                else:
                    raise PdsXException("LOG EVOLVE komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG TEMPORAL "):
                match = re.match(r"LOG TEMPORAL\s+(\w+)\s+(\w+)\s+(\d*\.?\d*)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    trace_id1, trace_id2, weight, var_name = match.groups()
                    weight = float(weight)
                    self.temporal_graph.add_relation(trace_id1, trace_id2, weight)
                    self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("LOG TEMPORAL komutunda sözdizimi hatası")
            elif command_upper.startswith("LOG PREDICT "):
                match = re.match(r"LOG PREDICT\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    context_str, var_name = match.groups()
                    context = eval(context_str, self.interpreter.current_scope())
                    trace = traceback.format_stack()[:-1]
                    is_anomaly = self.error_shield.predict(trace, context)
                    self.interpreter.current_scope()[var_name] = is_anomaly
                else:
                    raise PdsXException("LOG PREDICT komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen hata izleme komutu: {command}")
        except Exception as e:
            log.error(f"Hata izleme komut hatası: {str(e)}")
            raise PdsXException(f"Hata izleme komut hatası: {str(e)}")

if __name__ == "__main__":
    print("f11_backtrace_logger.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")