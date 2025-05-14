# graph.py - PDS-X BASIC v14u Grafik Veri Yapısı Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
import threading
import heapq
import graphviz
import uuid
import numpy as np
from pdsx_exception import PdsXException  # Hata yönetimi için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("graph")

class GraphVertex:
    """Grafik düğümü sınıfı."""
    def __init__(self, value: Any, vertex_id: Optional[str] = None):
        self.value = value
        self.vertex_id = vertex_id or str(uuid.uuid4())
        self.metadata = {}  # Ek özellikler için (örneğin, renk, ağırlık)

    def __str__(self) -> str:
        return f"Vertex({self.value}, id={self.vertex_id})"

class GraphEdge:
    """Grafik kenarı sınıfı."""
    def __init__(self, source_id: str, target_id: str, weight: float = 1.0, directed: bool = False):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.directed = directed
        self.metadata = {}  # Ek özellikler için

    def __str__(self) -> str:
        return f"Edge({self.source_id} -> {self.target_id}, weight={self.weight}, directed={self.directed})"

class Graph:
    """Grafik sınıfı."""
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.vertices = {}  # {vertex_id: GraphVertex}
        self.edges = defaultdict(list)  # {source_id: [(target_id, weight)]}
        self.vertex_count = 0
        self.edge_count = 0
        self.lock = threading.Lock()

    def create(self, value: Any, vertex_id: Optional[str] = None) -> str:
        """Grafik oluşturur ve ilk düğümü ekler."""
        with self.lock:
            if self.vertices:
                raise PdsXException("Grafik zaten mevcut")
            try:
                vertex = GraphVertex(value, vertex_id)
                self.vertices[vertex.vertex_id] = vertex
                self.vertex_count = 1
                log.debug(f"Grafik oluşturuldu: vertex_id={vertex.vertex_id}, directed={self.directed}")
                return vertex.vertex_id
            except Exception as e:
                log.error(f"Grafik oluşturma hatası: {str(e)}")
                raise PdsXException(f"Grafik oluşturma hatası: {str(e)}")

    def add_vertex(self, value: Any, vertex_id: Optional[str] = None) -> str:
        """Grafiğe yeni bir düğüm ekler."""
        with self.lock:
            try:
                vertex = GraphVertex(value, vertex_id)
                if vertex.vertex_id in self.vertices:
                    raise PdsXException(f"Düğüm zaten mevcut: {vertex.vertex_id}")
                self.vertices[vertex.vertex_id] = vertex
                self.vertex_count += 1
                log.debug(f"Düğüm eklendi: vertex_id={vertex.vertex_id}")
                return vertex.vertex_id
            except Exception as e:
                log.error(f"Düğüm ekleme hatası: {str(e)}")
                raise PdsXException(f"Düğüm ekleme hatası: {str(e)}")

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0) -> None:
        """Grafiğe yeni bir kenar ekler."""
        with self.lock:
            if source_id not in self.vertices or target_id not in self.vertices:
                raise PdsXException(f"Düğümlerden biri bulunamadı: source={source_id}, target={target_id}")
            try:
                self.edges[source_id].append((target_id, weight))
                if not self.directed:
                    self.edges[target_id].append((source_id, weight))
                self.edge_count += 1
                log.debug(f"Kenar eklendi: source={source_id}, target={target_id}, weight={weight}")
            except Exception as e:
                log.error(f"Kenar ekleme hatası: {str(e)}")
                raise PdsXException(f"Kenar ekleme hatası: {str(e)}")

    def remove_vertex(self, vertex_id: str) -> None:
        """Grafikten düğüm kaldırır."""
        with self.lock:
            if vertex_id not in self.vertices:
                raise PdsXException(f"Düğüm bulunamadı: {vertex_id}")
            try:
                # Bağlantılı kenarları kaldır
                for target_id, _ in self.edges[vertex_id]:
                    if not self.directed:
                        self.edges[target_id] = [(sid, w) for sid, w in self.edges[target_id] if sid != vertex_id]
                    self.edge_count -= 1
                self.edges.pop(vertex_id, None)
                
                # Diğer düğümlerden bu düğüme olan kenarları kaldır
                for source_id in self.edges:
                    self.edges[source_id] = [(tid, w) for tid, w in self.edges[source_id] if tid != vertex_id]
                    self.edge_count -= sum(1 for tid, _ in self.edges[source_id] if tid == vertex_id)
                
                # Düğümü kaldır
                self.vertices.pop(vertex_id)
                self.vertex_count -= 1
                log.debug(f"Düğüm kaldırıldı: vertex_id={vertex_id}")
            except Exception as e:
                log.error(f"Düğüm kaldırma hatası: {str(e)}")
                raise PdsXException(f"Düğüm kaldırma hatası: {str(e)}")

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Grafikten kenar kaldırır."""
        with self.lock:
            if source_id not in self.vertices or target_id not in self.vertices:
                raise PdsXException(f"Düğümlerden biri bulunamadı: source={source_id}, target={target_id}")
            try:
                self.edges[source_id] = [(tid, w) for tid, w in self.edges[source_id] if tid != target_id]
                if not self.directed:
                    self.edges[target_id] = [(sid, w) for sid, w in self.edges[target_id] if sid != source_id]
                self.edge_count -= 1
                log.debug(f"Kenar kaldırıldı: source={source_id}, target={target_id}")
            except Exception as e:
                log.error(f"Kenar kaldırma hatası: {str(e)}")
                raise PdsXException(f"Kenar kaldırma hatası: {str(e)}")

    def traverse(self, mode: str = "BFS", start_id: Optional[str] = None) -> List[Dict]:
        """Grafiği dolaşır (BFS veya DFS)."""
        with self.lock:
            result = []
            start_id = start_id or next(iter(self.vertices), None)
            if not start_id or start_id not in self.vertices:
                raise PdsXException(f"Başlangıç düğümü bulunamadı: {start_id}")
            
            try:
                visited = set()
                mode = mode.upper()
                
                if mode == "BFS":
                    queue = deque([start_id])
                    while queue:
                        vertex_id = queue.popleft()
                        if vertex_id not in visited:
                            visited.add(vertex_id)
                            vertex = self.vertices[vertex_id]
                            result.append({"id": vertex_id, "value": vertex.value, "metadata": vertex.metadata})
                            for target_id, _ in self.edges[vertex_id]:
                                if target_id not in visited:
                                    queue.append(target_id)
                
                elif mode == "DFS":
                    stack = [start_id]
                    while stack:
                        vertex_id = stack.pop()
                        if vertex_id not in visited:
                            visited.add(vertex_id)
                            vertex = self.vertices[vertex_id]
                            result.append({"id": vertex_id, "value": vertex.value, "metadata": vertex.metadata})
                            for target_id, _ in reversed(self.edges[vertex_id]):
                                if target_id not in visited:
                                    stack.append(target_id)
                
                else:
                    raise PdsXException(f"Desteklenmeyen dolaşım modu: {mode}")
                
                log.debug(f"Grafik dolaşıldı: mode={mode}, vertices={len(result)}")
                return result
            except Exception as e:
                log.error(f"Grafik dolaşım hatası: {str(e)}")
                raise PdsXException(f"Grafik dolaşım hatası: {str(e)}")

    def shortest_path(self, start_id: str, end_id: str) -> Dict:
        """Dijkstra algoritması ile en kısa yolu bulur."""
        with self.lock:
            if start_id not in self.vertices or end_id not in self.vertices:
                raise PdsXException(f"Düğümlerden biri bulunamadı: start={start_id}, end={end_id}")
            
            try:
                distances = {vid: float('inf') for vid in self.vertices}
                distances[start_id] = 0
                predecessors = {vid: None for vid in self.vertices}
                pq = [(0, start_id)]
                visited = set()
                
                while pq:
                    dist, current = heapq.heappop(pq)
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    if current == end_id:
                        break
                    
                    for neighbor_id, weight in self.edges[current]:
                        if neighbor_id in visited:
                            continue
                        new_dist = dist + weight
                        if new_dist < distances[neighbor_id]:
                            distances[neighbor_id] = new_dist
                            predecessors[neighbor_id] = current
                            heapq.heappush(pq, (new_dist, neighbor_id))
                
                # Yolu oluştur
                path = []
                current = end_id
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
                
                result = {
                    "path": path,
                    "length": distances[end_id] if distances[end_id] != float('inf') else -1,
                    "nodes": [self.vertices[vid].value for vid in path]
                }
                log.debug(f"En kısa yol bulundu: start={start_id}, end={end_id}, length={result['length']}")
                return result
            except Exception as e:
                log.error(f"En kısa yol bulma hatası: {str(e)}")
                raise PdsXException(f"En kısa yol bulma hatası: {str(e)}")

    def minimum_spanning_tree(self) -> List[Dict]:
        """Kruskal algoritması ile minimum yayılma ağacını bulur."""
        with self.lock:
            try:
                # Kenarları topla
                edges = []
                for source_id in self.edges:
                    for target_id, weight in self.edges[source_id]:
                        if self.directed or source_id < target_id:  # Yönsüz grafiklerde çift kenarları önle
                            edges.append((weight, source_id, target_id))
                edges.sort()  # Ağırlığa göre sırala
                
                # Birleşik-bul (union-find) veri yapısı
                parent = {vid: vid for vid in self.vertices}
                rank = {vid: 0 for vid in self.vertices}
                
                def find(vid: str) -> str:
                    if parent[vid] != vid:
                        parent[vid] = find(parent[vid])
                    return parent[vid]
                
                def union(vid1: str, vid2: str) -> None:
                    root1, root2 = find(vid1), find(vid2)
                    if root1 != root2:
                        if rank[root1] < rank[root2]:
                            parent[root1] = root2
                        elif rank[root1] > rank[root2]:
                            parent[root2] = root1
                        else:
                            parent[root2] = root1
                            rank[root1] += 1
                
                # Kruskal algoritması
                mst = []
                for weight, source_id, target_id in edges:
                    if find(source_id) != find(target_id):
                        union(source_id, target_id)
                        mst.append({
                            "source": source_id,
                            "target": target_id,
                            "weight": weight,
                            "source_value": self.vertices[source_id].value,
                            "target_value": self.vertices[target_id].value
                        })
                
                log.debug(f"Minimum yayılma ağacı bulundu: edges={len(mst)}")
                return mst
            except Exception as e:
                log.error(f"Minimum yayılma ağacı hesaplama hatası: {str(e)}")
                raise PdsXException(f"Minimum yayılma ağacı hesaplama hatası: {str(e)}")

    def topological_sort(self) -> List[str]:
        """Topolojik sıralama yapar (yönlü grafiklerde)."""
        with self.lock:
            if not self.directed:
                raise PdsXException("Topolojik sıralama yalnızca yönlü grafiklerde desteklenir")
            
            try:
                visited = set()
                stack = []
                
                def dfs(vertex_id: str):
                    visited.add(vertex_id)
                    for neighbor_id, _ in self.edges[vertex_id]:
                        if neighbor_id not in visited:
                            dfs(neighbor_id)
                    stack.append(vertex_id)
                
                for vertex_id in self.vertices:
                    if vertex_id not in visited:
                        dfs(vertex_id)
                
                result = stack[::-1]
                log.debug(f"Topolojik sıralama tamamlandı: vertices={len(result)}")
                return result
            except Exception as e:
                log.error(f"Topolojik sıralama hatası: {str(e)}")
                raise PdsXException(f"Topolojik sıralama hatası: {str(e)}")

    def visualize(self, output_path: str, format: str = "png") -> None:
        """Grafiği görselleştirir."""
        with self.lock:
            try:
                dot = graphviz.Digraph(format=format) if self.directed else graphviz.Graph(format=format)
                for vertex_id, vertex in self.vertices.items():
                    node_label = f"{vertex.value}\nID: {vertex_id}"
                    dot.node(vertex_id, node_label)
                
                seen_edges = set()
                for source_id in self.edges:
                    for target_id, weight in self.edges[source_id]:
                        edge_key = (min(source_id, target_id), max(source_id, target_id))
                        if self.directed or edge_key not in seen_edges:
                            dot.edge(source_id, target_id, label=str(weight))
                            if not self.directed:
                                seen_edges.add(edge_key)
                
                dot.render(output_path, cleanup=True)
                log.debug(f"Grafik görselleştirildi: path={output_path}.{format}")
            except Exception as e:
                log.error(f"Grafik görselleştirme hatası: {str(e)}")
                raise PdsXException(f"Grafik görselleştirme hatası: {str(e)}")

class GraphManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.graphs = {}  # {graph_id: Graph}
        self.lock = threading.Lock()
        self.metadata = {"graph": {"version": "1.0.0", "dependencies": ["graphviz", "numpy", "pdsx_exception"]}}

    def parse_graph_command(self, command: str) -> None:
        """Grafik komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("GRAPH CREATE "):
                match = re.match(r"GRAPH CREATE\s+(\w+)\s+AS\s+(DIRECTED|UNDIRECTED)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, directed, value, var_name = match.groups()
                    graph = Graph(directed=directed.upper() == "DIRECTED")
                    vertex_id = graph.create(self.interpreter.evaluate_expression(value))
                    with self.lock:
                        self.graphs[graph_id] = graph
                    self.interpreter.current_scope()[var_name] = vertex_id
                    self.interpreter.current_scope()[f"{graph_id}_GRAPH"] = graph_id
                    log.debug(f"Grafik oluşturuldu: graph_id={graph_id}, directed={directed}")
                else:
                    raise PdsXException("GRAPH CREATE komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH ADD VERTEX "):
                match = re.match(r"GRAPH ADD VERTEX\s+(\w+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, value, var_name = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    vertex_id = graph.add_vertex(self.interpreter.evaluate_expression(value))
                    self.interpreter.current_scope()[var_name] = vertex_id
                else:
                    raise PdsXException("GRAPH ADD VERTEX komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH ADD EDGE "):
                match = re.match(r"GRAPH ADD EDGE\s+(\w+)\s+(\w+)\s+(\w+)\s*(\d*\.?\d*)?", command, re.IGNORECASE)
                if match:
                    graph_id, source_id, target_id, weight = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    weight = float(weight) if weight else 1.0
                    graph.add_edge(source_id, target_id, weight)
                else:
                    raise PdsXException("GRAPH ADD EDGE komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH REMOVE VERTEX "):
                match = re.match(r"GRAPH REMOVE VERTEX\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, vertex_id = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    graph.remove_vertex(vertex_id)
                else:
                    raise PdsXException("GRAPH REMOVE VERTEX komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH REMOVE EDGE "):
                match = re.match(r"GRAPH REMOVE EDGE\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, source_id, target_id = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    graph.remove_edge(source_id, target_id)
                else:
                    raise PdsXException("GRAPH REMOVE EDGE komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH TRAVERSE "):
                match = re.match(r"GRAPH TRAVERSE\s+(\w+)\s+(\w+)\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, mode, start_id, var_name = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    result = graph.traverse(mode, start_id)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("GRAPH TRAVERSE komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH SHORTEST PATH "):
                match = re.match(r"GRAPH SHORTEST PATH\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, start_id, end_id, var_name = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    result = graph.shortest_path(start_id, end_id)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("GRAPH SHORTEST PATH komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH MINIMUM SPANNING TREE "):
                match = re.match(r"GRAPH MINIMUM SPANNING TREE\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, var_name = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    result = graph.minimum_spanning_tree()
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("GRAPH MINIMUM SPANNING TREE komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH TOPOLOGICAL SORT "):
                match = re.match(r"GRAPH TOPOLOGICAL SORT\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    graph_id, var_name = match.groups()
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    result = graph.topological_sort()
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("GRAPH TOPOLOGICAL SORT komutunda sözdizimi hatası")
            elif command_upper.startswith("GRAPH VISUALIZE "):
                match = re.match(r"GRAPH VISUALIZE\s+(\w+)\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    graph_id, output_path, format = match.groups()
                    format = format or "png"
                    graph = self.graphs.get(graph_id)
                    if not graph:
                        raise PdsXException(f"Grafik bulunamadı: {graph_id}")
                    graph.visualize(output_path, format)
                else:
                    raise PdsXException("GRAPH VISUALIZE komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen grafik komutu: {command}")
        except Exception as e:
            log.error(f"Grafik komut hatası: {str(e)}")
            raise PdsXException(f"Grafik komut hatası: {str(e)}")

if __name__ == "__main__":
    print("graph.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")