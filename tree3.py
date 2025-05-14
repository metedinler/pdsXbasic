# tree.py - PDS-X BASIC v14u Ağaç Veri Yapısı Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, deque
import threading
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
log = logging.getLogger("tree")

class TreeNode:
    """Genel ağaç düğümü sınıfı."""
    def __init__(self, value: Any, node_id: Optional[str] = None):
        self.value = value
        self.node_id = node_id or str(uuid.uuid4())
        self.children = []
        self.parent = None
        self.metadata = {}  # Ek özellikler için (örneğin, ağırlık, renk)

    def add_child(self, child: 'TreeNode') -> None:
        """Düğüme çocuk ekler."""
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: 'TreeNode') -> None:
        """Düğüme çocuğu kaldırır."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def __str__(self) -> str:
        return f"Node({self.value}, id={self.node_id})"

class BinaryTreeNode(TreeNode):
    """İkili ağaç düğümü sınıfı."""
    def __init__(self, value: Any, node_id: Optional[str] = None):
        super().__init__(value, node_id)
        self.left = None
        self.right = None
        self.balance_factor = 0  # AVL için

    def add_child(self, child: 'BinaryTreeNode') -> None:
        """İkili ağaç için çocuk ekler."""
        if self.left is None:
            self.left = child
            child.parent = self
        elif self.right is None:
            self.right = child
            child.parent = self
        else:
            raise PdsXException("İkili ağaç düğümünde zaten iki çocuk var")

class RedBlackNode(BinaryTreeNode):
    """Kırmızı-siyah ağaç düğümü sınıfı."""
    def __init__(self, value: Any, node_id: Optional[str] = None):
        super().__init__(value, node_id)
        self.color = "RED"  # RED veya BLACK
        self.nil = None  # Nil düğüm için

    def set_nil(self, nil_node: 'RedBlackNode') -> None:
        """Nil düğümünü ayarlar."""
        self.nil = nil_node

class BTreeNode:
    """B-ağaç düğümü sınıfı."""
    def __init__(self, degree: int, leaf: bool = True, node_id: Optional[str] = None):
        self.degree = degree  # Minimum derece (t)
        self.keys = []  # Anahtar listesi
        self.children = []  # Çocuk işaretçileri
        self.leaf = leaf  # Yaprak düğüm mü?
        self.node_id = node_id or str(uuid.uuid4())
        self.metadata = {}  # Ek özellikler

    def is_full(self) -> bool:
        """Düğümün dolu olup olmadığını kontrol eder."""
        return len(self.keys) >= 2 * self.degree - 1

    def split_child(self, i: int, child: 'BTreeNode') -> None:
        """Düğümün i’inci çocuğunu böler."""
        new_node = BTreeNode(self.degree, child.leaf)
        new_node.keys = child.keys[self.degree:]
        new_node.children = child.children[self.degree:] if not child.leaf else []
        child.keys = child.keys[:self.degree - 1]
        child.children = child.children[:self.degree] if not child.leaf else []
        self.children.insert(i + 1, new_node)
        self.keys.insert(i, child.keys[self.degree - 1])
        new_node.metadata = child.metadata.copy()

class TrieNode:
    """Trie düğümü sınıfı."""
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.children = {}  # {char: TrieNode}
        self.is_end = False  # Kelime sonu mu?
        self.value = None  # İlişkili veri
        self.metadata = {}  # Ek özellikler

class Tree:
    """Ağaç sınıfı (genel, ikili, kırmızı-siyah, B-ağaç, trie)."""
    def __init__(self, tree_type: str = "GENERAL", degree: int = 3):
        self.root = None
        self.tree_type = tree_type.upper()
        self.node_count = 0
        self.nodes = {}  # {node_id: node}
        self.lock = threading.Lock()
        self.nil = None  # Kırmızı-siyah ağaç için nil düğüm
        self.degree = degree  # B-ağaç için minimum derece

    def create(self, value: Any, node_id: Optional[str] = None) -> str:
        """Ağaç oluşturur."""
        with self.lock:
            if self.root:
                raise PdsXException("Ağaç zaten mevcut")
            try:
                if self.tree_type == "BINARY":
                    self.root = BinaryTreeNode(value, node_id)
                elif self.tree_type == "REDBLACK":
                    self.nil = RedBlackNode(None, "NIL")
                    self.nil.color = "BLACK"
                    self.root = RedBlackNode(value, node_id)
                    self.root.set_nil(self.nil)
                    self.root.color = "BLACK"
                    self.root.left = self.nil
                    self.root.right = self.nil
                elif self.tree_type == "BTREE":
                    self.root = BTreeNode(self.degree, True, node_id)
                    self.root.keys = [value]
                elif self.tree_type == "TRIE":
                    self.root = TrieNode(node_id)
                    if isinstance(value, str):
                        self._insert_trie_word(value)
                else:
                    self.root = TreeNode(value, node_id)
                self.node_count = 1
                self.nodes[self.root.node_id] = self.root
                if self.nil:
                    self.nodes[self.nil.node_id] = self.nil
                log.debug(f"Ağaç oluşturuldu: type={self.tree_type}, root_id={self.root.node_id}")
                return self.root.node_id
            except Exception as e:
                log.error(f"Ağaç oluşturma hatası: {str(e)}")
                raise PdsXException(f"Ağaç oluşturma hatası: {str(e)}")

    def add_node(self, parent_id: str, value: Any, node_id: Optional[str] = None) -> str:
        """Ağaca yeni bir düğüm ekler."""
        with self.lock:
            parent = self.nodes.get(parent_id)
            if not parent:
                raise PdsXException(f"Üst düğüm bulunamadı: {parent_id}")
            try:
                if self.tree_type == "BINARY":
                    node = BinaryTreeNode(value, node_id)
                    parent.add_child(node)
                elif self.tree_type == "REDBLACK":
                    node = RedBlackNode(value, node_id)
                    node.set_nil(self.nil)
                    node.left = self.nil
                    node.right = self.nil
                    parent.add_child(node)
                    self._rebalance_redblack(node)
                elif self.tree_type == "BTREE":
                    self._insert_btree(parent, value)
                    return parent.node_id  # B-ağaçta yeni düğüm ID’si döndürmek yerine anahtar eklenir
                elif self.tree_type == "TRIE":
                    if isinstance(value, str):
                        self._insert_trie_word(value, parent)
                        return parent.node_id  # Trie’de kelime eklenir, yeni düğüm ID’si gerekmez
                    else:
                        raise PdsXException("Trie için değer string olmalı")
                else:
                    node = TreeNode(value, node_id)
                    parent.add_child(node)
                
                if self.tree_type not in ("BTREE", "TRIE"):
                    self.node_count += 1
                    self.nodes[node.node_id] = node
                    log.debug(f"Düğüm eklendi: parent_id={parent_id}, node_id={node.node_id}")
                    return node.node_id
                else:
                    log.debug(f"Anahtar eklendi: parent_id={parent_id}, value={value}")
                    return parent_id
            except Exception as e:
                log.error(f"Düğüm ekleme hatası: {str(e)}")
                raise PdsXException(f"Düğüm ekleme hatası: {str(e)}")

    def _insert_btree(self, node: BTreeNode, key: Any) -> None:
        """B-ağaca anahtar ekler."""
        try:
            if node.is_full():
                if node == self.root:
                    new_root = BTreeNode(self.degree, False)
                    new_root.children.append(node)
                    new_root.split_child(0, node)
                    self.root = new_root
                    node = new_root
                else:
                    raise PdsXException("B-ağaç düğümü dolu, bölme gerekiyor")
            
            i = len(node.keys) - 1
            if node.leaf:
                while i >= 0 and node.keys[i] > key:
                    i -= 1
                node.keys.insert(i + 1, key)
            else:
                while i >= 0 and node.keys[i] > key:
                    i -= 1
                self._insert_btree(node.children[i + 1], key)
            
            self.node_count += 1
        except Exception as e:
            log.error(f"B-ağaç ekleme hatası: {str(e)}")
            raise PdsXException(f"B-ağaç ekleme hatası: {str(e)}")

    def _insert_trie_word(self, word: str, start_node: TrieNode = None) -> None:
        """Trie’ye kelime ekler."""
        try:
            current = start_node or self.root
            for char in word:
                if char not in current.children:
                    new_node = TrieNode()
                    current.children[char] = new_node
                    self.nodes[new_node.node_id] = new_node
                    self.node_count += 1
                current = current.children[char]
            current.is_end = True
            current.value = word
            log.debug(f"Trie’ye kelime eklendi: {word}")
        except Exception as e:
            log.error(f"Trie kelime ekleme hatası: {str(e)}")
            raise PdsXException(f"Trie kelime ekleme hatası: {str(e)}")

    def remove_node(self, node_id: str) -> None:
        """Ağaçtan düğüm kaldırır."""
        with self.lock:
            node = self.nodes.get(node_id)
            if not node:
                raise PdsXException(f"Düğüm bulunamadı: {node_id}")
            if node == self.root:
                self.root = None
                self.nodes.clear()
                self.node_count = 0
                if self.nil:
                    self.nodes[self.nil.node_id] = self.nil
                log.debug("Kök düğüm kaldırıldı, ağaç temizlendi")
                return
            
            try:
                parent = node.parent
                if parent:
                    parent.remove_child(node)
                
                # Çocukları temizle
                def clear_subtree(n: TreeNode):
                    for child in n.children:
                        if child != self.nil:
                            clear_subtree(child)
                            self.nodes.pop(child.node_id, None)
                    n.children.clear()
                
                if self.tree_type != "BTREE" and self.tree_type != "TRIE":
                    clear_subtree(node)
                    self.node_count -= 1
                    self.nodes.pop(node_id, None)
                
                if self.tree_type == "REDBLACK" and isinstance(node, RedBlackNode):
                    self._rebalance_redblack_remove(node)
                elif self.tree_type == "BTREE":
                    self._remove_btree(node, node.keys[0])  # İlk anahtarı kaldır
                elif self.tree_type == "TRIE":
                    self._remove_trie_word(node)
                
                log.debug(f"Düğüm kaldırıldı: node_id={node_id}")
            except Exception as e:
                log.error(f"Düğüm kaldırma hatası: {str(e)}")
                raise PdsXException(f"Düğüm kaldırma hatası: {str(e)}")

    def _remove_btree(self, node: BTreeNode, key: Any) -> None:
        """B-ağaçtan anahtar kaldırır."""
        try:
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                if node.leaf:
                    node.keys.pop(i)
                else:
                    raise PdsXException("B-ağaç iç düğümden anahtar kaldırma henüz desteklenmiyor")
            else:
                if node.leaf:
                    raise PdsXException(f"Anahtar bulunamadı: {key}")
                self._remove_btree(node.children[i], key)
            self.node_count -= 1
        except Exception as e:
            log.error(f"B-ağaç kaldırma hatası: {str(e)}")
            raise PdsXException(f"B-ağaç kaldırma hatası: {str(e)}")

    def _remove_trie_word(self, node: TrieNode) -> None:
        """Trie’den kelime kaldırır."""
        try:
            node.is_end = False
            node.value = None
            # Gerekirse düğümleri temizle
            while node and not node.is_end and not node.children and node != self.root:
                parent = node.parent
                for char, child in parent.children.items():
                    if child == node:
                        del parent.children[char]
                        self.nodes.pop(node.node_id, None)
                        self.node_count -= 1
                        break
                node = parent
            log.debug("Trie’den kelime kaldırıldı")
        except Exception as e:
            log.error(f"Trie kelime kaldırma hatası: {str(e)}")
            raise PdsXException(f"Trie kelime kaldırma hatası: {str(e)}")

    def traverse(self, mode: str = "PREORDER") -> List[Dict]:
        """Ağacı dolaşır."""
        with self.lock:
            result = []
            try:
                def preorder(node: TreeNode):
                    if node and node != self.nil:
                        result.append({"id": node.node_id, "value": node.value, "metadata": node.metadata})
                        for child in node.children:
                            preorder(child)
                
                def inorder(node: TreeNode):
                    if node and node != self.nil:
                        inorder(node.left)
                        result.append({"id": node.node_id, "value": node.value, "metadata": node.metadata})
                        inorder(node.right)
                
                def postorder(node: TreeNode):
                    if node and node != self.nil:
                        for child in node.children:
                            postorder(child)
                        result.append({"id": node.node_id, "value": node.value, "metadata": node.metadata})
                
                def btree_traverse(node: BTreeNode):
                    if node:
                        for i, key in enumerate(node.keys):
                            if not node.leaf:
                                btree_traverse(node.children[i])
                            result.append({"id": node.node_id, "value": key, "metadata": node.metadata})
                        if not node.leaf:
                            btree_traverse(node.children[-1])
                
                def trie_traverse(node: TrieNode):
                    if node:
                        if node.is_end:
                            result.append({"id": node.node_id, "value": node.value, "metadata": node.metadata})
                        for char, child in node.children.items():
                            trie_traverse(child)
                
                mode = mode.upper()
                if mode == "PREORDER":
                    if self.tree_type == "BTREE":
                        btree_traverse(self.root)
                    elif self.tree_type == "TRIE":
                        trie_traverse(self.root)
                    else:
                        preorder(self.root)
                elif mode == "INORDER" and self.tree_type in ("BINARY", "REDBLACK"):
                    inorder(self.root)
                elif mode == "POSTORDER":
                    postorder(self.root)
                else:
                    raise PdsXException(f"Desteklenmeyen dolaşım modu: {mode}")
                
                log.debug(f"Ağaç dolaşıldı: mode={mode}, nodes={len(result)}")
                return result
            except Exception as e:
                log.error(f"Ağaç dolaşım hatası: {str(e)}")
                raise PdsXException(f"Ağaç dolaşım hatası: {str(e)}")

    def search(self, value: Any) -> Optional[str]:
        """Ağaçta değer arar."""
        with self.lock:
            try:
                if self.tree_type == "BTREE":
                    return self._search_btree(self.root, value)
                elif self.tree_type == "TRIE":
                    return self._search_trie(value)
                
                def dfs(node: TreeNode):
                    if node and node != self.nil:
                        if node.value == value:
                            return node.node_id
                        for child in node.children:
                            result = dfs(child)
                            if result:
                                return result
                    return None
                
                node_id = dfs(self.root)
                if node_id:
                    log.debug(f"Değer bulundu: value={value}, node_id={node_id}")
                else:
                    log.debug(f"Değer bulunamadı: value={value}")
                return node_id
            except Exception as e:
                log.error(f"Ağaç arama hatası: {str(e)}")
                raise PdsXException(f"Ağaç arama hatası: {str(e)}")

    def _search_btree(self, node: BTreeNode, key: Any) -> Optional[str]:
        """B-ağaçta anahtar arar."""
        try:
            i = 0
            while i < len(node.keys) and node.keys[i] < key:
                i += 1
            if i < len(node.keys) and node.keys[i] == key:
                return node.node_id
            if node.leaf:
                return None
            return self._search_btree(node.children[i], key)
        except Exception as e:
            log.error(f"B-ağaç arama hatası: {str(e)}")
            raise PdsXException(f"B-ağaç arama hatası: {str(e)}")

    def _search_trie(self, word: str) -> Optional[str]:
        """Trie’de kelime arar."""
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    return None
                current = current.children[char]
            if current.is_end:
                return current.node_id
            return None
        except Exception as e:
            log.error(f"Trie arama hatası: {str(e)}")
            raise PdsXException(f"Trie arama hatası: {str(e)}")

    def search_optimized(self, value: Any) -> Optional[str]:
        """Ağaçta optimize edilmiş arama yapar."""
        with self.lock:
            try:
                if self.tree_type in ("BINARY", "REDBLACK"):
                    def bst_search(node: BinaryTreeNode, val: Any):
                        if not node or node == self.nil:
                            return None
                        if node.value == val:
                            return node.node_id
                        if val < node.value:
                            return bst_search(node.left, val)
                        return bst_search(node.right, val)
                    
                    node_id = bst_search(self.root, value)
                elif self.tree_type == "BTREE":
                    node_id = self._search_btree(self.root, value)
                elif self.tree_type == "TRIE":
                    node_id = self._search_trie(value)
                else:
                    node_id = self.search(value)  # Genel ağaç için varsayılan DFS
                
                if node_id:
                    log.debug(f"Optimize edilmiş arama bulundu: value={value}, node_id={node_id}")
                else:
                    log.debug(f"Optimize edilmiş arama bulunamadı: value={value}")
                return node_id
            except Exception as e:
                log.error(f"Optimize edilmiş arama hatası: {str(e)}")
                raise PdsXException(f"Optimize edilmiş arama hatası: {str(e)}")

    def path_sum(self) -> List[Dict]:
        """Ağaçtaki tüm yolların toplamını hesaplar."""
        with self.lock:
            result = []
            try:
                def compute_path_sums(node: TreeNode, current_path: List, current_sum: float):
                    if not node or node == self.nil:
                        return
                    current_path.append(node)
                    current_sum += node.value if isinstance(node.value, (int, float)) else 0
                    if not node.children or (isinstance(node, BinaryTreeNode) and node.left == self.nil and node.right == self.nil):
                        result.append({
                            "path": [n.node_id for n in current_path],
                            "sum": current_sum,
                            "nodes": [n.value for n in current_path]
                        })
                    for child in node.children:
                        if child != self.nil:
                            compute_path_sums(child, current_path.copy(), current_sum)
                
                if self.tree_type == "BTREE":
                    for key in self.root.keys:
                        result.append({"path": [self.root.node_id], "sum": key, "nodes": [key]})
                elif self.tree_type == "TRIE":
                    def trie_path_sums(node: TrieNode, current_path: List, current_word: str):
                        if node.is_end:
                            result.append({
                                "path": [n.node_id for n in current_path],
                                "sum": len(current_word),
                                "nodes": [current_word]
                            })
                        for char, child in node.children.items():
                            trie_path_sums(child, current_path + [child], current_word + char)
                    
                    trie_path_sums(self.root, [self.root], "")
                else:
                    compute_path_sums(self.root, [], 0.0)
                
                log.debug(f"Yol toplamları hesaplandı: paths={len(result)}")
                return result
            except Exception as e:
                log.error(f"Yol toplamı hesaplama hatası: {str(e)}")
                raise PdsXException(f"Yol toplamı hesaplama hatası: {str(e)}")

    def shortest_path(self, start_id: str, end_id: str) -> Dict:
        """İki düğüm arasındaki en kısa yolu bulur."""
        with self.lock:
            start = self.nodes.get(start_id)
            end = self.nodes.get(end_id)
            if not start or not end:
                raise PdsXException(f"Düğümlerden biri bulunamadı: start={start_id}, end={end_id}")
            
            try:
                queue = deque([(start, [start.node_id])])
                visited = set()
                
                while queue:
                    node, path = queue.popleft()
                    if node.node_id == end_id:
                        return {
                            "path": path,
                            "length": len(path) - 1,
                            "nodes": [self.nodes[nid].value for nid in path]
                        }
                    if node.node_id not in visited:
                        visited.add(node.node_id)
                        for child in node.children:
                            if child != self.nil:
                                queue.append((child, path + [child.node_id]))
                        if node.parent and node.parent != self.nil:
                            queue.append((node.parent, path + [node.parent.node_id]))
                
                log.debug(f"En kısa yol bulunamadı: start={start_id}, end={end_id}")
                return {"path": [], "length": -1, "nodes": []}
            except Exception as e:
                log.error(f"En kısa yol bulma hatası: {str(e)}")
                raise PdsXException(f"En kısa yol bulma hatası: {str(e)}")

    def diameter(self) -> Dict:
        """Ağacın çapını (en uzun yolu) hesaplar."""
        with self.lock:
            try:
                max_diameter = [0]
                longest_path = [[]]
                
                def compute_diameter(node: TreeNode) -> int:
                    if not node or node == self.nil:
                        return 0
                    heights = []
                    for child in node.children:
                        if child != self.nil:
                            h = compute_diameter(child)
                            heights.append(h)
                    heights.sort(reverse=True)
                    curr_diameter = sum(heights[:2]) if len(heights) >= 2 else (heights[0] if heights else 0)
                    if curr_diameter > max_diameter[0]:
                        max_diameter[0] = curr_diameter
                        longest_path[0] = [node.node_id]
                    return (heights[0] if heights else 0) + 1
                
                compute_diameter(self.root)
                result = {
                    "diameter": max_diameter[0],
                    "path": longest_path[0]
                }
                log.debug(f"Ağaç çapı hesaplandı: diameter={result['diameter']}")
                return result
            except Exception as e:
                log.error(f"Ağaç çapı hesaplama hatası: {str(e)}")
                raise PdsXException(f"Ağaç çapı hesaplama hatası: {str(e)}")

    def height(self) -> int:
        """Ağacın yüksekliğini döndürür."""
        with self.lock:
            try:
                def compute_height(node: TreeNode):
                    if not node or node == self.nil:
                        return 0
                    if self.tree_type == "BTREE":
                        return 1 + max(compute_height(child) for child in node.children if child)
                    if self.tree_type == "TRIE":
                        return 1 + max(compute_height(child) for child in node.children.values() if child)
                    return 1 + max(compute_height(child) for child in node.children if child != self.nil)
                
                height = compute_height(self.root) if self.root else 0
                log.debug(f"Ağaç yüksekliği: {height}")
                return height
            except Exception as e:
                log.error(f"Ağaç yüksekliği hesaplama hatası: {str(e)}")
                raise PdsXException(f"Ağaç yüksekliği hesaplama hatası: {str(e)}")

    def visualize(self, output_path: str, format: str = "png") -> None:
        """Ağacı görselleştirir."""
        with self.lock:
            try:
                dot = graphviz.Digraph(format=format)
                if not self.root:
                    return
                
                def add_nodes_edges(node: Union[TreeNode, BTreeNode, TrieNode]):
                    if node and node != self.nil:
                        node_label = f"{node.value if not isinstance(node, (BTreeNode, TrieNode)) else node.keys if isinstance(node, BTreeNode) else node.value or ''}\nID: {node.node_id}"
                        if isinstance(node, RedBlackNode):
                            node_label += f"\nColor: {node.color}"
                            dot.node(node.node_id, node_label, color="red" if node.color == "RED" else "black", 
                                     fontcolor="white" if node.color == "BLACK" else "black")
                        elif isinstance(node, BTreeNode):
                            node_label = f"Keys: {node.keys}\nID: {node.node_id}"
                            dot.node(node.node_id, node_label, shape="box")
                        elif isinstance(node, TrieNode):
                            node_label = f"{'[END]' if node.is_end else ''}\nID: {node.node_id}"
                            dot.node(node.node_id, node_label, shape="ellipse")
                        else:
                            dot.node(node.node_id, node_label)
                        
                        if isinstance(node, TrieNode):
                            for char, child in node.children.items():
                                dot.edge(node.node_id, child.node_id, label=char)
                                add_nodes_edges(child)
                        else:
                            for child in node.children:
                                if child != self.nil:
                                    dot.edge(node.node_id, child.node_id)
                                    add_nodes_edges(child)
                
                add_nodes_edges(self.root)
                dot.render(output_path, cleanup=True)
                log.debug(f"Ağaç görselleştirildi: path={output_path}.{format}")
            except Exception as e:
                log.error(f"Ağaç görselleştirme hatası: {str(e)}")
                raise PdsXException(f"Ağaç görselleştirme hatası: {str(e)}")

    def _rebalance_redblack(self, node: RedBlackNode) -> None:
        """Kırmızı-siyah ağaca düğüm eklendiğinde yeniden dengeler."""
        try:
            while node and node != self.root and node.parent and node.parent.color == "RED":
                parent = node.parent
                grandparent = parent.parent
                if not grandparent:
                    break
                
                if parent == grandparent.left:
                    uncle = grandparent.right
                    if uncle and uncle.color == "RED":
                        parent.color = "BLACK"
                        uncle.color = "BLACK"
                        grandparent.color = "RED"
                        node = grandparent
                    else:
                        if node == parent.right:
                            node = parent
                            self._rotate_left(node)
                            parent = node.parent
                        parent.color = "BLACK"
                        grandparent.color = "RED"
                        self._rotate_right(grandparent)
                else:
                    uncle = grandparent.left
                    if uncle and uncle.color == "RED":
                        parent.color = "BLACK"
                        uncle.color = "BLACK"
                        grandparent.color = "RED"
                        node = grandparent
                    else:
                        if node == parent.left:
                            node = parent
                            self._rotate_right(node)
                            parent = node.parent
                        parent.color = "BLACK"
                        grandparent.color = "RED"
                        self._rotate_left(grandparent)
            
            self.root.color = "BLACK"
            log.debug("Kırmızı-siyah ağaç dengelendi")
        except Exception as e:
            log.error(f"Kırmızı-siyah ağaç dengeleme hatası: {str(e)}")
            raise PdsXException(f"Kırmızı-siyah ağaç dengeleme hatası: {str(e)}")

    def _rebalance_redblack_remove(self, node: RedBlackNode) -> None:
        """Kırmızı-siyah ağaçtan düğüm kaldırıldığında yeniden dengeler."""
        try:
            with self.lock:
                def find_successor(n: RedBlackNode) -> RedBlackNode:
                    current = n.right
                    while current and current.left != self.nil:
                        current = current.left
                    return current
                
                if node.left == self.nil and node.right == self.nil:
                    child = self.nil
                elif node.left != self.nil and node.right != self.nil:
                    successor = find_successor(node)
                    node.value = successor.value
                    node.node_id = successor.node_id
                    self.nodes[node.node_id] = node
                    self.nodes.pop(successor.node_id, None)
                    node = successor
                    child = node.right
                else:
                    child = node.left if node.left != self.nil else node.right
                
                parent = node.parent
                if not parent:
                    self.root = child
                elif node == parent.left:
                    parent.left = child
                else:
                    parent.right = child
                if child != self.nil:
                    child.parent = parent
                
                if node.color == "BLACK":
                    self._fix_double_black(child, parent)
                
                if node != self.nil:
                    self.nodes.pop(node.node_id, None)
                    self.node_count -= 1
                
                log.debug(f"Kırmızı-siyah ağaç kaldırma sonrası dengelendi: node_id={node.node_id}")
        except Exception as e:
            log.error(f"Kırmızı-siyah ağaç kaldırma dengeleme hatası: {str(e)}")
            raise PdsXException(f"Kırmızı-siyah ağaç kaldırma dengeleme hatası: {str(e)}")

    def _fix_double_black(self, node: RedBlackNode, parent: Optional[RedBlackNode]) -> None:
        """Çift siyah durumu düzeltir."""
        try:
            while node != self.root and (node == self.nil or node.color == "BLACK"):
                if parent and node == parent.left:
                    sibling = parent.right
                    if sibling == self.nil:
                        break
                    
                    if sibling.color == "RED":
                        sibling.color = "BLACK"
                        parent.color = "RED"
                        self._rotate_left(parent)
                        sibling = parent.right
                    
                    if (sibling.left == self.nil or sibling.left.color == "BLACK") and \
                       (sibling.right == self.nil or sibling.right.color == "BLACK"):
                        sibling.color = "RED"
                        if parent.color == "RED":
                            parent.color = "BLACK"
                        else:
                            node = parent
                            parent = node.parent
                            continue
                    elif sibling.right != self.nil and sibling.right.color == "RED":
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.right.color = "BLACK"
                        self._rotate_left(parent)
                        break
                    elif sibling.left != self.nil and sibling.left.color == "RED":
                        sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._rotate_right(sibling)
                        sibling = parent.right
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.right.color = "BLACK"
                        self._rotate_left(parent)
                        break
                elif parent:
                    sibling = parent.left
                    if sibling == self.nil:
                        break
                    
                    if sibling.color == "RED":
                        sibling.color = "BLACK"
                        parent.color = "RED"
                        self._rotate_right(parent)
                        sibling = parent.left
                    
                    if (sibling.right == self.nil or sibling.right.color == "BLACK") and \
                       (sibling.left == self.nil or sibling.left.color == "BLACK"):
                        sibling.color = "RED"
                        if parent.color == "RED":
                            parent.color = "BLACK"
                        else:
                            node = parent
                            parent = node.parent
                            continue
                    elif sibling.left != self.nil and sibling.left.color == "RED":
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.left.color = "BLACK"
                        self._rotate_right(parent)
                        break
                    elif sibling.right != self.nil and sibling.right.color == "RED":
                        sibling.right.color = "BLACK"
                        sibling.color = "RED"
                        self._rotate_left(sibling)
                        sibling = parent.left
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.left.color = "BLACK"
                        self._rotate_right(parent)
                        break
                
                break
            
            if node and node != self.nil:
                node.color = "BLACK"
            log.debug("Çift siyah durum düzeltildi")
        except Exception as e:
            log.error(f"Çift siyah düzeltme hatası: {str(e)}")
            raise PdsXException(f"Çift siyah düzeltme hatası: {str(e)}")

    def _rotate_left(self, node: RedBlackNode) -> None:
        """Sol rotasyon yapar."""
        try:
            right = node.right
            node.right = right.left
            if right.left != self.nil:
                right.left.parent = node
            right.parent = node.parent
            if not node.parent:
                self.root = right
            elif node == node.parent.left:
                node.parent.left = right
            else:
                node.parent.right = right
            right.left = node
            node.parent = right
            log.debug(f"Sol rotasyon yapıldı: node_id={node.node_id}")
        except Exception as e:
            log.error(f"Sol rotasyon hatası: {str(e)}")
            raise PdsXException(f"Sol rotasyon hatası: {str(e)}")

    def _rotate_right(self, node: RedBlackNode) -> None:
        """Sağ rotasyon yapar."""
        try:
            left = node.left
            node.left = left.right
            if left.right != self.nil:
                left.right.parent = node
            left.parent = node.parent
            if not node.parent:
                self.root = left
            elif node == node.parent.right:
                node.parent.right = left
            else:
                node.parent.left = left
            left.right = node
            node.parent = left
            log.debug(f"Sağ rotasyon yapıldı: node_id={node.node_id}")
        except Exception as e:
            log.error(f"Sağ rotasyon hatası: {str(e)}")
            raise PdsXException(f"Sağ rotasyon hatası: {str(e)}")

class TreeManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.trees = {}  # {tree_id: Tree}
        self.lock = threading.Lock()
        self.metadata = {"tree": {"version": "1.0.0", "dependencies": ["graphviz", "numpy", "pdsx_exception"]}}

    def parse_tree_command(self, command: str) -> None:
        """Ağaç komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("TREE CREATE "):
                match = re.match(r"TREE CREATE\s+(\w+)\s+AS\s+(\w+)\s+(.+?)\s+(\w+)(?:\s+DEGREE\s+(\d+))?", command, re.IGNORECASE)
                if match:
                    tree_id, tree_type, value, var_name, degree = match.groups()
                    tree = Tree(tree_type, degree=int(degree) if degree else 3)
                    root_id = tree.create(self.interpreter.evaluate_expression(value))
                    with self.lock:
                        self.trees[tree_id] = tree
                    self.interpreter.current_scope()[var_name] = root_id
                    self.interpreter.current_scope()[f"{tree_id}_TREE"] = tree_id
                    log.debug(f"Ağaç oluşturuldu: tree_id={tree_id}, type={tree_type}, degree={degree or 3}")
                else:
                    raise PdsXException("TREE CREATE komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE ADD "):
                match = re.match(r"TREE ADD\s+(\w+)\s+(\w+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, parent_id, value, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    node_id = tree.add_node(parent_id, self.interpreter.evaluate_expression(value))
                    self.interpreter.current_scope()[var_name] = node_id
                else:
                    raise PdsXException("TREE ADD komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE REMOVE "):
                match = re.match(r"TREE REMOVE\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, node_id = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    tree.remove_node(node_id)
                else:
                    raise PdsXException("TREE REMOVE komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE TRAVERSE "):
                match = re.match(r"TREE TRAVERSE\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, mode, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    result = tree.traverse(mode)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("TREE TRAVERSE komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE SEARCH "):
                match = re.match(r"TREE SEARCH\s+(\w+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, value, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    node_id = tree.search(self.interpreter.evaluate_expression(value))
                    self.interpreter.current_scope()[var_name] = node_id
                else:
                    raise PdsXException("TREE SEARCH komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE SEARCH OPTIMIZED "):
                match = re.match(r"TREE SEARCH OPTIMIZED\s+(\w+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, value, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    node_id = tree.search_optimized(self.interpreter.evaluate_expression(value))
                    self.interpreter.current_scope()[var_name] = node_id
                else:
                    raise PdsXException("TREE SEARCH OPTIMIZED komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE PATH SUM "):
                match = re.match(r"TREE PATH SUM\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    result = tree.path_sum()
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("TREE PATH SUM komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE SHORTEST PATH "):
                match = re.match(r"TREE SHORTEST PATH\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, start_id, end_id, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    result = tree.shortest_path(start_id, end_id)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("TREE SHORTEST PATH komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE DIAMETER "):
                match = re.match(r"TREE DIAMETER\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    result = tree.diameter()
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("TREE DIAMETER komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE HEIGHT "):
                match = re.match(r"TREE HEIGHT\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, var_name = match.groups()
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    height = tree.height()
                    self.interpreter.current_scope()[var_name] = height
                else:
                    raise PdsXException("TREE HEIGHT komutunda sözdizimi hatası")
            elif command_upper.startswith("TREE VISUALIZE "):
                match = re.match(r"TREE VISUALIZE\s+(\w+)\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    tree_id, output_path, format = match.groups()
                    format = format or "png"
                    tree = self.trees.get(tree_id)
                    if not tree:
                        raise PdsXException(f"Ağaç bulunamadı: {tree_id}")
                    tree.visualize(output_path, format)
                else:
                    raise PdsXException("TREE VISUALIZE komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen ağaç komutu: {command}")
        except Exception as e:
            log.error(f"Ağaç komut hatası: {str(e)}")
            raise PdsXException(f"Ağaç komut hatası: {str(e)}")

if __name__ == "__main__":
    print("tree.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")