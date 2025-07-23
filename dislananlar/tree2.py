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
    """Ağaç düğümü sınıfı."""
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

class Tree:
    """Genel ağaç sınıfı."""
    def __init__(self, tree_type: str = "GENERAL"):
        self.root = None
        self.tree_type = tree_type.upper()
        self.node_count = 0
        self.nodes = {}  # {node_id: node}
        self.lock = threading.Lock()
        self.nil = None  # Kırmızı-siyah ağaç için nil düğüm

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
                else:
                    node = TreeNode(value, node_id)
                    parent.add_child(node)
                
                self.node_count += 1
                self.nodes[node.node_id] = node
                log.debug(f"Düğüm eklendi: parent_id={parent_id}, node_id={node.node_id}")
                return node.node_id
            except Exception as e:
                log.error(f"Düğüm ekleme hatası: {str(e)}")
                raise PdsXException(f"Düğüm ekleme hatası: {str(e)}")

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
                
                clear_subtree(node)
                self.node_count -= 1
                self.nodes.pop(node_id, None)
                
                if self.tree_type == "REDBLACK" and isinstance(node, RedBlackNode):
                    self._rebalance_redblack_remove(node)
                
                log.debug(f"Düğüm kaldırıldı: node_id={node_id}")
            except Exception as e:
                log.error(f"Düğüm kaldırma hatası: {str(e)}")
                raise PdsXException(f"Düğüm kaldırma hatası: {str(e)}")

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
                
                mode = mode.upper()
                if mode == "PREORDER":
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

    def height(self) -> int:
        """Ağacın yüksekliğini döndürür."""
        with self.lock:
            try:
                def compute_height(node: TreeNode):
                    if not node or node == self.nil:
                        return 0
                    if not node.children:
                        return 1
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
                
                def add_nodes_edges(node: TreeNode):
                    if node and node != self.nil:
                        node_label = f"{node.value}\nID: {node.node_id}"
                        if isinstance(node, RedBlackNode):
                            node_label += f"\nColor: {node.color}"
                            dot.node(node.node_id, node_label, color="red" if node.color == "RED" else "black", 
                                     fontcolor="white" if node.color == "BLACK" else "black")
                        else:
                            dot.node(node.node_id, node_label)
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
        """
        Kırmızı-siyah ağaçtan düğüm kaldırıldığında yeniden dengeler.
        PDS-X BASIC odaklı, maksimalist ve eksiksiz bir uygulama.
        """
        try:
            with self.lock:
                # Kaldırılacak düğümün yerine geçecek düğümü bul
                def find_successor(n: RedBlackNode) -> RedBlackNode:
                    current = n.right
                    while current and current.left != self.nil:
                        current = current.left
                    return current
                
                # Düğümün çocuk sayısına göre işlem yap
                if node.left == self.nil and node.right == self.nil:
                    # Yaprak düğüm (veya nil çocukları var)
                    child = self.nil
                elif node.left != self.nil and node.right != self.nil:
                    # İki çocuk var, ardılı bul ve kopyala
                    successor = find_successor(node)
                    node.value = successor.value
                    node.node_id = successor.node_id  # ID’yi de kopyala
                    self.nodes[node.node_id] = node
                    self.nodes.pop(successor.node_id, None)
                    node = successor
                    child = node.right  # Ardılın sağ çocuğu
                else:
                    # Tek çocuk var
                    child = node.left if node.left != self.nil else node.right
                
                # Düğümü ağaçtan çıkar
                parent = node.parent
                if not parent:
                    self.root = child
                elif node == parent.left:
                    parent.left = child
                else:
                    parent.right = child
                if child != self.nil:
                    child.parent = parent
                
                # Kaldırılan düğüm siyahsa, dengeleme gerekli
                if node.color == "BLACK":
                    self._fix_double_black(child, parent)
                
                # Nil düğüm değilse, düğüm silindiğinde referansları güncelle
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
                    
                    # Durum 1: Kardeş kırmızı
                    if sibling.color == "RED":
                        sibling.color = "BLACK"
                        parent.color = "RED"
                        self._rotate_left(parent)
                        sibling = parent.right
                    
                    # Durum 2: Kardeş siyah, her iki çocuk siyah
                    if (sibling.left == self.nil or sibling.left.color == "BLACK") and \
                       (sibling.right == self.nil or sibling.right.color == "BLACK"):
                        sibling.color = "RED"
                        if parent.color == "RED":
                            parent.color = "BLACK"
                        else:
                            node = parent
                            parent = node.parent
                            continue
                    # Durum 3: Kardeş siyah, sağ çocuk kırmızı
                    elif sibling.right != self.nil and sibling.right.color == "RED":
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.right.color = "BLACK"
                        self._rotate_left(parent)
                        break
                    # Durum 4: Kardeş siyah, sol çocuk kırmızı
                    elif sibling.left != self.nil and sibling.left.color == "RED":
                        sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._rotate_right(sibling)
                        sibling = parent.right
                        # Durum 3’e döner
                        sibling.color = parent.color
                        parent.color = "BLACK"
                        sibling.right.color = "BLACK"
                        self._rotate_left(parent)
                        break
                elif parent:
                    sibling = parent.left
                    if sibling == self.nil:
                        break
                    
                    # Simetrik durumlar (sağ taraf)
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
                
                break  # Döngüden çık
            
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
                match = re.match(r"TREE CREATE\s+(\w+)\s+AS\s+(\w+)\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_id, tree_type, value, var_name = match.groups()
                    tree = Tree(tree_type)
                    root_id = tree.create(self.interpreter.evaluate_expression(value))
                    with self.lock:
                        self.trees[tree_id] = tree
                    self.interpreter.current_scope()[var_name] = root_id
                    self.interpreter.current_scope()[f"{tree_id}_TREE"] = tree_id
                    log.debug(f"Ağaç oluşturuldu: tree_id={tree_id}, type={tree_type}")
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