# functional.py - PDS-X BASIC v14u Fonksiyonel Programlama Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading
import functools
import copy
import uuid
import numpy as np
from itertools import islice
from pdsx_exception import PdsXException  # Hata yönetimi için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("functional")

class Maybe:
    """Maybe monadı sınıfı."""
    def __init__(self, value: Optional[Any] = None):
        self.value = value

    def bind(self, func: Callable) -> 'Maybe':
        if self.value is None:
            return Maybe(None)
        try:
            return Maybe(func(self.value))
        except Exception as e:
            log.error(f"Maybe bind hatası: {str(e)}")
            return Maybe(None)

    def unwrap(self) -> Any:
        if self.value is None:
            raise PdsXException("Maybe monadından None değer çıkarılamaz")
        return self.value

    def __str__(self) -> str:
        return f"Maybe({self.value})"

class Either:
    """Either monadı sınıfı (Sağ veya Sol)."""
    def __init__(self, right: Optional[Any] = None, left: Optional[Any] = None):
        self.right = right
        self.left = left

    def bind(self, func: Callable) -> 'Either':
        if self.left is not None:
            return Either(left=self.left)
        try:
            return Either(right=func(self.right))
        except Exception as e:
            return Either(left=str(e))

    def unwrap(self) -> Any:
        if self.left is not None:
            raise PdsXException(f"Either monadında hata: {self.left}")
        return self.right

    def __str__(self) -> str:
        return f"Either(right={self.right}, left={self.left})"

class State:
    """State monadı sınıfı."""
    def __init__(self, run: Callable[[Any], Tuple[Any, Any]]):
        self.run = run

    def bind(self, func: Callable) -> 'State':
        def new_run(state: Any) -> Tuple[Any, Any]:
            value, new_state = self.run(state)
            return func(value).run(new_state)
        return State(new_run)

    def eval(self, state: Any) -> Any:
        value, _ = self.run(state)
        return value

    def exec(self, state: Any) -> Any:
        _, new_state = self.run(state)
        return new_state

    def __str__(self) -> str:
        return "State(...)"

class IO:
    """IO monadı sınıfı."""
    def __init__(self, action: Callable[[], Any]):
        self.action = action

    def bind(self, func: Callable) -> 'IO':
        def new_action():
            result = self.action()
            return func(result).action()
        return IO(new_action)

    def run(self) -> Any:
        return self.action()

    def __str__(self) -> str:
        return "IO(...)"

class Reader:
    """Reader monadı sınıfı."""
    def __init__(self, run: Callable[[Any], Any]):
        self.run = run

    def bind(self, func: Callable) -> 'Reader':
        def new_run(env: Any) -> Any:
            value = self.run(env)
            return func(value).run(env)
        return Reader(new_run)

    def run_reader(self, env: Any) -> Any:
        return self.run(env)

    def __str__(self) -> str:
        return "Reader(...)"

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

class ImmutableTree:
    """Değiştirilemez ağaç sınıfı."""
    def __init__(self, value: Any, children: List['ImmutableTree'] = None):
        self.value = value
        self.children = tuple(children or [])

    def add_child(self, child: 'ImmutableTree') -> 'ImmutableTree':
        return ImmutableTree(self.value, list(self.children) + [child])

    def __str__(self) -> str:
        return f"ImmutableTree({self.value}, children={len(self.children)})"

class ImmutableGraph:
    """Değiştirilemez grafik sınıfı."""
    def __init__(self, vertices: Dict[str, Any], edges: Dict[str, List[Tuple[str, float]]]):
        self.vertices = frozenset(vertices.items())
        self.edges = frozenset((src, tuple(targets)) for src, targets in edges.items())

    def add_vertex(self, vertex_id: str, value: Any) -> 'ImmutableGraph':
        new_vertices = dict(self.vertices)
        new_vertices[vertex_id] = value
        new_edges = dict(self.edges)
        if vertex_id not in new_edges:
            new_edges[vertex_id] = []
        return ImmutableGraph(new_vertices, new_edges)

    def add_edge(self, source_id: str, target_id: str, weight: float) -> 'ImmutableGraph':
        new_edges = dict(self.edges)
        new_edges[source_id] = list(new_edges.get(source_id, [])) + [(target_id, weight)]
        return ImmutableGraph(dict(self.vertices), new_edges)

    def __str__(self) -> str:
        return f"ImmutableGraph(vertices={len(self.vertices)}, edges={sum(len(targets) for _, targets in self.edges)})"

class Stream:
    """Tembel değerlendirilen veri akışı sınıfı."""
    def __init__(self, generator: Callable[[], Any]):
        self.generator = generator
        self.cache = []

    def take(self, n: int) -> List[Any]:
        """İlk n öğeyi alır."""
        try:
            while len(self.cache) < n:
                self.cache.append(next(self.generator()))
            return self.cache[:n]
        except StopIteration:
            return self.cache
        except Exception as e:
            log.error(f"Stream take hatası: {str(e)}")
            raise PdsXException(f"Stream take hatası: {str(e)}")

    def map(self, func: Callable) -> 'Stream':
        """Tembel map işlemi."""
        def new_generator():
            for item in self.generator():
                yield func(item)
        return Stream(new_generator)

    def filter(self, func: Callable) -> 'Stream':
        """Tembel filter işlemi."""
        def new_generator():
            for item in self.generator():
                if func(item):
                    yield item
        return Stream(new_generator)

    def __str__(self) -> str:
        return "Stream(...)"

class Zipper:
    """Comonadic zipper sınıfı (ağaç gezinmesi için)."""
    def __init__(self, tree: ImmutableTree, path: List[Tuple['ImmutableTree', int]] = None):
        self.focus = tree
        self.path = path or []

    def move_down(self, index: int) -> 'Zipper':
        """Çocuk düğüme iner."""
        if index >= len(self.focus.children):
            raise PdsXException(f"Geçersiz çocuk indeksi: {index}")
        new_path = self.path + [(self.focus, index)]
        return Zipper(self.focus.children[index], new_path)

    def move_up(self) -> 'Zipper':
        """Üst düğüme çıkar."""
        if not self.path:
            raise PdsXException("Zaten kökte")
        parent, index = self.path[-1]
        new_path = self.path[:-1]
        return Zipper(parent, new_path)

    def update(self, value: Any) -> 'Zipper':
        """Odaklanmış düğümün değerini günceller."""
        new_tree = ImmutableTree(value, self.focus.children)
        return Zipper(new_tree, self.path)

    def __str__(self) -> str:
        return f"Zipper(focus={self.focus.value}, path_len={len(self.path)})"

class Effect:
    """Deneysel efekt sistemi sınıfı."""
    def __init__(self, action: Callable, effects: List[str] = None):
        self.action = action
        self.effects = effects or []

    def run(self, context: Dict[str, Any]) -> Any:
        """Efekti çalıştırır, yan etkileri izler."""
        try:
            result = self.action(context)
            return {"result": result, "effects": self.effects}
        except Exception as e:
            log.error(f"Effect run hatası: {str(e)}")
            raise PdsXException(f"Effect run hatası: {str(e)}")

    def __str__(self) -> str:
        return f"Effect(effects={self.effects})"

class Lens:
    """Lens sınıfı (optik veri manipülasyonu)."""
    def __init__(self, getter: Callable, setter: Callable):
        self.getter = getter
        self.setter = setter

    def get(self, obj: Any) -> Any:
        return self.getter(obj)

    def set(self, obj: Any, value: Any) -> Any:
        return self.setter(obj, value)

    def compose(self, other: 'Lens') -> 'Lens':
        def new_getter(obj):
            return self.getter(other.getter(obj))
        def new_setter(obj, value):
            inner = other.getter(obj)
            updated_inner = self.setter(inner, value)
            return other.setter(obj, updated_inner)
        return Lens(new_getter, new_setter)

    def __str__(self) -> str:
        return "Lens(...)"

class Prism:
    """Prism sınıfı (optik veri manipülasyonu)."""
    def __init__(self, preview: Callable, review: Callable):
        self.preview = preview
        self.review = review

    def try_get(self, obj: Any) -> Maybe:
        try:
            return Maybe(self.preview(obj))
        except Exception:
            return Maybe(None)

    def set(self, value: Any) -> Any:
        return self.review(value)

    def __str__(self) -> str:
        return "Prism(...)"

class FreeMonad:
    """Free monad sınıfı."""
    def __init__(self, value: Any, is_pure: bool = True):
        self.value = value
        self.is_pure = is_pure

    def bind(self, func: Callable) -> 'FreeMonad':
        if self.is_pure:
            return func(self.value)
        return FreeMonad((self.value, func), is_pure=False)

    def run(self, interpreter: Callable) -> Any:
        if self.is_pure:
            return self.value
        value, func = self.value
        return func(interpreter(value)).run(interpreter)

    def __str__(self) -> str:
        return f"FreeMonad(is_pure={self.is_pure})"

class FunctionalManager:
    """Fonksiyonel programlama yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.functions = {}  # {func_id: callable}
        self.lambda_cache = {}  # {lambda_id: (func, closure)}
        self.lock = threading.Lock()
        self.metadata = {"functional": {"version": "1.0.0", "dependencies": ["numpy", "pdsx_exception"]}}

    def lambda_func(self, expr: str, closure: Dict[str, Any]) -> Callable:
        """Standart lambda fonksiyonu oluşturur."""
        with self.lock:
            try:
                def lambda_wrapper(*args, **kwargs):
                    local_scope = closure.copy()
                    for i, arg in enumerate(args):
                        local_scope[f"arg{i}"] = arg
                    local_scope.update(kwargs)
                    self.interpreter.current_scope().update(local_scope)
                    return self.interpreter.evaluate_expression(expr)
                
                lambda_id = str(uuid.uuid4())
                self.lambda_cache[lambda_id] = (lambda_wrapper, closure)
                log.debug(f"Lambda fonksiyonu oluşturuldu: id={lambda_id}, expr={expr}")
                return lambda_wrapper
            except Exception as e:
                log.error(f"Lambda oluşturma hatası: {str(e)}")
                raise PdsXException(f"Lambda oluşturma hatası: {str(e)}")

    def omega_func(self, expr: str, closure: Dict[str, Any]) -> Callable:
        """Kendi kendine uygulanan (omega) lambda fonksiyonu oluşturur."""
        with self.lock:
            try:
                def omega_wrapper(*args, **kwargs):
                    local_scope = closure.copy()
                    local_scope["self"] = omega_wrapper  # Kendi kendine referans
                    for i, arg in enumerate(args):
                        local_scope[f"arg{i}"] = arg
                    local_scope.update(kwargs)
                    self.interpreter.current_scope().update(local_scope)
                    return self.interpreter.evaluate_expression(expr)
                
                lambda_id = str(uuid.uuid4())
                self.lambda_cache[lambda_id] = (omega_wrapper, closure)
                log.debug(f"Omega fonksiyonu oluşturuldu: id={lambda_id}, expr={expr}")
                return omega_wrapper
            except Exception as e:
                log.error(f"Omega oluşturma hatası: {str(e)}")
                raise PdsXException(f"Omega oluşturma hatası: {str(e)}")

    def gamma_func(self, expr: str, closure: Dict[str, Any], partial_args: List[Any] = None) -> Callable:
        """Kısmi uygulamalı zincirleme (gamma) lambda fonksiyonu oluşturur."""
        with self.lock:
            try:
                partial_args = partial_args or []
                def gamma_wrapper(*args, **kwargs):
                    local_scope = closure.copy()
                    combined_args = partial_args + list(args)
                    for i, arg in enumerate(combined_args):
                        local_scope[f"arg{i}"] = arg
                    local_scope.update(kwargs)
                    self.interpreter.current_scope().update(local_scope)
                    result = self.interpreter.evaluate_expression(expr)
                    if callable(result):
                        return self.gamma_func(expr, local_scope, combined_args)
                    return result
                
                lambda_id = str(uuid.uuid4())
                self.lambda_cache[lambda_id] = (gamma_wrapper, closure)
                log.debug(f"Gamma fonksiyonu oluşturuldu: id={lambda_id}, expr={expr}")
                return gamma_wrapper
            except Exception as e:
                log.error(f"Gamma oluşturma hatası: {str(e)}")
                raise PdsXException(f"Gamma oluşturma hatası: {str(e)}")

    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Fonksiyonu bir iterable’a uygular."""
        try:
            result = [func(item) for item in iterable]
            log.debug(f"Map uygulandı: items={len(iterable)}")
            return result
        except Exception as e:
            log.error(f"Map hatası: {str(e)}")
            raise PdsXException(f"Map hatası: {str(e)}")

    def filter(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """Fonksiyonla iterable’ı filtreler."""
        try:
            result = [item for item in iterable if func(item)]
            log.debug(f"Filter uygulandı: items={len(iterable)}, filtered={len(result)}")
            return result
        except Exception as e:
            log.error(f"Filter hatası: {str(e)}")
            raise PdsXException(f"Filter hatası: {str(e)}")

    def reduce(self, func: Callable, iterable: List[Any], initial: Optional[Any] = None) -> Any:
        """Fonksiyonla iterable’ı indirger."""
        try:
            iterator = iter(iterable)
            value = initial if initial is not None else next(iterator)
            for item in iterator:
                value = func(value, item)
            log.debug(f"Reduce uygulandı: items={len(iterable)}")
            return value
        except Exception as e:
            log.error(f"Reduce hatası: {str(e)}")
            raise PdsXException(f"Reduce hatası: {str(e)}")

    def pipe(self, value: Any, *funcs: Callable) -> Any:
        """Fonksiyonları sırayla zincirler."""
        try:
            result = value
            for func in funcs:
                result = func(result)
            log.debug(f"Pipe uygulandı: funcs={len(funcs)}")
            return result
        except Exception as e:
            log.error(f"Pipe hatası: {str(e)}")
            raise PdsXException(f"Pipe hatası: {str(e)}")

    def compose(self, *funcs: Callable) -> Callable:
        """Fonksiyonları ters sırayla birleştirir."""
        try:
            def composed(*args, **kwargs) -> Any:
                result = funcs[-1](*args, **kwargs)
                for func in reversed(funcs[:-1]):
                    result = func(result)
                return result
            log.debug(f"Compose uygulandı: funcs={len(funcs)}")
            return composed
        except Exception as e:
            log.error(f"Compose hatası: {str(e)}")
            raise PdsXException(f"Compose hatası: {str(e)}")

    def curry(self, func: Callable, *fixed_args, **fixed_kwargs) -> Callable:
        """Fonksiyonu kısmi olarak uygular."""
        try:
            def curried(*args, **kwargs) -> Any:
                combined_args = fixed_args + args
                combined_kwargs = fixed_kwargs.copy()
                combined_kwargs.update(kwargs)
                return func(*combined_args, **combined_kwargs)
            log.debug(f"Curry uygulandı: func={func.__name__}")
            return curried
        except Exception as e:
            log.error(f"Curry hatası: {str(e)}")
            raise PdsXException(f"Curry hatası: {str(e)}")

    def fold(self, func: Callable, iterable: List[Any], initial: Any) -> Any:
        """Sol katlama (fold left) yapar."""
        try:
            result = initial
            for item in iterable:
                result = func(result, item)
            log.debug(f"Fold uygulandı: items={len(iterable)}")
            return result
        except Exception as e:
            log.error(f"Fold hatası: {str(e)}")
            raise PdsXException(f"Fold hatası: {str(e)}")

    def scan(self, func: Callable, iterable: List[Any], initial: Any) -> List[Any]:
        """Kümülatif katlama (scan) yapar."""
        try:
            result = [initial]
            current = initial
            for item in iterable:
                current = func(current, item)
                result.append(current)
            log.debug(f"Scan uygulandı: items={len(iterable)}")
            return result
        except Exception as e:
            log.error(f"Scan hatası: {str(e)}")
            raise PdsXException(f"Scan hatası: {str(e)}")

    def zip_with(self, func: Callable, *iterables: List[Any]) -> List[Any]:
        """Birden fazla iterable’ı fonksiyonla birleştirir."""
        try:
            result = [func(*items) for items in zip(*iterables)]
            log.debug(f"Zip_with uygulandı: iterables={len(iterables)}")
            return result
        except Exception as e:
            log.error(f"Zip_with hatası: {str(e)}")
            raise PdsXException(f"Zip_with hatası: {str(e)}")

    def parse_functional_command(self, command: str) -> None:
        """Fonksiyonel programlama komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("FUNC LAMBDA "):
                match = re.match(r"FUNC LAMBDA\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    closure = self.interpreter.current_scope().copy()
                    func = self.lambda_func(expr, closure)
                    self.interpreter.current_scope()[var_name] = func
                else:
                    raise PdsXException("FUNC LAMBDA komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC OMEGA "):
                match = re.match(r"FUNC OMEGA\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    closure = self.interpreter.current_scope().copy()
                    func = self.omega_func(expr, closure)
                    self.interpreter.current_scope()[var_name] = func
                else:
                    raise PdsXException("FUNC OMEGA komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC GAMMA "):
                match = re.match(r"FUNC GAMMA\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, args_str, var_name = match.groups()
                    closure = self.interpreter.current_scope().copy()
                    args = eval(args_str, self.interpreter.current_scope())
                    args = args if isinstance(args, list) else [args]
                    func = self.gamma_func(expr, closure, args)
                    self.interpreter.current_scope()[var_name] = func
                else:
                    raise PdsXException("FUNC GAMMA komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC MAP "):
                match = re.match(r"FUNC MAP\s+(\w+)\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, items_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    items = eval(items_str, self.interpreter.current_scope())
                    result = self.map(func, items)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC MAP komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC FILTER "):
                match = re.match(r"FUNC FILTER\s+(\w+)\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, items_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    items = eval(items_str, self.interpreter.current_scope())
                    result = self.filter(func, items)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC FILTER komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC REDUCE "):
                match = re.match(r"FUNC REDUCE\s+(\w+)\s+\[(.+?)\]\s*(.+?)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, items_str, initial_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    items = eval(items_str, self.interpreter.current_scope())
                    initial = self.interpreter.evaluate_expression(initial_str) if initial_str else None
                    result = self.reduce(func, items, initial)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC REDUCE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC PIPE "):
                match = re.match(r"FUNC PIPE\s+(.+?)\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    value_str, funcs_str, var_name = match.groups()
                    value = self.interpreter.evaluate_expression(value_str)
                    func_names = [f.strip() for f in funcs_str.split(",")]
                    funcs = [self.interpreter.current_scope().get(fn) for fn in func_names]
                    if not all(callable(f) for f in funcs):
                        raise PdsXException("Tüm pipe fonksiyonları callable olmalı")
                    result = self.pipe(value, *funcs)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC PIPE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC COMPOSE "):
                match = re.match(r"FUNC COMPOSE\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    funcs_str, var_name = match.groups()
                    func_names = [f.strip() for f in funcs_str.split(",")]
                    funcs = [self.interpreter.current_scope().get(fn) for fn in func_names]
                    if not all(callable(f) for f in funcs):
                        raise PdsXException("Tüm compose fonksiyonları callable olmalı")
                    result = self.compose(*funcs)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC COMPOSE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC CURRY "):
                match = re.match(r"FUNC CURRY\s+(\w+)\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, args_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    args = eval(args_str, self.interpreter.current_scope())
                    args = args if isinstance(args, tuple) else (args,)
                    result = self.curry(func, *args)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC CURRY komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC MAYBE "):
                match = re.match(r"FUNC MAYBE\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    value_str, var_name = match.groups()
                    value = self.interpreter.evaluate_expression(value_str)
                    result = Maybe(value)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC MAYBE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC EITHER "):
                match = re.match(r"FUNC EITHER\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    value_str, var_name = match.groups()
                    value = self.interpreter.evaluate_expression(value_str)
                    result = Either(right=value)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC EITHER komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC STATE "):
                match = re.match(r"FUNC STATE\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    def state_action(state):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["state"] = state
                        self.interpreter.current_scope().update(local_scope)
                        value = self.interpreter.evaluate_expression(expr)
                        return value, local_scope.get("state", state)
                    result = State(state_action)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC STATE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC IO "):
                match = re.match(r"FUNC IO\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    def io_action():
                        return self.interpreter.evaluate_expression(expr)
                    result = IO(io_action)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC IO komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC READER "):
                match = re.match(r"FUNC READER\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    def reader_action(env):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["env"] = env
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(expr)
                    result = Reader(reader_action)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC READER komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC STREAM "):
                match = re.match(r"FUNC STREAM\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, var_name = match.groups()
                    def stream_generator():
                        local_scope = self.interpreter.current_scope().copy()
                        self.interpreter.current_scope().update(local_scope)
                        while True:
                            yield self.interpreter.evaluate_expression(expr)
                    result = Stream(stream_generator)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC STREAM komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC ZIPPER "):
                match = re.match(r"FUNC ZIPPER\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    tree_str, var_name = match.groups()
                    tree = self.interpreter.evaluate_expression(tree_str)
                    if not isinstance(tree, ImmutableTree):
                        raise PdsXException("Zipper için ImmutableTree gerekli")
                    result = Zipper(tree)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC ZIPPER komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC EFFECT "):
                match = re.match(r"FUNC EFFECT\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    expr, effects_str, var_name = match.groups()
                    effects = eval(effects_str, self.interpreter.current_scope())
                    def effect_action(context):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope.update(context)
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(expr)
                    result = Effect(effect_action, effects)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC EFFECT komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC LENS "):
                match = re.match(r"FUNC LENS\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    getter_expr, setter_expr, var_name = match.groups()
                    def getter(obj):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["obj"] = obj
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(getter_expr)
                    def setter(obj, value):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["obj"] = obj
                        local_scope["value"] = value
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(setter_expr)
                    result = Lens(getter, setter)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC LENS komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC PRISM "):
                match = re.match(r"FUNC PRISM\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    preview_expr, review_expr, var_name = match.groups()
                    def preview(obj):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["obj"] = obj
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(preview_expr)
                    def review(value):
                        local_scope = self.interpreter.current_scope().copy()
                        local_scope["value"] = value
                        self.interpreter.current_scope().update(local_scope)
                        return self.interpreter.evaluate_expression(review_expr)
                    result = Prism(preview, review)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC PRISM komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC FREE "):
                match = re.match(r"FUNC FREE\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    value_str, var_name = match.groups()
                    value = self.interpreter.evaluate_expression(value_str)
                    result = FreeMonad(value)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC FREE komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC FOLD "):
                match = re.match(r"FUNC FOLD\s+(\w+)\s+\[(.+?)\]\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, items_str, initial_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    items = eval(items_str, self.interpreter.current_scope())
                    initial = self.interpreter.evaluate_expression(initial_str)
                    result = self.fold(func, items, initial)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC FOLD komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC SCAN "):
                match = re.match(r"FUNC SCAN\s+(\w+)\s+\[(.+?)\]\s+(.+?)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, items_str, initial_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    items = eval(items_str, self.interpreter.current_scope())
                    initial = self.interpreter.evaluate_expression(initial_str)
                    result = self.scan(func, items, initial)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC SCAN komutunda sözdizimi hatası")
            elif command_upper.startswith("FUNC ZIP WITH "):
                match = re.match(r"FUNC ZIP WITH\s+(\w+)\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    func_name, iterables_str, var_name = match.groups()
                    func = self.interpreter.current_scope().get(func_name)
                    if not callable(func):
                        raise PdsXException(f"Fonksiyon bulunamadı veya callable değil: {func_name}")
                    iterables = eval(iterables_str, self.interpreter.current_scope())
                    result = self.zip_with(func, *iterables)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("FUNC ZIP WITH komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen fonksiyonel komut: {command}")
        except Exception as e:
            log.error(f"Fonksiyonel komut hatası: {str(e)}")
            raise PdsXException(f"Fonksiyonel komut hatası: {str(e)}")

if __name__ == "__main__":
    print("functional.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")