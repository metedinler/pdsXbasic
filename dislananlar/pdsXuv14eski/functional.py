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
        """Fonksiyonu monad değerine uygular."""
        if self.value is None:
            return Maybe(None)
        try:
            return Maybe(func(self.value))
        except Exception as e:
            log.error(f"Maybe bind hatası: {str(e)}")
            return Maybe(None)

    def unwrap(self) -> Any:
        """Değeri çıkarır, None ise hata fırlatır."""
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
        """Fonksiyonu sağ değere uygular, sol ise korunur."""
        if self.left is not None:
            return Either(left=self.left)
        try:
            return Either(right=func(self.right))
        except Exception as e:
            return Either(left=str(e))

    def unwrap(self) -> Any:
        """Sağ değeri çıkarır, sol ise hata fırlatır."""
        if self.left is not None:
            raise PdsXException(f"Either monadında hata: {self.left}")
        return self.right

    def __str__(self) -> str:
        return f"Either(right={self.right}, left={self.left})"

class ImmutableList:
    """Değiştirilemez liste sınıfı."""
    def __init__(self, items: List[Any]):
        self.items = tuple(items)  # Değiştirilemez kopya

    def append(self, item: Any) -> 'ImmutableList':
        """Yeni bir liste döndürür."""
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
        self.items = frozenset(items.items())  # Değiştirilemez kopya

    def set(self, key: Any, value: Any) -> 'ImmutableDict':
        """Yeni bir sözlük döndürür."""
        new_items = dict(self.items)
        new_items[key] = value
        return ImmutableDict(new_items)

    def __getitem__(self, key: Any) -> Any:
        return dict(self.items)[key]

    def __str__(self) -> str:
        return f"ImmutableDict({dict(self.items)})"

class FunctionalManager:
    """Fonksiyonel programlama yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.functions = {}  # {func_id: callable}
        self.lambda_cache = {}  # {lambda_id: (func, closure)}
        self.lock = threading.Lock()
        self.metadata = {"functional": {"version": "1.0.0", "dependencies": ["numpy", "pdsx_exception"]}}

    def lambda_func(self, expr: str, closure: Dict[str, Any]) -> Callable:
        """Lambda fonksiyonu oluşturur."""
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