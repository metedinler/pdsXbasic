# pdsx_exception2.py - PDS-X BASIC v15 Hata Yönetim Kütüphanesi
# Version: 1.5.2
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import traceback
import json
import time
import threading
import asyncio
import aiofiles
import re
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
import graphviz
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from hashlib import sha256
import psutil  # Sistem metrikleri için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdsx_exception2")

class PdsXException(Exception):
    """PDS-X BASIC v15 temel hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = None):
        self.message = message
        self.context = context or {"lang": "en", "module": "unknown", "line_no": 0}
        self.code = code or "ERR_UNKNOWN"
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()[:-1]
        self.system_metrics = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent
        }
        super().__init__(self._format_message())
        log.error(self._format_message())

    @lru_cache(maxsize=256)
    def _format_message(self) -> str:
        """Hata mesajını formatlar (önbellekli)."""
        translations = {
            "en": {"ERROR": "Error", "CODE": "Code", "MODULE": "Module", "LINE": "Line"},
            "tr": {"ERROR": "Hata", "CODE": "Kod", "MODULE": "Modül", "LINE": "Satır"}
        }
        lang = self.context.get("lang", "en")
        prefix = translations[lang]["ERROR"]
        code_str = f"{translations[lang]['CODE']}: {self.code}" if self.code else ""
        module = f"{translations[lang]['MODULE']}: {self.context.get('module', 'unknown')}"
        line = f"{translations[lang]['LINE']}: {self.context.get('line_no', 0)}"
        return f"{prefix}: {self.message} ({code_str}) [{module}, {line}]"

class PdsXSyntaxError(PdsXException):
    """Sözdizimi hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_SYNTAX"):
        super().__init__(message, context, code)

class PdsXRuntimeError(PdsXException):
    """Çalışma zamanı hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_RUNTIME"):
        super().__init__(message, context, code)

class PdsXTypeError(PdsXException):
    """Tip hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_TYPE"):
        super().__init__(message, context, code)

class PdsXValueError(PdsXException):
    """Değer hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_VALUE"):
        super().__init__(message, context, code)

class PdsXMemoryError(PdsXException):
    """Bellek hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_MEMORY"):
        super().__init__(message, context, code)

class PdsXIOException(PdsXException):
    """Giriş/çıkış hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_IO"):
        super().__init__(message, context, code)

class PdsXSecurityError(PdsXException):
    """Güvenlik hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_SECURITY"):
        super().__init__(message, context, code)

class PdsXNetworkError(PdsXException):
    """Ağ hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_NETWORK"):
        super().__init__(message, context, code)

class PdsXDatabaseError(PdsXException):
    """Veritabanı hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_DATABASE"):
        super().__init__(message, context, code)

class PdsXLogicError(PdsXException):
    """Mantıksal programlama hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_LOGIC"):
        super().__init__(message, context, code)

class PdsXConcurrencyError(PdsXException):
    """Eşzamanlılık hataları için hata sınıfı."""
    def __init__(self, message: str, context: Optional[Dict] = None, code: Optional[str] = "ERR_CONCURRENCY"):
        super().__init__(message, context, code)

class ExceptionManager:
    """PDS-X BASIC v15 hata yönetimi sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.error_log: List[Dict] = []
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.code_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.suppressed_types: set = set()
        self.suppressed_codes: set = set()
        self.debug_mode: bool = False
        self.trace_mode: bool = False
        self.error_stats: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "last_time": 0.0, "modules": {}})
        self.lock = threading.Lock()
        self.metadata: Dict = {
            "pdsx_exception2": {
                "version": "1.5.0",
                "dependencies": ["sklearn", "tensorflow", "numpy", "graphviz", "aiofiles", "psutil"]
            }
        }
        self._init_neural_model()

    def _init_neural_model(self) -> None:
        """Nöral hata tahmini için LSTM modeli başlatır."""
        self.neural_model = Sequential([
            LSTM(64, input_shape=(20, 1), return_sequences=True),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        self.neural_model.compile(optimizer="adam", loss="binary_crossentropy")

    async def handle_error(self, error: Exception) -> None:
        """Hata işleme ve günlüğe kaydetme (asenkron)."""
        with self.lock:
            error_type = type(error).__name__
            error_code = getattr(error, "code", "ERR_UNKNOWN")
            if error_type in self.suppressed_types or error_code in self.suppressed_codes:
                log.debug(f"Bastırılmış hata: {error_type}, Kod: {error_code}")
                return

            error_info = {
                "type": error_type,
                "message": str(error),
                "code": error_code,
                "timestamp": time.time(),
                "stack_trace": traceback.format_tb(error.__traceback__),
                "context": getattr(error, "context", {"module": "unknown", "line_no": 0}),
                "system_metrics": getattr(error, "system_metrics", {"cpu": 0.0, "memory": 0.0}),
                "cause": str(error.__cause__) if error.__cause__ else None
            }
            self.error_log.append(error_info)
            module = error_info["context"].get("module", "unknown")
            self.error_stats[error_type]["count"] += 1
            self.error_stats[error_type]["last_time"] = error_info["timestamp"]
            self.error_stats[error_type]["modules"][module] = self.error_stats[error_type]["modules"].get(module, 0) + 1
            log.error(f"Hata işlendi: {error_type} - {error_info['message']} (Kod: {error_code})")

            # Asenkron günlüğe yazma
            await self._async_log_error(error_info)

            # Otomatik hata ayıklama
            if self.debug_mode:
                await self._async_auto_debug(error_info)

            # Hata türü için handler tetikleme
            for handler in self.handlers[error_type]:
                try:
                    handler(error_info)
                except Exception as e:
                    log.error(f"Tür handler hatası: {str(e)}")

            # Hata kodu için handler tetikleme
            for handler in self.code_handlers[error_code]:
                try:
                    handler(error_info)
                except Exception as e:
                    log.error(f"Kod handler hatası: {str(e)}")

    async def _async_log_error(self, error_info: Dict) -> None:
        """Hata bilgisini asenkron olarak günlüğe kaydeder."""
        async with aiofiles.open("pdsxu_errors.json", "a", encoding="utf-8") as f:
            await f.write(json.dumps(error_info, indent=2) + "\n")

    async def _async_auto_debug(self, error_info: Dict) -> None:
        """Otomatik hata ayıklama bilgisi asenkron kaydeder."""
        debug_info = {
            "variables": self.interpreter.current_scope(),
            "program_counter": self.interpreter.program_counter,
            "source_line": self.interpreter.program[self.interpreter.program_counter][0]
                           if self.interpreter.program_counter < len(self.interpreter.program) else "N/A",
            "system_metrics": error_info["system_metrics"],
            "context": error_info["context"]
        }
        async with aiofiles.open("pdsxu_debug.json", "a", encoding="utf-8") as f:
            await f.write(json.dumps(debug_info, indent=2) + "\n")
        log.debug(f"Otomatik hata ayıklama: {json.dumps(debug_info, indent=2)}")

    def get_error_log(self) -> List[Dict]:
        """Hata günlüğünü döndürür."""
        with self.lock:
            return self.error_log

    async def save_error_log(self, path: str) -> None:
        """Hata günlüğünü dosyaya asenkron kaydeder."""
        with self.lock:
            try:
                async with aiofiles.open(path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(self.error_log, indent=2))
                log.info(f"Hata günlüğü kaydedildi: {path}")
            except Exception as e:
                log.error(f"Hata günlüğü kaydetme hatası: {str(e)}")
                raise PdsXIOException(f"Hata günlüğü kaydetme hatası: {str(e)}", context={"source": "save_error_log"})

    def clear_error_log(self) -> None:
        """Hata günlüğünü sıfırlar."""
        with self.lock:
            self.error_log.clear()
            self.error_stats.clear()
            self.suppressed_types.clear()
            self.suppressed_codes.clear()
            log.info("Hata günlüğü sıfırlandı")

    def register_handler(self, error_type: str, handler: Callable) -> None:
        """Hata türü için işleyici kaydeder."""
        with self.lock:
            self.handlers[error_type].append(handler)
            log.debug(f"Hata türü işleyici kaydedildi: {error_type}")

    def register_code_handler(self, error_code: str, handler: Callable) -> None:
        """Hata kodu için işleyici kaydeder."""
        with self.lock:
            self.code_handlers[error_code].append(handler)
            log.debug(f"Hata kodu işleyici kaydedildi: {error_code}")

    def unregister_handler(self, error_type: str, handler: Callable) -> None:
        """Hata türü işleyicisini kaldırır."""
        with self.lock:
            if error_type in self.handlers and handler in self.handlers[error_type]:
                self.handlers[error_type].remove(handler)
                log.debug(f"Hata türü işleyici kaldırıldı: {error_type}")
            else:
                log.warning(f"Hata türü işleyici bulunamadı: {error_type}")

    def unregister_code_handler(self, error_code: str, handler: Callable) -> None:
        """Hata kodu işleyicisini kaldırır."""
        with self.lock:
            if error_code in self.code_handlers and handler in self.code_handlers[error_code]:
                self.code_handlers[error_code].remove(handler)
                log.debug(f"Hata kodu işleyici kaldırıldı: {error_code}")
            else:
                log.warning(f"Hata kodu işleyici bulunamadı: {error_code}")

    def suppress_error(self, error_type: str = None, error_code: str = None) -> None:
        """Hata türü veya kodunu bastırır."""
        with self.lock:
            if error_type:
                self.suppressed_types.add(error_type)
                log.info(f"Hata türü bastırıldı: {error_type}")
            if error_code:
                self.suppressed_codes.add(error_code)
                log.info(f"Hata kodu bastırıldı: {error_code}")

    def unsuppress_error(self, error_type: str = None, error_code: str = None) -> None:
        """Hata türü veya kodunun bastırılmasını kaldırır."""
        with self.lock:
            if error_type:
                self.suppressed_types.discard(error_type)
                log.info(f"Hata türü bastırma kaldırıldı: {error_type}")
            if error_code:
                self.suppressed_codes.discard(error_code)
                log.info(f"Hata kodu bastırma kaldırıldı: {error_code}")

    def enable_debug(self) -> None:
        """Hata ayıklama modunu etkinleştirir."""
        with self.lock:
            self.debug_mode = True
            log.info("Hata ayıklama modu etkinleştirildi")

    def disable_debug(self) -> None:
        """Hata ayıklama modunu devre dışı bırakır."""
        with self.lock:
            self.debug_mode = False
            log.info("Hata ayıklama modu devre dışı bırakıldı")

    def enable_trace(self) -> None:
        """Hata izleme modunu etkinleştirir."""
        with self.lock:
            self.trace_mode = True
            log.info("Hata izleme modu etkinleştirildi")

    def disable_trace(self) -> None:
        """Hata izleme modunu devre dışı bırakır."""
        with self.lock:
            self.trace_mode = False
            log.info("Hata izleme modu devre dışı bırakıldı")

    def get_error_stats(self) -> Dict:
        """Hata istatistiklerini döndürür."""
        with self.lock:
            return {k: dict(v) for k, v in self.error_stats.items()}

    def analyze_error(self, error_type: str = None, error_code: str = None) -> Dict:
        """Hata türü veya kodunu derinlemesine analiz eder."""
        with self.lock:
            errors = self.error_log
            if error_type:
                errors = [e for e in errors if e["type"] == error_type]
            if error_code:
                errors = [e for e in errors if e["code"] == error_code]
            if not errors:
                return {"status": "no_errors", "count": 0, "details": {}}
            
            timestamps = [e["timestamp"] for e in errors]
            modules = [e["context"].get("module", "unknown") for e in errors]
            codes = [e["code"] for e in errors]
            return {
                "status": "analyzed",
                "count": len(errors),
                "frequency": len(errors) / (max(timestamps) - min(timestamps) + 1e-6),
                "modules": {m: modules.count(m) for m in set(modules)},
                "codes": {c: codes.count(c) for c in set(codes)},
                "last_occurrence": max(timestamps)
            }

    def visualize_error(self, error_type: str = None, error_code: str = None, output_path: str = "error_graph", format: str = "png") -> None:
        """Hata zincirini görselleştirir (graphviz)."""
        with self.lock:
            dot = graphviz.Digraph(comment=f"Error Chain for {error_type or error_code}", format=format)
            errors = self.error_log
            if error_type:
                errors = [e for e in errors if e["type"] == error_type]
            if error_code:
                errors = [e for e in errors if e["code"] == error_code]
            for i, error in enumerate(errors):
                node_id = f"error_{i}"
                label = f"{error['type']} (Kod: {error['code']})\n{error['message']}\n{time.ctime(error['timestamp'])}"
                dot.node(node_id, label)
                if i > 0 and error.get("cause"):
                    dot.edge(f"error_{i-1}", node_id, label="cause")
            dot.render(output_path, cleanup=True)
            log.info(f"Hata zinciri görselleştirildi: {output_path}.{format}")

    # Deneysel/Bilimsel İşlevler
    def neural_error_forecast(self, history_window: int = 100) -> Dict:
        """Nöral ağlarla hata oluşum tahmini."""
        with self.lock:
            if len(self.error_log) < history_window:
                return {"prediction": "Yetersiz veri", "confidence": 0.0}
            
            X = np.array([log["timestamp"] for log in self.error_log[-history_window:]])
            X = (X - X.min()) / (X.max() - X.min() + 1e-6)
            X = X.reshape((1, history_window, 1))
            
            prediction = self.neural_model.predict(X, verbose=0)[0][0]
            confidence = float(prediction)
            result = "Hata bekleniyor" if prediction > 0.5 else "Normal"
            
            log.debug(f"Nöral hata tahmini: {result}, güven: {confidence}")
            return {"prediction": result, "confidence": confidence}

    def chaos_error_dynamics(self, error_type: str = None, error_code: str = None) -> List[Dict]:
        """Kaotik dinamiklerle hata zincir analizi."""
        with self.lock:
            errors = self.error_log
            if error_type:
                errors = [e for e in errors if e["type"] == error_type]
            if error_code:
                errors = [e for e in errors if e["code"] == error_code]
            chain = []
            for error in errors:
                chaos_factor = random.uniform(0.1, 0.9)  # Mock kaotik faktör
                chain.append({
                    "message": error["message"],
                    "code": error["code"],
                    "timestamp": error["timestamp"],
                    "context": error["context"],
                    "chaos_factor": chaos_factor
                })
            log.debug(f"Kaotik zincir analizi: {error_type or error_code}, {len(chain)} hata")
            return chain

    def quantum_error_modeling(self, error_types: List[str] = None, error_codes: List[str] = None) -> Dict:
        """Kuantum simülasyonu ile hata ilişkisi modelleme."""
        with self.lock:
            correlations = {}
            items = error_types or error_codes or []
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    correlations[f"{item1}_{item2}"] = random.uniform(0.0, 1.0)  # Mock kuantum korelasyon
            log.debug(f"Kuantum hata simülasyonu: {correlations}")
            return correlations

    def genetic_error_recovery(self, error_info: Dict) -> Dict:
        """Genetik algoritmalarla hata kurtarma stratejileri."""
        with self.lock:
            strategies = ["retry", "skip", "log", "abort", "fallback", "reconfigure"]
            fitness_scores = [random.uniform(0.7, 0.95) for _ in strategies]
            best_strategy = strategies[np.argmax(fitness_scores)]
            log.debug(f"Genetik hata optimizasyonu: {best_strategy}")
            return {
                "error": error_info["message"],
                "code": error_info["code"],
                "strategy": best_strategy,
                "fitness": max(fitness_scores)
            }

    def blockchain_error_integrity(self, error_info: Dict) -> str:
        """Blockchain tabanlı hata günlüğü doğrulama."""
        with self.lock:
            error_str = json.dumps(error_info, sort_keys=True)
            error_hash = sha256(error_str.encode("utf-8")).hexdigest()
            log.debug(f"Blockchain hata doğrulaması: {error_hash}")
            return error_hash

    def parse_exception_command(self, command: str) -> None:
        """Hata komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("ERROR "):
                match = re.match(r"ERROR\s+\"([^\"]+)\"\s*(?:CODE\s+(\w+))?", command, re.IGNORECASE)
                if match:
                    message, code = match.groups()
                    raise PdsXException(
                        message,
                        context={
                            "source": "parse_exception_command",
                            "lang": self.interpreter.language,
                            "module": "exception",
                            "line_no": self.interpreter.program_counter
                        },
                        code=code
                    )
                raise PdsXSyntaxError("ERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("MONERROR "):
                match = re.match(r"MONERROR\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    self.interpreter.current_scope()[var_name] = self.get_error_stats()
                else:
                    raise PdsXSyntaxError("MONERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("ANALYZEERROR "):
                match = re.match(r"ANALYZEERROR\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))", command, re.IGNORECASE)
                if match:
                    error_type, error_code = match.groups()
                    self.interpreter.current_scope()["_ANALYSIS"] = self.analyze_error(error_type, error_code)
                else:
                    raise PdsXSyntaxError("ANALYZEERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("VISUALIZEERROR "):
                match = re.match(r"VISUALIZEERROR\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))\s+AS\s+(\S+)(?:\s+FORMAT\s+(\w+))?", command, re.IGNORECASE)
                if match:
                    error_type, error_code, output_path, fmt = match.groups()
                    fmt = fmt or "png"
                    self.visualize_error(error_type, error_code, output_path, fmt)
                else:
                    raise PdsXSyntaxError("VISUALIZEERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("SUPPRESSERROR "):
                match = re.match(r"SUPPRESSERROR\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))", command, re.IGNORECASE)
                if match:
                    error_type, error_code = match.groups()
                    self.suppress_error(error_type, error_code)
                else:
                    raise PdsXSyntaxError("SUPPRESSERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("UNSUPPRESSERROR "):
                match = re.match(r"UNSUPPRESSERROR\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))", command, re.IGNORECASE)
                if match:
                    error_type, error_code = match.groups()
                    self.unsuppress_error(error_type, error_code)
                else:
                    raise PdsXSyntaxError("UNSUPPRESSERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("CATCHERROR "):
                match = re.match(r"CATCHERROR\s+CODE\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    error_code, var_name = match.groups()
                    self.interpreter.current_scope()[var_name] = next((e for e in self.error_log if e["code"] == error_code), None)
                else:
                    raise PdsXSyntaxError("CATCHERROR komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper == "TRACE ON":
                self.enable_trace()
            elif command_upper == "TRACE OFF":
                self.disable_trace()
            elif command_upper == "DEBUG ON":
                self.enable_debug()
            elif command_upper == "DEBUG OFF":
                self.disable_debug()
            elif command_upper == "CLEARERROR":
                self.clear_error_log()
            elif command_upper.startswith("TRY "):
                match = re.match(r"TRY\s+(.+)\s+CATCH\s+(.+)(?:\s+FINALLY\s+(.+))?", command, re.IGNORECASE)
                if match:
                    try_block, catch_block, finally_block = match.groups()
                    try:
                        self.interpreter.execute_command(try_block)
                    except PdsXException as e:
                        self.interpreter.current_scope()["_ERROR"] = {
                            "message": str(e),
                            "type": type(e).__name__,
                            "code": getattr(e, "code", "ERR_UNKNOWN"),
                            "context": getattr(e, "context", {})
                        }
                        self.interpreter.execute_command(catch_block)
                    finally:
                        if finally_block:
                            self.interpreter.execute_command(finally_block)
                else:
                    raise PdsXSyntaxError("TRY komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("ON ERROR GOTO "):
                match = re.match(r"ON ERROR GOTO\s+(\w+)(?:\s+CODE\s+(\w+))?", command, re.IGNORECASE)
                if match:
                    label, error_code = match.groups()
                    handler = lambda info: self.interpreter.execute_command(f"GOTO {label}")
                    if error_code:
                        self.register_code_handler(error_code, handler)
                    else:
                        self.register_handler("PdsXException", handler)
                else:
                    raise PdsXSyntaxError("ON ERROR GOTO komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("REGISTER ERROR HANDLER "):
                match = re.match(r"REGISTER ERROR HANDLER\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    error_type, error_code, handler_name = match.groups()
                    handler = lambda info: self.interpreter.execute_command(handler_name)
                    if error_type:
                        self.register_handler(error_type, handler)
                    if error_code:
                        self.register_code_handler(error_code, handler)
                else:
                    raise PdsXSyntaxError("REGISTER ERROR HANDLER komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            elif command_upper.startswith("UNREGISTER ERROR HANDLER "):
                match = re.match(r"UNREGISTER ERROR HANDLER\s+(?:TYPE\s+(\w+)|CODE\s+(\w+))\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    error_type, error_code, handler_name = match.groups()
                    handler = lambda info: self.interpreter.execute_command(handler_name)
                    if error_type:
                        self.unregister_handler(error_type, handler)
                    if error_code:
                        self.unregister_code_handler(error_code, handler)
                else:
                    raise PdsXSyntaxError("UNREGISTER ERROR HANDLER komutunda sözdizimi hatası", context={"source": "parse_exception_command"})
            else:
                raise PdsXSyntaxError(f"Bilinmeyen hata komutu: {command}", context={"source": "parse_exception_command"})
        except PdsXException as e:
            asyncio.create_task(self.handle_error(e))
            raise
        except Exception as e:
            asyncio.create_task(self.handle_error(e))
            raise PdsXRuntimeError(f"Hata komut hatası: {str(e)}", context={"source": "parse_exception_command"})

if __name__ == "__main__":
    print("pdsx_exception2.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")    