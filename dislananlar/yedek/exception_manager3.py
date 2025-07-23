# exception_manager.py - PDS-X BASIC v15 Hata Yönetim Modülü
# Version: 1.5.4
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import traceback
import threading
import asyncio
import sys
import time
import numpy as np
import os
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple, Type
from functools import lru_cache
from datetime import datetime
try:
    from cython import compiled
except ImportError:
    compiled = False

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("exception_manager")

# Cython için statik tipli yardımcı fonksiyon
if compiled:
    from cython.cimports import libc
    @cython.cfunc
    def fast_format_error(code: str, msg: str, context: Dict[str, Any]) -> str:
        return f"[{code}] {msg} | Context: {context}"
else:
    def fast_format_error(code: str, msg: str, context: Dict[str, Any]) -> str:
        return f"[{code}] {msg} | Context: {context}"

class PdsXException(Exception):
    """PDS-X BASIC v15 temel hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stack_trace = "".join(traceback.format_tb(sys.exc_info()[2])) if sys.exc_info()[2] else ""
        super().__init__(self.message)

    def enrich_context(self, additional_context: Dict[str, Any]) -> None:
        self.context.update(additional_context)
        log.debug(f"Hata bağlamı zenginleştirildi: {additional_context}")

    def get_formatted_error(self, lang: str = "tr") -> str:
        msg = self.message if lang == "tr" else f"Error: {self.message}"
        return fast_format_error(self.code, msg, self.context)

    def log_details(self) -> None:
        log.error(f"Hata: {self.get_formatted_error()} | Stack Trace: {self.stack_trace}")

class PdsXSyntaxError(PdsXException):
    """Sözdizimi hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.suggestion = self._generate_suggestion()

    def _generate_suggestion(self) -> Optional[str]:
        source = self.context.get("source", "")
        if source == "DIM":
            return "DIM yazımını kontrol edin: DIM var_name AS type [value] veya DIM var_name AS type ... END type"
        elif source == "FOR":
            return "FOR döngüsünü kontrol edin: FOR var = start TO end [STEP step] ... NEXT var"
        elif source == "NEXT":
            return "NEXT talimatı için eşleşen FOR döngüsü olduğundan emin olun"
        elif source == "SART":
            return "SART komutunu kontrol edin: SART condition ATLA pipe_id label"
        elif source == "TRY":
            return "TRY bloğunu kontrol edin: TRY ... CATCH [type] ... FINALLY ... END TRY"
        elif source == "ON ERROR":
            return "ON ERROR yazımını kontrol edin: ON ERROR GOTO label veya ON ERROR GOSUB sub_name"
        elif source == "CLASS" or source == "CLAZZ":
            return "Sınıf tanımını kontrol edin: CLASS/CLAZZ DEFINE \"name\" [parent] var_name"
        elif source == "YAPI":
            return "YAPI tanımını kontrol edin: YAPI name ... END YAPI"
        elif source == "FUNC":
            return "Fonksiyonel komut yazımını kontrol edin: FUNC LAMBDA/OMEGA/GAMMA ..."
        elif source == "SQLITE":
            return "SQLite komutunu kontrol edin: SQLITE EXECUTE conn_id \"query\" [params]"
        elif source == "TREE":
            return "Ağaç komutunu kontrol edin: TREE CREATE tree_id AS type value var_name"
        elif source == "NET":
            return "Ağ komutunu kontrol edin: NET GET \"url\" var_name"
        elif source == "LOWLEVEL":
            return "Düşük seviyeli komut yazımını kontrol edin: BITSET ptr, field, value, bits"
        elif source == "NLP":
            return "NLP komutunu kontrol edin: NLP ANALYZE \"text\" [lang]"
        elif source == "DB":
            return "Veritabanı komutunu kontrol edin: OPEN DATABASE type conn_id params"
        elif source == "EVENT":
            return "Olay komutunu kontrol edin: EVENT REGISTER event_name \"handler\" [priority]"
        elif source == "GUI":
            return "GUI komutunu kontrol edin: WINDOW name WIDTH=width,HEIGHT=height,TITLE=\"title\""
        elif source == "CONCURRENCY":
            return "Eşzamanlılık komutunu kontrol edin: THREAD thread_id, sub_name"
        elif source == "LOGIC":
            return "Mantıksal programlama komutunu kontrol edin: FACT fact veya RULE head :- body"
        elif source == "JIT":
            return "JIT komutunu kontrol edin: COMPILE language code AS \"output_name\""
        return None

    def suggest_fix(self) -> Optional[str]:
        return self.suggestion

    def analyze_syntax(self) -> Dict[str, Any]:
        analysis = {
            "line_no": self.context.get("line_no", -1),
            "source": self.context.get("source", "unknown"),
            "suggestion": self.suggestion or "Bilinmeyen sözdizimi hatası",
            "code_snippet": self.context.get("code_snippet", "")
        }
        log.debug(f"Sözdizimi analizi: {analysis}")
        return analysis

class PdsXRuntimeError(PdsXException):
    """Çalışma zamanı hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.variable_state = self._capture_variable_state()

    def _capture_variable_state(self) -> Dict[str, Any]:
        return {
            "scope": {k: str(v)[:100] for k, v in self.interpreter.current_scope().items() if isinstance(v, (int, float, str, bool))},
            "program_counter": self.interpreter.program_counter
        }

    def log_stack_trace(self) -> str:
        return self.stack_trace

    def analyze_runtime(self) -> Dict[str, Any]:
        analysis = {
            "variable_state": self.variable_state,
            "stack_trace": self.stack_trace,
            "opcode": self.context.get("opcode", "unknown"),
            "line_no": self.context.get("line_no", -1)
        }
        log.debug(f"Çalışma zamanı analizi: {analysis}")
        return analysis

class PdsXTypeError(PdsXException):
    """Tip uyumsuzlukları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.expected_type = self.context.get("expected_type", "unknown")
        self.actual_type = self.context.get("actual_type", "unknown")

    def get_type_suggestion(self) -> str:
        return f"Bu işlem için {self.expected_type} tipi bekleniyor, ancak {self.actual_type} bulundu."

    def compare_types(self) -> Dict[str, str]:
        comparison = {
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
            "suggestion": self.get_type_suggestion()
        }
        log.debug(f"Tip karşılaştırması: {comparison}")
        return comparison

class PdsXValueError(PdsXException):
    """Değer hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.value = self.context.get("value", None)

    def validate_value(self, value: Any = None) -> bool:
        value = value if value is not None else self.value
        valid = value is not None and (not isinstance(value, (int, float)) or value != 0)
        if not valid:
            log.debug(f"Geçersiz değer: {value}")
        return valid

    def analyze_value(self) -> Dict[str, Any]:
        analysis = {
            "value": self.value,
            "is_valid": self.validate_value(),
            "constraint": self.context.get("constraint", "unknown")
        }
        log.debug(f"Değer analizi: {analysis}")
        return analysis

class PdsXIOException(PdsXException):
    """Dosya/giriş-çıkış hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.file_path = self.context.get("file_path", "unknown")
        self.operation = self.context.get("operation", "unknown")

    def retry_possible(self) -> bool:
        return "file_not_found" not in self.message.lower() and "permission" not in self.message.lower()

    def check_file_status(self) -> Dict[str, Any]:
        status = {
            "file_path": self.file_path,
            "exists": os.path.exists(self.file_path) if self.file_path != "unknown" else False,
            "operation": self.operation,
            "retry_possible": self.retry_possible()
        }
        log.debug(f"Dosya durumu: {status}")
        return status

class PdsXNetworkError(PdsXException):
    """Ağ hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.retry_count: int = context.get("retry_count", 0) if context else 0
        self.url = self.context.get("url", "unknown")
        self.method = self.context.get("method", "unknown")

    def increment_retry(self) -> None:
        self.retry_count += 1
        self.context["retry_count"] = self.retry_count
        log.debug(f"Yeniden deneme sayısı: {self.retry_count}")

    def analyze_network(self) -> Dict[str, Any]:
        analysis = {
            "url": self.url,
            "method": self.method,
            "retry_count": self.retry_count,
            "status_code": self.context.get("status_code", "unknown")
        }
        log.debug(f"Ağ analizi: {analysis}")
        return analysis

class PdsXDatabaseError(PdsXException):
    """Veritabanı hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.db_type = self.context.get("db_type", "unknown")
        self.query = self.context.get("query", "unknown")

    def analyze_database(self) -> Dict[str, Any]:
        analysis = {
            "db_type": self.db_type,
            "query": self.query[:100],
            "connection_id": self.context.get("connection_id", "unknown")
        }
        log.debug(f"Veritabanı analizi: {analysis}")
        return analysis

class PdsXScientificError(PdsXException):
    """Bilimsel ve deneysel işlemler için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.computation_type = self.context.get("computation_type", "unknown")

    def validate_computation(self) -> bool:
        if self.computation_type == "QUANTUM_STATE":
            amplitudes = self.context.get("amplitudes", [])
            total_prob = sum(abs(a)**2 for a in amplitudes)
            return np.isclose(total_prob, 1.0, atol=1e-6)
        elif self.computation_type == "NEURAL_TENSOR":
            tensor = self.context.get("tensor", [])
            return not np.any(np.isnan(tensor))
        elif self.computation_type == "CHAOS_FIELD":
            initial_conditions = self.context.get("initial_conditions", [])
            return len(initial_conditions) >= 3
        elif self.computation_type == "HOLO_DATA":
            data = self.context.get("data", b"")
            return len(data) > 0
        elif self.computation_type == "BLOCKCHAIN_LEDGER":
            ledger = self.context.get("ledger", {})
            return len(ledger) > 0
        return True

    def analyze_computation(self) -> Dict[str, Any]:
        analysis = {
            "computation_type": self.computation_type,
            "is_valid": self.validate_computation(),
            "parameters": self.context.get("parameters", {})
        }
        log.debug(f"Hesaplama analizi: {analysis}")
        return analysis

class PdsXDataStructureError(PdsXException):
    """Veri yapıları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.structure_type = self.context.get("structure_type", "unknown")

    def check_structure(self) -> Dict[str, Any]:
        status = {
            "structure_type": self.structure_type,
            "size": self.context.get("size", -1),
            "index": self.context.get("index", None),
            "key": self.context.get("key", None)
        }
        log.debug(f"Veri yapısı durumu: {status}")
        return status

    def suggest_structure_fix(self) -> Optional[str]:
        if self.structure_type == "LIST" and "index" in self.context:
            return f"Liste indeksi {self.context['index']} sınırlar dışında; liste boyutunu kontrol edin."
        elif self.structure_type == "DICT" and "key" in self.context:
            return f"Sözlük anahtarı {self.context['key']} bulunamadı; anahtarları kontrol edin."
        elif self.structure_type == "STACK" or self.structure_type == "QUEUE":
            return "Yığın/kuyruk boş; veri eklemeyi deneyin."
        elif self.structure_type == "TREE" or self.structure_type == "GRAPH":
            return "Ağaç/grafik yapısını kontrol edin; düğüm veya kenar eksik olabilir."
        elif self.structure_type == "STREAM":
            return "Akış sonuna ulaşıldı; akış konumunu sıfırlamayı deneyin."
        return None

class PdsXClassError(PdsXException):
    """Sınıf ve nesne hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.class_type = self.context.get("class_type", "unknown")
        self.access_level = self.context.get("access_level", "unknown")

    def check_access(self) -> bool:
        return self.access_level in ("PUBLIC", "PROTECTED") or self._is_authorized()

    def _is_authorized(self) -> bool:
        return self.context.get("module", "") == self.interpreter.current_module

    def analyze_class(self) -> Dict[str, Any]:
        analysis = {
            "class_type": self.class_type,
            "access_level": self.access_level,
            "method": self.context.get("method", "unknown"),
            "property": self.context.get("property", "unknown")
        }
        log.debug(f"Sınıf analizi: {analysis}")
        return analysis

class PdsXFunctionalError(PdsXException):
    """Fonksiyonel programlama hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.monad_type = self.context.get("monad_type", "unknown")

    def validate_monad(self) -> bool:
        if self.monad_type == "MAYBE":
            return self.context.get("value", None) is not None
        elif self.monad_type == "EITHER":
            return self.context.get("left", None) is None
        return True

    def analyze_monad(self) -> Dict[str, Any]:
        analysis = {
            "monad_type": self.monad_type,
            "is_valid": self.validate_monad(),
            "value": self.context.get("value", None)
        }
        log.debug(f"Monad analizi: {analysis}")
        return analysis

class PdsXMemoryError(PdsXException):
    """Bellek hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.ptr = self.context.get("ptr", "unknown")

    def analyze_memory(self) -> Dict[str, Any]:
        analysis = {
            "ptr": self.ptr,
            "size": self.context.get("size", -1),
            "operation": self.context.get("operation", "unknown")
        }
        log.debug(f"Bellek analizi: {analysis}")
        return analysis

class PdsXEventError(PdsXException):
    """Olay yönetimi hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.event_name = self.context.get("event_name", "unknown")

    def analyze_event(self) -> Dict[str, Any]:
        analysis = {
            "event_name": self.event_name,
            "handler": self.context.get("handler", "unknown"),
            "priority": self.context.get("priority", 0)
        }
        log.debug(f"Olay analizi: {analysis}")
        return analysis

class PdsXGUIError(PdsXException):
    """GUI hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.window_name = self.context.get("window_name", "unknown")
        self.widget_name = self.context.get("widget_name", "unknown")

    def analyze_gui(self) -> Dict[str, Any]:
        analysis = {
            "window_name": self.window_name,
            "widget_name": self.widget_name,
            "event": self.context.get("event", "unknown")
        }
        log.debug(f"GUI analizi: {analysis}")
        return analysis

class PdsXConcurrencyError(PdsXException):
    """Eşzamanlılık hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.task_id = self.context.get("task_id", "unknown")

    def analyze_concurrency(self) -> Dict[str, Any]:
        analysis = {
            "task_id": self.task_id,
            "task_type": self.context.get("task_type", "unknown"),
            "status": self.context.get("status", "unknown")
        }
        log.debug(f"Eşzamanlılık analizi: {analysis}")
        return analysis

class PdsXLogicError(PdsXException):
    """Mantıksal programlama hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.goal = self.context.get("goal", "unknown")

    def analyze_logic(self) -> Dict[str, Any]:
        analysis = {
            "goal": self.goal,
            "bindings": self.context.get("bindings", {})
        }
        log.debug(f"Mantıksal programlama analizi: {analysis}")
        return analysis

class PdsXJITError(PdsXException):
    """JIT derleme hataları için hata sınıfı."""
    def __init__(self, message: str, code: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)
        self.language = self.context.get("language", "unknown")

    def analyze_jit(self) -> Dict[str, Any]:
        analysis = {
            "language": self.language,
            "output_name": self.context.get("output_name", "unknown")
        }
        log.debug(f"JIT analizi: {analysis}")
        return analysis

class ExceptionManager:
    """PDS-X BASIC v15 hata yönetim motoru."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.Lock()
        self.error_cache: Dict[str, str] = {}
        self.language: str = "tr"
        self.error_stats: Dict[str, int] = defaultdict(int)
        self.try_stack: List[Dict] = []
        self.error_handler: Optional[Dict] = None
        self.error_registry: Dict[str, Dict[str, str]] = {
            # Genel Hatalar
            "GENERIC001": {"tr": "Bilinmeyen hata", "en": "Unknown error"},
            "LANG001": {"tr": "Geçersiz dil", "en": "Invalid language"},
            "GENERIC002": {"tr": "Geçersiz hata işleyici tipi", "en": "Invalid error handler type"},
            "GENERIC003": {"tr": "Geçersiz RESUME modu", "en": "Invalid RESUME mode"},
            # Yürütme Hataları
            "EXEC001": {"tr": "Bilinmeyen bayt kodu", "en": "Unknown bytecode"},
            "EXEC002": {"tr": "Bayt kodu yürütme hatası", "en": "Bytecode execution error"},
            # Döngü Hataları
            "FOR001": {"tr": "Adım sıfır olamaz", "en": "Step cannot be zero"},
            "FOR002": {"tr": "FOR döngü hatası", "en": "FOR loop error"},
            "FOREACH001": {"tr": "Geçersiz koleksiyon", "en": "Invalid collection"},
            "FOREACH002": {"tr": "FOR EACH döngü hatası", "en": "FOR EACH loop error"},
            "NEXT001": {"tr": "Kapatılacak döngü bulunamadı", "en": "No loop found to close"},
            "NEXT002": {"tr": "Geçersiz döngü tipi", "en": "Invalid loop type"},
            # Değişken Tanımlama Hataları
            "DIM001": {"tr": "Geçersiz veri tipi", "en": "Invalid data type"},
            "DIM002": {"tr": "DIM tanımlama hatası", "en": "DIM definition error"},
            "UNDIM001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            # Yapı Hataları
            "SETFIELD001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "SETFIELD002": {"tr": "Geçersiz yapı", "en": "Invalid structure"},
            "GETFIELD001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "GETFIELD002": {"tr": "Geçersiz yapı", "en": "Invalid structure"},
            "ADDFIELD001": {"tr": "Geçersiz veri tipi", "en": "Invalid data type"},
            "ADDFIELD002": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "ADDFIELD003": {"tr": "Geçersiz yapı", "en": "Invalid structure"},
            "REMOVEFIELD001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "REMOVEFIELD002": {"tr": "Geçersiz yapı", "en": "Invalid structure"},
            "REMOVEFIELD003": {"tr": "Alan bulunamadı", "en": "Field not found"},
            # Nesne Hataları
            "NEWOBJ001": {"tr": "Sınıf bulunamadı", "en": "Class not found"},
            "NEWOBJ002": {"tr": "Geçersiz parametre sayısı", "en": "Invalid parameter count"},
            "COUNTOBJ001": {"tr": "Tip bulunamadı", "en": "Type not found"},
            # API ve DLL Hataları
            "INSPOBJ001": {"tr": "Nesne bulunamadı", "en": "Object not found"},            
            "CALLAPI001": {"tr": "HTTP isteği başarısız", "en": "HTTP request failed"},
            "CALLAPI002": {"tr": "HTTP isteği başarısız", "en": "HTTP request failed"},
            "CALLAPI003": {"tr": "API çağrısı hatası", "en": "API call error"},
            "CALLDLL001": {"tr": "DLL çağrısı hatası", "en": "DLL call error"},
            # Boru Hattı Hataları
            "SART001": {"tr": "Boru hattı bulunamadı", "en": "Pipeline not found"},
            "SART002": {"tr": "Etiket bulunamadı", "en": "Label not found"},
            "SART003": {"tr": "SART değerlendirme hatası", "en": "SART evaluation error"},
            # Değişken ve Kapsam Hataları
            "ALIAS001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "RESTRICT001": {"tr": "Geçersiz kapsam", "en": "Invalid scope"},
            "CLEAR_BASIC001": {"tr": "Geçersiz kapsam", "en": "Invalid scope"},
            "SEC_VAR001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "MON_VAR001": {"tr": "Değişken izleme hatası", "en": "Variable monitoring error"},
            # Tip Dönüşüm Hataları
            "CONVERT001": {"tr": "Geçersiz tip", "en": "Invalid type"},
            "CONVERT002": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "CONVERT003": {"tr": "Tip dönüşüm hatası", "en": "Type conversion error"},
            "CAST001": {"tr": "Geçersiz tip", "en": "Invalid type"},
            "CAST002": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "CAST003": {"tr": "Tip dönüşüm hatası", "en": "Type conversion error"},
            # Sınıf ve YAPI Hataları
            "CLASS001": {"tr": "SUB tanımı hatalı", "en": "SUB definition error"},
            "CLASS002": {"tr": "Geçersiz dönüş tipi", "en": "Invalid return type"},
            "CLASS003": {"tr": "FUNCTION tanımı hatalı", "en": "FUNCTION definition error"},
            "CLASS004": {"tr": "Geçersiz özellik tipi", "en": "Invalid property type"},
            "CLASS005": {"tr": "PROP tanımı hatalı", "en": "PROP definition error"},
            "CLAZZ001": {"tr": "CLAZZ komut hatası", "en": "CLAZZ command error"},
            "YAPI001": {"tr": "SUB tanımı hatalı", "en": "SUB definition error"},
            "YAPI002": {"tr": "Geçersiz dönüş tipi", "en": "Invalid return type"},
            "YAPI003": {"tr": "FUNCTION tanımı hatalı", "en": "FUNCTION definition error"},
            "YAPI004": {"tr": "Geçersiz dönüş tipi", "en": "Invalid return type"},
            "YAPI005": {"tr": "GAMMA tanımı hatalı", "en": "GAMMA definition error"},
            "YAPI006": {"tr": "Geçersiz dönüş tipi", "en": "Invalid return type"},
            "YAPI007": {"tr": "OMEGA tanımı hatalı", "en": "OMEGA definition error"},
            "YAPI008": {"tr": "Geçersiz özellik tipi", "en": "Invalid property type"},
            "YAPI009": {"tr": "DIM tanımı hatalı", "en": "DIM definition error"},
            "GAMMA001": {"tr": "YAPI bulunamadı", "en": "YAPI not found"},
            "GAMMA002": {"tr": "GAMMA fonksiyonu bulunamadı", "en": "GAMMA function not found"},
            "GAMMA003": {"tr": "GAMMA yürütme hatası", "en": "GAMMA execution error"},
            "OMEGA001": {"tr": "YAPI bulunamadı", "en": "YAPI not found"},
            "OMEGA002": {"tr": "OMEGA fonksiyonu bulunamadı", "en": "OMEGA function not found"},
            "OMEGA003": {"tr": "OMEGA yürütme hatası", "en": "OMEGA execution error"},
            # Deneysel Özellik Hataları
            "QUANTUM001": {"tr": "Geçersiz kuantum durumu", "en": "Invalid quantum state"},
            "QUANTUM002": {"tr": "Kuantum durum boyutları uyuşmuyor", "en": "Quantum state dimensions mismatch"},
            "NEURAL001": {"tr": "Geçersiz nöral tensor", "en": "Invalid neural tensor"},
            "NEURAL002": {"tr": "Nöral işleme hatası", "en": "Neural processing error"},
            "CHAOS001": {"tr": "Geçersiz kaos alanı", "en": "Invalid chaos field"},
            "CHAOS002": {"tr": "Kaotik desen hatası", "en": "Chaotic pattern error"},
            "HOLO001": {"tr": "Holografik veri bozulması", "en": "Holographic data corruption"},
            "HOLO002": {"tr": "Holografik kodlama hatası", "en": "Holographic encoding error"},
            "BLOCKCHAIN001": {"tr": "Geçersiz blockchain defteri", "en": "Invalid blockchain ledger"},
            "BLOCKCHAIN002": {"tr": "Blockchain doğrulama hatası", "en": "Blockchain validation error"},
            # Diğer Hatalar
            "PRINT001": {"tr": "PRINT değerlendirme hatası", "en": "PRINT evaluation error"},
            "GOTO001": {"tr": "Etiket bulunamadı", "en": "Label not found"},
            "GOSUB001": {"tr": "Etiket bulunamadı", "en": "Label not found"},
            "RETURN001": {"tr": "Geri dönülecek yordam yok", "en": "No subroutine to return from"},
            "READ001": {"tr": "Veri listesi sonu", "en": "End of data list"},
            "CHAIN001": {"tr": "Program yükleme hatası", "en": "Program loading error"},
            "COMMON001": {"tr": "Değişken bulunamadı", "en": "Variable not found"},
            "DECLARE001": {"tr": "Geçersiz dönüş tipi", "en": "Invalid return type"},
            "EXIT001": {"tr": "Çıkılacak döngü bulunamadı", "en": "No loop to exit"},
            "EXIT002": {"tr": "Çıkılacak yordam bulunamadı", "en": "No subroutine to exit"},
            "END001": {"tr": "Geçersiz END tipi", "en": "Invalid END type"},
            "END002": {"tr": "Kapatılacak döngü bulunamadı", "en": "No loop to close"},
            "END003": {"tr": "Kapatılacak IF bloğu bulunamadı", "en": "No IF block to close"},
            # Veri Yapısı Hataları
            "LIST001": {"tr": "Indeks hatası", "en": "Index error"},
            "DICT001": {"tr": "Anahtar bulunamadı", "en": "Key not found"},
            "SET001": {"tr": "Geçersiz küme işlemi", "en": "Invalid set operation"},
            "STACK001": {"tr": "Yığın boş", "en": "Stack empty"},
            "QUEUE001": {"tr": "Kuyruk boş", "en": "Queue empty"},
            "ARRAY001": {"tr": "Dizi boyut hatası", "en": "Array dimension error"},
            "MATRIX001": {"tr": "Matris boyut uyumsuzluğu", "en": "Matrix dimension mismatch"},
            "TENSOR001": {"tr": "Tensor boyut uyumsuzluğu", "en": "Tensor dimension mismatch"},
            "TREE001": {"tr": "Ağaç düğümü bulunamadı", "en": "Tree node not found"},
            "GRAPH001": {"tr": "Grafik kenarı/düğümü bulunamadı", "en": "Graph edge/node not found"},
            "STREAM001": {"tr": "Akış sonuna ulaşıldı", "en": "Stream end reached"},
            # Modül Spesifik Hatalar
            "EXPORT001": {"tr": "Desteklenmeyen ihracat formatı", "en": "Unsupported export format"},
            "REPORT001": {"tr": "PDF oluşturma hatası", "en": "PDF generation error"},
            "OOP001": {"tr": "Sınıf bulunamadı", "en": "Class not found"},
            "PIPE001": {"tr": "Boru hattı yürütme hatası", "en": "Pipeline execution error"},
            "GUI001": {"tr": "GUI penceresi başlatma hatası", "en": "GUI window initialization error"},
            "ML001": {"tr": "Model eğitimi hatası", "en": "Model training error"},
            "REPLY001": {"tr": "Yanıt komut hatası", "en": "Reply command error"},
            "FUNC001": {"tr": "Fonksiyonel komut hatası", "en": "Functional command error"},
            "SQLITE001": {"tr": "SQLite komut hatası", "en": "SQLite command error"},
            "NET001": {"tr": "Ağ komut hatası", "en": "Network command error"},
            "LOWLEVEL001": {"tr": "Düşük seviyeli komut hatası", "en": "Low-level command error"},
            "NLP001": {"tr": "NLP komut hatası", "en": "NLP command error"},
            "DB001": {"tr": "Veritabanı komut hatası", "en": "Database command error"},
            "EVENT001": {"tr": "Olay komut hatası", "en": "Event command error"},
            "CONCURRENCY001": {"tr": "Eşzamanlılık komut hatası", "en": "Concurrency command error"},
            "LOGIC001": {"tr": "Mantıksal programlama komut hatası", "en": "Logic command error"},
            "JIT001": {"tr": "JIT derleme hatası", "en": "JIT compilation error"},
        }

    @lru_cache(maxsize=8192)
    def get_error_message(self, code: str, lang: str) -> str:
        msg_dict = self.error_registry.get(code, {"tr": "Bilinmeyen hata", "en": "Unknown error"})
        return msg_dict.get(lang, msg_dict["tr"])

    async def handle_error(self, error: Exception) -> None:
        with self.lock:
            error_id = id(error)
            self.interpreter.object_counter["ERROR"] += 1
            self.interpreter.object_registry[error_id] = {
                "type": "ERROR",
                "name": error.__class__.__name__,
                "atom": str(error)[:100]
            }
            self.error_stats[error.__class__.__name__] += 1

        if isinstance(error, PdsXException):
            code = error.code
            msg = self.get_error_message(code, self.language)
            context = error.context
        else:
            code = "GENERIC001"
            msg = self.get_error_message(code, self.language)
            context = {
                "source": "unknown",
                "line_no": self.interpreter.program_counter,
                "opcode": "unknown"
            }

        if isinstance(error, PdsXSyntaxError):
            suggestion = error.suggest_fix()
            if suggestion:
                context["suggestion"] = suggestion
            context["syntax_analysis"] = error.analyze_syntax()
        elif isinstance(error, PdsXRuntimeError):
            context["runtime_analysis"] = error.analyze_runtime()
        elif isinstance(error, PdsXTypeError):
            context["type_comparison"] = error.compare_types()
        elif isinstance(error, PdsXValueError):
            context["value_analysis"] = error.analyze_value()
        elif isinstance(error, PdsXIOException):
            context["file_status"] = error.check_file_status()
        elif isinstance(error, PdsXNetworkError):
            error.increment_retry()
            context["network_analysis"] = error.analyze_network()
        elif isinstance(error, PdsXDatabaseError):
            context["database_analysis"] = error.analyze_database()
        elif isinstance(error, PdsXScientificError):
            context["computation_analysis"] = error.analyze_computation()
        elif isinstance(error, PdsXDataStructureError):
            context["structure_status"] = error.check_structure()
            suggestion = error.suggest_structure_fix()
            if suggestion:
                context["suggestion"] = suggestion
        elif isinstance(error, PdsXClassError):
            context["class_analysis"] = error.analyze_class()
        elif isinstance(error, PdsXFunctionalError):
            context["monad_analysis"] = error.analyze_monad()
        elif isinstance(error, PdsXMemoryError):
            context["memory_analysis"] = error.analyze_memory()
        elif isinstance(error, PdsXEventError):
            context["event_analysis"] = error.analyze_event()
        elif isinstance(error, PdsXGUIError):
            context["gui_analysis"] = error.analyze_gui()
        elif isinstance(error, PdsXConcurrencyError):
            context["concurrency_analysis"] = error.analyze_concurrency()
        elif isinstance(error, PdsXLogicError):
            context["logic_analysis"] = error.analyze_logic()
        elif isinstance(error, PdsXJITError):
            context["jit_analysis"] = error.analyze_jit()

        formatted_error = fast_format_error(code, f"{msg}: {str(error)}", context)
        
        await self.log_error(error, context)
        print(formatted_error, file=sys.stderr)
        
        if self.try_stack:
            try_block = self.try_stack[-1]
            if try_block["catch_type"] is None or isinstance(error, try_block["catch_type"]):
                self.interpreter.current_scope()["ERROR"] = formatted_error
                self.interpreter.execute_command(try_block["catch_block"])
                if try_block["finally_block"]:
                    self.interpreter.execute_command(try_block["finally_block"])
                self.try_stack.pop()
            else:
                self.try_stack.pop()
                await self.handle_error(error)
        elif self.error_handler:
            handler_type = self.error_handler["type"]
            if handler_type == "GOTO":
                self.interpreter.program_counter = self.interpreter.labels.get(self.error_handler["target"], self.interpreter.program_counter)
            elif handler_type == "GOSUB":
                self.interpreter.call_stack.append({"return_pc": self.interpreter.program_counter + 1})
                self.interpreter.program_counter = self.interpreter.labels.get(self.error_handler["target"], self.interpreter.program_counter)
        else:
            await self.recover(error)

    async def recover(self, error: Exception) -> None:
        with self.lock:
            self.interpreter.stack.clear()
            self.interpreter.call_stack.clear()
            
            if isinstance(error, PdsXSyntaxError):
                self.interpreter.running = False
                self.interpreter.paused = True
                log.warning("Sözdizimi hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXIOException):
                if error.retry_possible():
                    log.info("G/Ç hatası: Yeniden deneme mümkün")
                    self.interpreter.program_counter += 1
                else:
                    self.interpreter.running = False
                    self.interpreter.paused = True
                    log.warning("Kritik G/Ç hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXNetworkError):
                if error.retry_count < 3:
                    log.info(f"Ağ hatası: Yeniden deneme #{error.retry_count + 1}")
                    self.interpreter.program_counter += 1
                    await asyncio.sleep(1)
                else:
                    self.interpreter.paused = True
                    log.warning("Ağ hatası: Maksimum yeniden deneme aşıldı")
            elif isinstance(error, PdsXTypeError):
                self.interpreter.program_counter += 1
                log.info("Tip hatası: Bir sonraki talimata geçiliyor")
            elif isinstance(error, PdsXValueError):
                if error.validate_value():
                    self.interpreter.program_counter += 1
                    log.info("Değer hatası: Bir sonraki talimata geçiliyor")
                else:
                    self.interpreter.paused = True
                    log.warning("Kritik değer hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXDatabaseError):
                self.interpreter.paused = True
                log.warning("Veritabanı hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXScientificError):
                if error.validate_computation():
                    self.interpreter.program_counter += 1
                    log.info("Bilimsel hata: Bir sonraki talimata geçiliyor")
                else:
                    self.interpreter.paused = True
                    log.warning("Kritik bilimsel hata: Yürütme durduruldu")
            elif isinstance(error, PdsXDataStructureError):
                suggestion = error.suggest_structure_fix()
                if suggestion:
                    self.interpreter.program_counter += 1
                    log.info(f"Veri yapısı hatası: {suggestion}, devam ediliyor")
                else:
                    self.interpreter.paused = True
                    log.warning("Kritik veri yapısı hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXClassError):
                if error.check_access():
                    self.interpreter.program_counter += 1
                    log.info("Sınıf hatası: Bir sonraki talimata geçiliyor")
                else:
                    self.interpreter.paused = True
                    log.warning("Kritik sınıf hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXFunctionalError):
                if error.validate_monad():
                    self.interpreter.program_counter += 1
                    log.info("Fonksiyonel hata: Bir sonraki talimata geçiliyor")
                else:
                    self.interpreter.paused = True
                    log.warning("Kritik fonksiyonel hata: Yürütme durduruldu")
            elif isinstance(error, PdsXMemoryError):
                self.interpreter.paused = True
                log.warning("Bellek hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXEventError):
                self.interpreter.program_counter += 1
                log.info("Olay hatası: Bir sonraki talimata geçiliyor")
            elif isinstance(error, PdsXGUIError):
                self.interpreter.paused = True
                log.warning("GUI hatası: Yürütme durduruldu")
            elif isinstance(error, PdsXConcurrencyError):
                self.interpreter.program_counter += 1
                log.info("Eşzamanlılık hatası: Bir sonraki talimata geçiliyor")
            elif isinstance(error, PdsXLogicError):
                self.interpreter.program_counter += 1
                log.info("Mantıksal programlama hatası: Bir sonraki talimata geçiliyor")
            elif isinstance(error, PdsXJITError):
                self.interpreter.paused = True
                log.warning("JIT derleme hatası: Yürütme durduruldu")
            else:
                self.interpreter.program_counter += 1
                log.info("Hata sonrası devam ediliyor")
            
            if len(self.error_cache) > 8192:
                self.error_cache.clear()
                log.debug("Hata önbelleği temizlendi")

    def set_language(self, lang: str) -> None:
        if lang not in ("tr", "en"):
            raise PdsXValueError(f"Geçersiz dil: {lang}", code="LANG001")
        with self.lock:
            self.language = lang
        log.debug(f"Hata mesaj dili ayarlandı: {lang}")

    async def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        formatted_error = fast_format_error(
            getattr(error, "code", "GENERIC001"),
            str(error),
            context
        )
        log.error(formatted_error)
        if isinstance(error, PdsXRuntimeError):
            log.error(f"Stack Trace: {error.log_stack_trace()}")
        else:
            log.error(f"Traceback: {''.join(traceback.format_tb(error.__traceback__))}")

    async def register_error(self, code: str, messages: Dict[str, str]) -> None:
        with self.lock:
            self.error_registry[code] = messages
        log.debug(f"Yeni hata kodu kaydedildi: {code}")

    def get_error_stats(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.error_stats)

    async def analyze_error_frequency(self) -> Dict[str, float]:
        total_errors = sum(self.error_stats.values())
        if total_errors == 0:
            return {}
        analysis = {k: (v / total_errors) * 100 for k, v in self.error_stats.items()}
        log.debug(f"Hata sıklık analizi: {analysis}")
        return analysis

    def set_error_handler(self, handler_type: str, target: str) -> None:
        with self.lock:
            if handler_type not in ("GOTO", "GOSUB"):
                raise PdsXValueError(f"Geçersiz hata işleyici tipi: {handler_type}", code="GENERIC002")
            self.error_handler = {"type": handler_type, "target": target}
        log.debug(f"Hata işleyici ayarlandı: type={handler_type}, target={target}")

    def clear_error_handler(self) -> None:
        with self.lock:
            self.error_handler = None
        log.debug("Hata işleyici sıfırlandı")

    def push_try_block(self, catch_type: Optional[Type[Exception]], catch_block: str, finally_block: Optional[str]) -> None:
        with self.lock:
            self.try_stack.append({
                "catch_type": catch_type,
                "catch_block": catch_block,
                "finally_block": finally_block
            })
        log.debug("TRY bloğu yığına eklendi")

    def execute_resume(self, mode: str, target: Optional[str] = None) -> None:
        with self.lock:
            if mode == "RESUME":
                log.debug("RESUME: Hata oluşturan satırdan devam ediliyor")
            elif mode == "RESUME NEXT":
                self.interpreter.program_counter += 1
                log.debug("RESUME NEXT: Bir sonraki satıra geçiliyor")
            elif mode == "RESUME LABEL" and target:
                if target in self.interpreter.labels:
                    self.interpreter.program_counter = self.interpreter.labels[target]
                    log.debug(f"RESUME LABEL: {target} etiketine atlanıyor")
                else:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {target}", code="GOTO001")
            else:
                raise PdsXSyntaxError(f"Geçersiz RESUME modu: {mode}", code="GENERIC003")

if __name__ == "__main__":
    print("exception_manager.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")