# pdsx_exception.py - PDS-X BASIC v14u Hata Yönetim Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import traceback
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import threading
import json
import uuid

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdsx_exception")

class PdsXException(Exception):
    """PDS-X BASIC hata sınıfı."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code or f"ERR_{uuid.uuid4().hex[:8]}"
        self.context = context or {}
        self.timestamp = time.time()
        self.stack_trace = traceback.format_stack()[:-1]  # Çağrı yığınını al, son çağrıyı hariç tut
        self.source = self.context.get("source", "unknown")
        self.line_no = self.context.get("line_no", 0)
        super().__init__(self.message)

    def to_dict(self) -> Dict:
        """Hata bilgisini sözlük formatına dönüştürür."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "source": self.source,
            "line_no": self.line_no,
            "stack_trace": self.stack_trace,
            "context": self.context
        }

    def __str__(self) -> str:
        """Hata mesajını formatlar."""
        return f"[Error {self.error_code}] {self.message} at {self.source}:{self.line_no}\n" + \
               "\n".join(self.stack_trace)

class PdsXSyntaxError(PdsXException):
    """Sözdizimi hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "SYNTAX_ERR", context)

class PdsXRuntimeError(PdsXException):
    """Çalışma zamanı hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "RUNTIME_ERR", context)

class PdsXMemoryError(PdsXException):
    """Bellek hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "MEMORY_ERR", context)

class PdsXTypeError(PdsXException):
    """Veri tipi hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "TYPE_ERR", context)

class PdsXIOException(PdsXException):
    """Giriş/çıkış hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "IO_ERR", context)

class PdsXNetworkError(PdsXException):
    """Ağ hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "NETWORK_ERR", context)

class PdsXDatabaseError(PdsXException):
    """Veritabanı hataları için alt sınıf."""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message, error_code or "DB_ERR", context)

class ExceptionManager:
    """Hata yönetim sistemi."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.error_handlers = defaultdict(list)  # {error_code: [handler, ...]}
        self.global_handler = None
        self.error_log = []  # Hata geçmişi: [(timestamp, error)]
        self.lock = threading.Lock()
        self.metadata = {"pdsx_exception": {"version": "1.0.0", "dependencies": []}}
        self.max_log_size = 1000  # Maksimum hata kaydı sayısı
        self.error_codes = {
            "SYNTAX_ERR": "Sözdizimi hatası",
            "RUNTIME_ERR": "Çalışma zamanı hatası",
            "MEMORY_ERR": "Bellek hatası",
            "TYPE_ERR": "Veri tipi hatası",
            "IO_ERR": "Giriş/çıkış hatası",
            "NETWORK_ERR": "Ağ hatası",
            "DB_ERR": "Veritabanı hatası"
        }

    def register_handler(self, error_code: str, handler: str) -> None:
        """Hata kodu için bir işleyici kaydeder."""
        with self.lock:
            try:
                self.error_handlers[error_code.upper()].append(handler)
                log.debug(f"Hata işleyici kaydedildi: error_code={error_code}, handler={handler}")
            except Exception as e:
                log.error(f"Hata işleyici kayıt hatası: {str(e)}")
                raise PdsXException(f"Hata işleyici kayıt hatası: {str(e)}")

    def unregister_handler(self, error_code: str, handler: str) -> None:
        """Hata kodu için bir işleyiciyi kaldırır."""
        with self.lock:
            try:
                if error_code.upper() in self.error_handlers and handler in self.error_handlers[error_code.upper()]:
                    self.error_handlers[error_code.upper()].remove(handler)
                    log.debug(f"Hata işleyici kaldırıldı: error_code={error_code}, handler={handler}")
                else:
                    raise PdsXException(f"Hata işleyici bulunamadı: {error_code}, {handler}")
            except Exception as e:
                log.error(f"Hata işleyici kaldırma hatası: {str(e)}")
                raise PdsXException(f"Hata işleyici kaldırma hatası: {str(e)}")

    def set_global_handler(self, handler: str) -> None:
        """Genel hata işleyiciyi ayarlar."""
        with self.lock:
            try:
                self.global_handler = handler
                log.debug(f"Genel hata işleyici ayarlandı: {handler}")
            except Exception as e:
                log.error(f"Genel hata işleyici ayarlama hatası: {str(e)}")
                raise PdsXException(f"Genel hata işleyici ayarlama hatası: {str(e)}")

    def handle_error(self, error: PdsXException) -> Optional[str]:
        """Hata işleyicileri tetikler."""
        with self.lock:
            try:
                self.error_log.append((error.timestamp, error.to_dict()))
                if len(self.error_log) > self.max_log_size:
                    self.error_log.pop(0)
                
                log.error(f"Hata yakalandı: {error}")
                
                # Hata koduna özel işleyiciler
                handlers = self.error_handlers.get(error.error_code.upper(), [])
                for handler in handlers:
                    self.interpreter.execute_command(handler)
                
                # Genel işleyici
                if self.global_handler:
                    self.interpreter.current_scope()["_ERROR"] = error.to_dict()
                    return self.global_handler
                
                # Varsayılan hata işleme
                return None
            except Exception as e:
                log.error(f"Hata işleme hatası: {str(e)}")
                raise PdsXException(f"Hata işleme hatası: {str(e)}")

    def get_error_log(self, max_entries: Optional[int] = None) -> List[Dict]:
        """Hata günlüğünü döndürür."""
        with self.lock:
            try:
                return [error for _, error in self.error_log[-max_entries:]] if max_entries else [error for _, error in self.error_log]
            except Exception as e:
                log.error(f"Hata günlüğü alma hatası: {str(e)}")
                raise PdsXException(f"Hata günlüğü alma hatası: {str(e)}")

    def clear_error_log(self) -> None:
        """Hata günlüğünü temizler."""
        with self.lock:
            try:
                self.error_log.clear()
                log.debug("Hata günlüğü temizlendi")
            except Exception as e:
                log.error(f"Hata günlüğü temizleme hatası: {str(e)}")
                raise PdsXException(f"Hata günlüğü temizleme hatası: {str(e)}")

    def save_error_log(self, path: str, compress: bool = False) -> None:
        """Hata günlüğünü dosyaya kaydeder."""
        with self.lock:
            try:
                data = json.dumps([error for _, error in self.error_log], indent=2)
                if compress:
                    with gzip.open(path, "wt", encoding="utf-8") as f:
                        f.write(data)
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(data)
                log.debug(f"Hata günlüğü kaydedildi: path={path}, compress={compress}")
            except Exception as e:
                log.error(f"Hata günlüğü kaydetme hatası: {str(e)}")
                raise PdsXException(f"Hata günlüğü kaydetme hatası: {str(e)}")

    def load_error_log(self, path: str) -> None:
        """Hata günlüğünü dosyadan yükler."""
        with self.lock:
            try:
                if path.endswith(".gz"):
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                
                self.error_log.extend([(entry.get("timestamp", time.time()), entry) for entry in data])
                if len(self.error_log) > self.max_log_size:
                    self.error_log = self.error_log[-self.max_log_size:]
                log.debug(f"Hata günlüğü yüklendi: path={path}")
            except Exception as e:
                log.error(f"Hata günlüğü yükleme hatası: {str(e)}")
                raise PdsXException(f"Hata günlüğü yükleme hatası: {str(e)}")

    def parse_exception_command(self, command: str) -> None:
        """Hata yönetimi komutlarını ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("TRY "):
                match = re.match(r"TRY\s+(.+)\s+CATCH\s+(.+)(?:\s+FINALLY\s+(.+))?", command, re.IGNORECASE)
                if match:
                    try_block, catch_block, finally_block = match.groups()
                    try:
                        self.interpreter.execute_command(try_block)
                    except PdsXException as e:
                        self.interpreter.current_scope()["_ERROR"] = e.to_dict()
                        self.interpreter.execute_command(catch_block)
                    except Exception as e:
                        pdsx_error = PdsXRuntimeError(str(e), context={"source": "TRY", "line_no": self.interpreter.program_counter})
                        self.interpreter.current_scope()["_ERROR"] = pdsx_error.to_dict()
                        self.interpreter.execute_command(catch_block)
                    finally:
                        if finally_block:
                            self.interpreter.execute_command(finally_block)
                else:
                    raise PdsXException("TRY CATCH FINALLY komutunda sözdizimi hatası")
            elif command_upper.startswith("ON ERROR GOTO "):
                match = re.match(r"ON ERROR GOTO\s+(\w+)", command, re.IGNORECASE)
                if match:
                    handler = match.group(1)
                    self.set_global_handler(handler)
                else:
                    raise PdsXException("ON ERROR GOTO komutunda sözdizimi hatası")
            elif command_upper.startswith("RESUME"):
                # RESUME, hata işleyiciden sonra yürütmeye devam eder
                if self.global_handler:
                    log.debug("RESUME ile yürütme devam ediyor")
                else:
                    raise PdsXException("RESUME için hata işleyici tanımlı değil")
            elif command_upper.startswith("REGISTER ERROR HANDLER "):
                match = re.match(r"REGISTER ERROR HANDLER\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    error_code, handler = match.groups()
                    self.register_handler(error_code, handler)
                else:
                    raise PdsXException("REGISTER ERROR HANDLER komutunda sözdizimi hatası")
            elif command_upper.startswith("UNREGISTER ERROR HANDLER "):
                match = re.match(r"UNREGISTER ERROR HANDLER\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    error_code, handler = match.groups()
                    self.unregister_handler(error_code, handler)
                else:
                    raise PdsXException("UNREGISTER ERROR HANDLER komutunda sözdizimi hatası")
            elif command_upper.startswith("GET ERROR LOG "):
                match = re.match(r"GET ERROR LOG\s+(\w+)(?:\s+(\d+))?", command, re.IGNORECASE)
                if match:
                    var_name, max_entries = match.groups()
                    max_entries = int(max_entries) if max_entries else None
                    self.interpreter.current_scope()[var_name] = self.get_error_log(max_entries)
                else:
                    raise PdsXException("GET ERROR LOG komutunda sözdizimi hatası")
            elif command_upper.startswith("CLEAR ERROR LOG"):
                self.clear_error_log()
            elif command_upper.startswith("SAVE ERROR LOG "):
                match = re.match(r"SAVE ERROR LOG\s+\"([^\"]+)\"\s*(COMPRESS)?", command, re.IGNORECASE)
                if match:
                    path, compress = match.groups()
                    self.save_error_log(path, bool(compress))
                else:
                    raise PdsXException("SAVE ERROR LOG komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD ERROR LOG "):
                match = re.match(r"LOAD ERROR LOG\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    path = match.group(1)
                    self.load_error_log(path)
                else:
                    raise PdsXException("LOAD ERROR LOG komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen hata komutu: {command}")
        except Exception as e:
            log.error(f"Hata komut hatası: {str(e)}")
            raise PdsXException(f"Hata komut hatası: {str(e)}")

if __name__ == "__main__":
    print("pdsx_exception.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")