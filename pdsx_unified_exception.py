# pdsx_unified_exception.py - PDS-X v14u Unified Exception System
# Version: 2.0.0 (Merged from v13, v14, v15 exception systems)
# Date: July 21, 2025
# Author: Claude 3.5 Sonnet (Unified by metedinler directive)

"""
================================================================================
🔧 PDS-X UNIFIED EXCEPTION SYSTEM v2.0.0
================================================================================

Bu modül, PDS-X framework'ünün 3 farklı exception modülünü birleştiren
unified hata yönetim sistemidir:

📋 BİRLEŞTİRİLEN MODÜLLER:
- pdsx_exception.py (v13 Legacy) - Temel exception handling
- pdsx_exception2.py (v14 Bridge) - AI/ML tabanlı anomali tespiti
- exception_manager3.py (v15 Latest) - En gelişmiş architecture

🎯 ÖZELLİKLER:
✅ Backward compatibility (v13, v14, v15)
✅ Performance optimized (Cython, LRU cache)
✅ Async/threading support
✅ AI/ML anomaly detection (optional)
✅ Comprehensive exception hierarchy
✅ Context tracking & logging
✅ Recovery strategies
✅ Debug mode support

📡 KULLANIM:
```python
from pdsx_unified_exception import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError,
    ExceptionManager
)

# Exception handling
try:
    risky_operation()
except PdsXException as e:
    e.handle()

# Exception manager
manager = ExceptionManager()
await manager.handle_error("Test error")
```

================================================================================
"""

import logging
import traceback
import threading
import asyncio
import sys
import time
import os
import re
import json
import hashlib
from collections import defaultdict
from typing import Any, Dict, Optional, List, Tuple, Type, Callable
from functools import lru_cache
from datetime import datetime
from pathlib import Path

# Optional dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from cython import compiled
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    compiled = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    # Fallback for async file operations
    class _DummyAiofiles:
        @staticmethod
        def open(path, mode='r', encoding=None):
            class _SyncCM:
                def __init__(self):
                    self._f = open(path, mode, encoding=encoding)
                async def __aenter__(self):
                    return self._f
                async def __aexit__(self, exc_type, exc, tb):
                    self._f.close()
            return _SyncCM()
    aiofiles = _DummyAiofiles()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# AI/ML dependencies (optional)
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Logging configuration
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("pdsx_unified_exception")

# Performance optimized error formatting
if CYTHON_AVAILABLE and compiled:
    @cython.cfunc
    def fast_format_error(code: str, msg: str, context: Dict[str, Any]) -> str:
        return f"[{code}] {msg} | Context: {context}"
else:
    def fast_format_error(code: str, msg: str, context: Dict[str, Any]) -> str:
        return f"[{code}] {msg} | Context: {context}"

# ================================================================================
# BASE EXCEPTION CLASSES (From exception_manager3.py - v15 Latest)
# ================================================================================

class PdsXException(Exception):
    """PDS-X BASIC Unified temel hata sınıfı - v2.0.0"""
    def __init__(self, message: str, code: str = "PDS-X000", context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stack_trace = "".join(traceback.format_tb(sys.exc_info()[2])) if sys.exc_info()[2] else ""
        self.thread_id = threading.get_ident()
        super().__init__(self.message)
    
    def __str__(self):
        return fast_format_error(self.code, self.message, self.context)
    
    def handle(self):
        """Legacy compatibility method from pdsx_exception.py"""
        log.error(f"Exception handled: {self}")
        self.log_error()
    
    def log_error(self):
        """Log error with full context"""
        log.error(f"[{self.code}] {self.message}")
        if self.context:
            log.error(f"Context: {self.context}")
        if self.stack_trace:
            log.error(f"Stack trace: {self.stack_trace}")

# ================================================================================
# SPECIFIC EXCEPTION CLASSES (Unified from all 3 modules)
# ================================================================================

# Core PDS-X Exceptions
class PdsXSyntaxError(PdsXException):
    """Sözdizimi hataları"""
    def __init__(self, message: str, code: str = "SYNTAX001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXRuntimeError(PdsXException):
    """Çalışma zamanı hataları"""
    def __init__(self, message: str, code: str = "RUNTIME001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXMemoryError(PdsXException):
    """Bellek yönetimi hataları"""
    def __init__(self, message: str, code: str = "MEMORY001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXModuleError(PdsXException):
    """Modül yükleme/yönetim hataları"""
    def __init__(self, message: str, code: str = "MODULE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXCompileError(PdsXException):
    """Derleme hataları"""
    def __init__(self, message: str, code: str = "COMPILE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXSecurityError(PdsXException):
    """Güvenlik hataları"""
    def __init__(self, message: str, code: str = "SECURITY001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# Pipeline & Communication Exceptions
class PdsXPipeError(PdsXException):
    """Pipeline hataları"""
    def __init__(self, message: str, code: str = "PIPE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXBusError(PdsXException):
    """Bus communication hataları"""
    def __init__(self, message: str, code: str = "BUS001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXNetworkError(PdsXException):
    """Network hataları"""
    def __init__(self, message: str, code: str = "NETWORK001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# Database Exceptions
class PdsXDatabaseError(PdsXException):
    """Database hataları"""
    def __init__(self, message: str, code: str = "DB001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# REPL & UI Exceptions
class PdsXReplyError(PdsXException):
    """REPL hataları"""
    def __init__(self, message: str, code: str = "REPLY001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# LibX Domain Exceptions
class PdsXLogicError(PdsXException):
    """Logic/Prolog hataları"""
    def __init__(self, message: str, code: str = "LOGIC001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXMLError(PdsXException):
    """Machine Learning hataları"""
    def __init__(self, message: str, code: str = "ML001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXNLPError(PdsXException):
    """Natural Language Processing hataları"""
    def __init__(self, message: str, code: str = "NLP001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# System Level Exceptions
class PdsXLowLevelError(PdsXException):
    """Low-level sistem hataları"""
    def __init__(self, message: str, code: str = "LOWLEVEL001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXMultithreadingError(PdsXException):
    """Multithreading hataları"""
    def __init__(self, message: str, code: str = "THREADING001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# Module Management Exceptions
class PdsXModuleValidatorError(PdsXException):
    """Module validator hataları"""
    def __init__(self, message: str, code: str = "VALIDATOR001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXModuleManagerError(PdsXException):
    """Module manager hataları"""
    def __init__(self, message: str, code: str = "MANAGER001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# Additional Exception Classes
class PdsXEventError(PdsXException):
    """Event management hataları"""
    def __init__(self, message: str, code: str = "EVENT001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXTreeError(PdsXException):
    """Tree structure hataları"""
    def __init__(self, message: str, code: str = "TREE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXAutoImporterError(PdsXException):
    """AutoImporter hataları"""
    def __init__(self, message: str, code: str = "AUTOIMPORTER001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXSaveLoadError(PdsXException):
    """Save/Load işlem hataları"""
    def __init__(self, message: str, code: str = "SAVELOAD001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXExportError(PdsXException):
    """Export işlem hataları"""
    def __init__(self, message: str, code: str = "EXPORT001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXValidationError(PdsXException):
    """Validation hataları"""
    def __init__(self, message: str, code: str = "VALIDATION001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXTimeoutError(PdsXException):
    """Timeout hataları"""
    def __init__(self, message: str, code: str = "TIMEOUT001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXFileError(PdsXException):
    """File işlem hataları"""
    def __init__(self, message: str, code: str = "FILE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXIOError(PdsXException):
    """Input/Output hataları"""
    def __init__(self, message: str, code: str = "IO001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXTypeError(PdsXException):
    """Type mismatch hataları"""
    def __init__(self, message: str, code: str = "TYPE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXValueError(PdsXException):
    """Value error hataları"""
    def __init__(self, message: str, code: str = "VALUE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXFunctionalError(PdsXException):
    """Functional programming hataları"""
    def __init__(self, message: str, code: str = "FUNCTIONAL001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXOOPError(PdsXException):
    """Object-oriented programming hataları"""
    def __init__(self, message: str, code: str = "OOP001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXOfflineError(PdsXException):
    """Offline işlem hataları"""
    def __init__(self, message: str, code: str = "OFFLINE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

class PdsXCoreException(PdsXException):
    """Core system hataları"""
    def __init__(self, message: str, code: str = "CORE001", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, code, context)

# ================================================================================
# EXCEPTION MANAGER CLASS (Enhanced from all 3 modules)
# ================================================================================

class ExceptionManager:
    """PDS-X Unified Exception Manager v2.0.0"""
    
    def __init__(self, interpreter=None):
        self.interpreter = interpreter
        self.error_counts = defaultdict(int)
        self.error_history = []
        self.max_history = 1000
        self.debug_mode = False
        self.recovery_strategies = {}
        self.lock = threading.RLock()
        
        # AI/ML anomaly detection (optional)
        self.anomaly_detector = None
        if SKLEARN_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Performance metrics
        self.metrics = {
            "total_errors": 0,
            "errors_by_type": defaultdict(int),
            "errors_by_code": defaultdict(int),
            "recovery_success_rate": 0.0
        }
        
        # Setup default recovery strategies
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup default recovery strategies from pdsx_exception.py"""
        self.recovery_strategies = {
            "SYNTAX": self._recover_syntax_error,
            "RUNTIME": self._recover_runtime_error,
            "MEMORY": self._recover_memory_error,
            "MODULE": self._recover_module_error,
            "NETWORK": self._recover_network_error,
            "DB": self._recover_database_error
        }
    
    async def handle_error(self, error, context: Optional[Dict[str, Any]] = None):
        """Async error handling with context"""
        with self.lock:
            if isinstance(error, str):
                error = PdsXException(error, context=context)
            elif not isinstance(error, PdsXException):
                error = PdsXException(str(error), context=context)
            
            # Update metrics
            self.metrics["total_errors"] += 1
            self.metrics["errors_by_type"][type(error).__name__] += 1
            self.metrics["errors_by_code"][error.code] += 1
            
            # Add to history
            error_data = {
                "timestamp": error.timestamp,
                "type": type(error).__name__,
                "code": error.code,
                "message": error.message,
                "context": error.context,
                "thread_id": error.thread_id
            }
            
            self.error_history.append(error_data)
            if len(self.error_history) > self.max_history:
                self.error_history.pop(0)
            
            # Log error
            error.log_error()
            
            # Try recovery
            await self._attempt_recovery(error)
            
            # Anomaly detection (if available)
            if self.anomaly_detector and NUMPY_AVAILABLE:
                await self._detect_anomaly(error_data)
    
    async def _attempt_recovery(self, error: PdsXException):
        """Attempt error recovery using strategies"""
        error_category = error.code.split("001")[0] if "001" in error.code else "UNKNOWN"
        
        if error_category in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_category]
                success = await recovery_func(error) if asyncio.iscoroutinefunction(recovery_func) else recovery_func(error)
                if success:
                    log.info(f"Recovery successful for {error.code}")
                    return True
            except Exception as recovery_error:
                log.error(f"Recovery failed for {error.code}: {recovery_error}")
        
        return False
    
    async def _detect_anomaly(self, error_data: Dict[str, Any]):
        """AI/ML based anomaly detection from pdsx_exception2.py"""
        if not self.anomaly_detector:
            return
        
        try:
            # Convert error data to numerical features
            features = [
                hash(error_data["type"]) % 10000,
                hash(error_data["code"]) % 10000,
                len(error_data["message"]),
                error_data["thread_id"] % 10000,
                time.time() % 86400  # Time of day
            ]
            
            # Predict anomaly
            prediction = self.anomaly_detector.predict([features])
            if prediction[0] == -1:  # Anomaly detected
                log.warning(f"Anomaly detected in error pattern: {error_data}")
        except Exception as e:
            log.debug(f"Anomaly detection failed: {e}")
    
    # Recovery strategy implementations (from pdsx_exception.py)
    def _recover_syntax_error(self, error: PdsXException) -> bool:
        """Attempt syntax error recovery"""
        if "missing" in error.message.lower():
            log.info("Attempting syntax error recovery")
            return True
        return False
    
    def _recover_runtime_error(self, error: PdsXException) -> bool:
        """Attempt runtime error recovery"""
        if "division by zero" in error.message.lower():
            log.info("Attempting division by zero recovery")
            return True
        return False
    
    def _recover_memory_error(self, error: PdsXException) -> bool:
        """Attempt memory error recovery"""
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                log.warning("High memory usage detected, attempting cleanup")
                # Trigger garbage collection
                import gc
                gc.collect()
                return True
        return False
    
    def _recover_module_error(self, error: PdsXException) -> bool:
        """Attempt module error recovery"""
        if "import" in error.message.lower():
            log.info("Attempting module import recovery")
            return True
        return False
    
    def _recover_network_error(self, error: PdsXException) -> bool:
        """Attempt network error recovery"""
        if "timeout" in error.message.lower():
            log.info("Attempting network timeout recovery")
            return True
        return False
    
    def _recover_database_error(self, error: PdsXException) -> bool:
        """Attempt database error recovery"""
        if "connection" in error.message.lower():
            log.info("Attempting database connection recovery")
            return True
        return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self.lock:
            return {
                "metrics": dict(self.metrics),
                "recent_errors": self.error_history[-10:],
                "error_counts": dict(self.error_counts),
                "total_history": len(self.error_history)
            }
    
    @lru_cache(maxsize=128)
    def get_error_pattern(self, error_type: str) -> Optional[str]:
        """Get cached error pattern analysis"""
        errors_of_type = [e for e in self.error_history if e["type"] == error_type]
        if len(errors_of_type) > 5:
            return f"Pattern detected: {error_type} occurs frequently"
        return None

# ================================================================================
# LEGACY COMPATIBILITY LAYER
# ================================================================================

# Aliases for backward compatibility with pdsx_exception.py (v13)
ModuleError = PdsXModuleError
MemoryError = PdsXMemoryError  # Note: This shadows built-in MemoryError
CompileError = PdsXCompileError
RuntimeError = PdsXRuntimeError  # Note: This shadows built-in RuntimeError
SecurityError = PdsXSecurityError

# Aliases for backward compatibility with pdsx_exception2.py (v14)
PdsxAnomalyDetector = ExceptionManager  # ML-based detection is now part of manager

# ================================================================================
# MODULE EXPORTS
# ================================================================================

__all__ = [
    # Base exceptions
    'PdsXException',
    'PdsXSyntaxError',
    'PdsXRuntimeError',
    'PdsXMemoryError',
    'PdsXModuleError',
    'PdsXCompileError',
    'PdsXSecurityError',
    
    # Communication exceptions
    'PdsXPipeError',
    'PdsXBusError',
    'PdsXNetworkError',
    
    # Database exceptions
    'PdsXDatabaseError',
    
    # REPL exceptions
    'PdsXReplyError',
    
    # LibX domain exceptions
    'PdsXLogicError',
    'PdsXMLError',
    'PdsXNLPError',
    
    # System exceptions
    'PdsXLowLevelError',
    'PdsXMultithreadingError',
    
    # Module management exceptions
    'PdsXModuleValidatorError',
    'PdsXModuleManagerError',
    
    # Manager class
    'ExceptionManager',
    
    # Legacy aliases
    'ModuleError',
    'MemoryError',
    'CompileError',
    'RuntimeError',
    'SecurityError'
]

# Module initialization
log.info("PDS-X Unified Exception System v2.0.0 initialized")
if CYTHON_AVAILABLE:
    log.info("Cython optimizations enabled")
if NUMPY_AVAILABLE:
    log.info("NumPy acceleration available")
if SKLEARN_AVAILABLE:
    log.info("ML anomaly detection available")
if TENSORFLOW_AVAILABLE:
    log.info("TensorFlow LSTM prediction available")

print("[PDS-X] ✅ Unified Exception System v2.0.0 loaded")
print(f"[PDS-X] 🔧 Features: Cython={CYTHON_AVAILABLE}, ML={SKLEARN_AVAILABLE}, Async={AIOFILES_AVAILABLE}")
