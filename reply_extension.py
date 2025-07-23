# reply_extension.py - PDS-X BASIC v14u Yanıt Uzantısı Kütüphanesi
# Version: 1.0.0
# Date: May 13, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import threading
import asyncio
import time
import json
import yaml
import xml.etree.ElementTree as ET
import base64
import sys
import subprocess
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Program manager import
from program_manager import MultiLineProgramManager


class BaseResponse:
    """Base response class for all response types."""
    def __init__(self, response_id: str, data: Any, timestamp: float):
        self.response_id = response_id
        self.data = data
        self.timestamp = timestamp
        self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        return f"Response(id={self.response_id}, data={self.data}, timestamp={self.timestamp})"

    def __repr__(self) -> str:
        return self.__str__()


class TextResponse(BaseResponse):
    """Text response for simple messages."""
    def __init__(self):
        super().__init__("text_response", "", time.time())
        self.content = ""
        
    def set_content(self, content: str):
        self.content = content
        self.data = content


class DataResponse(BaseResponse):
    """Data response for structured information."""
    def __init__(self):
        super().__init__("data_response", {}, time.time())
        
    def add_data(self, key: str, value: Any):
        if isinstance(self.data, dict):
            self.data[key] = value


class BinaryResponse(BaseResponse):
    """Binary response for file data."""
    def __init__(self):
        super().__init__("binary_response", b"", time.time())


class ErrorResponse(BaseResponse):
    """Error response for exceptions."""
    def __init__(self):
        super().__init__("error_response", "", time.time())
        self.error_type = None
        self.error_message = ""
        
    def set_error(self, error_type: str, message: str):
        self.error_type = error_type
        self.error_message = message
        self.data = {"error_type": error_type, "message": message}


class LogResponse(BaseResponse):
    """Log response for logging information."""
    def __init__(self):
        super().__init__("log_response", [], time.time())
        self.logs = []
        
    def add_log(self, level: str, message: str, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.logs.append({"level": level, "message": message, "timestamp": timestamp})
        self.data = self.logs


class QuantumResponseCorrelator:
    """Advanced response correlation system."""
    def __init__(self):
        self.correlations = {}
        self.lock = threading.Lock()
    
    def correlate(self, request_id: str, response: BaseResponse) -> None:
        with self.lock:
            self.correlations[request_id] = response
    
    def get_correlation(self, request_id: str) -> Optional[BaseResponse]:
        with self.lock:
            return self.correlations.get(request_id)


class HoloResponse:
    """Holographic response processor."""
    def __init__(self):
        self.dimensions = {}
        
    def add_dimension(self, name: str, data: Any):
        self.dimensions[name] = data
    
    def render(self) -> Dict[str, Any]:
        return self.dimensions.copy()


class SmartRouter:
    """Smart response routing system."""
    def __init__(self):
        self.routes = {}
        
    def add_route(self, pattern: str, handler: Callable):
        self.routes[pattern] = handler
        
    def route(self, message: str) -> Any:
        for pattern, handler in self.routes.items():
            if re.search(pattern, message):
                return handler(message)
        return None


class TemporalResponseGraph:
    """Temporal response graph analysis."""
    def __init__(self):
        self.graph = {}
        self.timeline = []
        
    def add_node(self, node_id: str, data: Any):
        self.graph[node_id] = data
        self.timeline.append({"id": node_id, "timestamp": time.time()})
        
    def get_timeline(self) -> List[Dict]:
        return self.timeline.copy()


class ResponseShield:
    """Response security and validation."""
    def __init__(self):
        self.filters = []
        
    def add_filter(self, filter_func: Callable):
        self.filters.append(filter_func)
        
    def validate(self, response: BaseResponse) -> bool:
        for filter_func in self.filters:
            if not filter_func(response):
                return False
        return True


class ReplyExtension:
    """Yanıt uzantısı yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.responses = {}  # {response_id: Response}
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.quantum_correlator = QuantumResponseCorrelator()
        self.holo_response = HoloResponse()
        self.smart_router = SmartRouter()
        self.temporal_graph = TemporalResponseGraph()
        self.response_shield = ResponseShield()
        self.lock = threading.Lock()
        
        # Çok satırlı program sistemi
        self.program_manager = MultiLineProgramManager()
        
        self.metadata = {
            "reply_extension": {
                "version": "1.0.0",
                "dependencies": ["graphviz", "numpy", "scikit-learn", "boto3", "websockets", "pyyaml", "pycryptodome", "pdsx_exception"],
                "supported_formats": [".basx", ".libx", ".pdsx", ".py", ".js", ".sql", ".txt"],
                "features": ["multiline_programs", "encryption", "compression", "basic_interpreter"]
            }
        }
        self.max_response_size = 10000

    def start_async_processing(self):
        """Start async processing thread."""
        if self.async_thread is None or not self.async_thread.is_alive():
            self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.async_thread.start()

    def _run_async_loop(self):
        """Run async event loop in separate thread."""
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()

    def create_response(self, response_type: str = "text") -> BaseResponse:
        """Create a new response object."""
        response_classes = {
            "text": TextResponse,
            "data": DataResponse,
            "binary": BinaryResponse,
            "error": ErrorResponse,
            "log": LogResponse
        }
        
        response_class = response_classes.get(response_type, TextResponse)
        response = response_class()
        
        with self.lock:
            self.responses[response.response_id] = response
        
        return response

    def get_response(self, response_id: str) -> Optional[BaseResponse]:
        """Get a response by ID."""
        with self.lock:
            return self.responses.get(response_id)

    def delete_response(self, response_id: str) -> bool:
        """Delete a response."""
        with self.lock:
            if response_id in self.responses:
                del self.responses[response_id]
                return True
        return False

    def list_responses(self) -> List[str]:
        """List all response IDs."""
        with self.lock:
            return list(self.responses.keys())

    def clear_responses(self):
        """Clear all responses."""
        with self.lock:
            self.responses.clear()

    async def process_async_response(self, data: Any) -> BaseResponse:
        """Process response asynchronously."""
        response = self.create_response("data")
        if hasattr(response, 'add_data'):
            response.add_data("async_result", data)
            response.add_data("processed_at", time.time())
        return response

    def process_multipart_response(self, parts: List[Any]) -> BaseResponse:
        """Process multipart response."""
        response = self.create_response("data")
        for i, part in enumerate(parts):
            response.add_data(f"part_{i}", part)
        return response

    def compress_response(self, response: BaseResponse) -> BaseResponse:
        """Compress response data."""
        import gzip
        
        data_str = json.dumps(response.to_dict())
        compressed = gzip.compress(data_str.encode())
        
        compressed_response = self.create_response("binary")
        compressed_response.data = base64.b64encode(compressed).decode()
        compressed_response.metadata["compression"] = "gzip"
        compressed_response.metadata["original_size"] = len(data_str)
        compressed_response.metadata["compressed_size"] = len(compressed)
        
        return compressed_response

    def encrypt_response(self, response: BaseResponse, key: str) -> BaseResponse:
        """Encrypt response data."""
        try:
            from Crypto.Cipher import AES
            from Crypto.Random import get_random_bytes
            from Crypto.Util.Padding import pad
            
            # Convert key to bytes
            key_bytes = key.encode()[:32].ljust(32, b'\0')
            
            # Prepare data
            data_str = json.dumps(response.to_dict())
            data_bytes = data_str.encode()
            
            # Encrypt
            cipher = AES.new(key_bytes, AES.MODE_CBC)
            encrypted = cipher.encrypt(pad(data_bytes, AES.block_size))
            
            # Create encrypted response
            encrypted_response = self.create_response("binary")
            encrypted_response.data = base64.b64encode(cipher.iv + encrypted).decode()
            encrypted_response.metadata["encryption"] = "AES-CBC"
            encrypted_response.metadata["encrypted"] = True
            
            return encrypted_response
            
        except ImportError:
            # Fallback to simple encoding
            data_str = json.dumps(response.to_dict())
            encoded = base64.b64encode(data_str.encode()).decode()
            
            fallback_response = self.create_response("binary")
            fallback_response.data = encoded
            fallback_response.metadata["encoding"] = "base64"
            
            return fallback_response

    def batch_process_responses(self, responses: List[BaseResponse]) -> BaseResponse:
        """Process multiple responses in batch."""
        batch_response = self.create_response("data")
        
        results = []
        for i, response in enumerate(responses):
            results.append({
                "index": i,
                "response_id": response.response_id,
                "data": response.data,
                "timestamp": response.timestamp
            })
        
        batch_response.add_data("batch_results", results)
        batch_response.add_data("batch_size", len(responses))
        batch_response.add_data("processed_at", time.time())
        
        return batch_response

    def validate_response_format(self, response: BaseResponse) -> bool:
        """Validate response format."""
        if not hasattr(response, 'response_id') or not response.response_id:
            return False
        if not hasattr(response, 'timestamp') or response.timestamp <= 0:
            return False
        if not hasattr(response, 'data'):
            return False
        return True

    def get_response_stats(self) -> Dict[str, Any]:
        """Get response statistics."""
        with self.lock:
            total_responses = len(self.responses)
            response_types = {}
            
            for response in self.responses.values():
                response_type = type(response).__name__
                response_types[response_type] = response_types.get(response_type, 0) + 1
        
        return {
            "total_responses": total_responses,
            "response_types": response_types,
            "memory_usage": sys.getsizeof(self.responses),
            "max_response_size": self.max_response_size
        }

    def export_responses(self, format_type: str = "json") -> str:
        """Export all responses."""
        with self.lock:
            if format_type == "json":
                export_data = {}
                for response_id, response in self.responses.items():
                    export_data[response_id] = response.to_dict()
                return json.dumps(export_data, indent=2)
            
            elif format_type == "xml":
                root = ET.Element("responses")
                for response_id, response in self.responses.items():
                    response_elem = ET.SubElement(root, "response", id=response_id)
                    data_elem = ET.SubElement(response_elem, "data")
                    data_elem.text = str(response.data)
                    timestamp_elem = ET.SubElement(response_elem, "timestamp")
                    timestamp_elem.text = str(response.timestamp)
                return ET.tostring(root, encoding='unicode')
            
            elif format_type == "yaml":
                export_data = {}
                for response_id, response in self.responses.items():
                    export_data[response_id] = response.to_dict()
                return yaml.dump(export_data)
        
        return ""

    def import_responses(self, data: str, format_type: str = "json"):
        """Import responses from data."""
        try:
            if format_type == "json":
                import_data = json.loads(data)
                for response_id, response_dict in import_data.items():
                    response = self.create_response("data")
                    response.response_id = response_id
                    response.data = response_dict.get("data", {})
                    response.timestamp = response_dict.get("timestamp", time.time())
                    response.metadata = response_dict.get("metadata", {})
            
            elif format_type == "yaml":
                import_data = yaml.safe_load(data)
                for response_id, response_dict in import_data.items():
                    response = self.create_response("data")
                    response.response_id = response_id
                    response.data = response_dict.get("data", {})
                    response.timestamp = response_dict.get("timestamp", time.time())
                    response.metadata = response_dict.get("metadata", {})
                    
        except Exception as e:
            logging.error(f"Import error: {e}")

    def backup_responses(self, filename: str):
        """Backup responses to file."""
        try:
            backup_data = self.export_responses("json")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(backup_data)
            logging.info(f"Responses backed up to {filename}")
        except Exception as e:
            logging.error(f"Backup error: {e}")

    def restore_responses(self, filename: str):
        """Restore responses from backup."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            self.import_responses(backup_data, "json")
            logging.info(f"Responses restored from {filename}")
        except Exception as e:
            logging.error(f"Restore error: {e}")

    def cleanup_old_responses(self, max_age_hours: int = 24):
        """Clean up responses older than specified hours."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        with self.lock:
            old_responses = [
                response_id for response_id, response in self.responses.items()
                if response.timestamp < cutoff_time
            ]
            
            for response_id in old_responses:
                del self.responses[response_id]
        
        logging.info(f"Cleaned up {len(old_responses)} old responses")
        return len(old_responses)

    def add_program_line(self, line: str) -> bool:
        """
        Program modunda satır ekle
        
        Args:
            line: Eklenecek kod satırı
            
        Returns: True if still in program mode, False if exited
        """
        try:
            if self.program_manager.in_program_mode:
                if line.strip().upper() == 'END PROGRAM':
                    return self.program_manager.end_program()
                else:
                    return self.program_manager.add_line(line)
            return False
        except Exception as e:
            print(f"[PDS-X] ❌ Program girdi hatası: {e}")
            return False


if __name__ == "__main__":
    print("reply_extension.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")
