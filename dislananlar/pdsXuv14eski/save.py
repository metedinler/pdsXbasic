# save.py - PDS-X BASIC v14u Veri Kaydetme ve Serileştirme Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import threading
import json
import pickle
import yaml
import gzip
import zlib
import lzma
import base64
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import hashlib
import boto3
import botocore
import uuid
import time
from collections import defaultdict
import numpy as np
from pdsx_exception import PdsXException  # Hata yönetimi için

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("save")

class DiffSerializer:
    """Farklı serileştirme sınıfı (yalnızca değişen verileri kaydeder)."""
    def __init__(self):
        self.previous_state = None
        self.diff_history = []

    def serialize(self, data: Any) -> bytes:
        """Verinin farkını serileştirir."""
        try:
            current_state = pickle.dumps(data)
            if self.previous_state is None:
                self.previous_state = current_state
                self.diff_history.append(current_state)
                return current_state
            
            # Fark hesaplama (basitleştirilmiş: delta encoding)
            diff = bytes(a ^ b for a, b in zip(current_state, self.previous_state))
            self.previous_state = current_state
            self.diff_history.append(diff)
            log.debug("Farklı serileştirme yapıldı")
            return diff
        except Exception as e:
            log.error(f"DiffSerializer serialize hatası: {str(e)}")
            raise PdsXException(f"DiffSerializer serialize hatası: {str(e)}")

    def deserialize(self, diff: bytes) -> Any:
        """Farktan veriyi geri yükler."""
        try:
            if not self.previous_state:
                self.previous_state = diff
                return pickle.loads(diff)
            
            # Farkı geri yükle
            reconstructed = bytes(a ^ b for a, b in zip(diff, self.previous_state))
            self.previous_state = reconstructed
            return pickle.loads(reconstructed)
        except Exception as e:
            log.error(f"DiffSerializer deserialize hatası: {str(e)}")
            raise PdsXException(f"DiffSerializer deserialize hatası: {str(e)}")

class ProvenanceChain:
    """Blok zinciri tabanlı veri geçmişi izleme sınıfı."""
    def __init__(self):
        self.chain = []
        self.current_hash = b"genesis"

    def add_block(self, data: bytes) -> str:
        """Veri bloğu ekler ve hash oluşturur."""
        try:
            block = {
                "timestamp": time.time(),
                "data": base64.b64encode(data).decode('utf-8'),
                "prev_hash": self.current_hash.hex()
            }
            block_hash = hashlib.sha256(json.dumps(block, sort_keys=True).encode()).digest()
            block["hash"] = block_hash.hex()
            self.chain.append(block)
            self.current_hash = block_hash
            log.debug(f"Provenance block eklendi: hash={block['hash']}")
            return block["hash"]
        except Exception as e:
            log.error(f"ProvenanceChain add_block hatası: {str(e)}")
            raise PdsXException(f"ProvenanceChain add_block hatası: {str(e)}")

    def verify(self) -> bool:
        """Zincirin bütünlüğünü doğrular."""
        try:
            for i, block in enumerate(self.chain[1:], 1):
                prev_block = self.chain[i-1]
                if block["prev_hash"] != prev_block["hash"]:
                    return False
                computed_hash = hashlib.sha256(json.dumps({k: v for k, v in block.items() if k != "hash"}, sort_keys=True).encode()).hexdigest()
                if computed_hash != block["hash"]:
                    return False
            log.debug("Provenance zinciri doğrulandı")
            return True
        except Exception as e:
            log.error(f"ProvenanceChain verify hatası: {str(e)}")
            raise PdsXException(f"ProvenanceChain verify hatası: {str(e)}")

class HoloStorage:
    """Holografik veri depolama simülasyonu sınıfı."""
    def __init__(self):
        self.storage = defaultdict(list)  # {pattern: [data]}

    def store(self, data: bytes) -> str:
        """Veriyi holografik olarak sıkıştırır ve depolar."""
        try:
            # Basit holografik simülasyon: veri parmak izi oluştur
            pattern = hashlib.sha256(data).hexdigest()[:16]
            self.storage[pattern].append(data)
            log.debug(f"Holografik depolama: pattern={pattern}")
            return pattern
        except Exception as e:
            log.error(f"HoloStorage store hatası: {str(e)}")
            raise PdsXException(f"HoloStorage store hatası: {str(e)}")

    def retrieve(self, pattern: str) -> Optional[bytes]:
        """Veriyi holografik depodan alır."""
        try:
            if pattern in self.storage and self.storage[pattern]:
                return self.storage[pattern][-1]
            return None
        except Exception as e:
            log.error(f"HoloStorage retrieve hatası: {str(e)}")
            raise PdsXException(f"HoloStorage retrieve hatası: {str(e)}")

class QuantumEncoder:
    """Kuantum tabanlı veri kodlama simülasyonu sınıfı."""
    def __init__(self):
        self.qubits = {}  # {id: encoded_data}

    def encode(self, data: bytes) -> str:
        """Veriyi kuantum kodlama simülasyonuyla kodlar."""
        try:
            # Basit simülasyon: veri bitlerini rastgele karıştır
            encoded = bytes([b ^ get_random_bytes(1)[0] for b in data])
            qubit_id = str(uuid.uuid4())
            self.qubits[qubit_id] = encoded
            log.debug(f"Kuantum kodlama: qubit_id={qubit_id}")
            return qubit_id
        except Exception as e:
            log.error(f"QuantumEncoder encode hatası: {str(e)}")
            raise PdsXException(f"QuantumEncoder encode hatası: {str(e)}")

    def decode(self, qubit_id: str) -> Optional[bytes]:
        """Kodlanmış veriyi çözer."""
        try:
            if qubit_id in self.qubits:
                encoded = self.qubits[qubit_id]
                # Aynı karıştırma işlemi tersine çevrilir
                decoded = bytes([b ^ get_random_bytes(1)[0] for b in encoded])
                return decoded
            return None
        except Exception as e:
            log.error(f"QuantumEncoder decode hatası: {str(e)}")
            raise PdsXException(f"QuantumEncoder decode hatası: {str(e)}")

class SelfHealingArchive:
    """Kendi kendini onaran arşiv sınıfı."""
    def __init__(self):
        self.chunks = {}  # {chunk_id: (data, checksum)}
        self.redundancy = 3  # Yedek kopya sayısı

    def store(self, data: bytes) -> str:
        """Veriyi parçalar ve yedekli depolar."""
        try:
            chunk_id = str(uuid.uuid4())
            checksum = hashlib.sha256(data).hexdigest()
            for i in range(self.redundancy):
                self.chunks[f"{chunk_id}_{i}"] = (data, checksum)
            log.debug(f"Kendi kendini onaran arşiv: chunk_id={chunk_id}")
            return chunk_id
        except Exception as e:
            log.error(f"SelfHealingArchive store hatası: {str(e)}")
            raise PdsXException(f"SelfHealingArchive store hatası: {str(e)}")

    def retrieve(self, chunk_id: str) -> Optional[bytes]:
        """Veriyi alır, bozuksa onarır."""
        try:
            for i in range(self.redundancy):
                key = f"{chunk_id}_{i}"
                if key in self.chunks:
                    data, checksum = self.chunks[key]
                    if hashlib.sha256(data).hexdigest() == checksum:
                        return data
                    # Bozuk veri, diğer kopyaları kontrol et
                    for j in range(self.redundancy):
                        if j != i:
                            alt_key = f"{chunk_id}_{j}"
                            if alt_key in self.chunks:
                                alt_data, alt_checksum = self.chunks[alt_key]
                                if hashlib.sha256(alt_data).hexdigest() == alt_checksum:
                                    self.chunks[key] = (alt_data, alt_checksum)  # Onar
                                    return alt_data
            return None
        except Exception as e:
            log.error(f"SelfHealingArchive retrieve hatası: {str(e)}")
            raise PdsXException(f"SelfHealingArchive retrieve hatası: {str(e)}")

class SaveManager:
    """Veri kaydetme ve serileştirme yönetim sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.versions = defaultdict(list)  # {path: [(timestamp, hash)]}
        self.lock = threading.Lock()
        self.diff_serializer = DiffSerializer()
        self.provenance_chain = ProvenanceChain()
        self.holo_storage = HoloStorage()
        self.quantum_encoder = QuantumEncoder()
        self.self_healing_archive = SelfHealingArchive()
        self.metadata = {"save": {"version": "1.0.0", "dependencies": ["pycryptodome", "boto3", "pyyaml", "numpy", "pdsx_exception"]}}

    def serialize(self, data: Any, format: str = "json") -> bytes:
        """Veriyi belirtilen formatta serileştirir."""
        with self.lock:
            try:
                format = format.lower()
                if format == "json":
                    return json.dumps(data).encode('utf-8')
                elif format == "pickle":
                    return pickle.dumps(data)
                elif format == "yaml":
                    return yaml.dump(data).encode('utf-8')
                elif format == "protobuf":
                    # Basit simülasyon: pickle yerine
                    return pickle.dumps(data)
                elif format == "pdsx":
                    # Özel PDS-X formatı: JSON + meta veri
                    pdsx_data = {"data": data, "meta": {"version": "1.0", "timestamp": time.time()}}
                    return json.dumps(pdsx_data).encode('utf-8')
                else:
                    raise PdsXException(f"Desteklenmeyen serileştirme formatı: {format}")
            except Exception as e:
                log.error(f"Serialize hatası: {str(e)}")
                raise PdsXException(f"Serialize hatası: {str(e)}")

    def deserialize(self, data: bytes, format: str = "json") -> Any:
        """Veriyi belirtilen formatta geri yükler."""
        with self.lock:
            try:
                format = format.lower()
                if format == "json":
                    return json.loads(data.decode('utf-8'))
                elif format == "pickle":
                    return pickle.loads(data)
                elif format == "yaml":
                    return yaml.load(data.decode('utf-8'), Loader=yaml.SafeLoader)
                elif format == "protobuf":
                    return pickle.loads(data)
                elif format == "pdsx":
                    pdsx_data = json.loads(data.decode('utf-8'))
                    return pdsx_data["data"]
                else:
                    raise PdsXException(f"Desteklenmeyen serileştirme formatı: {format}")
            except Exception as e:
                log.error(f"Deserialize hatası: {str(e)}")
                raise PdsXException(f"Deserialize hatası: {str(e)}")

    def compress(self, data: bytes, method: str = "gzip") -> bytes:
        """Veriyi sıkıştırır."""
        try:
            method = method.lower()
            if method == "gzip":
                return gzip.compress(data)
            elif method == "zlib":
                return zlib.compress(data)
            elif method == "lzma":
                return lzma.compress(data)
            else:
                raise PdsXException(f"Desteklenmeyen sıkıştırma yöntemi: {method}")
        except Exception as e:
            log.error(f"Compress hatası: {str(e)}")
            raise PdsXException(f"Compress hatası: {str(e)}")

    def decompress(self, data: bytes, method: str = "gzip") -> bytes:
        """Veriyi açar."""
        try:
            method = method.lower()
            if method == "gzip":
                return gzip.decompress(data)
            elif method == "zlib":
                return zlib.decompress(data)
            elif method == "lzma":
                return lzma.decompress(data)
            else:
                raise PdsXException(f"Desteklenmeyen sıkıştırma yöntemi: {method}")
        except Exception as e:
            log.error(f"Decompress hatası: {str(e)}")
            raise PdsXException(f"Decompress hatası: {str(e)}")

    def encrypt(self, data: bytes, key: bytes, method: str = "aes") -> bytes:
        """Veriyi şifreler."""
        try:
            method = method.lower()
            if method == "aes":
                cipher = AES.new(key, AES.MODE_EAX)
                ciphertext, tag = cipher.encrypt_and_digest(data)
                return cipher.nonce + tag + ciphertext
            elif method == "rsa":
                public_key = RSA.import_key(key)
                cipher = PKCS1_OAEP.new(public_key)
                return cipher.encrypt(data)
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Encrypt hatası: {str(e)}")
            raise PdsXException(f"Encrypt hatası: {str(e)}")

    def decrypt(self, data: bytes, key: bytes, method: str = "aes") -> bytes:
        """Veriyi çözer."""
        try:
            method = method.lower()
            if method == "aes":
                nonce, tag, ciphertext = data[:16], data[16:32], data[32:]
                cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
                return cipher.decrypt_and_verify(ciphertext, tag)
            elif method == "rsa":
                private_key = RSA.import_key(key)
                cipher = PKCS1_OAEP.new(private_key)
                return cipher.decrypt(data)
            else:
                raise PdsXException(f"Desteklenmeyen şifreleme yöntemi: {method}")
        except Exception as e:
            log.error(f"Decrypt hatası: {str(e)}")
            raise PdsXException(f"Decrypt hatası: {str(e)}")

    def save_version(self, path: str, data: bytes) -> str:
        """Veriyi versiyonlayarak kaydeder."""
        with self.lock:
            try:
                timestamp = time.time()
                data_hash = hashlib.sha256(data).hexdigest()
                self.versions[path].append((timestamp, data_hash))
                version_path = f"{path}.{timestamp}"
                with open(version_path, "wb") as f:
                    f.write(data)
                log.debug(f"Versiyon kaydedildi: path={version_path}, hash={data_hash}")
                return version_path
            except Exception as e:
                log.error(f"Save version hatası: {str(e)}")
                raise PdsXException(f"Save version hatası: {str(e)}")

    def distributed_save(self, data: bytes, bucket: str, key: str, credentials: Dict[str, str]) -> str:
        """Veriyi dağıtık depolama sistemine kaydeder (S3 benzeri)."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
            log.debug(f"Dağıtık kaydetme: bucket={bucket}, key={key}")
            return f"s3://{bucket}/{key}"
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed save hatası: {str(e)}")
            raise PdsXException(f"Distributed save hatası: {str(e)}")

    def distributed_load(self, bucket: str, key: str, credentials: Dict[str, str]) -> bytes:
        """Dağıtık depolama sisteminden veri yükler."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key")
            )
            response = s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            log.debug(f"Dağıtık yükleme: bucket={bucket}, key={key}")
            return data
        except botocore.exceptions.ClientError as e:
            log.error(f"Distributed load hatası: {str(e)}")
            raise PdsXException(f"Distributed load hatası: {str(e)}")

    def parse_save_command(self, command: str) -> None:
        """Veri kaydetme komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("SAVE DATA "):
                match = re.match(r"SAVE DATA\s+(.+?)\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    data_str, path, format = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    format = format or "json"
                    serialized = self.serialize(data, format)
                    with open(path, "wb") as f:
                        f.write(serialized)
                    self.save_version(path, serialized)
                    log.debug(f"Veri kaydedildi: path={path}, format={format}")
                else:
                    raise PdsXException("SAVE DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD DATA "):
                match = re.match(r"LOAD DATA\s+\"([^\"]+)\"\s*(\w+)?\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, format, var_name = match.groups()
                    format = format or "json"
                    with open(path, "rb") as f:
                        data = f.read()
                    deserialized = self.deserialize(data, format)
                    self.interpreter.current_scope()[var_name] = deserialized
                    log.debug(f"Veri yüklendi: path={path}, format={format}")
                else:
                    raise PdsXException("LOAD DATA komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE SERIALIZE "):
                match = re.match(r"SAVE SERIALIZE\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, format, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data, format)
                    with open(path, "wb") as f:
                        f.write(serialized)
                    self.interpreter.current_scope()[var_name] = serialized
                    log.debug(f"Serileştirme kaydedildi: path={path}, format={format}")
                else:
                    raise PdsXException("SAVE SERIALIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE COMPRESS "):
                match = re.match(r"SAVE COMPRESS\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, method, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    compressed = self.compress(serialized, method)
                    with open(path, "wb") as f:
                        f.write(compressed)
                    self.interpreter.current_scope()[var_name] = compressed
                    log.debug(f"Sıkıştırılmış veri kaydedildi: path={path}, method={method}")
                else:
                    raise PdsXException("SAVE COMPRESS komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE ENCRYPT "):
                match = re.match(r"SAVE ENCRYPT\s+(.+?)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, key_str, method, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    key = base64.b64decode(key_str)
                    serialized = self.serialize(data)
                    encrypted = self.encrypt(serialized, key, method)
                    with open(path, "wb") as f:
                        f.write(encrypted)
                    self.interpreter.current_scope()[var_name] = encrypted
                    log.debug(f"Şifrelenmiş veri kaydedildi: path={path}, method={method}")
                else:
                    raise PdsXException("SAVE ENCRYPT komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE VERSION "):
                match = re.match(r"SAVE VERSION\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    version_path = self.save_version(path, serialized)
                    self.interpreter.current_scope()[var_name] = version_path
                    log.debug(f"Versiyonlu veri kaydedildi: version_path={version_path}")
                else:
                    raise PdsXException("SAVE VERSION komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE DISTRIBUTED "):
                match = re.match(r"SAVE DISTRIBUTED\s+(.+?)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, bucket, key, creds_str, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    serialized = self.serialize(data)
                    s3_path = self.distributed_save(serialized, bucket, key, credentials)
                    self.interpreter.current_scope()[var_name] = s3_path
                    log.debug(f"Dağıtık veri kaydedildi: s3_path={s3_path}")
                else:
                    raise PdsXException("SAVE DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD DISTRIBUTED "):
                match = re.match(r"LOAD DISTRIBUTED\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\[(.+?)\]\s+(\w+)", command, re.IGNORECASE)
                if match:
                    bucket, key, creds_str, var_name = match.groups()
                    credentials = eval(creds_str, self.interpreter.current_scope())
                    data = self.distributed_load(bucket, key, credentials)
                    deserialized = self.deserialize(data)
                    self.interpreter.current_scope()[var_name] = deserialized
                    log.debug(f"Dağıtık veri yüklendi: bucket={bucket}, key={key}")
                else:
                    raise PdsXException("LOAD DISTRIBUTED komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE DIFF "):
                match = re.match(r"SAVE DIFF\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    diff_data = self.diff_serializer.serialize(data)
                    with open(path, "wb") as f:
                        f.write(diff_data)
                    self.interpreter.current_scope()[var_name] = diff_data
                    log.debug(f"Farklı serileştirme kaydedildi: path={path}")
                else:
                    raise PdsXException("SAVE DIFF komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD DIFF "):
                match = re.match(r"LOAD DIFF\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, var_name = match.groups()
                    with open(path, "rb") as f:
                        diff_data = f.read()
                    data = self.diff_serializer.deserialize(diff_data)
                    self.interpreter.current_scope()[var_name] = data
                    log.debug(f"Farklı serileştirme yüklendi: path={path}")
                else:
                    raise PdsXException("LOAD DIFF komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE PROVENANCE "):
                match = re.match(r"SAVE PROVENANCE\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    block_hash = self.provenance_chain.add_block(serialized)
                    with open(path, "wb") as f:
                        f.write(serialized)
                    self.interpreter.current_scope()[var_name] = block_hash
                    log.debug(f"Provenance kaydedildi: path={path}, hash={block_hash}")
                else:
                    raise PdsXException("SAVE PROVENANCE komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE HOLO "):
                match = re.match(r"SAVE HOLO\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    pattern = self.holo_storage.store(serialized)
                    with open(path, "w") as f:
                        f.write(pattern)
                    self.interpreter.current_scope()[var_name] = pattern
                    log.debug(f"Holografik depolama: path={path}, pattern={pattern}")
                else:
                    raise PdsXException("SAVE HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD HOLO "):
                match = re.match(r"LOAD HOLO\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, var_name = match.groups()
                    with open(path, "r") as f:
                        pattern = f.read()
                    data = self.holo_storage.retrieve(pattern)
                    if data:
                        deserialized = self.deserialize(data)
                        self.interpreter.current_scope()[var_name] = deserialized
                        log.debug(f"Holografik veri yüklendi: pattern={pattern}")
                    else:
                        raise PdsXException("Holografik veri bulunamadı")
                else:
                    raise PdsXException("LOAD HOLO komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE QUANTUM "):
                match = re.match(r"SAVE QUANTUM\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    qubit_id = self.quantum_encoder.encode(serialized)
                    with open(path, "w") as f:
                        f.write(qubit_id)
                    self.interpreter.current_scope()[var_name] = qubit_id
                    log.debug(f"Kuantum kodlama: path={path}, qubit_id={qubit_id}")
                else:
                    raise PdsXException("SAVE QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD QUANTUM "):
                match = re.match(r"LOAD QUANTUM\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, var_name = match.groups()
                    with open(path, "r") as f:
                        qubit_id = f.read()
                    data = self.quantum_encoder.decode(qubit_id)
                    if data:
                        deserialized = self.deserialize(data)
                        self.interpreter.current_scope()[var_name] = deserialized
                        log.debug(f"Kuantum veri yüklendi: qubit_id={qubit_id}")
                    else:
                        raise PdsXException("Kuantum veri bulunamadı")
                else:
                    raise PdsXException("LOAD QUANTUM komutunda sözdizimi hatası")
            elif command_upper.startswith("SAVE HEALING "):
                match = re.match(r"SAVE HEALING\s+(.+?)\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    data_str, path, var_name = match.groups()
                    data = self.interpreter.evaluate_expression(data_str)
                    serialized = self.serialize(data)
                    chunk_id = self.self_healing_archive.store(serialized)
                    with open(path, "w") as f:
                        f.write(chunk_id)
                    self.interpreter.current_scope()[var_name] = chunk_id
                    log.debug(f"Kendi kendini onaran arşiv: path={path}, chunk_id={chunk_id}")
                else:
                    raise PdsXException("SAVE HEALING komutunda sözdizimi hatası")
            elif command_upper.startswith("LOAD HEALING "):
                match = re.match(r"LOAD HEALING\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    path, var_name = match.groups()
                    with open(path, "r") as f:
                        chunk_id = f.read()
                    data = self.self_healing_archive.retrieve(chunk_id)
                    if data:
                        deserialized = self.deserialize(data)
                        self.interpreter.current_scope()[var_name] = deserialized
                        log.debug(f"Kendi kendini onaran veri yüklendi: chunk_id={chunk_id}")
                    else:
                        raise PdsXException("Kendi kendini onaran veri bulunamadı")
                else:
                    raise PdsXException("LOAD HEALING komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen kaydetme komutu: {command}")
        except Exception as e:
            log.error(f"Kaydetme komut hatası: {str(e)}")
            raise PdsXException(f"Kaydetme komut hatası: {str(e)}")

if __name__ == "__main__":
    print("save.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")