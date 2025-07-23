# lib_db.py - PDS-X BASIC v14u Veritabanı Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import sqlite3
import mysql.connector
import psycopg2
import pandas as pd
import numpy as np
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from contextlib import contextmanager
import threading
import pickle
from collections import defaultdict
from datetime import datetime

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("lib_db")

class PdsXException(Exception):
    pass

class ISAMSession:
    """ISAM veritabanı oturumu."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = defaultdict(list)
        self.index = {}
        self.lock = threading.Lock()
        self._load_data()

    def _load_data(self) -> None:
        """ISAM dosyasını yükler."""
        try:
            if Path(self.file_path).exists():
                with open(self.file_path, "rb") as f:
                    self.data.update(pickle.load(f))
                log.debug(f"ISAM dosyası yüklendi: {self.file_path}")
        except Exception as e:
            log.error(f"ISAM yükleme hatası: {str(e)}")
            raise PdsXException(f"ISAM yükleme hatası: {str(e)}")

    def write(self, table: str, key: Any, value: Any) -> None:
        """ISAM tablosuna veri yazar."""
        with self.lock:
            try:
                self.data[table].append((key, value))
                self.index[(table, key)] = len(self.data[table]) - 1
                with open(self.file_path, "wb") as f:
                    pickle.dump(dict(self.data), f)
                log.debug(f"ISAM yazma tamamlandı: {table}, {key}")
            except Exception as e:
                log.error(f"ISAM yazma hatası: {str(e)}")
                raise PdsXException(f"ISAM yazma hatası: {str(e)}")

    def read(self, table: str, key: Any) -> Any:
        """ISAM tablosundan veri okur."""
        with self.lock:
            try:
                idx = self.index.get((table, key))
                if idx is not None:
                    return self.data[table][idx][1]
                log.debug(f"ISAM okuma: {table}, {key} bulunamadı")
                return None
            except Exception as e:
                log.error(f"ISAM okuma hatası: {str(e)}")
                raise PdsXException(f"ISAM okuma hatası: {str(e)}")

    def close(self) -> None:
        """ISAM oturumunu kapatır."""
        with self.lock:
            self.data.clear()
            self.index.clear()
            log.debug(f"ISAM oturumu kapatıldı: {self.file_path}")

class DBSession:
    """SQL veritabanı oturumu."""
    def __init__(self):
        self.connections = {}
        self.cursors = {}
        self.lock = threading.Lock()
        self.auto_connect = False
        self.connection_pool = defaultdict(list)

    def connect(self, db_type: str, **kwargs) -> str:
        """Veritabanına bağlanır."""
        with self.lock:
            conn_id = f"{db_type}_{id(kwargs)}"
            try:
                if db_type.lower() == "sqlite":
                    conn = sqlite3.connect(kwargs.get("database", ":memory:"))
                elif db_type.lower() == "mysql":
                    conn = mysql.connector.connect(
                        host=kwargs.get("host", "localhost"),
                        user=kwargs.get("user", "root"),
                        password=kwargs.get("password", ""),
                        database=kwargs.get("database", "mysql")
                    )
                elif db_type.lower() == "postgresql":
                    conn = psycopg2.connect(
                        host=kwargs.get("host", "localhost"),
                        user=kwargs.get("user", "postgres"),
                        password=kwargs.get("password", ""),
                        database=kwargs.get("database", "postgres")
                    )
                else:
                    raise PdsXException(f"Desteklenmeyen veritabanı tipi: {db_type}")
                
                cursor = conn.cursor()
                self.connections[conn_id] = conn
                self.cursors[conn_id] = cursor
                self.connection_pool[db_type].append(conn)
                log.debug(f"Veritabanı bağlantısı kuruldu: {conn_id}")
                return conn_id
            except Exception as e:
                log.error(f"Veritabanı bağlantı hatası: {str(e)}")
                raise PdsXException(f"Veritabanı bağlantı hatası: {str(e)}")

    @contextmanager
    def get_connection(self, db_type: str) -> Any:
        """Bağlantı havuzundan bağlantı sağlar."""
        with self.lock:
            if self.connection_pool[db_type]:
                conn = self.connection_pool[db_type].pop(0)
                try:
                    yield conn
                finally:
                    self.connection_pool[db_type].append(conn)
            else:
                raise PdsXException(f"Boşta bağlantı yok: {db_type}")

    def execute(self, conn_id: str, query: str, params: Optional[tuple] = None) -> None:
        """SQL sorgusu yürütür."""
        cursor = self.cursors.get(conn_id)
        conn = self.connections.get(conn_id)
        if not cursor or not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.execute(query, params or ())
            conn.commit()
            log.debug(f"SQL sorgusu yürütüldü: {query[:50]}...")
        except Exception as e:
            conn.rollback()
            log.error(f"SQL yürütme hatası: {str(e)}")
            raise PdsXException(f"SQL yürütme hatası: {str(e)}")

    def query(self, conn_id: str, query: str, params: Optional[tuple] = None) -> List:
        """SQL sorgusu ile veri çeker."""
        cursor = self.cursors.get(conn_id)
        if not cursor:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            log.debug(f"SQL sorgusu tamamlandı: {query[:50]}...")
            return result
        except Exception as e:
            log.error(f"SQL sorgu hatası: {str(e)}")
            raise PdsXException(f"SQL sorgu hatası: {str(e)}")

    def sql_result_to_array(self, result: List) -> np.ndarray:
        """SQL sonucunu NumPy dizisine dönüştürür."""
        try:
            return np.array(result)
        except Exception as e:
            log.error(f"SQL to array dönüşüm hatası: {str(e)}")
            raise PdsXException(f"SQL to array dönüşüm hatası: {str(e)}")

    def sql_result_to_struct(self, result: List, columns: List[str]) -> List[Dict]:
        """SQL sonucunu yapı listesine dönüştürür."""
        try:
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            log.error(f"SQL to struct dönüşüm hatası: {str(e)}")
            raise PdsXException(f"SQL to struct dönüşüm hatası: {str(e)}")

    def sql_result_to_dataframe(self, result: List, columns: List[str]) -> pd.DataFrame:
        """SQL sonucunu pandas DataFrame'e dönüştürür."""
        try:
            return pd.DataFrame(result, columns=columns)
        except Exception as e:
            log.error(f"SQL to DataFrame dönüşüm hatası: {str(e)}")
            raise PdsXException(f"SQL to DataFrame dönüşüm hatası: {str(e)}")

    def begin_transaction(self, conn_id: str) -> None:
        """İşlemi başlatır."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.autocommit = False
            log.debug(f"İşlem başlatıldı: {conn_id}")
        except Exception as e:
            log.error(f"İşlem başlatma hatası: {str(e)}")
            raise PdsXException(f"İşlem başlatma hatası: {str(e)}")

    def commit(self, conn_id: str) -> None:
        """İşlemi onaylar."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.commit()
            conn.autocommit = True
            log.debug(f"İşlem onaylandı: {conn_id}")
        except Exception as e:
            log.error(f"İşlem onaylama hatası: {str(e)}")
            raise PdsXException(f"İşlem onaylama hatası: {str(e)}")

    def rollback(self, conn_id: str) -> None:
        """İşlemi geri alır."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.rollback()
            conn.autocommit = True
            log.debug(f"İşlem geri alındı: {conn_id}")
        except Exception as e:
            log.error(f"İşlem geri alma hatası: {str(e)}")
            raise PdsXException(f"İşlem geri alma hatası: {str(e)}")

    def close(self, conn_id: str) -> None:
        """Veritabanı bağlantısını kapatır."""
        conn = self.connections.get(conn_id)
        cursor = self.cursors.get(conn_id)
        if not conn or not cursor:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.close()
            conn.close()
            with self.lock:
                del self.connections[conn_id]
                del self.cursors[conn_id]
                self.connection_pool[conn_id.split("_")[0]].remove(conn)
            log.debug(f"Veritabanı bağlantısı kapatıldı: {conn_id}")
        except Exception as e:
            log.error(f"Veritabanı kapatma hatası: {str(e)}")
            raise PdsXException(f"Veritabanı kapatma hatası: {str(e)}")

class LibDB:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.db_session = DBSession()
        self.isam_sessions = {}
        self.metadata = {"lib_db": {"version": "1.0.0", "dependencies": ["sqlite3", "mysql-connector-python", "psycopg2", "pandas"]}}
        self.lock = threading.Lock()

    def open_database(self, db_type: str, **kwargs) -> str:
        """Veritabanı bağlantısı açar."""
        return self.db_session.connect(db_type, **kwargs)

    def open_isam(self, file_path: str) -> str:
        """ISAM veritabanı açar."""
        with self.lock:
            isam_id = f"isam_{id(file_path)}"
            if isam_id not in self.isam_sessions:
                self.isam_sessions[isam_id] = ISAMSession(file_path)
                log.debug(f"ISAM veritabanı açıldı: {isam_id}")
            return isam_id

    def parse_db_command(self, command: str) -> None:
        """Veritabanı komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("OPEN DATABASE "):
                match = re.match(r"OPEN DATABASE\s+(\w+)\s+(\w+)(?:\s+(.+))?", command, re.IGNORECASE)
                if match:
                    db_type, conn_id, params_str = match.groups()
                    params = {}
                    if params_str:
                        for param in params_str.split(","):
                            key, value = param.strip().split("=")
                            params[key.strip().lower()] = value.strip().strip('"')
                    conn_id = self.open_database(db_type, **params)
                    self.interpreter.current_scope()[conn_id] = conn_id
                else:
                    raise PdsXException("OPEN DATABASE komutunda sözdizimi hatası")
            elif command_upper.startswith("SET SQL AUTO "):
                match = re.match(r"SET SQL AUTO\s+(ON|OFF)", command, re.IGNORECASE)
                if match:
                    self.db_session.auto_connect = (match.group(1).upper() == "ON")
                    log.debug(f"SQL otomatik bağlantı: {self.db_session.auto_connect}")
                else:
                    raise PdsXException("SET SQL AUTO komutunda sözdizimi hatası")
            elif command_upper.startswith("DBEXEC "):
                match = re.match(r"DBEXEC\s+(\w+)\s+\"([^\"]+)\"\s*(\(.+\))?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    params = tuple(eval(params_str)) if params_str else None
                    self.db_session.execute(conn_id, query, params)
                    if var_name:
                        self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("DBEXEC komutunda sözdizimi hatası")
            elif command_upper.startswith("DBQUERY "):
                match = re.match(r"DBQUERY\s+(\w+)\s+\"([^\"]+)\"\s*(\(.+\))?\s*(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    params = tuple(eval(params_str)) if params_str else None
                    result = self.db_session.query(conn_id, query, params)
                    self.interpreter.current_scope()[var_name] = result
                else:
                    raise PdsXException("DBQUERY komutunda sözdizimi hatası")
            elif command_upper.startswith("SQL RESULT TO ARRAY "):
                match = re.match(r"SQL RESULT TO ARRAY\s+(\w+)\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    var_name, result_var = match.groups()
                    result = self.interpreter.current_scope().get(result_var or "_SQL_RESULT", [])
                    array = self.db_session.sql_result_to_array(result)
                    self.interpreter.current_scope()[var_name] = array
                else:
                    raise PdsXException("SQL RESULT TO ARRAY komutunda sözdizimi hatası")
            elif command_upper.startswith("SQL RESULT TO STRUCT "):
                match = re.match(r"SQL RESULT TO STRUCT\s+(\w+)\s*(\w+)?\s+\[(.*?)\]", command, re.IGNORECASE)
                if match:
                    var_name, result_var, columns_str = match.groups()
                    columns = [c.strip() for c in columns_str.split(",")]
                    result = self.interpreter.current_scope().get(result_var or "_SQL_RESULT", [])
                    struct = self.db_session.sql_result_to_struct(result, columns)
                    self.interpreter.current_scope()[var_name] = struct
                else:
                    raise PdsXException("SQL RESULT TO STRUCT komutunda sözdizimi hatası")
            elif command_upper.startswith("SQL RESULT TO DATAFRAME "):
                match = re.match(r"SQL RESULT TO DATAFRAME\s+(\w+)\s*(\w+)?\s+\[(.*?)\]", command, re.IGNORECASE)
                if match:
                    var_name, result_var, columns_str = match.groups()
                    columns = [c.strip() for c in columns_str.split(",")]
                    result = self.interpreter.current_scope().get(result_var or "_SQL_RESULT", [])
                    df = self.db_session.sql_result_to_dataframe(result, columns)
                    self.interpreter.current_scope()[var_name] = df
                else:
                    raise PdsXException("SQL RESULT TO DATAFRAME komutunda sözdizimi hatası")
            elif command_upper.startswith("BEGIN TRANSACTION "):
                match = re.match(r"BEGIN TRANSACTION\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.db_session.begin_transaction(conn_id)
                else:
                    raise PdsXException("BEGIN TRANSACTION komutunda sözdizimi hatası")
            elif command_upper.startswith("COMMIT "):
                match = re.match(r"COMMIT\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.db_session.commit(conn_id)
                else:
                    raise PdsXException("COMMIT komutunda sözdizimi hatası")
            elif command_upper.startswith("ROLLBACK "):
                match = re.match(r"ROLLBACK\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.db_session.rollback(conn_id)
                else:
                    raise PdsXException("ROLLBACK komutunda sözdizimi hatası")
            elif command_upper.startswith("CLOSE DATABASE "):
                match = re.match(r"CLOSE DATABASE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.db_session.close(conn_id)
                else:
                    raise PdsXException("CLOSE DATABASE komutunda sözdizimi hatası")
            elif command_upper.startswith("ISAMOPEN "):
                match = re.match(r"ISAMOPEN\s+\"([^\"]+)\"\s+(\w+)", command, re.IGNORECASE)
                if match:
                    file_path, isam_id = match.groups()
                    isam_id = self.open_isam(file_path)
                    self.interpreter.current_scope()[isam_id] = isam_id
                else:
                    raise PdsXException("ISAMOPEN komutunda sözdizimi hatası")
            elif command_upper.startswith("ISAMWRITE "):
                match = re.match(r"ISAMWRITE\s+(\w+)\s+(\w+)\s+(\w+)\s+(.+)", command, re.IGNORECASE)
                if match:
                    isam_id, table, key, value = match.groups()
                    value = self.interpreter.evaluate_expression(value)
                    isam = self.isam_sessions.get(isam_id)
                    if isam:
                        isam.write(table, key, value)
                    else:
                        raise PdsXException(f"ISAM oturumu bulunamadı: {isam_id}")
                else:
                    raise PdsXException("ISAMWRITE komutunda sözdizimi hatası")
            elif command_upper.startswith("ISAMREAD "):
                match = re.match(r"ISAMREAD\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    isam_id, table, key, var_name = match.groups()
                    isam = self.isam_sessions.get(isam_id)
                    if isam:
                        result = isam.read(table, key)
                        self.interpreter.current_scope()[var_name] = result
                    else:
                        raise PdsXException(f"ISAM oturumu bulunamadı: {isam_id}")
                else:
                    raise PdsXException("ISAMREAD komutunda sözdizimi hatası")
            elif command_upper.startswith("ISAMCLOSE "):
                match = re.match(r"ISAMCLOSE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    isam_id = match.group(1)
                    isam = self.isam_sessions.get(isam_id)
                    if isam:
                        isam.close()
                        with self.lock:
                            del self.isam_sessions[isam_id]
                    else:
                        raise PdsXException(f"ISAM oturumu bulunamadı: {isam_id}")
                else:
                    raise PdsXException("ISAMCLOSE komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen veritabanı komutu: {command}")
        except Exception as e:
            log.error(f"Veritabanı komut hatası: {str(e)}")
            raise PdsXException(f"Veritabanı komut hatası: {str(e)}")

if __name__ == "__main__":
    print("lib_db.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")