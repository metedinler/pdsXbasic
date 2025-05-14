# sqlite.py - PDS-X BASIC v14u SQLite Veritabanı Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import sqlite3
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
import threading
from collections import defaultdict
import pandas as pd
import numpy as np

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("sqlite")

class PdsXException(Exception):
    pass

class SQLiteSession:
    """SQLite veritabanı oturumu."""
    def __init__(self):
        self.connections = {}
        self.cursors = {}
        self.transactions = {}
        self.lock = threading.Lock()
        self.connection_pool = []
        self.max_pool_size = 10

    def connect(self, db_path: str, conn_id: str) -> None:
        """SQLite veritabanına bağlanır."""
        with self.lock:
            if conn_id in self.connections:
                raise PdsXException(f"Bağlantı ID zaten kullanımda: {conn_id}")
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cursor = conn.cursor()
                self.connections[conn_id] = conn
                self.cursors[conn_id] = cursor
                self.connection_pool.append(conn)
                if len(self.connection_pool) > self.max_pool_size:
                    oldest_conn = self.connection_pool.pop(0)
                    oldest_conn.close()
                log.debug(f"SQLite bağlantısı kuruldu: {conn_id}, {db_path}")
            except sqlite3.Error as e:
                log.error(f"SQLite bağlantı hatası: {str(e)}")
                raise PdsXException(f"SQLite bağlantı hatası: {str(e)}")

    @contextmanager
    def get_connection(self) -> Any:
        """Bağlantı havuzundan bağlantı sağlar."""
        with self.lock:
            if self.connection_pool:
                conn = self.connection_pool.pop(0)
                try:
                    yield conn
                finally:
                    self.connection_pool.append(conn)
            else:
                raise PdsXException("Boşta SQLite bağlantısı yok")

    def execute(self, conn_id: str, query: str, params: Optional[Tuple] = None) -> None:
        """SQLite sorgusu yürütür."""
        cursor = self.cursors.get(conn_id)
        conn = self.connections.get(conn_id)
        if not cursor or not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.execute(query, params or ())
            conn.commit()
            log.debug(f"SQLite sorgusu yürütüldü: {query[:50]}...")
        except sqlite3.Error as e:
            conn.rollback()
            log.error(f"SQLite yürütme hatası: {str(e)}")
            raise PdsXException(f"SQLite yürütme hatası: {str(e)}")

    def query(self, conn_id: str, query: str, params: Optional[Tuple] = None) -> List:
        """SQLite sorgusu ile veri çeker."""
        cursor = self.cursors.get(conn_id)
        if not cursor:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.execute(query, params or ())
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            result = cursor.fetchall()
            log.debug(f"SQLite sorgusu tamamlandı: {query[:50]}...")
            return result, columns
        except sqlite3.Error as e:
            log.error(f"SQLite sorgu hatası: {str(e)}")
            raise PdsXException(f"SQLite sorgu hatası: {str(e)}")

    def begin_transaction(self, conn_id: str) -> None:
        """SQLite işlem başlatır."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.execute("BEGIN TRANSACTION")
            self.transactions[conn_id] = True
            log.debug(f"SQLite işlem başlatıldı: {conn_id}")
        except sqlite3.Error as e:
            log.error(f"SQLite işlem başlatma hatası: {str(e)}")
            raise PdsXException(f"SQLite işlem başlatma hatası: {str(e)}")

    def commit(self, conn_id: str) -> None:
        """SQLite işlemini onaylar."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.commit()
            self.transactions.pop(conn_id, None)
            log.debug(f"SQLite işlem onaylandı: {conn_id}")
        except sqlite3.Error as e:
            log.error(f"SQLite işlem onaylama hatası: {str(e)}")
            raise PdsXException(f"SQLite işlem onaylama hatası: {str(e)}")

    def rollback(self, conn_id: str) -> None:
        """SQLite işlemini geri alır."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            conn.rollback()
            self.transactions.pop(conn_id, None)
            log.debug(f"SQLite işlem geri alındı: {conn_id}")
        except sqlite3.Error as e:
            log.error(f"SQLite işlem geri alma hatası: {str(e)}")
            raise PdsXException(f"SQLite işlem geri alma hatası: {str(e)}")

    def close(self, conn_id: str) -> None:
        """SQLite bağlantısını kapatır."""
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
                self.connection_pool.remove(conn)
            log.debug(f"SQLite bağlantısı kapatıldı: {conn_id}")
        except sqlite3.Error as e:
            log.error(f"SQLite kapatma hatası: {str(e)}")
            raise PdsXException(f"SQLite kapatma hatası: {str(e)}")

    def vacuum(self, conn_id: str) -> None:
        """SQLite veritabanını optimize eder."""
        cursor = self.cursors.get(conn_id)
        if not cursor:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            cursor.execute("VACUUM")
            log.debug(f"SQLite VACUUM yürütüldü: {conn_id}")
        except sqlite3.Error as e:
            log.error(f"SQLite VACUUM hatası: {str(e)}")
            raise PdsXException(f"SQLite VACUUM hatası: {str(e)}")

    def backup(self, conn_id: str, backup_path: str) -> None:
        """SQLite veritabanını yedekler."""
        conn = self.connections.get(conn_id)
        if not conn:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                conn.backup(backup_conn)
            backup_conn.close()
            log.debug(f"SQLite yedekleme tamamlandı: {conn_id}, {backup_path}")
        except sqlite3.Error as e:
            log.error(f"SQLite yedekleme hatası: {str(e)}")
            raise PdsXException(f"SQLite yedekleme hatası: {str(e)}")

    def get_schema(self, conn_id: str, table: Optional[str] = None) -> Dict:
        """Veritabanı veya tablo şemasını döndürür."""
        cursor = self.cursors.get(conn_id)
        if not cursor:
            raise PdsXException(f"Bağlantı bulunamadı: {conn_id}")
        try:
            if table:
                cursor.execute("PRAGMA table_info(?)", (table,))
                schema = {"table": table, "columns": cursor.fetchall()}
            else:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                schema = {"tables": tables}
            log.debug(f"SQLite şema alındı: {conn_id}, {table or 'all'}")
            return schema
        except sqlite3.Error as e:
            log.error(f"SQLite şema alma hatası: {str(e)}")
            raise PdsXException(f"SQLite şema alma hatası: {str(e)}")

class SQLiteManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.session = SQLiteSession()
        self.metadata = {"sqlite": {"version": "1.0.0", "dependencies": ["sqlite3", "pandas", "numpy"]}}
        self.lock = threading.Lock()

    def parse_sqlite_command(self, command: str) -> None:
        """SQLite komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("SQLITE CONNECT "):
                match = re.match(r"SQLITE CONNECT\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    db_path, conn_id = match.groups()
                    self.session.connect(db_path, conn_id)
                    self.interpreter.current_scope()[conn_id] = conn_id
                else:
                    raise PdsXException("SQLITE CONNECT komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE EXECUTE "):
                match = re.match(r"SQLITE EXECUTE\s+(\w+)\s+\"([^\"]+)\"\s*(\(.+\))?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    params = tuple(eval(params_str)) if params_str else None
                    self.session.execute(conn_id, query, params)
                    if var_name:
                        self.interpreter.current_scope()[var_name] = True
                else:
                    raise PdsXException("SQLITE EXECUTE komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE QUERY "):
                match = re.match(r"SQLITE QUERY\s+(\w+)\s+\"([^\"]+)\"\s*(\(.+\))?\s*(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id, query, params_str, var_name = match.groups()
                    params = tuple(eval(params_str)) if params_str else None
                    result, columns = self.session.query(conn_id, query, params)
                    self.interpreter.current_scope()[var_name] = result
                    self.interpreter.current_scope()[f"{var_name}_COLUMNS"] = columns
                else:
                    raise PdsXException("SQLITE QUERY komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE BEGIN TRANSACTION "):
                match = re.match(r"SQLITE BEGIN TRANSACTION\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.session.begin_transaction(conn_id)
                else:
                    raise PdsXException("SQLITE BEGIN TRANSACTION komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE COMMIT "):
                match = re.match(r"SQLITE COMMIT\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.session.commit(conn_id)
                else:
                    raise PdsXException("SQLITE COMMIT komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE ROLLBACK "):
                match = re.match(r"SQLITE ROLLBACK\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.session.rollback(conn_id)
                else:
                    raise PdsXException("SQLITE ROLLBACK komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE CLOSE "):
                match = re.match(r"SQLITE CLOSE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.session.close(conn_id)
                else:
                    raise PdsXException("SQLITE CLOSE komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE VACUUM "):
                match = re.match(r"SQLITE VACUUM\s+(\w+)", command, re.IGNORECASE)
                if match:
                    conn_id = match.group(1)
                    self.session.vacuum(conn_id)
                else:
                    raise PdsXException("SQLITE VACUUM komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE BACKUP "):
                match = re.match(r"SQLITE BACKUP\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    conn_id, backup_path = match.groups()
                    self.session.backup(conn_id, backup_path)
                else:
                    raise PdsXException("SQLITE BACKUP komutunda sözdizimi hatası")
            elif command_upper.startswith("SQLITE SCHEMA "):
                match = re.match(r"SQLITE SCHEMA\s+(\w+)\s*(\w+)?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    conn_id, table, var_name = match.groups()
                    result = self.session.get_schema(conn_id, table)
                    self.interpreter.current_scope()[var_name or "_SQLITE_SCHEMA"] = result
                else:
                    raise PdsXException("SQLITE SCHEMA komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen SQLite komutu: {command}")
        except Exception as e:
            log.error(f"SQLite komut hatası: {str(e)}")
            raise PdsXException(f"SQLite komut hatası: {str(e)}")

if __name__ == "__main__":
    print("sqlite.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")