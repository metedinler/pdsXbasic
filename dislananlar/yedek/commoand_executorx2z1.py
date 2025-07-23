
# command_executor3.py - PDS-X Enhanced Command Executor (command_executor.py için devam)
# Version: 1.0.0
# Date: June 14, 2025
import re
import asyncio
import json
import numpy as np
import pandas as pd
import aiohttp
import aiofiles
import ctypes
import time
import os
import subprocess
import tkinter as tk
from typing import Dict, Callable, Optional
from pdsx_exception import PdsXException
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import paho.mqtt.client as mqtt
import networkx as nx
import matplotlib.pyplot as plt
import sqlite3
import graphviz
from concurrent.futures import ThreadPoolExecutor

class CommandExecutor:
    async def handle_edit_module(self, command: str, scope_name: str) -> None:
        """EDIT MODULE komutunu işler."""
        match = re.match(r"EDIT\s+MODULE\s+(\w+)\s+SOURCE\s+\"([^\"]+)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"EDIT MODULE komutunda sözdizimi hatası: {command}")
        module_name, source_path = match.groups()
        try:
            async with aiofiles.open(source_path, "r", encoding="utf-8") as f:
                new_source = await f.read()
            self.interpreter.module_manager.edit_module(module_name, new_source)
            self.interpreter.object_counter["EDIT MODULE"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "EDIT MODULE",
                "name": module_name,
                "atom": command
            }
        except FileNotFoundError:
            raise PdsXException(f"Kaynak dosya bulunamadı: {source_path}")
        except Exception as e:
            raise PdsXException(f"Modül düzenleme hatası: {str(e)}")

    async def handle_thread_create(self, command: str, scope_name: str) -> None:
        """THREAD CREATE komutunu işler."""
        match = re.match(r"THREAD\s+CREATE\s+(\w+)\s+EXECUTE\s+(\w+)\s+\((.*?)\)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"THREAD CREATE komutunda sözdizimi hatası: {command}")
        thread_id, sub_name, params = match.groups()
        if sub_name not in self.interpreter.subs:
            raise PdsXException(f"Alt program bulunamadı: {sub_name}")
        param_values = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()]
        async def thread_task():
            new_scope = dict(zip(self.interpreter.subs[sub_name]["params"], param_values))
            self.interpreter.local_scopes.append(new_scope)
            await self.execute(self.interpreter.subs[sub_name]["body"], scope_name)
            self.interpreter.local_scopes.pop()
        with ThreadPoolExecutor() as executor:
            executor.submit(lambda: asyncio.run(thread_task()))
        scope = self.interpreter.current_scope()
        scope[thread_id] = thread_id
        self.interpreter.object_counter["THREAD CREATE"] += 1
        self.interpreter.object_registry[id(thread_id)] = {
            "type": "THREAD CREATE",
            "name": thread_id,
            "atom": command
        }

    async def handle_db_connect(self, command: str, scope_name: str) -> None:
        """DB CONNECT komutunu işler."""
        match = re.match(r"DB\s+CONNECT\s+\"([^\"]+)\"\s+TYPE\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB CONNECT komutunda sözdizimi hatası: {command}")
        db_url, db_type, conn_var = match.groups()
        conn = await self.interpreter.database.connect(db_url, db_type.upper())
        self.interpreter.db_connections[conn_var] = conn
        scope = self.interpreter.current_scope()
        scope[conn_var] = conn
        self.interpreter.object_counter["DB CONNECT"] += 1
        self.interpreter.object_registry[id(conn)] = {
            "type": "DB CONNECT",
            "name": conn_var,
            "atom": f"{db_type}:{db_url}"
        }

    async def handle_db_query(self, command: str, scope_name: str) -> None:
        """DB QUERY komutunu işler."""
        match = re.match(r"DB\s+QUERY\s+(\w+)\s+SQL\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB QUERY komutunda sözdizimi hatası: {command}")
        conn_var, sql, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        result = await self.interpreter.database.execute_query(self.interpreter.db_connections[conn_var], sql)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB QUERY"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB QUERY",
            "name": result_var,
            "atom": sql[:100]
        }

    async def handle_db_async_query(self, command: str, scope_name: str) -> None:
        """DB ASYNC QUERY komutunu işler."""
        match = re.match(r"DB\s+ASYNC\s+QUERY\s+(\w+)\s+SQL\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB ASYNC QUERY komutunda sözdizimi hatası: {command}")
        conn_var, sql, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        result = await self.interpreter.database.execute_async_query(self.interpreter.db_connections[conn_var], sql)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB ASYNC QUERY"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB ASYNC QUERY",
            "name": result_var,
            "atom": sql[:100]
        }

    async def handle_db_create_table(self, command: str, scope_name: str) -> None:
        """DB CREATE TABLE komutunu işler."""
        match = re.match(r"DB\s+CREATE\s+TABLE\s+(\w+)\s+\[(.+?)\]\s+ON\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB CREATE TABLE komutunda sözdizimi hatası: {command}")
        table_name, columns_str, conn_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        columns = []
        for col_def in columns_str.split(","):
            col_match = re.match(r"(\w+)\s+(\w+)", col_def.strip(), re.IGNORECASE)
            if col_match:
                col_name, col_type = col_match.groups()
                sql_type = {
                    "STRING": "TEXT",
                    "INTEGER": "INTEGER",
                    "DOUBLE": "REAL",
                    "BOOLEAN": "BOOLEAN"
                }.get(col_type.upper(), "TEXT")
                columns.append(f"{col_name} {sql_type}")
        sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        await self.interpreter.database.execute_query(self.interpreter.db_connections[conn_var], sql)
        self.interpreter.object_counter["DB CREATE TABLE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "DB CREATE TABLE",
            "name": table_name,
            "atom": command
        }

    async def handle_db_isam_create(self, command: str, scope_name: str) -> None:
        """DB ISAM CREATE komutunu işler."""
        match = re.match(r"DB\s+ISAM\s+CREATE\s+(\w+)\s+\[(.+?)\]\s+ON\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB ISAM CREATE komutunda sözdizimi hatası: {command}")
        table_name, columns_str, conn_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        await self.interpreter.database.create_isam_table(self.interpreter.db_connections[conn_var], table_name, columns_str)
        self.interpreter.object_counter["DB ISAM CREATE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "DB ISAM CREATE",
            "name": table_name,
            "atom": command
        }

    async def handle_db_isam_insert(self, command: str, scope_name: str) -> None:
        """DB ISAM INSERT komutunu işler."""
        match = re.match(r"DB\s+ISAM\s+INSERT\s+(\w+)\s+DATA\s+\[(.+?)\]\s+ON\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB ISAM INSERT komutunda sözdizimi hatası: {command}")
        table_name, data_str, conn_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        data = json.loads(f"[{data_str}]")
        await self.interpreter.database.insert_isam_data(self.interpreter.db_connections[conn_var], table_name, data)
        self.interpreter.object_counter["DB ISAM INSERT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "DB ISAM INSERT",
            "name": table_name,
            "atom": command
        }

    async def handle_db_isam_search(self, command: str, scope_name: str) -> None:
        """DB ISAM SEARCH komutunu işler."""
        match = re.match(r"DB\s+ISAM\s+SEARCH\s+(\w+)\s+KEY\s+(.+)\s+ON\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB ISAM SEARCH komutunda sözdizimi hatası: {command}")
        table_name, key_expr, conn_var, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        key = self.interpreter.evaluate_expression(key_expr, scope_name)
        result = await self.interpreter.database.search_isam_data(self.interpreter.db_connections[conn_var], table_name, key)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB ISAM SEARCH"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB ISAM SEARCH",
            "name": result_var,
            "atom": str(result)[:100]
        }

    async def handle_db_analyze(self, command: str, scope_name: str) -> None:
        """DB ANALYZE komutunu işler."""
        match = re.match(r"DB\s+ANALYZE\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB ANALYZE komutunda sözdizimi hatası: {command}")
        conn_var, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        stats = await self.interpreter.database.analyze(self.interpreter.db_connections[conn_var])
        scope = self.interpreter.current_scope()
        scope[result_var] = stats
        self.interpreter.object_counter["DB ANALYZE"] += 1
        self.interpreter.object_registry[id(stats)] = {
            "type": "DB ANALYZE",
            "name": result_var,
            "atom": str(stats)[:100]
        }

    async def handle_db_visualize(self, command: str, scope_name: str) -> None:
        """DB VISUALIZE komutunu işler."""
        match = re.match(r"DB\s+VISUALIZE\s+(\w+)\s+TABLE\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB VISUALIZE komutunda sözdizimi hatası: {command}")
        conn_var, table_name, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        dot = graphviz.Digraph()
        conn = self.interpreter.db_connections[conn_var]
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        dot.node(table_name, f"{table_name}\n" + "\n".join([f"{col[1]}: {col[2]}" for col in columns]))
        cursor.close()
        graph_path = f"{table_name}.gv"
        dot.render(graph_path, format="png", cleanup=True)
        scope = self.interpreter.current_scope()
        scope[result_var] = graph_path + ".png"
        self.interpreter.object_counter["DB VISUALIZE"] += 1
        self.interpreter.object_registry[id(graph_path)] = {
            "type": "DB VISUALIZE",
            "name": result_var,
            "atom": graph_path
        }

    async def handle_db_quantum(self, command: str, scope_name: str) -> None:
        """DB QUANTUM komutunu işler."""
        match = re.match(r"DB\s+QUANTUM\s+(\w+)\s+QUERY\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB QUANTUM komutunda sözdizimi hatası: {command}")
        conn_var, query, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        result = await self.interpreter.database.quantum_query(self.interpreter.db_connections[conn_var], query)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB QUANTUM"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB QUANTUM",
            "name": result_var,
            "atom": str(result)[:100]
        }

    async def handle_db_holo(self, command: str, scope_name: str) -> None:
        """DB HOLO komutunu işler."""
        match = re.match(r"DB\s+HOLO\s+(\w+)\s+DATA\s+(.+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB HOLO komutunda sözdizimi hatası: {command}")
        conn_var, data_expr, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        data = self.interpreter.evaluate_expression(data_expr, scope_name)
        compressed = await self.interpreter.database.holo_compress(self.interpreter.db_connections[conn_var], data)
        scope = self.interpreter.current_scope()
        scope[result_var] = compressed
        self.interpreter.object_counter["DB HOLO"] += 1
        self.interpreter.object_registry[id(compressed)] = {
            "type": "DB HOLO",
            "name": result_var,
            "atom": str(compressed)[:100]
        }

    async def handle_db_smart(self, command: str, scope_name: str) -> None:
        """DB SMART komutunu işler."""
        match = re.match(r"DB\s+SMART\s+(\w+)\s+QUERY\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB SMART komutunda sözdizimi hatası: {command}")
        conn_var, query, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        optimized_query = await self.interpreter.database.optimize_query(query)
        result = await self.interpreter.database.execute_query(self.interpreter.db_connections[conn_var], optimized_query)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB SMART"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB SMART",
            "name": result_var,
            "atom": optimized_query[:100]
        }

    async def handle_db_temporal(self, command: str, scope_name: str) -> None:
        """DB TEMPORAL komutunu işler."""
        match = re.match(r"DB\s+TEMPORAL\s+(\w+)\s+QUERY\s+\"([^\"]+)\"\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB TEMPORAL komutunda sözdizimi hatası: {command}")
        conn_var, query, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        result = await self.interpreter.database.temporal_query(self.interpreter.db_connections[conn_var], query)
        scope = self.interpreter.current_scope()
        scope[result_var] = result
        self.interpreter.object_counter["DB TEMPORAL"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "DB TEMPORAL",
            "name": result_var,
            "atom": str(result)[:100]
        }

    async def handle_db_predict(self, command: str, scope_name: str) -> None:
        """DB PREDICT komutunu işler."""
        match = re.match(r"DB\s+PREDICT\s+(\w+)\s+TABLE\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DB PREDICT komutunda sözdizimi hatası: {command}")
        conn_var, table_name, result_var = match.groups()
        if conn_var not in self.interpreter.db_connections:
            raise PdsXException(f"Veritabanı bağlantısı bulunamadı: {conn_var}")
        predictions = await self.interpreter.database.predict(self.interpreter.db_connections[conn_var], table_name)
        scope = self.interpreter.current_scope()
        scope[result_var] = predictions
        self.interpreter.object_counter["DB PREDICT"] += 1
        self.interpreter.object_registry[id(predictions)] = {
            "type": "DB PREDICT",
            "name": result_var,
            "atom": str(predictions)[:100]
        }
