```python
# command_executor2.py - PDS-X Enhanced Command Executor (command_executor.py için devam)
# Version: 1.0.0
# Date: June 14, 2025
import re
import asyncio
from typing import Dict, Callable, Optional
from pdsx_exception import PdsXException
import json
import numpy as np
import pandas as pd
import aiohttp
import aiofiles
import ctypes
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import paho.mqtt.client as mqtt

class CommandExecutor:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.command_handlers: Dict[str, Callable] = {}
        self.trace_mode = interpreter.trace_mode
        self.backtrace_logger = interpreter.backtrace
        self.program_counter = interpreter.program_counter
        self.register_commands()

    def register_commands(self):
        """Komut handler'larýný kaydeder."""
        self.command_handlers["PRINT"] = self.handle_print
        self.command_handlers["LET"] = self.handle_let
        self.command_handlers["DIM"] = self.handle_dim
        self.command_handlers["INPUT"] = self.handle_input
        self.command_handlers["IF"] = self.handle_if
        self.command_handlers["ELSE"] = self.handle_else
        self.command_handlers["ENDIF"] = self.handle_endif
        self.command_handlers["FOR"] = self.handle_for
        self.command_handlers["NEXT"] = self.handle_next
        self.command_handlers["WHILE"] = self.handle_while
        self.command_handlers["WEND"] = self.handle_wend
        self.command_handlers["DO"] = self.handle_do
        self.command_handlers["LOOP"] = self.handle_loop
        self.command_handlers["SELECT CASE"] = self.handle_select_case
        self.command_handlers["CASE"] = self.handle_case
        self.command_handlers["END SELECT"] = self.handle_end_select
        self.command_handlers["GOTO"] = self.handle_goto
        self.command_handlers["GOSUB"] = self.handle_gosub
        self.command_handlers["RETURN"] = self.handle_return
        self.command_handlers["ON ERROR"] = self.handle_on_error
        self.command_handlers["SUB"] = self.handle_sub
        self.command_handlers["FUNCTION"] = self.handle_function
        self.command_handlers["CALL"] = self.handle_call
        self.command_handlers["END"] = self.handle_end
        self.command_handlers["CLASS"] = self.handle_class
        self.command_handlers["YAPI"] = self.handle_yapi
        self.command_handlers["DATA"] = self.handle_data
        self.command_handlers["READ"] = self.handle_read
        self.command_handlers["RESTORE"] = self.handle_restore
        self.command_handlers["CHAIN"] = self.handle_chain
        self.command_handlers["CONT"] = self.handle_cont
        self.command_handlers["STOP"] = self.handle_stop
        self.command_handlers["TRON"] = self.handle_tron
        self.command_handlers["TROFF"] = self.handle_troff
        self.command_handlers["COMMON"] = self.handle_common
        self.command_handlers["DECLARE"] = self.handle_declare
        self.command_handlers["DEF"] = self.handle_def
        self.command_handlers["EXIT"] = self.handle_exit
        self.command_handlers["UNDIM"] = self.handle_undim
        self.command_handlers["SETFIELD"] = self.handle_setfield
        self.command_handlers["GETFIELD"] = self.handle_getfield
        self.command_handlers["ADDFIELD"] = self.handle_addfield
        self.command_handlers["REMOVEFIELD"] = self.handle_removefield
        self.command_handlers["NEWOBJ"] = self.handle_newobj
        self.command_handlers["COUNTOBJ"] = self.handle_countobj
        self.command_handlers["INSPOBJ"] = self.handle_inspobj
        self.command_handlers["CALLAPI"] = self.handle_callapi
        self.command_handlers["CALLDLL"] = self.handle_calldll
        self.command_handlers["SART"] = self.handle_sart
        self.command_handlers["ALIAS"] = self.handle_alias
        self.command_handlers["RESTRICT"] = self.handle_restrict
        self.command_handlers["CLEAR BASIC"] = self.handle_clear_basic
        self.command_handlers["LISTFILES"] = self.handle_listfiles
        self.command_handlers["LISTPROG"] = self.handle_listprog
        self.command_handlers["CHECKFILE"] = self.handle_checkfile
        self.command_handlers["SCREEN"] = self.handle_screen
        self.command_handlers["SOUND"] = self.handle_sound
        self.command_handlers["SEC VAR"] = self.handle_sec_var
        self.command_handlers["MON VAR"] = self.handle_mon_var
        self.command_handlers["CONVERT"] = self.handle_convert
        self.command_handlers["CAST"] = self.handle_cast
        self.command_handlers["ALTER TABLE"] = self.handle_alter_table
        self.command_handlers["CREATE VIEW"] = self.handle_create_view
        self.command_handlers["ENCRYPT"] = self.handle_encrypt
        self.command_handlers["DECRYPT"] = self.handle_decrypt
        self.command_handlers["SIGN"] = self.handle_sign
        self.command_handlers["VERIFY"] = self.handle_verify
        self.command_handlers["SECURE_VAR"] = self.handle_secure_var
        self.command_handlers["CONNECT_IOT"] = self.handle_connect_iot
        self.command_handlers["PUBLISH_IOT"] = self.handle_publish_iot
        self.command_handlers["SUBSCRIBE_IOT"] = self.handle_subscribe_iot
        self.command_handlers["BUS DEFINE"] = self.handle_bus_define
        self.command_handlers["BUS PUBLISH"] = self.handle_bus_publish
        self.command_handlers["BUS SUBSCRIBE"] = self.handle_bus_subscribe
        self.command_handlers["BUS UNSUBSCRIBE"] = self.handle_bus_unsubscribe
        self.command_handlers["BUS START"] = self.handle_bus_start
        self.command_handlers["BUS STOP"] = self.handle_bus_stop
        self.command_handlers["BUS PAUSE"] = self.handle_bus_pause
        self.command_handlers["BUS RESUME"] = self.handle_bus_resume
        self.command_handlers["BUS MONITOR"] = self.handle_bus_monitor
        self.command_handlers["SECURE BUS"] = self.handle_secure_bus
        self.command_handlers["UNSECURE BUS"] = self.handle_unsecure_bus
        self.command_handlers["BUS EVENT TRIGGER"] = self.handle_bus_event_trigger
        self.command_handlers["BUS RECURSE"] = self.handle_bus_recurse
        self.command_handlers["BUS PRIORITIZE"] = self.handle_bus_prioritize
        self.command_handlers["BUS LOCK"] = self.handle_bus_lock
        self.command_handlers["BUS UNLOCK"] = self.handle_bus_unlock
        self.command_handlers["BUS DATA SEND"] = self.handle_bus_data_send
        self.command_handlers["BUS DATA RECEIVE"] = self.handle_bus_data_receive
        self.command_handlers["BUS IOT CONNECT"] = self.handle_bus_iot_connect
        self.command_handlers["BUS CLOUD SYNC"] = self.handle_bus_cloud_sync
        self.command_handlers["BUS NETWORK CONNECT"] = self.handle_bus_network_connect
        self.command_handlers["BUS EXPERIMENT"] = self.handle_bus_experiment
        self.command_handlers["BUS PARADIGM SET"] = self.handle_bus_paradigm_set
        self.command_handlers["BUS DATA STRUCTURE USE"] = self.handle_bus_data_structure_use
        self.command_handlers["BUS FLOW CONTROL"] = self.handle_bus_flow_control
        self.command_handlers["BUS RESOURCE ALLOCATE"] = self.handle_bus_resource_allocate
        self.command_handlers["BUS RESOURCE FREE"] = self.handle_bus_resource_free
        self.command_handlers["BUS TIMEOUT SET"] = self.handle_bus_timeout_set
        self.command_handlers["BUS RETRY"] = self.handle_bus_retry
        self.command_handlers["BUS ALERT"] = self.handle_bus_alert
        self.command_handlers["BUS QUANTUM SEND"] = self.handle_bus_quantum_send
        self.command_handlers["COMPILE C"] = self.handle_compile_c
        self.command_handlers["SQLITE CONNECT"] = self.handle_sqlite_connect
        self.command_handlers["EVENT REGISTER"] = self.handle_event_register
        self.command_handlers["THREAD"] = self.handle_thread
        self.command_handlers["OPEN DATABASE"] = self.handle_open_database
        self.command_handlers["TIMER SET"] = self.handle_timer_set
        self.command_handlers["PIPE"] = self.handle_pipe
        self.command_handlers["LOG TRACE"] = self.handle_log_trace
        self.command_handlers["GRAPH CREATE"] = self.handle_graph_create
        self.command_handlers["EXPORT DATA"] = self.handle_export_data
        self.command_handlers["FACT"] = self.handle_fact
        self.command_handlers["RULE"] = self.handle_rule
        self.command_handlers["QUERY"] = self.handle_query
        self.command_handlers["WINDOW"] = self.handle_window
        self.command_handlers["SIMD ADD"] = self.handle_simd_add
        self.command_handlers["NET GET"] = self.handle_net_get
        self.command_handlers["NET POST"] = self.handle_net_post
        self.command_handlers["AI GENERATE"] = self.handle_ai_generate
        self.command_handlers["BITSET"] = self.handle_bitset
        self.command_handlers["NLP ANALYZE"] = self.handle_nlp_analyze
        self.command_handlers["VALIDATE MODULE"] = self.handle_validate_module
        self.command_handlers["TYPE"] = self.handle_type
        self.command_handlers["LOAD MODULE"] = self.handle_load_module
        self.command_handlers["UNLOAD MODULE"] = self.handle_unload_module
        self.command_handlers["LIST MODULES"] = self.handle_list_modules
        self.command_handlers["SAVE MODULE"] = self.handle_save_module
        self.command_handlers["EDIT MODULE"] = self.handle_edit_module
        self.command_handlers["THREAD CREATE"] = self.handle_thread_create
        self.command_handlers["DB CONNECT"] = self.handle_db_connect
        self.command_handlers["DB QUERY"] = self.handle_db_query
        self.command_handlers["DB ASYNC QUERY"] = self.handle_db_async_query
        self.command_handlers["DB CREATE TABLE"] = self.handle_db_create_table
        self.command_handlers["DB ISAM CREATE"] = self.handle_db_isam_create
        self.command_handlers["DB ISAM INSERT"] = self.handle_db_isam_insert
        self.command_handlers["DB ISAM SEARCH"] = self.handle_db_isam_search
        self.command_handlers["DB ANALYZE"] = self.handle_db_analyze
        self.command_handlers["DB VISUALIZE"] = self.handle_db_visualize
        self.command_handlers["DB QUANTUM"] = self.handle_db_quantum
        self.command_handlers["DB HOLO"] = self.handle_db_holo
        self.command_handlers["DB SMART"] = self.handle_db_smart
        self.command_handlers["DB TEMPORAL"] = self.handle_db_temporal
        self.command_handlers["DB PREDICT"] = self.handle_db_predict

    async def execute_async(self, command: str, scope_name: str = None) -> Optional[int]:
        """Komutu asenkron çalýþtýrýr."""
        result = self.execute(command, scope_name)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def execute(self, command: str, scope_name: str = None) -> Optional[int]:
        """Komut yorumlayýcý"""
        if isinstance(command, dict):
            return None

        command = command.strip()
        if not command:
            return None
        command_upper = command.upper()

        if self.trace_mode:
            self.backtrace_logger.log(f"TRACE: Satýr {self.program_counter + 1}: {command}")

        try:
            for cmd, handler in self.command_handlers.items():
                if command_upper.startswith(cmd):
                    result = handler(command, scope_name)
                    if asyncio.iscoroutine(result):
                        return asyncio.run(result)
                    return result

            if "=" in command and not command_upper.startswith(("IF", "FOR", "SELECT")):
                var_name, expr = command.split("=", 1)
                var_name = var_name.strip()
                value = self.interpreter.evaluate_expression(expr.strip(), scope_name)
                if scope_name and scope_name in self.interpreter.modules:
                    self.interpreter.modules[scope_name]["variables"][var_name] = value
                else:
                    self.interpreter.current_scope()[var_name] = value
                self.interpreter.object_counter["VARIABLE"] += 1
                self.interpreter.object_registry[id(value)] = {
                    "type": "VARIABLE",
                    "name": var_name,
                    "atom": str(value)[:100]
                }
                return None

            if command_upper.startswith("LIBX."):
                parts = command.split(".")
                if len(parts) >= 3:
                    module_name = parts[1].lower()
                    func_name = parts[2].split("(")[0].upper()
                    if module_name in self.interpreter.libx_modules:
                        module = self.interpreter.libx_modules[module_name]
                        if hasattr(module, func_name):
                            args_match = re.search(r"\((.*?)\)", command)
                            args = []
                            if args_match:
                                args = [self.interpreter.evaluate_expression(a.strip(), scope_name)
                                        for a in args_match.group(1).split(",")]
                            result = getattr(module, func_name)(*args)
                            self.interpreter.object_counter["LIBX_CALL"] += 1
                            self.interpreter.object_registry[id(result)] = {
                                "type": "LIBX_CALL",
                                "name": f"{module_name}.{func_name}",
                                "atom": str(result)[:100]
                            }
                            return result
                    raise PdsXException(f"LIBX modülü veya fonksiyonu bulunamadý: {module_name}.{func_name}")

            if command_upper.startswith("BYTECODE."):
                parts = command[9:].split(".")
                if parts:
                    op = parts[0].upper()
                    if op in self.interpreter.bytecode_opcodes:
                        args_match = re.search(r"\((.*?)\)", command)
                        args = []
                        if args_match:
                            args = [self.interpreter.evaluate_expression(a.strip(), scope_name)
                                    for a in args_match.group(1).split(",")]
                        result = self.interpreter.bytecode_opcodes[op](*args)
                        self.interpreter.object_counter["BYTECODE_OP"] += 1
                        self.interpreter.object_registry[id(result)] = {
                            "type": "BYTECODE_OP",
                            "name": op,
                            "atom": str(result)[:100]
                        }
                        return result
                    raise PdsXException(f"Bilinmeyen bytecode operasyonu: {op}")

            if command_upper.startswith(("SIMD.", "NEURAL.", "QUANTUM.", "GENETIC.", "BLOCKCHAIN.")):
                feature_type = command.split(".")[0].upper()
                if feature_type in self.interpreter.bytecode_opcodes:
                    op = command.split(".")[1].split("(")[0].upper()
                    args_match = re.search(r"\((.*?)\)", command)
                    args = []
                    if args_match:
                        args = [self.interpreter.evaluate_expression(a.strip(), scope_name)
                                for a in args_match.group(1).split(",")]
                    if op in self.interpreter.bytecode_opcodes[feature_type]:
                        result = self.interpreter.bytecode_opcodes[feature_type][op](*args)
                        self.interpreter.object_counter[f"{feature_type}_OP"] += 1
                        self.interpreter.object_registry[id(result)] = {
                            "type": f"{feature_type}_OP",
                            "name": op,
                            "atom": str(result)[:100]
                        }
                        return result
                    raise PdsXException(f"Bilinmeyen {feature_type} operasyonu: {op}")

            raise PdsXException(f"Bilinmeyen komut: {command}")

        except PdsXException as e:
            if self.interpreter.error_handler is not None:
                self.program_counter = self.interpreter.error_handler
                return None
            raise

    def handle_print(self, command: str, scope_name: str) -> Optional[int]:
        """PRINT komutunu iþler."""
        match = re.match(r"PRINT\s*(.+)?", command, re.IGNORECASE)
        if match:
            expr = match.group(1)
            if expr:
                result = self.interpreter.evaluate_expression(expr, scope_name)
                print(result)
            else:
                print()
            self.interpreter.object_counter["PRINT"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "PRINT",
                "name": "PRINT",
                "atom": command
            }
        return None

    def handle_let(self, command: str, scope_name: str) -> Optional[int]:
        """LET komutunu iþler."""
        match = re.match(r"(?:LET\s+)?((?:\w+\s*(?:,\s*\w+)*))\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"LET/Atama komutunda sözdizimi hatasý: {command}")
        var_names, expr = match.groups()
        value = self.interpreter.evaluate_expression(expr, scope_name)
        var_names = [v.strip() for v in var_names.split(",")]
        scope = self.interpreter.modules[scope_name]["variables"] if scope_name and scope_name in self.interpreter.modules else self.interpreter.current_scope()
        for var in var_names:
            scope[var] = value
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_registry[id(value)] = {
                "type": "VARIABLE",
                "name": var,
                "atom": str(value)[:100]
            }
        self.interpreter.object_counter["LET"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "LET",
            "name": ", ".join(var_names),
            "atom": command
        }
        return None

    def handle_dim(self, command: str, scope_name: str) -> Optional[int]:
        """DIM komutunu iþler."""
        match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)(?:\s*\[\s*(.+)\s*\])?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DIM komutunda sözdizimi hatasý: {command}")
        var_name, var_type, dims = match.groups()
        if var_type.upper() not in self.interpreter.type_table:
            raise PdsXException(f"Geçersiz veri tipi: {var_type}")
        scope = self.interpreter.modules[scope_name]["variables"] if scope_name and scope_name in self.interpreter.modules else self.interpreter.current_scope()
        if dims:
            dims = [self.interpreter.evaluate_expression(d.strip(), scope_name) for d in dims.split(",")]
            value = np.zeros(dims) if var_type.upper() in ["SINGLE", "DOUBLE", "ARRAY"] else [None] * dims[0]
        else:
            value = self.interpreter.type_table[var_type.upper()]()
        scope[var_name] = value
        self.interpreter.object_counter["VARIABLE"] += 1
        self.interpreter.object_registry[id(value)] = {
            "type": "VARIABLE",
            "name": var_name,
            "atom": str(value)[:100]
        }
        self.interpreter.object_counter["DIM"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "DIM",
            "name": var_name,
            "atom": command
        }
        return None

    def handle_input(self, command: str, scope_name: str) -> Optional[int]:
        """INPUT komutunu iþler."""
        match = re.match(r"INPUT\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"INPUT komutunda sözdizimi hatasý: {command}")
        var_name = match.group(1)
        value = input()
        scope = self.interpreter.modules[scope_name]["variables"] if scope_name and scope_name in self.interpreter.modules else self.interpreter.current_scope()
        scope[var_name] = value
        self.interpreter.object_counter["VARIABLE"] += 1
        self.interpreter.object_registry[id(value)] = {
            "type": "VARIABLE",
            "name": var_name,
            "atom": str(value)[:100]
        }
        self.interpreter.object_counter["INPUT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "INPUT",
            "name": var_name,
            "atom": command
        }
        return None

    def handle_if(self, command: str, scope_name: str) -> Optional[int]:
        """IF komutunu iþler."""
        match = re.match(r"IF\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+?))?(?:\s+END\s+IF)?$", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXException(f"IF komutunda sözdizimi hatasý: {command}")
        condition, then_block, else_block = match.groups()
        result = self.interpreter.evaluate_expression(condition, scope_name)
        self.interpreter.if_stack.append(bool(result))
        if result:
            self.execute(then_block.strip(), scope_name)
        elif else_block:
            self.execute(else_block.strip(), scope_name)
        self.interpreter.object_counter["IF"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "IF",
            "name": "IF",
            "atom": command
        }
        return None

    def handle_else(self, command: str, scope_name: str) -> Optional[int]:
        """ELSE komutunu iþler."""
        if self.interpreter.if_stack and self.interpreter.if_stack[-1]:
            while self.program_counter < len(self.interpreter.program):
                next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
                if next_cmd == "ENDIF":
                    break
                self.program_counter += 1
        self.interpreter.object_counter["ELSE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "ELSE",
            "name": "ELSE",
            "atom": command
        }
        return None

    def handle_endif(self, command: str, scope_name: str) -> Optional[int]:
        """ENDIF komutunu iþler."""
        if self.interpreter.if_stack:
            self.interpreter.if_stack.pop()
        self.interpreter.object_counter["ENDIF"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "ENDIF",
            "name": "ENDIF",
            "atom": command
        }
        return None

    def handle_for(self, command: str, scope_name: str) -> Optional[int]:
        """FOR komutunu iþler."""
        match = re.match(r"FOR\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"FOR komutunda sözdizimi hatasý: {command}")
        var_name, start, end, step = match.groups()
        start_val = self.interpreter.evaluate_expression(start, scope_name)
        end_val = self.interpreter.evaluate_expression(end, scope_name)
        step_val = self.interpreter.evaluate_expression(step, scope_name) if step else 1
        scope = self.interpreter.modules[scope_name]["variables"] if scope_name and scope_name in self.interpreter.modules else self.interpreter.current_scope()
        scope[var_name] = start_val
        self.interpreter.loop_stack.append({
            "var": var_name,
            "current": start_val,
            "end": end_val,
            "step": step_val,
            "start_pc": self.program_counter,
            "scope_name": scope_name
        })
        self.interpreter.object_counter["FOR"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "FOR",
            "name": var_name,
            "atom": command
        }
        return None

    def handle_next(self, command: str, scope_name: str) -> Optional[int]:
        """NEXT komutunu iþler."""
        if self.interpreter.loop_stack:
            loop_info = self.interpreter.loop_stack[-1]
            var_name = loop_info["var"]
            current = loop_info["current"] + loop_info["step"]
            scope_name = loop_info["scope_name"]
            scope = self.interpreter.modules[scope_name]["variables"] if scope_name and scope_name in self.interpreter.modules else self.interpreter.current_scope()
            if (loop_info["step"] > 0 and current <= loop_info["end"]) or (loop_info["step"] < 0 and current >= loop_info["end"]):
                loop_info["current"] = current
                scope[var_name] = current
                self.program_counter = loop_info["start_pc"]
            else:
                self.interpreter.loop_stack.pop()
        self.interpreter.object_counter["NEXT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "NEXT",
            "name": "NEXT",
            "atom": command
        }
        return None

    def handle_while(self, command: str, scope_name: str) -> Optional[int]:
        """WHILE komutunu iþler."""
        match = re.match(r"WHILE\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"WHILE komutunda sözdizimi hatasý: {command}")
        condition = match.group(1)
        result = self.interpreter.evaluate_expression(condition, scope_name)
        if result:
            self.interpreter.loop_stack.append({
                "type": "while",
                "condition": condition,
                "start_pc": self.program_counter,
                "scope_name": scope_name
            })
        else:
            nested = 1
            while self.program_counter < len(self.interpreter.program):
                next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
                if next_cmd.startswith("WHILE"):
                    nested += 1
                elif next_cmd == "WEND":
                    nested -= 1
                    if nested == 0:
                        break
                self.program_counter += 1
        self.interpreter.object_counter["WHILE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "WHILE",
            "name": "WHILE",
            "atom": command
        }
        return None

    def handle_wend(self, command: str, scope_name: str) -> Optional[int]:
        """WEND komutunu iþler."""
        if self.interpreter.loop_stack and self.interpreter.loop_stack[-1]["type"] == "while":
            loop_info = self.interpreter.loop_stack[-1]
            if self.interpreter.evaluate_expression(loop_info["condition"], loop_info["scope_name"]):
                self.program_counter = loop_info["start_pc"]
            else:
                self.interpreter.loop_stack.pop()
        self.interpreter.object_counter["WEND"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "WEND",
            "name": "WEND",
            "atom": command
        }
        return None

    def handle_do(self, command: str, scope_name: str) -> Optional[int]:
        """DO komutunu iþler."""
        self.interpreter.loop_stack.append({
            "type": "do",
            "start_pc": self.program_counter,
            "scope_name": scope_name
        })
        self.interpreter.object_counter["DO"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "DO",
            "name": "DO",
            "atom": command
        }
        return None

    def handle_loop(self, command: str, scope_name: str) -> Optional[int]:
        """LOOP komutunu iþler."""
        match = re.match(r"LOOP\s+(?:UNTIL|WHILE)\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"LOOP komutunda sözdizimi hatasý: {command}")
        condition = match.group(1)
        if self.interpreter.loop_stack and self.interpreter.loop_stack[-1]["type"] == "do":
            is_until = "UNTIL" in command_upper
            result = self.interpreter.evaluate_expression(condition, scope_name)
            if (is_until and not result) or (not is_until and result):
                self.program_counter = self.interpreter.loop_stack[-1]["start_pc"]
            else:
                self.interpreter.loop_stack.pop()
        self.interpreter.object_counter["LOOP"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "LOOP",
            "name": "LOOP",
            "atom": command
        }
        return None

    def handle_select_case(self, command: str, scope_name: str) -> Optional[int]:
        """SELECT CASE komutunu iþler."""
        match = re.match(r"SELECT CASE\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"SELECT CASE komutunda sözdizimi hatasý: {command}")
        expr = match.group(1)
        value = self.interpreter.evaluate_expression(expr, scope_name)
        self.interpreter.select_stack.append({
            "value": value,
            "matched": False,
            "scope_name": scope_name
        })
        self.interpreter.object_counter["SELECT CASE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "SELECT CASE",
            "name": "SELECT CASE",
            "atom": command
        }
        return None

    def handle_case(self, command: str, scope_name: str) -> Optional[int]:
        """CASE komutunu iþler."""
        if self.interpreter.select_stack:
            select_info = self.interpreter.select_stack[-1]
            if not select_info["matched"]:
                match = re.match(r"CASE\s+(.+)", command, re.IGNORECASE)
                if not match:
                    raise PdsXException(f"CASE komutunda sözdizimi hatasý: {command}")
                case_expr = match.group(1)
                if case_expr.upper() == "ELSE":
                    select_info["matched"] = True
                else:
                    case_value = self.interpreter.evaluate_expression(case_expr, scope_name)
                    if case_value == select_info["value"]:
                        select_info["matched"] = True
                    else:
                        while self.program_counter < len(self.interpreter.program):
                            next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
                            if next_cmd.startswith("CASE") or next_cmd == "END SELECT":
                                break
                            self.program_counter += 1
        self.interpreter.object_counter["CASE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "CASE",
            "name": "CASE",
            "atom": command
        }
        return None

    def handle_end_select(self, command: str, scope_name: str) -> Optional[int]:
        """END SELECT komutunu iþler."""
        if self.interpreter.select_stack:
            self.interpreter.select_stack.pop()
        self.interpreter.object_counter["END SELECT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "END SELECT",
            "name": "END SELECT",
            "atom": command
        }
        return None

    def handle_goto(self, command: str, scope_name: str) -> Optional[int]:
        """GOTO komutunu iþler."""
        match = re.match(r"GOTO\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"GOTO komutunda sözdizimi hatasý: {command}")
        label = match.group(1)
        if label in self.interpreter.labels:
            self.program_counter = self.interpreter.labels[label]
            self.interpreter.object_counter["GOTO"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "GOTO",
                "name": label,
                "atom": command
            }
            return self.program_counter
        raise PdsXException(f"Etiket bulunamadý: {label}")

    def handle_gosub(self, command: str, scope_name: str) -> Optional[int]:
        """GOSUB komutunu iþler."""
        match = re.match(r"GOSUB\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"GOSUB komutunda sözdizimi hatasý: {command}")
        label = match.group(1)
        if label in self.interpreter.labels:
            self.interpreter.call_stack.append(self.program_counter)
            self.program_counter = self.interpreter.labels[label]
            self.interpreter.object_counter["GOSUB"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "GOSUB",
                "name": label,
                "atom": command
            }
            return self.program_counter
        raise PdsXException(f"Alt program etiketi bulunamadý: {label}")

    def handle_return(self, command: str, scope_name: str) -> Optional[int]:
        """RETURN komutunu iþler."""
        if self.interpreter.call_stack:
            self.program_counter = self.interpreter.call_stack.pop()
            self.interpreter.object_counter["RETURN"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "RETURN",
                "name": "RETURN",
                "atom": command
            }
            return self.program_counter
        raise PdsXException("Dönecek çaðrý yýðýný yok")

    def handle_on_error(self, command: str, scope_name: str) -> Optional[int]:
        """ON ERROR komutunu iþler."""
        if command_upper == "ON ERROR RESUME NEXT":
            self.interpreter.error_handler = None
        else:
            match = re.match(r"ON ERROR GOTO\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXException(f"ON ERROR komutunda sözdizimi hatasý: {command}")
            label = match.group(1)
            if label in self.interpreter.labels:
                self.interpreter.error_handler = self.interpreter.labels[label]
                self.interpreter.object_counter["ON ERROR"] += 1
                self.interpreter.object_registry[id(command)] = {
                    "type": "ON ERROR",
                    "name": label,
                    "atom": command
                }
            else:
                raise PdsXException(f"Hata iþleyici etiketi bulunamadý: {label}")
        return None

    def handle_sub(self, command: str, scope_name: str) -> Optional[int]:
        """SUB komutunu iþler."""
        match = re.match(r"SUB\s+(\w+)(?:\((.*?)\))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"SUB komutunda sözdizimi hatasý: {command}")
        name, params = match.groups()
        params = [p.strip() for p in params.split(",")] if params else []
        self.interpreter.subs[name] = {
            "params": params,
            "body": [],
            "start_pc": self.program_counter
        }
        while self.program_counter < len(self.interpreter.program):
            next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
            if next_cmd == "END SUB":
                break
            self.program_counter += 1
        self.interpreter.object_counter["SUB"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "SUB",
            "name": name,
            "atom": command
        }
        return None

    def handle_function(self, command: str, scope_name: str) -> Optional[int]:
        """FUNCTION komutunu iþler."""
        match = re.match(r"FUNCTION\s+(\w+)(?:\((.*?)\))?\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"FUNCTION komutunda sözdizimi hatasý: {command}")
        name, params, return_type = match.groups()
        if return_type.upper() not in self.interpreter.type_table:
            raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
        params = [p.strip() for p in params.split(",")] if params else []
        self.interpreter.functions[name] = {
            "params": params,
            "return_type": return_type,
            "body": [],
            "start_pc": self.program_counter
        }
        while self.program_counter < len(self.interpreter.program):
            next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
            if next_cmd == "END FUNCTION":
                break
            self.program_counter += 1
        self.interpreter.object_counter["FUNCTION"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "FUNCTION",
            "name": name,
            "atom": command
        }
        return None

    def handle_call(self, command: str, scope_name: str) -> Optional[int]:
        """CALL komutunu iþler."""
        match = re.match(r"CALL\s+(\w+)(?:\((.*?)\))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"CALL komutunda sözdizimi hatasý: {command}")
        name, args = match.groups()
        args = [self.interpreter.evaluate_expression(a.strip(), scope_name) for a in args.split(",") if a.strip()]
        if name in self.interpreter.subs:
            sub_info = self.interpreter.subs[name]
            if len(args) != len(sub_info["params"]):
                raise PdsXException(f"Geçersiz parametre sayýsý: {name}")
            new_scope = dict(zip(sub_info["params"], args))
            self.interpreter.local_scopes.append(new_scope)
            self.interpreter.call_stack.append(self.program_counter)
            self.program_counter = sub_info["start_pc"]
            self.interpreter.object_counter["CALL"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "CALL",
                "name": name,
                "atom": command
            }
            return self.program_counter
        raise PdsXException(f"Alt program bulunamadý: {name}")

    async def handle_end(self, command: str, scope_name: str) -> None:
        """END komutunu iþler."""
        match = re.match(r"END(?:\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"END komutunda sözdizimi hatasý: {command}")
        block_type = match.group(1).upper() if match.group(1) else None
        if block_type:
            valid_types = list(self.interpreter.type_table.keys()) + ["FOR", "WHILE", "IF", "CLASS", "YAPI", "SELECT", "SUB", "FUNCTION"]
            if block_type not in valid_types:
                raise PdsXException(f"Geçersiz END tipi: {block_type}")
            if block_type in ["FOR", "WHILE"]:
                if not self.interpreter.loop_stack or self.interpreter.loop_stack[-1]["type"] != block_type.lower():
                    raise PdsXException(f"Kapatýlacak {block_type} bloðu yok")
                self.interpreter.loop_stack.pop()
            elif block_type == "IF":
                if self.interpreter.if_stack:
                    self.interpreter.if_stack.pop()
            elif block_type == "SELECT":
                if self.interpreter.select_stack:
                    self.interpreter.select_stack.pop()
        else:
            self.interpreter.running = False
        self.interpreter.object_counter["END"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "END",
            "name": block_type or "PROGRAM",
            "atom": command
        }

    async def handle_class(self, command: str, scope_name: str) -> None:
        """CLASS komutunu iþler."""
        match = re.match(r"CLASS\s+(\w+)\s+(.+?)\s+END\s+CLASS", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXException(f"CLASS komutunda sözdizimi hatasý: {command}")
        class_name, body = match.groups()
        class_def = {"methods": {}, "properties": {}, "subs": {}, "functions": {}}
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("SUB "):
                sub_match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)\s*(.+?)\s+END\s+SUB", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if sub_match:
                    sub_name, params, sub_body = sub_match.groups()
                    class_def["subs"][sub_name] = {"params": params, "body": sub_body}
                    self.interpreter.object_counter["SUB"] += 1
                    self.interpreter.object_registry[id(sub_body)] = {
                        "type": "SUB",
                        "name": sub_name,
                        "atom": sub_body[:100]
                    }
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXException(f"SUB tanýmý hatalý: {line}")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
                    class_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body[:100]
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXException(f"FUNCTION tanýmý hatalý: {line}")
            elif line.upper().startswith("PROP "):
                prop_match = re.match(r"PROP\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if prop_match:
                    prop_name, prop_type = prop_match.groups()
                    if prop_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz özellik tipi: {prop_type}")
                    class_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type.upper()] += 1
                    i += 1
                else:
                    raise PdsXException(f"PROP tanýmý hatalý: {line}")
            else:
                i += 1
        self.interpreter.classes[class_name] = class_def
        self.interpreter.object_counter["CLASS"] += 1
        self.interpreter.object_registry[id(class_def)] = {
            "type": "CLASS",
            "name": class_name,
            "atom": class_name
        }

    async def handle_yapi(self, command: str, scope_name: str) -> None:
        """YAPI komutunu iþler."""
        match = re.match(r"YAPI\s+(\w+)\s+(.+?)\s+END\s+YAPI", command, re.IGNORECASE | re.DOTALL)
        if not match:
            raise PdsXException(f"YAPI komutunda sözdizimi hatasý: {command}")
        yapi_name, body = match.groups()
        yapi_def = {"methods": {}, "properties": {}, "subs": {}, "functions": {}, "gamma": {}, "omega": {}}
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith("SUB "):
                sub_match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)\s*(.+?)\s+END\s+SUB", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if sub_match:
                    sub_name, params, sub_body = sub_match.groups()
                    yapi_def["subs"][sub_name] = {"params": params, "body": sub_body}
                    self.interpreter.object_counter["SUB"] += 1
                    self.interpreter.object_registry[id(sub_body)] = {
                        "type": "SUB",
                        "name": sub_name,
                        "atom": sub_body[:100]
                    }
                    i += sub_body.count("\n") + 2
                else:
                    raise PdsXException(f"SUB tanýmý hatalý: {line}")
            elif line.upper().startswith("FUNCTION "):
                func_match = re.match(r"FUNCTION\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNCTION", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body[:100]
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXException(f"FUNCTION tanýmý hatalý: {line}")
            elif line.upper().startswith("FUNC "):
                func_match = re.match(r"FUNC\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+FUNC", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if func_match:
                    func_name, params, return_type, func_body = func_match.groups()
                    if return_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
                    yapi_def["functions"][func_name] = {"params": params, "return_type": return_type, "body": func_body}
                    self.interpreter.object_counter["FUNCTION"] += 1
                    self.interpreter.object_registry[id(func_body)] = {
                        "type": "FUNCTION",
                        "name": func_name,
                        "atom": func_body[:100]
                    }
                    i += func_body.count("\n") + 2
                else:
                    raise PdsXException(f"FUNC tanýmý hatalý: {line}")
            elif line.upper().startswith("GAMMA "):
                gamma_match = re.match(r"GAMMA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+GAMMA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if gamma_match:
                    gamma_name, params, return_type, gamma_body = gamma_match.groups()
                    if return_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
                    yapi_def["gamma"][gamma_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": gamma_body,
                        "partial": lambda *args: lambda *rest: self.interpreter.evaluate_expression(f"{gamma_body}({','.join(map(str, args + rest))})")
                    }
                    self.interpreter.object_counter["GAMMA"] += 1
                    self.interpreter.object_registry[id(gamma_body)] = {
                        "type": "GAMMA",
                        "name": gamma_name,
                        "atom": gamma_body[:100]
                    }
                    i += gamma_body.count("\n") + 2
                else:
                    raise PdsXException(f"GAMMA tanýmý hatalý: {line}")
            elif line.upper().startswith("OMEGA "):
                omega_match = re.match(r"OMEGA\s+(\w+)\s*\((.*?)\)\s+AS\s+(\w+)\s*(.+?)\s+END\s+OMEGA", "\n".join(lines[i:]), re.IGNORECASE | re.DOTALL)
                if omega_match:
                    omega_name, params, return_type, omega_body = omega_match.groups()
                    if return_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
                    yapi_def["omega"][omega_name] = {
                        "params": params,
                        "return_type": return_type,
                        "body": omega_body,
                        "self_apply": lambda x: self.interpreter.evaluate_expression(f"{omega_body}({x})")
                    }
                    self.interpreter.object_counter["OMEGA"] += 1
                    self.interpreter.object_registry[id(omega_body)] = {
                        "type": "OMEGA",
                        "name": omega_name,
                        "atom": omega_body[:100]
                    }
                    i += omega_body.count("\n") + 2
                else:
                    raise PdsXException(f"OMEGA tanýmý hatalý: {line}")
            elif line.upper().startswith("DIM "):
                dim_match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)", line, re.IGNORECASE)
                if dim_match:
                    prop_name, prop_type = dim_match.groups()
                    if prop_type.upper() not in self.interpreter.type_table:
                        raise PdsXException(f"Geçersiz özellik tipi: {prop_type}")
                    yapi_def["properties"][prop_name] = prop_type
                    self.interpreter.object_counter[prop_type.upper()] += 1
                    i += 1
                else:
                    raise PdsXException(f"DIM tanýmý hatalý: {line}")
            else:
                i += 1
        self.interpreter.classes[yapi_name] = yapi_def
        self.interpreter.object_counter["YAPI"] += 1
        self.interpreter.object_registry[id(yapi_def)] = {
            "type": "YAPI",
            "name": yapi_name,
            "atom": yapi_name
        }

    async def handle_data(self, command: str, scope_name: str) -> None:
        """DATA komutunu iþler."""
        match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DATA komutunda sözdizimi hatasý: {command}")
        values = [v.strip() for v in match.group(1).split(",")]
        self.interpreter.data_list.extend(values)
        self.interpreter.object_counter["DATA"] += len(values)
        for v in values:
            self.interpreter.object_registry[id(v)] = {
                "type": "DATA",
                "name": "DATA",
                "atom": v[:100]
            }

    async def handle_read(self, command: str, scope_name: str) -> None:
        """READ komutunu iþler."""
        match = re.match(r"READ\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"READ komutunda sözdizimi hatasý: {command}")
        var_names = [v.strip() for v in match.group(1).split(",")]
        scope = self.interpreter.current_scope()
        for var in var_names:
            if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                raise PdsXException("Veri listesi sonu")
            value = self.interpreter.data_list[self.interpreter.data_pointer]
            scope[var] = value
            self.interpreter.data_pointer += 1
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_registry[id(value)] = {
                "type": "VARIABLE",
                "name": var,
                "atom": value[:100]
            }
        self.interpreter.object_counter["READ"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "READ",
            "name": ", ".join(var_names),
            "atom": command
        }

    async def handle_restore(self, command: str, scope_name: str) -> None:
        """RESTORE komutunu iþler."""
        self.interpreter.data_pointer = 0
        self.interpreter.object_counter["RESTORE"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "RESTORE",
            "name": "RESTORE",
            "atom": command
        }

    async def handle_chain(self, command: str, scope_name: str) -> None:
        """CHAIN komutunu iþler."""
        match = re.match(r"CHAIN\s+\"([^\"]+)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"CHAIN komutunda sözdizimi hatasý: {command}")
        program_file = match.group(1)
        async with aiofiles.open(program_file, "r", encoding="utf-8") as f:
            program_text = await f.read()
        self.interpreter.parse_program(program_text)
        self.interpreter.object_counter["CHAIN"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "CHAIN",
            "name": program_file,
            "atom": command
        }

    async def handle_cont(self, command: str, scope_name: str) -> None:
        """CONT komutunu iþler."""
        self.interpreter.paused = False
        self.interpreter.object_counter["CONT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "CONT",
            "name": "CONT",
            "atom": command
        }

    async def handle_stop(self, command: str, scope_name: str) -> None:
        """STOP komutunu iþler."""
        self.interpreter.paused = True
        self.interpreter.object_counter["STOP"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "STOP",
            "name": "STOP",
            "atom": command
        }

    async def handle_tron(self, command: str, scope_name: str) -> None:
        """TRON komutunu iþler."""
        self.interpreter.trace_mode = True
        self.trace_mode = True
        self.interpreter.object_counter["TRON"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "TRON",
            "name": "TRON",
            "atom": command
        }

    async def handle_troff(self, command: str, scope_name: str) -> None:
        """TROFF komutunu iþler."""
        self.interpreter.trace_mode = False
        self.trace_mode = False
        self.interpreter.object_counter["TROFF"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "TROFF",
            "name": "TROFF",
            "atom": command
        }

    async def handle_common(self, command: str, scope_name: str) -> None:
        """COMMON komutunu iþler."""
        match = re.match(r"COMMON\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"COMMON komutunda sözdizimi hatasý: {command}")
        var_names = [v.strip() for v in match.group(1).split(",")]
        scope = self.interpreter.current_scope()
        for var in var_names:
            if var in scope:
                self.interpreter.shared_vars[var].append(scope[var])
                self.interpreter.object_counter["COMMON"] += 1
                self.interpreter.object_registry[id(var)] = {
                    "type": "COMMON",
                    "name": var,
                    "atom": var
                }

    async def handle_declare(self, command: str, scope_name: str) -> None:
        """DECLARE komutunu iþler."""
        match = re.match(r"DECLARE\s+(SUB|FUNCTION)\s+(\w+)(?:\((.*?)\))?\s*(?:AS\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DECLARE komutunda sözdizimi hatasý: {command}")
        decl_type, name, params, return_type = match.groups()
        params = [p.strip() for p in params.split(",")] if params else []
        if decl_type.upper() == "FUNCTION" and return_type and return_type.upper() not in self.interpreter.type_table:
            raise PdsXException(f"Geçersiz dönüþ tipi: {return_type}")
        if decl_type.upper() == "SUB":
            self.interpreter.subs[name] = {"params": params, "body": [], "declared": True}
        else:
            self.interpreter.functions[name] = {"params": params, "return_type": return_type, "body": [], "declared": True}
        self.interpreter.object_counter[decl_type.upper()] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": decl_type.upper(),
            "name": name,
            "atom": command
        }

    async def handle_def(self, command: str, scope_name: str) -> None:
        """DEF komutunu iþler."""
        match = re.match(r"DEF\s+(\w+)\s*\((.*?)\)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"DEF komutunda sözdizimi hatasý: {command}")
        name, params, body = match.groups()
        params = [p.strip() for p in params.split(",")] if params else []
        self.interpreter.functions[name] = {
            "params": params,
            "return_type": "VARIABLE",
            "body": body,
            "inline": True
        }
        self.interpreter.object_counter["FUNCTION"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "FUNCTION",
            "name": name,
            "atom": command
        }

    async def handle_exit(self, command: str, scope_name: str) -> None:
        """EXIT komutunu iþler."""
        match = re.match(r"EXIT\s+(FOR|WHILE|SUB|FUNCTION|PROGRAM)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"EXIT komutunda sözdizimi hatasý: {command}")
        exit_type = match.group(1).upper()
        if exit_type == "PROGRAM":
            self.interpreter.running = False
        elif exit_type in ["FOR", "WHILE"]:
            if self.interpreter.loop_stack:
                self.interpreter.loop_stack.pop()
                while self.program_counter < len(self.interpreter.program):
                    next_cmd = self.interpreter.program[self.program_counter][0].strip().upper()
                    if (exit_type == "FOR" and next_cmd == "NEXT") or (exit_type == "WHILE" and next_cmd == "WEND"):
                        break
                    self.program_counter += 1
        elif exit_type in ["SUB", "FUNCTION"]:
            if self.interpreter.call_stack:
                self.program_counter = self.interpreter.call_stack.pop()
        self.interpreter.object_counter["EXIT"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "EXIT",
            "name": exit_type,
            "atom": command
        }

    async def handle_undim(self, command: str, scope_name: str) -> None:
        """UNDIM komutunu iþler."""
        match = re.match(r"UNDIM\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"UNDIM komutunda sözdizimi hatasý: {command}")
        var_name = match.group(1)
        scope = self.interpreter.current_scope()
        if var_name in scope:
            del scope[var_name]
            self.interpreter.object_counter["VARIABLE"] -= 1
        self.interpreter.object_counter["UNDIM"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "UNDIM",
            "name": var_name,
            "atom": command
        }

    async def handle_setfield(self, command: str, scope_name: str) -> None:
        """SETFIELD komutunu iþler."""
        match = re.match(r"SETFIELD\s+(\w+)\.(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"SETFIELD komutunda sözdizimi hatasý: {command}")
        var_name, field, value_expr = match.groups()
        scope = self.interpreter.current_scope()
        if var_name not in scope:
            raise PdsXException(f"Deðiþken bulunamadý: {var_name}")
        struct = scope[var_name]
        if not isinstance(struct, dict):
            raise PdsXException(f"Geçersiz yapý: {var_name}")
        value = self.interpreter.evaluate_expression(value_expr, scope_name)
        struct[field] = value
        self.interpreter.object_counter["SETFIELD"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "SETFIELD",
            "name": f"{var_name}.{field}",
            "atom": command
        }

    async def handle_getfield(self, command: str, scope_name: str) -> None:
        """GETFIELD komutunu iþler."""
        match = re.match(r"GETFIELD\s+(\w+)\.(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"GETFIELD komutunda sözdizimi hatasý: {command}")
        var_name, field, new_var = match.groups()
        scope = self.interpreter.current_scope()
        if var_name not in scope:
            raise PdsXException(f"Deðiþken bulunamadý: {var_name}")
        struct = scope[var_name]
        if not isinstance(struct, dict):
            raise PdsXException(f"Geçersiz yapý: {var_name}")
        if field in struct:
            scope[new_var] = struct[field]
            self.interpreter.object_counter["VARIABLE"] += 1
            self.interpreter.object_registry[id(struct[field])] = {
                "type": "VARIABLE",
                "name": new_var,
                "atom": str(struct[field])[:100]
            }
        self.interpreter.object_counter["GETFIELD"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "GETFIELD",
            "name": f"{var_name}.{field}",
            "atom": command
        }

    async def handle_addfield(self, command: str, scope_name: str) -> None:
        """ADDFIELD komutunu iþler."""
        match = re.match(r"ADDFIELD\s+(\w+)\.(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"ADDFIELD komutunda sözdizimi hatasý: {command}")
        var_name, field, type_name = match.groups()
        scope = self.interpreter.current_scope()
        if var_name not in scope:
            raise PdsXException(f"Deðiþken bulunamadý: {var_name}")
        struct = scope[var_name]
        if not isinstance(struct, dict):
            raise PdsXException(f"Geçersiz yapý: {var_name}")
        struct[field] = None
        self.interpreter.object_counter[type_name.upper()] += 1
        self.interpreter.object_counter["ADDFIELD"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "ADDFIELD",
            "name": f"{var_name}.{field}",
            "atom": command
        }

    async def handle_removefield(self, command: str, scope_name: str) -> None:
        """REMOVEFIELD komutunu iþler."""
        match = re.match(r"REMOVEFIELD\s+(\w+)\.(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"REMOVEFIELD komutunda sözdizimi hatasý: {command}")
        var_name, field = match.groups()
        scope = self.interpreter.current_scope()
        if var_name not in scope:
            raise PdsXException(f"Deðiþken bulunamadý: {var_name}")
        struct = scope[var_name]
        if not isinstance(struct, dict):
            raise PdsXException(f"Geçersiz yapý: {var_name}")
        if field in struct:
            value = struct.pop(field)
            type_name = next((t for t, v in self.interpreter.type_table.items() if isinstance(value, v)), "VARIABLE")
            self.interpreter.object_counter[type_name] -= 1
        self.interpreter.object_counter["REMOVEFIELD"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "REMOVEFIELD",
            "name": f"{var_name}.{field}",
            "atom": command
        }

    async def handle_newobj(self, command: str, scope_name: str) -> None:
        """NEWOBJ komutunu iþler."""
        match = re.match(r"NEWOBJ\s+(\w+)\((.*?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"NEWOBJ komutunda sözdizimi hatasý: {command}")
        class_name, params, var_name = match.groups()
        if class_name not in self.interpreter.classes:
            raise PdsXException(f"Sýnýf bulunamadý: {class_name}")
        class_def = self.interpreter.classes[class_name]
        instance = {
            "class": class_name,
            "properties": {prop: None for prop, prop_type in class_def["properties"].items()},
            "subs": class_def["subs"],
            "functions": class_def["functions"],
            "gamma": class_def.get("gamma", {}),
            "omega": class_def.get("omega", {})
        }
        param_values = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()]
        if "Init" in class_def["subs"]:
            init_params = class_def["subs"]["Init"]["params"].split(",")
            if len(param_values) != len([p for p in init_params if p.strip()]):
                raise PdsXException(f"Geçersiz parametre sayýsý: {class_name}.Init")
            self.interpreter.current_scope().update({f"_{i}": v for i, v in enumerate(param_values)})
            await self.execute(class_def["subs"]["Init"]["body"], scope_name)
            for i in range(len(param_values)):
                self.interpreter.current_scope().pop(f"_{i}", None)
        scope = self.interpreter.current_scope()
        scope[var_name] = instance
        self.interpreter.object_counter["OBJECT"] += 1
        self.interpreter.object_registry[id(instance)] = {
            "type": "OBJECT",
            "name": var_name,
            "class": class_name,
            "atom": f"obj_{class_name}_{var_name}"
        }
        self.interpreter.object_counter["NEWOBJ"] += 1
        self.interpreter.object_registry[id(command)] = {
            "type": "NEWOBJ",
            "name": var_name,
            "atom": command
        }

    async def handle_countobj(self, command: str, scope_name: str) -> None:
        """COUNTOBJ komutunu iþler."""
        match = re.match(r"COUNTOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"COUNTOBJ komutunda sözdizimi hatasý: {command}")
        type_name, var_name = match.groups()
        scope = self.interpreter.current_scope()
        count = self.interpreter.object_counter.get(type_name, 0)
        scope[var_name] = count
        self.interpreter.object_counter["COUNTOBJ"] += 1
        self.interpreter.object_registry[id(count)] = {
            "type": "COUNTOBJ",
            "name": var_name,
            "atom": str(count)
        }

    async def handle_inspobj(self, command: str, scope_name: str) -> None:
        """INSPOBJ komutunu iþler."""
        match = re.match(r"INSPOBJ\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"INSPOBJ komutunda sözdizimi hatasý: {command}")
        obj_id, var_name = match.groups()
        scope = self.interpreter.current_scope()
        obj = self.interpreter.object_registry.get(int(obj_id), {})
        if not obj:
            raise PdsXException(f"Nesne bulunamadý: {obj_id}")
        scope[var_name] = obj
        self.interpreter.object_counter["INSPOBJ"] += 1
        self.interpreter.object_registry[id(obj)] = {
            "type": "INSPOBJ",
            "name": var_name,
            "atom": json.dumps(obj)[:100]
        }

    async def handle_callapi(self, command: str, scope_name: str) -> None:
        """CALLAPI komutunu iþler."""
        match = re.match(r"CALLAPI\s+\"([^\"]+)\"\s*,\s*METHOD=\"(\w+)\"\s*,\s*HEADERS=\{(.+?)\}\s*,\s*DATA=\{(.+?)\}\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"CALLAPI komutunda sözdizimi hatasý: {command}")
        url, method, headers, data, var_name = match.groups()
        headers = json.loads(f"{{{headers}}}")
        data = json.loads(f"{{{data}}}")
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url, headers=headers, params=data) as resp:
                    response = await resp.json()
            elif method.upper() == "POST":
                async with session.post(url, headers=headers, json=data) as resp:
                    response = await resp.json()
            else:
                raise PdsXException(f"Desteklenmeyen metod: {method}")
        scope = self.interpreter.current_scope()
        scope[var_name] = response
        self.interpreter.object_counter["CALLAPI"] += 1
        self.interpreter.object_registry[id(response)] = {
            "type": "CALLAPI",
            "name": var_name,
            "atom": json.dumps(response)[:100]
        }

    async def handle_calldll(self, command: str, scope_name: str) -> None:
        """CALLDLL komutunu iþler."""
        match = re.match(r"CALLDLL\s+\"([^\"]+)\"\s*,\s*\"(\w+)\"\s*,\s*PARAMS=\((.+?)\)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"CALLDLL komutunda sözdizimi hatasý: {command}")
        dll_name, func_name, params, var_name = match.groups()
        dll = ctypes.WinDLL(dll_name)
        func = getattr(dll, func_name)
        param_values = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()]
        result = func(*param_values)
        scope = self.interpreter.current_scope()
        scope[var_name] = result
        self.interpreter.object_counter["CALLDLL"] += 1
        self.interpreter.object_registry[id(result)] = {
            "type": "CALLDLL",
            "name": var_name,
            "atom": str(result)[:100]
        }

    async def handle_sart(self, command: str, scope_name: str) -> Optional[int]:
        """SART komutunu iþler."""
        match = re.match(r"SART\s+(.+)\s+ATLA\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"SART komutunda sözdizimi hatasý: {command}")
        condition, pipe_id, label = match.groups()
        result = self.interpreter.evaluate_expression(condition, scope_name)
        if result:
            if pipe_id not in self.interpreter.labels:
                raise PdsXException(f"Boru hattý bulunamadý: {pipe_id}")
            if label not in self.interpreter.labels:
                raise PdsXException(f"Etiket bulunamadý: {label}")
            self.interpreter.object_counter["SART"] += 1
            self.interpreter.object_registry[id(command)] = {
                "type": "SART",
                "name": label,
                "atom": command
            }
            self.program_counter = self.interpreter.labels[label]
            return self.program_counter
        return None

    async def handle_alias(self, command: str, scope_name: str) -> None:
        """ALIAS komutunu iþler."""
        match = re.match(r"ALIAS\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXException(f"ALIAS komutunda sözdizimi hatasý: {command}")
        old_name, new_name = match.groups()
        scope = self.interpreter.current_scope()
        if old_name in scope:
            scope[new_name] = scope[old_name]
            self.interpreter.object_counter["ALIAS"] += 1
            self.interpreter.object_registry[id(new_name)] = {
                "type": "ALIAS",
                "name": new_name,
                "atom": old_name
            }
        else:
            raise PdsXException(f"Deðiþken bulunamadý: {old_name}")

    async def handle_restrict(self, command: str, scope_name: str) -> None: