```python
# command_executor.py - PDS-X BASIC v15 Komut Yürütme Motoru
# Version: 1.0.0
# Date: June 14, 2025

import re
import asyncio
import logging
from typing import Dict, Optional, Any
from pdsx_exception import PdsXException
from pdsx_exception2 import PdsXSyntaxError, PdsXRuntimeError, PdsXPipeError
from exception_manager3 import PdsXReplyError, PdsXDatabaseError, PdsXLogicError, PdsXNetworkError, PdsXMLError, PdsXNLPError, PdsXModuleValidatorError, PdsXModuleManagerError, PdsXMultithreadingError, PdsXBusError, PdsXLowLevelError

logging.basicConfig(filename='pdsxe_errors.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("command_executor")

class CommandExecutor:
    """PDS-X BASIC v15 komut yürütme motoru."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.command_handlers: Dict[str, callable] = {
            "PRINT": self.handle_print,
            "LET": self.handle_let,
            "DIM": self.handle_dim,
            "IF": self.handle_if,
            "FOR": self.handle_for,
            "FOREACH": self.handle_foreach,
            "WHILE": self.handle_while,
            "END": self.handle_end,
            "CLASS": self.handle_class,
            "YAPI": self.handle_yapi,
            "GOTO": self.handle_goto,
            "GOSUB": self.handle_gosub,
            "RETURN": self.handle_return,
            "SELECT CASE": self.handle_select_case,
            "DATA": self.handle_data,
            "READ": self.handle_read,
            "RESTORE": self.handle_restore,
            "CHAIN": self.handle_chain,
            "CONT": self.handle_cont,
            "STOP": self.handle_stop,
            "TRON": self.handle_tron,
            "TROFF": self.handle_troff,
            "COMMON": self.handle_common,
            "DECLARE": self.handle_declare,
            "DEF": self.handle_def,
            "EXIT": self.handle_exit,
            "UNDIM": self.handle_undim,
            "SETFIELD": self.handle_setfield,
            "GETFIELD": self.handle_getfield,
            "ADDFIELD": self.handle_addfield,
            "REMOVEFIELD": self.handle_removefield,
            "NEWOBJ": self.handle_newobj,
            "COUNTOBJ": self.handle_countobj,
            "INSPOBJ": self.handle_inspobj,
            "CALLAPI": self.handle_callapi,
            "CALLDLL": self.handle_calldll,
            "SART": self.handle_sart,
            "ALIAS": self.handle_alias,
            "RESTRICT": self.handle_restrict,
            "CLEAR BASIC": self.handle_clear_basic,
            "LISTFILES": self.handle_listfiles,
            "LISTPROG": self.handle_listprog,
            "CHECKFILE": self.handle_checkfile,
            "SCREEN": self.handle_screen,
            "SOUND": self.handle_sound,
            "SEC VAR": self.handle_sec_var,
            "LISTENER": self.handle_listener,
            "CONVERT": self.handle_convert,
            "CAST": self.handle_cast,
            "ALTER TABLE": self.handle_alter_table,
            "CREATE VIEW": self.handle_create_view,
            "ENCRYPT": self.handle_encrypt,
            "DECRYPT": self.handle_decrypt,
            "SIGN": self.handle_sign,
            "VERIFY": self.handle_verify,
            "SECURE_VAR": self.handle_secure_var,
            "CONNECT_IOT": self.handle_connect_iot,
            "PUBLISH_IOT": self.handle_publish_iot,
            "SUBSCRIBE_IOT": self.handle_subscribe_iot,
            "INPUT": self.handle_input,
            "SUB": self.handle_sub,
            "FUNCTION": self.handle_function,
            "CALL": self.handle_call,
            "ON ERROR": self.handle_on_error,
            "SYSINFO": self.handle_sysinfo,
            "CPUINFO": self.handle_cpuinfo,
            "DISKINFO": self.handle_diskinfo,
            "ASSERT": self.handle_assert,
            "WATCH": self.handle_watch,
            "MERGE": self.handle_merge,
            "SORT": self.handle_sort,
            "MAP": self.handle_map,
            "FILTER": self.handle_filter,
            "REDUCE": self.handle_reduce,
            "TRY": self.handle_try,
            "TRACE": self.handle_trace,
            "DATEDIFF": self.handle_datediff,
            "WAIT": self.handle_wait,
            "PUBLISH_SYSINFO": self.handle_publish_sysinfo,
            "SIMPLE_MODE": self.handle_simple_mode,
            "TYPE": self.handle_type,
            "ENUM": self.handle_enum,
            "NEW": self.handle_new,
            "DELETE": self.handle_delete,
            "SIZEOF": self.handle_sizeof,
            "PIPE DEFINE": self.handle_pipe,
            "START PIPE": self.handle_pipe,
            "STOP PIPE": self.handle_pipe,
            "PAUSE PIPE": self.handle_pipe,
            "RESUME PIPE": self.handle_pipe,
            "PIPE BRANCH": self.handle_pipe,
            "PIPE REDIRECT": self.handle_pipe,
            "PIPE JUMP": self.handle_pipe,
            "PIPE IOT CONNECT": self.handle_pipe,
            "PIPE PARALLEL": self.handle_pipe,
            "PIPE RECURSE": self.handle_pipe,
            "PIPE MONITOR LISTENER": self.handle_pipe,
            "PIPE EXECUTE": self.handle_pipe,
            "PIPE FUNCTION CALL": self.handle_pipe,
            "PIPE SCOPE CREATE": self.handle_pipe,
            "PIPE SCOPE DESTROY": self.handle_pipe,
            "PIPE DATA SYNTH": self.handle_pipe,
            "PIPE DATA SEND": self.handle_pipe,
            "PIPE DATA RECEIVE": self.handle_pipe,
            "PIPE INTERRUPT": self.handle_pipe,
            "PIPE CLOUD SYNC": self.handle_pipe,
            "PIPE NETWORK": self.handle_pipe,
            "PIPE EXPERIMENT": self.handle_pipe,
            "PIPE PARADIGM SET": self.handle_pipe,
            "PIPE DATA STRUCTURE USE": self.handle_pipe,
            "PIPE FLOW CONTROL": self.handle_pipe,
            "PIPE RESOURCE_ALLOCATOR": self.handle_pipe,
            "PIPE RESOURCE_FREE": self.handle_pipe,
            "PIPE TIMEOUT_SET": self.handle_pipe,
            "PIPE RETRY": self.handle_pipe,
            "PIPE ALERT": self.handle_pipe,
            "PIPE AUTO_SCALE": self.handle_pipe,
            "PIPE DISTRIBUTE": self.handle_pipe,
            "PIPE SYNC": self.handle_pipe,
            "PIPE ASYNC": self.handle_pipe,
            "PIPE LOG": self.handle_pipe,
            "PIPE EVENT_TRIGGER": self.handle_pipe,
            "PIPE ERROR": self.handle_pipe,
            "PIPE SECURE": self.handle_pipe,
            "PIPE UNSECURE": self.handle_pipe,
            "PIPE GROUP": self.handle_pipe,
            "PIPE UNGROUP": self.handle_pipe,
            "PIPE BROADCAST": self.handle_pipe,
            "PIPE QUANTUM EXECUTE": self.handle_pipe,
            "REPLY SENDER": self.handle_reply,
            "REPLY ASYNC": self.handle_reply,
            "REPLY DISTRIBUTED": self.handle_reply,
            "REPLY WEBSOCKET": self.handle_reply,
            "REPLY ENCRYPTION": self.handle_reply,
            "REPLY ANALYSE": self.handle_reply,
            "REPLY VISUALISATION": self.handle_reply,
            "REPLY QUANTUM": self.handle_reply,
            "REPLY HOLO": self.handle_reply,
            "REPLY SMART": self.handle_reply,
            "REPLY TEMPORAL": self.handle_reply,
            "REPLY PREDICT": self.handle_reply,
            "DB CONNECT": self.handle_db_connect,
            "DB QUERY": self.handle_db_query,
            "DB ASYNC QUERY": self.handle_db_async_query,
            "DB CREATE TABLE": self.handle_db_create_table,
            "DB ISAM CREATE": self.handle_db_isam_create,
            "DB ISAM INSERT": self.handle_db_isam_insert,
            "DB ISAM SEARCH": self.handle_db_isam_search,
            "DB ANALYZE": self.handle_db_analyze,
            "DB VISUALIZE": self.handle_db_visualize,
            "DB QUANTUM": self.handle_db_quantum,
            "DB HOLO": self.handle_db_holo,
            "DB SMART": self.handle_db_smart,
            "DB TEMPORAL": self.handle_db_temporal,
            "DB PREDICT": self.handle_db_predict,
            "FACT": self.handle_fact,
            "RULE": self.handle_rule,
            "BUS_DEFINE": self.handle_bus,
            "BUS_PUBLISH": self.handle_bus,
            "BUS_SUBSCRIBE": self.handle_bus,
            "BUS_UNSUBSCRIBE": self.handle_bus,
            "BUS_START": self.handle_bus,
            "BUS_STOP": self.handle_bus,
            "BUS_PAUSE": self.handle_bus,
            "BUS_RESUME": self.handle_bus,
            "BUS_MONITOR": self.handle_bus,
            "SECURE_BUS": self.handle_bus,
            "UNSECURE_BUS": self.handle_bus,
            "BUS_EVENT_TRIGGER": self.handle_bus,
            "BUS_RECURSE": self.handle_bus,
            "BUS_PRIORITIZE": self.handle_bus,
            "BUS_LOCK": self.handle_bus,
            "BUS_UNLOCK": self.handle_bus,
            "BUS_DATA_SEND": self.handle_bus,
            "BUS_DATA_RECEIVE": self.handle_bus,
            "BUS_IOT_CONNECT": self.handle_bus,
            "BUS_CLOUD_SYNC": self.handle_bus,
            "BUS_NETWORK_CONNECT": self.handle_bus,
            "BUS_EXPERIMENT": self.handle_bus,
            "BUS_PARADIGM_SET": self.handle_bus,
            "BUS_DATA_STRUCTURE_USE": self.handle_bus,
            "BUS_FLOW_CONTROL": self.handle_bus,
            "BUS_RESOURCE_ALLOCATE": self.handle_bus,
            "BUS_RESOURCE_FREE": self.handle_bus,
            "BUS_TIMEOUT_SET": self.handle_bus,
            "BUS_RETRY": self.handle_bus,
            "BUS_ALERT": self.handle_bus,
            "BUS_QUANTUM_SEND": self.handle_bus
        }

    def handle_print(self, command: str, scope_name: Optional[str] = None) -> None:
        """PRINT komutunu işler."""
        try:
            match = re.match(r"PRINT\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("PRINT komutunda sözdizimi hatası", code="PRINT001")
            expr = match.group(1)
            result = self.interpreter.evaluate_expression(expr, scope_name)
            print(result)
            log.debug(f"PRINT yürütüldü: {expr} -> {result}")
        except PdsXException as e:
            log.error(f"PRINT yürütme hatası: {str(e)}")
            raise

    def handle_let(self, command: str, scope_name: Optional[str] = None) -> None:
        """LET komutunu işler."""
        try:
            match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("LET komutunda sözdizimi hatası", code="LET001")
            var_name, expr = match.groups()
            value = self.interpreter.evaluate_expression(expr, scope_name)
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            scope[var_name] = value
            log.debug(f"LET yürütüldü: {var_name} = {value}")
        except PdsXException as e:
            log.error(f"LET yürütme hatası: {str(e)}")
            raise

    def handle_dim(self, command: str, scope_name: Optional[str] = None) -> None:
        """DIM komutunu işler."""
        try:
            match = re.match(r"DIM\s+(\w+)\s+AS\s+(\w+)(?:\s*,\s*([\d,]+))?(?:\s*=\s*(.+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("DIM komutunda sözdizimi hatası", code="DIM001")
            var_name, type_name, dim_str, init_value = match.groups()
            type_obj = self.interpreter.type_table.get(type_name.upper())
            if not type_obj:
                raise PdsXRuntimeError(f"Geçersiz veri tipi: {type_name}", code="DIM002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if dim_str:
                dims = [int(x) for x in dim_str.split(",")]
                scope[var_name] = np.zeros(dims, dtype=type_obj)
            else:
                scope[var_name] = type_obj() if not init_value else self.interpreter.evaluate_expression(init_value, scope_name)
            log.debug(f"DIM yürütüldü: {var_name} AS {type_name}")
        except PdsXException as e:
            log.error(f"DIM yürütme hatası: {str(e)}")
            raise

    def handle_if(self, command: str, scope_name: Optional[str] = None) -> None:
        """IF komutunu işler."""
        try:
            match = re.match(r"IF\s+(.+?)\s+THEN", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("IF komutunda sözdizimi hatası", code="IF001")
            condition = match.group(1)
            result = self.interpreter.evaluate_expression(condition, scope_name)
            self.interpreter.if_stack.append(bool(result))
            if not result:
                self.interpreter.program_counter = self.find_matching_else_or_end()
            log.debug(f"IF yürütüldü: {condition} -> {result}")
        except PdsXException as e:
            log.error(f"IF yürütme hatası: {str(e)}")
            raise

    def handle_for(self, command: str, scope_name: Optional[str] = None) -> None:
        """FOR komutunu işler."""
        try:
            match = re.match(r"FOR\s+(\w+)\s*=\s*(.+?)\s+TO\s+(.+?)(?:\s+STEP\s+(.+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("FOR komutunda sözdizimi hatası", code="FOR001")
            var_name, start, end, step = match.groups()
            start_val = self.interpreter.evaluate_expression(start, scope_name)
            end_val = self.interpreter.evaluate_expression(end, scope_name)
            step_val = self.interpreter.evaluate_expression(step, scope_name) if step else 1
            if step_val == 0:
                raise PdsXRuntimeError("Adım sıfır olamaz", code="FOR002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            scope[var_name] = start_val
            self.interpreter.loop_stack.append({
                "type": "FOR",
                "var": var_name,
                "end": end_val,
                "step": step_val,
                "start_pc": self.interpreter.program_counter
            })
            log.debug(f"FOR yürütüldü: {var_name} = {start_val} TO {end_val} STEP {step_val}")
        except PdsXException as e:
            log.error(f"FOR yürütme hatası: {str(e)}")
            raise

    def handle_foreach(self, command: str, scope_name: Optional[str] = None) -> None:
        """FOREACH komutunu işler."""
        try:
            match = re.match(r"FOREACH\s+(\w+)\s+IN\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("FOREACH komutunda sözdizimi hatası", code="FOREACH001")
            var_name, collection = match.groups()
            coll_val = self.interpreter.evaluate_expression(collection, scope_name)
            if not isinstance(coll_val, (list, tuple, set, dict)):
                raise PdsXRuntimeError("Geçersiz koleksiyon", code="FOREACH002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            self.interpreter.loop_stack.append({
                "type": "FOREACH",
                "var": var_name,
                "collection": iter(coll_val),
                "start_pc": self.interpreter.program_counter
            })
            try:
                scope[var_name] = next(self.interpreter.loop_stack[-1]["collection"])
                log.debug(f"FOREACH yürütüldü: {var_name} IN {collection}")
            except StopIteration:
                self.interpreter.program_counter = self.find_matching_next()
        except PdsXException as e:
            log.error(f"FOREACH yürütme hatası: {str(e)}")
            raise

    def handle_while(self, command: str, scope_name: Optional[str] = None) -> None:
        """WHILE komutunu işler."""
        try:
            match = re.match(r"WHILE\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("WHILE komutunda sözdizimi hatası", code="WHILE001")
            condition = match.group(1)
            result = self.interpreter.evaluate_expression(condition, scope_name)
            self.interpreter.loop_stack.append({
                "type": "WHILE",
                "condition": condition,
                "start_pc": self.interpreter.program_counter
            })
            if not result:
                self.interpreter.program_counter = self.find_matching_wend()
            log.debug(f"WHILE yürütüldü: {condition} -> {result}")
        except PdsXException as e:
            log.error(f"WHILE yürütme hatası: {str(e)}")
            raise

    def handle_end(self, command: str, scope_name: Optional[str] = None) -> None:
        """END komutunu işler."""
        try:
            match = re.match(r"END\s+(\w+)", command, re.IGNORECASE)
            if not match:
                self.interpreter.running = False
                log.debug("END yürütüldü: Program sonlandırıldı")
                return
            block_type = match.group(1).upper()
            if block_type == "IF":
                if not self.interpreter.if_stack:
                    raise PdsXRuntimeError("Kapatılacak IF bloğu bulunamadı", code="END003")
                self.interpreter.if_stack.pop()
            elif block_type in ("FOR", "FOREACH", "WHILE"):
                if not self.interpreter.loop_stack:
                    raise PdsXRuntimeError("Kapatılacak döngü bulunamadı", code="END002")
                self.interpreter.loop_stack.pop()
            else:
                raise PdsXSyntaxError(f"Geçersiz END tipi: {block_type}", code="END001")
            log.debug(f"END yürütüldü: {block_type}")
        except PdsXException as e:
            log.error(f"END yürütme hatası: {str(e)}")
            raise

    def handle_class(self, command: str, scope_name: Optional[str] = None) -> None:
        """CLASS komutunu işler."""
        try:
            match = re.match(r"CLASS\s+(\w+)\s*(?:PARENT\s+(\w+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("CLASS komutunda sözdizimi hatası", code="CLASS001")
            class_name, parent = match.groups()
            self.interpreter.memory_manager.type_manager.define_class(
                class_name,
                methods={},
                private_methods={},
                static_vars={},
                parent=parent,
                abstract=False
            )
            log.debug(f"CLASS yürütüldü: {class_name}")
        except PdsXException as e:
            log.error(f"CLASS yürütme hatası: {str(e)}")
            raise

    def handle_yapi(self, command: str, scope_name: Optional[str] = None) -> None:
        """YAPI komutunu işler."""
        try:
            match = re.match(r"YAPI\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("YAPI komutunda sözdizimi hatası", code="YAPI001")
            name = match.group(1)
            self.interpreter.memory_manager.type_manager.define_type(name, fields=[])
            log.debug(f"YAPI yürütüldü: {name}")
        except PdsXException as e:
            log.error(f"YAPI yürütme hatası: {str(e)}")
            raise

    def handle_goto(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOTO komutunu işler."""
        try:
            match = re.match(r"GOTO\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("GOTO komutunda sözdizimi hatası", code="GOTO001")
            label = match.group(1)
            if label not in self.interpreter.labels:
                raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOTO002")
            self.interpreter.program_counter = self.interpreter.labels[label]
            log.debug(f"GOTO yürütüldü: {label}")
        except PdsXException as e:
            log.error(f"GOTO yürütme hatası: {str(e)}")
            raise

    def handle_gosub(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOSUB komutunu işler."""
        try:
            match = re.match(r"GOSUB\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("GOSUB komutunda sözdizimi hatası", code="GOSUB001")
            label = match.group(1)
            if label not in self.interpreter.labels:
                raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOSUB002")
            self.interpreter.call_stack.append({"return_pc": self.interpreter.program_counter + 1})
            self.interpreter.program_counter = self.interpreter.labels[label]
            log.debug(f"GOSUB yürütüldü: {label}")
        except PdsXException as e:
            log.error(f"GOSUB yürütme hatası: {str(e)}")
            raise

    def handle_return(self, command: str, scope_name: Optional[str] = None) -> None:
        """RETURN komutunu işler."""
        try:
            if not self.interpreter.call_stack:
                raise PdsXRuntimeError("Geri dönülecek yordam yok", code="RETURN001")
            call_info = self.interpreter.call_stack.pop()
            self.interpreter.program_counter = call_info["return_pc"]
            log.debug("RETURN yürütüldü")
        except PdsXException as e:
            log.error(f"RETURN yürütme hatası: {str(e)}")
            raise

    def handle_select_case(self, command: str, scope_name: Optional[str] = None) -> None:
        """SELECT CASE komutunu işler."""
        try:
            match = re.match(r"SELECT CASE\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("SELECT CASE komutunda sözdizimi hatası", code="SELECT001")
            expr = match.group(1)
            value = self.interpreter.evaluate_expression(expr, scope_name)
            self.interpreter.select_stack.append({"value": value, "start_pc": self.interpreter.program_counter})
            log.debug(f"SELECT CASE yürütüldü: {expr} -> {value}")
        except PdsXException as e:
            log.error(f"SELECT CASE yürütme hatası: {str(e)}")
            raise

    def handle_data(self, command: str, scope_name: Optional[str] = None) -> None:
        """DATA komutunu işler."""
        try:
            match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("DATA komutunda sözdizimi hatası", code="DATA001")
            values = [x.strip() for x in match.group(1).split(",")]
            self.interpreter.data_list.extend(values)
            log.debug(f"DATA yürütüldü: {values}")
        except PdsXException as e:
            log.error(f"DATA yürütme hatası: {str(e)}")
            raise

    def handle_read(self, command: str, scope_name: Optional[str] = None) -> None:
        """READ komutunu işler."""
        try:
            match = re.match(r"READ\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("READ komutunda sözdizimi hatası", code="READ001")
            var_names = [x.strip() for x in match.group(1).split(",")]
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            for var_name in var_names:
                if self.interpreter.data_pointer >= len(self.interpreter.data_list):
                    raise PdsXRuntimeError("Veri listesi sonu", code="READ002")
                value = self.interpreter.evaluate_expression(self.interpreter.data_list[self.interpreter.data_pointer])
                scope[var_name] = value
                self.interpreter.data_pointer += 1
            log.debug(f"READ yürütüldü: {var_names}")
        except PdsXException as e:
            log.error(f"READ yürütme hatası: {str(e)}")
            raise

    def handle_restore(self, command: str, scope_name: Optional[str] = None) -> None:
        """RESTORE komutunu işler."""
        try:
            self.interpreter.data_pointer = 0
            log.debug("RESTORE yürütüldü")
        except PdsXException as e:
            log.error(f"RESTORE yürütme hatası: {str(e)}")
            raise

    def handle_chain(self, command: str, scope_name: Optional[str] = None) -> None:
        """CHAIN komutunu işler."""
        try:
            match = re.match(r"CHAIN\s+\"(.+?)\"", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("CHAIN komutunda sözdizimi hatası", code="CHAIN001")
            file_name = match.group(1)
            with open(file_name, 'r', encoding='utf-8') as f:
                code = f.read()
            self.interpreter.parse_program(code)
            self.interpreter.program_counter = 0
            log.debug(f"CHAIN yürütüldü: {file_name}")
        except FileNotFoundError:
            log.error(f"Program yükleme hatası: {file_name}")
            raise PdsXRuntimeError(f"Program yükleme hatası: {file_name}", code="CHAIN002")

    def handle_cont(self, command: str, scope_name: Optional[str] = None) -> None:
        """CONT komutunu işler."""
        try:
            self.interpreter.running = True
            log.debug("CONT yürütüldü")
        except PdsXException as e:
            log.error(f"CONT yürütme hatası: {str(e)}")
            raise

    def handle_stop(self, command: str, scope_name: Optional[str] = None) -> None:
        """STOP komutunu işler."""
        try:
            self.interpreter.running = False
            log.debug("STOP yürütüldü")
        except PdsXException as e:
            log.error(f"STOP yürütme hatası: {str(e)}")
            raise

    def handle_tron(self, command: str, scope_name: Optional[str] = None) -> None:
        """TRON komutunu işler."""
        try:
            self.interpreter.trace_mode = True
            log.debug("TRON yürütüldü")
        except PdsXException as e:
            log.error(f"TRON yürütme hatası: {str(e)}")
            raise

    def handle_troff(self, command: str, scope_name: Optional[str] = None) -> None:
        """TROFF komutunu işler."""
        try:
            self.interpreter.trace_mode = False
            log.debug("TROFF yürütüldü")
        except PdsXException as e:
            log.error(f"TROFF yürütme hatası: {str(e)}")
            raise

    def handle_common(self, command: str, scope_name: Optional[str] = None) -> None:
        """COMMON komutunu işler."""
        try:
            match = re.match(r"COMMON\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("COMMON komutunda sözdizimi hatası", code="COMMON001")
            var_names = [x.strip() for x in match.group(1).split(",")]
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            for var_name in var_names:
                if var_name not in scope:
                    raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="COMMON002")
                self.interpreter.shared_vars[var_name].append(scope[var_name])
            log.debug(f"COMMON yürütüldü: {var_names}")
        except PdsXException as e:
            log.error(f"COMMON yürütme hatası: {str(e)}")
            raise

    def handle_declare(self, command: str, scope_name: Optional[str] = None) -> None:
        """DECLARE komutunu işler."""
        try:
            match = re.match(r"DECLARE\s+(FUNCTION|SUB)\s+(\w+)\s*\((.*?)\)\s*(?:AS\s+(\w+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("DECLARE komutunda sözdizimi hatası", code="DECLARE001")
            kind, name, params, return_type = match.groups()
            if return_type and return_type.upper() not in self.interpreter.type_table:
                raise PdsXRuntimeError(f"Geçersiz dönüş tipi: {return_type}", code="DECLARE002")
            self.interpreter.functions[name.upper()] = {
                "kind": kind.upper(),
                "params": [p.strip() for p in params.split(",") if p.strip()],
                "return_type": return_type.upper() if return_type else None
            }
            log.debug(f"DECLARE yürütüldü: {kind} {name}")
        except PdsXException as e:
            log.error(f"DECLARE yürütme hatası: {str(e)}")
            raise

    def handle_def(self, command: str, scope_name: Optional[str] = None) -> None:
        """DEF komutunu işler."""
        try:
            match = re.match(r"DEF\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("DEF komutunda sözdizimi hatası", code="DEF001")
            var_name, expr = match.groups()
            value = self.interpreter.evaluate_expression(expr, scope_name)
            self.interpreter.global_vars[var_name] = value
            log.debug(f"DEF yürütüldü: {var_name} = {value}")
        except PdsXException as e:
            log.error(f"DEF yürütme hatası: {str(e)}")
            raise

    def handle_exit(self, command: str, scope_name: Optional[str] = None) -> None:
        """EXIT komutunu işler."""
        try:
            match = re.match(r"EXIT\s+(\w+)", command, re.IGNORECASE)
            if not match:
                self.interpreter.running = False
                log.debug("EXIT yürütüldü: Program sonlandırıldı")
                return
            exit_type = match.group(1).upper()
            if exit_type in ("FOR", "FOREACH", "WHILE"):
                if not self.interpreter.loop_stack:
                    raise PdsXRuntimeError("Çıkılacak döngü bulunamadı", code="EXIT001")
                self.interpreter.program_counter = self.find_matching_next()
                self.interpreter.loop_stack.pop()
            elif exit_type == "SUB":
                if not self.interpreter.call_stack:
                    raise PdsXRuntimeError("Çıkılacak yordam bulunamadı", code="EXIT002")
                call_info = self.interpreter.call_stack.pop()
                self.interpreter.program_counter = call_info["return_pc"]
            else:
                raise PdsXSyntaxError(f"Geçersiz EXIT tipi: {exit_type}", code="EXIT003")
            log.debug(f"EXIT yürütüldü: {exit_type}")
        except PdsXException as e:
            log.error(f"EXIT yürütme hatası: {str(e)}")
            raise

    def handle_undim(self, command: str, scope_name: Optional[str] = None) -> None:
        """UNDIM komutunu işler."""
        try:
            match = re.match(r"UNDIM\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("UNDIM komutunda sözdizimi hatası", code="UNDIM001")
            var_name = match.group(1)
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if var_name not in scope:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="UNDIM002")
            del scope[var_name]
            log.debug(f"UNDIM yürütüldü: {var_name}")
        except PdsXException as e:
            log.error(f"UNDIM yürütme hatası: {str(e)}")
            raise

    def handle_setfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """SETFIELD komutunu işler."""
        try:
            match = re.match(r"SETFIELD\s+(\w+)\s*,\s*(\w+)\s*,\s*(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("SETFIELD komutunda sözdizimi hatası", code="SETFIELD001")
            struct_name, field_name, value_expr = match.groups()
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if struct_name not in scope:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {struct_name}", code="SETFIELD002")
            struct = scope[struct_name]
            if not isinstance(struct, dict):
                raise PdsXRuntimeError(f"Geçersiz yapı: {struct_name}", code="SETFIELD003")
            value = self.interpreter.evaluate_expression(value_expr, scope_name)
            struct[field_name] = value
            log.debug(f"SETFIELD yürütüldü: {struct_name}.{field_name} = {value}")
        except PdsXException as e:
            log.error(f"SETFIELD yürütme hatası: {str(e)}")
            raise

    def handle_getfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """GETFIELD komutunu işler."""
        try:
            match = re.match(r"GETFIELD\s+(\w+)\s*,\s*(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("GETFIELD komutunda sözdizimi hatası", code="GETFIELD001")
            struct_name, field_name, var_name = match.groups()
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if struct_name not in scope:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {struct_name}", code="GETFIELD002")
            struct = scope[struct_name]
            if not isinstance(struct, dict):
                raise PdsXRuntimeError(f"Geçersiz yapı: {struct_name}", code="GETFIELD003")
            scope[var_name] = struct.get(field_name)
            log.debug(f"GETFIELD yürütüldü: {var_name} = {struct_name}.{field_name}")
        except PdsXException as e:
            log.error(f"GETFIELD yürütme hatası: {str(e)}")
            raise

    def handle_addfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """ADDFIELD komutunu işler."""
        try:
            match = re.match(r"ADDFIELD\s+(\w+)\s*,\s*(\w+)\s+AS\s+(\w+)(?:\s*=\s*(.+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("ADDFIELD komutunda sözdizimi hatası", code="ADDFIELD001")
            struct_name, field_name, type_name, init_value = match.groups()
            type_obj = self.interpreter.type_table.get(type_name.upper())
            if not type_obj:
                raise PdsXRuntimeError(f"Geçersiz veri tipi: {type_name}", code="ADDFIELD002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if struct_name not in scope:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {struct_name}", code="ADDFIELD003")
            struct = scope[struct_name]
            if not isinstance(struct, dict):
                raise PdsXRuntimeError(f"Geçersiz yapı: {struct_name}", code="ADDFIELD004")
            struct[field_name] = type_obj() if not init_value else self.interpreter.evaluate_expression(init_value, scope_name)
            log.debug(f"ADDFIELD yürütüldü: {struct_name}.{field_name} AS {type_name}")
        except PdsXException as e:
            log.error(f"ADDFIELD yürütme hatası: {str(e)}")
            raise

    def handle_removefield(self, command: str, scope_name: Optional[str] = None) -> None:
        """REMOVEFIELD komutunu işler."""
        try:
            match = re.match(r"REMOVEFIELD\s+(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("REMOVEFIELD komutunda sözdizimi hatası", code="REMOVEFIELD001")
            struct_name, field_name = match.groups()
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if struct_name not in scope:
                raise PdsXRuntimeError(f"Değişken bulunamadı: {struct_name}", code="REMOVEFIELD002")
            struct = scope[struct_name]
            if not isinstance(struct, dict):
                raise PdsXRuntimeError(f"Geçersiz yapı: {struct_name}", code="REMOVEFIELD003")
            if field_name not in struct:
                raise PdsXRuntimeError(f"Alan bulunamadı: {field_name}", code="REMOVEFIELD004")
            del struct[field_name]
            log.debug(f"REMOVEFIELD yürütüldü: {struct_name}.{field_name}")
        except PdsXException as e:
            log.error(f"REMOVEFIELD yürütme hatası: {str(e)}")
            raise

    def handle_newobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """NEWOBJ komutunu işler."""
        try:
            match = re.match(r"NEWOBJ\s+(\w+)\s+AS\s+(\w+)(?:\s*\((.*?)\))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("NEWOBJ komutunda sözdizimi hatası", code="NEWOBJ001")
            class_name, var_name, params = match.groups()
            if class_name.upper() not in self.interpreter.type_table:
                raise PdsXRuntimeError(f"Sınıf bulunamadı: {class_name}", code="NEWOBJ002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            params_list = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()] if params else []
            instance_id = self.interpreter.memory_manager.type_manager.create_class_instance(class_name)
            scope[var_name] = instance_id
            log.debug(f"NEWOBJ yürütüldü: {class_name} -> {var_name}")
        except PdsXException as e:
            log.error(f"NEWOBJ yürütme hatası: {str(e)}")
            raise

    def handle_countobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """COUNTOBJ komutunu işler."""
        try:
            match = re.match(r"COUNTOBJ\s+(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("COUNTOBJ komutunda sözdizimi hatası", code="COUNTOBJ001")
            type_name, var_name = match.groups()
            if type_name.upper() not in self.interpreter.type_table:
                raise PdsXRuntimeError(f"Tip bulunamadı: {type_name}", code="COUNTOBJ002")
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            count = len(self.interpreter.memory_manager.type_manager.instances[type_name.upper()])
            scope[var_name] = count
            log.debug(f"COUNTOBJ yürütüldü: {type_name} -> {count}")
        except PdsXException as e:
            log.error(f"COUNTOBJ yürütme hatası: {str(e)}")
            raise

    def handle_inspobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """INSPOBJ komutunu işler."""
        try:
            match = re.match(r"INSPOBJ\s+(\w+)\s*,\s*(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("INSPOBJ komutunda sözdizimi hatası", code="INSPOBJ001")
            instance_id, var_name = match.groups()
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            if instance_id not in scope:
                raise PdsXRuntimeError(f"Nesne bulunamadı: {instance_id}", code="INSPOBJ002")
            instance = self.interpreter.memory_manager.type_manager.get_instance(scope[instance_id])
            scope[var_name] = instance
            log.debug(f"INSPOBJ yürütüldü: {instance_id} -> {var_name}")
        except PdsXException as e:
            log.error(f"INSPOBJ yürütme hatası: {str(e)}")
            raise

    def handle_callapi(self, command: str, scope_name: Optional[str] = None) -> None:
        """CALLAPI komutunu işler."""
        try:
            match = re.match(r"CALLAPI\s+\"(.+?)\"\s+(\w+)\s+\[(.+?)\]", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("CALLAPI komutunda sözdizimi hatası", code="CALLAPI001")
            url, var_name, params = match.groups()
            params_dict = eval(params, self.interpreter.current_scope())
            response = self.interpreter.network.web_get(url, **params_dict)
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            scope[var_name] = response
            log.debug(f"CALLAPI yürütüldü: {url} -> {var_name}")
        except PdsXException as e:
            log.error(f"CALLAPI yürütme hatası: {str(e)}")
            raise PdsXNetworkError(f"HTTP isteği başarısız: {str(e)}", code="CALLAPI002")

    def handle_calldll(self, command: str, scope_name: Optional[str] = None) -> None:
        """CALLDLL komutunu işler."""
        try:
            match = re.match(r"CALLDLL\s+\"(.+?)\"\s*,\s*\"(.+?)\"\s*,\s*(\w+)(?:\s*\((.*?)\))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("CALLDLL komutunda sözdizimi hatası", code="CALLDLL001")
            dll_name, func_name, var_name, params = match.groups()
            params_list = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()] if params else []
            result = self.interpreter.dll_manager.call(dll_name, func_name, *params_list)
            scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
            scope[var_name] = result
            log.debug(f"CALLDLL yürütüldü: {dll_name}.{func_name} -> {var_name}")
        except PdsXException as e:
            log.error(f"CALLDLL yürütme hatası: {str(e)}")
            raise PdsXLowLevelError(f"DLL çağrısı hatası: {str(e)}", code="CALLDLL002")

    def handle_sart(self, command: str, scope_name: Optional[str] = None) -> None:
        """SART komutunu işler."""
        try:
            match = re.match(r"SART\s+(.+?)\s+ATLA\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("SART komutunda sözdizimi hatası, lütfen koşul ve etiket yazımını kontrol edin", code="SART001")
            condition, pipe_id, label = match.groups()
            if pipe_id not in self.interpreter.pipe_manager.pipes:
                raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", code="SART002")
            result = self.interpreter.evaluate_expression(condition, scope_name)
            if result:
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı