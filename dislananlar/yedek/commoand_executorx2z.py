```python
# command_executor.py - PDS-X BASIC v15 Komut Yürütme Motoru
# Version: 1.0.0
# Date: June 14, 2025
import re
import asyncio
import logging
from typing import Dict, Optional, Any
from pdsx_exception import PdsXException
from pdsx_exception2 import PdsXSyntaxError, PdsXRuntimeError

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
            "ALIAS": self.handle_alias
        }

    def handle_print(self, command: str, scope_name: Optional[str] = None) -> None:
        """PRINT komutunu işler."""
        match = re.match(r"PRINT\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("PRINT komutunda sözdizimi hatası", code="PRINT001")
        expr = match.group(1)
        result = self.interpreter.evaluate_expression(expr, scope_name)
        print(result)
        log.debug(f"PRINT yürütüldü: {expr} -> {result}")

    def handle_let(self, command: str, scope_name: Optional[str] = None) -> None:
        """LET komutunu işler."""
        match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("LET komutunda sözdizimi hatası", code="LET001")
        var_name, expr = match.groups()
        value = self.interpreter.evaluate_expression(expr, scope_name)
        scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
        scope[var_name] = value
        log.debug(f"LET yürütüldü: {var_name} = {value}")

    def handle_dim(self, command: str, scope_name: Optional[str] = None) -> None:
        """DIM komutunu işler."""
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

    def handle_if(self, command: str, scope_name: Optional[str] = None) -> None:
        """IF komutunu işler."""
        match = re.match(r"IF\s+(.+?)\s+THEN", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("IF komutunda sözdizimi hatası", code="IF001")
        condition = match.group(1)
        result = self.interpreter.evaluate_expression(condition, scope_name)
        self.interpreter.if_stack.append(bool(result))
        if not result:
            self.interpreter.program_counter = self.find_matching_else_or_end()
        log.debug(f"IF yürütüldü: {condition} -> {result}")

    def handle_for(self, command: str, scope_name: Optional[str] = None) -> None:
        """FOR komutunu işler."""
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

    def handle_foreach(self, command: str, scope_name: Optional[str] = None) -> None:
        """FOREACH komutunu işler."""
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
        except StopIteration:
            self.interpreter.program_counter = self.find_matching_next()
        log.debug(f"FOREACH yürütüldü: {var_name} IN {collection}")

    def handle_while(self, command: str, scope_name: Optional[str] = None) -> None:
        """WHILE komutunu işler."""
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

    def handle_end(self, command: str, scope_name: Optional[str] = None) -> None:
        """END komutunu işler."""
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

    def handle_goto(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOTO komutunu işler."""
        match = re.match(r"GOTO\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GOTO komutunda sözdizimi hatası", code="GOTO001")
        label = match.group(1)
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOTO002")
        self.interpreter.program_counter = self.interpreter.labels[label]
        log.debug(f"GOTO yürütüldü: {label}")

    def handle_gosub(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOSUB komutunu işler."""
        match = re.match(r"GOSUB\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("GOSUB komutunda sözdizimi hatası", code="GOSUB001")
        label = match.group(1)
        if label not in self.interpreter.labels:
            raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="GOSUB002")
        self.interpreter.call_stack.append({"return_pc": self.interpreter.program_counter + 1})
        self.interpreter.program_counter = self.interpreter.labels[label]
        log.debug(f"GOSUB yürütüldü: {label}")

    def handle_return(self, command: str, scope_name: Optional[str] = None) -> None:
        """RETURN komutunu işler."""
        if not self.interpreter.call_stack:
            raise PdsXRuntimeError("Geri dönülecek yordam yok", code="RETURN001")
        call_info = self.interpreter.call_stack.pop()
        self.interpreter.program_counter = call_info["return_pc"]
        log.debug("RETURN yürütüldü")

    def handle_select_case(self, command: str, scope_name: Optional[str] = None) -> None:
        """SELECT CASE komutunu işler."""
        match = re.match(r"SELECT CASE\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SELECT CASE komutunda sözdizimi hatası", code="SELECT001")
        expr = match.group(1)
        value = self.interpreter.evaluate_expression(expr, scope_name)
        self.interpreter.select_stack.append({"value": value, "start_pc": self.interpreter.program_counter})
        log.debug(f"SELECT CASE yürütüldü: {expr} -> {value}")

    def handle_data(self, command: str, scope_name: Optional[str] = None) -> None:
        """DATA komutunu işler."""
        match = re.match(r"DATA\s+(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DATA komutunda sözdizimi hatası", code="DATA001")
        values = [x.strip() for x in match.group(1).split(",")]
        self.interpreter.data_list.extend(values)
        log.debug(f"DATA yürütüldü: {values}")

    def handle_read(self, command: str, scope_name: Optional[str] = None) -> None:
        """READ komutunu işler."""
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

    def handle_restore(self, command: str, scope_name: Optional[str] = None) -> None:
        """RESTORE komutunu işler."""
        self.interpreter.data_pointer = 0
        log.debug("RESTORE yürütüldü")

    def handle_chain(self, command: str, scope_name: Optional[str] = None) -> None:
        """CHAIN komutunu işler."""
        match = re.match(r"CHAIN\s+\"(.+?)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CHAIN komutunda sözdizimi hatası", code="CHAIN001")
        file_name = match.group(1)
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                code = f.read()
            self.interpreter.parse_program(code)
            self.interpreter.program_counter = 0
            log.debug(f"CHAIN yürütüldü: {file_name}")
        except FileNotFoundError:
            raise PdsXRuntimeError(f"Program yükleme hatası: {file_name}", code="CHAIN002")

    def handle_cont(self, command: str, scope_name: Optional[str] = None) -> None:
        """CONT komutunu işler."""
        self.interpreter.running = True
        log.debug("CONT yürütüldü")

    def handle_stop(self, command: str, scope_name: Optional[str] = None) -> None:
        """STOP komutunu işler."""
        self.interpreter.running = False
        log.debug("STOP yürütüldü")

    def handle_tron(self, command: str, scope_name: Optional[str] = None) -> None:
        """TRON komutunu işler."""
        self.interpreter.trace_mode = True
        log.debug("TRON yürütüldü")

    def handle_troff(self, command: str, scope_name: Optional[str] = None) -> None:
        """TROFF komutunu işler."""
        self.interpreter.trace_mode = False
        log.debug("TROFF yürütüldü")

    def handle_common(self, command: str, scope_name: Optional[str] = None) -> None:
        """COMMON komutunu işler."""
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

    def handle_declare(self, command: str, scope_name: Optional[str] = None) -> None:
        """DECLARE komutunu işler."""
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

    def handle_def(self, command: str, scope_name: Optional[str] = None) -> None:
        """DEF komutunu işler."""
        match = re.match(r"DEF\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("DEF komutunda sözdizimi hatası", code="DEF001")
        var_name, expr = match.groups()
        value = self.interpreter.evaluate_expression(expr, scope_name)
        self.interpreter.global_vars[var_name] = value
        log.debug(f"DEF yürütüldü: {var_name} = {value}")

    def handle_exit(self, command: str, scope_name: Optional[str] = None) -> None:
        """EXIT komutunu işler."""
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

    def handle_undim(self, command: str, scope_name: Optional[str] = None) -> None:
        """UNDIM komutunu işler."""
        match = re.match(r"UNDIM\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("UNDIM komutunda sözdizimi hatası", code="UNDIM001")
        var_name = match.group(1)
        scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
        if var_name not in scope:
            raise PdsXRuntimeError(f"Değişken bulunamadı: {var_name}", code="UNDIM002")
        del scope[var_name]
        log.debug(f"UNDIM yürütüldü: {var_name}")

    def handle_alias(self, command: str, scope_name: Optional[str] = None) -> None:
        """ALIAS komutunu işler."""
        match = re.match(r"ALIAS\s+\"(.+?)\"\s+AS\s+\"(.+?)\"", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("ALIAS komutunda sözdizimi hatası", code="ALIAS001")
        command_name, alias_name = match.groups()
        self.interpreter.alias(command_name, alias_name, scope_name)
        self.command_handlers[alias_name.upper()] = self.command_handlers.get(command_name.upper())
        log.debug(f"ALIAS yürütüldü: {command_name} -> {alias_name}")
        
        # Yapı ve nesne yönetimi komutları
    def handle_setfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """SETFIELD komutunu işler."""
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

    def handle_getfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """GETFIELD komutunu işler."""
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

    def handle_addfield(self, command: str, scope_name: Optional[str] = None) -> None:
        """ADDFIELD komutunu işler."""
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

    def handle_removefield(self, command: str, scope_name: Optional[str] = None) -> None:
        """REMOVEFIELD komutunu işler."""
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

    def handle_newobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """NEWOBJ komutunu işler."""
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

    def handle_countobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """COUNTOBJ komutunu işler."""
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

    def handle_inspobj(self, command: str, scope_name: Optional[str] = None) -> None:
        """INSPOBJ komutunu işler."""
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

    def handle_callapi(self, command: str, scope_name: Optional[str] = None) -> None:
        """CALLAPI komutunu işler."""
        match = re.match(r"CALLAPI\s+\"(.+?)\"\s+(\w+)\s+\[(.+?)\]", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLAPI komutunda sözdizimi hatası", code="CALLAPI001")
        url, var_name, params = match.groups()
        params_dict = eval(params, self.interpreter.current_scope())
        response = self.interpreter.network.web_get(url, **params_dict)
        scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
        scope[var_name] = response
        log.debug(f"CALLAPI yürütüldü: {url} -> {var_name}")

    def handle_calldll(self, command: str, scope_name: Optional[str] = None) -> None:
        """CALLDLL komutunu işler."""
        match = re.match(r"CALLDLL\s+\"(.+?)\"\s*,\s*\"(.+?)\"\s*,\s*(\w+)(?:\s*\((.*?)\))?", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("CALLDLL komutunda sözdizimi hatası", code="CALLDLL001")
        dll_name, func_name, var_name, params = match.groups()
        params_list = [self.interpreter.evaluate_expression(p.strip(), scope_name) for p in params.split(",") if p.strip()] if params else []
        result = self.interpreter.dll_manager.call(dll_name, func_name, *params_list)
        scope = self.interpreter.current_scope() if not scope_name else self.interpreter.modules[scope_name]["variables"]
        scope[var_name] = result
        log.debug(f"CALLDLL yürütüldü: {dll_name}.{func_name} -> {var_name}")

    def handle_sart(self, command: str, scope_name: Optional[str] = None) -> None:
        """SART komutunu işler."""
        match = re.match(r"SART\s+(.+?)\s+ATLA\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SART komutunda sözdizimi hatası", code="SART001")
        condition, pipe_id, label = match.groups()
        if pipe_id not in self.interpreter.pipe_manager.pipes:
            raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", code="SART002")
        result = self.interpreter.evaluate_expression(condition, scope_name)
        if result:
            if label not in self.interpreter.labels:
                raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="SART003")
            self.interpreter.program_counter = self.interpreter.labels[label]
        log.debug(f"SART yürütüldü: {condition} -> {pipe_id} {label}")

    def handle_sart(self, command: str, scope_name: Optional[str] = None) -> None:
        """SART komutunu işler."""
        match = re.match(r"SART\s+(.+?)\s+ATLA\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
        if not match:
            raise PdsXSyntaxError("SART komutunda sözdizimi hatası, lütfen koşul ve etiket yazımını kontrol edin", code="SART001")
        condition, pipe_id, label = match.groups()
        if pipe_id not in self.interpreter.pipe_manager.pipes:
            raise PdsXRuntimeError(f"Boru hattı bulunamadı: {pipe_id}", code="SART002")
        try:
            result = self.interpreter.evaluate_expression(condition, scope_name)
            if result:
                if label not in self.interpreter.labels:
                    raise PdsXRuntimeError(f"Etiket bulunamadı: {label}", code="SART003")
                self.interpreter.program_counter = self.interpreter.labels[label]
                log.debug(f"SART yürütüldü: {condition} -> {pipe_id} {label}")
        except PdsXException as e:
            log.error(f"SART yürütme hatası: {str(e)}")
            raise

    def handle_type(self, command: str, scope_name: Optional[str] = None) -> None:
        """TYPE komutunu işler."""
        try:
            self.interpreter.memory_manager.parse_memory_command(command, scope_name)
            log.debug(f"TYPE yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Tür tanımlama hatası: {str(e)}")
            raise PdsXException(f"Tür tanımlama başarısız: {str(e)}", code="TYPE001")

    def handle_enum(self, command: str, scope_name: Optional[str] = None) -> None:
        """ENUM komutunu işler."""
        try:
            self.interpreter.memory_manager.parse_memory_command(command, scope_name)
            log.debug(f"ENUM yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Enum tanımlama hatası: {str(e)}")
            raise PdsXException(f"Enum tanımlama başarısız: {str(e)}", code="ENUM001")

    def handle_new(self, command: str, scope_name: Optional[str] = None) -> None:
        """NEW komutunu işler."""
        try:
            self.interpreter.memory_manager.parse_memory_command(command, scope_name)
            log.debug(f"NEW yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Yeni örnek oluşturma hatası: {str(e)}")
            raise PdsXException(f"Yeni örnek oluşturma başarısız: {str(e)}", code="NEW001")

    def handle_delete(self, command: str, scope_name: Optional[str] = None) -> None:
        """DELETE komutunu işler."""
        try:
            self.interpreter.memory_manager.parse_memory_command(command, scope_name)
            log.debug(f"DELETE yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Örnek silme hatası: {str(e)}")
            raise PdsXException(f"Örnek silme başarısız: {str(e)}", code="DELETE001")

    def handle_sizeof(self, command: str, scope_name: Optional[str] = None) -> None:
        """SIZEOF komutunu işler."""
        try:
            self.interpreter.memory_manager.parse_memory_command(command, scope_name)
            log.debug(f"SIZEOF yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Boyut alma hatası: {str(e)}")
            raise PdsXException(f"Boyut alma başarısız: {str(e)}", code="SIZEOF001")

    def handle_pipe(self, command: str, scope_name: Optional[str] = None) -> None:
        """PIPE komutlarını işler."""
        try:
            self.interpreter.pipe_manager.parse_pipe_command(command, self.interpreter)
            log.debug(f"PIPE yürütüldü: {command}")
        except PdsXPipeError as e:
            log.error(f"Boru hattı yürütme hatası: {str(e)}")
            raise

    def handle_reply(self, command: str, scope_name: Optional[str] = None) -> None:
        """REPLY komutlarını işler."""
        try:
            self.interpreter.reply_extension.parse_reply_command(command)
            log.debug(f"REPLY yürütüldü: {command}")
        except PdsXException as e:
            log.error(f"Yanıt yürütme hatası: {str(e)}")
            raise
    
    def execute(self, command: str, scope_name: Optional[str] = None) -> Optional[int]:
        """Komutu yürütür."""
        command_upper = command.strip().upper()
        if not command:
            return
        try:
            handler = self.command_handlers.get(command_upper.split()[0])
            if handler:
                handler(command, scope_name)
            else:
                raise PdsXSyntaxError(f"Bilinmeyen komut: {command}", code="GENERIC001")
        except PdsXException as e:
            self.interpreter.exception_manager.handle_error(e)
        log.debug(f"Komut yürütüldü: {command}")
        return self.interpreter.program_counter

    async def execute_async(self, command: str, scope_name: Optional[str] = None) -> Optional[int]:
        """Komutu asenkron yürütür."""
        command_upper = command.strip().upper()
        if not command:
            return
        try:
            handler = self.command_handlers.get(command_upper.split()[0])
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(command, scope_name)
                else:
                    handler(command, scope_name)
            else:
                raise PdsXSyntaxError(f"Bilinmeyen komut: {command}", code="GENERIC001")
        except PdsXException as e:
            await self.interpreter.exception_manager.handle_error(e)
        log.debug(f"Asenkron komut yürütüldü: {command}")
        return self.interpreter.program_counter

    def find_matching_else_or_end(self) -> int:
        """Eşleşen ELSE veya END IF bulur."""
        depth = 0
        for i in range(self.interpreter.program_counter + 1, len(self.interpreter.program)):
            cmd, _ = self.interpreter.program[i]
            cmd_upper = cmd.strip().upper()
            if cmd_upper.startswith("IF"):
                depth += 1
            elif cmd_upper.startswith("ELSE") and depth == 0:
                return i
            elif cmd_upper.startswith("END IF"):
                if depth == 0:
                    return i
                depth -= 1
        return len(self.interpreter.program)

    def find_matching_next(self) -> int:
        """Eşleşen NEXT bulur."""
        depth = 0
        for i in range(self.interpreter.program_counter + 1, len(self.interpreter.program)):
            cmd, _ = self.interpreter.program[i]
            cmd_upper = cmd.strip().upper()
            if cmd_upper.startswith("FOR") or cmd_upper.startswith("FOREACH"):
                depth += 1
            elif cmd_upper.startswith("NEXT"):
                if depth == 0:
                    return i
                depth -= 1
        return len(self.interpreter.program)

    def find_matching_wend(self) -> int:
        """Eşleşen WEND bulur."""
        depth = 0
        for i in range(self.interpreter.program_counter + 1, len(self.interpreter.program)):
            cmd, _ = self.interpreter.program[i]
            cmd_upper = cmd.strip().upper()
            if cmd_upper.startswith("WHILE"):
                depth += 1
            elif cmd_upper.startswith("WEND"):
                if depth == 0:
                    return i
if __name__ == "__main__":
    print("command_executor.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")

if __name__ == "__main__":
    print("command_executor.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")