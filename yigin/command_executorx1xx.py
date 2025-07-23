# command_executorx1.py - PDS-X BASIC v15 Komut Yürütme Motoru
# Version: 2.0.0 (Rebuilt with unified exception system)
# Date: July 21, 2025
# Author: Claude 3.5 Sonnet (Rebuilt by metedinler directive)

import re
import asyncio
import logging
from typing import Dict, Optional, Any

# PDS-X Unified Exception System v2.0.0
from pdsx_unified_exception import (
    PdsXException, PdsXSyntaxError, PdsXRuntimeError, PdsXPipeError,
    PdsXReplyError, PdsXDatabaseError, PdsXLogicError, PdsXNetworkError,
    PdsXMLError, PdsXNLPError, PdsXModuleValidatorError, PdsXModuleManagerError,
    PdsXMultithreadingError, PdsXBusError, PdsXLowLevelError
)

logging.basicConfig(filename='pdsxe_errors.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("command_executor_v2")

class CommandExecutor:
    """PDS-X BASIC v15 komut yürütme motoru - Unified Exception System v2.0.0"""
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.command_handlers: Dict[str, Any] = {
            # Temel komutlar
            "PRINT": self.handle_print,
            "LET": self.handle_let,
            "DIM": self.handle_dim,
            "IF": self.handle_if,
            "FOR": self.handle_for,
            "WHILE": self.handle_while,
            "END": self.handle_end,
            
            # OOP komutları
            "CLASS": self.handle_class,
            "YAPI": self.handle_yapi,
            
            # Control flow
            "GOTO": self.handle_goto,
            "GOSUB": self.handle_gosub,
            "RETURN": self.handle_return,
            
            # Data handling
            "DATA": self.handle_data,
            "READ": self.handle_read,
            
            # I/O
            "INPUT": self.handle_input,
            
            # Functions
            "DEF": self.handle_def,
            "FUNCTION": self.handle_function,
            "SUB": self.handle_sub,
            "CALL": self.handle_call,
            
            # Database komutları
            "DB CONNECT": self.handle_db_connect,
            "DB QUERY": self.handle_db_query,
            
            # Pipeline komutları
            "PIPE DEFINE": self.handle_pipe,
            "PIPE START": self.handle_pipe,
            
            # Bus komutları  
            "BUS DEFINE": self.handle_bus,
            "BUS PUBLISH": self.handle_bus,
            "BUS SUBSCRIBE": self.handle_bus,
            
            # REPL komutları
            "REPLY": self.handle_reply,
            
            # LibX komutları
            "LIBX": self.handle_libx,
            
            # Error handling
            "ON ERROR": self.handle_on_error,
            "TRY": self.handle_try,
            
            # System
            "EXIT": self.handle_exit,
        }
    
    def execute_command(self, command: str, scope_name: Optional[str] = None) -> Any:
        """Ana komut çalıştırma metodu"""
        try:
            if not command or not command.strip():
                return None
            
            command = command.strip()
            
            # Komut tipini belirle
            command_parts = command.split()
            if not command_parts:
                return None
            
            command_type = command_parts[0].upper()
            
            # Multi-word komutları handle et
            if len(command_parts) > 1:
                two_word = f"{command_parts[0]} {command_parts[1]}".upper()
                if two_word in self.command_handlers:
                    command_type = two_word
            
            # Handler'ı bul ve çalıştır
            if command_type in self.command_handlers:
                handler = self.command_handlers[command_type]
                return handler(command, scope_name)
            else:
                # Bilinmeyen komut - interpreter'a gönder
                if hasattr(self.interpreter, 'execute_command_async'):
                    import asyncio
                    return asyncio.run(self.interpreter.execute_command_async(command, scope_name))
                else:
                    raise PdsXSyntaxError(f"Bilinmeyen komut: {command_type}", code="CMD001")
        
        except PdsXException:
            raise  # PDS-X exception'ları olduğu gibi tekrar fırlat
        except Exception as e:
            raise PdsXRuntimeError(f"Komut çalıştırma hatası: {str(e)}", code="CMD002", context={"command": command})
    
    # ============================================================================
    # TEMEL KOMUT HANDLER'LARI
    # ============================================================================
    
    def handle_print(self, command: str, scope_name: Optional[str] = None) -> None:
        """PRINT komutunu işler"""
        try:
            match = re.match(r"PRINT\s+(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("PRINT komutunda sözdizimi hatası", code="PRINT001")
            
            expr = match.group(1)
            # Basit string handling
            if expr.startswith('"') and expr.endswith('"'):
                result = expr[1:-1]  # String literal
            else:
                # Expression evaluation (basitleştirilmiş)
                result = expr
            
            print(result)
            log.debug(f"PRINT yürütüldü: {expr} -> {result}")
            
        except PdsXException:
            raise
        except Exception as e:
            raise PdsXRuntimeError(f"PRINT yürütme hatası: {str(e)}", code="PRINT002")
    
    def handle_let(self, command: str, scope_name: Optional[str] = None) -> None:
        """LET komutunu işler"""
        try:
            match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("LET komutunda sözdizimi hatası", code="LET001")
            
            var_name, expr = match.groups()
            
            # Basit value assignment
            if expr.startswith('"') and expr.endswith('"'):
                value = expr[1:-1]  # String literal
            elif expr.isdigit():
                value = int(expr)  # Integer
            elif '.' in expr and expr.replace('.', '').isdigit():
                value = float(expr)  # Float
            else:
                value = expr  # Expression as string
            
            # Store in interpreter variables
            if hasattr(self.interpreter, 'global_vars'):
                self.interpreter.global_vars[var_name] = value
            
            log.debug(f"LET yürütüldü: {var_name} = {value}")
            
        except PdsXException:
            raise
        except Exception as e:
            raise PdsXRuntimeError(f"LET yürütme hatası: {str(e)}", code="LET002")
    
    def handle_dim(self, command: str, scope_name: Optional[str] = None) -> None:
        """DIM komutunu işler"""
        try:
            match = re.match(r"DIM\s+(\w+)(?:\s*\[(.*?)\])?\s*(?:AS\s+(\w+))?", command, re.IGNORECASE)
            if not match:
                raise PdsXSyntaxError("DIM komutunda sözdizimi hatası", code="DIM001")
            
            var_name, dimensions, type_name = match.groups()
            
            # Create variable with type
            if type_name:
                if type_name.upper() == "STRING":
                    value = ""
                elif type_name.upper() in ["INTEGER", "INT"]:
                    value = 0
                elif type_name.upper() in ["FLOAT", "SINGLE", "DOUBLE"]:
                    value = 0.0
                else:
                    value = None
            else:
                value = None
            
            # Store in interpreter
            if hasattr(self.interpreter, 'global_vars'):
                self.interpreter.global_vars[var_name] = value
            
            log.debug(f"DIM yürütüldü: {var_name} AS {type_name}")
            
        except PdsXException:
            raise
        except Exception as e:
            raise PdsXRuntimeError(f"DIM yürütme hatası: {str(e)}", code="DIM002")
    
    # ============================================================================
    # STUB HANDLERS (Geliştirilecek)
    # ============================================================================
    
    def handle_if(self, command: str, scope_name: Optional[str] = None) -> None:
        """IF komutunu işler - Stub"""
        log.debug(f"IF komut işlendi: {command}")
    
    def handle_for(self, command: str, scope_name: Optional[str] = None) -> None:
        """FOR komutunu işler - Stub"""
        log.debug(f"FOR komut işlendi: {command}")
    
    def handle_while(self, command: str, scope_name: Optional[str] = None) -> None:
        """WHILE komutunu işler - Stub"""
        log.debug(f"WHILE komut işlendi: {command}")
    
    def handle_end(self, command: str, scope_name: Optional[str] = None) -> None:
        """END komutunu işler - Stub"""
        log.debug(f"END komut işlendi: {command}")
    
    def handle_class(self, command: str, scope_name: Optional[str] = None) -> None:
        """CLASS komutunu işler - Stub"""
        log.debug(f"CLASS komut işlendi: {command}")
    
    def handle_yapi(self, command: str, scope_name: Optional[str] = None) -> None:
        """YAPI komutunu işler - Stub"""
        log.debug(f"YAPI komut işlendi: {command}")
    
    def handle_goto(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOTO komutunu işler - Stub"""
        log.debug(f"GOTO komut işlendi: {command}")
    
    def handle_gosub(self, command: str, scope_name: Optional[str] = None) -> None:
        """GOSUB komutunu işler - Stub"""
        log.debug(f"GOSUB komut işlendi: {command}")
    
    def handle_return(self, command: str, scope_name: Optional[str] = None) -> None:
        """RETURN komutunu işler - Stub"""
        log.debug(f"RETURN komut işlendi: {command}")
    
    def handle_data(self, command: str, scope_name: Optional[str] = None) -> None:
        """DATA komutunu işler - Stub"""
        log.debug(f"DATA komut işlendi: {command}")
    
    def handle_read(self, command: str, scope_name: Optional[str] = None) -> None:
        """READ komutunu işler - Stub"""
        log.debug(f"READ komut işlendi: {command}")
    
    def handle_input(self, command: str, scope_name: Optional[str] = None) -> None:
        """INPUT komutunu işler - Stub"""
        log.debug(f"INPUT komut işlendi: {command}")
    
    def handle_def(self, command: str, scope_name: Optional[str] = None) -> None:
        """DEF komutunu işler - Stub"""
        log.debug(f"DEF komut işlendi: {command}")
    
    def handle_function(self, command: str, scope_name: Optional[str] = None) -> None:
        """FUNCTION komutunu işler - Stub"""
        log.debug(f"FUNCTION komut işlendi: {command}")
    
    def handle_sub(self, command: str, scope_name: Optional[str] = None) -> None:
        """SUB komutunu işler - Stub"""
        log.debug(f"SUB komut işlendi: {command}")
    
    def handle_call(self, command: str, scope_name: Optional[str] = None) -> None:
        """CALL komutunu işler - Stub"""
        log.debug(f"CALL komut işlendi: {command}")
    
    def handle_db_connect(self, command: str, scope_name: Optional[str] = None) -> None:
        """DB CONNECT komutunu işler - Stub"""
        log.debug(f"DB CONNECT komut işlendi: {command}")
    
    def handle_db_query(self, command: str, scope_name: Optional[str] = None) -> None:
        """DB QUERY komutunu işler - Stub"""
        log.debug(f"DB QUERY komut işlendi: {command}")
    
    def handle_pipe(self, command: str, scope_name: Optional[str] = None) -> None:
        """PIPE komutlarını işler - Stub"""
        log.debug(f"PIPE komut işlendi: {command}")
    
    def handle_bus(self, command: str, scope_name: Optional[str] = None) -> None:
        """BUS komutlarını işler - Stub"""
        log.debug(f"BUS komut işlendi: {command}")
    
    def handle_reply(self, command: str, scope_name: Optional[str] = None) -> None:
        """REPLY komutunu işler - Stub"""
        log.debug(f"REPLY komut işlendi: {command}")
    
    def handle_libx(self, command: str, scope_name: Optional[str] = None) -> None:
        """LIBX komutlarını işler - Stub"""
        log.debug(f"LIBX komut işlendi: {command}")
    
    def handle_on_error(self, command: str, scope_name: Optional[str] = None) -> None:
        """ON ERROR komutunu işler - Stub"""
        log.debug(f"ON ERROR komut işlendi: {command}")
    
    def handle_try(self, command: str, scope_name: Optional[str] = None) -> None:
        """TRY komutunu işler - Stub"""
        log.debug(f"TRY komut işlendi: {command}")
    
    def handle_exit(self, command: str, scope_name: Optional[str] = None) -> None:
        """EXIT komutunu işler - Stub"""
        log.debug(f"EXIT komut işlendi: {command}")

# Module initialization
log.info("PDS-X CommandExecutor v2.0.0 initialized with Unified Exception System")
print("[PDS-X] ✅ CommandExecutor v2.0.0 loaded with unified exceptions")