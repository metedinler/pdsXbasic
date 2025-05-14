# bytecode_compiler.py - PDS-X BASIC v14u Bytecode Derleyici Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import ast
import pickle
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from restrictedpython import compile_restricted, safe_globals, utility_builtins
from collections import defaultdict

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bytecode_compiler")

class PdsXException(Exception):
    pass

class BytecodeCompiler:
    def __init__(self):
        self.bytecode_cache = defaultdict(list)
        self.metadata = {"bytecode_compiler": {"version": "1.0.0", "dependencies": ["restrictedpython"]}}
        self.supported_commands = {
            "PRINT", "LET", "IF", "ELSE", "ENDIF", "FOR", "NEXT", "WHILE", "WEND",
            "DIM", "GOTO", "GOSUB", "RETURN", "END", "CLS", "BEEP", "OPEN", "CLOSE",
            "WRITE", "INPUT", "SEEK", "GET", "PUT", "LINE INPUT", "READ", "RESTORE",
            "SWAP", "KILL", "NAME", "MKDIR", "RMDIR", "CHDIR", "FILES", "CHAIN",
            "COMMON", "DECLARE", "DEF", "REDIM", "ERASE", "SELECT CASE", "CASE",
            "END SELECT", "CALL", "FUNCTION", "SUB", "EXIT", "TYPE", "STRUCT",
            "UNION", "ENUM", "NEW", "SET", "CONST", "INCLUDE", "IMPORT", "TRACE",
            "DEBUG", "OPTION", "LOCK", "UNLOCK", "THREAD", "ASYNC", "WAIT",
            "BITSET", "BITGET", "MEMCPY", "MEMSET", "PTR", "DEREF", "PIPE",
            "WINDOW", "BUTTON", "LABEL", "MENU", "SHOW", "NLP", "FACT", "RULE",
            "QUERY", "REGISTER PROLOG HANDLER", "UNREGISTER PROLOG HANDLER",
            "ON INTERRUPT", "END INTERRUPT", "SET SQL AUTO", "SQL RESULT TO ARRAY",
            "SQL RESULT TO STRUCT", "SQL RESULT TO DATAFRAME", "BEGIN TRANSACTION",
            "COMMIT", "ROLLBACK", "COMPILE", "SAVE BYTECODE", "LOAD BYTECODE"
        }

    def compile(self, code: str, module_name: str = "main") -> List[Tuple[str, str, Optional[bytes]]]:
        """
        BASIC kodunu bytecode'a derler.

        Args:
            code: Derlenecek BASIC kodu
            module_name: Modül adı

        Returns:
            Bytecode listesi: [(cmd_type, line, compiled_tree), ...]
        """
        try:
            bytecode = []
            lines = code.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.upper().startswith("'") or line.upper().startswith("REM "):
                    i += 1
                    continue
                
                cmd_upper = line.upper()
                cmd_type = "RAW"
                compiled_tree = None

                # BASIC komutlarını kontrol et
                for cmd in self.supported_commands:
                    if cmd_upper.startswith(cmd + " ") or cmd_upper == cmd:
                        cmd_type = cmd
                        break
                
                # Özel komutlar için ek ayrıştırma
                if cmd_type == "COMPILE":
                    match = re.match(r"COMPILE\s+(\w+)\s+(.+)?", line, re.IGNORECASE)
                    if match:
                        lang, rest = match.groups()
                        j = i + 1
                        block = []
                        while j < len(lines) and not lines[j].strip().upper().startswith(f"END {lang}"):
                            block.append(lines[j])
                            j += 1
                        output_name = re.match(r"AS\s+\"([^\"]+)\"", lines[j].strip(), re.IGNORECASE).group(1)
                        bytecode.append(("COMPILE", line, pickle.dumps({"language": lang, "code": "\n".join(block), "output_name": output_name})))
                        i = j + 1
                        continue
                elif cmd_type == "IF":
                    # IF bloğunu tam yakala
                    if_block = [line]
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().upper().startswith("ENDIF"):
                        if_block.append(lines[j])
                        if lines[j].strip().upper().startswith("ELSE"):
                            cmd_type = "IF_ELSE"
                        j += 1
                    if_block.append(lines[j])  # ENDIF
                    try:
                        compiled_tree = pickle.dumps(ast.parse("\n".join(if_block), mode="exec"))
                    except SyntaxError:
                        compiled_tree = None
                    bytecode.append((cmd_type, "\n".join(if_block), compiled_tree))
                    i = j + 1
                    continue
                elif cmd_type == "FOR":
                    # FOR bloğunu tam yakala
                    for_block = [line]
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().upper().startswith("NEXT"):
                        for_block.append(lines[j])
                        j += 1
                    for_block.append(lines[j])  # NEXT
                    try:
                        compiled_tree = pickle.dumps(ast.parse("\n".join(for_block), mode="exec"))
                    except SyntaxError:
                        compiled_tree = None
                    bytecode.append((cmd_type, "\n".join(for_block), compiled_tree))
                    i = j + 1
                    continue
                elif cmd_type == "WHILE":
                    # WHILE bloğunu tam yakala
                    while_block = [line]
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().upper().startswith("WEND"):
                        while_block.append(lines[j])
                        j += 1
                    while_block.append(lines[j])  # WEND
                    try:
                        compiled_tree = pickle.dumps(ast.parse("\n".join(while_block), mode="exec"))
                    except SyntaxError:
                        compiled_tree = None
                    bytecode.append((cmd_type, "\n".join(while_block), compiled_tree))
                    i = j + 1
                    continue
                elif cmd_type == "FUNCTION" or cmd_type == "SUB":
                    # FUNCTION/SUB bloğunu tam yakala
                    block = [line]
                    j = i + 1
                    end_cmd = f"END {cmd_type}"
                    while j < len(lines) and not lines[j].strip().upper().startswith(end_cmd):
                        block.append(lines[j])
                        j += 1
                    block.append(lines[j])  # END FUNCTION/SUB
                    try:
                        compiled_tree = pickle.dumps(ast.parse("\n".join(block), mode="exec"))
                    except SyntaxError:
                        compiled_tree = None
                    bytecode.append((cmd_type, "\n".join(block), compiled_tree))
                    i = j + 1
                    continue
                
                # Tek satırlık komutlar için güvenli derleme
                try:
                    # restrictedpython ile güvenli derleme
                    byte_code = compile_restricted(line, filename=f"<{module_name}>", mode="exec")
                    compiled_tree = pickle.dumps(byte_code)
                    cmd_type = "EXEC"
                except SyntaxError as e:
                    log.warning(f"Derleme hatası, RAW olarak kaydediliyor: {line}, {str(e)}")
                    compiled_tree = None
                
                bytecode.append((cmd_type, line, compiled_tree))
                i += 1
            
            self.bytecode_cache[module_name] = bytecode
            log.debug(f"Bytecode derleme tamamlandı: {module_name}, {len(bytecode)} komut")
            return bytecode
        except Exception as e:
            log.error(f"Bytecode derleme hatası: {str(e)}")
            raise PdsXException(f"Bytecode derleme hatası: {str(e)}")

    def optimize(self, bytecode: List[Tuple[str, str, Optional[bytes]]]) -> List[Tuple[str, str, Optional[bytes]]]:
        """
        Bytecode'u optimize eder (gereksiz komutları kaldırır, sabitleri katlar).
        """
        try:
            optimized = []
            i = 0
            while i < len(bytecode):
                cmd_type, line, tree = bytecode[i]
                
                # Boş veya yorum satırlarını atla
                if cmd_type == "RAW" and (not line.strip() or line.strip().upper().startswith(("'", "REM "))):
                    i += 1
                    continue
                
                # Sabit ifadeleri katla (örneğin, LET x = 2 + 3 -> LET x = 5)
                if cmd_type == "LET":
                    match = re.match(r"LET\s+(\w+)\s*=\s*(.+)", line, re.IGNORECASE)
                    if match:
                        var_name, expr = match.groups()
                        try:
                            value = eval(expr, safe_globals)
                            optimized.append(("LET", f"LET {var_name} = {value}", None))
                            i += 1
                            continue
                        except:
                            pass
                
                # Ardışık PRINT'leri birleştir
                if cmd_type == "PRINT" and i + 1 < len(bytecode) and bytecode[i + 1][0] == "PRINT":
                    combined = [line]
                    j = i + 1
                    while j < len(bytecode) and bytecode[j][0] == "PRINT":
                        combined.append(bytecode[j][1])
                        j += 1
                    optimized.append(("PRINT", "; ".join(combined), None))
                    i = j
                    continue
                
                optimized.append((cmd_type, line, tree))
                i += 1
            
            log.debug(f"Bytecode optimizasyonu tamamlandı: {len(bytecode)} -> {len(optimized)} komut")
            return optimized
        except Exception as e:
            log.error(f"Bytecode optimizasyon hatası: {str(e)}")
            raise PdsXException(f"Bytecode optimizasyon hatası: {str(e)}")

    def validate(self, bytecode: List[Tuple[str, str, Optional[bytes]]]) -> bool:
        """
        Bytecode'un geçerliliğini kontrol eder.
        """
        try:
            stack = []
            for cmd_type, line, _ in bytecode:
                if cmd_type in ("IF", "FOR", "WHILE", "FUNCTION", "SUB"):
                    stack.append(cmd_type)
                elif cmd_type == "ENDIF" and stack and stack[-1] == "IF":
                    stack.pop()
                elif cmd_type == "NEXT" and stack and stack[-1] == "FOR":
                    stack.pop()
                elif cmd_type == "WEND" and stack and stack[-1] == "WHILE":
                    stack.pop()
                elif cmd_type in ("END FUNCTION", "END SUB") and stack and stack[-1] in ("FUNCTION", "SUB"):
                    stack.pop()
            
            if stack:
                log.error(f"Bytecode geçersiz: Eşleşmeyen bloklar {stack}")
                return False
            
            log.debug("Bytecode doğrulama başarılı")
            return True
        except Exception as e:
            log.error(f"Bytecode doğrulama hatası: {str(e)}")
            raise PdsXException(f"Bytecode doğrulama hatası: {str(e)}")

    def get_bytecode_info(self, bytecode: List[Tuple[str, str, Optional[bytes]]]) -> Dict:
        """
        Bytecode hakkında bilgi döndürür.
        """
        try:
            info = {
                "command_count": len(bytecode),
                "command_types": list(set(cmd_type for cmd_type, _, _ in bytecode)),
                "size": sum(len(line) for _, line, _ in bytecode),
                "has_compiled": sum(1 for _, _, tree in bytecode if tree is not None)
            }
            log.debug(f"Bytecode bilgisi alındı: {info}")
            return info
        except Exception as e:
            log.error(f"Bytecode bilgi alma hatası: {str(e)}")
            raise PdsXException(f"Bytecode bilgi alma hatası: {str(e)}")

if __name__ == "__main__":
    print("bytecode_compiler.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")