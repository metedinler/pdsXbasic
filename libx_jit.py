# libx_jit.py - PDS-X BASIC v14u JIT ve Inline Derleme Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import subprocess
import ctypes
import tempfile
import os
import platform
import pickle
import ast
import logging
from typing import Callable, Optional
from restrictedpython import compile_restricted, safe_globals, utility_builtins
from pathlib import Path

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_jit")

class PdsXException(Exception):
    pass

class LibXJIT:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.compiler = "gcc" if platform.system() != "Windows" else "clang"
        self.asm_assembler = "as"
        self.linker = "ld" if platform.system() != "Windows" else self.compiler
        self.temp_files = []
        self.supported_languages = ["ASM", "C", "JIT"]
        
    def compile(self, language: str, code: str, output_name: str) -> Optional[Callable]:
        """
        Belirtilen dilde kodu derler ve çalıştırılabilir bir fonksiyon döndürür.
        
        Args:
            language: Derleme dili ("ASM", "C", "JIT")
            code: Derlenecek kod
            output_name: Çıktı dosya adı (.so, .dll, .bc)
        
        Returns:
            Çalıştırılabilir fonksiyon veya None
        """
        language = language.upper()
        if language not in self.supported_languages:
            raise PdsXException(f"Desteklenmeyen dil: {language}")
        
        try:
            if language == "ASM":
                return self.compile_asm(code, output_name)
            elif language == "C":
                return self.compile_c(code, output_name)
            elif language == "JIT":
                return self.compile_jit(code, output_name)
        except Exception as e:
            log.error(f"Derleme hatası: {language}, {str(e)}")
            raise PdsXException(f"Derleme hatası: {str(e)}")
        finally:
            self.cleanup_temp_files()

    def compile_asm(self, code: str, output_name: str) -> Callable:
        """
        Assembly kodunu derler ve çalıştırılabilir bir fonksiyon döndürür.
        """
        asm_path = tempfile.mktemp(suffix=".s")
        obj_path = tempfile.mktemp(suffix=".o")
        so_path = output_name if output_name.endswith((".so", ".dll")) else output_name + (".so" if platform.system() != "Windows" else ".dll")
        self.temp_files.extend([asm_path, obj_path])
        
        try:
            # Assembly kodunu dosyaya yaz
            with open(asm_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Assembly derleme
            subprocess.run([self.asm_assembler, asm_path, "-o", obj_path], check=True, capture_output=True, text=True)
            
            # Bağlama
            ld_cmd = [self.linker, "-shared", obj_path, "-o", so_path]
            if platform.system() == "Windows":
                ld_cmd = [self.compiler, "-shared", obj_path, "-o", so_path]
            subprocess.run(ld_cmd, check=True, capture_output=True, text=True)
            
            # Dinamik kütüphane yükleme
            lib = ctypes.CDLL(so_path)
            if hasattr(lib, "run"):
                return lib.run
            raise PdsXException("Derlenen ASM kodunda 'run' fonksiyonu bulunamadı")
        
        except subprocess.CalledProcessError as e:
            log.error(f"ASM derleme hatası: {e.stderr}")
            raise PdsXException(f"ASM derleme hatası: {e.stderr}")
        
    def compile_c(self, code: str, output_name: str) -> Callable:
        """
        C kodunu derler ve çalıştırılabilir bir fonksiyon döndürür.
        """
        c_path = tempfile.mktemp(suffix=".c")
        so_path = output_name if output_name.endswith((".so", ".dll")) else output_name + (".so" if platform.system() != "Windows" else ".dll")
        self.temp_files.append(c_path)
        
        try:
            # C kodunu dosyaya yaz
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # C derleme
            cmd = [self.compiler, "-shared", "-fPIC", c_path, "-o", so_path]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Dinamik kütüphane yükleme
            lib = ctypes.CDLL(so_path)
            if hasattr(lib, "run"):
                return lib.run
            raise PdsXException("Derlenen C kodunda 'run' fonksiyonu bulunamadı")
        
        except subprocess.CalledProcessError as e:
            log.error(f"C derleme hatası: {e.stderr}")
            raise PdsXException(f"C derleme hatası: {e.stderr}")

    def compile_jit(self, code: str, output_name: str) -> Callable:
        """
        JIT modunda kodu derler ve çalıştırılabilir bir fonksiyon döndürür.
        """
        try:
            # Güvenli derleme için restrictedpython kullanımı
            byte_code = compile_restricted(
                code,
                filename="<inline>",
                mode="exec",
                policy=None
            )
            
            # Bytecode'u dosyaya kaydet (isteğe bağlı)
            if output_name:
                with open(output_name if output_name.endswith(".bc") else output_name + ".bc", "wb") as f:
                    pickle.dump(byte_code, f)
            
            # Güvenli yürütme ortamı
            safe_env = safe_globals.copy()
            safe_env.update(utility_builtins)
            safe_env.update(self.interpreter.current_scope())
            safe_env.update(self.interpreter.function_table)
            
            # Kodu yürüt
            exec(byte_code, safe_env)
            
            # Çalıştırılabilir fonksiyon döndür
            return lambda: exec(byte_code, safe_env)
        
        except SyntaxError as e:
            log.error(f"JIT derleme hatası: {str(e)}")
            raise PdsXException(f"JIT derleme hatası: {str(e)}")
        except Exception as e:
            log.error(f"JIT yürütme hatası: {str(e)}")
            raise PdsXException(f"JIT yürütme hatası: {str(e)}")

    def cleanup_temp_files(self) -> None:
        """Geçici dosyaları temizler."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError as e:
                log.warning(f"Geçici dosya silinemedi: {temp_file}, {str(e)}")
        self.temp_files.clear()

if __name__ == "__main__":
    print("libx_jit.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")