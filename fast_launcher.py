#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X BASIC v14u - Lightweight Fast Launcher
============================================

Bu dosya sadece REPL'i hızlı başlatmak için minimal import'larla yazılmıştır.
Büyük modüller lazy-loading ile ihtiyaç olunca yüklenir.
"""

import sys
import os
import importlib

print(f"[PDS-X] 🚀 PDS-X BASIC v14u - Fast Launcher")
print(f"[PDS-X] Python: {sys.version_info.major}.{sys.version_info.minor}")

def fast_repl():
    """Hızlı REPL - minimal import'larla"""
    print("""
[PDS-X] ⚡ Fast REPL Mode
========================
Minimal imports, maximum speed!
Type 'help()' for commands, 'exit' to quit.
""")
    
    # Global environment
    repl_globals = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
        'help': lambda: print("""
PDS-X Fast REPL Commands:
  help()           - Show this help
  load_module(name) - Load a PDS-X module
  list_files()     - List Python files
  run_file(name)   - Run a Python file
  system(cmd)      - Run system command
  exit             - Exit REPL
  
Python code works too:
  print("Hello!")
  x = 1 + 1
  import math; print(math.pi)
"""),
        'load_module': lambda name: __import__(name),
        'list_files': lambda: [f for f in os.listdir('.') if f.endswith('.py')][:20],
        'run_file': lambda name: exec(open(name).read()),
        'system': lambda cmd: os.system(cmd),
        'exit': lambda: sys.exit(0),
        'quit': lambda: sys.exit(0),
    }
    
    while True:
        try:
            user_input = input("PDS-X⚡> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ('exit', 'quit'):
                print("[PDS-X] 👋 Fast REPL shutting down...")
                break
                
            # Special commands
            if user_input == 'load_full_system':
                print("[PDS-X] 🔄 Loading full PDS-X system...")
                try:
                    # Lazy load büyük sistem
                    from pdsXuv14 import PdsXv14uInterpreter
                    interpreter = PdsXv14uInterpreter()
                    repl_globals['interpreter'] = interpreter
                    print("[PDS-X] ✅ Full system loaded! Use 'interpreter' variable.")
                except Exception as e:
                    print(f"[PDS-X] ❌ Load error: {e}")
                continue
                
            if user_input == 'load_auto_importer':
                print("[PDS-X] 🔄 Loading AutoImporter...")
                try:
                    from auto_importer import AutoImporter
                    ai = AutoImporter(mode="NORMAL")
                    repl_globals['auto_importer'] = ai
                    print("[PDS-X] ✅ AutoImporter loaded! Use 'auto_importer' variable.")
                except Exception as e:
                    print(f"[PDS-X] ❌ Load error: {e}")
                continue
                
            try:
                # Try eval first (for expressions)
                result = eval(user_input, repl_globals)
                if result is not None:
                    print("=>", result)
            except SyntaxError:
                # If eval fails, try exec (for statements)
                exec(user_input, repl_globals)
            except Exception as e:
                print(f"[ERROR] {e}")
                
        except KeyboardInterrupt:
            print("\\n[PDS-X] Ctrl+C - Exiting...")
            break
        except EOFError:
            print("\\n[PDS-X] EOF - Exiting...")
            break

if __name__ == "__main__":
    fast_repl()
