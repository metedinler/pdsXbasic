#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Simple REPL
=================

Basit REPL implementasyonu - sadece minimal bağımlılıklarla
"""

import sys
import os
import importlib
from pathlib import Path

# Basit logging
def log_info(msg):
    print(f"[PDS-X] {msg}")

def log_error(msg):
    print(f"[PDS-X ERROR] {msg}")

# Simple module loader
class SimplePDSXLoader:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.loaded_modules = {}
        
    def load_module(self, name):
        """Basit modül yükleme"""
        try:
            if name in self.loaded_modules:
                return self.loaded_modules[name]
                
            # Try direct import first
            try:
                module = importlib.import_module(name)
                self.loaded_modules[name] = module
                return module
            except ImportError:
                pass
                
            # Try from file
            module_path = self.base_path / f"{name}.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    self.loaded_modules[name] = module
                    return module
                    
            return None
        except Exception as e:
            log_error(f"Modül yükleme hatası {name}: {e}")
            return None

def simple_repl():
    """PDS-X Simple REPL"""
    log_info("🔧 PDS-X Simple REPL Başlatılıyor...")
    
    loader = SimplePDSXLoader()
    
    # Basic environment
    global_vars = {
        '__name__': '__main__',
        '__doc__': 'PDS-X Simple REPL Environment',
        'load': loader.load_module,
        'loader': loader,
        'help': lambda: print("PDS-X Simple REPL\\nKomutlar: load('module_name'), exit(), help()"),
        'exit': lambda: sys.exit(0),
        'quit': lambda: sys.exit(0),
    }
    
    log_info("✅ REPL hazır - 'help()' yazın")
    log_info("Çıkmak için: exit(), quit() veya Ctrl+C")
    
    while True:
        try:
            user_input = input("PDS-X> ").strip()
            
            if not user_input:
                continue
                
            if user_input in ['exit', 'quit', 'q']:
                log_info("👋 REPL kapatılıyor...")
                break
                
            try:
                # Try eval first (for expressions)
                result = eval(user_input, global_vars)
                if result is not None:
                    print("=>", result)
            except SyntaxError:
                # If eval fails, try exec (for statements)
                exec(user_input, global_vars)
            except Exception as e:
                log_error(f"Hata: {e}")
                
        except KeyboardInterrupt:
            print("\\n[PDS-X] Ctrl+C - Çıkış...")
            break
        except EOFError:
            print("\\n[PDS-X] EOF - Çıkış...")
            break

if __name__ == "__main__":
    import importlib.util
    simple_repl()
