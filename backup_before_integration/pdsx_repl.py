#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X REPL Module
=================

PDS-X için REPL implementasyonu
"""

import sys
import os
import importlib
import importlib.util
from pathlib import Path

def log_info(msg):
    """Log info mesajı"""
    print(f"[PDS-X REPL] {msg}")

def log_error(msg):
    """Log error mesajı"""
    print(f"[PDS-X REPL ERROR] {msg}")

class PDSXREPLEnvironment:
    """PDS-X REPL Ortamı"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.modules = {}
        self.variables = {}
        
    def load_module(self, name):
        """Modül yükle"""
        try:
            if name in self.modules:
                return self.modules[name]
                
            # Try direct import
            try:
                module = importlib.import_module(name)
                self.modules[name] = module
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
                    self.modules[name] = module
                    return module
                    
            log_error(f"Modül bulunamadı: {name}")
            return None
            
        except Exception as e:
            log_error(f"Modül yükleme hatası {name}: {e}")
            return None
            
    def get_help(self):
        """Yardım bilgisi"""
        return """
PDS-X REPL - Yardım
===================

Kullanılabilir komutlar:
  help()          - Bu yardımı göster
  load('module')  - Modül yükle
  modules()       - Yüklü modülleri listele
  vars()          - Değişkenleri listele
  clear()         - Ekranı temizle
  exit()          - REPL'den çık
  quit()          - REPL'den çık

Python ifadeleri ve komutları:
  1 + 1           - Hesaplama
  x = 10          - Değişken atama
  print("hello")  - Yazdırma
  
Modül kullanımı:
  core = load('core2_6')
  mem = load('memory_manager')
"""

    def list_modules(self):
        """Yüklü modülleri listele"""
        return list(self.modules.keys())
        
    def clear_screen(self):
        """Ekranı temizle"""
        os.system('cls' if os.name == 'nt' else 'clear')

def start_pdsx_repl():
    """PDS-X REPL'ini başlat"""
    log_info("🚀 PDS-X REPL Başlatılıyor...")
    
    env = PDSXREPLEnvironment()
    
    # REPL global environment
    repl_globals = {
        '__name__': '__main__',
        '__doc__': 'PDS-X REPL Environment',
        'load': env.load_module,
        'modules': env.list_modules,
        'help': lambda: print(env.get_help()),
        'clear': env.clear_screen,
        'exit': lambda: sys.exit(0),
        'quit': lambda: sys.exit(0),
        'vars': lambda: list(repl_globals.keys()),
    }
    
    log_info("✅ REPL hazır - 'help()' yazarak yardım alabilirsiniz")
    log_info("Çıkmak için: exit(), quit() veya Ctrl+C")
    
    while True:
        try:
            user_input = input("PDS-X> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                log_info("👋 REPL kapatılıyor...")
                break
                
            try:
                # Try eval first (for expressions)
                result = eval(user_input, repl_globals)
                if result is not None:
                    print("=>", result)
            except SyntaxError:
                # If eval fails, try exec (for statements)
                exec(user_input, repl_globals)
            except Exception as e:
                log_error(f"Hata: {e}")
                
        except KeyboardInterrupt:
            print("\\n[PDS-X] Ctrl+C ile çıkış...")
            break
        except EOFError:
            print("\\n[PDS-X] EOF ile çıkış...")
            break

if __name__ == "__main__":
    start_pdsx_repl()
