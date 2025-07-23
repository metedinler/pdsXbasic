#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Clean Launcher
====================

Sadece pdsx_repl kullanan temiz launcher.
AutoImporter ve reply_extension çakışması çözüldü.
"""

import sys
import os
from pathlib import Path

def main():
    """Clean PDS-X launcher - sadece pdsx_repl"""
    print("[PDS-X] 🚀 Clean PDS-X Launcher")
    print("[PDS-X] Python:", sys.version)
    print("[PDS-X] Working directory:", os.getcwd())
    
    # AutoImporter'ı kontrollü şekilde yükle
    auto_importer_loaded = False
    try:
        from auto_importer import AutoImporter
        print("[PDS-X] ✅ AutoImporter yüklendi")
        auto_importer_loaded = True
    except ImportError as e:
        print(f"[PDS-X] ⚠️ AutoImporter yüklenemedi: {e}")
        auto_importer_loaded = False
    
    # Sadece pdsx_repl kullan
    try:
        from pdsx_repl import start_pdsx_repl
        print("[PDS-X] ✅ pdsx_repl yüklendi")
        
        # REPL'i başlat
        print("[PDS-X] 🚀 REPL başlatılıyor...")
        start_pdsx_repl()
        
    except ImportError as e:
        print(f"[PDS-X] ❌ pdsx_repl bulunamadı: {e}")
        print("[PDS-X] 🔧 Minimal REPL başlatılıyor...")
        
        # Minimal REPL
        print("Minimal PDS-X REPL - exit ile çıkış")
        while True:
            try:
                cmd = input("PDS-X> ").strip()
                if cmd.lower() in ('exit', 'quit'):
                    break
                elif cmd == 'help()':
                    print("Komutlar: exit, quit, help()")
                else:
                    try:
                        result = eval(cmd)
                        if result is not None:
                            print("=>", result)
                    except SyntaxError:
                        exec(cmd)
                    except Exception as e:
                        print("Hata:", e)
            except (KeyboardInterrupt, EOFError):
                break
        
        print("[PDS-X] 👋 REPL kapatıldı")

if __name__ == "__main__":
    main()
