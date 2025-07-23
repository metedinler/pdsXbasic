#!/usr/bin/env python3
"""
PDS-X REPL Multi-Line Program Demo
Çok satırlı program özelliklerini gösterir
"""

import sys
import os

# PDS-X modüllerini import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from program_manager import MultiLineProgramManager


def demo_multiline_programs():
    """Multi-line program özelliklerini demo et"""
    print("🚀 PDS-X Multi-Line Program Demo")
    print("=" * 50)
    
    manager = MultiLineProgramManager()
    
    # Demo 1: Python Hello World
    print("\n📝 Demo 1: Python Hello World")
    print("-" * 30)
    
    manager.start_program("PROGRAM hello.py")
    manager.add_line("print('Merhaba PDS-X!')")
    manager.add_line("print('Bu bir multi-line program!')")
    manager.add_line("print('Python 🐍 ile yazıldı')")
    manager.end_program()
    
    # Demo 2: JavaScript Console Log
    print("\n📝 Demo 2: JavaScript Console")
    print("-" * 30)
    
    manager.start_program("PROGRAM app.js")
    manager.add_line("console.log('JavaScript ile Merhaba!');")
    manager.add_line("let version = 'PDS-X v14u';")
    manager.add_line("console.log('Versiyon:', version);")
    manager.end_program()
    
    # Demo 3: PDS-X BASIC
    print("\n📝 Demo 3: PDS-X BASIC")
    print("-" * 30)
    
    manager.start_program("PROGRAM math.basx")
    manager.add_line("10 LET A = 5")
    manager.add_line("20 LET B = 3")
    manager.add_line("30 LET C = A + B")
    manager.add_line("40 PRINT \"Sonuç:\", C")
    manager.end_program()
    
    # Demo 4: Program listeleme
    print("\n📋 Demo 4: Program Listesi")
    print("-" * 30)
    
    manager.list_programs()
    
    # Demo 5: Program içerik gösterme
    print("\n📄 Demo 5: Program İçeriği")
    print("-" * 30)
    
    manager.show_program("hello")
    
    # Demo 6: Python program çalıştırma
    print("\n⚡ Demo 6: Program Çalıştırma")
    print("-" * 30)
    
    print("Python programı çalıştırılıyor...")
    manager.run_program("hello")
    
    print("\n" + "=" * 50)
    print("🎉 Demo Tamamlandı!")
    print("💡 REPL'de şu komutları kullanabilirsiniz:")
    print("   • PROGRAM <name>.<ext>  - Program yazma başlat")
    print("   • END PROGRAM           - Program yazma bitir")
    print("   • LIST                  - Tüm programları listele")
    print("   • LIST <name>           - Program içeriğini göster")
    print("   • RUN PROGRAM <name>    - Program çalıştır")
    

if __name__ == "__main__":
    demo_multiline_programs()
