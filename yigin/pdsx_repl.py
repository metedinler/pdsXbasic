# PDS-X REPL - Interactive Mode
# ===============================
# PDS-X BASIC v14u İnteraktif Çok-Satırlı REPL Ortamı
# Bu modül --interactive anahtarı ile çağrılır

import os
import sys
import re
import time
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque

# AutoImporter entegrasyonu - TAMAMEN OTOMATİK
try:
    from auto_importer import AutoImporter, EnvManager
    print("🔧 PDS-X AutoImporter REPL desteği yüklendi")
    AUTO_IMPORTER_AVAILABLE = True
    
    # Wrapper functions için geçici çözümler
    def auto_setup_for_repl():
        return True
    
    def check_requirements_status():
        return {"status": "ok", "missing": []}
        
except ImportError:
    print("⚠️  AutoImporter bulunamadı, temel REPL modu")
    AUTO_IMPORTER_AVAILABLE = False
    AutoImporter = None
    
    def auto_setup_for_repl():
        return False
    
    def check_requirements_status():
        return {"status": "error", "missing": ["auto_importer"]}

# PDS-X Core imports (dinamik)
try:
    from pdsXuv14 import PDSXInterpreter
    PDSX_AVAILABLE = True
except ImportError:
    print("⚠️  PDS-X Core bulunamadı, emülatör modunda çalışıyor")
    PDSX_AVAILABLE = False

class PDSXREPLError(Exception):
    """REPL spesifik hataları"""
    pass

class PDSXREPLSession:
    """PDS-X İnteraktif REPL Oturumu - Otomatik Kurulum"""
    
    def __init__(self, auto_importer=None):
        print("🚀 PDS-X REPL v1.0 başlatılıyor...")
        
        # Otomatik kurulum başlat (sessizce)
        if AUTO_IMPORTER_AVAILABLE:
            print("📦 Gereksinimler otomatik kontrol ediliyor...")
            try:
                if auto_setup_for_repl:
                    stats = auto_setup_for_repl()
                    if 'error' not in stats:
                        print(f"✅ Kurulum: {stats.get('installed', 0)}/{stats.get('total', 0)} paket")
                    else:
                        print(f"⚠️  Kurulum hatası: {stats['error']}")
            except Exception as e:
                print(f"⚠️  Otomatik kurulum atlandı: {e}")
        
        self.auto_importer = auto_importer
        self.interpreter = None
        self.history = []
        self.multiline_buffer = []
        self.in_multiline = False
        self.multiline_keywords = [
            'CLASS', 'YAPI', 'FUNCTION', 'SUB', 'FOR', 'WHILE', 'IF', 'SELECT',
            'GAMMA', 'OMEGA', 'STRUCT', 'UNION', 'ENUM'
        ]
        self.variables = {}
        self.commands_executed = 0
        self.session_start = time.time()
        
        # Çok satırlı program yönetimi
        self.program_manager = None
        self.in_program_mode = False
        
        # REPL konfigürasyonu
        self.config = {
            'prompt_primary': 'PDS-X> ',
            'prompt_multiline': '  ... ',
            'auto_indent': True,
            'syntax_highlighting': False,  # Basit terminal için
            'auto_completion': True,
            'history_size': 1000,
            'debug_mode': False
        }
        
        # Komut geçmişi dosyası
        self.history_file = Path('.pdsx_repl_history.json')
        self.load_history()
        
        # ReplyExtension'dan program manager'ı initialize et
        try:
            from reply_extension import MultiLineProgramManager
            self.program_manager = MultiLineProgramManager(self.interpreter)
            print("✅ Çok satırlı program sistemi yüklendi")
        except ImportError:
            print("⚠️  Çok satırlı program sistemi kullanılamıyor")
        
        print("✅ PDS-X REPL hazır!")
        if PDSX_AVAILABLE:
            try:
                self.interpreter = PDSXInterpreter()
                print("✅ PDS-X Core yorumlayıcı yüklendi")
            except Exception as e:
                print(f"⚠️  PDS-X Core yüklenemedi: {e}")
                self.interpreter = None
        
    def load_history(self):
        """Komut geçmişini yükle"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])[-self.config['history_size']:]
                    print(f"📜 {len(self.history)} komut geçmişi yüklendi")
        except Exception as e:
            print(f"⚠️  Geçmiş yükleme hatası: {e}")
            
    def save_history(self):
        """Komut geçmişini kaydet"""
        try:
            data = {
                'history': self.history[-self.config['history_size']:],
                'session_info': {
                    'commands_executed': self.commands_executed,
                    'session_duration': time.time() - self.session_start,
                    'last_save': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Geçmiş kaydetme hatası: {e}")
            
    def is_multiline_start(self, line: str) -> bool:
        """Çok-satırlı blok başlangıcını kontrol et"""
        line_upper = line.strip().upper()
        
        # Çok-satırlı anahtar kelimeler
        for keyword in self.multiline_keywords:
            if line_upper.startswith(keyword):
                return True
                
        # Parantez, köşeli parantez, süslü parantez kontrolü
        open_chars = line.count('(') + line.count('[') + line.count('{')
        close_chars = line.count(')') + line.count(']') + line.count('}')
        
        return open_chars > close_chars
        
    def is_multiline_end(self, line: str) -> bool:
        """Çok-satırlı blok bitişini kontrol et"""
        line_upper = line.strip().upper()
        
        # END ile başlayan satırlar
        if line_upper.startswith('END'):
            return True
            
        # Boş satır (bazı durumlarda bitirici)
        if not line.strip():
            return True
            
        return False
        
    def auto_indent_level(self, lines: List[str]) -> int:
        """Otomatik girinti seviyesi hesapla"""
        if not lines:
            return 0
            
        last_line = lines[-1].strip().upper()
        
        # Girinti artıran kelimeler
        indent_keywords = [
            'CLASS', 'YAPI', 'FUNCTION', 'SUB', 'FOR', 'WHILE', 'IF', 'ELSE',
            'SELECT', 'CASE', 'GAMMA', 'OMEGA', 'STRUCT', 'UNION'
        ]
        
        for keyword in indent_keywords:
            if last_line.startswith(keyword):
                return 1
                
        return 0
        
    def execute_repl_command(self, command: str) -> bool:
        """REPL özel komutlarını çalıştır"""
        cmd = command.strip()
        
        if cmd.startswith('.'):
            # Meta komutlar
            meta_cmd = cmd[1:].lower()
            
            if meta_cmd == 'help':
                self.show_help()
                return True
                
            elif meta_cmd == 'exit' or meta_cmd == 'quit':
                print("👋 PDS-X REPL kapatılıyor...")
                return False
                
            elif meta_cmd == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                return True
                
            elif meta_cmd == 'history':
                self.show_history()
                return True
                
            elif meta_cmd == 'vars':
                self.show_variables()
                return True
                
            elif meta_cmd == 'config':
                self.show_config()
                return True
                
            elif meta_cmd == 'stats':
                self.show_stats()
                return True
                
            elif meta_cmd == 'debug':
                self.config['debug_mode'] = not self.config['debug_mode']
                print(f"🐛 Debug modu: {'Açık' if self.config['debug_mode'] else 'Kapalı'}")
                return True
                
            elif meta_cmd.startswith('load '):
                filename = meta_cmd[5:].strip()
                self.load_program(filename)
                return True
                
            elif meta_cmd.startswith('save '):
                filename = meta_cmd[5:].strip()
                self.save_session(filename)
                return True
                
            else:
                print(f"❌ Bilinmeyen meta komut: {meta_cmd}")
                print("💡 .help yazarak yardım alabilirsiniz")
                return True
        
        # Çok satırlı program komutları kontrolü
        if self.program_manager:
            cmd_upper = cmd.upper()
            
            if cmd_upper.startswith('PROGRAM '):
                # Program yazma modunu başlat
                if self.program_manager.start_program(cmd):
                    print("📝 Program yazma modu aktif. 'END PROGRAM' ile bitirin.")
                    return True
                else:
                    print("❌ Program başlatma hatası")
                    return True
                    
            elif cmd_upper.startswith('RUN PROGRAM '):
                # Program çalıştır
                import re
                match = re.match(r'RUN PROGRAM\s+([a-zA-Z_][a-zA-Z0-9_]*)', cmd, re.IGNORECASE)
                if match:
                    program_name = match.group(1)
                    if self.program_manager.run_program(program_name):
                        print(f"✅ Program çalıştırıldı: {program_name}")
                    else:
                        print(f"❌ Program çalıştırma hatası: {program_name}")
                else:
                    print("❌ Geçersiz RUN PROGRAM sözdizimi")
                return True
                
            elif cmd_upper == 'LIST':
                # Tüm programları listele
                self.program_manager.list_programs()
                return True
                
            elif cmd_upper.startswith('LIST '):
                # Program listele/göster
                import re
                # LIST program_name veya LIST program_name.ext
                match = re.match(r'LIST\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\.[a-zA-Z]+)?', cmd, re.IGNORECASE)
                if match:
                    program_name = match.group(1)
                    extension = match.group(2)
                    if extension:
                        self.program_manager.show_program(program_name, extension)
                    else:
                        self.program_manager.show_program(program_name)
                else:
                    # LIST .extension (uzantıya göre filtrele)
                    ext_match = re.match(r'LIST\s+(\.[a-zA-Z]+)', cmd, re.IGNORECASE)
                    if ext_match:
                        extension = ext_match.group(1)
                        self.program_manager.list_programs(extension)
                    else:
                        print("❌ Geçersiz LIST sözdizimi")
                return True
        
        return False  # Normal PDS-X komutu
        
    def execute_pdsx_command(self, command: str) -> bool:
        """PDS-X BASIC komutunu çalıştır"""
        try:
            if self.interpreter:
                # Gerçek PDS-X yorumlayıcı
                result = self.interpreter.execute_command(command)
                if result is not None:
                    print(f"📤 {result}")
            else:
                # Emülatör modu
                self.emulate_command(command)
                
            self.commands_executed += 1
            return True
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            if self.config['debug_mode']:
                traceback.print_exc()
            return True
            
    def emulate_command(self, command: str):
        """PDS-X komutlarını emüle et (test modu)"""
        cmd_upper = command.strip().upper()
        
        if cmd_upper.startswith('PRINT '):
            # PRINT emülasyonu
            expr = command[6:].strip()
            try:
                # Basit değişken değerlendirmesi
                if expr in self.variables:
                    print(self.variables[expr])
                else:
                    print(expr.strip('"\''))  # String literal
            except:
                print(expr)
                
        elif cmd_upper.startswith('LET ') or '=' in command:
            # Değişken atama emülasyonu
            if cmd_upper.startswith('LET '):
                expr = command[4:].strip()
            else:
                expr = command.strip()
                
            if '=' in expr:
                var_name, value = expr.split('=', 1)
                var_name = var_name.strip()
                value = value.strip()
                
                # Basit değer ataması
                try:
                    if value.isdigit():
                        self.variables[var_name] = int(value)
                    elif value.replace('.', '').isdigit():
                        self.variables[var_name] = float(value)
                    else:
                        self.variables[var_name] = value.strip('"\'')
                    print(f"✅ {var_name} = {self.variables[var_name]}")
                except:
                    self.variables[var_name] = value
                    
        elif cmd_upper.startswith('DIM '):
            # Değişken tanımlama emülasyonu
            parts = command[4:].strip().split()
            if len(parts) >= 3 and parts[1].upper() == 'AS':
                var_name = parts[0]
                var_type = parts[2]
                self.variables[var_name] = f"<{var_type}>"
                print(f"✅ {var_name} tanımlandı ({var_type})")
                
        else:
            print(f"🔧 Emülatör: '{command}' komutu işlendi")
            
    def show_help(self):
        """Yardım menüsünü göster"""
        print("""
🆘 PDS-X REPL Yardım
==================

Meta Komutlar:
  .help       - Bu yardım menüsü
  .exit/.quit - REPL'den çık
  .clear      - Ekranı temizle
  .history    - Komut geçmişini göster
  .vars       - Tanımlı değişkenleri listele
  .config     - REPL konfigürasyonu
  .stats      - Oturum istatistikleri
  .debug      - Debug modunu aç/kapat
  .load <file> - Program dosyası yükle
  .save <file> - Oturumu kaydet

PDS-X Komutları:
  LET x = 10         - Değişken ata
  DIM x AS INTEGER   - Değişken tanımla
  PRINT x            - Değişken yazdır
  FOR ... END FOR    - Döngü (çok-satırlı)
  IF ... END IF      - Koşul (çok-satırlı)
  CLASS ... END CLASS - Sınıf tanımla
  
Çok-satırlı Mod:
  - CLASS, FOR, IF, WHILE vb. ile başlayın
  - Enter ile yeni satır ekleyin
  - END ile bitirin veya boş satır girin
  
Örnekler:
  PDS-X> LET x = 42
  PDS-X> PRINT x
  PDS-X> FOR i = 1 TO 10
    ...   PRINT i
    ... END FOR
""")

    def show_history(self, count: int = 10):
        """Komut geçmişini göster"""
        print(f"\n📜 Son {count} komut:")
        for i, cmd in enumerate(self.history[-count:], 1):
            print(f"  {i:2d}. {cmd}")
        print()
        
    def show_variables(self):
        """Tanımlı değişkenleri göster"""
        if not self.variables:
            print("📭 Henüz tanımlı değişken yok")
            return
            
        print(f"\n📊 Tanımlı Değişkenler ({len(self.variables)} adet):")
        for name, value in self.variables.items():
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"  {name:15} = {value_str}")
        print()
        
    def show_config(self):
        """REPL konfigürasyonunu göster"""
        print("\n⚙️  REPL Konfigürasyonu:")
        for key, value in self.config.items():
            print(f"  {key:20} = {value}")
        print()
        
    def show_stats(self):
        """Oturum istatistiklerini göster"""
        duration = time.time() - self.session_start
        print(f"""
📈 Oturum İstatistikleri:
  ⏱️  Süre: {duration:.1f} saniye
  🔢 Komut sayısı: {self.commands_executed}
  📜 Geçmiş: {len(self.history)} komut
  💾 Değişken: {len(self.variables)} adet
  🔧 Modu: {'PDS-X Core' if self.interpreter else 'Emülatör'}
""")

    def load_program(self, filename: str):
        """Program dosyası yükle"""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                print(f"❌ Dosya bulunamadı: {filename}")
                return
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            print(f"📂 {filename} yükleniyor...")
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    print(f"  {line_num:3d}> {line}")
                    if not self.execute_pdsx_command(line):
                        break
                        
            print(f"✅ {filename} yüklendi ({len(lines)} satır)")
            
        except Exception as e:
            print(f"❌ Dosya yükleme hatası: {e}")
            
    def save_session(self, filename: str):
        """Oturumu dosyaya kaydet"""
        try:
            session_data = {
                'history': self.history,
                'variables': self.variables,
                'config': self.config,
                'stats': {
                    'commands_executed': self.commands_executed,
                    'session_duration': time.time() - self.session_start,
                    'created': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Oturum kaydedildi: {filename}")
            
        except Exception as e:
            print(f"❌ Oturum kaydetme hatası: {e}")
            
    def run(self):
        """REPL ana döngüsü"""
        print(f"""
🚀 PDS-X İnteraktif REPL Modu
============================
Çok-satırlı komut desteği aktif
Meta komutlar için .help yazın
Çıkmak için .exit yazın

""")
        
        try:
            while True:
                try:
                    # Prompt belirleme
                    if self.in_multiline:
                        prompt = self.config['prompt_multiline']
                        if self.config['auto_indent']:
                            indent_level = self.auto_indent_level(self.multiline_buffer)
                            prompt += '  ' * indent_level
                    else:
                        prompt = self.config['prompt_primary']
                    
                    # Kullanıcı girişi al
                    try:
                        user_input = input(prompt)
                    except EOFError:
                        print("\n👋 REPL kapatılıyor...")
                        break
                    except KeyboardInterrupt:
                        if self.in_multiline:
                            print("\n❌ Çok-satırlı mod iptal edildi")
                            self.multiline_buffer.clear()
                            self.in_multiline = False
                        else:
                            print("\n👋 REPL kapatılıyor...")
                            break
                        continue
                    
                    # Boş satır kontrolü
                    if not user_input.strip():
                        if self.in_multiline:
                            # Çok-satırlı modu bitir
                            command = '\n'.join(self.multiline_buffer)
                            self.multiline_buffer.clear()
                            self.in_multiline = False
                            
                            if command.strip():
                                self.history.append(command)
                                if not self.process_command(command):
                                    break
                        continue
                    
                    # Çok-satırlı program modu kontrolü
                    if self.program_manager and self.program_manager.in_program_mode:
                        # Program yazma modunda
                        if self.program_manager.add_program_line(user_input):
                            # Hala program modunda, devam et
                            continue
                        else:
                            # Program modu bitti
                            continue
                    
                    # Çok-satırlı mod kontrolü
                    if self.in_multiline:
                        if self.is_multiline_end(user_input):
                            # Son satırı ekle ve çok-satırlı modu bitir
                            self.multiline_buffer.append(user_input)
                            command = '\n'.join(self.multiline_buffer)
                            self.multiline_buffer.clear()
                            self.in_multiline = False
                            
                            self.history.append(command)
                            if not self.process_command(command):
                                break
                        else:
                            # Çok-satırlı buffer'a ekle
                            self.multiline_buffer.append(user_input)
                    else:
                        # Tek satırlı veya çok-satırlı başlangıç kontrolü
                        if self.is_multiline_start(user_input):
                            # Çok-satırlı modu başlat
                            self.multiline_buffer.append(user_input)
                            self.in_multiline = True
                        else:
                            # Tek satırlı komut
                            self.history.append(user_input)
                            if not self.process_command(user_input):
                                break
                
                except Exception as e:
                    print(f"❌ REPL hatası: {e}")
                    if self.config['debug_mode']:
                        traceback.print_exc()
                    
                    # Çok-satırlı moddan çık
                    if self.in_multiline:
                        self.multiline_buffer.clear()
                        self.in_multiline = False
                    
        finally:
            # Temizlik işlemleri
            self.save_history()
            if self.auto_importer:
                self.auto_importer.cleanup()
            print("✅ PDS-X REPL oturumu temizlendi")
            
    def process_command(self, command: str) -> bool:
        """Komutu işle ve devam edip etmeyeceğini döndür"""
        # REPL komutlarını kontrol et
        repl_result = self.execute_repl_command(command)
        if repl_result is not None:
            return repl_result
        
        # PDS-X komutlarını çalıştır
        return self.execute_pdsx_command(command)

# ===================================================
# Ana REPL başlatıcı fonksiyonu
# ===================================================

def start_pdsx_repl(auto_importer=None):
    """PDS-X REPL'i başlat - tamamen otomatik"""
    print("🔧 PDS-X İnteraktif REPL başlatılıyor...")
    
    # AutoImporter setup - otomatik mode
    if auto_importer is None and AUTO_IMPORTER_AVAILABLE:
        if 'AutoImporter' in globals() and AutoImporter:
            try:
                auto_importer = AutoImporter()
                print("✅ AutoImporter REPL modu başlatıldı")
            except Exception as e:
                print(f"⚠️  AutoImporter başlatılamadı: {e}")
                auto_importer = None
    
    # REPL oturumu başlat
    repl = PDSXREPLSession(auto_importer)
    repl.run()

if __name__ == "__main__":
    # Doğrudan çalıştırılırsa REPL'i başlat
    start_pdsx_repl()
