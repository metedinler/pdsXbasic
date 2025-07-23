"""
Çok satırlı program yazma ve yönetme sistemi - PDS-X v14u Entegre
"""

import re
import time
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# PDS-X v14u GERÇEK INTERPRETER ENTEGRASYONU
try:
    # Ana PDS-X v14u interpreter'ı import et
    from pdsXuv14 import PdsXv14uInterpreter
    from command_executorx1 import CommandExecutor  # En gelişmiş v15 executor
    # Core system wrapper kullan
    from core_system import (
        get_core_manager, get_libx_core, get_memory_manager, get_auto_importer, get_auto_installer, get_hybrid_executor,
        CORE_SYSTEM_AVAILABLE, LIBX_AVAILABLE, MEMORY_MANAGER_AVAILABLE, AUTO_IMPORTER_AVAILABLE, AUTO_INSTALLER_AVAILABLE, HYBRID_EXECUTOR_AVAILABLE
    )
    REAL_PDSX_AVAILABLE = True
    print(f"[PDS-X] ✅ PDS-X v14u interpreter + CommandExecutor v15 yüklendi (Core: {'✅' if CORE_SYSTEM_AVAILABLE else '❌'})")
except ImportError as e:
    print(f"[PDS-X] ❌ PDS-X v14u interpreter yüklenemedi: {e}")
    try:
        # Fallback: command_executor kullan
        from command_executor import execute_command as fallback_executor
        FALLBACK_AVAILABLE = True
        CORE_SYSTEM_AVAILABLE = False
        print("[PDS-X] ⚠️ Fallback command executor yüklendi")
    except ImportError:
        FALLBACK_AVAILABLE = False
        CORE_SYSTEM_AVAILABLE = False
        print("[PDS-X] ❌ Hiçbir executor bulunamadı!")
    REAL_PDSX_AVAILABLE = False
    get_core_manager = None


class MultiLineProgramManager:
    """
    Çok satırlı program yazma ve yönetme sistemi
    
    Desteklenen format:
    PROGRAM <program_name>.<extension>
    <kod satırları>
    END PROGRAM
    
    Desteklenen uzantılar:
    - .basX  : PDS-X BASIC (varsayılan)
    - .libx  : LibX library pseudocode  
    - .pdsx  : PDS-X komutları ve makrolar
    - .py    : Python
    - .js    : JavaScript
    - .sql   : SQL queries
    """
    
    def __init__(self, programs_dir: Optional[str] = None):
        self.programs = {}  # Program ismi -> {extension -> {'code': str, 'metadata': dict}}
        self.programs_dir = Path(programs_dir or "programs")
        self.programs_dir.mkdir(exist_ok=True)
        
        # Desteklenen uzantılar ve özellikleri
        self.supported_extensions = {
            '.basx': {'name': 'PDS-X BASIC', 'executable': True, 'encrypt': True, 'compress': True},
            '.libx': {'name': 'LibX Library', 'executable': True, 'encrypt': True, 'compress': True},
            '.pdsx': {'name': 'PDS-X Commands', 'executable': True, 'encrypt': True, 'compress': True},
            '.py': {'name': 'Python', 'executable': True, 'encrypt': False, 'compress': False},
            '.js': {'name': 'JavaScript', 'executable': True, 'encrypt': False, 'compress': False},
            '.sql': {'name': 'SQL Queries', 'executable': False, 'encrypt': False, 'compress': False},
            '.txt': {'name': 'Plain Text', 'executable': False, 'encrypt': False, 'compress': False}
        }
        
        self.current_program = None
        self.current_code_lines = []
        self.in_program_mode = False
        
        # Şifreleme ve sıkıştırma için
        self.compression_enabled = True
        self.encryption_enabled = True
        
        # PDS-X v14u BASIC Interpreter - gerçek interpreter entegrasyonu
        if REAL_PDSX_AVAILABLE:
            try:
                # Ana PDS-X v14u interpreter'ını kullan
                self.pdsx_interpreter = PdsXv14uInterpreter()
                self.command_executor = CommandExecutor(self.pdsx_interpreter)
                print("[PDS-X] ✅ PDS-X v14u interpreter + CommandExecutor v15 başlatıldı")
                
                # AutoInstaller'ı önce başlat (hibrit executor için gerekli)
                if AUTO_INSTALLER_AVAILABLE and get_auto_installer:
                    try:
                        import os
                        self.auto_installer = get_auto_installer(os.getcwd())
                        if self.auto_installer:
                            print("[PDS-X] ✅ AutoInstaller entegre edildi")
                        else:
                            print("[PDS-X] ⚠️ AutoInstaller yüklenemedi")
                            self.auto_installer = None
                    except Exception as installer_error:
                        print(f"[PDS-X] ⚠️ AutoInstaller entegrasyon hatası: {installer_error}")
                        self.auto_installer = None
                else:
                    self.auto_installer = None
                
                # Hibrit Command Executor - Tüm executor'ları birleştir
                if HYBRID_EXECUTOR_AVAILABLE and get_hybrid_executor:
                    try:
                        self.hybrid_executor = get_hybrid_executor(self.pdsx_interpreter, self.auto_installer)
                        if self.hybrid_executor:
                            print("[PDS-X] 🚀 Hibrit Command Executor aktif (X1+X2Z1+X2 birleşik)")
                            # Ana executor olarak hibrit'i kullan
                            self.command_executor = self.hybrid_executor
                        else:
                            print("[PDS-X] ⚠️ Hibrit Command Executor yüklenemedi")
                            self.hybrid_executor = None
                    except Exception as hybrid_error:
                        print(f"[PDS-X] ⚠️ Hibrit Executor entegrasyon hatası: {hybrid_error}")
                        self.hybrid_executor = None
                else:
                    self.hybrid_executor = None
                
                # Core system entegrasyonu
                if CORE_SYSTEM_AVAILABLE and get_core_manager:
                    try:
                        self.core_manager = get_core_manager(self.pdsx_interpreter)
                        if self.core_manager:
                            print("[PDS-X] ✅ Core system (CoreManager) entegre edildi")
                        else:
                            print("[PDS-X] ⚠️ Core system yüklenemedi")
                            self.core_manager = None
                    except Exception as core_error:
                        print(f"[PDS-X] ⚠️ Core system entegrasyon hatası: {core_error}")
                        self.core_manager = None
                else:
                    self.core_manager = None
                
                # LibXCore entegrasyonu
                if LIBX_AVAILABLE and get_libx_core:
                    try:
                        self.libx_core = get_libx_core(self.pdsx_interpreter)
                        if self.libx_core:
                            print("[PDS-X] ✅ LibXCore entegre edildi")
                        else:
                            print("[PDS-X] ⚠️ LibXCore yüklenemedi")
                            self.libx_core = None
                    except Exception as libx_error:
                        print(f"[PDS-X] ⚠️ LibXCore entegrasyon hatası: {libx_error}")
                        self.libx_core = None
                else:
                    self.libx_core = None
                
                # MemoryManager entegrasyonu
                if MEMORY_MANAGER_AVAILABLE and get_memory_manager:
                    try:
                        self.memory_manager = get_memory_manager(self.pdsx_interpreter)
                        if self.memory_manager:
                            print("[PDS-X] ✅ MemoryManager entegre edildi")
                        else:
                            print("[PDS-X] ⚠️ MemoryManager yüklenemedi")
                            self.memory_manager = None
                    except Exception as memory_error:
                        print(f"[PDS-X] ⚠️ MemoryManager entegrasyon hatası: {memory_error}")
                        self.memory_manager = None
                else:
                    self.memory_manager = None
                    
                # AutoImporter entegrasyonu
                if AUTO_IMPORTER_AVAILABLE and get_auto_importer:
                    try:
                        self.auto_importer = get_auto_importer()
                        if self.auto_importer:
                            print("[PDS-X] ✅ AutoImporter entegre edildi")
                            # Import kontrolü yap
                            self._check_imports()
                        else:
                            print("[PDS-X] ⚠️ AutoImporter yüklenemedi")
                            self.auto_importer = None
                    except Exception as auto_error:
                        print(f"[PDS-X] ⚠️ AutoImporter entegrasyon hatası: {auto_error}")
                        self.auto_importer = None
                else:
                    self.auto_importer = None
                
                # Core systems status
                core_systems = []
                if self.core_manager:
                    core_systems.append("CoreManager")
                if self.libx_core:
                    core_systems.append("LibXCore")
                if self.memory_manager:
                    core_systems.append("MemoryManager")
                if self.auto_importer:
                    core_systems.append("AutoImporter")
                    
                if core_systems:
                    print(f"[PDS-X] 🔧 Entegre core systems: {', '.join(core_systems)}")
                else:
                    print("[PDS-X] ❌ Core systems kullanılamıyor")
                    
            except Exception as e:
                print(f"[PDS-X] ⚠️ PDS-X interpreter oluşturma hatası: {e}")
                self.pdsx_interpreter = None
                self.command_executor = None
                self.core_manager = None
                self.libx_core = None
                self.memory_manager = None
        else:
            self.pdsx_interpreter = None
            self.command_executor = None
            self.core_manager = None
            self.libx_core = None
            self.memory_manager = None
            print("[PDS-X] ❌ PDS-X v14u interpreter kullanılamıyor")
        
    def start_program(self, program_declaration: str) -> bool:
        """
        Program yazma modunu başlat
        
        Format: PROGRAM <n>.<extension>
        Örnek: PROGRAM hello.basX
               PROGRAM mylib.libx
               PROGRAM scripts.pdsx
        """
        try:
            # Program deklarasyonunu parse et
            match = re.match(r'PROGRAM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z]+)?)', program_declaration.strip())
            if not match:
                print("[PDS-X] ❌ Geçersiz program deklarasyonu. Format: PROGRAM <n>.<extension>")
                return False
            
            program_name_with_ext = match.group(1)
            
            # Uzantı kontrolü
            if '.' in program_name_with_ext:
                program_name, extension = program_name_with_ext.rsplit('.', 1)
                extension = '.' + extension.lower()
            else:
                program_name = program_name_with_ext
                extension = '.basx'  # Varsayılan uzantı
            
            # Desteklenen uzantı kontrolü
            if extension not in self.supported_extensions:
                print(f"[PDS-X] ⚠️ Desteklenmeyen uzantı: {extension}")
                print(f"[PDS-X] 📋 Desteklenen uzantılar: {', '.join(self.supported_extensions.keys())}")
                return False
            
            # Program modu başlat
            self.current_program = {
                'name': program_name,
                'extension': extension,
                'full_name': f"{program_name}{extension}"
            }
            self.current_code_lines = []
            self.in_program_mode = True
            
            ext_info = self.supported_extensions[extension]
            print(f"[PDS-X] 📝 Program yazma modu başlatıldı: {self.current_program['full_name']}")
            print(f"[PDS-X] 🔤 Dil: {ext_info['name']}")
            print(f"[PDS-X] 💾 Şifreleme: {'✅' if ext_info['encrypt'] else '❌'}")
            print(f"[PDS-X] 🗜️ Sıkıştırma: {'✅' if ext_info['compress'] else '❌'}")
            print(f"[PDS-X] ⚡ Çalıştırılabilir: {'✅' if ext_info['executable'] else '❌'}")
            print(f"[PDS-X] 📋 'END PROGRAM' yazarak tamamlayın")
            
            return True
            
        except Exception as e:
            print(f"[PDS-X] ❌ Program başlatma hatası: {e}")
            return False
    
    def add_line(self, line: str) -> bool:
        """Program modunda satır ekle"""
        if not self.in_program_mode:
            return False
        self.current_code_lines.append(line)
        return True
        
    def end_program(self) -> bool:
        """Program yazma modunu sonlandır ve kaydet"""
        if not self.in_program_mode or self.current_program is None:
            return False
        
        # Program kodunu birleştir
        code = '\n'.join(self.current_code_lines)
        
        # Program bilgilerini kaydet
        program_key = self.current_program['name']
        if program_key not in self.programs:
            self.programs[program_key] = {}
        
        self.programs[program_key][self.current_program['extension']] = {
            'code': code,
            'metadata': {
                'created': time.time(),
                'lines': len(self.current_code_lines),
                'size': len(code)
            }
        }
        
        # Dosyaya kaydet
        try:
            self._save_program_to_file(program_key, self.current_program['extension'], code)
        except Exception as e:
            print(f"[PDS-X] ⚠️ Dosya kaydetme hatası: {e}")
        
        print(f"[PDS-X] ✅ Program kaydedildi: {self.current_program['full_name']}")
        print(f"[PDS-X] 📊 {len(self.current_code_lines)} satır, {len(code)} karakter")
        
        # Modu sıfırla
        self.in_program_mode = False
        self.current_program = None
        self.current_code_lines = []
        
        return True
        
    def _get_all_programs(self) -> dict:
        """Tüm programları döndür"""
        return self.programs.copy()
        
    def _get_program_content(self, name: str, extension: str) -> str:
        """Program içeriğini döndür"""
        if name in self.programs and extension in self.programs[name]:
            return self.programs[name][extension]['code']
        return ""
        
    def _compress_content(self, content: str) -> bytes:
        """İçeriği sıkıştır"""
        import gzip
        return gzip.compress(content.encode('utf-8'))
        
    def _decompress_content(self, compressed: bytes) -> str:
        """Sıkıştırılmış içeriği aç"""
        import gzip
        return gzip.decompress(compressed).decode('utf-8')
        
    def _encrypt_content(self, content: str, password: str) -> bytes:
        """İçeriği şifrele (basit XOR)"""
        content_bytes = content.encode('utf-8')
        password_bytes = password.encode('utf-8')
        result = bytearray()
        for i, byte in enumerate(content_bytes):
            result.append(byte ^ password_bytes[i % len(password_bytes)])
        return bytes(result)
        
    def _decrypt_content(self, encrypted: bytes, password: str) -> str:
        """Şifrelenmiş içeriği çöz"""
        password_bytes = password.encode('utf-8')
        result = bytearray()
        for i, byte in enumerate(encrypted):
            result.append(byte ^ password_bytes[i % len(password_bytes)])
        return result.decode('utf-8')
        
    def _save_program_to_file(self, name: str, extension: str, code: str):
        """Programı dosyaya kaydet"""
        filename = f"{name}{extension}"
        filepath = self.programs_dir / filename
        
        # Basit metin dosyası olarak kaydet
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
            
    def list_programs(self, filter_ext: Optional[str] = None) -> None:
        """Tüm programları listele"""
        if not self.programs:
            print("[PDS-X] 📋 Henüz program yok")
            return
        
        print("[PDS-X] 📋 Kaydedilmiş Programlar:")
        print("-" * 50)
        
        for prog_name, extensions in self.programs.items():
            for ext, data in extensions.items():
                if filter_ext is None or ext == filter_ext:
                    metadata = data['metadata']
                    lang = self.supported_extensions.get(ext, {}).get('name', 'Unknown')
                    print(f"  📄 {prog_name}{ext} ({lang})")
                    print(f"      📊 {metadata['lines']} satır, {metadata['size']} karakter")
                    created = time.strftime('%Y-%m-%d %H:%M', time.localtime(metadata['created']))
                    print(f"      📅 Oluşturulma: {created}")
    
    def show_program(self, program_name: str, extension: Optional[str] = None) -> None:
        """Program içeriğini göster"""
        if program_name not in self.programs:
            print(f"[PDS-X] ❌ Program bulunamadı: {program_name}")
            return
        
        prog_data = self.programs[program_name]
        
        if extension:
            if extension not in prog_data:
                print(f"[PDS-X] ❌ Uzantı bulunamadı: {program_name}{extension}")
                return
            extensions_to_show = [extension]
        else:
            extensions_to_show = list(prog_data.keys())
        
        for ext in extensions_to_show:
            data = prog_data[ext]
            print(f"\n[PDS-X] 📄 Program: {program_name}{ext}")
            print("-" * 40)
            lines = data['code'].split('\n')
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}: {line}")
    
    def run_program(self, program_name: str) -> bool:
        """Programı çalıştır - PDS-X v14u entegre"""
        if program_name not in self.programs:
            print(f"[PDS-X] ❌ Program bulunamadı: {program_name}")
            return False
        
        prog_data = self.programs[program_name]
        
        # PDS-X BASIC programları (.basx/.libx/.pdsx)
        if '.basx' in prog_data or '.libx' in prog_data or '.pdsx' in prog_data:
            try:
                # Önce .basx, sonra .libx, sonra .pdsx dene
                for ext in ['.basx', '.libx', '.pdsx']:
                    if ext in prog_data:
                        code = prog_data[ext]['code']
                        print(f"[PDS-X] 🔧 {ext} programı çalıştırılıyor...")
                        
                        # Gerçek PDS-X v14u entegrasyonu
                        if self.command_executor:
                            # CommandExecutor v15 ile satır satır çalıştır
                            try:
                                lines = [line.strip() for line in code.split('\n') if line.strip()]
                                for line_num, line in enumerate(lines, 1):
                                    if line and not line.startswith('REM') and not line.startswith("'"):
                                        try:
                                            # Hibrit executor ile komut yürütme
                                            if hasattr(self.command_executor, 'execute_command'):
                                                # Hibrit executor
                                                import asyncio
                                                result = asyncio.run(self.command_executor.execute_command(line))
                                                if result:
                                                    print(f"[PDS-X] 📝 Satır {line_num}: {result}")
                                            elif hasattr(self.command_executor, 'command_handlers'):
                                                # Standard executor  
                                                command_type = line.split()[0].upper()
                                                if command_type in self.command_executor.command_handlers:
                                                    handler = self.command_executor.command_handlers[command_type]
                                                    result = handler(line)
                                                    if result:
                                                        print(f"[PDS-X] 📝 Satır {line_num}: {result}")
                                                else:
                                                    print(f"[PDS-X] ⚠️ Bilinmeyen komut: {command_type}")
                                            else:
                                                print(f"[PDS-X] ⚠️ Executor interface bulunamadı")
                                        except Exception as line_error:
                                            print(f"[PDS-X] ⚠️ Satır {line_num} hatası: {line_error}")
                                print(f"[PDS-X] ✅ {ext} programı CommandExecutor v15 ile tamamlandı")
                                return True
                            except Exception as e:
                                print(f"[PDS-X] ⚠️ CommandExecutor v15 hatası: {e}")
                                
                        elif self.pdsx_interpreter:
                            # PDS-X v14u interpreter ile async çalıştır
                            try:
                                import asyncio
                                lines = [line.strip() for line in code.split('\n') if line.strip()]
                                for line_num, line in enumerate(lines, 1):
                                    if line and not line.startswith('REM') and not line.startswith("'"):
                                        try:
                                            result = asyncio.run(self.pdsx_interpreter.execute_command_async(line))
                                            if result:
                                                print(f"[PDS-X] 📝 Satır {line_num}: {result}")
                                        except Exception as line_error:
                                            print(f"[PDS-X] ⚠️ Satır {line_num} hatası: {line_error}")
                                print(f"[PDS-X] ✅ {ext} programı PDS-X v14u ile tamamlandı")
                                return True
                            except Exception as e:
                                print(f"[PDS-X] ⚠️ PDS-X v14u interpreter hatası: {e}")
                        else:
                            print(f"[PDS-X] ❌ PDS-X interpreter ve CommandExecutor mevcut değil")
                            return False
                        
                        print(f"[PDS-X] ❌ {ext} programı çalıştırılamadı")
                        continue
                            
                print(f"[PDS-X] ❌ Program çalıştırılamadı: {program_name}")
                return False
                
            except Exception as e:
                print(f"[PDS-X] ❌ PDS-X BASIC çalıştırma hatası: {e}")
                return False
        
        # Python programı varsa çalıştır
        elif '.py' in prog_data:
            try:
                code = prog_data['.py']['code']
                print("[PDS-X] 🐍 Python programı çalıştırılıyor...")
                exec(code)
                return True
            except Exception as e:
                print(f"[PDS-X] ❌ Python çalıştırma hatası: {e}")
                return False
        
        # JavaScript programı varsa çalıştır (Node.js gerekli)
        elif '.js' in prog_data:
            try:
                code = prog_data['.js']['code']
                print("[PDS-X] 🟨 JavaScript programı çalıştırılıyor...")
                
                # Geçici dosya oluştur
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Node.js ile çalıştır
                result = subprocess.run(['node', temp_file], 
                                      capture_output=True, text=True, timeout=30)
                
                # Geçici dosyayı sil
                os.unlink(temp_file)
                
                if result.returncode == 0:
                    print(result.stdout, end='')
                    return True
                else:
                    print(f"[PDS-X] ❌ JavaScript hatası: {result.stderr}")
                    return False
                    
            except FileNotFoundError:
                print("[PDS-X] ❌ Node.js bulunamadı. JavaScript çalıştırmak için Node.js yükleyin.")
                return False
            except Exception as e:
                print(f"[PDS-X] ❌ JavaScript çalıştırma hatası: {e}")
                return False
        
        # Diğer uzantılar için
        available_exts = list(prog_data.keys())
        print(f"[PDS-X] ℹ️ Program mevcut uzantıları: {', '.join(available_exts)}")
        print(f"[PDS-X] ℹ️ Şu anda sadece .py ve .js programları çalıştırılabilir")
        
        return False
    
    def _check_imports(self):
        """Tüm modüllerdeki import'ları kontrol et ve eksikleri yükle"""
        if not self.auto_importer:
            return
        
        print("[PDS-X] 🔍 Import analizi başlatılıyor...")
        
        try:
            # Aktif modülleri tara
            modules_to_check = [
                "pdsXuv14.py", "core2-6.py", "libxcore.py", 
                "memory_manager.py", "command_executorx1.py",
                "pdsx_unified_exception.py", "tree3.py",
                "save_load_system2.py", "quantum_pdsX.py"
            ]
            
            missing_packages = set()
            
            for module in modules_to_check:
                if os.path.exists(module):
                    missing = self._analyze_module_imports(module)
                    missing_packages.update(missing)
            
            if missing_packages:
                print(f"[PDS-X] ⚠️ Eksik paketler bulundu: {', '.join(missing_packages)}")
                print("[PDS-X] 📦 AutoImporter ile yükleniyor...")
                
                # AutoImporter ile yükle
                for package in missing_packages:
                    try:
                        self.auto_importer.install_package(package)
                        print(f"[PDS-X] ✅ {package} yüklendi")
                    except Exception as e:
                        print(f"[PDS-X] ❌ {package} yüklenemedi: {e}")
            else:
                print("[PDS-X] ✅ Tüm import'lar mevcut")
                
        except Exception as e:
            print(f"[PDS-X] ⚠️ Import analizi hatası: {e}")
    
    def _analyze_module_imports(self, module_path: str) -> set:
        """Bir modüldeki import'ları analiz et ve eksikleri döndür"""
        missing = set()
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Import pattern'leri
            import_patterns = [
                r'^import\s+(\w+)',
                r'^from\s+(\w+)\s+import',
                r'^\s*import\s+(\w+)',
                r'^\s*from\s+(\w+)\s+import'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    # Standard library olmayan paketleri kontrol et
                    if not self._is_standard_library(match):
                        try:
                            __import__(match)
                        except ImportError:
                            missing.add(match)
            
        except Exception as e:
            print(f"[PDS-X] ⚠️ {module_path} analiz hatası: {e}")
        
        return missing
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Modülün standard library olup olmadığını kontrol et"""
        stdlib_modules = {
            'os', 'sys', 're', 'json', 'time', 'datetime', 'threading',
            'pathlib', 'collections', 'functools', 'itertools', 'logging',
            'traceback', 'subprocess', 'shutil', 'platform', 'socket',
            'urllib', 'http', 'email', 'hashlib', 'uuid', 'random',
            'decimal', 'math', 'statistics', 'typing', 'dataclasses'
        }
        return module_name in stdlib_modules
