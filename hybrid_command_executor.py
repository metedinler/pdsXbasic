# hybrid_command_executor.py - PDS-X Hibrit Command Executor Sistemi
# Version: 1.0.0
# Date: July 21, 2025
# Tüm command executor'ların en iyi özelliklerini birleştiren hibrit sistem

"""
PDS-X Hibrit Command Executor

Bu sistem farklı command executor'ların güçlü yanlarını birleştirerek
optimize edilmiş bir komut yürütme sistemi sağlar:

- command_executorx1.py -> Temel komutlar ve performans
- command_executorx2z1.py -> Async ve gelişmiş özellikler  
- command_executor2.py -> Güvenilirlik ve hata yönetimi

Ayrıca AutoInstaller entegrasyonu ile:
- Çift alias sistemi
- Dinamik çakışma çözümü
- Modül bazlı komut yönetimi
"""

import re
import asyncio
import json
import time
import os
import subprocess
from typing import Dict, Callable, Optional, List, Any, Union
from concurrent.futures import ThreadPoolExecutor
import logging

# PDS-X imports
try:
    from pdsx_unified_exception import UnifiedException, PdsXExceptionHandler
except ImportError:
    from pdsx_exception import PdsXException as UnifiedException
    PdsXExceptionHandler = None

# Setup logging
log = logging.getLogger("hybrid_executor")

class HybridCommandExecutor:
    """
    Hibrit Command Executor - Tüm executor'ların en iyi özelliklerini birleştir
    """
    
    def __init__(self, interpreter, auto_installer=None):
        self.interpreter = interpreter
        self.auto_installer = auto_installer
        
        # Command routing system - hangi komut hangi executor'a gidecek
        self.command_routes = {
            "basic": [],      # Temel komutlar (x1'den)
            "async": [],      # Async komutlar (x2z1'den)  
            "advanced": [],   # Gelişmiş komutlar (x2'den)
            "hybrid": []      # Özel hibrit komutlar
        }
        
        # Handler registries
        self.primary_handlers: Dict[str, Callable] = {}
        self.alias_handlers: Dict[str, str] = {}  # alias -> primary_command
        self.secondary_aliases: Dict[str, List[str]] = {}  # primary -> [aliases]
        
        # Executor instances
        self.executors = {}
        self._initialize_executors()
        self._register_all_commands()
        self._setup_alias_system()
        
        log.info("🚀 Hibrit Command Executor başlatıldı")
    
    @property
    def command_handlers(self) -> Dict[str, Callable]:
        """Geriye uyumluluk için command_handlers property'si"""
        return self.primary_handlers
    
    def _initialize_executors(self):
        """Alt executor'ları başlat"""
        try:
            # Primary executor (x1) - Temel komutlar için
            from command_executorx1 import CommandExecutor as X1Executor
            self.executors['x1'] = X1Executor(self.interpreter)
            log.info("✅ CommandExecutorX1 yüklendi (temel komutlar)")
        except Exception as e:
            log.warning(f"⚠️ CommandExecutorX1 yüklenemedi: {e}")
        
        try:
            # Async executor (x2z1) - Async komutlar için
            from command_executorx2z1 import CommandExecutor as X2Z1Executor  
            self.executors['x2z1'] = X2Z1Executor()
            log.info("✅ CommandExecutorX2Z1 yüklendi (async komutlar)")
        except Exception as e:
            log.warning(f"⚠️ CommandExecutorX2Z1 yüklenemedi: {e}")
            
        try:
            # Advanced executor (x2) - Gelişmiş komutlar için
            from command_executor2 import CommandExecutor as X2Executor
            self.executors['x2'] = X2Executor(self.interpreter)
            log.info("✅ CommandExecutor2 yüklendi (gelişmiş komutlar)")
        except Exception as e:
            log.warning(f"⚠️ CommandExecutor2 yüklenemedi: {e}")
    
    def _register_all_commands(self):
        """Tüm komutları kaydet ve route'ları belirle"""
        
        # Temel komutlar (X1'den) - Hızlı ve güvenilir
        basic_commands = [
            "PRINT", "LET", "DIM", "INPUT", "IF", "ELSE", "ENDIF",
            "FOR", "NEXT", "WHILE", "WEND", "DO", "LOOP", "GOTO", "GOSUB", "RETURN"
        ]
        
        # Async komutlar (X2Z1'den) - Performans gerektiren
        async_commands = [
            "EDIT MODULE", "THREAD CREATE", "DB CONNECT", "HTTP REQUEST",
            "FILE ASYNC", "CRYPTO ENCRYPT", "MQTT PUBLISH"
        ]
        
        # Gelişmiş komutlar (X2'den) - Karmaşık mantık
        advanced_commands = [
            "SELECT CASE", "CASE", "END SELECT", "TRY", "CATCH", "FINALLY",
            "CLASS", "FUNCTION", "SUB", "MODULE", "IMPORT"
        ]
        
        # Route'ları kaydet
        self.command_routes["basic"] = basic_commands
        self.command_routes["async"] = async_commands  
        self.command_routes["advanced"] = advanced_commands
        
        # Primary handler'ları kaydet
        for cmd in basic_commands:
            if 'x1' in self.executors:
                self.primary_handlers[cmd] = self._route_to_x1
        
        for cmd in async_commands:
            if 'x2z1' in self.executors:
                self.primary_handlers[cmd] = self._route_to_x2z1
                
        for cmd in advanced_commands:
            if 'x2' in self.executors:
                self.primary_handlers[cmd] = self._route_to_x2
    
    def _setup_alias_system(self):
        """Çift alias sistemi kur"""
        # Ana komutlar için alias'lar
        alias_map = {
            # Temel komutlar
            "PRINT": ["P", "ECHO", "WRITE", "OUTPUT"],
            "LET": ["SET", "ASSIGN", "="],
            "DIM": ["DECLARE", "VAR"],
            "INPUT": ["READ", "GET", "SCAN"],
            "IF": ["WHEN", "CHECK"],
            "FOR": ["ITERATE", "LOOP"],
            
            # Async komutlar  
            "EDIT MODULE": ["EDITMOD", "MODIFYMODULE", "CHANGEMOD"],
            "THREAD CREATE": ["CREATETHREAD", "NEWTHREAD", "SPAWN"],
            "DB CONNECT": ["CONNECTDB", "DBOPEN", "DATABASE"],
            
            # Gelişmiş komutlar
            "SELECT CASE": ["SWITCH", "MATCH", "CHOOSE"],
            "FUNCTION": ["FUNC", "DEF", "PROCEDURE"],
            "MODULE": ["MOD", "UNIT", "PACKAGE"]
        }
        
        # Alias mapping'i kaydet
        for primary, aliases in alias_map.items():
            self.secondary_aliases[primary] = aliases
            for alias in aliases:
                self.alias_handlers[alias.upper()] = primary
        
        log.info(f"📝 {len(self.alias_handlers)} alias kaydedildi")
    
    async def execute_command(self, command: str, scope_name: str = "global") -> Any:
        """
        Ana komut yürütme metodu - hibrit routing ile
        """
        try:
            # Komut normalizasyonu
            command = command.strip()
            if not command:
                return None
            
            # Komut tipini belirle
            cmd_word = self._extract_command_word(command)
            
            # Alias kontrolü
            if cmd_word in self.alias_handlers:
                actual_cmd = self.alias_handlers[cmd_word]
                command = command.replace(cmd_word, actual_cmd, 1)
                cmd_word = actual_cmd
            
            # Route belirleme ve yürütme
            if cmd_word in self.primary_handlers:
                handler = self.primary_handlers[cmd_word]
                return await handler(command, scope_name)
            else:
                # Fallback - X1 executor'ı dene
                if 'x1' in self.executors:
                    return await self._route_to_x1(command, scope_name)
                else:
                    raise UnifiedException(f"Bilinmeyen komut: {cmd_word}")
                    
        except Exception as e:
            if PdsXExceptionHandler:
                PdsXExceptionHandler.handle_exception(e)
            else:
                log.error(f"Komut yürütme hatası: {e}")
            raise
    
    def _extract_command_word(self, command: str) -> str:
        """Komutun ilk kelimesini çıkart"""
        parts = command.split()
        if not parts:
            return ""
        
        # Multi-word commands için (SELECT CASE, EDIT MODULE vb.)
        if len(parts) >= 2:
            two_word = f"{parts[0]} {parts[1]}"
            if two_word.upper() in self.primary_handlers or two_word.upper() in self.alias_handlers:
                return two_word.upper()
        
        return parts[0].upper()
    
    # Route metodları
    async def _route_to_x1(self, command: str, scope_name: str) -> Any:
        """X1 executor'a yönlendir"""
        executor = self.executors['x1']
        # X1'in execute metodunu çağır
        if hasattr(executor, 'execute_command'):
            return await executor.execute_command(command, scope_name)
        else:
            return executor.execute(command, scope_name)
    
    async def _route_to_x2z1(self, command: str, scope_name: str) -> Any:
        """X2Z1 executor'a yönlendir (async)"""
        executor = self.executors['x2z1']
        cmd_word = self._extract_command_word(command)
        
        # Specific handler metodunu çağır
        if cmd_word == "EDIT MODULE":
            return await executor.handle_edit_module(command, scope_name)
        elif cmd_word == "THREAD CREATE":
            return await executor.handle_thread_create(command, scope_name)
        elif cmd_word == "DB CONNECT":
            return await executor.handle_db_connect(command, scope_name)
        else:
            # Generic execute
            if hasattr(executor, 'execute'):
                return await executor.execute(command, scope_name)
    
    async def _route_to_x2(self, command: str, scope_name: str) -> Any:
        """X2 executor'a yönlendir"""
        executor = self.executors['x2']
        if hasattr(executor, 'execute_command'):
            return await executor.execute_command(command, scope_name)
        else:
            return executor.execute(command, scope_name)
    
    def register_module_commands(self, module_name: str, commands: Dict[str, Callable]):
        """Modül bazlı komut kayıtı (AutoInstaller entegrasyonu için)"""
        for cmd_name, handler in commands.items():
            # Çakışma kontrolü
            if cmd_name in self.primary_handlers:
                # Alias üret
                alias = f"{module_name}_{cmd_name}"
                self.alias_handlers[alias] = cmd_name
                self.secondary_aliases.setdefault(cmd_name, []).append(alias)
                log.warning(f"⚠️ Komut çakışması: {cmd_name} -> {alias} alias'ı oluşturuldu")
            else:
                self.primary_handlers[cmd_name] = handler
        
        log.info(f"📦 {module_name} modülü için {len(commands)} komut kaydedildi")
    
    def get_command_info(self, command: Optional[str] = None) -> Dict:
        """Komut bilgilerini döndür"""
        if command:
            cmd_word = self._extract_command_word(command)
            if cmd_word in self.alias_handlers:
                cmd_word = self.alias_handlers[cmd_word]
            
            info = {
                "command": cmd_word,
                "aliases": self.secondary_aliases.get(cmd_word, []),
                "route": self._get_command_route(cmd_word),
                "available": cmd_word in self.primary_handlers
            }
            return info
        else:
            # Tüm komutlar
            return {
                "total_commands": len(self.primary_handlers),
                "total_aliases": len(self.alias_handlers),
                "routes": {
                    "basic": len(self.command_routes["basic"]),
                    "async": len(self.command_routes["async"]),
                    "advanced": len(self.command_routes["advanced"])
                },
                "executors": list(self.executors.keys())
            }
    
    def _get_command_route(self, cmd_word: str) -> str:
        """Komutun hangi route'ta olduğunu bul"""
        for route_name, commands in self.command_routes.items():
            if cmd_word in commands:
                return route_name
        return "unknown"
    
    def install_module_with_commands(self, module_path: str):
        """AutoInstaller ile modül kur ve komutlarını kaydet"""
        if not self.auto_installer:
            log.warning("AutoInstaller bulunamadı")
            return False
        
        try:
            # AutoInstaller ile modülü analiz et ve kur
            dependencies = self.auto_installer.analyze_new_module(module_path)
            success = self.auto_installer.install_dependencies(dependencies)
            
            if success:
                # Modülden komutları çıkart ve kaydet
                module_name = os.path.basename(module_path).replace('.py', '')
                # Module'i import et ve __pdsX_exports__ varsa kaydet
                # TODO: Dynamic import and command extraction
                log.info(f"✅ {module_name} modülü başarıyla kuruldu")
                return True
            else:
                module_name = os.path.basename(module_path).replace('.py', '')
                log.error(f"❌ {module_name} modülü kurulamadı")
                return False
                
        except Exception as e:
            log.error(f"Modül kurulum hatası: {e}")
            return False

# Singleton instance
_hybrid_executor = None

def get_hybrid_executor(interpreter=None, auto_installer=None):
    """Hibrit executor singleton'ını döndür"""
    global _hybrid_executor
    if _hybrid_executor is None:
        _hybrid_executor = HybridCommandExecutor(interpreter, auto_installer)
    return _hybrid_executor

# Export
__all__ = ['HybridCommandExecutor', 'get_hybrid_executor']
