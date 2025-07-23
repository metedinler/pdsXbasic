# pdsXu_v15.py
# Version: 1.5.0
# Description: PDS-X BASIC v15 yorumlayıcısı, QBASIC 7.1 benzeri, özelleştirilmiş.
# Author: metedinler
# License: MIT

import argparse
import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional
from autoimporter import install_library
from core2 import CoreV15
from pdsx_exception2 import PdsXException, PdsXSyntaxError, PdsXRuntimeError

# Loglama ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("pdsXu")

class PDSXInterpreter:
    """PDS-X BASIC v15 yorumlayıcısı."""
    
    def __init__(self):
        self.core = CoreV15(self)
        self.program: List[str] = []
        self.program_counter: int = 0
        self.global_vars: Dict = {}
        self.shared_vars: Dict = {}
        self.scopes: List[Dict] = [{}]
        self.labels: Dict[str, int] = {}
        self.file_handles: Dict[int, asyncio.StreamReader] = {}
        self.object_counter: Dict[str, int] = {}
        self.object_registry: Dict[int, Dict] = {}
        self.restricted_scopes: set = set()
        self.restricted_vars: set = set()
        self.exception_manager = PdsXException(self)
        self.config: Dict = {}

    def current_scope(self) -> Dict:
        """Geçerli kapsamı döndürür."""
        return self.scopes[-1]

    async def load_config(self, config_file: str) -> None:
        """Yapılandırma dosyasını yükler."""
        try:
            async with aiofiles.open(config_file, "r", encoding="utf-8") as f:
                self.config = json.loads(await f.read())
            log.debug(f"Yapılandırma yüklendi: {config_file}")
        except Exception as e:
            raise PdsXRuntimeError(f"Yapılandırma yükleme hatası: {str(e)}")

    async def load_program(self, file_path: str) -> None:
        """Program dosyasını yükler."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                self.program = [line.strip() for line in await f.readlines() if line.strip()]
            self.labels = {line.split(":")[0].strip(): i for i, line in enumerate(self.program) if ":" in line}
            log.debug(f"Program yüklendi: {file_path}, Satır sayısı: {len(self.program)}")
        except Exception as e:
            raise PdsXRuntimeError(f"Program yükleme hatası: {str(e)}")

    async def execute_command(self, command: str) -> Optional[int]:
        """Komutu yürütür."""
        try:
            return await self.core.parse_core_command(command)
        except PdsXException as e:
            await self.exception_manager.handle_error(e)
            return None

    async def run(self) -> None:
        """Programı çalıştırır."""
        self.program_counter = 0
        while 0 <= self.program_counter < len(self.program):
            command = self.program[self.program_counter]
            next_line = await self.execute_command(command)
            self.program_counter = next_line if next_line is not None else self.program_counter + 1
            log.debug(f"Komut yürütüldü: {command}, PC: {self.program_counter}")

    async def interactive_shell(self) -> None:
        """Etkileşimli kabuğu başlatır."""
        print("PDS-X BASIC v15.0 - Etkileşimli Kabuk")
        while True:
            try:
                command = input("PDSX> ")
                if command.lower() in ("exit", "quit"):
                    break
                await self.execute_command(command)
            except KeyboardInterrupt:
                print("\nÇıkış için 'exit' yazın.")
            except Exception as e:
                log.error(f"Kabuk hatası: {str(e)}")
                print(f"Hata: {str(e)}")

def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(description="PDS-X BASIC v15 Yorumlayıcısı")
    parser.add_argument("--version", action="version", version="PDS-X BASIC v15.0")
    parser.add_argument("--file", type=str, help=".basX dosyasını çalıştırır")
    parser.add_argument("--debug", action="store_true", help="Hata ayıklama modu")
    parser.add_argument("--interactive", action="store_true", help="Etkileşimli kabuk")
    parser.add_argument("--output", type=str, help="Çıktı dosyası (CSV/JSON/YAML)")
    parser.add_argument("--config", type=str, help="Yapılandırma dosyası (JSON/TXT/YAML)")
    parser.add_argument("--silent", action="store_true", help="Konsol çıktısını kapatır")
    parser.add_argument("--profile", action="store_true", help="Performans profili")
    parser.add_argument("--trace", action="store_true", help="Komut izleme")
    parser.add_argument("--help", action="store_true", help="Çift dilli yardım (tr/en)")
    parser.add_argument("--plugin", type=str, help="Eklenti yükler")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log seviyesi")
    parser.add_argument("--lang", type=str, default="tr", help="Dil seçimi (tr/en)")
    parser.add_argument("--theme", type=str, default="dark", help="Tema (dark/light)")
    parser.add_argument("--test", type=str, help="Test çalıştırır")
    parser.add_argument("--bytecode", type=str, help="Bytecode derler/çalıştırır")
    parser.add_argument("--secure", action="store_true", help="Güvenli mod")
    parser.add_argument("--monitor", action="store_true", help="Kaynak izleme")
    return parser.parse_args()

async def main():
    """Ana yürütme fonksiyonu."""
    args = parse_args()
    
    # Bağımlılıkları yükle
    required_libs = ["numpy", "pandas", "scipy", "aiohttp", "aiofiles", "pyyaml", "pdfplumber"]
    for lib in required_libs:
        install_library(lib)

    interpreter = PDSXInterpreter()

    # Yapılandırmayı yükle
    if args.config:
        await interpreter.load_config(args.config)

    # Log seviyesini ayarla
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.DEBUG))

    # Programı çalıştır
    if args.file:
        await interpreter.load_program(args.file)
        await interpreter.run()
    elif args.interactive:
        await interpreter.interactive_shell()
    elif args.test:
        # Test çalıştırma (pytest entegrasyonu)
        import pytest
        sys.exit(pytest.main([args.test]))
    elif args.bytecode:
        # Bytecode çalıştırma (bytecode_engine.py ile)
        from bytecode_engine import BytecodeEngine
        engine = BytecodeEngine(interpreter)
        await engine.execute_bytecode(args.bytecode)

if __name__ == "__main__":
    asyncio.run(main())