# -*- coding: utf-8 -*-
"""
pdsXuextended.py - Birleştirilmiş ve Genişletilmiş PDS-X Yorumlayıcı

Bu dosya, "pdsX" ile başlayan ana modüllerin tüm temel özelliklerini tek bir
birleşik yorumlayıcı sınıfında toplar. Aşağıdaki modüllerin içerdikleri sınıf,
metot ve fonksiyonlar bu yorumlayıcıya aktarılmıştır:

  - pdsXuv14.py (PdsXv14uInterpreter)
  - pdsxeu_v14.py (PdsXe_uv14)
  - pdsx_interpreter.py (PDSXInterpreter)
  - pdsXuv14xxx.py (geliştirilmiş eklenti yöneticileri, core entegrasyonları)
  - pdsX_iyilestirme.py (iyileştirme, optimizasyon araçları)

Not: pdsX_exection.py ve pdsX_exection2.py modülleri bu birleştirmeye dahil edilmemiştir.
"""

import asyncio
from typing import Optional
import os
import importlib
import pkgutil
import inspect
from bytecode_compiler import BytecodeCompiler
from bytecode_manager import BytecodeManager
from pdsXuv14 import PdsXv14uInterpreter as Interp14u
from pdsxeu_v14 import PdsXe_uv14 as InterpXe
from pdsx_interpreter import PdsXInterpreter as Interp15
from pdsXuv14xxx import PluginManager as XxxPluginManager
from pdsX_iyilestirme import PDSXIntegrator as IyiIntegrator
from pdsXuv14 import _add_core_module_exports_to_tables


class PDSXExtended(Interp14u, InterpXe if InterpXe is not None else object, Interp15 if Interp15 is not None else object):
    """Genişletilmiş PDS-X yorumlayıcı sınıfı."""
    def __init__(self, **kwargs):
        # 14u yorumlayıcı başlat
        Interp14u.__init__(self, **kwargs)
        # Xe versiyonu varsa başlat
        if InterpXe:
            InterpXe.__init__(self)
        # ek iyileştirme modülleri
        self.iyilestirme = IyiIntegrator()
        self.xxx_plugins = XxxPluginManager(self)
        # Otomatik pdsX modüllerinden export entegrasyonu (dinamik)
        package = __package__ or ''
        module_dir = os.path.dirname(__file__)
        own_name = os.path.splitext(os.path.basename(__file__))[0].lower()
        for finder, mod_name, ispkg in pkgutil.iter_modules([module_dir]):
            m_lower = mod_name.lower()
            if m_lower.startswith('pdsx') and m_lower != own_name:
                full_name = f"{package}.{mod_name}" if package else mod_name
                try:
                    mod = importlib.import_module(full_name)
                    exports = getattr(mod, '__pdsX_exports__', {})
                    for fname, func in exports.get('functions', {}).items():
                        self.function_table[fname.upper()] = func
                    for cname, cls in exports.get('classes', {}).items():
                        self.type_table[cname.upper()] = cls
                except Exception:
                    continue
        # Çekirdek modül fonksiyon ve tip tablolarını ekle
        _add_core_module_exports_to_tables(self.function_table, self.type_table)
        # Bytecode compiler ve manager başlat
        self.bytecode_compiler = BytecodeCompiler()
        self.bytecode_manager = BytecodeManager(self)
        self.bytecode_manager.start_async_loop()
        # Bytecode manager core özelliklerini kaydet
        try:
            self.bytecode_manager.register_core_features(self.core)
        except Exception:
            pass
        # Fonksiyon tablolarına bytecode işlemleri ekle
        self.function_table['COMPILE'] = self.compile_code
        self.function_table['EXECUTE_BYTECODE'] = self.execute_bytecode
        self.function_table['EXEC_BC'] = self.execute_bytecode

    # Ortak API metodları
    def run(self, *args, **kwargs):
        """Birleştirilmiş çalıştırma: önce senkron, yoksa asenkron."""
        # Öncelikle 14u senkron run
        if hasattr(super(), 'run'):
            try:
                return super().run(*args, **kwargs)
            except Exception:
                pass
        # Asenkron run varsa
        if hasattr(self, 'run_async'):
            return asyncio.run(self.run_async(*args, **kwargs))
        raise RuntimeError("Çalıştırma yöntemi bulunamadı")

    def execute_command(self, command: str, scope_name: Optional[str] = None):
        """Birleştirilmiş komut çalıştırma: önce 14u, sonra Xe, sonra 15."""
        # 14u ve pdsXuv14 içindeki execute_command
        if hasattr(Interp14u, 'execute_command'):
            res = super().execute_command(command, scope_name)
            if inspect.iscoroutine(res):
                return asyncio.run(res)
            return res
        # Xe sürümü
        if InterpXe and hasattr(InterpXe, 'execute'):  # Xe execute
            res = InterpXe.execute(self, command)
            if inspect.iscoroutine(res):
                return asyncio.run(res)
            return res
        # placeholder 15
        if Interp15 and hasattr(Interp15, 'execute_command'):
            res = Interp15.execute_command(self, command)
            if inspect.iscoroutine(res):
                return asyncio.run(res)
            return res
        raise RuntimeError(f"Komut çalıştıralamadı: {command}")

    def compile_code(self, code: str) -> str:
        """Kod derlenir ve bytecode_id döner."""
        return self.bytecode_compiler.compile(code)

    def execute_bytecode(self, bytecode_id: str) -> Optional[any]:
        """Derlenmiş bytecode'u yürütür ve sonucu döner."""
        return asyncio.run(self.bytecode_manager.execute(bytecode_id))

    def repl(self):
        """Birleştirilmiş REPL: senkron ve asenkron komutlar arasında geçiş yapar."""
        print("PDS-X Extended REPL (diller: en/tr)")
        buffer = []
        while True:
            prompt = '... ' if buffer else 'PDS-X> '
            line = input(prompt)
            if not buffer and line.strip().lower() in ['exit', 'quit']:
                break
            # Çok satırlı komut desteği
            if line.endswith(':') or (buffer and line.startswith(' ')):
                buffer.append(line)
                continue
            if buffer:
                if line:
                    buffer.append(line)
                    continue
                code = '\n'.join(buffer)
                buffer.clear()
            else:
                code = line
            try:
                res = self.execute_command(code)
                if res is not None:
                    print(res)
            except Exception as e:
                print(f"[HATA] {e}")


if __name__ == '__main__':
    # Örnek kullanım
    ext = PDSXExtended()
    print("PDS-X Extended Interpreter hazır.")
    # REPL başlat
    ext.repl()
