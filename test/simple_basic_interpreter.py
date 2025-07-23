"""
Basit PDS-X BASIC Interpreter - program_manager.py için
"""

import re
import ast
import sys
import time
from typing import Dict, Any, List, Optional


class SimpleBasicInterpreter:
    """Basit PDS-X BASIC interpreter (.basx/.libx/.pdsx dosyaları için)"""
    
    def __init__(self):
        self.variables = {}  # Değişkenler
        self.program_lines = []  # Program satırları  
        self.program_counter = 0  # Şu anki satır
        self.stack = []  # Çağrı yığını
        self.loop_stack = []  # Döngü yığını
        self.if_stack = []  # IF yığını
        self.function_table = {
            # Matematik fonksiyonları
            'SIN': lambda x: __import__('math').sin(x),
            'COS': lambda x: __import__('math').cos(x),
            'TAN': lambda x: __import__('math').tan(x),
            'SQR': lambda x: __import__('math').sqrt(x),
            'ABS': lambda x: abs(x),
            'INT': lambda x: int(x),
            'RND': lambda: __import__('random').random(),
            'LEN': lambda x: len(str(x)),
            'MID': lambda s, start, length=None: str(s)[start-1:start-1+length] if length else str(s)[start-1:],
            'LEFT': lambda s, n: str(s)[:n],
            'RIGHT': lambda s, n: str(s)[-n:],
            'STR': lambda x: str(x),
            'VAL': lambda x: float(x) if '.' in str(x) else int(x),
            'CHR': lambda x: chr(int(x)),
            'ASC': lambda x: ord(str(x)[0]),
            'TIME': lambda: time.strftime("%H:%M:%S"),
            'DATE': lambda: time.strftime("%Y-%m-%d")
        }
        
    def run_code(self, code: str) -> bool:
        """PDS-X BASIC kodu çalıştır"""
        try:
            # Kodu satırlara böl
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            
            # Satır numaralarını ayıkla
            self.program_lines = []
            for line in lines:
                # Satır numarası varsa (10 PRINT "Merhaba")
                if re.match(r'^\d+\s+', line):
                    line_num, command = line.split(' ', 1)
                    self.program_lines.append((int(line_num), command))
                else:
                    # Satır numarası yoksa (PRINT "Merhaba")
                    self.program_lines.append((len(self.program_lines) * 10, line))
            
            # Satır numarasına göre sırala
            self.program_lines.sort(key=lambda x: x[0])
            
            # Programı çalıştır
            self.program_counter = 0
            while self.program_counter < len(self.program_lines):
                line_num, command = self.program_lines[self.program_counter]
                
                # Komut çalıştır
                if not self.execute_command(command):
                    break
                    
                self.program_counter += 1
                
            return True
            
        except Exception as e:
            print(f"[PDS-X BASIC] ❌ Çalıştırma hatası: {e}")
            return False
    
    def execute_command(self, command: str) -> bool:
        """Tek bir komut çalıştır"""
        command = command.strip()
        if not command:
            return True
            
        command_upper = command.upper()
        
        try:
            # PRINT komutu
            if command_upper.startswith('PRINT'):
                return self._handle_print(command)
                
            # LET komutu (değişken atama)
            elif command_upper.startswith('LET') or '=' in command:
                return self._handle_let(command)
                
            # IF komutu
            elif command_upper.startswith('IF'):
                return self._handle_if(command)
                
            # ELSE komutu
            elif command_upper == 'ELSE':
                return self._handle_else()
                
            # ENDIF komutu
            elif command_upper == 'ENDIF':
                return self._handle_endif()
                
            # FOR komutu
            elif command_upper.startswith('FOR'):
                return self._handle_for(command)
                
            # NEXT komutu
            elif command_upper.startswith('NEXT'):
                return self._handle_next(command)
                
            # GOTO komutu
            elif command_upper.startswith('GOTO'):
                return self._handle_goto(command)
                
            # END komutu
            elif command_upper == 'END':
                return False  # Programı sonlandır
                
            # CLS komutu
            elif command_upper == 'CLS':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                return True
                
            # INPUT komutu
            elif command_upper.startswith('INPUT'):
                return self._handle_input(command)
                
            # REM komutu (yorum)
            elif command_upper.startswith('REM') or command.startswith("'"):
                return True  # Yorumları atla
                
            # Bilinmeyen komut
            else:
                print(f"[PDS-X BASIC] ⚠️ Bilinmeyen komut: {command}")
                return True
                
        except Exception as e:
            print(f"[PDS-X BASIC] ❌ Komut hatası '{command}': {e}")
            return True  # Hataya rağmen devam et
    
    def _handle_print(self, command: str) -> bool:
        """PRINT komutunu işle"""
        match = re.match(r'PRINT\s*(.*)', command, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            if expr:
                # Birden fazla öğe (virgül/noktalı virgül ile ayrılmış)
                if ';' in expr or ',' in expr:
                    parts = re.split(r'[;,]', expr)
                    output = []
                    for part in parts:
                        part = part.strip()
                        if part:
                            result = self._evaluate_expression(part)
                            output.append(str(result))
                    print(' '.join(output))
                else:
                    # Tek öğe
                    result = self._evaluate_expression(expr)
                    print(result)
            else:
                print()  # Boş satır
        return True
    
    def _handle_let(self, command: str) -> bool:
        """LET komutunu işle"""
        # LET A = 5 veya A = 5
        if command.upper().startswith('LET'):
            expr = command[3:].strip()
        else:
            expr = command.strip()
            
        match = re.match(r'(\w+)\s*=\s*(.+)', expr)
        if match:
            var_name, value_expr = match.groups()
            value = self._evaluate_expression(value_expr)
            self.variables[var_name] = value
            return True
        return True
    
    def _handle_if(self, command: str) -> bool:
        """IF komutunu işle"""
        match = re.match(r'IF\s+(.+)\s+THEN\s*(.*)', command, re.IGNORECASE)
        if match:
            condition, then_part = match.groups()
            condition_result = self._evaluate_expression(condition)
            
            self.if_stack.append({
                'condition': bool(condition_result),
                'line': self.program_counter
            })
            
            # THEN kısmı varsa çalıştır
            if then_part and bool(condition_result):
                self.execute_command(then_part)
                
            return True
        return True
    
    def _handle_else(self) -> bool:
        """ELSE komutunu işle"""
        if self.if_stack:
            # IF condition false ise ELSE bloğunu çalıştır
            pass
        return True
    
    def _handle_endif(self) -> bool:
        """ENDIF komutunu işle"""
        if self.if_stack:
            self.if_stack.pop()
        return True
    
    def _handle_for(self, command: str) -> bool:
        """FOR komutunu işle"""
        match = re.match(r'FOR\s+(\w+)\s*=\s*(.+)\s+TO\s+(.+)(?:\s+STEP\s+(.+))?', command, re.IGNORECASE)
        if match:
            var_name, start, end, step = match.groups()
            start_val = self._evaluate_expression(start)
            end_val = self._evaluate_expression(end)
            step_val = self._evaluate_expression(step) if step else 1
            
            self.variables[var_name] = start_val
            self.loop_stack.append({
                'type': 'FOR',
                'var': var_name,
                'end': end_val,
                'step': step_val,
                'start_line': self.program_counter
            })
        return True
    
    def _handle_next(self, command: str) -> bool:
        """NEXT komutunu işle"""
        if self.loop_stack and self.loop_stack[-1]['type'] == 'FOR':
            loop = self.loop_stack[-1]
            var_name = loop['var']
            
            # Değişkeni artır
            self.variables[var_name] += loop['step']
            
            # Döngü devam etsin mi?
            if ((loop['step'] > 0 and self.variables[var_name] <= loop['end']) or
                (loop['step'] < 0 and self.variables[var_name] >= loop['end'])):
                # Döngü başına dön
                self.program_counter = loop['start_line']
            else:
                # Döngüyü bitir
                self.loop_stack.pop()
        return True
    
    def _handle_goto(self, command: str) -> bool:
        """GOTO komutunu işle"""
        match = re.match(r'GOTO\s+(\d+)', command, re.IGNORECASE)
        if match:
            target_line = int(match.group(1))
            # Hedef satırı bul
            for i, (line_num, _) in enumerate(self.program_lines):
                if line_num == target_line:
                    self.program_counter = i - 1  # -1 çünkü döngü sonunda +1 olacak
                    break
        return True
    
    def _handle_input(self, command: str) -> bool:
        """INPUT komutunu işle"""
        match = re.match(r'INPUT\s*(?:"([^"]+)")?\s*[;,]?\s*(\w+)', command, re.IGNORECASE)
        if match:
            prompt, var_name = match.groups()
            if prompt:
                user_input = input(prompt + " ")
            else:
                user_input = input("? ")
            
            # Sayı ise sayıya çevir
            try:
                value = float(user_input) if '.' in user_input else int(user_input)
            except ValueError:
                value = user_input  # String olarak kal
                
            self.variables[var_name] = value
        return True
    
    def _evaluate_expression(self, expr: str) -> Any:
        """İfade değerlendirme"""
        expr = expr.strip()
        
        # String literal
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1]
        
        # Sayı
        try:
            if '.' in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            pass
        
        # Değişken
        if expr in self.variables:
            return self.variables[expr]
        
        # Değişken (büyük/küçük harf duyarsız)
        for var_name, var_value in self.variables.items():
            if var_name.upper() == expr.upper():
                return var_value
        
        # Fonksiyon çağrısı
        for func_name, func in self.function_table.items():
            if expr.upper().startswith(func_name + '('):
                # Basit fonksiyon parsing
                try:
                    # Parantez içini al
                    match = re.match(rf'{func_name}\s*\(([^)]*)\)', expr, re.IGNORECASE)
                    if match:
                        args_str = match.group(1)
                        if args_str.strip():
                            # Argümanları değerlendir
                            args = [self._evaluate_expression(arg.strip()) for arg in args_str.split(',')]
                            return func(*args)
                        else:
                            return func()
                except:
                    pass
        
        # Matematiksel ifade
        try:
            # Güvenli eval için namespace oluştur
            safe_namespace = {
                **self.variables,
                **self.function_table,
                '__builtins__': {},
                'abs': abs, 'int': int, 'float': float, 'str': str, 'len': len
            }
            
            # Basit operatörler için
            expr = expr.replace(' AND ', ' and ').replace(' OR ', ' or ').replace(' NOT ', ' not ')
            expr = expr.replace('=', '==').replace('<==', '<=').replace('>==', '>=').replace('!==', '!=')
            
            # Değişken referanslarını namespace'e ekle
            for var_name, var_value in self.variables.items():
                safe_namespace[var_name] = var_value
            
            return eval(expr, safe_namespace)
        except:
            # Son çare: string olarak döndür
            return expr


if __name__ == "__main__":
    # Test
    interpreter = SimpleBasicInterpreter()
    
    test_code = '''
10 PRINT "PDS-X BASIC Test"
20 LET A = 10
30 LET B = 5
40 PRINT "A =", A
50 PRINT "B =", B
60 PRINT "Toplam:", A + B
70 FOR I = 1 TO 3
80 PRINT "Sayı:", I
90 NEXT I
100 END
'''
    
    print("Test çalıştırılıyor...")
    interpreter.run_code(test_code)
