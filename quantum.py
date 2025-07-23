# pdsxq_advanced_quantum.py - PDS-X Gelişmiş Kuantum Simülatör Modülü
# PDS-X BASIC v14u için gerçekçi ve ileri düzey kuantum programlama yetenekleri sağlar.
# Qiskit ve Cirq gibi kütüphanelerle entegre.

import numpy as np # NumPy 1.26.4 uyumlu olmalı
import math
import random
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import asyncio # Asenkron işlemler için

# PDS-X Exception'ı dışarıdan import et
try:
    from pdsx_unified_exception import PdsXException # PdsX Interpreter'ın kendi exception'ı
    # Eğer auto_importerX'ten geliyorsa, PdsXException olarak tanımlanmalı
    # Eğer bu modül pdsx_exception2'den alıyorsa:
    # from pdsx_exception2 import PdsXException
except ImportError:
    class PdsXException(Exception):
        def __init__(self, message, code="ERR_UNKNOWN", context=None):
            super().__init__(message)
            self.code = code
            self.context = context or {}
            
# Logger instance
log = logging.getLogger("pdsxq_advanced_quantum")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("pdsxq_advanced_quantum")

# Kuantum Kütüphanesi Importları (Python 3.10 uyumlu versiyonlar)
# Genellikle Qiskit ve Cirq'in en son versiyonları Python 3.10 ile uyumludur.
# AutoImporter'ın bunları kurduğundan emin olmalıyız.
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile, assemble, Aer
    from qiskit.visualization import plot_histogram, plot_bloch_multivector # Görselleştirmeler için
    from qiskit.providers.aer import AerSimulator # AerSimulator for more control
    log.info(f"Qiskit yüklendi: {qiskit.__version__}")
except ImportError:
    QuantumCircuit = Aer = transpile = assemble = plot_histogram = plot_bloch_multivector = AerSimulator = None
    log.error("Qiskit kütüphanesi yüklenemedi. Kuantum simülasyonu sınırlı/devre dışı.")

try:
    import cirq
    from cirq import Simulator as CirqSimulator, LineQubit, Circuit
    log.info(f"Cirq yüklendi: {cirq.__version__}")
except ImportError:
    cirq = CirqSimulator = LineQubit = Circuit = None
    log.error("Cirq kütüphanesi yüklenemedi. Bazı gelişmiş simülasyonlar devre dışı.")

# NumPy Sürümü Kontrolü (AutoImporter'a göre)
# AutoImporter'dan gelen NumPy sürümü 1.26.4
# if np.__version__ != '1.26.4':
#     log.warning(f"NumPy sürümü beklenenden farklı: {np.__version__}. (Beklenen: 1.26.4)")

class PDSXAdvancedQuantumSimulator:
    """
    PDS-X BASIC için gerçekçi ve gelişmiş kuantum simülatörü çekirdeği.
    Qiskit veya Cirq gibi kütüphaneleri kullanarak kuantum devrelerini yönetir.
    """
    def __init__(self, logger_instance: logging.Logger = None):
        self._qc = None # Qiskit QuantumCircuit nesnesi
        self._cirq_circuit = None # Cirq Circuit nesnesi
        self._num_qubits = 0
        self.logger = logger_instance if logger_instance else log
        
        # Simülatör backendi seçimi
        if AerSimulator:
            self._qiskit_simulator = AerSimulator() # Daha fazla kontrol için AerSimulator
            self.logger.info("Qiskit AerSimulator kullanılıyor.")
        else:
            self._qiskit_simulator = None
            self.logger.warning("Qiskit AerSimulator mevcut değil, kuantum simülasyonu yapılamayacak.")
        
        if CirqSimulator:
            self._cirq_simulator = CirqSimulator()
            self.logger.info("Cirq Simulator kullanılıyor.")
        else:
            self._cirq_simulator = None
            self.logger.warning("Cirq Simulator mevcut değil, bazı gelişmiş simülasyonlar yapılamayacak.")

    def QINIT(self, num_qubits: int):
        """
        Kuantum sistemini belirli sayıda qubitle başlatır.
        Mevcut devreyi sıfırlar.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise PdsXException("QINIT için qubit sayısı pozitif bir tam sayı olmalıdır.")
        
        self._num_qubits = num_qubits
        if self._qiskit_simulator:
            self._qc = QuantumCircuit(num_qubits, num_qubits) # N qubit, N klasik bit
            self.logger.info(f"Qiskit: {num_qubits} qubitle yeni kuantum devresi başlatıldı.")
        else:
            self.logger.error("Qiskit simülatörü mevcut değil, QINIT başarısız.")
            raise PdsXException("Qiskit mevcut değil, kuantum sistemi başlatılamadı.")
        
        if self._cirq_simulator:
            self._cirq_qubits = [LineQubit(i) for i in range(num_qubits)]
            self._cirq_circuit = Circuit() # Boş Cirq devresi
            self.logger.info(f"Cirq: {num_qubits} qubitle yeni Cirq devresi başlatıldı.")

        self.logger.info(f"Kuantum sistemi {num_qubits} qubitle başlatıldı.")
        
        # Donanım kısıtlamaları hakkında uyarı
        if num_qubits > 25:
            self.logger.warning(f"Yüksek qubit sayısı ({num_qubits}): Bu simülatör belleği tüketebilir veya yavaşlayabilir.")
            self.logger.warning("Tipik ev bilgisayarında 28-30 qubit üzeri pratik değildir (8GB RAM için yaklaşık 28 qubit sınırı).")
        elif num_qubits > 15:
            self.logger.info(f"Orta qubit sayısı ({num_qubits}): Simülasyon süreleri uzayabilir.")

    def QGATE(self, gate_type: str, *qubits_and_params: Union[str, float]):
        """
        Kuantum kapılarını uygular. Qiskit veya Cirq'in kapılarını kullanır.
        Kullanım örnekleri:
        QGATE H Q0
        QGATE CNOT Q0 Q1
        QGATE RX Q0 THETA 1.5708
        QGATE U3 Q0 THETA 0.5 PHI 1.0 LAMBDA 1.5
        QGATE MEASURE Q0 C0 # Ölçüm de bir 'kapı' olarak eklenebilir, klasik bite düşürme
        QGATE SWAP Q0 Q1
        QGATE TOFFOLI Q0 Q1 Q2 # CCNOT
        """
        if self._qc is None or self._cirq_circuit is None:
            raise PdsXException("Kuantum sistemi başlatılmadı. Önce QINIT kullanın.")
        
        gate_type_upper = gate_type.upper()
        
        try:
            # Qubit indekslerini ve klasik bit indekslerini ayrıştırma yardımcı fonksiyonu
            def parse_qubit_and_cbit(arg_str: str) -> Tuple[int, Optional[int]]:
                if arg_str.startswith('Q'): return int(arg_str[1:]), None
                if arg_str.startswith('C'): return None, int(arg_str[1:])
                raise PdsXException(f"Geçersiz qubit/cbit formatı: {arg_str}")

            if gate_type_upper == "H":
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                self._qc.h(target_q_idx)
                self._cirq_circuit.append(cirq.H(self._cirq_qubits[target_q_idx]))
            elif gate_type_upper == "X":
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                self._qc.x(target_q_idx)
                self._cirq_circuit.append(cirq.X(self._cirq_qubits[target_q_idx]))
            # ... (Diğer tek qubit kapıları: Y, Z, S, T, Sdg, Tdg)
            elif gate_type_upper == "RX":
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                if str(qubits_and_params[1]).upper() != "THETA": raise PdsXException("RX için THETA parametresi bekleniyor.")
                theta = float(qubits_and_params[2])
                self._qc.rx(theta, target_q_idx)
                self._cirq_circuit.append(cirq.rx(theta)(self._cirq_qubits[target_q_idx]))
            # ... (Diğer rotasyon kapıları: RY, RZ)
            elif gate_type_upper == "U3": # Genel tek-qubit kapısı
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                # Argümanları parse et: THETA val, PHI val, LAMBDA val
                params = {}
                for i in range(1, len(qubits_and_params), 2):
                    if i + 1 < len(qubits_and_params):
                        params[str(qubits_and_params[i]).upper()] = float(qubits_and_params[i+1])
                theta = params.get("THETA", 0.0)
                phi = params.get("PHI", 0.0)
                lamb = params.get("LAMBDA", 0.0)
                self._qc.u(theta, phi, lamb, target_q_idx)
                # Cirq'te bu kapıyı doğrudan karşılığı olmayabilir, temel kapılarla inşa edilmelidir.
                self.logger.warning("Cirq'te U3 doğrudan desteklenmeyebilir.")

            elif gate_type_upper == "CNOT":
                control_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[1]))
                self._qc.cx(control_q_idx, target_q_idx)
                self._cirq_circuit.append(cirq.CNOT(self._cirq_qubits[control_q_idx], self._cirq_qubits[target_q_idx]))
            elif gate_type_upper == "SWAP":
                q_idx1, _ = parse_qubit_and_cbit(str(qubits_and_params[0]))
                q_idx2, _ = parse_qubit_and_cbit(str(qubits_and_params[1]))
                self._qc.swap(q_idx1, q_idx2)
                self._cirq_circuit.append(cirq.SWAP(self._cirq_qubits[q_idx1], self._cirq_qubits[q_idx2]))
            elif gate_type_upper == "TOFFOLI": # CCNOT
                q_idx1, _ = parse_qubit_and_cbit(str(qubits_and_params[0])) # Kontrol 1
                q_idx2, _ = parse_qubit_and_cbit(str(qubits_and_params[1])) # Kontrol 2
                target_q_idx, _ = parse_qubit_and_cbit(str(qubits_and_params[2])) # Hedef
                self._qc.ccx(q_idx1, q_idx2, target_q_idx)
                self._cirq_circuit.append(cirq.TOFFOLI(self._cirq_qubits[q_idx1], self._cirq_qubits[q_idx2], self._cirq_qubits[target_q_idx]))
            
            elif gate_type_upper == "MEASURE": # Kuantum devresine ölçüm kapısı ekler
                target_q_idx, c_bit_idx = parse_qubit_and_cbit(str(qubits_and_params[0]))
                if c_bit_idx is None:
                    raise PdsXException("MEASURE komutu için klasik bit (C<idx>) belirtilmeli.")
                self._qc.measure(target_q_idx, c_bit_idx) # Qiskit'te ölçüm böyle eklenir.
                self._cirq_circuit.append(cirq.measure(self._cirq_qubits[target_q_idx], key=f"q{target_q_idx}_measurement"))
                self.logger.info(f"Devreye ölçüm kapısı eklendi: Q{target_q_idx} -> C{c_bit_idx}")
                return # Bu bir ölçüm kapısı, aşağıda yürütme yok
            
            else:
                raise PdsXException(f"Desteklenmeyen kuantum kapısı: {gate_type}")
            
            self.logger.info(f"Kuantum Kapısı '{gate_type}' devreyere eklendi. Qubits: {qubits_and_params}")

        except PdsXException:
            raise # Kendi exception'ımızı yeniden fırlat
        except (IndexError, ValueError) as e:
            raise PdsXException(f"QGATE sözdizimi hatası: {gate_type} {qubits_and_params} -> {e}")
        except Exception as e:
            self.logger.error(f"QGATE yürütme hatası: {e}")
            raise PdsXException(f"QGATE yürütme hatası: {e}")

    async def QRUN_SIMULATOR(self, shots: int = 1024, simulator_type: str = "QISKIT") -> Dict[str, int]:
        """
        Mevcut kuantum devresini simülatör üzerinde çalıştırır ve sonuçları döndürür.
        Bu komut, QGATE komutlarından sonra devreyi çalıştırmak için çağrılır.
        Örn: QRUN_SIMULATOR SHOTS 1024 TYPE CIRQ
        """
        if self._qc is None:
            raise PdsXException("Kuantum devresi oluşturulmadı. Önce QINIT kullanın.")
        
        simulator_type_upper = simulator_type.upper()
        
        if simulator_type_upper == "QISKIT":
            if not self._qiskit_simulator:
                raise PdsXException("Qiskit AerSimulator mevcut değil.")
            try:
                # Transpile etme (isteğe bağlı ama gerçekçilik için önemli)
                transpiled_circuit = transpile(self._qc, self._qiskit_simulator)
                
                # Async olarak çalıştırmak için loop.run_in_executor kullan
                loop = asyncio.get_event_loop()
                job = await loop.run_in_executor(None, self._qiskit_simulator.run, transpiled_circuit, shots=shots)
                
                # Sonuçları al
                result = await loop.run_in_executor(None, job.result)
                counts = result.get_counts(self._qc)
                
                self.logger.info(f"Qiskit simülasyonu tamamlandı ({shots} atış). Sonuçlar: {counts}")
                return counts
            except Exception as e:
                self.logger.error(f"Qiskit simülasyon hatası: {e}")
                raise PdsXException(f"Qiskit simülasyon hatası: {e}")
        
        elif simulator_type_upper == "CIRQ":
            if not self._cirq_simulator or not self._cirq_circuit:
                raise PdsXException("Cirq Simulator mevcut değil.")
            try:
                # Cirq'te simulate komutu blocking'dir, executor ile çalıştıralım
                loop = asyncio.get_event_loop()
                # simulate_samples ile doğrudan örnekler alabiliriz
                results = await loop.run_in_executor(None, self._cirq_simulator.run, self._cirq_circuit, repetitions=shots)
                
                # Cirq'ten sonuçları sayımlara dönüştürme (Qiskit formatına benzer)
                # Cirq sonuçları DataFrame olarak gelir, 'key'lere göre gruplayıp sayacağız.
                counts = results.histogram(key=lambda k: k) # Tüm ölçüm anahtarlarını histograma çevir
                # Binary string'e dönüştürmek için:
                # counts = {bin(k)[2:].zfill(self._num_qubits): v for k, v in cirq_counts.items()}
                # Cirq histogramı doğrudan iyi bir formatta olabilir, basitleştirelim.
                
                processed_counts = {}
                for measure_key, count in counts.items():
                    # ölçüm anahtarı genellikle tuple (0,1,0) veya int formatında gelir.
                    # bunu PDS-X'in anlayacağı binary string'e çevirelim
                    if isinstance(measure_key, int):
                         binary_repr = bin(measure_key)[2:].zfill(self._num_qubits)
                         processed_counts[binary_repr] = count
                    else: # Eğer tuple veya başka bir format ise, daha karmaşık bir dönüşüm gerekir
                        # Basitlik için sadece tek bir ölçüm anahtarını varsayalım veya Cirq'in varsayılanını kullanalım
                        self.logger.warning(f"Cirq histogram anahtarı beklenenden farklı format: {measure_key}")
                        processed_counts[str(measure_key)] = count # Direkt stringe çevir
                
                self.logger.info(f"Cirq simülasyonu tamamlandı ({shots} atış). Sonuçlar: {processed_counts}")
                return processed_counts
            except Exception as e:
                self.logger.error(f"Cirq simülasyon hatası: {e}")
                raise PdsXException(f"Cirq simülasyon hatası: {e}")
        
        else:
            raise PdsXException(f"Bilinmeyen simülatör tipi: {simulator_type}. 'QISKIT' veya 'CIRQ' olmalı.")

    def QRESET(self):
        """Kuantum devresini sıfırlar."""
        if self._qc is None:
            raise PdsXException("Kuantum sistemi başlatılmadı.")
        self._qc = QuantumCircuit(self._num_qubits, self._num_qubits) # Yeniden başlat
        if self._cirq_circuit:
            self._cirq_circuit = Circuit() # Cirq devresini de sıfırla
        self.logger.info("Kuantum devresi sıfırlandı.")

    def QPRINT_CIRCUIT(self, output_type: str = "TEXT"):
        """Kuantum devresinin görselini veya metin temsilini yazdırır."""
        if self._qc is None:
            self.logger.warning("Kuantum devresi başlatılmadı.")
            print("Kuantum devresi boş.")
            return

        output_type_upper = output_type.upper()
        try:
            if output_type_upper == "TEXT":
                print("\n--- Qiskit Devre Şeması (Metin) ---")
                print(self._qc.draw(output='text'))
                if self._cirq_circuit:
                    print("\n--- Cirq Devre Şeması ---")
                    print(self._cirq_circuit)
            elif output_type_upper == "ASCII" and self._qc:
                print("\n--- Qiskit Devre Şeması (ASCII) ---")
                print(self._qc.draw(output='mpl', style={'filename': 'circuit_ascii.png'})) # ASCII çıktı için matplotlib'e ihtiyacı var
                self.logger.info("ASCII çıktı dosyasına kaydedildi (circuit_ascii.png).")
            # MPL ve diğer görselleştirmeler için dosya kaydetme mantığı eklenebilir.
            else:
                raise PdsXException(f"Desteklenmeyen QPRINT_CIRCUIT çıktı tipi: {output_type}")
        except Exception as e:
            self.logger.error(f"Devre görselleştirme hatası: {e}")
            raise PdsXException(f"Devre görselleştirme hatası: {e}")
            
    async def QVISUALIZE_RESULTS(self, counts: Dict[str, int], plot_type: str = "HISTOGRAM", filename: str = "quantum_results.png"):
        """
        Simülasyon sonuçlarını görselleştirir ve dosyaya kaydeder.
        Kullanım: QVISUALIZE_RESULTS COUNTS <dict_var> TYPE HISTOGRAM FILENAME "plot.png"
        """
        if not plot_histogram and not plot_bloch_multivector:
            self.logger.error("Qiskit görselleştirme araçları yüklenemedi. Görselleştirme yapılamaz.")
            raise PdsXException("Qiskit görselleştirme araçları eksik.")
        
        plot_type_upper = plot_type.upper()
        try:
            if plot_type_upper == "HISTOGRAM":
                if not counts:
                    self.logger.warning("Görselleştirilecek sonuç yok.")
                    return
                # Qiskit'in plot_histogram'ı otomatik olarak matplotlib kullanarak çizim yapar
                fig = plot_histogram(counts)
                # Matplotlib figürü göstermek veya kaydetmek için
                import matplotlib.pyplot as plt
                fig.savefig(filename)
                plt.close(fig) # Figürü kapat
                self.logger.info(f"Histogram '{filename}' olarak kaydedildi.")
            # Bloch küresi için daha karmaşık durum vektörü gerekir, counts'tan doğrudan olmaz.
            # elif plot_type_upper == "BLOCH":
            #     if self._num_qubits > 3: # Çok fazla qubit için Bloch küresi anlamlı değil
            #         self.logger.warning("Bloch küresi görselleştirmesi sadece 1-3 qubit için uygundur.")
            #         return
            #     # Bloch küresi için durumu yeniden simüle etmek gerekebilir veya state_vector'u almak.
            #     # Bu, Qiskit'in statevector_simulator'ü ile yapılır.
            #     pass
            else:
                raise PdsXException(f"Desteklenmeyen görselleştirme tipi: {plot_type}")
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {e}")
            raise PdsXException(f"Görselleştirme hatası: {e}")

# --- PDSXv14uInterpreter ile Entegrasyon için Modül Dışa Aktarımı ---
# PDSXQBasicModule sınıfını kaldırıp doğrudan PDSXAdvancedQuantumModule olarak yapılandırıyoruz
# ve interpreter'ın function_table'ına kaydolacak şekilde ayarlıyoruz.

class PDSXAdvancedQuantumModule:
    """
    PDS-X QBasic dil komutlarını Gelişmiş Kuantum Simülatörüne bağlayan modül.
    Bu sınıf, PDSXv14uInterpreter tarafından doğrudan çağrılacak metodları sağlar
    ve __pdsX_exports__ protokolünü uygular.
    """
    def __init__(self, interpreter_instance: Any):
        self.interpreter = interpreter_instance
        self.simulator = PDSXAdvancedQuantumSimulator(logger_instance=self._get_logger())

    def _get_logger(self):
        """Interpreter'ın logger'ını güvenli bir şekilde alır."""
        if hasattr(self.interpreter, 'logger') and self.interpreter.logger is not None:
            return self.interpreter.logger
        return log # Varsayılan modül logger'ı

    def _get_var_value(self, var_name: str):
        """Interpreter'ın scope'undan değişken değerini alır."""
        # Değişken adı formatını kontrol et: 'MYVAR' veya 'Q0' gibi
        if self.interpreter.current_scope() and var_name in self.interpreter.current_scope():
            return self.interpreter.current_scope()[var_name]
        if self.interpreter.global_vars and var_name in self.interpreter.global_vars:
            return self.interpreter.global_vars[var_name]
        
        # Eğer bir numara veya boolean stringi ise, doğrudan dönüştür
        if re.match(r"^-?\d+(\.\d+)?$", var_name):
            return float(var_name) if '.' in var_name else int(var_name)
        if var_name.upper() == "TRUE": return True
        if var_name.upper() == "FALSE": return False
        
        raise PdsXException(f"Değişken veya geçerli değer '{var_name}' bulunamadı.")

    def _set_var_value(self, var_name: str, value: Any):
        """Interpreter'ın scope'una değişken değeri atar."""
        if self.interpreter.current_scope() and var_name in self.interpreter.current_scope():
            self.interpreter.current_scope()[var_name] = value
        elif self.interpreter.global_vars and var_name in self.interpreter.global_vars:
            self.interpreter.global_vars[var_name] = value
        else: # Yeni değişkeni yerel veya global scope'a ekle (basitlik için yerel)
            self.interpreter.current_scope()[var_name] = value

    # --- PDS-X QBasic Komut Metodları ---
    async def QINIT_CMD(self, num_qubits_str: str):
        """QINIT num_qubits"""
        num_qubits = int(self._get_var_value(num_qubits_str))
        self.simulator.QINIT(num_qubits)
        
    async def QGATE_CMD(self, *args):
        """QGATE H Q0, QGATE CNOT Q0 Q1, QGATE RX Q0 THETA 1.57"""
        gate_type = str(args[0])
        parsed_args = []
        for arg in args[1:]:
            if isinstance(arg, str) and self._is_var_name(arg): # Değişken adıysa değerini al
                parsed_args.append(self._get_var_value(arg))
            else: # Sayı veya qubit ismi gibi doğrudan değerler
                parsed_args.append(arg)
        
        # QGATE'in argüman yapısı biraz karmaşık, direkt QGATE'i çağırmadan önce
        # parametreleri ayrıştırmamız gerekecek.
        # Örneğin, QGATE RX Q0 THETA 1.57 için:
        # gate_type = "RX", qubits_and_params = ["Q0", "THETA", 1.57]
        # Bu, PDS-X interpreter'ının command parser'ının bu argümanları doğru şekilde iletmesini gerektirir.
        
        # Geçici çözüm: args[0] kapı tipi, geri kalanı *qubits_and_params olarak QGATE'e ilet.
        # QGATE ("H", "Q0"), ("CNOT", "Q0", "Q1"), ("RX", "Q0", "THETA", 1.57) şeklinde çağrılır
        # Normalde PDS-X interpreter'ı komutları ayrıştırırken 'parse_qubit_and_cbit' gibi işlevler kullanmalıydı.

        # QGATE metodunun beklediği gibi argümanları hazırlama
        if gate_type.upper() in ["H", "X", "Y", "Z", "U3", "RX", "RY", "RZ", "PHASE"]:
            # Tek qubit kapıları ve parametreli rotasyonlar
            if gate_type.upper() in ["RX", "RY", "RZ", "PHASE"]:
                 # QGATE RX Q0 THETA 1.57 gibi
                if len(parsed_args) != 3 or not (isinstance(parsed_args[2], (int, float))):
                    raise PdsXException(f"{gate_type} için sözdizimi hatası: Q<idx> PARAM_KEYWORD VAL")
                await self.simulator.QGATE(gate_type, parsed_args[0], parsed_args[1], parsed_args[2])
            else: # H,X,Y,Z
                if len(parsed_args) != 1: raise PdsXException(f"{gate_type} için sözdizimi hatası: Q<idx>")
                await self.simulator.QGATE(gate_type, parsed_args[0])

        elif gate_type.upper() in ["CNOT", "SWAP"]:
            if len(parsed_args) != 2: raise PdsXException(f"{gate_type} için sözdizimi hatası: Q<idx1> Q<idx2>")
            await self.simulator.QGATE(gate_type, parsed_args[0], parsed_args[1])

        elif gate_type.upper() == "TOFFOLI": # CCNOT
            if len(parsed_args) != 3: raise PdsXException(f"{gate_type} için sözdizimi hatası: Q<idx1> Q<idx2> Q<idx3>")
            await self.simulator.QGATE(gate_type, parsed_args[0], parsed_args[1], parsed_args[2])
        
        elif gate_type.upper() == "MEASURE": # Ölçüm kapısı ekleme
            if len(parsed_args) != 2: raise PdsXException(f"MEASURE için sözdizimi hatası: Q<idx> C<idx>")
            await self.simulator.QGATE(gate_type, parsed_args[0], parsed_args[1]) # QGATE bunu işleyecek
        else:
            raise PdsXException(f"Bilinmeyen veya desteklenmeyen QGATE tipi: {gate_type}")

    async def QRUN_SIMULATOR_CMD(self, *args):
        """QRUN_SIMULATOR SHOTS 1024 TYPE QISKIT INTO RESULT_VAR"""
        shots = 1024
        sim_type = "QISKIT"
        result_var_name = None

        # Argümanları parse et (anahtar kelime tabanlı)
        i = 0
        while i < len(args):
            arg = str(args[i]).upper()
            if arg == "SHOTS" and i + 1 < len(args):
                shots = int(self._get_var_value(str(args[i+1])))
                i += 2
            elif arg == "TYPE" and i + 1 < len(args):
                sim_type = str(args[i+1]).upper()
                i += 2
            elif arg == "INTO" and i + 1 < len(args):
                result_var_name = str(args[i+1])
                i += 2
            else:
                raise PdsXException(f"QRUN_SIMULATOR için bilinmeyen argüman: {arg}")
        
        if result_var_name is None:
            raise PdsXException("QRUN_SIMULATOR için INTO <var_name> gerekli.")

        counts = await self.simulator.QRUN_SIMULATOR(shots, sim_type)
        self._set_var_value(result_var_name, counts)
        self._get_logger().info(f"Simülasyon sonuçları '{result_var_name}'e kaydedildi.")

    async def QMEASURE_RESULT_CMD(self, target_qubit_str: str, result_var_name: str):
        """
        Devredeki son ölçüm sonuçlarından belirli bir qubitin sonucunu klasik değişkene atar.
        Qiskit'te bu, QRUN_SIMULATOR'dan sonra counts dictionary'sinden ayrıştırılır.
        Bu komut, devreden doğrudan ölçüm yapmaz, simülasyon çıktısını işler.
        Kullanım: QMEASURE_RESULT Q0 INTO <var_name>
        """
        # Qiskit'te tek tek qubit ölçümü, devrenin sonuna MEASURE kapısı eklenip
        # ardından QRUN_SIMULATOR ile tüm devre çalıştırıldıktan sonra counts objesinden okunur.
        # Bu yüzden bu komut, daha önce çalıştırılmış bir QRUN_SIMULATOR'ın sonucuna (bir sözlük) ihtiyaç duyar.
        
        # Varsayım: Simülasyon sonucu bir değişkende tutuluyor (örn. `RESULT_COUNTS`)
        # Bu durumda, PDS-X QBASIC dilinde:
        # QRUN_SIMULATOR SHOTS 1024 INTO RESULT_COUNTS
        # QMEASURE_RESULT Q0 RESULT_COUNTS INTO FINAL_Q0_RESULT
        
        if len(target_qubit_str) < 2 or target_qubit_str[0].upper() != 'Q':
            raise PdsXException(f"QMEASURE_RESULT için geçersiz qubit formatı: {target_qubit_str}")
        target_q_idx = int(target_qubit_str[1:])

        # Result var_name'in bir dictionary olduğunu varsayarak (counts'ı içeren)
        # Örnek: QMEASURE_RESULT Q0 MY_COUNTS INTO BIT_0_RESULT
        # args[0] = Q0, args[1] = MY_COUNTS, args[2] = INTO, args[3] = BIT_0_RESULT
        
        # result_dict_name = args[1]
        # final_var_name = args[3]
        
        # PDS-X'in komut ayrıştırmasına göre bu methodun argümanlarını ayarlayın.
        # Şimdilik, sadece `counts` objesini doğrudan alıyoruz.
        
        # NOT: Bu metodun çağrılma şekli, PDS-X Interpreter'ın `QMEASURE` komutunu nasıl ayrıştırdığına bağlıdır.
        # Eğer `QMEASURE Q0 RES0` komutu doğrudan `measure_qubit` fonksiyonunu çağıracaksa,
        # bu metodun görevi değişir ve daha çok "post-processing" (işleme sonrası) için kullanılır.
        # Şimdiki tasarımda QMEASURE (kapı ekleyen) ve QRUN_SIMULATOR (simülasyonu çalıştıran) var.
        # Bu QMEASURE_RESULT_CMD, QRUN_SIMULATOR'ın sayım sonuçlarından tek bir qubitin baskın sonucunu çıkarır.

        # QRUN_SIMULATOR'dan alınan 'counts' sözlüğünü bir değişkenden almalıyız
        counts_dict_name = "LAST_Q_COUNTS" # Varsayılan olarak bir önceki QRUN_SIMULATOR'ın sonucu
        if self.interpreter.current_scope() and counts_dict_name in self.interpreter.current_scope():
            counts = self.interpreter.current_scope()[counts_dict_name]
        elif self.interpreter.global_vars and counts_dict_name in self.interpreter.global_vars:
            counts = self.interpreter.global_vars[counts_dict_name]
        else:
            raise PdsXException(f"QMEASURE_RESULT için simülasyon sonuçları bulunamadı (varsayılan '{counts_dict_name}' yok).")
            
        if not isinstance(counts, dict) or not counts:
            raise PdsXException("QMEASURE_RESULT için geçerli simülasyon sonuçları (counts dictionary) gerekli.")

        # Counts objesinden belirli bir qubitin sonucunu çıkar (basit bir yaklaşım)
        # Genellikle counts = {'00': 500, '11': 524} şeklindedir.
        # Belirli bir qubitin sonucu için, o qubitin değeri 0 veya 1 olan durumların toplamına bakılır.
        
        # Örnek: Q0'ı ölçersek, '00' ve '01' durumlarındaki Q0 değeri 0'dır. '10' ve '11' de 1'dir.
        # Bunu doğru yapmak için counts anahtarlarının formatını bilmek gerekir ('00', '01' gibi ikili stringler)
        
        # En çok ölçülen sonucu bul
        most_common_outcome = max(counts, key=counts.get) # Örn: '00' veya '11'

        # Bu en çok çıkan sonucun target_q_idx'deki bitini al
        # Qiskit'te sağdan sola indeksleme (0 en sağdaki bit)
        if len(most_common_outcome) != self.simulator._num_qubits:
             raise PdsXException("QMEASURE_RESULT: Simülasyon çıktısı qubit sayısıyla uyuşmuyor.")
             
        # İndeksleme (PDS-X'in 'Q0' = 0. qubit gibi)
        # Qiskit'te string '00' ise 0. qubit sağda, 1. qubit soldadır.
        # Yani '00' stringinde Q0 = 0, Q1 = 0. '01' stringinde Q0=1, Q1=0.
        # most_common_outcome[self.simulator._num_qubits - 1 - target_q_idx]
        
        result_bit = int(most_common_outcome[self.simulator._num_qubits - 1 - target_q_idx])

        self._set_var_value(result_var_name, result_bit)
        self._get_logger().info(f"Simülasyon sonucundan Qubit {target_q_idx} için sonuç '{result_bit}' '{result_var_name}'e atandı.")
        
    async def QRESET_CMD(self):
        """QRESET"""
        self.simulator.QRESET()

    async def QPRINT_CMD(self, output_type_str: str = "TEXT"):
        """QPRINT (Kuantum durumunu yazdırır) veya QPRINT CIRCUIT [TYPE TEXT|ASCII] (Devreyi yazdırır)"""
        # Argümanları kontrol et
        if output_type_str.upper() == "STATE": # QPRINT STATE
            if self.simulator._qc:
                # Qiskit'in durum vektörü simülatörünü çalıştırıp anlık durumu al
                try:
                    qiskit_state_sim = AerSimulator(method='statevector')
                    transpiled_circuit = transpile(self.simulator._qc, qiskit_state_sim)
                    job = await asyncio.get_event_loop().run_in_executor(None, qiskit_state_sim.run, transpiled_circuit)
                    statevector = await asyncio.get_event_loop().run_in_executor(None, job.result().get_statevector, transpiled_circuit)
                    
                    output_str = f"Kuantum Durumu ({self.simulator._num_qubits} Qubit):\n"
                    for i, amp in enumerate(statevector):
                        basis_state = bin(i)[2:].zfill(self.simulator._num_qubits)
                        prob = abs(amp)**2
                        output_str += f"  |{basis_state}>: ({amp.real:.4f} {'+' if amp.imag >= 0 else ''}{amp.imag:.4f}j) [Olasılık: {prob:.4f}]\n"
                    print(output_str)
                    self._get_logger().info("Kuantum durum vektörü yazdırıldı.")
                except Exception as e:
                    self._get_logger().error(f"QPRINT STATE hatası: {e}")
                    raise PdsXException(f"QPRINT STATE hatası: {e}")
            else:
                 print("Kuantum sistemi başlatılmadı.")
        elif output_type_str.upper() == "CIRCUIT": # QPRINT CIRCUIT [TYPE TEXT|ASCII]
            # Bu komut için parser'ın QPRINT CIRCUIT TYPE TEXT gibi gelmesini bekler.
            # Şu an için QPRINT CIRCUIT (varsayılan TEXT) veya QPRINT CIRCUIT ASCII olarak düşünelim.
            # Normalde PDS-X interpreter'ı bunu QPRINT("CIRCUIT", "TEXT") gibi çevirmeli.
            
            # Argümanları tek bir string olarak alıp ayırmamız gerekiyor (eğer PDS-X parser'ı böyle veriyorsa)
            # Örneğin: QPRINT CIRCUIT TYPE ASCII -> output_type_str = "CIRCUIT TYPE ASCII"
            
            parts = output_type_str.split()
            cmd_part = parts[0].upper() # CIRCUIT
            type_part = "TEXT" # Varsayılan
            
            if len(parts) > 2 and parts[1].upper() == "TYPE":
                type_part = parts[2].upper()

            if cmd_part == "CIRCUIT":
                await self.simulator.QPRINT_CIRCUIT(type_part)
            else:
                raise PdsXException(f"QPRINT için geçersiz alt komut: {output_type_str}")
        else:
            raise PdsXException(f"QPRINT için geçersiz çıktı tipi: {output_type_str}. 'STATE' veya 'CIRCUIT [TYPE TEXT|ASCII]' olmalı.")

    async def QVISUALIZE_RESULTS_CMD(self, *args):
        """
        Simülasyon sonuçlarını görselleştirir ve dosyaya kaydeder.
        Kullanım: QVISUALIZE_RESULTS COUNTS <dict_var_name> TYPE HISTOGRAM FILENAME "plot.png"
        """
        if not self.simulator._qiskit_simulator:
            raise PdsXException("Qiskit görselleştirme için hazır değil.")
        
        # Argümanları parse et
        counts_var_name = None
        plot_type = "HISTOGRAM"
        filename = "quantum_results.png"

        i = 0
        while i < len(args):
            arg = str(args[i]).upper()
            if arg == "COUNTS" and i + 1 < len(args):
                counts_var_name = str(args[i+1])
                i += 2
            elif arg == "TYPE" and i + 1 < len(args):
                plot_type = str(args[i+1])
                i += 2
            elif arg == "FILENAME" and i + 1 < len(args):
                filename = str(args[i+1])
                i += 2
            else:
                raise PdsXException(f"QVISUALIZE_RESULTS için bilinmeyen argüman: {arg}")
        
        if counts_var_name is None:
            raise PdsXException("QVISUALIZE_RESULTS için COUNTS <dict_var_name> gerekli.")

        counts_data = self._get_var_value(counts_var_name)
        if not isinstance(counts_data, dict):
            raise PdsXException(f"Değişken '{counts_var_name}' bir sözlük (counts) değil.")

        await self.simulator.QVISUALIZE_RESULTS(counts_data, plot_type, filename)
        self._get_logger().info(f"Kuantum sonuçları '{filename}' olarak görselleştirildi.")

    # --- PDS-X'in __pdsX_exports__ protokolünü uygula ---
    __pdsX_exports__ = {
        "functions": {
            "QINIT": QINIT_CMD,
            "QGATE": QGATE_CMD,
            "QRUN_SIMULATOR": QRUN_SIMULATOR_CMD,
            "QMEASURE_RESULT": QMEASURE_RESULT_CMD,
            "QRESET": QRESET_CMD,
            "QPRINT": QPRINT_CMD,
            "QVISUALIZE_RESULTS": QVISUALIZE_RESULTS_CMD
        },
        "classes": {
            # Kuantum modülü kendi içindeki sınıfları dışa aktarmaz, sadece fonksiyonel API sunar.
        },
        "variables": {
            # Modül başlatıldığında global değişkenler tanımlanabilir (örn. PI_QUANTUM = 3.14159)
        }
    }


# --- Modülün PDS-X Interpreter'ına Yüklenme Şekli ---
# PDSXv14uInterpreter'ın __init__ metodunda veya LOAD MODULE komutunda bu modül yüklendiğinde,
# __pdsX_exports__ kullanılarak fonksiyonlar function_table'a otomatik olarak eklenir.
# Örn:
# from pdsxq_advanced_quantum import PDSXAdvancedQuantumModule
# self.quantum_module_instance = PDSXAdvancedQuantumModule(self)
# self.function_table.update(self.quantum_module_instance.__pdsX_exports__["functions"])

# Yorumlayıcının Komut Ayrıştırması için Notlar:
# Yorumlayıcının execute_command metodu, komutları doğru şekilde parse etmeli ve bu fonksiyonları çağırmalıdır.
# Örneğin:
# "QINIT 2" -> self.function_table["QINIT"](2)
# "QGATE H Q0" -> self.function_table["QGATE"]("H", "Q0")
# "QGATE RX Q0 THETA 1.57" -> self.function_table["QGATE"]("RX", "Q0", "THETA", "1.57")
# "QRUN_SIMULATOR SHOTS 1024 TYPE QISKIT INTO MY_COUNTS" -> self.function_table["QRUN_SIMULATOR"]("SHOTS", "1024", "TYPE", "QISKIT", "INTO", "MY_COUNTS")
# (Değişken değerlerinin evaluate_expression ile alınması parse_args_for_qgate gibi yardımcı fonksiyonlarla yapılmalı)