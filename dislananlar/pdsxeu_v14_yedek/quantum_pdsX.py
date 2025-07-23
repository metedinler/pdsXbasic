# pdsxq_advanced_quantum.py - PDS-X Gelişmiş Kuantum Simülatör Modülü
# PDS-X BASIC v14u için gerçekçi ve ileri düzey kuantum programlama yetenekleri sağlar.
# Qiskit ve Cirq gibi kütüphanelerle entegre.

import numpy as np
import math
import random
import logging
import re
import asyncio
import os # Dosya kaydetme için

from typing import List, Dict, Tuple, Any, Optional, Union

# PdsXException'ı dışarıdan import et (PDS-X Interpreter'ın kendi exception sınıfı)
try:
    from pdsx_exception import PdsXException # AutoImporter tarafından sağlanan
except ImportError:
    # Eğer bulunamazsa, modülün kendi içinde temel bir PdsXException tanımla
    class PdsXException(Exception):
        def __init__(self, message, code="ERR_UNKNOWN", context=None):
            super().__init__(message)
            self.code = code
            self.context = context or {}
            
# Modül için logger ayarları
log = logging.getLogger("pdsxq_advanced_quantum")
if not log.handlers:
    # Eğer henüz bir handler ayarlanmamışsa (PDS-X'in ana log sistemi tarafından),
    # varsayılan olarak basit bir konsol handler'ı ekle.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("pdsxq_advanced_quantum")

# Kuantum Kütüphanesi Importları (Python 3.10 ve AutoImporter uyumlu versiyonlar)
# Genellikle Qiskit ve Cirq'in en son versiyonları Python 3.10 ile uyumludur.
# AutoImporter'ın bunları kurduğundan emin olmalıyız.
try:
    import qiskit
    from qiskit import QuantumCircuit, transpile, assemble, Aer
    from qiskit.providers.aer import AerSimulator # Daha fazla kontrol ve metodlar için
    from qiskit.visualization import plot_histogram, plot_bloch_multivector, circuit_drawer # Görselleştirmeler
    # Qiskit 1.0+ için Sampler ve Estimator gibi primitives'ler önerilir, ancak AerSimulator da kullanılabilir.
    log.info(f"Qiskit yüklendi: {qiskit.__version__}")
except ImportError:
    QuantumCircuit = Aer = transpile = assemble = AerSimulator = plot_histogram = plot_bloch_multivector = circuit_drawer = None
    log.error("Qiskit kütüphanesi yüklenemedi. Kuantum simülasyonu sınırlı/devre dışı.")

try:
    import cirq
    from cirq import Simulator as CirqSimulator, LineQubit, Circuit, ops # ops: gate operations
    log.info(f"Cirq yüklendi: {cirq.__version__}")
except ImportError:
    cirq = CirqSimulator = LineQubit = Circuit = ops = None
    log.error("Cirq kütüphanesi yüklenemedi. Bazı gelişmiş simülasyonlar devre dışı.")

# NumPy Sürümü Kontrolü (AutoImporter tarafından 1.26.4 bekleniyor)
if np.__version__ != '1.26.4':
    log.warning(f"NumPy sürümü beklenenden farklı: {np.__version__}. (Beklenen: 1.26.4). Bu durum uyumsuzluklara yol açabilir.")

# --- Kuantum Simülatörü Çekirdeği ---
class PDSXAdvancedQuantumSimulator:
    """
    PDS-X BASIC için gerçekçi ve gelişmiş kuantum simülatörü çekirdeği.
    Qiskit ve Cirq gibi kütüphaneleri kullanarak kuantum devrelerini yönetir ve çalıştırır.
    """
    def __init__(self, logger_instance: logging.Logger = None):
        self._qc: Optional[QuantumCircuit] = None # Qiskit QuantumCircuit nesnesi
        self._cirq_circuit: Optional[Circuit] = None # Cirq Circuit nesnesi
        self._cirq_qubits: Optional[List[LineQubit]] = None # Cirq qubit nesneleri
        self._num_qubits: int = 0
        self.logger = logger_instance if logger_instance else log
        
        # Simülatör backendi seçimi
        if AerSimulator:
            self._qiskit_simulator = AerSimulator() # AerSimulator for more control
            self.logger.info("Qiskit AerSimulator başarıyla başlatıldı.")
        else:
            self._qiskit_simulator = None
            self.logger.error("Qiskit AerSimulator mevcut değil, kuantum simülasyonu yapılamayacak.")
        
        if CirqSimulator:
            self._cirq_simulator = CirqSimulator()
            self.logger.info("Cirq Simulator başarıyla başlatıldı.")
        else:
            self._cirq_simulator = None
            self.logger.error("Cirq Simulator mevcut değil, bazı gelişmiş simülasyonlar yapılamayacak.")

    def QINIT(self, num_qubits: int):
        """
        Kuantum sistemini belirli sayıda qubitle başlatır. Mevcut devreyi sıfırlar.
        num_qubits: Oluşturulacak qubit sayısı.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise PdsXException("QINIT için qubit sayısı pozitif bir tam sayı olmalıdır.")
        
        self._num_qubits = num_qubits
        
        # Qiskit devresini başlat
        if self._qiskit_simulator:
            # Her qubit için bir klasik bit de oluşturulur (ölçüm sonuçları için)
            self._qc = QuantumCircuit(num_qubits, num_qubits)
            self.logger.info(f"Qiskit: {num_qubits} qubitle yeni kuantum devresi başlatıldı.")
        else:
            self.logger.error("Qiskit AerSimulator mevcut değil, QINIT başarısız.")
            raise PdsXException("Qiskit mevcut değil, kuantum sistemi başlatılamadı.")
        
        # Cirq devresini başlat
        if self._cirq_simulator:
            self._cirq_qubits = [LineQubit(i) for i in range(num_qubits)]
            self._cirq_circuit = Circuit() # Boş Cirq devresi
            self.logger.info(f"Cirq: {num_qubits} qubitle yeni Cirq devresi başlatıldı.")

        self.logger.info(f"Kuantum sistemi {num_qubits} qubitle başlatıldı.")
        
        # Donanım kısıtlamaları hakkında uyarı (tasarımsal sınırlama değil, pratik sınırlama)
        if num_qubits > 28: # Ortalama 8GB RAM'li PC için pratik sınır
            self.logger.warning(f"YÜKSEK QUBIT SAYISI UYARISI ({num_qubits}): Bu kadar qubitin simülasyonu belleği tüketebilir ve çok yavaşlayabilir.")
            self.logger.warning("Tipik bir ev bilgisayarında (8GB RAM) 28-30 qubit üzeri simülasyonlar pratik değildir.")
        elif num_qubits > 18:
            self.logger.info(f"ORTA QUBIT SAYISI ({num_qubits}): Simülasyon süreleri uzayabilir.")

    def QGATE(self, gate_type: str, *qubits_and_params: Union[str, float]):
        """
        Kuantum kapılarını mevcut devreye uygular.
        gate_type: Kapı tipi (örn. "H", "CNOT", "RX", "U3", "SWAP", "TOFFOLI", "CRZ", "MEASURE")
        qubits_and_params: Kapının uygulanacağı qubitler ve parametreler.
            Q<idx>: Qubit indeksi.
            C<idx>: Klasik bit indeksi (sadece MEASURE için).
            ANGLE <val>: Açı parametreleri için (THETA, PHI, LAMBDA).
        """
        if self._qc is None or self._cirq_circuit is None:
            raise PdsXException("Kuantum devresi başlatılmadı. Önce QINIT kullanın.")
        
        gate_type_upper = gate_type.upper()
        
        # Qubit indekslerini ve klasik bit indekslerini ayrıştırma yardımcı fonksiyonu
        def parse_q_c_idx(arg_str: str) -> Tuple[Optional[int], Optional[int]]:
            if isinstance(arg_str, str):
                if arg_str.upper().startswith('Q'): return int(arg_str[1:]), None
                if arg_str.upper().startswith('C'): return None, int(arg_str[1:])
            raise PdsXException(f"Geçersiz qubit/cbit formatı: {arg_str}")

        try:
            if gate_type_upper in ["H", "X", "Y", "Z", "S", "T", "SDG", "TDG"]:
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                if target_q_idx >= self._num_qubits: raise PdsXException(f"Qubit indeksi sistem dışı: Q{target_q_idx}")
                
                # Qiskit
                if gate_type_upper == "H": self._qc.h(target_q_idx)
                elif gate_type_upper == "X": self._qc.x(target_q_idx)
                elif gate_type_upper == "Y": self._qc.y(target_q_idx)
                elif gate_type_upper == "Z": self._qc.z(target_q_idx)
                elif gate_type_upper == "S": self._qc.s(target_q_idx)
                elif gate_type_upper == "T": self._qc.t(target_q_idx)
                elif gate_type_upper == "SDG": self._qc.sdg(target_q_idx)
                elif gate_type_upper == "TDG": self._qc.tdg(target_q_idx)
                
                # Cirq
                if self._cirq_simulator:
                    cirq_q = self._cirq_qubits[target_q_idx]
                    if gate_type_upper == "H": self._cirq_circuit.append(cirq.H(cirq_q))
                    elif gate_type_upper == "X": self._cirq_circuit.append(cirq.X(cirq_q))
                    elif gate_type_upper == "Y": self._cirq_circuit.append(cirq.Y(cirq_q))
                    elif gate_type_upper == "Z": self._cirq_circuit.append(cirq.Z(cirq_q))
                    elif gate_type_upper == "S": self._cirq_circuit.append(cirq.S(cirq_q))
                    elif gate_type_upper == "T": self._cirq_circuit.append(cirq.T(cirq_q))
                    elif gate_type_upper == "SDG": self._cirq_circuit.append(cirq.S(cirq_q)**-1) # Sdagger
                    elif gate_type_upper == "TDG": self._cirq_circuit.append(cirq.T(cirq_q)**-1) # Tdagger

            elif gate_type_upper in ["RX", "RY", "RZ"]:
                if len(qubits_and_params) != 3 or str(qubits_and_params[1]).upper() != "ANGLE":
                    raise PdsXException(f"{gate_type_upper} için sözdizimi: Q<idx> ANGLE <val>")
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                angle = float(qubits_and_params[2])
                if target_q_idx >= self._num_qubits: raise PdsXException(f"Qubit indeksi sistem dışı: Q{target_q_idx}")

                if gate_type_upper == "RX": self._qc.rx(angle, target_q_idx)
                elif gate_type_upper == "RY": self._qc.ry(angle, target_q_idx)
                elif gate_type_upper == "RZ": self._qc.rz(angle, target_q_idx)

                if self._cirq_simulator:
                    cirq_q = self._cirq_qubits[target_q_idx]
                    if gate_type_upper == "RX": self._cirq_circuit.append(cirq.rx(angle)(cirq_q))
                    elif gate_type_upper == "RY": self._cirq_circuit.append(cirq.ry(angle)(cirq_q))
                    elif gate_type_upper == "RZ": self._cirq_circuit.append(cirq.rz(angle)(cirq_q))
            
            elif gate_type_upper == "U3": # U3 Q<idx> THETA <val> PHI <val> LAMBDA <val>
                if len(qubits_and_params) != 7 or \
                   str(qubits_and_params[1]).upper() != "THETA" or \
                   str(qubits_and_params[3]).upper() != "PHI" or \
                   str(qubits_and_params[5]).upper() != "LAMBDA":
                    raise PdsXException("U3 için sözdizimi: Q<idx> THETA <val> PHI <val> LAMBDA <val>")
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                theta = float(qubits_and_params[2])
                phi = float(qubits_and_params[4])
                lamb = float(qubits_and_params[6])
                if target_q_idx >= self._num_qubits: raise PdsXException(f"Qubit indeksi sistem dışı: Q{target_q_idx}")

                self._qc.u(theta, phi, lamb, target_q_idx)
                # Cirq'te U3 doğrudan karşılığı olmayabilir, temel kapılarla inşa edilmelidir.
                self.logger.warning("Cirq'te U3 kapısı doğrudan desteklenmeyebilir. Temel kapılarla inşa etmeniz önerilir.")

            elif gate_type_upper == "CX": # CNOT
                if len(qubits_and_params) != 2: raise PdsXException("CX için sözdizimi: Q<control> Q<target>")
                control_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[1]))
                if control_q_idx >= self._num_qubits or target_q_idx >= self._num_qubits: raise PdsXException("CX için qubit indeksleri sistem dışı.")
                if control_q_idx == target_q_idx: raise PdsXException("CX için kontrol ve hedef qubit aynı olamaz.")

                self._qc.cx(control_q_idx, target_q_idx)
                if self._cirq_simulator: self._cirq_circuit.append(cirq.CNOT(self._cirq_qubits[control_q_idx], self._cirq_qubits[target_q_idx]))
            
            elif gate_type_upper == "SWAP":
                if len(qubits_and_params) != 2: raise PdsXException("SWAP için sözdizimi: Q<idx1> Q<idx2>")
                q_idx1, _ = parse_q_c_idx(str(qubits_and_params[0]))
                q_idx2, _ = parse_q_c_idx(str(qubits_and_params[1]))
                if q_idx1 >= self._num_qubits or q_idx2 >= self._num_qubits: raise PdsXException("SWAP için qubit indeksleri sistem dışı.")
                if q_idx1 == q_idx2: raise PdsXException("SWAP için qubitler aynı olamaz.")

                self._qc.swap(q_idx1, q_idx2)
                if self._cirq_simulator: self._cirq_circuit.append(cirq.SWAP(self._cirq_qubits[q_idx1], self._cirq_qubits[q_idx2]))
            
            elif gate_type_upper == "CCNOT": # TOFFOLI
                if len(qubits_and_params) != 3: raise PdsXException("CCNOT için sözdizimi: Q<control1> Q<control2> Q<target>")
                c1_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                c2_idx, _ = parse_q_c_idx(str(qubits_and_params[1]))
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[2]))
                if any(idx >= self._num_qubits for idx in [c1_idx, c2_idx, target_q_idx]): raise PdsXException("CCNOT için qubit indeksleri sistem dışı.")
                if len(set([c1_idx, c2_idx, target_q_idx])) != 3: raise PdsXException("CCNOT için tüm qubitler farklı olmalıdır.")

                self._qc.ccx(c1_idx, c2_idx, target_q_idx)
                if self._cirq_simulator: self._cirq_circuit.append(cirq.TOFFOLI(self._cirq_qubits[c1_idx], self._cirq_qubits[c2_idx], self._cirq_qubits[target_q_idx]))
            
            elif gate_type_upper == "CRZ": # Controlled-RZ
                if len(qubits_and_params) != 3 or str(qubits_and_params[2]).upper() != "ANGLE": # Hata düzeltme: angle olması bekleniyor
                     raise PdsXException("CRZ için sözdizimi: Q<control> Q<target> ANGLE <val>")
                control_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[1]))
                angle = float(qubits_and_params[3]) # Hata düzeltme: angle değeri 3. argümanda değil 4. argümanda.

                if control_q_idx >= self._num_qubits or target_q_idx >= self._num_qubits: raise PdsXException("CRZ için qubit indeksleri sistem dışı.")
                if control_q_idx == target_q_idx: raise PdsXException("CRZ için kontrol ve hedef qubit aynı olamaz.")

                self._qc.crz(angle, control_q_idx, target_q_idx)
                if self._cirq_simulator: self._cirq_circuit.append(cirq.rz(angle).on(self._cirq_qubits[target_q_idx]).controlled_by(self._cirq_qubits[control_q_idx]))

            elif gate_type_upper == "MEASURE": # QGATE MEASURE Q<idx> C<idx>
                if len(qubits_and_params) != 2: raise PdsXException("MEASURE için sözdizimi: Q<idx> C<idx>")
                target_q_idx, _ = parse_q_c_idx(str(qubits_and_params[0]))
                _, c_bit_idx = parse_q_c_idx(str(qubits_and_params[1]))
                if target_q_idx is None or c_bit_idx is None: raise PdsXException("MEASURE için hem qubit hem klasik bit gerekli.")
                if target_q_idx >= self._num_qubits or c_bit_idx >= self._num_qubits: raise PdsXException("MEASURE için indeksler sistem dışı.")

                self._qc.measure(target_q_idx, c_bit_idx)
                if self._cirq_simulator: self._cirq_circuit.append(cirq.measure(self._cirq_qubits[target_q_idx], key=f"q{target_q_idx}_meas"))
                self.logger.info(f"Devreye ölçüm kapısı eklendi: Q{target_q_idx} -> C{c_bit_idx}")

            elif gate_type_upper == "CUSTOM": # QGATE CUSTOM <gate_name> Q<idx...> MATRIX <matrix_var>
                # Bu çok ileri düzey bir komut, matrix_var'ın bir numpy array olması beklenir.
                # args[0] = CUSTOM, args[1] = gate_name, args[2...] = qubits, args[son-1] = MATRIX, args[son] = matrix_var_name
                if len(qubits_and_params) < 3 or str(qubits_and_params[-2]).upper() != "MATRIX":
                    raise PdsXException("CUSTOM gate için sözdizimi: CUSTOM <name> Q<idx...> MATRIX <matrix_var>")
                
                gate_name = str(qubits_and_params[0])
                matrix_var_name = str(qubits_and_params[-1])
                target_q_indices = [parse_q_c_idx(str(q))[0] for q in qubits_and_params[1:-2] if str(q).upper().startswith('Q')]
                
                if not target_q_indices:
                     raise PdsXException("CUSTOM gate için hedef qubit(ler) belirtilmeli.")
                
                # Matrisi PDS-X değişkeninden al
                matrix_data = self.interpreter._get_var_value(matrix_var_name) # Interpreter'dan değişken değeri çekme
                if not isinstance(matrix_data, np.ndarray) or matrix_data.shape[0] != matrix_data.shape[1] or \
                   not math.isclose(matrix_data.shape[0], 2**len(target_q_indices)):
                    raise PdsXException(f"CUSTOM gate için geçersiz matris. Kare olmalı ve 2^N boyutunda olmalı (N: {len(target_q_indices)}).")
                
                from qiskit.quantum_info import Operator
                custom_operator = Operator(matrix_data)
                self._qc.unitary(custom_operator, target_q_indices, label=gate_name)
                
                # Cirq'te özel unitari uygulamak biraz farklıdır, UnitaryGate kullanılabilir.
                if self._cirq_simulator:
                    cirq_qubits_for_gate = [self._cirq_qubits[i] for i in target_q_indices]
                    self._cirq_circuit.append(cirq.MatrixGate(matrix_data).on(*cirq_qubits_for_gate))
                self.logger.info(f"CUSTOM Kapı '{gate_name}' devreye eklendi.")

            else:
                raise PdsXException(f"Desteklenmeyen kuantum kapısı tipi: {gate_type}")
            
            self.logger.info(f"Kuantum Kapısı '{gate_type_upper}' devreye eklendi.")

        except PdsXException:
            raise # Kendi exception'ımızı yeniden fırlat
        except (IndexError, ValueError) as e:
            raise PdsXException(f"QGATE sözdizimi hatası: {gate_type_upper} {qubits_and_params} -> {e}")
        except Exception as e:
            self.logger.error(f"QGATE yürütme hatası: {e}")
            raise PdsXException(f"QGATE yürütme hatası: {e}")

    async def QRUN_SIMULATOR(self, shots: int = 1024, simulator_type: str = "QISKIT") -> Dict[str, int]:
        """
        Mevcut kuantum devresini simülatör üzerinde çalıştırır ve sonuçları döndürür.
        shots: Simülasyon atış sayısı (ölçüm için).
        simulator_type: Kullanılacak simülatör ("QISKIT" veya "CIRQ").
        Dönüş: Ölçüm sonuçlarının sayımları (örn. {'00': 500, '11': 524}).
        """
        if self._qc is None:
            raise PdsXException("Kuantum devresi oluşturulmadı. Önce QINIT kullanın.")
        if self._num_qubits == 0:
            raise PdsXException("Kuantum sistemi başlatılmadı veya qubit sayısı sıfır.")
        
        simulator_type_upper = simulator_type.upper()
        
        if simulator_type_upper == "QISKIT":
            if not self._qiskit_simulator:
                raise PdsXException("Qiskit AerSimulator mevcut değil.")
            try:
                # Transpile etme (isteğe bağlı ama gerçekçilik ve performans için önemli)
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
                results = await loop.run_in_executor(None, self._cirq_simulator.run, self._cirq_circuit, repetitions=shots)
                
                # Cirq sonuçlarını Qiskit formatına benzer sayımlara dönüştürme
                counts = results.histogram(key=lambda k: k) # Tüm ölçüm anahtarlarını histograma çevir
                
                processed_counts = {}
                for measure_key, count in counts.items():
                    # Cirq'ten gelen ölçüm anahtarları (int veya tuple). Binary string'e çevirelim.
                    if isinstance(measure_key, (int, np.integer)):
                        # Qiskit genellikle '001' gibi stringler kullanır.
                        # Qiskit'te qubitler sağdan sola (0 en sağda) sıralanır.
                        # Cirq'ün varsayılan ölçüm anahtarı sırası farklı olabilir, kontrol etmek gerekir.
                        # Varsayalım ki Cirq de sağdan sola (MSB solda, LSB sağda) bir int veriyor.
                        binary_repr = bin(measure_key)[2:].zfill(self._num_qubits)
                        processed_counts[binary_repr] = count
                    else: # Tuple (0,1,0) veya başka bir format ise
                        # Daha karmaşık bir dönüşüm gerekir veya Cirq'in varsayılanını kullanalım.
                        # Şimdilik, stringe çevirerek basitleştirelim.
                        processed_counts[str(measure_key)] = count
                
                self.logger.info(f"Cirq simülasyonu tamamlandı ({shots} atış). Sonuçlar: {processed_counts}")
                return processed_counts
            except Exception as e:
                self.logger.error(f"Cirq simülasyon hatası: {e}")
                raise PdsXException(f"Cirq simülasyon hatası: {e}")
        
        else:
            raise PdsXException(f"Bilinmeyen simülatör tipi: {simulator_type}. 'QISKIT' veya 'CIRQ' olmalı.")

    async def QGET_STATEVECTOR(self) -> np.ndarray:
        """
        Mevcut kuantum devresinin durum vektörünü döndürür.
        Yalnızca durum vektörü simülatörleri ile kullanılabilir ve ölçüm kapıları olmadan çağrılmalıdır.
        """
        if self._qc is None:
            raise PdsXException("Kuantum devresi başlatılmadı.")
        if not self._qiskit_simulator:
            raise PdsXException("Qiskit AerSimulator mevcut değil.")
        
        try:
            # Ölçüm kapılarını geçici olarak kaldırarak statevector al (eğer devrede varsa)
            # Daha doğru bir yaklaşım, ayrı bir statevector simülatör backendi kullanmaktır.
            statevector_sim = AerSimulator(method='statevector')
            transpiled_circuit = transpile(self._qc, statevector_sim)
            
            # Devrede ölçüm kapısı varsa, statevector almadan önce kaldırılması gerekir.
            # Qiskit 1.0+'da get_statevector() doğrudan results objesinden çağrılır.
            
            # job = await asyncio.get_event_loop().run_in_executor(None, statevector_sim.run, transpiled_circuit) # Hata: result()'tan get_statevector çağırılmalı
            
            job = statevector_sim.run(transpiled_circuit)
            result = await asyncio.get_event_loop().run_in_executor(None, job.result)
            statevector = result.get_statevector(transpiled_circuit)
            
            self.logger.info("Kuantum durum vektörü başarıyla alındı.")
            return statevector
        except Exception as e:
            self.logger.error(f"Durum vektörü alma hatası: {e}. Devrede ölçüm kapıları olabilir veya simülatör tipi uygun olmayabilir.")
            raise PdsXException(f"Durum vektörü alma hatası: {e}. Devrede ölçüm kapıları olabilir veya simülatör tipi uygun olmayabilir.")

    def QRESET(self):
        """Kuantum devresini sıfırlar (tüm qubitleri |0> durumuna getirir)."""
        if self._num_qubits == 0:
             raise PdsXException("Kuantum sistemi henüz başlatılmadı (QINIT).")

        if self._qiskit_simulator:
            self._qc = QuantumCircuit(self._num_qubits, self._num_qubits)
            self.logger.info("Qiskit devresi sıfırlandı.")
        if self._cirq_circuit:
            self._cirq_qubits = [LineQubit(i) for i in range(self._num_qubits)] # Qubit objelerini yeniden oluştur
            self._cirq_circuit = Circuit()
            self.logger.info("Cirq devresi sıfırlandı.")
        self.logger.info("Kuantum devresi sıfırlandı.")

    def QDESTROY(self):
        """Kuantum simülatörü örneğini tamamen yok eder (kaynakları serbest bırakır)."""
        self._qc = None
        self._cirq_circuit = None
        self._cirq_qubits = None
        self._num_qubits = 0
        self._qiskit_simulator = None # Simülatör nesnesini de sıfırla
        self._cirq_simulator = None
        self.logger.info("Kuantum simülatörü örneği yok edildi ve kaynaklar serbest bırakıldı.")

    async def QPRINT_CIRCUIT(self, output_type: str = "TEXT", filename: Optional[str] = None):
        """
        Kuantum devresinin görselini veya metin temsilini yazdırır/kaydeder.
        output_type: "TEXT", "ASCII", "MPL" (Matplotlib), "LATEX", "LATEX_SQUARE", "FLIP_TEXT"
        filename: Dosyaya kaydetmek için dosya adı (MPL, LATEX için).
        """
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
            elif output_type_upper in ["ASCII", "MPL", "LATEX", "LATEX_SQUARE", "FLIP_TEXT"]:
                if not circuit_drawer:
                    raise PdsXException("Qiskit devre çizim araçları yüklenemedi. 'matplotlib' veya 'qiskit' paketlerini kontrol edin.")
                
                # MPL ve LATEX için dosya kaydetme
                if output_type_upper in ["MPL", "LATEX", "LATEX_SQUARE"]:
                    if not filename:
                        raise PdsXException(f"'{output_type_upper}' çıktısı için 'FILENAME' belirtilmeli.")
                    if output_type_upper == "MPL":
                        fig = self._qc.draw(output='mpl')
                        import matplotlib.pyplot as plt
                        fig.savefig(filename)
                        plt.close(fig) # Figürü kapat
                    else: # LATEX, LATEX_SQUARE
                        # LaTeX çıktısı doğrudan kaydedilir, matplotlib figürü oluşturmaz
                        self._qc.draw(output=output_type_upper.lower(), filename=filename) # filename parametresi yok
                        # Qiskit'te filename parametresi draw metodunun çıktısına göre değişir.
                        # Doğrusu: file=open(filename, "w") gibi yapmak veya çizim motorunun doğrudan kaydetmesini beklemek.
                        self.logger.info(f"Devre şeması '{output_type_upper}' formatında kaydedildi: {filename} (Manuel kontrol gerekebilir).")
                        return

                else: # Sadece konsola yazdırılabilen tipler
                    print(self._qc.draw(output=output_type_upper.lower()))
                self.logger.info(f"Devre şeması '{output_type_upper}' formatında yazdırıldı.")
            else:
                raise PdsXException(f"Desteklenmeyen QPRINT_CIRCUIT çıktı tipi: {output_type}")
        except PdsXException:
            raise
        except Exception as e:
            self.logger.error(f"Devre görselleştirme hatası: {e}")
            raise PdsXException(f"Devre görselleştirme hatası: {e}")
            
    async def QVISUALIZE_RESULTS(self, counts: Dict[str, int], plot_type: str = "HISTOGRAM", filename: str = "quantum_results.png"):
        """
        Simülasyon sonuçlarını görselleştirir ve dosyaya kaydeder.
        counts: QRUN_SIMULATOR'dan alınan sayımlar sözlüğü.
        plot_type: "HISTOGRAM" veya "BLOCH" (Bloch için Statevector gerekli).
        filename: Kaydedilecek dosya adı (örn. "my_plot.png").
        """
        if not plot_histogram: # plot_bloch_multivector da kontrol edilebilir
            self.logger.error("Qiskit görselleştirme araçları yüklenemedi. Görselleştirme yapılamaz. 'matplotlib' kurulu mu?")
            raise PdsXException("Qiskit görselleştirme araçları eksik.")
        
        plot_type_upper = plot_type.upper()
        
        try:
            if plot_type_upper == "HISTOGRAM":
                if not counts:
                    self.logger.warning("Görselleştirilecek sonuç (counts) boş.")
                    return
                # plot_histogram otomatik olarak matplotlib kullanarak çizim yapar
                fig = plot_histogram(counts)
                import matplotlib.pyplot as plt
                fig.savefig(filename)
                plt.close(fig) # Figürü kapat
                self.logger.info(f"Histogram '{filename}' olarak kaydedildi.")
            elif plot_type_upper == "BLOCH":
                if self._num_qubits > 3:
                    raise PdsXException("Bloch küresi görselleştirmesi sadece 1-3 qubit için uygundur.")
                # Bloch küresi için durum vektörü gereklidir, counts'tan doğrudan olmaz.
                # QGET_STATEVECTOR çağrılmalı ve sonucu buraya iletilmelidir.
                # Bu durumda counts yerine state_vector argümanı beklemelidir.
                self.logger.warning("Bloch küresi için QGET_STATEVECTOR sonucunu kullanmanız önerilir.")
                
                # Geçici olarak mevcut durum vektörünü almaya çalışalım
                try:
                    statevector = await self.QGET_STATEVECTOR()
                    fig = plot_bloch_multivector(statevector)
                    import matplotlib.pyplot as plt
                    fig.savefig(filename)
                    plt.close(fig)
                    self.logger.info(f"Bloch küresi '{filename}' olarak kaydedildi.")
                except Exception as e:
                    raise PdsXException(f"Bloch küresi görselleştirme hatası (statevector alınamadı/uygun değil): {e}")

            else:
                raise PdsXException(f"Desteklenmeyen görselleştirme tipi: {plot_type}. 'HISTOGRAM' veya 'BLOCH' olmalı.")
        except PdsXException:
            raise
        except Exception as e:
            self.logger.error(f"Görselleştirme hatası: {e}")
            raise PdsXException(f"Görselleştirme hatası: {e}")

# --- PDS-X'in Interpreter'ı ile Entegrasyon için Modül Dışa Aktarımı ---
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
        if isinstance(var_name, str):
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
        else: # Yeni değişkeni yerel veya global scope'a ekle (PDS-X Basic davranışına göre)
            self.interpreter.current_scope()[var_name] = value

    # --- PDS-X QBasic Komut Metodları (PDS-X Interpreter tarafından çağrılacak) ---
    async def QINIT_CMD(self, num_qubits_val: Union[str, int]):
        """QINIT num_qubits_val"""
        num_qubits = int(self._get_var_value(str(num_qubits_val)))
        await self.simulator.QINIT(num_qubits) # await ekledik

    async def QGATE_CMD(self, *args):
        """QGATE H Q0, QGATE CNOT Q0 Q1, QGATE RX Q0 ANGLE 1.57, QGATE CUSTOM ..."""
        # Argümanları PDS-X interpreter'ının evaluate_expression'ından gelen ham halleriyle alıyoruz.
        # Bunları QGATE metodunun beklediği formata dönüştürmeliyiz.
        
        processed_args = []
        for arg in args:
            if isinstance(arg, str): # Eğer string ise, değişken adı mı yoksa doğrudan değer mi kontrol et
                try:
                    processed_args.append(self._get_var_value(arg))
                except PdsXException: # Değişken değilse, doğrudan argüman olarak kullan
                    processed_args.append(arg)
            else: # Sayı veya başka bir primitif ise
                processed_args.append(arg)

        # processed_args[0] = gate_type, geri kalanı parametreler.
        gate_type = str(processed_args[0])
        qubits_and_params = processed_args[1:] # Kalan argümanlar
        
        await self.simulator.QGATE(gate_type, *qubits_and_params)

    async def QRUN_SIMULATOR_CMD(self, *args):
        """QRUN_SIMULATOR SHOTS 1024 TYPE QISKIT INTO RESULT_VAR"""
        # Argümanlar PDS-X tarafından ayrıştırılmış olarak gelir.
        # Örnek: QRUN_SIMULATOR_CMD("SHOTS", "1024", "TYPE", "QISKIT", "INTO", "MY_COUNTS")
        
        # Varsayılan değerler
        shots = 1024
        sim_type = "QISKIT"
        result_var_name = None

        i = 0
        while i < len(args):
            arg_key = str(args[i]).upper()
            if arg_key == "SHOTS" and i + 1 < len(args):
                shots = int(self._get_var_value(str(args[i+1]))) # Argümanın değerini al
                i += 2
            elif arg_key == "TYPE" and i + 1 < len(args):
                sim_type = str(self._get_var_value(str(args[i+1]))).upper()
                i += 2
            elif arg_key == "INTO" and i + 1 < len(args):
                result_var_name = str(args[i+1]) # INTO'dan sonraki argüman değişkendir, değeri değil.
                i += 2
            else:
                raise PdsXException(f"QRUN_SIMULATOR için bilinmeyen argüman: {arg_key}")
        
        if result_var_name is None:
            raise PdsXException("QRUN_SIMULATOR için INTO <var_name> gerekli.")

        counts = await self.simulator.QRUN_SIMULATOR(shots, sim_type)
        self._set_var_value(result_var_name, counts)
        self._get_logger().info(f"Simülasyon sonuçları '{result_var_name}'e kaydedildi.")

    async def QGET_STATEVECTOR_CMD(self, result_var_name: str):
        """QGET_STATEVECTOR INTO <result_var>"""
        statevector = await self.simulator.QGET_STATEVECTOR()
        self._set_var_value(result_var_name, statevector)
        self._get_logger().info(f"Durum vektörü '{result_var_name}'e kaydedildi.")

    async def QMEASURE_RESULT_CMD(self, *args):
        """
        QMEASURE_RESULT Q<idx> COUNTS <counts_dict_var> INTO <classical_var>
        """
        target_qubit_str = None
        counts_dict_var_name = None
        result_var_name = None

        i = 0
        while i < len(args):
            arg = str(args[i]).upper()
            if arg.startswith('Q') and target_qubit_str is None:
                target_qubit_str = args[i]
                i += 1
            elif arg == "COUNTS" and i + 1 < len(args):
                counts_dict_var_name = str(args[i+1])
                i += 2
            elif arg == "INTO" and i + 1 < len(args):
                result_var_name = str(args[i+1])
                i += 2
            else:
                raise PdsXException(f"QMEASURE_RESULT için bilinmeyen argüman: {arg}")
        
        if target_qubit_str is None or counts_dict_var_name is None or result_var_name is None:
            raise PdsXException("QMEASURE_RESULT için eksik argüman: Q<idx> COUNTS <dict_var> INTO <var>")

        if not target_qubit_str.upper().startswith('Q'):
             raise PdsXException(f"Geçersiz qubit formatı: {target_qubit_str}. Q<idx> olmalı.")
        target_q_idx = int(target_qubit_str[1:])

        counts_data = self._get_var_value(counts_dict_var_name)
        if not isinstance(counts_data, dict) or not counts_data:
            raise PdsXException(f"Değişken '{counts_dict_var_name}' geçerli bir counts sözlüğü değil.")
        
        # En çok ölçülen sonucu bul
        # Qiskit'te counts = {'00': 500, '11': 524} şeklindedir.
        # most_common_outcome string bir ikili sayıdır (örn. '001', '110').
        most_common_outcome = max(counts_data, key=counts_data.get) 
        
        # Qiskit'te qubitler genellikle sağdan sola indekslenir (0 en sağdaki bit).
        # Bu yüzden string üzerinde tersine indeksleme kullanırız.
        if len(most_common_outcome) != self.simulator._num_qubits:
             raise PdsXException("QMEASURE_RESULT: Simülasyon çıktısı qubit sayısıyla uyuşmuyor.")
             
        result_bit = int(most_common_outcome[self.simulator._num_qubits - 1 - target_q_idx])

        self._set_var_value(result_var_name, result_bit)
        self._get_logger().info(f"Simülasyon sonucundan Qubit {target_q_idx} için sonuç '{result_bit}' '{result_var_name}'e atandı.")

    async def QRESET_CMD(self):
        """QRESET"""
        await self.simulator.QRESET()

    async def QDESTROY_CMD(self):
        """QDESTROY"""
        await self.simulator.QDESTROY()

    async def QPRINT_CMD(self, *args):
        """QPRINT STATE, QPRINT CIRCUIT [TYPE TEXT|ASCII|MPL] [FILENAME <path>]"""
        if not args:
            raise PdsXException("QPRINT için alt komut (STATE veya CIRCUIT) gerekli.")
        
        sub_command = str(args[0]).upper()
        
        if sub_command == "STATE": # QPRINT STATE
            result_var_name = None
            if len(args) > 1 and str(args[1]).upper() == "INTO":
                if len(args) > 2:
                    result_var_name = str(args[2])
                else:
                    raise PdsXException("QPRINT STATE INTO için değişken adı gerekli.")

            statevector = await self.simulator.QGET_STATEVECTOR()
            if result_var_name:
                self._set_var_value(result_var_name, statevector)
                self._get_logger().info(f"Kuantum durum vektörü '{result_var_name}'e kaydedildi.")
            else:
                output_str = f"Kuantum Durumu ({self.simulator._num_qubits} Qubit):\n"
                for i, amp in enumerate(statevector):
                    basis_state = bin(i)[2:].zfill(self.simulator._num_qubits)
                    prob = abs(amp)**2
                    output_str += f"  |{basis_state}>: ({amp.real:.4f} {'+' if amp.imag >= 0 else ''}{amp.imag:.4f}j) [Olasılık: {prob:.4f}]\n"
                print(output_str)
                self._get_logger().info("Kuantum durum vektörü konsola yazdırıldı.")
        elif sub_command == "CIRCUIT": # QPRINT CIRCUIT [TYPE TEXT|ASCII|MPL] [FILENAME <path>]
            output_type = "TEXT"
            filename = None
            i = 1
            while i < len(args):
                arg_key = str(args[i]).upper()
                if arg_key == "TYPE" and i + 1 < len(args):
                    output_type = str(args[i+1]).upper()
                    i += 2
                elif arg_key == "FILENAME" and i + 1 < len(args):
                    filename = str(args[i+1])
                    i += 2
                else:
                    raise PdsXException(f"QPRINT CIRCUIT için bilinmeyen argüman: {arg_key}")
            await self.simulator.QPRINT_CIRCUIT(output_type, filename)
        else:
            raise PdsXException(f"QPRINT için geçersiz alt komut: {sub_command}. 'STATE' veya 'CIRCUIT' olmalı.")

    async def QVISUALIZE_RESULTS_CMD(self, *args):
        """
        QVISUALIZE_RESULTS COUNTS <dict_var_name> TYPE HISTOGRAM FILENAME "plot.png"
        QVISUALIZE_RESULTS STATEVECTOR <array_var_name> TYPE BLOCH FILENAME "bloch.png"
        """
        data_type_key = None # COUNTS veya STATEVECTOR
        data_var_name = None
        plot_type = "HISTOGRAM"
        filename = "quantum_results.png"

        i = 0
        while i < len(args):
            arg_key = str(args[i]).upper()
            if arg_key in ["COUNTS", "STATEVECTOR"] and i + 1 < len(args):
                data_type_key = arg_key
                data_var_name = str(args[i+1])
                i += 2
            elif arg_key == "TYPE" and i + 1 < len(args):
                plot_type = str(args[i+1]).upper()
                i += 2
            elif arg_key == "FILENAME" and i + 1 < len(args):
                filename = str(args[i+1])
                i += 2
            else:
                raise PdsXException(f"QVISUALIZE_RESULTS için bilinmeyen argüman: {arg_key}")
        
        if data_type_key is None or data_var_name is None:
            raise PdsXException("QVISUALIZE_RESULTS için COUNTS/STATEVECTOR ve ilgili değişken adı gerekli.")

        data_to_plot = self._get_var_value(data_var_name)

        if plot_type == "HISTOGRAM" and data_type_key == "COUNTS":
            if not isinstance(data_to_plot, dict):
                raise PdsXException(f"Değişken '{data_var_name}' bir counts sözlüğü değil, histogram için.")
            await self.simulator.QVISUALIZE_RESULTS(data_to_plot, "HISTOGRAM", filename)
        elif plot_type == "BLOCH" and data_type_key == "STATEVECTOR":
            if not isinstance(data_to_plot, np.ndarray):
                raise PdsXException(f"Değişken '{data_var_name}' bir NumPy array (statevector) değil, Bloch için.")
            # Bloch küresi için _qc ve _num_qubits gerekiyor, PDSXAdvancedQuantumSimulator'a doğrudan parametre iletilmeli
            await self.simulator.QVISUALIZE_RESULTS(data_to_plot, "BLOCH", filename) # data_to_plot burada counts değil statevector
        else:
            raise PdsXException(f"Desteklenmeyen görselleştirme kombinasyonu: '{data_type_key}' ve '{plot_type}'.")
        
        self._get_logger().info(f"Kuantum sonuçları '{filename}' olarak görselleştirildi.")

    # --- PDS-X'in __pdsX_exports__ protokolünü uygula ---
    __pdsX_exports__ = {
        "functions": {
            "QINIT": QINIT_CMD,
            "QGATE": QGATE_CMD,
            "QRUN_SIMULATOR": QRUN_SIMULATOR_CMD,
            "QGET_STATEVECTOR": QGET_STATEVECTOR_CMD,
            "QMEASURE_RESULT": QMEASURE_RESULT_CMD,
            "QRESET": QRESET_CMD,
            "QDESTROY": QDESTROY_CMD,
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

