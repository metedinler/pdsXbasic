import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import psutil
import threading
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ScientificUtils:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1)
        self._lock = threading.Lock()
          # rem #25.1. Kuantum benzetimi ile yük analizi
    def quantum_load_simulation(self, metrics: List[float]) -> Dict[str, float]:
        """Gerçek sistem performans metriklerini kuantum simülasyonla analiz eder."""
        with self._lock:
            try:
                if not metrics:
                    raise ValueError("Metrik listesi boş olamaz")
                    
                # Metrikleri normalize et
                normalized = self.scaler.fit_transform(np.array(metrics).reshape(-1, 1))
                
                # İstatistiksel analiz
                mean = np.mean(metrics)
                std = np.std(metrics)
                z_scores = stats.zscore(metrics)
                
                # Aykırı değer tespiti 
                outliers = self.isolation_forest.fit_predict(normalized)
                
                # İleri analiz
                quantiles = np.percentile(metrics, [25, 50, 75])
                top_metrics = sorted(metrics, reverse=True)[:3]
                
                return {
                    "mean": float(mean),
                    "std": float(std),
                    "z_scores": z_scores.tolist(),
                    "outliers": outliers.tolist(),
                    "q1": float(quantiles[0]),
                    "median": float(quantiles[1]), 
                    "q3": float(quantiles[2]),
                    "top_metrics": top_metrics,
                    "sample_size": len(metrics)
                }
            except Exception as e:
                logger.error(f"Quantum load simulation error: {str(e)}")
                return {"error": str(e)}    # rem #25.2. Kaotik sistem analizi ile yük tahmini
    def chaos_load_prediction(self) -> Dict[str, float]:
        """Sistem kaynak kullanımını kaos teorisi ile analiz eder."""
        try:
            # Anlık sistem durumu
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Proses bilgileri
            process_count = len(psutil.Process().children())
            thread_count = psutil.Process().num_threads()
            
            # Swap kullanımı
            swap = psutil.swap_memory()
            
            # Ağ istatistikleri
            net = psutil.net_io_counters()
            
            return {
                "cpu_usage": cpu_percent,
                "memory_used": memory.percent,
                "disk_used": disk.percent,
                "swap_used": swap.percent,
                "process_count": process_count,
                "thread_count": thread_count,
                "net_packets_sent": net.packets_sent,
                "net_packets_recv": net.packets_recv,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Kaos yük tahmini hatası: {str(e)}")
            return {"error": str(e)}

    def genetic_dependency_optimizer(self, deps: List[Tuple[str, str]]) -> List[str]:
        """Bağımlılıkları optimize eder."""
        try:
            # Çevrimsel bağımlılıkları tespit et
            graph = {}
            for dep, target in deps:
                if dep not in graph:
                    graph[dep] = []
                graph[dep].append(target)
                
            visited = set()
            temp = set()
            
            def has_cycle(node: str) -> bool:
                if node in temp:
                    return True
                if node in visited:
                    return False
                    
                temp.add(node)
                for neighbor in graph.get(node, []):
                    if has_cycle(neighbor):
                        return True
                temp.remove(node)
                visited.add(node)
                return False

            # Optimize edilmiş sıralama
            optimized = []
            for dep in graph:
                if not has_cycle(dep):
                    optimized.append(dep)
                    
            return optimized
        except Exception as e:
            logger.error(f"Genetic dependency optimizer error: {str(e)}")
            return []    # rem #25.3. Yapay sinir ağı tabanlı yük dengeleyici
    def neural_load_balancer(self, resources: List[Dict[str, float]], threshold: float = 0.8) -> Dict[str, List[int]]:
        """Kaynak yönetimi ve yük dengeleme yapar."""
        try:
            if not resources:
                raise ValueError("Kaynak listesi boş olamaz")
                
            # Kaynakları normalize et
            resource_matrix = np.array([[r['cpu'], r['memory'], r['disk']] for r in resources])
            normalized = self.scaler.fit_transform(resource_matrix)
            
            # Aşırı yüklenmiş kaynakları tespit et
            overloaded = np.where(normalized > threshold)[0]
            underloaded = np.where(normalized < threshold)[0]
            
            # Yük dağılımı analizi
            load_distribution = {
                "mean_load": float(normalized.mean()),
                "std_load": float(normalized.std()),
                "max_load": float(normalized.max()),
                "min_load": float(normalized.min())
            }
            
            # Kaynak kullanım tahminleri
            predictions = {
                "overloaded_risk": len(overloaded) / len(resources),
                "underloaded_ratio": len(underloaded) / len(resources),
                "balance_score": 1 - abs(len(overloaded) - len(underloaded)) / len(resources)
            }
            
            return {
                "overloaded": overloaded.tolist(),
                "underloaded": underloaded.tolist(),
                "scores": normalized.mean(axis=1).tolist(),
                "distribution": load_distribution,
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Nöral yük dengeleyici hatası: {str(e)}")
            return {"error": str(e)}    # rem #25.4. Blockchain tabanlı modül doğrulama sistemi
    def blockchain_module_validation(self, modules: List[Dict]) -> Dict[str, List[str]]:
        """Modül güvenlik doğrulaması yapar ve hash zinciri oluşturur."""
        try:
            if not modules:
                raise ValueError("Modül listesi boş olamaz")
                
            valid_modules = []
            invalid_modules = []
            validation_chain = []
            prev_hash = None
            
            for module in modules:
                # Modül bütünlüğünü kontrol et
                is_valid = self._verify_module_integrity(module)
                
                # Modül hash'i ve metadata oluştur
                module_info = {
                    "name": module.get('name', 'unknown'),
                    "version": module.get('version', 'unknown'),
                    "timestamp": datetime.now().isoformat(),
                    "prev_hash": prev_hash
                }
                
                # Hash hesapla
                current_hash = hash(str(module_info))
                module_info["hash"] = current_hash
                
                if is_valid:
                    valid_modules.append(module['name'])
                else:
                    invalid_modules.append(module['name'])
                
                validation_chain.append(module_info)
                prev_hash = current_hash
            
            return {
                "valid": valid_modules,
                "invalid": invalid_modules,
                "validation_chain": validation_chain,
                "chain_length": len(validation_chain),
                "genesis_hash": validation_chain[0]["hash"] if validation_chain else None,
                "latest_hash": prev_hash,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Blockchain modül doğrulama hatası: {str(e)}")
            return {"error": str(e)}
            
    def _verify_module_integrity(self, module: Dict) -> bool:
        """Modül bütünlüğünü kontrol eder."""
        required_fields = ['name', 'version', 'dependencies']
        try:
            # Gerekli alanları kontrol et
            if not all(field in module for field in required_fields):
                return False
                
            # Versiyon formatını kontrol et
            version_parts = module['version'].split('.')
            if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
                return False
                
            # Bağımlılıkları kontrol et
            if not isinstance(module['dependencies'], list):
                return False
                
            return True
        except Exception:
            return False
