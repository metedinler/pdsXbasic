# PDS-X Auto Importer ve Installer Geliştirme Planı

## Plan Özeti ve İlerleme Durumu

### Güncel Durum (18 Haziran 2025)
- [x] Temel modül yönetimi tamamlandı
- [x] Log rotasyon sistemi eklendi
- [x] Deneysel bilimsel sistemler entegre edildi
- [ ] Çakışma çözümleyici geliştiriliyor
- [ ] Import versiyon kaydı sistemi kurulum aşamasında
- [ ] AutoInstaller geliştirilecek

### Planlanmış Adımlar

1. Çakışma Çözümleme Sistemi
   - Terminal çıktıları ve log dosyaları analizi
   - Çakışma veritabanı oluşturma
   - Neural ağ tabanlı çözüm önerileri
   - Blockchain tabanlı doğrulama

2. Import Versiyon Kayıt Sistemi
   - Merkezi versiyon veritabanı
   - Çakışma geçmişi takibi
   - Otomatik rollback özelliği
   - Versiyon ağacı görselleştirme

3. Deneysel/Bilimsel Sistemler
   - Quantum yük analizi
   - Genetik algoritma optimizasyonu
   - Neural ağ tabanlı yük dengeleme
   - Kaos teorisi tabanlı anomali tespiti

4. AutoInstaller Geliştirme
   - İç içe hata kontrolü
   - Çoklu seviye kurtarma
   - Otomatik bağımlılık çözümleme
   - Güvenli mod kurulum desteği

### Bir Sonraki Adım
auto_importer.py modülünün geliştirilmesi:
1. Temel sınıf yapısı ve hata kontrolü
2. Bağımlılık yönetimi
3. Güvenli kurulum modu
4. Kurtarma sistemi

## Mevcut Analiz

### Auto Importer Versiyon Analiz Tablosu

| Özellik | İlk Versiyon | Orta Versiyon | Son Versiyon |
|---------|--------------|---------------|--------------|
| **Temel İşlevler** | • Basit paket yükleme<br>• Manuel Python kontrolü<br>• Minimum hata denetimi | • İzole ortam desteği<br>• Otomatik Python 3.10<br>• Temel bağımlılık yönetimi | • Gelişmiş paket yönetimi<br>• Thread-safe tasarım<br>• Akıllı çakışma önleme |
| **Güvenlik** | • Temel kontroller | • Güvenli mod desteği<br>• İzinli yol kontrolü | • Blockchain doğrulama<br>• Neural ağ tabanlı kontrol<br>• Gelişmiş yetkilendirme |
| **Performans** | • Sıralı yükleme | • Önbellekleme<br>• Paralel yükleme | • Quantum simülasyon<br>• Genetik optimizasyon<br>• Yük dengeleme |
| **Hata Yönetimi** | • Basit hata mesajları | • Yapılandırılmış hatalar<br>• Log dosyası desteği | • Ayrıntılı loglama<br>• Otomatik kurtarma<br>• Hata tahminleme |
| **Bağımlılık Yönetimi** | • Manuel kontrol<br>• Sabit liste | • JSON tabanlı yapı<br>• Versiyon kontrolü | • Dinamik analiz<br>• Çakışma çözümleme<br>• Otomatik güncelleme |
| **Modül Yönetimi** | • Basit import | • Güvenli yükleme<br>• Önbellek desteği | • İzole yükleme<br>• Akıllı önbellekleme<br>• Olay tetikleme |
| **İzolasyon** | • Yok | • Temel sanal ortam<br>• Path izolasyonu | • Tam izolasyon<br>• Kaynak kontrolü<br>• Güvenli mod |
| **Analiz Özellikleri** | • Yok | • Temel istatistikler | • Kuantum analiz<br>• Performans izleme<br>• Usage metrics |

## 1. Genel Bakış

`auto_importer.py`, pdsX sisteminin modül ve paket yönetiminden sorumlu temel bileşenidir. Ana görevleri:

1. Python 3.10 ortam kontrolü ve kurulumu
2. İzole çalışma ortamı (.pdsx_isolated_env) yönetimi
3. Paket bağımlılıklarının yönetimi
4. Güvenli modül yükleme mekanizması

## 2. Versiyon Karşılaştırması

### İlk Versiyon (auto_importer_ilk versiyor)
- Basit paket yükleme
- Minimum hata kontrolü
- Manuel Python kontrolü

### Orta Versiyon (dislananlar/auto_importer.py)
- İzole ortam desteği eklenmiş
- Otomatik Python 3.10 kurulumu
- Temel bağımlılık yönetimi
- Güvenli mod implementasyonu

### Son Versiyon (auto_importer.py)
- Gelişmiş paket yönetimi
- Thread-safe tasarım
- Çakışma önleme
- İyileştirilmiş hata yönetimi
- Ayrıntılı loglama

## 3. Temel Sınıflar ve İşlevleri

### ModuleAutoImporter
- Modül yükleme/kaldırma
- Bağımlılık kontrolü
- Önbellekleme
- Kaynak temizleme

### IsolatedEnvManager 
- Ortam oluşturma/yapılandırma
- Paket kurulumu
- Çakışma kontrolü
- Temizlik işlemleri

### SystemStartupManager
- Başlangıç sırası kontrolü
- Ortam doğrulama
- Güvenlik kontrolleri

## 4. Başlıca Fonksiyonlar

### Python Ortam Yönetimi
- find_python310()
- download_and_install_python310()
- add_python_to_path()

### Paket Yönetimi
- install_missing_packages()
- ensure_venv()
- clean_and_reinstall_packages()

### Güvenlik ve Doğrulama
- validate_environment()
- check_package_conflicts()
- secure_mode_enable/disable()

## 5. Bağımlılık Listesi 

### Ana Bağımlılıklar
- Temel Python paketleri
- Gerekli sistem kütüphaneleri 
- İzole ortam gereksinimleri

## 6. Önemli Süreçler

### 1. Başlangıç Sırası
```
pdsX başlat
  |
  +-> auto_importer çağır
        |
        +-> Python 3.10 kontrol/kur
        +-> İzole ortam kontrolü
        +-> Paket kurulumu
        +-> Modül yönetimi
```

### 2. Paket Kurulum Süreci
```
install_missing_packages()
  |
  +-> Paket listesi kontrol
  +-> Paralel kurulum
  +-> Çakışma kontrolü
  +-> Doğrulama
```

### 3. Modül Yükleme Süreci
```
import_module()
  |
  +-> Bağımlılık kontrolü
  +-> Güvenlik kontrolü
  +-> Yükleme ve cache
  +-> Hata yönetimi
```

## 7. Önemli Noktalar ve Öneriler

1. pdsX ile Entegrasyon
- Auto_importer pdsX'in bir modülü
- Bağımsız çalıştırılamaz
- pdsX başlangıcında otomatik çağrılır

2. İzolasyon Önemi
- .pdsx_isolated_env kullanımı
- Paket çakışmalarından kaçınma
- Temiz ortam garantisi

3. Güvenlik
- Güvenli mod desteği
- İzinli yol kontrolü
- Doğrulanmış modüller

4. İyileştirme Önerileri
- Başarısız kurulumlar için retry mekanizması
- Daha detaylı hata raporlama
- Offline mod desteği
- Paket önbellekleme geliştirmesi

## 8. Kritik Rem Satırları

- rem #0: PDS-X sisteminin başlangıç aşamaları
- rem #17: Tüm bağımlılıklar ve sürüm sabitleme
- rem #23: Dinamik modül yükleme sistemi
- rem #24: İzole ortam yönetimi
- rem #25: ModuleAutoImporter sınıfı
- rem #28: Sistem başlatma ve kontrol

## 9. Yeni Geliştirme Planı

### A. Önbellekleme ve İndirme Yönetimi
1. **Önbellek Klasörü Yapısı**
   ```
   .pdsx_cache/
   ├── packages/         # İndirilen pip paketleri
   ├── downloads/        # İndirilen Python ve diğer araçlar
   ├── metadata/        # Paket metadataları
   └── temp/            # Geçici dosyalar
   ```

2. **Önbellek Yönetimi**
   - İndirilen dosyaların hash doğrulaması
   - Önbellek boyutu limitleme
   - Otomatik temizleme politikası
   - Last-Modified kontrolü

3. **Paralel İndirme**
   ```python
   def parallel_download():
       with ThreadPoolExecutor(max_workers=4) as executor:
           future_to_url = {executor.submit(download_file, url): url 
                          for url in download_urls}
   ```

### B. Yükleme ve Doğrulama Süreci
1. **Paket Kurulum Adımları**
   ```
   Kontrol Et -> İndir -> Doğrula -> Kur -> Test Et -> Rapor
   ```

2. **Retry Mekanizması**
   - Maksimum 3 deneme
   - Üstel geri çekilme (exponential backoff)
   - Hata türüne göre özel işlem

3. **Hata Raporlama**
   - Detaylı log tutma
   - Hata kategorileri
   - Çözüm önerileri

### C. Çok İşlemcili Kurulum

1. **Process Pool Yönetimi**
   ```python
   def multiprocess_install():
       with ProcessPoolExecutor() as executor:
           packages = split_into_chunks(package_list)
           results = executor.map(install_chunk, packages)
   ```

2. **İş Parçacığı Havuzu**
   - Dinamik iş dağıtımı
   - Yük dengeleme
   - Kaynak yönetimi

3. **Senkronizasyon**
   - Kilit mekanizması
   - Paylaşılan durum yönetimi
   - Deadlock önleme

### D. Offline Mod Desteği

1. **Önbellek Yönetimi**
   - Paket indeksi önbellekleme
   - Bağımlılık ağacı
   - Wheel dosyaları

2. **Offline Kurulum**
   ```
   .pdsx_offline/
   ├── index/          # Paket indeksi
   ├── wheels/         # Wheel dosyaları
   └── metadata/       # Metadata bilgileri
   ```

### E. pyvenv.cfg Yönetimi

1. **Otomatik Onarım**
   ```python
   def repair_venv():
       if not os.path.exists('pyvenv.cfg'):
           create_pyvenv_cfg()
       validate_venv_structure()
   ```

2. **Doğrulama ve İzleme**
   - Düzenli kontroller
   - Otomatik düzeltme
   - Hata bildirimi

### F. Renkli Özet Tablo

1. **Kurulum Özeti**
   ```
   ┌────────────────────────────────────┐
   │ PDS-X Kurulum Özeti               │
   ├────────────────┬─────────┬────────┤
   │ Paket          │ Durum   │ Süre   │
   ├────────────────┼─────────┼────────┤
   │ numpy          │ ✓ Başarılı│ 2.3s  │
   │ pandas         │ ✓ Başarılı│ 3.1s  │
   └────────────────┴─────────┴────────┘
   ```

2. **Renk Kodları**
   - Yeşil: Başarılı
   - Sarı: Uyarı
   - Kırmızı: Hata
   - Mavi: İşlemde

### G. İlerleme Takibi

1. **Progress Bar**
   ```
   [====================] 100% Tamamlandı
   numpy    [====        ]  40%
   pandas   [========    ]  80%
   ```

2. **Detaylı Durum**
   - İndirme hızı
   - Kalan süre
   - İşlem detayları

### H. Başarısız Kurulum Yönetimi

1. **Yeniden Deneme Stratejisi**
   ```python
   def retry_with_backoff(func, max_tries=3):
       for attempt in range(max_tries):
           try:
               return func()
           except Exception as e:
               wait_time = (2 ** attempt) * 1  # Üstel artış
               time.sleep(wait_time)
   ```

2. **Alternatif Kaynaklar**
   - Farklı mirror'lar
   - Alternatif paket versiyonları
   - Offline cache

3. **Kurtarma Modu**
   - Minimal kurulum
   - Kritik bağımlılıklar
   - Güvenli mod

## 10. Uygulama Planı

1. **Aşama 1: Altyapı**
   - Önbellek sistemi
   - Paralel indirme
   - Hata yönetimi

2. **Aşama 2: Optimize Etme**
   - Çok işlemcili destek
   - Retry mekanizması
   - Progress tracking

3. **Aşama 3: İyileştirmeler**
   - Offline mod
   - Renkli raporlama
   - Detaylı logging

4. **Aşama 4: Test ve Deploy**
   - Unit testler
   - Entegrasyon testleri
   - Dokümantasyon

## 11. Loglama Sistemi Tasarımı

### A. Log Yapılandırması

1. **Log Dosyaları**
   ```
   logs/
   ├── pdsxu_terminal.log    # Terminal çıktıları
   ├── pdsxu_errors.log      # Hata ve debug logları
   ├── pdsxu_terminal.bak    # Terminal yedeği
   └── analytics/            # Log analizleri
       ├── daily/            # Günlük raporlar
       ├── warnings/         # Uyarı logları
       └── errors/          # Hata logları
   ```

2. **Log Formatları**
   ```python
   {
     "timestamp": "2025-06-18T10:30:00Z",
     "level": "ERROR",
     "module": "auto_importer",
     "message": "Paket kurulum hatası",
     "context": {
       "package": "numpy",
       "error_code": "E101",
       "stack_trace": "..."
     }
   }
   ```

### B. Loglama Stratejileri

1. **Terminal Loglama (Tee Sınıfı)**
   - Eşzamanlı ekran ve dosya çıktısı
   - Otomatik flush
   - UTF-8 encoding desteği

2. **Hata İzleme (BacktraceLogger)**
   - Hata yığını takibi
   - Bağlam bilgisi
   - Elasticsearch entegrasyonu
   - Görselleştirme

3. **Akıllı Loglama (EvolvingLogger)**
   - Anomali tespiti
   - Örüntü öğrenme 
   - Otomatik kategorizasyon
   - Tahmine dayalı uyarılar

### C. Log Yönetimi

1. **Rotasyon Politikası**
   ```python
   {
     "max_size": "100MB",
     "backup_count": 5,
     "compression": True,
     "retention_days": 30
   }
   ```

2. **Log Temizleme**
   - Otomatik arşivleme
   - Sıkıştırma
   - Disk alan yönetimi

3. **Performans Optimizasyonu**
   - Asenkron yazma
   - Buffer yönetimi
   - Batch işleme

### D. Analiz ve Raporlama

1. **Real-time Analiz**
   - Canlı hata takibi
   - Anlık uyarılar
   - Dashboard entegrasyonu

2. **İstatistiksel Analiz**
   ```python
   {
     "daily_summary": {
       "error_count": 120,
       "warning_count": 350,
       "success_rate": "98.5%"
     },
     "trends": {
       "top_errors": [...],
       "peak_times": [...],
       "common_patterns": [...]
     }
   }
   ```

3. **Görselleştirme**
   - Zaman çizelgeleri
   - Hata grafikleri
   - İlişki haritaları

### E. Güvenlik ve Uyum

1. **Veri Koruma**
   - Hassas veri maskeleme
   - Şifreleme
   - Erişim kontrolü

2. **Denetim**
   - İşlem logları
   - Güvenlik olayları
   - Yetkilendirme kayıtları

### F. Entegrasyon

1. **Elasticsearch**
   ```python
   def connect_elasticsearch():
       es = Elasticsearch([{
           'host': 'localhost',
           'port': 9200,
           'scheme': 'http'
       }])
       return es
   ```

2. **Monitoring Sistemleri**
   - Prometheus entegrasyonu
   - Grafana dashboards
   - Alert manager


module_analysor.py

import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

class ModuleAnalyzer:
    def __init__(self, log_file: str = "pdsxu_terminal.log"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
    def analyze_logs(self) -> Dict:
        """Log dosyalarını analiz eder ve rapor oluşturur."""
        try:
            log_stats = {
                "errors": [],
                "warnings": [],
                "info": [],
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0
            }
            
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "[ERROR]" in line:
                            log_stats["errors"].append(line.strip())
                            log_stats["error_count"] += 1
                        elif "[WARNING]" in line:
                            log_stats["warnings"].append(line.strip())
                            log_stats["warning_count"] += 1
                        elif "[INFO]" in line:
                            log_stats["info"].append(line.strip())
                            log_stats["info_count"] += 1
                            
            return log_stats
        except Exception as e:
            self.logger.error(f"Log analizi hatası: {str(e)}")
            return {"error": str(e)}
            
    def generate_module_report(self, modules: List[Dict]) -> Dict:
        """Modül analiz raporu oluşturur."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "modules": {},
                "total_modules": len(modules),
                "issues": [],
                "recommendations": []
            }
            
            for module in modules:
                module_name = module.get("name", "unknown")
                report["modules"][module_name] = {
                    "version": module.get("version", "unknown"),
                    "dependencies": module.get("dependencies", []),
                    "status": self._check_module_status(module)
                }
                
                # Sorunları tespit et
                issues = self._detect_issues(module)
                if issues:
                    report["issues"].extend(issues)
                    
                # Öneriler oluştur
                recommendations = self._generate_recommendations(module)
                if recommendations:
                    report["recommendations"].extend(recommendations)
                    
            return report
        except Exception as e:
            self.logger.error(f"Rapor oluşturma hatası: {str(e)}")
            return {"error": str(e)}
            
    def _check_module_status(self, module: Dict) -> str:
        """Modül durumunu kontrol eder."""
        try:
            if not all(key in module for key in ["name", "version", "dependencies"]):
                return "invalid"
                
            if not module.get("dependencies"):
                return "warning"
                
            return "ok"
        except Exception:
            return "error"
            
    def _detect_issues(self, module: Dict) -> List[str]:
        """Modüldeki sorunları tespit eder."""
        issues = []
        try:
            # Temel alan kontrolü
            if "name" not in module:
                issues.append(f"Module missing name field")
            if "version" not in module:
                issues.append(f"Module {module.get('name', 'unknown')} missing version")
            if "dependencies" not in module:
                issues.append(f"Module {module.get('name', 'unknown')} missing dependencies")
                
            # Versiyon format kontrolü
            version = module.get("version", "")
            if version and not self._is_valid_version(version):
                issues.append(f"Module {module.get('name', 'unknown')} has invalid version format: {version}")
                
            # Bağımlılık kontrolü
            deps = module.get("dependencies", [])
            if isinstance(deps, list) and len(deps) == 0:
                issues.append(f"Module {module.get('name', 'unknown')} has no dependencies")
                
            return issues
        except Exception as e:
            self.logger.error(f"Issue detection error: {str(e)}")
            return [f"Error analyzing module: {str(e)}"]
            
    def _generate_recommendations(self, module: Dict) -> List[str]:
        """Modül için öneriler oluşturur."""
        recommendations = []
        try:
            # Eksik alanlar için öneriler
            if "version" not in module:
                recommendations.append(f"Add version information for module {module.get('name', 'unknown')}")
            if "dependencies" not in module:
                recommendations.append(f"Specify dependencies for module {module.get('name', 'unknown')}")
                
            # Bağımlılık önerileri
            deps = module.get("dependencies", [])
            if isinstance(deps, list):
                if len(deps) == 0:
                    recommendations.append(f"Consider adding required dependencies for module {module.get('name', 'unknown')}")
                elif len(deps) > 10:
                    recommendations.append(f"Consider reducing dependencies for module {module.get('name', 'unknown')}")
                    
            return recommendations
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return [f"Error generating recommendations: {str(e)}"]
            
    def _is_valid_version(self, version: str) -> bool:
        """Versiyon formatının geçerliliğini kontrol eder."""
        try:
            parts = version.split(".")
            return len(parts) >= 2 and all(part.isdigit() for part in parts)
        except Exception:
            return False
            
    def export_report(self, report: Dict, output_file: str = "module_analysis.json") -> None:
        """Raporu JSON dosyasına kaydeder."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Report exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Report export error: {str(e)}")
            raise


scientific utils
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import psutil
import threading
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ScientificUtils:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1)
        self._lock = threading.Lock()
        
    def quantum_load_simulation(self, metrics: List[float]) -> Dict[str, float]:
        """Gerçek sistem performans metriklerini analiz eder."""
        with self._lock:
            try:
                # Metrikleri normalize et
                normalized = self.scaler.fit_transform(np.array(metrics).reshape(-1, 1))
                
                # İstatistiksel analiz
                mean = np.mean(metrics)
                std = np.std(metrics)
                z_scores = stats.zscore(metrics)
                
                # Aykırı değer tespiti
                outliers = self.isolation_forest.fit_predict(normalized)
                
                return {
                    "mean": float(mean),
                    "std": float(std),
                    "z_scores": z_scores.tolist(),
                    "outliers": outliers.tolist()
                }
            except Exception as e:
                logger.error(f"Quantum load simulation error: {str(e)}")
                return {"error": str(e)}

    def chaos_load_prediction(self) -> Dict[str, float]:
        """Sistem kaynak kullanımını analiz eder."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_percent,
                "memory_used": memory.percent,
                "disk_used": disk.percent
            }
        except Exception as e:
            logger.error(f"Chaos load prediction error: {str(e)}")
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
            return []

    def neural_load_balancer(self, resources: List[Dict[str, float]], threshold: float = 0.8) -> Dict[str, List[int]]:
        """Kaynak yönetimi ve yük dengeleme yapar."""
        try:
            # Kaynakları normalize et
            resource_matrix = np.array([[r['cpu'], r['memory'], r['disk']] for r in resources])
            normalized = self.scaler.fit_transform(resource_matrix)
            
            # Aşırı yüklenmiş kaynakları tespit et
            overloaded = np.where(normalized > threshold)[0]
            underloaded = np.where(normalized < threshold)[0]
            
            return {
                "overloaded": overloaded.tolist(),
                "underloaded": underloaded.tolist(),
                "scores": normalized.mean(axis=1).tolist()
            }
        except Exception as e:
            logger.error(f"Neural load balancer error: {str(e)}")
            return {"error": str(e)}

    def blockchain_module_validation(self, modules: List[Dict]) -> Dict[str, List[str]]:
        """Modül güvenlik doğrulaması yapar."""
        try:
            valid_modules = []
            invalid_modules = []
            
            for module in modules:
                # Modül bütünlüğünü kontrol et
                if self._verify_module_integrity(module):
                    valid_modules.append(module['name'])
                else:
                    invalid_modules.append(module['name'])
                    
            return {
                "valid": valid_modules,
                "invalid": invalid_modules
            }
        except Exception as e:
            logger.error(f"Blockchain module validation error: {str(e)}")
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


libxcore degisiklik

import asyncio
from typing import Any, Callable, List, Dict, Optional, Union
from collections import deque

       self.log_file = "pdsxu_terminal.log"
        self.error_log = "pdsxu_errors.log"
        self.backup_log = "pdsxu_terminal.bak"
        self.log_queue = deque(maxlen=1000)  # Son 1000 logu tut

def log(self, message: str, level: str = "INFO", target: Optional[str] = None) -> None:
        """Log mesajı kaydeder ve yedekler."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # Log dosyasına yaz
        target_file = target or self.log_file
        try:
            with open(target_file, "a", encoding=self.default_encoding) as f:
                f.write(log_entry + "\n")
            
            # Log kuyruğuna ekle
            self.log_queue.append(log_entry)
            
            # Dosya boyutu kontrolü (10MB üzeri)
            if os.path.getsize(target_file) > 10 * 1024 * 1024:
                self._rotate_logs(target_file)
                
        except Exception as e:
            # Hata durumunda error loga yaz
            with open(self.error_log, "a", encoding=self.default_encoding) as f:
                f.write(f"[{timestamp}] [ERROR] Log yazma hatası: {str(e)}\n")
                
    def _rotate_logs(self, log_file: str) -> None:
        """Log dosyalarını yedekler ve yeni log dosyası oluşturur."""
        try:
            if os.path.exists(log_file):
                backup = f"{log_file}.{int(time.time())}.bak"
                shutil.copy2(log_file, backup)
                with open(log_file, "w", encoding=self.default_encoding) as f:
                    f.write("")  # Dosyayı temizle
                # En son 5 yedeği tut
                self._cleanup_old_backups(log_file)
        except Exception as e:
            print(f"[PDS-X] Log rotasyon hatası: {e}")
            
    def _cleanup_old_backups(self, log_file: str) -> None:
        """Eski log yedeklerini temizler, son 5 yedeği tutar."""
        pattern = f"{log_file}.*.bak"
        backups = sorted(glob.glob(pattern), key=os.path.getctime, reverse=True)
        for old_backup in backups[5:]:  # İlk 5'i hariç sil
            try:
                os.remove(old_backup)
            except Exception:
                pass

requitments.txt
# Core Dependencies
numpy>=1.21.0,<2.2.0  # TensorFlow uyumluluğu için üst sınır
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0
graphviz>=0.16
requests>=2.26.0
aiohttp>=3.8.0
websockets>=9.1
aiofiles>=0.8.0
pdfplumber>=0.7.0
tensorflow==2.19.0
psycopg2-binary>=2.9.0
PyYAML>=6.0
psutil>=5.8.0
paho-mqtt>=1.6.0
boto3>=1.20.0
botocore>=1.23.0
language-data>=1.1

# Optional Dependencies
transformers>=4.12.0
nltk>=3.6.0
spacy>=3.1.0
gensim>=4.1.0

# Development Dependencies
pytest>=6.2.0
black>=21.5b2
pylint>=2.8.0
mypy>=0.910
coverage>=6.2


yerek alma nasil yapiliyor (log yedegi) ? Ayrica.json formatli bir dosya olusturacaksan yeni bir isim ile yaratilsin. deneysel ozellikleri autoimporter calisirken hatalar uyarilar bilgileri (terminal ciktilarini log kayilarini) inceleyerek duzeltmek icin gereken cozumleri almasini saglamalisin. bunlari yapmadan once tum modulleri ve ana programi incelemelisin. kararlarini incelemelerinin sonucuna gore al.

senden onceki yapay zeka asagidaki satirlari silmek icin bana yalan soyledi ve sildi. REQUIRED_PACKAGES = [

Temel bilimsel ve ML kütüphaneleri
("numpy<2.2.0", "numpy"), # pin numpy version <2.2.0 for TF compatibility
("scipy==1.11.4", "scipy"),
("pandas==2.1.4", "pandas"),
("scikit-learn==1.3.2", "sklearn"),
("joblib", "joblib"),
("threadpoolctl", "threadpoolctl"),
("matplotlib==3.8.4", "matplotlib"),
("kiwisolver", "kiwisolver"),
("cycler", "cycler"),
("pyparsing", "pyparsing"),
("python-dateutil", "dateutil"),
("pillow", "PIL"),
("packaging", "packaging"),
("seaborn", "seaborn"),
("statsmodels", "statsmodels"),
("tornado", "tornado"),
("plotly", "plotly"),
("tenacity", "tenacity"),
("dash", "dash"),
("flask", "flask"),
("jinja2", "jinja2"),
("werkzeug", "werkzeug"),
("itsdangerous", "itsdangerous"),
("markupsafe", "markupsafe"),
("click", "click"),
("grpcio", "grpc"),
("protobuf", "google.protobuf"),
("aiohttp", "aiohttp"),
("async-timeout", "async_timeout"),
("yarl", "yarl"),
("multidict", "multidict"),
("attrs", "attr"),
("frozenlist", "frozenlist"),
("pyzmq", "zmq"),
("websocket-client", "websocket"),
("paho-mqtt", "paho.mqtt.client"), # MQTT client support
("boto3", "boto3"), # AWS SDK
("botocore", "botocore"), # boto3 dependency
("kafka-python", "kafka"),
("river", "river"),
("qiskit", "qiskit"),
("networkx", "networkx"),
("boto3", "boto3"),
("botocore", "botocore"),
("websockets", "websockets"),
("rich", "rich"),
("colorama", "colorama"),
("textblob", "textblob"),
("mysql-connector-python", "mysql.connector"),
("psutil", "psutil"),
("pyyaml", "yaml"),
("graphviz", "graphviz"),
("aiofiles==23.2.1", "aiofiles"),
("RestrictedPython>=6.2,<8.0", "RestrictedPython"),
("pdfplumber", "pdfplumber"),
("requests", "requests"),
("psycopg2-binary==2.9.9", "psycopg2"),
("elasticsearch", "elasticsearch"),
("elastic-transport", "elastic_transport"),
("nltk", "nltk"), # textblob için zorunlu bağımlılık

Derin öğrenme ve bilimsel
("tensorflow==2.15.0", "tensorflow"),
("torch==2.2.2", "torch"),
("torch-geometric==2.5.3", "torch_geometric"),
("spacy==3.5.3", "spacy"), # pin spacy version to avoid conflicts
("transformers", "transformers"),
("pycryptodome", "Crypto"),

--- TRANSITIVE/SECONDARY DEPENDENCIES ---
("jmespath", "jmespath"), # boto3/botocore için
("pdfminer.six==20250327", "pdfminer"), # pdfplumber için
("pypdfium2>=4.18.0", "pypdfium2"), # pdfplumber için
("markdown-it-py>=2.2.0", "markdown_it_py"), # rich için
("pygments>=2.13.0,<3.0.0", "pygments"), # rich için
("catalogue<2.1.0,>=2.0.6", "catalogue"), # spacy için
("cymem<2.1.0,>=2.0.2", "cymem"), # spacy için
("langcodes<4.0.0,>=3.2.0", "langcodes"), # spacy için
("murmurhash<1.1.0,>=0.28.0", "murmurhash"),# spacy için
("preshed<3.1.0,>=3.0.2", "preshed"), # spacy için
("pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4", "pydantic"), # spacy için
("spacy-legacy<3.1.0,>=3.0.11", "spacy_legacy"), # spacy için
("spacy-loggers<2.0.0,>=1.0.0", "spacy_loggers"), # spacy için
("srsly<3.0.0,>=2.4.3", "srsly"), # spacy için
("thinc<8.4.0,>=8.3.4", "thinc"), # spacy için
("typer<1.0.0,>=0.3.0", "typer"), # spacy için
("wasabi<1.2.0,>=0.9.1", "wasabi"), # spacy için
("weasel<0.5.0,>=0.1.0", "weasel"), # spacy için
("huggingface-hub<1.0,>=0.30.0", "huggingface_hub"), # transformers için
("regex!=2019.12.17", "regex"), # transformers için
("safetensors>=0.4.3", "safetensors"), # transformers için
("tokenizers<0.22,>=0.21", "tokenizers"), # transformers için
("cryptography", "cryptography"), # pdfminer.six için
("cffi", "cffi"), # cryptography için
("pycparser", "pycparser"), # cffi için
("six", "six"), # cryptography için
("pyasn1", "pyasn1"), # cryptography için
("pyasn1-modules", "pyasn1_modules"), # cryptography için
("idna", "idna"), # requests için
("charset_normalizer", "charset_normalizer"), # requests için
("urllib3", "urllib3"), # requests için
("certifi", "certifi"), # requests için
("chardet", "chardet"), # textblob için
("blis<1.4.0,>=1.3.0", "blis"), # thinc için
("confection<1.0.0,>=0.0.1", "confection"), # thinc, weasel için
("shellingham>=1.3.0", "shellingham"), # typer için
("smart-open<8.0.0,>=5.2.1", "smart_open"), # weasel için
("cloudpathlib<1.0.0,>=0.7.0", "cloudpathlib"), # weasel için
("annotated-types>=0.6.0", "annotated_types"), # pydantic için
("pydantic-core==2.33.2", "pydantic_core"), # pydantic için
("typing-inspection>=0.4.0", "typing_inspect"), # pydantic için
]

bu satirlardaki kutuphaneleri neden silmis olabilir yerine yerlestir. bu satirlarin tum modulleri inceleye ve neden bu sekilde yuklendigini anlamaya calis?