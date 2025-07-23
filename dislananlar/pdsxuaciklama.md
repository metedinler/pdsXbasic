# PDS-X BASIC v14u Yorumlayıcı Analizi

## Program Genel Bakış

PDS-X BASIC v14u, çok paradigmalı bir programlama dili yorumlayıcısıdır. Python 3.10 üzerinde çalışmak üzere tasarlanmış ve modüler bir yapıya sahiptir.

## Temel Bileşenler

### 1. Başlangıç ve Ortam Kontrolü
```python
from auto_importer import find_python310, CORE_DEPENDENCIES, install_missing_packages
```
- Program başlangıçta Python 3.10 ortamını kontrol eder
- Gerekli bağımlılıkları otomatik olarak yönetir
- İzole bir Python ortamı oluşturur (.pdsx_isolated_env)

### 2. İzole Ortam Yönetimi (`_ensure_isolated_env()`)
- İzole Python ortamı oluşturur veya mevcut ortamı kontrol eder
- Temel paketleri otomatik olarak yükler
- Ortam değişikliği gerektiğinde programı yeni ortamda yeniden başlatır

### 3. Modül Yönetimi Sistemi
Aşağıdaki temel modül grupları bulunur:
- Core Modules (Çekirdek Modüller)
- LibX Modules (Genişletilmiş Kütüphane Modülleri)
- Utility Modules (Yardımcı Modüller)

## Önemli Sınıflar

### 1. ModuleAutoImporter
- Dinamik modül yükleme ve yönetimi
- Bağımlılık çözümleme
- Versiyon kontrolü

### 2. PdsXv14uInterpreter
- Ana yorumlayıcı sınıfı
- Komut işleme ve yürütme
- Değişken ve kapsam yönetimi

### 3. PDSXIntegrator
- Modül entegrasyonu
- Bileşen başlatma
- Sistem hazırlığı

### 4. Float Sınıfları (Float128, Float256, Float512)
- Yüksek hassasiyetli kayan nokta desteği
- decimal.Decimal tabanlı özel sayı tipleri

## Modül Yapısı

### Core Modüller
1. **BytecodeCompiler & BytecodeManager**
   - Bytecode derleme ve yönetimi
   - Performans optimizasyonu

2. **ModuleManager**
   - Modül yaşam döngüsü yönetimi
   - Bağımlılık çözümleme

3. **CoreManager**
   - Temel sistem operasyonları
   - Bellek ve kaynak yönetimi

### LibX Modülleri
1. **LibXJIT**
   - Dinamik kod derleme
   - JIT optimizasyonu

2. **LibXData & LibXLogic**
   - Veri yapıları
   - Mantık işlemleri

3. **LibXConcurrency**
   - Paralel işlem yönetimi
   - Asenkron operasyonlar

### Yardımcı Modüller
1. **PluginManager**
   - Eklenti yükleme/kaldırma
   - Plugin yaşam döngüsü yönetimi

2. **ExceptionManager**
   - Hata yakalama ve loglama
   - Hata raporlama

## Özel Özellikler

### 1. Güvenlik Özellikleri
- İzole ortam kullanımı
- Güvenli modül yükleme
- Kaynak kısıtlamaları

### 2. Performans Özellikleri
- JIT derleme desteği
- Bytecode optimizasyonu
- Önbellekleme mekanizmaları

### 3. Geliştirici Araçları
- Detaylı loglama
- Hata ayıklama araçları
- Performans profili

## Program Akışı

1. **Başlangıç**
   - Python 3.10 kontrolü
   - İzole ortam hazırlığı
   - Temel bağımlılıkların kurulumu

2. **Sistem Hazırlığı**
   - Modül yükleyicilerin başlatılması
   - Bytecode sisteminin hazırlanması
   - Loglama sisteminin başlatılması

3. **Çalışma Zamanı**
   - Komut yorumlama
   - Modül yönetimi
   - Hata yönetimi

## CLI Argümanları

Program aşağıdaki komut satırı argümanlarını destekler:
- `--interactive`: Etkileşimli kabuk modu
- `--debug`: Hata ayıklama modu
- `--file`: .basX dosyası çalıştırma
- `--config`: Yapılandırma dosyası kullanımı
- `--profile`: Performans profili
- `--log-level`: Log seviyesi ayarı
- `--lang`: Dil seçimi (tr/en)

## Loglama ve Hata Yönetimi

- Detaylı log kayıtları
- Hata izleme ve raporlama
- Performans metrikleri

## Güvenlik Önlemleri

1. **İzole Ortam**
   - Bağımlılıkların izole yönetimi
   - Sistem Python'dan bağımsız çalışma

2. **Modül Doğrulama**
   - Güvenlik kontrolleri
   - Versiyon uyumluluk kontrolleri

## Modül Bağımlılıkları

### Temel Bağımlılıklar
- numpy
- pandas
- pytest
- pylint
- autopep8
- transformers
- nltk
- spacy
- gensim

### Ek Bağımlılıklar
- boto3
- botocore
- paho-mqtt

## Geliştirici Notları

1. **Katkıda Bulunma**
   - Modüler yapı
   - Test odaklı geliştirme
   - Dokümantasyon önemi

2. **Performans İyileştirmeleri**
   - JIT optimizasyonu
   - Bellek yönetimi
   - Önbellekleme stratejileri

3. **Bakım ve Güncellemeler**
   - Versiyon kontrolü
   - Bağımlılık yönetimi
   - Geriye dönük uyumluluk

# pdsXuv14.py - Detaylı Program Analizi

## Program Yapısı ve Bileşenler

### 1. Başlangıç ve Ortam Hazırlığı
- Program başlangıçta Python 3.10 ortamının varlığını kontrol eder
- İzole bir Python ortamı (.pdsx_isolated_env) oluşturur veya mevcut olanı kullanır
- Temel bağımlılıkların kurulumunu otomatik olarak gerçekleştirir

### 2. Sınıf Yapıları

#### 2.1 PdsXv14uInterpreter Ana Sınıfı
```python
class PdsXv14uInterpreter:
    def __init__(self):
        # Değişken ve Kapsam Yönetimi
        self.global_vars = {}
        self.shared_vars = defaultdict(list)
        self.local_scopes = [{}]
        
        # Tip ve Fonksiyon Yönetimi
        self.types = {}
        self.classes = {}
        self.interfaces = {}
        self.functions = {}
```
Bu ana sınıf aşağıdaki özellikleri içerir:
- Global ve yerel değişken yönetimi
- Tip sistemi (Float128, Float256, Float512 gibi özel tipler dahil)
- Fonksiyon ve komut yönetimi
- Bytecode derleme ve optimizasyon
- Asenkron çalışma desteği
- Çok dilli arayüz desteği

#### 2.2 PluginManager
```python
class PluginManager:
    def __init__(self, plugin_dir="plugins"):
        self.plugin_dir = plugin_dir
        self.plugins = {}
```
- Dinamik eklenti yükleme/kaldırma yönetimi
- Plugin keşfi ve yaşam döngüsü kontrolü
- Güvenli eklenti yükleme mekanizmaları

#### 2.3 ExceptionManager
```python
class ExceptionManager:
    async def handle_error(self, exc):
        log.error(f"Exception: {exc}")
        print(f"[HATA] {exc}")
```
- Asenkron hata yönetimi
- Detaylı hata loglama
- Kullanıcı dostu hata mesajları

### 3. Modül Sistemi

#### 3.1 Çekirdek Modüller
```python
CORE_MODULES = [
    "pdsx_exception", "pdsx_exception2", "module_manager",
    "core2-6", "memory_manager", "save_load_system2",
    # ...diğer çekirdek modüller
]
```
- Temel sistem fonksiyonları
- Bellek ve kaynak yönetimi
- Bytecode işleme ve optimizasyon

#### 3.2 Yüklenebilir Modüller
```python
LOADABLE_MODULES = [
    "core2-5", "database_sql_isam", "bytecode_engine",
    "functional2", "graph2", "libx_ml",
    # ...diğer yüklenebilir modüller
]
```
- İsteğe bağlı özellikler
- Uzantı ve genişletme modülleri
- Özel amaçlı kütüphaneler

### 4. Fonksiyon ve Komut Sistemi

#### 4.1 Fonksiyon Tablosu
Program, 200'den fazla yerleşik fonksiyon içerir:
- Matematik fonksiyonları (SIN, COS, TAN, LOG vb.)
- String işleme (MID$, LEFT$, RIGHT$ vb.)
- Veri yapıları (LIST, DICT, ARRAY vb.)
- Sistem fonksiyonları (SYSTEM, TIME_NOW, DATE_NOW vb.)
- İstatistik fonksiyonları (MEAN, MEDIAN, MODE vb.)

#### 4.2 Operatör Tablosu
```python
self.operator_table = {
    '++': lambda x: x + 1,
    '--': lambda x: x - 1,
    '<<': lambda x, y: x << y,
    # ...diğer operatörler
}
```
- Aritmetik operatörler
- Mantıksal operatörler
- Bit işlem operatörleri
- Özel mantık operatörleri (IMP, EQV, XNOR vb.)

### 5. Performans Özellikleri

#### 5.1 Bytecode Optimizasyonu
- Bytecode derleme ve yönetimi
- JIT (Just-In-Time) derleme desteği
- Kod önbellekleme

#### 5.2 Asenkron Çalışma
```python
async def run_async(self):
    self.running = True
    while self.running and self.program_counter < len(self.program):
        command = self.program[self.program_counter]
        # ...komut yürütme
```
- Asenkron program yürütme
- Paralel işlem desteği
- Event-loop yönetimi

### 6. Güvenlik Özellikleri

#### 6.1 İzole Ortam
```python
def _ensure_isolated_env():
    venv_dir = os.path.join(here, '.pdsx_isolated_env')
    # ...izole ortam kurulumu
```
- Güvenli paket yönetimi
- Sistem Python'dan izole çalışma
- Bağımlılık çakışmalarını önleme

#### 6.2 Modül Doğrulama
```python
def validate_modules():
    validator = ModuleVersionValidator()
    core_results = validate_all_modules(CORE_MODULES)
    # ...modül doğrulama
```
- Modül versiyon kontrolü
- Güvenlik doğrulamaları
- Bütünlük kontrolleri

### 7. Geliştirici Araçları

#### 7.1 REPL (Read-Eval-Print Loop)
```python
async def interactive_shell(self):
    self.repl_mode = True
    while self.repl_mode:
        command = input("[pdsX-Basic]>>> ")
        # ...komut işleme
```
- Etkileşimli geliştirme ortamı
- Anlık kod yürütme
- Hata ayıklama desteği

#### 7.2 Loglama ve İzleme
- Detaylı log kayıtları
- Performans metrikleri
- Hata izleme ve raporlama

### 8. Veri Yapıları ve Tipler

#### 8.1 Özel Sayı Tipleri
```python
class Float128(decimal.Decimal):
    def __new__(cls, value=0):
        context = decimal.Context(prec=34)
        return decimal.Decimal.__new__(cls, str(value), context)
```
- Yüksek hassasiyetli sayılar
- Özel ondalık işlemler
- Bilimsel hesaplamalar

#### 8.2 Veri Yapıları
- İleri düzey listeler
- Ağaç yapıları
- Graf yapıları
- Kuyruk ve yığın implementasyonları

### 9. Dil Desteği ve Uluslararasılaştırma

#### 9.1 Çoklu Dil Desteği
```python
def load_translations(self, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # ...varsayılan çeviriler
```
- Dinamik dil değiştirme
- Çeviri dosyası desteği
- UTF-8 ve diğer kodlama destekleri

### 10. Entegrasyon ve Genişletilebilirlik

#### 10.1 Modül Entegrasyonu
```python
def _add_core_module_exports_to_tables():
    for mod_name, mod_obj in CORE_MODULES_LIST:
        # ...modül fonksiyonlarını entegre et
```
- Otomatik modül entegrasyonu
- Plugin sistemi
- API genişletme desteği

#### 10.2 Komut Ayrıştırma
```python
class CommandParser:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.command_registry = {}
        # ...komut kayıt sistemi
```
- Esnek komut sistemi
- Alias desteği
- Dinamik komut yükleme

### 11. Performans ve Optimizasyon

#### 11.1 İfade Önbellekleme
```python
def evaluate_expression(self, expr, scope_name=None):
    cache_key = (expr, scope_name)
    if cache_key not in self.expr_cache:
        # ...ifade derleme ve önbellekleme
```
- İfade önbellekleme
- Bytecode optimizasyonu
- Bellek yönetimi

#### 11.2 Asenkron İşlem Yönetimi
- Paralel işlem desteği
- Event-loop optimizasyonu
- Kaynak yönetimi

Verdiğiniz `auto_importer.py` kodunda çakışma önleme ve çakışma çözme işlemleri, özellikle bağımlılık yönetimi sırasında ortaya çıkan kütüphane çakışmalarını (örneğin, bir paketin farklı sürümlerinin gereksinimleri) tespit etmek ve çözmek için tasarlanmış bir sistem tarafından gerçekleştiriliyor. Bu işlemler, `ConflictManager` sınıfıを中心に yürütülüyor ve diğer sınıflarla (örneğin, `AdvancedLogger`, `DependencyRegistry`, `CacheManager`) entegre bir şekilde çalışıyor. Aşağıda, çakışma önleme ve çözme işlemlerinin nasıl çalıştığını, hangi dosyaları kullandığı ve ilgili kod parçalarını detaylı bir şekilde açıklayacağım. Her şey Türkçe olacak ve hiçbir özetleme veya kısaltma yapılmayacak.

---

### **Çakışma Önleme ve Çözme İşlemleri Nasıl Çalışıyor?**

`auto_importer.py` modülünde çakışma önleme ve çözme, Python kütüphanelerinin (örneğin, `numpy`, `pandas`) kurulumunda bağımlılık çakışmalarını (dependency conflicts) yönetmek için kullanılır. Çakışmalar, genellikle bir paketin belirli bir sürüm gerektirmesi (örneğin, `numpy==1.26.4`), ancak başka bir paketin farklı bir sürüm gerektirmesi (örneğin, `numpy>=2.0`) durumunda ortaya çıkar. Bu süreç, aşağıdaki adımlarla gerçekleştirilir:

1. **Çakışma Önleme**: Kurulum öncesi, mevcut bağımlılıkların durumu kontrol edilerek çakışmaların önüne geçilmeye çalışılır.
2. **Çakışma Tespiti**: Kurulum sırasında veya sonrasında, `pip check` komutu kullanılarak çakışmalar tespit edilir.
3. **Çakışma Çözme**: Tespit edilen çakışmalar, karar ağacı (`DecisionTreeClassifier`) ve nöral ağ (`MLPClassifier`) tabanlı yöntemlerle çözülmeye çalışılır.

Bu işlemler, `ConflictManager` sınıfı tarafından koordine edilir ve diğer bileşenlerle (loglama, önbellek, bağımlılık kaydı) entegre edilir.

#### **1. Çakışma Önleme**
Çakışma önleme, paket kurulumundan önce mevcut bağımlılıkların durumunu kontrol ederek potansiyel çakışmaları azaltmayı amaçlar. Bu süreçte kullanılan mekanizmalar:

- **Bağımlılık Kayıt Kontrolü (`DependencyRegistry`)**:
  - `DependencyRegistry.check_package` metodu, bir paketin son 24 saat içinde başarıyla kurulup kurulmadığını kontrol eder:
    ```python
    def check_package(self, package: str) -> bool:
        try:
            if package in self.registry["packages"]:
                pkg_info = self.registry["packages"][package]
                if pkg_info["status"] == "Başarılı":
                    elapsed = datetime.now() - datetime.fromisoformat(pkg_info["timestamp"])
                    if elapsed.total_seconds() < 24 * 60 * 60:  # 24 saat
                        self.logger.log("info", f"{package} zaten yüklü ve güncel.")
                        return True
            return False
        except Exception as e:
            self.logger.log("error", f"{package} kontrol hatası: {e}")
            return False
    ```
  - Eğer paket zaten yüklüyse, kurulum atlanır (`install_package` veya `async_install_package` metodlarında):
    ```python
    if self.dependency_registry.check_package(package):
        self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
        return
    ```
  - **Amaç**: Gereksiz yeniden kurulumları önleyerek çakışma riskini azaltmak.
  - **Kullanılan Dosya**: `dependencies.json` (`CACHE_DIR / "dependencies.json"`, örneğin, `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_cache\dependencies.json`).
    - Bu dosya, kurulu paketlerin sürüm, durum ve zaman damgası bilgilerini içerir:
      ```json
      {
        "packages": {
          "numpy==1.26.4": {
            "version": "1.26.4",
            "status": "Başarılı",
            "dependencies": [],
            "timestamp": "2025-06-21T17:06:00"
          }
        },
        "resolutions": {},
        "status": "conflict_free",
        "timestamp": "2025-06-21T17:06:00"
      }
      ```

- **Önbellek Kontrolü (`CacheManager`)**:
  - `CacheManager.install_from_cache` metodu, paketin önbellekte olup olmadığını kontrol eder:
    ```python
    cache_file = self.cache_dir / f"{package}.whl"
    if cache_file.exists():
        self.logger.log("info", f"{package} önbellekten yükleniyor.")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(cache_file)], check=True, capture_output=True, text=True
        )
        self.logger.log("info", f"{package} önbellekten kuruldu: {result.stdout}")
        return True
    ```
  - Önbellekteki paket, çakışmaya neden olabilecek eski bir sürüm değilse, doğrudan kurulur ve çakışma riski azalır.
  - **Kullanılan Dosyalar**:
    - Önbellek dosyaları: `.pdsx_cache/wheels` dizininde (örneğin, `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_cache\wheels\numpy-1.26.4.whl`).
    - Metadata dosyası: `.pdsx_cache/packages.json`, paketlerin hash, sürüm ve zaman damgası bilgilerini içerir:
      ```json
      {
        "numpy==1.26.4": {
          "version": "1.26.4",
          "timestamp": "2025-06-21T17:06:00",
          "hash": "<sha256_hash>"
        }
      }
      ```

- **Loglama ile İzleme (`AdvancedLogger`)**:
  - Çakışma önleme adımları, `AdvancedLogger` ile loglanır. Örneğin, bir paket zaten yüklüyse:
    ```python
    self.logger.log("info", f"{package} zaten yüklü, kurulum atlanıyor.")
    ```
  - **Kullanılan Dosyalar**:
    - `logs/pdsXu_terminal.log`: Düz metin loglar.
    - `logs/pdsxu_info.jsonl`, `logs/pdsxu_warnings.jsonl`, `logs/pdsxu_errors.jsonl`: JSONL formatında loglar.
    - Örnek JSONL log:
      ```json
      {"time": "2025-06-21T17:06:00", "level": "INFO", "message": "numpy==1.26.4 zaten yüklü, kurulum atlanıyor."}
      ```

**Sonuç**: Çakışma önleme, `DependencyRegistry` ile kurulu paketlerin kontrol edilmesi ve `CacheManager` ile önbellekten kurulum yapılmasıyla gerçekleştirilir. Bu, gereksiz kurulumları engelleyerek çakışma riskini azaltır. Kullanılan dosyalar: `dependencies.json`, `packages.json` ve log dosyaları.

#### **2. Çakışma Tespiti**
Çakışmalar, kurulum sırasında veya sonrasında `pip check` komutu kullanılarak tespit edilir. Bu süreç, `ConflictManager` sınıfının `detect_conflicts` metodu tarafından yönetilir.

- **Metod: `detect_conflicts`**
  ```python
  def detect_conflicts(self, module_name: str, deps: List[str]) -> Dict:
      try:
          self.logger.log("info", f"{module_name} için çakışma kontrolü başlatılıyor.")
          conflicts = {}
          for dep in deps:
              result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
              if "no conflicts" not in result.stdout.lower():
                  conflicts[dep] = result.stdout
                  self.logger.log("warning", f"Çakışma tespit edildi: {dep}, {result.stdout}")
              else:
                  self.logger.log("info", f"{dep} için çakışma bulunamadı.")
          self.conflicts[module_name] = conflicts
          return conflicts
      except Exception as e:
          self.logger.log("error", f"Çakışma kontrol hatası: {e}")
          return {}
  ```
- **Çalışma Mantığı**:
  - `pip check` komutu, mevcut ortamda yüklü paketlerin bağımlılık gereksinimlerini kontrol eder. Örneğin, eğer `numpy==1.26.4` yüklüyse, ancak başka bir paket `numpy>=2.0` gerektiriyorsa, bu bir çakışma olarak raporlanır.
  - Her paket (`deps` listesindeki) için `pip check` çalıştırılır ve çakışma varsa, `conflicts` sözlüğüne eklenir.
  - Çakışma yoksa, logda `"info", f"{dep} için çakışma bulunamadı."` mesajı yazılır.
  - Çakışma varsa, logda `"warning", f"Çakışma tespit edildi: {dep}, {result.stdout}"` mesajı yazılır ve çakışma detayları kaydedilir.
- **Kullanılan Dosyalar**:
  - **Log Dosyaları**: Çakışma tespit sonuçları loglanır:
    - `logs/pdsxu_warnings.jsonl`: Çakışma uyarıları.
    - `logs/pdsxu_errors.jsonl`: Hata durumunda.
    - `logs/pdsXu_terminal.log`: Düz metin loglar.
    - Örnek:
      ```json
      {"time": "2025-06-21T17:06:00", "level": "WARNING", "message": "Çakışma tespit edildi: numpy==1.26.4, numpy>=2.0 gerektiren paket bulundu."}
      ```
  - **Geçici Dosyalar**: `pip check` komutu, ortamın `site-packages` dizinini (örneğin, `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_isolated_env\Lib\site-packages`) tarar, ancak yeni dosya oluşturmaz.
- **Ne Zaman Çağrılır?**:
  - `install_package` ve `async_install_package` metodlarında, kurulum tamamlandıktan sonra çakışmalar kontrol edilir:
    ```python
    conflicts = self.conflict_manager.detect_conflicts(package, [package])
    ```
  - `run` metodunda, birden fazla paket için çakışmalar toplu kontrol edilir:
    ```python
    conflicts = self.conflict_manager.detect_conflicts("pdsX", packages)
    ```

**Sonuç**: Çakışma tespiti, `pip check` ile yapılır ve `ConflictManager.detect_conflicts` tarafından yönetilir. Çakışmalar, log dosyalarına (`pdsxu_warnings.jsonl`, `pdsxu_errors.jsonl`, `pdsXu_terminal.log`) kaydedilir. Ortamın `site-packages` dizini taranır, ancak ek dosya oluşturulmaz.

#### **3. Çakışma Çözme**
Çakışmalar tespit edildikten sonra, `ConflictManager` sınıfının `resolve_conflicts` ve `neural_conflict_resolution` metodları ile çözülmeye çalışılır. Bu süreç, makine öğrenmesi tabanlı yöntemler (karar ağacı ve nöral ağ) kullanır.

- **Metod: `resolve_conflicts`**
  ```python
  def resolve_conflicts(self, module_name: str, conflicts: Dict) -> Dict:
      try:
          resolutions = {}
          clf = self.build_decision_tree(conflicts)
          if clf:
              for dep, issue in conflicts.items():
                  features = [len(issue), issue.count("=="), issue.count("<"), issue.count(">")]
                  prediction = clf.predict([features])[0]
                  cmd = f"pip install {self.clean_version(dep)} --force-reinstall" if prediction == 1 else f"pip install {self.clean_version(dep)}"
                  try:
                      result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
                      resolutions[dep] = {"command": cmd, "reason": issue, "output": result.stdout}
                      self.logger.log("info", f"Çakışma çözüldü: {dep}, {issue}")
                  except subprocess.CalledProcessError as e:
                      self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
          neural_resolutions = self.neural_conflict_resolution(module_name, conflicts)
          resolutions.update(neural_resolutions)
          self.resolutions[module_name] = resolutions
          metrics = [len(conflicts), sum(len(v) for v in conflicts.values())]
          quantum_result = self.scientific_utils.quantum_load_simulation(metrics)
          if "error" not in quantum_result:
              self.logger.log("info", f"Kuantum analizi sonucu: {quantum_result}")
          return resolutions
      except Exception as e:
          self.logger.log("error", f"Çakışma çözüm hatası: {e}")
          return {}
  ```
- **Çalışma Mantığı**:
  1. **Karar Ağacı ile Çözüm (`build_decision_tree`)**:
     - Çakışma mesajlarının özelliklerini (uzunluk, `==`, `<`, `>` sayısı) kullanarak bir karar ağacı (`DecisionTreeClassifier`) oluşturulur:
       ```python
       def build_decision_tree(self, conflicts: Dict) -> DecisionTreeClassifier:
           try:
               X = []
               y = []
               for dep, issue in conflicts.items():
                   features = [len(issue), issue.count("=="), issue.count("<"), issue.count(">")]
                   X.append(features)
                   y.append(1 if "force-reinstall" in issue.lower() else 0)
               clf = DecisionTreeClassifier(max_depth=5)
               clf.fit(X, y)
               self.logger.log("info", "Karar ağacı oluşturuldu.")
               return clf
           except Exception as e:
               self.logger.log("error", f"Karar ağacı oluşturma hatası: {e}")
               return None
       ```
     - Karar ağacı, çakışmanın `--force-reinstall` ile mi yoksa normal kurulumla mı çözüleceğini tahmin eder.
     - Örnek: Eğer `numpy==1.26.4` ile çakışma varsa, karar ağacı `--force-reinstall` önerirse:
       ```python
       cmd = f"pip install {self.clean_version(dep)} --force-reinstall"
       ```
       Bu, `pip install numpy==1.26.4 --force-reinstall` komutunu çalıştırır.
     - Komut başarıyla çalışırsa, çözüm `resolutions` sözlüğüne eklenir ve loglanır:
       ```python
       resolutions[dep] = {"command": cmd, "reason": issue, "output": result.stdout}
       self.logger.log("info", f"Çakışma çözüldü: {dep}, {issue}")
       ```
     - Başarısız olursa, hata loglanır:
       ```python
       self.logger.log("error", f"Çakışma çözme hatası: {dep}, {e.output}")
       ```
  2. **Nöral Ağ ile Çözüm (`neural_conflict_resolution`)**:
     - Eğer karar ağacı yeterli değilse, nöral ağ (`MLPClassifier`) devreye girer:
       ```python
       def neural_conflict_resolution(self, module_name: str, conflicts: Dict) -> Dict:
           try:
               if not conflicts:
                   self.logger.log("info", f"{module_name} için nöral ağ çakışma çözümü: Çakışma yok.")
                   return {}
               X = []
               for v in conflicts.values():
                   features = [len(conflicts), sum(len(val) for val in conflicts.values()), len(v.split())]
                   X.append(features)
               y = [1 if any(dep.lower() in ["numpy", "tensorflow", "thinc"] for dep in conflicts) else 0]
               clf = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=500)
               clf.fit(X, y)
               prediction = clf.predict(X)
               resolutions = {}
               if prediction[0] == 1:
                   for dep in conflicts:
                       if "numpy" in dep.lower():
                           resolutions[dep] = {"command": "pip install numpy==1.26.4 --force-reinstall", "reason": "Nöral ağ önerisi: numpy çakışması"}
                       elif "tensorflow" in dep.lower():
                           resolutions[dep] = {"command": "pip install tensorflow==2.15.0 --force-reinstall", "reason": "Nöral ağ önerisi: tensorflow çakışması"}
                       elif "thinc" in dep.lower():
                           resolutions[dep] = {"command": "pip install thinc==8.3.2 --force-reinstall", "reason": "Nöral ağ önerisi: thinc çakışması"}
               self.logger.log("info", f"Nöral ağ çakışma çözümü: {resolutions}")
               return resolutions
           except Exception as e:
               self.logger.log("error", f"Nöral ağ çakışma çözüm hatası: {e}")
               return {}
       ```
     - Nöral ağ, özellikle `numpy`, `tensorflow` ve `thinc` gibi bilinen paketlerde çakışmalar için özel çözümler önerir (örneğin, `numpy==1.26.4` için `--force-reinstall`).
     - Çözümler, `resolutions` sözlüğüne eklenir ve loglanır.
  3. **Kuantum Analizi (`quantum_load_simulation`)**:
     - Çakışmaların yoğunluğunu analiz etmek için `ScientificUtils.quantum_load_simulation` çağrılır:
       ```python
       metrics = [len(conflicts), sum(len(v) for v in conflicts.values())]
       quantum_result = self.scientific_utils.quantum_load_simulation(metrics)
       if "error" not in quantum_result:
           self.logger.log("info", f"Kuantum analizi sonucu: {quantum_result}")
       ```
     - Bu, çakışmaların istatistiksel analizini yapar (ortalama, standart sapma, aykırı değerler) ve kullanıcıya ek bilgi sağlar.
- **Kullanılan Dosyalar**:
  - **Log Dosyaları**:
    - `logs/pdsxu_info.jsonl`: Çakışma çözüm başarıları.
    - `logs/pdsxu_errors.jsonl`: Çözüm başarısızlıkları.
    - `logs/pdsXu_terminal.log`: Düz metin loglar.
    - Örnek:
      ```json
      {"time": "2025-06-21T17:06:00", "level": "INFO", "message": "Çakışma çözüldü: numpy==1.26.4, numpy>=2.0 gerektiren paket bulundu."}
      ```
  - **Bağımlılık Kaydı**: Çözümler, `DependencyRegistry.register_resolution` ile `dependencies.json`’a kaydedilir:
    ```python
    self.dependency_registry.register_resolution(package, resolutions)
    ```
    - Örnek:
      ```json
      {
        "resolutions": {
          "numpy==1.26.4": {
            "command": "pip install numpy==1.26.4 --force-reinstall",
            "reason": "numpy>=2.0 gerektiren paket bulundu.",
            "output": "<pip çıktısı>"
          }
        }
      }
      ```
  - **Önbellek Dosyaları**: Çözüm için yeniden kurulum gerekiyorsa, `.pdsx_cache/wheels` dizininden paket alınabilir.
  - **Ortam Dosyaları**: Yeniden kurulum, `site-packages` dizinini günceller (örneğin, `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_isolated_env\Lib\site-packages`).
- **Ne Zaman Çağrılır?**:
  - `install_package` ve `async_install_package` metodlarında, çakışma tespit edildikten sonra:
    ```python
    if conflicts:
        resolutions = self.conflict_manager.resolve_conflicts(package, conflicts)
        self.dependency_registry.register_resolution(package, resolutions)
    ```
  - `run` metodunda, toplu paket kurulumu sonrası:
    ```python
    if conflicts:
        resolutions = self.conflict_manager.resolve_conflicts("pdsX", conflicts)
        for pkg, res in resolutions.items():
            self.dependency_registry.update_on_conflict(pkg, str(conflicts.get(pkg, "")), res["command"])
    ```

**Sonuç**: Çakışma çözme, `ConflictManager.resolve_conflicts` ve `neural_conflict_resolution` ile yapılır. Karar ağacı ve nöral ağ, çakışmaları çözmek için `--force-reinstall` veya normal kurulum önerir. Kullanılan dosyalar: `dependencies.json`, log dosyaları (`pdsxu_info.jsonl`, `pdsxu_errors.jsonl`, `pdsXu_terminal.log`), önbellek dosyaları ve `site-packages`.

---

### **Hangi Dosyalar Kullanılıyor?**

Aşağıda, çakışma önleme ve çözme işlemlerinde kullanılan tüm dosyalar listeleniyor:

1. **`dependencies.json`**:
   - **Yer**: `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_cache\dependencies.json`
   - **Amaç**: Kurulan paketlerin durumunu ve çakışma çözümlerini kaydeder.
   - **Kullanıldığı Yer**:
     - `DependencyRegistry.check_package`: Zaten yüklü paketleri kontrol eder.
     - `DependencyRegistry.register_resolution`: Çakışma çözümlerini kaydeder.
     - `DependencyRegistry.update_on_conflict`: Çakışma bilgilerini günceller.

2. **`packages.json`**:
   - **Yer**: `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_cache\packages.json`
   - **Amaç**: Önbellek dosyalarının metadata’sını (sürüm, hash, zaman damgası) saklar.
   - **Kullanıldığı Yer**:
     - `CacheManager.install_from_cache`: Önbellekten kurulum.
     - `CacheManager._download_and_cache`: Yeni paketlerin metadata’sını kaydeder.

3. **Önbellek Dosyaları**:
   - **Yer**: `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_cache\wheels`
   - **Amaç**: İndirilen paketlerin `.whl` dosyalarını saklar.
   - **Kullanıldığı Yer**:
     - `CacheManager.install_from_cache`: Önbellekten kurulum.
     - `CacheManager._download_and_cache`: Paket indirme ve kaydetme.

4. **Log Dosyaları**:
   - **Yer**: `C:\Users\mete\Zotero\basic\pdsXuv14\logs`
   - **Dosyalar**:
     - `pdsXu_terminal.log`: Düz metin loglar.
     - `pdsxu_terminal.jsonl`: Genel JSONL loglar.
     - `pdsxu_info.jsonl`: Bilgi logları.
     - `pdsxu_warnings.jsonl`: Uyarı logları (örneğin, çakışma tespitleri).
     - `pdsxu_errors.jsonl`: Hata logları (örneğin, çakışma çözüm başarısızlıkları).
   - **Amaç**: Çakışma önleme, tespit ve çözüm adımlarını kaydeder.
   - **Kullanıldığı Yer**:
     - `AdvancedLogger.log`: Tüm loglama işlemleri.
     - `ConflictManager.detect_conflicts`: Çakışma tespit logları.
     - `ConflictManager.resolve_conflicts`: Çakışma çözüm logları.

5. **Ortam Dosyaları (`site-packages`)**:
   - **Yer**: `C:\Users\mete\Zotero\basic\pdsXuv14\.pdsx_isolated_env\Lib\site-packages`
   - **Amaç**: Kurulan paketlerin dosyalarını saklar.
   - **Kullanıldığı Yer**:
     - `pip check`: Çakışma tespiti için ortam taranır.
     - `pip install`: Paket kurulumları ve `--force-reinstall` işlemleri.

---

### **Genel İş Akışı**

1. **Çakışma Önleme**:
   - `DependencyRegistry.check_package`, paketin zaten yüklü olup olmadığını kontrol eder (`dependencies.json`).
   - `CacheManager.install_from_cache`, önbellekte uygun sürüm varsa kurar (`packages.json`, `.whl` dosyaları).
   - Loglama: `AdvancedLogger` ile kaydedilir (`logs` dizini).

2. **Çakışma Tespiti**:
   - `ConflictManager.detect_conflicts`, `pip check` ile çakışmaları tespit eder (`site-packages` taranır).
   - Çakışmalar, `conflicts` sözlüğüne eklenir ve loglanır (`pdsxu_warnings.jsonl`, `pdsXu_terminal.log`).

3. **Çakışma Çözme**:
   - `ConflictManager.resolve_conflicts`, karar ağacı ve nöral ağ ile çözüm önerir.
   - Önerilen komutlar (`pip install --force-reinstall`) çalıştırılır.
   - Çözümler, `dependencies.json`’a kaydedilir ve loglanır (`pdsxu_info.jsonl`, `pdsxu_errors.jsonl`).
   - Önbellek gerekiyorsa kullanılır (`.pdsx_cache/wheels`).

---

### **Örnek Senaryo**

**Senaryo**: `numpy==1.26.4` gerektiren bir modül ve `numpy>=2.0` gerektiren başka bir modül var.

1. **Önleme**:
   - `DependencyRegistry.check_package("numpy==1.26.4")`, paketin yüklü olduğunu tespit eder ve loglar:
     ```json
     {"time": "2025-06-21T17:06:00", "level": "INFO", "message": "numpy==1.26.4 zaten yüklü, kurulum atlanıyor."}
     ```
   - `dependencies.json` güncellenir.

2. **Tespit**:
   - `ConflictManager.detect_conflicts`, `pip check` ile çakışma bulur:
     ```json
     {"time": "2025-06-21T17:06:00", "level": "WARNING", "message": "Çakışma tespit edildi: numpy==1.26.4, numpy>=2.0 gerektiren paket bulundu."}
     ```

3. **Çözüm**:
   - `resolve_conflicts`, nöral ağ ile `pip install numpy==1.26.4 --force-reinstall` önerir.
   - Komut çalışır ve çözüm loglanır:
     ```json
     {"time": "2025-06-21T17:06:00", "level": "INFO", "message": "Çakışma çözüldü: numpy==1.26.4, numpy>=2.0 gerektiren paket bulundu."}
     ```
   - `dependencies.json`’a çözüm kaydedilir:
     ```json
     {
       "resolutions": {
         "numpy==1.26.4": {
           "command": "pip install numpy==1.26.4 --force-reinstall",
           "reason": "numpy>=2.0 gerektiren paket bulundu.",
           "output": "<pip çıktısı>"
         }
       }
     }
     ```

---

### **Sonuç**
- **Çakışma Önleme**: `DependencyRegistry` ve `CacheManager` ile yapılır. Kullanılan dosyalar: `dependencies.json`, `packages.json`, önbellek dosyaları, log dosyaları.
- **Çakışma Tespiti**: `ConflictManager.detect_conflicts` ile `pip check` kullanılır. Kullanılan dosyalar: log dosyaları, `site-packages`.
- **Çakışma Çözme**: `ConflictManager.resolve_conflicts` ve `neural_conflict_resolution` ile karar ağacı ve nöral ağ tabanlı çözümler uygulanır. Kullanılan dosyalar: `dependencies.json`, log dosyaları, önbellek dosyaları, `site-packages`.

Eğer ek sorularınız varsa veya başka bir detayı açıklamamı isterseniz, lütfen belirtin!