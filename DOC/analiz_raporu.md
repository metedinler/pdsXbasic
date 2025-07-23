### **Referans Dosya 4: `auto_importer.py` (Sürüm 1.7.9.5)**

- **Dosya Yolu**: `c:\Users\mete\Zotero\basic\pdsXuv14\auto_importer.py`
- **Analiz Tarihi**: 26.07.2024

#### a. Genel Bakış ve Mimari

Bu sürüm, projedeki en kapsamlı ve gelişmiş `auto_importer` uygulamasıdır. Yüksek derecede otonom, kendi kendini iyileştiren ve sağlam bir sistem olarak tasarlanmıştır. Neredeyse tüm önceki sürümlerin özelliklerini bir araya getirir ve üzerine önemli eklemeler yapar. Mimari, ortam yönetimi, akıllı kurulum, hata kurtarma, kapsamlı günlük kaydı ve önbelleğe almayı birleştiren modüler bir yapıya dayanır.

#### b. Temel Özellikler ve Mekanizmalar

- **Graceful Shutdown Yöneticisi (`GracefulShutdownManager`)**:
  - **İşlev**: Sinyal yakalama (`SIGINT`, `SIGTERM`, `SIGBREAK`) ve özel bir klavye kısayolu (`Ctrl+Shift+Q`) ile programın güvenli bir şekilde sonlandırılmasını sağlar.
  - **Mekanizma**: `signal` ve `atexit` modüllerini kullanarak temizlik fonksiyonlarını ve aktif süreçlerin sonlandırılmasını yönetir. Acil durumlar için ayrı bir "kill switch" mantığı içerir.

- **Gelişmiş Günlükleme (`AdvancedLogger`)**:
  - **İşlev**: Seviye tabanlı (INFO, WARNING, ERROR), JSONL formatında yapılandırılmış günlükler oluşturur. Ayrıca düz metin bir terminal günlüğü de tutar.
  - **Mekanizma**: `logging` modülünü kullanır, `stdout` ve `stderr`'i `Tee` sınıfı aracılığıyla hem konsola hem de dosyalara yönlendirir. Log rotasyonu, spam koruması ve isteğe bağlı Elasticsearch entegrasyonu gibi gelişmiş özelliklere sahiptir.

- **Ortam Yöneticisi (`EnvManager`)**:
  - **İşlev**: `.pdsx_isolated_env` adında yalıtılmış bir sanal ortamı yönetir.
  - **Mekanizma**: Gerekli Python 3.10 sürümünü sistemde arar (PATH ve Windows Registry), bulamazsa indirip kurar. Belirli bir hata sayısına ulaşıldığında ortamı otomatik olarak silip yeniden oluşturarak "kendi kendini iyileştirme" yeteneği sunar. `pip` sürümünü de yönetir.

- **Önbellek Yöneticisi (`CacheManager`)**:
  - **İşlev**: Paket "wheel" dosyalarını yerel bir önbellekte saklayarak kurulumları hızlandırır.
  - **Mekanizma**: İndirilen `.whl` dosyalarını `.pdsx_cache/wheels` dizininde saklar. Kurulumdan önce bu önbelleği kontrol eder. Dosya bütünlüğünü SHA256 hash'leri ile doğrular. Paketleri geri alma (`rollback`) ve eski önbellek dosyalarını temizleme yeteneğine sahiptir.

- **Pip Çıktı Analizcisi (`PipOutputAnalyzer`)**:
  - **İşlev**: `pip` kurulumu sırasında oluşan yaygın hataları tanır ve otomatik olarak düzeltmeye çalışır.
  - **Mekanizma**: Sık karşılaşılan hata mesajlarını (örn: "Permission denied", "deadlock detected") ve bunlara karşılık gelen düzeltme komutlarını (örn: `--user`, `--no-cache-dir`) içeren bir sözlük kullanır.

- **Bağımlılık ve Terminal Analizi**:
  - **İşlev**: Kapsamlı bir çekirdek bağımlılık listesi (`REQUIRED_PACKAGES`) içerir. Terminal çıktısını analiz ederek `ModuleNotFoundError` gibi hatalardan eksik modülleri tespit eder.
  - **Mekanizma**: Regex kullanarak terminal loglarını tarar ve `pip install` önerilerini veya import hatalarını yakalar.

- **Asenkron Yetenekler ve Tembel Yükleme (Lazy Loading)**:
  - **İşlev**: `asyncio` kullanarak paket kurulumu gibi işlemleri asenkron yapma potansiyeli sunar. `numpy`, `sklearn` gibi büyük kütüphaneleri sadece gerektiğinde yükler.
  - **Mekanizma**: `async def` fonksiyonlar ve `get_numpy()` gibi sarmalayıcı (wrapper) fonksiyonlar kullanır.

#### c. `auto_importer_fixed.py` ile Karşılaştırma ve Öne Çıkan Farklar

- **Otonom Kurulum**: Bu sürüm, `EnvManager` aracılığıyla Python 3.10'u bile kendi başına kurabilir. Bu, `auto_importer_fixed.py`'de bulunmayan çok ileri bir otonomi seviyesidir.
- **Kendi Kendini İyileştirme**: Sanal ortamın bozulması durumunda otomatik olarak yeniden oluşturulması, bu sürümü daha dayanıklı kılar.
- **Güvenli Kapatma**: `GracefulShutdownManager`, özellikle uzun süren veya karmaşık işlemleri yönetirken sistemin kararlılığını artıran kritik bir özelliktir.
- **Gelişmiş Hata Düzeltme**: `PipOutputAnalyzer`, `auto_importer_fixed.py`'deki genel hata yakalamadan daha spesifik ve proaktif bir hata düzeltme mantığı sunar.
- **Terminal Analizi**: Terminal loglarını canlı olarak analiz etme yeteneği, `auto_importer_fixed.py`'nin statik analizine kıyasla daha dinamik bir bağımlılık tespiti sağlar.
- **Genel Karmaşıklık**: Bu sürüm, diğer tüm referanslardan önemli ölçüde daha karmaşık ve daha fazla özelliğe sahiptir. Neredeyse bir işletim sistemi alt katmanı gibi davranır.

### 5. auto_importerv17941.py (Sürüm 1.7.9.4.001)

- **Dosya Yolu**: `c:\Users\mete\Zotero\basic\pdsXuv14\auto_importerv17941.py`
- **Analiz Tarihi**: 26.07.2024

#### a. Genel Bakış ve Mimari

Bu sürüm, `auto_importer.py` (v1.7.9.5) ile çok büyük benzerlikler taşıyan, ondan bir önceki ana sürüm gibi görünen bir yapıya sahiptir. Temel mimari yine modülerdir ve otonom çalışmayı hedefler. Bu sürümün en ayırt edici özelliği, terminal loglarını proaktif olarak analiz ederek eksik bağımlılıkları "öğrenmesi" ve komut satırı argümanlarını kaydedip tekrar yürütebilmesidir.

#### b. Temel Özellikler ve Mekanizmalar

- **Terminal Log Analizcisi (`TerminalLogAnalyzer`)**:
  - **İşlev**: Hem düz metin (`pdsXu_terminal.log`) hem de JSONL (`pdsxu_terminal.jsonl`) formatındaki log dosyalarını tarar. `ModuleNotFoundError`, `ImportError` ve `pip install` önerilerini tespit eder.
  - **Mekanizma**: Regex kullanarak log satırlarını ayrıştırır ve eksik paket adlarını ve potansiyel versiyon numaralarını çıkarır. Bu, sistemin çalışma zamanında karşılaştığı hatalardan öğrenmesini sağlar.

- **Öğrenen Bağımlılık Kaydı (`DependencyRegistry`)**:
  - **İşlev**: `TerminalLogAnalyzer` tarafından tespit edilen paketleri `dependencies.json` dosyasına `auto_discovered` anahtarı altında kaydeder.
  - **Mekanizma**: Bu, sistemin zamanla daha akıllı hale gelmesini sağlar. Bir kere karşılaşılan ve çözülen bir bağımlılık hatası, kayıt defterine eklenir ve gelecekteki çalıştırmalarda otomatik olarak dikkate alınabilir.

- **Argüman Kaydetme ve Tekrar Yürütme (`argparse`)**:
  - **İşlev**: Komut satırından alınan argümanları `.pdsx_last_args.json` dosyasına kaydeder. `--replay` argümanı ile son komutu aynı argümanlarla tekrar çalıştırır.
  - **Mekanizma**: `argparse` modülü ile CLI argümanlarını yönetir. `AutoImporter` sınıfının `save_last_args` ve `replay_last_command` metotları bu işlevselliği sağlar. Bu, özellikle hata ayıklama ve tekrarlanabilirlik için güçlü bir özelliktir.

- **Agresif Tembel Yükleme (Lazy Loading)**:
  - **İşlev**: `numpy`, `sklearn` gibi kütüphaneleri sadece gerektiğinde yükler.
  - **Mekanizma**: Diğer sürümlerden farklı olarak, `get_numpy()` gibi bir fonksiyon çağrıldığında modül yüklü değilse, sadece `ImportError` vermekle kalmaz, `subprocess` kullanarak o anda paketi yüklemeye çalışır. Bu, daha proaktif bir yaklaşımdır.

- **Diğer Ortak Özellikler**:
  - `GracefulShutdownManager`, `EnvManager`, `CacheManager` ve `PipOutputAnalyzer` gibi sınıflar, `auto_importer.py` (v1.7.9.5) sürümündekine çok benzer şekilde mevcuttur ve otonom ortam yönetimi, önbellekleme ve hata düzeltme yetenekleri sunar.

#### c. `auto_importer_fixed.py` ile Karşılaştırma ve Öne Çıkan Farklar

- **Dinamik vs. Statik Analiz**: `auto_importer_fixed.py` statik kod analizi (`CodeAnalyzer`) ile importları bulurken, bu sürüm çalışma zamanı hatalarını (`TerminalLogAnalyzer`) analiz eder. Bu, dinamik olarak yüklenen veya `exec` ile çalıştırılan kodlardaki bağımlılıkları bile yakalayabilme potansiyeli sunar.
- **Öğrenme Yeteneği**: Terminal analiziyle öğrenip bunu bir JSON dosyasına kaydetme özelliği, `auto_importer_fixed.py`'de bulunmayan, sistemi zamanla evrimleştiren önemli bir farktır.
- **Tekrarlanabilirlik (`--replay`)**: Argüman kaydetme ve tekrar yürütme, `auto_importer_fixed.py`'de olmayan, otomasyon ve hata ayıklama için değerli bir araçtır.
- **Otonomi Seviyesi**: Bu sürüm de Python'u ve sanal ortamı kendi kendine yönetebilme yeteneğiyle `auto_importer_fixed.py`'den çok daha otonomdur.

---

### 6. `auto_importer.py` (v1.7.9.5 - Birleştirilmiş Sürüm)

Bu sürüm, projedeki en karmaşık ve kapsamlı implementasyondur. Diğer referans dosyalardaki birçok özelliği birleştirir ve üzerine yeni yetenekler ekler. Neredeyse tam otonom bir sistem hedeflenmiştir.

**Anahtar Özellikler ve Mekanizmalar:**

1.  **Graceful Shutdown Manager (Zarif Kapatma Yöneticisi):**
    *   **Sinyal Yakalama:** `SIGINT` (Ctrl+C), `SIGTERM` (normal kapatma) ve Windows'a özel `SIGBREAK` sinyallerini yakalayarak programın aniden sonlanmasını engeller.
    *   **Acil Durum Kapatma:** `keyboard` kütüphanesini kullanarak `Ctrl+Shift+Q` klavye kısayolu ile tetiklenen bir "kill switch" içerir. Bu, sistemin donduğu veya yanıt vermediği durumlarda bile güvenli bir çıkış sağlar.
    *   **Kaynak Temizliği:** `atexit` modülü ile programın çıkışında çalışacak temizlik fonksiyonları kaydeder. Bu fonksiyonlar, açık dosyaları kapatır ve başlatılan alt işlemleri sonlandırır.

2.  **AdvancedLogger (Gelişmiş Kayıt Sistemi):**
    *   **Çoklu Dosya Kaydı:** Logları seviyelerine (`INFO`, `WARNING`, `ERROR`) göre ayrı JSONL formatındaki dosyalara yazar. Bu, logların yapısal olarak analiz edilmesini kolaylaştırır.
    *   **Düz Metin Terminal Kaydı:** Tüm terminal çıktılarını, zaman damgalarıyla birlikte `pdsXu_terminal.log` adlı bir düz metin dosyasına da kaydeder.
    *   **Stdout/Stderr Yönlendirme:** `sys.stdout` ve `sys.stderr` akışlarını kendi `Tee` sınıfı aracılığıyla hem konsola hem de ilgili log dosyalarına yönlendirir.
    *   **Elasticsearch Entegrasyonu (Opsiyonel):** Logları merkezi bir Elasticsearch sunucusuna gönderme yeteneğine sahiptir. Bu, büyük ölçekli log analizi için kullanışlıdır.
    *   **Log Rotasyonu ve Yedekleme:** Log dosyaları belirli bir boyuta ulaştığında (`MAX_LOG_SIZE`) otomatik olarak yedeklenir ve yeni bir log dosyası oluşturulur. Eski yedekler (`MAX_BACKUPS`) silinir.
    *   **Sessiz Mod (Silent Mode):** Kullanıcının sadece kritik hata ve uyarıları görmesini sağlayan bir sessiz mod içerir.

3.  **Bağımlılık ve Önbellek Yönetimi:**
    *   **DependencyRegistry:** Kurulan paketlerin adı, sürümü, durumu ve bağımlılıkları gibi bilgileri `dependencies.json` dosyasında saklar. Bu, gereksiz yere tekrar kurulum yapılmasını önler.
    *   **PipOutputAnalyzer:** `pip` komutlarının çıktısını analiz ederek sık karşılaşılan hataları ("ModuleNotFoundError", "Permission denied", "deadlock detected" vb.) tespit eder ve otomatik düzeltme stratejileri uygular (örn. `--force-reinstall`, `--no-cache-dir`, `--user`).
    *   **CacheManager (Wheel Önbelleği):** İndirilen paketlerin `.whl` (wheel) dosyalarını yerel bir önbellek dizininde (`.pdsx_cache/wheels`) saklar.
        *   **Bütünlük Doğrulama:** Önbellekteki dosyaların bütünlüğünü SHA256 hash'leri ile doğrular. Bozuk dosyaları silip yeniden indirir.
        *   **Sürüm Geri Alma (Rollback):** Bir paketin önceki bir sürümüne geri dönme yeteneği sunar.
        *   **Bağımlılık Ağacı Görselleştirme:** `graphviz` kütüphanesini kullanarak bir modülün bağımlılık ağacını görselleştirebilir.

4.  **EnvManager (İzole Ortam Yöneticisi):**
    *   **Sanal Ortam Yönetimi:** Tüm bağımlılıkları `.pdsx_isolated_env` adlı bir sanal ortam (venv) içinde yönetir. Bu, sistemin genel Python kurulumunu etkilemeyi önler.
    *   **Otomatik Ortam Onarımı:** Sanal ortamda belirli bir sayıda hata (`max_errors`) tespit edildiğinde, ortamı otomatik olarak silip yeniden oluşturur.
    *   **Python 3.10 Bulma ve Kurma:** Sistemde Python 3.10'u arar (PATH ve Windows Registry). Bulamazsa, Windows için Python 3.10 kurulum dosyasını indirip sessiz kurulum yapabilir.
    *   **PATH Güncelleme Yardımcıları:** Kullanıcının Python'ı ve sanal ortamın `Scripts` klasörünü sistemsel `PATH`'e eklemesini kolaylaştırmak için `.bat` ve `.ps1` scriptleri oluşturur.
    *   **Pip Güncelleme:** Sanal ortamdaki `pip`'i, Python 3.10 ile uyumluluğu test edilmiş belirli bir sürüme (`21.2.4`) veya en son sürüme günceller.

5.  **TerminalLogAnalyzer (Terminal Kayıt Analizörü):**
    *   **Gerçek Zamanlı Analiz:** `RealTimeLogMonitor` (watchdog tabanlı) ile log dosyasını sürekli izler.
    *   **Hata Tespiti:** Regex kullanarak loglardan `ModuleNotFoundError`, `ImportError` gibi hataları ve `pip install` önerilerini ayıklar.
    *   **Otonom Öğrenme ve Kurulum:** Tespit edilen eksik modülleri "öğrenir" ve bunları otonom olarak kurmayı tetikleyebilir. Kurulum kararını vermek için basit bir makine öğrenmesi modeli (Karar Ağacı) kullanır.

6.  **Argparse ve Komut Tekrar Oynatma (Replay):**
    *   **Gelişmiş Komut Satırı Arayüzü:** `argparse` kullanarak `--install`, `--check`, `--analyze`, `--replay` gibi çeşitli komut satırı argümanlarını destekler.
    *   **Son Komutu Kaydetme:** Çalıştırılan son komutu argümanlarıyla birlikte `.pdsx_last_args.json` dosyasına kaydeder.
    *   **Tekrar Oynatma:** `--replay` argümanı, başarısız olan veya tekrarlanması gereken son komutun kolayca yeniden çalıştırılmasını sağlar.

7.  **AutoImporter (Ana Sınıf):**
    *   **Entegrasyon:** Yukarıda listelenen tüm bileşenleri (Logger, EnvManager, CacheManager vb.) bir araya getirir ve orkestrasyonunu sağlar.
    *   **Asenkron İşlemler:** `asyncio` kullanarak paket indirme ve kurma gibi işlemleri asenkron olarak yürütebilir.
    *   **İstatistik ve Özet:** Kurulum süreleri, başarı/hata oranları gibi istatistikleri toplayan ve sonunda bir özet raporu sunan `SummaryGenerator` içerir.
    *   **Bootstrap Mekanizması:** AutoImporter'ın kendisinin çalışması için gereken temel bağımlılıkların (örn. `requests`, `colorama`) mevcut olduğundan emin olan bir başlangıç mekanizmasına sahiptir.

**Özet:** Bu sürüm, sadece bir "otomatik kurucu" olmanın çok ötesine geçerek, kendi kendini yöneten, onaran, öğrenen ve optimize eden bir sistem olmayı hedefler. Hata toleransı, kullanıcı etkileşimini en aza indirme ve şeffaflık (gelişmiş loglama ile) temel tasarım prensipleridir. Neredeyse tüm olası senaryolar (bozuk dosyalar, ağ hataları, izin sorunları, bağımlılık çakışmaları) düşünülerek tasarlanmıştır. `auto_importer_fixed.py` için en iyi referans ve hedef model budur.

---

### 7. `auto_importerv17941z.py` (v1.7.9.4.1 - Terminal Analizi ve Öğrenme)

Bu sürüm, `auto_importerv17941.py`'nin bir iyileştirmesidir ve `v1.7.9.5`'in temelini oluşturan birçok önemli özelliği içerir. Odak noktası, terminal çıktılarından öğrenerek otonom bir şekilde bağımlılıkları yönetmektir.

**Anahtar Özellikler ve Mekanizmalar:**

1.  **Lazy Loading ile Otomatik Kurulum:**
    *   `get_numpy()`, `get_sklearn_components()` gibi fonksiyonlar, bir modül bulunamadığında sadece hata vermekle kalmaz, `pip install` komutunu çalıştırarak eksik paketi anında kurmayı dener. Bu, sistemin kendi kendine yeterliliğini artıran önemli bir adımdır.

2.  **TerminalLogAnalyzer ve Bağımlılık Öğrenme:**
    *   Bu sürümün en belirgin özelliğidir. Hem düz metin (`.log`) hem de yapısal (`.jsonl`) log dosyalarını düzenli ifadeler (regex) kullanarak analiz eder.
    *   `ModuleNotFoundError`, `ImportError` ve `pip install` önerilerini tespit eder.
    *   **`DependencyRegistry` Entegrasyonu:** Tespit edilen bu yeni bağımlılıkları, `dependencies.json` dosyası içindeki `auto_discovered` adlı özel bir bölüme kaydeder. Bu, sistemin zamanla "öğrenmesini" ve gelecekteki çalışmalar için bu bilgiyi kalıcı hale getirmesini sağlar.

3.  **Gelişmiş Ortam Yönetimi (EnvManager):**
    *   **Pip Uyumluluk Testi:** `update_pip_if_needed` fonksiyonu, `pip`'i güncelledikten sonra `pip check` komutunu çalıştırarak herhangi bir uyumsuzluk olup olmadığını kontrol eder. Eğer bir sorun tespit ederse, otomatik olarak bilinen kararlı bir sürüme (`21.2.4`) geri döner. Bu, çok güçlü bir kendini onarma mekanizmasıdır.
    *   **Kurulum Döngüsü:** `setup_environment` metodu, sadece önceden tanımlanmış paketleri kurmakla kalmaz, aynı zamanda `TerminalLogAnalyzer` tarafından öğrenilen paketleri de kurmaya çalışır. Bu işlemi, tüm bağımlılıklar başarıyla kurulana kadar bir döngü içinde tekrarlayabilir.

4.  **Argparse Komut Satırı Arayüzü ve Tekrar Oynatma (Replay):**
    *   `v1.7.9.5`'e benzer şekilde, `--install`, `--check`, `--analyze-logs`, `--replay`, `--update-deps` (öğrenilen bağımlılıkları kurmak için) gibi zengin bir komut satırı arayüzü sunar.
    *   Çalıştırılan son komutu `.pdsx_last_args.json` dosyasına kaydederek `--replay` argümanı ile kolayca tekrar çalıştırılmasına olanak tanır.

5.  **Asenkron İşlemler ve Graceful Shutdown:**
    *   `asyncio` ve `ThreadPoolExecutor` kullanarak ağ ve dosya sistemi işlemlerini arka planda yürüterek ana programın kilitlenmesini önler.
    *   Sinyal yakalama (`SIGINT`, `SIGTERM`) mekanizması ile programın güvenli bir şekilde sonlandırılmasını, başlatılan alt işlemlerin ve asenkron görevlerin temizlenmesini sağlar.

**Özet:**

Bu sürüm, `v1.7.9.5`'in bir öncülüdür ve "öğrenme" yeteneğini sisteme kazandıran kritik bir versiyondur. Terminal loglarını proaktif bir şekilde analiz edip bağımlılık listesini dinamik olarak güncellemesi, onu statik bir kurucudan adaptif bir yardımcıya dönüştürür. `pip` uyumluluk kontrolü gibi kendini onarma mekanizmaları, sistemin kararlılığını önemli ölçüde artırır. `auto_importer_fixed.py`'ye entegre edilecek "otonom öğrenme" özelliği için mükemmel bir referanstır.

---

### 8. `auto.py`

Bu dosya boştur. Herhangi bir kod veya işlevsellik içermemektedir. Analiz için dikkate alınmayacaktır.
