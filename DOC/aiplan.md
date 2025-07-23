# PDS-X AutoImporter Geliştirme Master Planı (v3.1 - Durum Takibi)

**Tarih:** 29.06.2025
**Durum:** Proje planı, her bir madde için durum takibi içerecek şekilde güncellenmiştir. Eleştiriler doğrultusunda yeniden odaklanılmıştır.

---

## Bölüm 1: Proje Vizyonu ve Temel Prensipler (Değişmedi)

1.  **Tam Otonomi:** `AutoImporter`, PDS-X yorumlayıcısının bir parçasıdır. Dışarıdan minimum müdahale ile kendi kararlarını alabilmeli, hataları çözebilmeli ve ortamı PDS-X'in çalışması için hazır tutmalıdır.
2.  **Çift Kontrol Mekanizması:** Sistem, hem PDS-X tarafından programatik olarak (API aracılığıyla) hem de geliştirici tarafından manuel olarak (CLI aracılığıyla) yönetilebilmelidir.
3.  **Üçlü Kurulum Stratejisi:** Sistem, aşağıdaki üç stratejiyi de uyum içinde desteklemelidir:
    *   **Strateji 1 (Toplu Kurulum):** PDS-X betiği çalışmadan önce, `requirements.txt` veya modül listesinden tüm bağımlılıkların proaktif olarak kurulması.
    *   **Strateji 2 (Anında Kurulum):** Kod çalışırken karşılaşılan `ImportError` anında, eksik modülün anlık olarak kurulup çalışmaya devam edilmesi (`sys.meta_path` kancası).
    *   **Strateji 3 (Bağımlılık Zinciri):** Herhangi bir kurulum sırasında, `pip` çıktısından tespit edilen alt bağımlılıkların otomatik olarak kurulum kuyruğuna eklenmesi.
4.  **Gelişmiş Zeka ve Dayanıklılık:** Sistem, basit `pip install` komutlarından ibaret olmamalıdır. Hata analizi, çok adımlı çözüm denemeleri, kaynak durumuna göre adaptasyon ve çakışma yönetimi gibi zeki mekanizmalar içermelidir.
5.  **Kayıt ve Geriye Dönük Analiz:** `dependencies.json` gibi bir kayıt dosyası ile kurulan her paketin versiyonu, bağımlılıkları ve kurulum durumu kaydedilmelidir. Bu, hem tutarlılık sağlar hem de gelecekteki analizler için veri biriktirir.

---

## Bölüm 2: Referans Kodlardan Miras Alınacak Değerler (Değişmedi)

| Kaynak Dosya | Miras Alınacak Kilit Yetenek / Kod | Hedef Sınıf/Modül (`auto_importer_fixed.py`) |
| :--- | :--- | :--- |
| **`toplu*.py` Serisi | - Dosyadan (`.txt`) ve Python listesinden modül okuma.<br>- Sırayla `pip install` komutlarını tetikleme mantığı. | `AutoImporter` (Yeni `install_from_file` ve `install_from_list` metotları) |
| **`v1793(calisan).py`** | - `sys.meta_path` üzerine kurulan `PDSXFinder` sınıfının temel iskeleti. | `PDSXFinder` (Mevcut yapının temeli olarak korunacak) |
| **`v1795` & `(28).py`** | - **Gelişmiş `pip` Çıktı Analizi**<br>- **Çok Adımlı Çözüm Mantığı**<br>- **Kaynak İzleme**<br>- **Wheel Cache Fikri** | `PipOutputAnalyzer`<br>`AutoImporter._install_package_with_retry()`<br>`ResourceMonitor`<br>`WheelCacheManager` |
| **`aiozel.md` (Yeni Fikirler)** | - **Genetik Algoritma (`DependencyOptimizer`)**<br>- **Kaos Analizi (`ChaosLoadAnalyzer`)** | `DependencyOptimizer`<br>`ScientificUtils` |

---

## Bölüm 3: Nihai Uygulama Planı (Adım Adım)

### **Faz 1: Çekirdek Zekanın İnşası**

- **Madde 1: `PipOutputAnalyzer`'ı Güçlendirme**
    - **Durum:** <span style="color:green">**TAMAMLANDI**</span> - Sınıf, çeşitli pip hatalarını tanıyıp çözüm önerileri üretebiliyor.
- **Madde 2: Akıllı Kurulum Döngüsü Oluşturma**
    - **Durum:** <span style="color:green">**TAMAMLANDI**</span> - `_install_package_with_retry` metodu, `PipOutputAnalyzer`'dan gelen önerilere göre yeniden deneme mantığını içeriyor.

### **Faz 2: Stratejilerin ve Kontrol Mekanizmalarının Entegrasyonu**

- **Madde 3: Toplu Kurulum Metotları Ekleme**
    - **Durum:** <span style="color:green">**TAMAMLANDI**</span> - `install_from_list` ve `install_from_file` metotları eklendi. Kurulumların bitmesini beklemek için senkronizasyon mekanizması (`threading.Event`) entegre edildi.
- **Madde 4: Çift Kontrol Arayüzü Oluşturma**
    - **Durum:** <span style="color:green">**TAMAMLANDI**</span> - CLI argümanları ve programatik API (`trigger_task`) artık kurulumların tamamlanmasını bekleyebiliyor.

### **Faz 3: Gelişmiş Analiz ve Yardımcı Modüllerin Tamamlanması**

- **Madde 5: `DependencyOptimizer` Entegrasyonu**
    - **Durum:** <span style="color:orange">**DEVAM EDİYOR**</span>
    - **Açıklama:** Kurulum sırasını optimize edecek, çakışmaları önleyecek ve gereksiz bağımlılıkları tespit edecek bir sınıf. Bu, `learned_dependencies.json` dosyasını kullanarak daha akıllı kararlar verebilir.
- **Madde 6: `CodeAnalyzer` ve Statik Analiz**
    - **Durum:** <span style="color:red">**BAŞLANMADI**</span>
    - **Açıklama:** Kod kalitesini artırmak ve potansiyel hataları önceden tespit etmek için statik analiz araçlarının entegrasyonu.

### **Faz 4: Son Dokunuşlar ve Sağlamlaştırma**

- **Madde 7: Test ve Dokümantasyon**
    - **Durum:** <span style="color:grey">BEKLEMEDE</span>
    - **Açıklama:** Geliştirilen tüm bileşenler için birim (unit) ve entegrasyon testleri yazılacak. Kod içi dokümantasyon (docstrings) ve genel `README.md` dosyası güncellenecektir. Bu süreç, diğer fazlarla paralel olarak ilerleyecektir.

---

## **Bölüm 4: Stratejik Geliştirme ve Nihai Yetenekler**

**Amaç:** `AutoImporter`'ı sadece bir paket kurucudan, kendi kendine yeten, proaktif ve akıllı bir ortam yöneticisine dönüştürmek. Bu bölüm, sistemin en temel güvenilirlik sorunlarını ele alan ve `aiozel.md`'deki ileri seviye konseptlere zemin hazırlayan stratejik hedefleri içerir.

### **Faz 5: Otonom Yetenekler**

- **Madde 8: Otonom Derleyici Yönetimi**
    - **Durum:** <span style="color:red">**BAŞLANMADI**</span>
    - **Hedef:** C/C++ derleyicisi gerektiren paketlerdeki kurulum hatalarını reaktif olarak çözmek yerine, sorunu proaktif olarak ortadan kaldırmak.
    - **Adımlar:**
        1.  **Proaktif Tespit:** `AutoImporter` başlangıcında, sistemde `cl.exe` (MSVC) veya `gcc` gibi bilinen derleyicilerin varlığını ve PATH'de olup olmadığını kontrol et.
        2.  **Otonom Kurulum:** Eğer derleyiciler eksikse, kullanıcıya sormadan, arka planda ve sessiz modda Visual Studio Build Tools gibi gerekli araç setlerini indirip kuracak bir mekanizma geliştir.
        3.  **Ortam Entegrasyonu:** Kurulum sonrası, `pip`'in derleyiciyi görebilmesi için gerekli ortam değişkenlerini (`PATH`, `INCLUDE`, `LIB`) dinamik olarak ayarla veya `pip` komutuna bu yolları parametre olarak geçir.

- **Madde 9: Tam Kendi Kendine Yeterlilik (Bootstrap Mekanizması)**
    - **Durum:** <span style="color:red">**BAŞLANMADI**</span>
    - **Hedef:** `AutoImporter`'ın, hiçbir temel bağımlılığı (`psutil`, `packaging` vb.) kurulu olmasa bile, herhangi bir temiz Python ortamında kendini çalıştırabilir hale getirmesini sağlamak.
    - **Adımlar:**
        1.  **Çekirdek Liste Tanımlama:** Kod içinde, `AutoImporter`'ın çalışması için mutlak zorunlu olan paketlerin bir listesini (`CORE_DEPENDENCIES`) tanımla.
        2.  **Önyükleme (Bootstrap) Fonksiyonu:** `AutoImporter`'ın `__init__` metodunun en başında çalışacak özel bir `_bootstrap()` fonksiyonu oluştur.
        3.  **Sessiz Kurulum:** Bu fonksiyon, listedeki her paketin varlığını `importlib.util.find_spec` ile kontrol etmeli. Eksikse, loglama veya diğer karmaşık özellikleri kullanmadan, basit bir `subprocess.run` çağrısı ile paketi kurmalı.
        4.  **Güvenli Başlatma:** Ancak `_bootstrap()` başarıyla tamamlandıktan sonra `AutoImporter`'ın geri kalan (`Logger`, `DependencyOptimizer` vb.) bileşenleri yüklenmelidir.

### **Faz 6: Derinlemesine Analiz, Karşılaştırma ve İleri Seviye Entegrasyon**

- **Madde 10: Mevcut Sistemin (`auto_importer_fixed.py`) Detaylı Dokümantasyonu**
    - **Durum:** <span style="color:green">**TAMAMLANDI**</span>
    - **Görev:** `auto_importer_fixed.py` içerisindeki her sınıf, her metot, her mekanizma ve her iş akışını (örneğin, bir `ImportError`'un nasıl yakalandığı, analiz edildiği, kuyruğa eklendiği, denendiği ve çözüldüğü) hiçbir varsayıma yer bırakmayacak şekilde, açık ve net bir dille belgelemek. Bu, sonraki karşılaştırmalar için temel "doğruluk kaynağı" olacaktır.
    - **Çıktı:** Bu maddenin tüm adımları doğrudan `aiplan.md` içine yazılmıştır. (Aşağıdaki alt bölümlerde detaylandırılmıştır).

    #### **1. Genel Mimarî ve Çalışma Prensibi**
    `AutoImporter`, bir Python ortamında karşılaşılan `ImportError` ve `ModuleNotFoundError` hatalarını proaktif ve reaktif olarak çözmek üzere tasarlanmış çok bileşenli bir sistemdir. Temel amacı, eksik olan Python paketlerini kullanıcı müdahalesine gerek kalmadan, akıllı bir şekilde tespit edip kurmaktır. Sistem, aşağıdaki temel prensipler üzerine kurulmuştur:
    - **Singleton Tasarım Deseni:** Tüm sistem, `AutoImporter` ana sınıfının tek bir örneği (instance) üzerinden yönetilir. Bu, kaynakların (log dosyaları, thread havuzları, konfigürasyonlar) merkezi ve tutarlı bir şekilde kontrol edilmesini sağlar.
    - **Çok Katmanlı Yönetim:** Sistem, her biri belirli bir göreve odaklanmış çok sayıda yönetici (manager) ve yardımcı (helper) sınıftan oluşur. Örneğin, `DependencyRegistry` veritabanını yönetirken, `ResourceMonitor` sistem kaynaklarını izler, `DependencyOptimizer` ise kurulum sırasını optimize eder. Bu, "Separation of Concerns" (Sorumlulukların Ayrılması) ilkesine uygun, modüler ve bakımı kolay bir yapı sağlar.
    - **Asenkron ve Paralel Çalışma:** Sistem, I/O-bağlantılı (örn. paket indirme) ve CPU-bağlantılı (örn. bilimsel hesaplama) görevler için sırasıyla `ThreadPoolExecutor` ve `ProcessPoolExecutor` kullanarak ana uygulama iş parçacığını (main thread) kilitlemez. Bu, özellikle PDS-X gibi interaktif bir yorumlayıcı içinde çalışırken sistemin donmasını engeller.
    - **Reaktif ve Proaktif Modlar:**
        - **Reaktif:** `RealTimeLogMonitor` aracılığıyla, çalışan uygulamaların ürettiği logları (özellikle `stderr`) gerçek zamanlı olarak izler. Bir `ImportError` tespit ettiğinde, eksik modülün kurulumunu otomatik olarak tetikler.
        - **Proaktif:** `CodeAnalyzer` aracılığıyla, bir Python kaynak dosyasını statik olarak analiz ederek (kodu çalıştırmadan) içindeki tüm `import` ifadelerini bulur ve bu bağımlılıkların kurulu olup olmadığını kontrol ederek eksik olanları önceden kurabilir.

    #### **2. Temel Yapılandırma ve Yardımcı Sınıflar**
    Bu bileşenler, sistemin temel iskeletini ve çalışma ortamını oluşturur.
    - **`PdsXException` Sınıfları:** Standart Python hataları (`ImportError`, `FileNotFoundError` vb.) yerine, PDS-X ortamına özgü, daha fazla bağlam (context) ve özel bir hata kodu (`code`) içeren özel hata sınıfları tanımlanmıştır. Bu, hata yönetimini standartlaştırır ve hata ayıklamayı kolaylaştırır.
    - **Lazy Loading (Tembel Yükleme) Mekanizması:** `psutil`, `numpy`, `scikit-learn` gibi ağır ve her zaman gerekmeyebilecek kütüphaneler, program başlar başlamaz `import` edilmez. Bunun yerine, `get_psutil()` gibi özel fonksiyonlar aracılığıyla, sadece ilgili kütüphaneye ilk kez ihtiyaç duyulduğunda yüklenirler. Eğer kütüphane sistemde yoksa, program çökmez; bunun yerine bir uyarı mesajı basılır ve ilgili özellik devre dışı bırakılır. Bu, `AutoImporter`'ın başlangıç süresini önemli ölçüde kısaltır.
    - **`ModeManager` (Çalışma Modu Yöneticisi):** Sistemin ne kadar "konuşkan" olacağını kontrol eder. `NORMAL` modda tüm çıktılar gösterilirken, `SILENT` veya `TOTAL_SILENT` modlarda kullanıcı arayüzü çıktısı bastırılır. Bu, sistemin farklı otomasyon seviyelerinde veya kullanıcı tercihlerine göre çalışmasına olanak tanır.
    - **`AdvancedLogger` ve `Tee` Sınıfı:** Bu ikili, sistemin en kritik bileşenlerindendir.
        - `AdvancedLogger`, singleton yapıda bir loglama sistemidir. Olayları birden fazla hedefe aynı anda yazabilir: 
            1.  **Standart Log Dosyası (`pdsx_auto_importer.log`):** İnsan tarafından okunabilir, detaylı çalışma günlüğü.
            2.  **JSONL Dosyası (`pdsx_terminal.jsonl`):** Yapısal (structured) loglama için. Her log kaydı bir JSON nesnesidir. Bu, logların otomatik sistemler tarafından kolayca işlenmesini ve analiz edilmesini sağlar.
            3.  **Konsol (stdout):** Kullanıcıya anlık bilgi vermek için.
        - `Tee` sınıfı, `sys.stdout` ve `sys.stderr` akışlarını ele geçirir. Bu sayede, `print()` ile yazılan veya bir alt işlemden (subprocess) gelen herhangi bir çıktı, hem orijinal hedefine (konsol) hem de bir log dosyasına (`pdsX_terminal.log`) eş zamanlı olarak yazılır. Bu, sistemde olan *her şeyin* kaydının tutulmasını garanti altına alır ve `RealTimeLogMonitor`'un çalışması için temel oluşturur.

    #### **3. Çekirdek Veri ve Kurulum Yönetimi Bileşenleri**
    Bu sınıflar, AutoImporter'ın durumunu yönetir, çevresiyle etkileşime girer ve gerçekleştirdiği eylemlerin sonuçlarını anlamlandırır.
    - **`EnvManager` (Ortam Yöneticisi):**
        - **Amaç:** Sistemin içinde çalıştığı Python ortamının temel yollarını (`python.exe` ve `pip.exe`) tespit etmek ve merkezi bir yerden erişilebilir kılmak.
        - **Mekanizma:** Çalıştırıldığı anda `sys.executable` global değişkenini kullanarak Python yorumlayıcısının tam yolunu bulur. Ardından, bu yoldan yola çıkarak `pip` yürütülebilir dosyasının standart konumda (`Scripts` klasörü altında) olduğunu varsayar. Eğer bulamazsa, bunu loglar. 
        - **Önemi:** Bu sınıf olmadan, sistem hangi `pip` komutunu çalıştıracağını bilemez. Tüm alt işlem (subprocess) komutları, bu sınıfın sağladığı yolları kullanarak doğru Python ortamını hedef aldığından emin olur.
    - **`DependencyRegistry` (Bağımlılık Kayıt Defteri):**
        - **Amaç:** Sistemin uzun süreli hafızasıdır. `pip` aracılığıyla başarıyla kurulan her paketin adını, sürümünü, bağımlılıklarını ve kurulum tarihini kalıcı olarak saklar.
        - **Mekanizma:** Tüm verileri `dependencies.json` adlı bir dosyada saklar. Sistem başladığında bu dosyayı okur, çalışma sırasında yeni bir paket kurulduğunda günceller ve dosyaya geri yazar. Bu, sistemin yeniden başlatılsa bile geçmiş kurulumları hatırlamasını sağlar.
        - **Önemi:** Bu kayıt defteri, gereksiz kurulumları önler (bir paket zaten kayıtlıysa tekrar kurulmaz) ve `DependencyOptimizer` gibi daha gelişmiş bileşenler için hayati bir veri kaynağıdır. Hangi paketin neye bağlı olduğunu bilmek, gelecekteki kurulum kararlarını daha akıllı hale getirir.
    - **`PipOutputAnalyzer` (Pip Çıktı Analizcisi):**
        - **Amaç:** `pip install` komutunun çalıştırıldıktan sonra ürettiği metin tabanlı çıktıyı (stdout) analiz ederek yapısal bilgilere dönüştürmek.
        - **Mekanizma:** Önceden derlenmiş düzenli ifadeler (regex) kullanarak `pip` çıktısını satır satır tarar. "Collecting..." ile başlayan satırlardan paketin bağımlılıklarını, "Successfully installed..." ile biten satırdan ise kurulan ana paketin adını ve tam sürüm numarasını çıkarır.
        - **Önemi:** `pip` komutu, programatik olarak kullanılabilecek yapısal bir çıktı (JSON gibi) vermez. Bu sınıf, insan tarafından okunması için tasarlanmış bu metni, makine tarafından okunabilir bir sözlüğe (`Dict`) çevirir. Bu sözlük, daha sonra `DependencyRegistry`'yi güncellemek için kullanılır. Bu olmadan, sistem neyi, hangi sürümde kurduğunu ve bu paketin neleri beraberinde getirdiğini bilemezdi.
    - **`ModuleSummaryGenerator` (Modül Özet Üreticisi):**
        - **Amaç:** Bir veya daha fazla kurulum işlemi tamamlandıktan sonra kullanıcıya okunabilir, temiz bir özet rapor sunmak.
        - **Mekanizma:** Tamamlanan kurulumların sonuçlarını (başarı/hata durumu, süre vb.) içeren bir liste alır. Bu listeyi işleyerek başarılı ve başarısız kurulumları gruplandırır, toplam harcanan zamanı hesaplar ve tüm bu bilgileri biçimlendirilmiş bir metin bloğu olarak log sistemine basar.
        - **Önemi:** Otomatik bir sistemde şeffaflık çok önemlidir. Bu sınıf, kullanıcıya sistemin ne yaptığını, ne kadar sürdüğünü ve başarılı olup olmadığını net bir şekilde bildirerek sistemin "kara kutu" gibi davranmasını engeller.

    #### **4. Gelişmiş Analiz ve Optimizasyon Bileşenleri**
    Bu katman, AutoImporter'ı basit bir "kurulum otomatından" akıllı bir "paket yöneticisine" dönüştüren beyin takımını içerir. Bu bileşenler, proaktif olarak bağımlılıkları tespit eder, en verimli kurulum yolunu planlar ve performansı artırmak için önbellekleme (caching) yapar.
    - **`CodeAnalyzer` (Kod Analizcisi):**
        - **Amaç:** Bir Python betiğini çalıştırmadan, kaynak kodunu statik olarak analiz ederek hangi modüllere ihtiyaç duyduğunu tespit etmek.
        - **Mekanizma:** Bu sınıf, önceki basit regex tabanlı `ModuleAnalyzer`'ın yerini alan çok daha güçlü bir yaklaşım kullanır. Python'un dahili `ast` (Abstract Syntax Tree - Soyut Sözdizimi Ağacı) modülünü kullanır. Bir kaynak dosyanın içeriğini okur, bu metni Python'un anladığı bir ağaç yapısına dönüştürür. Daha sonra bu ağacı gezerek `import ...` ve `from ... import ...` gibi tüm import ifadelerini bulur. Bu yöntem, kodun yorum satırlarında veya metin dizeleri içinde geçen modül adlarını yanlışlıkla tespit etme riskini ortadan kaldırır ve sadece gerçek kod olan importları bulur.
        - **Önemi:** Bu, sistemin **proaktif** yeteneğidir. Kullanıcı bir betiği çalıştırmadan önce, AutoImporter bu analizciyi kullanarak betiğin tüm bağımlılıklarını bulabilir ve eksik olanları önceden kurarak `ImportError` hatasının hiç oluşmamasını sağlayabilir.
    - **`DependencyOptimizer` (Bağımlılık Optimize Edici):**
        - **Amaç:** Kurulacak bir paket listesi verildiğinde, bu paketlerin ve onların alt bağımlılıklarının hangi sırayla kurulması gerektiğini belirlemek. Yanlış sıra, kurulum hatalarına yol açabilir.
        - **Mekanizma:** İki ana veri kaynağını birleştirerek bir bağımlılık grafiği oluşturur: `DependencyRegistry`'den gelen (gerçekte kurulmuş) ve `learned_dependencies.json` dosyasından gelen (geçmiş tecrübelerden öğrenilmiş) bağımlılıklar. Bu grafiği oluşturduktan sonra, meşhur bir bilgisayar bilimi algoritması olan **Topolojik Sıralama (Topological Sort)** kullanır. Bu algoritma, grafiği analiz ederek "önce A kurulmalı, çünkü B, A'ya bağlı" gibi ilişkileri çözer ve en mantıklı kurulum sırasını içeren bir liste döndürür. Ayrıca, bağımlılıklar arasında bir döngü (A, B'ye bağlı; B de A'ya bağlı gibi) tespit ederse uyarı verir.
        - **Önemi:** Özellikle karmaşık bağımlılık ağlarına sahip kütüphanelerde (örneğin, veri bilimi paketleri) doğru kurulum sırası hayati önem taşır. Bu sınıf, `pip`'in kendi çözümleme mekanizmasına ek bir zeka katmanı ekleyerek kurulumların daha sağlam ve hatasız olmasını sağlar.
    - **`WheelCacheManager` (Wheel Önbellek Yöneticisi):**
        - **Amaç:** Python paketlerinin derlenmiş versiyonları olan "wheel" (`.whl`) dosyalarını yerel bir önbellekte (cache) saklayarak, aynı paketin tekrar tekrar indirilip derlenmesini önlemek ve kurulum sürelerini dramatik şekilde azaltmak.
        - **Mekanizma:** `.pdsx_cache/wheels` dizinini kullanır. Bir paket kurulmadan önce, bu yönetici önce bu dizinde pakete uygun bir wheel dosyasının olup olmadığını kontrol eder. Varsa, `pip`'e paketi internetten indirmek yerine bu yerel dosyayı kullanmasını söyler. Eğer dosya yoksa, `pip wheel` komutunu kullanarak paketin sadece `.whl` dosyasını (bağımlılıkları olmadan) indirir ve bu dizine kaydeder. Böylece bir sonraki kurulumda hazır olur.
        - **Önemi:** Bu, sistemin verimliliğini en çok artıran bileşenlerden biridir. Özellikle büyük paketler veya yavaş internet bağlantıları söz konusu olduğunda, kurulum sürelerini saniyelere indirebilir. Ayrıca, çevrimdışı (offline) çalışma yeteneği için de bir temel oluşturur.
    - **`ConflictManager` (Çakışma Yöneticisi):**
        - **Amaç:** Kurulmak istenen bir paketin, sistemde zaten kurulu olan başka bir paketle sürüm çakışması yaratma potansiyelini (ilkel bir seviyede de olsa) kontrol etmek.
        - **Mekanizma:** `packaging` kütüphanesini (lazy loading ile yüklenir) kullanarak kurulacak paket isteğini (`requests==2.25.1` gibi) analiz eder. Daha sonra `DependencyRegistry`'deki kayıtlara bakarak aynı isimde bir paketin zaten kurulu olup olmadığını kontrol eder. Eğer kuruluysa, bir uyarı logu basar. 
        - **Not:** Mevcut implementasyon oldukça basittir ve `pip`'in kendi gelişmiş çakışma çözümleme mekanizmasının yerini almaz. Daha çok bir erken uyarı sistemi olarak görev yapar.
        - **Önemi:** Projenin gelecekte daha sofistike bir bağımlılık çözümleyiciye sahip olması için bir temel ve yer tutucu görevi görür. Potansiyel sorunları erkenden fark etme potansiyeli taşır.

    #### **5. Yürütme, İzleme ve Güvenlik Bileşenleri**
    Bu sınıflar, AutoImporter'ın canlı operasyonlarını yürütür, sistemin durumunu ve dış olayları izler ve beklenmedik durumlara karşı koruma mekanizmaları sağlar.
    - **`TerminalLogAnalyzer` ve `RealTimeLogMonitor` (Reaktif Mekanizma):**
        - **Amaç:** Bu ikili, sistemin **reaktif** gücünü oluşturur. PDS-X yorumlayıcısı veya başka bir betik çalışırken oluşan `ImportError` hatalarını anında yakalayıp çözmekle görevlidirler.
        - **Mekanizma:**
            1.  `Tee` sınıfı sayesinde, konsola yazılan her hata (`stderr`) aynı zamanda `pdsX_terminal.log` dosyasına da yazılır.
            2.  `RealTimeLogMonitor`, ayrı bir arka plan iş parçacığında (background thread) bu log dosyasını sürekli olarak "izler" (tail komutu gibi).
            3.  Dosyaya yeni bir satır eklendiğinde, bu satırı `TerminalLogAnalyzer`'a gönderir.
            4.  `TerminalLogAnalyzer`, gelen satırın içinde `ImportError: No module named '...'` gibi bir ifade olup olmadığını çok spesifik bir regex ile kontrol eder.
            5.  Eğer bir eşleşme bulursa, eksik modülün adını (`'...'` kısmı) çıkarır ve ana `AutoImporter` örneğinin `auto_install_package` metodunu bu modül adıyla çağırarak kurulum sürecini tetikler.
        - **Önemi:** Bu mekanizma, AutoImporter'ın PDS-X ile kusursuz bir şekilde entegre olmasını sağlar. Kullanıcı, kodunu çalıştırırken bir modülün eksik olduğunu fark ettiğinde, AutoImporter çoktan arka planda o modülü kurmaya başlamış olur. Bu, kesintisiz bir geliştirme deneyimi sunar.
    - **`ResourceMonitor` (Kaynak İzleyici):**
        - **Amaç:** Kurulum işlemleri sırasında sistemin CPU ve bellek kullanımını izleyerek, sistemin aşırı yüklenip yüklenmediğini kontrol etmek ve bu verileri loglamak.
        - **Mekanizma:** `psutil` kütüphanesini (lazy loading ile) kullanarak, periyodik olarak (örn. her 5 saniyede bir) CPU ve sanal bellek kullanım yüzdelerini alır. Bu işlemi, ana programı engellememek için kendi arka plan iş parçacığında (thread) yapar. Topladığı verileri `debug` seviyesinde loglar.
        - **Önemi:** Özellikle kaynakları kısıtlı sistemlerde veya çok büyük paketlerin kurulumu sırasında, sistemin performansını analiz etmek ve olası darboğazları tespit etmek için değerli veriler sağlar. Bir nevi sistemin "sağlık monitörü"dür.
    - **`AsyncDownloadManager` ve `ScientificUtils` (Görev Yöneticileri):**
        - **Amaç:** Farklı nitelikteki görevleri, en verimli şekilde çalıştırmak üzere özel işçi havuzlarına (worker pools) yönlendirmek.
        - **Mekanizma:**
            - `AsyncDownloadManager`: `ThreadPoolExecutor` kullanır. Bu havuz, ağdan dosya indirme gibi G/Ç'ye bağlı (I/O-bound) işlemler için idealdir. Birçok indirme işlemini aynı anda başlatabilir, çünkü işlemcinin büyük bir kısmı dosyaların inmesini beklerken boşta kalır. `WheelCacheManager`'ın `download_wheel` görevi bu havuza gönderilir.
            - `ScientificUtils`: `ProcessPoolExecutor` kullanır. Bu havuz, karmaşık matematiksel hesaplamalar veya veri analizi gibi işlemciyi yoğun kullanan (CPU-bound) görevler için tasarlanmıştır. Bu görevleri ayrı proseslerde (processes) çalıştırarak Python'un Global Interpreter Lock (GIL) kısıtlamasını aşar ve çok çekirdekli işlemcilerden tam olarak faydalanır.
        - **Önemi:** Bu ayrım, sistemin kaynaklarını en verimli şekilde kullanmasını sağlar. Doğru görevi doğru havuza yönlendirerek, uygulamanın genel performansını ve yanıt verme yeteneğini en üst düzeye çıkarır.
    - **`KillSwitch` ve `GracefulShutdownManager` (Güvenlik ve Sonlandırma):**
        - **Amaç:** Sistemin hem acil durumlarda anında durdurulabilmesini hem de normal kapatma senaryolarında tüm kaynaklarını temiz ve düzenli bir şekilde serbest bırakmasını sağlamak.
        - **Mekanizma:**
            - `KillSwitch`: Basit ama etkili bir güvenlik önlemidir. `is_active` adında bir bayrak (flag) tutar. Eğer bu bayrak `True` olarak ayarlanırsa, kurulum döngüsü gibi kritik işlemlerin başındaki `check()` metodu bir `InterruptedError` fırlatarak tüm operasyonları anında keser.
            - `GracefulShutdownManager`: Sistemin "temizlik ekibidir". Başlatılan her arka plan servisi (`ResourceMonitor`, `LogMonitor` vb.) kendi `stop` fonksiyonunu bu yöneticiye kaydettirir. Programın sonlandırılması gerektiğinde, `shutdown()` metodu çağrılır ve kaydedilen tüm `stop` fonksiyonları, kaydedilme sıralarının tersine göre (LIFO - Son Giren İlk Çıkar) tek tek çalıştırılır. Bu, önce dış servislerin durdurulmasını, en son ise log dosyaları gibi temel kaynakların kapatılmasını garanti eder.
        - **Önemi:** Bu ikili, sistemin sağlamlığını (robustness) ve güvenilirliğini artırır. `KillSwitch`, kontrolden çıkan bir durumu engellerken, `GracefulShutdownManager` ise sistemin arkasında açık dosyalar, öksüz iş parçacıkları veya bozuk kaynaklar bırakmadan temiz bir şekilde kapanmasını sağlar.

    #### **6. Ana İş Akışları (Core Workflows)**
    Bu bölümde, yukarıda tanımlanan bileşenlerin belirli senaryolarda bir orkestra gibi nasıl birlikte çalıştığı adım adım açıklanmaktadır.
    **Senaryo 1: Reaktif Kurulum (Bir `ImportError` Anında)**
    Bu, sistemin en yaygın kullanım senaryosudur. Kullanıcı, PDS-X'te bir kod çalıştırır ve bu kod, kurulu olmayan bir modülü import etmeye çalışır.
    1.  **Hata Oluşur:** Python yorumlayıcısı `ImportError: No module named 'requests'` gibi bir hata verir. Bu hata mesajı `stderr` akışına yazdırılır.
    2.  **`Tee` Devreye Girer:** `Tee` sınıfı, `stderr`'e giden bu mesajı yakalar. Mesajı hem orijinal hedefine (kullanıcının konsolu) gönderir hem de anında `pdsX_terminal.log` dosyasına yazar.
    3.  **`RealTimeLogMonitor` Uyanır:** Ayrı bir iş parçacığında çalışan bu izleyici, `pdsX_terminal.log` dosyasındaki değişikliği (yeni eklenen hata satırını) fark eder.
    4.  **`TerminalLogAnalyzer` Analiz Eder:** İzleyici, yeni satırı analizciye verir. Analizci, regex kullanarak satırın bir "No module named" hatası içerip içermediğini kontrol eder. Eşleşme başarılı olur ve eksik modülün adını (`'requests'`) çıkarır.
    5.  **Kurulum Tetiklenir:** Analizci, bulduğu modül adıyla ana `AutoImporter` nesnesinin `auto_install_package('requests')` metodunu çağırır.
    6.  **Merkezi Süreç Başlar:** Buradan sonra süreç, aşağıda açıklanan "Merkezi Kurulum Süreci"ne devredilir.

    **Senaryo 2: Proaktif Kurulum (Statik Kod Analizi ile)**
    Bu senaryo, bir `ImportError` oluşmadan önce, bir betiğin ihtiyaç duyacağı paketleri önden kurmayı amaçlar.
    1.  **Analiz Başlatılır:** Kullanıcı veya PDS-X, `AutoImporter`'ın `analyze_and_install_from_file('path/to/script.py')` gibi bir metodunu (bu metot varsayımsaldır, mevcut kodda bu işlevsellik `CodeAnalyzer` üzerinden manuel olarak tetiklenir) çağırır.
    2.  **`CodeAnalyzer` Çalışır:** `CodeAnalyzer`, verilen betiğin kaynak kodunu `ast` modülü ile analiz eder ve içindeki tüm import ifadelerinden oluşan bir liste (`['pandas', 'numpy', 'matplotlib']`) çıkarır.
    3.  **Paketler Kontrol Edilir:** Sistem, bu listedeki her bir modül için `check_package_installed()` metodunu kullanarak hangilerinin zaten kurulu olduğunu kontrol eder.
    4.  **Kurulumlar Tetiklenir:** Kurulu olmayan her bir modül için (`'matplotlib'` diyelim), `auto_install_package('matplotlib')` metodu çağrılır.
    5.  **Merkezi Süreç Başlar:** Her bir eksik paket için süreç, "Merkezi Kurulum Süreci"ne devredilir.

    **Senaryo 3: Merkezi Kurulum Süreci (Tüm Senaryoların Kalbi)**
    İster reaktif ister proaktif olarak tetiklensin, tüm kurulum istekleri bu merkezi ve sağlam iş akışından geçer.
    1.  **İstek Alınır (`auto_install_package`):** Metot, kurulacak paket adı (`'requests'`) ile çağrılır. Bir `threading.Lock` ile aynı anda sadece bir iş parçacığının bu bölüme girmesi sağlanır.
    2.  **İsim Çözümlenir ve Kontrol Edilir:** Modül adı, bilinen takma adlar (`aliases`) kullanılarak gerçek paket adına (`bs4` -> `beautifulsoup4`) çevrilir. Paketin daha önce denenip başarısız olup olmadığı (`failed_packages`) veya zaten kurulum kuyruğunda olup olmadığı kontrol edilir. Eğer öyleyse, işlem iptal edilir.
    3.  **Kuyruğa Ekleme:** Paket, `self.installation_queue` (`queue.Queue` nesnesi) adlı güvenli kurulum kuyruğuna eklenir. `installation_complete_event` adlı `threading.Event` nesnesi, yeni bir kurulum başladığını belirtmek için temizlenir (clear).
    4.  **İşleyici (Worker) Başlatma:** Sistem, o anda aktif bir kurulum işleyici iş parçacığı (`_process_installation_queue`) olup olmadığını kontrol eder. Yoksa, kuyruğu işlemeye başlaması için yeni bir tane başlatır.
    5.  **Kuyruk İşlenir (`_process_installation_queue`):**
        a.  İşleyici, kuyruktan bir paket alır.
        b.  **Optimizasyon:** Aldığı paket listesini (`['requests']`) `DependencyOptimizer`'a verir. Optimize edici, bilinen bağımlılıkları da (`['certifi', 'charset-normalizer', 'idna', 'urllib3']`) hesaba katarak ve topolojik sıralama yaparak en verimli kurulum sırasını (`['certifi', 'charset-normalizer', 'idna', 'urllib3', 'requests']`) oluşturur.
        c.  **Kurulum Döngüsü:** İşleyici, bu optimize edilmiş listedeki her bir paket için sırayla şunları yapar:
            i.  **Önbellek Kontrolü:** `WheelCacheManager`'a bu paket için yerel bir `.whl` dosyasının olup olmadığını sorar.
            ii. **`pip` Komutu Hazırlanır:** `pip install` komutu oluşturulur. Eğer önbellekte wheel varsa, komuta paket adı yerine `.whl` dosyasının yolu eklenir. Yoksa, paket adı eklenir.
            iii. **`pip` Çalıştırılır:** `subprocess.run` ile `pip` komutu çalıştırılır ve tüm çıktı (stdout ve stderr) yakalanır.
            iv. **Sonuç Analizi:** `pip`'in çıktısı ve dönüş kodu (`returncode`) `PipOutputAnalyzer`'a verilir. Analizci, kurulan ana paketin adını, sürümünü ve diğer bağımlılıklarını çıkarır.
            v.  **Kayıt Güncelleme:** Eğer kurulum başarılıysa (`returncode == 0`), analizciden gelen yapısal veri kullanılarak `DependencyRegistry`'ye yeni paket bilgileri eklenir ve `dependencies.json` dosyası güncellenir.
            vi. **Hata Yönetimi:** Kurulum başarısız olursa, paket `failed_packages` sözlüğüne eklenir ve bir daha denenmez.
    6.  **İşlem Sonu ve Özet:** Kuyruk boşaldığında, işleyici iş parçacığı sonlanır. `ModuleSummaryGenerator` çağrılarak kullanıcıya başarılı/başarısız kurulumları ve toplam süreyi gösteren bir özet rapor sunulur. Son olarak, `installation_complete_event` tekrar kurulur (set), böylece `wait_for_installs()` gibi bir metodun beklemesini sonlandırır.

- **Madde 11: Referans `AutoImporter` Implementasyonlarının Analizi**
    - **Durum:** <span style="color:orange">**DEVAM EDİYOR**</span>
    - **Görev:** `inceleme.md` dosyasının başında listelenen tüm eski `auto_importer` sürümlerini (`v1793`, `v1795` vb.) tek tek, satır satır incelemek. Her birinin kendine özgü yaklaşımlarını, güçlü ve zayıf yönlerini, denedikleri mekanizmaları (başarılı veya başarısız) tespit etmek.
    - **Çıktı:** Tüm bulgular, her dosya için ayrı bir başlık altında, `referans_analizi.md` adında yeni bir dosyaya detaylı olarak yazılacaktır.

- **Madde 12: Kapsamlı Karşılaştırma Raporu**
    - **Durum:** <span style="color:red">**BAŞLANMADI**</span>
    - **Görev:** Madde 10'da oluşturulan `auto_importer_fixed.py` dokümantasyonu ile Madde 11'de oluşturulan `referans_analizi.md` dosyasındaki bilgileri madde madde karşılaştırmak.
    - **Örnek Karşılaştırma Maddeleri:**
        - "Hata Analizi: `v1795`'teki `pip` çıktı analizi basit bir string kontrolüne dayanırken, `auto_importer_fixed.py`'deki `PipOutputAnalyzer` sınıfı, 15'ten fazla farklı hata senaryosunu regex ile tanıyabilmekte ve her biri için özel çözüm stratejileri üretebilmektedir."
        - "Asenkron Çalışma: Referans kodların hiçbiri gerçek anlamda bir asenkron veya paralel kurulum yeteneğine sahip değilken, `auto_importer_fixed.py`'deki `ThreadPoolExecutor` ve `threading.Event` kullanımı, birden fazla kurulumu eş zamanlı yöneterek ve ana iş parçacığını kilitlemeyerek modern bir çözüm sunmaktadır."
    - **Çıktı:** Bu karşılaştırma raporu, `aiplan.md` içine, bu maddenin altına detaylı olarak yazılacaktır.

- **Madde 13: `aiozel.md` Konseptlerinin Karşılaştırmalı Değerlendirmesi ve Entegrasyon Planı**
    - **Durum:** <span style="color:red">**BAŞLANMADI**</span>
    - **Görev:** `aiozel.md`'de yer alan Genetik Algoritmalar, Kaos Teorisi gibi ileri seviye optimizasyon ve analiz fikirlerini, `auto_importer_fixed.py`'de halihazırda geliştirdiğimiz `DependencyOptimizer` ve `ScientificUtils` gibi yapılarla karşılaştırmak. Hangi yaklaşımın hangi senaryoda daha üstün olduğunu teorik ve pratik olarak analiz etmek.
    - **Çıktı:** Analiz ve entegrasyon planı bu maddenin altına yazılacaktır.
