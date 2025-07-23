# PDS-X Akıllı Modül Yükleyici Özellikleri

Bu belge, PDS-X Akıllı Modül Yükleyici programının (auto_importer_fixed.py) temel özelliklerini programlama bilmeyen kullanıcılar için açıklamaktadır.

---

## PDS-X Akıllı Modül Yükleyici Nedir?

Bu program, PDS-X sisteminin veya diğer Python projelerinizin çalışması için gerekli olan ek parçaları (modülleri veya kütüphaneleri) otomatik olarak bulup kuran akıllı bir yardımcıdır. Normalde bu parçaları bilgisayarınıza kendiniz yüklemeniz gerekirken, bu program bu işi sizin yerinize yapar.

---

## Temel Özellikler

1.  **Otomatik Kurulum:**
    *   Bir program çalışırken eksik bir parça (modül) olduğunu fark ederse, bu program otomatik olarak devreye girer ve o parçayı internetten bulup bilgisayarınıza yükler. Böylece programınızın çalışması için gerekenleri manuel olarak aramak ve kurmak zorunda kalmazsınız.

2.  **Bağımlılık Yönetimi:**
    *   Her parçanın (modülün) çalışması için başka parçalara ihtiyacı olabilir. Bu program, hangi parçanın hangi parçaya bağlı olduğunu bilir ve kurulumları doğru sırada yapar. Böylece parçalar arasındaki uyumsuzluklardan kaynaklanan hataları önler.

3.  **Akıllı Kurulum Sırası (SmartInstallManager):**
    *   Kurulacak birçok parça olduğunda, bilgisayarınızın o anki durumuna (ne kadar meşgul olduğuna) bakarak en hızlı ve sorunsuz kurulum sırasını belirler. Bazı parçaları aynı anda (eş zamanlı) kurarak toplam kurulum süresini kısaltabilir.

4.  **Hata Yakalama ve Düzeltme:**
    *   Kurulum sırasında bir hata oluşursa (örneğin, internet bağlantısı sorunu, uyumsuz versiyonlar), program bu hatayı anlamaya çalışır ve mümkünse otomatik olarak düzeltme yolları dener. Hata detaylarını size bildirir.

5.  **Loglama ve Raporlama:**
    *   Programın yaptığı her şeyi (hangi parçayı kurmaya çalıştığı, başarılı olup olmadığı, hatalar vb.) kaydeder. Bu kayıtlara bakarak ne olduğunu anlayabilir ve sorunları teşhis edebilirsiniz. Kurulum tamamlandığında size bir özet sunar.

6.  **Önbellekleme (Caching):**
    *   İnternetten indirdiği parçaları bilgisayarınızda özel bir alanda saklar. Aynı parçaya tekrar ihtiyaç duyulduğunda, internetten yeniden indirmek yerine bu sakladığı kopyayı kullanır. Bu, kurulumları çok daha hızlı hale getirir.

7.  **Çakışma Tespiti ve Geri Alma:**
    *   Farklı parçaların birbiriyle uyumsuz versiyonlarını tespit etmeye çalışır. Eğer bir çakışma olursa, programın çalışmasını sağlamak için önceki uyumlu versiyona geri dönme gibi çözümler deneyebilir.

8.  **Gerçek Zamanlı İzleme:**
    *   Program çalışırken terminalde (komut ekranı) çıkan yazıları sürekli takip eder. Eğer bir hata mesajı (özellikle eksik modül hatası) görürse, hemen o modülü kurmak için harekete geçer.

9.  **Komut Satırı Kullanımı:**
    *   Programı çalıştırırken bazı özel komutlar (argümanlar) kullanarak ne yapması gerektiğini söyleyebilirsiniz. Örneğin, belirli bir parçayı kurmasını veya bir listedeki tüm parçaları yüklemesini isteyebilirsiniz.

---

## Kısacası

PDS-X Akıllı Modül Yükleyici, Python projelerinizin ihtiyaç duyduğu ek parçaları sizin için otomatik olarak yöneten, sorunları tespit edip çözmeye çalışan ve kurulum sürecini kolaylaştıran akıllı bir araçtır. Amacı, teknik detaylarla uğraşmadan projelerinizi daha rahat çalıştırmanızı sağlamaktır.
