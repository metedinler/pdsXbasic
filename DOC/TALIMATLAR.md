# PDS-X Auto_Importer Analiz Şablonu ve Yazım Kuralları (GÜNCEL)

## GENEL KURALLAR
- Her analiz, aşağıdaki başlık ve sıralama ile yapılacaktır:
  1. GENEL FELSEFE VE AMAÇ
  2. MİMARİ YAPI
  3. SINIFLAR, FONKSİYONLAR VE METOTLAR
     - Sınıfların, fonksiyonların ve metotların detaylı listesi ve açıklaması
  4. SABİTLER VE VERİ YAPILARI
  5. LAZY/EAGER IMPORT MEKANİZMALARI
  6. GİRİŞ/ÇIKIŞ (I/O) OPERASYONLARI
  7. KOMUT SATIRI ARAYÜZÜ (CLI)
  8. İŞLETİLEBİLİRLİK DURUMU
  9. EVRİMSEL GELİŞİM
  10. TEKNİK MİMARİDE ÖNE ÇIKAN 3 YAPI (varsa)
- Her başlık altında, ilgili sınıf, fonksiyon, metot, değişken, sabit ve teknik terimler renkli ve etiketli olarak gösterilecektir.
- İngilizce teknik terimler <span style="color:#ffe082;">açık sarı</span> ile, yanına Türkçe açıklama ve amacı eklenerek yazılacaktır.
- Sınıf isimleri: <span style="color:#1976d2;font-weight:bold;">mavi</span>, fonksiyon isimleri: <span style="color:#388e3c;font-weight:bold;">yeşil</span>, metot isimleri: <span style="color:#f57c00;font-weight:bold;">turuncu</span>, değişken isimleri: <span style="color:#8e24aa;font-weight:bold;">mor</span>, sabitler: <span style="color:#ff5252;font-weight:bold;">açık kırmızı</span> ile gösterilecektir.
- Teknik terimler, İngilizce olarak <span style="color:#ffe082;">açık sarı</span> renkte, yanında Türkçe açıklama ve amacı ile birlikte yazılacaktır.
- Kod bloklarıyla dosya doldurulmayacak, sadece teknik ve açıklayıcı analiz yapılacaktır.
- Her analiz, checklist sırasına göre ve numaralandırılmış şekilde ilerleyecektir.
- Hiçbir açıklama veya analiz silinmeyecek, yeni bulgular eski açıklamaların altına eklenecektir.
- Kullanıcıdan gelen yeni uyarı ve örnekler sürekli dikkate alınacaktır.

## ŞABLON (Her Dosya İçin Kullanılacak)

> **Renkli Etiket Açıklaması:**
> - <span style="color:#1976d2;font-weight:bold;">[Sınıf]</span> Mavi
> - <span style="color:#388e3c;font-weight:bold;">[Fonksiyon]</span> Yeşil
> - <span style="color:#f57c00;font-weight:bold;">[Metot]</span> Turuncu
> - <span style="color:#8e24aa;font-weight:bold;">[Değişken]</span> Mor
> - <span style="color:#ff5252;font-weight:bold;">[Sabit]</span> Açık Kırmızı
> - <span style="color:#ffe082;">Teknik terimler</span> Açık Sarı

### 1. GENEL FELSEFE VE AMAÇ
Kısa ve öz şekilde, programın genel amacı ve felsefesi, teknik terimler renkli ve açıklamalı olarak.

### 2. MİMARİ YAPI
Modülerlik, katmanlar, bağımlılıklar, alt sistemler ve mimari özet.

### 3. SINIFLAR, FONKSİYONLAR VE METOTLAR
- Sınıflar, fonksiyonlar ve metotlar renkli ve etiketli olarak listelenir.
- Her biri için sistemdeki rolü, amacı ve mimarideki önemi açıklanır.

### 4. SABİTLER VE VERİ YAPILARI
- Tüm sabitler ve ana veri yapıları, görevleri ve operasyonel kullanımları ile birlikte açıklanır.

### 5. LAZY/EAGER IMPORT MEKANİZMALARI
- Lazy ve eager import örnekleri, hangi fonksiyonlarda/sınıflarda nasıl uygulandığı.

### 6. GİRİŞ/ÇIKIŞ (I/O) OPERASYONLARI
- Dosya okuma/yazma, loglama, terminal çıktısı, argüman kaydı vb. işlemler ve ilgili sınıf/fonksiyon rolleri.

### 7. KOMUT SATIRI ARAYÜZÜ (CLI)
- CLI desteği, kullanılan argümanlar, ilgili fonksiyonlar ve akış.

### 8. İŞLETİLEBİLİRLİK DURUMU
- Platform uyumluluğu, hata yönetimi, loglama, gerçek zamanlı izleme vb.

### 9. EVRİMSEL GELİŞİM
- Önceki sürümlerle ilişkisi, yenilikler, evrimsel değişiklikler.

### 10. TEKNİK MİMARİDE ÖNE ÇIKAN 3 YAPI (varsa)
- Dosyaya özgü öne çıkan teknik mimari yapılar, renkli ve açıklamalı.

---

Bu şablon ve kurallar, bundan sonraki tüm analizlerde birebir uygulanacaktır. Her yeni analizde, eski açıklamalar silinmeyecek, yeni bulgular ilgili başlık altına eklenerek ilerleme sağlanacaktır.
