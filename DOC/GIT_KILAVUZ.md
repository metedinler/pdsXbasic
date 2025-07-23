# PDS-X BASIC v14u Git Kılavuzu

## Başlangıç Kurulumu

Proje için yapılan ilk Git kurulumu:

1. Git deposu oluşturma:
   ```
   git init
   git add *.py dependencies.json
   git commit -m "İlk commit: PDS-X BASIC v14u projesinin başlangıç kodu"
   ```

2. Dal yapısının kurulumu:
   ```
   git branch -M main
   git checkout -b develop
   ```

3. .gitignore dosyası oluşturma:
   ```
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .Python
   *.log
   *.bak
   *.zip
   *.rar
   *.backup.*
   ```

## Dal (Branch) Yapısı

- `main`: Kararlı sürümler
- `develop`: Aktif geliştirme
- `feature/*`: Yeni özellikler
- `hotfix/*`: Acil düzeltmeler

## Commit Mesaj Formatı

```
[TÜR]: Kısa açıklama

- Detaylı açıklama
- Değişiklik nedenleri
- Etkilenen alanlar
```

### Mesaj Türleri

- YENI: Yeni özellikler
- DÜZELT: Hata düzeltmeleri
- BELGE: Belgelendirme
- GELIŞTIR: İyileştirmeler

## Temel İş Akışı

1. Geliştirme için yeni dal oluştur:
   ```
   git checkout develop
   git checkout -b feature/ozellik-adi
   ```

2. Değişiklikleri kaydet:
   ```
   git add .
   git commit -m "[TÜR]: Açıklama"
   ```

3. Değişiklikleri birleştir:
   ```
   git checkout develop
   git merge feature/ozellik-adi
   ```

## Sürüm Yönetimi

1. Sürüm etiketleme:
   ```
   git tag -a v1.0 -m "PDS-X BASIC v14u ilk kararlı sürüm"
   ```

2. Önceki sürüme dönme:
   ```
   git checkout <commit-hash>  # Belirli bir commit'e dönmek için
   git checkout v1.0          # Belirli bir sürüme dönmek için
   ```

3. Yedekleme:
   ```
   git archive --format=zip HEAD -o pdsXuv14.zip  # Projeyi ZIP olarak yedekle
   ```

4. Uzak depo işlemleri:
   ```
   git remote add origin <uzak-depo-url>
   git push -u origin main
   git push --tags  # Etiketleri uzak depoya gönder
   ```

## İyi Uygulamalar

1. Sık sık commit yapın
2. Her commit'in tek bir amacı olsun
3. Açıklayıcı commit mesajları yazın
4. Değişiklikleri düzenli olarak push edin
5. Birleştirmeden önce test edin
