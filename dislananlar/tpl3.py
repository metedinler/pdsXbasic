import sys
import os
import re
import math
import random
import time
import datetime
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, Canvas
import winsound
import struct
import sqlite3
from io import StringIO
from collections import defaultdict

class TPLInterpreter:
    def __init__(self):
        # Temel veri yapıları
        self.degiskenler = {}
        self.fonksiyonlar = {}
        self.rutinler = {}
        self.siniflar = {}
        self.tipler = {}
        self.etiketler = {}
        self.hata_yonetimi = None
        self.dosyalar = {}
        self.kutuphaneler = {}
        self.sabitler = {}
        
        # REPL özellikleri
        self.repl_modu = False
        self.cikis_istegi = False
        
        # GUI desteği
        self.gui_nesneleri = {}
        self.gui_pencere = None
        self.gui_canvas = None
        self.olaylar = defaultdict(dict)
        self.gui_renkler = {
            'siyah': 'black', 'beyaz': 'white', 'kırmızı': 'red', 'yeşil': 'green',
            'mavi': 'blue', 'sarı': 'yellow', 'mor': 'purple', 'turkuaz': 'cyan'
        }
        self.gui_ekran_modu = "metin"
        
        # Grafik desteği
        self.grafik_noktasi = (0, 0)
        self.grafik_renk = 'black'
        self.grafik_dolgu = None
        
        # Ses desteği
        self.ses_notalari = {
            'do': 262, 're': 294, 'mi': 330, 'fa': 349, 
            'sol': 392, 'la': 440, 'si': 494
        }
        
        # Paralel işlemler
        self.paralel_islemler = []
        self.ana_thread = threading.current_thread()
        
        # Standart kütüphane fonksiyonlarını yükle
        self.standart_kutuphaneyi_yukle()
    
    def standart_kutuphaneyi_yukle(self):
        # Matematik fonksiyonları
        self.fonksiyonlar['karekök'] = math.sqrt
        self.fonksiyonlar['mutlak'] = abs
        self.fonksiyonlar['üs'] = pow
        self.fonksiyonlar['sinüs'] = math.sin
        self.fonksiyonlar['kosinüs'] = math.cos
        self.fonksiyonlar['tanjant'] = math.tan
        self.fonksiyonlar['logaritma'] = math.log
        self.fonksiyonlar['yuvarla'] = round
        self.fonksiyonlar['tavan'] = math.ceil
        self.fonksiyonlar['taban'] = math.floor
        
        # Rastgele sayılar
        self.fonksiyonlar['rastgele'] = random.random
        self.fonksiyonlar['rastgele_tam'] = random.randint
        self.fonksiyonlar['rastgele_sec'] = random.choice
        
        # Tarih ve zaman
        self.fonksiyonlar['tarih'] = lambda: datetime.datetime.now().strftime("%d.%m.%Y")
        self.fonksiyonlar['saat'] = lambda: datetime.datetime.now().strftime("%H:%M:%S")
        self.fonksiyonlar['zamanlayıcı'] = time.time
        self.fonksiyonlar['bekle'] = time.sleep
        
        # Giriş/çıkış
        self.fonksiyonlar['yaz'] = self.yaz
        self.fonksiyonlar['sor'] = self.sor
        self.fonksiyonlar['temizle'] = self.temizle
        
        # Tip dönüşümleri
        self.fonksiyonlar['sayıyaçevir'] = float
        self.fonksiyonlar['tamsayıyaçevir'] = int
        self.fonksiyonlar['metneyap'] = str
        self.fonksiyonlar['karakter'] = chr
        self.fonksiyonlar['kod'] = ord
        self.fonksiyonlar['baytyap'] = lambda x: bytes([int(x)])
        
        # Dosya işlemleri
        self.fonksiyonlar['dosya_listele'] = os.listdir
        self.fonksiyonlar['klasöroluştur'] = os.mkdir
        self.fonksiyonlar['klasördegistir'] = os.chdir
        self.fonksiyonlar['klasörsil'] = os.rmdir
        self.fonksiyonlar['dosya_sil'] = os.remove
        self.fonksiyonlar['dosya_varmı'] = os.path.exists
        self.fonksiyonlar['dosya_yenidenadlandır'] = os.rename
        
        # Sistem komutları
        self.fonksiyonlar['komut'] = os.system
        self.fonksiyonlar['çıkış'] = self.cikis
        
        # Grafik komutları
        self.fonksiyonlar['nokta_çiz'] = self.nokta_ciz
        self.fonksiyonlar['çizgi_çiz'] = self.cizgi_ciz
        self.fonksiyonlar['daire_çiz'] = self.daire_ciz
        self.fonksiyonlar['dikdörtgen_çiz'] = self.dikdortgen_ciz
        self.fonksiyonlar['boya'] = self.boya
        self.fonksiyonlar['ekran_modu'] = self.ekran_modu
        self.fonksiyonlar['renk_ayarla'] = self.renk_ayarla
        self.fonksiyonlar['dolgu_ayarla'] = self.dolgu_ayarla
        self.fonksiyonlar['konumlandır'] = self.konumlandir
        self.fonksiyonlar['grafik_temizle'] = self.grafik_temizle
        
        # Ses komutları
        self.fonksiyonlar['bip'] = self.bip
        self.fonksiyonlar['nota_çal'] = self.nota_cal
        self.fonksiyonlar['müzik_çal'] = self.muzik_cal
        
        # Veritabanı işlemleri
        self.fonksiyonlar['veritabanı_aç'] = self.veritabani_ac
        self.fonksiyonlar['veritabanı_kapat'] = self.veritabani_kapat
        self.fonksiyonlar['sorgu_çalıştır'] = self.sorgu_calistir
        self.fonksiyonlar['tablo_oluştur'] = self.tablo_olustur
        self.fonksiyonlar['kayıt_ekle'] = self.kayit_ekle
        self.fonksiyonlar['kayıt_güncelle'] = self.kayit_guncelle
        self.fonksiyonlar['kayıt_sil'] = self.kayit_sil
        
        # Binary dosya işlemleri
        self.fonksiyonlar['binary_yükle'] = self.binary_yukle
        self.fonksiyonlar['binary_kaydet'] = self.binary_kaydet
        self.fonksiyonlar['belleğe_yaz'] = self.bellege_yaz
        self.fonksiyonlar['bellekten_oku'] = self.bellekten_oku
    
    def calistir(self, kod, dosya_adi=None):
        try:
            if isinstance(kod, str):
                satirlar = kod.split('\n')
            else:
                satirlar = kod
                
            self.etiketleri_bul(satirlar)
            
            i = 0
            while i < len(satirlar):
                satir = satirlar[i].strip()
                if not satir or satir.startswith(("'", "/*")):
                    i += 1
                    continue
                    
                # Çoklu komutları ayır
                komutlar = [k.strip() for k in satir.split(':') if k.strip()]
                
                for komut in komutlar:
                    if self.cikis_istegi:
                        return
                        
                    sonuc = self.komut_islet(komut)
                    if sonuc == "DONGU_ATLA":
                        break
                    elif sonuc == "DONGU_CIK":
                        return
                    elif isinstance(sonuc, int):
                        i = sonuc - 1  # Etikete git (i artırılacak)
                        break
                        
                i += 1
                
        except Exception as e:
            if self.hata_yonetimi:
                self.hata_yonetimi(str(e))
            else:
                print(f"Hata: {str(e)}")
    
    def komut_islet(self, komut):
        # Değişken atama
        esitlik = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*=\s*(.+)$', komut)
        if esitlik:
            degisken = esitlik.group(1)
            deger = self.ifade_hesapla(esitlik.group(2))
            self.degiskenler[degisken] = deger
            return
        
        # Tip belirterek değişken tanımlama
        tip_tanim = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s+(tam\s+sayı|gerçeksayı|metin|boolean|byte|liste|kume|sozluk|dizi)\s+tanımla$', komut)
        if tip_tanim:
            degisken = tip_tanim.group(1)
            tip = tip_tanim.group(2)
            
            if tip == "tam sayı":
                self.degiskenler[degisken] = 0
            elif tip == "gerçeksayı":
                self.degiskenler[degisken] = 0.0
            elif tip == "metin":
                self.degiskenler[degisken] = ""
            elif tip == "boolean":
                self.degiskenler[degisken] = False
            elif tip == "byte":
                self.degiskenler[degisken] = bytes(1)
            elif tip == "liste":
                self.degiskenler[degisken] = []
            elif tip == "kume":
                self.degiskenler[degisken] = set()
            elif tip == "sozluk":
                self.degiskenler[degisken] = {}
            elif tip == "dizi":
                self.degiskenler[degisken] = []  # Basit dizi
            return
        
        # Sabit tanımlama
        sabit_tanim = re.match(r'^sabit\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*=\s*(.+)$', komut)
        if sabit_tanim:
            sabit_adi = sabit_tanim.group(1)
            deger = self.ifade_hesapla(sabit_tanim.group(2))
            self.sabitler[sabit_adi] = deger
            return
        
        # Fonksiyon tanımlama
        fonk_tanim = re.match(r'^fonksiyon\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*(.*)$', komut)
        if fonk_tanim:
            fonk_adi = fonk_tanim.group(1)
            parametreler = [p.strip() for p in fonk_tanim.group(2).split(',') if p.strip()]
            
            # Fonksiyon gövdesini topla
            gövde = []
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "fonksiyon son":
                    break
                gövde.append(komut)
                
            self.fonksiyonlar[fonk_adi] = {
                'parametreler': parametreler,
                'gövde': gövde
            }
            return
        
        # Rutin (sub) tanımlama
        rutin_tanim = re.match(r'^rutin\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*(.*)$', komut)
        if rutin_tanim:
            rutin_adi = rutin_tanim.group(1)
            parametreler = [p.strip() for p in rutin_tanim.group(2).split(',') if p.strip()]
            
            # Rutin gövdesini topla
            gövde = []
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "rutinson":
                    break
                gövde.append(komut)
                
            self.rutinler[rutin_adi] = {
                'parametreler': parametreler,
                'gövde': gövde
            }
            return
        
        # Sınıf tanımlama
        sinif_tanim = re.match(r'^sınıf\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if sinif_tanim:
            sinif_adi = sinif_tanim.group(1)
            
            # Sınıf gövdesini topla
            ozellikler = []
            metotlar = {}
            
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "sınıf son":
                    break
                    
                # Özellik tanımlama
                ozellik = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s+(tam\s+sayı|gerçeksayı|metin|boolean|byte|liste|kume|sozluk|dizi)\s+tanımla$', komut)
                if ozellik:
                    ozellikler.append({
                        'ad': ozellik.group(1),
                        'tip': ozellik.group(2)
                    })
                    continue
                    
                # Metot tanımlama
                metot = re.match(r'^metot\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*(.*)$', komut)
                if metot:
                    metot_adi = metot.group(1)
                    parametreler = [p.strip() for p in metot.group(2).split(',') if p.strip()]
                    
                    # Metot gövdesini topla
                    gövde = []
                    while True:
                        komut = input("... " if self.repl_modu else "").strip()
                        if komut == "metot son":
                            break
                        gövde.append(komut)
                        
                    metotlar[metot_adi] = {
                        'parametreler': parametreler,
                        'gövde': gövde
                    }
                    continue
            
            self.siniflar[sinif_adi] = {
                'ozellikler': ozellikler,
                'metotlar': metotlar
            }
            return
        
        # Tip tanımlama
        tip_tanim = re.match(r'^tip\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if tip_tanim:
            tip_adi = tip_tanim.group(1)
            
            # Tip gövdesini topla
            ozellikler = []
            
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "tip son":
                    break
                    
                ozellik = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s+(tam\s+sayı|gerçeksayı|metin|boolean|byte|liste|kume|sozluk|dizi)\s+tanımla$', komut)
                if ozellik:
                    ozellikler.append({
                        'ad': ozellik.group(1),
                        'tip': ozellik.group(2)
                    })
            
            self.tipler[tip_adi] = ozellikler
            return
        
        # Eğer koşulu
        if komut.startswith('eğer '):
            kosul = komut[4:].split(' ise')[0].strip()
            sonuc = self.ifade_hesapla(kosul)
            
            if sonuc:
                return
            else:
                return "DONGU_ATLA"
        
        # Değilse
        if komut == 'değilse':
            return "DONGU_ATLA"
        
        # Eğer son
        if komut == 'eger son':
            return
        
        # Seç yapısı
        if komut.startswith('seç '):
            self.sec_degisken = komut[4:].strip()
            self.sec_durum = False
            return
        
        # Durum
        if komut.startswith('durum '):
            if not hasattr(self, 'sec_degisken'):
                raise Exception("Seç yapısı olmadan durum kullanılamaz")
                
            deger = komut[6:].strip()
            if self.ifade_hesapla(deger) == self.degiskenler.get(self.sec_degisken):
                self.sec_durum = True
                return
            else:
                self.sec_durum = False
                return "DONGU_ATLA"
        
        # Seç son
        if komut == 'seç son':
            if hasattr(self, 'sec_degisken'):
                del self.sec_degisken
            if hasattr(self, 'sec_durum'):
                del self.sec_durum
            return
        
        # Döngüler
        if komut.startswith('iken '):
            kosul = komut[5:].strip()
            sonuc = self.ifade_hesapla(kosul)
            
            if not sonuc:
                return "DONGU_ATLA"
            return
        
        if komut.startswith('için '):
            # için sayaç = başlangıç to bitiş adım adım
            parts = komut[5:].split()
            if len(parts) < 4 or parts[1] != '=' or parts[3] != 'to':
                raise Exception("Geçersiz için döngüsü sözdizimi")
            
            degisken = parts[0]
            baslangic = int(self.ifade_hesapla(parts[2]))
            bitis = int(self.ifade_hesapla(parts[4]))
            adim = 1 if len(parts) < 7 else int(self.ifade_hesapla(parts[6]))
            
            # Döngü durumunu kaydet
            self.dongu_durumu = {
                'tip': 'FOR',
                'degisken': degisken,
                'mevcut': baslangic,
                'bitis': bitis,
                'adim': adim
            }
            
            # İlk değeri ata
            self.degiskenler[degisken] = baslangic
            
            # Sınır kontrolü
            if (adim > 0 and baslangic > bitis) or (adim < 0 and baslangic < bitis):
                return "DONGU_ATLA"
            
            return
        
        if komut == 'say son':
            if hasattr(self, 'dongu_durumu') and self.dongu_durumu['tip'] == 'FOR':
                # Sonraki değer
                self.dongu_durumu['mevcut'] += self.dongu_durumu['adim']
                self.degiskenler[self.dongu_durumu['degisken']] = self.dongu_durumu['mevcut']
                
                # Sınır kontrolü
                if (self.dongu_durumu['adim'] > 0 and 
                    self.dongu_durumu['mevcut'] > self.dongu_durumu['bitis']):
                    del self.dongu_durumu
                    return
                elif (self.dongu_durumu['adim'] < 0 and 
                      self.dongu_durumu['mevcut'] < self.dongu_durumu['bitis']):
                    del self.dongu_durumu
                    return
                
                # Döngü başına dön
                return "DONGU_ATLA"
            return
        
        if komut == 'döngü son':
            return "DONGU_ATLA"
        
        # Fonksiyon çağırma
        fonk_cagri = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*\((.*)\)$', komut)
        if fonk_cagri:
            fonk_adi = fonk_cagri.group(1)
            parametreler = [p.strip() for p in fonk_cagri.group(2).split(',') if p.strip()]
            
            # Yerel değişkenler
            yerel_degiskenler = {}
            
            if fonk_adi in self.fonksiyonlar:
                fonk = self.fonksiyonlar[fonk_adi]
                
                # Parametreleri yerel değişkenlere ata
                for i, param in enumerate(fonk['parametreler']):
                    if i < len(parametreler):
                        yerel_degiskenler[param] = self.ifade_hesapla(parametreler[i])
                    else:
                        yerel_degiskenler[param] = None
                
                # Eski değişkenleri sakla
                eski_degiskenler = self.degiskenler.copy()
                
                # Yerel değişkenleri ekle
                self.degiskenler.update(yerel_degiskenler)
                
                # Fonksiyon gövdesini çalıştır
                sonuc = None
                try:
                    self.calistir(fonk['gövde'])
                except ReturnException as e:
                    sonuc = e.value
                
                # Değişkenleri geri yükle
                self.degiskenler = eski_degiskenler
                
                return sonuc
            
            elif fonk_adi in self.rutinler:
                rutin = self.rutinler[fonk_adi]
                
                # Parametreleri yerel değişkenlere ata
                for i, param in enumerate(rutin['parametreler']):
                    if i < len(parametreler):
                        yerel_degiskenler[param] = self.ifade_hesapla(parametreler[i])
                    else:
                        yerel_degiskenler[param] = None
                
                # Eski değişkenleri sakla
                eski_degiskenler = self.degiskenler.copy()
                
                # Yerel değişkenleri ekle
                self.degiskenler.update(yerel_degiskenler)
                
                # Rutin gövdesini çalıştır
                self.calistir(rutin['gövde'])
                
                # Değişkenleri geri yükle
                self.degiskenler = eski_degiskenler
                return
            
            elif fonk_adi in self.siniflar:
                # Sınıf örneği oluştur
                sinif = self.siniflar[fonk_adi]
                ornek = {}
                
                # Özellikleri başlat
                for ozellik in sinif['ozellikler']:
                    if ozellik['tip'] == "tam sayı":
                        ornek[ozellik['ad']] = 0
                    elif ozellik['tip'] == "gerçeksayı":
                        ornek[ozellik['ad']] = 0.0
                    elif ozellik['tip'] == "metin":
                        ornek[ozellik['ad']] = ""
                    elif ozellik['tip'] == "boolean":
                        ornek[ozellik['ad']] = False
                    elif ozellik['tip'] == "byte":
                        ornek[ozellik['ad']] = bytes(1)
                    elif ozellik['tip'] == "liste":
                        ornek[ozellik['ad']] = []
                    elif ozellik['tip'] == "kume":
                        ornek[ozellik['ad']] = set()
                    elif ozellik['tip'] == "sozluk":
                        ornek[ozellik['ad']] = {}
                    elif ozellik['tip'] == "dizi":
                        ornek[ozellik['ad']] = []
                
                # Metotları ekle
                for metot_adi, metot in sinif['metotlar'].items():
                    ornek[metot_adi] = lambda *args, m=metot, o=ornek: self.metot_calistir(m, o, args)
                
                return ornek
            
            else:
                raise Exception(f"Tanımlanmamış fonksiyon/rutin/sınıf: {fonk_adi}")
        
        # Metot çağırma
        metot_cagri = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\.([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*\((.*)\)$', komut)
        if metot_cagri:
            nesne_adi = metot_cagri.group(1)
            metot_adi = metot_cagri.group(2)
            parametreler = [p.strip() for p in metot_cagri.group(3).split(',') if p.strip()]
            
            if nesne_adi in self.degiskenler:
                nesne = self.degiskenler[nesne_adi]
                
                if isinstance(nesne, dict) and metot_adi in nesne:
                    # Metodu çağır
                    return nesne[metot_adi](*[self.ifade_hesapla(p) for p in parametreler])
                else:
                    raise Exception(f"Nesnede tanımlı olmayan metot: {metot_adi}")
            else:
                raise Exception(f"Tanımlanmamış nesne: {nesne_adi}")
        
        # Döngü kontrolü
        if komut == 'çık':
            return "DONGU_CIK"
        if komut == 'devam':
            return "DONGU_ATLA"
        
        # Git komutu
        git = re.match(r'^git\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if git:
            etiket = git.group(1)
            if etiket in self.etiketler:
                return self.etiketler[etiket]
            else:
                raise Exception(f"Tanımlanmamış etiket: {etiket}")
        
        # Alt programa git
        gosub = re.match(r'^altprograma_git\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if gosub:
            rutin_adi = gosub.group(1)
            if rutin_adi in self.rutinler:
                # Rutini çalıştır
                eski_degiskenler = self.degiskenler.copy()
                self.calistir(self.rutinler[rutin_adi]['gövde'])
                self.degiskenler = eski_degiskenler
            else:
                raise Exception(f"Tanımlanmamış rutin: {rutin_adi}")
            return
        
        # Alt programdan dön
        if komut == 'dön':
            return
        
        # Ver komutu
        if komut.startswith('ver '):
            deger = self.ifade_hesapla(komut[4:].strip())
            raise ReturnException(deger)
        
        # Diğer komutlar
        if komut.endswith(' yaz'):
            if komut.startswith("'"):
                print(komut[1:-4])
            else:
                deger = self.ifade_hesapla(komut[:-4].strip())
                print(deger)
            return
        
        if komut.endswith(' sor'):
            prompt = komut[:-4].strip()
            if prompt.startswith("'"):
                prompt = prompt[1:-1]
            deger = input(prompt + " ")
            return deger
        
        # Olay tanımlama
        olay_tanim = re.match(r'^olay\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if olay_tanim:
            olay_adi = olay_tanim.group(1)
            
            # Olay gövdesini topla
            govde = []
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "olay son":
                    break
                govde.append(komut)
            
            self.olaylar[olay_adi]['govde'] = govde
            return
        
        # Olay tetikleme
        olay_tetikle = re.match(r'^olay_tetikle\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if olay_tetikle:
            olay_adi = olay_tetikle.group(1)
            if olay_adi in self.olaylar and 'govde' in self.olaylar[olay_adi]:
                self.calistir(self.olaylar[olay_adi]['govde'])
            return
        
        # Paralel işlem
        if komut == 'paralel başla':
            # Paralel işlem gövdesini topla
            govde = []
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "paralel son":
                    break
                govde.append(komut)
            
            # Yeni thread oluştur
            t = threading.Thread(target=self.calistir, args=(govde,))
            t.start()
            self.paralel_islemler.append(t)
            return
        
        # Hata yönetimi
        if komut.startswith('hata yakala'):
            # Hata yönetimi gövdesini topla
            govde = []
            while True:
                komut = input("... " if self.repl_modu else "").strip()
                if komut == "hata son":
                    break
                govde.append(komut)
            
            self.hata_yonetimi = lambda hata: self.calistir(govde + [f"'Hata: '; {hata} yaz"])
            return
        
        # Kütüphane yükleme
        kutuphane = re.match(r'^kütüphane\s+"(.*)"$', komut)
        if kutuphane:
            dosya_adi = kutuphane.group(1)
            if not dosya_adi.endswith('.lbp'):
                dosya_adi += '.lbp'
            
            try:
                with open(dosya_adi, 'r', encoding='utf-8') as f:
                    kod = f.read()
                self.calistir(kod.split('\n'), dosya_adi)
            except FileNotFoundError:
                raise Exception(f"Kütüphane dosyası bulunamadı: {dosya_adi}")
            return
        
        # Modül yükleme
        modul = re.match(r'^modül\s+"(.*)"$', komut)
        if modul:
            dosya_adi = modul.group(1)
            if not dosya_adi.endswith('.tpl'):
                dosya_adi += '.tpl'
            
            try:
                with open(dosya_adi, 'r', encoding='utf-8') as f:
                    kod = f.read()
                self.calistir(kod.split('\n'), dosya_adi)
            except FileNotFoundError:
                raise Exception(f"Modül dosyası bulunamadı: {dosya_adi}")
            return
        
        # Dosya açma
        dosya_ac = re.match(r'^"(.*)"\s+(okuma|yazma|ekleme)\s+ac$', komut)
        if dosya_ac:
            dosya_adi = dosya_ac.group(1)
            mod = dosya_ac.group(2)
            
            if mod == "okuma":
                mod_str = 'r'
            elif mod == "yazma":
                mod_str = 'w'
            elif mod == "ekleme":
                mod_str = 'a'
            
            try:
                self.dosyalar[dosya_adi] = open(dosya_adi, mod_str, encoding='utf-8')
            except Exception as e:
                raise Exception(f"Dosya açılamadı: {str(e)}")
            return
        
        # Dosya kapatma
        dosya_kapat = re.match(r'^"(.*)"\s+kapat$', komut)
        if dosya_kapat:
            dosya_adi = dosya_kapat.group(1)
            if dosya_adi in self.dosyalar:
                self.dosyalar[dosya_adi].close()
                del self.dosyalar[dosya_adi]
            return
        
        # Dosyadan okuma
        dosya_oku = re.match(r'^dosya\s+oku\s+([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)$', komut)
        if dosya_oku:
            degisken = dosya_oku.group(1)
            if len(self.dosyalar) == 0:
                raise Exception("Açık dosya yok")
            dosya = list(self.dosyalar.values())[0]  # İlk açık dosya
            self.degiskenler[degisken] = dosya.readline().strip()
            return
        
        # Dosyaya yazma
        dosya_yaz = re.match(r'^dosya\s+yazdır\s+(.*)$', komut)
        if dosya_yaz:
            if len(self.dosyalar) == 0:
                raise Exception("Açık dosya yok")
            dosya = list(self.dosyalar.values())[0]  # İlk açık dosya
            icerik = self.ifade_hesapla(dosya_yaz.group(1))
            dosya.write(str(icerik) + '\n')
            return
        
        # Pencere oluşturma
        pencere_olustur = re.match(r'^pencere\s*=\s*yeni\s+Pencere\("(.*)",\s*(\d+),\s*(\d+)\)$', komut)
        if pencere_olustur:
            baslik = pencere_olustur.group(1)
            genislik = int(pencere_olustur.group(2))
            yukseklik = int(pencere_olustur.group(3))
            
            if self.gui_pencere is None:
                self.gui_pencere = tk.Tk()
                self.gui_pencere.title(baslik)
                self.gui_pencere.geometry(f"{genislik}x{yukseklik}")
                self.gui_canvas = Canvas(self.gui_pencere, width=genislik, height=yukseklik, bg='white')
                self.gui_canvas.pack()
            return
        
        # Pencere gösterme
        if komut == 'pencere.göster()':
            if self.gui_pencere:
                self.gui_pencere.mainloop()
            return
        
        # Buton oluşturma
        buton_olustur = re.match(r'^buton\s*=\s*yeni\s+Buton\("(.*)",\s*(\d+),\s*(\d+)\)$', komut)
        if buton_olustur:
            if not self.gui_pencere:
                raise Exception("Önce pencere oluşturulmalı")
            text = buton_olustur.group(1)
            x = int(buton_olustur.group(2))
            y = int(buton_olustur.group(3))
            
            btn = tk.Button(self.gui_pencere, text=text)
            btn.place(x=x, y=y)
            self.gui_nesneleri[f"buton_{len(self.gui_nesneleri)+1}"] = btn
            return
        
        # Diğer tüm durumlar için ifade hesaplama
        return self.ifade_hesapla(komut)
    
    def ifade_hesapla(self, ifade):
        # Değişken değeri
        if ifade in self.degiskenler:
            return self.degiskenler[ifade]
        if ifade in self.sabitler:
            return self.sabitler[ifade]
        
        # Metin değeri
        if ifade.startswith("'") and ifade.endswith("'"):
            return ifade[1:-1]
        
        # Sayısal değerler
        if ifade.replace('.', '', 1).isdigit():
            return float(ifade) if '.' in ifade else int(ifade)
        
        # Boolean değerler
        if ifade.lower() == 'doğru':
            return True
        if ifade.lower() == 'yanlış':
            return False
        
        # Liste tanımı
        if ifade.startswith('liste(') and ifade.endswith(')'):
            return []
        
        # Küme tanımı
        if ifade.startswith('kume(') and ifade.endswith(')'):
            return set()
        
        # Sözlük tanımı
        if ifade.startswith('sozluk(') and ifade.endswith(')'):
            return {}
        
        # Dizi tanımı
        dizi_tanim = re.match(r'^dizi\((\d+)\s*,\s*(\d+)\)$', ifade)
        if dizi_tanim:
            satir = int(dizi_tanim.group(1))
            sutun = int(dizi_tanim.group(2))
            return [[0 for _ in range(sutun)] for _ in range(satir)]
        
        # Aritmetik işlemler
        for op in ['+', '-', '*', '/', '%', '^']:
            if op in ifade:
                parcalar = ifade.split(op)
                sol = self.ifade_hesapla(parcalar[0].strip())
                sag = self.ifade_hesapla(op.join(parcalar[1:]).strip())
                
                if op == '+':
                    return sol + sag
                elif op == '-':
                    return sol - sag
                elif op == '*':
                    return sol * sag
                elif op == '/':
                    return sol / sag
                elif op == '%':
                    return sol % sag
                elif op == '^':
                    return sol ** sag
        
        # Artırma ve azaltma
        if ifade.endswith('++'):
            degisken = ifade[:-2].strip()
            self.degiskenler[degisken] = self.degiskenler.get(degisken, 0) + 1
            return self.degiskenler[degisken]
        if ifade.endswith('--'):
            degisken = ifade[:-2].strip()
            self.degiskenler[degisken] = self.degiskenler.get(degisken, 0) - 1
            return self.degiskenler[degisken]
        
        # Karşılaştırma operatörleri
        for op in ['==', '!=', '>', '<', '>=', '<=']:
            if op in ifade:
                parcalar = ifade.split(op)
                sol = self.ifade_hesapla(parcalar[0].strip())
                sag = self.ifade_hesapla(op.join(parcalar[1:]).strip())
                
                if op == '==':
                    return sol == sag
                elif op == '!=':
                    return sol != sag
                elif op == '>':
                    return sol > sag
                elif op == '<':
                    return sol < sag
                elif op == '>=':
                    return sol >= sag
                elif op == '<=':
                    return sol <= sag
        
        # Mantıksal operatörler
        if ' ve ' in ifade:
            parcalar = ifade.split(' ve ')
            return all(self.ifade_hesapla(p.strip()) for p in parcalar)
        if ' veya ' in ifade:
            parcalar = ifade.split(' veya ')
            return any(self.ifade_hesapla(p.strip()) for p in parcalar)
        if ifade.startswith('değil '):
            return not self.ifade_hesapla(ifade[6:].strip())
        
        # Fonksiyon çağırma
        fonk_cagri = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*\((.*)\)$', ifade)
        if fonk_cagri:
            fonk_adi = fonk_cagri.group(1)
            parametreler = [p.strip() for p in fonk_cagri.group(2).split(',') if p.strip()]
            
            if fonk_adi in self.fonksiyonlar:
                # Yerleşik fonksiyon
                args = [self.ifade_hesapla(p) for p in parametreler]
                return self.fonksiyonlar[fonk_adi](*args)
            else:
                raise Exception(f"Tanımlanmamış fonksiyon: {fonk_adi}")
        
        # Metot çağırma
        metot_cagri = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\.([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*\((.*)\)$', ifade)
        if metot_cagri:
            nesne_adi = metot_cagri.group(1)
            metot_adi = metot_cagri.group(2)
            parametreler = [p.strip() for p in metot_cagri.group(3).split(',') if p.strip()]
            
            if nesne_adi in self.degiskenler:
                nesne = self.degiskenler[nesne_adi]
                
                if isinstance(nesne, dict) and metot_adi in nesne:
                    # Metodu çağır
                    args = [self.ifade_hesapla(p) for p in parametreler]
                    return nesne[metot_adi](*args)
                else:
                    raise Exception(f"Nesnede tanımlı olmayan metot: {metot_adi}")
            else:
                raise Exception(f"Tanımlanmamış nesne: {nesne_adi}")
        
        # Liste/Küme/Sözlük erişimi
        if '[' in ifade and ']' in ifade:
            nesne_adi = ifade.split('[')[0].strip()
            indeks = self.ifade_hesapla(ifade[ifade.index('[')+1:ifade.index(']')].strip())
            
            if nesne_adi in self.degiskenler:
                nesne = self.degiskenler[nesne_adi]
                
                if isinstance(nesne, (list, tuple)) and isinstance(indeks, int):
                    return nesne[indeks]
                elif isinstance(nesne, dict):
                    return nesne[indeks]
                else:
                    raise Exception(f"Geçersiz indeksleme: {ifade}")
            else:
                raise Exception(f"Tanımlanmamış nesne: {nesne_adi}")
        
        # Tip dönüşümleri
        if ifade.startswith('sayıyaçevir(') and ifade.endswith(')'):
            deger = ifade[12:-1].strip()
            return float(self.ifade_hesapla(deger))
        
        if ifade.startswith('tamsayıyaçevir(') and ifade.endswith(')'):
            deger = ifade[16:-1].strip()
            return int(self.ifade_hesapla(deger))
        
        if ifade.startswith('metneyap(') and ifade.endswith(')'):
            deger = ifade[9:-1].strip()
            return str(self.ifade_hesapla(deger))
        
        if ifade.startswith('karakter(') and ifade.endswith(')'):
            kod = ifade[9:-1].strip()
            return chr(int(self.ifade_hesapla(kod)))
        
        if ifade.startswith('kod(') and ifade.endswith(')'):
            karakter = ifade[4:-1].strip()
            return ord(self.ifade_hesapla(karakter))
        
        if ifade.startswith('baytyap(') and ifade.endswith(')'):
            deger = ifade[8:-1].strip()
            return bytes([int(self.ifade_hesapla(deger))])
        
        # Grafik komutları
        if ifade.startswith('nokta_çiz(') and ifade.endswith(')'):
            params = ifade[10:-1].split(',')
            x = int(self.ifade_hesapla(params[0].strip()))
            y = int(self.ifade_hesapla(params[1].strip()))
            return self.nokta_ciz(x, y)
        
        if ifade.startswith('çizgi_çiz(') and ifade.endswith(')'):
            params = ifade[10:-1].split(',')
            x1 = int(self.ifade_hesapla(params[0].strip()))
            y1 = int(self.ifade_hesapla(params[1].strip()))
            x2 = int(self.ifade_hesapla(params[2].strip()))
            y2 = int(self.ifade_hesapla(params[3].strip()))
            return self.cizgi_ciz(x1, y1, x2, y2)
        
        if ifade.startswith('daire_çiz(') and ifade.endswith(')'):
            params = ifade[10:-1].split(',')
            x = int(self.ifade_hesapla(params[0].strip()))
            y = int(self.ifade_hesapla(params[1].strip()))
            r = int(self.ifade_hesapla(params[2].strip()))
            return self.daire_ciz(x, y, r)
        
        # Ses komutları
        if ifade.startswith('bip(') and ifade.endswith(')'):
            frekans = int(self.ifade_hesapla(ifade[4:-1].strip()))
            return self.bip(frekans)
        
        if ifade.startswith('nota_çal(') and ifade.endswith(')'):
            nota = ifade[9:-1].strip().strip("'")
            return self.nota_cal(nota)
        
        # Binary komutları
        if ifade.startswith('binary_yükle(') and ifade.endswith(')'):
            dosya_adi = ifade[13:-1].strip().strip("'")
            return self.binary_yukle(dosya_adi)
        
        raise Exception(f"Tanınmayan ifade: {ifade}")
    
    def etiketleri_bul(self, satirlar):
        self.etiketler = {}
        for i, satir in enumerate(satirlar):
            satir = satir.strip()
            if satir and not satir.startswith(("'", "/*")) and ':' not in satir:
                etiket = re.match(r'^([a-zA-ZğüşıöçĞÜŞİÖÇ_][a-zA-ZğüşıöçĞÜŞİÖÇ0-9_]*)\s*$', satir)
                if etiket:
                    self.etiketler[etiket.group(1)] = i + 1  # 1-based index
    
    def metot_calistir(self, metot, nesne, args):
        # Yerel değişkenler
        yerel_degiskenler = nesne.copy()
        
        # Parametreleri yerel değişkenlere ata
        for i, param in enumerate(metot['parametreler']):
            if i < len(args):
                yerel_degiskenler[param] = self.ifade_hesapla(args[i])
            else:
                yerel_degiskenler[param] = None
        
        # Eski değişkenleri sakla
        eski_degiskenler = self.degiskenler.copy()
        
        # Yerel değişkenleri ekle
        self.degiskenler.update(yerel_degiskenler)
        
        # Metot gövdesini çalıştır
        sonuc = None
        try:
            self.calistir(metot['gövde'])
        except ReturnException as e:
            sonuc = e.value
        
        # Değişkenleri geri yükle
        self.degiskenler = eski_degiskenler
        
        # Nesne durumunu güncelle
        for k, v in yerel_degiskenler.items():
            if k in nesne and k not in metot['parametreler']:
                nesne[k] = v
        
        return sonuc
    
    # Grafik fonksiyonları
    def nokta_ciz(self, x, y):
        if not self.gui_canvas:
            raise Exception("Önce ekran modu ayarlanmalı")
        self.gui_canvas.create_oval(x-1, y-1, x+1, y+1, fill=self.grafik_renk)
        self.grafik_noktasi = (x, y)
        return True
    
    def cizgi_ciz(self, x1, y1, x2, y2):
        if not self.gui_canvas:
            raise Exception("Önce ekran modu ayarlanmalı")
        self.gui_canvas.create_line(x1, y1, x2, y2, fill=self.grafik_renk)
        self.grafik_noktasi = (x2, y2)
        return True
    
    def daire_ciz(self, x, y, r):
        if not self.gui_canvas:
            raise Exception("Önce ekran modu ayarlanmalı")
        self.gui_canvas.create_oval(x-r, y-r, x+r, y+r, 
                                  outline=self.grafik_renk, 
                                  fill=self.grafik_dolgu)
        return True
    
    def dikdortgen_ciz(self, x1, y1, x2, y2):
        if not self.gui_canvas:
            raise Exception("Önce ekran modu ayarlanmalı")
        self.gui_canvas.create_rectangle(x1, y1, x2, y2, 
                                       outline=self.grafik_renk, 
                                       fill=self.grafik_dolgu)
        return True
    
    def boya(self, x, y, renk):
        if not self.gui_canvas:
            raise Exception("Önce ekran modu ayarlanmalı")
        # Basit doldurma uygulaması
        self.gui_canvas.create_rectangle(x-1, y-1, x+1, y+1, fill=renk, outline=renk)
        return True
    
    def ekran_modu(self, mod):
        if mod == "grafik":
            if not self.gui_pencere:
                self.gui_pencere = tk.Tk()
                self.gui_pencere.title("TPL Grafik Ekranı")
                self.gui_pencere.geometry("800x600")
                self.gui_canvas = Canvas(self.gui_pencere, width=800, height=600, bg='white')
                self.gui_canvas.pack()
            self.gui_ekran_modu = "grafik"
        else:
            self.gui_ekran_modu = "metin"
        return True
    
    def renk_ayarla(self, renk):
        if renk in self.gui_renkler:
            self.grafik_renk = self.gui_renkler[renk]
        else:
            self.grafik_renk = renk
        return True
    
    def dolgu_ayarla(self, renk):
        if renk in self.gui_renkler:
            self.grafik_dolgu = self.gui_renkler[renk]
        else:
            self.grafik_dolgu = renk
        return True
    
    def konumlandir(self, x, y):
        self.grafik_noktasi = (x, y)
        return True
    
    def grafik_temizle(self):
        if self.gui_canvas:
            self.gui_canvas.delete("all")
        return True
    
    # Ses fonksiyonları
    def bip(self, frekans=1000, sure=100):
        winsound.Beep(frekans, sure)
        return True
    
    def nota_cal(self, nota, sure=500):
        if nota in self.ses_notalari:
            winsound.Beep(self.ses_notalari[nota], sure)
        else:
            raise Exception(f"Bilinmeyen nota: {nota}")
        return True
    
    def muzik_cal(self, notalar):
        for nota in notalar.split():
            if nota in self.ses_notalari:
                winsound.Beep(self.ses_notalari[nota], 300)
            else:
                winsound.Beep(0, 100)  # Durak
        return True
    
    # Veritabanı fonksiyonları
    def veritabani_ac(self, dosya_adi):
        conn = sqlite3.connect(dosya_adi)
        self.degiskenler["_veritabani"] = conn
        return conn
    
    def veritabani_kapat(self):
        conn = self.degiskenler.get("_veritabani")
        if conn:
            conn.close()
            del self.degiskenler["_veritabani"]
        return True
    
    def sorgu_calistir(self, sorgu):
        conn = self.degiskenler.get("_veritabani")
        if not conn:
            raise Exception("Veritabanı bağlantısı yok")
        cursor = conn.cursor()
        cursor.execute(sorgu)
        conn.commit()
        return cursor.fetchall()
    
    def tablo_olustur(self, tablo_adi, alanlar):
        conn = self.degiskenler.get("_veritabani")
        if not conn:
            raise Exception("Veritabanı bağlantısı yok")
        
        alan_tanimi = ", ".join([f"{alan} TEXT" for alan in alanlar])
        sorgu = f"CREATE TABLE IF NOT EXISTS {tablo_adi} ({alan_tanimi})"
        
        cursor = conn.cursor()
        cursor.execute(sorgu)
        conn.commit()
        return True
    
    def kayit_ekle(self, tablo_adi, degerler):
        conn = self.degiskenler.get("_veritabani")
        if not conn:
            raise Exception("Veritabanı bağlantısı yok")
        
        placeholders = ", ".join(["?" for _ in degerler])
        sorgu = f"INSERT INTO {tablo_adi} VALUES ({placeholders})"
        
        cursor = conn.cursor()
        cursor.execute(sorgu, degerler)
        conn.commit()
        return cursor.lastrowid
    
    def kayit_guncelle(self, tablo_adi, set_deger, kosul):
        conn = self.degiskenler.get("_veritabani")
        if not conn:
            raise Exception("Veritabanı bağlantısı yok")
        
        sorgu = f"UPDATE {tablo_adi} SET {set_deger} WHERE {kosul}"
        
        cursor = conn.cursor()
        cursor.execute(sorgu)
        conn.commit()
        return cursor.rowcount
    
    def kayit_sil(self, tablo_adi, kosul):
        conn = self.degiskenler.get("_veritabani")
        if not conn:
            raise Exception("Veritabanı bağlantısı yok")
        
        sorgu = f"DELETE FROM {tablo_adi} WHERE {kosul}"
        
        cursor = conn.cursor()
        cursor.execute(sorgu)
        conn.commit()
        return cursor.rowcount
    
    # Binary dosya fonksiyonları
    def binary_yukle(self, dosya_adi):
        with open(dosya_adi, 'rb') as f:
            data = f.read()
        return data
    
    def binary_kaydet(self, dosya_adi, data):
        with open(dosya_adi, 'wb') as f:
            f.write(data)
        return True
    
    def bellege_yaz(self, adres, deger):
        # Basit bellek benzetimi
        if '_bellek' not in self.degiskenler:
            self.degiskenler['_bellek'] = bytearray(65536)  # 64KB bellek
        
        if adres < 0 or adres >= len(self.degiskenler['_bellek']):
            raise Exception("Geçersiz bellek adresi")
            
        self.degiskenler['_bellek'][adres] = deger
        return True
    
    def bellekten_oku(self, adres):
        if '_bellek' not in self.degiskenler:
            self.degiskenler['_bellek'] = bytearray(65536)  # 64KB bellek
        
        if adres < 0 or adres >= len(self.degiskenler['_bellek']):
            raise Exception("Geçersiz bellek adresi")
            
        return self.degiskenler['_bellek'][adres]
    
    # Diğer yardımcı fonksiyonlar
    def yaz(self, *args):
        print(' '.join(str(arg) for arg in args))
    
    def sor(self, prompt=''):
        return input(prompt)
    
    def temizle(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    def cikis(self):
        self.cikis_istegi = True
    
    def repl(self):
        self.repl_modu = True
        print("TPL (Türkçe Programlama Dili) REPL Ortamı")
        print("Yardım için 'yardım' yazın, çıkmak için 'ayrıl'")
        
        kod_buffer = []
        while not self.cikis_istegi:
            try:
                girdi = input(">>> " if not kod_buffer else "... ").strip()
                
                if girdi == 'ayrıl':
                    break
                elif girdi == 'yardım':
                    self.yardim_goster()
                    continue
                elif girdi == 'temizle':
                    self.temizle()
                    continue
                elif girdi == 'listele':
                    if kod_buffer:
                        print('\n'.join(kod_buffer))
                    continue
                elif girdi == 'calistir':
                    if kod_buffer:
                        self.calistir(kod_buffer)
                        kod_buffer = []
                    continue
                
                if girdi.endswith(':') or kod_buffer:
                    kod_buffer.append(girdi)
                    if not girdi.endswith(':'):
                        self.calistir(kod_buffer)
                        kod_buffer = []
                else:
                    self.calistir([girdi])
                    
            except KeyboardInterrupt:
                print("\nÇıkmak için 'ayrıl' yazın")
                kod_buffer = []
            except Exception as e:
                print(f"Hata: {str(e)}")
                kod_buffer = []
        
        self.repl_modu = False
    
    def yardim_goster(self):
        yardim_metni = """
===== TPL (Türkçe Programlama Dili) Yardım Sistemi =====

TEMEL KOMUTLAR:
----------------
1. Değişken Tanımlama:
   degisken_adi = deger
   x tam sayı tanımla  # Tip belirterek tanımlama
   PI = 3.14159        # Sabit değer

   Örnek:
     sayi = 10
     isim = 'Ahmet'
     liste = [1, 2, 3]

2. Giriş/Çıkış:
   'mesaj' yaz         # Ekrana yazdırma
   'soru' sor; cevap   # Kullanıcıdan giriş alma

   Örnek:
     'Adınız nedir?' sor; ad
     'Merhaba '; ad yaz

KONTROL YAPILARI:
-----------------
1. Eğer Yapısı:
   eğer koşul ise
      # doğruysa yapılacaklar
   değilse
      # yanlışsa yapılacaklar
   eger son

   Örnek:
     eğer sayi > 0 ise
        'Pozitif' yaz
     değilse
        'Negatif veya sıfır' yaz
     eger son

2. Seç Yapısı:
   seç değişken
   durum deger1
      # deger1 durumu
   durum deger2
      # deger2 durumu
   seç son

   Örnek:
     seç gun
     durum 1
        'Pazartesi' yaz
     durum 2
        'Salı' yaz
     seç son

DÖNGÜLER:
---------
1. İçin Döngüsü:
   sayaç için başlangıç to bitiş adım adım
      # döngü gövdesi
   say son

   Örnek:
     i için 1 to 10 adım 2
        i yaz
     say son

2. İken Döngüsü:
   iken koşul
      # döngü gövdesi
   döngü son

   Örnek:
     sayac = 1
     iken sayac < 5
        sayac yaz
        sayac = sayac + 1
     döngü son

FONKSİYONLAR VE RUTİNLER:
-------------------------
1. Fonksiyon Tanımı:
   fonksiyon isim(param1, param2)
      # fonksiyon gövdesi
      sonuc ver  # değer döndürme
   fonksiyon son

   Örnek:
     fonksiyon kare(x)
        sonuc = x * x
        sonuc ver
     fonksiyon son

     sonuc = kare(5)
     sonuc yaz

2. Rutin (Alt Program):
   rutin isim(param1)
      # rutin gövdesi
   rutinson

   Örnek:
     rutin selamla(isim)
        'Merhaba '; isim yaz
     rutinson

     selamla('Ayşe')

NESNE YÖNELİMLİ PROGRAMLAMA:
----------------------------
1. Sınıf Tanımı:
   sınıf SınıfAdı
      özellik1 tip tanımla
      özellik2 tip tanımla

      metot metot_adi(param)
         # metot gövdesi
      metot son
   sınıf son

   Örnek:
     sınıf Dikdörtgen
        en tam sayı tanımla
        boy tam sayı tanımla

        metot alan()
           en * boy ver
        metot son
     sınıf son

     d = Dikdörtgen()
     d.en = 10
     d.boy = 5
     alan = d.alan()

DOSYA İŞLEMLERİ:
----------------
1. Dosya Açma/Kapama:
   'dosya.txt' okuma aç
   dosya oku veri
   dosya kapat

   'dosya.txt' yazma aç
   dosya yazdır 'İçerik'
   dosya kapat

   Örnek:
     'veriler.txt' okuma aç
     dosya oku satir
     satir yaz
     dosya kapat

VERİTABANI İŞLEMLERİ:
---------------------
1. Veritabanı Bağlantısı:
   vt = veritabanı_aç('veriler.db')
   tablo_oluştur('kullanicilar', ['id', 'ad', 'yas'])
   kayıt_ekle('kullanicilar', [1, 'Ahmet', 25])
   sonuc = sorgu_çalıştır('SELECT * FROM kullanicilar')
   sonuc yaz
   veritabanı_kapat()

GRAFİK İŞLEMLERİ:
-----------------
1. Grafik Modu:
   ekran_modu('grafik')
   pencere = yeni Pencere('Başlık', 800, 600)
   renk_ayarla('kırmızı')
   nokta_çiz(100, 100)
   çizgi_çiz(50, 50, 150, 150)
   daire_çiz(300, 300, 50)
   pencere.göster()

SES İŞLEMLERİ:
--------------
1. Ses Komutları:
   bip()                   # Standart bip sesi
   nota_çal('do')          # Do notasını çal
   müzik_çal('do re mi')   # Nota dizisini çal

BINARY VE BELLEK İŞLEMLERİ:
---------------------------
1. Binary Dosyalar:
   veri = binary_yükle('dosya.bin')
   binary_kaydet('kopya.bin', veri)

2. Bellek Erişimi:
   bellege_yaz(100, 255)   # 100. adrese 255 yaz
   deger = bellekten_oku(100)

REPL KOMUTLARI:
---------------
   yardım    : Bu yardım mesajını göster
   temizle   : Ekranı temizle
   listele   : Kod bufferını göster
   calistir  : Kod bufferını çalıştır
   ayrıl     : REPL'den çık

=======================================================
"""
        print(yardim_metni)


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class TPLGUI:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.root = tk.Tk()
        self.root.title("TPL (Türkçe Programlama Dili)")
        self.root.geometry("1000x800")
        
        self.setup_ui()
    
    def setup_ui(self):
        # Menü çubuğu
        menubar = tk.Menu(self.root)
        
        # Dosya menüsü
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Aç", command=self.dosya_ac, accelerator="Ctrl+O")
        filemenu.add_command(label="Kaydet", command=self.dosya_kaydet, accelerator="Ctrl+S")
        filemenu.add_command(label="Farklı Kaydet", command=self.dosya_farkli_kaydet)
        filemenu.add_separator()
        filemenu.add_command(label="Çıkış", command=self.root.quit, accelerator="Alt+F4")
        menubar.add_cascade(label="Dosya", menu=filemenu)
        
        # Düzen menüsü
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Kes", command=self.kes, accelerator="Ctrl+X")
        editmenu.add_command(label="Kopyala", command=self.kopyala, accelerator="Ctrl+C")
        editmenu.add_command(label="Yapıştır", command=self.yapistir, accelerator="Ctrl+V")
        editmenu.add_separator()
        editmenu.add_command(label="Tümünü Seç", command=self.tumunu_sec, accelerator="Ctrl+A")
        menubar.add_cascade(label="Düzen", menu=editmenu)
        
        # Çalıştır menüsü
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Çalıştır", command=self.calistir, accelerator="F5")
        runmenu.add_command(label="Durdur", command=self.durdur, accelerator="Ctrl+Break")
        runmenu.add_command(label="Temizle", command=self.temizle)
        menubar.add_cascade(label="Çalıştır", menu=runmenu)
        
        # Yardım menüsü
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Yardım", command=self.yardim, accelerator="F1")
        helpmenu.add_command(label="Komut Listesi", command=self.komut_listesi)
        helpmenu.add_command(label="Hakkında", command=self.hakkinda)
        menubar.add_cascade(label="Yardım", menu=helpmenu)
        
        self.root.config(menu=menubar)
        
        # PanedWindow ile bölünmüş arayüz
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)
        
        # Sol panel - Kod editörü
        left_frame = tk.Frame(main_pane)
        main_pane.add(left_frame)
        
        editor_label = tk.Label(left_frame, text="Kod Editörü", font=("Arial", 10, "bold"))
        editor_label.pack(pady=5)
        
        editor_frame = tk.Frame(left_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.kod_editoru = tk.Text(editor_frame, wrap=tk.WORD, undo=True, font=("Courier New", 12))
        self.kod_editoru.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(editor_frame, command=self.kod_editoru.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.kod_editoru.config(yscrollcommand=scrollbar.set)
        
        # Sağ panel - Çıktı ve Yardım
        right_pane = tk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane)
        
        # Çıktı alanı
        cikti_frame = tk.Frame(right_pane)
        right_pane.add(cikti_frame)
        
        cikti_label = tk.Label(cikti_frame, text="Program Çıktısı", font=("Arial", 10, "bold"))
        cikti_label.pack(pady=5)
        
        cikti_inner_frame = tk.Frame(cikti_frame)
        cikti_inner_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cikti_alani = tk.Text(cikti_inner_frame, wrap=tk.WORD, state=tk.DISABLED, 
                                 font=("Courier New", 11), bg='#f0f0f0')
        self.cikti_alani.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        cikti_scrollbar = tk.Scrollbar(cikti_inner_frame, command=self.cikti_alani.yview)
        cikti_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cikti_alani.config(yscrollcommand=cikti_scrollbar.set)
        
        # Yardım alanı
        yardim_frame = tk.Frame(right_pane)
        right_pane.add(yardim_frame)
        
        yardim_label = tk.Label(yardim_frame, text="Hızlı Yardım", font=("Arial", 10, "bold"))
        yardim_label.pack(pady=5)
        
        yardim_inner_frame = tk.Frame(yardim_frame)
        yardim_inner_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.yardim_alani = tk.Text(yardim_inner_frame, wrap=tk.WORD, state=tk.DISABLED,
                                  font=("Arial", 10), height=10, bg='#ffffe0')
        self.yardim_alani.pack(fill=tk.BOTH, expand=True)
        
        # Yardım içeriğini yükle
        self.yardim_alani.config(state=tk.NORMAL)
        self.yardim_alani.insert(tk.END, "F1: Tam Yardım | F2: Komut Listesi\n\n")
        self.yardim_alani.insert(tk.END, "Temel Komutlar:\n")
        self.yardim_alani.insert(tk.END, "  degisken = deger\n")
        self.yardim_alani.insert(tk.END, "  'mesaj' yaz\n")
        self.yardim_alani.insert(tk.END, "  eğer ... ise ... değilse ... eger son\n")
        self.yardim_alani.config(state=tk.DISABLED)
        
        # Durum çubuğu
        self.durum_cubugu = tk.Label(self.root, text="Hazır", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.durum_cubugu.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Hızlandırıcı tuşlar
        self.root.bind('<Control-o>', lambda event: self.dosya_ac())
        self.root.bind('<Control-s>', lambda event: self.dosya_kaydet())
        self.root.bind('<F5>', lambda event: self.calistir())
        self.root.bind('<F1>', lambda event: self.yardim())
        self.root.bind('<F2>', lambda event: self.komut_listesi())
    
    def dosya_ac(self):
        dosya_adi = filedialog.askopenfilename(
            filetypes=[("TPL Dosyaları", "*.tpl"), ("Tüm Dosyalar", "*.*")]
        )
        if dosya_adi:
            with open(dosya_adi, 'r', encoding='utf-8') as f:
                self.kod_editoru.delete(1.0, tk.END)
                self.kod_editoru.insert(tk.END, f.read())
            self.durum_cubugu.config(text=f"Açılan dosya: {dosya_adi}")
            self.root.title(f"TPL - {dosya_adi}")
    
    def dosya_kaydet(self):
        if hasattr(self, 'mevcut_dosya') and self.mevcut_dosya:
            with open(self.mevcut_dosya, 'w', encoding='utf-8') as f:
                f.write(self.kod_editoru.get(1.0, tk.END))
            self.durum_cubugu.config(text=f"Kaydedilen dosya: {self.mevcut_dosya}")
        else:
            self.dosya_farkli_kaydet()
    
    def dosya_farkli_kaydet(self):
        dosya_adi = filedialog.asksaveasfilename(
            defaultextension=".tpl", 
            filetypes=[("TPL Dosyaları", "*.tpl"), ("Tüm Dosyalar", "*.*")]
        )
        if dosya_adi:
            with open(dosya_adi, 'w', encoding='utf-8') as f:
                f.write(self.kod_editoru.get(1.0, tk.END))
            self.mevcut_dosya = dosya_adi
            self.durum_cubugu.config(text=f"Kaydedilen dosya: {dosya_adi}")
            self.root.title(f"TPL - {dosya_adi}")
    
    def kes(self):
        self.kod_editoru.event_generate("<<Cut>>")
    
    def kopyala(self):
        self.kod_editoru.event_generate("<<Copy>>")
    
    def yapistir(self):
        self.kod_editoru.event_generate("<<Paste>>")
    
    def tumunu_sec(self):
        self.kod_editoru.tag_add(tk.SEL, "1.0", tk.END)
        self.kod_editoru.mark_set(tk.INSERT, "1.0")
        self.kod_editoru.see(tk.INSERT)
        return "break"
    
    def calistir(self):
        kod = self.kod_editoru.get(1.0, tk.END).split('\n')
        self.cikti_alani.config(state=tk.NORMAL)
        self.cikti_alani.delete(1.0, tk.END)
        
        # Çıktıyı yönlendir
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.interpreter.calistir(kod)
            output = sys.stdout.getvalue()
            self.cikti_alani.insert(tk.END, output)
            self.durum_cubugu.config(text="Çalıştırma başarıyla tamamlandı")
        except Exception as e:
            self.cikti_alani.insert(tk.END, f"Hata: {str(e)}")
            self.durum_cubugu.config(text=f"Hata: {str(e)}")
        finally:
            sys.stdout = old_stdout
            self.cikti_alani.config(state=tk.DISABLED)
    
    def durdur(self):
        self.interpreter.cikis_istegi = True
        self.durum_cubugu.config(text="Çalıştırma durduruldu")
    
    def temizle(self):
        self.cikti_alani.config(state=tk.NORMAL)
        self.cikti_alani.delete(1.0, tk.END)
        self.cikti_alani.config(state=tk.DISABLED)
        self.durum_cubugu.config(text="Çıktı temizlendi")
    
    def yardim(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("TPL Yardım Sistemi")
        help_window.geometry("800x600")
        
        # Sekmeli arayüz
        notebook = tk.ttk.Notebook(help_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Temel Komutlar sekmesi
        temel_frame = tk.Frame(notebook)
        notebook.add(temel_frame, text="Temel Komutlar")
        
        temel_text = tk.Text(temel_frame, wrap=tk.WORD)
        temel_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        temel_content = """
=== TEMEL KOMUTLAR ===

1. Değişken Tanımlama:
   degisken_adi = deger
   x tam sayı tanımla  # Tip belirterek tanımlama
   PI = 3.14159        # Sabit değer

   Örnek:
     sayi = 10
     isim = 'Ahmet'
     liste = [1, 2, 3]

2. Giriş/Çıkış:
   'mesaj' yaz         # Ekrana yazdırma
   'soru' sor; cevap   # Kullanıcıdan giriş alma

   Örnek:
     'Adınız nedir?' sor; ad
     'Merhaba '; ad yaz
"""
        temel_text.insert(tk.END, temel_content)
        temel_text.config(state=tk.DISABLED)
        
        # Kontrol Yapıları sekmesi
        kontrol_frame = tk.Frame(notebook)
        notebook.add(kontrol_frame, text="Kontrol Yapıları")
        
        kontrol_text = tk.Text(kontrol_frame, wrap=tk.WORD)
        kontrol_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        kontrol_content = """
=== KONTROL YAPILARI ===

1. Eğer Yapısı:
   eğer koşul ise
      # doğruysa yapılacaklar
   değilse
      # yanlışsa yapılacaklar
   eger son

   Örnek:
     eğer sayi > 0 ise
        'Pozitif' yaz
     değilse
        'Negatif veya sıfır' yaz
     eger son

2. Seç Yapısı:
   seç değişken
   durum deger1
      # deger1 durumu
   durum deger2
      # deger2 durumu
   seç son

   Örnek:
     seç gun
     durum 1
        'Pazartesi' yaz
     durum 2
        'Salı' yaz
     seç son
"""
        kontrol_text.insert(tk.END, kontrol_content)
        kontrol_text.config(state=tk.DISABLED)
        
        # Diğer sekmeler benzer şekilde eklenebilir...
        
        close_button = tk.Button(help_window, text="Kapat", command=help_window.destroy)
        close_button.pack(pady=10)
    
    def komut_listesi(self):
        list_window = tk.Toplevel(self.root)
        list_window.title("TPL Komut Listesi")
        list_window.geometry("600x400")
        
        # Treeview ile komut listesi
        tree = tk.ttk.Treeview(list_window, columns=("Açıklama"), show="headings")
        tree.heading("#0", text="Komut")
        tree.heading("Açıklama", text="Açıklama")
        
        tree.column("#0", width=150)
        tree.column("Açıklama", width=450)
        
        # Komutları ekle
        komutlar = [
            ("yaz", "Ekrana çıktı verir"),
            ("sor", "Kullanıcı girişi alır"),
            ("eğer/ise/değilse/eger son", "Koşul yapısı"),
            ("için/to/say son", "For döngüsü"),
            ("iken/döngü son", "While döngüsü"),
            ("fonksiyon/fonksiyon son", "Fonksiyon tanımlama"),
            ("rutin/rutinson", "Alt program tanımlama"),
            ("sınıf/sınıf son", "Sınıf tanımlama"),
            ("ac/oku/yazdır/kapat", "Dosya işlemleri"),
            ("veritabanı_aç/sorgu_çalıştır", "Veritabanı işlemleri"),
            ("nokta_çiz/çizgi_çiz/daire_çiz", "Grafik komutları"),
            ("bip/nota_çal", "Ses komutları")
        ]
        
        for komut, aciklama in komutlar:
            tree.insert("", "end", text=komut, values=(aciklama,))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        close_button = tk.Button(list_window, text="Kapat", command=list_window.destroy)
        close_button.pack(pady=10)
    
    def hakkinda(self):
        messagebox.showinfo("Hakkında", 
            "TPL (Türkçe Programlama Dili)\n\nSürüm 2.0\n\nPython ile geliştirilmiştir.")
    
    def run(self):
        self.root.mainloop()


def main():
    if len(sys.argv) > 1:
        # Dosya modu
        interpreter = TPLInterpreter()
        try:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                kod = f.read()
            interpreter.calistir(kod.split('\n'), sys.argv[1])
        except FileNotFoundError:
            print(f"Dosya bulunamadı: {sys.argv[1]}")
    else:
        # GUI modu
        interpreter = TPLInterpreter()
        gui = TPLGUI(interpreter)
        gui.run()


if __name__ == "__main__":
    # Tkinter ttk stilini yükle
    if 'ttk' in dir(tk):
        style = tk.ttk.Style()
        style.theme_use('clam')  # Modern bir tema
    
    main()