import sys
import os
import re
import math
import random
import time
import datetime
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from queue import Queue
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
        
        # REPL özellikleri
        self.repl_modu = False
        self.cikis_istegi = False
        
        # GUI desteği
        self.gui_nesneleri = {}
        self.olaylar = defaultdict(dict)
        
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
        
        # Rastgele sayılar
        self.fonksiyonlar['rastgele'] = random.random
        self.fonksiyonlar['rastgele_tam'] = random.randint
        
        # Tarih ve zaman
        self.fonksiyonlar['tarih'] = lambda: datetime.datetime.now().strftime("%d.%m.%Y")
        self.fonksiyonlar['saat'] = lambda: datetime.datetime.now().strftime("%H:%M:%S")
        self.fonksiyonlar['zamanlayıcı'] = time.time
        
        # Giriş/çıkış
        self.fonksiyonlar['yaz'] = self.yaz
        self.fonksiyonlar['sor'] = self.sor
        self.fonksiyonlar['temizle'] = self.temizle
        
        # Tip dönüşümleri
        self.fonksiyonlar['sayıyaçevir'] = float
        self.fonksiyonlar['metneyap'] = str
        self.fonksiyonlar['karakter'] = chr
        self.fonksiyonlar['kod'] = ord
        
        # Dosya işlemleri
        self.fonksiyonlar['dosya_listele'] = os.listdir
        self.fonksiyonlar['klasöroluştur'] = os.mkdir
        self.fonksiyonlar['klasördegistir'] = os.chdir
        self.fonksiyonlar['klasörsil'] = os.rmdir
        self.fonksiyonlar['dosya_sil'] = os.remove
        
        # Sistem komutları
        self.fonksiyonlar['komut'] = os.system
        self.fonksiyonlar['çıkış'] = self.cikis
    
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
        
        # Döngüler
        if komut.startswith('iken '):
            kosul = komut[5:].strip()
            sonuc = self.ifade_hesapla(kosul)
            
            if not sonuc:
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
        
        # Diğer tüm durumlar için ifade hesaplama
        return self.ifade_hesapla(komut)
    
    def ifade_hesapla(self, ifade):
        # Değişken değeri
        if ifade in self.degiskenler:
            return self.degiskenler[ifade]
        
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
        
        if ifade.startswith('metneyap(') and ifade.endswith(')'):
            deger = ifade[9:-1].strip()
            return str(self.ifade_hesapla(deger))
        
        if ifade.startswith('karakter(') and ifade.endswith(')'):
            kod = ifade[9:-1].strip()
            return chr(int(self.ifade_hesapla(kod)))
        
        if ifade.startswith('kod(') and ifade.endswith(')'):
            karakter = ifade[4:-1].strip()
            return ord(self.ifade_hesapla(karakter))
        
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
        print("Commodore.gen.tr icin BASIC Destegi Saglar")
        print("      Dusundugun gibi programlama yap")
        print("Yardım için 'yardım' yazın, çıkmak için 'ayrıl' yazın")
        
        # REPL durumu
        repl_lines = []  # Tüm kod satırlarını saklar
        current_block = None  # Şu anki blok tipi
        block_lines = []  # Blok içindeki satırlar
        block_stack = []  # İç içe bloklar için yığın
        
        # Blok sonlarını tanımla
        block_ends = {
            "fonksiyon": "fonksiyon son",
            "rutin": "rutinson",
            "sınıf": "sınıf son",
            "tip": "tip son",
            "olay": "olay son",
            "paralel başla": "paralel son",
            "hata yakala": "hata son",
            "eğer": "eger son",
            "iken": "döngü son",
            "için": "say son"
        }
        
        while not self.cikis_istegi:
            try:
                # Blok durumuna göre uygun istemi göster
                if current_block:
                    prompt = "... "
                else:
                    prompt = "TPL> "
                
                line = input(prompt).strip()
                
                # REPL komutlarını kontrol et (sadece blok dışında)
                if not current_block and line:
                    # Yardım komutu
                    if line == "yardım":
                        print("TPL REPL Komutları:")
                        print("  yükle 'dosya.tpl' - Program yükle")
                        print("  kaydet 'dosya.tpl' - Programı kaydet")
                        print("  listele - Kodu görüntüle")
                        print("  çalıştır - Programı çalıştır")
                        print("  temizle - Ekranı temizle")
                        print("  ayrıl - REPL'den çık")
                        print("  yardım - Bu mesajı göster")
                        continue
                    
                    # Yükle komutu
                    if line.startswith("yükle "):
                        match = re.match(r'yükle\s+"(.+)"', line)
                        if match:
                            filename = match.group(1)
                            if not filename.endswith('.tpl'):
                                filename += '.tpl'
                            try:
                                with open(filename, 'r', encoding='utf-8') as f:
                                    repl_lines = f.read().splitlines()
                                print(f"{filename} yüklendi")
                            except FileNotFoundError:
                                print(f"Dosya bulunamadı: {filename}")
                        else:
                            print("Kullanım: yükle \"dosya.tpl\"")
                        continue
                    
                    # Kaydet komutu
                    if line.startswith("kaydet "):
                        match = re.match(r'kaydet\s+"(.+)"', line)
                        if match:
                            filename = match.group(1)
                            if not filename.endswith('.tpl'):
                                filename += '.tpl'
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(repl_lines))
                            print(f"Program {filename} olarak kaydedildi")
                        else:
                            print("Kullanım: kaydet \"dosya.tpl\"")
                        continue
                    
                    # Listele komutu
                    if line == "listele":
                        print("\nProgram Kodu:")
                        for i, code_line in enumerate(repl_lines):
                            print(f"{i+1}: {code_line}")
                        continue
                    
                    # Çalıştır komutu
                    if line == "çalıştır":
                        print("Program çalıştırılıyor...")
                        self.calistir(repl_lines)
                        print("Program tamamlandı")
                        continue
                    
                    # Temizle komutu
                    if line == "temizle":
                        self.temizle()
                        continue
                    
                    # Ayrıl komutu
                    if line == "ayrıl":
                        self.cikis_istegi = True
                        continue
                
                # Blok başlangıcını kontrol et
                block_started = False
                for block_type, start_pattern in [
                    ("fonksiyon", r"fonksiyon\s+\w+"),
                    ("rutin", r"rutin\s+\w+"),
                    ("sınıf", r"sınıf\s+\w+"),
                    ("tip", r"tip\s+\w+"),
                    ("olay", r"olay\s+\w+"),
                    ("paralel başla", r"paralel başla"),
                    ("hata yakala", r"hata yakala"),
                    ("eğer", r"eğer\s+.+\s+ise"),
                    ("iken", r"iken\s+.+"),
                    ("için", r"için\s+.+\s+kadar")
                ]:
                    if re.match(start_pattern, line):
                        if not current_block:  # Yeni dış blok
                            current_block = block_type
                            block_stack = [block_type]
                        else:  # İç içe blok
                            block_stack.append(block_type)
                        block_started = True
                        break
                
                # Satırı uygun listeye ekle
                if current_block:
                    block_lines.append(line)
                else:
                    repl_lines.append(line)
                
                # Blok sonunu kontrol et
                if current_block and line == block_ends[block_stack[-1]]:
                    block_stack.pop()
                    if not block_stack:  # Blok tamamlandı
                        repl_lines.extend(block_lines)
                        self.calistir(block_lines)
                        current_block = None
                        block_lines = []
                
                # Blok dışında ve tek satırlık komut
                if not current_block and not block_started:
                    self.calistir([line])
            
            except KeyboardInterrupt:
                print("\nİptal edildi. Çıkmak için 'ayrıl' yazın")
                current_block = None
                block_lines = []
                block_stack = []
            
            except Exception as e:
                if self.hata_yonetimi:
                    self.hata_yonetimi(str(e))
                else:
                    print(f"Hata: {str(e)}")
        
        print("TPL REPL'den çıkılıyor...")
        self.repl_modu = False
