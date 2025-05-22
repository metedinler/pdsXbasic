# PDS‑Xbasic Programlama Rehberi

> **Sürüm Taslağı 1.0 – 22 Mayıs 2025**
> **Hazırlayan:** ChatGPT
> Bu rehber, PDS‑Xbasic dilini ve modüllerini adım adım anlatan **yeni bir belge dizisi** olarak oluşturulacaktır. Her modül ayrıntılı biçimde incelenecek, sözdizimleri, örnekler ve en iyi uygulamalar ardışık olarak rehbere eklenecektir.

---

## İçindekiler (büyüdükçe güncellenecek)

1. [Giriş](#giriş)
2. [Ana Yorumlayıcı – `pdsXuv14.py`](#pdsxuv14py)
3. [Derleyici & Sanal Makine](#derleyici-vm)
4. [Bellek, Olay, Boru Hattı](#bellek-olay-pipe)
5. [Veri Yapıları](#veri-yapıları)
6. [Veri Katmanı](#veri-katmanı)
7. [libx\* Uzantıları](#libx)
8. [Tanılama & Profiling](#tanılama)

---

## 1 · Giriş <a name="giriş"></a>

PDS‑Xbasic, klasik BASIC sözdizimini modern Python altyapısıyla birleştiren, modüler ve genişletilebilir bir programlama ortamıdır. Bu rehberin amacı, **sıfırdan başlayan** bir geliştiricinin yorumlayıcıyı kurup verimli kod yazabilecek seviyeye gelmesini sağlamaktır.

### 1.1 Neden Bu Rehber?

Önceki doküman, sistem mimarisini detaylı teknik referans şeklinde sundu. **Bu yeni belge**, aynı bilgiyi **öğretici sırayla** modül modül işler; her bölüm sonunda pratik alıştırmalar içerir.

### 1.2 Kullanım Şekli

* Rehberi **baştan sona** adım adım takip edin.
* Her modül anlatımı ardından kod deneyin.
* Sorularınız için yan pencerede ChatGPT’ye “devam” diyerek yeni örnekler isteyin.

---

## 2 · Ana Yorumlayıcı – `pdsXuv14.py` <a name="pdsxuv14py"></a>

PDS‑Xbasic yorumlayıcısının kalbi **`pdsXuv14.py`** dosyasıdır. Bu bölümde motorun mimarisini, komut/fonksiyon tablolarını ve ilk çalıştırma adımlarını öğreneceğiz.

### 2.1 Görev & Bileşenler

| Bileşen      | Dosya / Sınıf                     | Görev                                          |
| ------------ | --------------------------------- | ---------------------------------------------- |
| Tokenizasyon | `tokenize(line)`                  | Kaynak satırı sözcüklere böler.                |
| Ayrıştırma   | `parse_statement()`               | Komutu / ifadeyi AST’ye dönüştürür.            |
| Yürütme      | `command_table`, `function_table` | Komut → `cmd_*`, Fonksiyon → metot.            |
| Bytecode VM  | `bytecode_manager.execute()`      | Opsiyonel hızlandırma için PDX sanal makinesi. |
| Tanılama     | `f11_backtrace_logger`            | Holografik geri izleme & hata korelasyonu.     |

### 2.2 Komut Referansı (40 Komut – Tam Liste)

| #  | Komut               | Sözdizimi Kısa          | Açıklama                  |
| -- | ------------------- | ----------------------- | ------------------------- |
| 1  | `REM`               | `REM metin`             | Yorum satırı.             |
| 2  | `RUN`               | `RUN "dosya.basX"`      | Kaynağı derle & çalıştır. |
| 3  | `COMPILE`           | `COMPILE "src.basX"`    | Kaynağı bytecode’a çevir. |
| 4  | `LOAD BYTECODE`     | `LOAD BYTECODE "a.pdx"` | PDX yükle.                |
| 5  | `SAVE BYTECODE`     | `SAVE BYTECODE "a.pdx"` | Bytecode kaydet.          |
| 6  | `LIST`              | `LIST`                  | Yüklü betiği listeler.    |
| 7  | `CLEAR`             | `CLEAR`                 | Ortam sıfırlar.           |
| 8  | `IF`                | `IF expr THEN ...`      | Koşullu yürütme.          |
| 9  | `FOR` / `NEXT`      | `FOR i=...`             | Sayaç döngüsü.            |
| 10 | `WHILE` / `WEND`    | `WHILE expr`            | Koşullu döngü.            |
| 11 | `GOTO`              | `GOTO etiket`           | Satır atlama.             |
| 12 | `GOSUB` / `RETURN`  | `GOSUB label`           | Alt program.              |
| 13 | `STOP`              | `STOP`                  | Yürütmeyi durdurur.       |
| 14 | `END`               | `END`                   | Program sonu.             |
| 15 | `SUB` / `END SUB`   | Alt prosedür tanımı.    |                           |
| 16 | `FUNCTION`          | Fonksiyon tanımı.       |                           |
| 17 | `CLASS`             | Nesne tanımı.           |                           |
| 18 | `DIM`               | Dizi ayırma.            |                           |
| 19 | `EVENT`             | `EVENT slot,Handler`    | Slot bağlama.             |
| 20 | `TRIGGEREVENT`      | Tetikle.                |                           |
| 21 | `WAITEVENT`         | Bekle.                  |                           |
| 22 | `PIPE`              | `PIPE src→dst`          | Akış hattı.               |
| 23 | `MEMALLOC`          | Bellek tahsis.          |                           |
| 24 | `MEMFREE`           | Bellek serbest.         |                           |
| 25 | `PEEK` / `POKE`     | Low-level bellek.       |                           |
| 26 | `DBOPEN` etc.       | DB işlemleri.           |                           |
| 27 | `GRAPH NEW`         | Graf oluştur.           |                           |
| 28 | `GRAPH ADDNODE`     | Düğüm ekle.             |                           |
| 29 | `TREE BUILD`        | Ağaç oluştur.           |                           |
| 30 | `TIMER START`       | Zamanlayıcı.            |                           |
| 31 | `BACKTRACE`         | Son hatayı göster.      |                           |
| 32 | `IMPORT LIBX_<...>` | Uzantı yükle.           |                           |
| 33 | `SAVE STATE`        | Ortam kaydet.           |                           |
| 34 | `RESTORE`           | Ortam yükle.            |                           |
| 35 | `BYTECODE INFO`     | Bytecode meta.          |                           |
| 36 | `BYTECODE DUMP`     | Disasm dosya.           |                           |
| 37 | `VM PROFILE`        | Profil sayacı.          |                           |
| 38 | `EVENT MASK`        | Olay maskeleme.         |                           |
| 39 | `PIPE SPLIT`        | Akışı branşla.          |                           |
| 40 | `PIPE MERGE`        | Akış birleştir.         |                           |

*(Komut sözdizimleri detaylı açıklamaları 4. bölümde genişletilmiştir.)*

### 2.3 Yerleşik Fonksiyon Referansı (140 Fonksiyon – Tam Liste)

Fonksiyonları kategori başlıkları altında **tamamen** listeliyoruz. Her satır: `adı(sözdizimi) – görev`.

#### 2.3.1 Dize İşleme

| Fonksiyon | Sözdizimi            | Açıklama                                                                                                      | Örnek                             |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `LEN`     | `LEN(x)`             | Dize, dizi veya liste uzunluğunu döndürür (integer).                                                          | `PRINT LEN("ANKARA") '→ 6`        |
| `MID$`    | `MID$(s,start,len)`  | `s` dizgesinin `start` (1‑tabanlı) konumundan itibaren `len` karakterlik alt dizge döndürür.                  | `PRINT MID$("ANKARA",2,3) '→ NKA` |
| `LEFT$`   | `LEFT$(s,n)`         | Dizgenin en solundaki `n` karakteri döndürür.                                                                 | `LEFT$("Merhaba",3) '→ Mer`       |
| `RIGHT$`  | `RIGHT$(s,n)`        | Dizgenin en sağındaki `n` karakter.                                                                           | `RIGHT$("12345",2) '→ 45`         |
| `LTRIM$`  | `LTRIM$(s)`          | Başındaki boşlukları kırpar.                                                                                  | `LTRIM$("  a") '→ "a"`            |
| `RTRIM$`  | `RTRIM$(s)`          | Sonundaki boşlukları kırpar.                                                                                  | `RTRIM$("a  ")`                   |
| `INSTR`   | `INSTR(start,s,sub)` | `sub` alt dizgesinin `s` içinde `start` pozisyonundan itibaren ilk görüldüğü yerin 1‑tabanlı indisi, yoksa 0. | `INSTR(1,"bananas","na") '→ 3`    |
| `UCASE$`  | `UCASE$(s)`          | Tüm harfleri büyük yapar.                                                                                     | `UCASE$("abc") '→ ABC`            |
| `LCASE$`  | `LCASE$(s)`          | Tüm harfleri küçük yapar.                                                                                     | `LCASE$("İSTANBUL")`              |
| `STR$`    | `STR$(n)`            | Sayıyı dizeye çevirir.                                                                                        | `STR$(3.14) '→ "3.14"`            |
| `VAL`     | `VAL(s)`             | Dizgeyi sayıya çevirir (geçersizse 0).                                                                        | `VAL("42") '→ 42`                 |
| `ASC`     | `ASC(c)`             | Tek karakterin ASCII/UTF‑8 kodu.                                                                              | `ASC("A") '→ 65`                  |
| `CHR$`    | `CHR$(code)`         | ASCII kodu karaktere çevirir.                                                                                 | `CHR$(10) '→ LF`                  |

#### 2.3.2 Matematik & İstatistik

| Fonksiyon    | Sözdizimi               | Açıklama                              | Örnek              |
| ------------ | ----------------------- | ------------------------------------- | ------------------ |
| `ABS`        | `ABS(x)`                | Mutlak değer.                         | `ABS(-3) '→ 3`     |
| `SQR`        | `SQR(x)`                | Pozitif sayının karekökü.             | `SQR(9) '→ 3`      |
| `SIN`        | `SIN(rad)`              | Radyan cinsinden sinüs.               | `SIN(PI/2) '→ 1`   |
| `COS`        | `COS(rad)`              | Kosinüs.                              | `COS(0)`           |
| `TAN`        | `TAN(rad)`              | Tanjant.                              | `TAN(PI/4)`        |
| `ATN`        | `ATN(x)`                | Ark tanjant (arctan).                 | `ATN(1) '→ 0.785…` |
| `LOG`        | `LOG(x)`                | Doğal log (ln).                       | `LOG(E)`           |
| `EXP`        | `EXP(x)`                | eⁿ.                                   | `EXP(1)`           |
| `ROUND`      | `ROUND(x,n)`            | x’i n ondalıkla yuvarla.              | `ROUND(3.14159,2)` |
| `MOD`        | `MOD(a,b)`              | Kalan (a mod b).                      | `MOD(10,3) '→ 1`   |
| `MEAN`       | `MEAN(arr())`           | Ortalama.                             | `MEAN([1,2,3])`    |
| `STD`        | `STD(arr())`            | Standart sapma.                       | `STD([1,2,3])`     |
| `VAR`        | `VAR(arr())`            | Varyans.                              | `VAR([1,2,3])`     |
| `SUM`        | `SUM(arr())`            | Eleman toplamı.                       | `SUM([1,2,3])`     |
| `PROD`       | `PROD(arr())`           | Eleman çarpımı.                       | `PROD([2,3,4])`    |
| `REGRESS`    | `REGRESS(x(),y())`      | Doğrusal regresyon katsayı + p değer. | `REGRESS(xs,ys)`   |
| `ANOVA`      | `ANOVA(g1(),g2(),g3())` | Tek yönlü ANOVA F & p.                | `ANOVA(A,B,C)`     |
| `PERCENTILE` | `PERCENTILE(a(),q)`     | q (0-100) yüzdelik değeri.            | `PERCENTILE(A,90)` |
| `CORR`       | `CORR(x(),y())`         | Pearson korelasyonu.                  | `CORR(A,B)`        |

*(Diğer bilimsel fonksiyonlar Ek‑A’da tam liste, örnek ve formülle yer almaktadır.)*

Matematik & İstatistik
`ABS(x)`, `SQR(x)`, `SIN(x)`, `COS(x)`, `TAN(x)`, `ATN(x)`, `LOG(x)`, `EXP(x)`, `ROUND(x,n)`, `MOD(a,b)`, `MEAN(a())`, `STD(a())`, `VAR(a())`, `SUM(a())`, `PROD(a())`, `PERCENTILE(a(),q)`, `CORR(x(),y())`, `REGRESS(x(),y())`, `ANOVA(g1(),g2())`, …

#### 2.3.3 Tarih & Saat

| Fonksiyon | Sözdizimi | Açıklama                                                     | Örnek                           |
| --------- | --------- | ------------------------------------------------------------ | ------------------------------- |
| `DATE$`   | `DATE$()` | Geçerli tarihi `YYYY-MM-DD` biçiminde verir.                 | `PRINT DATE$() '→ "2025-05-22"` |
| `TIME$`   | `TIME$()` | Geçerli saati `HH:MM:SS` biçiminde verir.                    | `PRINT TIME$() '→ "14:32:05"`   |
| `TIMER`   | `TIMER()` | Program başından itibaren geçen saniyeyi ondalıklı döndürür. | `PRINT TIMER()`                 |

#### 2.3.4 Sistem & Bellek

| Fonksiyon   | Sözdizimi            | Açıklama                                                                | Örnek                        |
| ----------- | -------------------- | ----------------------------------------------------------------------- | ---------------------------- |
| `HEXDUMP$`  | `HEXDUMP$(addr,len)` | `addr` adresinden itibaren `len` baytı renkli HEX dize olarak döndürür. | `PRINT HEXDUMP$(&H1000, 16)` |
| `CPU_INFO$` | `CPU_INFO$()`        | JSON olarak CPU adı, mimari, çekirdek, SIMD özelliklerini verir.        | `PRINT CPU_INFO$()`          |
| `PORT_IN`   | `PORT_IN(addr)`      | x86 I/O portundan byte okur (donanım/OS destekliyse).                   | `PORT_IN(&H60)`              |
| `PORT_OUT`  | `PORT_OUT(addr,val)` | x86 I/O portuna byte yazar.                                             | `PORT_OUT(&H60, &HFF)`       |

> NOT: `PORT_IN/OUT` yalnızca `libx_lowlevel.py` etkin ve uygun erişim izni varsa çalışır.
> Sistem & Bellek
> `HEXDUMP$(addr,len)`, `CPU_INFO$()`, `PORT_IN(addr)` / `PORT_OUT(addr,val)`

#### 2.3.5 Veri Analizi, Makine Öğrenmesi ve Doğal Dil İşleme (ML / NLP)

Bu bölüm, `libx_data.py`, `libx_ml.py` ve `libx_nlp.py` uzantılarının sağladığı fonksiyonları içerir. Fonksiyonlar veriye dayalı modelleme, tahmin, metin çözümleme, kümeleme ve istatistiksel modelleme işlevleri sunar.

##### A) Veri Fonksiyonları (`libx_data`)

| Fonksiyon                     | Açıklama                                                                       | Örnek                                |
| ----------------------------- | ------------------------------------------------------------------------------ | ------------------------------------ |
| `DATA_LOAD(path$, fmt$)`      | CSV, JSON, Parquet gibi veri dosyalarını yükler. Geriye DataFrame ID’si döner. | `id$ = DATA_LOAD("data.csv", "csv")` |
| `DATA_FILTER(id$, expr$)`     | Koşullu filtre uygular, yeni alt-DataFrame döner.                              | `DATA_FILTER(id$, "val > 50")`       |
| `DATA_AGG(id$, method$)`      | `mean`, `sum`, `count` gibi özet fonksiyon.                                    | `DATA_AGG(id$, "mean")`              |
| `DATA_GROUP(id$, cols$)`      | Gruplama işlemi (`groupby`) için sütun tanımlanır.                             | `DATA_GROUP(id$, "region")`          |
| `DATA_JOIN(id1$, id2$, on$)`  | İki DataFrame’i `on` sütununda birleştirir.                                    | `DATA_JOIN(a$, b$, "id")`            |
| `DATA_HEAD$(id$, n)`          | İlk `n` satırı JSON dizisi olarak verir.                                       | `DATA_HEAD$(id$, 5)`                 |
| `DATA_SAVE(id$, path$, fmt$)` | CSV veya Parquet olarak veri kaydeder.                                         | `DATA_SAVE(id$, "out.csv", "csv")`   |

##### B) Makine Öğrenmesi Fonksiyonları (`libx_ml`)

| Fonksiyon                  | Açıklama                                                 | Örnek                              |
| -------------------------- | -------------------------------------------------------- | ---------------------------------- |
| `ML_TRAIN(model$, X$, y$)` | Gözetimli model (LR, SVM, RF) eğitir. Model adı verilir. | `ML_TRAIN("model1", X$, y$)`       |
| `ML_PREDICT(model$, X$)`   | Verilen model ile tahmin yapar.                          | `res$ = ML_PREDICT("model1", X$)`  |
| `ML_SCORE(model$, X$, y$)` | Doğruluk / AUC gibi metrikleri verir.                    | `ML_SCORE("model1", X$, y$)`       |
| `ML_EXPORT(model$, path$)` | Eğitilen modeli `.pkl` olarak kaydeder.                  | `ML_EXPORT("model1", "model.pkl")` |
| `ML_IMPORT(path$)`         | Dış model dosyasını yükler.                              | `ML_IMPORT("model.pkl")`           |
| `ML_CLUSTER(X$, k)`        | K-means kümeleme uygular (`k` kümeye ayırır).            | `ML_CLUSTER(data$, 3)`             |

##### C) Doğal Dil İşleme Fonksiyonları (`libx_nlp`)

| Fonksiyon                       | Açıklama                                                  | Örnek                                   |
| ------------------------------- | --------------------------------------------------------- | --------------------------------------- |
| `NLP_TOKENIZE(text$, lang$)`    | `lang$ = "tr"` veya `"en"`. Metni kelime dizisine ayırır. | `NLP_TOKENIZE("Merhaba dünya","tr")`    |
| `NLP_SENTIMENT(text$)`          | Duygu analizi: pozitif, nötr, negatif + skor.             | `NLP_SENTIMENT("Bu harika!")`           |
| `NLP_TAG(text$)`                | Kelimelere sözcük türü (POS) etiketi verir.               | `NLP_TAG("Ali koşuyor")`                |
| `NLP_SUMMARY(text$, sentcount)` | Belirtilen cümle sayısında özet döner.                    | `NLP_SUMMARY(bigtext$, 2)`              |
| `NLP_TOPICS(text$, k)`          | Anahtar konu/etiket çıkarımı (`k` adet).                  | `NLP_TOPICS(text$, 3)`                  |
| `NLP_VECTORIZE(text$)`          | Vektör temsili (`TF-IDF`, `Word2Vec`, model bazlı).       | `vec$ = NLP_VECTORIZE("Akıllı sistem")` |

##### D) Uygulama Örneği – Veri + Model + NLP Zinciri

```basic
csv$ = "veri.csv"
id$ = DATA_LOAD(csv$, "csv")
id2$ = DATA_FILTER(id$, "puan > 75")
X$ = DATA_HEAD$(id2$, 5)
CALL NLP_SENTIMENT("Metin çok iyi yazılmış.")
CALL ML_CLUSTER(X$, 3)
```

Bu örnek veri yüklüyor, filtreliyor, başlığı okuyor, duygu analizi yapıyor ve kümeliyor.

##### E) Gelişmiş Kullanım

* `DATA_LOAD` sonrası `DATA_GROUP` → `DATA_AGG` zincirlenebilir.
* NLP çıktıları (`VECTORIZE`) ML modellerine girdi olur.

---

## 6 · Grafik Veri Yapıları ve Algoritmaları <a name="graph"></a>

Bu bölüm, `graph.py` ve `GraphManager` modülünün sunduğu grafik (graph) yapılarını, komutlarını ve algoritmalarını açıklar. Bu sistemde yönlü/yönsüz grafikler oluşturulabilir, düğümler ve kenarlar eklenebilir, grafik üzerinde algoritmalar (Dijkstra, Bellman-Ford, DFS, BFS, GNN) çalıştırılabilir.

### 6.1 Temel Komutlar

| Komut                         | Sözdizimi                                   | Açıklama                                       |
| ----------------------------- | ------------------------------------------- | ---------------------------------------------- |
| `GRAPH CREATE`                | `GRAPH CREATE id$ AS DIRECTED value$ var$`  | Yeni bir yönlü/yönsüz grafik oluşturur.        |
| `GRAPH ADD VERTEX`            | `GRAPH ADD VERTEX id$ value$ var$`          | Grafiğe yeni düğüm ekler.                      |
| `GRAPH ADD EDGE`              | `GRAPH ADD EDGE id$ src$ dst$ weight?`      | Grafiğe kenar ekler (varsayılan ağırlık: 1.0). |
| `GRAPH REMOVE VERTEX`         | `GRAPH REMOVE VERTEX id$ vertex$`           | Belirtilen düğümü siler.                       |
| `GRAPH REMOVE EDGE`           | `GRAPH REMOVE EDGE id$ src$ dst$`           | İki düğüm arasındaki kenarı siler.             |
| `GRAPH TRAVERSE`              | `GRAPH TRAVERSE id$ mode$ start$ result$`   | `BFS` veya `DFS` modunda gezinti.              |
| `GRAPH SHORTEST PATH`         | `GRAPH SHORTEST PATH id$ src$ dst$ result$` | Dijkstra algoritması ile yol bulur.            |
| `GRAPH BELLMAN FORD`          | `GRAPH BELLMAN FORD id$ start$ result$`     | Negatif ağırlıklı yol için.                    |
| `GRAPH FLOYD WARSHALL`        | `GRAPH FLOYD WARSHALL id$ result$`          | Tüm çiftler için en kısa yollar.               |
| `GRAPH GNN`                   | `GRAPH GNN id$ dim agg$ [weights] result$`  | Özellik vektörleri üzerinden GNN geçişi.       |
| `GRAPH MINIMUM SPANNING TREE` | `GRAPH MINIMUM SPANNING TREE id$ result$`   | Kruskal ile minimum yayılma ağacı.             |
| `GRAPH TOPOLOGICAL SORT`      | `GRAPH TOPOLOGICAL SORT id$ result$`        | DAG üzerinde sıralama.                         |
| `GRAPH VISUALIZE`             | `GRAPH VISUALIZE id$ "path" format$`        | PNG, SVG gibi görsel çıktı.                    |

### 6.2 Fonksiyonlar

| Fonksiyon           | Açıklama                  |
| ------------------- | ------------------------- |
| `GRAPH NODES$(id$)` | Tüm düğümleri verir.      |
| `GRAPH EDGES$(id$)` | Kenar listesini döndürür. |
| `GRAPH RESULT$()`   | Son algoritmanın çıktısı. |

### 6.3 Kullanım Örneği — Kısa Yol ve Görselleştirme

```basic
GRAPH CREATE G1 AS DIRECTED "Start" v1$
GRAPH ADD VERTEX G1 "Middle" v2$
GRAPH ADD VERTEX G1 "End" v3$
GRAPH ADD EDGE G1 v1$ v2$ 3
GRAPH ADD EDGE G1 v2$ v3$ 5
GRAPH SHORTEST PATH G1 v1$ v3$ yol$
PRINT GRAPH RESULT$()
GRAPH VISUALIZE G1 "graph_output" PNG
```

### 6.4 GNN Uygulaması

```basic
GRAPH GNN G1 4 SUM [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] gnnres$
PRINT GRAPH RESULT$()
```

### 6.5 Alıştırmalar

1. `GRAPH TRAVERSE` ile DFS ve BFS farklarını test edin.
2. GNN kullanarak düğüm sınıflandırmasını deneyin (`features` ile).
3. `GRAPH TOPOLOGICAL SORT` sonucunu DAG grafiği üzerinde yorumlayın.

---

## 7 · PDSX Ana Yorumlayıcı (pdsXuv14.py) <a name="interpreter"></a>

### 7.1 Genel Yapı

Ana yorumlayıcı sınıfı `PdsXv14uInterpreter`, tüm alt sistemleri bağlar ve komutları çalıştırır:

* `parse_statement()` → satırı komut veya ifadeye çevirir
* `execute_line()` → komutu yürütür
* `eval_expression()` → matematiksel/lojik ifade değerlendirir
* `command_table` → tüm PDS komutları (RUN, IF, FOR, GOTO…)
* `function_table` → tüm yerleşik fonksiyonlar (LEN, MID\$, SQR, DATE\$…)

Yorumlayıcı, tüm sistemin omurgasıdır. Her komut buradan geçer.

### 7.2 Yüklü Modüller

| Alt Sistem      | Sınıf                                 | Görev                                      |
| --------------- | ------------------------------------- | ------------------------------------------ |
| Bellek          | `MemoryManager`                       | `MEMALLOC`, `MEMFREE`, `PEEK`...           |
| Boru Hattı      | `PipeManager`                         | `PIPE`, `PIPE SPLIT`...                    |
| Olay Sistemi    | `EventManager`                        | `EVENT`, `WAITEVENT`...                    |
| Ağaç & Grafik   | `TreeManager`, `GraphManager`         | `TREE BUILD`, `GRAPH CREATE`...            |
| Veri / ML / NLP | `LibXData`, `LibXML`, `LibXNLP`       | `DATA_LOAD`, `ML_TRAIN`, `NLP_TOKENIZE`... |
| GUI             | `LibXGui`                             | Arayüz oluşturma, görsel çıkışlar.         |
| DB              | `LibDB`, `SQLiteManager`              | `DBOPEN`, `DBQUERY`                        |
| Derleyici       | `BytecodeCompiler`, `BytecodeManager` | `COMPILE`, `BYTECODE INFO`                 |
| Düşük Seviye    | `LowLevelManager`                     | `PORT_IN`, `PORT_OUT`                      |
| Zamanlayıcı     | `TimerManager`                        | `TIMER START`, `TICK$`                     |
| Geri İzleme     | `BacktraceLogger`                     | `BACKTRACE`, `DEBUG TRACE`                 |

### 7.3 Komut Haritası (command\_table)

`command_table`, 40+ komutu anahtar-string → fonksiyon eşlemesi olarak tutar. Her satırın başı bu tabloyla karşılaştırılır.

| Komut                                                  | Fonksiyon         | Açıklama                      |
| ------------------------------------------------------ | ----------------- | ----------------------------- |
| `RUN`                                                  | `cmd_run()`       | Programı çalıştırır           |
| `IF`                                                   | `cmd_if()`        | Koşullu dallanma              |
| `FOR`                                                  | `cmd_for()`       | Sayaç döngüsü                 |
| `WHILE`                                                | `cmd_while()`     | Koşullu döngü                 |
| `SUB`                                                  | `cmd_sub()`       | Alt prosedür                  |
| `FUNCTION`                                             | `cmd_function()`  | Fonksiyon bildirimi           |
| `DIM`                                                  | `cmd_dim()`       | Dizi bildirimi                |
| `PRINT`                                                | `cmd_print()`     | Konsola yazdır                |
| `INPUT`                                                | `cmd_input()`     | Girdi al                      |
| `IMPORT`                                               | `cmd_import()`    | Kütüphane yükle               |
| `TREE`, `GRAPH`, `PIPE`, `EVENT`, `TIMER`, `BACKTRACE` | `cmd_x_manager()` | Modül tarafından ayrıştırılır |

> Komutlar `parse_statement` ile ayrıştırılır ve `execute_line` üzerinden uygun `cmd_*` işlevi çalıştırılır.

### 7.4 Fonksiyon Haritası (function\_table)

`function_table`, yerleşik fonksiyonları (örn. `LEN`, `SQR`, `DATE$`) eşler. `evaluate_expression()` fonksiyonu bu tabloyu kullanır.

* Her fonksiyon `"LEN": self.fn_len` gibi tanımlanır.
* 140+ fonksiyon kategorilere göre ayrılmıştır (2.3 bölümünde açıklanmıştır).

### 7.5 Yorumlayıcı Akışı

1. Satır `REPL`’de girilir veya dosyadan okunur.
2. `parse_statement` çağrılır.
3. `command_table` veya `function_table` eşleşmesi aranır.
4. İlgili `cmd_*` veya `fn_*` metodu yürütülür.
5. Durum (scope, stack, timers…) güncellenir.

### 7.6 Örnek – Karma Komut Zinciri

```basic
IMPORT LIBX_DATA
DATA_LOAD "data.csv", "csv" res$
IF LEN(res$)>0 THEN
    PIPE Sensor → Alarm
    TIMER START t$,1,1,TRIG30
    SUB TRIG30() : TRIGGEREVENT 30 : END SUB
END IF
```

---

### 7.7 Komutların Yürütme Mantığı (Scope, Stack, Jump)

PDSX yorumlayıcısında her komut çalıştırılırken kontrol akışı aşağıdaki yapılara dayanır:

#### Scope (Kapsam)

* `self.stack` → çağrı yığını (`GOSUB`, `SUB`, `FUNCTION` çağrılarında kullanılır)
* `self.current_scope()` → geçici sözlük (`DIM`, `LET`, `INPUT`, vs.)
* Değişkenler `{name: value}` yapısında tutulur

#### Jump Sistemi

* `self.line_map` → satır numaraları ile eşleşen konumlar
* `self.goto(label)` → doğrudan satır atlama
* `self.sub_map`, `self.func_map` → prosedür/fonksiyon yerleri

#### CALL ve RETURN

```basic
SUB Merhaba()
  PRINT "Selam"
END SUB
CALL Merhaba()
```

Alt prosedür çağrıldığında `self.stack.append(current_line)` yapılır ve `RETURN` komutu bu satıra geri döner.

---

### 7.8 Hata İzleme & Backtrace Sistemi

* `BacktraceLogger` (`f11_backtrace_logger.py`) sınıfı sayesinde her hata çağrı zinciriyle birlikte loglanır
* `BACKTRACE` komutu ile en son hata detaylıca dökülür
* Her hata `PdsXException` sınıfı üzerinden işlenir

```basic
x = 5 / 0
BACKTRACE
```

Bu örnekte, sıfıra bölme hatası loga işlenir, sonra `BACKTRACE` ile analiz edilir.

---

### 7.9 REPL ve Dosya Modları

Yorumlayıcı şu modlarda çalışabilir:

| Mod              | Açıklama                                                                        |
| ---------------- | ------------------------------------------------------------------------------- |
| REPL             | Satır satır interaktif mod (CLI)                                                |
| Kaynak Modu      | `.basX` dosyaları tek seferde yüklenip çalıştırılır                             |
| Bytecode Modu    | `.pdx` bytecode dosyaları çalıştırılır                                          |
| JSON/Script Modu | `PIPE`, `TREE`, `GRAPH`, `NLP` gibi sistemlerle JSON & yapılandırma desteklenir |

REPL örneği:

```
> LET A = 10
> PRINT A*2
20
```

Dosya modunda:

```python
interpreter.load_source("myprogram.basX")
interpreter.run()
```

---

## 8 · Grafik Arayüz Sistemi (GUI – libx\_gui.py) <a name="gui"></a>

### 8.1 Giriş

PDS-X GUI sistemi, `libx_gui.py` modülü ile `tkinter` tabanlı pencere, buton ve etiket destekli basit grafik arayüzler üretmeyi sağlar. Arayüzler `WindowManager` sınıfı tarafından yönetilir.

### 8.2 Pencere Oluşturma

| Komut                                     | Açıklama                | Örnek                                  |
| ----------------------------------------- | ----------------------- | -------------------------------------- |
| `GUI CREATE name$, width, height, title$` | Yeni pencere oluşturur. | `GUI CREATE "ANA", 400, 300, "Başlık"` |

Bu komut, arkaplanda şu sınıf metodunu çağırır:

```python
WindowManager.create(name="ANA", width=400, height=300, title="Başlık")
```

### 8.3 Widget Ekleme

| Komut                                  | Açıklama      | Örnek                                        |
| -------------------------------------- | ------------- | -------------------------------------------- |
| `GUI BUTTON win$, btn$, "Metin", x, y` | Buton ekler.  | `GUI BUTTON "ANA", "BTN1", "Tıkla", 50, 20`  |
| `GUI LABEL win$, lbl$, "Metin", x, y`  | Etiket ekler. | `GUI LABEL "ANA", "LBL1", "Merhaba", 20, 80` |

Kod karşılıkları:

```python
WindowManager.add_button("ANA", "BTN1", "Tıkla", 50, 20)
WindowManager.add_label("ANA", "LBL1", "Merhaba", 20, 80)
```

### 8.4 Pencereyi Gösterme ve Çalıştırma

GUI pencereleri arkaplanda `Tk()` ana kökü ile başlatılır. `Tk.mainloop()` çağrısı otomatik yapılır.

Komutlar yorumlayıcıdan geçtikçe pencere oluşur ve aktif olur. REPL modunda arayüzler anında görünür.

### 8.5 Hata Yönetimi

Her işlem `try-except` bloğu ile korunur. GUI bileşeni mevcut değilse veya konumlandırma hatası varsa `PdsXException` fırlatılır ve log’a yazılır.

### 8.6 Genişletme Alanları (Planlanan)

* `GUI INPUT` → Girdi kutuları
* `GUI DRAW` → Tuval üzerine çizim
* `GUI TABLE` → Tablo nesnesi
* `GUI PLOT` → Grafik çizimi (veri ile)
* `GUI EVENT` → Butona tıklama, pencere kapama, zamanlayıcı tetikleme gibi olay bağlama

### 8.7 Tam Örnek – Basit Arayüz

Aşağıdaki örnekte basit bir pencere oluşturulmakta, bir metin etiketi ve bir buton eklenmektedir. Program çalıştığında pencere açılır ve kullanıcı butona tıklayabilir.

#### Örnek: Pencere + Etiket + Buton

```basic
GUI CREATE "ANA", 320, 240, "Deneme Arayüz"
GUI LABEL "ANA", "L1", "Merhaba Dünya", 40, 50
GUI BUTTON "ANA", "B1", "Kapat", 100, 120
```

Yukarıdaki örnek çalıştırıldığında 320x240 boyutunda bir pencere açılır. İçerisinde "Merhaba Dünya" yazan bir etiket ve "Kapat" adlı bir buton yer alır. Henüz olay bağlama (`GUI EVENT`) sistemi devrede değilse buton etkileşimi sınırlı olabilir.

---

#### Ek Örnek 1: Çoklu Bileşen Yerleşimi

```basic
GUI CREATE "P1", 400, 300, "Form"
GUI LABEL "P1", "isimLbl", "İsim:", 30, 30
GUI LABEL "P1", "yasLbl", "Yaş:", 30, 70
GUI BUTTON "P1", "btnGonder", "Gönder", 150, 120
```

Bu örnekte form tarzında yerleştirilmiş etiketler ve bir gönder butonu görülmektedir. `GUI INPUT` komutu henüz mevcut değilse giriş alanları sadece temsilîdir.

---

```basic
GUI CREATE "ANA", 320, 240, "Deneme"
GUI LABEL "ANA", "L1", "Merhaba Dünya", 40, 50
GUI BUTTON "ANA", "B1", "Kapat", 100, 120
```

---

## 9 · PIPE İzleme ve GUI Yönetimi (pipe\_monitor\_gui.py) <a name="pipe_gui"></a>

### 9.1 Giriş

Bu modül, PIPE sisteminin veri akışını izlemenizi, bağlantıları GUI ile denetlemenizi sağlar. WebSocket, dosya, named pipe gibi farklı kanalları destekler.

### 9.2 PIPE Bağlantı Tipleri

| Tip          | Açıklama                                               |
| ------------ | ------------------------------------------------------ |
| `websocket`  | Asenkron WebSocket sunucusuna bağlanır.                |
| `file`       | Dosyaya veri yazar/okur (binary modda).                |
| `named_pipe` | Windows/Linux uyumlu adlandırılmış kanalları kullanır. |

#### Örnek:

```basic
PIPE CONNECT "P1" TYPE "file" PATH "logdata.dat"
PIPE WRITE "P1", "Sıcaklık=25"
```

### 9.3 Temel İşlevler (PipeConnection)

| Metot                    | Açıklama                                           |
| ------------------------ | -------------------------------------------------- |
| `connect()`              | Belirtilen kanal tipine göre bağlantı kurar.       |
| `write(data, compress?)` | Veriyi yazar; JSON + opsiyonel sıkıştırma uygular. |
| `read()`                 | Kanaldan veri okur.                                |
| `close()`                | Bağlantıyı kapatır.                                |

Desteklenen sıkıştırmalar: `gzip`, `zlib`, `base64`, `none`

### 9.4 GUI Destekli İzleme Paneli (Planlanan Yapı)

Kütüphane, tkinter tabanlı GUI ile boru bağlantılarını görselleştirir:

* Her `pipe_id` bir widget olarak temsil edilir.
* Kanal türü, durumu, tampon doluluğu gibi bilgiler ekranda canlı olarak gösterilir.
* Gelecekte `matplotlib` ile veri grafikleri, `graphviz` ile akış şeması çıkışı sağlanabilir.

#### Örnek Kullanım Akışı (teorik)

```basic
PIPE CONNECT "T1" TYPE "websocket" URL "ws://localhost:8000"
PIPE MONITOR "T1"
PIPE STATS$ "T1" res$
PRINT res$
```

### 9.5 Teknik Özellikler

* Thread-safe erişim için `@synchronized` dekoratörü kullanılır.
* `deque()` ile buffer sistemi mevcuttur.
* `multiprocessing` desteğiyle aynı anda çok sayıda kanal işlenebilir.
* `pdsx_exception` ile sağlam hata yönetimi sağlanır.

---

## 10 · PIPE Sistemi – Boru Hattı Yönetimi <a name="pipe"></a>

PIPE sistemi, modüller arası veri akışını ve işlemler zincirini yönetir. `pipe.py` dosyasında tanımlıdır ve aşağıdaki bileşenleri içerir:

### 10.1 PIPE Komutları

| Komut          | Açıklama                               | Sözdizimi                                   |
| -------------- | -------------------------------------- | ------------------------------------------- |
| `PIPE CREATE`  | Yeni pipe tanımlar                     | `PIPE CREATE id$`                           |
| `PIPE CONNECT` | Pipe’a kaynak atar                     | `PIPE CONNECT id$ TYPE "file" PATH "a.dat"` |
| `PIPE WRITE`   | Pipe’a veri yazar                      | `PIPE WRITE id$, "Veri"`                    |
| `PIPE READ`    | Pipe’tan veri okur                     | `PIPE READ id$, var$`                       |
| `PIPE CLOSE`   | Pipe’ı kapatır                         | `PIPE CLOSE id$`                            |
| `PIPE SPLIT`   | Birden fazla hedefe yönlendirir        | `PIPE SPLIT id$ TO id2$, id3$`              |
| `PIPE MERGE`   | Çeşitli kaynaklardan tek pipe’a toplar | `PIPE MERGE id$, id1$, id2$`                |
| `PIPE STATUS$` | Pipe’ın durumu (JSON)                  | `PIPE STATUS$ id$`                          |

### 10.2 Fonksiyonlar (PipeManager)

| Fonksiyon             | Açıklama                  |
| --------------------- | ------------------------- |
| `write(data)`         | Pipe’a veri gönderir      |
| `read()`              | Pipe’tan veri çeker       |
| `connect(type, path)` | Fiziksel kaynakla eşleşir |
| `disconnect()`        | Bağlantıyı koparır        |

Pipe sisteminde her `pipe_id` bir nesneye karşılık gelir. Bu nesneler `PipeManager` tarafından saklanır.

### 10.3 PIPE Kullanım Örnekleri

#### Örnek 1: Dosya Üzerinden Veri Aktarımı

```basic
PIPE CREATE "P1"
PIPE CONNECT "P1" TYPE "file" PATH "veriler.dat"
PIPE WRITE "P1", "Sıcaklık=23.5"
PIPE CLOSE "P1"
```

#### Örnek 2: WebSocket ile Canlı Yayın

```basic
PIPE CREATE "W1"
PIPE CONNECT "W1" TYPE "websocket" URL "ws://localhost:9000"
PIPE WRITE "W1", "Merhaba WebSocket"
PIPE CLOSE "W1"
```

#### Örnek 3: Çok Kaynaktan Tek Pipe

```basic
PIPE CREATE "P2"
PIPE MERGE "P2", "SENSOR1", "SENSOR2"
```

#### Örnek 4: Tek Kaynaktan Çoklu Hedef

```basic
PIPE CREATE "P3"
PIPE SPLIT "P3" TO "MONITOR", "LOGGER"
```

---

### 10.4 PIPE Sistem Mimarisi

* Tüm `pipe_id`’ler merkezi `pipe_table` sözlüğünde tutulur.
* Her pipe bağımsız olarak veri buffer’ı, sıkıştırma biçimi ve kanal tipine sahiptir.
* Sıkıştırma destekleri: `none`, `gzip`, `zlib`, `base64`
* Pipe’lar `thread-safe` biçimde veri aktarır. Her işlem kilitlenebilir yapıdadır (`Lock`).

---

### 10.5 Alıştırmalar

1. 3 farklı sensör verisini `PIPE MERGE` ile birleştirip tek dosyaya yazın.
2. Bir pipe üzerinden 10 adet JSON mesaj gönderin, `PIPE READ` ile okuyun.
3. `PIPE STATUS$` ile aktif pipe’ların yapılarını görüntüleyin.

---

## 11 · EVENT Sistemi – Olay ve Kesme Yönetimi <a name="event"></a>

### 11.1 Giriş

PDS-X yorumlayıcısının `event.py` modülü, zamanlayıcılar, dış tetikleyiciler ve kullanıcı olayları gibi işlemleri yönetmek için olay tabanlı bir sistem sunar. Bu sistem, çoklu işleyici, önceliklendirme, zamanlama ve sinyal işleme gibi özelliklerle birlikte gelir.

### 11.2 EVENT Komutları

| Komut          | Açıklama                                                | Sözdizimi                                  |
| -------------- | ------------------------------------------------------- | ------------------------------------------ |
| `ON EVENT`     | Belirli bir olay gerçekleştiğinde altprogramı çağırır.  | `ON EVENT "sensor" CALL AlarmAç`           |
| `TRIGGEREVENT` | Bir olayı tetikler (manuel).                            | `TRIGGEREVENT "sensor"`                    |
| `WAITEVENT`    | Belirli bir olayın oluşmasını bekler.                   | `WAITEVENT "user_ok"`                      |
| `TIMER START`  | Zamanlayıcı başlatır ve belirli süre sonra olay üretir. | `TIMER START id$, delay, repeat, handler$` |
| `TIMER STOP`   | Belirtilen zamanlayıcıyı durdurur.                      | `TIMER STOP id$`                           |

### 11.3 Fonksiyonlar (EventManager)

| Fonksiyon                                  | Açıklama                                               |
| ------------------------------------------ | ------------------------------------------------------ |
| `register_event(name, handler, priority)`  | Olayı ve işleyicisini kaydeder.                        |
| `trigger_event(name)`                      | Olayı tetikler.                                        |
| `wait_for_event(name)`                     | Bloklayıcı bekleme sağlar.                             |
| `add_timer(id, interval, repeat, handler)` | Belirtilen aralıkla olay tetikleyen zamanlayıcı kurar. |

### 11.4 EVENT Kullanım Örnekleri

#### Örnek 1: Kullanıcı Olayı

```basic
ON EVENT "KAPI_ACILDI" CALL AlarmCal
TRIGGEREVENT "KAPI_ACILDI"
SUB AlarmCal()
  PRINT "Hırsız alarmı çalıştı!"
END SUB
```

#### Örnek 2: Zamanlayıcı ile Otomatik Olay

```basic
TIMER START "z1", 5, 1, ZilCal
SUB ZilCal()
  PRINT "5 saniyede bir çalışıyorum"
END SUB
```

#### Örnek 3: Olay Bekleyici

```basic
PRINT "Onay için kullanıcıdan OK sinyali bekleniyor..."
WAITEVENT "KULLANICI_OK"
PRINT "Kullanıcı onay verdi."
```

---

### 11.5 Teknik Detaylar

* Tüm olaylar `deque()` ile kuyruklanır.
* `priority` parametresiyle çok sayıda işleyici sıralanabilir.
* Zamanlayıcılar arkaplanda `asyncio` + `threading` ile çalışır.
* `signal`, `psutil` entegrasyonu ile gerçek sistem olayları da bağlanabilir (örneğin CPU %90 → `ON EVENT "YUKSEK_CPU"`).

---

### 11.6 Alıştırmalar

1. Her 3 saniyede bir rastgele sayı üretip pipe’a yazan bir zamanlayıcı kurun.
2. Kullanıcı bir tuşa basınca tetiklenen bir `TRIGGEREVENT` yazın.
3. Öncelikli olay işleyicilerini test edin: `ON EVENT "acil" CALL X`, `ON EVENT "acil" CALL Y, PRIORITY 1`.

---

## 12 · TREE Sistemi – Ağaç Veri Yapıları <a name="tree"></a>

`tree3.py` modülü, PDS-X içerisinde çeşitli ağaç yapılarını destekler. Genel ağaçlar, ikili ağaçlar, kırmızı-siyah ağaçlar ve B-ağaçları gibi farklı türler bu sistemde tanımlanmıştır.

### 12.1 Sınıflar

#### TreeNode – Genel Ağaç Düğümü

```python
node = TreeNode("Veri")
node.add_child(child_node)
```

* `value`: Düğümdeki veri.
* `children`: Alt düğümler listesi.
* `metadata`: Ek bilgiler.

#### BinaryTreeNode – İkili Ağaç

```python
root = BinaryTreeNode("Kök")
root.add_child(left_node)
root.add_child(right_node)
```

* En fazla 2 çocuk düğüm.
* `balance_factor`: AVL için kullanılabilir.

#### RedBlackNode – Kırmızı Siyah Ağaç

```python
n = RedBlackNode("Veri")
n.color = "RED" or "BLACK"
```

* `color`: Renk kontrolü (denge).
* `nil`: Yaprak dışı düğüm işaretçisi.

#### BTreeNode – B-Ağacı

```python
n = BTreeNode(degree=3)
n.keys.append("A")
n.children.append(child_node)
```

* `degree`: Minimum çocuk sayısı (t)
* `keys`: Anahtarlar
* `children`: Alt dallar
* `leaf`: Yaprak mı?

---

### 12.2 Komutlar (Planlanmış Kullanım)

| Komut              | Açıklama                         | Sözdizimi                                 |
| ------------------ | -------------------------------- | ----------------------------------------- |
| `TREE BUILD`       | Yeni genel ağaç oluşturur        | `TREE BUILD id$ AS GENERAL value$ var$`   |
| `TREE ADD NODE`    | Düğüm ekler                      | `TREE ADD NODE id$, parent$, value$ var$` |
| `TREE REMOVE NODE` | Düğüm siler                      | `TREE REMOVE NODE id$, node$`             |
| `TREE SET NIL`     | Kırmızı-siyah ağaçta nil ayarı   | `TREE SET NIL id$, node$`                 |
| `TREE BALANCE`     | AVL dengeleme / sıralama uygular | `TREE BALANCE id$`                        |
| `TREE KEYS$`       | Anahtarları listeler             | `TREE KEYS$ id$`                          |

---

### 12.3 Örnekler

#### Örnek 1: Genel Ağaç Yapısı

```basic
TREE BUILD "T1" AS GENERAL "Kök" root$
TREE ADD NODE "T1", root$, "Alt1" n1$
TREE ADD NODE "T1", root$, "Alt2" n2$
```

#### Örnek 2: İkili Ağaç

```basic
TREE BUILD "T2" AS BINARY "Root" r$
TREE ADD NODE "T2", r$, "Sol" left$
TREE ADD NODE "T2", r$, "Sağ" right$
```

#### Örnek 3: Kırmızı-Siyah Ağaç

```basic
TREE BUILD "T3" AS RBTREE "X" x$
TREE ADD NODE "T3", x$, "Y" y$
TREE SET NIL "T3", y$
```

---

### 12.4 Fonksiyonlar (Yapısal İşlevler)

| Fonksiyon        | Açıklama                                |
| ---------------- | --------------------------------------- |
| `add_child()`    | Yeni alt düğüm ekler                    |
| `remove_child()` | Belirli alt düğümü siler                |
| `is_full()`      | BTreeNode için dolu kontrolü            |
| `set_nil()`      | RedBlackNode’da nil göstergesi tanımlar |
| `balance_factor` | AVL yapılar için denge ölçer            |

---

### 12.5 Teknik Notlar

* Her düğüm UUID ile benzersiz ID alır.
* `graphviz` desteklidir, `visualize()` ile çizim yapılabilir.
* Düğümler arası ilişki `parent` ve `children` ile kurulmuştur.
* BTreeNode içinde anahtarlar sıralı tutulur.

---

### 12.6 Alıştırmalar

1. 3 seviyeli genel ağaç inşa edin ve her düğümün değerini `PRINT` ile gösterin.
2. AVL yapısında dengesiz yapı kurup `TREE BALANCE` uygulayın.
3. `BTreeNode` sınıfını kullanarak anahtar yerleştirme simülasyonu yapın.

---

## 13 · DATA Sistemi – Veri Yükleme, Dönüştürme ve Analiz <a name="data"></a>

`libx_data.py` modülü, PDS-X BASIC içerisinde veri setlerini yüklemek, analiz etmek, filtrelemek ve pipe tabanlı veri akışlarına bağlamak için kullanılır. Ana yapı `PipelineInstance` sınıfıdır.

### 13.1 Temel Yapılar

#### PipelineInstance

Bu sınıf, bir dizi adımı (komutu) sırayla veya paralel olarak çalıştırmak için kullanılır.

* `commands`: İşlenecek veri adımları listesi
* `status`: Hangi adımlar işlendi, hangileri beklemede
* `execute()`: Tüm adımları sırayla veya paralel çalıştırır

### 13.2 Komutlar (DATA Komutları)

| Komut            | Açıklama                              | Sözdizimi                                          |
| ---------------- | ------------------------------------- | -------------------------------------------------- |
| `DATA LOAD`      | CSV, JSON, Excel gibi verileri yükler | `DATA LOAD "veri.csv", "csv", result$`             |
| `DATA FILTER`    | Şarta göre satırları filtreler        | `DATA FILTER result$, "sutun > 10", filtrelenmis$` |
| `DATA SELECT`    | Belirli sütunları seçer               | `DATA SELECT result$, "yas, gelir", secilen$`      |
| `DATA STATS`     | İstatistik hesaplar (ortalama, std)   | `DATA STATS result$, "gelir", istatistik$`         |
| `DATA PLOT`      | Veriyi çizer (planlı)                 | `DATA PLOT result$, "yas", "gelir"`                |
| `DATA TRANSFORM` | Normalize, ölçekleme, log alma        | `DATA TRANSFORM result$, "normalize"`              |

### 13.3 Örnekler

#### Örnek 1: CSV Dosyası Yükleme

```basic
DATA LOAD "ornek.csv", "csv", veri$
PRINT veri$
```

#### Örnek 2: Filtreleme ve Seçim

```basic
DATA FILTER veri$, "gelir > 5000", yuksekGelir$
DATA SELECT yuksekGelir$, "isim, gelir", rapor$
```

#### Örnek 3: İstatistik ve Dönüşüm

```basic
DATA STATS veri$, "yas", istatistik$
DATA TRANSFORM veri$, "log"
```

#### Örnek 4: Pipe Entegrasyonu

```basic
PIPE CREATE "D1"
PIPE CONNECT "D1" TYPE "file" PATH "log.csv"
PIPE WRITE "D1", veri$
```

---

### 13.4 PipelineInstance Metotları

| Metot                                   | Açıklama                               |
| --------------------------------------- | -------------------------------------- |
| `add_command(cmd, step_no?, position?)` | İşlem adımı ekler (baş/son/araya)      |
| `remove_command(step_no)`               | Adımı çıkarır                          |
| `execute(parallel=False)`               | Zinciri sırayla veya eşzamanlı yürütür |

Bu yapı sayesinde PDS-X içinde hem veri işleme zincirleri hem de pipe-temelli görevler düzenli bir yapı içinde kontrol edilebilir.

---

### 13.5 Alıştırmalar

1. `data.csv` adlı dosyayı yükleyin, "yaş" > 40 olanları filtreleyin, sadece "isim" ve "yaş" sütunlarını seçin.
2. Veriyi normalize edip PIPE’a yazın, ardından PIPE’tan geri okuyun.
3. Bir dataset üzerinde `DATA STATS` ile tüm sayısal sütunların ortalama ve standart sapmalarını hesaplayın.

---

## 14 · NLP Sistemi – Doğal Dil İşleme Modülü <a name="nlp"></a>

`libx_nlp.py` modülü, PDS-X içerisinde çok dilli doğal dil işleme (NLP) görevlerini gerçekleştirmek için kullanılır. Bu sistem dört büyük motoru entegre eder:

* `spaCy` → sözcük analizi, sözdizimsel ayrıştırma
* `NLTK` → tokenizasyon, etiketleme
* `TextBlob` → duygu analizi, çeviri
* `transformers` → modern modellerle özetleme, sınıflandırma, çeviri

### 14.1 Desteklenen İşlemler

| Görev       | Açıklama                            | Örnek Komut                                |
| ----------- | ----------------------------------- | ------------------------------------------ |
| `TOKENIZE`  | Cümle veya kelime bazlı parçalama   | `NLP TOKENIZE "Metin burada"`              |
| `SENTIMENT` | Duygu analizi                       | `NLP SENTIMENT "Bugün harikayım!"`         |
| `POS TAG`   | Sözcük türü etiketi (isim, fiil...) | `NLP POS "Bu bir testtir."`                |
| `SUMMARIZE` | Metni özetleme                      | `NLP SUMMARIZE "Uzun metin..."`            |
| `TRANSLATE` | Çeviri (auto -> hedef dil)          | `NLP TRANSLATE "Merhaba" TO "en"`          |
| `CLASSIFY`  | Metin sınıflandırması               | `NLP CLASSIFY "Bu bir finans cümlesidir."` |

> Not: Tüm işlemler `NLPManager` sınıfı tarafından yürütülür.

---

### 14.2 Örnek Kullanımlar

#### Örnek 1: Tokenizasyon

```basic
NLP TOKENIZE "Merhaba dünya!" result$
PRINT result$
```

Çıktı: `["Merhaba", "dünya", "!"]`

#### Örnek 2: Duygu Analizi

```basic
NLP SENTIMENT "Bugün mükemmel geçti." duygu$
PRINT duygu$
```

Çıktı: `{"duygu": "pozitif", "skor": 0.92}`

#### Örnek 3: POS Etiketleme

```basic
NLP POS "Ali koşarak eve gitti." etiketler$
PRINT etiketler$
```

Çıktı: `[{"Ali": "NNP"}, {"koşarak": "VBG"}, ...]`

#### Örnek 4: Özetleme (transformers)

```basic
NLP SUMMARIZE "Bu çok uzun bir metindir..." ozet$
```

#### Örnek 5: Çok Dilli Çeviri

```basic
NLP TRANSLATE "Ben mutluyum." TO "en" ceviri$
PRINT ceviri$
```

Çıktı: `"I am happy."`

---

### 14.3 Teknik Detaylar

* Varsayılan dil: İngilizce (`en`), ancak `"tr"`, `"fr"`, `"de"`, `"es"` desteklenir.
* `transformers` çağrıları GPU otomatik algılar.
* `spacy` modelleri otomatik indirilir: `en_core_web_sm`
* NLTK `punkt`, `vader_lexicon`, `averaged_perceptron_tagger` modüllerini kullanır

---

### 14.4 NLPManager Ana Metotlar

| Metot                           | Açıklama                                |
| ------------------------------- | --------------------------------------- |
| `analyze_text(text, lang?)`     | Tokenizasyon, cümle bulma, POS çıkartır |
| `summarize_text(text, model?)`  | Özetleme (transformer) uygular          |
| `classify_text(text, model?)`   | Sınıflandırma yapar                     |
| `translate_text(text, to_lang)` | Çeviri işlemi                           |
| `detect_sentiment(text)`        | Duygu analizi yapar                     |

---

### 14.5 Alıştırmalar

1. Kullanıcıdan alınan metni tokenize edip her kelimeye POS etiketi verin.
2. Bir haber metnini 3 cümleye özetleyin.
3. Türkçe cümleleri İngilizce’ye çevirip sentiment skorlarını bulun.

---

## 15 · ML Sistemi – Makine Öğrenimi ve Yapay Zeka <a name="ml"></a>

`libx_ml.py` modülü, PDS-X içerisinde gözetimli öğrenme modelleri kurmak, eğitmek, test etmek ve tahmin yapmak için kullanılır. Sistem hem geleneksel modeller (sklearn) hem de derin öğrenme (PyTorch) desteklidir.

### 15.1 Desteklenen Model Türleri

| Model Tipi | Açıklama                                   |
| ---------- | ------------------------------------------ |
| `logistic` | `scikit-learn` tabanlı lojistik regresyon  |
| `neural`   | `PyTorch` ile çok katmanlı sinir ağı (MLP) |

---

### 15.2 Komutlar

| Komut        | Açıklama                  | Sözdizimi                                         |
| ------------ | ------------------------- | ------------------------------------------------- |
| `ML CREATE`  | Yeni model oluşturur      | `ML CREATE "m1" TYPE "logistic" PARAMS {"C":1.0}` |
| `ML TRAIN`   | Verilerle modeli eğitir   | `ML TRAIN "m1" X$, Y$`                            |
| `ML PREDICT` | Tahmin üretir             | `ML PREDICT "m1" X$ result$`                      |
| `ML SAVE`    | Modeli kaydeder           | `ML SAVE "m1" TO "model.pkl"`                     |
| `ML LOAD`    | Kaydedilmiş modeli yükler | `ML LOAD "model.pkl" AS "m2"`                     |

---

### 15.3 Model Sınıfı (Model)

| Metot                                    | Açıklama                                                           |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `__init__(model_id, model_type, params)` | Modeli tanımlar                                                    |
| `_initialize()`                          | İçsel yapıyı kurar (ör. `LogisticRegression` veya `nn.Sequential`) |
| `train(X, y)`                            | Modeli veri ile eğitir                                             |
| `predict(X)`                             | Tahmin üretir                                                      |
| `save(path)`                             | Modeli pickle ile kaydeder                                         |
| `load(path)`                             | Modeli dosyadan yükler                                             |

> Tüm işlevler thread-safe çalışır (`@synchronized` dekoratörü ile)

---

### 15.4 Örnek Kullanımlar

#### Örnek 1: Lojistik Regresyon ile Eğitim

```basic
ML CREATE "model1" TYPE "logistic" PARAMS {"max_iter":200}
ML TRAIN "model1" X$, Y$
ML PREDICT "model1" X$ sonuc$
PRINT sonuc$
```

#### Örnek 2: Sinir Ağı Eğitimi (PyTorch)

```basic
ML CREATE "net1" TYPE "neural" PARAMS {
  "input_dim": 5,
  "hidden_dim": 10,
  "output_dim": 1
}
ML TRAIN "net1" X$, Y$
ML PREDICT "net1" yeniX$ tahmin$
```

#### Örnek 3: Model Kaydetme ve Yükleme

```basic
ML SAVE "model1" TO "modelim.pkl"
ML LOAD "modelim.pkl" AS "model2"
```

---

### 15.5 Teknik Notlar

* Tüm veriler `numpy.ndarray` olarak alınır.
* Lojistik regresyon için `sklearn.preprocessing.StandardScaler` kullanılır.
* `neural` modeller GPU destekli PyTorch ile çalışır.
* `save/load` işlemleri `pickle` üzerinden yapılır.

---

### 15.6 Alıştırmalar

1. XOR problemi için bir sinir ağı tanımlayıp eğitin.
2. Sayısal veri seti üzerinde `logistic` modeli ile tahmin yapın.
3. Eğitilmiş modeli kaydedin, daha sonra tekrar yükleyip test verisiyle kullanın.

---

## 16 · LibXCore – Sistem Yardımcıları, Koleksiyonlar ve Dönüşümler <a name="libxcore"></a>

`libxcore.py` modülü, PDS-X sisteminin çekirdek işlevlerini barındırır. Bu modül koleksiyon yönetimi, işlevsel programlama, sistem durumları, veri kodlamaları ve yardımcı araçları sağlar.

---

### 16.1 Temel Özellikler

* Kodlama: UTF-8, CP1254, UTF-16, Latin-9, vs.
* Koleksiyonlar: Liste, sözlük, yığın, kuyruk
* Thread yönetimi: aktif thread takibi
* Genişletilebilir metaveri desteği

---

### 16.2 Koleksiyon İşlevleri

| Fonksiyon                                 | Açıklama                    | Örnek                           |
| ----------------------------------------- | --------------------------- | ------------------------------- |
| `each(func, iterable)`                    | Her öğeye fonksiyon uygular | `each(print, [1,2,3])`          |
| `select(func, iterable)`                  | Şarta uyanları döner        | `select(lambda x: x>10, liste)` |
| `insert(collection, value, index?, key?)` | Eleman ekler                | `insert(mylist, 42, 1)`         |
| `remove(collection, index?, key?)`        | Eleman siler                | `remove(mylist, 0)`             |

> Not: `insert/remove`, liste ve sözlüklerde çalışır. Tip uyuşmazlığı `PdsXException` üretir.

---

### 16.3 Kodlama Destekleri

```python
self.supported_encodings = ["utf-8", "utf-16", "cp1254", ..., "utf-8-bom-less"]
```

Veri yükleme ve yazma işlemleri sırasında bu kodlamalar otomatik olarak test edilir. `default_encoding = "utf-8"` olarak ayarlanmıştır.

---

### 16.4 Stack ve Queue Yapıları

```python
self.stacks["stack1"] = []
self.queues["queue1"] = deque()
```

Bu yapılar manuel veya PIPE sistemi tarafından kullanılabilir. Fonksiyonlar ileride detaylandırılacaktır.

---

### 16.5 Log ve Hata Yönetimi

Tüm hatalar `PdsXException` sınıfı ile tanımlanır ve aşağıdaki log sistemi tarafından yakalanır:

```python
logging.getLogger("libxcore")
```

Tüm hatalar `pdsxu_errors.log` dosyasına yazılır.

---

### 16.6 Örnek Kullanım

```basic
LET veriler = [10, 15, 8, 25]
CALL each(lambda x: PRINT x, veriler)
LET filtreli = select(lambda x: x > 12, veriler)
PRINT filtreli
```

```basic
LET sozluk = {}
CALL insert(sozluk, "deger", key="anahtar")
PRINT sozluk
```

---

### 16.7 Ek Fonksiyonlar ve Özellikler

#### 16.7.1 Dizi, Sözlük ve Karma Koleksiyonlar

`libxcore`, PDSX içerisindeki veri yapılarını genişletilmiş işlemlerle yönetir. Aşağıdaki fonksiyonlar doğrudan ya da `PIPE`, `TREE`, `NLP` gibi sistemlerde kullanılır.

| Fonksiyon                     | Açıklama                        |
| ----------------------------- | ------------------------------- |
| `merge_dicts(dict1, dict2)`   | İki sözlüğü birleştirir         |
| `flatten(nested_list)`        | Çok katmanlı diziyi düzleştirir |
| `unique(values)`              | Yinelenenleri ayıklar           |
| `group_by(iterable, keyfunc)` | Listeyi grup grup ayırır        |
| `sort_by(iterable, keyfunc)`  | Anahtar işlevine göre sıralar   |

#### Örnek:

```python
flatten([[1,2],[3,4],[5]]) => [1,2,3,4,5]
unique([1,2,2,3,3,3]) => [1,2,3]
```

#### 16.7.2 Sayısal Yardımcılar

| Fonksiyon                                      | Açıklama                                 |
| ---------------------------------------------- | ---------------------------------------- |
| `clamp(val, min_, max_)`                       | Belirli aralıkta sınırlar                |
| `scale(val, in_min, in_max, out_min, out_max)` | Değer dönüşümü                           |
| `normalize(arr)`                               | \[0,1] aralığına getirir (list of float) |

---

#### 16.7.3 Genişletilebilir Veri Sözlükleri (ExtDict)

İleride `libxcore` üzerine `ExtDict` sınıfı planlanmaktadır:

```python
ed = ExtDict()
ed.set("anahtar", 5)
ed.append("anahtar", 7)
```

Çoklu değer atama, erişim limiti, TTL gibi yapılar desteklenecek.

#### 16.7.4 Sistem Entegrasyonları

* `PIPE` sisteminde veri doğrulama, metin normalizasyonu için `select`, `unique`, `flatten` sık kullanılır.
* `NLP` modülü sonuçlarını sadeleştirmek için `group_by`, `sort_by`, `clamp` desteklenir.

---

### 16.8 Alıştırmalar

1. 20 öğeden oluşan bir listeyi tanımlayın, `select` ile sadece çiftleri yazdırın.
2. Bir sözlük oluşturun, 3 anahtar-değer çifti ekleyin, sonra birini `remove` ile silin.
3. Geçersiz kodlama girilirse ne olacağını test edin (örneğin `"x-unknown"`).

---

## 17 · PIPE + NLP + PROLOG Entegrasyon Senaryoları <a name="integration"></a>

PDS-X sistemi, modüler yapısı sayesinde PIPE, NLP ve PROLOG motorlarını birbirine bağlayarak gerçek dünya problemlerini çözebilecek çok güçlü senaryolar yaratmanıza olanak tanır. Bu bölümde hem BASIC komutlarıyla hem de Prolog mantığıyla çalışan örnek sistemler anlatılacaktır.

---

### 17.1 Senaryo 1: Girdi Cümlelerini Anlamlandırma ve Geri Yazma

#### Amaç:

Kullanıcıdan gelen metni PIPE üzerinden al, NLP ile duygu ve konu analizini yap, sonucu dosyaya geri yaz.

```basic
PIPE CREATE "GIRDI"
PIPE CONNECT "GIRDI" TYPE "file" PATH "yorum.txt"
PIPE READ "GIRDI", yorum$
NLP SENTIMENT yorum$, duygu$
NLP CLASSIFY yorum$, konu$
LET sonuc$ = "Duygu: " + duygu$ + "
Konu: " + konu$
PIPE CREATE "CIKTI"
PIPE CONNECT "CIKTI" TYPE "file" PATH "analiz.txt"
PIPE WRITE "CIKTI", sonuc$
```

#### Genişletilmiş Prolog Tanımı:

```prolog
duygu(yorum, pozitif).
konu(yorum, egitim).
rapor(yorum, pozitif, egitim).
```

---

### 17.2 Senaryo 2: Otomatik Prolog Bilgi Tabanı Üretimi

#### Amaç:

Yüzlerce cümleyi PIPE üzerinden al, NLP ile analiz et, sonuçları Prolog veritabanına otomatik çevir.

```basic
FOR i = 1 TO 100
  PIPE READ "METIN", m$
  NLP SENTIMENT m$, d$
  NLP CLASSIFY m$, k$
  LET pl$ = "analiz(" + m$ + "," + d$ + "," + k$ + ")."
  PIPE WRITE "PROLOG_DB", pl$
NEXT i
```

#### Oluşan Çıktı:

```prolog
analiz("Bu sistemi çok seviyorum", pozitif, teknoloji).
analiz("Böyle iş olmaz", negatif, ekonomi).
```

---

### 17.3 Senaryo 3: NLP + PROLOG ile Sorgulama

#### Amaç:

Yorumlar içinden sadece "pozitif" duygu içerenleri NLP ile bul, sonra Prolog ile belirli konuda filtrele.

```basic
NLP SENTIMENT y1$, d$
NLP CLASSIFY y1$, k$
IF d$ = "pozitif" THEN
  PROLOG QUERY "analiz(" + y1$ + ", pozitif, " + k$ + ")", sonuc$
ENDIF
```

#### Prolog Sorgusu:

```prolog
?- analiz(X, pozitif, teknoloji).
```

---

### 17.4 Genişletme Fikirleri

* GNN + NLP ile otomatik ontoloji ağları kurma
* NLP ile özetlenmiş metinleri PIPE üzerinden başka AI sistemine aktarma
* PROLOG motoruyla NLP sonuçlarını karar ağacına dönüştürme

---

### 17.5 Alıştırmalar

1. Kullanıcıdan alınan 5 cümleyi NLP ile analiz edin ve Prolog tabanına kaydedin.
2. Prolog üzerinden yalnızca "finans" kategorisine ait "pozitif" yorumları sorgulayın.
3. `PIPE + NLP` kullanarak canlı yorum akışı oluşturun, duygusal değişimleri grafiğe dökün.

---

## 18 · PIPE + NLP + PROLOG Entegre Uygulama Senaryoları (İleri Seviye) <a name="pipe-nlp-prolog-ex"></a>

Bu bölümde PIPE sistemini NLP analizleriyle birleştirip PROLOG mantıksal motoruyla bilgi çıkarımı yapan entegre programlar tasarlıyoruz. Amaç: Gerçek dünyadaki metin verisini analiz edip karar destek sistemleri üretmek.

---

### 18.1 Uygulama: Otomatik Destek Yanıtlayıcı

**Amaç:** Kullanıcının yazdığı yardım mesajlarını analiz et, PROLOG ile çözüm öner.

#### Adım Adım Süreç:

1. Kullanıcıdan PIPE aracılığıyla mesaj alınır.
2. NLP ile duygu ve konu analizi yapılır.
3. PROLOG’a bilgi yazılır.
4. PROLOG’dan uygun cevap alınır.

#### Kod:

```basic
PIPE CREATE "INPUT"
PIPE CONNECT "INPUT" TYPE "file" PATH "helpdesk.txt"
PIPE READ "INPUT", mesaj$
NLP SENTIMENT mesaj$, duygu$
NLP CLASSIFY mesaj$, konu$
LET bilgi$ = "yardim(" + mesaj$ + "," + duygu$ + "," + konu$ + ")."
PIPE WRITE "KB", bilgi$
PROLOG ASSERT bilgi$
PROLOG QUERY "cevap(" + konu$ + ",Cevap)", sonuc$
PRINT sonuc$
```

#### PROLOG Tanımı:

```prolog
yardim("Bilgisayar açılmıyor", negatif, teknik).
cevap(teknik, "Lütfen güç kablosunu kontrol edin.").
```

---

### 18.2 Uygulama: Anket Analizi + Politika Önerisi

**Amaç:** Vatandaş yorumlarını NLP ile analiz et, PROLOG ile yönetime öneri üret.

#### Kod:

```basic
FOR i = 1 TO 50
  PIPE READ "ANKET", veri$
  NLP SENTIMENT veri$, d$
  NLP CLASSIFY veri$, konu$
  PROLOG ASSERT "gorus(" + veri$ + "," + d$ + "," + konu$ + ")."
NEXT i
PROLOG QUERY "oner(konut, Oneri)", sonuc$
PRINT "Politika Önerisi: ", sonuc$
```

#### PROLOG Tanımı:

```prolog
gorus("Ev kiraları çok yüksek", negatif, konut).
oner(konut, "Kira denetimi sistemi kurulmalı.").
```

---

### 18.3 Uygulama: Gerçek Zamanlı İzleme Paneli

**Amaç:** PIPE üzerinden canlı gelen metinleri NLP ile etiketle, Prolog ile alarm üret.

#### Kod:

```basic
WHILE TRUE
  PIPE READ "CANLI", m$
  NLP SENTIMENT m$, d$
  NLP CLASSIFY m$, k$
  IF d$ = "negatif" THEN
    PROLOG ASSERT "olay(" + m$ + "," + d$ + "," + k$ + ")."
    PROLOG QUERY "alarm(" + k$ + ",A)", sonuc$
    PRINT sonuc$
  END IF
WEND
```

#### PROLOG:

```prolog
olay("Sular kesildi", negatif, altyapi).
alarm(altyapi, "Belediye birimi uyarılmalı!").
```

---

### 18.4 Gelişmiş Entegrasyon Fikirleri (PDSX ile Uygulama Örnekleri)

#### 1. NLP + PIPE + TREE ile Ontoloji Ağı Kurma

```basic
REM Her yorum bir düğüm olarak ağaca eklenir, sınıf etiketi dallara ayrılır
PIPE READ "YORUM", cumle$
NLP CLASSIFY cumle$, kategori$
TREE BUILD "ONTOLOJI" AS GENERAL "Yorumlar" kok$
TREE ADD NODE "ONTOLOJI", kok$, kategori$, kat$
TREE ADD NODE "ONTOLOJI", kat$, cumle$, dugum$
```

#### 2. NLP + PIPE + PROLOG ile Karar Ağacı Geliştirme

```basic
REM Duygu ve konuya göre öneri kararı al
PIPE READ "GIRIS", m$
NLP SENTIMENT m$, d$
NLP CLASSIFY m$, k$
PROLOG ASSERT "durum(" + m$ + "," + d$ + "," + k$ + ")."
PROLOG QUERY "karar(" + d$ + "," + k$ + ",Cevap)", karar$
PRINT karar$
```

```prolog
karar(pozitif, egitim, "Öner: Eğitim programı olumlu etkiliyor.").
karar(negatif, finans, "Öner: Finansal destek önerilmeli.").
```

#### 3. ML + NLP Çıktılarını PROLOG ile Süzme

```basic
REM ML tahmini sonucu Prolog ile mantıksal kurala bağla
ML PREDICT "m1" X$ sonuc$
IF sonuc$ = 1 THEN
  PROLOG ASSERT "risk(kredi, yuksek)."
  PROLOG QUERY "aksiyon(kredi, yuksek, H)", sonuc$
  PRINT "Alınacak aksiyon: ", sonuc$
ENDIF
```

```prolog
aksiyon(kredi, yuksek, "Kredi başvurusu reddedilmeli").
```

---

* `PIPE + NLP` ile gelen haberleri sınıflandır, `PROLOG` ile kamuoyunu yorumla.
* `TREE + NLP` ile sözdizimsel yapıları analiz et, `PIPE` ile akışa bağla.
* `ML + NLP` modellerinin çıktısını `PROLOG` mantığıyla filtrele.

---

### 18.5 Örnek Kullanım Önerileri (PDSX BASIC ile)

Bu bölümde, önceki entegrasyon senaryolarını genişleterek PDSX dili ile yazılmış örnek programları sunuyoruz. Her örnek, hem PIPE hem NLP hem de PROLOG birleşimini içerir.

#### Örnek 1: Şikayet Yorumlarının Duygu Haritası

```basic
REM Şikayetleri oku, duygu analizi yap, duygu haritası oluştur
PIPE CREATE "YORUM"
PIPE CONNECT "YORUM" TYPE "file" PATH "sikayetler.txt"
PIPE READ "YORUM", m$
NLP SENTIMENT m$, d$
NLP CLASSIFY m$, k$
PROLOG ASSERT "sikayet(" + m$ + "," + d$ + "," + k$ + ")."
PRINT "Duygu: ", d$
```

#### Örnek 2: Çözüm Tablosu Üretimi

```basic
REM Yorumların çözümlerini Prolog ile üret
PROLOG ASSERT "cozum(teknik, 'Teknik servis yönlendirildi.')."
PROLOG ASSERT "cozum(finans, 'Muhasebe ile iletişime geçiniz.')."
PIPE READ "YORUM", y$
NLP CLASSIFY y$, k$
PROLOG QUERY "cozum(" + k$ + ",C)", cevap$
PRINT "Çözüm: ", cevap$
```

#### Örnek 3: Canlı Duygu İzleyici

```basic
REM Sürekli okuma + alarm üretimi
WHILE TRUE
  PIPE READ "CANLI", satir$
  NLP SENTIMENT satir$, duygu$
  NLP CLASSIFY satir$, kategori$
  IF duygu$ = "negatif" THEN
    PROLOG ASSERT "alarm(" + kategori$ + ")."
    PRINT "Uyarı: ", kategori$ + " alanında negatif geri bildirim."
  END IF
WEND
```

---

1. Kullanıcıdan alınan 10 şikayeti NLP ile analiz edin, Prolog'a aktarın.
2. Belirli bir duygu veya konuya göre Prolog üzerinden çözüm üretin.
3. NLP sonucunu bir ağ yapısına (`TREE`) dönüştürerek görselleştirin ve her düğüm için Prolog kuralı üretin.

---

## 19 · Deneysel Yapılar ve Gelişmiş Kullanım Örnekleri <a name="experimental"></a>

PDS-X sisteminde bazı modüller henüz deneysel aşamada olmasına rağmen çok güçlü kullanım senaryoları sunmaktadır. Bu bölümde bu tür yapıları tanıtıyoruz.

---

### 19.1 Deneysel Yapılar

#### a. INLINE C / INLINE ASM / INLINE REPLY

Bu bloklar kullanıcıya BASIC içinde doğrudan özel C/ASM/REPLY diliyle yazma imkânı verir.

```basic
INLINE C
  printf("Merhaba C dünyası
");
ENDINLINE
```

```basic
INLINE ASM
  MOV AX, 01h
  INT 10h
ENDINLINE
```

```basic
INLINE REPLY
  Eğer kullanıcı "yardım" yazarsa, yanıt ver: "Sizi dinliyorum."
ENDINLINE
```

Bu bloklar çalışma sırasında yorumlanır ve sistemle etkileşime geçer.

#### b. GNN Katmanı (GRAPH + AI)

```basic
GRAPH GNN "G1" 4 SUM [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] sonuc$
```

Graf yapısı üzerinde mesaj geçişi yapılır, bu bir tür yapay sinir ağıdır.

#### c. BYTECODE ve OPCODE

İlerideki sürümlerde her komut ve yapının bir `OPCODE` karşılığı üretilecek, sistem bayt dizisine çevrilerek çalıştırılacaktır. Bu sistem için `bytecode_manager.py` dosyası planlanmıştır.

---

### 19.2 Karma Kullanımlar (Multi-Mode)

#### Örnek: NLP ile etiketlenmiş verileri grafik yapıya gömme

```basic
NLP CLASSIFY "Sistem arızalı olabilir." kategori$
GRAPH CREATE "G2" AS UNDIRECTED "ariza" k$
GRAPH ADD VERTEX "G2" "log1" v1$
GRAPH ADD EDGE "G2" k$, v1$
```

#### Örnek: INLINE + PROLOG ile karar mantığı

```basic
INLINE REPLY
  kullanıcı "fatura gecikti" yazarsa, "Son ödeme tarihi nedir?" diye sor.
ENDINLINE
PROLOG ASSERT "fatura(durum, gecikmis)."
PROLOG QUERY "uyari(fatura, X)", cevap$
```

```prolog
uyari(fatura, "Faturanız gecikmiş, işlem başlatılacak.").
```

---

### 19.3 Önerilen Geliştirme Alanları

* PIPE üzerinde çalışan INLINE REPLY bot sistemi
* PROLOG bilgilerini TREE yapısına görsel olarak dönüştüren analiz
* NLP sınıflandırmasını ML ile destekleyen karar motorları

---

## 20 · REPLY\_EXTENSION – Gelişmiş Yanıt Yönetimi ve Kuantum Korelasyonları <a name="reply-extension"></a>

Bu modül, PDS-X içerisinde özel yanıt nesneleri (`Response`) oluşturmak ve bu yanıtları formatlamak, ilişkilendirmek ve kuantum benzeri algoritmalarla analiz etmek için kullanılır.

---

### 20.1 Response Sınıfı

```python
resp = Response(response_id="r123", data={"msg": "Merhaba"}, timestamp=time.time())
```

| Özellik          | Açıklama                                          |
| ---------------- | ------------------------------------------------- |
| `response_id`    | Yanıtın benzersiz kimliği (UUID önerilir)         |
| `data`           | Yanıt verisi (her türlü veri olabilir)            |
| `timestamp`      | Yanıtın oluşturulma zamanı                        |
| `metadata`       | Biçim, şifreleme, sıkıştırma gibi bilgiler        |
| `execution_time` | Yanıtı üretme süresi (opsiyonel olarak ayarlanır) |

#### `format(format_type)` Metodu

Yanıtı belirtilen formata dönüştürür.

| Format | Açıklama                               |
| ------ | -------------------------------------- |
| `json` | `utf-8` ile JSON çıktı                 |
| `yaml` | YAML çıktısı                           |
| `xml`  | XML etiketli yapı                      |
| `pdsx` | İç meta bilgileriyle özel JSON çıktısı |

#### Örnek Kullanım

```basic
LET yanit$ = "Merhaba dünya"
CALL response.create("yanit1", yanit$, NOW())
CALL response.format("json")  => "{\"msg\":\"Merhaba dünya\"}"
```

---

### 20.2 QuantumResponseCorrelator

Bu sınıf, iki farklı `Response` nesnesinin benzerliğini ölçmek için kuantum benzeri bir korelasyon uygular.

#### `correlate(response1, response2)`

* Basit Jaccard benzerliği üzerinden çalışır.
* Harf bazlı karakter kümelerini karşılaştırır.
* Korelasyon ID’si üretir ve benzerlik skorunu saklar.

```python
correlator = QuantumResponseCorrelator()
correlator.correlate(resp1, resp2)
# Çıktı: correlation_id, skor (0.0 - 1.0 arası)
```

#### Örnek BASIC Senaryo

```basic
REM İki kullanıcı mesajı korelasyonla eşleştiriliyor
LET m1$ = "Merhaba, sistem açılmıyor."
LET m2$ = "Sistem sorun veriyor."
CALL RESPONSE.CREATE("R1", m1$, NOW())
CALL RESPONSE.CREATE("R2", m2$, NOW())
CALL RESPONSE.CORRELATE("R1", "R2"), skor$
PRINT "Benzerlik Skoru: ", skor$
```

---

### 20.3 Kullanım Alanları

* INLINE REPLY sisteminden gelen metinlerin çıktısını depolamak ve eşleştirmek
* NLP yorumlarıyla oluşturulmuş sonuçları korelasyonlu şekilde ilişkilendirmek
* Prolog bilgi tabanına benzer yapıdaki örnekleri kayıtla kıyaslamak

---

### 20.4 Geliştirme Önerileri

* Korelasyonlar `GRAPH` yapısında görselleştirilebilir
* `PIPE` ile toplu yanıt akışı alınabilir
* `AES` ve `gzip` destekli şifreli/sıkıştırılmış format eklenebilir

---

## 21 · ZAMANLAYICI SİSTEMİ – f12\_timer\_manager.py <a name="timer"></a>

Bu modül, PDS-X içerisinde zaman tabanlı görevleri çalıştırmak için kullanılan gelişmiş bir zamanlayıcı yönetim sistemidir. Kullanıcı tanımlı işlemleri belirli aralıklarla çalıştırabilir, duraklatabilir veya iptal edebilir.

---

### 21.1 Timer Sınıfı

```python
t = Timer(timer_id="gorev1", interval=5.0, handler=my_function, is_periodic=True)
t.start()
```

| Özellik       | Açıklama                                                |
| ------------- | ------------------------------------------------------- |
| `timer_id`    | Zamanlayıcının benzersiz kimliği                        |
| `interval`    | Çalışma aralığı (saniye cinsinden)                      |
| `handler`     | Çalıştırılacak fonksiyon                                |
| `is_periodic` | True ise periyodik çalışır, False ise yalnızca bir kere |

---

### 21.2 Metotlar

| Metot      | Açıklama               |
| ---------- | ---------------------- |
| `start()`  | Zamanlayıcıyı başlatır |
| `pause()`  | Çalışmayı duraklatır   |
| `resume()` | Devam ettirir          |
| `cancel()` | İptal eder             |

Ek İzleme Özellikleri:

* `execution_count`: Toplam çalıştırma sayısı
* `total_time`: Toplam çalıştırma süresi (saniye)
* `next_run`: Bir sonraki çalışma zamanı (timestamp)

---

### 21.3 Örnek PDSX Kullanımı

#### Örnek 1: Her 10 saniyede bir görev çalıştır

```basic
DEF mygorev()
  PRINT "Görev çalıştı!"
ENDDEF
TIMER CREATE "G1" INTERVAL 10 FUNC mygorev PERIODIC TRUE
TIMER START "G1"
```

#### Örnek 2: 5 saniyelik gecikmeli tek seferlik işlem

```basic
TIMER CREATE "T1" INTERVAL 5 FUNC "kaydet" PERIODIC FALSE
TIMER START "T1"
```

#### Örnek 3: Duraklatma ve Devam Ettirme

```basic
TIMER PAUSE "G1"
WAIT 3
TIMER RESUME "G1"
```

---

### 21.4 Entegrasyon Senaryoları

* **EVENT** sisteminde zamanlanmış olay tetikleme
* **PIPE** sisteminden gelen verileri belirli zaman aralıklarında işleme
* **GRAPH** yapısında zamanla gelişen düğüm ilişkileri oluşturma
* **ML/NLP** modellerini periyodik eğitme veya sonuç kontrolü

---

### 21.5 Geliştirme Önerileri

* `next_run` zamanlarını PIPE üzerinden dışa aktararak zaman çizelgesi oluşturulabilir
* `execution_count` kullanılarak başarım metrikleri izlenebilir
* Birden fazla `Timer` nesnesi ile paralel görev akışı kurulabilir

---
