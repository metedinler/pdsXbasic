# pdsXbasic

**pdsXbasic** is a feature-rich, modular BASIC interpreter and runtime environment implemented in Python. It provides a comprehensive set of functionalities inspired by QBasic/PDS systems, enhanced with modern Python libraries and techniques for concurrency, data structures, persistence, and extensibility.

## 🚀 Features

* **Bytecode Compilation & Execution**: Compile PDSX BASIC code into bytecode and execute using `bytecode_compiler.py` and `bytecode_manager.py`.
* **Data Structures**: Built-in support for lists, stacks, queues, trees, graphs and custom data types (`data_structures.py`, `tree.py`, `graph.py`).
* **Memory Management**: Low-level memory operations and allocation management (`memory_manager.py`, `lowlevel.py`).
* **Event Handling**: 64-slot event/event interrupt system (`event.py`).
* **Concurrency & Parallelism**: Threading and multiprocessing support via `libx_concurrency.py`, `multithreading_process.py`, asyncio integration.
* **Pipeline Processing**: Flexible pipe/pipeline patterns (`pipe.py`, `pipe_monitor_gui.py`).
* **Save & Load System**: Persistence with multiple encodings and compression methods (`save_load_system.py`, `save.py`).
* **Database Integration**: SQL/ISAM support (`database_sql_isam.py`, `sqlite.py`) and high-level DB APIs (`lib_db.py`).
* **Logging & Debugging**: Backtrace logger (F11) and timer manager (F12) modules for robust debugging (`f11_backtrace_logger.py`, `f12_timer_manager.py`).
* **Library Extensions**: Modular `libx_` packages for GUI, JIT, logic, ML, network, NLP and more.
* **Reporting**: Export reports to DOC/TeX formats (`export_report_doc.py`, `export_report_doc.tex`).
* **Error Handling**: Custom exception hierarchy via `pdsx_exception.py`.

## 📂 Directory Structure

```
.gitignore
bytecode_compiler.py
bytecode_manager.py
clazz.py
data_structures.py
database_sql_isam.py
dosya_listesi.txt
event.py
export_report_doc.py
export_report_doc.tex
f11_backtrace_logger.py
f12_timer_manager.py
functional.py
functional2.py
graph.py
graph2.py
lib_db.py
libx_concurrency.py
libx_data.py
libx_gui.py
libx_jit.py
libx_logic.py
libx_ml.py
libx_network.py
libx_nlp.py
libxcore.py
lowlevel.py
memory_manager.py
multithreading_process.py
oop_and_class.py
oop_and_class2.py
oop_and_class2x.py
pdsXu.zip
pdsXuv14.py
pdsx_exception.py
pipe.py
pipe_monitor_gui.py
reply_extension.py
save.py
save_load_system.py
save_load_system2.py
sqlite.py
tree.py
tree2.py
tree3.py
```

## 🛠 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/metedinler/pdsXbasic.git
   cd pdsXbasic
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt  # (create this file listing necessary packages)
   ```

3. **Run the interpreter**:

   ```bash
   python pdsXuv14.py  # Launch the BASIC interpreter UI/CLI
   ```

## 📝 Usage

* **Compile BASIC code**: Use `bytecode_compiler.py` to compile `.basX`-style scripts into bytecode files.
* **Execute Bytecode**: Load and run bytecode via `bytecode_manager.py` or through the main interpreter script.
* **Interactive Mode**: Launch `pdsXuv14.py` for an interactive REPL, with support for commands, event triggers, and GUI monitoring.
* **Scripting**: Integrate PDSX BASIC modules into Python workflows by importing modules such as `pipe`, `event`, and `memory_manager`.

## 📐 Architecture & Modules

Each module is designed to work independently or together; key components include:

| Module Group         | Files                                              | Purpose                                        |
| -------------------- | -------------------------------------------------- | ---------------------------------------------- |
| Core Engine          | `pdsXuv14.py`, `bytecode_*`, `pdsx_exception.py`   | Main interpreter, compiler, exception handling |
| Data & Structures    | `data_structures.py`, `tree.py`, `graph.py`        | Custom collections and graph/tree algorithms   |
| Memory & Low-Level   | `memory_manager.py`, `lowlevel.py`                 | Memory allocation, pointer-like operations     |
| Concurrency          | `libx_concurrency.py`, `multithreading_process.py` | Thread/process management, asyncio integration |
| Event & Pipeline     | `event.py`, `pipe.py`                              | Event loop, pipeline/pipe execution patterns   |
| Persistence          | `save_load_system.py`, `save.py`                   | Saving/loading state with compression          |
| Database             | `database_sql_isam.py`, `sqlite.py`, `lib_db.py`   | SQL/ISAM and SQLite interactions               |
| Logging & Debugging  | `f11_backtrace_logger.py`, `f12_timer_manager.py`  | Debug logs, performance timing                 |
| Extensions (`libx_`) | GUI, JIT, logic, ML, network, NLP modules          | Domain-specific libraries                      |
| Reporting            | `export_report_doc.py`, `.tex`                     | Generate DOC/TeX reports                       |

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add YourFeature"`)
4. Push to your branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Generated by ChatGPT based on the repository analysis.*

# pdsXbasic

**pdsXbasic**, Python ile yazılmış, modüler ve özellik dolu bir BASIC yorumlayıcısı ve çalışma zamanıdır. QBasic/PDS sistemlerinden esinlenerek geliştirilmiş; eşzamanlılık, veri yapıları, kalıcılık ve genişletilebilirlik için modern Python kütüphanelerini kullanır.

## 🚀 Özellikler

* **Bytecode Derleme ve Çalıştırma**: PDSX BASIC kodunu bytecode’a çevirir ve `bytecode_compiler.py` ile derleyip `bytecode_manager.py` ile çalıştırır.
* **Veri Yapıları**: Listeler, yığınlar, kuyruklar, ağaçlar, grafikler ve özel veri tipleri desteği (`data_structures.py`, `tree.py`, `graph.py`).
* **Bellek Yönetimi**: Düşük seviyede bellek ayırma ve yönetim işlevleri (`memory_manager.py`, `lowlevel.py`).
* **Olay Yönetimi**: 64 slotlu olay ve kesme sistemi (`event.py`).
* **Eşzamanlılık & Paralelleştirme**: `libx_concurrency.py`, `multithreading_process.py` ile threading ve multiprocessing; `asyncio` desteği.
* **Boru Hattı İşleme**: Esnek pipe/pipeline desenleri (`pipe.py`, `pipe_monitor_gui.py`).
* **Kaydetme & Yükleme**: Çeşitli kodlamalar ve sıkıştırma yöntemleriyle kalıcılık (`save_load_system.py`, `save.py`).
* **Veritabanı Entegrasyonu**: SQL/ISAM ve SQLite desteği (`database_sql_isam.py`, `sqlite.py`, `lib_db.py`).
* **Loglama & Hata Ayıklama**: Geri izleme (F11) ve zamanlayıcı (F12) modülleri (`f11_backtrace_logger.py`, `f12_timer_manager.py`).
* **Libx Uzantıları**: GUI, JIT, mantık, makine öğrenmesi, ağ, NLP gibi alanlara yönelik modüler `libx_` paketleri.
* **Raporlama**: DOC/TeX formatında rapor oluşturma (`export_report_doc.py`, `export_report_doc.tex`).
* **Hata Yönetimi**: Özel istisna hiyerarşisi (`pdsx_exception.py`).

## 📂 Dizin Yapısı

```
.gitignore
bytecode_compiler.py
bytecode_manager.py
clazz.py
data_structures.py
database_sql_isam.py
dosya_listesi.txt
event.py
export_report_doc.py
export_report_doc.tex
f11_backtrace_logger.py
f12_timer_manager.py
functional.py
functional2.py
graph.py
graph2.py
lib_db.py
libx_concurrency.py
libx_data.py
libx_gui.py
libx_jit.py
libx_logic.py
libx_ml.py
libx_network.py
libx_nlp.py
libxcore.py
lowlevel.py
memory_manager.py
multithreading_process.py
oop_and_class.py
oop_and_class2.py
oop_and_class2x.py
pdsXu.zip
pdsXuv14.py
pdsx_exception.py
pipe.py
pipe_monitor_gui.py
reply_extension.py
save.py
save_load_system.py
save_load_system2.py
sqlite.py
tree.py
tree2.py
tree3.py
```

## 🛠 Kurulum

1. **Depoyu klonlayın**:

   ```bash
   git clone https://github.com/metedinler/pdsXbasic.git
   cd pdsXbasic
   ```

2. **Bağımlılıkları yükleyin**:

   ```bash
   pip install -r requirements.txt  # Gerekli paketleri listeleyen dosya
   ```

3. **Yorumlayıcıyı çalıştırın**:

   ```bash
   python pdsXuv14.py  # BASIC yorumlayıcısını başlatır
   ```

## 📝 Kullanım

* **Kod Derleme**: `.basX` uzantılı betikleri `bytecode_compiler.py` ile bytecode’a çevirin.
* **Bytecode Çalıştırma**: `bytecode_manager.py` veya ana yorumlayıcı ile yükleyip çalıştırın.
* **Etkileşimli Mod**: `pdsXuv14.py` ile REPL ortamına girin; komut, olay ve GUI izleme desteği sağlar.
* **Betik Entegrasyonu**: `pipe`, `event`, `memory_manager` gibi modülleri Python projelerinize import ederek kullanın.

## 📐 Mimari & Modüller

| Modül Grubu             | Dosyalar                                           | Amaç                                             |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------ |
| Çekirdek Motor          | `pdsXuv14.py`, `bytecode_*`, `pdsx_exception.py`   | Yorumlayıcı, derleyici, istisna yönetimi         |
| Veri & Yapılar          | `data_structures.py`, `tree.py`, `graph.py`        | Özel koleksiyon ve grafik/ağaç algoritmaları     |
| Bellek & Düşük Seviye   | `memory_manager.py`, `lowlevel.py`                 | Bellek ayırma, gösterge benzeri işlemler         |
| Eşzamanlılık            | `libx_concurrency.py`, `multithreading_process.py` | İş parçacığı/süreç yönetimi, asyncio             |
| Olay & Boru Hattı       | `event.py`, `pipe.py`                              | Olay döngüsü, boru hattı/pipe yürütme desenleri  |
| Kalıcılık               | `save_load_system.py`, `save.py`                   | Sıkıştırma ve kodlama ile durum kaydetme/yükleme |
| Veritabanı              | `database_sql_isam.py`, `sqlite.py`, `lib_db.py`   | SQL/ISAM ve SQLite işlemleri                     |
| Loglama & Hata Ayıklama | `f11_backtrace_logger.py`, `f12_timer_manager.py`  | Hata günlükleri, performans ölçümleri            |
| Uzantılar (`libx_`)     | GUI, JIT, mantık, ML, ağ, NLP modülleri            | Alan bazlı genişletmeler                         |
| Raporlama               | `export_report_doc.py`, `.tex`                     | DOC/TeX rapor üretme                             |

## 🤝 Katkıda Bulunma

Katkılar, hata bildirimleri ve özellik istekleri memnuniyetle karşılanır:

1. Depoyu fork’layın
2. Özellik dalı açın (`git checkout -b feature/Özelliğiniz`)
3. Değişiklikleri commit’leyin (`git commit -m "Özelliğinizi ekle"`)
4. Branch’i push’layın (`git push origin feature/Özelliğiniz`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

*ChatGPT tarafından depoyu inceleyip oluşturulmuştur.*

zmetedinler@gmail.com

