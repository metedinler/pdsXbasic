"""
PDSxV15 Event Sistemi Kullanım Kılavuzu (eventxkullanim.py)
-----------------------------------------------------------
Bu dosya, PDSxV15'in ultra güçlü event (olay) sisteminin temel ve gelişmiş komutlarının nasıl kullanılacağını, örneklerle ve açıklamalarla anlatır.
Her komutun başında, ne işe yaradığını ve nasıl çalıştığını açıklayan detaylı REMARK (açıklama) satırları bulunur.

Başlangıç: EventManager ve EventCompat kullanımı
------------------------------------------------
"""

from eventx import EventManager, EventCompat

# REMARK: EventManager, olayları kaydetmek, tetiklemek, analiz etmek, zamanlamak, önceliklendirmek ve daha fazlası için ana sınıftır.
# REMARK: EventCompat, eski event.py API'si ile uyumluluk sağlar (gerekirse kullanılır).

# Olay sistemi başlatılır (örnek bir interpreter nesnesi ile):
interpreter = ...  # PDSxInterpreter veya benzeri bir nesne olmalı
em = EventManager(interpreter)

# 1. Olay Kaydı (EVENT REGISTER)
# REMARK: Bir olayı sisteme kaydeder. Handler, olay tetiklendiğinde çalışacak fonksiyonun adıdır.
em.register_event("sensor_alert", "handle_alert", alias="myalert")

# 2. Olay Tetikleme (EVENT TRIGGER)
# REMARK: Kayıtlı bir olayı tetikler (örneğin bir sensör alarmı geldiğinde).
import asyncio
asyncio.run(em.trigger_event("sensor_alert"))

# 3. Olay Analizi (EVENT ANALYZE)
# REMARK: Bir olay üzerinde analiz (ör. korelasyon, anomali, NLP, uzamsal) yapar.
asyncio.run(em.analyze_event({"event_id": "sensor_alert"}, method="correlation", config={"type": "granger"}))

# 4. Olay Öngörüsü (EVENT FORECAST)
# REMARK: Olay verisiyle öngörü (ör. LSTM, ARIMA, kuantum) yapar.
asyncio.run(em.forecast_event({"event_id": "sensor_alert"}, model="lstm", horizon=10, config={}))

# 5. Anomali Tespiti (EVENT DETECT ANOMALY)
# REMARK: Olayda anomali olup olmadığını tespit eder.
asyncio.run(em.detect_anomaly({"event_id": "sensor_alert"}, method="z_score", config={}))

# 6. Olay Görselleştirme (EVENT VISUALIZE)
# REMARK: Olayı veya analiz sonucunu 3D/4D grafik, event tree vb. ile görselleştirir.
asyncio.run(em.visualize_event({"event_id": "sensor_alert"}, vis_type="event_tree", config={"output": "tree.html"}))

# 7. Olay Zamanlama (EVENT SCHEDULE)
# REMARK: Bir olayı belirli bir zamanda veya periyodik olarak tetikler.
asyncio.run(em.schedule("sensor_alert", "2025-06-12 10:00", config="repeat: daily"))

# 8. Olay Önceliklendirme (EVENT PRIORITIZE)
# REMARK: Bir olayın kuyruğundaki önceliğini değiştirir.
asyncio.run(em.prioritize("sensor_alert", priority=5, config=""))

# 9. Olayda Kesme (EVENT INTERRUPT)
# REMARK: Donanım/yazılım kesmesi gibi bir olay tetikler.
asyncio.run(em.interrupt("sensor_alert", interrupt_type="hardware", config="urgent: true"))

# 10. Olay Örneği Oluşturma/Yok Etme (EVENT INSTANCE)
# REMARK: Bir olayın yeni bir örneğini oluşturur veya yok eder (eşzamanlılık için).
asyncio.run(em.instance_manager.create_instance("sensor_alert", "alert_1", priority=1.0))
asyncio.run(em.instance_manager.destroy_instance("sensor_alert", "alert_1"))

# 11. Olay Zinciri (EVENT CHAIN)
# REMARK: Bir olay tetiklendiğinde başka bir olayı otomatik tetiklemek için zincir kurar.
em.dependency_graph.add_edge("sensor_alert", "system_alert")

# 12. Olayı Disk Kuyruğuna Toplu Yayınla (BULK_PUBLISH)
# REMARK: Birden fazla olayı disk tabanlı kuyruk veya ZeroMQ ile topluca yayınlar.
bulk_events = [
    {"event_id": "sensor_alert", "priority": 1.0, "data": {"value": 42}},
    {"event_id": "system_alert", "priority": 2.0, "data": {"value": 99}}
]
asyncio.run(em.bulk_publish(bulk_events, use_disk_queue=True, use_zmq=False))

# 13. Olay Bayrağı Yönetimi (FLAG SET/CLEAR)
# REMARK: Olaylara özel bayraklar atanabilir ve bus ile sistem çapında duyurulur.
em.flag_manager.set_flag("sensor_alert", "URGENT", True)
em.flag_manager.clear_flag("sensor_alert", "URGENT")

# 14. Olayı Kuyruktan İşle (process_spool)
# REMARK: Disk tabanlı kuyruktaki bekleyen olayları işleme alır.
asyncio.run(em.process_spool(max_count=10))

# 15. Olayı Zamanlayıcıda Arka Planda İşle (start_spool_processor)
# REMARK: Disk kuyruğunu arka planda sürekli işler.
em.start_spool_processor(interval=1.0, max_count=10)

# 16. Olay Durumu ve Bilgisi (EVENT_STATUS, EVENT_INFO)
# REMARK: Bir olayın tetiklenip tetiklenmediğini ve detaylarını sorgular.
status = em.event_status("sensor_alert")
info = em.event_info("sensor_alert")
print(f"Olay durumu: {status}, Bilgi: {info}")

# 17. Komut Satırı ile Olay Komutları (parse_event_command)
# REMARK: Tüm PDSx komutları string olarak parse_event_command ile çalıştırılabilir.
em.parse_event_command('EVENT REGISTER sensor_alert "handle_alert" ALIAS myalert', interpreter)
em.parse_event_command('EVENT TRIGGER sensor_alert', interpreter)
em.parse_event_command('EVENT SCHEDULE sensor_alert AT "2025-06-12 10:00" CONFIG "repeat: daily"', interpreter)

# 18. Eski API ile Uyumluluk (EventCompat)
# REMARK: Eski event.py API'si ile uyumlu kullanım için EventCompat kullanılabilir.
compat = EventCompat(em)
compat.parse_event_command('EVENTX BULK_PUBLISH [{"event_id": "sensor_alert"}] DISK', interpreter)

# 19. Gelişmiş: Olay Bus Sistemi ile Entegrasyon
# REMARK: Bayrak değişiklikleri ve olaylar bus.py ile sistem çapında yayınlanabilir.
# (BusManager entegrasyonu için em.flag_manager.bus_manager = bus_manager yapılmalıdır.)

# 20. Gelişmiş: Olay Zinciri, Grup, Broadcast, Sync, Aggregate, Flow Analyze gibi yeni komutlar
# REMARK: Bu komutlar için EventManager'a yeni metodlar eklenebilir ve parse_event_command ile çağrılabilir.
# Örnek: em.parse_event_command('EVENT BROADCAST system_alert', interpreter)

# =====================
# PDSx Komutları ile Olay Sistemi Kullanımı (Yüksek Seviyeli ve Entegre Örnekler)
# =====================

# REMARK: PDSx komutları, EventManager'ın parse_event_command fonksiyonu ile doğrudan string olarak çalıştırılabilir.
# REMARK: Komutlar, klasik BASIC gibi yazılır ve sistemde olayları, analizleri, zamanlamayı, zincirlemeyi, bayrakları ve daha fazlasını yönetir.

# --- Temel Komutlar ---
# REMARK: Olay kaydı, tetikleme ve analiz için temel PDSx komutları:
em.parse_event_command('EVENT REGISTER temp_alert "handle_temp" ALIAS temp', interpreter)
em.parse_event_command('EVENT TRIGGER temp_alert', interpreter)
em.parse_event_command('EVENT ANALYZE temp_alert METHOD "correlation" CONFIG "type: granger"', interpreter)
em.parse_event_command('EVENT FORECAST temp_alert MODEL "lstm" HORIZON "10"', interpreter)
em.parse_event_command('EVENT DETECT ANOMALY temp_alert METHOD "z_score"', interpreter)
em.parse_event_command('EVENT VISUALIZE temp_alert TYPE "event_tree" CONFIG "output: tree.html"', interpreter)

# --- Zamanlama, Öncelik, Kesme, Bayrak ---
# REMARK: Olayı zamanla, önceliğini değiştir, kesme tetikle, bayrak ata/temizle:
em.parse_event_command('EVENT SCHEDULE temp_alert AT "2025-06-12 12:00" CONFIG "repeat: daily"', interpreter)
em.parse_event_command('EVENT PRIORITIZE temp_alert LEVEL 10 CONFIG ""', interpreter)
em.parse_event_command('EVENT INTERRUPT temp_alert TYPE "hardware" CONFIG "urgent: true"', interpreter)
# REMARK: Bayrak atama/temizleme (örnek, doğrudan FlagManager ile de yapılabilir):
em.flag_manager.set_flag("temp_alert", "URGENT", True)
em.flag_manager.clear_flag("temp_alert", "URGENT")

# --- Olay Zinciri, Grup, Broadcast, Sync, Aggregate, Flow Analyze ---
# REMARK: Olay zinciri, grup, broadcast, senkronizasyon ve akış analizi için ileri komutlar:
em.parse_event_command('EVENT CHAIN temp_alert TO system_alert', interpreter)
# (Grup, broadcast, sync, aggregate, flow analyze gibi komutlar için EventManager'a ek metodlar eklenmeli ve burada çağrılmalı)
# em.parse_event_command('EVENT GROUP alerts temp_alert system_alert', interpreter)
# em.parse_event_command('EVENT BROADCAST system_alert', interpreter)
# em.parse_event_command('EVENT SYNC temp_alert WITH system_alert', interpreter)
# em.parse_event_command('EVENT AGGREGATE temp_alert METHOD "average"', interpreter)
# em.parse_event_command('EVENT FLOW ANALYZE system_events', interpreter)

# --- Olay Okuma ve Desen Tanımlama ---
# REMARK: Olay verisini dosyadan oku ve özel desen tanımla:
em.parse_event_command('EVENT READ SOURCE "iot_events.json" FORMAT "json" CONFIG "remove_duplicates: true" AS events', interpreter)
em.parse_event_command('EVENT PATTERN CUSTOM temp_alert RULE "value > 100" AS high_temp_pattern', interpreter)

# --- Gerçek Zamanlı Akış ve Abonelik ---
# REMARK: Gerçek zamanlı akışlardan olay oku ve bus üzerinden abone ol:
# em.parse_event_command('EVENT STREAM ANALYZE SOURCE "mqtt://broker/topic" CALLBACK "handle_stream" AS stream_events', interpreter)
# em.parse_event_command('EVENT SUBSCRIBE alerts CALLBACK "handle_alert"', interpreter)
# em.parse_event_command('EVENT PUBLISH alerts DATA "Alarm!"', interpreter)

# --- Kuantum ve Federatif Analiz ---
# REMARK: Kuantum ve federatif analiz komutları:
em.parse_event_command('EVENT QUANTUM ANALYZE temp_alert METHOD "quantum_corr"', interpreter)
em.parse_event_command('EVENT FEDERATED ANALYZE temp_alert METHOD "federated"', interpreter)

# --- Koşullu Olay ve Döngü ---
# REMARK: Koşullu olay ve döngüsel tetikleme (örnek, interpreter tarafından destekleniyorsa):
# em.parse_event_command('EVENT COND temp_alert WHEN "value > 100" ... ENDCOND', interpreter)
# em.parse_event_command('EVENT LOOP temp_alert UNTIL "value < 10" ... ENDLOOP', interpreter)

# --- Bayrak ve Bus Entegrasyonu ---
# REMARK: Bayrak değişiklikleri bus ile sistem çapında yayınlanır. BusManager entegrasyonu için:
# bus_manager = ...  # bus.py'den alınır
# em.flag_manager.bus_manager = bus_manager
# em.bus_manager = bus_manager

# --- Gelişmiş Senaryo: Olay Tabanlı Otomasyon ---
# REMARK: Tüm bu komutlar, PDSx ekosisteminde olay tabanlı otomasyon, dağıtık işleme, gerçek zamanlı izleme ve güvenlik için kullanılabilir.
# Örneğin, bir IoT ağı için olaylar zincirlenip, analiz edilip, anomali tespit edilip, sonuçlar görselleştirilebilir ve sistem çapında broadcast yapılabilir.

# --- Notlar ---
# - Komutlar, klasik BASIC gibi string olarak yazılır ve parse_event_command ile çalıştırılır.
# - Her komutun başında REMARK ile ne yaptığı açıklanır.
# - Gelişmiş entegrasyonlar için EventManager'a yeni metodlar eklenebilir.
# - Daha fazla örnek ve ileri seviye kullanım için PDSx dokümantasyonuna bakınız.

"""
Bu dosya, PDSxV15 event sisteminin tüm anahtar komutlarını ve kullanım örneklerini, sıfırdan başlayan bir kullanıcı için açıklamalı olarak sunar.
Her komutun başında REMARK ile detaylı açıklama verilmiştir.
Daha fazla örnek ve ileri seviye kullanım için PDSx dokümantasyonuna bakınız.
"""
