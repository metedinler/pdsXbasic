# bus.py - PDS-X BASIC v15 Ultra Güçlü Veri Yolu Kütüphanesi
# Version: 2.2.0
# Date: June 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import asyncio
import threading
import time
import json
import uuid
import logging
import re
from collections import defaultdict, deque
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union, Callable
from pdsx_exception2 import PdsXPipeError as PdsXBusError  # PdsXBusError yerine PdsXPipeError kullanıyoruz
import zmq.asyncio
from prometheus_client import Counter, Gauge, start_http_server
import paho.mqtt.client as mqtt
from kafka import KafkaConsumer, KafkaProducer
import grpc
import websocket
import qiskit
from qiskit import QuantumCircuit, execute, Aer

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_bus.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("bus")

# BUS_DATA Veri Yapısı
BUS_DATA = {
    "topic": str,         # Veri yolu konusu
    "status": str,        # Durum (active, paused, vb.)
    "mode": str,          # async/sync
    "config": dict,       # Yapılandırma seçenekleri
    "subscribers": list,  # Abone kimlikleri
    "flags": dict,        # Bayraklar (PUBLISHED, SUBSCRIBED, vb.)
    "priority": float,    # Öncelik seviyesi
    "timestamp": float,   # Oluşturma zamanı
    "data": Any,          # Yayınlanan veri
    "subroutines": dict   # Alt programlar
}

class FlagManager:
    """Dinamik bayrak yönetimi sınıfı."""
    def __init__(self):
        self.flags: Dict[str, Dict[str, bool]] = defaultdict(lambda: {
            "QUEUED": False, "PUBLISHED": False, "SUBSCRIBED": False, "DELIVERED": False,
            "FAILED": False, "PAUSED": False, "SECURED": False, "MONITORED": False,
            "LOCKED": False, "UNLOCKED": True
        })
        self._lock = threading.RLock()

    def set_flag(self, topic: str, flag: str, value: bool = True) -> None:
        """Belirtilen konu için bayrağı ayarlar."""
        with self._lock:
            self.flags[topic][flag] = value
            log.debug(f"Bayrak ayarlandı: {topic}, {flag} = {value}")

    def get_flag(self, topic: str, flag: str) -> bool:
        """Belirtilen konu için bayrağın değerini döndürür."""
        with self._lock:
            return self.flags[topic].get(flag, False)

    def clear_flag(self, topic: str, flag: str) -> None:
        """Belirtilen konu için bayrağı temizler."""
        with self._lock:
            self.flags[topic].pop(flag, None)
            log.debug(f"Bayrak temizlendi: {topic}, {flag}")

class BusInstanceManager:
    """Veri yolu abonelik örnek yönetimi sınıfı."""
    def __init__(self, max_instances: int = 65536):
        self.instances: Dict[str, List[Dict]] = defaultdict(list)
        self.max_instances = max_instances
        self.lock = asyncio.Lock()

    async def create_instance(self, topic: str, instance_id: str, priority: float) -> bool:
        """Yeni bir abonelik örneği oluşturur."""
        async with self.lock:
            if len(self.instances[topic]) >= self.max_instances:
                raise PdsXBusError(f"Maksimum abonelik sınırına ulaşıldı: {topic} (BUS701)", context={"source": "create_instance"})
            self.instances[topic].append({"id": instance_id, "priority": priority, "start_time": time.time()})
            log.debug(f"Abonelik örneği oluşturuldu: {topic}, Örnek: {instance_id}")
            return True

    async def destroy_instance(self, topic: str, instance_id: str) -> None:
        """Belirtilen abonelik örneğini yok eder."""
        async with self.lock:
            self.instances[topic] = [inst for inst in self.instances[topic] if inst["id"] != instance_id]
            log.debug(f"Abonelik örneği yok edildi: {topic}, Örnek: {instance_id}")

class BusManager:
    """PDS-X BASIC v15 Ultra Güçlü Veri Yolu İşleme sınıfı."""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.lock = threading.RLock()
        self.async_lock = asyncio.Lock()
        self.topics: Dict[str, Dict] = {}
        self.buffers: Dict[str, Any] = {}
        self.event_queue = asyncio.PriorityQueue()
        self.instance_manager = BusInstanceManager()
        self.flag_manager = FlagManager()
        self.event_loop = asyncio.get_event_loop()
        self.zmq_ctx = zmq.asyncio.Context()
        self.zmq_sock = self.zmq_ctx.socket(zmq.asyncio.PUB)
        self.zmq_sock.bind("tcp://*:5558")
        self.mqtt_client = mqtt.Client(client_id=f"pdsx_{uuid.uuid4().hex[:8]}")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            retries=3,
            acks='all'
        )
        self.prom_counter = Counter("busx_messages_total", "Toplam işlenen mesaj sayısı", ["topic"])
        self.prom_status = Gauge("busx_status", "Veri yolu durumu", ["topic", "status"])
        start_http_server(8003)
        self._init_bus()

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT bağlantı callback fonksiyonu."""
        log.debug(f"MQTT bağlantısı sağlandı: Kod {rc}")
        if rc != 0:
            log.error(f"MQTT bağlantı hatası: Kod {rc}")
            client.reconnect()

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT bağlantı kesilme callback fonksiyonu."""
        log.warning(f"MQTT bağlantısı kesildi: Kod {rc}")
        try:
            client.reconnect()
        except Exception as e:
            log.error(f"MQTT yeniden bağlanma hatası: {str(e)}")

    def _init_bus(self) -> None:
        """Veri yolu sistemini başlatır."""
        try:
            with self.lock:
                self.interpreter.object_counter["BUS_INIT"] = self.interpreter.object_counter.get("BUS_INIT", 0) + 1
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
            log.info("Veri yolu yöneticisi başlatıldı")
            asyncio.create_task(self._process_queue())
        except Exception as e:
            raise PdsXBusError(f"Veri yolu başlatma hatası: {str(e)} (BUS001)", context={"source": "_init_bus"})

    @lru_cache(maxsize=1024)
    async def _process_queue(self) -> None:
        """Veri yolu kuyruğunu işler."""
        while True:
            try:
                priority, bus_data = await self.event_queue.get()
                topic = bus_data["topic"]
                instance_id = bus_data.get("instance_id")
                await self._process_message(topic, instance_id, bus_data["data"])
                self.event_queue.task_done()
                self.prom_counter.labels(topic=topic).inc()
            except PdsXBusError as e:
                log.error(f"Kuyruk işleme hatası: {str(e)}")
            except Exception as e:
                log.error(f"Beklenmeyen kuyruk hatası: {str(e)} (BUS900)", context={"source": "_process_queue"})

    @lru_cache(maxsize=512)
    async def _process_message(self, topic: str, instance_id: str, data: Any) -> None:
        """Veri yolu mesajını işler."""
        try:
            async with self.async_lock:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS002)", context={"source": "_process_message"})
                for subscriber in self.topics[topic]["subscribers"]:
                    callback = subscriber["name"]
                    if callback in self.interpreter.function_table:
                        await self.interpreter.function_table[callback](data)
                    else:
                        log.warning(f"Geri çağrım bulunamadı: {callback} for topic {topic}")
                self.flag_manager.set_flag(topic, "DELIVERED")
                await self.instance_manager.destroy_instance(topic, instance_id)
                log.debug(f"Mesaj işlendi: {topic}, Örnek: {instance_id}")
            except PdsXBusError as e:
                self.flag_manager.set_flag(topic, "FAILED")
                raise
            except Exception as e:
                self.flag_manager.set_flag(topic, "FAILED")
                raise PdsXBusError(f"Mesaj işleme hatası: {str(e)} (BUS003)", context={"source": "_process_message"})

    @lru_cache(maxsize=1024)
    def _parse_config(self, config: str) -> Dict[str, Any]:
        """Konfigürasyon dizesini sözlüğe çevirir."""
        try:
            config_dict = {}
            if config:
                pairs = config.split()
                for pair in pairs:
                    if ":" in pair:
                        key, value = pair.split(":", 1)
                        if key.lower() in ("max_instances", "max_recursion"):
                            value = int(value)
                        elif key.lower() in ("priority"):
                            value = float(value)
                        elif key.lower() in ("encrypt", "live"):
                            value = value.lower() == "true"
                        config_dict[key.lower()] = value
            return config_dict
        except Exception as e:
            raise PdsXBusError(f"Konfigürasyon ayrıştırma hatası: {str(e)} (BUS004)", context={"source": "_parse_config"})

    async def define(self, topic: str, config: str, alias: Optional[str] = None) -> str:
        """BUS DEFINE <topic> CONFIG "<options>" AS <var>
        Yeni bir veri yolu konusu tanımlar."""
        async with self.async_lock:
            try:
                if topic in self.topics:
                    raise PdsXBusError(f"Konu zaten mevcut: {topic} (BUS005)", context={"source": "define"})
                config_dict = self._parse_config(config)
                topic = topic or f"topic_{uuid.uuid4().hex[:8]}"
                self.topics[topic] = {
                    "topic": topic,
                    "status": "ready",
                    "mode": config_dict.get("mode", "async"),
                    "config": config_dict,
                    "subscribers": [],
                    "instances": [],
                    "max_instances": config_dict.get("max_instances", 65536),
                    "priority": config_dict.get("priority", 1.0),
                    "timestamp": time.time(),
                    "alias": alias,
                    "subroutines": {},
                    "commands": []
                }
                self.buffers[topic] = None
                self.flag_manager.set_flag(topic, "QUEUED")
                with self.lock:
                    self.interpreter.object_counter["BUS_DEFINE"] = self.interpreter.object_counter.get("BUS_DEFINE", 0) + 1
                    self.interpreter.object_registry[topic] = {
                        "type": "BUS_DATA",
                        "name": topic,
                        "atom": f"bus:{topic}"
                    }
                log.debug(f"Konu tanımlandı: {topic}, Konfigürasyon: {config_dict}")
                return topic
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Konu tanımlama hatası: {str(e)} (BUS006)", context={"source": "define"})

    @lru_cache(maxsize=512)
    async def publish(self, topic: str, data: Any, config: str = "") -> None:
        """BUS PUBLISH <topic> DATA "<data>" CONFIG "<options>"
        Belirtilen konuya veri yayınlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS007)", context={"source": "publish"})
                config_dict = self._parse_config(config)
                instance_id = f"instance_{uuid.uuid4().hex[:8]}"
                await self.instance_manager.create_instance(topic, instance_id, self.topics[topic]["priority"])
                if config_dict.get("encrypt", False):
                    security_manager = self.interpreter.get_manager("security")
                    data = await security_manager.encrypt_data(data)
                self.buffers[topic] = data
                await self.event_queue.put((
                    self.topics[topic]["priority"],
                    {"topic": topic, "instance_id": instance_id, "data": data}
                ))
                await self.zmq_sock.send_json({"topic": topic, "data": data})
                if config_dict.get("mqtt", False):
                    self.mqtt_client.publish(topic, json.dumps(data))
                if config_dict.get("kafka", False):
                    self.kafka_producer.send(topic, json.dumps(data).encode('utf-8'))
                if config_dict.get("federated", False):
                    federated_manager = self.interpreter.get_manager("federated")
                    await federated_manager.share_data(topic, data)
                self.flag_manager.set_flag(topic, "PUBLISHED")
                self.prom_status.labels(topic=topic, status="published").set(1)
                log.debug(f"Veri yayınlandı: {topic}, Veri: {data}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Yayınlama hatası: {str(e)} (BUS008)", context={"source": "publish"})

    @lru_cache(maxsize=512)
    async def subscribe(self, topic: str, callback: str, config: str = "") -> None:
        """BUS SUBSCRIBE <topic> CALLBACK "<callback>" CONFIG "<options>"
        Belirtilen konuya abone olur."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS009)", context={"source": "subscribe"})
                config_dict = self._parse_config(config)
                subscriber_id = f"sub_{uuid.uuid4().hex[:8]}"
                self.topics[topic]["subscribers"].append({"id": subscriber_id, "name": callback})
                self.flag_manager.set_flag(topic, "SUBSCRIBED")
                log.debug(f"Abone olundu: {topic}, Geri çağrı: {callback}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Abonelik hatası: {str(e)} (BUS010)", context={"source": "subscribe"})

    async def unsubscribe(self, topic: str, subscriber_id: Optional[str] = None) -> None:
        """BUS UNSUBSCRIBE <topic>
        Belirtilen konudan aboneliği iptal eder."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS011)", context={"source": "unsubscribe"})
                if subscriber_id:
                    self.topics[topic]["subscribers"] = [
                        sub for sub in self.topics[topic]["subscribers"] if sub["id"] != subscriber_id
                    ]
                else:
                    self.topics[topic]["subscribers"] = []
                if not self.topics[topic]["subscribers"]:
                    self.flag_manager.clear_flag(topic, "SUBSCRIBED")
                log.debug(f"Abonelik iptal edildi: {topic}, Abone: {subscriber_id}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Abonelik iptal hatası: {str(e)} (BUS012)", context={"source": "unsubscribe"})

    async def start(self, topic: str, mode: str = "async", config: str = "") -> None:
        """BUS START <topic> MODE "<async/sync>" CONFIG "<options>"
        Konu yayınını başlatır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS013)", context={"source": "start"})
                config_dict = self._parse_config(config)
                self.topics[topic]["mode"] = mode
                self.topics[topic]["status"] = "active"
                self.flag_manager.set_flag(topic, "RUNNING")
                self.prom_status.labels(topic=topic, status="active").set(1)
                log.debug(f"Konu başlatıldı: {topic}, Mod: {mode}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Konu başlatma hatası: {str(e)} (BUS014)", context={"source": "start"})

    async def stop(self, topic: str, config: str = "") -> None:
        """BUS STOP <topic> CONFIG "<options>"
        Konu yayınını durdurur."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS015)", context={"source": "stop"})
                config_dict = self._parse_config(config)
                self.topics[topic]["status"] = "stopped"
                self.flag_manager.clear_flag(topic, "RUNNING")
                self.prom_status.labels(topic=topic, status="stopped").set(0)
                log.debug(f"Konu durduruldu: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Konu durdurma hatası: {str(e)} (BUS016)", context={"source": "stop"})

    async def pause(self, topic: str, config: str = "") -> None:
        """BUS PAUSE <topic> CONFIG "<options>"
        Konu yayınını duraklatır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS017)", context={"source": "pause"})
                config_dict = self._parse_config(config)
                self.topics[topic]["status"] = "paused"
                self.flag_manager.set_flag(topic, "PAUSED")
                self.prom_status.labels(topic=topic, status="paused").set(1)
                log.debug(f"Konu duraklatıldı: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Konu duraklatma hatası: {str(e)} (BUS018)", context={"source": "pause"})

    async def resume(self, topic: str, config: str = "") -> None:
        """BUS RESUME <topic> CONFIG "<options>"
        Konu yayınını devam ettirir."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS019)", context={"source": "resume"})
                config_dict = self._parse_config(config)
                self.topics[topic]["status"] = "active"
                self.flag_manager.clear_flag(topic, "PAUSED")
                self.flag_manager.set_flag(topic, "RUNNING")
                self.prom_status.labels(topic=topic, status="active").set(1)
                log.debug(f"Konu devam ettirildi: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Konu devam ettirme hatası: {str(e)} (BUS020)", context={"source": "resume"})

    async def monitor(self, topic: str, config: str, var: Optional[str] = None) -> str:
        """BUS MONITOR <topic> CONFIG "<options>" AS <var>
        Konuyu izler ve monitör bağlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS021)", context={"source": "monitor"})
                config_dict = self._parse_config(config)
                monitor_manager = self.interpreter.get_manager("monitor")
                monitor_id = await monitor_manager.connect(
                    topic, config_dict.get("monitor_id", f"monitor_{uuid.uuid4().hex[:8]}"), config
                )
                self.flag_manager.set_flag(topic, "MONITORED")
                if var:
                    self.interpreter.current_scope()[var] = monitor_id
                log.debug(f"Konu izleniyor: {topic}, Monitör: {monitor_id}")
                return monitor_id
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Monitör hatası: {str(e)} (BUS022)", context={"source": "monitor"})

    async def secure(self, topic: str, config: str) -> None:
        """SECURE BUS <topic> CONFIG "<encrypt: true>"
        Konuyu şifreler."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS023)", context={"source": "secure"})
                config_dict = self._parse_config(config)
                security_manager = self.interpreter.get_manager("security")
                await security_manager.encrypt_bus(topic, config)
                self.flag_manager.set_flag(topic, "SECURED")
                log.debug(f"Konu güvenli hale getirildi: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Güvenlik hatası: {str(e)} (BUS024)", context={"source": "secure"})

    async def unsecure(self, topic: str, config: str) -> None:
        """UNSECURE BUS <topic> CONFIG "<options>"
        Konu şifrelemesini kaldırır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS025)", context={"source": "unsecure"})
                config_dict = self._parse_config(config)
                security_manager = self.interpreter.get_manager("security")
                await security_manager.decrypt_bus(topic, config)
                self.flag_manager.clear_flag(topic, "SECURED")
                log.debug(f"Konu güvensiz hale getirildi: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Güvensiz hale getirme hatası: {str(e)} (BUS026)", context={"source": "unsecure"})

    async def event_trigger(self, topic: str, event: str, config: str = "") -> None:
        """BUS EVENT TRIGGER <topic> <event> CONFIG "<options>"
        Konudan bir olay tetikler."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS027)", context={"source": "event_trigger"})
                config_dict = self._parse_config(config)
                event_manager = self.interpreter.get_manager("event")
                await event_manager.trigger(event)
                log.debug(f"Olay tetiklendi: {topic}, Olay: {event}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Olay tetikleme hatası: {str(e)} (BUS028)", context={"source": "event_trigger"})

    async def recurse(self, topic: str, depth: int, config: str = "") -> None:
        """BUS RECURSE <topic> DEPTH <depth> CONFIG "<options>"
        Konuya rekürsif veri aktarımı yapar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS029)", context={"source": "recurse"})
                config_dict = self._parse_config(config)
                self.flag_manager.set_flag(topic, "RECURSING")
                for _ in range(depth):
                    await self.publish(topic, self.buffers.get(topic), config)
                self.flag_manager.clear_flag(topic, "RECURSING")
                log.debug(f"Rekürsif yayın: {topic}, Derinlik: {depth}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Rekürsif yayın hatası: {str(e)} (BUS030)", context={"source": "recurse"})

    async def prioritize(self, topic: str, level: float, config: str = "") -> None:
        """BUS PRIORITIZE <topic> LEVEL <priority> CONFIG "<options>"
        Konu önceliğini ayarlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS031)", context={"source": "prioritize"})
                config_dict = self._parse_config(config)
                self.topics[topic]["priority"] = level
                self.flag_manager.set_flag(topic, "PRIORITY_UPDATED")
                log.debug(f"Öncelik güncellendi: {topic}, Seviye: {level}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Öncelik güncelleme hatası: {str(e)} (BUS032)", context={"source": "prioritize"})

    async def lock(self, topic: str, config: str = "") -> None:
        """BUS LOCK <topic> CONFIG "<options>"
        Konuyu kilitler."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS033)", context={"source": "lock"})
                config_dict = self._parse_config(config)
                self.flag_manager.set_flag(topic, "LOCKED")
                self.flag_manager.clear_flag(topic, "UNLOCKED")
                log.debug(f"Konu kilitlendi: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Kilit hatası: {str(e)} (BUS034)", context={"source": "lock"})

    async def unlock(self, topic: str, config: str = "") -> None:
        """BUS UNLOCK <topic> CONFIG "<options>"
        Konu kilidini açar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS035)", context={"source": "unlock"})
                config_dict = self._parse_config(config)
                self.flag_manager.set_flag(topic, "UNLOCKED")
                self.flag_manager.clear_flag(topic, "LOCKED")
                log.debug(f"Konu kilidi açıldı: {topic}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Kilit açma hatası: {str(e)} (BUS036)", context={"source": "unlock"})

    async def data_send(self, topic: str, target: str, data: Any, config: str = "") -> None:
        """BUS DATA SEND <topic> TO "<target>" DATA "<data>" CONFIG "<options>"
        Belirli bir hedefe veri gönderir."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS037)", context={"source": "data_send"})
                config_dict = self._parse_config(config)
                if target.startswith("topic:"):
                    target_topic = target.replace("topic:", "")
                    await self.publish(target_topic, data, config)
                else:
                    await self.zmq_sock.send_json({"target": target, "data": data})
                log.debug(f"Veri gönderildi: {topic}, Hedef: {target}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Veri gönderme hatası: {str(e)} (BUS038)", context={"source": "data_send"})

    async def data_receive(self, topic: str, source: str, config: str = "") -> None:
        """BUS DATA RECEIVE <topic> FROM "<source>" CONFIG "<options>"
        Belirtilen kaynaktan veri alır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS039)", context={"source": "data_receive"})
                config_dict = self._parse_config(config)
                if source.startswith("topic:"):
                    source_topic = source.replace("topic:", "")
                    if source_topic in self.topics:
                        self.buffers[topic] = self.buffers.get(source_topic)
                else:
                    consumer = KafkaConsumer(
                        source,
                        bootstrap_servers=['localhost:9092'],
                        auto_offset_reset='latest',
                        enable_auto_commit=True,
                        consumer_timeout_ms=1000
                    )
                    for msg in consumer:
                        self.buffers[topic] = json.loads(msg.value.decode('utf-8'))
                        break
                log.debug(f"Veri alındı: {topic}, Kaynak: {source}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Veri alma hatası: {str(e)} (BUS040)", context={"source": "data_receive"})

    async def iot_connect(self, topic: str, device_id: str, config: str = "") -> None:
        """BUS IOT CONNECT <topic> DEVICE "<device_id>" CONFIG "<options>"
        Konuyu bir IoT cihazına bağlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS041)", context={"source": "iot_connect"})
                config_dict = self._parse_config(config)
                iot_manager = self.interpreter.get_manager("iot")
                await iot_manager.connect(device_id, config)
                self.buffers[topic] = iot_manager.get_data(device_id)
                log.debug(f"IoT bağlantısı: {topic}, Cihaz: {device_id}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"IoT bağlantı hatası: {str(e)} (BUS042)", context={"source": "iot_connect"})

    async def cloud_sync(self, topic: str, cloud_service: str, config: str = "") -> None:
        """BUS CLOUD SYNC <topic> TO "<cloud_service>" CONFIG "<options>"
        Konu verilerini buluta senkronize eder."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS043)", context={"source": "cloud_sync"})
                config_dict = self._parse_config(config)
                network_manager = self.interpreter.get_manager("network")
                await network_manager.sync_to_cloud(self.buffers.get(topic), cloud_service, config)
                log.debug(f"Bulut senkronizasyonu: {topic}, Servis: {cloud_service}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Bulut senkronizasyon hatası: {str(e)} (BUS044)", context={"source": "cloud_sync"})

    async def network_connect(self, topic: str, network: str, config: str = "") -> None:
        """BUS NETWORK CONNECT <topic> TO "<network>" CONFIG "<options>"
        Konuyu bir ağa bağlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS045)", context={"source": "network_connect"})
                config_dict = self._parse_config(config)
                network_manager = self.interpreter.get_manager("network")
                await network_manager.connect(topic, network, config)
                log.debug(f"Ağ bağlantısı: {topic}, Ağ: {network}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Ağ bağlantı hatası: {str(e)} (BUS046)", context={"source": "network_connect"})

    async def experiment(self, topic: str, method: str, config: str = "") -> None:
        """BUS EXPERIMENT <topic> METHOD "<method>" CONFIG "<options>"
        Deneysel yöntemle veri aktarımı yapar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS047)", context={"source": "experiment"})
                config_dict = self._parse_config(config)
                experiment_manager = self.interpreter.get_manager("experiment")
                await experiment_manager.run(topic, method, config)
                log.debug(f"Deneysel yöntem: {topic}, Yöntem: {method}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Deneysel yöntem hatası: {str(e)} (BUS048)", context={"source": "experiment"})

    async def paradigm_set(self, topic: str, paradigm: str, config: str = "") -> None:
        """BUS PARADIGM SET <topic> TO "<paradigm>" CONFIG "<options>"
        Veri yolu paradigmasını ayarlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS049)", context={"source": "paradigm_set"})
                config_dict = self._parse_config(config)
                self.topics[topic]["paradigm"] = paradigm.lower()
                log.debug(f"Paradigma ayarlandı: {topic}, Paradigma: {paradigm}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Paradigma ayarlama hatası: {str(e)} (BUS050)", context={"source": "paradigm_set"})

    async def data_structure_use(self, topic: str, data_type: str, config: str = "") -> None:
        """BUS DATA STRUCTURE USE <topic> TYPE "<type>" CONFIG "<options>"
        Veri yolu için veri yapısı kullanır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS051)", context={"source": "data_structure_use"})
                config_dict = self._parse_config(config)
                data_manager = self.interpreter.get_manager("data")
                if data_type.lower() == "dataframe":
                    self.buffers[topic] = pd.DataFrame(self.buffers.get(topic, []))
                elif data_type.lower() == "quantum_state":
                    quantum_manager = self.interpreter.get_manager("quantum")
                    self.buffers[topic] = await quantum_manager.convert_to_quantum(self.buffers.get(topic))
                else:
                    self.buffers[topic] = await data_manager.create_structure(data_type, config)
                log.debug(f"Veri yapısı kullanıldı: {topic}, Tür: {data_type}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Veri yapısı kullanma hatası: {str(e)} (BUS052)", context={"source": "data_structure_use"})

    async def flow_control(self, topic: str, action: str, config: str = "") -> None:
        """BUS FLOW CONTROL <topic> ACTION "<action>" CONFIG "<options>"
        Veri akışını kontrol eder."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS053)", context={"source": "flow_control"})
                config_dict = self._parse_config(config)
                if action.lower() == "hold":
                    await self.pause(topic)
                elif action.lower() == "release":
                    await self.resume(topic)
                log.debug(f"Akış kontrolü: {topic}, Eylem: {action}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Akış kontrol hatası: {str(e)} (BUS054)", context={"source": "flow_control"})

    async def resource_allocate(self, topic: str, resource: str, config: str = "") -> None:
        """BUS RESOURCE ALLOCATE <topic> RESOURCE "<resource>" CONFIG "<options>"
        Konu için kaynak ayırır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS055)", context={"source": "resource_allocate"})
                config_dict = self._parse_config(config)
                resource_manager = self.interpreter.get_manager("resource")
                await resource_manager.allocate(topic, resource, config)
                log.debug(f"Kaynak ayrıldı: {topic}, Kaynak: {resource}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Kaynak ayırma hatası: {str(e)} (BUS056)", context={"source": "resource_allocate"})

    async def resource_free(self, topic: str, resource: str, config: str = "") -> None:
        """BUS RESOURCE FREE <topic> RESOURCE "<resource>" CONFIG "<options>"
        Konu kaynaklarını serbest bırakır."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS057)", context={"source": "resource_free"})
                config_dict = self._parse_config(config)
                resource_manager = self.interpreter.get_manager("resource")
                await resource_manager.free(topic, resource, config)
                log.debug(f"Kaynak serbest bırakıldı: {topic}, Kaynak: {resource}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Kaynak serbest bırakma hatası: {str(e)} (BUS058)", context={"source": "resource_free"})

    async def timeout_set(self, topic: str, time: str, config: str = "") -> None:
        """BUS TIMEOUT SET <topic> TIME "<time>" CONFIG "<options>"
        Konu için zaman aşımı ayarlar."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS059)", context={"source": "timeout_set"})
                config_dict = self._parse_config(config)
                timeout_manager = self.interpreter.get_manager("timeout")
                await timeout_manager.set_timeout(topic, time, config)
                log.debug(f"Zaman aşımı ayarlandı: {topic}, Süre: {time}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Zaman aşımı ayarlama hatası: {str(e)} (BUS060)", context={"source": "timeout_set"})

    async def retry(self, topic: str, attempts: int, config: str = "") -> None:
        """BUS RETRY <topic> ATTEMPTS <attempts> CONFIG "<options>"
        Başarısız işlemleri tekrar dener."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS061)", context={"source": "retry"})
                config_dict = self._parse_config(config)
                for attempt in range(attempts):
                    try:
                        await self.publish(topic, self.buffers.get(topic), config)
                        break
                    except PdsXBusError as e:
                        log.warning(f"Yeniden deneme {attempt + 1}/{attempts} başarısız: {str(e)}")
                        if attempt == attempts - 1:
                            raise
                log.debug(f"Yeniden deneme: {topic}, Deneme sayısı: {attempts}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Yeniden deneme hatası: {str(e)} (BUS062)", context={"source": "retry"})

    async def alert(self, topic: str, condition: str, config: str = "") -> None:
        """BUS ALERT <topic> CONDITION "<condition>" CONFIG "<options>"
        Koşula bağlı uyarı üretir."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS063)", context={"source": "alert"})
                config_dict = self._parse_config(config)
                if await self.interpreter.evaluate_condition(condition):
                    event_manager = self.interpreter.get_manager("event")
                    await event_manager.trigger(f"alert_{topic}")
                    self.flag_manager.set_flag(topic, "ALERTED")
                log.debug(f"Uyarı tetiklendi: {topic}, Koşul: {condition}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Uyarı hatası: {str(e)} (BUS064)", context={"source": "alert"})

    async def quantum_send(self, topic: str, quantum_data: Dict[str, Any], config: str = "") -> None:
        """BUS QUANTUM SEND <topic> DATA "<quantum_data>" CONFIG "<options>"
        Kuantum verilerini gönderir."""
        async with self.async_lock:
            try:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS065)", context={"source": "quantum_send"})
                config_dict = self._parse_config(config)
                quantum_manager = self.interpreter.get_manager("quantum")
                circuit = QuantumCircuit(quantum_data.get("qubits", 2), quantum_data.get("clbits", 2))
                for gate in quantum_data.get("gates", []):
                    if gate["type"] == "H":
                        circuit.h(gate["qubit"])
                    elif gate["type"] == "CX":
                        circuit.cx(gate["control"], gate["target"])
                backend = Aer.get_backend("qasm_simulator")
                job = execute(circuit, backend, shots=1024)
                result = job.result()
                quantum_result = result.get_counts()
                self.buffers[topic] = quantum_result
                if config_dict.get("federated", False):
                    federated_manager = self.interpreter.get_manager("federated")
                    await federated_manager.share_data(topic, quantum_result)
                await self.publish(topic, quantum_result, config)
                log.debug(f"Kuantum veri gönderildi: {topic}, Veri: {quantum_result}")
            except PdsXBusError as e:
                raise
            except Exception as e:
                raise PdsXBusError(f"Kuantum veri gönderme hatası: {str(e)} (BUS066)", context={"source": "quantum_send"})

    def get_status(self, topic: str) -> str:
        """GET_BUS_STATUS(topic)
        Konunun anlık durumunu döndürür."""
        try:
            with self.lock:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS101)", context={"source": "get_status"})
                status = self.topics[topic]["status"]
                log.debug(f"Konu durumu alındı: {topic}, Durum: {status}")
                return status
        except PdsXBusError as e:
            raise
        except Exception as e:
            raise PdsXBusError(f"Durum alma hatası: {str(e)} (BUS102)", context={"source": "get_status"})

    def get_stats(self, topic: str) -> Dict[str, Any]:
        """GET_BUS_STATS(topic)
        Konu istatistiklerini döndürür."""
        try:
            with self.lock:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS103)", context={"source": "get_stats"})
                stats = {
                    "runtime": time.time() - self.topics[topic]["timestamp"],
                    "subscribers": len(self.topics[topic]["subscribers"]),
                    "messages": self.prom_counter.labels(topic=topic)._value.get()
                }
                log.debug(f"Konu istatistikleri alındı: {topic}, İstatistikler: {stats}")
                return stats
        except PdsXBusError as e:
            raise
        except Exception as e:
            raise PdsXBusError(f"İstatistik alma hatası: {str(e)} (BUS104)", context={"source": "get_stats"})

    def get_subscribers(self, topic: str) -> List[str]:
        """GET_BUS_SUBSCRIBERS(topic)
        Konunun abone listesini döndürür."""
        try:
            with self.lock:
                if topic not in self.topics:
                    raise PdsXBusError(f"Konu bulunamadı: {topic} (BUS105)", context={"source": "get_subscribers"})
                subscribers = [sub["id"] for sub in self.topics[topic]["subscribers"]]
                log.debug(f"Abone listesi alındı: {topic}, Aboneler: {subscribers}")
                return subscribers
        except PdsXBusError as e:
            raise
        except Exception as e:
            raise PdsXBusError(f"Abone listesi alma hatası: {str(e)} (BUS106)", context={"source": "get_subscribers"})

    async def parse_bus_block(self, block: str, interpreter):
        """Yapısal BUS ... END BUS veya kompakt BUS (...) bloğunu ayrıştırır."""
        block = block.strip()
        if block.upper().startswith("BUS ") and "END BUS" in block.upper():
            # Geleneksel yapı
            lines = block.split('\n')
            if not lines[-1].upper().strip() == "END BUS":
                raise PdsXBusError(f"Geçersiz BUS bloğu sonu (BUS902)", context={"source": "parse_bus_block"})
            header = lines[0].strip()
            match = re.match(r"BUS\s+(\w+)(?:\s*CONFIG\s+\"(.+?)\")?(?:\s*ALIAS\s+(\w+))?", header, re.IGNORECASE)
            if not match:
                raise PdsXBusError(f"Geçersiz BUS başlığı (BUS902)", context={"source": "parse_bus_block"})
            topic, config, alias = match.groups()
            config_dict = self._parse_config(config or "")
            topic = await self.define(topic, config, alias)
            commands = []
            subroutines = {}
            current_sub = None
            for line in lines[1:-1]:
                line = line.strip()
                if line.upper().startswith("SUB "):
                    match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)", line, re.IGNORECASE)
                    if match:
                        sub_name, params = match.groups()
                        current_sub = sub_name
                        subroutines[sub_name] = {"params": params, "commands": []}
                    continue
                elif line.upper() == "END SUB":
                    current_sub = None
                    continue
                if current_sub:
                    subroutines[current_sub]["commands"].append(line)
                else:
                    commands.append(line)
            self.topics[topic]["commands"] = commands
            self.topics[topic]["subroutines"] = subroutines
            for cmd in commands:
                if cmd.strip():
                    await self.parse_bus_command(cmd, interpreter)
            log.debug(f"Veri yolu bloğu kaydedildi: {topic}")
        elif block.upper().startswith("BUS ") and block.endswith(")"):
            # Kompakt yapı
            match = re.match(r"BUS\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?\s*\((.*?)\)", block, re.IGNORECASE)
            if not match:
                raise PdsXBusError(f"Geçersiz kompakt BUS sözdizimi (BUS903)", context={"source": "parse_bus_block"})
            topic, config, content = match.groups()
            config = config or ""
            topic = await self.define(topic, config)
            parts = content.split("|")
            commands = parts[0].split(":") if parts[0] else []
            subroutines = {}
            current_sub = None
            if len(parts) > 1:
                for part in parts[1].split(":"):
                    part = part.strip()
                    if part:
                        if part.upper().startswith("SUB "):
                            match = re.match(r"SUB\s+(\w+)\s*\((.*?)\)", part, re.IGNORECASE)
                            if match:
                                sub_name, params = match.groups()
                                current_sub = sub_name
                                subroutines[sub_name] = {"params": params, "commands": []}
                        elif current_sub:
                            subroutines[current_sub]["commands"].append(part)
                        else:
                            commands.append(part)
            self.topics[topic]["commands"] = commands
            self.topics[topic]["subroutines"] = subroutines
            for cmd in commands:
                if cmd.strip():
                    await self.parse_bus_command(f"BUS {cmd}", interpreter)
            log.debug(f"Kompakt veri yolu kaydedildi: {topic}")

    async def parse_bus_command(self, command: str, interpreter):
        """Veri yolu komutlarını ayrıştırır ve çalıştırır."""
        command_upper = command.strip().upper()
        if command_upper.startswith("BUS ") and "END BUS" in command_upper:
            await self.parse_bus_block(command, interpreter)
            return
        try:
            match = re.match(r"BUS DEFINE\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?(?:\s+AS\s+(\w+))?", command, re.IGNORECASE)
            if match:
                topic, config, var = match.groups()
                topic = await self.define(topic, config or "")
                if var:
                    self.interpreter.current_scope()[var] = topic
                return
            match = re.match(r"BUS PUBLISH\s+(\w+)\s+DATA\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, data, config = match.groups()
                await self.publish(topic, data, config or "")
                return
            match = re.match(r"BUS SUBSCRIBE\s+(\w+)\s+CALLBACK\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, callback, config = match.groups()
                await self.subscribe(topic, callback, config or "")
                return
            match = re.match(r"BUS UNSUBSCRIBE\s+(\w+)(?:\s+(\w+))?", command, re.IGNORECASE)
            if match:
                topic, subscriber_id = match.groups()
                await self.unsubscribe(topic, subscriber_id)
                return
            match = re.match(r"BUS START\s+(\w+)\s+MODE\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, mode, config = match.groups()
                await self.start(topic, mode, config or "")
                return
            match = re.match(r"BUS STOP\s+(\w+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.stop(topic, config or "")
                return
            match = re.match(r"BUS PAUSE\s+(\w+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.pause(topic, config or "")
                return
            match = re.match(r"BUS RESUME\s+(\w+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.resume(topic, config or "")
                return
            match = re.match(r"BUS MONITOR\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?(?:\s+AS\s+(\w+))?", command, re.IGNORECASE)
            if match:
                topic, config, var = match.groups()
                monitor_id = await self.monitor(topic, config or "", var)
                if var:
                    self.interpreter.current_scope()[var] = monitor_id
                return
            match = re.match(r"SECURE BUS\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.secure(topic, config or "")
                return
            match = re.match(r"UNSECURE BUS\s+(\w+)\s*(?:CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.unsecure(topic, config or "")
                return
            match = re.match(r"BUS EVENT TRIGGER\s+(\w+)\s+(.+?)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, event, config = match.groups()
                await self.event_trigger(topic, event, config or "")
                return
            match = re.match(r"BUS RECURSE\s+(\w+)\s+DEPTH\s+(\d+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, depth, config = match.groups()
                await self.recurse(topic, int(depth), config or "")
                return
            match = re.match(r"BUS PRIORITIZE\s+(\w+)\s+LEVEL\s+(\d+\.\d+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, level, config = match.groups()
                await self.prioritize(topic, float(level), config or "")
                return
            match = re.match(r"BUS LOCK\s+(\w+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.lock(topic, config or "")
                return
            match = re.match(r"BUS UNLOCK\s+(\w+)(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, config = match.groups()
                await self.unlock(topic, config or "")
                return
            match = re.match(r"BUS DATA SEND\s+(\w+)\s+TO\s+\"(.+?)\"\s+DATA\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, target, data, config = match.groups()
                await self.data_send(topic, target, data, config or "")
                return
            match = re.match(r"BUS DATA RECEIVE\s+(\w+)\s+FROM\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, source, config = match.groups()
                await self.data_receive(topic, source, config or "")
                return
            match = re.match(r"BUS IOT CONNECT\s+(\w+)\s+DEVICE\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, device_id, config = match.groups()
                await self.iot_connect(topic, device_id, config or "")
                return
            match = re.match(r"BUS CLOUD SYNC\s+(\w+)\s+TO\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, cloud_service, config = match.groups()
                await self.cloud_sync(topic, cloud_service, config or "")
                return
            match = re.match(r"BUS NETWORK CONNECT\s+(\w+)\s+TO\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, network, config = match.groups()
                await self.network_connect(topic, network, config or "")
                return
            match = re.match(r"BUS EXPERIMENT\s+(\w+)\s+METHOD\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, method, config = match.groups()
                await self.experiment(topic, method, config or "")
                return
            match = re.match(r"BUS PARADIGM SET\s+(\w+)\s+TO\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, paradigm, config = match.groups()
                await self.paradigm_set(topic, paradigm, config or "")
                return
            match = re.match(r"BUS DATA STRUCTURE USE\s+(\w+)\s+TYPE\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, data_type, config = match.groups()
                await self.data_structure_use(topic, data_type, config or "")
                return
            match = re.match(r"BUS FLOW CONTROL\s+(\w+)\s+ACTION\s+\"(.+?)\"(?:\s+CONFIG\s+\"(.+?)\")?", command, re.IGNORECASE)
            if match:
                topic, action, config = match.groups()