# event.py - PDS-X BASIC v14u Olay ve Kesme Yönetim Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import uuid
import signal
import psutil
from pdsx_exception import PdsXException  # Hata yönetimi için
import re

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("event")

__pdsX_exports__ = {
    "classes": {},
    "functions": {},
    "variables": {
        "dependencies": []
    }
}

class EventManager:
    """Olay ve kesme yönetim sistemi."""
    def __init__(self):
        self.interpreter = None  # Interpreter, parse_event_command ile ayarlanacak
        self.events = defaultdict(list)  # {event_name: [(handler, priority), ...]}
        self.signal_handlers = {}  # {signal: handler}
        self.timers = {}  # {timer_id: (handler, interval, is_periodic, next_run)}
        self.event_queue = deque()  # Olay kuyruğu: [(event_name, args, kwargs, timestamp)]
        self.active_events = 0
        self.lock = threading.Lock()
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = None
        self.metadata = {"event": {"version": "1.0.0", "dependencies": ["psutil", "pdsx_exception"]}}
        self.max_queue_size = 1000  # Maksimum kuyruk boyutu
        self.event_log = []  # Olay geçmişi: [(timestamp, event_name, handler)]
        self.max_log_size = 1000  # Maksimum olay kaydı sayısı

    def set_interpreter(self, interpreter):
        """Interpreter'ı ayarlar."""
        self.interpreter = interpreter
        log.debug("EventManager için interpreter ayarlandı")

    def start_async_loop(self) -> None:
        """Asenkron olay döngüsünü başlatır."""
        def run_loop():
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        with self.lock:
            if not self.async_thread or not self.async_thread.is_alive():
                self.async_thread = threading.Thread(target=run_loop, daemon=True)
                self.async_thread.start()
                log.debug("Asenkron olay döngüsü başlatıldı")

    def register_event(self, event_name: str, handler: str, priority: int = 0) -> str:
        """Olay için işleyici kaydeder."""
        with self.lock:
            try:
                event_id = str(uuid.uuid4())
                self.events[event_name.upper()].append((handler, priority, event_id))
                self.events[event_name.upper()].sort(key=lambda x: x[1], reverse=True)  # Öncelik sırasına göre sırala
                log.debug(f"Olay işleyici kaydedildi: event={event_name}, handler={handler}, priority={priority}, id={event_id}")
                return event_id
            except Exception as e:
                log.error(f"Olay işleyici kayıt hatası: {str(e)}")
                raise PdsXException(f"Olay işleyici kayıt hatası: {str(e)}")

    def unregister_event(self, event_name: str, event_id: str) -> None:
        """Olay işleyiciyi kaldırır."""
        with self.lock:
            try:
                event_name = event_name.upper()
                self.events[event_name] = [(h, p, eid) for h, p, eid in self.events[event_name] if eid != event_id]
                if not self.events[event_name]:
                    del self.events[event_name]
                log.debug(f"Olay işleyici kaldırıldı: event={event_name}, id={event_id}")
            except Exception as e:
                log.error(f"Olay işleyici kaldırma hatası: {str(e)}")
                raise PdsXException(f"Olay işleyici kaldırma hatası: {str(e)}")

    def raise_event(self, event_name: str, *args, **kwargs) -> None:
        """Olayı tetikler."""
        with self.lock:
            try:
                event_name = event_name.upper()
                timestamp = time.time()
                self.event_queue.append((event_name, args, kwargs, timestamp))
                self.active_events += 1
                self.event_log.append((timestamp, event_name, args))
                if len(self.event_log) > self.max_log_size:
                    self.event_log.pop(0)
                if len(self.event_queue) > self.max_queue_size:
                    self.event_queue.popleft()
                
                # Olay işleyicileri yürüt
                for handler, _, _ in self.events.get(event_name, []):
                    self.interpreter.execute_command(handler, *args, **kwargs)
                
                log.debug(f"Olay tetiklendi: event={event_name}, args={args}, kwargs={kwargs}")
            except Exception as e:
                log.error(f"Olay tetikleme hatası: {str(e)}")
                raise PdsXException(f"Olay tetikleme hatası: {str(e)}")

    async def raise_event_async(self, event_name: str, *args, **kwargs) -> None:
        """Olayı asenkron olarak tetikler."""
        try:
            event_name = event_name.upper()
            timestamp = time.time()
            with self.lock:
                self.event_queue.append((event_name, args, kwargs, timestamp))
                self.active_events += 1
                self.event_log.append((timestamp, event_name, args))
                if len(self.event_log) > self.max_log_size:
                    self.event_log.pop(0)
                if len(self.event_queue) > self.max_queue_size:
                    self.event_queue.popleft()
            
            for handler, _, _ in self.events.get(event_name, []):
                await asyncio.to_thread(self.interpreter.execute_command, handler, *args, **kwargs)
            
            log.debug(f"Asenkron olay tetiklendi: event={event_name}, args={args}, kwargs={kwargs}")
        except Exception as e:
            log.error(f"Asenkron olay tetikleme hatası: {str(e)}")
            raise PdsXException(f"Asenkron olay tetikleme hatası: {str(e)}")

    def map_signal(self, signal_name: str, handler: str) -> None:
        """Sistem sinyalini olay işleyiciye bağlar."""
        with self.lock:
            try:
                signal_map = {
                    "SIGINT": signal.SIGINT,
                    "SIGTERM": signal.SIGTERM,
                    "SIGHUP": signal.SIGHUP,
                    "SIGUSR1": signal.SIGUSR1,
                    "SIGUSR2": signal.SIGUSR2
                }
                sig = signal_map.get(signal_name.upper())
                if not sig:
                    raise PdsXException(f"Desteklenmeyen sinyal: {signal_name}")
                
                def signal_handler(signum, frame):
                    self.raise_event(signal_name, signum)
                
                signal.signal(sig, signal_handler)
                self.signal_handlers[signal_name.upper()] = handler
                log.debug(f"Sinyal eşleştirildi: signal={signal_name}, handler={handler}")
            except Exception as e:
                log.error(f"Sinyal eşleştirme hatası: {str(e)}")
                raise PdsXException(f"Sinyal eşleştirme hatası: {str(e)}")

    def set_timer(self, interval: float, handler: str, is_periodic: bool = False) -> str:
        """Zamanlayıcı ayarlar."""
        with self.lock:
            try:
                timer_id = str(uuid.uuid4())
                next_run = time.time() + interval
                self.timers[timer_id] = (handler, interval, is_periodic, next_run)
                
                def timer_task():
                    while timer_id in self.timers:
                        current_time = time.time()
                        handler, interval, is_periodic, next_run = self.timers[timer_id]
                        if current_time >= next_run:
                            try:
                                self.interpreter.execute_command(handler)
                                if is_periodic:
                                    self.timers[timer_id] = (handler, interval, is_periodic, current_time + interval)
                                else:
                                    del self.timers[timer_id]
                                    break
                            except Exception as e:
                                log.error(f"Zamanlayıcı yürütme hatası: {str(e)}")
                                raise PdsXException(f"Zamanlayıcı yürütme hatası: {str(e)}")
                        time.sleep(0.01)  # CPU kullanımını azalt
                
                threading.Thread(target=timer_task, daemon=True).start()
                log.debug(f"Zamanlayıcı ayarlandı: id={timer_id}, interval={interval}, periodic={is_periodic}")
                return timer_id
            except Exception as e:
                log.error(f"Zamanlayıcı ayarlama hatası: {str(e)}")
                raise PdsXException(f"Zamanlayıcı ayarlama hatası: {str(e)}")

    def cancel_timer(self, timer_id: str) -> None:
        """Zamanlayıcıyı iptal eder."""
        with self.lock:
            try:
                if timer_id in self.timers:
                    del self.timers[timer_id]
                    log.debug(f"Zamanlayıcı iptal edildi: id={timer_id}")
                else:
                    raise PdsXException(f"Zamanlayıcı bulunamadı: {timer_id}")
            except Exception as e:
                log.error(f"Zamanlayıcı iptal hatası: {str(e)}")
                raise PdsXException(f"Zamanlayıcı iptal hatası: {str(e)}")

    def get_event_log(self, max_entries: Optional[int] = None) -> List[Tuple[float, str, Any]]:
        """Olay günlüğünü döndürür."""
        with self.lock:
            try:
                return self.event_log[-max_entries:] if max_entries else self.event_log[:]
            except Exception as e:
                log.error(f"Olay günlüğü alma hatası: {str(e)}")
                raise PdsXException(f"Olay günlüğü alma hatası: {str(e)}")

    def clear_event_log(self) -> None:
        """Olay günlüğünü temizler."""
        with self.lock:
            try:
                self.event_log.clear()
                log.debug("Olay günlüğü temizlendi")
            except Exception as e:
                log.error(f"Olay günlüğü temizleme hatası: {str(e)}")
                raise PdsXException(f"Olay günlüğü temizleme hatası: {str(e)}")

    def parse_event_command(self, command: str, interpreter) -> None:
        """Olay yönetimi komutlarını ayrıştırır ve yürütür."""
        if not self.interpreter:
            self.set_interpreter(interpreter)
        
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("EVENT REGISTER "):
                match = re.match(r"EVENT REGISTER\s+(\w+)\s+\"([^\"]+)\"\s*(\d+)?", command, re.IGNORECASE)
                if match:
                    event_name, handler, priority = match.groups()
                    priority = int(priority) if priority else 0
                    event_id = self.register_event(event_name, handler, priority)
                    self.interpreter.current_scope()[f"{event_name}_EVENT_ID"] = event_id
                else:
                    raise PdsXException("EVENT REGISTER komutunda sözdizimi hatası")
            elif command_upper.startswith("EVENT UNREGISTER "):
                match = re.match(r"EVENT UNREGISTER\s+(\w+)\s+(\w+)", command, re.IGNORECASE)
                if match:
                    event_name, event_id = match.groups()
                    self.unregister_event(event_name, event_id)
                else:
                    raise PdsXException("EVENT UNREGISTER komutunda sözdizimi hatası")
            elif command_upper.startswith("EVENT TRIGGER "):
                match = re.match(r"EVENT TRIGGER\s+(\w+)\s*(.*)", command, re.IGNORECASE)
                if match:
                    event_name, args_str = match.groups()
                    args = eval(args_str) if args_str else ()
                    if isinstance(args, tuple):
                        self.raise_event(event_name, *args)
                    else:
                        self.raise_event(event_name, args)
                else:
                    raise PdsXException("EVENT TRIGGER komutunda sözdizimi hatası")
            elif command_upper.startswith("EVENT TRIGGER ASYNC "):
                match = re.match(r"EVENT TRIGGER ASYNC\s+(\w+)\s*(.*)", command, re.IGNORECASE)
                if match:
                    event_name, args_str = match.groups()
                    args = eval(args_str) if args_str else ()
                    self.start_async_loop()
                    if isinstance(args, tuple):
                        asyncio.run_coroutine_threadsafe(self.raise_event_async(event_name, *args), self.async_loop)
                    else:
                        asyncio.run_coroutine_threadsafe(self.raise_event_async(event_name, args), self.async_loop)
                else:
                    raise PdsXException("EVENT TRIGGER ASYNC komutunda sözdizimi hatası")
            elif command_upper.startswith("ON INTERRUPT "):
                match = re.match(r"ON INTERRUPT\s+(\w+)\s+AS\s+(\w+)", command, re.IGNORECASE)
                if match:
                    signal_name, handler = match.groups()
                    self.map_signal(signal_name, handler)
                else:
                    raise PdsXException("ON INTERRUPT komutunda sözdizimi hatası")
            elif command_upper.startswith("TIMER SET "):
                match = re.match(r"TIMER SET\s+(\d+\.?\d*)\s+\"([^\"]+)\"\s*(PERIODIC)?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    interval, handler, periodic, var_name = match.groups()
                    is_periodic = bool(periodic)
                    timer_id = self.set_timer(float(interval), handler, is_periodic)
                    if var_name:
                        self.interpreter.current_scope()[var_name] = timer_id
                else:
                    raise PdsXException("TIMER SET komutunda sözdizimi hatası")
            elif command_upper.startswith("TIMER CANCEL "):
                match = re.match(r"TIMER CANCEL\s+(\w+)", command, re.IGNORECASE)
                if match:
                    timer_id = match.group(1)
                    self.cancel_timer(timer_id)
                else:
                    raise PdsXException("TIMER CANCEL komutunda sözdizimi hatası")
            elif command_upper.startswith("GET EVENT LOG "):
                match = re.match(r"GET EVENT LOG\s+(\w+)(?:\s+(\d+))?", command, re.IGNORECASE)
                if match:
                    var_name, max_entries = match.groups()
                    max_entries = int(max_entries) if max_entries else None
                    self.interpreter.current_scope()[var_name] = self.get_event_log(max_entries)
                else:
                    raise PdsXException("GET EVENT LOG komutunda sözdizimi hatası")
            elif command_upper.startswith("CLEAR EVENT LOG"):
                self.clear_event_log()
            else:
                raise PdsXException(f"Bilinmeyen olay komutu: {command}")
        except Exception as e:
            log.error(f"Olay komut hatası: {str(e)}")
            raise PdsXException(f"Olay komut hatası: {str(e)}")

    def shutdown(self) -> None:
        """Olay yönetim sistemini kapatır."""
        with self.lock:
            try:
                self.events.clear()
                self.signal_handlers.clear()
                self.timers.clear()
                self.event_queue.clear()
                self.active_events = 0
                if self.async_thread and self.async_thread.is_alive():
                    self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                    self.async_loop.run_until_complete(self.async_loop.shutdown_asyncgens())
                    self.async_loop.close()
                log.info("Olay yönetim sistemi kapatıldı")
            except Exception as e:
                log.error(f"Olay sistemi kapatma hatası: {str(e)}")
                raise PdsXException(f"Olay sistemi kapatma hatası: {str(e)}")

if __name__ == "__main__":
    print("event.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")