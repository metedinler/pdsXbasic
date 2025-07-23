import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] HATA: Bu modül sadece Python 3.10 ortamında çalışır! Lütfen pdsX'in ana başlatıcısını kullanın.")
    sys.exit(1)

# libx_gui.py - PDS-X BASIC v14u Kullanıcı Arayüzü Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import tkinter as tk
from typing import Any, Callable, Dict, Optional, List
import logging
from threading import Thread
from pathlib import Path
import re

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_gui")

class PdsXException(Exception):
    pass

class WindowManager:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.windows = {}
        self.widgets = {}
        self.event_manager = interpreter.event_manager
        self.root = None

    def create(self, name: str, width: int, height: int, title: str = "") -> None:
        """Yeni bir pencere oluşturur."""
        if name.upper() in self.windows:
            raise PdsXException(f"Pencere zaten mevcut: {name}")
        try:
            if not self.root:
                self.root = tk.Tk()
            window = tk.Toplevel(self.root) if self.root else tk.Tk()
            window.title(title)
            window.geometry(f"{width}x{height}")
            self.windows[name.upper()] = window
            self.widgets[name.upper()] = {}
            log.info(f"Pencere oluşturuldu: {name}, {width}x{height}, {title}")
        except Exception as e:
            log.error(f"Pencere oluşturma hatası: {str(e)}")
            raise PdsXException(f"Pencere oluşturma hatası: {str(e)}")

    def add_button(self, window_name: str, widget_name: str, text: str, x: int, y: int) -> None:
        """Pencereye düğme ekler."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            button = tk.Button(window, text=text)
            button.place(x=x, y=y)
            self.widgets[window_name.upper()][widget_name.upper()] = button
            log.debug(f"Düğme eklendi: {window_name}.{widget_name}")
        except Exception as e:
            log.error(f"Düğme ekleme hatası: {str(e)}")
            raise PdsXException(f"Düğme ekleme hatası: {str(e)}")

    def add_label(self, window_name: str, widget_name: str, text: str, x: int, y: int) -> None:
        """Pencereye etiket ekler."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            label = tk.Label(window, text=text)
            label.place(x=x, y=y)
            self.widgets[window_name.upper()][widget_name.upper()] = label
            log.debug(f"Etiket eklendi: {window_name}.{widget_name}")
        except Exception as e:
            log.error(f"Etiket ekleme hatası: {str(e)}")
            raise PdsXException(f"Etiket ekleme hatası: {str(e)}")

    def add_input(self, window_name: str, widget_name: str, x: int, y: int, width: int = 100) -> None:
        """Pencereye giriş alanı ekler."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            entry = tk.Entry(window, width=width // 10)
            entry.place(x=x, y=y)
            self.widgets[window_name.upper()][widget_name.upper()] = entry
            log.debug(f"Giriş alanı eklendi: {window_name}.{widget_name}")
        except Exception as e:
            log.error(f"Giriş alanı ekleme hatası: {str(e)}")
            raise PdsXException(f"Giriş alanı ekleme hatası: {str(e)}")

    def add_menu(self, window_name: str, menu_name: str, items: List[Dict]) -> None:
        """Pencereye menü ekler."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            menubar = tk.Menu(window)
            menu = tk.Menu(menubar, tearoff=0)
            for item in items:
                label = item.get("label", "")
                command = item.get("command", "")
                if command:
                    menu.add_command(label=label, command=lambda: self.interpreter.execute_command(command))
                else:
                    menu.add_command(label=label)
            menubar.add_cascade(label=menu_name, menu=menu)
            window.config(menu=menubar)
            self.widgets[window_name.upper()][menu_name.upper()] = menubar
            log.debug(f"Menü eklendi: {window_name}.{menu_name}")
        except Exception as e:
            log.error(f"Menü ekleme hatası: {str(e)}")
            raise PdsXException(f"Menü ekleme hatası: {str(e)}")

    def show(self, window_name: str) -> None:
        """Pencereyi gösterir."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            window.deiconify()
            log.debug(f"Pencere gösterildi: {window_name}")
        except Exception as e:
            log.error(f"Pencere gösterme hatası: {str(e)}")
            raise PdsXException(f"Pencere gösterme hatası: {str(e)}")

    def close(self, window_name: str) -> None:
        """Pencereyi kapatır."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            window.destroy()
            del self.windows[window_name.upper()]
            del self.widgets[window_name.upper()]
            log.debug(f"Pencere kapatıldı: {window_name}")
        except Exception as e:
            log.error(f"Pencere kapatma hatası: {str(e)}")
            raise PdsXException(f"Pencere kapatma hatası: {str(e)}")

    def bind_event(self, window_name: str, widget_name: str, event: str, handler: str) -> None:
        """Widget'a olay bağlar."""
        window_widgets = self.widgets.get(window_name.upper(), {})
        widget = window_widgets.get(widget_name.upper())
        if not widget:
            raise PdsXException(f"Widget bulunamadı: {window_name}.{widget_name}")
        try:
            tk_event = self._map_event(event)
            widget.bind(tk_event, lambda e: self.interpreter.execute_command(handler))
            log.debug(f"Olay bağlandı: {window_name}.{widget_name}, {event}")
        except Exception as e:
            log.error(f"Olay bağlama hatası: {str(e)}")
            raise PdsXException(f"Olay bağlama hatası: {str(e)}")

    def bind_system_event(self, window_name: str, event_type: str, handler: str) -> None:
        """Pencereye sistem olayı bağlar."""
        window = self.windows.get(window_name.upper())
        if not window:
            raise PdsXException(f"Pencere bulunamadı: {window_name}")
        try:
            if event_type.lower() == "mouse_clicked":
                window.bind("<Button-1>", lambda e: self.interpreter.execute_command(handler))
            elif event_type.lower() == "key_pressed":
                window.bind("<Key>", lambda e: self.interpreter.execute_command(handler))
            else:
                raise PdsXException(f"Desteklenmeyen sistem olayı: {event_type}")
            log.debug(f"Sistem olayı bağlandı: {window_name}, {event_type}")
        except Exception as e:
            log.error(f"Sistem olayı bağlama hatası: {str(e)}")
            raise PdsXException(f"Sistem olayı bağlama hatası: {str(e)}")

    def run(self) -> None:
        """GUI ana döngüsünü başlatır."""
        if self.root:
            try:
                self.root.mainloop()
                log.info("GUI ana döngüsü başlatıldı")
            except Exception as e:
                log.error(f"GUI çalıştırma hatası: {str(e)}")
                raise PdsXException(f"GUI çalıştırma hatası: {str(e)}")

    def _map_event(self, event: str) -> str:
        """Olay adını Tkinter formatına çevirir."""
        event_map = {
            "CLICK": "<Button-1>",
            "DOUBLE_CLICK": "<Double-Button-1>",
            "KEY_PRESS": "<Key>",
            "ENTER": "<Return>",
            "FOCUS_IN": "<FocusIn>",
            "FOCUS_OUT": "<FocusOut>"
        }
        return event_map.get(event.upper(), event)

    def _parse_params(self, param_str: str) -> Dict:
        """Parametre dizesini ayrıştırır."""
        params = {}
        for param in param_str.split(","):
            key, value = param.strip().split("=")
            key = key.strip().upper()
            value = value.strip().strip('"')
            if key in ("WIDTH", "HEIGHT"):
                params[key] = int(value)
            else:
                params[key] = value
        return params

class LibXGui:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.window_manager = WindowManager(interpreter)
        self.metadata = {"libx_gui": {"version": "1.0.0", "dependencies": ["tkinter"]}}
        self.gui_thread = None

    def parse_gui_command(self, command: str) -> None:
        """GUI komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("WINDOW "):
                match = re.match(r"WINDOW\s+(\w+)\s+(.*)", command, re.IGNORECASE)
                if match:
                    name, rest = match.groups()
                    params = self.window_manager._parse_params(rest)
                    self.window_manager.create(name, params["WIDTH"], params["HEIGHT"], params.get("TITLE", ""))
                else:
                    raise PdsXException("WINDOW komutunda sözdizimi hatası")
            elif command_upper.startswith("BUTTON "):
                match = re.match(r"BUTTON\s+(\w+)\s+(\w+)\s+\"([^\"]+)\"\s+(\d+)\s+(\d+)", command, re.IGNORECASE)
                if match:
                    window_name, widget_name, text, x, y = match.groups()
                    self.window_manager.add_button(window_name, widget_name, text, int(x), int(y))
                else:
                    raise PdsXException("BUTTON komutunda sözdizimi hatası")
            elif command_upper.startswith("LABEL "):
                match = re.match(r"LABEL\s+(\w+)\s+(\w+)\s+\"([^\"]+)\"\s+(\d+)\s+(\d+)", command, re.IGNORECASE)
                if match:
                    window_name, widget_name, text, x, y = match.groups()
                    self.window_manager.add_label(window_name, widget_name, text, int(x), int(y))
                else:
                    raise PdsXException("LABEL komutunda sözdizimi hatası")
            elif command_upper.startswith("INPUT "):
                match = re.match(r"INPUT\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)(?:\s+(\d+))?", command, re.IGNORECASE)
                if match:
                    window_name, widget_name, x, y, width = match.groups()
                    width = int(width) if width else 100
                    self.window_manager.add_input(window_name, widget_name, int(x), int(y), width)
                else:
                    raise PdsXException("INPUT komutunda sözdizimi hatası")
            elif command_upper.startswith("MENU "):
                match = re.match(r"MENU\s+(\w+)\s+(\w+)\s+\[(.*)\]", command, re.IGNORECASE)
                if match:
                    window_name, menu_name, items_str = match.groups()
                    items = eval(f"[{items_str}]")  # Güvenli ayrıştırma için JSON kullanılabilir
                    self.window_manager.add_menu(window_name, menu_name, items)
                else:
                    raise PdsXException("MENU komutunda sözdizimi hatası")
            elif command_upper.startswith("SHOW "):
                match = re.match(r"SHOW\s+(\w+)", command, re.IGNORECASE)
                if match:
                    window_name = match.group(1)
                    self.window_manager.show(window_name)
                else:
                    raise PdsXException("SHOW komutunda sözdizimi hatası")
            elif command_upper.startswith("CLOSE "):
                match = re.match(r"CLOSE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    window_name = match.group(1)
                    self.window_manager.close(window_name)
                else:
                    raise PdsXException("CLOSE komutunda sözdizimi hatası")
            elif command_upper.startswith("BIND "):
                match = re.match(r"BIND\s+(\w+)\s+(\w+)\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    window_name, widget_name, event, handler = match.groups()
                    self.window_manager.bind_event(window_name, widget_name, event, handler)
                else:
                    raise PdsXException("BIND komutunda sözdizimi hatası")
            elif command_upper.startswith("BIND SYSTEM "):
                match = re.match(r"BIND SYSTEM\s+(\w+)\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    window_name, event_type, handler = match.groups()
                    self.window_manager.bind_system_event(window_name, event_type, handler)
                else:
                    raise PdsXException("BIND SYSTEM komutunda sözdizimi hatası")
            elif command_upper == "RUN":
                self.start_gui_thread()
            else:
                raise PdsXException(f"Bilinmeyen GUI komutu: {command}")
        except Exception as e:
            log.error(f"GUI komut hatası: {str(e)}")
            raise PdsXException(f"GUI komut hatası: {str(e)}")

    def start_gui_thread(self) -> None:
        """GUI döngüsünü ayrı bir iş parçacığında başlatır."""
        if not self.gui_thread or not self.gui_thread.is_alive():
            self.gui_thread = Thread(target=self.window_manager.run, daemon=True)
            self.gui_thread.start()
            log.debug("GUI iş parçacığı başlatıldı")

if __name__ == "__main__":
    print("libx_gui.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")