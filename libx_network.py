import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] HATA: Bu modül sadece Python 3.10 ortamında çalışır! Lütfen pdsX'in ana başlatıcısını kullanın.")
    sys.exit(1)

# libx_network.py - PDS-X BASIC v14u Ağ ve İletişim Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import requests
import socket
import asyncio
import aiohttp
import websockets
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode
import json
import time
from pathlib import Path
try:
    import oauthlib.oauth1
except ImportError as e:
    print('oauthlib.oauth1 import error:', e)
    oauthlib = None
try:
    from requests_oauthlib import OAuth1Session
except ImportError as e:
    print('requests_oauthlib import error:', e)
    OAuth1Session = None
import threading
from types import SimpleNamespace
import uuid

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_network")

class PdsXException(Exception):
    pass

# Sınıf adı NetworkManager'dı, LibXNetwork olarak değiştirildi
class LibXNetwork:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.session = None  # ClientSession başlatılmadı
        self.oauth_sessions = {}
        self.websocket_connections = {}
        self.api_cache = {}
        self.metadata = {"libx_network": {"version": "1.0.0", "dependencies": ["requests", "aiohttp", "websockets", "requests_oauthlib"]}}
        self.lock = threading.Lock()

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def fetch(self, url, **kwargs):
        session = await self.get_session()
        async with session.get(url, **kwargs) as response:
            return await response.text()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def shutdown(self) -> None:
        """Ağ bağlantılarını kapatır."""
        try:
            await self.close()
            for ws in self.websocket_connections.values():
                await ws.close()
            self.websocket_connections.clear()
            log.info("Ağ bağlantıları kapatıldı")
        except Exception as e:
            log.error(f"Ağ kapatma hatası: {str(e)}")
            raise PdsXException(f"Ağ kapatma hatası: {str(e)}")

    def web_get(self, url: str, headers: Optional[Dict] = None, timeout: float = 10.0) -> str:
        """Senkron HTTP GET isteği yapar."""
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            log.debug(f"HTTP GET tamamlandı: {url}")
            return response.text
        except requests.RequestException as e:
            log.error(f"HTTP GET hatası: {url}, {str(e)}")
            raise PdsXException(f"HTTP GET hatası: {str(e)}")

    async def web_get_async(self, url: str, headers: Optional[Dict] = None, timeout: float = 10.0) -> str:
        """Asenkron HTTP GET isteği yapar."""
        try:
            session = await self.get_session()
            async with session.get(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                text = await response.text()
                log.debug(f"Asenkron HTTP GET tamamlandı: {url}")
                return text
        except aiohttp.ClientError as e:
            log.error(f"Asenkron HTTP GET hatası: {url}, {str(e)}")
            raise PdsXException(f"Asenkron HTTP GET hatası: {str(e)}")

    def web_post(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: float = 10.0) -> str:
        """Senkron HTTP POST isteği yapar."""
        try:
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
            response.raise_for_status()
            log.debug(f"HTTP POST tamamlandı: {url}")
            return response.text
        except requests.RequestException as e:
            log.error(f"HTTP POST hatası: {url}, {str(e)}")
            raise PdsXException(f"HTTP POST hatası: {str(e)}")

    async def web_post_async(self, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: float = 10.0) -> str:
        """Asenkron HTTP POST isteği yapar."""
        try:
            session = await self.get_session()
            async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                text = await response.text()
                log.debug(f"Asenkron HTTP POST tamamlandı: {url}")
                return text
        except aiohttp.ClientError as e:
            log.error(f"Asenkron HTTP POST hatası: {url}, {str(e)}")
            raise PdsXException(f"Asenkron HTTP POST hatası: {str(e)}")

    def ping(self, host: str, timeout: float = 2.0) -> bool:
        """Host'a ping atar."""
        try:
            socket.setdefaulttimeout(timeout)
            socket.gethostbyname(host)
            log.debug(f"Ping başarılı: {host}")
            return True
        except socket.error as e:
            log.debug(f"Ping başarısız: {host}, {str(e)}")
            return False

    def load_api(self, url: str, api_id: Optional[str] = None) -> SimpleNamespace:
        """API uç noktasını yükler ve bir nesne döndürür."""
        api_id = api_id or str(uuid.uuid4())
        try:
            def ask(query: str) -> Dict:
                try:
                    response = requests.post(url, json={"query": query}, timeout=10.0)
                    response.raise_for_status()
                    return response.json().get("response", {})
                except requests.RequestException as e:
                    log.error(f"API sorgu hatası: {url}, {str(e)}")
                    raise PdsXException(f"API sorgu hatası: {str(e)}")
            
            api = SimpleNamespace(ask=ask)
            self.api_cache[api_id] = api
            log.debug(f"API yüklendi: {api_id}, {url}")
            return api
        except Exception as e:
            log.error(f"API yükleme hatası: {url}, {str(e)}")
            raise PdsXException(f"API yükleme hatası: {str(e)}")

    def oauth_register(self, api_id: str, consumer_key: str, consumer_secret: str, access_token: str, access_token_secret: str) -> None:
        """OAuth 1.0 oturumu kaydeder."""
        try:
            oauth = OAuth1Session(
                consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=access_token,
                resource_owner_secret=access_token_secret
            )
            with self.lock:
                self.oauth_sessions[api_id] = oauth
            log.debug(f"OAuth oturumu kaydedildi: {api_id}")
        except Exception as e:
            log.error(f"OAuth kayıt hatası: {api_id}, {str(e)}")
            raise PdsXException(f"OAuth kayıt hatası: {str(e)}")

    def oauth_request(self, api_id: str, url: str, method: str = "GET", data: Optional[Dict] = None) -> str:
        """OAuth ile HTTP isteği yapar."""
        oauth = self.oauth_sessions.get(api_id)
        if not oauth:
            raise PdsXException(f"OAuth oturumu bulunamadı: {api_id}")
        try:
            if method.upper() == "GET":
                response = oauth.get(url, data=urlencode(data or {}))
            elif method.upper() == "POST":
                response = oauth.post(url, json=data)
            else:
                raise PdsXException(f"Desteklenmeyen HTTP yöntemi: {method}")
            response.raise_for_status()
            log.debug(f"OAuth isteği tamamlandı: {api_id}, {url}")
            return response.text
        except requests.RequestException as e:
            log.error(f"OAuth istek hatası: {api_id}, {url}, {str(e)}")
            raise PdsXException(f"OAuth istek hatası: {str(e)}")

    async def websocket_connect(self, ws_id: str, uri: str) -> None:
        """WebSocket bağlantısı kurar."""
        try:
            ws = await websockets.connect(uri)
            with self.lock:
                self.websocket_connections[ws_id] = ws
            log.debug(f"WebSocket bağlantısı kuruldu: {ws_id}, {uri}")
        except Exception as e:
            log.error(f"WebSocket bağlantı hatası: {ws_id}, {uri}, {str(e)}")
            raise PdsXException(f"WebSocket bağlantı hatası: {str(e)}")

    async def websocket_send(self, ws_id: str, message: str) -> None:
        """WebSocket üzerinden mesaj gönderir."""
        ws = self.websocket_connections.get(ws_id)
        if not ws:
            raise PdsXException(f"WebSocket bağlantısı bulunamadı: {ws_id}")
        try:
            await ws.send(message)
            log.debug(f"WebSocket mesajı gönderildi: {ws_id}, {message[:50]}...")
        except Exception as e:
            log.error(f"WebSocket gönderme hatası: {ws_id}, {str(e)}")
            raise PdsXException(f"WebSocket gönderme hatası: {str(e)}")

    async def websocket_receive(self, ws_id: str, timeout: float = 10.0) -> str:
        """WebSocket üzerinden mesaj alır."""
        ws = self.websocket_connections.get(ws_id)
        if not ws:
            raise PdsXException(f"WebSocket bağlantısı bulunamadı: {ws_id}")
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=timeout)
            log.debug(f"WebSocket mesajı alındı: {ws_id}, {message[:50]}...")
            return message
        except asyncio.TimeoutError:
            log.error(f"WebSocket alma zaman aşımı: {ws_id}")
            raise PdsXException(f"WebSocket alma zaman aşımı: {ws_id}")
        except Exception as e:
            log.error(f"WebSocket alma hatası: {ws_id}, {str(e)}")
            raise PdsXException(f"WebSocket alma hatası: {str(e)}")

    async def websocket_close(self, ws_id: str) -> None:
        """WebSocket bağlantısını kapatır."""
        ws = self.websocket_connections.get(ws_id)
        if not ws:
            raise PdsXException(f"WebSocket bağlantısı bulunamadı: {ws_id}")
        try:
            await ws.close()
            with self.lock:
                del self.websocket_connections[ws_id]
            log.debug(f"WebSocket bağlantısı kapatıldı: {ws_id}")
        except Exception as e:
            log.error(f"WebSocket kapatma hatası: {ws_id}, {str(e)}")
            raise PdsXException(f"WebSocket kapatma hatası: {str(e)}")

    def reply(self, query: str, endpoint: str, model: str = "default") -> Dict:
        """İnteraktif ağ sorgusu yapar."""
        cache_key = f"{endpoint}_{query}_{model}"
        if cache_key in self.api_cache:
            log.debug(f"Önbellekten alındı: {cache_key}")
            return self.api_cache[cache_key]
        
        try:
            # Örnek bir REPLY endpoint'i
            response = requests.post(endpoint, json={"query": query, "model": model}, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            self.api_cache[cache_key] = result
            log.debug(f"REPLY tamamlandı: {query[:50]}...")
            return result
        except requests.RequestException as e:
            log.error(f"REPLY hatası: {endpoint}, {str(e)}")
            raise PdsXException(f"REPLY hatası: {str(e)}")

    def parse_network_command(self, command: str) -> None:
        """Ağ komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("NET GET "):
                match = re.match(r"NET GET\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    url, var_name = match.groups()
                    result = self.web_get(url)
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET GET komutunda sözdizimi hatası")
            elif command_upper.startswith("NET POST "):
                match = re.match(r"NET POST\s+\"([^\"]+)\"\s*(\{.*?\})?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    url, data_str, var_name = match.groups()
                    data = json.loads(data_str) if data_str else None
                    result = self.web_post(url, data)
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET POST komutunda sözdizimi hatası")
            elif command_upper.startswith("NET PING "):
                match = re.match(r"NET PING\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    host, var_name = match.groups()
                    result = self.ping(host)
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET PING komutunda sözdizimi hatası")
            elif command_upper.startswith("NET API "):
                match = re.match(r"NET API\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    url, api_id = match.groups()
                    result = self.load_api(url, api_id)
                    self.interpreter.current_scope()[api_id or "_NET_API"] = result
                else:
                    raise PdsXException("NET API komutunda sözdizimi hatası")
            elif command_upper.startswith("NET OAUTH REGISTER "):
                match = re.match(r"NET OAUTH REGISTER\s+(\w+)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    api_id, consumer_key, consumer_secret, access_token, access_token_secret = match.groups()
                    self.oauth_register(api_id, consumer_key, consumer_secret, access_token, access_token_secret)
                else:
                    raise PdsXException("NET OAUTH REGISTER komutunda sözdizimi hatası")
            elif command_upper.startswith("NET OAUTH REQUEST "):
                match = re.match(r"NET OAUTH REQUEST\s+(\w+)\s+\"([^\"]+)\"\s+(\w+)\s*(\{.*?\})?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    api_id, url, method, data_str, var_name = match.groups()
                    data = json.loads(data_str) if data_str else None
                    result = self.oauth_request(api_id, url, method, data)
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET OAUTH REQUEST komutunda sözdizimi hatası")
            elif command_upper.startswith("NET WEBSOCKET CONNECT "):
                match = re.match(r"NET WEBSOCKET CONNECT\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    ws_id, uri = match.groups()
                    asyncio.run(self.websocket_connect(ws_id, uri))
                else:
                    raise PdsXException("NET WEBSOCKET CONNECT komutunda sözdizimi hatası")
            elif command_upper.startswith("NET WEBSOCKET SEND "):
                match = re.match(r"NET WEBSOCKET SEND\s+(\w+)\s+\"([^\"]+)\"", command, re.IGNORECASE)
                if match:
                    ws_id, message = match.groups()
                    asyncio.run(self.websocket_send(ws_id, message))
                else:
                    raise PdsXException("NET WEBSOCKET SEND komutunda sözdizimi hatası")
            elif command_upper.startswith("NET WEBSOCKET RECEIVE "):
                match = re.match(r"NET WEBSOCKET RECEIVE\s+(\w+)\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    ws_id, var_name = match.groups()
                    result = asyncio.run(self.websocket_receive(ws_id))
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET WEBSOCKET RECEIVE komutunda sözdizimi hatası")
            elif command_upper.startswith("NET WEBSOCKET CLOSE "):
                match = re.match(r"NET WEBSOCKET CLOSE\s+(\w+)", command, re.IGNORECASE)
                if match:
                    ws_id = match.group(1)
                    asyncio.run(self.websocket_close(ws_id))
                else:
                    raise PdsXException("NET WEBSOCKET CLOSE komutunda sözdizimi hatası")
            elif command_upper.startswith("NET REPLY "):
                match = re.match(r"NET REPLY\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*(\w+)?\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    query, endpoint, model, var_name = match.groups()
                    model = model or "default"
                    result = self.reply(query, endpoint, model)
                    self.interpreter.current_scope()[var_name or "_NET_RESULT"] = result
                else:
                    raise PdsXException("NET REPLY komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen ağ komutu: {command}")
        except Exception as e:
            log.error(f"Ağ komut hatası: {str(e)}")
            raise PdsXException(f"Ağ komut hatası: {str(e)}")

if __name__ == "__main__":
    print("libx_network.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")

# Dinamik yükleme için ihraç edilecek öğeler
__pdsX_exports__ = {
    "classes": {
        "LibXNetwork": LibXNetwork,
        "PdsXException": PdsXException
    },
    "functions": {
    },
    "variables": {
        "version": "1.0.0",
        "dependencies": [
            "requests", "aiohttp", "websockets", 
            "oauthlib", "requests_oauthlib"
        ]
    }
}