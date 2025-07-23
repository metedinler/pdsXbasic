"""
PDS-X Paralel İndirme Yöneticisi
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Optional

@dataclass
class DownloadTask:
    url: str
    package_name: str
    version: str
    retries: int = 3
    retry_delay: float = 1.0
    
class AsyncDownloadManager:
    def __init__(self, max_workers: int = 4, cache_dir: Path = None):
        self.max_workers = max_workers
        self.cache_dir = cache_dir or Path(".pdsx_cache/downloads")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.download_stats = {}
        
    async def download_package(self, task: DownloadTask) -> Dict:
        """Tek bir paketi indirir."""
        retry_count = 0
        last_error = None
        
        while retry_count < task.retries:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(task.url) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # Önbelleğe kaydet
                            cache_path = self.cache_dir / f"{task.package_name}-{task.version}.whl"
                            cache_path.write_bytes(content)
                            
                            duration = time.time() - start_time
                            stats = {
                                "success": True,
                                "duration": duration,
                                "size": len(content),
                                "retries": retry_count
                            }
                            
                            self.download_stats[task.package_name] = stats
                            return stats
                            
                        else:
                            last_error = f"HTTP {response.status}"
                            
            except Exception as e:
                last_error = str(e)
            
            retry_count += 1
            if retry_count < task.retries:
                await asyncio.sleep(task.retry_delay * (2 ** retry_count))  # Exponential backoff
                
        stats = {
            "success": False,
            "error": last_error,
            "retries": retry_count
        }
        self.download_stats[task.package_name] = stats
        return stats
        
    async def download_all(self, tasks: List[DownloadTask]) -> Dict[str, Dict]:
        """Birden fazla paketi paralel olarak indirir."""
        async with asyncio.TaskGroup() as tg:
            download_tasks = [
                tg.create_task(self.download_package(task))
                for task in tasks
            ]
            
        return self.download_stats
        
    def get_stats(self) -> Dict[str, Dict]:
        """İndirme istatistiklerini döndürür."""
        return self.download_stats.copy()
        
    def clear_stats(self):
        """İstatistikleri sıfırlar."""
        self.download_stats.clear()
