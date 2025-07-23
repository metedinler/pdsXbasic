#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS-X Memory Optimization Analyzer
==================================

RAM kullanımını analiz eder ve modül yükleme optimizasyonu önerir.
"ne kadar ram okadar cok modul kulanma" prensibine göre çalışır.

Version: 1.0
Date: 22 Temmuz 2025
"""

import sys
import os
import psutil
import importlib
import gc
import tracemalloc
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple

class MemoryOptimizer:
    """RAM optimizasyonu ve modül yönetimi"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory()
        self.module_memory_usage = {}
        self.optimization_recommendations = []
        
        # Sistem RAM bilgisi
        self.system_memory = psutil.virtual_memory()
        self.available_ram_gb = self.system_memory.available / (1024**3)
        self.total_ram_gb = self.system_memory.total / (1024**3)
        self.memory_usage_percent = self.system_memory.percent
        
        print(f"[Memory Optimizer] 🧠 Sistem RAM: {self.total_ram_gb:.1f}GB")
        print(f"[Memory Optimizer] 📊 Kullanılan: {self.memory_usage_percent:.1f}%")
        print(f"[Memory Optimizer] 🆓 Mevcut: {self.available_ram_gb:.1f}GB")
        
    def get_current_memory(self) -> float:
        """Mevcut memory kullanımını MB olarak döndür"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def analyze_module_impact(self, module_name: str) -> Dict:
        """Modül yükleme etkisini analiz et"""
        # Memory tracking başlat
        tracemalloc.start()
        memory_before = self.get_current_memory()
        time_before = time.time()
        
        try:
            # Modülü import et
            module = importlib.import_module(module_name)
            
            # Memory kullanımını ölç
            memory_after = self.get_current_memory()
            time_after = time.time()
            
            # Memory snapshot al
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            impact = {
                "module": module_name,
                "memory_increase_mb": memory_after - memory_before,
                "load_time_ms": (time_after - time_before) * 1000,
                "total_memory_mb": memory_after,
                "success": True,
                "memory_hotspots": [stat.traceback.format()[:2] for stat in top_stats[:3]]
            }
            
        except Exception as e:
            impact = {
                "module": module_name,
                "memory_increase_mb": 0,
                "load_time_ms": 0,
                "total_memory_mb": self.get_current_memory(),
                "success": False,
                "error": str(e)
            }
        
        finally:
            tracemalloc.stop()
        
        self.module_memory_usage[module_name] = impact
        return impact
    
    def analyze_pdsx_modules(self) -> Dict:
        """PDS-X modüllerini analiz et"""
        print("\n[Memory Optimizer] 🔍 PDS-X Modül Analizi Başlıyor...")
        
        # PDS-X modülleri
        pdsx_modules = [
            "auto_importer_lite", "auto_importer_heavy",
            "pdsx_repl", "reply_extension", "program_manager",
            "core_system", "memory_manager", "event3",
            "libxcore", "module_manager", "data_structures"
        ]
        
        results = {}
        total_memory_increase = 0
        
        for module in pdsx_modules:
            print(f"[Memory Optimizer] 📦 {module} analiz ediliyor...")
            impact = self.analyze_module_impact(module)
            results[module] = impact
            
            if impact["success"]:
                total_memory_increase += impact["memory_increase_mb"]
                print(f"  ✅ {impact['memory_increase_mb']:.1f}MB (+{impact['load_time_ms']:.0f}ms)")
            else:
                print(f"  ❌ Yüklenemedi: {impact.get('error', 'Unknown')}")
        
        results["summary"] = {
            "total_modules_tested": len(pdsx_modules),
            "successful_loads": sum(1 for r in results.values() if isinstance(r, dict) and r.get("success")),
            "total_memory_increase_mb": total_memory_increase,
            "average_memory_per_module_mb": total_memory_increase / len(pdsx_modules)
        }
        
        print(f"\n[Memory Optimizer] 📊 Toplam Memory Artışı: {total_memory_increase:.1f}MB")
        return results
    
    def generate_optimization_strategy(self) -> Dict:
        """RAM miktarına göre optimizasyon stratejisi"""
        
        if self.available_ram_gb < 2:
            strategy = "ULTRA_LITE"
            max_modules = 5
            priority_modules = ["auto_importer_lite", "pdsx_repl", "core_system"]
        elif self.available_ram_gb < 4:
            strategy = "LITE"
            max_modules = 10
            priority_modules = ["auto_importer_lite", "pdsx_repl", "program_manager", "memory_manager"]
        elif self.available_ram_gb < 8:
            strategy = "BALANCED"
            max_modules = 15
            priority_modules = ["auto_importer_heavy", "reply_extension", "libxcore", "event3"]
        else:
            strategy = "FULL"
            max_modules = 25
            priority_modules = ["auto_importer_heavy", "reply_extension", "libxcore", "all_features"]
        
        optimization = {
            "strategy": strategy,
            "available_ram_gb": self.available_ram_gb,
            "max_recommended_modules": max_modules,
            "priority_modules": priority_modules,
            "memory_budget_mb": self.available_ram_gb * 1024 * 0.3,  # %30 RAM kullan
            "recommendations": []
        }
        
        # Tavsiyeler
        if strategy == "ULTRA_LITE":
            optimization["recommendations"] = [
                "Sadece temel modülleri yükle",
                "AutoImporter Lite kullan",
                "Background servisleri devre dışı bırak",
                "Cache boyutunu sınırla"
            ]
        elif strategy == "LITE":
            optimization["recommendations"] = [
                "AutoImporter Lite tercih et",
                "Lazy loading kullan",
                "Ağır modülleri isteğe bağlı yükle"
            ]
        elif strategy == "BALANCED":
            optimization["recommendations"] = [
                "AutoImporter Heavy kullanılabilir",
                "Seçici modül yükleme",
                "Memory monitoring aktif"
            ]
        else:
            optimization["recommendations"] = [
                "Tüm özellikler kullanılabilir",
                "Agresif pre-loading",
                "Full feature set"
            ]
        
        return optimization
    
    def create_optimized_config(self) -> Dict:
        """Optimized config dosyası oluştur"""
        strategy = self.generate_optimization_strategy()
        
        config = {
            "pdsx_optimization": {
                "strategy": strategy["strategy"],
                "memory_budget_mb": strategy["memory_budget_mb"],
                "max_modules": strategy["max_recommended_modules"],
                "auto_importer_mode": "LITE" if strategy["strategy"] in ["ULTRA_LITE", "LITE"] else "HEAVY",
                "modules": {
                    "priority": strategy["priority_modules"],
                    "lazy_load": strategy["strategy"] in ["ULTRA_LITE", "LITE"],
                    "background_services": strategy["strategy"] not in ["ULTRA_LITE"],
                    "cache_size_mb": min(500, strategy["memory_budget_mb"] * 0.2)
                },
                "performance": {
                    "aggressive_gc": strategy["strategy"] == "ULTRA_LITE",
                    "memory_monitoring": True,
                    "startup_optimization": True
                }
            },
            "generated": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_ram_gb": self.total_ram_gb,
                "available_ram_gb": self.available_ram_gb,
                "memory_usage_percent": self.memory_usage_percent
            }
        }
        
        return config
    
    def save_optimization_report(self, module_analysis: Dict, config: Dict):
        """Optimizasyon raporunu kaydet"""
        report = {
            "memory_analysis": module_analysis,
            "optimization_config": config,
            "system_info": {
                "total_ram_gb": self.total_ram_gb,
                "available_ram_gb": self.available_ram_gb,
                "memory_usage_percent": self.memory_usage_percent,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # JSON dosyasına kaydet
        with open("pdsx_memory_optimization_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Memory Optimizer] 💾 Rapor kaydedildi: pdsx_memory_optimization_report.json")
    
    def run_full_analysis(self):
        """Tam optimizasyon analizi çalıştır"""
        print("="*60)
        print("🧠 PDS-X MEMORY OPTIMIZATION ANALYSIS")
        print("="*60)
        
        # 1. Modül analizi
        module_analysis = self.analyze_pdsx_modules()
        
        # 2. Optimizasyon stratejisi
        optimization_strategy = self.generate_optimization_strategy()
        
        # 3. Config oluştur
        optimized_config = self.create_optimized_config()
        
        # 4. Sonuçları göster
        print(f"\n📈 ÖNERİLEN STRATEJİ: {optimization_strategy['strategy']}")
        print(f"🎯 Max Modül: {optimization_strategy['max_recommended_modules']}")
        print(f"💾 Memory Budget: {optimization_strategy['memory_budget_mb']:.0f}MB")
        print(f"🔧 AutoImporter Mode: {optimized_config['pdsx_optimization']['auto_importer_mode']}")
        
        print(f"\n💡 TAVSİYELER:")
        for i, rec in enumerate(optimization_strategy['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # 5. Raporu kaydet
        self.save_optimization_report(module_analysis, optimized_config)
        
        print("\n" + "="*60)
        print("✅ OPTIMIZATION ANALYSIS TAMAMLANDI")
        print("="*60)
        
        return {
            "module_analysis": module_analysis,
            "strategy": optimization_strategy,
            "config": optimized_config
        }

def main():
    """Ana analiz fonksiyonu"""
    optimizer = MemoryOptimizer()
    results = optimizer.run_full_analysis()
    
    # En çok memory kullanan modülleri göster
    module_usage = [(name, data.get("memory_increase_mb", 0)) 
                   for name, data in results["module_analysis"].items() 
                   if isinstance(data, dict) and "memory_increase_mb" in data]
    
    module_usage.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 EN ÇOK MEMORY KULLANAN MODÜLLER:")
    for i, (module, memory_mb) in enumerate(module_usage[:5], 1):
        print(f"  {i}. {module}: {memory_mb:.1f}MB")
    
    # Hızlı optimizasyon önerisi
    strategy = results["strategy"]["strategy"]
    print(f"\n⚡ HIZLI OPTİMİZASYON:")
    print(f"  📋 Strateji: {strategy}")
    
    if strategy in ["ULTRA_LITE", "LITE"]:
        print("  🔧 auto_importer_lite.py kullan")
        print("  🚫 Ağır modülleri yükleme")
        print("  ⚡ Lazy loading aktif et")
    else:
        print("  🔥 auto_importer_heavy.py kullanılabilir")
        print("  ✅ Tüm özellikler aktif")
        print("  🚀 Full performance mode")

if __name__ == "__main__":
    main()
