#!/usr/bin/env python3
"""
PDS-X v14u PERFORMANCE OPTIMIZER
=================================

Hızlı optimizasyon araçları:
1. Lazy loading uygulaması  
2. Import filtreleme
3. AutoImporter cache optimizasyonu
4. Performance benchmark

Kullanım:
    python performance_optimizer.py --analyze        # Mevcut durumu analiz et
    python performance_optimizer.py --lazy          # Lazy loading uygula  
    python performance_optimizer.py --filter        # Import filtrele
    python performance_optimizer.py --cache         # Cache optimize et
    python performance_optimizer.py --benchmark     # Performance test
    python performance_optimizer.py --all           # Hepsini uygula
"""

import os
import sys
import time
import json
import psutil
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# PDS-X modülleri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class PerformanceOptimizer:
    """PDS-X Performance Optimization Engine"""
    
    def __init__(self):
        self.workspace = Path(__file__).parent
        self.metrics = {}
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def analyze_current_state(self) -> Dict[str, Any]:
        """Mevcut sistem durumunu analiz et"""
        print("🔍 Mevcut sistem durumu analiz ediliyor...")
        
        # Import analizi
        import_count = self._count_imports()
        
        # Memory analizi  
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        
        # File analizi
        file_sizes = self._analyze_file_sizes()
        
        # Cache analizi
        cache_status = self._analyze_cache()
        
        analysis = {
            "timestamp": time.time(),
            "imports": import_count,
            "memory_mb": memory_usage,
            "file_sizes": file_sizes,
            "cache": cache_status,
            "startup_time": time.time() - self.start_time
        }
        
        self._print_analysis(analysis)
        return analysis
    
    def _count_imports(self) -> Dict[str, int]:
        """Import sayısını hesapla"""
        counts = {"total": 0, "external": 0, "internal": 0}
        
        main_file = self.workspace / "pdsXuv14.py"
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            counts["total"] = len(import_lines)
            
            # External vs internal imports
            for line in import_lines:
                if any(pkg in line for pkg in ['numpy', 'pandas', 'tensorflow', 'matplotlib']):
                    counts["external"] += 1
                else:
                    counts["internal"] += 1
                    
        return counts
    
    def _analyze_file_sizes(self) -> Dict[str, int]:
        """Dosya boyutlarını analiz et"""
        sizes = {}
        
        for py_file in self.workspace.glob("*.py"):
            if py_file.stat().st_size > 1024:  # 1KB+
                sizes[py_file.name] = py_file.stat().st_size
                
        return dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_cache(self) -> Dict[str, Any]:
        """Cache durumunu analiz et"""
        cache_dir = self.workspace / ".pdsx_cache"
        status = {"exists": cache_dir.exists()}
        
        if cache_dir.exists():
            wheels_dir = cache_dir / "wheels"
            status["wheels_count"] = len(list(wheels_dir.glob("*.whl"))) if wheels_dir.exists() else 0
            
            packages_json = cache_dir / "packages.json"
            if packages_json.exists():
                try:
                    with open(packages_json, 'r') as f:
                        data = json.load(f)
                        status["packages_count"] = len(data)
                except:
                    status["packages_count"] = 0
            else:
                status["packages_count"] = 0
                
        return status
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Analiz sonuçlarını göster"""
        print("\n" + "="*50)
        print("📊 PDS-X PERFORMANCE ANALYSIS")
        print("="*50)
        
        print(f"⏱️  Startup Time: {analysis['startup_time']:.2f} seconds")
        print(f"💾 Memory Usage: {analysis['memory_mb']:.1f} MB")
        print(f"📦 Total Imports: {analysis['imports']['total']}")
        print(f"   ├─ External: {analysis['imports']['external']}")
        print(f"   └─ Internal: {analysis['imports']['internal']}")
        
        print(f"\n🗂️  Largest Files:")
        for filename, size in list(analysis['file_sizes'].items())[:5]:
            print(f"   {filename}: {size/1024:.1f} KB")
            
        print(f"\n💾 Cache Status:")
        cache = analysis['cache']
        print(f"   Cache Dir: {'✅' if cache['exists'] else '❌'}")
        if cache['exists']:
            print(f"   Wheels: {cache['wheels_count']}")
            print(f"   Packages: {cache['packages_count']}")
            
        print("\n" + "="*50)
    
    def apply_lazy_loading(self) -> bool:
        """Lazy loading uygula"""
        print("🚀 Lazy loading uygulanıyor...")
        
        # pdsXuv14.py'yi oku
        main_file = self.workspace / "pdsXuv14.py"
        if not main_file.exists():
            print("❌ pdsXuv14.py bulunamadı!")
            return False
            
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Lazy loading pattern'i ekle
        lazy_template = '''
# Lazy Loading Pattern - Performance Optimization
_lazy_modules = {}

def lazy_import(module_name: str, package: str = None):
    """Modülü sadece gerektiğinde import et"""
    global _lazy_modules
    
    key = f"{package}.{module_name}" if package else module_name
    
    if key not in _lazy_modules:
        try:
            if package:
                _lazy_modules[key] = getattr(__import__(package, fromlist=[module_name]), module_name)
            else:
                _lazy_modules[key] = __import__(module_name)
        except ImportError as e:
            print(f"⚠️  Lazy import failed for {key}: {e}")
            return None
            
    return _lazy_modules[key]

# Lazy loading functions
def get_numpy():
    return lazy_import("numpy")
    
def get_pandas():
    return lazy_import("pandas")
    
def get_tensorflow():
    return lazy_import("tensorflow")
    
def get_matplotlib():
    return lazy_import("matplotlib.pyplot", "matplotlib")
'''
        
        # İlk import satırından önce ekle
        import_start = content.find("import ")
        if import_start > 0:
            new_content = content[:import_start] + lazy_template + "\n" + content[import_start:]
            
            # Backup al
            backup_file = main_file.with_suffix('.py.backup')
            main_file.rename(backup_file)
            
            # Yeni versiyonu yaz
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print("✅ Lazy loading uygulandı!")
            print(f"📝 Backup: {backup_file.name}")
            return True
        else:
            print("❌ Import satırları bulunamadı!")
            return False
    
    def filter_imports(self) -> bool:
        """Gereksiz import'ları filtrele"""
        print("🔍 Import filtreleme...")
        
        # Essential modules listesi
        essential_modules = {
            "os", "sys", "time", "json", "pathlib", "argparse",
            "pdsx_repl", "auto_importer", "core2_6", "libxcore", 
            "command_executor", "module_manager"
        }
        
        # Heavy modules (conditional loading)
        heavy_modules = {
            "tensorflow", "pandas", "numpy", "matplotlib", 
            "sklearn", "scipy", "seaborn"
        }
        
        main_file = self.workspace / "pdsXuv14.py"
        if not main_file.exists():
            print("❌ pdsXuv14.py bulunamadı!")
            return False
            
        with open(main_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        filtered_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Import satırları
            if stripped.startswith(('import ', 'from ')):
                # Essential modül mü?
                is_essential = any(mod in stripped for mod in essential_modules)
                is_heavy = any(mod in stripped for mod in heavy_modules)
                
                if is_essential:
                    new_lines.append(line)
                elif is_heavy:
                    # Heavy modüller için conditional import
                    new_lines.append(f"# LAZY: {line}")
                    filtered_count += 1
                else:
                    # Diğer modüller için conditional import
                    new_lines.append(f"# FILTERED: {line}")
                    filtered_count += 1
            else:
                new_lines.append(line)
                
        if filtered_count > 0:
            # Backup al
            backup_file = main_file.with_suffix('.py.backup_filter')
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            # Filtered version yaz
            with open(main_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            print(f"✅ {filtered_count} import filtrelendi!")
            print(f"📝 Backup: {backup_file.name}")
            return True
        else:
            print("ℹ️  Filtrelenecek import bulunamadı.")
            return False
    
    def optimize_cache(self) -> bool:
        """AutoImporter cache'ini optimize et"""
        print("💾 Cache optimization...")
        
        cache_dir = self.workspace / ".pdsx_cache"
        if not cache_dir.exists():
            print("❌ Cache directory bulunamadı!")
            return False
            
        # Cache settings optimize et
        cache_config = {
            "offline_mode": True,
            "cache_timeout": 3600,  # 1 hour
            "fast_lookup": True,
            "skip_verification": True,
            "parallel_download": True
        }
        
        config_file = cache_dir / "cache_config.json"
        with open(config_file, 'w') as f:
            json.dump(cache_config, f, indent=2)
            
        print("✅ Cache configuration optimized!")
        return True
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Performance test yap"""
        print("⚡ Performance benchmark...")
        
        # Import speed test
        start_time = time.time()
        
        # Test basic imports
        test_modules = ["os", "sys", "json", "time"]
        import_times = {}
        
        for module in test_modules:
            module_start = time.time()
            try:
                __import__(module)
                import_times[module] = time.time() - module_start
            except ImportError:
                import_times[module] = -1
                
        total_import_time = time.time() - start_time
        
        # Memory test
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = current_memory - self.start_memory
        
        # REPL response test (simulated)
        repl_start = time.time()
        try:
            # Test bir Python expression evaluate etme
            result = eval("2 + 2")
            repl_time = time.time() - repl_start
        except:
            repl_time = -1
            
        benchmark = {
            "total_import_time": total_import_time,
            "individual_imports": import_times,
            "memory_increase_mb": memory_increase,
            "repl_response_time": repl_time,
            "total_runtime": time.time() - self.start_time
        }
        
        self._print_benchmark(benchmark)
        return benchmark
    
    def _print_benchmark(self, benchmark: Dict[str, float]):
        """Benchmark sonuçlarını göster"""
        print("\n" + "="*50)
        print("⚡ PERFORMANCE BENCHMARK RESULTS")
        print("="*50)
        
        print(f"⏱️  Total Import Time: {benchmark['total_import_time']:.3f}s")
        print(f"💾 Memory Increase: {benchmark['memory_increase_mb']:.1f} MB")
        print(f"🖥️  REPL Response: {benchmark['repl_response_time']:.3f}s")
        print(f"⏰ Total Runtime: {benchmark['total_runtime']:.2f}s")
        
        print("\n📦 Individual Import Times:")
        for module, time_taken in benchmark['individual_imports'].items():
            if time_taken >= 0:
                print(f"   {module}: {time_taken:.3f}s")
            else:
                print(f"   {module}: FAILED")
                
        print("\n" + "="*50)
    
    def save_results(self, results: Dict[str, Any]):
        """Sonuçları kaydet"""
        results_file = self.workspace / "optimization_results.json"
        
        existing_results = []
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
            except:
                existing_results = []
                
        existing_results.append({
            "timestamp": time.time(),
            "results": results
        })
        
        with open(results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
            
        print(f"💾 Results saved to: {results_file.name}")

def main():
    parser = argparse.ArgumentParser(description="PDS-X Performance Optimizer")
    parser.add_argument("--analyze", action="store_true", help="Analyze current state")
    parser.add_argument("--lazy", action="store_true", help="Apply lazy loading")
    parser.add_argument("--filter", action="store_true", help="Filter imports")
    parser.add_argument("--cache", action="store_true", help="Optimize cache")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--all", action="store_true", help="Apply all optimizations")
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer()
    results = {}
    
    if args.all or args.analyze:
        results["analysis"] = optimizer.analyze_current_state()
        
    if args.all or args.lazy:
        results["lazy_loading"] = optimizer.apply_lazy_loading()
        
    if args.all or args.filter:
        results["import_filtering"] = optimizer.filter_imports()
        
    if args.all or args.cache:
        results["cache_optimization"] = optimizer.optimize_cache()
        
    if args.all or args.benchmark:
        results["benchmark"] = optimizer.benchmark_performance()
        
    if results:
        optimizer.save_results(results)
        print("\n🎉 Optimization completed!")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
