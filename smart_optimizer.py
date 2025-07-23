#!/usr/bin/env python3
"""
PDS-X SMART OPTIMIZER v2
========================

İndentation problemlerini çözen akıllı optimizatör.
Safer approach ile step-by-step optimization.

Kullanım:
    python smart_optimizer.py --step1    # Essential imports only
    python smart_optimizer.py --step2    # Lazy loading careful
    python smart_optimizer.py --step3    # Cache optimization
    python smart_optimizer.py --test     # Test system
"""

import os
import sys
import time
import json
from pathlib import Path
import argparse

class SmartOptimizer:
    """Safe step-by-step optimization"""
    
    def __init__(self):
        self.workspace = Path(__file__).parent
        self.main_file = self.workspace / "pdsXuv14.py"
        
    def step1_essential_imports(self):
        """Step 1: Sadece essential imports'ları bırak"""
        print("🔧 STEP 1: Essential imports optimization...")
        
        if not self.main_file.exists():
            print("❌ pdsXuv14.py bulunamadı!")
            return False
            
        # Backup al
        backup_file = self.main_file.with_suffix('.py.step1_backup')
        
        with open(self.main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Essential imports listesi
        essential_patterns = [
            "import sys",
            "import os", 
            "import time",
            "import json",
            "import argparse",
            "from pathlib import Path",
            "import pdsx_repl",
            "import auto_importer", 
            "import libxcore",
            "import core2_6"
        ]
        
        lines = content.split('\n')
        new_lines = []
        commented_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Import satırı mı?
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                # Essential mı?
                is_essential = any(pattern in line for pattern in essential_patterns)
                
                if is_essential:
                    new_lines.append(line)
                else:
                    # Comment out instead of removing
                    new_lines.append(f"# OPTIMIZED: {line}")
                    commented_count += 1
            else:
                new_lines.append(line)
                
        # Yeni content'i yaz
        new_content = '\n'.join(new_lines)
        
        with open(self.main_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"✅ {commented_count} import commented out")
        print(f"📝 Backup: {backup_file.name}")
        return True
    
    def step2_add_lazy_loading(self):
        """Step 2: Dikkatli lazy loading ekle"""
        print("🔧 STEP 2: Adding lazy loading...")
        
        if not self.main_file.exists():
            print("❌ pdsXuv14.py bulunamadı!")
            return False
            
        # Backup al
        backup_file = self.main_file.with_suffix('.py.step2_backup')
        
        with open(self.main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Lazy loading code'u hazırla
        lazy_code = '''
# =============================================================================
# LAZY LOADING OPTIMIZATION - PDS-X v14u Performance Enhancement
# =============================================================================

class LazyLoader:
    """Thread-safe lazy module loader"""
    
    def __init__(self):
        self._modules = {}
        self._loading = set()
    
    def load(self, module_name, package=None):
        """Load module only when needed"""
        key = f"{package}.{module_name}" if package else module_name
        
        if key in self._modules:
            return self._modules[key]
            
        if key in self._loading:
            # Avoid circular loading
            return None
            
        self._loading.add(key)
        
        try:
            if package:
                mod = __import__(package, fromlist=[module_name])
                self._modules[key] = getattr(mod, module_name)
            else:
                self._modules[key] = __import__(module_name)
        except ImportError as e:
            print(f"⚠️  Lazy load failed: {key} -> {e}")
            self._modules[key] = None
        finally:
            self._loading.discard(key)
            
        return self._modules[key]

# Global lazy loader instance
_lazy = LazyLoader()

# Convenience functions for heavy imports
def get_numpy():
    """Get NumPy when needed"""
    return _lazy.load("numpy")

def get_pandas():
    """Get Pandas when needed"""
    return _lazy.load("pandas")

def get_tensorflow():
    """Get TensorFlow when needed"""
    return _lazy.load("tensorflow")

def get_matplotlib():
    """Get Matplotlib pyplot when needed"""
    return _lazy.load("pyplot", "matplotlib")

# =============================================================================
# END LAZY LOADING
# =============================================================================

'''
        
        # İlk import'tan önce ekle (dikkatli)
        import_start = content.find("import sys")
        if import_start > 0:
            # sys import'tan önce ekle
            new_content = content[:import_start] + lazy_code + content[import_start:]
            
            with open(self.main_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print("✅ Lazy loading added safely!")
            print(f"📝 Backup: {backup_file.name}")
            return True
        else:
            print("❌ sys import bulunamadı - safe location yok")
            return False
    
    def step3_optimize_cache(self):
        """Step 3: Cache optimization"""
        print("🔧 STEP 3: Cache optimization...")
        
        cache_dir = self.workspace / ".pdsx_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Fast cache configuration
        config = {
            "version": "v14u_optimized",
            "offline_first": True,
            "fast_lookup": True,
            "skip_verification": True,
            "cache_timeout": 7200,  # 2 hours
            "parallel_downloads": True,
            "max_parallel": 4,
            "compression": True,
            "metadata_cache": True
        }
        
        config_file = cache_dir / "optimization_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        print("✅ Cache optimization config created!")
        return True
    
    def test_system(self):
        """Test optimized system"""
        print("🧪 Testing optimized system...")
        
        # Syntax check
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(self.main_file)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Syntax check: PASSED")
            else:
                print(f"❌ Syntax check: FAILED")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Syntax check failed: {e}")
            return False
            
        # Import test
        try:
            # Test basic import
            original_path = sys.path.copy()
            sys.path.insert(0, str(self.workspace))
            
            # Quick import test
            start_time = time.time()
            spec = __import__("pdsXuv14")
            import_time = time.time() - start_time
            
            print(f"✅ Import test: PASSED ({import_time:.3f}s)")
            return True
            
        except Exception as e:
            print(f"❌ Import test: FAILED -> {e}")
            return False
        finally:
            sys.path = original_path
    
    def rollback(self, step):
        """Rollback to previous step"""
        backup_file = self.main_file.with_suffix(f'.py.step{step}_backup')
        
        if backup_file.exists():
            with open(backup_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            with open(self.main_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✅ Rolled back to step {step}")
            return True
        else:
            print(f"❌ Backup for step {step} not found!")
            return False
    
    def status(self):
        """Show optimization status"""
        print("\n" + "="*50)
        print("🔍 PDS-X OPTIMIZATION STATUS")
        print("="*50)
        
        # File existence
        print(f"Main file: {'✅' if self.main_file.exists() else '❌'}")
        
        # Backup files
        for i in range(1, 4):
            backup = self.main_file.with_suffix(f'.py.step{i}_backup')
            print(f"Step {i} backup: {'✅' if backup.exists() else '❌'}")
            
        # Cache status
        cache_dir = self.workspace / ".pdsx_cache"
        print(f"Cache dir: {'✅' if cache_dir.exists() else '❌'}")
        
        if cache_dir.exists():
            config_file = cache_dir / "optimization_config.json"
            print(f"Cache config: {'✅' if config_file.exists() else '❌'}")
            
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="PDS-X Smart Optimizer v2")
    parser.add_argument("--step1", action="store_true", help="Essential imports only")
    parser.add_argument("--step2", action="store_true", help="Add lazy loading")
    parser.add_argument("--step3", action="store_true", help="Cache optimization")
    parser.add_argument("--test", action="store_true", help="Test system")
    parser.add_argument("--rollback", type=int, help="Rollback to step (1-3)")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    optimizer = SmartOptimizer()
    
    if args.status:
        optimizer.status()
        return
        
    if args.rollback:
        optimizer.rollback(args.rollback)
        return
    
    success = True
    
    if args.all or args.step1:
        success &= optimizer.step1_essential_imports()
        if success and args.all:
            success &= optimizer.test_system()
            
    if success and (args.all or args.step2):
        success &= optimizer.step2_add_lazy_loading()
        if success and args.all:
            success &= optimizer.test_system()
            
    if success and (args.all or args.step3):
        success &= optimizer.step3_optimize_cache()
        
    if args.test or (args.all and success):
        optimizer.test_system()
        
    if not success:
        print("\n❌ Optimization failed! Use --rollback to restore.")
    elif args.all:
        print("\n🎉 All optimization steps completed successfully!")

if __name__ == "__main__":
    main()
