#!/usr/bin/env python3
"""
PDS-X Modül Analiz Sistemi
Ana dizindeki modülleri kategorize ederek sistemin durumunu analiz eder.
"""

import os
import glob
import importlib.util
from pathlib import Path
import json

def analyze_modules():
    """Ana dizindeki Python modüllerini analiz et"""
    
    print("🔍 PDS-X MODÜL ANALİZİ")
    print("=" * 50)
    
    # Python dosyalarını bul
    py_files = glob.glob('*.py')
    print(f"📊 Toplam Python dosyası: {len(py_files)}")
    
    # Ana modülleri kategorize et
    categories = {
        'core': [],
        'libx': [],
        'managers': [],
        'utilities': [],
        'repls': [],
        'tests': [],
        'disabled': [],
        'others': []
    }
    
    # Kategorilendirme kuralları
    for file in sorted(py_files):
        name = file.replace('.py', '')
        
        if name.startswith('libx'):
            categories['libx'].append(name)
        elif 'manager' in name.lower():
            categories['managers'].append(name)
        elif name in ['core2_6', 'core_system', 'bytecode_compiler', 'bytecode_manager']:
            categories['core'].append(name)
        elif name in ['auto_importer_lite', 'auto_importer_heavy', 'memory_manager', 'data_structures']:
            categories['utilities'].append(name)
        elif 'repl' in name.lower() or 'reply' in name.lower():
            categories['repls'].append(name)
        elif 'test' in name.lower():
            categories['tests'].append(name)
        elif 'disabled' in name.lower():
            categories['disabled'].append(name)
        else:
            categories['others'].append(name)
    
    # Sonuçları göster
    print(f"\n🔥 ÇEKİRDEK MODÜLLER ({len(categories['core'])}):")
    for mod in categories['core']:
        print(f"  • {mod}")
    
    print(f"\n📚 LIBX KÜTÜPHANELERİ ({len(categories['libx'])}):")
    for mod in categories['libx']:
        print(f"  • {mod}")
    
    print(f"\n⚙️ YÖNETİCİ MODÜLLER ({len(categories['managers'])}):")
    for mod in categories['managers']:
        print(f"  • {mod}")
    
    print(f"\n🔧 YARDIMCI MODÜLLER ({len(categories['utilities'])}):")
    for mod in categories['utilities']:
        print(f"  • {mod}")
    
    print(f"\n💬 REPL MODÜLLER ({len(categories['repls'])}):")
    for mod in categories['repls']:
        print(f"  • {mod}")
    
    print(f"\n🧪 TEST MODÜLLER ({len(categories['tests'])}):")
    for mod in categories['tests'][:5]:  # İlk 5'i göster
        print(f"  • {mod}")
    if len(categories['tests']) > 5:
        print(f"  ... ve {len(categories['tests'])-5} test daha")
    
    print(f"\n📦 DİĞER MODÜLLER ({len(categories['others'])}):")
    for mod in categories['others'][:10]:  # İlk 10'u göster
        print(f"  • {mod}")
    if len(categories['others']) > 10:
        print(f"  ... ve {len(categories['others'])-10} modül daha")
    
    return categories

def test_imports(categories):
    """Temel modüllerin import edilebilirliğini test et"""
    
    print("\n🔬 MODÜL İMPORT TESTİ")
    print("=" * 30)
    
    critical_modules = categories['core'] + categories['utilities'][:2]  # Kritik modüller
    
    working_modules = []
    broken_modules = []
    
    for module_name in critical_modules:
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            if spec and spec.loader:
                print(f"  ✅ {module_name} - Import OK")
                working_modules.append(module_name)
            else:
                print(f"  ❌ {module_name} - Spec Error")
                broken_modules.append(module_name)
        except Exception as e:
            print(f"  ❌ {module_name} - {str(e)[:50]}...")
            broken_modules.append(module_name)
    
    print(f"\n📊 ÖZET:")
    print(f"  ✅ Çalışan: {len(working_modules)}")
    print(f"  ❌ Hatalı: {len(broken_modules)}")
    
    return working_modules, broken_modules

def generate_integration_plan(categories, working_modules):
    """Entegrasyon planı oluştur"""
    
    plan = {
        "integration_priority": [],
        "current_status": {
            "total_modules": sum(len(cat) for cat in categories.values()),
            "working_modules": len(working_modules),
            "core_ready": len([m for m in working_modules if m in categories['core']]),
            "libx_available": len(categories['libx']),
            "utilities_ready": len([m for m in working_modules if m in categories['utilities']])
        },
        "next_steps": []
    }
    
    # Öncelik sırası
    plan["integration_priority"].extend([
        {"phase": 1, "modules": categories['core'], "description": "Çekirdek sistemler"},
        {"phase": 2, "modules": categories['utilities'], "description": "Yardımcı modüller"},
        {"phase": 3, "modules": categories['libx'][:3], "description": "Temel LibX kütüphaneleri"},
        {"phase": 4, "modules": categories['repls'], "description": "REPL sistemleri"},
        {"phase": 5, "modules": categories['managers'], "description": "Yönetici modüller"}
    ])
    
    # Sonraki adımlar
    plan["next_steps"] = [
        "1. Git repository oluştur",
        "2. Mevcut çalışma durumunu commit et",
        "3. Faz 1 modülleri pdsXuv14.py'ye entegre et",
        "4. Her faz sonrası test et ve commit et",
        "5. Tam entegrasyon sonrası final test"
    ]
    
    return plan

if __name__ == "__main__":
    try:
        # Modül analizi
        categories = analyze_modules()
        
        # Import testi
        working_modules, broken_modules = test_imports(categories)
        
        # Entegrasyon planı
        plan = generate_integration_plan(categories, working_modules)
        
        print("\n🚀 ENTEGRASYON PLANI")
        print("=" * 25)
        print(f"📊 Toplam Modül: {plan['current_status']['total_modules']}")
        print(f"✅ Çalışan Modül: {plan['current_status']['working_modules']}")
        print(f"🔥 Çekirdek Hazır: {plan['current_status']['core_ready']}")
        print(f"📚 LibX Mevcut: {plan['current_status']['libx_available']}")
        
        print(f"\n📋 SONRAKİ ADIMLAR:")
        for step in plan['next_steps']:
            print(f"  {step}")
        
        # JSON olarak kaydet
        with open('integration_plan.json', 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Plan kaydedildi: integration_plan.json")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
