# pdsx_repl_demo.py - PDS-X REPL Interactive Demo
# Version: 1.0.0
# Date: July 21, 2025

"""
PDS-X REPL Interactive Demo

Bu script PDS-X sistemini interactive olarak test etmek için kullanılır.
"""

def main():
    print("🚀 PDS-X v14u REPL Demo Starting...")
    print("=" * 50)
    
    # Core System Test
    print("\n🔧 Loading Core System...")
    try:
        from core_system import (
            CORE_SYSTEM_AVAILABLE, LIBX_AVAILABLE, MEMORY_MANAGER_AVAILABLE,
            AUTO_IMPORTER_AVAILABLE, AUTO_INSTALLER_AVAILABLE, HYBRID_EXECUTOR_AVAILABLE
        )
        
        components = {
            "Core2_6": CORE_SYSTEM_AVAILABLE,
            "LibX": LIBX_AVAILABLE,
            "MemoryManager": MEMORY_MANAGER_AVAILABLE,
            "AutoImporter": AUTO_IMPORTER_AVAILABLE,
            "AutoInstaller": AUTO_INSTALLER_AVAILABLE,
            "HybridExecutor": HYBRID_EXECUTOR_AVAILABLE
        }
        
        print("📊 Component Status:")
        for name, available in components.items():
            status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
            print(f"   {name}: {status}")
        
        available_count = sum(components.values())
        total_count = len(components)
        print(f"\n📈 Summary: {available_count}/{total_count} components operational")
        
        if available_count == total_count:
            print("🎉 ALL SYSTEMS OPERATIONAL!")
        else:
            print("⚠️ Some components not available")
            
    except Exception as e:
        print(f"❌ Core system error: {e}")
        return False
    
    # Hybrid Executor Test
    print("\n🎭 Testing Hybrid Executor...")
    try:
        from core_system import get_hybrid_executor
        
        # Test availability
        if HYBRID_EXECUTOR_AVAILABLE:
            print("✅ Hybrid Executor available for testing")
            
            # Initialize with minimal setup
            hybrid = get_hybrid_executor()
            if hybrid:
                print("✅ Hybrid Executor instance created")
                
                # Test command info
                info = hybrid.get_command_info()
                print(f"📊 Command Info: {info['total_commands']} commands, {info['total_aliases']} aliases")
                print(f"🔀 Available routes: {list(info['routes'].keys())}")
                print(f"⚙️ Active executors: {info['executors']}")
            else:
                print("⚠️ Could not create hybrid executor instance")
        else:
            print("❌ Hybrid Executor not available")
            
    except Exception as e:
        print(f"⚠️ Hybrid executor test error: {e}")
    
    # AutoInstaller Test
    print("\n📦 Testing AutoInstaller...")
    try:
        from core_system import get_auto_installer
        
        if AUTO_INSTALLER_AVAILABLE:
            print("✅ AutoInstaller available")
            
            installer = get_auto_installer()
            if installer:
                print("✅ AutoInstaller instance created")
                print(f"📁 Workspace: {installer.workspace_path}")
            else:
                print("⚠️ Could not create auto installer instance")
        else:
            print("❌ AutoInstaller not available")
            
    except Exception as e:
        print(f"⚠️ AutoInstaller test error: {e}")
    
    # Interactive Mode
    print("\n🎮 Entering Interactive Mode...")
    print("Commands: 'test', 'status', 'help', 'exit'")
    
    while True:
        try:
            command = input("\nPDS-X> ").strip().lower()
            
            if command == "exit":
                print("👋 Goodbye!")
                break
            elif command == "help":
                print("Available commands:")
                print("  test - Run system tests")
                print("  status - Show system status")
                print("  hybrid - Test hybrid executor")
                print("  exit - Exit demo")
            elif command == "status":
                print("📊 System Status:")
                for name, available in components.items():
                    status = "OPERATIONAL" if available else "NOT AVAILABLE"
                    print(f"  {name}: {status}")
            elif command == "test":
                print("🧪 Running quick test...")
                try:
                    # Test core system availability
                    test_passed = all(components.values())
                    if test_passed:
                        print("✅ All tests passed!")
                    else:
                        print("❌ Some tests failed!")
                except Exception as e:
                    print(f"❌ Test error: {e}")
            elif command == "hybrid":
                print("🎭 Testing Hybrid Executor commands...")
                try:
                    if HYBRID_EXECUTOR_AVAILABLE:
                        hybrid = get_hybrid_executor()
                        if hybrid:
                            # Test a few command info requests
                            commands_to_test = ["PRINT", "LET", "FOR"]
                            for cmd in commands_to_test:
                                info = hybrid.get_command_info(cmd)
                                if info.get('available'):
                                    print(f"  ✅ {cmd}: {info.get('route', 'unknown')} route")
                                else:
                                    print(f"  ❌ {cmd}: not available")
                        else:
                            print("❌ Could not initialize hybrid executor")
                    else:
                        print("❌ Hybrid executor not available")
                except Exception as e:
                    print(f"❌ Hybrid test error: {e}")
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
