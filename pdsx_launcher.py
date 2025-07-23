# pdsx_launcher.py - PDS-X Production Launcher
# Version: 1.0.0
# Date: July 21, 2025
# Production-ready PDS-X system launcher

"""
PDS-X Production Launcher

Bu launcher PDS-X sistemini production ortamında güvenli bir şekilde başlatır:
- Environment validation
- System health checks  
- Graceful error handling
- Performance monitoring
- Automatic recovery
"""

import os
import sys
import time
import json
import traceback
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup production logging
def setup_production_logging():
    """Production logging setup"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / "pdsx_production.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("pdsx_launcher")

class PdsXLauncher:
    """PDS-X Production Launcher"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.log = setup_production_logging()
        self.config = self._load_config(config_file)
        self.system_status = {}
        self.program_manager = None
        
        self.log.info("🚀 PDS-X Production Launcher initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load launcher configuration"""
        default_config = {
            "environment": "production",
            "max_retries": 3,
            "health_check_interval": 30,
            "auto_recovery": True,
            "performance_monitoring": True,
            "required_components": [
                "core2_6.py",
                "libxcore.py", 
                "memory_manager.py",
                "auto_importer.py",
                "autoinstaller.py",
                "hybrid_command_executor.py"
            ]
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.log.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.log.warning(f"Failed to load config from {config_file}: {e}")
        
        return default_config
    
    def validate_environment(self) -> bool:
        """Validate production environment"""
        self.log.info("🔍 Validating production environment...")
        
        # Python version check
        python_version = sys.version_info
        if python_version < (3, 10):
            self.log.error(f"❌ Python 3.10+ required, found {python_version}")
            return False
        self.log.info(f"✅ Python version: {python_version}")
        
        # Required files check
        missing_files = []
        for component in self.config["required_components"]:
            if not os.path.exists(component):
                missing_files.append(component)
        
        if missing_files:
            self.log.error(f"❌ Missing required files: {missing_files}")
            return False
        self.log.info("✅ All required components present")
        
        # Core system availability
        try:
            from core_system import (
                CORE_SYSTEM_AVAILABLE, LIBX_AVAILABLE, MEMORY_MANAGER_AVAILABLE,
                AUTO_IMPORTER_AVAILABLE, AUTO_INSTALLER_AVAILABLE, HYBRID_EXECUTOR_AVAILABLE
            )
            
            availability = {
                "Core2_6": CORE_SYSTEM_AVAILABLE,
                "LibX": LIBX_AVAILABLE,
                "MemoryManager": MEMORY_MANAGER_AVAILABLE,
                "AutoImporter": AUTO_IMPORTER_AVAILABLE,
                "AutoInstaller": AUTO_INSTALLER_AVAILABLE,
                "HybridExecutor": HYBRID_EXECUTOR_AVAILABLE
            }
            
            unavailable = [k for k, v in availability.items() if not v]
            if unavailable:
                self.log.error(f"❌ Unavailable components: {unavailable}")
                return False
            
            self.log.info("✅ All core systems available")
            self.system_status.update(availability)
            
        except Exception as e:
            self.log.error(f"❌ Core system validation failed: {e}")
            return False
        
        return True
    
    def initialize_system(self) -> bool:
        """Initialize PDS-X system"""
        self.log.info("🔧 Initializing PDS-X system...")
        
        for attempt in range(self.config["max_retries"]):
            try:
                # Import program manager
                from program_manager import MultLineProgramManager
                
                # Initialize with error handling
                self.program_manager = MultLineProgramManager()
                
                # Verify initialization
                if not self.program_manager:
                    raise Exception("Program manager initialization returned None")
                
                # Check critical components
                if hasattr(self.program_manager, 'pdsx_interpreter'):
                    self.log.info("✅ PDS-X interpreter loaded")
                if hasattr(self.program_manager, 'hybrid_executor'):
                    self.log.info("✅ Hybrid executor loaded")
                if hasattr(self.program_manager, 'auto_installer'):
                    self.log.info("✅ Auto installer loaded")
                
                self.log.info("🎉 PDS-X system successfully initialized")
                return True
                
            except Exception as e:
                self.log.error(f"❌ Initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.config["max_retries"] - 1:
                    self.log.info(f"🔄 Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    self.log.error("❌ All initialization attempts failed")
                    traceback.print_exc()
        
        return False
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run system health check"""
        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
            "performance": {}
        }
        
        try:
            # Component health
            if self.program_manager:
                health_status["components"]["program_manager"] = "operational"
                
                if hasattr(self.program_manager, 'pdsx_interpreter') and self.program_manager.pdsx_interpreter:
                    health_status["components"]["pdsx_interpreter"] = "operational"
                
                if hasattr(self.program_manager, 'hybrid_executor') and self.program_manager.hybrid_executor:
                    health_status["components"]["hybrid_executor"] = "operational"
                    
                    # Test hybrid executor
                    try:
                        executor_info = self.program_manager.hybrid_executor.get_command_info()
                        health_status["performance"]["total_commands"] = executor_info.get("total_commands", 0)
                        health_status["performance"]["total_aliases"] = executor_info.get("total_aliases", 0)
                    except Exception as e:
                        self.log.warning(f"Hybrid executor health check failed: {e}")
                        health_status["components"]["hybrid_executor"] = "degraded"
            
            # Check for any failed components
            failed_components = [k for k, v in health_status["components"].items() if v != "operational"]
            if failed_components:
                health_status["overall_status"] = "degraded"
                
        except Exception as e:
            self.log.error(f"Health check failed: {e}")
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.config["performance_monitoring"]:
            return
        
        self.log.info("📊 Starting performance monitoring...")
        
        # Simple monitoring loop
        try:
            while True:
                health = self.run_health_check()
                
                if health["overall_status"] != "healthy":
                    self.log.warning(f"⚠️ System health: {health['overall_status']}")
                    
                    if self.config["auto_recovery"] and health["overall_status"] == "unhealthy":
                        self.log.info("🔄 Attempting auto-recovery...")
                        if self.initialize_system():
                            self.log.info("✅ Auto-recovery successful")
                        else:
                            self.log.error("❌ Auto-recovery failed")
                
                time.sleep(self.config["health_check_interval"])
                
        except KeyboardInterrupt:
            self.log.info("🛑 Monitoring stopped by user")
        except Exception as e:
            self.log.error(f"❌ Monitoring error: {e}")
    
    def launch(self, interactive: bool = True) -> bool:
        """Launch PDS-X system"""
        self.log.info("🚀 Starting PDS-X production launch sequence...")
        
        # Phase 1: Environment validation
        if not self.validate_environment():
            self.log.error("❌ Environment validation failed")
            return False
        
        # Phase 2: System initialization
        if not self.initialize_system():
            self.log.error("❌ System initialization failed")
            return False
        
        # Phase 3: Final health check
        health = self.run_health_check()
        self.log.info(f"📊 System health: {health['overall_status']}")
        
        if health["overall_status"] == "unhealthy":
            self.log.error("❌ System is unhealthy, aborting launch")
            return False
        
        # Phase 4: Launch complete
        self.log.info("🎉 PDS-X system successfully launched!")
        self.log.info("📊 System status:")
        for component, status in health["components"].items():
            self.log.info(f"   {component}: {status}")
        
        if interactive:
            self.log.info("🎮 Entering interactive mode...")
            self.interactive_mode()
        
        return True
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n" + "="*60)
        print("🎮 PDS-X Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  health - Show system health")
        print("  status - Show system status") 
        print("  monitor - Start monitoring")
        print("  exit - Exit interactive mode")
        print("")
        
        while True:
            try:
                command = input("PDS-X> ").strip().lower()
                
                if command == "exit":
                    break
                elif command == "health":
                    health = self.run_health_check()
                    print(f"System Health: {health['overall_status']}")
                    for comp, status in health["components"].items():
                        print(f"  {comp}: {status}")
                elif command == "status":
                    print("System Status:")
                    for comp, available in self.system_status.items():
                        status = "AVAILABLE" if available else "UNAVAILABLE"
                        print(f"  {comp}: {status}")
                elif command == "monitor":
                    self.start_monitoring()
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.log.info("🛑 Shutting down PDS-X system...")
        
        if self.program_manager:
            # Cleanup program manager if needed
            try:
                if hasattr(self.program_manager, 'cleanup'):
                    self.program_manager.cleanup()
            except Exception as e:
                self.log.warning(f"Cleanup warning: {e}")
        
        self.log.info("✅ PDS-X system shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PDS-X Production Launcher")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    parser.add_argument("--monitor-only", action="store_true", help="Start monitoring only")
    args = parser.parse_args()
    
    launcher = PdsXLauncher(args.config)
    
    try:
        if args.monitor_only:
            if launcher.validate_environment() and launcher.initialize_system():
                launcher.start_monitoring()
        else:
            success = launcher.launch(interactive=not args.no_interactive)
            if not success:
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        launcher.log.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        launcher.shutdown()

if __name__ == "__main__":
    main()
