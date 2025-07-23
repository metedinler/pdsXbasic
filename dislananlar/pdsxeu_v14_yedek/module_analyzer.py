import json
import os
import logging
import logging.handlers
from typing import Dict, List, Optional
from datetime import datetime

# rem #24.1. ModuleAnalyzer sınıfı - Modül analizi ve loglama sistemi
# rem #24.2. Rotasyonlu loglama ve gelişmiş modül analizi yetenekleri
class ModuleAnalyzer:    # rem #24.3. Sınıf başlatma ve loglama konfigürasyonu
    def __init__(self, log_file: str = "pdsxu_terminal.log"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Rotasyonlu dosya handler'ı ekle
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
          # rem #24.4. Gelişmiş log analizi sistemi
    def analyze_logs(self) -> Dict:
        """Log dosyalarını analiz eder ve detaylı rapor oluşturur."""
        try:
            log_stats = {
                "errors": [],
                "warnings": [],
                "info": [],
                "debug": [],
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "debug_count": 0,
                "total_size": 0,
                "last_modified": None,
                "backup_files": []
            }
            
            # Ana log dosyası analizi
            if os.path.exists(self.log_file):
                log_stats["total_size"] = os.path.getsize(self.log_file) / 1024  # KB cinsinden
                log_stats["last_modified"] = datetime.fromtimestamp(
                    os.path.getmtime(self.log_file)
                ).isoformat()
                
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "[ERROR]" in line:
                            log_stats["errors"].append(line.strip())
                            log_stats["error_count"] += 1
                        elif "[WARNING]" in line:
                            log_stats["warnings"].append(line.strip())
                            log_stats["warning_count"] += 1
                        elif "[INFO]" in line:
                            log_stats["info"].append(line.strip())
                            log_stats["info_count"] += 1
                        elif "[DEBUG]" in line:
                            log_stats["debug"].append(line.strip())
                            log_stats["debug_count"] += 1
            
            # Yedek dosyaları analiz et
            base_dir = os.path.dirname(self.log_file)
            base_name = os.path.basename(self.log_file)
            for backup_num in range(1, 4):  # 3 yedek dosya için
                backup_file = f"{base_dir}/{base_name}.{backup_num}"
                if os.path.exists(backup_file):
                    log_stats["backup_files"].append({
                        "file": backup_file,
                        "size": os.path.getsize(backup_file) / 1024,  # KB cinsinden
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(backup_file)
                        ).isoformat()
                    })
                            
            return log_stats
        except Exception as e:
            self.logger.error(f"Log analizi hatası: {str(e)}")
            return {"error": str(e)}
            
    def generate_module_report(self, modules: List[Dict]) -> Dict:
        """Modül analiz raporu oluşturur."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "modules": {},
                "total_modules": len(modules),
                "issues": [],
                "recommendations": []
            }
            
            for module in modules:
                module_name = module.get("name", "unknown")
                report["modules"][module_name] = {
                    "version": module.get("version", "unknown"),
                    "dependencies": module.get("dependencies", []),
                    "status": self._check_module_status(module)
                }
                
                # Sorunları tespit et
                issues = self._detect_issues(module)
                if issues:
                    report["issues"].extend(issues)
                    
                # Öneriler oluştur
                recommendations = self._generate_recommendations(module)
                if recommendations:
                    report["recommendations"].extend(recommendations)
                    
            return report
        except Exception as e:
            self.logger.error(f"Rapor oluşturma hatası: {str(e)}")
            return {"error": str(e)}
            
    def _check_module_status(self, module: Dict) -> str:
        """Modül durumunu kontrol eder."""
        try:
            if not all(key in module for key in ["name", "version", "dependencies"]):
                return "invalid"
                
            if not module.get("dependencies"):
                return "warning"
                
            return "ok"
        except Exception:
            return "error"
            
    def _detect_issues(self, module: Dict) -> List[str]:
        """Modüldeki sorunları tespit eder."""
        issues = []
        try:            # rem #24.5. Temel modül alan kontrolü
            # Temel alan kontrolü
            if "name" not in module:
                issues.append("Module missing name field")
            if "version" not in module:
                module_name = module.get('name', 'unknown')
                issues.append(f"Module {module_name} missing version")
            if "dependencies" not in module:
                module_name = module.get('name', 'unknown')
                issues.append(f"Module {module_name} missing dependencies")
                
            # Versiyon format kontrolü
            version = module.get("version", "")
            if version and not self._is_valid_version(version):
                issues.append(f"Module {module.get('name', 'unknown')} has invalid version format: {version}")
                
            # Bağımlılık kontrolü
            deps = module.get("dependencies", [])
            if isinstance(deps, list) and len(deps) == 0:
                issues.append(f"Module {module.get('name', 'unknown')} has no dependencies")
                
            return issues
        except Exception as e:
            self.logger.error(f"Issue detection error: {str(e)}")
            return [f"Error analyzing module: {str(e)}"]
            
    def _generate_recommendations(self, module: Dict) -> List[str]:
        """Modül için öneriler oluşturur."""
        recommendations = []
        try:
            # Eksik alanlar için öneriler
            if "version" not in module:
                recommendations.append(f"Add version information for module {module.get('name', 'unknown')}")
            if "dependencies" not in module:
                recommendations.append(f"Specify dependencies for module {module.get('name', 'unknown')}")
                
            # Bağımlılık önerileri
            deps = module.get("dependencies", [])
            if isinstance(deps, list):
                if len(deps) == 0:
                    recommendations.append(f"Consider adding required dependencies for module {module.get('name', 'unknown')}")
                elif len(deps) > 10:
                    recommendations.append(f"Consider reducing dependencies for module {module.get('name', 'unknown')}")
                    
            return recommendations
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return [f"Error generating recommendations: {str(e)}"]
            
    def _is_valid_version(self, version: str) -> bool:
        """Versiyon formatının geçerliliğini kontrol eder."""
        try:
            parts = version.split(".")
            return len(parts) >= 2 and all(part.isdigit() for part in parts)
        except Exception:
            return False
            
    def export_report(self, report: Dict, output_file: str = "module_analysis.json") -> None:
        """Raporu JSON dosyasına kaydeder."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Report exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Report export error: {str(e)}")
            raise
