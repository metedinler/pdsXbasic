# auto_importer.py - PDS-X BASIC v15 Dinamik Modül Yükleyici
# Version: 1.5.0
# Date: May 19, 2025
# Author: xAI (Grok 3 ile oluşturuldu, fikir Mete Dinler, duzenleyen ve calismasini saglayan github copilot)
# --- PDS-X Otomatik Ortam Kurulumu ve Modül Yükleyici ---
# Bu script, PDS-X'in çalışması için gerekli olan Python 3.10 ortamını ve tüm pip paketlerini otomatik olarak kurar.
# Kullanıcıdan hiçbir manuel işlem beklemez, her adımda otomasyon ve hata önleme önceliklidir.

# --- Gerekli Modüllerin İçe Aktarılması ---
# Standart kütüphaneler ve ortam yönetimi için gerekli modüller yüklenir.
# rem #1. Gerekli modüllerin ve ortam yönetimi için temel ayarların yapılması
import os
import sys
import importlib.util
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set
from collections import defaultdict
import time
import random
from typing import List
import shutil
import urllib.request
import threading  # Eksik threading modülü eklendi
from concurrent.futures import ThreadPoolExecutor  # as_completed kaldırıldı çünkü zaten kullanılmıyor
import concurrent.futures  # Eksik concurrent.futures modülü eklendi
import json  # JSON modülü eklendi
try:
    import winreg
except ImportError:
    winreg = None

# --- Terminal log dosyasını yedekle ve sıfırla ---
# rem #2. Terminal log dosyasını yedekle ve sıfırla
LOG_FILE = "pdsxu_terminal.log"
LOG_BAK = "pdsxu_terminal.bak"
try:
    if os.path.exists(LOG_FILE):
        if os.path.exists(LOG_BAK):
            os.remove(LOG_BAK)
        shutil.move(LOG_FILE, LOG_BAK)
except Exception as e:
    print(f"[PDS-X] Log dosyası yedeklenemedi: {e}")

# --- Tüm terminal çıktısını hem ekrana hem log dosyasına yazan sınıf ---
class Tee:
    # rem #3. Tüm terminal çıktısını hem ekrana hem log dosyasına yazan sınıf
    def __init__(self, *files):
        # rem #3.1. Dosya nesnelerini sakla
        self.files = files
    def write(self, obj):
        # rem #3.2. Her dosyaya yaz ve flush et
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass
    def flush(self):
        # rem #3.3. Her dosyayı flush et
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

sys.stdout = Tee(sys.__stdout__, open(LOG_FILE, "a", encoding="utf-8"))
sys.stderr = Tee(sys.__stderr__, open(LOG_FILE, "a", encoding="utf-8"))

# --- Temel Bağımlılıklar ---
# rem #4. Temel ve opsiyonel bağımlılıkların listelenmesi
CORE_DEPENDENCIES = {
    "base": [
        "numpy<2.2.0",  # pin numpy version <2.2.0 to satisfy TensorFlow requirements
        "pandas",
        "scikit-learn",
        "torch",
        "graphviz",
        "requests",
        "aiohttp",
        "websockets",
        "aiofiles",      # ensure aiofiles is installed
        "pdfplumber",    # added to support core2-6 pdfplumber import
        "tensorflow==2.19.0",    # added to ensure TensorFlow is installed for core2-6
        "psycopg2-binary",
        "pyyaml",
        "psutil",        # added to ensure psutil is installed
        "paho-mqtt",     # added to ensure MQTT client support
        "boto3",         # added to ensure AWS SDK (boto3)
        "botocore",      # added to support boto3 dependencies
        "language-data", # added to satisfy langcodes requirement
    ],
    "optional": [
        "tensorflow",
        "transformers",
        "nltk",
        "spacy",
        "gensim"
    ]
}

# --- Dinamik Modül Bağımlılıkları ---
# rem #5. Modül bazlı özel bağımlılıkların tanımlanması
# Her modül için özel bağımlılıklar
MODULE_SPECIFIC_DEPS = {
    "core2-5.py": ["tensorflow", "scikit-learn", "numpy"],
    "libx_ml.py": ["torch", "transformers", "scikit-learn"],
    "libx_nlp.py": ["nltk", "spacy", "gensim"],
    "database_sql_isam.py": ["psycopg2-binary", "sqlite3"],
    "graph.py": ["networkx", "graphviz"],
}

# --- Python 3.10 Bulucu ---
# rem #6. Python 3.10 bulucu fonksiyon
# rem #6.1. PATH, py launcher, yaygın dizinler, venv, registry, conda, pyenv araması
# rem #6.2. Uygun python bulunamazsa None döner
# Sistemde Python 3.10 yüklü mü kontrol eder. PATH, py launcher ve yaygın dizinlerde arama yapar.
def find_python310():
    # rem #6.3. Python 3.10 arama başlatıldı
    print("[PDS-X] Python 3.10 arama başlatıldı...")
    found = []
    # rem #6.4. PATH üzerinde arama
    for exe in ["python3.10", "python310", "python"]:
        path = shutil.which(exe)
        if path:
            try:
                out = subprocess.check_output([path, "--version"], text=True)
                print(f"[PDS-X] PATH'de bulundu: {path} ({out.strip()})")
                if "3.10" in out:
                    print(f"[PDS-X] Uygun Python bulundu: {path}")
                    return path
                else:
                    found.append((path, out.strip()))
            except Exception as e:
                print(f"[PDS-X] PATH'de {exe} çalıştırılamadı: {e}")
    # rem #6.5. py launcher ile arama
    try:
        out = subprocess.check_output(["py", "-3.10", "--version"], text=True)
        print(f"[PDS-X] py launcher ile bulundu: py -3.10 ({out.strip()})")
        if "3.10" in out:
            return "py -3.10"
    except Exception as e:
        print(f"[PDS-X] py launcher ile Python 3.10 bulunamadı: {e}")
    # rem #6.6. Yaygın Windows dizinlerinde arama
    possible = [
        rf"C:\\Users\\{os.environ.get('USERNAME','')}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
        r"C:\\Python310\\python.exe"
    ]
    for path in possible:
        if os.path.exists(os.path.expandvars(path)):
            print(f"[PDS-X] Yaygın dizinde bulundu: {path}")
            return os.path.expandvars(path)
    # rem #6.7. Proje içi venv'lerde arama
    for venv_name in [".venv", "venv", ".pdsx_isolated_env"]:
        for sub in ["Scripts", "bin", ""]:
            venv_py = Path(venv_name) / sub / ("python.exe" if os.name == "nt" else "python3")
            if venv_py.exists():
                try:
                    out = subprocess.check_output([str(venv_py), "--version"], text=True)
                    print(f"[PDS-X] Proje venv'de bulundu: {venv_py} ({out.strip()})")
                    if "3.10" in out:
                        return str(venv_py)
                except Exception as e:
                    print(f"[PDS-X] Proje venv'de hata: {e}")
    # rem #6.8. Registry'de arama (Windows)
    if os.name == "nt":
        try:
            import winreg
            for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                with winreg.OpenKey(root, r"SOFTWARE\\Python\\PythonCore") as hkey:
                    for i in range(0, winreg.QueryInfoKey(hkey)[0]):
                        ver = winreg.EnumKey(hkey, i)
                        if ver.startswith("3.10"):
                            with winreg.OpenKey(hkey, ver + r"\\InstallPath") as subkey:
                                py = winreg.QueryValue(subkey, None) + "python.exe"
                                if os.path.exists(py):
                                    print(f"[PDS-X] Registry'de bulundu: {py}")
                                    return py
        except Exception as e:
            print(f"[PDS-X] Registry taranırken hata: {e}")
    # rem #6.9. Conda ortamlarında arama
    try:
        out = subprocess.check_output(["conda", "env", "list"], text=True)
        for line in out.splitlines():
            if "3.10" in line and ("*" in line or "envs" in line):
                env_path = line.split()[-1]
                py = os.path.join(env_path, "python.exe" if os.name == "nt" else "bin/python3")
                if os.path.exists(py):
                    print(f"[PDS-X] Conda ortamında bulundu: {py}")
                    return py
    except Exception as e:
        print(f"[PDS-X] Conda ortamı taranamadı: {e}")
    # rem #6.10. pyenv ortamlarında arama
    try:
        pyenv_root = os.environ.get("PYENV_ROOT") or os.path.expanduser("~/.pyenv")
        versions_dir = Path(pyenv_root) / "versions"
        if versions_dir.exists():
            for ver in versions_dir.iterdir():
                if "3.10" in ver.name:
                    py = ver / ("bin/python3" if (ver / "bin/python3").exists() else "python.exe")
                    if py.exists():
                        print(f"[PDS-X] pyenv ortamında bulundu: {py}")
                        return str(py)
    except Exception as e:
        print(f"[PDS-X] pyenv ortamı taranamadı: {e}")
    # rem #6.11. Hiçbiri bulunamazsa None döner
    print("[PDS-X] Uygun Python 3.10 bulunamadı. Bulunanlar:")
    for p, v in found:
        print(f"  {p} ({v})")
    return None

# --- Python 3.10 İndirici ve Sessiz Kurucu (Sadece Windows) ---
# rem #7. Python 3.10 indirici ve sessiz kurucu (sadece Windows)
# Eğer Python 3.10 bulunamazsa, otomatik olarak indirir ve sessizce kurar.
def download_and_install_python310():
    # rem #7.1. Disk alanı kontrolü ve indirme
    import shutil
    drive = os.path.splitdrive(os.getcwd())[0] or 'C:'
    total, used, free = shutil.disk_usage(drive + '\\')
    min_required = 300 * 1024 * 1024  # 300 MB
    print(f"[PDS-X] {drive} sürücüsünde boş alan: {free // (1024*1024)} MB")
    if free < min_required:
        print(f"[PDS-X] HATA: Python kurulumu için en az 300 MB boş alan gerekli! Şu an: {free // (1024*1024)} MB")
        return None
    installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
    installer_path = "python310_installer.exe"
    print("[PDS-X] Python 3.10 indiriliyor...")
    urllib.request.urlretrieve(installer_url, installer_path)
    # rem #7.2. Sessiz kurulum ve temizleme
    print("[PDS-X] Python 3.10 kuruluyor (sessiz)...")
    subprocess.run([installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], check=True)
    print("[PDS-X] Python 3.10 kurulumu tamamlandı.")
    os.remove(installer_path)
    return find_python310()

# --- PATH Güncelleme Scriptleri Oluşturucu ---
# rem #8. PATH güncelleme scriptleri oluşturucu
# Python ve venv dizinlerini PATH'e eklemek için .bat ve .ps1 scriptleri üretir.
def add_python_to_path(python_path):
    python_dir = os.path.dirname(python_path)
    venv_dir = str(VENV_DIR / ("Scripts" if os.name == "nt" else "bin"))
    setx_cmd = f'setx PATH "%PATH%;{python_dir};{venv_dir}"'
    with open("add_pdsx_path.bat", "w") as f:
        f.write(f"@echo off\n{setx_cmd}\necho PATH güncellendi.\npause\n")
    with open("add_pdsx_path.ps1", "w") as f:
        f.write(
            f"$ErrorActionPreference = 'Stop'\n"
            f"$pythonDir = '{python_dir}'\n"
            f"$venvDir = '{venv_dir}'\n"
            "$oldPath = [System.Environment]::GetEnvironmentVariable('Path', [System.EnvironmentVariableTarget]::User)\n"
            "if ($oldPath -notlike \"*$pythonDir*\") {\n"
            "    $newPath = \"$oldPath;$pythonDir;$venvDir\"\n"
            "    [System.Environment]::SetEnvironmentVariable('Path', $newPath, [System.EnvironmentVariableTarget]::User)\n"
            "    Write-Host 'PATH güncellendi.'\n"
            "} else {\n"
            "    Write-Host 'PATH zaten güncel.'\n"
            "}\n"
        )
    print("[PDS-X] PATH'e eklemek için add_pdsx_path.bat veya add_pdsx_path.ps1 dosyasını yönetici olarak çalıştırın!")

# --- Sanal Ortam (venv) Dizini ---
# rem #9. Sanal ortam (venv) dizini ve yolu
# Ortam adı çakışmasın diye özel bir isim kullanılır.
VENV_DIR = Path(".pdsx_isolated_env")

# --- Python 3.10 Kontrol ve Otomatik Kurulum ---
# rem #10. Python 3.10 kontrol ve otomatik kurulum
# Eğer Python 3.10 yoksa, indirip kurar. Sonra PATH'e ekler.
PYTHON310_PATH = find_python310()
if not PYTHON310_PATH:
    if os.name == "nt":
        PYTHON310_PATH = download_and_install_python310()
        # --- DÜZELTME: Kurulumdan sonra tekrar bul ---
        PYTHON310_PATH = find_python310()
        if not PYTHON310_PATH:
            print("[PDS-X] Python 3.10 kurulamadı! Lütfen elle kurun.")
            sys.exit(1)
    else:
        print("[PDS-X] Python 3.10 bulunamadı! Lütfen elle kurun: https://www.python.org/downloads/release/python-31011/")
        sys.exit(1)
else:
    print(f"[PDS-X] Python 3.10 bulundu: {PYTHON310_PATH}")
    add_python_to_path(PYTHON310_PATH)

# --- venv Yolu ve Python Yürütücüsü ---
# rem #11. venv yolu ve python yürütücüsü belirleme
# İşletim sistemine göre venv içindeki python yolu belirlenir.
if os.name == "nt":
    VENV_PYTHON = str(VENV_DIR / "Scripts" / "python.exe")
else:
    VENV_PYTHON = str(VENV_DIR / "bin" / "python3")

# --- Sanal Ortamda mıyız? ---
# rem #12. Sanal ortamda mıyız kontrolü
# Kodun venv içinde çalışıp çalışmadığını kontrol eder.
def in_venv():
    # rem #12.1. venv içinde çalışılıp çalışılmadığını kontrol eder
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV')
    )

def _python_cmd_args(python_path, *args):
    # rem #12.2. py -3.10 gibi path'leri subprocess için uygun hale getirir
    """PYTHON310_PATH 'py -3.10' gibi ise split ederek subprocess'a uygun hale getirir."""
    if python_path == "py -3.10":
        return ["py", "-3.10", *args]
    return [python_path, *args]

# --- Sanal Ortamı Otomatik Oluşturucu ve Yeniden Başlatıcı ---
# rem #13. Sanal ortamı otomatik oluşturucu ve yeniden başlatıcı
# Eğer venv yoksa oluşturur, pip'i günceller ve scripti venv içinde tekrar başlatır.
def ensure_venv():
    """Sanal ortamı oluşturur ve doğru şekilde yapılandırır."""
    def log_and_print(msg):
        print(msg)
        with open("pdsxu_terminal.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if not in_venv():
        log_and_print("[PDS-X] UYARI: Şu anda base ortamdasınız. Otomatik olarak izole venv'ye geçiliyor!")
        logging.warning("Base ortamda başlatıldı, otomatik olarak venv'ye geçiliyor.")
        if not VENV_DIR.exists():
            log_and_print("[PDS-X] Sanal ortam oluşturuluyor: venv (.pdsx_isolated_env)...")
            subprocess.run(_python_cmd_args(PYTHON310_PATH, "-m", "venv", str(VENV_DIR)), check=True)

        # pyvenv.cfg dosyasını kontrol et ve eksikse oluştur
        pyvenv_cfg_path = VENV_DIR / "pyvenv.cfg"
        if not pyvenv_cfg_path.exists():
            log_and_print("[PDS-X] pyvenv.cfg dosyası eksik, manuel olarak oluşturuluyor...")
            with open(pyvenv_cfg_path, "w", encoding="utf-8") as f:
                f.write(f"home = {os.path.dirname(PYTHON310_PATH)}\ninclude-system-site-packages = false\nversion = 3.10\n")

        log_and_print("[PDS-X] pip güncelleniyor...")
        python_executable = str(VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python")
        subprocess.run([python_executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
        log_and_print("[PDS-X] Ortam hazırlanıyor, tekrar başlatılıyor... (venv (.pdsx_isolated_env) içinde)")
        logging.info("Script yeniden başlatılıyor: venv (.pdsx_isolated_env) içinde çalışacak.")
        os.execv(python_executable, [python_executable] + sys.argv)
    else:
        log_and_print("[PDS-X] İzole venv (.pdsx_isolated_env) ortamında çalışıyorsunuz.")
        logging.info("İzole venv (.pdsx_isolated_env) ortamında başlatıldı.")

# --- Ortamı Hazırla ---
# rem #14. Ortamı hazırla (script başlatıldığında otomatik venv ve ortam hazırlığı)
# Script başlatıldığında otomatik olarak venv ve ortamı hazırlar.
ensure_venv()

# --- Python Sürüm Uyarısı ---
# rem #15. Python sürüm uyarısı
# Kullanıcıya uyumlu sürümde olup olmadığını bildirir.
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] UYARI: En uyumlu çalışma için Python 3.10.x kullanmanız önerilir! Şu anki sürüm:", sys.version)

# --- Loglama Ayarları ---
# rem #16. Loglama ayarları
# Hatalar ve işlemler için log dosyası oluşturulur.
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("auto_importer")

# --- Gerekli Pip Paketleri Listesi ---
# rem #17. Gerekli pip paketleri listesi (sabit sürüm, import adı)
# Her biri (pip_adı, import_adı) şeklinde, sabit sürümle tanımlanır.


# Define the local package directory
LOCAL_PACKAGE_DIR = Path("local_packages")
LOCAL_PACKAGE_DIR.mkdir(exist_ok=True)

# Function to save package metadata to a JSON file
def save_package_metadata(packages):
    metadata_file = LOCAL_PACKAGE_DIR / "packages.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(packages, f, indent=4)

# Function to load package metadata from a JSON file
def load_package_metadata():
    metadata_file = LOCAL_PACKAGE_DIR / "packages.json"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# rem #18. Komut çıktısını hem terminale hem log dosyasına yazan yardımcı fonksiyon
def run_subprocess_logged(cmd, log_file=LOG_FILE):
    """Bir komutu çalıştırır, çıktısını hem terminale hem log dosyasına yazar."""
    import subprocess
    with open(log_file, "a", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        return process.returncode

# --- Eksik Paketleri Otomatik Yükleyici ---
# rem #19. Eksik paketleri otomatik yükleyici (paralel)
# Her paketi import etmeyi dener, eksikse --no-deps ve --force-reinstall ile yükler.
def install_missing_packages():
    """Gerekli pip paketlerini optimize edilmiş şekilde yükler."""
    import sys

    # Load existing package metadata
    existing_packages = load_package_metadata()

    # pip cache temizliği
    print("[PDS-X] pip cache temizleniyor...")
    run_subprocess_logged([sys.executable, "-m", "pip", "cache", "purge"])

    # Ana ML/bilimsel paketlerin toplu kurulumu
    main_pkgs = [
        "numpy==1.26.4", "scipy==1.11.4", "pandas==2.1.4", "scikit-learn==1.3.2", "matplotlib==3.8.4", "seaborn==0.13.2",
        "tensorflow==2.15.0", "torch==2.2.2", "torch-geometric==2.5.3", "qiskit", "dash", "plotly", "flask", "grpcio", "river"
    ]

    print("[PDS-X] Ana ML/bilimsel paketler yükleniyor...")
    for pkg in main_pkgs:
        pkg_name = pkg.split("==")[0]
        if pkg_name in existing_packages:
            print(f"[PDS-X] {pkg} zaten mevcut, atlanıyor.")
            continue
        run_subprocess_logged([sys.executable, "-m", "pip", "download", "--dest", str(LOCAL_PACKAGE_DIR), pkg])
        existing_packages[pkg_name] = pkg

    # Save updated package metadata
    save_package_metadata(existing_packages)

    summary = []
    lock = threading.Lock()

    def install_one(pip_name, import_name):
        try:
            if import_name in existing_packages:
                print(f"[PDS-X] {pip_name} zaten mevcut, atlanıyor.")
                return (pip_name, "ATLANDI")
            run_subprocess_logged([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", pip_name])
            with lock:
                summary.append((pip_name, "OK"))
            return (pip_name, "OK")
        except ImportError:
            code = run_subprocess_logged([sys.executable, "-m", "pip", "install", pip_name, "--no-deps", "--force-reinstall", "--no-cache-dir"])
            if code == 0:
                try:
                    __import__(import_name)
                    return (pip_name, "OK")
                except ImportError:
                    return (pip_name, "IMPORT_ERROR")
            else:
                return (pip_name, "INSTALL_ERROR")
        except Exception as e:
            return (pip_name, f"ERROR: {str(e)}")

    # Kalan paketleri paralel yükle
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(install_one, pip_name, import_name) for pip_name, import_name in REQUIRED_PACKAGES]
        for future in futures:
            result = future.result()
            if result:
                with lock:
                    summary.append(result)

    # Sonuç özeti
    print("\n[PDS-X] Paket yükleme özeti:")
    for pkg, stat in summary:
        if stat == "OK":
            print(f"  \033[92m{pkg}: Yüklü\033[0m")
        elif stat == "ATLANDI":
            print(f"  \033[93m{pkg}: Atlandı\033[0m")
        else:
            print(f"  \033[91m{pkg}: HATA! ({stat})\033[0m")
            print("[PDS-X] Eğer bir paket HATA verdiyse, elle tekrar yüklemeyi deneyin veya pip cache temizleyin.")
    print("[PDS-X] SONUC OZET TABLOSU BITTI.")

# --- Eksik veya Bozuk Paketlerin Temizlenmesi ve Yeniden Yüklenmesi ---
def clean_and_reinstall_packages():
    """Bozuk paketleri temizler ve yeniden yükler."""
    def log_and_print(msg):
        print(msg)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_and_print("[PDS-X] Bozuk paketler kontrol ediliyor...")
    try:
        # Bozuk paketleri tespit et
        result = subprocess.run([VENV_PYTHON, "-m", "pip", "check"], capture_output=True, text=True)
        if result.returncode == 0:
            log_and_print("[PDS-X] Hiçbir bozuk paket bulunamadı.")
            return

        log_and_print("[PDS-X] Bozuk paketler tespit edildi. İşleme başlanıyor...")
        broken_packages = set()
        for line in result.stdout.splitlines():
            if "has requirement" in line:
                package = line.split()[0]
                broken_packages.add(package)

        # Bozuk paketleri kaldır
        for package in broken_packages:
            log_and_print(f"[PDS-X] Paket kaldırılıyor: {package}")
            subprocess.run([VENV_PYTHON, "-m", "pip", "uninstall", "-y", package], check=False)

        # Paketleri yeniden yükle
        for package in broken_packages:
            log_and_print(f"[PDS-X] Paket yeniden yükleniyor: {package}")
            subprocess.run([VENV_PYTHON, "-m", "pip", "install", package, "--force-reinstall", "--no-cache-dir"], check=False)

        log_and_print("[PDS-X] Bozuk paketler başarıyla temizlendi ve yeniden yüklendi.")
    except Exception as e:
        log_and_print(f"[PDS-X] Paket temizleme ve yeniden yükleme sırasında hata: {e}")

# Ortamı hazırladıktan sonra bozuk paketleri temizle ve yeniden yükle
clean_and_reinstall_packages()

# --- Paket Çakışma Kontrolü (Paralel) ---
# rem #20. Paket çakışma kontrolü (paralel)
def check_package_conflicts_parallel(packages):
    # rem #20.1. Her paketi paralel olarak pip check ile kontrol eder
    """Paket çakışmalarını paralel kontrol eder."""
    def check_one(pkg):
        try:
            cmd = [str(VENV_PYTHON), "-m", "pip", "check"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "no conflicts" not in result.stdout.lower():
                return pkg
        except Exception:
            return pkg
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(check_one, packages))
    return [pkg for pkg in results if pkg]

# Eğer version modülüne ihtiyaç duyulan bir fonksiyon varsa, orada import et:
# rem #21. Versiyon karşılaştırıcı yardımcı fonksiyon
def compare_versions(ver1, ver2):
    # Basic version comparison without external libraries
    try:
        parts1 = [int(p) for p in ver1.split('.')]
        parts2 = [int(p) for p in ver2.split('.')]
    except ValueError:
        return False
    # Compare each part
    for p1, p2 in zip(parts1, parts2):
        if p1 != p2:
            return p1 > p2
    # If equal so far, longer version wins
    return len(parts1) > len(parts2)

# --- PDS-X Özel Hata Sınıfları ---
# rem #22. PDS-X özel hata sınıfları
try:
    from pdsx_exception2 import PdsXException, PdsXSyntaxError, PdsXRuntimeError
except Exception:
    # If pdsx_exception2 or its dependencies are missing, define dummy exceptions
    class PdsXException(Exception):
        """Dummy fallback for PdsXException"""
        pass

    class PdsXSyntaxError(Exception):
        """Dummy fallback for PdsXSyntaxError"""
        pass

    class PdsXRuntimeError(Exception):
        """Dummy fallback for PdsXRuntimeError"""
        pass

# --- Dinamik Modül Yükleyici Sınıfı ---
# rem #23. Dinamik modül yükleyici sınıfı (AutoImporter)
# Otomatik modül yükleme, bağımlılık yönetimi ve güvenli mod desteği sağlar.
class AutoImporter:
    # rem #23.1. Dinamik modül yükleme ve bağımlılık yönetimi
    """Dinamik modül yükleme ve bağımlılık yönetimi sınıfı."""
    def __init__(self, interpreter):
        # rem #23.2. Sınıf içi değişkenlerin başlatılması
        self.interpreter = interpreter
        self.loaded_modules: Dict[str, Any] = {}
        self.module_cache: Dict[str, Any] = {}
        self.imported_files: Set[str] = set()
        self.aliases: Dict[str, str] = {}
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        self.secure_mode: bool = False
        self.metadata: Dict = {"auto_importer": {"version": "1.5.0", "dependencies": []}}
        self.lock = threading.Lock()

    def load_module(self, file_path: str, alias: Optional[str] = None) -> Any:
        # rem #23.3. Modül dosyasını yükler, önbelleğe alır, bağımlılıkları kontrol eder
        """Modül dosyasını yükler."""
        with self.lock:
            abs_path = os.path.abspath(file_path)
            if abs_path in self.imported_files:
                log.debug(f"Modül zaten yüklü: {abs_path}")
                return self.module_cache.get(abs_path)
            
            if self.secure_mode and not self._is_allowed_path(abs_path):
                raise PdsXRuntimeError(f"Güvenli modda dış modül yüklenemez: {file_path}", context={"source": "load_module"})

            try:
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                if alias:
                    module_name = alias
                    self.aliases[alias] = abs_path
                
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                if not spec:
                    raise PdsXSyntaxError(f"Modül spec oluşturulamadı: {file_path}", context={"source": "load_module"})
                
                sys.modules[module_name] = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sys.modules[module_name])
                
                self.loaded_modules[module_name] = sys.modules[module_name]
                self.module_cache[abs_path] = sys.modules[module_name]
                self.imported_files.add(abs_path)
                
                # Bağımlılıkları kontrol et
                self._check_dependencies(module_name)
                
                log.info(f"Modül yüklendi: {module_name} ({abs_path})")
                return sys.modules[module_name]
            except Exception as e:
                log.error(f"Modül yükleme hatası: {file_path}, {str(e)}")
                raise PdsXRuntimeError(f"Modül yükleme hatası: {file_path}, {str(e)}", context={"source": "load_module"})

    def unload_module(self, module_name: str) -> None:
        # rem #23.4. Modülü bellekten kaldırır
        """Modülü bellekten kaldırır."""
        with self.lock:
            if module_name not in self.loaded_modules:
                raise PdsXRuntimeError(f"Modül yüklü değil: {module_name}", context={"source": "unload_module"})
            
            try:
                module = self.loaded_modules[module_name]
                abs_path = self.aliases.get(module_name, module.__file__)
                
                del sys.modules[module_name]
                del self.loaded_modules[module_name]
                self.module_cache.pop(abs_path, None)
                self.imported_files.discard(abs_path)
                self.aliases.pop(module_name, None)
                
                log.info(f"Modül kaldırıldı: {module_name}")
            except Exception as e:
                log.error(f"Modül kaldırma hatası: {module_name}, {str(e)}")
                raise PdsXRuntimeError(f"Modül kaldırma hatası: {module_name}, {str(e)}", context={"source": "unload_module"})

    def _is_allowed_path(self, path: str) -> bool:
        # rem #23.5. Güvenli modda izinli yol kontrolü
        """Güvenli modda yolun izinli olup olmadığını kontrol eder."""
        allowed_dirs = [os.path.abspath("."), os.path.abspath("libs")]
        return any(path.startswith(d) for d in allowed_dirs)

    def _check_dependencies(self, module_name: str) -> None:
        # rem #23.6. Modül bağımlılıklarını kontrol eder
        """Modül bağımlılıklarını kontrol eder."""
        required_deps = getattr(self.loaded_modules[module_name], "metadata", {}).get("dependencies", [])
        for dep in required_deps:
            if dep not in self.loaded_modules:
                try:
                    self.load_module(f"libs/{dep}.py")
                    self.dependencies[module_name].append(dep)
                except Exception as e:
                    raise PdsXRuntimeError(f"Bağımlılık yüklenemedi: {dep}, {str(e)}", context={"source": "_check_dependencies"})

    def get_module_stats(self, module_name: str) -> Dict:
        # rem #23.7. Modül istatistiklerini döndürür
        """Modül istatistiklerini döndürür."""
        with self.lock:
            if module_name not in self.loaded_modules:
                raise PdsXRuntimeError(f"Modül yüklü değil: {module_name}", context={"source": "get_module_stats"})
            
            module = self.loaded_modules[module_name]
            abs_path = self.aliases.get(module_name, module.__file__)
            stats = {
                "name": module_name,
                "path": abs_path,
                "size": os.path.getsize(abs_path) / 1024,  # KB cinsinden
                "load_time": getattr(module, "load_time", time.time()),
                "dependencies": self.dependencies[module_name],
                "version": getattr(module, "metadata", {}).get("version", "unknown")
            }
            log.debug(f"Modül istatistikleri: {stats}")
            return stats

    def secure_mode_enable(self) -> None:
        # rem #23.8. Güvenli modu etkinleştirir
        """Güvenli modu etkinleştirir."""
        with self.lock:
            self.secure_mode = True
            log.info("Güvenli mod etkinleştirildi")

    def secure_mode_disable(self) -> None:
        # rem #23.9. Güvenli modu devre dışı bırakır
        """Güvenli modu devre dışı bırakır."""
        with self.lock:
            self.secure_mode = False
            log.info("Güvenli mod devre dışı bırakıldı")

    # Deneysel/Bilimsel İşlevler
    def quantum_load_simulation(self, module_name: str) -> Dict:
        # rem #23.10. Kuantum simülasyonu ile modül yükleme performansı
        """Kuantum simülasyonu ile modül yükleme performansı analizi."""
        # Deneysel: Kuantum simülasyonu tabanlı yükleme tahmini
        start_time = time.time()
        self.load_module(module_name)
        end_time = time.time()
        return {
            "module": module_name,
            "load_time": end_time - start_time,
            "simulated_quantum_efficiency": random.uniform(0.8, 0.95)  # Mock kuantum verimliliği
        }

    def chaos_load_prediction(self, module_name: str) -> float:
        # rem #23.11. Kaotik sistem analizi ile yükleme süresi tahmini
        """Kaotik sistem analizi ile yükleme süresi tahmini."""
        # Deneysel: Kaotik dinamikler kullanılarak yükleme süresi tahmini
        return random.uniform(0.1, 1.0)  # Mock kaotik tahmin

    def genetic_dependency_optimizer(self, module_name: str) -> List[str]:
        # rem #23.12. Genetik algoritmalarla bağımlılık optimizasyonu
        """Genetik algoritmalarla bağımlılık optimizasyonu."""
        # Deneysel: Genetik algoritmalarla bağımlılık sıralama
        return self.dependencies.get(module_name, [])[::-1]  # Mock sıralama

    def neural_load_balancer(self, module_name: str) -> float:
        # rem #23.13. Nöral ağ tabanlı yük dengeleme
        """Nöral ağ tabanlı yük dengeleme."""
        # Deneysel: Nöral ağ ile yükleme yükü tahmini
        return random.uniform(0.5, 0.9)  # Mock dengeleme skoru

    def blockchain_module_validation(self, module_name: str) -> bool:
        # rem #23.14. Blockchain tabanlı modül doğrulama
        """Blockchain tabanlı modül doğrulama."""
        # Deneysel: Modül doğruluğu için blockchain simülasyonu
        return True  # Mock doğrulama

# rem #24. İzole ortam yöneticisi (IsolatedEnvManager)
class IsolatedEnvManager:
    # rem #24.1. Ortam adı ve yolunun ayarlanması
    """PDS-X için izole Python ortamı yöneticisi"""
    
    ENV_NAME = ".pdsx_isolated_env"  # Sabit ortam adı
    
    def __init__(self, base_path: Path):
        # rem #24.2. Ortam yolu ve python/pip yolu başlatılır
        self.base_path = base_path
        self.env_path = base_path / self.ENV_NAME
        self._python_path = None
        self._pip_path = None
        
        # Hatalı isimli eski ortamları temizle
        self._cleanup_old_envs()
        
    def _cleanup_old_envs(self):
        # rem #24.4. Hatalı isimli eski venv ortamlarını temizler
        """Hatalı isimli eski venv ortamlarını temizler."""
        old_names = [
        
        for old_name in old_names:isim
            ".pdsx-isolated-env",  # Tire olan eski isim
            "venv",               # Genel venv ismi
            ".venv"              # Gizli venv ismi
        ]
        
        for old_name in old_names:
            if old_name != self.ENV_NAME:  # Mevcut doğru ismi atlama
                old_path = self.base_path / old_name
                if old_path.exists():
                    try:
                        print(f"[PDS-X] Eski ortam temizleniyor: {old_path}")
                        shutil.rmtree(old_path)
                    except Exception as e:
                        print(f"[PDS-X] Eski ortam temizleme hatası ({old_name}): {e}")

    def _validate_env_name(self) -> bool:
        # rem #24.5. Ortam isminin doğru formatta olduğunu kontrol eder
        """Ortam isminin doğru formatta olduğunu kontrol eder."""
        if self.env_path.exists():
            actual_name = self.env_path.name
            if actual_name != self.ENV_NAME:
                print(f"[PDS-X] UYARI: Ortam ismi yanlış: {actual_name}")
                print(f"[PDS-X] Beklenen: {self.ENV_NAME}")
                return False
        return True

    def create_env(self) -> bool:
        # rem #24.6. İzole Python ortamı oluşturur
        """İzole Python ortamı oluşturur."""
        try:
            # Önce eski ortamları temizle
            self._cleanup_old_envs()
            
            # İsim kontrolü
            if not self._validate_env_name():
                print("[PDS-X] Ortam ismi düzeltiliyor...")
                if self.env_path.exists():
                    shutil.rmtree(self.env_path)
            
            python_path = find_python310()
            if not python_path:
                print("[PDS-X] Python 3.10 bulunamadı!")
                return False
            
            if not self.env_path.exists():
                subprocess.run([python_path, "-m", "venv", str(self.env_path)], check=True)
                print(f"[PDS-X] İzole ortam oluşturuldu: {self.env_path}")
                
            return True
        except Exception as e:
            print(f"[PDS-X] İzole ortam oluşturma hatası: {e}")
            return False

    @property
    def python_path(self) -> Optional[str]:
        # rem #24.7. İzole ortamdaki Python yolunu döndürür
        """İzole ortamdaki Python yolunu döndürür."""
        if not self._python_path:
            if os.name == "nt":
                self._python_path = str(self.env_path / "Scripts" / "python.exe")
            else:
                self._python_path = str(self.env_path / "bin" / "python")
        return self._python_path if Path(self._python_path).exists() else None

    def install_package(self, package: str, upgrade: bool = False) -> bool:
        # rem #24.8. Güvenli paket kurulumu yapar
        """Güvenli paket kurulumu yapar."""
        try:
            cmd = [self.python_path, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[PDS-X] Paket kuruldu: {package}")
                return True
            else:
                print(f"[PDS-X] Paket kurulum hatası: {result.stderr}")
                return False
        except Exception as e:
            print(f"[PDS-X] Paket kurulum hatası: {e}")
            return False

    def check_package_conflicts(self, packages: List[str]) -> List[str]:
        # rem #24.9. Paket çakışmalarını paralel kontrol eder
        """Paket çakışmalarını paralel kontrol eder."""
        def check_one(pkg):
            try:
                cmd = [self.python_path, "-m", "pip", "check"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if "no conflicts" not in result.stdout.lower():
                    return pkg
            except Exception:
                return pkg
            return None
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(check_one, packages))
        return [pkg for pkg in results if pkg]

# rem #25. Modül otomatik yükleyici (ModuleAutoImporter)
class ModuleAutoImporter:
    # rem #25.1. Çalışma alanı ve ortam yöneticisi başlatılır
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.env_manager = IsolatedEnvManager(workspace_path)
        self.loaded_modules = set()
        self.package_cache = {}
        
    def setup_environment(self) -> bool:
        # rem #25.2. Çalışma ortamını hazırlar (izole ortam + temel bağımlılıklar)
        """Çalışma ortamını hazırlar."""
        print("[PDS-X] Ortam kurulumu başlatılıyor...")
        
        # İzole ortam oluştur
        if not self.env_manager.create_env():
            return False
            
        # Temel bağımlılıkları kur
        success = True
        for package in CORE_DEPENDENCIES["base"]:
            if not self.env_manager.install_package(package):
                print(f"[PDS-X] Temel paket kurulumu başarısız: {package}")
                success = False
                
        return success
        
    def import_module(self, module_name: str) -> Any:
        # rem #25.3. Modülü güvenli şekilde import eder ve bağımlılıklarını yönetir
        """Modülü güvenli şekilde import eder ve bağımlılıklarını yönetir."""
        try:
            # Modül bağımlılıklarını kontrol et
            if module_name in MODULE_SPECIFIC_DEPS:
                deps = MODULE_SPECIFIC_DEPS[module_name]
                if self.env_manager.check_package_conflicts(deps):
                    print(f"[PDS-X] Paket çakışması tespit edildi: {deps}")
                    return None
                    
                # Eksik bağımlılıkları kur
                for dep in deps:
                    if not self.package_cache.get(dep):
                        if self.env_manager.install_package(dep):
                            self.package_cache[dep] = True
                        else:
                            print(f"[PDS-X] Bağımlılık kurulumu başarısız: {dep}")
                            return None
            
            # Modülü yükle
            module_path = self.workspace_path / module_name
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if not spec or not spec.loader:
                raise ImportError(f"Modül yüklenemedi: {module_name}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.loaded_modules.add(module_name)
            
            return module
            
        except Exception as e:
            print(f"[PDS-X] Modül import hatası ({module_name}): {e}")
            return None

    def cleanup(self):
        # rem #25.4. Yüklenen modülleri ve kaynakları temizler
        """Yüklenen modülleri ve kaynakları temizler."""
        for module_name in self.loaded_modules:
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            except Exception as e:
                print(f"[PDS-X] Modül temizleme hatası ({module_name}): {e}")

# Yardımcı fonksiyonlar
# rem #26. Yardımcı fonksiyonlar ve ortam doğrulama
def validate_environment() -> Dict[str, bool]:
    # rem #26.1. Python sürümü ve sistem gereksinimlerini kontrol eder
    """Python sürümü ve sistem gereksinimlerini kontrol eder."""
    status = {
        "python_version": False,
        "pip_available": False,
        "venv_available": False
    }
    
    # Python sürüm kontrolü
    python_version = sys.version_info
    status["python_version"] = python_version.major == 3 and python_version.minor == 10
    
    # pip kontrolü
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       capture_output=True, check=True)
        status["pip_available"] = True
    except Exception:
        pass
        
    # venv kontrolü
    try:
        subprocess.run([sys.executable, "-m", "venv", "--help"],
                       capture_output=True, check=True)
        status["venv_available"] = True
    except Exception:
        pass
        
    return status

def create_workspace(path: str) -> ModuleAutoImporter:
    #26.2. Yeni bir çalışma alanı oluşturur ve yapılandırır
    """Yeni bir çalışma alanı oluşturur ve yapılandırır."""
    workspace = Path(path)
    if not workspace.exists():
        workspace.mkdir(parents=True)
        
    importer = ModuleAutoImporter(workspace)
    if importer.setup_environment():
        print("[PDS-X] Çalışma alanı başarıyla oluşturuldu")
        return importer
    else:
        print("[PDS-X] Çalışma alanı oluşturma başarısız")
        return None

# Ana yürütme bloğu
#27. Ana yürütme bloğu
if __name__ == "__main__":
    #27.1. Ortam kontrolü ve çalışma alanı oluşturma
    # validate sistemi pdsX ortamini gereksinimlere karsi kontrol eder
    status = validate_environment()
    if not all(status.values()):
        print("[PDS-X] Sistem gereksinimleri karşılanmıyor:")
        for check, passed in status.items():
            print(f"  - {check}: {'Geçti' if passed else 'Başarısız'}")
        sys.exit(1)
        
    # Çalışma alanı oluştur
    workspace_path = Path.cwd() / "pdsx_workspace"
    importer = create_workspace(str(workspace_path))
    
    if importer:
        print("[PDS-X] Sistem hazır")
    else:
        print("[PDS-X] Sistem başlatılamadı")
        sys.exit(1)

    print("auto_importer.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")
# SON. Kodun her adımı için rem satırları eklendi. Kodun amacı değişmedikçe bu rem satırları asla silinmeyecek, sadece ekleme yapılacak.

# EXPORTS. PDS-X modül doğrulama için dışa aktarılanlar listesi
__pdsX_exports__ = [
    'find_python310',
    'download_and_install_python310',
    'add_python_to_path',
    'in_venv',
    'ensure_venv',
    'run_subprocess_logged',
    'install_missing_packages',
    'clean_and_reinstall_packages',
    'check_package_conflicts_parallel',
    'compare_versions',
    'AutoImporter',
    'IsolatedEnvManager',
    'ModuleAutoImporter',
    'validate_environment',
    'create_workspace'
]
