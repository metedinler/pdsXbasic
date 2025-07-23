# libx_data.py - PDS-X BASIC v14u Veri İşleme ve Boru Hattı Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Any, Callable, List, Dict, Optional, Union
from collections import deque
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_data")

class PdsXException(Exception):
    pass

class PipelineInstance:
    def __init__(self, commands: List[Dict], interpreter, alias: Optional[str] = None, return_var: Optional[str] = None):
        self.commands = commands
        self.interpreter = interpreter
        self.current_index = 0
        self.data = []
        self.active = False
        self.status = {"executed": [], "pending": commands.copy()}
        self.id = id(self)
        self.priority = "NORMAL"
        self.alias = alias
        self.return_var = return_var
        self.labels = {}
        self.lock = threading.Lock()

    def add_command(self, command: Dict, step_no: Optional[int] = None, position: Optional[str] = None) -> None:
        """Boru hattına yeni bir komut ekler."""
        with self.lock:
            if step_no is not None:
                self.commands.insert(int(step_no), command)
                self.status["pending"].insert(int(step_no), command)
            elif position == "START":
                self.commands.insert(0, command)
                self.status["pending"].insert(0, command)
            elif position == "END":
                self.commands.append(command)
                self.status["pending"].append(command)
            else:
                self.commands.append(command)
                self.status["pending"].append(command)

    def remove_command(self, step_no: int) -> None:
        """Boru hattından bir komutu siler."""
        with self.lock:
            if 0 <= step_no < len(self.commands):
                self.status["pending"].remove(self.commands[step_no])
                self.commands.pop(step_no)
            else:
                raise PdsXException(f"Geçersiz adım numarası: {step_no}")

    def execute(self, parallel: bool = False) -> Any:
        """Boru hattını yürütür."""
        self.active = True
        try:
            if parallel:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.interpreter.execute_command, cmd) for cmd in self.commands[self.current_index:]]
                    for future, cmd in zip(futures, self.commands[self.current_index:]):
                        future.result()  # Hata yakalama için
                        self.current_index += 1
                        self.status["executed"].append(cmd)
                        self.status["pending"].remove(cmd)
            else:
                for cmd in self.commands[self.current_index:]:
                    self.interpreter.execute_command(cmd)
                    self.current_index += 1
                    self.status["executed"].append(cmd)
                    self.status["pending"].remove(cmd)
            if self.return_var:
                return self.interpreter.current_scope().get(self.return_var, None)
        except Exception as e:
            log.error(f"Boru hattı yürütme hatası: {str(e)}")
            raise PdsXException(f"Boru hattı yürütme hatası: {str(e)}")
        finally:
            self.active = False

    def next(self) -> None:
        """Bir sonraki komutu yürütür."""
        with self.lock:
            if self.current_index < len(self.commands):
                self.interpreter.execute_command(self.commands[self.current_index])
                self.status["executed"].append(self.commands[self.current_index])
                self.status["pending"].remove(self.commands[self.current_index])
                self.current_index += 1

    def set_label(self, label: Union[str, int], step_no: int) -> None:
        """Boru hattına etiket ekler."""
        with self.lock:
            if isinstance(label, (str, int)):
                self.labels[str(label)] = step_no.to_bytes(1, 'big')
            elif isinstance(label, bytes):
                self.labels[label.decode()] = label

    def get_label(self, label: str) -> int:
        """Etiketin adım numarasını döndürür."""
        with self.lock:
            return int.from_bytes(self.labels.get(str(label), b'\x00'), 'big')

    def get_status(self) -> Dict:
        """Boru hattı durumunu döndürür."""
        with self.lock:
            return self.status

class LibXData:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.pipelines = {}
        self.active_pipes = 0
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.metadata = {"libx_data": {"version": "1.0.0", "dependencies": ["numpy", "pandas", "scipy"]}}

    def parse_pipe_command(self, command: str) -> None:
        """PIPE komutunu ayrıştırır ve boru hattı oluşturur."""
        match = re.match(r"PIPE\s+(\w+)(?:\s+RETURN\s+(\w+))?(?:\s+PARALLEL)?(?:\s+ONERROR\s+(\w+))?", command, re.IGNORECASE)
        if not match:
            raise PdsXException("PIPE komutunda sözdizimi hatası")
        
        pipe_id, return_var, onerror = match.groups()
        parallel = "PARALLEL" in command.upper()
        commands = []
        
        # Boru hattı komutlarını ayrıştır
        lines = command.split("\n")[1:]  # İlk satırı atla
        for line in lines:
            line = line.strip()
            if not line or line.upper().startswith("END PIPE"):
                break
            if line.upper().startswith("STEP"):
                cmd = {"type": "step", "command": line[5:].strip()}
                commands.append(cmd)
            elif line.upper().startswith("LABEL"):
                label, step_no = re.match(r"LABEL\s+(\w+)\s+(\d+)", line, re.IGNORECASE).groups()
                commands.append({"type": "label", "label": label, "step_no": int(step_no)})
        
        # Boru hattı örneği oluştur
        pipeline = PipelineInstance(commands, self.interpreter, alias=pipe_id, return_var=return_var)
        if onerror:
            pipeline.onerror = onerror
        self.pipelines[pipe_id] = pipeline
        self.active_pipes += 1
        
        # Boru hattını yürüt
        if parallel:
            self.executor.submit(pipeline.execute, parallel=True)
        else:
            pipeline.execute(parallel=False)

    def filter_data(self, data: Any, condition: Callable) -> Any:
        """Veriyi filtreler."""
        if isinstance(data, pd.DataFrame):
            return data[data.apply(condition, axis=1)]
        elif isinstance(data, np.ndarray):
            return data[condition(data)]
        elif isinstance(data, list):
            return [x for x in data if condition(x)]
        else:
            raise PdsXException("Desteklenmeyen veri tipi")

    def aggregate_data(self, data: Any, key: str, agg_func: Callable) -> Any:
        """Veriyi gruplar ve toplar."""
        if isinstance(data, pd.DataFrame):
            return data.groupby(key).agg(agg_func)
        elif isinstance(data, list):
            grouped = {}
            for item in data:
                k = item[key] if isinstance(item, dict) else getattr(item, key)
                if k not in grouped:
                    grouped[k] = []
                grouped[k].append(item)
            return {k: agg_func(v) for k, v in grouped.items()}
        else:
            raise PdsXException("Desteklenmeyen veri tipi")

    def transform_data(self, data: Any, func: Callable) -> Any:
        """Veriyi dönüştürür."""
        if isinstance(data, pd.DataFrame):
            return data.apply(func)
        elif isinstance(data, np.ndarray):
            return np.vectorize(func)(data)
        elif isinstance(data, list):
            return [func(x) for x in data]
        else:
            raise PdsXException("Desteklenmeyen veri tipi")

    def merge_data(self, data1: Any, data2: Any, on: Optional[str] = None) -> Any:
        """İki veri setini birleştirir."""
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            return pd.merge(data1, data2, on=on)
        elif isinstance(data1, list) and isinstance(data2, list):
            return data1 + data2
        elif isinstance(data1, dict) and isinstance(data2, dict):
            return {**data1, **data2}
        else:
            raise PdsXException("Desteklenmeyen veri tipi")

    def sort_data(self, data: Any, key: Optional[Callable] = None, column: Optional[str] = None) -> Any:
        """Veriyi sıralar."""
        if isinstance(data, pd.DataFrame):
            return data.sort_values(column)
        elif isinstance(data, list):
            return sorted(data, key=key)
        else:
            raise PdsXException("Desteklenmeyen veri tipi")

    # Veri Bilimi Fonksiyonları
    def mean(self, data: Any) -> float:
        """Ortalama hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.mean().to_dict()
        elif isinstance(data, np.ndarray):
            return np.mean(data)
        elif isinstance(data, list):
            return sum(data) / len(data) if data else 0
        raise PdsXException("Desteklenmeyen veri tipi")

    def median(self, data: Any) -> float:
        """Medyan hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.median().to_dict()
        elif isinstance(data, np.ndarray):
            return np.median(data)
        elif isinstance(data, list):
            return np.median(data)
        raise PdsXException("Desteklenmeyen veri tipi")

    def mode(self, data: Any) -> Any:
        """Mod hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.mode().iloc[0].to_dict()
        elif isinstance(data, np.ndarray):
            return stats.mode(data)[0][0]
        elif isinstance(data, list):
            return stats.mode(data)[0][0]
        raise PdsXException("Desteklenmeyen veri tipi")

    def std(self, data: Any) -> float:
        """Standart sapma hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.std().to_dict()
        elif isinstance(data, np.ndarray):
            return np.std(data)
        elif isinstance(data, list):
            return np.std(data)
        raise PdsXException("Desteklenmeyen veri tipi")

    def var(self, data: Any) -> float:
        """Varyans hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.var().to_dict()
        elif isinstance(data, np.ndarray):
            return np.var(data)
        elif isinstance(data, list):
            return np.var(data)
        raise PdsXException("Desteklenmeyen veri tipi")

    def sum(self, data: Any) -> float:
        """Toplam hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.sum().to_dict()
        elif isinstance(data, np.ndarray):
            return np.sum(data)
        elif isinstance(data, list):
            return sum(data)
        raise PdsXException("Desteklenmeyen veri tipi")

    def prod(self, data: Any) -> float:
        """Çarpım hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.prod().to_dict()
        elif isinstance(data, np.ndarray):
            return np.prod(data)
        elif isinstance(data, list):
            return np.prod(data)
        raise PdsXException("Desteklenmeyen veri tipi")

    def percentile(self, data: Any, q: float) -> float:
        """Persentil hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.quantile(q / 100).to_dict()
        elif isinstance(data, np.ndarray):
            return np.percentile(data, q)
        elif isinstance(data, list):
            return np.percentile(data, q)
        raise PdsXException("Desteklenmeyen veri tipi")

    def quantile(self, data: Any, q: float) -> float:
        """Kuantil hesaplar."""
        if isinstance(data, pd.DataFrame):
            return data.quantile(q).to_dict()
        elif isinstance(data, np.ndarray):
            return np.quantile(data, q)
        elif isinstance(data, list):
            return np.quantile(data, q)
        raise PdsXException("Desteklenmeyen veri tipi")

    def corr(self, data1: Any, data2: Any) -> float:
        """Korelasyon hesaplar."""
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            return data1.corrwith(data2).to_dict()
        elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
            return np.corrcoef(data1, data2)[0, 1]
        elif isinstance(data1, list) and isinstance(data2, list):
            return np.corrcoef(data1, data2)[0, 1]
        raise PdsXException("Desteklenmeyen veri tipi")

    def cov(self, data1: Any, data2: Any) -> float:
        """Kovaryans hesaplar."""
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            return data1.cov(data2).to_dict()
        elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
            return np.cov(data1, data2)[0, 1]
        elif isinstance(data1, list) and isinstance(data2, list):
            return np.cov(data1, data2)[0, 1]
        raise PdsXException("Desteklenmeyen veri tipi")

    def describe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Veri setinin istatistiksel özetini döndürür."""
        if isinstance(data, pd.DataFrame):
            return data.describe()
        raise PdsXException("Yalnızca DataFrame için geçerli")

    def groupby(self, data: pd.DataFrame, column: str) -> pd.core.groupby.GroupBy:
        """Veriyi sütuna göre gruplar."""
        if isinstance(data, pd.DataFrame):
            return data.groupby(column)
        raise PdsXException("Yalnızca DataFrame için geçerli")

    def pivot_table(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Pivot tablo oluşturur."""
        if isinstance(data, pd.DataFrame):
            return data.pivot_table(**kwargs)
        raise PdsXException("Yalnızca DataFrame için geçerli")

    def ttest(self, sample1: Any, sample2: Any) -> Dict:
        """T-testi yapar."""
        stat, p = stats.ttest_ind(sample1, sample2)
        return {"statistic": stat, "pvalue": p}

    def chisquare(self, observed: Any) -> Dict:
        """Ki-kare testi yapar."""
        stat, p = stats.chisquare(observed)
        return {"statistic": stat, "pvalue": p}

    def anova(self, *groups: Any) -> Dict:
        """ANOVA testi yapar."""
        stat, p = stats.f_oneway(*groups)
        return {"statistic": stat, "pvalue": p}

    def regress(self, x: Any, y: Any) -> Dict:
        """Lineer regresyon yapar."""
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }

if __name__ == "__main__":
    print("libx_data.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")