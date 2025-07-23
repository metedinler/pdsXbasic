import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] HATA: Bu modül sadece Python 3.10 ortamında çalışır! Lütfen pdsX'in ana başlatıcısını kullanın.")
    sys.exit(1)

# libx_logic.py - PDS-X BASIC v14u Mantıksal Programlama Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

from collections import defaultdict

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_logic")

class PdsXException(Exception):
    pass

class PrologEngine:
    def __init__(self):
        self.facts = []
        self.rules = []
        self.debug = False
        self.handlers = defaultdict(list)  # Olay tabanlı handler'lar

    def add_fact(self, fact: str) -> None:
        """Bilgi tabanına bir gerçek ekler."""
        self.facts.append(fact)
        if self.debug:
            log.debug(f"Fact eklendi: {fact}")

    def add_rule(self, head: str, body: List[str], func: Optional[Callable] = None) -> None:
        """Bilgi tabanına bir kural ekler."""
        func = func or (lambda x: x)
        self.rules.append((head, body, func))
        if self.debug:
            log.debug(f"Rule eklendi: {head} :- {body}")

    def query(self, goal: str) -> bool:
        """Bilgi tabanında bir sorgu yürütür."""
        if self.debug:
            log.debug(f"Sorgu başlatılıyor: {goal}")
        bindings = self.backtrack(goal, {})
        if bindings:
            log.info(f"Evet: {goal}")
            if bindings:
                log.info("Sonuçlar:")
                for var, val in bindings.items():
                    log.info(f"  {var} = {val}")
            return True
        log.info(f"Hayır: {goal}")
        return False

    def backtrack(self, goal: Any, bindings: Dict) -> Optional[Dict]:
        """Geri izleme ile sorgu çözümlemesi yapar."""
        if self.debug:
            log.debug(f"Backtrack: Goal={goal}, Bindings={bindings}")

        if isinstance(goal, tuple) and goal[0] == "AND":
            local_bindings = bindings.copy()
            for subgoal in goal[1:]:
                result = self.backtrack(subgoal, local_bindings)
                if result is None:
                    if self.debug:
                        log.debug(f"AND başarısız: {subgoal}")
                    return None
                local_bindings.update(result)
            return local_bindings

        elif isinstance(goal, tuple) and goal[0] == "OR":
            for subgoal in goal[1:]:
                result = self.backtrack(subgoal, bindings.copy())
                if result:
                    return result
            return None

        elif isinstance(goal, tuple) and goal[0] == "NOT":
            result = self.backtrack(goal[1], bindings.copy())
            return {} if result is None else None

        elif isinstance(goal, tuple) and goal[0] == "XOR":
            successes = [self.backtrack(subgoal, bindings.copy()) for subgoal in goal[1:]]
            count = sum(1 for r in successes if r)
            return {} if count == 1 else None

        elif isinstance(goal, tuple) and goal[0] == "IMP":
            if not self.backtrack(goal[1], bindings.copy()) or self.backtrack(goal[2], bindings.copy()):
                return {}
            return None

        elif isinstance(goal, tuple) and goal[0] == "BI-COND":
            a = self.backtrack(goal[1], bindings.copy())
            b = self.backtrack(goal[2], bindings.copy())
            return {} if (bool(a) == bool(b)) else None

        else:
            for fact in self.facts:
                new_bindings = self.unify(goal, fact, bindings.copy())
                if new_bindings is not None:
                    return new_bindings

            for head, body, func in self.rules:
                head_bindings = self.unify(goal, head, bindings.copy())
                if head_bindings is not None:
                    for subgoal in body:
                        subgoal_bindings = self.backtrack(subgoal, head_bindings)
                        if subgoal_bindings is None:
                            break
                        head_bindings.update(subgoal_bindings)
                    else:
                        return func(head_bindings)
            return None

    def unify(self, term1: Any, term2: Any, bindings: Dict) -> Optional[Dict]:
        """İki terimi birleştirir."""
        if self.debug:
            log.debug(f"Unify: {term1} <=> {term2}, Bindings={bindings}")

        if term1 == term2:
            return bindings

        if isinstance(term1, str) and term1.startswith("#"):
            if term1 in bindings:
                return self.unify(bindings[term1], term2, bindings)
            bindings[term1] = term2
            return bindings

        if isinstance(term2, str) and term2.startswith("#"):
            if term2 in bindings:
                return self.unify(term1, bindings[term2], bindings)
            bindings[term2] = term1
            return bindings

        if isinstance(term1, tuple) and isinstance(term2, tuple) and len(term1) == len(term2):
            for t1, t2 in zip(term1, term2):
                bindings = self.unify(t1, t2, bindings)
                if bindings is None:
                    return None
            return bindings

        if term1 == term2:
            return bindings
        return None

    def clear(self) -> None:
        """Bilgi tabanını temizler."""
        self.facts.clear()
        self.rules.clear()
        if self.debug:
            log.debug("Bilgi tabanı temizlendi.")

    def dump(self) -> None:
        """Bilgi tabanını döker."""
        log.info("Bilgi Tabanı:")
        log.info("  Gerçekler:")
        for fact in self.facts:
            log.info(f"    {fact}")
        log.info("  Kurallar:")
        for head, body, _ in self.rules:
            log.info(f"    {head} :- {body}")

    def count(self, goal: str) -> int:
        """Belirtilen hedef için eşleşme sayısını döndürür."""
        count = 0
        for fact in self.facts:
            if self.unify(goal, fact, {}) is not None:
                count += 1
        return count

    def exists(self, goal: str) -> bool:
        """Hedefin var olup olmadığını kontrol eder."""
        return self.query(goal)

    def forall(self, condition: str, items: List) -> bool:
        """Tüm öğeler için koşulu kontrol eder."""
        return all(self.query((condition, item)) for item in items)

    def enable_debug(self) -> None:
        """Hata ayıklama modunu etkinleştirir."""
        self.debug = True
        log.info("Debug modu aktif.")

    def disable_debug(self) -> None:
        """Hata ayıklama modunu devre dışı bırakır."""
        self.debug = False
        log.info("Debug modu kapalı.")

    def register_handler(self, event: str, handler: Callable) -> None:
        """Olay tabanlı bir handler kaydeder."""
        self.handlers[event].append(handler)
        if self.debug:
            log.debug(f"Handler kaydedildi: {event}")

    def unregister_handler(self, event: str, handler: Callable) -> None:
        """Olay tabanlı bir handler'ı kaldırır."""
        if event in self.handlers and handler in self.handlers[event]:
            self.handlers[event].remove(handler)
            if self.debug:
                log.debug(f"Handler kaldırıldı: {event}")

    def trigger_handlers(self, event: str, *args, **kwargs) -> None:
        """Olay için kayıtlı handler'ları tetikler."""
        if event in self.handlers:
            for handler in self.handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    log.error(f"Handler hatası: {event}, {str(e)}")

class LibXLogic:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.prolog_engine = PrologEngine()
        self.metadata = {"libx_logic": {"version": "1.0.0", "dependencies": []}}

    def parse_prolog_command(self, command: str) -> None:
        """Prolog komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("FACT "):
                fact = command[5:].strip()
                self.prolog_engine.add_fact(fact)
            elif command_upper.startswith("RULE "):
                match = re.match(r"RULE\s+(\w+)\s*:-\s*(.+)", command, re.IGNORECASE)
                if match:
                    head, body = match.groups()
                    body_terms = [term.strip() for term in body.split(",")]
                    self.prolog_engine.add_rule(head, body_terms)
                else:
                    raise PdsXException("RULE komutunda sözdizimi hatası")
            elif command_upper.startswith("QUERY "):
                goal = command[6:].strip()
                self.prolog_engine.query(goal)
            elif command_upper.startswith("REGISTER PROLOG HANDLER "):
                event = command[22:].strip()
                handler = lambda *args, **kwargs: self.interpreter.execute_command(event)
                self.prolog_engine.register_handler(event, handler)
            elif command_upper.startswith("UNREGISTER PROLOG HANDLER "):
                event = command[24:].strip()
                handler = lambda *args, **kwargs: self.interpreter.execute_command(event)
                self.prolog_engine.unregister_handler(event, handler)
            elif command_upper == "CLEAR":
                self.prolog_engine.clear()
            elif command_upper == "DUMP":
                self.prolog_engine.dump()
            elif command_upper.startswith("COUNT "):
                goal = command[6:].strip()
                count = self.prolog_engine.count(goal)
                self.interpreter.current_scope()["_COUNT"] = count
            elif command_upper.startswith("EXISTS "):
                goal = command[7:].strip()
                exists = self.prolog_engine.exists(goal)
                self.interpreter.current_scope()["_EXISTS"] = exists
            elif command_upper == "DEBUG ON":
                self.prolog_engine.enable_debug()
            elif command_upper == "DEBUG OFF":
                self.prolog_engine.disable_debug()
            else:
                raise PdsXException(f"Bilinmeyen Prolog komutu: {command}")
        except Exception as e:
            log.error(f"Prolog komut hatası: {str(e)}")
            raise PdsXException(f"Prolog komut hatası: {str(e)}")

    def prolog_query(self, goal: str) -> bool:
        """Prolog sorgusu yürütür (eski API için)."""
        return self.prolog_engine.query(goal)

    def prolog_assert(self, fact: str) -> None:
        """Prolog gerçeği ekler (eski API için)."""
        self.prolog_engine.add_fact(fact)

if __name__ == "__main__":
    print("libx_logic.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")