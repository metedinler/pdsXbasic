import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
    print("[PDS-X] HATA: Bu modül sadece Python 3.10 ortamında çalışır! Lütfen pdsX'in ana başlatıcısını kullanın.")
    sys.exit(1)

# libx_nlp.py - PDS-X BASIC v14u Doğal Dil İşleme Kütüphanesi
# Version: 1.0.0
# Date: May 12, 2025
# Author: xAI (Grok 3 ile oluşturuldu, Mete Dinler için)

import logging
import re
import subprocess
import sys
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import nltk
try:
    import spacy
except ImportError as e:
    print('spacy import error:', e)
    spacy = None
from transformers import pipeline
from textblob import TextBlob
import torch

# Loglama Ayarları
logging.basicConfig(
    filename="pdsxu_errors.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("libx_nlp")

class PdsXException(Exception):
    pass

class LibXNLP:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.spacy_nlp = None
        self.nltk_initialized = False
        self.transformers_pipelines = {}
        self.supported_languages = ["en", "tr", "fr", "de", "es"]
        self.model_cache = {}
        self.metadata = {"libx_nlp": {"version": "1.0.0", "dependencies": ["spacy", "nltk", "transformers", "textblob"]}}
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """NLP bağımlılıklarını başlatır."""
        try:
            # spaCy başlatma
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                log.info("spaCy modeli yükleniyor: en_core_web_sm")
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.spacy_nlp = spacy.load("en_core_web_sm")
            
            # NLTK başlatma
            if not self.nltk_initialized:
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("vader_lexicon", quiet=True)
                self.nltk_initialized = True
            
            log.info("NLP bağımlılıkları başarıyla başlatıldı")
        except Exception as e:
            log.error(f"NLP başlatma hatası: {str(e)}")
            raise PdsXException(f"NLP başlatma hatası: {str(e)}")

    def _load_transformer_pipeline(self, task: str, model: str) -> pipeline:
        """Transformers pipeline'ını yükler veya önbellekten alır."""
        cache_key = f"{task}_{model}"
        if cache_key not in self.transformers_pipelines:
            try:
                self.transformers_pipelines[cache_key] = pipeline(task, model=model, device=0 if torch.cuda.is_available() else -1)
                log.debug(f"Transformers pipeline yüklendi: {task}, {model}")
            except Exception as e:
                log.error(f"Transformers pipeline yükleme hatası: {task}, {model}, {str(e)}")
                raise PdsXException(f"Transformers pipeline yükleme hatası: {str(e)}")
        return self.transformers_pipelines[cache_key]

    def analyze_text(self, text: str, lang: str = "en") -> Dict:
        """Metni analiz eder (tokenizasyon, POS, NER, bağımlılık)."""
        if lang not in self.supported_languages:
            raise PdsXException(f"Desteklenmeyen dil: {lang}")
        
        try:
            if lang == "en":
                doc = self.spacy_nlp(text)
            else:
                # Çok dilli model için spaCy yükleme
                model_map = {"tr": "tr_core_news_sm", "fr": "fr_core_news_sm", "de": "de_core_news_sm", "es": "es_core_news_sm"}
                try:
                    doc = spacy.load(model_map[lang])(text)
                except OSError:
                    log.info(f"spaCy modeli yükleniyor: {model_map[lang]}")
                    subprocess.run([sys.executable, "-m", "spacy", "download", model_map[lang]], check=True)
                    doc = spacy.load(model_map[lang])(text)
            
            result = {
                "tokens": [token.text for token in doc],
                "pos": [(token.text, token.pos_) for token in doc],
                "ner": [(ent.text, ent.label_) for ent in doc.ents],
                "dependencies": [(token.text, token.dep_, token.head.text) for token in doc]
            }
            log.debug(f"Metin analizi tamamlandı: {text[:50]}...")
            return result
        except Exception as e:
            log.error(f"Metin analizi hatası: {str(e)}")
            raise PdsXException(f"Metin analizi hatası: {str(e)}")

    def tokenize(self, text: str, lang: str = "en") -> List[str]:
        """Metni token'lara ayırır."""
        try:
            if lang == "en":
                return nltk.word_tokenize(text)
            else:
                # spaCy ile çok dilli tokenizasyon
                doc = self.spacy_nlp(text) if lang == "en" else spacy.load(f"{lang}_core_news_sm")(text)
                return [token.text for token in doc]
        except Exception as e:
            log.error(f"Tokenizasyon hatası: {str(e)}")
            raise PdsXException(f"Tokenizasyon hatası: {str(e)}")

    def sentiment_analysis(self, text: str, lang: str = "en") -> Dict:
        """Metnin duygu analizini yapar."""
        try:
            if lang == "en":
                blob = TextBlob(text)
                sentiment = blob.sentiment
                result = {
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity
                }
            else:
                # Transformers ile çok dilli duygu analizi
                model = "nlptown/bert-base-multilingual-uncased-sentiment"
                pipe = self._load_transformer_pipeline("sentiment-analysis", model)
                result = pipe(text)[0]
                result = {"label": result["label"], "score": result["score"]}
            log.debug(f"Duygu analizi tamamlandı: {text[:50]}...")
            return result
        except Exception as e:
            log.error(f"Duygu analizi hatası: {str(e)}")
            raise PdsXException(f"Duygu analizi hatası: {str(e)}")

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Metni özetler."""
        try:
            summarizer = self._load_transformer_pipeline("summarization", "facebook/bart-large-cnn")
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
            log.debug(f"Metin özeti tamamlandı: {text[:50]}...")
            return summary
        except Exception as e:
            log.error(f"Metin özetleme hatası: {str(e)}")
            raise PdsXException(f"Metin özetleme hatası: {str(e)}")

    def named_entity_recognition(self, text: str, lang: str = "en") -> List[Tuple[str, str]]:
        """Metindeki adlandırılmış varlıkları tanır."""
        try:
            doc = self.spacy_nlp(text) if lang == "en" else spacy.load(f"{lang}_core_news_sm")(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            log.debug(f"NER tamamlandı: {text[:50]}...")
            return entities
        except Exception as e:
            log.error(f"NER hatası: {str(e)}")
            raise PdsXException(f"NER hatası: {str(e)}")

    def pos_tagging(self, text: str, lang: str = "en") -> List[Tuple[str, str]]:
        """Metindeki kelimelere POS etiketleri atar."""
        try:
            if lang == "en":
                tokens = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)
            else:
                doc = self.spacy_nlp(text) if lang == "en" else spacy.load(f"{lang}_core_news_sm")(text)
                pos_tags = [(token.text, token.pos_) for token in doc]
            log.debug(f"POS etiketleme tamamlandı: {text[:50]}...")
            return pos_tags
        except Exception as e:
            log.error(f"POS etiketleme hatası: {str(e)}")
            raise PdsXException(f"POS etiketleme hatası: {str(e)}")

    def dependency_parsing(self, text: str, lang: str = "en") -> List[Tuple[str, str, str]]:
        """Metindeki bağımlılık ilişkilerini analiz eder."""
        try:
            doc = self.spacy_nlp(text) if lang == "en" else spacy.load(f"{lang}_core_news_sm")(text)
            deps = [(token.text, token.dep_, token.head.text) for token in doc]
            log.debug(f"Bağımlılık analizi tamamlandı: {text[:50]}...")
            return deps
        except Exception as e:
            log.error(f"Bağımlılık analizi hatası: {str(e)}")
            raise PdsXException(f"Bağımlılık analizi hatası: {str(e)}")

    def text_classification(self, text: str, model: str = "distilbert-base-uncased-finetuned-sst-2-english") -> Dict:
        """Metni sınıflandırır."""
        try:
            classifier = self._load_transformer_pipeline("text-classification", model)
            result = classifier(text)[0]
            log.debug(f"Metin sınıflandırma tamamlandı: {text[:50]}...")
            return result
        except Exception as e:
            log.error(f"Metin sınıflandırma hatası: {str(e)}")
            raise PdsXException(f"Metin sınıflandırma hatası: {str(e)}")

    def generate_text(self, prompt: str, model: str = "gpt2", max_length: int = 50) -> str:
        """Metin üretir."""
        try:
            generator = self._load_transformer_pipeline("text-generation", model)
            result = generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
            log.debug(f"Metin üretimi tamamlandı: {prompt[:50]}...")
            return result
        except Exception as e:
            log.error(f"Metin üretimi hatası: {str(e)}")
            raise PdsXException(f"Metin üretimi hatası: {str(e)}")

    def parse_nlp_command(self, command: str) -> None:
        """NLP komutunu ayrıştırır ve yürütür."""
        command_upper = command.upper().strip()
        try:
            if command_upper.startswith("NLP ANALYZE "):
                match = re.match(r"NLP ANALYZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.analyze_text(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP ANALYZE komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP TOKENIZE "):
                match = re.match(r"NLP TOKENIZE\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.tokenize(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP TOKENIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP SENTIMENT "):
                match = re.match(r"NLP SENTIMENT\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.sentiment_analysis(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP SENTIMENT komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP SUMMARIZE "):
                match = re.match(r"NLP SUMMARIZE\s+\"([^\"]+)\"\s*(\d+)?\s*(\d+)?", command, re.IGNORECASE)
                if match:
                    text, max_len, min_len = match.groups()
                    max_len = int(max_len) if max_len else 150
                    min_len = int(min_len) if min_len else 40
                    result = self.summarize_text(text, max_len, min_len)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP SUMMARIZE komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP NER "):
                match = re.match(r"NLP NER\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.named_entity_recognition(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP NER komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP POS "):
                match = re.match(r"NLP POS\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.pos_tagging(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP POS komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP DEP "):
                match = re.match(r"NLP DEP\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, lang = match.groups()
                    lang = lang or "en"
                    result = self.dependency_parsing(text, lang)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP DEP komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP CLASSIFY "):
                match = re.match(r"NLP CLASSIFY\s+\"([^\"]+)\"\s*(\w+)?", command, re.IGNORECASE)
                if match:
                    text, model = match.groups()
                    model = model or "distilbert-base-uncased-finetuned-sst-2-english"
                    result = self.text_classification(text, model)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP CLASSIFY komutunda sözdizimi hatası")
            elif command_upper.startswith("NLP GENERATE "):
                match = re.match(r"NLP GENERATE\s+\"([^\"]+)\"\s*(\w+)?\s*(\d+)?", command, re.IGNORECASE)
                if match:
                    prompt, model, max_len = match.groups()
                    model = model or "gpt2"
                    max_len = int(max_len) if max_len else 50
                    result = self.generate_text(prompt, model, max_len)
                    self.interpreter.current_scope()["_NLP_RESULT"] = result
                else:
                    raise PdsXException("NLP GENERATE komutunda sözdizimi hatası")
            else:
                raise PdsXException(f"Bilinmeyen NLP komutu: {command}")
        except Exception as e:
            log.error(f"NLP komut hatası: {str(e)}")
            raise PdsXException(f"NLP komut hatası: {str(e)}")

    def boot_nlp(self) -> None:
        """NLP sistemini başlatır (eski API için)."""
        self._initialize_nlp()

if __name__ == "__main__":
    print("libx_nlp.py bağımsız çalıştırılamaz. pdsXu ile kullanın.")