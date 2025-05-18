# pii_detection/flair_detector.py
from flair.data import Sentence
from flair.models import SequenceTagger
import time
import logging
# Configure logging level
logging.basicConfig(level=logging.ERROR)

class FlairDetector:
    def __init__(self):
        # Measure loading time
        start_time = time.time()
        self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
        end_time = time.time()
        print(f"Flair model loaded in {end_time - start_time:.2f} seconds.")

    def detect_pii(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        return [
            {
                "text": entity.text,
                "entity_type": entity.tag,
                "start": entity.start_position,
                "end": entity.end_position
            }
            for entity in sentence.get_spans('ner')
        ]