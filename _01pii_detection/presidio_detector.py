# pii_detection/presidio_detector.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
# from presidio_anonymizer import AnonymizerEngine
import time
import logging

# Silence all warning messages
logging.basicConfig(level=logging.ERROR)


class CustomPhoneRecognizer(PatternRecognizer):
    def __init__(self):
        # Define regex patterns using Pattern objects
        patterns = [
            Pattern(name="phone_pattern", regex=r"\b\d{3}-\d{3}-\d{3}\b", score=0.7)
        ]
        super().__init__(supported_entity="PHONE_NUMBER", patterns=patterns)

class PresidioDetector:
    def __init__(self):
        # Measure loading time
        start_time = time.time()
        self.analyzer = AnalyzerEngine()
        # Add custom PHONE_NUMBER recognizer to support more formats
        custom_phone_recognizer = CustomPhoneRecognizer()
        self.analyzer.registry.add_recognizer(custom_phone_recognizer)
        end_time = time.time()
        print(f"presidio model loaded in {end_time - start_time:.2f} seconds.")

    def detect_pii(self, text):
        # Do not specify the entities parameter to enable all supported entity types
        
        results = self.analyzer.analyze(text=text,language='en')#, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        
        # Create a list of results with deduplication logic
        unique_results = []

        seen_spans = set()  # Used to store (start, end) ranges of all entities

        for result in results:
            span = (result.start, result.end)
            # Check if the entity range overlaps with existing ranges
            if not any(self._is_overlap(span, existing_span) for existing_span in seen_spans):
                seen_spans.add(span)  # Add new range
                unique_results.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "text": text[result.start:result.end]
                })

        return unique_results
    
    def _is_overlap(self, span1, span2):
        """Check if two ranges overlap"""
        start1, end1 = span1
        start2, end2 = span2
        return not (end1 <= start2 or end2 <= start1)

