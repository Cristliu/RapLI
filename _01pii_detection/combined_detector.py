try:
    from .flair_detector import FlairDetector
    from .presidio_detector import PresidioDetector
except ImportError:
    from flair_detector import FlairDetector
    from presidio_detector import PresidioDetector


class CombinedPIIDetector:
    def __init__(self):
        self.flair_detector = FlairDetector()
        self.presidio_detector = PresidioDetector()
        self.categories = [
            "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", 
            "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", 
            "QUANTITY", "TIME", "WORK_OF_ART", "IP_ADDRESS", "PHONE_NUMBER", 
            "URL", "EMAIL_ADDRESS", "DATE_TIME", "CREDIT_CARD", "BANK_ACCOUNT"
        ]

    def detect_pii(self, text):
    # """Only return the union of presido_results and flair_results without type or conflict handling."""
        presidio_results = self.presidio_detector.detect_pii(text)
        flair_results = self.flair_detector.detect_pii(text)

        # Use a dictionary to store (start, end) -> entity_info for deduplication and merging
        combined_map = {}
        for res in presidio_results:
            combined_map[(res["start"], res["end"])] = res
        for res in flair_results:
            if (res["start"], res["end"]) not in combined_map:
                combined_map[(res["start"], res["end"])] = res

        combined_results = list(combined_map.values())
        return combined_results

