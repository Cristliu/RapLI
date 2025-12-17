# _02risk_assessment/risk_assessor.py
import random

# Define random seed
random.seed(42)

class RiskAssessor:
    def __init__(self):
        pass

    def assess_risk(self, combined_results):
        assessed_results = []
        
        for entity in combined_results:
            # Assign random risk level (1-5)
            risk_level = 5  # For identified PII entities, a higher risk level should be assigned, here set to 5

            assessed_results.append({
                "entity": entity['text'],
                "type": entity['entity_type'],
                'start': entity['start'],
                'end': entity['end'],
                "risk_level": risk_level,
            })
        
        return assessed_results
