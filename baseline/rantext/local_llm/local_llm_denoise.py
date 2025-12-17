# local_llm_denoise.py
import requests
import json

class LocalLLMDenoise:
    def __init__(self, task, base_url, api_key=None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.task = task

    def define_system_message(self):
        if self.task == "topic_classification":
            content = "You are a topic classification expert. Your task is to classify the topic of the 'Original sentence' provided by the user. The 'Original sentence' has been transformed into a 'Disturbed sentence', which GPT analyzed to produce reasoning and classification in the 'GPT's analysis and classification'. Reference GPT's reasoning approach while independently evaluating the 'Original sentence'. Classify the text into one of the following categories: ['World', 'Sports', 'Business', 'Science/Technology']. Even if the text is unclear, classify it into the most likely category. IMPORTANT: At the very end of your response, provide exactly one JSON dictionary in the format: {'label': *}, where * corresponds to the selected category (0 for World, 1 for Sports, 2 for Business, 3 for Science/Technology). This dictionary must appear only once, and no other JSON-like structures should be included in your response. If you fail to provide the JSON dictionary {'label': *} at the end, your answer will be considered incomplete or invalid."

        elif self.task == "piidocs_classification":
            content = "You are a document classification expert. Your task is to classify the 'Original sentence' into one of the following document categories: ['healthcare', 'legal-documents', 'travel-hospitality']. The 'Original sentence' has been transformed into a 'Disturbed sentence', which GPT analyzed to produce reasoning and classification in the 'GPT's analysis and classification'. Reference GPT's reasoning approach while independently evaluating the 'Original sentence'. Classify the document into one of the following categories: 0 for healthcare, 1 for legal-documents, or 2 for travel-hospitality. Even if the text is unclear, classify it into the most likely category. IMPORTANT: At the very end of your response, provide exactly one JSON dictionary in the format: {'label': *}, where * corresponds to the selected document category (0 for healthcare, 1 for legal-documents, 2 for travel-hospitality). This dictionary must appear only once, and no other JSON-like structures should be included in your response. If you fail to provide the JSON dictionary {'label': *} at the end, your answer will be considered incomplete or invalid."

        return content

    def denoise_output(self, model, prompt):
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.define_system_message()},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(self.base_url, json=data, headers=self.headers)
            if response.status_code == 200:
                # Logic to concatenate the complete sentence
                full_content = ""
                try:
                    for line in response.text.splitlines():
                        # Try to parse each line of JSON
                        try:
                            json_line = json.loads(line)
                            if "message" in json_line and "content" in json_line["message"]:
                                full_content += json_line["message"]["content"]
                        except json.JSONDecodeError as e:
                            print("Error parsing line:", str(e))
                except Exception as e:
                    print("Error parsing line by line:", str(e))

                return full_content.strip()
            else:
                print("Error:", response.status_code, response.text)
                return None
        except Exception as e:
            print("Request error:", str(e))
            return None