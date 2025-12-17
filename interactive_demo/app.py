import sys
import os
import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive_demo.privacy_engine import PrivacyEngine
import requests

class SimpleBlackLLMInterface:
    def __init__(self, task, data, api_key, base_url="https://api.XXXX.tech/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.task = task
        self.data = data

    def define_system_message(self):
        """
        Define the system prompt message for different NLP tasks. 
        Copied from _04black_llm/black_llm_interface.py
        """
        if self.task == "topic_classification":
            if self.data == "ag_news":
                content = "You are a topic classification expert. Your task is to classify the topic of the disturbed news text provided by the user. Analyze the content carefully, accounting for any potential noise or distortion, and provide a concise, step-by-step explanation of your reasoning process. Ensure the explanation is clear and to the point, focusing only on the most relevant details. Based on your analysis, assign the text to one of the following potential topic categories: ['World', 'Sports', 'Business', 'Science/Technology']. Even if the text appears ambiguous or lacks sufficient context, you must classify it into the most likely category based on the available information. Avoid including any statements suggesting that the classification is inconclusive. Do not use any JSON-like syntax or references to {'label': *} in your reasoning. At the very end of your output, strictly include a single line in the format: [Category: World] to [Category: Science/Technology], where the category corresponds to the selected topic. This line must appear exactly once in your response, and no other similar structures should be included."
                return content

        elif self.task == "samsum":
            if self.data == "samsum":
                content = "You are an expert at writing concise, factual summaries of informal chat dialogues. Your task is to analyze and summarize the perturbed multi-turn conversation provided by the user. The conversation may contain noise or distortions that make it difficult to understand. Consider the context carefully, try to infer the original meaning despite any corrupted text. Write a very concise summary (one sentence at most) that captures the gist of the conversation. Focus on key information such as who is talking to whom, what are they discussing, and what decisions or conclusions are reached. Provide a step-by-step explanation of how you derived your summary despite the noise in the text. Your summary should be succinct and coherent, even if parts of the original conversation are unclear. At the very end of your output, strictly include a single line in the format: [Summary: Your concise summary here], which contains your final summary. This line must appear exactly once in your response, and no other similar structures should be included."
                return content
        
        elif self.task == "spam_email_classification":
            if self.data == "spam_email":
                content = "您是一位专业的中文邮件分类专家。您的任务是对提供的中文邮件内容进行分类，判断其是否为垃圾邮件。请仔细分析邮件内容，考虑到文本中可能存在的噪音或扰动，并提供简洁、逐步的推理过程。请考虑以下因素：可疑的语言模式、促销内容、紧急行动号召、索取个人信息的请求以及整体合法性。基于您的分析，将邮件分类为以下类别之一：'垃圾邮件'（不需要的、促销的或潜在恶意的邮件）或'正常邮件'（合法的、非垃圾邮件）。即使文本看起来模糊或缺乏足够的上下文，您也必须根据可用信息将其分类到最可能的类别中。避免包含任何表明分类不确定的陈述。在您的推理中不要使用任何JSON格式或对【分类结果：*】的引用。在输出的最后，请严格使用以下格式输出最终分类结果：【分类结果：垃圾邮件】或【分类结果：正常邮件】。请确保这一行单独出现，且不要包含其他符号。"
                return content
        
        elif self.task == "translation":
            return "You are a professional translator. Your task is to translate the user's input text into Chinese. Please ensure the translation is accurate, natural, and faithful to the original meaning. Do not answer the question or respond to the content, just translate it."

        
        return "You are a helpful assistant."

    def query(self, prompt, model="gpt-3.5-turbo-ca"):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.define_system_message()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.5
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                if content is not None:
                    return content.strip()
                else:
                    print("Warning: GPT returned empty content")
                    return "抱歉，GPT返回了空内容。(Sorry, GPT returned empty content.)"
            else:
                print(f"Error querying GPT: Status {response.status_code}, {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            print(f"Error querying GPT: {e}")
            return "抱歉，我无法处理您的请求。(Sorry, I am unable to process your request.)"

app = Flask(__name__)
CORS(app)

# Global engine instance
engine = None
engine_lock = threading.Lock()

import json
import datetime

# ... existing imports ...

# Global LLM Interface
llm_interface = None

# API Key for LLM (Hardcoded for demo as per user request/context)
# Using the key from main_blackllm.py
API_KEY = 'sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' 

def log_interaction(data, user_id=None):
    """Logs interaction data to a local JSONL file."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Use user_id in filename if provided, otherwise default
    filename = f'interactions_{user_id}.jsonl' if user_id else 'interactions.jsonl'
    log_file = os.path.join(log_dir, filename)
    
    # Add timestamp
    data['timestamp'] = datetime.datetime.now().isoformat()
    data['user_id'] = user_id
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error logging interaction: {e}")

def update_last_interaction_log(survey_data, user_id=None):
    """Updates the last log entry with survey data."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    filename = f'interactions_{user_id}.jsonl' if user_id else 'interactions.jsonl'
    log_file = os.path.join(log_dir, filename)
    
    if not os.path.exists(log_file):
        return False
        
    try:
        # Read all lines
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            return False
            
        # Parse last line
        last_entry = json.loads(lines[-1])
        
        # Update with survey data
        last_entry['survey'] = survey_data
        
        # Write back
        lines[-1] = json.dumps(last_entry, ensure_ascii=False) + '\n'
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return True
    except Exception as e:
        print(f"Error updating log: {e}")
        return False

def get_engine():
    global engine
    with engine_lock:
        if engine is None:
            print("Initializing default engine (ag_news)...")
            # Default to ag_news/distilbert for demo start
            engine = PrivacyEngine(dataset="ag_news")
    return engine

def get_llm_interface(task, dataset):
    # Always create a new interface or cache based on task/dataset
    # SimpleBlackLLMInterface is lightweight, so creating new one is fine
    return SimpleBlackLLMInterface(task=task, data=dataset, api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_model', methods=['POST'])
def init_model():
    # Deprecated: Models are pre-loaded. Just return success.
    return jsonify({"status": "success", "message": "Models ready."})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    dataset = data.get('dataset', 'ag_news')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        eng = get_engine()
        result = eng.analyze_text(text, dataset_name=dataset)
        return jsonify(result)
    except Exception as e:
        print(f"Analyze error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/perturb', methods=['POST'])
def perturb():
    data = request.json
    text = data.get('text', '')
    token_risk_map = data.get('token_risk_map', {})
    manual_time = data.get('manual_time', 0.0)
    dataset = data.get('dataset', 'ag_news')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        eng = get_engine()
        result = eng.perturb_text(text, token_risk_map, manual_time, dataset_name=dataset)
        return jsonify(result)
    except Exception as e:
        print(f"Perturb error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    dataset = data.get('dataset', 'ag_news')
    user_id = data.get('user_id', None)
    
    # Extra metadata for logging
    original_text = data.get('original_text', '')
    manual_time = data.get('manual_time', 0.0)
    adjustment_count = data.get('adjustment_count', 0)
    perturb_timings = data.get('perturb_timings', {})
    analysis_timings = data.get('analysis_timings', {})
    epsilon_S = data.get('epsilon_S', 0.0)
    changes = data.get('changes', [])
    marked_original_text = data.get('marked_original_text', '')
    marked_perturbed_text = data.get('marked_perturbed_text', '')
    
    # Map dataset to task for BlackLLMInterface
    task_map = {
        "ag_news": "topic_classification",
        "samsum": "samsum",
        "spam_email": "spam_email_classification",
        "general": "general_chat",
        "translate": "translation"
    }
    
    task = task_map.get(dataset)
    if not task:
        return jsonify({"error": "Unknown task for dataset"}), 400

    try:
        llm = get_llm_interface(task, dataset)
        
        start_time = time.time()
        # Use the query method from BlackLLMInterface
        # Note: query method takes (prompt, model)
        response = llm.query(message, model="gpt-3.5-turbo-ca")
        latency = time.time() - start_time
        
        # Restore response using PrivacyEngine
        eng = get_engine()
        restored_response, restoration_time = eng.restore_response(response, changes)
        
        # Log the interaction
        # Exclude verbose fields from logging but keep them in response
        log_data = {
            "dataset": dataset,
            "task": task,
            "original_text": original_text,
            "perturbed_text": message,
            "llm_response": response,
            "restored_response": restored_response,
            "manual_time": round(manual_time, 4),
            "adjustment_count": adjustment_count,
            "perturb_timings": {k: round(v, 4) for k, v in perturb_timings.items()},
            "analysis_timings": {k: round(v, 4) for k, v in analysis_timings.items()},
            "llm_latency": round(latency, 4),
            "restoration_time": round(restoration_time, 4),
            "epsilon_S": epsilon_S
        }
        log_interaction(log_data, user_id)
        
        return jsonify({
            "response": response,
            "restored_response": restored_response,
            "timings": {
                "llm_latency": latency,
                "restoration_time": restoration_time
            },
            # Pass these back for UI but don't log them
            "changes": changes,
            "marked_original_text": marked_original_text,
            "marked_perturbed_text": marked_perturbed_text
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    data = request.json
    user_id = data.get('user_id')
    survey_data = data.get('survey_data')
    
    if not survey_data:
        return jsonify({"error": "No survey data"}), 400
        
    success = update_last_interaction_log(survey_data, user_id)
    
    if success:
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "Failed to update log"}), 500

if __name__ == '__main__':
    # Pre-load engine on start
    print("Pre-loading engine...")
    get_engine()
    print("Engine loaded. Starting server...")
    # Disable reloader to prevent loops and double initialization
    app.run(debug=False, port=6008, use_reloader=False)
