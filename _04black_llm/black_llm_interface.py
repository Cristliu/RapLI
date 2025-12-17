# _04black_llm/black_llm_interface.py
from openai import OpenAI

class BlackLLMInterface:
    def __init__(self, task, data, api_key, base_url="https://api.openai.com/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.task = task
        self.data = data

    def define_system_message(self): 
        """
        Define the system prompt message for different NLP tasks. 
        Each prompt emphasizes:
        1. Clear label and classification mapping.
        2. Concise, step-by-step reasoning instructions.
        3. Single JSON dictionary with {'label': X} output exactly once at the end.
        """
        
        if self.task == "topic_classification":
            if self.data == "ag_news":
                content = "You are a topic classification expert. Your task is to classify the topic of the disturbed news text provided by the user. Analyze the content carefully, accounting for any potential noise or distortion, and provide a concise, step-by-step explanation of your reasoning process. Ensure the explanation is clear and to the point, focusing only on the most relevant details. Based on your analysis, assign the text to one of the following potential topic categories: ['World', 'Sports', 'Business', 'Science/Technology']. Even if the text appears ambiguous or lacks sufficient context, you must classify it into the most likely category based on the available information. Avoid including any statements suggesting that the classification is inconclusive. Do not use any JSON-like syntax or references to {'label': *} in your reasoning. At the very end of your output, strictly include a single JSON dictionary in the format: {'label': 0} to {'label': 3}, where the label corresponds to the selected topic category. This JSON dictionary must appear exactly once in your response, and no other similar structures should be included."
                return content


        elif self.task == "piidocs_classification":
            if self.data == "piidocs":
                content = "You are a document classification expert. Your task is to classify the category of the disturbed document text provided by the user. Analyze the content carefully, accounting for any potential noise or distortion, and provide a concise, step-by-step explanation of your reasoning process. Ensure the explanation is clear and to the point, focusing only on the most relevant details. Based on your analysis, assign the text to one of the following potential document categories: ['healthcare', 'legal-documents', 'travel-hospitality']. Here is a brief summary of possible document types under each category: healthcare: typically medical reports, test results, vaccination records, discharge summaries, insurance claims, billing statements, administrative forms, appointment requests, etc. legal-documents: typically contracts, agreements, subpoenas, settlements, judgments, etc. travel-hospitality: typically itineraries, e-tickets, hotel reservations, baggage policies, feedback forms, etc. Even if the text appears unclear or lacks sufficient context, you must classify it into the most likely category based on the available information. Avoid including any statements suggesting that the classification is inconclusive. Do not use any JSON-like syntax or references to {'label': *} in your reasoning. At the very end of your output, strictly include a single JSON dictionary in the format: {'label': 0} to {'label': 2}, where the label corresponds to the selected document category. This JSON dictionary must appear exactly once in your response, and no other similar structures should be included."
                return content 
        
        
        elif self.task == "samsum":
            if self.data == "samsum":
                
                content = "You are an expert at writing concise, factual summaries of informal chat dialogues. Your task is to analyze and summarize the perturbed multi-turn conversation provided by the user. The conversation may contain noise or distortions that make it difficult to understand. Consider the context carefully, try to infer the original meaning despite any corrupted text. Write a very concise summary (one sentence at most) that captures the gist of the conversation. Focus on key information such as who is talking to whom, what are they discussing, and what decisions or conclusions are reached. Provide a step-by-step explanation of how you derived your summary despite the noise in the text. Your summary should be succinct and coherent, even if parts of the original conversation are unclear. At the very end of your output, strictly include a single JSON dictionary in the format: {'summary': 'Your concise summary here'}, which contains your final summary. This JSON dictionary must appear exactly once in your response, and no other similar structures should be included."
                return content
        
        elif self.task == "spam_email_classification":
            if self.data == "spam_email":
                content = "你是一位专业的中文邮件分类专家。你的任务是对提供的中文邮件内容进行分类，判断其是否为垃圾邮件。请仔细分析邮件内容，考虑到文本中可能存在的噪音或扰动，并提供简洁、逐步的推理过程。请考虑以下因素：可疑的语言模式、促销内容、紧急行动号召、索取个人信息的请求以及整体合法性。基于你的分析，将邮件分类为以下类别之一：'垃圾邮件'（不需要的、促销的或潜在恶意的邮件）或'正常邮件'（合法的、非垃圾邮件）。即使文本看起来模糊或缺乏足够的上下文，你也必须根据可用信息将其分类到最可能的类别中。避免包含任何表明分类不确定的陈述。在你的推理中不要使用任何JSON格式或对{'label': *}的引用。在输出的最后，严格包含一个JSON字典，格式为：{'label': 1}（垃圾邮件）或{'label': 0}（正常邮件），其中标签对应你的分类结果。这个JSON字典必须在你的响应中只出现一次，不应包含其他类似的结构。"
                return content



    # def define_system_message(self): ### Prompt set for the original unperturbed dataset

    #     if self.task == "topic_classification":
    #         if self.data == "ag_news":
    #             content = "You are a topic classification expert. Your task is to classify the topic of the news text provided by the user. Analyze the content carefully and provide a concise, step-by-step explanation of your reasoning process. Ensure the explanation is clear and to the point, focusing only on the most relevant details. Based on your analysis, assign the text to one of the following potential topic categories: ['World', 'Sports', 'Business', 'Science/Technology']. You must classify it into the most likely category based on the available information. Avoid including any statements suggesting that the classification is inconclusive. Do not use any JSON-like syntax or references to {'label': *} in your reasoning. At the very end of your output, strictly include a single JSON dictionary in the format: {'label': 0} to {'label': 3}, where the label corresponds to the selected topic category. This JSON dictionary must appear exactly once in your response, and no other similar structures should be included."
    #             return content


    #     elif self.task == "piidocs_classification":
    #         if self.data == "piidocs":
    #             content = "You are a document classification expert. Your task is to classify the category of the document text provided by the user. Analyze the content carefully and provide a concise, step-by-step explanation of your reasoning process. Ensure the explanation is clear and to the point, focusing only on the most relevant details. Based on your analysis, assign the text to one of the following potential document categories: ['healthcare', 'legal-documents', 'travel-hospitality']. Here is a brief summary of possible document types under each category: healthcare: typically medical reports, test results, vaccination records, discharge summaries, insurance claims, billing statements, administrative forms, appointment requests, etc. legal-documents: typically contracts, agreements, subpoenas, settlements, judgments, etc. travel-hospitality: typically itineraries, e-tickets, hotel reservations, baggage policies, feedback forms, etc. You must classify it into the most likely category based on the available information. Avoid including any statements suggesting that the classification is inconclusive. Do not use any JSON-like syntax or references to {'label': *} in your reasoning. At the very end of your output, strictly include a single JSON dictionary in the format: {'label': 0} to {'label': 2}, where the label corresponds to the selected document category. This JSON dictionary must appear exactly once in your response, and no other similar structures should be included."
    #             return content 




    def query(self, prompt, model="gpt-3.5-turbo"):
        try:
            response = self.client.chat.completions.create(
                model=model,  
                messages=[
                    {
                    "role": "system",
                    "content": self.define_system_message()  
                    },
                    {
                    "role": "user", "content":  prompt
                    }
                    ],
                temperature=0.5, ###It is not supported in deepseek-reasoner
            )
            # Get reply content
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"Error querying GPT: {e}")
            return "I'm sorry, but I'm unable to handle your request."
        


