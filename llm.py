# llm.py

from openai import OpenAI
from anthropic import Anthropic
import os
import json
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """LLM基类，定义通用接口"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    @abstractmethod
    def generate_text(self, prompt, max_tokens=1500, temperature=0.2):
        pass
    
    @abstractmethod
    def generate_json(self, prompt, schema, max_tokens=1500, temperature=0.2):
        pass
    
    @abstractmethod
    def generate_conversation(self, messages, max_tokens=3500, temperature=0.2):
        pass
    
    def _validate_json(self, json_output, schema):
        """验证生成的JSON是否符合给定的schema"""
        if schema:
            for key, expected_type in schema.items():
                if key not in json_output or not isinstance(json_output[key], expected_type):
                    print(f"键 {key} 的类型错误，期望 {expected_type}，实际 {type(json_output.get(key))}")
                    return False
        return True

class OpenAILLM(BaseLLM):
    """OpenAI API实现"""
    
    def __init__(self, api_key=os.environ.get('GPT_API_KEY'), verbose=False):
        super().__init__(verbose)
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def generate_text(self, prompt, max_tokens=1500, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a reliable writing assistant. Please strictly follow the instructions to generate content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = response.choices[0].message.content
        if self.verbose: print(text)
        return text

    def generate_json(self, prompt, schema, max_tokens=1500, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature
        )
        json_output = response.choices[0].message.content
        if self.verbose: print(json_output)

        try:
            json_output = json.loads(json_output)
        except json.JSONDecodeError:
            print("无法解析 JSON 输出。")
            return None
        
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("生成的 JSON 不符合预期的 schema。")
            return None

    def generate_conversation(self, messages, max_tokens=3500, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        if self.verbose: print(reply)
        messages.append({"role": "assistant", "content": reply})
        return messages, reply

class ClaudeLLM(BaseLLM):
    """Claude API实现"""
    
    def __init__(self, api_key=os.environ.get('ANTHROPIC_API_KEY'), verbose=False):
        super().__init__(verbose)
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-haiku-20240307"
    
    def generate_text(self, prompt, max_tokens=1500, temperature=0.2):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a reliable writing assistant. Please strictly follow the instructions to generate content.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        text = response.content[0].text
        if self.verbose: print(text)
        return text

    def generate_json(self, prompt, schema, max_tokens=1500, temperature=0.2):
        # Claude没有直接的JSON response_format，需要在提示中说明
        json_prompt = f"Please respond with a valid JSON object. {prompt}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful assistant designed to output JSON.",
            messages=[
                {"role": "user", "content": json_prompt}
            ]
        )
        
        try:
            json_output = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            print("无法解析 JSON 输出。")
            return None
            
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("生成的 JSON 不符合预期的 schema。")
            return None

    def generate_conversation(self, messages, max_tokens=3500, temperature=0.2):
        # 转换消息格式以适应Claude API
        claude_messages = []
        for msg in messages:
            if msg["role"] != "system":  # Claude的system prompt需要单独处理
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=claude_messages
        )
        
        reply = response.content[0].text
        if self.verbose: print(reply)
        messages.append({"role": "assistant", "content": reply})
        return messages, reply

def LLM(model_type='gpt', api_key=None, verbose=False):
    """工厂函数，根据model_type返回对应的LLM实例"""
    if model_type.lower() == 'gpt':
        return OpenAILLM(api_key=api_key, verbose=verbose)
    elif model_type.lower() == 'claude':
        return ClaudeLLM(api_key=api_key, verbose=verbose)
    else:
        raise ValueError("Unsupported model type. Use 'gpt' or 'claude'.")