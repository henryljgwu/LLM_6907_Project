# llm.py

from openai import OpenAI
from anthropic import Anthropic
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

# 全局默认参数
DEFAULT_MAX_TOKENS = 2500
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SYS_PROMPT = "You are a reliable writing assistant. Please strictly follow the instructions to generate content. You could write done your thinking or reasoning before you generate the final output if neccessay."

# 模型配置
MODEL_CONFIGS = {
    'gpt': {
        'default': 'gpt-4o-mini',
        'models': ('gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt-o1'),
        'mapping': {
            'gpt-4': 'gpt-4',
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-o1-mini': 'o1-mini-2024-09-12',
            'gpt-o1': 'o1-preview'
        }
    },
    'claude': {
        'default': 'claude-haiku',
        'models': ('claude-haiku', 'claude-sonnet'),
        'mapping': {
            'claude-haiku': 'claude-3-5-haiku-latest',
            'claude-sonnet': 'claude-3-5-sonnet-latest'
        }
    },
    'llama': {
        'default': 'llama3-8b',
        'models': ('llama3-70b', 'llama3-8b'),
        'mapping': {
            'llama3-70b': 'meta-llama/Llama-3.1-70B-Instruct',
            'llama3-8b': 'meta-llama/Llama-3.1-8B-Instruct'
        }
    },
    'qwen': {
        'default': 'qwen-72b',
        'models': ('qwen-72b',),
        'mapping': {
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct'
        }
    }
}

class BaseLLM(ABC):
    """LLM基类，定义通用接口"""
    
    def __init__(self, verbose: bool = False):
        """
        初始化基类
        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, 
                     temperature: float = DEFAULT_TEMPERATURE) -> str:
        """生成文本接口"""
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, schema: Dict, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict]:
        """生成JSON接口"""
        pass
    
    @abstractmethod
    def generate_conversation(self, messages: List[Dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                            temperature: float = DEFAULT_TEMPERATURE) -> tuple:
        """生成对话接口"""
        pass
    
    def _validate_json(self, json_output: Dict, schema: Dict) -> bool:
        """验证JSON输出是否符合schema"""
        return True

class OpenAILLM(BaseLLM):
    """OpenAI API实现"""
    
    def __init__(self, model_name: str = None, api_key: str = os.environ.get('GPT_API_KEY'),
                 verbose: bool = False):
        """
        初始化OpenAI LLM
        Args:
            model_name: 模型名称，如果不指定则使用默认模型
            api_key: API密钥
            verbose: 是否打印详细信息
        """
        super().__init__(verbose)
        if not api_key: api_key = os.environ.get('GPT_API_KEY')
        self.client = OpenAI(api_key=api_key)
        
        # 确定使用的模型
        if model_name and model_name in MODEL_CONFIGS['gpt']['mapping']:
            self.model = MODEL_CONFIGS['gpt']['mapping'][model_name]
        else:
            default_name = MODEL_CONFIGS['gpt']['default']
            self.model = MODEL_CONFIGS['gpt']['mapping'][default_name]
    
    def generate_text(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        生成文本
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            生成的文本
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = response.choices[0].message.content
        if self.verbose:
            print(f"Generated text: {text}")
        return text

    def generate_json(self, prompt: str, schema: Dict, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict]:
        """
        生成JSON
        Args:
            prompt: 输入提示
            schema: JSON schema
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            生成的JSON字典或None（生成失败时）
        """
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
        if self.verbose:
            print(f"Generated JSON: {json_output}")

        try:
            json_output = json.loads(json_output)
        except json.JSONDecodeError:
            print("Failed to parse JSON output.")
            return None
        
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("Generated JSON does not match the expected schema.")
            return None

    def generate_conversation(self, messages: List[Dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                            temperature: float = DEFAULT_TEMPERATURE) -> tuple:
        """
        生成对话
        Args:
            messages: 对话历史
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            更新后的对话历史和最新回复的元组
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        if self.verbose:
            print(f"Generated reply: {reply}")
        messages.append({"role": "assistant", "content": reply})
        return messages, reply

class ClaudeLLM(BaseLLM):
    """Claude API实现"""
    
    def __init__(self, model_name: str = None, 
                 api_key: str = os.environ.get('LLM_Project_Claude'),
                 verbose: bool = False):
        """
        初始化Claude LLM
        Args:
            model_name: 模型名称，如果不指定则使用默认模型
            api_key: API密钥
            verbose: 是否打印详细信息
        """
        super().__init__(verbose)
        if not api_key: api_key = os.environ.get('LLM_Project_Claude')
        self.client = Anthropic(api_key=api_key)
        
        # 确定使用的模型
        if model_name and model_name in MODEL_CONFIGS['claude']['mapping']:
            self.model = MODEL_CONFIGS['claude']['mapping'][model_name]
        else:
            default_name = MODEL_CONFIGS['claude']['default']
            self.model = MODEL_CONFIGS['claude']['mapping'][default_name]
    
    def generate_text(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        生成文本
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            生成的文本
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=DEFAULT_SYS_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        text = response.content[0].text
        if self.verbose:
            print(f"Generated text: {text}")
        return text

    def generate_json(self, prompt: str, schema: Dict, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict]:
        """
        生成JSON
        Args:
            prompt: 输入提示
            schema: JSON schema
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            生成的JSON字典或None（生成失败时）
        """
        json_prompt = (
            "Please complete the following JSON object. "
            "Your response must start with the opening brace and end with the closing brace.\n\n"
            f"Complete this JSON based on the following request: {prompt}"
        )
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful assistant designed to output JSON. Always complete the JSON starting with the opening brace.",
            messages=[
                {"role": "user", "content": json_prompt},
                {"role": "assistant", "content": "{"}
            ]
        )
        
        response_text = response.content[0].text.strip()
        if response_text.startswith('{'):
            response_text = response_text[1:]
        
        full_json_str = "{" + response_text
        if not full_json_str.strip().endswith('}'):
            full_json_str += "}"
        
        try:
            json_output = json.loads(full_json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON output: {e}")
            print(f"Raw response: {full_json_str}")
            return None
        
        if self.verbose:
            print("Generated JSON:")
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
            
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("Generated JSON does not match the expected schema.")
            return None

    def generate_conversation(self, messages: List[Dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                            temperature: float = DEFAULT_TEMPERATURE) -> tuple:
        """
        生成对话
        Args:
            messages: 对话历史
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            更新后的对话历史和最新回复的元组
        """
        claude_messages = []
        for msg in messages:
            if msg["role"] != "system":
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=claude_messages
        )
        
        reply = response.content[0].text
        if self.verbose:
            print(f"Generated reply: {reply}")
        messages.append({"role": "assistant", "content": reply})
        return messages, reply
    
class LlamaLLM(BaseLLM):
    """Llama-3 API implementation using Hugging Face Inference API"""
    
    def __init__(self, model_name: str = None, 
                 api_key: str = os.environ.get('HF_API_KEY'),
                 verbose: bool = False):
        """
        Initialize Llama LLM
        Args:
            model_name: Model name, uses default if not specified
            api_key: API key
            verbose: Whether to print detailed information
        """
        super().__init__(verbose)
        if not api_key: api_key = os.environ.get('HF_API_KEY')
        self.client = OpenAI(
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=api_key
        )
        
        # Determine model to use
        if model_name and model_name in MODEL_CONFIGS['llama']['mapping']:
            self.model = MODEL_CONFIGS['llama']['mapping'][model_name]
        else:
            default_name = MODEL_CONFIGS['llama']['default']
            self.model = MODEL_CONFIGS['llama']['mapping'][default_name]
    
    def generate_text(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generate text
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        text = response.choices[0].message.content
        if self.verbose:
            print(f"Generated text: {text}")
        return text

    def generate_json(self, prompt: str, schema: Dict, max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float = DEFAULT_TEMPERATURE) -> Optional[Dict]:
        """
        Generate JSON
        Args:
            prompt: Input prompt
            schema: JSON schema
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
        Returns:
            Generated JSON dictionary or None (if generation fails)
        """
        json_prompt = (
            "Please complete the following JSON object. "
            "Your response must be valid JSON.\n\n"
            f"Complete this JSON based on the following request: {prompt}"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": json_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            json_str = response.choices[0].message.content.strip()
            json_output = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON output: {e}")
            return None
        
        if self.verbose:
            print("Generated JSON:")
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
            
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("Generated JSON does not match the expected schema.")
            return None

    def generate_conversation(self, messages: List[Dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                            temperature: float = DEFAULT_TEMPERATURE) -> tuple:
        """
        Generate conversation
        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
        Returns:
            Updated conversation history and latest reply tuple
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        reply = response.choices[0].message.content
        if self.verbose:
            print(f"Generated reply: {reply}")
        messages.append({"role": "assistant", "content": reply})
        return messages, reply

def LLM(model_type: str = 'gpt', api_key: Optional[str] = None, verbose: bool = False):
    """
    工厂函数，根据model_type返回对应的LLM实例
    Args:
        model_type: 模型类型，支持'gpt'、'gpt-4'等或'claude'、'claude-haiku'等
        api_key: API密钥
        verbose: 是否打印详细信息
    Returns:
        相应的LLM实例
    """
    # 解析模型类型和名称
    model_name = None
    if model_type.startswith('gpt'):
        provider = 'gpt'
        model_name = model_type if model_type in MODEL_CONFIGS['gpt']['models'] else None
        return OpenAILLM(model_name=model_name, api_key=api_key, verbose=verbose)
    elif model_type.startswith('claude'):
        provider = 'claude'
        model_name = model_type if model_type in MODEL_CONFIGS['claude']['models'] else None
        return ClaudeLLM(model_name=model_name, api_key=api_key, verbose=verbose)
    elif model_type.startswith(('llama', 'qwen')):  # 合并 llama 和 qwen 的处理
        provider = 'llama' if model_type.startswith('llama') else 'qwen'
        model_name = model_type if model_type in MODEL_CONFIGS[provider]['models'] else None
        return LlamaLLM(model_name=model_name, api_key=api_key, verbose=verbose)
    else:
        raise ValueError("Unsupported model type. Use 'gpt[-*]', 'claude[-*]', or 'llama[-*]'.")