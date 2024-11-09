# llm.py

from openai import OpenAI
import os
import json

class LLM:
    def __init__(self, api_key=os.environ.get('GPT_API_KEY'), verbose = False):
        self.client = OpenAI(api_key=api_key)
        self.verbose = verbose

    def generate_text(self, prompt, max_tokens=1500, temperature=0.2):
        """
        生成文本响应
        :param prompt: 用户输入的提示
        :param max_tokens: 最大 tokens 数
        :param temperature: 生成的随机性
        :return: 生成的文本
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
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
        """
        生成符合 schema 的 JSON 格式响应
        :param prompt: 用户输入的提示
        :param schema: 期望输出的 JSON 模式
        :param max_tokens: 最大 tokens 数
        :param temperature: 生成的随机性
        :return: 生成的 JSON 格式响应
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
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

        # 验证 JSON 格式
        try:
            json_output = json.loads(json_output)
        except json.JSONDecodeError:
            print("无法解析 JSON 输出。")
            return None
        
        return json_output

        # 验证生成的 JSON 是否符合给定 schema
        # 假设 schema 是 Python 字典中的类型描述
        if self._validate_json(json_output, schema):
            return json_output
        else:
            print("生成的 JSON 不符合预期的 schema。")
            return None

    def _validate_json(self, json_output, schema):
        """
        验证生成的 JSON 是否符合给定的 schema
        :param json_output: 生成的 JSON 数据
        :param schema: 预期的 schema
        :return: True 如果符合，否则 False
        """
        for key, expected_type in schema.items():
            if key not in json_output or not isinstance(json_output[key], expected_type):
                print(f"键 {key} 的类型错误，期望 {expected_type}，实际 {type(json_output.get(key))}")
                return False
        return True

    def generate_conversation(self, messages, max_tokens=3500, temperature=0.2):
        """
        生成多轮对话响应，保持上下文
        :param messages: 当前的消息列表，包括之前的对话
        :param max_tokens: 最大 tokens 数
        :param temperature: 生成的随机性
        :return: 更新后的消息列表和最新生成的内容
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        if self.verbose: print(reply)
        # 更新消息列表
        messages.append({"role": "assistant", "content": reply})
        return messages, reply
