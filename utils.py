import json
import re


def parse_llm_output(llm_output_str):
    """
    从 LLM 的回复中提取 JSON 数据。
    """
    try:
        # 1. 尝试直接解析
        return json.loads(llm_output_str)
    except json.JSONDecodeError:
        print("[Warning] 直接解析 JSON 失败，尝试清洗数据...")

    # 2. 如果失败，尝试提取 ```json ... ``` 之间的内容
    # 正则表达式：寻找被 ```json 和 ``` 包裹的内容
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, llm_output_str, re.DOTALL)

    if match:
        clean_json_str = match.group(1)
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            print("[Error] 清洗后依然无法解析 JSON")
            return None
    else:
        print("[Error] 未找到 Markdown 代码块")
        return None


# --- 测试 ---
if __name__ == "__main__":
    # 模拟一个脏数据：LLM 在 JSON 外面说了废话
    dirty_output = """
    好的，这是为您生成的代码：
    ```json
    {
        "thought": "Testing extraction",
        "code": "print('hello')"
    }
    ```
    希望这对您有帮助！
    """

    result = parse_llm_output(dirty_output)
    print("解析结果:", result)
    print("代码部分:", result['code'])