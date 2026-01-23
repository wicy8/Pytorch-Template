import time
import json
import random
import utils

class MockLLM:
    """
    这是一个用于测试的假 LLM。
    它不消耗显存，不需要 GPU，但能模拟 LLM 的输入输出行为。
    """
    def __init__(self, simulation_time=1.5):
        self.simulation_time = simulation_time # 模拟推理延迟，单位秒

    def chat(self, user_prompt, system_prompt=""):
        """
        模拟对话接口。
        
        Args:
            user_prompt (str): 用户发给模型的话
            system_prompt (str): 系统指令
            
        Returns:
            str: 模型的回复（模拟 JSON 格式）
        """
        print(f"\n[MockLLM] 正在思考... (模拟延迟 {self.simulation_time}s)")
        print(f"[MockLLM] 收到 Prompt: {user_prompt[:50]}...") # 只打印前50个字
        
        time.sleep(self.simulation_time)
        
        # --- 核心 Mock 逻辑 ---
        # 我们假设我们在做 CartPole 任务，LLM 应该返回一段 Python 代码
        # 这里我们硬编码一个简单的策略代码作为返回
        
        mock_code = """
            def custom_reward(params):
                # 这是一个 Mock 生成的奖励函数
                # 策略：如果杆子越直 (angle 越小)，奖励越高
                angle = params['angle']
                return 1.0 - abs(angle)
            """
        
        # 模拟 LLM 经常会把代码包在 Markdown 块里，甚至带点废话
        # response_content = {
        #     "thought": "根据物理学原理，保持杆子垂直是关键，所以我设计了一个基于角度惩罚的奖励函数。",
        #     "code": mock_code
        # }

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
        response_content = utils.parse_llm_output(dirty_output)
        
        # 返回 JSON 字符串
        return json.dumps(response_content, ensure_ascii=False)

# --- 单元测试 ---
if __name__ == "__main__":
    # 自测代码：只有直接运行这个文件时才会执行
    llm = MockLLM()
    response = llm.chat("请帮我写一个 CartPole 的奖励函数")
    print("\n[MockLLM] 回复内容:")
    print(response)