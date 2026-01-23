from dotenv import load_dotenv
import pandas as pd
import json
import time
import os
from openai import OpenAI
from tqdm import tqdm  # 进度条库，pip install tqdm


# ================= 配置区 =================
load_dotenv()  # 加载 .env 文件
# BASE_URL = "https://api.deepseek.com"  # 或者智谱的 url
#
# client = OpenAI(api_key=os.getenv("API_KEY"), base_url=BASE_URL)

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),
)

def ask_llm_expert(row):
    """
    将一行数据转化为 Prompt，发送给 LLM，返回决策
    """
    # 1. 构造 Prompt
    # 技巧：把具体数字放入 prompt，并明确要求 JSON 格式
    system_prompt = (
        "你是一个边缘计算调度专家。请根据任务状态决定是否卸载任务。"
        "如果任务大、带宽低、服务器负载高，建议本地处理或者丢弃（根据具体策略）。"
        "如果带宽充足且服务器空闲，建议卸载到边缘服务器。"
        "请务必返回 JSON 格式，包含两个字段：'action' (0代表本地处理, 1代表卸载) 和 'reason' (简短理由)。"
    )

    user_prompt = f"""
    当前状态：
    - 任务大小: {row['task_size']} MB
    - 时延要求: {row['latency_limit']} ms
    - 当前带宽: {row['bandwidth']} Mbps
    - 服务器负载: {row['server_load']}%

    请给出决策。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-v3-2-251201",  # 或 glm-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},  # 关键！强制返回 JSON
            temperature=0.1  # 低随机性
        )

        # 解析返回的内容
        content = response.choices[0].message.content
        result_json = json.loads(content)  # 把字符串变成字典

        return result_json['action'], result_json['reason']

    except Exception as e:
        print(f"API 请求失败: {e}")
        return None, "Error"


def main():
    # 1. 读取原始数据
    try:
        df = pd.read_csv("raw_scenarios.csv")
    except FileNotFoundError:
        print("❌ 没找到 raw_scenarios.csv，请先运行 Step 1 的代码！")
        return

    print("🚀 开始请求 LLM 进行标注（Teacher 模式）...")

    actions = []
    reasons = []

    # 2. 循环处理每一行 (使用 tqdm 显示进度条)
    # df.iterrows() 遍历每一行数据
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        action, reason = ask_llm_expert(row)

        actions.append(action)
        reasons.append(reason)

        # ⚠️ 重要：加上延时，防止触发 API 的速率限制 (Rate Limit)
        # DeepSeek 比较宽松，但建议还是停 0.5 秒
        time.sleep(0.5)

        # 3. 保存结果
    df['label_action'] = actions
    df['label_reason'] = reasons

    # 去除失败的行 (None)
    df = df.dropna()

    output_file = "expert_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 标注完成！高质量训练数据已保存至: {output_file}")
    print(f"共获得 {len(df)} 条有效数据。")
    print("-" * 20)
    print(df[['task_size', 'bandwidth', 'label_action', 'label_reason']].head())


if __name__ == "__main__":
    main()