import pandas as pd
import numpy as np
import random


def generate_edge_scenarios(num_samples=100):
    """
    生成模拟的边缘计算环境状态。
    假设状态包含 4 个特征：
    1. task_size: 任务大小 (MB)
    2. required_latency: 任务要求的最大延迟 (ms)
    3. bandwidth: 当前网络带宽 (Mbps)
    4. server_load: 边缘服务器当前负载 (0-100%)
    """
    data = []

    print(f"正在生成 {num_samples} 条模拟数据...")

    for _ in range(num_samples):
        # 模拟随机场景
        task_size = round(random.uniform(1.0, 50.0), 2)  # 1MB 到 50MB
        req_latency = round(random.uniform(10, 200), 2)  # 10ms 到 200ms
        bandwidth = round(random.uniform(5.0, 100.0), 2)  # 5Mbps 到 100Mbps
        server_load = round(random.uniform(0.0, 100.0), 2)  # 0% 到 100%

        # 存储一条记录
        record = {
            "task_size": task_size,
            "latency_limit": req_latency,
            "bandwidth": bandwidth,
            "server_load": server_load
        }
        data.append(record)

    # 转为 DataFrame 并保存
    df = pd.DataFrame(data)
    df.to_csv("raw_scenarios.csv", index=False)
    print("✅ 原始数据生成完毕！保存为 'raw_scenarios.csv'")
    print(df.head())  # 打印前5行看看


if __name__ == "__main__":
    generate_edge_scenarios(50)  # 先生成 50 条测试一下