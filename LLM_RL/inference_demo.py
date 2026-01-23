import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ================= 1. 模型定义 (必须和训练时一模一样) =================
# 就像读档一样，你的“空壳”结构必须和存档时的结构完全一致，才能把参数装进去
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# ================= 2. 准备“尺子” (StandardScaler) =================
def get_scaler():
    """
    重新加载训练数据，fit 一遍 scaler，确保推理时的数据缩放比例和训练时一致。
    在工业界，通常会把 scaler 保存为文件，但科研Demo里这样写最快。
    """
    try:
        df = pd.read_csv("expert_data.csv")
        X = df[['task_size', 'latency_limit', 'bandwidth', 'server_load']].values
        scaler = StandardScaler()
        scaler.fit(X)  # 记住数据的均值和方差
        return scaler
    except FileNotFoundError:
        print("❌ 错误：找不到 expert_data.csv，无法校准数据！")
        return None


# ================= 3. 核心推理函数 =================
def predict_decision(model, scaler, task_state, device): # <--- 🔧 多传一个 device 参数
    """
    输入：一个具体的任务状态 [大小, 时延, 带宽, 负载]
    输出：决策 (0/1) 和 置信度
    """
    # 1. 切换到评估模式 (告诉模型：现在是考试，不要进行训练时的随机操作)
    model.eval()

    # 2. 数据预处理
    # 变成 numpy 二维数组 (1行, 4列)
    state_array = np.array([task_state])
    # 用之前的尺子进行标准化
    state_scaled = scaler.transform(state_array)
    # 变成 Tensor，创建 Tensor 时，直接把它送到和模型一样的 device 上
    input_tensor = torch.tensor(state_scaled, dtype=torch.float32).to(device)

    # 3. 推理 (Inference)
    with torch.no_grad():  # ⚠️ 关键！考试时不需要计算梯度，能省内存并加速
        outputs = model(input_tensor)

        # 使用 Softmax 把得分变成概率 (百分比)
        probabilities = torch.softmax(outputs, dim=1)

        # 拿到最大概率的索引 (0 或 1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), confidence.item()


# ================= 4. 主程序 =================
if __name__ == "__main__":
    # A. 加载模型结构，自动检测设备 (有显卡就用显卡，没显卡用CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前运行设备: {device}")
    # 加载模型结构到设备
    model = StudentNet().to(device)

    # B. 加载模型权重 (读取 .pth 存档)
    # map_location='cpu' 保证即使你在服务器训练(GPU)，回笔记本(CPU)也能跑
    try:
        # 🔧 修复警告：添加 weights_only=True
        # map_location 确保权重被加载到正确的设备
        model.load_state_dict(torch.load("student_model.pth", map_location='cpu', weights_only=True))
        print("✅ 模型加载成功！准备进行推理...")
    except FileNotFoundError:
        print("❌ 找不到 student_model.pth，请先运行任务二训练模型！")
        exit()

    # C. 准备尺子
    scaler = get_scaler()
    if scaler is None:
        print("scaler 生成失败")
        exit()

    print("-" * 50)
    print("🤖 Edge-LLM-Student 边缘调度系统已启动")
    print("-" * 50)

    # D. 模拟几个新任务场景
    test_cases = [
        # 格式: [任务大小(MB), 时延要求(ms), 带宽(Mbps), 负载(%)]
        [15.0, 50.0, 80.0, 20.0],  # Case 1: 任务小，网好，负载低 -> 应该卸载 (1)
        [45.0, 20.0, 5.0, 90.0],  # Case 2: 任务巨大，网烂，负载高 -> 必须本地处理/丢弃 (0)
        [10.0, 200.0, 50.0, 50.0],  # Case 3: 中规中矩 -> 看模型判断
        [16.0, 200.0, 50.0, 50.0]   # Case 4: 极限位置 -> 根据Case 3探索模型判断的边界值
    ]

    labels = {0: "本地处理 (Local)", 1: "卸载到边缘 (Edge)"}

    for i, state in enumerate(test_cases):
        action, conf = predict_decision(model, scaler, state, device)

        print(f"\n📝 任务 {i + 1}: 大小={state[0]}MB, 时延={state[1]}ms, 带宽={state[2]}Mbps, 负载={state[3]}%")
        print(f"🧠 模型决策: 【{labels[action]}】")
        print(f"📊 置信度: {conf * 100:.2f}%")  # 看看模型有多确定