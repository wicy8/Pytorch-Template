import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= 1. 数据加载与预处理 =================
print("Loading data...")

# [Pandas]: 读取 CSV 文件
df = pd.read_csv("expert_data.csv")

# 提取特征 (Input) 和 标签 (Label)
# 特征是我们生成的环境状态：任务大小, 时延要求, 带宽, 负载
X_raw = df[['task_size', 'latency_limit', 'bandwidth', 'server_load']].values  # .values 把表格变成了 Numpy 数组
# 标签是老师给出的动作：0 或 1
y_raw = df['label_action'].values

# [数据标准化]: 这一步很重要！
# 因为 '任务大小' 是 1-50，而 '时延' 是 10-200。数值范围差距太大，神经网络会晕。
# StandardScaler 把它们都缩放到 0 附近 (均值为0，方差为1)。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# [Sklearn]: 划分训练集和测试集
# 80% 的数据用来训练(学习)，20% 用来测试(考试)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)

# [PyTorch]: 把 Numpy 数组变成 PyTorch 能认识的 Tensor (张量)
# float32 是标准小数格式，long 是整数格式(用于分类标签)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"数据准备完毕！训练集数量: {len(X_train)}, 测试集数量: {len(X_test)}")


# ================= 2. 定义神经网络 (Student Model) =================
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        # 这是一个简单的 3 层全连接网络 (MLP)
        # 输入层 (4个特征) -> 隐藏层 (64个神经元) -> 输出层 (2个动作: 0或1)
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()  # 激活函数 (给网络一点非线性思考能力)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 输出2个值，分别代表选0和选1的“得分”

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# 初始化模型
student_model = StudentNet()
# 定义损失函数：交叉熵损失 (CrossEntropyLoss)，专门用于分类任务
criterion = nn.CrossEntropyLoss()
# 定义优化器：Adam (目前最常用的优化器)，lr是学习率
optimizer = optim.Adam(student_model.parameters(), lr=0.01)

# ================= 3. 开始训练 (Training Loop) =================
epochs = 100  # 训练 100 轮
losses = []  # 用于画图，记录每一轮的 Loss

print("\n🚀 开始训练 Student 模型...")

for epoch in range(epochs):
    # --- 正向传播 ---
    optimizer.zero_grad()  # 梯度清零 (每次重新算)
    outputs = student_model(X_train_tensor)  # 喂数据，得结果
    loss = criterion(outputs, y_train_tensor)  # 算算和标准答案差多少

    # --- 反向传播 ---
    loss.backward()  # 算出误差怎么传导
    optimizer.step()  # 更新网络参数

    losses.append(loss.item())  # 记录 Loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# ================= 4. 考试与画图 (Result Visualization) =================

# 在测试集上考试
with torch.no_grad():  # 考试时不需要算梯度
    test_outputs = student_model(X_test_tensor)
    # torch.max 返回最大值的索引 (即网络认为概率最大的动作)
    _, predicted = torch.max(test_outputs, 1)
    # 算出准确率
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\n✅ 训练结束！")
print(f"🎓 Student 模型在测试集上的准确率: {accuracy * 100:.2f}%")

# 保存模型
torch.save(student_model.state_dict(), "student_model.pth")
print("💾 模型已保存为 student_model.pth")

# [Matplotlib]: 画出 Loss 下降曲线
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()  # 这会弹出一个窗口显示图片