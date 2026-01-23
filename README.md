# Pytorch-Template

# LLM_RL项目
## 1.生成原始数据
使用"generate_scenarios.py"生成原始数据"raw_scenarios.csv"

## 2.生成专家数据
"label_with_llm.py"，使用deepseek-v3.2大模型训练出专家数据，并保存至"expert_data.csv"

## 3.生成自己的离线神经网络
"train_student.py"，根据专家数据，使用SGD训练新的神经网络，并把参数保存至"student_model.pth"

## 4.测试离线神经网络
"inference_demo.py"，加载训练好的神经网络对新的数据进行预测，并输出模型对自己判断的置信度

