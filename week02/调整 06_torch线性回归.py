import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的模拟数据
# 生成0到2π范围内的1000个随机点，形状为(1000, 1)
X_numpy = np.random.rand(1000, 1) * 2 * np.pi  # 输入范围[0, 2π]
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数+少量噪声，更贴近真实场景

# 转换为torch张量（float类型）
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("数据形状：X=", X.shape, " y=", y.shape)
print("---" * 10)

# 2. 构建多层神经网络（替代原来的单层线性模型）
# 结构：输入层(1) -> 隐藏层1(32) -> 隐藏层2(16) -> 输出层(1)
class SinFittingNet(torch.nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        # 第一层全连接：1个输入特征 -> 32个隐藏节点，激活函数用ReLU
        self.fc1 = torch.nn.Linear(1, 32)
        # 第二层全连接：32个节点 -> 16个节点
        self.fc2 = torch.nn.Linear(32, 16)
        # 输出层：16个节点 -> 1个输出（拟合sin函数的结果）
        self.fc3 = torch.nn.Linear(16, 1)
        # 激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 前向传播：输入 -> 隐藏层1(ReLU) -> 隐藏层2(ReLU) -> 输出层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = SinFittingNet()
print("多层网络结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 回归任务仍用均方误差
# 优化器：传入模型所有参数，学习率可适当调整
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器比SGD更适合拟合非线性函数

# 4. 训练模型
num_epochs = 1000
loss_history = []  # 记录损失变化，方便后续可视化
for epoch in range(num_epochs):
    # 前向传播：用模型预测
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    loss_history.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 测试模型（生成平滑的预测曲线）
# 生成0到2π范围内的1000个均匀点，用于绘制平滑的拟合曲线
X_test = torch.linspace(0, 2 * np.pi, 1000).reshape(-1, 1).float()
with torch.no_grad():  # 推理阶段不需要计算梯度，节省资源
    y_pred_test = model(X_test)

# 转换为numpy数组，方便绘图
X_test_numpy = X_test.numpy()
y_pred_test_numpy = y_pred_test.numpy()
y_true_numpy = np.sin(X_test_numpy)  # 真实的sin函数值

# 6. 可视化结果
plt.figure(figsize=(12, 8))

# 子图1：损失变化曲线
plt.subplot(2, 1, 1)
plt.plot(loss_history, color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)

# 子图2：sin函数拟合效果
plt.subplot(2, 1, 2)
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='lightblue', alpha=0.5, s=10)
plt.plot(X_test_numpy, y_true_numpy, label='True sin(x)', color='red', linewidth=2)
plt.plot(X_test_numpy, y_pred_test_numpy, label='Fitted Model', color='green', linewidth=2, linestyle='--')
plt.xlabel('x (0 to 2π)')
plt.ylabel('y = sin(x)')
plt.legend()
plt.grid(True)
plt.title('Multi-layer Network Fitting sin(x)')

plt.tight_layout()  # 调整子图间距
plt.show()

# 打印最终的损失值
print("\n训练完成！最终损失值：", loss_history[-1])
