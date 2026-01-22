import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 数据加载与预处理（原逻辑保留）=====================
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


# ===================== 2. 数据集类（原逻辑保留）=====================
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# ===================== 3. 可灵活调整层数/节点数的分类器 =====================
class AdjustableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        可调整层数和节点数的分类器
        :param input_dim: 输入维度（词袋向量维度）
        :param hidden_dims: 隐藏层配置列表，如[128]（单层128节点）、[256,128]（两层，256→128）
        :param output_dim: 输出维度（类别数）
        """
        super(AdjustableClassifier, self).__init__()
        # 构建隐藏层
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        # 拼接所有隐藏层
        self.hidden_layers = nn.Sequential(*layers)
        # 输出层
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        out = self.hidden_layers(x)
        out = self.output_layer(out)
        return out


# ===================== 4. 模型训练函数（通用，适配不同配置）=====================
def train_model(model_config, vocab_size, output_dim, dataloader, num_epochs=10, lr=0.01):
    """
    训练指定配置的模型，返回每轮Loss
    :param model_config: 模型配置字典，如{"name": "单层128", "hidden_dims": [128]}
    :return: 每轮的平均Loss列表
    """
    # 初始化模型
    model = AdjustableClassifier(
        input_dim=vocab_size,
        hidden_dims=model_config["hidden_dims"],
        output_dim=output_dim
    )
    # 损失函数+优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 记录每轮Loss
    epoch_losses = []

    print(f"\n========== 开始训练：{model_config['name']} ==========")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 打印批次Loss（可选）
            if idx % 50 == 0 and idx > 0:
                print(f"  Batch {idx}, Batch Loss: {loss.item():.4f}")

        # 计算本轮平均Loss
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    return model, epoch_losses


# ===================== 5. 实验配置（不同层数/节点数）=====================
# 配置1：单层+128节点（原模型）
config1 = {"name": "单层_128节点", "hidden_dims": [128]}
# 配置2：两层+[256,128]节点（加深+更多节点）
config2 = {"name": "两层_256-128节点", "hidden_dims": [256, 128]}
# 配置3：三层+[512,256,128]节点（更深+更多节点）
config3 = {"name": "三层_512-256-128节点", "hidden_dims": [512, 256, 128]}
# 配置4：单层+64节点（对比：更少节点）
config4 = {"name": "单层_64节点", "hidden_dims": [64]}

all_configs = [config1, config2, config3, config4]

# ===================== 6. 执行实验 =====================
# 初始化数据集和DataLoader
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)
num_epochs = 10

# 存储所有模型的Loss结果
all_loss_results = {}
# 存储训练好的模型（可选）
trained_models = {}

# 遍历所有配置训练
for config in all_configs:
    model, loss_list = train_model(config, vocab_size, output_dim, dataloader, num_epochs)
    all_loss_results[config["name"]] = loss_list
    trained_models[config["name"]] = model

# ===================== 7. Loss可视化对比 =====================
plt.figure(figsize=(10, 6))
for model_name, loss_list in all_loss_results.items():
    plt.plot(range(1, num_epochs + 1), loss_list, marker='o', label=model_name)

plt.title('不同模型配置的Loss变化对比', fontsize=14)
plt.xlabel('Epoch（训练轮数）', fontsize=12)
plt.ylabel('Average Loss（平均损失）', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('模型Loss对比图.png', dpi=300)
plt.show()


# ===================== 8. 预测函数（原逻辑保留）=====================
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# ===================== 9. 测试不同模型的预测效果（可选）=====================
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试文本
test_texts = ["帮我导航到北京", "查询明天北京的天气"]

# 打印各模型预测结果
for model_name, model in trained_models.items():
    print(f"\n========== {model_name} 预测结果 ==========")
    for text in test_texts:
        pred_label = classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label)
        print(f"输入 '{text}' → 预测类别: '{pred_label}'")
