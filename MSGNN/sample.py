# 原始句子
sentence_tokens = ['It', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'Buy', 'and', 'Accorsi', '.']

# 在前面添加[CLS]，在后面添加[SEP]
tokens_with_cls_sep = ["[CLS]"] + sentence_tokens + ["[SEP]"]

# 打印结果
print(tokens_with_cls_sep)

import torch
import torch.nn as nn

import torch
import torch_geometric
from torch_geometric.nn import GCNConv

# 定义输入节点特征和边的索引
num_nodes = 10
in_channels = 64
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])

x = torch.randn(num_nodes, in_channels)

# 创建 GCNConv 层
gcn_layer = GCNConv(in_channels, out_channels=128)

# 使用 GCNConv 层进行前向传播
output = gcn_layer(x, edge_index)
print(output)
average_accuracy=1
average_loss=1
num_epochs=50
for epoch in range(num_epochs):
    with open('out.txt', 'a') as file:
        # print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {average_accuracy}")
        output_string = f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {average_accuracy}\n"
        file.write(output_string)


import matplotlib.pyplot as plt
true_labels=[3,4,3,4,3,4,1,2,1,2]
predicted_labels=[1,1,1,4,4,4,2,2,1,2]

# 绘制标签的散点图
plt.figure(figsize=(10, 5))
plt.scatter(range(len(true_labels)), true_labels, label='True Labels', marker='o', color='blue')
plt.scatter(range(len(predicted_labels)), predicted_labels, label='Predicted Labels', marker='x', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.title('True vs. Predicted Labels')
plt.legend()
plt.show()
