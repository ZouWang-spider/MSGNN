import csv
import dgl
import torch
import nltk
from supar import Parser
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
# from MSGNN.BaseModel.GGNN import GGNNModel


#Read SST1-1 dataset
def read(filepath):
    # 打开CSV文件
    with open(filepath, mode='r') as file:
        # 创建CSV读取器
        reader = csv.reader(file)

        # 假设第一行是标题行，跳过它（如果没有标题行，可以省略这一步）
        next(reader, None)

        # 初始化两个空列表，用于存储输入和输出数据
        train_data = []
        train_label = []

        # 逐行读取CSV文件
        for row in reader:
            # 假设输入位于第一列，输出位于第二列
            input_value = row[0]
            output_value = row[1]

            # 将数据添加到相应的列表中
            train_data.append(input_value)
            train_label.append(output_value)
    return train_data, train_label

filepath = '/MSGNN/dataset/SST1/val.csv'
train_data, train_label = read(filepath)
# print("Input Data:", train_label[:5000])   #train_data[:5]
# print("Output Data:", train_label[:2])   #['3', '2']


#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'D:\StanfordCoreNLP\stanford-corenlp-4.5.4', lang='en')

# 初始化两个空列表，用于存储输入和输出数据
Word_feature = []
POS_feature = []

# 循环遍历每个评论语句
for sentence in train_data:
    # 使用 CoreNLP 进行词性标注
    ann = nlp.pos_tag(sentence)
    # print(nlp.pos_tag(sentence))
    # 提取单词和词性
    words = [pair[0] for pair in ann]
    pos_tags = [pair[1] for pair in ann]

    Word_feature.append(words)
    POS_feature.append(pos_tags)


#Word-GNN模型参数
num_nodes = 10
input_size = 768
hidden_size = 768
output_dim = 1
num_layers = 2
num_steps = 5  # Number of GGNN propagation steps
num_etypes = 1  # Number of edge types (can be adjusted based on your dataset)

num_classes = 5   #标签类别数量
num_epochs = 20
learning_rate = 0.001


import torch.optim as optim
import torch.nn.functional as F
from MSGNN.BaseModel.GGNN import GGNN_FCModel

# 定义优化器和损失函数
word_ggnn = GGNN_FCModel(input_size, hidden_size, num_layers, num_steps, num_etypes,num_classes)  # 替换成你的模型
optimizer = optim.Adam(word_ggnn.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

number = round(len(train_label)/num_epochs)


# 加载BERT模型和分词器
model_name = 'bert-base-uncased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

total_loss = 0.0  # 用于累积损失值
total_accuracy = 0.0  # 用于累积正确预测的样本数量
for epoch in range(num_epochs):

    word_ggnn.train()
    start_idx = number * epoch
    end_idx = number * (epoch + 1)

    # Word_embedding_feature =  []
    for i, (text, label) in enumerate(zip(Word_feature[start_idx:end_idx],train_label[start_idx:end_idx])):
        # sentence = "It's a lovely film with lovely performances by Buy and Accorsi."
        # 标记化句子
        marked_text = ["[CLS]"] + text + ["[SEP]"]
        # print("Tokens:", marked_text)

        #将标签转化为张量
        label = int(label)
        label_tensor = torch.tensor([label])
        # print(label_tensor)
        # print('label dim:',label_tensor.shape)

        # 将分词转化为词向量
        # tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = torch.tensor(tokenizer.encode(marked_text, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)

        # 获取词向量
        word_embeddings = outputs.last_hidden_state

        # 提取单词对应的词向量（去掉特殊标记的部分）
        word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        word_list = [item for item in marked_text if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        Word_embedding_feature = word_embeddings[0][1:-1, :]  # 节点特征


        # 使用BiAffine对句子进行处理得到arcs、rels、probs
        # text = nltk.word_tokenize('text')
        parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
        dataset = parser.predict([text], prob=True, verbose=True)
        # print(dataset.sentences[0])
        # print(f"arcs:  {dataset.arcs[0]}\n"
        #       f"rels:  {dataset.rels[0]}\n"
        #       f"probs: {dataset.probs[0].gather(1, torch.tensor(dataset.arcs[0]).unsqueeze(1)).squeeze(-1)}")

        # 构建句子的图，由弧-->节点
        arcs = dataset.arcs[0]  # 边的信息
        edges = [i + 1 for i in range(len(arcs))]
        for i in range(len(arcs)):
            if arcs[i] == 0:
                arcs[i] = edges[i]

        # 将节点的序号减一，以便适应DGL graph从0序号开始
        arcs = [arc - 1 for arc in arcs]
        edges = [edge - 1 for edge in edges]
        graph = (arcs, edges)
        # graph_line = '({}, {})\n'.format(graph[0], graph[1])  # 将图信息转为字符串
        # print("graph:", graph)
        # print(graph_line)

        # Create a DGL graph
        g = dgl.graph(graph)  # 句子的图结构
        # print(g)

        # 训练Word-GNN模型
        optimizer.zero_grad()
        # ggnn_model = word_ggnn(input_size, hidden_size, num_layers, num_steps, num_etypes)

        # Forward pass
        output = word_ggnn(g, Word_embedding_feature)  # GGNN模型输入结果为图、节点特征  output.unsqueeze(0)
        # print(output)    tensor(1,5)
        # print('GGNN model output\n', output)
        loss = loss_function(output, label_tensor)   #tensor(1)
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(output, 1)  # 获取最大值的索引为计算的类别
        accuracy = (predicted == label_tensor).sum().item()
        total_accuracy += accuracy  #累计准确率
        total_loss += loss.item()  # 累积损失值
        # 打印每个 epoch 的损失值
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy / len(label_tensor)}")

    # 计算并打印每个 epoch 的平均损失值,平均准确率
    average_loss = total_loss / number
    average_accuracy = total_accuracy / number
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {average_accuracy}")
    total_loss = 0.0
    total_accuracy = 0.0

# 保存模型的状态字典到文件
torch.save(word_ggnn.state_dict(), 'word_ggnn_model.pth')