# 从0搭建RAG

这是一个从0搭建的 RAG（检索增强生成）迷你项目，实现了文档向量化存储、相似度检索和智能问答功能。

## 主要功能

- 📄 多格式文档读取（PDF、Markdown、TXT）
- 🔍 文档智能分块处理
- 🧮 OpenAI Embedding 向量化
- 💾 向量数据库存储与检索
- 🤖 GPT-4o 智能问答

### 文档分块策略
- 🔢 基于 Token 数量的智能分块
- 📏 可配置的最大长度和重叠长度
- 🔄 支持跨行分割和内容重叠

### 向量检索算法
- 📐 余弦相似度计算
- 🎯 Top-K 相似文档检索
- ⚡ 高效的向量搜索

### 问答生成策略
- 📝 结构化的 Prompt 模板
- 🧠 上下文感知的回答生成
- 🌐 中文优化的回答质量

## 🚀 快速开始

### python版本

- Python >= 3.13

### 环境变量配置

- 创建 `.env` 文件并配置API密钥：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 或您的自定义API端点
```

## 主要文件说明

### RAG.py
- 包含所有核心类的封装实现
- 可以直接运行查看完整的RAG系统演示
- 包含 ReadFiles、OpenAIEmbedding、VectorStore、GPT4oChat 等主要类

### RAG.ipynb
- 交互式 Jupyter Notebook 演示文件
- 详细展示如何调用各个类和方法
- 适合学习和实验RAG系统的各个组件
- 支持逐步执行和结果查看

## 🏗️ 项目结构

```
.
├── RAG.py                      # 主要类的封装实现
├── RAG.ipynb                   # 交互式演示和方法调用示例
├── README.md                   # 项目说明文档
├── pyproject.toml              # 项目依赖配置
├── uv.lock                     # 依赖锁定文件
├── .env                        # 环境变量配置（不提交到版本控制）
├── .env.example                # 环境变量模板
├── data/                       # 数据文件目录
│   └── *.txt                   # 文档数据文件
└── storage/                    # 向量存储目录
    ├── documents.txt           # 分块后的文档内容
    └── vectors.npy             # 文档向量数据
```

## 核心功能模块

### 1. 文档处理模块 (ReadFiles)

支持多种文件格式的读取和智能分块：

```python
# 初始化文件读取器
file_reader = ReadFiles(path="./data")

# 获取支持的文件列表
file_list = file_reader.get_files()

# 将文档分块处理
document_chunks = file_reader.get_content(max_token_len=600, cover_content=150)
```

**支持的文件格式：**
- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.pdf` - PDF 文档

### 2. 向量化模块 (OpenAIEmbedding)

基于 OpenAI Embedding API 的文本向量化：

```python
# 初始化 Embedding 模型
embedding_model = OpenAIEmbedding()

# 获取文本向量
vector = embedding_model.get_embedding("示例文本")

# 计算余弦相似度
similarity = OpenAIEmbedding.cosine_similarity(vector1, vector2)
```

### 3. 向量存储模块 (VectorStore)

实现向量的存储、检索和持久化：

```python
# 创建向量数据库
vector_store = VectorStore(document=document_chunks)

# 文档向量化
vector_store.get_vector(embedding_model)

# 持久化存储
vector_store.persist('storage')

# 相似度检索
results = vector_store.query("查询问题", embedding_model, k=3)
```

### 4. 智能问答模块 (GPT4oChat)

基于 GPT-4o 的智能问答系统：

```python
# 初始化聊天模型
chat = GPT4oChat(api_key=api_key)

# 生成回答
answer = chat.chat("用户问题", [], context_content)
```

## 使用方法

### 完整 RAG 流程示例

```python
# 1. 文档处理
file_reader = ReadFiles(path="./data")
document_chunks = file_reader.get_content()

# 2. 向量化存储
vector_store = VectorStore(document=document_chunks)
embedding_model = OpenAIEmbedding()
vector_store.get_vector(embedding_model)
vector_store.persist('storage')

# 3. 智能问答
chat = GPT4oChat(api_key=api_key)
query = "您的问题"
results = vector_store.query(query, embedding_model)
answer = chat.chat(query, [], results[0])
print(answer)
```

## ❓ 常见问题

### Q: 如何更换 Embedding 模型？
A: 在 `OpenAIEmbedding.get_embedding()` 方法中修改 `model` 参数。

### Q: 如何调整文档分块大小？
A: 在 `ReadFiles.get_content()` 方法中调整 `max_token_len` 和 `cover_content` 参数。

### Q: 如何修改问答模板？
A: 修改 `PROMPT_TEMPLATE` 中的模板内容。

### Q: 支持哪些文档格式？
A: 目前支持 PDF、Markdown 和 TXT 文件格式。