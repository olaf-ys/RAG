# RAG (Retrieval-Augmented Generation) 项目

这是一个基于 OpenAI API 的 RAG（检索增强生成）项目，实现了文档向量化存储、相似度检索和智能问答功能。

## 项目简介

本项目实现了一个完整的 RAG 系统，包括：
- 📄 多格式文档读取（PDF、Markdown、TXT）
- 🔍 文档智能分块处理
- 🧮 OpenAI Embedding 向量化
- 💾 向量数据库存储与检索
- 🤖 GPT-4o 智能问答

## 环境变量配置

为了安全地管理 API 密钥，本项目使用环境变量配置。请按照以下步骤设置：

### 1. 创建 .env 文件

复制 `.env.example` 文件为 `.env`：

```bash
cp .env.example .env
```

### 2. 配置 API 密钥

在 `.env` 文件中填入您的实际 API 密钥：

```
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

**注意**: 本项目支持多种 API 端点，包括 OpenAI 官方 API 和兼容的代理服务。

### 3. 运行项目

```bash
# RAG.py 包含主要类的封装，可以直接运行查看演示
python RAG.py

# RAG.ipynb 演示如何调用这些类和方法，推荐使用 Jupyter Notebook 进行交互式学习
jupyter notebook RAG.ipynb
```

## 安全提示

- ✅ `.env` 文件已添加到 `.gitignore`，不会被提交到版本控制
- ✅ 使用环境变量而非硬编码API密钥
- ✅ 提供了 `.env.example` 模板文件
- ❌ 切勿将真实的API密钥提交到代码仓库

## 项目结构

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

## 项目文件说明

### RAG.py
- 包含所有核心类的封装实现
- 可以直接运行查看完整的RAG系统演示
- 包含 ReadFiles、OpenAIEmbedding、VectorStore、GPT4oChat 等主要类

### RAG.ipynb
- 交互式 Jupyter Notebook 演示文件
- 详细展示如何调用各个类和方法
- 适合学习和实验RAG系统的各个组件
- 支持逐步执行和结果查看

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

## 技术特性

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

## API 配置说明

### 支持的模型
- **Embedding 模型**: `text-embedding-3-large`
- **聊天模型**: `gpt-4o-mini`

### 支持的 API 端点
- OpenAI 官方 API: `https://api.openai.com/v1`
- 兼容的代理 API 服务

## 项目结构

## 依赖安装

### Python 环境要求
本项目推荐使用 Python 3.8+ 版本。

### 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
uv install

### 主要依赖包
- `openai` - OpenAI API 客户端
- `python-dotenv` - 环境变量管理
- `numpy` - 数值计算
- `tiktoken` - Token 计数
- `PyPDF2` - PDF 文件处理
- `markdown` - Markdown 文件处理
- `beautifulsoup4` - HTML 解析
- `tqdm` - 进度条显示

## 常见问题

### Q: 如何更换 Embedding 模型？
A: 在 `OpenAIEmbedding.get_embedding()` 方法中修改 `model` 参数。

### Q: 如何调整文档分块大小？
A: 在 `ReadFiles.get_content()` 方法中调整 `max_token_len` 和 `cover_content` 参数。

### Q: 如何修改问答模板？
A: 修改 `PROMPT_TEMPLATE` 中的模板内容。

### Q: 支持哪些文档格式？
A: 目前支持 PDF、Markdown 和 TXT 文件格式。

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 许可证

[MIT License](LICENSE)

---

**重要提示**: 使用本项目需要有效的 OpenAI API 密钥。请确保您的 API 密钥有足够的额度来支持 Embedding 和聊天功能的调用。