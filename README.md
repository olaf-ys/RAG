# RAG 项目

这是一个基于 OpenAI API 的 RAG（Retrieval-Augmented Generation）项目。

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
OPENAI_BASE_URL=https://ai.devtool.tech/proxy/v1
```

### 3. 运行项目

```bash
# 运行主程序
python main.py

# 或者使用 Jupyter Notebook
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
├── main.py              # 主程序
├── RAG.ipynb            # Jupyter Notebook
├── .env                 # 环境变量配置（不提交到版本控制）
├── .env.example         # 环境变量模板
├── pyproject.toml       # 项目依赖
├── uv.lock             # 依赖锁定文件
└── README.md           # 项目说明
```

## 依赖安装

```bash
# 使用 uv 安装依赖
uv install

# 或者使用 pip
pip install -r requirements.txt
```