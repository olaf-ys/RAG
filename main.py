import os
from openai import OpenAI
from dotenv import load_dotenv

def main():
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
    if not base_url:
        raise ValueError("请在 .env 文件中设置 OPENAI_BASE_URL")
    
    # 创建客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    print("Hello from RAG!")
    print(f"使用API密钥: {api_key[:10]}...")
    print(f"Base URL: {base_url}")


if __name__ == "__main__":
    main()
