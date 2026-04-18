import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:14b-custom")

    # 邮件配置
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

    # 安全配置
    ALLOWED_FILE_DIR = os.getenv("ALLOWED_FILE_DIR", "./data/documents")
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", 50))

config = Config()