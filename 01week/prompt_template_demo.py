import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="qwen3.5-9b-gpu:latest",
    base_url="http://192.168.3.8:11434/v1",
    api_key="ollama",
    temperature=0.7,
)

# 定义一个提示词模板
prompt = ChatPromptTemplate.from_template(
    "请用{style}的语言解释什么是{concept}，并给出一个例子。"
)

# 构建链
chain = prompt | llm | StrOutputParser()

# 调用
result = chain.invoke({
    "style": "通俗易懂",
    "concept": "提示词注入"
})
print(result)