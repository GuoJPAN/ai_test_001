from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys
import io
# 设置标准输出编码为UTF-8，解决Windows中文输出问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 连接你的主力机模型
llm = ChatOpenAI(
    model="qwen2.5:14b-custom",
    base_url="http://192.168.3.8:11434/v1",
    api_key="ollama",
)

# 1. PromptTemplate
prompt = ChatPromptTemplate.from_template("请用{style}的语言解释：{concept}")

# 2. Chain（用 LCEL 的管道操作符）
chain = prompt | llm | StrOutputParser()

# 3. 调用
result = chain.invoke({
    "style": "通俗易懂",
    "concept": "提示词注入"
})
print(result)
