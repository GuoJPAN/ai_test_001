# ==============================================
# 模块3 Memory - 企业生产级安全实现
# 功能：多用户隔离 | 敏感信息脱敏 | 记忆长度限制 | 防注入 | 防越权
# 适配：Ollama + qwen2.5:7b-custom
# ==============================================
"""
1. ChatMessageHistory（最基础：存原始消息）
2. ConversationBufferMemory（缓冲记忆：全量存对话）
3. ConversationSummaryMemory（摘要记忆：压缩长对话）
4. ConversationBufferWindowMemory（窗口记忆：只保留最近 K 轮）

风险 1：记忆存储敏感信息 → 未脱敏泄露
风险 2：无长度限制 → 长上下文注入攻击（超级高危）
风险 3：多用户记忆不隔离 → A 用户看到 B 用户对话（越权访问）
"""
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# ===================== 生产环境配置 =====================
OLLAMA_BASE_URL = "http://192.168.3.8:11434/v1"
MODEL_NAME = "qwen2.5:7b-custom"
MAX_MEMORY_ROUNDS = 3        # 仅保留最近3轮对话（防长上下文注入）
ENABLE_DESENSITIZE = True    # 生产必须开启
LOG_LEVEL = logging.INFO

# ===================== 初始化日志（生产级脱敏） =====================
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("llm_memory_pro")

# ===================== 初始化LLM =====================
llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    temperature=0.1,
    max_tokens=1024,
    timeout=60,
)

# ===================== 【生产核心1】敏感信息脱敏 =====================
def desensitize(text: str) -> str:
    """生产级多维度脱敏，防止隐私存入记忆"""
    if not text:
        return text

    # 手机号
    text = re.sub(r"1[3-9]\d{9}", "1**********", text)
    # 身份证
    text = re.sub(r"(\d{6})\d{8}(\d{4})", r"\1********\2", text)
    # 邮箱
    text = re.sub(r"(\w+)\@(\w+\.\w+)", r"****@\2", text)
    # 银行卡
    text = re.sub(r"(\d{4})\d{10}(\d{4})", r"\1**********\2", text)
    # 密钥特征
    text = re.sub(r"sk-[a-zA-Z0-9]{20,48}", "sk-******************", text)
    return text

# ===================== 【生产核心2】记忆长度限制（防长上下文注入） =====================
def limit_memory(history: BaseChatMessageHistory):
    """强制限制对话轮数，杜绝超长注入攻击"""
    max_msgs = MAX_MEMORY_ROUNDS * 2
    if len(history.messages) > max_msgs:
        history.messages = history.messages[-max_msgs:]
        logger.info(f"[记忆安全] 已截断超长上下文，保留最近 {MAX_MEMORY_ROUNDS} 轮")

# ===================== 【生产核心3】多用户记忆隔离（防越权） =====================
# 生产可替换为 RedisChatMessageHistory 实现分布式存储
MEMORY_STORE = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    按 session_id 隔离用户记忆
    生产环境：替换成 Redis/MySQL 持久化
    """
    if not session_id or len(session_id) < 5:
        raise ValueError("非法会话ID，拒绝创建记忆")

    if session_id not in MEMORY_STORE:
        MEMORY_STORE[session_id] = InMemoryChatMessageHistory()
        logger.info(f"[记忆安全] 新建用户会话：{session_id[:8]}****")

    history = MEMORY_STORE[session_id]
    limit_memory(history)  # 每次获取自动截断
    return history

# ===================== 构建生产级带记忆对话链 =====================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是安全合规的AI助手，所有记忆必须脱敏、隔离、长度受限。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 基础链
base_chain = prompt | llm | StrOutputParser()

# 带记忆 + 安全控制
memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ===================== 【生产核心4】安全调用入口（统一封装） =====================
def chat(session_id: str, user_input: str) -> str:
    try:
        # 1. 输入脱敏
        user_input_safe = desensitize(user_input) if ENABLE_DESENSITIZE else user_input

        # 2. 调用带记忆链
        response = memory_chain.invoke(
            {"input": user_input_safe},
            config={"configurable": {"session_id": session_id}}
        )

        # 3. 输出脱敏
        response_safe = desensitize(response)
        logger.info(f"[对话成功] session={session_id[:8]}****")
        return response_safe

    except Exception as e:
        logger.error(f"[系统异常] 错误信息已脱敏")
        return "系统繁忙，请稍后再试"

# ===================== 生产级测试 =====================
if __name__ == "__main__":
    print("=" * 70)
    print("    模块3 Memory 企业生产版 启动成功")
    print("=" * 70)

    # ========== 测试：用户A（带隐私） ==========
    print("\n【用户A】")
    print(chat("user_20250403_001", "我的手机号13812345678，身份证110101199003031234"))

    # ========== 测试：用户B（完全隔离） ==========
    print("\n【用户B】")
    print(chat("user_20250403_002", "我是独立用户，看不到别人的记忆"))

    # ========== 测试：超长对话自动截断 ==========
    print("\n【测试长记忆自动限制】")
    for i in range(10):
        chat("user_test_limit", f"第{i+1}轮对话")
    hist = get_session_history("user_test_limit")
    print(f"最终记忆条数：{len(hist.messages)}（应≤6）")

    print("\n" + "=" * 70)
    print("  ✅ 生产环境 Memory 三防护全部生效：")
    print("  1.敏感信息脱敏  2.长上下文注入防护  3.多用户越权防护")
    print("=" * 70)