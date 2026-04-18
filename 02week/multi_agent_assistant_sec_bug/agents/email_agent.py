# agents/email_agent.py
"""
EmailAgent - 邮件管理子 Agent
专门负责邮件发送任务
"""

from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from tools.email_tools import send_email
from config import config


# ========== 1. 定义 EmailAgentState ==========
class EmailAgentState(TypedDict):
    """EmailAgent 状态"""
    messages: list
    # 任务相关上下文
    task_context: str  # 从 Supervisor 传来的任务描述
    task_result: str  # 任务执行结果


# ========== 2. 定义 EmailAgent 的工具列表 ==========
email_tools = [send_email]


# ========== 3. System Prompt ==========
EMAIL_AGENT_SYSTEM_PROMPT = """你是一个专业的邮件助手（EmailAgent）。

## 你的职责
- 专门负责邮件发送任务
- 只处理与邮件相关的请求，不要处理文件操作或数据库查询任务

## 可用工具
- send_email: 发送邮件（使用 SMTP）

## 工作流程
1. 理解用户的邮件发送需求（收件人、主题、正文）
2. 调用 send_email 工具发送邮件
3. 返回发送结果

## 约束
- 如果用户要求你处理非邮件任务（如读取文件、查询数据库），请明确告知用户这不属于你的职责范围
- 收件人邮箱地址需要符合基本格式规范
- 邮件主题和正文不能包含敏感信息（如密码、密钥等）
- 发送失败时给出明确的错误原因
- 发送成功后确认邮件已发送
"""


# ========== 4. 初始化 EmailAgent 模型 ==========
email_model = ChatOpenAI(
    model=config.MODEL_NAME,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
    timeout=120
)
email_model_with_tools = email_model.bind_tools(email_tools)


# ========== 5. Agent 节点函数 ==========
def email_agent_node(state: EmailAgentState):
    """EmailAgent 模型调用节点"""
    messages = state["messages"]
    
    # 添加系统提示（首次调用时）
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        system_prompt = {"role": "system", "content": EMAIL_AGENT_SYSTEM_PROMPT}
        messages = [system_prompt] + list(messages)
    
    response = email_model_with_tools.invoke(messages)
    return {"messages": [response]}


# ========== 6. 条件边函数 ==========
def email_should_continue(state: EmailAgentState):
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 7. 构建 EmailAgent 图 ==========
email_workflow = StateGraph(EmailAgentState)
email_workflow.add_node("agent", email_agent_node)
email_workflow.add_node("tools", ToolNode(email_tools))
email_workflow.add_edge(START, "agent")
email_workflow.add_conditional_edges("agent", email_should_continue, ["tools", END])
email_workflow.add_edge("tools", "agent")

email_memory = MemorySaver()
email_agent_app = email_workflow.compile(checkpointer=email_memory)


# ========== 8. 辅助函数：调用 EmailAgent ==========
def call_email_agent(task: str) -> str:
    """
    调用 EmailAgent 执行邮件任务
    
    Args:
        task: 任务描述，如 "发送邮件给 test@example.com，主题：测试，内容：Hello World"
    
    Returns:
        str: 执行结果
    """
    input_state = {
        "messages": [("user", task)],
        "task_context": task,
        "task_result": ""
    }
    
    # 配置 checkpointer 的 thread_id
    config = {"configurable": {"thread_id": f"email_agent_{hash(task) % 10000}"}}
    
    result = ""
    for event in email_agent_app.stream(input_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                result = last_msg.content
    
    return result


# ========== 9. 创建 LangChain Tool compatible 工具 ==========
from langchain_core.tools import tool


@tool
def call_email_agent_tool(task: str) -> str:
    """
    调用 EmailAgent 执行邮件发送任务
    
    Args:
        task: 任务描述，如 "发送邮件给 test@example.com，主题：测试，内容：Hello"
    
    Returns:
        str: EmailAgent 的执行结果
    """
    return call_email_agent(task)