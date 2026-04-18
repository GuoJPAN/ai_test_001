# agents/db_agent.py
"""
DBAgent - 数据库管理子 Agent
专门负责数据库查询和数据分析任务
"""

from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from tools.db_tools import query_sales_data, execute_safe_query
from config import config


# ========== 1. 定义 DBAgentState ==========
class DBAgentState(TypedDict):
    """DBAgent 状态"""
    messages: list
    # 任务相关上下文
    task_context: str  # 从 Supervisor 传来的任务描述
    task_result: str  # 任务执行结果


# ========== 2. 定义 DBAgent 的工具列表 ==========
db_tools = [query_sales_data, execute_safe_query]


# ========== 3. System Prompt ==========
DB_AGENT_SYSTEM_PROMPT = """你是一个专业的数据库助手（DBAgent）。

## 你的职责
- 专门负责数据库查询和数据分析任务
- 只处理与数据库相关的请求，不要处理文件操作或邮件发送任务

## 可用工具
- query_sales_data: 查询销售记录（预设的安全查询）
- execute_safe_query: 执行只读SELECT查询（安全防护，只允许 SELECT 语句）

## 工作流程
1. 理解用户的数据库查询需求
2. 选择合适的工具执行查询
3. 返回查询结果

## 约束
- 如果用户要求你处理非数据库任务（如读取文件、发送邮件），请明确告知用户这不属于你的职责范围
- execute_safe_query 仅允许 SELECT 语句，不允许 INSERT、UPDATE、DELETE 或其他危险操作
- 返回结果要清晰，标明查询条件和结果条数
- 如果查询较为复杂，先向用户解释查询逻辑再执行
"""


# ========== 4. 初始化 DBAgent 模型 ==========
db_model = ChatOpenAI(
    model=config.MODEL_NAME,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
    timeout=120
)
db_model_with_tools = db_model.bind_tools(db_tools)


# ========== 5. Agent 节点函数 ==========
def db_agent_node(state: DBAgentState):
    """DBAgent 模型调用节点"""
    messages = state["messages"]
    
    # 添加系统提示（首次调用时）
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        system_prompt = {"role": "system", "content": DB_AGENT_SYSTEM_PROMPT}
        messages = [system_prompt] + list(messages)
    
    response = db_model_with_tools.invoke(messages)
    return {"messages": [response]}


# ========== 6. 条件边函数 ==========
def db_should_continue(state: DBAgentState):
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 7. 构建 DBAgent 图 ==========
db_workflow = StateGraph(DBAgentState)
db_workflow.add_node("agent", db_agent_node)
db_workflow.add_node("tools", ToolNode(db_tools))
db_workflow.add_edge(START, "agent")
db_workflow.add_conditional_edges("agent", db_should_continue, ["tools", END])
db_workflow.add_edge("tools", "agent")

db_memory = MemorySaver()
db_agent_app = db_workflow.compile(checkpointer=db_memory)


# ========== 8. 辅助函数：调用 DBAgent ==========
def call_db_agent(task: str) -> str:
    """
    调用 DBAgent 执行数据库任务
    
    Args:
        task: 任务描述，如 "查询最近10条销售���录"
    
    Returns:
        str: 执行结果
    """
    input_state = {
        "messages": [("user", task)],
        "task_context": task,
        "task_result": ""
    }
    
    # 配置 checkpointer 的 thread_id
    config = {"configurable": {"thread_id": f"db_agent_{hash(task) % 10000}"}}
    
    result = ""
    for event in db_agent_app.stream(input_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                result = last_msg.content
    
    return result


# ========== 9. 创建 LangChain Tool compatible 工具 ==========
from langchain_core.tools import tool


@tool
def call_db_agent_tool(task: str) -> str:
    """
    调用 DBAgent 执行数据库查询任务
    
    Args:
        task: 任务描述，如 "查询最近5条销售记录" 或 "执行 SELECT * FROM sales WHERE amount > 1000"
    
    Returns:
        str: DBAgent 的执行结果
    """
    return call_db_agent(task)