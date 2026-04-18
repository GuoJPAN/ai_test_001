# agents/file_agent.py
"""
FileAgent - 文件管理子 Agent
专门负责文件读取、列出等文件操作任务
"""

from typing import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from tools.file_tools import read_local_file, list_files
from config import config


# ========== 1. 定义 FileAgentState ==========
class FileAgentState(TypedDict):
    """FileAgent 状态"""
    messages: list
    # 任务相关上下文
    task_context: str  # 从 Supervisor 传来的任务描述
    task_result: str  # 任务执行结果


# ========== 2. 定义 FileAgent 的工具列表 ==========
file_tools = [read_local_file, list_files]


# ========== 3. System Prompt ==========
FILE_AGENT_SYSTEM_PROMPT = """你是一个专业的文件管理助手（FileAgent）。

## 你的职责
- 专门负责读取文件和列出目录文件的任务
- 只处理与文件操作相关的请求，不要处理数据库或邮件相关任务

## 可用工具
- read_local_file: 读取文件内容（支持 .txt, .md, .py, .json, .yaml, .csv, .docx, .pdf 等格式）
- list_files: 列出指定目录下的文件

## 工作流程
1. 理解用户要读取或查看的文件任务
2. 调用相应工具完成操作
3. 返回文件内容或文件列表

## 约束
- 如果用户要求你处理非文件操作任务（如数据库查询、发送邮件），请明确告知用户这不属于你的职责范围，建议联系对应的 Agent
- 如果文件不存在或无法读取，请明确告知错误原因
- 返回结果要清晰完整，标明文件路径和内容摘要
"""


# ========== 4. 初始化 FileAgent 模型 ==========
file_model = ChatOpenAI(
    model=config.MODEL_NAME,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
    timeout=120
)
file_model_with_tools = file_model.bind_tools(file_tools)


# ========== 5. Agent 节点函数 ==========
def file_agent_node(state: FileAgentState):
    """FileAgent 模型调用节点"""
    messages = state["messages"]
    
    # 添加系统提示（首次调用时）
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        system_prompt = {"role": "system", "content": FILE_AGENT_SYSTEM_PROMPT}
        messages = [system_prompt] + list(messages)
    
    response = file_model_with_tools.invoke(messages)
    return {"messages": [response]}


# ========== 6. 条件边函数 ==========
def file_should_continue(state: FileAgentState):
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 7. 构建 FileAgent 图 ==========
file_workflow = StateGraph(FileAgentState)
file_workflow.add_node("agent", file_agent_node)
file_workflow.add_node("tools", ToolNode(file_tools))
file_workflow.add_edge(START, "agent")
file_workflow.add_conditional_edges("agent", file_should_continue, ["tools", END])
file_workflow.add_edge("tools", "agent")

file_memory = MemorySaver()
file_agent_app = file_workflow.compile(checkpointer=file_memory)


# ========== 8. 辅助函数：调用 FileAgent ==========
def call_file_agent(task: str) -> str:
    """
    调用 FileAgent 执行文件任务
    
    Args:
        task: 任务��述，如 "读取 ./data/documents/report.txt"
    
    Returns:
        str: 执行结果
    """
    input_state = {
        "messages": [("user", task)],
        "task_context": task,
        "task_result": ""
    }
    
    # 配置 checkpointer 的 thread_id
    config = {"configurable": {"thread_id": f"file_agent_{hash(task) % 10000}"}}
    
    result = ""
    for event in file_agent_app.stream(input_state, config, stream_mode="values"):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                result = last_msg.content
    
    return result


# ========== 9. 创建 LangChain Tool compatible 工具 ==========
from langchain_core.tools import tool


@tool
def call_file_agent_tool(task: str) -> str:
    """
    调用 FileAgent 执行文件操作任务
    
    Args:
        task: 任务描述，如 "读取 ./data/documents/report.txt" 或 "列出 ./data/documents 目录下的文件"
    
    Returns:
        str: FileAgent 的执行结果
    """
    return call_file_agent(task)