# agents/supervisor.py
"""
Supervisor Agent - 多 Agent 协调器
负责解析用户意图，将任务分发给对应的子 Agent (FileAgent, DBAgent, EmailAgent)
子 Agent 之间不直接通信，都通过 Supervisor 协调
"""

from typing import Literal, Any, TypedDict, List, Union
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

from config import config
from security.permissions import has_permission, Role
from security.audit import log_audit

# 导入子 Agent 和它们的工具
from agents.file_agent import call_file_agent_tool, file_agent_app
from agents.db_agent import call_db_agent_tool, db_agent_app
from agents.email_agent import call_email_agent_tool, email_agent_app


# ========== 0. 定义 SupervisorState（包含任务路由信息）==========
class SupervisorState(TypedDict):
    """Supervisor 状态，包含用户消息和任务路由信息"""
    messages: list
    user_role: str  # 用户角色
    # 任务路由相关字段
    target_agents: List[str]  # 目标子 Agent 列表: ["file", "db", "email"]
    task_results: dict  # 各子 Agent 的执行结果: {"file": "...", "db": "...", "email": "..."}
    pending_tasks: List[str]  # 待执行的任务队列


# ========== 1. 定义子 Agent 工具列表（Supervisor 可调用的工具）==========
# 这些工具允许 Supervisor 调用对应的子 Agent
supervisor_tools = [call_file_agent_tool, call_db_agent_tool, call_email_agent_tool]


# ========== 2. Supervisor System Prompt ==========
SUPERVISOR_SYSTEM_PROMPT = """你是一个多 Agent 协调器（Supervisor）。

## 你的职责
- 解析用户意图，将任务路由到对应的子 Agent
- 协调多个子 Agent 协作完成复杂任务
- 只负责分发任务，不直接处理文件、数据库或邮件等具体操作

## 子 Agent 类型
1. **FileAgent** - 负责文件操作（读取文件、列出目录）
   - 工具: call_file_agent_tool
   
2. **DBAgent** - 负责数据库查询
   - 工具: call_db_agent_tool
   
3. **EmailAgent** - 负责邮件发送
   - 工具: call_email_agent_tool

## 路由规则
根据用户输入判断需要调用哪些 Agent：

| 用户意图 | 目标 Agent |
|---------|-----------|
| 读取文件、列出目录 | FileAgent |
| 查询数据库、数据分析 | DBAgent |
| 发送邮件 | EmailAgent |
| 复杂任务（如先读取文件再发送报告）| 多个 Agent 协作 |

## 工作流程
1. 理解用户请求
2. 识别需要调用的子 Agent
3. 调用对应的子 Agent 工具（使用 call_*_agent_tool）
4. 汇总结果返回给用户

## 重要约束
- 每个子 Agent 工具接受一个 task 参数，描述要执行的具体任务
- 例如：call_file_agent_tool(task="读取 ./data/documents/report.txt")
- 如果需要多个 Agent 协作，按顺序调用，每个 Agent 完成后再调用下一个
- 不要尝试直接调用底层工具（read_local_file, query_sales_data 等），必须通过子 Agent 工具调用
"""


# ========== 3. 权限感知的工具节点（带审计日志）==========
class PermissionAwareToolNode(ToolNode):
    """在执行工具前检查用户权限，并记录审计日志"""
    
    def __init__(self, tools, get_user_role):
        super().__init__(tools)
        self.get_user_role = get_user_role

    def __call__(self, state: SupervisorState):
        # 获取最后一条消息中的工具调用
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        user_role = self.get_user_role(state)
        results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_call_id = tool_call.get("id", "")
            
            # 检查是否是子 Agent 工具
            if tool_name in ["call_file_agent_tool", "call_db_agent_tool", "call_email_agent_tool"]:
                # 子 Agent 工具需要特殊处理：直接执行
                # 注意：这里直接调用底层函数，不走 LLM 工具调用流程
                result = self._execute_sub_agent(tool_call)
                results.append(result)
            elif has_permission(user_role, tool_name):
                # 权限通过，正常执行工具
                result = self._execute_tool(tool_call)
                # 记录成功的审计日志
                log_audit(
                    user_id=user_role,
                    tool_name=tool_name,
                    params=tool_call.get("args", {}),
                    result=str(result.content)[:200] if hasattr(result, "content") else str(result)[:200],
                    status="success"
                )
                results.append(result)
            else:
                # 权限不足，返回错误消息
                error_msg = ToolMessage(
                    content=f"❌ 权限不足：角色 '{user_role}' 无权调用工具 '{tool_name}'",
                    tool_call_id=tool_call_id
                )
                log_audit(
                    user_id=user_role,
                    tool_name=tool_name,
                    params=tool_call.get("args", {}),
                    result=f"权限不足：角色 '{user_role}' 无权调用工具 '{tool_name}'",
                    status="denied"
                )
                results.append(error_msg)
        
        return {"messages": results}

    def _execute_sub_agent(self, tool_call):
        """执行子 Agent 调用"""
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call["id"]
        
        task = tool_args.get("task", "")
        
        try:
            if tool_name == "call_file_agent_tool":
                result = call_file_agent(task)
            elif tool_name == "call_db_agent_tool":
                result = call_db_agent(task)
            elif tool_name == "call_email_agent_tool":
                result = call_email_agent(task)
            else:
                result = f"未知子 Agent: {tool_name}"
        except Exception as e:
            result = f"子 Agent 执行失败: {str(e)}"
        
        return ToolMessage(content=str(result), tool_call_id=tool_call_id)

    def _execute_tool(self, tool_call):
        """执行单个工具"""
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_map = {tool.name: tool for tool in self.tools}
        tool = tool_map.get(tool_name)
        if tool:
            try:
                result = tool.invoke(tool_args)
            except Exception as e:
                result = f"工具执行异常: {str(e)}"
            return ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        else:
            return ToolMessage(content=f"未知工具: {tool_name}", tool_call_id=tool_call["id"])


# ========== 4. 获取用户角色的函数 ==========
def get_current_user_role(state: SupervisorState) -> str:
    """从 state 中获取用户角色，默认为 'user'"""
    return state.get("user_role", Role.USER)


# ========== 5. 初始化 Supervisor 模型 ==========
supervisor_model = ChatOpenAI(
    model=config.MODEL_NAME,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
    timeout=120
)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)


# ========== 6. Supervisor Agent 节点函数 ==========
def supervisor_agent_node(state: SupervisorState):
    """Supervisor 模型调用节点"""
    messages = state["messages"]
    
    # 添加系统提示（首次调用时）
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        system_prompt = {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}
        messages = [system_prompt] + list(messages)
    
    response = supervisor_model_with_tools.invoke(messages)
    return {"messages": [response]}


# ========== 7. 条件边函数 ==========
def supervisor_should_continue(state: SupervisorState) -> Literal["tools", "__end__"]:
    """判断是否继续调用工具（包括子 Agent 工具）"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 8. 构建 Supervisor 图 ==========
supervisor_workflow = StateGraph(SupervisorState)
supervisor_workflow.add_node("agent", supervisor_agent_node)

# 使用权限感知的工具节点
tool_node = PermissionAwareToolNode(supervisor_tools, get_current_user_role)
supervisor_workflow.add_node("tools", tool_node)

supervisor_workflow.add_edge(START, "agent")
supervisor_workflow.add_conditional_edges("agent", supervisor_should_continue, ["tools", END])
supervisor_workflow.add_edge("tools", "agent")

memory = MemorySaver()

# 不使用 interrupt_before，因为子 Agent 调用已经封装了完整逻辑
# 如果需要对特定操作（如发送邮件）进行审批，可以后续扩展
supervisor_app = supervisor_workflow.compile(
    checkpointer=memory,
)


# ========== 9. 辅助函数 ==========
def needs_approval(state: SupervisorState) -> bool:
    """检查是否包含需要审批的工具调用"""
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return False
    
    APPROVAL_REQUIRED_TOOLS = {"call_email_agent_tool"}  # 需要审批的工具
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        if tool_name in APPROVAL_REQUIRED_TOOLS:
            return True
    return False


# ========== 10. 子 Agent 状态传递说明 ==========
"""
## 子 Agent 之间的状态传递方式

### 方式1：共享 Supervisor 状态（推荐）
- Supervisor 维护一个共享的 state，其中包含 task_results 字典
- 每个子 Agent 执行完成后，结果存储在 task_results 中
- 后续 Agent 可以读取前面 Agent 的结果

### 方式2：独立上下文（通过 task 参数）
- 每个子 Agent 接收一个独立的 task 参数
- task 参数中可以包含前面 Agent 的结果信息
- 例如："根据文件 ./report.txt 中的销售数据，发送邮件给 boss@example.com"

### 实现示例（任务链）：
```
# 1. Supervisor 调用 FileAgent 读取文件
file_result = call_file_agent("读取 ./data/documents/report.txt")

# 2. 将结果传递给 DBAgent 进行分析（如果需要）
db_task = f"分析以下数据：{file_result}"
db_result = call_db_agent(db_task)

# 3. 将最终结果通过 EmailAgent 发送邮件
email_task = f"发送邮件给 manager@example.com，主题：报告，内容：{db_result}"
email_result = call_email_agent(email_task)
```

### State 结构：
{
    "messages": [...],           # 对话历史
    "user_role": "user",        # 用户角色
    "target_agents": ["file"],  # 本次任务涉及的 Agent
    "task_results": {         # 各 Agent 的执行结果
        "file": "文件内容...",
        "db": "查询结果...",
        "email": "发送成功"
    },
    "pending_tasks": [...]      # 待执行的任务队列
}
"""