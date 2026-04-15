import os
import sqlite3
from langgraph.errors import GraphInterrupt
from langchain_core.messages import ToolMessage

from agents.supervisor import supervisor_app, needs_approval
from agents.file_agent import call_file_agent
from agents.db_agent import call_db_agent
from agents.email_agent import call_email_agent
from security.permissions import Role


def init_database():
    """初始化示例数据库（如果不存在）"""
    os.makedirs("./data", exist_ok=True)
    db_path = "./data/business.db"
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                product TEXT,
                amount REAL,
                date TEXT
            )
        """)
        # 插入示例数据
        conn.execute("INSERT INTO sales (product, amount, date) VALUES ('笔记本电脑', 12999, '2025-12-01')")
        conn.execute("INSERT INTO sales (product, amount, date) VALUES ('无线鼠标', 299, '2025-12-01')")
        conn.commit()
        conn.close()
        print("✅ 示例数据库已创建")
    else:
        print("✅ 数据库已存在")


def handle_approval_prompt(state: dict, config: dict) -> bool:
    """
    处理审批提示
    
    Args:
        state: 中断时的状态
        config: LangGraph 配置
    
    Returns:
        bool: True 表示用户确认，False 表示用户拒绝
    """
    messages = state.get("messages", [])
    if not messages:
        return False
    
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return False
    
    # 提取 call_email_agent_tool 的调用信息
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        if tool_name == "call_email_agent_tool":
            print("\n" + "=" * 60)
            print("📧 待审批操作：发送邮件")
            print("=" * 60)
            print(f"  任务描述: {tool_args.get('task', 'N/A')}")
            print("=" * 60)
            
            while True:
                confirm = input("\n是否确认发送邮件？(y/n): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    return True
                elif confirm in ['n', 'no', '否']:
                    return False
                else:
                    print("请输入 y 或 n")
    
    return True  # 如果不是邮件，默认通过


def run_cli(user_role: str = Role.USER):
    """
    运行 CLI 交互界面（Supervisor + 子 Agent 模式）
    
    Args:
        user_role: 用户角色，默认为 "user"，可设置为 "admin"
    """
    print("=" * 60)
    print("🤖 企业内部智能助手（Supervisor + 子 Agent 架构）")
    print("=" * 60)
    print(f"\n当前用户角色: {user_role}")
    print("\n支持的操作示例：")
    print("  📄 读取文件：'读取 ./data/documents/report.txt'")
    print("  📂 列出文件：'列出 ./data/documents 目录下的文件'")
    print("  📊 查询销售：'查询最近10条销售记录'")
    print("  📧 发送邮件：'发送邮件给 test@example.com，主题：测试，内容：Hello'")
    print("  🔄 复杂任务：'读取销售报告，然后发邮件给 manager@example.com'")
    print("  🚪 退出：输入 'exit' 或 'quit'\n")

    # 会话配置
    config = {
        "configurable": {
            "thread_id": "user_session_1",
        }
    }

    while True:
        try:
            user_input = input("👤 用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 再见！")
                break
            if not user_input:
                continue

            # 将 user_role 合并到消息中
            input_state = {
                "messages": [("user", user_input)],
                "user_role": user_role,
                "target_agents": [],
                "task_results": {},
                "pending_tasks": []
            }

            # 流式输出
            for event in supervisor_app.stream(
                input_state,
                config,
                stream_mode="values"
            ):
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"\n🤖 助手: {last_msg.content}")

        except GraphInterrupt as e:
            # 图被中断
            current_state = supervisor_app.get_state(config)
            
            if current_state and needs_approval(current_state):
                approved = handle_approval_prompt(current_state, config)
                
                if approved:
                    print("\n⏳ 继续执行...")
                    for event in supervisor_app.stream(
                        None,
                        config,
                        stream_mode="values"
                    ):
                        if "messages" in event and event["messages"]:
                            last_msg = event["messages"][-1]
                            if hasattr(last_msg, "content") and last_msg.content:
                                print(f"\n🤖 助手: {last_msg.content}")
                else:
                    print("\n❌ 操作已取消")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 系统错误：{str(e)}")


def demo_direct_sub_agent_calls():
    """演示直接调用子 Agent（不通过 Supervisor）"""
    print("\n" + "=" * 60)
    print("📝 演示：直接调用子 Agent")
    print("=" * 60)
    
    # 1. 直接调用 FileAgent
    print("\n--- 1. 直接调用 FileAgent ---")
    result = call_file_agent("读取 ./data/documents/report.txt")
    print(f"结果: {result}")
    
    # 2. 直接调用 DBAgent
    print("\n--- 2. 直接调用 DBAgent ---")
    result = call_db_agent("查询最近5条销售记录")
    print(f"结果: {result}")
    
    # 3. 演示任务链（FileAgent -> DBAgent -> EmailAgent）
    print("\n--- 3. 任务链协作示例 ---")
    # Step 1: 读取文件
    file_result = call_file_agent("列出 ./data/documents 目录下的文件")
    print(f"[FileAgent] {file_result}")
    
    # Step 2: 基于文件结果查询数据库（可以在 task 中嵌入前面结果）
    db_task = f"查询销售数据（已读取文件内容：{file_result[:50]}...）"
    db_result = call_db_agent(db_task)
    print(f"[DBAgent] {db_result}")
    
    # Step 3: 发送邮件（可以嵌入前面结果）
    email_task = f"发送邮件给 test@example.com，主题：数据报告，摘要：{db_result[:100]}..."
    email_result = call_email_agent(email_task)
    print(f"[EmailAgent] {email_result}")


def main():
    """主函数，支持选择用户角色和演示模式"""
    print("=" * 60)
    print("🤖 选择运行模式")
    print("=" * 60)
    print("1. 交互模式 (Supervisor + 子 Agent)")
    print("2. 演示模式 (直接调用子 Agent)")
    print("3. 退出")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 运行交互模式
        print("\n" + "=" * 60)
        print("🤖 选择用户角色")
        print("=" * 60)
        print("1. 普通用户 (user) - 默认权限")
        print("2. 管理员 (admin) - 更高权限")
        
        role_choice = input("\n请选择 (1/2): ").strip()
        
        if role_choice == "2":
            user_role = Role.ADMIN
            print("\n✅ 已选择管理员角色")
        else:
            user_role = Role.USER
            print("\n✅ 已选择普通用户角色")
        
        run_cli(user_role=user_role)
    
    elif choice == "2":
        # 运行演示模式
        demo_direct_sub_agent_calls()
    
    else:
        print("👋 再见！")


if __name__ == "__main__":
    init_database()
    # 确保允许的目录存在
    allowed_dir = os.getenv("ALLOWED_FILE_DIR", "./data/documents")
    os.makedirs(allowed_dir, exist_ok=True)
    main()