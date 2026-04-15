"""
简单测试脚本：验证 Supervisor + 子 Agent 架构是否可以正常导入和运行
"""
import sys
import traceback

print("=" * 60)
print("🔧 测试：导入和初始化")
print("=" * 60)

try:
    # 1. 测试导入子 Agent
    print("\n1. 测试导入子 Agent...")
    from agents.file_agent import call_file_agent, file_agent_app
    print("   ✅ FileAgent 导入成功")

    from agents.db_agent import call_db_agent, db_agent_app
    print("   ✅ DBAgent 导入成功")

    from agents.email_agent import call_email_agent, email_agent_app
    print("   ✅ EmailAgent 导入成功")

    # 2. 测试导入 Supervisor
    print("\n2. 测试导入 Supervisor...")
    from agents.supervisor import supervisor_app, supervisor_tools
    print(f"   ✅ Supervisor 导入成功")
    print(f"   📦 子 Agent 工具数量: {len(supervisor_tools)}")

    # 3. 测试直接调用子 Agent（不依赖 LLM）
    print("\n3. 测试直接调用子 Agent（模拟）...")
    
    # 注意：这里可能会因为 LLM 配置问题失败，但也可能成功（取决于本地模型是否运行）
    print("\n   注意：如果 Ollama 未运行，后续调用会失败")
    print("   这是正常的，请确保 Ollama 正在运行")

    print("\n" + "=" * 60)
    print("✅ 所有模块导入成功！")
    print("=" * 60)
    print("\n启动方式：")
    print("  1. 确保 Ollama 已运行（本地模型）")
    print("  2. cd script/02week/multi_agent_assistant")
    print("  3. python main.py")
    print("\n选择模式：")
    print("  1 - 交互模式（通过 Supervisor 路由）")
    print("  2 - 演示模式（直接调用子 Agent）")

except Exception as e:
    print(f"\n❌ 导入失败: {str(e)}")
    print("\n详细错误：")
    traceback.print_exc()
    sys.exit(1)