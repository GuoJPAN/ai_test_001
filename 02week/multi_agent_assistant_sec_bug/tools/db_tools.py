import sqlite3
import os
from langchain_core.tools import tool
from security.validator import sanitize_sql
from security.audit import log_audit
from config import config

# 确保 data 目录存在
DB_PATH = "./data/business.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@tool
def query_sales_data(limit: int = 10) -> str:
    """查询销售记录，limit最多50条"""
    user_id = "unknown"
    limit = min(limit, config.MAX_QUERY_RESULTS)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT product, amount, date FROM sales ORDER BY date DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "暂无销售数据"
        result = "\n".join([f"产品：{r[0]} | 金额：{r[1]}元 | 日期：{r[2]}" for r in rows])
        log_audit(user_id, "query_sales_data", {"limit": limit}, "success", "success")
        return f"📊 最新{len(rows)}条销售记录：\n{result}"
    except Exception as e:
        log_audit(user_id, "query_sales_data", {"limit": limit}, str(e), "error")
        return f"❌ 查询失败：{str(e)}"

@tool
def execute_safe_query(sql: str) -> str:
    """执行只读SQL查询（仅SELECT）"""
    user_id = "unknown"
    if not sanitize_sql(sql):
        msg = "❌ 安全拦截：仅允许只读SELECT查询"
        log_audit(user_id, "execute_safe_query", {"sql": sql}, msg, "denied")
        return msg
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        result = "\n".join(str(row) for row in rows[:20])
        log_audit(user_id, "execute_safe_query", {"sql": sql}, "success", "success")
        return result if result else "查询无结果"
    except Exception as e:
        log_audit(user_id, "execute_safe_query", {"sql": sql}, str(e), "error")
        return f"❌ 查询失败：{str(e)}"