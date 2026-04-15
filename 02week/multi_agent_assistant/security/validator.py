# security/validator.py
import re
import os

def validate_file_path(file_path: str, allowed_dir: str) -> bool:
    """防止路径遍历攻击，确保文件在允许目录内"""
    abs_path = os.path.abspath(file_path)
    abs_allowed = os.path.abspath(allowed_dir)
    return abs_path.startswith(abs_allowed)

def sanitize_sql(sql: str) -> bool:
    """只允许SELECT，拦截危险关键字"""
    sql_upper = sql.strip().upper()
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "EXEC", "UNION"]
    for kw in dangerous:
        if kw in sql_upper:
            return False
    # 防止多语句
    if ";" in sql and not sql.rstrip().endswith(";"):
        return False
    return True

def validate_email(email: str) -> bool:
    """简单邮箱格式校验"""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None