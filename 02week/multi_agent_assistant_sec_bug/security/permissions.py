class Role:
    USER = "user"
    ADMIN = "admin"

PERMISSIONS = {
    Role.USER: ["read_local_file", "list_files", "query_sales_data", "send_email"],
    Role.ADMIN: ["read_local_file", "list_files", "query_sales_data", "send_email", "delete_file"]
}

def has_permission(role: str, tool_name: str) -> bool:
    return tool_name in PERMISSIONS.get(role, [])