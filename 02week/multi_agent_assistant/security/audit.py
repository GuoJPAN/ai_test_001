import json
import logging
from datetime import datetime

audit_logger = logging.getLogger("agent_audit")
audit_logger.setLevel(logging.INFO)
handler = logging.FileHandler("agent_audit.log", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

def log_audit(user_id: str, tool_name: str, params: dict, result: str, status: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "tool": tool_name,
        "params": {k: str(v)[:100] for k, v in params.items()},
        "result_preview": str(result)[:200],
        "status": status
    }
    audit_logger.info(json.dumps(log_entry, ensure_ascii=False))