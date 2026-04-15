import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
from security.validator import validate_email
from security.audit import log_audit
from config import config

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（使用SMTP）"""
    user_id = "unknown"
    if not validate_email(to):
        msg = f"❌ 无效的邮箱地址：{to}"
        log_audit(user_id, "send_email", {"to": to, "subject": subject}, msg, "denied")
        return msg

    if not config.SMTP_SERVER or not config.SENDER_EMAIL:
        log_audit(user_id, "send_email", {"to": to, "subject": subject}, "模拟发送", "success")
        return f"📧 [模拟] 邮件已发送至 {to}（SMTP未配置）"

    try:
        msg = MIMEMultipart()
        msg['From'] = config.SENDER_EMAIL
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 使用 587 端口 + STARTTLS
        server = smtplib.SMTP(config.SMTP_SERVER, 587)
        server.starttls()  # 升级为 TLS 连接
        server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        log_audit(user_id, "send_email", {"to": to, "subject": subject}, "success", "success")
        return f"✅ 邮件已发送至 {to}"
    except Exception as e:
        log_audit(user_id, "send_email", {"to": to, "subject": subject}, str(e), "error")
        return f"❌ 发送失败：{str(e)}"