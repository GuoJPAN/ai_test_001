import os
from langchain_core.tools import tool
from security.validator import validate_file_path
from security.audit import log_audit
from config import config

# 如果已经注释掉安全校验，可以忽略 validate_file_path 的调用，但这里保留结构

def read_text_file(file_path: str) -> str:
    """尝试多种编码读取文本文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return "❌ 无法识别文件编码，请转换为 UTF-8 编码"

def read_docx(file_path: str) -> str:
    """读取 Word 文档，增强错误处理"""
    try:
        from docx import Document
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except ImportError:
        return "❌ 缺少 python-docx 库，请运行 pip install python-docx"
    except Exception as e:
        error_msg = str(e)
        if "no relationship" in error_msg and "officeDocument" in error_msg:
            return "❌ 文件可能已损坏或不是有效的 Word 文档（无法找到文档主体关系）。请检查文件是否可以用 Microsoft Word 正常打开。"
        return f"❌ 读取 Word 文档失败：{error_msg}"

def read_pdf(file_path: str) -> str:
    """读取 PDF 文件（提取文本）"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return '\n'.join(text)
    except ImportError:
        return "❌ 缺少 PyPDF2 库，请运行 pip install PyPDF2"
    except Exception as e:
        return f"❌ 读取 PDF 失败：{str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """读取文件内容，支持 .txt, .md, .docx, .pdf 等格式"""
    user_id = "unknown"
    try:
        # 如果已注释安全校验，可以跳过下面两行，但为了完整性保留
        # if not validate_file_path(file_path, config.ALLOWED_FILE_DIR):
        #     msg = "❌ 安全限制：只能访问指定目录内的文件"
        #     log_audit(user_id, "read_local_file", {"file_path": file_path}, msg, "denied")
        #     return msg

        if not os.path.exists(file_path):
            msg = f"❌ 文件不存在：{file_path}"
            log_audit(user_id, "read_local_file", {"file_path": file_path}, msg, "fail")
            return msg

        # 文件大小限制
        size = os.path.getsize(file_path)
        if size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            msg = f"❌ 文件过大（超过{config.MAX_FILE_SIZE_MB}MB）"
            log_audit(user_id, "read_local_file", {"file_path": file_path}, msg, "fail")
            return msg

        # 根据扩展名选择读取方式
        ext = os.path.splitext(file_path)[1].lower()
        content = ""

        if ext in ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.log', '.csv']:
            content = read_text_file(file_path)
        elif ext == '.docx':
            content = read_docx(file_path)
        elif ext == '.pdf':
            content = read_pdf(file_path)
        else:
            # 尝试作为文本文件读取，失败则提示不支持
            try:
                content = read_text_file(file_path)
                # 如果内容很少且包含大量非打印字符，可能是二进制文件
                if content and sum(c.isprintable() for c in content[:100]) < 50:
                    content = f"❌ 不支持的文件类型：{ext}，文件可能是二进制格式"
            except:
                content = f"❌ 不支持的文件类型：{ext}"

        # 截断过长内容
        if len(content) > 2000:
            content = content[:2000] + "\n...(内容已截断)"
        
        log_audit(user_id, "read_local_file", {"file_path": file_path}, "success", "success")
        return content
    except Exception as e:
        msg = f"❌ 读取失败：{str(e)}"
        log_audit(user_id, "read_local_file", {"file_path": file_path}, msg, "error")
        return msg

@tool
def list_files(directory: str = None) -> str:
    """列出目录下的文件"""
    user_id = "unknown"
    target_dir = directory or config.ALLOWED_FILE_DIR
    # 如果安全校验已注释，可以跳过，但保留路径存在检查
    try:
        if not os.path.exists(target_dir):
            return f"❌ 目录不存在：{target_dir}"
        files = os.listdir(target_dir)
        if not files:
            return "目录为空"
        result = "\n".join([f"📄 {f}" for f in files])
        log_audit(user_id, "list_files", {"directory": target_dir}, "success", "success")
        return result
    except Exception as e:
        return f"❌ 列出文件失败：{str(e)}"