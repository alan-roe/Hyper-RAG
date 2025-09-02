"""
Backend translation system
"""
import json
import os

# Translation dictionaries
translations = {
    "en-US": {
        "hypergraph_suffix": "Hypergraph",
        "unsupported_file_type": "Unsupported file type: {filename}",
        "file_not_found": "File not found: {path}",
        "pdf_read_failed": "Failed to read PDF file: {error}",
        "docx_read_failed": "Failed to read DOCX file: {error}",
        "get_file_list_failed": "Failed to get file list: {error}",
        "file_not_exist": "File does not exist",
        "file_delete_failed": "Failed to delete file: {error}",
        "settings_save_success": "Settings saved successfully",
        "vertex_not_found": "Vertex {vertex_id} not found",
        "hyperedge_not_found": "Hyperedge not found",
        "vertex_already_exists": "Vertex {vertex_id} already exists",
        "hyperedge_already_exists": "Hyperedge already exists",
        "database_not_found": "Database {database} not found",
    },
    "zh-CN": {
        "hypergraph_suffix": "超图",
        "unsupported_file_type": "不支持的文件类型: {filename}",
        "file_not_found": "文件不存在: {path}",
        "pdf_read_failed": "PDF文件读取失败: {error}",
        "docx_read_failed": "DOCX文件读取失败: {error}",
        "get_file_list_failed": "获取文件列表失败: {error}",
        "file_not_exist": "文件不存在",
        "file_delete_failed": "文件删除失败: {error}",
        "settings_save_success": "设置保存成功",
        "vertex_not_found": "未找到顶点 {vertex_id}",
        "hyperedge_not_found": "未找到超边",
        "vertex_already_exists": "顶点 {vertex_id} 已存在",
        "hyperedge_already_exists": "超边已存在",
        "database_not_found": "未找到数据库 {database}",
    }
}

def get_current_language():
    """
    Get the current language setting from settings.json
    """
    settings_file = "settings.json"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return settings.get('language', 'en-US')
        except:
            pass
    return 'en-US'

def t(key: str, **kwargs) -> str:
    """
    Get translated string for the given key
    
    Args:
        key: Translation key
        **kwargs: Format parameters for the translation string
    
    Returns:
        Translated string
    """
    lang = get_current_language()
    
    # Default to English if language not found
    if lang not in translations:
        lang = "en-US"
    
    # Get the translation
    translation = translations.get(lang, {}).get(key, key)
    
    # Format with provided parameters
    if kwargs:
        try:
            return translation.format(**kwargs)
        except:
            return translation
    
    return translation