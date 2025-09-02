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
        "create_new_hyperrag_instance": "Creating new HyperRAG instance, database: {database}",
        "hyperrag_instance_created": "HyperRAG instance created, database: {database}",
        "use_existing_hyperrag_instance": "Using existing HyperRAG instance, database: {database}",
        "reading_file_content": "Reading file content...",
        "saving_file_locally": "Saving file locally...",
        "file_upload_complete": "File upload complete, success: {success}/{total}",
        "update_file_status_processing": "Updating file status to processing...",
        "get_file_info": "Getting file information...",
        "start_document_embedding": "Starting document embedding...",
        "document_embedding_complete": "Document embedding complete",
        "document_embedding_summary": "Document embedding complete, success: {success}/{total}",
        "log_redirect_enabled": "Log redirect enabled",
        "log_redirect_disabled": "Log redirect disabled",
        "hyperrag_log_config_complete": "HyperRAG log configuration complete",
        "initializing_hyperrag_instance": "Initializing HyperRAG instance...",
        "hyperrag_instance_initialized": "HyperRAG instance initialized",
        "reading_file_content_progress": "Reading file content...",
        "document_embedding_processing": "Starting document embedding processing...",
        "document_embedding_wait": "This process may take some time, please be patient...",
        "upload_file": "Uploading file {current}/{total}: {filename}",
        "file_size": "File size: {size} bytes",
        "file_content_read_complete": "File content read complete, actual size: {size} bytes",
        "file_save_success": "File saved successfully: {filename}",
        "file_id": "File ID: {file_id}",
        "save_path": "Save path: {path}",
        "database": "Database: {database}",
        "file_upload_failed": "File upload failed: {filename}, Error: {error}",
        "file_delete_success": "File deleted successfully",
        "start_document_embedding_batch": "Starting document embedding, file count: {count}",
        "config_params": "Configuration parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}",
        "processing_file": "Processing file {current}/{total}: {file_id}",
        "file_info": "File info: {filename} ({size} bytes)",
        "target_database": "Target database: {database}",
        "content_length": "Content length: {length} characters",
        "file_embedded_success": "File {filename} embedded successfully",
        "file_embedding_failed": "File embedding failed: {file_id}, Error: {error}",
        "batch_embedding_failed": "Batch embedding failed: {error}",
        "using_default_database": "Using default database: {database}",
        "hyperrag_working_dir": "HyperRAG working directory: {dir}",
        "hyperrag_unavailable": "HyperRAG unavailable",
        "processing_file_message": "Processing file {current}/{total}",
        "getting_file_info": "Getting file information: {file_id}",
        "file_info_success": "File information retrieved successfully:",
        "filename": "Filename: {filename}",
        "upload_time": "Upload time: {time}",
        "starting_file_processing": "Starting file processing: {filename} ({size} bytes), using database: {database}",
        "initializing_hyperrag_with_db": "Initializing HyperRAG instance, database: {database}",
        "hyperrag_initialized_with_db": "HyperRAG instance initialized, using database: {database}",
        "reading_file_message": "Reading file: {filename} (database: {database})",
        "file_read_complete": "File read complete, content length: {length} characters",
        "content_preview": "Content preview: {preview}",
        "embedding_document_message": "Embedding document: {filename} (database: {database})",
        "document_chunking": "Document chunking in progress...",
        "api_test_success": "API connection test successful",
        "anthropic_api_test_success": "Anthropic API connection test successful",
        "api_test_failed": "API connection test failed: {error}",
        "database_test_success": "Database connection test successful",
        "database_test_failed": "Database connection test failed: {error}",
        "llm_call_start": "Starting LLM call, prompt length: {length} characters",
        "system_prompt_length": "System prompt length: {length} characters",
        "using_model": "Using model: {model}, API URL: {url}",
        "llm_call_complete": "LLM call complete, response length: {length} characters",
        "llm_call_failed": "LLM call failed: {error}",
        "text_embedding_start": "Starting text embedding, text count: {count}",
        "text_total_length": "Total text length: {length} characters",
        "using_embedding_model": "Using embedding model: {model}",
        "text_embedding_complete": "Text embedding complete, embedding dimensions: {dimensions}",
        "text_embedding_failed": "Text embedding failed: {error}",
        "file_upload_start": "Starting file upload, file count: {count}",
        "document_embed_start": "Starting document embedding, file count: {count}",
        "unknown": "unknown",
        "enable_log_redirect": "Enable log redirect",
        "disable_log_redirect": "Disable log redirect",
        "send_progress_update": "Send progress updates to all connected clients",
        "send_log_message": "Send log messages to all connected clients",
        "setup_comprehensive_logging": "Setup comprehensive logging configuration",
        "configure_hyperrag_logging": "Configure HyperRAG related detailed log output",
        "cannot_import_hyperrag_module": "Cannot import HyperRAG module for log configuration: {error}",
        "hyperrag_log_config_failed": "HyperRAG log configuration failed: {error}",
        "document_embedding_started": "Document embedding processing has started",
        "async_file_processing": "Asynchronously process file embedding and send progress updates",
        "batch_file_embed_start": "Starting batch file embedding task",
        "total_files": "Total files: {count}",
        "processing_embed_task": "Processing embedding task for {count} files",
        "config_parameters": "Configuration parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}",
        "processing_file_num": "Processing file {current}/{total}",
        "file_id_label": "File ID: {file_id}",
        "error_label": "Error: {error}",
        "file_embed_complete": "File embedding complete: {filename} (database: {database})",
        "file_process_success": "File {filename} processed successfully!",
        "file_process_failed": "File processing failed: {file_id}, Error: {error}",
        "batch_document_complete": "Batch document processing complete!",
        "success_process": "Successfully processed: {count}",
        "failed_process": "Failed to process: {count}",
        "success_rate": "Success rate: {rate}%",
        "all_documents_complete": "All documents processed! Total: {total} files, Success: {success}, Failed: {failed}",
        "all_complete_message": "All documents processed (Success: {success}, Failed: {failed})",
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
        "create_new_hyperrag_instance": "创建新的HyperRAG实例，数据库: {database}",
        "hyperrag_instance_created": "HyperRAG实例创建完成，数据库: {database}",
        "use_existing_hyperrag_instance": "使用现有HyperRAG实例，数据库: {database}",
        "reading_file_content": "正在读取文件内容...",
        "saving_file_locally": "正在保存文件到本地...",
        "file_upload_complete": "文件上传完成，成功: {success}/{total}",
        "update_file_status_processing": "更新文件状态为处理中...",
        "get_file_info": "获取文件信息...",
        "start_document_embedding": "开始文档嵌入...",
        "document_embedding_complete": "文档嵌入完成",
        "document_embedding_summary": "文档嵌入完成，成功: {success}/{total}",
        "log_redirect_enabled": "日志重定向已启用",
        "log_redirect_disabled": "日志重定向已禁用",
        "hyperrag_log_config_complete": "HyperRAG日志配置完成",
        "initializing_hyperrag_instance": "正在初始化 HyperRAG 实例...",
        "hyperrag_instance_initialized": "HyperRAG 实例初始化完成",
        "reading_file_content_progress": "正在读取文件内容...",
        "document_embedding_processing": "开始文档嵌入处理...",
        "document_embedding_wait": "这个过程可能需要一些时间，请耐心等待...",
        "upload_file": "上传文件 {current}/{total}: {filename}",
        "file_size": "文件大小: {size} bytes",
        "file_content_read_complete": "文件内容读取完成，实际大小: {size} bytes",
        "file_save_success": "文件保存成功: {filename}",
        "file_id": "文件ID: {file_id}",
        "save_path": "保存路径: {path}",
        "database": "数据库: {database}",
        "file_upload_failed": "文件上传失败: {filename}, 错误: {error}",
        "file_delete_success": "文件删除成功",
        "start_document_embedding_batch": "开始文档嵌入，文件数量: {count}",
        "config_params": "配置参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}",
        "processing_file": "处理文件 {current}/{total}: {file_id}",
        "file_info": "文件信息: {filename} ({size} bytes)",
        "target_database": "目标数据库: {database}",
        "content_length": "内容长度: {length} 字符",
        "file_embedded_success": "文件 {filename} 嵌入成功",
        "file_embedding_failed": "文件嵌入失败: {file_id}, 错误: {error}",
        "batch_embedding_failed": "批量嵌入失败: {error}",
        "using_default_database": "使用默认数据库: {database}",
        "hyperrag_working_dir": "HyperRAG工作目录: {dir}",
        "hyperrag_unavailable": "HyperRAG 不可用",
        "processing_file_message": "正在处理文件 {current}/{total}",
        "getting_file_info": "获取文件信息: {file_id}",
        "file_info_success": "文件信息获取成功:",
        "filename": "文件名: {filename}",
        "upload_time": "上传时间: {time}",
        "starting_file_processing": "开始处理文件: {filename} ({size} bytes)，使用数据库: {database}",
        "initializing_hyperrag_with_db": "正在初始化 HyperRAG 实例，数据库: {database}",
        "hyperrag_initialized_with_db": "HyperRAG 实例初始化完成，使用数据库: {database}",
        "reading_file_message": "正在读取文件: {filename} (数据库: {database})",
        "file_read_complete": "文件读取完成，内容长度: {length} 字符",
        "content_preview": "内容预览: {preview}",
        "embedding_document_message": "正在嵌入文档: {filename} (数据库: {database})",
        "document_chunking": "正在进行文档分块...",
        "api_test_success": "API连接测试成功",
        "anthropic_api_test_success": "Anthropic API连接测试成功",
        "api_test_failed": "API连接测试失败: {error}",
        "database_test_success": "数据库连接测试成功",
        "database_test_failed": "数据库连接测试失败: {error}",
        "llm_call_start": "开始LLM调用，prompt长度: {length} 字符",
        "system_prompt_length": "系统提示词长度: {length} 字符",
        "using_model": "使用模型: {model}, API地址: {url}",
        "llm_call_complete": "LLM调用完成，响应长度: {length} 字符",
        "llm_call_failed": "LLM调用失败: {error}",
        "text_embedding_start": "开始文本嵌入，文本数量: {count}",
        "text_total_length": "文本总长度: {length} 字符",
        "using_embedding_model": "使用嵌入模型: {model}",
        "text_embedding_complete": "文本嵌入完成，嵌入维度: {dimensions}",
        "text_embedding_failed": "文本嵌入失败: {error}",
        "file_upload_start": "开始文件上传，文件数量: {count}",
        "document_embed_start": "开始文档嵌入，文件数量: {count}",
        "unknown": "未知",
        "enable_log_redirect": "启用日志重定向",
        "disable_log_redirect": "禁用日志重定向",
        "send_progress_update": "发送进度更新到所有连接的客户端",
        "send_log_message": "发送日志消息到所有连接的客户端",
        "setup_comprehensive_logging": "设置全面的日志配置",
        "configure_hyperrag_logging": "配置HyperRAG相关的详细日志输出",
        "cannot_import_hyperrag_module": "无法导入HyperRAG模块进行日志配置: {error}",
        "hyperrag_log_config_failed": "HyperRAG日志配置失败: {error}",
        "document_embedding_started": "文档嵌入处理已开始",
        "async_file_processing": "异步处理文件嵌入并发送进度更新",
        "batch_file_embed_start": "开始批量文件嵌入任务",
        "total_files": "文件总数: {count}",
        "processing_embed_task": "开始处理 {count} 个文件的嵌入任务",
        "config_parameters": "配置参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}",
        "processing_file_num": "处理文件 {current}/{total}",
        "file_id_label": "文件ID: {file_id}",
        "error_label": "错误: {error}",
        "file_embed_complete": "文件嵌入完成: {filename} (数据库: {database})",
        "file_process_success": "文件 {filename} 处理成功！",
        "file_process_failed": "文件处理失败: {file_id}, 错误: {error}",
        "batch_document_complete": "批量文档处理完成！",
        "success_process": "成功处理: {count}",
        "failed_process": "处理失败: {count}",
        "success_rate": "成功率: {rate}%",
        "all_documents_complete": "所有文档处理完成！总计: {total} 个文件，成功: {success}，失败: {failed}",
        "all_complete_message": "所有文档处理完成 (成功: {success}, 失败: {failed})",
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