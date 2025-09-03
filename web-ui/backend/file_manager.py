import os
import uuid
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import aiofiles
import asyncio
from translations import t

class FileManager:
    def __init__(self, storage_dir: str = "uploads", metadata_file: str = "file_metadata.json"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 元数据文件路径
        self.metadata_file = Path(metadata_file)
        self.metadata_lock = asyncio.Lock()
        
        # 支持的文件类型
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.md', '.doc', '.json'}
        self.supported_mime_types = {
            'text/plain', 'application/pdf', 
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword', 'text/markdown', 'application/json'
        }
        
        # 初始化元数据文件
        self._init_metadata_file()
    
    def _init_metadata_file(self):
        """初始化元数据文件"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_metadata(self, metadata: Dict):
        """保存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def generate_file_id(self) -> str:
        """生成唯一的文件ID"""
        return str(uuid.uuid4())
    
    def get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_supported_file(self, filename: str, mime_type: str = None) -> bool:
        """检查文件是否支持"""
        ext = Path(filename).suffix.lower()
        return ext in self.supported_extensions or (mime_type and mime_type in self.supported_mime_types)
    
    def generate_database_name(self, filename: str) -> str:
        """根据文件名前5个字符生成数据库名"""
        # 去除文件扩展名
        name_without_ext = Path(filename).stem
        # 只保留字母、数字和中文字符，移除特殊字符
        clean_name = re.sub(r'[^\w\u4e00-\u9fff]', '', name_without_ext)
        # 取前5个字符
        db_name = clean_name[:5]
        # 如果少于5个字符，用原文件名
        if len(db_name) < 1:
            db_name = "default"
        return db_name
    
    async def save_uploaded_file(self, file_content: bytes, original_filename: str) -> Dict:
        """保存上传的文件"""
        try:
            # 根据文件扩展名推断MIME类型
            ext = Path(original_filename).suffix.lower()
            mime_type_map = {
                '.txt': 'text/plain',
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.md': 'text/markdown',
                '.json': 'application/json'
            }
            mime_type = mime_type_map.get(ext, 'application/octet-stream')
            
            if not self.is_supported_file(original_filename, mime_type):
                raise ValueError(t('unsupported_file_type', filename=original_filename))
            
            # 生成数据库名
            database_name = self.generate_database_name(original_filename)
            
            # 生成文件ID和存储路径
            file_id = self.generate_file_id()
            file_ext = Path(original_filename).suffix
            filename = f"{file_id}{file_ext}"
            file_path = self.storage_dir / filename
            
            # 异步保存文件
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            # 计算文件大小和哈希
            file_size = len(file_content)
            file_hash = self.get_file_hash(str(file_path))
            
            # 保存到元数据文件
            async with self.metadata_lock:
                metadata = self._load_metadata()
                
                file_record = {
                    "file_id": file_id,
                    "filename": filename,
                    "original_filename": original_filename,
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "file_type": file_ext,
                    "mime_type": mime_type,
                    "database_name": database_name,
                    "upload_time": datetime.utcnow().isoformat(),
                    "status": "uploaded",
                    "processed_time": None,
                    "error_message": None,
                    "file_metadata": {"hash": file_hash}
                }
                
                metadata[file_id] = file_record
                self._save_metadata(metadata)
                
                return {
                    "file_id": file_id,
                    "filename": original_filename,
                    "file_path": str(file_path),
                    "database_name": database_name,
                    "file_size": file_size,
                    "status": "uploaded",
                    "upload_time": file_record["upload_time"]
                }
                
        except Exception as e:
            # 如果保存失败，删除已创建的文件
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            raise e
    
    def get_all_files(self) -> List[Dict]:
        """获取所有文件列表"""
        metadata = self._load_metadata()
        files = []
        
        for file_id, file_record in metadata.items():
            files.append({
                "file_id": file_record["file_id"],
                "filename": file_record["original_filename"],
                "database_name": file_record["database_name"],
                "file_size": file_record["file_size"],
                "file_type": file_record["file_type"],
                "mime_type": file_record["mime_type"],
                "upload_time": file_record["upload_time"],
                "status": file_record["status"],
                "processed_time": file_record.get("processed_time"),
                "error_message": file_record.get("error_message")
            })
        
        # 按上传时间降序排序
        files.sort(key=lambda x: x["upload_time"], reverse=True)
        return files
    
    def get_file_by_id(self, file_id: str) -> Optional[Dict]:
        """根据ID获取文件信息"""
        metadata = self._load_metadata()
        file_record = metadata.get(file_id)
        
        if not file_record:
            return None
        
        return {
            "file_id": file_record["file_id"],
            "filename": file_record["original_filename"],
            "database_name": file_record["database_name"],
            "file_path": file_record["file_path"],
            "file_size": file_record["file_size"],
            "file_type": file_record["file_type"],
            "mime_type": file_record["mime_type"],
            "upload_time": file_record["upload_time"],
            "status": file_record["status"],
            "processed_time": file_record.get("processed_time"),
            "error_message": file_record.get("error_message")
        }
    
    def update_file_status(self, file_id: str, status: str, error_message: str = None):
        """更新文件状态"""
        metadata = self._load_metadata()
        
        if file_id in metadata:
            metadata[file_id]["status"] = status
            if error_message:
                metadata[file_id]["error_message"] = error_message
            if status == "embedded":
                metadata[file_id]["processed_time"] = datetime.utcnow().isoformat()
            
            self._save_metadata(metadata)
    
    def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        metadata = self._load_metadata()
        
        if file_id not in metadata:
            return False
        
        file_record = metadata[file_id]
        
        # 删除文件
        file_path = Path(file_record["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # 删除元数据记录
        del metadata[file_id]
        self._save_metadata(metadata)
        
        return True
    
    async def read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(t('file_not_found', path=str(file_path)))
        
        # 根据文件类型选择不同的读取方式
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx']:
            return self._read_docx(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        elif file_path.suffix.lower() == '.json':
            return await self._read_json(file_path)
        else:
            raise ValueError(t('unsupported_file_type', filename=file_path.suffix))
    
    def _read_pdf(self, file_path: Path) -> str:
        """读取PDF文件内容"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise ValueError(t('pdf_read_failed', error=str(e)))
    
    def _read_docx(self, file_path: Path) -> str:
        """读取DOCX文件内容"""
        try:
            import docx2txt
            return docx2txt.process(str(file_path))
        except Exception as e:
            raise ValueError(t('docx_read_failed', error=str(e)))
    
    async def _read_json(self, file_path: Path) -> str:
        """读取JSON文件内容并格式化为LLM友好的文本"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                json_content = await f.read()
                json_data = json.loads(json_content)
            
            # 检测是否为transcription JSON
            if 'transcriptionId' in json_data or 'transcriptionTitle' in json_data:
                return self._format_transcription_json(json_data)
            else:
                return self._format_generic_json(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to read JSON file: {str(e)}")
    
    def _format_transcription_json(self, json_data: dict) -> str:
        """格式化transcription JSON为LLM优化的文本"""
        # 格式化sections
        sections_text = '\n'.join([
            f"- {s['title']} (confidence: {s.get('confidenceScore', 'N/A')}): {s['content']}"
            for s in json_data.get('sections', [])
        ]) if json_data.get('sections') else 'No sections available'
        
        # 格式化relationships
        relationships_text = '\n'.join([
            f"- {r['connection']}" 
            for r in json_data.get('relationships', [])
        ]) if json_data.get('relationships') else 'None identified'
        
        # 格式化action items
        actions_text = '\n'.join([
            f"- {task}" 
            for task in json_data.get('actionableTasks', [])
        ]) if json_data.get('actionableTasks') else 'None identified'
        
        # 格式化low confidence items
        low_conf_text = '\n'.join([
            f"- {item['item']} ({item.get('context', 'N/A')}): {item['reason']}"
            for item in json_data.get('lowConfidenceItems', [])
        ]) if json_data.get('lowConfidenceItems') else ''
        
        # 构建最终文本
        result = f"""Document: {json_data.get('fileName', 'Unknown')} - {json_data.get('transcriptionTitle', 'Untitled')}
Date: {json_data.get('updatedAt', 'Unknown date')}

SUMMARY:
{json_data.get('summary', 'No summary available')}

MAIN CONTENT:
{sections_text}

KEY RELATIONSHIPS:
{relationships_text}

ACTION ITEMS:
{actions_text}"""
        
        if low_conf_text:
            result += f"\n\nLOW CONFIDENCE ITEMS:\n{low_conf_text}"
        
        # 如果有原始文本且sections为空，添加原始文本
        if json_data.get('rawText') and not json_data.get('sections'):
            result += f"\n\nRAW TRANSCRIPT:\n{json_data['rawText']}"
        
        return result
    
    def _format_generic_json(self, json_data: dict, max_depth: int = 3) -> str:
        """格式化通用JSON为可读的叙述性文本"""
        def format_value(value, depth=0):
            if depth > max_depth:
                return "..."
            
            if isinstance(value, dict):
                items = []
                for k, v in value.items():
                    formatted = format_value(v, depth + 1)
                    items.append(f"{k}: {formatted}")
                return "; ".join(items) if items else "{}"
            elif isinstance(value, list):
                if not value:
                    return "[]"
                if len(value) > 5:
                    return f"[{len(value)} items]"
                return ", ".join([str(format_value(v, depth + 1)) for v in value])
            elif isinstance(value, str):
                # 对长字符串进行截断
                if len(value) > 200:
                    return value[:200] + "..."
                return value
            else:
                return str(value)
        
        lines = ["JSON Document Content:"]
        for key, value in json_data.items():
            formatted_value = format_value(value)
            lines.append(f"{key}: {formatted_value}")
        
        return "\n".join(lines)

# 全局文件管理器实例
file_manager = FileManager() 