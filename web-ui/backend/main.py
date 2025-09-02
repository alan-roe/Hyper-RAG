from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from db import get_hypergraph, getFrequentVertices, get_vertices, get_hyperedges, get_vertice, get_vertice_neighbor, get_hyperedge_neighbor_server, add_vertex, add_hyperedge, delete_vertex, delete_hyperedge, update_vertex, update_hyperedge, get_hyperedge_detail, db_manager
from file_manager import file_manager
from translations import t
import json
import os
import asyncio
import numpy as np
import logging
import sys
import importlib.util
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from io import StringIO

# 添加 HyperRAG 相关导入
# 若尚不可导入，则向上逐级查找含有 hyperrag 包的目录，并把“其父目录”加到 sys.path
if importlib.util.find_spec("hyperrag") is None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "hyperrag" / "__init__.py").exists():
            sys.path.insert(0, str(parent))  # 注意是父目录，不是 …/hyperrag
            break

try:
    from hyperrag import HyperRAG, QueryParam
    from hyperrag.utils import EmbeddingFunc
    from hyperrag.llm import openai_embedding, openai_complete_if_cache
    HYPERRAG_AVAILABLE = True
except ImportError as e:
    print(f"HyperRAG not available: {e}")
    HYPERRAG_AVAILABLE = False


# 设置文件路径
SETTINGS_FILE = "settings.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hyper-RAG"}


@app.get("/db")
async def db(database: str = None):
    """
    Get complete hypergraph data in JSON format
    获取全部数据json
    """
    try:
        data = get_hypergraph(database)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/vertices")
async def get_vertices_function(database: str = None, page: int = None, page_size: int = None):
    """
    Get list of vertices
    获取vertices列表
    """
    try:
        data = getFrequentVertices(database, page, page_size)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/hyperedges")
async def get_hypergraph_function(database: str = None, page: int = None, page_size: int = None):
    """
    Get list of hyperedges
    获取hyperedges列表
    """
    try:
        data = get_hyperedges(database, page, page_size)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/hyperedges/{hyperedge_id}")
async def get_hyperedge(hyperedge_id: str, database: str = None):
    """
    Get details of a specific hyperedge
    获取指定hyperedge的详情
    """
    try:
        hyperedge_id = hyperedge_id.replace("%20", " ")
        vertices = hyperedge_id.split("|*|")
        data = get_hyperedge_detail(vertices, database)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/vertices/{vertex_id}")
async def get_vertex(vertex_id: str, database: str = None):
    """
    Get JSON data for a specific vertex
    获取指定vertex的json
    """
    vertex_id = vertex_id.replace("%20", " ")
    try:
        data = get_vertice(vertex_id, database)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/vertices_neighbor/{vertex_id}")
async def get_vertex_neighbor(vertex_id: str, database: str = None):
    """
    Get neighbors of a specific vertex
    获取指定vertex的neighbor
    """
    vertex_id = vertex_id.replace("%20", " ")
    try:
        data = get_vertice_neighbor(vertex_id, database)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/db/hyperedge_neighbor/{hyperedge_id}")
async def get_hyperedge_neighbor(hyperedge_id: str, database: str = None):
    """
    Get neighbors of a specific hyperedge
    获取指定hyperedge的neighbor
    """
    hyperedge_id = hyperedge_id.replace("%20", " ")
    hyperedge_id = hyperedge_id.replace("*", "#")
    print(hyperedge_id)
    try:
        data = get_hyperedge_neighbor_server(hyperedge_id, database)
        return data
    except Exception as e:
        return {"error": str(e)}

class VertexModel(BaseModel):
    vertex_id: str
    entity_name: str = ""
    entity_type: str = ""
    description: str = ""
    additional_properties: str = ""
    database: str = None

class HyperedgeModel(BaseModel):
    vertices: list
    keywords: str = ""
    summary: str = ""
    database: str = None

class VertexUpdateModel(BaseModel):
    entity_name: str = ""
    entity_type: str = ""
    description: str = ""
    additional_properties: str = ""
    database: str = None

class HyperedgeUpdateModel(BaseModel):
    keywords: str = ""
    summary: str = ""
    database: str = None

@app.post("/db/vertices")
async def create_vertex(vertex: VertexModel):
    """
    Create a new vertex
    创建新的vertex
    """
    try:
        result = add_vertex(vertex.vertex_id, {
            "entity_name": vertex.entity_name,
            "entity_type": vertex.entity_type,
            "description": vertex.description,
            "additional_properties": vertex.additional_properties
        }, vertex.database)
        return {"success": True, "message": "Vertex created successfully", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/db/hyperedges")
async def create_hyperedge(hyperedge: HyperedgeModel):
    """
    Create a new hyperedge
    创建新的hyperedge
    """
    try:
        result = add_hyperedge(hyperedge.vertices, {
            "keywords": hyperedge.keywords,
            "summary": hyperedge.summary
        }, hyperedge.database)
        return {"success": True, "message": "Hyperedge created successfully", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.put("/db/vertices/{vertex_id}")
async def update_vertex_endpoint(vertex_id: str, vertex: VertexUpdateModel):
    """
    Update vertex information
    更新vertex信息
    """
    try:
        vertex_id = vertex_id.replace("%20", " ")
        result = update_vertex(vertex_id, {
            "entity_name": vertex.entity_name,
            "entity_type": vertex.entity_type,
            "description": vertex.description,
            "additional_properties": vertex.additional_properties
        }, vertex.database)
        return {"success": True, "message": "Vertex updated successfully", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.put("/db/hyperedges/{hyperedge_id}")
async def update_hyperedge_endpoint(hyperedge_id: str, hyperedge: HyperedgeUpdateModel):
    """
    Update hyperedge information
    更新hyperedge信息
    """
    try:
        hyperedge_id = hyperedge_id.replace("%20", " ")
        vertices = hyperedge_id.split("|*|")
        result = update_hyperedge(vertices, {
            "keywords": hyperedge.keywords,
            "summary": hyperedge.summary
        }, hyperedge.database)
        return {"success": True, "message": "Hyperedge updated successfully", "data": result}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.delete("/db/vertices/{vertex_id}")
async def delete_vertex_endpoint(vertex_id: str, database: str = None):
    """
    Delete a vertex
    删除vertex
    """
    try:
        vertex_id = vertex_id.replace("%20", " ")
        result = delete_vertex(vertex_id, database)
        return {"success": True, "message": "Vertex deleted successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.delete("/db/hyperedges/{hyperedge_id}")
async def delete_hyperedge_endpoint(hyperedge_id: str, database: str = None):
    """
    Delete a hyperedge
    删除hyperedge
    """
    try:
        hyperedge_id = hyperedge_id.replace("%20", " ")
        vertices = hyperedge_id.split("|*|")
        result = delete_hyperedge(vertices, database)
        return {"success": True, "message": "Hyperedge deleted successfully"}
    except Exception as e:
        return {"success": False, "message": str(e)}

# 设置相关的API接口

class SettingsModel(BaseModel):
    apiKey: str = ""
    modelProvider: str = "openai"
    modelName: str = "gpt-5-mini"
    baseUrl: str = "https://api.openai.com/v1"
    selectedDatabase: str = ""
    maxTokens: int = 2000
    temperature: float = 0.7
    # HyperRAG embedding model settings
    embeddingModel: str = "text-embedding-3-small"
    embeddingDim: int = 1536
    # Language setting for backend
    language: str = "en-US"  # en-US or zh-CN

class APITestModel(BaseModel):
    apiKey: str
    baseUrl: str
    modelName: str
    modelProvider: str

class DatabaseTestModel(BaseModel):
    database: str

@app.get("/settings")
async def get_settings():
    """
    Get system settings
    获取系统设置
    """
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # 不返回敏感信息如API Key
            settings_safe = settings.copy()
            if 'apiKey' in settings_safe:
                settings_safe['apiKey'] = '***' if settings_safe['apiKey'] else ''
            return settings_safe
        else:
            # 返回默认设置
            return {
                "apiKey": "",
                "modelProvider": "openai",
                "modelName": "gpt-4o-mini",
                "baseUrl": "https://api.openai.com/v1",
                "selectedDatabase": "",
                "maxTokens": 2000,
                "temperature": 0.7,
                "embeddingModel": "text-embedding-3-small",
                "embeddingDim": 1536,
                "language": "en-US"
            }
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/settings")
async def save_settings(settings: SettingsModel):
    """
    Save system settings
    保存系统设置
    """
    try:
        settings_dict = settings.dict()
        
        # 如果apiKey是***，则保持原有的apiKey不变
        if settings_dict.get('apiKey') == '***':
            # 读取现有设置中的apiKey
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    existing_settings = json.load(f)
                # 保持原有的apiKey
                settings_dict['apiKey'] = existing_settings.get('apiKey', '')
            else:
                # 如果没有现有设置文件，则设为空字符串
                settings_dict['apiKey'] = ''
        
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=2)
        return {"success": True, "message": t('settings_save_success')}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/databases")
async def get_databases(lang: str = "en-US"):
    """
    Get list of available databases
    获取可用数据库列表
    """
    try:
        databases = []
        
        # 使用db_manager获取数据库列表
        database_files = db_manager.list_databases()
        
        for file in database_files:
            # Generate description based on language setting from translations
            base_name = file.replace('.hgdb', '')
            description = f"{base_name} {t('hypergraph_suffix')}"
            
            databases.append({
                "name": file,
                "description": description
            })
        
        # 如果没有找到数据库文件，返回默认列表
        if not databases:
            databases = []
        
        return databases
    except Exception as e:
        return {"success": False, "message": str(e), "data": []}

class CreateDatabaseRequest(BaseModel):
    """Request model for creating a new database"""
    name: str = Field(..., description="Name of the database to create (without .hgdb extension)", example="my_knowledge_base")

@app.post("/create-database")
async def create_database(request: CreateDatabaseRequest):
    """
    Create a new empty database
    创建新的空数据库
    """
    try:
        database_name = request.name
        if not database_name:
            return {"success": False, "message": "Database name is required"}
        
        # Ensure it doesn't have .hgdb extension (we'll add it)
        if database_name.endswith('.hgdb'):
            database_name = database_name[:-5]
        
        # Check if database already exists
        existing_databases = db_manager.list_databases()
        if database_name in existing_databases:
            return {"success": False, "message": f"Database '{database_name}' already exists"}
        
        # Create directory structure
        database_dir = os.path.join(db_manager.cache_dir, database_name)
        Path(database_dir).mkdir(parents=True, exist_ok=True)
        
        # Create empty .hgdb file path
        database_path = os.path.join(database_dir, "hypergraph_chunk_entity_relation.hgdb")
        
        # Initialize empty HypergraphDB (it will create the file when saved)
        from hyperdb import HypergraphDB
        new_db = HypergraphDB(storage_file=database_path)
        
        # Save empty database to create the file
        new_db.save(Path(database_path))
        
        main_logger.info(f"Created new database: {database_name}")
        
        return {
            "success": True, 
            "message": f"Database '{database_name}' created successfully",
            "database": database_name
        }
    except Exception as e:
        main_logger.error(f"Failed to create database: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/test-api")
async def test_api_connection(api_test: APITestModel):
    """
    Test API connection
    测试API连接
    """
    try:
        from openai import OpenAI
        
        # 根据不同的模型提供商进行测试
        if api_test.modelProvider == "gemini":
            # Test Gemini API
            from google import genai
            client = genai.Client(api_key=api_test.apiKey)
            # Simple test to check if API key is valid
            response = client.models.generate_content(
                model=api_test.modelName or "gemini-2.5-flash-lite",
                contents="Hello, this is a test. Reply with 'API connection successful'."
            )
            return {"success": True, "message": "Gemini API connection successful"}
        elif api_test.modelProvider == "openai":
            client = OpenAI(
                api_key=api_test.apiKey,
                base_url=api_test.baseUrl
            )
            
            # 发送一个简单的测试请求
            response = client.chat.completions.create(
                model=api_test.modelName,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {"success": True, "message": t('api_test_success')}
            
        elif api_test.modelProvider == "anthropic":
            # 对于Anthropic，可以添加相应的测试逻辑
            return {"success": True, "message": t('anthropic_api_test_success')}
            
        else:
            # 对于其他提供商，进行通用测试
            return {"success": True, "message": t('api_test_success')}
            
    except Exception as e:
        return {"success": False, "message": t('api_test_failed', error=str(e))}

@app.post("/test-database")
async def test_database_connection(db_test: DatabaseTestModel):
    """
    Test database connection
    测试数据库连接
    """
    try:
        # 使用db_manager测试数据库连接
        db = db_manager.get_database(db_test.database)
        
        # 尝试获取数据库的基本信息来验证连接
        vertices_count = len(db.all_v)
        edges_count = len(db.all_e)
        
        return {
            "success": True, 
            "message": t('database_test_success'),
            "info": {
                "vertices_count": vertices_count,
                "edges_count": edges_count,
                "database": db_test.database
            }
        }
        
    except Exception as e:
        return {"success": False, "message": t('database_test_failed', error=str(e))}


# 全局 HyperRAG 实例 - 改为字典来支持多数据库
hyperrag_instances = {}
hyperrag_working_dir = "hyperrag_cache"

async def get_hyperrag_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """
    HyperRAG-specific LLM function, using async version
    HyperRAG 专用的 LLM 函数，使用异步版本
    """
    try:
        main_logger.info(t('llm_call_start', length=len(prompt)))
        if system_prompt:
            main_logger.info(t('system_prompt_length', length=len(system_prompt)))
        
        # 从设置文件读取配置
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        model_name = settings.get("modelName", "gpt-4o-mini")
        api_key = settings.get("apiKey")
        # If api_key is a list (for rotation), use the first one for LLM calls
        if isinstance(api_key, list) and api_key:
            llm_api_key = api_key[0]
        else:
            llm_api_key = api_key
        base_url = settings.get("baseUrl")
        model_provider = settings.get("modelProvider", "openai")
        
        main_logger.info(t('using_model', model=model_name, url=base_url))
        
        if model_provider == "gemini":
            # Use Gemini for text generation
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=llm_api_key)
            
            # Build conversation history using proper Content types
            contents = []
            
            # Add system prompt as initial user/model exchange if provided
            if system_prompt:
                contents.append(types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=system_prompt)]
                ))
                contents.append(types.Content(
                    role='model', 
                    parts=[types.Part.from_text(text="Understood. I'll follow these instructions.")]
                ))
            
            # Add history messages
            for msg in history_messages:
                role = 'user' if msg.get("role") == "user" else 'model'
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))]
                ))
            
            # Add current prompt
            contents.append(types.Content(
                role='user',
                parts=[types.Part.from_text(text=prompt)]
            ))
            
            # Retry logic for 503 errors
            max_retries = 100
            retry_delay = 30  # seconds (increased by factor of 6)
            
            for attempt in range(max_retries):
                try:
                    response_obj = client.models.generate_content(
                        model=model_name or "gemini-2.5-flash-lite",
                        contents=contents
                    )
                    response = response_obj.text
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "overloaded" in error_str.lower() or "UNAVAILABLE" in error_str:
                        if attempt < max_retries - 1:
                            main_logger.info(f"Model overloaded (503), retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            main_logger.error(f"Max retries reached for 503 error")
                            raise
                    else:
                        # Not a 503 error, raise immediately
                        raise
        else:
            # Use OpenAI-compatible API
            response = await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=llm_api_key,
                base_url=base_url,
                **kwargs,
            )
        
        main_logger.info(t('llm_call_complete', length=len(response)))
        return response
        
    except Exception as e:
        main_logger.error(t('llm_call_failed', error=str(e)))
        raise

async def get_hyperrag_embedding_func(texts: list[str]) -> np.ndarray:
    """
    HyperRAG-specific embedding function
    HyperRAG 专用的嵌入函数
    """
    try:
        main_logger.info(t('text_embedding_start', count=len(texts)))
        main_logger.info(t('text_total_length', length=sum(len(text) for text in texts)))
        
        # 从设置文件读取配置
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        embedding_model = settings.get("embeddingModel", "text-embedding-3-small")
        api_key = settings.get("apiKey")
        base_url = settings.get("baseUrl")
        model_provider = settings.get("modelProvider", "openai")
        embedding_dim = settings.get("embeddingDim", 768)
        
        main_logger.info(t('using_embedding_model', model=embedding_model))
        
        # Check model provider
        if model_provider == "gemini":
            # Use Gemini embeddings
            from hyperrag.llm import gemini_embedding
            embeddings = await gemini_embedding(
                texts,
                model=embedding_model,
                api_key=api_key,
                embedding_dim=embedding_dim
            )
        else:
            # Use OpenAI embeddings (for openai, azure, custom, etc.)
            # If api_key is a list, use the first one (OpenAI doesn't support rotation yet)
            embedding_api_key = api_key[0] if isinstance(api_key, list) and api_key else api_key
            embeddings = await openai_embedding(
                texts,
                model=embedding_model,
                api_key=embedding_api_key,
                base_url=base_url,
            )
        
        main_logger.info(t('text_embedding_complete', dimensions=embeddings.shape))
        return embeddings
        
    except Exception as e:
        main_logger.error(t('text_embedding_failed', error=str(e)))
        raise

def get_or_create_hyperrag(database: str = None):
    """
    Get or create HyperRAG instance for specified database
    获取或创建指定数据库的 HyperRAG 实例
    """
    global hyperrag_instances
    
    if not HYPERRAG_AVAILABLE:
        main_logger.error(t('hyperrag_unavailable'))
        raise RuntimeError("HyperRAG is not available")
    
    # 如果没有指定数据库，使用默认数据库
    if database is None:
        database = db_manager.default_database
        main_logger.info(t('using_default_database', database=database))
    
    # 检查是否已存在该数据库的实例
    if database not in hyperrag_instances:
        main_logger.info(t('create_new_hyperrag_instance', database=database))
        
        # 使用数据库名作为工作目录（去掉.hgdb后缀）
        if database.endswith('.hgdb'):
            db_dir_name = database.replace('.hgdb', '')
        else:
            db_dir_name = database
            
        # HyperRAG 工作目录直接使用 hyperrag_cache 下的数据库文件夹
        db_working_dir = os.path.join(hyperrag_working_dir, db_dir_name)
        Path(db_working_dir).mkdir(parents=True, exist_ok=True)
        
        main_logger.info(t('hyperrag_working_dir', dir=db_working_dir))
        
        # 初始化 HyperRAG 实例
        hyperrag_instances[database] = HyperRAG(
            working_dir=db_working_dir,
            llm_model_func=get_hyperrag_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,  # text-embedding-3-small 的维度
                max_token_size=8192,
                func=get_hyperrag_embedding_func
            ),
        )
        
        main_logger.info(t('hyperrag_instance_created', database=database))
    else:
        main_logger.info(t('use_existing_hyperrag_instance', database=database))
    
    return hyperrag_instances[database]


class Message(BaseModel):
    message: str

@app.post("/process_message")
async def process_message(msg: Message):
    user_message = msg.message
    try:
        response_message = await get_hyperrag_llm_func(prompt=user_message)
    except Exception as e:
        return {"response": str(e)} 
    return {"response": response_message}

# HyperRAG 问答相关接口

class DocumentModel(BaseModel):
    content: str
    retries: int = 3
    database: str = None  # 添加数据库参数

class QueryModel(BaseModel):
    question: str
    mode: str = "hyper"  # hyper, hyper-lite, naive
    top_k: int = 60
    max_token_for_text_unit: int = 1600
    max_token_for_entity_context: int = 300
    max_token_for_relation_context: int = 1600
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    database: str = None  # 添加数据库参数

@app.post("/hyperrag/insert")
async def insert_document(doc: DocumentModel):
    """
    Insert document into specified database's HyperRAG
    向指定数据库的 HyperRAG 插入文档
    """
    if not HYPERRAG_AVAILABLE:
        return {"success": False, "message": "HyperRAG is not available"}
    
    try:
        rag = get_or_create_hyperrag(doc.database)
        
        # 重试机制
        for attempt in range(doc.retries):
            try:
                await rag.ainsert(doc.content)
                return {
                    "success": True, 
                    "message": "Document inserted successfully",
                    "database": doc.database or "default"
                }
            except Exception as e:
                if attempt == doc.retries - 1:
                    raise e
                print(f"Insert attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(2)
                
    except Exception as e:
        return {"success": False, "message": f"Failed to insert document: {str(e)}"}

@app.post("/hyperrag/query")
async def query_hyperrag(query: QueryModel):
    """
    Query HyperRAG using specified database for Q&A
    使用指定数据库的 HyperRAG 进行问答查询
    """
    if not HYPERRAG_AVAILABLE:
        return {"success": False, "message": "HyperRAG is not available"}
    
    try:
        rag = get_or_create_hyperrag(query.database)
        
        # 创建查询参数
        param = QueryParam(
            mode=query.mode,
            top_k=query.top_k,
            max_token_for_text_unit=query.max_token_for_text_unit,
            max_token_for_entity_context=query.max_token_for_entity_context,
            max_token_for_relation_context=query.max_token_for_relation_context,
            only_need_context=query.only_need_context,
            response_type=query.response_type,
            return_type='json'
        )
        
        # 执行查询
        result = await rag.aquery(query.question, param)
        
        # 处理结果格式
        return {
            "success": True,
            "response": result.get("response", ""),
            "entities": result.get("entities", []),
            "hyperedges": result.get("hyperedges", []),
            "text_units": result.get("text_units", []),
            "mode": query.mode,
            "question": query.question,
            "database": query.database or "default"
        }
        
    except Exception as e:
        return {"success": False, "message": f"Query failed: {str(e)}"}

@app.get("/hyperrag/status")
async def get_hyperrag_status(database: str = None):
    """
    Get HyperRAG instance status for specified database
    获取指定数据库的 HyperRAG 实例状态
    """
    try:
        status = {
            "available": HYPERRAG_AVAILABLE,
            "database": database or "default",
            "working_dir": hyperrag_working_dir,
            "instances": list(hyperrag_instances.keys())
        }
        
        if database:
            # 获取特定数据库的状态
            if database in hyperrag_instances:
                instance = hyperrag_instances[database]
                status["initialized"] = True
                try:
                    status["details"] = {
                        "chunk_token_size": instance.chunk_token_size,
                        "llm_model_name": instance.llm_model_name,
                        "embedding_func_available": instance.embedding_func is not None,
                        "working_dir": os.path.join(hyperrag_working_dir, database.replace('.hgdb', ''))
                    }
                except Exception as e:
                    status["details"] = f"Error getting details: {str(e)}"
            else:
                status["initialized"] = False
        else:
            # 获取所有实例的概览
            status["initialized"] = len(hyperrag_instances) > 0
            status["total_instances"] = len(hyperrag_instances)
        
        return status
        
    except Exception as e:
        return {"success": False, "message": f"Failed to get status: {str(e)}"}

@app.delete("/hyperrag/reset")
async def reset_hyperrag(database: str = None):
    """
    Reset HyperRAG instance for specified database, or reset all instances
    重置指定数据库的 HyperRAG 实例，或重置所有实例
    """
    global hyperrag_instances
    
    try:
        if database:
            # 重置特定数据库的实例
            if database in hyperrag_instances:
                del hyperrag_instances[database]
                return {
                    "success": True, 
                    "message": f"HyperRAG instance for database '{database}' reset successfully"
                }
            else:
                return {
                    "success": False, 
                    "message": f"No HyperRAG instance found for database '{database}'"
                }
        else:
            # 重置所有实例
            hyperrag_instances = {}
            return {"success": True, "message": "All HyperRAG instances reset successfully"}
            
    except Exception as e:
        return {"success": False, "message": f"Failed to reset: {str(e)}"}

# 文件管理相关的API接口

class FileEmbedRequest(BaseModel):
    file_ids: List[str]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    database: str = None  # Add database parameter

@app.get("/files")
async def get_files():
    """
    Get list of all uploaded files
    获取所有上传的文件列表
    """
    try:
        files = file_manager.get_all_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=t('get_file_list_failed', error=str(e)))

@app.post("/files/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    File upload interface
    上传文件接口
    """
    print(f"\n{'='*50}")
    print(t('file_upload_start', count=len(files)))
    print(f"{'='*50}")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            print(f"\n{t('upload_file', current=i+1, total=len(files), filename=file.filename)}")
            print(t('file_size', size=file.size if hasattr(file, 'size') else t('unknown')))
            
            # 读取文件内容
            print(t('reading_file_content'))
            content = await file.read()
            print(f"✅ {t('file_content_read_complete', size=len(content))}")
            
            # 保存文件
            print(t('saving_file_locally'))
            file_info = await file_manager.save_uploaded_file(content, file.filename)
            file_info["status"] = "uploaded"
            print(f"✅ {t('file_save_success', filename=file_info['filename'])}")
            print(f"  - {t('file_id', file_id=file_info['file_id'])}")
            print(f"  - {t('save_path', path=file_info['file_path'])}")
            print(f"  - {t('database', database=file_info['database_name'])}")
            
            results.append(file_info)
            
        except Exception as e:
            error_msg = t('file_upload_failed', filename=file.filename, error=str(e))
            print(f"❌ {error_msg}")
            main_logger.error(error_msg)
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    success_count = len([r for r in results if r.get('status') == 'uploaded'])
    print(f"\n{t('file_upload_complete', success=success_count, total=len(files))}")
    print(f"{'='*50}")
    
    return {"files": results}

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete specified file
    删除指定的文件
    """
    try:
        success = file_manager.delete_file(file_id)
        if success:
            return {"success": True, "message": t('file_delete_success')}
        else:
            raise HTTPException(status_code=404, detail=t('file_not_exist'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=t('file_delete_failed', error=str(e)))

@app.post("/files/embed")
async def embed_files(request: FileEmbedRequest):
    """
    Batch embed documents into HyperRAG
    批量嵌入文档到HyperRAG
    """
    if not HYPERRAG_AVAILABLE:
        raise HTTPException(status_code=500, detail="HyperRAG is not available")
    
    print(f"\n{'='*50}")
    print(t('document_embed_start', count=len(request.file_ids)))
    print(t('config_params', chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap))
    print(f"{'='*50}")
    
    results = []
    
    try:
        for i, file_id in enumerate(request.file_ids):
            try:
                print(f"\n{t('processing_file', current=i+1, total=len(request.file_ids), file_id=file_id)}")
                
                # 更新文件状态为处理中
                print(t('update_file_status_processing'))
                file_manager.update_file_status(file_id, "processing")
                
                # 获取文件信息
                print(t('get_file_info'))
                file_info = file_manager.get_file_by_id(file_id)
                if not file_info:
                    error_msg = t('file_not_found', path=file_id)
                    print(f"❌ {error_msg}")
                    results.append({
                        "file_id": file_id,
                        "status": "error",
                        "error": t('file_not_exist')
                    })
                    continue
                
                print(f"✅ {t('file_info', filename=file_info['filename'], size=file_info['file_size'])}")
                
                # 使用文件对应的数据库名
                database_name = file_info["database_name"]
                print(t('target_database', database=database_name))
                rag = get_or_create_hyperrag(database_name)
                
                # 读取文件内容
                print(t('reading_file_content'))
                content = await file_manager.read_file_content(file_info["file_path"])
                print(f"✅ {t('content_length', length=len(content))}")
                
                # 插入到HyperRAG
                print(t('start_document_embedding'))
                await rag.ainsert(content)
                print(f"✅ {t('document_embedding_complete')}")
                
                # 更新文件状态为已嵌入
                file_manager.update_file_status(file_id, "embedded")
                
                results.append({
                    "file_id": file_id,
                    "filename": file_info["filename"],
                    "database_name": database_name,
                    "status": "embedded"
                })
                
                print(f"✅ {t('file_embedded_success', filename=file_info['filename'])}")
                
            except Exception as e:
                # 更新文件状态为错误
                error_msg = t('file_embedding_failed', file_id=file_id, error=str(e))
                print(f"❌ {error_msg}")
                file_manager.update_file_status(file_id, "error", str(e))
                
                results.append({
                    "file_id": file_id,
                    "status": "error",
                    "error": str(e)
                })
        
        successful = len([r for r in results if r.get('status') == 'embedded'])
        print(f"\n{t('document_embedding_summary', success=successful, total=len(request.file_ids))}")
        print(f"{'='*50}")
        
        return {"embedded_files": results}
        
    except Exception as e:
        error_msg = t('batch_embedding_failed', error=str(e))
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# 自定义日志处理器，将日志通过WebSocket发送
class WebSocketLogHandler(logging.Handler):
    def __init__(self, connection_manager):
        super().__init__()
        self.connection_manager = connection_manager
        
    def emit(self, record):
        try:
            log_message = self.format(record)
            # 异步发送日志消息
            asyncio.create_task(self.connection_manager.send_log_message({
                "type": "log",
                "level": record.levelname,
                "message": log_message,
                "timestamp": record.created,
                "logger_name": record.name
            }))
        except Exception:
            pass  # 避免日志处理器自身错误影响主程序

# 自定义流处理器，捕获print语句和其他输出
class WebSocketStreamHandler:
    def __init__(self, connection_manager, stream_type="stdout"):
        self.connection_manager = connection_manager
        self.stream_type = stream_type
        self.original_stream = sys.stdout if stream_type == "stdout" else sys.stderr
        
    def write(self, message):
        # 同时写入原始流
        self.original_stream.write(message)
        self.original_stream.flush()
        
        # 发送到WebSocket（去除空行）
        if message.strip():
            asyncio.create_task(self.connection_manager.send_log_message({
                "type": "console",
                "level": "ERROR" if self.stream_type == "stderr" else "INFO",
                "message": message.strip(),
                "timestamp": asyncio.get_event_loop().time(),
                "source": self.stream_type
            }))
    
    def flush(self):
        self.original_stream.flush()

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logging_enabled = False
        self.original_stdout = None
        self.original_stderr = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 如果是第一个连接，启用日志重定向
        if len(self.active_connections) == 1 and not self.logging_enabled:
            self.enable_logging_redirect()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # 如果没有连接了，禁用日志重定向
        if len(self.active_connections) == 0 and self.logging_enabled:
            self.disable_logging_redirect()

    def enable_logging_redirect(self):
        """Enable log redirection / 启用日志重定向"""
        if not self.logging_enabled:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
            # 重定向标准输出和错误输出
            sys.stdout = WebSocketStreamHandler(self, "stdout")
            sys.stderr = WebSocketStreamHandler(self, "stderr")
            
            self.logging_enabled = True
            print(t('log_redirect_enabled'))

    def disable_logging_redirect(self):
        """Disable log redirection / 禁用日志重定向"""
        if self.logging_enabled and self.original_stdout and self.original_stderr:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.logging_enabled = False
            print(t('log_redirect_disabled'))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # 如果连接已断开，标记为移除
                disconnected.append(connection)
        
        # 移除断开的连接
        for conn in disconnected:
            self.disconnect(conn)

    async def send_progress_update(self, progress_data: dict):
        """Send progress updates to all connected clients / 发送进度更新到所有连接的客户端"""
        message = json.dumps(progress_data)
        await self.broadcast(message)
    
    async def send_log_message(self, log_data: dict):
        """Send log messages to all connected clients / 发送日志消息到所有连接的客户端"""
        message = json.dumps(log_data)
        await self.broadcast(message)

manager = ConnectionManager()

# 设置全面的日志配置
def setup_comprehensive_logging():
    """Setup comprehensive logging configuration / 设置全面的日志配置"""
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建WebSocket处理器
    ws_handler = WebSocketLogHandler(manager)
    ws_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ws_handler.setFormatter(formatter)
    
    # 创建控制台处理器（保留控制台输出）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(ws_handler)
    root_logger.addHandler(console_handler)
    
    # 设置特定模块的日志级别
    logging.getLogger('hyperrag').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)  # 减少HTTP请求日志
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # 确保HyperRAG相关的所有子模块都能输出日志
    hyperrag_modules = [
        'hyperrag.base',
        'hyperrag.hyperrag', 
        'hyperrag.llm',
        'hyperrag.operate',
        'hyperrag.prompt',
        'hyperrag.storage',
        'hyperrag.utils'
    ]
    
    for module_name in hyperrag_modules:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(logging.INFO)
        # 确保模块日志也会传播到根记录器
        module_logger.propagate = True
    
    return root_logger

def configure_hyperrag_logging():
    """Configure HyperRAG-related detailed log output / 配置HyperRAG相关的详细日志输出"""
    try:
        # 如果HyperRAG可用，配置其内部日志
        if HYPERRAG_AVAILABLE:
            # 导入HyperRAG相关模块并设置日志
            try:
                import hyperrag
                import hyperrag.base
                import hyperrag.storage
                import hyperrag.llm
                import hyperrag.utils
                
                # 为HyperRAG的主要模块设置日志记录器
                modules_to_configure = [
                    hyperrag,
                    hyperrag.base,
                    hyperrag.storage, 
                    hyperrag.llm,
                    hyperrag.utils
                ]
                
                for module in modules_to_configure:
                    if hasattr(module, '__name__'):
                        logger = logging.getLogger(module.__name__)
                        logger.setLevel(logging.INFO)
                        logger.propagate = True
                        
                print(f"✅ {t('hyperrag_log_config_complete')}")
                        
            except ImportError as e:
                print(f"⚠️  {t('cannot_import_hyperrag_module', error=e)}")
                
    except Exception as e:
        print(f"⚠️  {t('hyperrag_log_config_failed', error=e)}")

# 初始化日志系统
main_logger = setup_comprehensive_logging()

# 配置HyperRAG日志
configure_hyperrag_logging()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 这里可以处理客户端发送的消息
            await manager.send_personal_message(f"Message received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 带实时进度通知的文档嵌入接口
@app.post("/files/embed-with-progress")
async def embed_files_with_progress(request: FileEmbedRequest):
    """
    Batch embed documents into HyperRAG with real-time progress notifications
    批量嵌入文档到HyperRAG，带实时进度通知
    """
    if not HYPERRAG_AVAILABLE:
        raise HTTPException(status_code=500, detail="HyperRAG is not available")
    
    # 立即返回处理开始的响应
    total_files = len(request.file_ids)
    
    # 异步处理文件嵌入
    asyncio.create_task(process_files_with_progress(request, total_files))
    
    return {
        "message": t('document_embedding_started'),
        "total_files": total_files,
        "processing": True
    }

async def process_files_with_progress(request: FileEmbedRequest, total_files: int):
    """Asynchronously process file embedding and send progress updates / 异步处理文件嵌入并发送进度更新"""
    try:
        print(f"="*60)
        print(t('batch_file_embed_start'))
        print(t('total_files', count=total_files))
        print(t('config_params', chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap))
        print(f"="*60)
        
        main_logger.info(t('processing_embed_task', count=total_files))
        main_logger.info(t('config_parameters', chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap))
        
        successful_files = 0
        failed_files = 0
        
        for i, file_id in enumerate(request.file_ids):
            try:
                print(f"\n{'='*40}")
                print(t('processing_file_num', current=i + 1, total=total_files))
                print(t('file_id_label', file_id=file_id))
                print(f"{'='*40}")
                
                # 发送进度更新
                await manager.send_progress_update({
                    "type": "progress",
                    "file_id": file_id,
                    "current": i + 1,
                    "total": total_files,
                    "percentage": ((i + 1) / total_files) * 100,
                    "status": "processing",
                    "message": t('processing_file_message', current=i + 1, total=total_files)
                })
                
                # 更新文件状态为处理中
                print(t('update_file_status_processing'))
                file_manager.update_file_status(file_id, "processing")
                
                # 获取文件信息
                print(t('reading_file_content_progress'))
                main_logger.info(t('getting_file_info', file_id=file_id))
                file_info = file_manager.get_file_by_id(file_id)
                if not file_info:
                    error_msg = t('file_not_found', path=file_id)
                    print(f"❌ {t('error_label', error=error_msg)}")
                    main_logger.error(error_msg)
                    await manager.send_progress_update({
                        "type": "error",
                        "file_id": file_id,
                        "error": t('file_not_exist'),
                        "current": i + 1,
                        "total": total_files
                    })
                    failed_files += 1
                    continue
                
                print(f"✅ {t('file_info_success')}")
                print(f"  - {t('filename', filename=file_info['filename'])}")
                print(f"  - {t('file_size', size=file_info['file_size'])}")
                print(f"  - {t('upload_time', time=file_info['upload_time'])}")
                
                # 使用请求中指定的数据库，如果没有则使用文件对应的数据库名
                database_name = request.database if request.database else file_info.get("database_name", "default")
                print(f"  - {t('target_database', database=database_name)}")
                
                main_logger.info(t('starting_file_processing', filename=file_info['filename'], size=file_info['file_size'], database=database_name))
                
                # 为每个文件初始化对应的HyperRAG实例
                print(t('initializing_hyperrag_instance'))
                main_logger.info(t('initializing_hyperrag_with_db', database=database_name))
                rag = get_or_create_hyperrag(database_name)
                print(f"✅ {t('hyperrag_instance_initialized')}")
                main_logger.info(t('hyperrag_initialized_with_db', database=database_name))
                
                # 发送详细进度信息
                await manager.send_progress_update({
                    "type": "file_processing",
                    "file_id": file_id,
                    "filename": file_info["filename"],
                    "database_name": database_name,
                    "stage": "reading",
                    "message": t('reading_file_message', filename=file_info['filename'], database=database_name)
                })
                
                # 读取文件内容
                print(t('reading_file_content'))
                main_logger.info(t('reading_file_message', filename=file_info['filename'], database=database_name))
                content = await file_manager.read_file_content(file_info["file_path"])
                print(f"✅ {t('file_read_complete', length=len(content))}")
                main_logger.info(t('file_read_complete', length=len(content)))
                
                # 显示内容预览
                preview = content[:200] + "..." if len(content) > 200 else content
                print(t('content_preview', preview=preview))
                
                # 发送嵌入阶段的进度
                await manager.send_progress_update({
                    "type": "file_processing",
                    "file_id": file_id,
                    "filename": file_info["filename"],
                    "database_name": database_name,
                    "stage": "embedding",
                    "message": t('embedding_document_message', filename=file_info['filename'], database=database_name)
                })
                
                # 插入到HyperRAG
                print(t('document_embedding_processing'))
                print(t('document_embedding_wait'))
                main_logger.info(t('embedding_document_message', filename=file_info['filename'], database=database_name))
                main_logger.info(t('document_chunking'))
                
                # 这里会触发HyperRAG的详细处理过程
                await rag.ainsert(content)
                
                print(f"✅ {t('document_embedding_complete')}")
                main_logger.info(t('embedding_document_message', filename=file_info['filename'], database=database_name))
                
                # 更新文件状态为已嵌入
                file_manager.update_file_status(file_id, "embedded")
                
                # 发送成功完成的进度更新
                await manager.send_progress_update({
                    "type": "file_completed",
                    "file_id": file_id,
                    "filename": file_info["filename"],
                    "database_name": database_name,
                    "status": "completed",
                    "message": t('file_embed_complete', filename=file_info['filename'], database=database_name)
                })
                
                successful_files += 1
                print(f"✅ {t('file_process_success', filename=file_info['filename'])}")
                
            except Exception as e:
                # 更新文件状态为错误
                error_msg = t('file_process_failed', file_id=file_id, error=str(e))
                print(f"❌ {error_msg}")
                main_logger.error(error_msg)
                file_manager.update_file_status(file_id, "error", str(e))
                
                # 发送错误进度更新
                await manager.send_progress_update({
                    "type": "file_error",
                    "file_id": file_id,
                    "error": str(e),
                    "current": i + 1,
                    "total": total_files
                })
                
                failed_files += 1
        
        # 发送整体完成的进度更新
        print(f"\n{'='*60}")
        print(t('batch_document_complete'))
        print(t('total_files', count=total_files))
        print(t('success_process', count=successful_files))
        print(t('failed_process', count=failed_files))
        print(t('success_rate', rate=f"{(successful_files/total_files)*100:.1f}"))
        print(f"{'='*60}")
        
        main_logger.info(t('all_documents_complete', total=total_files, success=successful_files, failed=failed_files))
        await manager.send_progress_update({
            "type": "all_completed",
            "message": t('all_complete_message', success=successful_files, failed=failed_files),
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files
        })
        
    except Exception as e:
        # 发送整体错误信息
        error_msg = t('batch_embedding_failed', error=str(e))
        print(f"❌ {error_msg}")
        main_logger.error(error_msg)
        await manager.send_progress_update({
            "type": "error",
            "error": error_msg
        })