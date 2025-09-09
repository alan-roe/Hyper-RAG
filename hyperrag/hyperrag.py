import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .operate import (
    chunking_by_token_size,
    extract_entities,
    hyper_query_lite,
    hyper_query,
    naive_query,
    graph_query,
    llm_query,
)
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    HypergraphStorage,
)


from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .logging_utils import CleanLogger
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    BaseHypergraphStorage,
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class HyperRAG:
    _clean_logger = CleanLogger("hyperrag")
    working_dir: str = field(
        default_factory=lambda: f"./HyperRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    entity_additional_properties_to_max_tokens: int = 250
    relation_summary_to_max_tokens: int = 750
    relation_keywords_to_max_tokens: int = 100

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete#
    # llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_name: str = ""
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    hypergraph_storage_cls: Type[BaseHypergraphStorage] = HypergraphStorage
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def _get_serializable_config(self) -> dict:
        """Get a serializable dictionary representation of the config, excluding callables."""
        config = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            # Include all non-callable fields
            if not callable(value):
                config[field_name] = value
            # For callable fields, include them as-is for internal use
            # but they won't be serialized for logging
            else:
                config[field_name] = value
        return config

    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "HyperRAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        # Create a safe config dict for logging (excluding callables)
        safe_config = {}
        for k, v in self._get_serializable_config().items():
            if not callable(v):
                try:
                    safe_config[k] = str(v)
                except:
                    safe_config[k] = f"<{type(v).__name__}>"
        
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in safe_config.items()])
        logger.debug(f"HyperRAG init with param:\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=self._get_serializable_config()
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=self._get_serializable_config()
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=self._get_serializable_config()
            )
            if self.enable_llm_cache
            else None
        )
        """
            download from hgdb_path
        """
        self.chunk_entity_relation_hypergraph = self.hypergraph_storage_cls(
            namespace="chunk_entity_relation", global_config=self._get_serializable_config()
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=self._get_serializable_config(),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=self._get_serializable_config(),
            embedding_func=self.embedding_func,
            meta_fields={"id_set"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=self._get_serializable_config(),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            logger.info(f"[Chunk Storage] Embedding {len(inserting_chunks)} chunks into vector database...")
            await self.chunks_vdb.upsert(inserting_chunks)
            # ----------------------------------------------------------------------------
            self._clean_logger.info("Starting entity extraction", chunks=len(inserting_chunks))
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_hypergraph_inst=self.chunk_entity_relation_hypergraph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=self._get_serializable_config(),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            # ----------------------------------------------------------------------------
            self.chunk_entity_relation_hypergraph = maybe_new_kg
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
            
            # Log successful completion
            self._clean_logger.info(f"Document insertion completed successfully", 
                                  documents=len(new_docs), 
                                  chunks=len(inserting_chunks))
        finally:
            await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_hypergraph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def get_context(self, query: str, param: QueryParam = QueryParam(), max_tokens: int = None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aget_context(query, param, max_tokens))
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aget_context(self, query: str, param: QueryParam = QueryParam(), max_tokens: int = None):
        """
        Get the enriched context for a query without generating the final response.
        This is useful for integration with external systems that want to handle
        response generation themselves.
        
        Args:
            query: The query string
            param: Query parameters
            max_tokens: Optional max tokens for LLM calls (defaults to model's default if not specified)
            
        Returns:
            Dictionary containing the enriched context with entities, relationships, and text units
        """
        # Set the parameter to only return context
        context_param = QueryParam(
            **{**param.__dict__, "only_need_context": True, "return_type": "json"}
        )
        
        # Get config and add max_tokens if specified
        config = self._get_serializable_config()
        if max_tokens is not None:
            config = {**config, "max_tokens": max_tokens}
        
        # Get context based on mode
        if param.mode == "hyper":
            context = await hyper_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                context_param,
                config,
            )
        elif param.mode == "hyper-lite":
            context = await hyper_query_lite(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.text_chunks,
                context_param,
                config,
            )
        elif param.mode == "graph":
            context = await graph_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                context_param,
                config,
            )
        elif param.mode == "naive":
            context = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                context_param,
                config,
            )
        elif param.mode == "llm":
            # LLM mode doesn't have context, just returns a response
            raise ValueError("LLM mode does not support context-only queries")
        else:
            raise ValueError(f"Unknown mode {param.mode}")
            
        await self._query_done()
        return context

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        
        if param.mode == "hyper":
            response = await hyper_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self._get_serializable_config(),
            )
        elif param.mode == "hyper-lite":
            response = await hyper_query_lite(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.text_chunks,
                param,
                self._get_serializable_config(),
            )
        elif param.mode == "graph":
            response = await graph_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self._get_serializable_config(),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self._get_serializable_config(),
            )
        elif param.mode == "llm":
            response = await llm_query(
                query,
                param,
                self._get_serializable_config(),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).query_done_callback())
        await asyncio.gather(*tasks)
