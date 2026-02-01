import json
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    content: str
    embedding: List[float]
    source: str
    metadata: Dict[str, Any]

class EmbeddingLoader:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.index_path = os.path.join(repo_path, ".copilot-index")
        self.chunks: List[Chunk] = []

    def load(self) -> List[Chunk]:
        if not os.path.exists(self.index_path):
            logger.warning(f"Index path {self.index_path} does not exist.")
            return []

        json_files = glob.glob(os.path.join(self.index_path, "*.json"))
        logger.info(f"Found index files: {json_files}")

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._parse_data(data, json_file)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        logger.info(f"Loaded {len(self.chunks)} chunks.")
        return self.chunks

    def _parse_data(self, data: Any, filename: str):
        # Strategy 0: Orama / Obsidian Copilot v3+ Format
        # Structure: { "docs": { "docs": { "id": { ... } } }, "index": { "vectorIndexes": { "embedding": { "vectors": { "id": [...] } } } } }
        if (isinstance(data, dict)
            and "docs" in data
            and "index" in data
            and isinstance(data["docs"], dict)
            and "docs" in data["docs"]):

            logger.info(f"Detected Orama format in {filename}")

            docs_map = data["docs"]["docs"]
            # Vectors path: index -> vectorIndexes -> embedding -> vectors
            vectors_map = {}
            try:
                vectors_map = data["index"]["vectorIndexes"]["embedding"]["vectors"]
            except (KeyError, TypeError):
                logger.warning(f"Could not find vectors in Orama file {filename}")
                return

            for doc_key, doc_data in docs_map.items():
                if not isinstance(doc_data, dict):
                    continue

                # Content usually in 'content' or 'text'
                content = doc_data.get("content") or doc_data.get("text")
                # Source path
                source = doc_data.get("filepath") or doc_data.get("path") or filename

                # IMPORTANT: Use the internal document ID to look up the embedding,
                # not the key in the docs map (which might be a sequential index).
                doc_id = doc_data.get("id")

                embedding = None
                if doc_id:
                     embedding = vectors_map.get(doc_id)

                # Fallback: if vectors keyed by doc_key (less likely but robust)
                if not embedding:
                     embedding = vectors_map.get(doc_key)

                if content and embedding:
                    self._add_validated_chunk(content, embedding, source, doc_data)
            return

        # Strategy 1: Dict of file paths -> object with chunks
        if isinstance(data, dict):
            for key, value in data.items():
                # Check if key looks like a file path or just a string key
                # We assume if it has 'chunks', it's the structure we want
                if isinstance(value, dict) and "chunks" in value:
                    for chunk_data in value["chunks"]:
                        self._add_chunk(chunk_data, source=key)
                elif isinstance(value, list):
                     # Maybe the value is the list of chunks directly
                     for item in value:
                         self._add_chunk(item, source=key)
                elif isinstance(value, dict):
                     # Nested structure? Try recursively or check for embedding directly?
                     self._add_chunk(value, source=key)

        # Strategy 2: List of objects
        elif isinstance(data, list):
            for item in data:
                self._add_chunk(item, source=filename)

    def _add_chunk(self, item: Any, source: str):
        if not isinstance(item, dict):
            return

        content = item.get("content") or item.get("text")
        embedding = item.get("embedding") or item.get("vector")

        self._add_validated_chunk(content, embedding, source, item)

    def _add_validated_chunk(self, content: Any, embedding: Any, source: str, metadata: Dict[str, Any]):
        # Ensure embedding is a list of floats
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except:
                pass

        if content and isinstance(embedding, list) and len(embedding) > 0:
            # Check if elements are numbers
            if all(isinstance(x, (int, float)) for x in embedding):
                self.chunks.append(Chunk(
                    content=content,
                    embedding=embedding,
                    source=source,
                    metadata=metadata
                ))
            else:
                 logger.debug(f"Skipping chunk with non-numeric embedding in {source}")
