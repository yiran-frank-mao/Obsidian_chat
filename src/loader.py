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

        # Sometimes embedding might be in a nested 'metadata' or similar, but let's stick to top level for now

        if content and embedding:
            self.chunks.append(Chunk(
                content=content,
                embedding=embedding,
                source=source,
                metadata=item
            ))
