import os
import numpy as np
from typing import List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from loader import Chunk
import logging

logger = logging.getLogger(__name__)

class ChatEngine:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        if chunks:
            # Pad embeddings if they have inconsistent lengths (shouldn't happen in valid index)
            # But let's check first dimension
            dim = len(chunks[0].embedding)
            valid_chunks = [c for c in chunks if len(c.embedding) == dim]
            if len(valid_chunks) != len(chunks):
                logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} chunks with inconsistent embedding dimensions.")

            self.chunks = valid_chunks
            self.embeddings = np.array([c.embedding for c in self.chunks])
        else:
            self.embeddings = None

    def search(self, query_embedding: List[float], k: int = 3) -> List[Chunk]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Ensure query embedding matches dimension
        if len(query_embedding) != self.embeddings.shape[1]:
            logger.error(f"Query embedding dimension {len(query_embedding)} does not match index dimension {self.embeddings.shape[1]}")
            return []

        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k indices
        # If fewer than k chunks, take all
        k = min(k, len(self.chunks))
        top_k_indices = similarities.argsort()[-k:][::-1]

        return [self.chunks[i] for i in top_k_indices]

    def generate_response(self, query: str, context_chunks: List[Chunk], provider: str, api_key: str, **kwargs) -> str:
        context_text = "\n\n".join([f"Source: {c.source}\n{c.content}" for c in context_chunks])

        system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know."
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        if provider == "OpenAI":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=kwargs.get("model", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI: {e}"

        elif provider == "Azure OpenAI":
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=kwargs.get("api_version", "2023-05-15"),
                    azure_endpoint=kwargs.get("azure_endpoint")
                )
                response = client.chat.completions.create(
                    model=kwargs.get("model"), # In Azure this is the deployment name
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling Azure OpenAI: {e}"

        elif provider == "Anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=kwargs.get("model", "claude-3-opus-20240229"),
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                return f"Error calling Anthropic: {e}"

        elif provider == "Mock":
            return f"Mock response. Context size: {len(context_text)} chars. Query: {query}"

        else:
            return "Provider not supported."

    def get_embedding(self, text: str, provider: str, api_key: str, **kwargs) -> List[float]:
        if provider == "OpenAI":
             try:
                 from openai import OpenAI
                 client = OpenAI(api_key=api_key)
                 response = client.embeddings.create(
                     input=text,
                     model=kwargs.get("embedding_model", "text-embedding-ada-002")
                 )
                 return response.data[0].embedding
             except Exception as e:
                 logger.error(f"OpenAI embedding error: {e}")
                 raise e

        elif provider == "Azure OpenAI":
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=kwargs.get("api_version", "2023-05-15"),
                    azure_endpoint=kwargs.get("azure_endpoint")
                )
                response = client.embeddings.create(
                    input=text,
                    model=kwargs.get("embedding_model") # In Azure this is the deployment name
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Azure OpenAI embedding error: {e}")
                raise e

        elif provider == "Mock":
            # Return a random vector of appropriate size for testing
            # We need to know the target dimension.
            # If we have loaded chunks, we can use their dimension.
            if self.embeddings is not None and len(self.embeddings) > 0:
                dim = self.embeddings.shape[1]
                return list(np.random.rand(dim))
            return [0.1, 0.2, 0.3]

        else:
            # For other providers, we might need specific implementations.
            # Assuming OpenAI compatible for now or failing.
            raise ValueError(f"Embedding provider {provider} not implemented")
