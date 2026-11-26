"""
Memory Bank: Stores and retrieves memory items
"""
import os

from functools import lru_cache
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import copy

# Retriever embedding dimensions
RETRIEVER_EMBEDDING_DIMS = {
    'contriever': 768,
    'dpr': 768,
    'dragon': 1024,
    'Qwen/Qwen3-Embedding-0.6B': 1024,
    # Add more retrievers as needed
}


def canonicalize_retriever_name(retriever_name: str) -> str:
    name = str(retriever_name or "").strip()
    lowered = name.lower()
    if lowered in {
        "qwen/qwen3-embedding-0.6b",
        "qwen_qwen3-embedding-0.6b",
        "qwen3-embedding-0.6b",
    }:
        return "Qwen/Qwen3-Embedding-0.6B"
    if lowered in {"contriever", "dpr", "dragon"}:
        return lowered
    return name


def get_retriever_embedding_dim(retriever_name: str) -> int:
    """Get embedding dimension for a retriever, with fallback"""
    canonical_name = canonicalize_retriever_name(retriever_name)
    return RETRIEVER_EMBEDDING_DIMS.get(canonical_name, 768)


@lru_cache(maxsize=1)
def _skylakex_dot_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dot_rows(matrix, query, partials, rows):
        row = tl.program_id(0)
        lane = tl.arange(0, 8)
        # OpenBLAS SGEMV handles blocks of four rows, then a pair and a final single row.
        full_block = row < rows - rows % 4
        width = tl.where(full_block | (row >= rows - rows % 2), 8, 4)
        acc = tl.full((8,), 0, tl.float32)
        for step in range(1024 // width):
            col = step * width + lane % width
            x = tl.load(matrix + row * 1024 + col)
            y = tl.load(query + col)
            if full_block:
                acc = tl.fma(x, y, acc)
            else:
                acc = acc + x * y
        tl.store(partials + row * 8 + lane, tl.where(lane < width, acc, 0.0))

    return dot_rows


def _numpy_norm_1024(matrix):
    """Reproduce NumPy's contiguous FP32 row reduction on the tensor's device."""
    squares = matrix.square().reshape(-1, 8, 16, 8)
    lanes = squares[:, :, 0, :]
    for step in range(1, 16):
        lanes = lanes + squares[:, :, step, :]
    while lanes.shape[-1] > 1:
        lanes = lanes[..., ::2] + lanes[..., 1::2]
    blocks = lanes.squeeze(-1)
    while blocks.shape[-1] > 1:
        blocks = blocks[:, ::2] + blocks[:, 1::2]
    return blocks.sqrt()


def _skylakex_cuda_order(embeddings, query_embedding, normalize_memories, k):
    """Use 251's single-threaded OpenBLAS 0.3.29 accumulation order for Qwen embeddings."""
    import torch

    matrix = torch.as_tensor(embeddings, device="cuda", dtype=torch.float32).contiguous()
    query = torch.as_tensor(query_embedding, device=matrix.device, dtype=matrix.dtype).reshape(1, -1)
    if matrix.shape[1] != 1024 or query.shape[1] != 1024:
        raise ValueError("cuda-skylakex requires 1024-dimensional memory and query embeddings")
    if len(matrix) == 1:
        return np.arange(1)[:k]
    if normalize_memories:
        matrix = matrix / (_numpy_norm_1024(matrix) + 1e-8)
    query = (query / (_numpy_norm_1024(query) + 1e-8)).contiguous()
    partials = torch.empty((len(matrix), 8), device=matrix.device, dtype=matrix.dtype)
    kernel = _skylakex_dot_kernel()
    kernel[(len(matrix),)](matrix, query, partials, len(matrix), num_warps=1, enable_fp_fusion=False)
    pairs = partials[:, :4] + partials[:, 4:]
    scores = (pairs[:, 0] + pairs[:, 1]) + (pairs[:, 2] + pairs[:, 3])
    # NumPy's default unstable argsort determines which equal-score memory versions are returned.
    return np.argsort(scores.cpu().numpy())[::-1][:k]


class MemoryItem:
    """Single memory item with dual embeddings (retriever + state encoder)."""
    def __init__(self, content: str, embedding: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None,
                 state_encoder_embedding: Optional[np.ndarray] = None):
        self.content = content
        self.embedding = embedding  # Retriever embedding (for QA evaluation)
        self.state_encoder_embedding = state_encoder_embedding
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = 0  # timestep
        self.created_at = 0
        # History tracking
        self.content_history: List[str] = []  # Previous contents before updates (chronological order)
        self.operation_history: List[str] = []  # Sequence of operations applied (e.g., ['insert', 'update', 'update'])

    def to_dict(self):
        return {
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'state_encoder_embedding': (
                self.state_encoder_embedding.tolist()
                if self.state_encoder_embedding is not None else None
            ),
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'created_at': self.created_at,
            'content_history': self.content_history,
            'operation_history': self.operation_history
        }

    @classmethod
    def from_dict(cls, data):
        state_emb = data.get('state_encoder_embedding')
        if state_emb is None:
            for key, value in data.items():
                if key.endswith('_embedding') and key not in ('embedding', 'state_encoder_embedding'):
                    state_emb = value
                    break
        item = cls(
            content=data['content'],
            embedding=np.array(data['embedding']) if data.get('embedding') is not None else None,
            metadata=data.get('metadata', {}),
            state_encoder_embedding=(
                np.array(state_emb) if state_emb is not None else None
            )
        )
        item.access_count = data.get('access_count', 0)
        item.last_accessed = data.get('last_accessed', 0)
        item.created_at = data.get('created_at', 0)
        item.content_history = data.get('content_history', [])
        item.operation_history = data.get('operation_history', [])
        return item


class MemoryBank:
    """
    Memory Bank stores memory items and supports retrieval via semantic search.
    Supports dual embeddings: retriever (for QA eval) and state encoder (for training).
    """
    def __init__(self, retriever_name: str = 'contriever', top_k: int = 5,
                 state_encoder=None):
        self.memories: List[MemoryItem] = []
        self.retriever_name = retriever_name
        self.top_k = top_k
        self.trace = None
        self.timestep = 0
        self._faiss_index = None
        self._state_encoder = state_encoder  # StateEncoder for computing state embeddings

    @property
    def state_encoder(self):
        return self._state_encoder

    @state_encoder.setter
    def state_encoder(self, encoder):
        self._state_encoder = encoder

    def set_state_encoder(self, encoder):
        """Set the state encoder for computing embeddings."""
        self.state_encoder = encoder

    def _compute_state_encoder_embedding(self, content: str) -> Optional[np.ndarray]:
        """Compute state encoder embedding for a single content string."""
        if self.state_encoder is None:
            return None
        emb = self.state_encoder._encode_texts(content)
        if emb.ndim == 2:
            return emb[0]
        return emb

    def initialize_from_sessions(self, sessions: List[str], embeddings: np.ndarray,
                                  state_encoder_embeddings: Optional[np.ndarray] = None):
        """Initialize memory bank from conversation sessions/turns."""
        assert len(sessions) == embeddings.shape[0], "Sessions and embeddings length mismatch"

        self.memories = []
        for i, (session, emb) in enumerate(zip(sessions, embeddings)):
            state_emb = state_encoder_embeddings[i] if state_encoder_embeddings is not None else None
            memory_item = MemoryItem(
                content=session,
                embedding=emb,
                state_encoder_embedding=state_emb,
                metadata={'source': 'initial', 'session_idx': i}
            )
            memory_item.created_at = self.timestep
            self.memories.append(memory_item)

        self._rebuild_index()

    def add_memory(self, content: str, embedding: np.ndarray,
                   metadata: Optional[Dict] = None,
                   operation_name: Optional[Union[str, List[str]]] = None,
                   state_encoder_embedding: Optional[np.ndarray] = None):
        """Add a new memory item with optional state encoder embedding.

        Args:
            content: Memory content text
            embedding: Retriever embedding
            metadata: Optional metadata dict
            operation_name: Operation name or list of names that created this memory
        """
        # Compute state encoder embedding if encoder is available and not provided
        if state_encoder_embedding is None and self.state_encoder is not None:
            state_encoder_embedding = self._compute_state_encoder_embedding(content)

        memory_item = MemoryItem(
            content=content,
            embedding=embedding,
            state_encoder_embedding=state_encoder_embedding,
            metadata=metadata
        )
        # print(memory_item.content)
        memory_item.created_at = self.timestep
        # Record the operation that created this memory
        if operation_name:
            if isinstance(operation_name, (list, tuple, set)):
                for name in operation_name:
                    if name:
                        memory_item.operation_history.append(str(name))
            else:
                memory_item.operation_history.append(str(operation_name))
        self.memories.append(memory_item)
        if self.trace is not None:
            self.trace.change("INSERT", len(self.memories)-1, None, content)
        self._rebuild_index()

    def update_memory(self, index: int, new_content: str, new_embedding: np.ndarray,
                      operation_name: Optional[Union[str, List[str]]] = None,
                      new_state_encoder_embedding: Optional[np.ndarray] = None):
        """Update an existing memory item.

        Args:
            index: Index of the memory to update
            new_content: New content text
            new_embedding: New retriever embedding
            operation_name: Operation name or list of names that performed this update
        """
        if 0 <= index < len(self.memories):
            mem = self.memories[index]
            # Save old content to history before updating
            mem.content_history.append(mem.content)
            # Update content and embeddings
            mem.content = new_content
            mem.embedding = new_embedding
            # Compute state encoder embedding if encoder is available and not provided
            if new_state_encoder_embedding is None and self.state_encoder is not None:
                new_state_encoder_embedding = self._compute_state_encoder_embedding(new_content)
            mem.state_encoder_embedding = new_state_encoder_embedding
            mem.last_accessed = self.timestep
            # Record the operation
            if operation_name:
                if isinstance(operation_name, (list, tuple, set)):
                    for name in operation_name:
                        if name:
                            mem.operation_history.append(str(name))
                else:
                    mem.operation_history.append(str(operation_name))
            if self.trace is not None:
                self.trace.change("UPDATE", index, mem.content_history[-1], new_content)
            self._rebuild_index()
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def delete_memory(self, index: int):
        """Delete a memory item"""
        if 0 <= index < len(self.memories):
            if self.trace is not None:
                self.trace.change("DELETE", index, self.memories[index].content, None)
            del self.memories[index]
            self._rebuild_index()
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def retrieve(self, query_embedding: np.ndarray, top_k: Optional[int] = None,
                 use_state_encoder: bool = False, return_embeddings: bool = False) -> Union[
                     Tuple[List[str], List[int]],
                     Tuple[List[str], List[int], Optional[np.ndarray]]
                 ]:
        """
        Retrieve top-k most relevant memories

        Args:
            query_embedding: Query embedding vector
            top_k: Number of memories to retrieve (default: self.top_k)
            use_state_encoder: If True, use state encoder embeddings for retrieval (training).
                              If False, use retriever embeddings (QA evaluation).
            return_embeddings: If True, also return embeddings aligned with retrieved memories.

        Returns:
            (memory_contents, memory_indices) or
            (memory_contents, memory_indices, memory_embeddings) if return_embeddings=True
        """
        if top_k is None:
            top_k = self.top_k

        if len(self.memories) == 0:
            if return_embeddings:
                return [], [], None
            return [], []

        # Get embeddings matrix based on embedding type
        embeddings = self._get_embeddings_matrix(use_state_encoder=use_state_encoder)
        normalize_memories = use_state_encoder or self.retriever_name == 'dragon'
        actual_k = min(top_k, len(self.memories))
        top_indices = None
        search = os.environ.get("MEMSKILL_MEMORY_SEARCH")
        if search == "cuda-skylakex":
            top_indices = _skylakex_cuda_order(embeddings, query_embedding, normalize_memories, actual_k)
        elif search == "cuda":
            import torch

            matrix = torch.as_tensor(embeddings, device="cuda", dtype=torch.float32)
            query = torch.as_tensor(query_embedding, device=matrix.device, dtype=torch.float32).reshape(-1)
            if normalize_memories:
                matrix = matrix / (torch.linalg.vector_norm(matrix, dim=1, keepdim=True) + 1e-8)
            query = query / (torch.linalg.vector_norm(query) + 1e-8)
            similarities = torch.mv(matrix, query)
            values, indices = similarities.sort(descending=True)
            boundary = values[:actual_k + 1]
            # A changed order of equal/near-equal scores can change which memory version gets credit.
            # Bound float32 norm/dot rounding by the vector length and the largest possible inner product.
            max_norm = 1.0 if normalize_memories else torch.linalg.vector_norm(matrix, dim=1).amax()
            margin = 4 * torch.finfo(matrix.dtype).eps * matrix.shape[1] * max_norm
            gaps = boundary[:-1] - boundary[1:]
            if actual_k >= 0 and (torch.isfinite(boundary).all() & (gaps > margin).all()).item():
                top_indices = indices[:actual_k].cpu().numpy()

        if top_indices is None:
            # Preserve upstream BLAS accumulation and NumPy's tie order when ranking is ambiguous.
            embeddings_for_sim = embeddings
            if normalize_memories:
                emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings_for_sim = embeddings / (emb_norms + 1e-8)
            q = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(q_norm, embeddings_for_sim.T)[0]
            top_indices = np.argsort(similarities)[::-1][:actual_k]
        # Update access statistics
        for idx in top_indices:
            self.memories[idx].access_count += 1
            self.memories[idx].last_accessed = self.timestep

        # Return contents and indices
        contents = [self.memories[idx].content for idx in top_indices]
        if return_embeddings:
            if len(top_indices) > 0:
                retrieved_embeddings = embeddings[top_indices].copy()
            else:
                retrieved_embeddings = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
            return contents, top_indices.tolist(), retrieved_embeddings
        return contents, top_indices.tolist()

    def _get_embeddings_matrix(self, use_state_encoder: bool = False) -> np.ndarray:
        """
        Get embeddings as a matrix [N, D]

        Args:
            use_state_encoder: If True, use state encoder embeddings. Otherwise use retriever embeddings.
        """
        embeddings = []
        embedding_dim = None

        # First pass: collect embeddings and determine dimension
        for mem in self.memories:
            if use_state_encoder:
                emb = mem.state_encoder_embedding
            else:
                emb = mem.embedding

            if emb is not None:
                embeddings.append(emb)
                if embedding_dim is None:
                    embedding_dim = emb.shape[0]
            else:
                embeddings.append(None)  # Placeholder for missing embeddings

        # Fallback dimension based on retriever if no embeddings found
        if embedding_dim is None:
            if use_state_encoder:
                if self.state_encoder is not None:
                    embedding_dim = getattr(self.state_encoder, 'embedding_dim', None)
                if embedding_dim is None:
                    embedding_dim = 768  # Default state encoder hidden size
            else:
                embedding_dim = get_retriever_embedding_dim(self.retriever_name)

        # Second pass: replace None with zero vectors of correct dimension
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = np.zeros(embedding_dim, dtype=np.float32)

        return np.vstack(embeddings).astype('float32')

    def _rebuild_index(self):
        """Rebuild FAISS index if needed"""
        # Optional: build FAISS index for faster retrieval
        # For now, we use simple numpy operations which are sufficient for moderate sizes
        self._faiss_index = None

    def get_all_contents(self) -> List[str]:
        """Get all memory contents"""
        return [mem.content for mem in self.memories]

    def get_memory_at(self, index: int) -> MemoryItem:
        """Get memory item at specific index"""
        if 0 <= index < len(self.memories):
            return self.memories[index]
        else:
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")

    def __len__(self):
        return len(self.memories)

    def step(self):
        """Increment timestep"""
        self.timestep += 1

    def to_dict(self):
        """Serialize to dict"""
        return {
            'memories': [mem.to_dict() for mem in self.memories],
            'retriever_name': self.retriever_name,
            'top_k': self.top_k,
            'timestep': self.timestep
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict"""
        bank = cls(
            retriever_name=data.get('retriever_name', 'contriever'),
            top_k=data.get('top_k', 5)
        )
        bank.memories = [MemoryItem.from_dict(m) for m in data.get('memories', [])]
        bank.timestep = data.get('timestep', 0)
        bank._rebuild_index()
        return bank

    def copy(self):
        """Create a deep copy of the memory bank"""
        return copy.deepcopy(self)
