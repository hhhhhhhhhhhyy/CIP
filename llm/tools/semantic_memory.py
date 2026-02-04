try:
    from pypinyin import lazy_pinyin
except ModuleNotFoundError:
    lazy_pinyin = None
from bs4 import BeautifulSoup
import pickle
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_store"
MODEL_PATH = PROJECT_ROOT / "m3e-base"
print("Chroma DB path:", CHROMA_PATH)
print("Model path:", MODEL_PATH)
import chromadb

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=str(MODEL_PATH))


def cover_name(name):

    s = str(name) if name is not None else ""

    if lazy_pinyin is not None and any("\u4e00" <= ch <= "\u9fff" for ch in s):
        return "_".join(lazy_pinyin(s))

    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower())
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    normalized = "".join(out).strip("_")
    return normalized or "collection"


def _normalize_collection_name(name: str | None) -> str:
    """Keep already-safe names as-is; normalize others for Chroma."""

    if name is None:
        return "SemanticMemory"
    s = str(name)
    if s and all(ch.isalnum() or ch in "_-" for ch in s):
        return s
    return cover_name(s)

class SemanticMemory:

    def __init__(self, name: str = "SemanticMemory"):

        self.name = _normalize_collection_name(name)
        self.collection = client.get_or_create_collection(self.name, embedding_function=embedding_fn,  metadata={"metric": "cosine"})


    def store_step(
        self,
        step_type: str,
        step_ai_desc: str,
        step_raw_desc: str,
        step_success_reason: str,
    ) -> str:

        step_type = "" if step_type is None else str(step_type)
        step_ai_desc = "" if step_ai_desc is None else str(step_ai_desc)
        step_raw_desc = "" if step_raw_desc is None else str(step_raw_desc)
        step_success_reason = "" if step_success_reason is None else str(step_success_reason)

        # Dedupe: if this raw step already exists, do not store again.
        raw_key = step_raw_desc.strip()
        if raw_key:
            try:
                existing = self.collection.get(where={"step_raw_desc": raw_key}, include=[])
                if isinstance(existing, dict):
                    existing_ids = existing.get("ids", []) or []
                    if existing_ids:
                        return existing_ids[0]
            except Exception:
                # If `get(where=...)` isn't supported in this chroma version, fall back to query.
                try:
                    hit = self.collection.query(
                        query_texts=[raw_key],
                        n_results=3,
                        include=["metadatas"],
                        where={"step_raw_desc": raw_key},
                    )
                    ids0 = (hit.get("ids", [[]]) or [[]])[0]
                    if ids0:
                        return ids0[0]
                except Exception:
                    pass

        doc = (
            f"STEP_TYPE: {step_type}\n"
            f"STEP_RAW_DESC: {step_raw_desc}\n"
            f"STEP_AI_DESC: {step_ai_desc}\n"
            f"STEP_SUCCESS_REASON: {step_success_reason}\n"
        )

        meta = {
            "step_type": step_type,
            "step_ai_desc": step_ai_desc,
            "step_raw_desc": raw_key,
            "step_success_reason": step_success_reason,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }


        record_id = f"{self.name}_{int(datetime.utcnow().timestamp() * 1000)}"

        try:
            self.collection.upsert(ids=[record_id], documents=[doc], metadatas=[meta])
        except Exception:
            # Some chroma versions/collections may not support upsert; fall back to add.
            self.collection.add(ids=[record_id], documents=[doc], metadatas=[meta])

        return record_id

    def query_steps(self, query: str, topn: int = 3, where: dict | None = None) -> pd.DataFrame:

        query = "" if query is None else str(query)
        if not query.strip():
            return pd.DataFrame(
                [],
                columns=[
                    "id",
                    "distance",
                    "step_type",
                    "step_raw_desc",
                    "step_ai_desc",
                    "step_success_reason",
                ],
            )

        kwargs = {
            "query_texts": [query],
            "n_results": int(topn),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0] if "ids" in result else [None] * len(docs)
        dists = result.get("distances", [[]])[0] if "distances" in result else [None] * len(docs)

        rows = []
        for i in range(len(docs)):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            rows.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "distance": dists[i] if i < len(dists) else None,
                    "step_type": meta.get("step_type", ""),
                    "step_raw_desc": meta.get("step_raw_desc", ""),
                    "step_ai_desc": meta.get("step_ai_desc", ""),
                    "step_success_reason": meta.get("step_success_reason", ""),
                }
            )

        # Sort by distance (smaller = closer for cosine)
        rows = sorted(rows, key=lambda x: (x["distance"] is None, x["distance"]))
        return pd.DataFrame(rows)

    def search(self, content, topn=3):

        if isinstance(content, str):

            result = self.collection.query(query_texts=[content], n_results=topn, include=["documents", "metadatas", "distances"])
        else:

            emb = np.array(content, dtype='float32')

            if emb.ndim > 1:
                emb = emb.flatten()
            result = self.collection.query(query_embeddings=[emb.tolist()], n_results=topn, include=["documents", "metadatas", "distances"])


        docs = result.get('documents', [[]])[0]
        metas = result.get('metadatas', [[]])[0]
        ids = result.get('ids', [[]])[0] if 'ids' in result else [None] * len(docs)
        dists = result.get('distances', [[]])[0] if 'distances' in result else [None] * len(docs)

        rows = []
        for i in range(len(docs)):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            rows.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "document": docs[i] if i < len(docs) else None,
                    "distance": dists[i] if i < len(dists) else None,
                    **meta,
                }
            )
        rows_sorted = sorted(rows, key=lambda x: (x["distance"] is None, x["distance"]))
        rows_sorted = pd.DataFrame(rows_sorted)

        return rows_sorted


    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            # Older chroma versions still usually support count(); if not, fall back to best-effort.
            try:
                result = self.collection.get(limit=1, include=[])
                ids = result.get("ids", []) if isinstance(result, dict) else []
                return len(ids)
            except Exception:
                return 0


    def get_all(
        self,
        where: dict | None = None,
        batch_size: int = 1000,
        limit: int | None = None,
        include_documents: bool = True,
        include_embeddings: bool = False,
    ) -> pd.DataFrame:
        """Fetch the entire collection (paged) as a DataFrame.

        Notes:
        - Embeddings can be very large; keep `include_embeddings=False` unless you really need them.
        - `limit` caps total returned rows.
        """

        include = ["metadatas"]
        if include_documents:
            include.append("documents")
        if include_embeddings:
            include.append("embeddings")

        all_rows: list[dict] = []
        offset = 0
        remaining = None if limit is None else int(limit)

        while True:
            if remaining is not None and remaining <= 0:
                break

            page_limit = int(batch_size)
            if remaining is not None:
                page_limit = min(page_limit, remaining)

            kwargs = {"include": include, "limit": page_limit, "offset": offset}
            if where:
                kwargs["where"] = where

            try:
                result = self.collection.get(**kwargs)
            except TypeError:
                # Some chroma versions don't support offset/limit in get(); fall back to peek.
                result = self.collection.peek(limit=page_limit)
            except Exception:
                break

            if not isinstance(result, dict):
                break

            ids = result.get("ids", []) or []
            if not ids:
                break

            documents = result.get("documents", None)
            metadatas = result.get("metadatas", None)
            embeddings = result.get("embeddings", None)

            for i in range(len(ids)):
                row = {"id": ids[i]}
                if include_documents:
                    row["document"] = documents[i] if isinstance(documents, list) and i < len(documents) else None
                if include_embeddings:
                    row["embedding"] = embeddings[i] if isinstance(embeddings, list) and i < len(embeddings) else None
                if isinstance(metadatas, list) and i < len(metadatas) and metadatas[i]:
                    row.update(metadatas[i])
                all_rows.append(row)

            offset += len(ids)
            if remaining is not None:
                remaining -= len(ids)

            if len(ids) < page_limit:
                break

        return pd.DataFrame(all_rows)


def search_serviceteams_two_stage(text, topn=3):
    raise NotImplementedError("Two-stage serviceteams search is not implemented in this workspace")


class ChromaRAGDatabase:
    def __init__(self, name=None):
        if name is None:
            name = "incidents"

        self.name = name
        self.emb_database = None

    def create_emb_database(self):
        self.emb_database = SemanticMemory(self.name)

    def search(self, text, topn=3):

        if not self.emb_database:
            self.emb_database = SemanticMemory(name=self.name)
        return self.emb_database.search(text, topn)


if __name__ == "__main__":

    dbRAGDatabase = ChromaRAGDatabase(name="incidents")
    aa = dbRAGDatabase.search("[Global Support] [Web UI] | Fr", topn=3)
    print(aa)