# search_engine.py
import os
import pickle
from typing import List
import jieba
from Engine_copy import InvertedIndex, build_or_load_index, filename_to_url, HTML_PATH, INDEX_FILE, get_postings_list

# 全局缓存，模块导入时加载一次
_inv = None
_files = []

def _is_valid_index(inv) -> bool:
    """简单校验 pickled 对象是不是我们能用的索引"""
    try:
        return isinstance(inv, InvertedIndex) \
               and getattr(inv, "doc_count", 0) > 0 \
               and isinstance(getattr(inv, "doc_length", None), list) \
               and len(inv.doc_length) == inv.doc_count
    except Exception:
        return False

def _load_index_once():
    global _inv, _files
    if _inv is not None:
        return
    
    try:
        inv = build_or_load_index()
        if not _is_valid_index(inv):
            raise ValueError("Loaded index not valid, rebuilding.")
    except Exception:
        try:
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
        except Exception:
            pass
        inv = build_or_load_index()
    if not hasattr(inv, "avgdl") or getattr(inv, "avgdl", 0) == 0:
        try:
            inv.compute_avgdl()
        except Exception:
            pass

    _inv = inv
    _files = [f for f in os.listdir(HTML_PATH) if f.endswith(('.html', '.htm'))]
    _files.sort()

_load_index_once()


def evaluate(query: str) -> List[str]:
    global _inv, _files
    if _inv is None:
        _load_index_once()

    # 判断 AND/OR
    and_queries = ["和", "与", "且", "并且", "以及", "同时"]
    or_queries  = ["或", "或者", "而且"]
    mode = "OR"
    if any(op in query for op in and_queries):
        mode = "AND"
    elif any(op in query for op in or_queries):
        mode = "OR"

    # 去掉连接词
    query_terms = query
    for op in and_queries + or_queries:
        query_terms = query_terms.replace(op, " ")
    query_terms = " ".join(query_terms.split())

    # 检索
    if mode == "AND":
        # 找交集
        doc_sets = [set(p.docid for p in get_postings_list(_inv.index, t))
                    for t in jieba.lcut(query_terms) if t in _inv.index]
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        results = _inv.bm25_search(query_terms, k=20, allowed_docs=common_docs)
    else:
        results = _inv.bm25_search(query_terms, k=20)

    # 转 URL
    urls = []
    for docid, score in results:
        if 0 <= docid < len(_files):
            urls.append(filename_to_url.get(_files[docid], _files[docid]))
    while len(urls) < 20:
        urls.append(urls[-1] if urls else "")
    return urls[:20]
