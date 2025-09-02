# search_engine.py  
import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import jieba
import heapq
from collections import defaultdict, Counter
import numpy as np
import pickle
import json
from tqdm import tqdm
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "htmls")  
INDEX_FILE = os.path.join(BASE_DIR, "index.pkl")
URL_FILE = os.path.join(BASE_DIR, "htmls", "urls.json")

with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f)

with open(URL_FILE, "r", encoding="utf-8") as f:
    url_list = json.load(f)
filename_to_url = {item["filename"]: item["url"] for item in url_list}

log_func = np.vectorize(lambda x: 1.0 + np.log(x) if x > 0 else 0.0)

def _safe_pickle_dump(obj, path):
    """原子性写入"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  
        print(f"DEBUG: 成功写入索引文件 -> {path}")
    except Exception as e:
        # 清理 tmp
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        print("ERROR: 写入 index.pkl 失败：", repr(e))
        raise

def get_text(fpath: str) -> Tuple[str, str]:
    """读取HTML并返回 (标题, 正文+meta信息)"""
    with open(fpath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title else ""

    # 针对科研处等情况
    if title in ["科研处", "中国人民大学科研处"]:
        h3 = soup.find("h3")
        if h3 and h3.get_text(strip=True):
            title = h3.get_text(strip=True)

    # 提取正文
    body = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

    # 提取 meta keywords 和 description
    meta_keywords = " ".join([m["content"] for m in soup.find_all("meta", attrs={"name": "keywords"}) if m.get("content")])
    meta_description = " ".join([m["content"] for m in soup.find_all("meta", attrs={"name": "description"}) if m.get("content")])

    # 将 meta 信息拼接到正文（可以给它加权重）
    extra = (meta_keywords + " ") * 3 + (meta_description + " ") * 2
    body = extra + body  

    return title, body


    return title, body

def get_postings_list(inverted_index: Dict[str, List[Dict[str, int]]], term: str) -> List[Dict[str, int]]:
    """获取某个词的倒排列表"""
    try:
        return inverted_index[term]
    except KeyError:
        return []
    

class postings:
    def __init__(self, docid: int, tf: int):
        self.docid = docid
        self.tf = tf

    def __repr__(self):
        return f"({self.docid}, {self.tf})"
    
class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[postings]] = defaultdict(list)
        self.doc_count = 0   
        self.df = {}         
        self.idf = {}       
        self.doc_length = []

    def get_avgdl(self):
        self.avgdl = sum(self.doc_length) / max(1, self.doc_count)

    def bm25_search(self, query: str, k1=0.6, b=0.6, k=20, allowed_docs: set=None):
        # 若尚未计算 avgdl，自动计算
        if not getattr(self, "avgdl", 0):
            try:
                self.get_avgdl()
            except Exception:
                self.avgdl = 1.0 

        tokens = [t for t in jieba.cut_for_search(query) if len(t) > 1 and t not in stopwords]
        qtf = Counter(tokens)
        scores = defaultdict(float)

        for i, (term, qf) in enumerate(qtf.items()):
            posting_list = self.index.get(term, [])
            df = len(posting_list)
            if df == 0:
                continue

            idf = max(0, np.log((self.doc_count - df + 0.5) / (df + 0.5)))

            for p in posting_list:
                if allowed_docs is not None and p.docid not in allowed_docs:
                    continue
                tf = p.tf
                dl = self.doc_length[p.docid] if p.docid < len(self.doc_length) else self.avgdl
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                score = idf * tf * (k1 + 1) / denom
                weight = 1.2 if i == 0 else 1.0
                scores[p.docid] += score * qf * weight

        topk = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
        return [(docid, score) for docid, score in topk]


    def add(self, word: str, docid: int) -> None:
        """把一个词加入倒排索引"""
        if self.index[word] and self.index[word][-1].docid == docid:
            self.index[word][-1].tf += 1
        else:
            self.index[word].append(postings(docid, 1))

    def build_index(self, html_path: str):
        """从文档集合里构建倒排索引"""
        files = [f for f in os.listdir(html_path) if f.endswith(('.html', '.htm'))]
        files.sort()
        self.doc_count = len(files)

        for docid, filename in enumerate(tqdm(files, desc="Building index")):
            title, content = get_text(os.path.join(html_path, filename))
            terms = jieba.lcut(content)  
            seen = set()
            for term in terms:
                term = term.strip()
                if len(term) <= 1: 
                    continue
                if not re.match(r"^[\u4e00-\u9fa5a-zA-Z0-9]+$", term): 
                    continue

                self.add(term, docid)
                if term not in seen:
                    self.df[term] = self.df.get(term, 0) + 1
                    seen.add(term)

        for term, df in self.df.items():
            self.idf[term] = np.log(self.doc_count / (1 + df))

        return self.index, files
    
    def get_length(self, html_path: str) -> List[float]:
        self.doc_length = [] 
        htmls = [f for f in os.listdir(html_path) if f.endswith(('.html', '.htm'))]
        htmls.sort()

        for docid, filename in enumerate(htmls):
            title, content = get_text(os.path.join(html_path, filename))
            terms = jieba.lcut(content)

            term_counts = Counter()
            for term in terms:
                term = term.strip()
                if len(term) <= 1: continue
                if not re.match(r"^[\u4e00-\u9fa5a-zA-Z0-9]+$", term): continue
                term_counts[term] += 1

            length_sq = 0.0
            for term, tf in term_counts.items():
                if term not in self.idf:  
                    continue
                w = (1 + np.log(tf)) * self.idf[term]  
                length_sq += w * w

            self.doc_length.append(np.sqrt(length_sq))

        return self.doc_length

def build_or_load_index():
    if os.path.exists(INDEX_FILE) and os.path.getsize(INDEX_FILE) > 0:
        try:
            with open(INDEX_FILE, "rb") as f:
                inv = pickle.load(f)
            if isinstance(inv, InvertedIndex):
                print("成功加载index.pkl")
                # 确保 avgdl 存在
                if not getattr(inv, "avgdl", 0):
                    try:
                        inv.get_avgdl_avgdl()
                    except Exception as e:
                        print("get_avgdl 异常:", repr(e))
                return inv
            else:
                print("index.pkl 内容不是 InvertedIndex，重建索引")
        except Exception as e:
            print("加载失败：", repr(e))
            
    print("正在构建索引")
    inv = InvertedIndex()
    inv.build_index(HTML_PATH)
    inv.get_length(HTML_PATH)
    inv.get_avgdl()

    try:
        _safe_pickle_dump(inv, INDEX_FILE)
    except Exception:
        import traceback; traceback.print_exc()
        print("虽然构建成功但保存 index.pkl 失败")

    print("索引构建完成")
    return inv

def query_or(inverted_index: InvertedIndex, files_path: str, query_term: str, k: int = 20):
    results = inverted_index.bm25_search(query_term, k=k)
    
    output = []
    files = [f for f in os.listdir(files_path) if f.endswith(('.html', '.htm'))]
    files.sort()
    
    for docid, score in results:
        fname = files[docid]
        fpath = os.path.join(files_path, fname)
        
        title, body = get_text(fpath)
        snippet = body[:150] + "..." if len(body) > 150 else body
        
        url = filename_to_url.get(fname, fname)
        
        output.append({
            "title": title if title else fname,
            "snippet": snippet,
            "url": url
        })
    
    return output


def query_and(inverted_index: InvertedIndex, files_path: str, query_term: str, k: int = 20):
    query_terms = [t for t in jieba.cut_for_search(query_term) if t in inverted_index.index]
    if not query_terms:
        return []

    doc_sets = [set(p.docid for p in get_postings_list(inverted_index.index, t)) for t in query_terms]
    common_docs = set.intersection(*doc_sets)
    if not common_docs:
        return []

    results = inverted_index.bm25_search(query_term, k=k, allowed_docs=common_docs)

    output = []
    files = [f for f in os.listdir(files_path) if f.endswith(('.html', '.htm'))]
    files.sort()

    for docid, score in results:
        fname = files[docid]
        fpath = os.path.join(files_path, fname)

        title, body = get_text(fpath)
        snippet = body[:150] + "..." if len(body) > 150 else body

        url = filename_to_url.get(fname, fname)

        output.append({
            "title": title if title else fname,
            "snippet": snippet,
            "url": url
        })

    return output


def query_bm25(inverted_index: InvertedIndex, files_path: str, query_term: str, k: int = 10):
    files = [f for f in os.listdir(files_path) if f.endswith(('.html', '.htm'))]
    files.sort()
    results = inverted_index.bm25_search(query_term, k=k)

    res = []
    for docid, score in results:
        fname = files[docid]
        url = filename_to_url.get(fname, None)
        res.append((url if url else fname, score))

    return res

_inv = None
_files = []

def _is_valid_index(inv) -> bool:
    """校验 pickled 对象是不是可用的索引"""
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
            inv.get_avgdl()
        except Exception:
            pass

    _inv = inv
    _files = [f for f in os.listdir(HTML_PATH) if f.endswith(('.html', '.htm'))]
    _files.sort()

def ensure_index_loaded(func):
    def wrapper(*args, **kwargs):
        global _inv, _files
        if _inv is None or not _files:
            _load_index_once()
        return func(*args, **kwargs)
    return wrapper

@ensure_index_loaded
def evaluate(query: str) -> List[str]:
    global _inv, _files
    # 判断 AND/OR
    and_queries = ["和", "与", "且", "并且", "以及", "同时"]
    or_queries  = ["或", "或者", "而且"]
    mode = "OR"
    if any(op in query for op in and_queries):
        mode = "AND"
    elif any(op in query for op in or_queries):
        mode = "OR"

    query_terms = query
    for op in and_queries + or_queries:
        query_terms = query_terms.replace(op, " ")
    query_terms = " ".join(query_terms.split())

    if mode == "AND":
        doc_sets = [set(p.docid for p in get_postings_list(_inv.index, t))
                    for t in jieba.cut_for_search(query_terms) if t in _inv.index]
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        results = _inv.bm25_search(query_terms, k=20, allowed_docs=common_docs)
    else:
        results = _inv.bm25_search(query_terms, k=20)

    urls = []
    for docid, score in results:
        if 0 <= docid < len(_files):
            urls.append(filename_to_url.get(_files[docid], _files[docid]))
    return urls
