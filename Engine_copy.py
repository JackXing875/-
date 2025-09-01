import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import jieba
import heapq
import os
from collections import defaultdict, Counter
import numpy as np
import pickle
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import jieba
import re
import json
import os
from typing import List, Dict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 把这些用绝对路径（避免从 client.py 不同的 cwd 导致写到别处）
HTML_PATH = os.path.join(BASE_DIR, "htmls")   # 如果你 htmls 在别处，请改这里
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
        os.replace(tmp, path)  # 原子替换
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
    """读取HTML"""
    with open(fpath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")

    # 标题
    title = soup.title.string.strip() if soup.title else ""

    if title in ["科研处", "中国人民大学科研处"]:
        h3 = soup.find("h3")
        if h3 and h3.get_text(strip=True):
            title = h3.get_text(strip=True)

    body = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

    return title, body

def get_postings_list(inverted_index: Dict[str, List[Dict[str, int]]], term: str) -> List[Dict[str, int]]:
    """获取某个词的倒排列表"""
    try:
        return inverted_index[term]
    except KeyError:
        return []
    

def cosine_scores(inverted_index: Dict[str, List[Dict[str, int]]], query: str, doc_count: int, doc_length: List[float],  allowed_docs: set = None, k: int = 10) -> List[tuple]:
    """计算余弦相似度得分"""
    scores = defaultdict(lambda: 0.0)
    query_terms = Counter(term for term in jieba.cut(query) if len(term) > 1 and term not in stopwords)
    for q in query_terms:
        postings_list = get_postings_list(inverted_index, q)
        idf = np.log(doc_count / (1 + len(postings_list))) if postings_list else 0.0
        wq = (1 + np.log(query_terms[q])) * idf
        for p in postings_list:
            tf = p.tf
            docid = p.docid
            if allowed_docs is not None and docid not in allowed_docs:
                continue
            wd = (1 + np.log(tf)) * idf
            scores[docid] += wq * wd
    
    res = [(docid, score / doc_length[docid]) for docid, score in scores.items() if doc_length[docid] > 0]
    res.sort(key=lambda x: -x[1])
    return res[:k]


def retrival_by_cosine_sim(inverted_index: Dict[str, List[Dict[str, int]]], files: List[str], query: str, doc_length: List[float], allowed_docs: set = None, k: int = 10) -> List[str]:
    top_scores = cosine_scores(inverted_index, query, len(files), doc_length, allowed_docs = allowed_docs, k = k)
    res = [(files[docid], score) for docid, score in top_scores]
    return res


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

    def compute_avgdl(self):
        self.avgdl = sum(self.doc_length) / max(1, self.doc_count)

    def bm25_search(self, query: str, k1=1.5, b=0.8, k=20, allowed_docs: set=None):
        # 若尚未计算 avgdl，自动计算一次
        if not getattr(self, "avgdl", 0):
            try:
                self.compute_avgdl()
            except Exception:
                self.avgdl = 1.0 

        tokens = [t for t in jieba.lcut(query) if len(t) > 1 and t not in stopwords]
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

        print(f"共发现 {self.doc_count} 个文档，开始建立索引…")

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
            with open(os.path.join(html_path, filename), "r", encoding="utf-8") as f:
                terms = jieba.lcut(get_text(os.path.join(html_path, filename)))

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
    print("DEBUG: 当前工作目录 cwd =", os.getcwd())
    print("DEBUG: INDEX_FILE 绝对路径 =", INDEX_FILE)

    if os.path.exists(INDEX_FILE) and os.path.getsize(INDEX_FILE) > 0:
        try:
            with open(INDEX_FILE, "rb") as f:
                inv = pickle.load(f)
            if isinstance(inv, InvertedIndex):
                print("DEBUG: 成功加载现有 index.pkl")
                # 确保 avgdl 存在
                if not getattr(inv, "avgdl", 0):
                    try:
                        inv.compute_avgdl()
                    except Exception as e:
                        print("DEBUG: compute_avgdl 异常:", repr(e))
                return inv
            else:
                print("DEBUG: index.pkl 内容不是 InvertedIndex（可能旧格式或在 __main__ 时保存），将重建索引")
        except Exception as e:
            print("DEBUG: 加载 index.pkl 失败，错误：", repr(e))
            

    
    print("DEBUG: 正在构建索引，这可能需要一段时间…")
    inv = InvertedIndex()
    inv.build_index(HTML_PATH)
    inv.get_length(HTML_PATH)
    inv.compute_avgdl()

    
    try:
        dirpath = os.path.dirname(INDEX_FILE) or "."
        if not os.access(dirpath, os.W_OK):
            print(f"WARNING: 没有写权限到目录：{dirpath}")
    except Exception:
        pass


    try:
        _safe_pickle_dump(inv, INDEX_FILE)
    except Exception:
        import traceback; traceback.print_exc()
        print("WARNING: 虽然构建成功但保存 index.pkl 失败（请检查权限与磁盘）。返回内存索引。")

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
        
        # 提取标题和正文
        title, body = get_text(fpath)
        snippet = body[:150] + "..." if len(body) > 150 else body
        
        # 转换文件名为url
        url = filename_to_url.get(fname, fname)
        
        output.append({
            "title": title if title else fname,
            "snippet": snippet,
            "url": url
        })
    
    return output



def query_and(inverted_index: InvertedIndex, files_path: str, query_term: str, k: int = 20):
    # 分词并过滤出在索引里的词
    query_terms = [t for t in jieba.lcut(query_term) if t in inverted_index.index]
    if not query_terms:
        return []

    # 获取每个词的文档集合
    doc_sets = [set(p.docid for p in get_postings_list(inverted_index.index, t)) for t in query_terms]
    common_docs = set.intersection(*doc_sets)
    if not common_docs:
        return []

    # 只在交集文档中做 BM25 检索
    results = inverted_index.bm25_search(query_term, k=k, allowed_docs=common_docs)

    output = []
    files = [f for f in os.listdir(files_path) if f.endswith(('.html', '.htm'))]
    files.sort()

    for docid, score in results:
        fname = files[docid]
        fpath = os.path.join(files_path, fname)

        # 提取标题和正文
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
