from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from url_normalize import url_normalize
import jieba
import re
import requests
import time
import random
import lxml
from queue import Queue, Empty
import json
import hashlib
import os
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import chardet

DEFAULT_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/91.0.4472.124 Safari/537.36')
}

ALLOWED_DOMAINS = ["keyan.ruc.edu.cn", "xsc.ruc.edu.cn"]

lock = threading.Lock()  
count = 0      
seen_hashes = set()

def normalize_url(url: str) -> str:
    """规范"""
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}{u.path}"

def extract_url(html: str, cur_url: str = None) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for item in soup.find_all("a", href=True):
        href = item["href"].strip()
        if not href:
            continue

        if cur_url and not href.startswith("http"):
            href = urljoin(cur_url, href)

        if not href.lower().endswith(('.html', '.htm')):
            continue

        domain = urlparse(href).netloc
        if any(domain == allowed for allowed in ALLOWED_DOMAINS): 
            links.add(normalize_url(href))

    return list(links)


def clean_words(words: List[str]) -> List[str]:
    """去掉标点、空白"""
    return [w for w in words if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', w)]


def body(html: str) -> List[str]:
    """
    提取正文并分词
    """
    soup = BeautifulSoup(html, "lxml")
    parsed_text = " ".join([p.get_text() for p in soup.find_all('p')])
    words = jieba.lcut(parsed_text) if parsed_text else []
    return clean_words(words)


def title(html: str) -> List[str]:
    """
    提取标题并分词
    """
    soup = BeautifulSoup(html, "lxml")
    words: List[str] = []

    if soup.title and soup.title.string:
        temp = jieba.lcut(soup.title.string.strip())
        words += temp

    for item in soup.find_all(["h1", "h2", "h3"]):
        t = item.get_text(strip = True)
        if t:
            temp = jieba.lcut(t)
            words += temp

    return clean_words(words)


def set_filename(url: str, index: int) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"{index}_{h}.html"


def crawl(url: str, index: int, save_path: str, headers: Dict[str, str], all_url: Set[str], q: Queue, mapping: List[Dict], lock: threading.Lock):
    global count, seen_hashes
    try:
        print(f"[{index}] 正在爬取: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        detect = chardet.detect(response.content)
        encoding = detect["encoding"] if detect["encoding"] else "utf-8"
        html = response.content.decode(encoding)

        content_hash = hashlib.md5(html.encode("utf-8")).hexdigest()

        with lock:
            if content_hash in seen_hashes:
                return
            seen_hashes.add(content_hash)

        filename = f"{index}_{hashlib.md5(url.encode()).hexdigest()}.html"
        filepath = os.path.join(save_path, filename)

        with lock:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            mapping.append({
                "id": index,
                "url": url,
                "filename": filename,
                "hash": content_hash
                })

        new_links = extract_url(html, url)

        with lock:
            for new_url in new_links:
                if new_url not in all_url:
                    all_url.add(new_url)
                    q.put(new_url)
                    print(f"新链接: {new_url}")

    except Exception as e:
        print(f"[{index}] 抓取失败 {url}: {e}")


def all_urls(url_seeds: List[str], save_path: str = "htmls", headers: Dict[str, str] = None, max_workers: int = 5, max_pages: int = 50) -> Set[str]:
    """多线程爬虫"""
    global count

    if not headers:
        headers = DEFAULT_HEADERS

    os.makedirs(save_path, exist_ok=True)

    q = Queue()
    mapping = []
    all_url = set()
    lock = threading.Lock()
    
    for seed in url_seeds:
            if not seed.startswith("http"):
                seed = "https://" + seed
            q.put(seed)
            all_url.add(seed)
        
    def worker():
        nonlocal mapping, all_url
        global count, seen_hashes
        while True:
            try:
                url = q.get(timeout=2)
                with lock:
                    if count >= max_pages:
                        q.task_done()
                        break
                    count += 1
                    current_index = count
                
                crawl(url, current_index, save_path, headers, all_url, q, mapping, lock)
                q.task_done()
                time.sleep(random.uniform(1.5, 3.0))

            except Empty:
                break

            except Exception as e:
                print(f"Worker异常: {e}")
                q.task_done()
                continue

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker) for _ in range(max_workers)]
        
        try:
            while any(not f.done() for f in futures):
                with lock:
                    if count >= max_pages:
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            print("停止爬取")


    with open(os.path.join(save_path, "urls.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"爬取完成！共爬取 {len(mapping)} 个页面，发现 {len(all_url)} 个链接")
    return all_url



if __name__ == "__main__":
    seeds = ["https://keyan.ruc.edu.cn","https://xsc.ruc.edu.cn"]
    LinksList = all_urls(seeds, headers=DEFAULT_HEADERS, max_pages=10000, max_workers=5)
