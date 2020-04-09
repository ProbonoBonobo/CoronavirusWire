import json
import csv
from concurrent import futures
from src.coronaviruswire.common import default_headers
import trio
import cssselect
import httpx
from lxml.html import fromstring, clean
from unidecode import unidecode
from html import unescape
from urllib.parse import urlparse, urljoin
from url_normalize import url_normalize
import lxml
import re
from src.coronaviruswire.common import db
from collections import Counter


def async_fetch(*urls,
                max_requests=25,
                headers=default_headers,
                timeout=60,
                **kwargs):
    if isinstance(urls, str):
        urls = [urls]
    urls = [url_normalize(url) for url in urls]
    chan = {}

    async def fetch():
        limit = trio.CapacityLimiter(max_requests)

        async def async_fetch(url, headers, timeout, **kwargs):
            async with httpx.AsyncClient() as client:
                chan[url] = await client.get(url,
                                             headers=default_headers,
                                             timeout=timeout,
                                             **kwargs)

        async with trio.open_nursery() as nursery:
            for url in urls:
                async with limit:
                    nursery.start_soon(async_fetch, url, headers, timeout,
                                       **kwargs)

    trio.run(fetch)
    chan = [chan[url] for url in urls]
    if len(chan) == 1:
        chan = chan[0]
    return chan


def parse_html(responses):
    if isinstance(responses, httpx.Response):
        responses = [responses]
    contents = [res.content for res in responses]
    base_urls = [f"{res.url.scheme}://{res.url.host}/" for res in responses]
    for response, html, url in zip(responses, contents, base_urls):
        response.tree = fromstring(html, base_url=url)
        response.base = url

    return responses


def iter_csv(fp="/home/kz/projects/coronaviruswire/lib/newspapers.csv",
             delimiter="\t"):
    with open(fp, 'r') as f:
        f_csv = csv.DictReader(f, delimiter=delimiter)
        cols = f_csv.fieldnames
        for row in f_csv:
            yield dict(row.items())


def load_csv(fp="/home/kz/projects/coronaviruswire/lib/newspapers.tsv",
             delimiter="\t"):
    return list(iter_csv(fp, delimiter))


def remove_cruft(list_of_articles):
    lines = []
    counts = None
    for article in list_of_articles:
        if not article:
            continue
        lines.extend([line.strip() for line in article.split("\n")])
    counts = Counter(lines)
    output = []
    for article in list_of_articles:
        curr = []
        if not article:
            continue
        for line in article.split("\n"):
            line = line.strip()
            if counts[line] > 2:
                continue
            curr.append(line)
        output.append("\n".join(curr))

    return list(reversed(output))


from collections import deque


def format_text(txt):
    """Go away, weird ASCII unicode transliterations"""
    if isinstance(txt, (list, tuple, set, deque)):
        # when txt is a vector of strings, process its items recursively
        return [format_text(s) for s in txt]
    elif isinstance(txt, str):
        # in many cases the text values of a schema.org metadata object not only eliminate line breaks, but
        # smoosh the lines together without preserving whitespace. Here we check for end punctuation characters
        # that are right adjacent to something that looks like the start of a new line, inserting a space between
        # the two capturing groups when we get a match.
        html_chars_padded = re.sub(r"(\S?)(&\w+\;)(\S?)", r"\1 \2 \3",
                                   txt.strip())
        padded = re.sub(r"(\S{4,})([\.\!\?]+|\-{3,})(\'?\"?)([A-Z])",
                        r"\1\2 \n \3", unidecode(unescape(html_chars_padded)))
        no_camels = re.sub(r"([a-z]{2,})([A-Z]\w*)", r"\1. \2", padded)
        no_html_tags = re.sub(r"(\s\<[^\>]*\>\s*|\s{2,})", " ",
                              no_camels).strip()
        return no_html_tags
    else:
        return txt


def extract_entities(s):
    days = r"((Mon(d|\s)|Tue|Wed(n|\b)|Thur|Fri|Sat|Sun(\b|d))[\w\.\,]*\s*\d*\s*)|((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\.]*\s*\d*\s)"
    cleaned = [
        re.sub(days, "", tok[0]) for tok in re.findall(
            r"(([A-Z]([a-zA-Z]+|\.|\'|\-\,)+)+(\s[A-Z][a-zA-Z]+)+)|([A-Z]{1,})|([a-zA-Z][A-Z])[a-zA-Z]*[A-Z][a-z]*",
            unidecode(s)) if tok[0] and len(tok[0]) > 4 and not "\n" in tok
    ]
    return [tok.strip() for tok in cleaned if '\n' not in tok and len(tok) > 5]


def flatten(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


import datetime
from dateutil.parser import parse


def parse_schemata(row):
    def dt(timestamp):
        if not timestamp:
            return None
        elif isinstance(timestamp, str):
            return parse(timestamp)
        elif isinstance(timestamp, datetime.datetime):
            return timestamp
        else:
            return None

    def coll(keywords):
        if isinstance(keywords, list):
            return coll(str(keywords))
        elif isinstance(keywords, str):
            keywords = format_text(keywords)
            return [
                re.sub("(^\'|^\"|\'$|\"$)", "", kw.strip()) for kw in re.split(
                    r"\s*(,|_|\/|\n|\"|\-|.com|\/|\:|\;|\[|\]|\)|\(|\{|\})\s*",
                    keywords) if len(kw.strip()) > 5
            ]
        else:
            raise TypeError(
                f"Weird type for keywords: {type(keywords).__class__.__name__} :: {keywords}"
            )

    metadata = row['metadata']
    objects = metadata['schemata']

    typemap = {
        'articleBody': str,
        'headline': str,
        'description': str,
        'keywords': coll,
        'datePublished': dt,
        'dateModified': dt
    }

    target_attributes = set(typemap.keys())
    for obj in flatten(objects):
        found_attrs = list(target_attributes.intersection(obj.keys()))
        tx = [typemap[k] for k in found_attrs]
        info = {
            attr: format_text(func(obj[attr]))
            for attr, func in zip(found_attrs, tx)
        }
        if found_attrs:
            return info
    return {}


def deduplicate_content(index, max_count=3):
    """Given a list of article strings, split each article into lines and count the occurrences of each line.
       if a line appears more than 3 times, erase all its occurrences. Join the remaining lines of each article
       before returning."""
    print(f"Deduplicating {len(list(index.keys()))} articles...")
    lines = []
    counts = {}
    _formatted = {}
    output = []
    for id, article in index.items():
        _lines = [format_text(line.strip()) for line in article.split("\n")]
        _formatted[id] = _lines
        lines.extend(_lines)

    counts = Counter(lines)
    for id, article in _formatted.items():
        uniques = [line for line in article if counts[line] <= max_count]
        deduplicated = '\n'.join(uniques)
        output.append({"id": id, "before": index[id], "after": deduplicated})

    return output


def deduplicate_table(tab):
    print(f"Indexing article contents...")
    updates = {row['id']: row['articlebody'] for row in tab}

    processed = deduplicate_content(updates)
    return processed


if __name__ == '__main__':
    responses = async_fetch("msn.com", "yahoo.com", "nytimes.com",
                            "news.ycombinator.com")
    crawldb = db['crawldb']
    rows = [(row['headline'], row['url'], row['articlebody'])
            for row in crawldb]
    headlines = [row[0] for row in rows]
    articles = [row[-1] for row in rows]
    cleaned = remove_cruft(articles)
    for before, after in zip(articles, cleaned):
        print(f" ============= BEFORE ============== ")
        print(before)
        print(f" ============= AFTER =============== ")
        print(after)
        print("\n\n")
    with open("../../outputs/cleaned.json", "w") as f:
        out = []
        for before, after, headline in zip(articles, cleaned, headlines):
            obj = {"headline": headline, "before": before, "after": after}
            out.append(obj)
        json.dump({"output": out}, f, indent=4)
