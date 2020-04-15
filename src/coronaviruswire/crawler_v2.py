from src.coronaviruswire.common import (
    patterns,
    default_headers,
    create_sitemaps_table,
    create_crawldb_table,
    create_moderation_table,
    db,
)
from src.coronaviruswire.utils import (
    parse_schemata,
    format_text,
    load_news_sources,
    deduplicate_content,
    deduplicate_moderation_table,
    deduplicate_table,
)
from url_normalize import url_normalize
from src.coronaviruswire.utils import load_csv
from munch import Munch
from dateutil.parser import parse as parse_timestamp
from lxml.html import fromstring as parse_html
from urllib.parse import urlparse
import random
import datetime
import os
import weakref
from copy import copy
from collections import deque
import httpx
from string import punctuation
from lxml.html import fromstring as parse_html
from bs4 import BeautifulSoup
import trio
from collections import defaultdict
import json
from unidecode import unidecode
from html import unescape
from collections import Counter
import re
import termcolor
import uuid

# this is a global variable because we need to reference its contents when building the database entry
news_sources = load_news_sources()

# i recommend dropping the moderation table before proceding, there are some small updates to the schema
create_moderation_table()

crawldb = db["moderationtable"]
seen = set([row["article_url"] for row in crawldb])

max_sources = 20
max_articles_per_source = 100


class chan:
    queue = deque()
    output = deque()
    seen = set()


def flatten_list(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


class Article:
    """Build a 2d database representation from HTTP responses and extracted metadata. If a metadata field
       isn't available, fallback to parsing the HTML manually using the property methods below."""

    # map schema attributes to the current database schema fields
    required_attrs = {
        "articleBody": "content",
        "headline": "title",
        "datePublished": "published_at",
        "dateModified": "updated_at",
        "description": "summary",
        "keywords": "keywords"
    }

    def __init__(self, url, dom, schema):
        self._dom = dom
        for k, v in self.required_attrs.items():
            if k in schema and schema[k]:
                value = schema[k]
            else:
                # compute the value from this object's property methods
                value = getattr(self, f"_{k}")
            if isinstance(value, str):
                value = format_text(value)
            setattr(self, v, value)
        site = re.sub(r"(https?://|www\.)", "", url_normalize(urlparse(url).netloc))
        self.latitude = float(news_sources[site]["lat"].split("deg")[0])
        self.longitude = float(news_sources[site]["long"].split("deg")[0])
        self.city = news_sources[site]["loc"]
        self.country = "us"
        self.article_url = url_normalize(url)
        self.author = news_sources[site]["name"]
        self.article_id = str(uuid.uuid4())
        self.source_id = self.author
        self.raw_content = copy(self.content)
        self.mod_status = 'pending'

        del self._dom
        super().__init__()
    @property
    def _keywords(self):
        return []

    @property
    def _description(self):
        try:
            text = [
                node.text_content().strip()
                for node in self._dom.xpath("//p")
                if node.text_content() and node.text_content().strip()
            ]
            return text[0] if text else ""

        except Exception as e:
            print(e.__class__.__name__, e)
            return None

    @property
    def _dateModified(self):
        return datetime.datetime.now()

    @property
    def _datePublished(self):
        return datetime.datetime.now()

    @property
    def _headline(self):
        return "\n".join(
            [node.text.strip() for node in self._dom.xpath("//h1") if node.text]
        )

    @property
    def _articleBody(self):
        body = []
        for node in self._dom.xpath("//p"):
            if node and node.text_content():
                for line in node.text_content().strip().split("\n"):
                    txt = line.strip()
                    if txt and len(txt) > 10:
                        body.append(txt)
        return "\n".join(body)


def extract_schemata(dom):
    """Returns the first schema object that contains one or more interesting keys"""

    def dt(timestamp):
        if not timestamp:
            return None
        elif isinstance(timestamp, str):
            return parse_timestamp(timestamp)
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
                re.sub("(^'|^\"|'$|\"$)", "", kw.strip())
                for kw in re.split(
                    r"\s*(,|_|\/|\n|\"|\-|.com|\/|\:|\;|\[|\]|\)|\(|\{|\})\s*", keywords
                )
                if len(kw.strip()) > 5
            ]
        else:
            raise TypeError(
                f"Weird type for keywords: {type(keywords).__class__.__name__} :: {keywords}"
            )

    schemata = dom.xpath("//script[contains(@type, 'json')]")
    objects = []
    for schema in schemata:
        schema = schema.text

        try:
            parsed = json.loads(schema)
            print(json.dumps(parsed))
            objects.append(parsed)
        except Exception as e:
            print(f"Couldn't parse {schema}")
            parsed = {}

    typemap = {
        "articleBody": str,
        "headline": str,
        "description": str,
        "keywords": coll,
        "datePublished": dt,
        "dateModified": dt,
    }

    target_attributes = set(typemap.keys())
    for obj in flatten_list(objects):
        found_attrs = list(target_attributes.intersection(obj.keys()))
        tx = [typemap[k] for k in found_attrs]
        info = {
            attr: format_text(func(obj[attr])) for attr, func in zip(found_attrs, tx)
        }
        if found_attrs:
            return info
    return {}


async def fetch_sitemap(url):
    """Delegate function for sitemap urls, which parses the http response and adds new urls to the
       queue channel"""
    async with httpx.AsyncClient() as client:
        res = await client.get(url, timeout=120, headers=default_headers)
        print(f"Got response from {url}")

    soup = BeautifulSoup(res.content, "xml")
    urls = soup.find_all("url")
    for i, url in enumerate(urls):
        if i > max_articles_per_source:
            break
        text = url_normalize(url.find("loc").text).strip()
        print(f"url #{i} :: {text}")
        if text in chan.seen:
            continue
        chan.queue.append(text)
        chan.seen.add(text)


async def fetch_content(url):
    """Delegate function for fetching a content URL, which appends the response to the output channel"""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, timeout=120, headers=default_headers)
            print(f"Got response from {url}")
        chan.output.append((url, res.content))

    except Exception as e:
        print({e.__class__.__name__}, e)
        pass


async def main():
    global news_sources
    keep_going = True

    print(f"Loaded {len(news_sources)} sources")
    queue = list(flatten_list([row["sitemap_urls"] for row in news_sources.values()]))
    print(queue)
    if max_sources:
        queue = random.sample(queue, max_sources)
    sitemap_urls = set(queue)
    chan.queue = deque(queue)
    while keep_going:
        print(f"Initializing nursery")
        async with trio.open_nursery() as nursery:
            for i in range(min(len(chan.queue), 50)):
                print(f"Processing item {i}")
                next_url = chan.queue.popleft()
                if next_url in sitemap_urls:
                    print(f"Starting {next_url}")
                    nursery.start_soon(fetch_sitemap, next_url)
                    print(f"Continuing")
                else:
                    nursery.start_soon(fetch_content, next_url)
                    print(f"Got url {next_url}")
        if len(chan.output) > 10 or not bool(chan.queue):
            processed = []
            for url, html in chan.output:
                dom = parse_html(html)
                metadata = extract_schemata(dom)
                parsed = Article(url, dom, metadata)
                print(f"Url: {url}")
                print(f"Parsed: {parsed}")
                print(json.dumps(parsed.__dict__, indent=4, default=str))
                processed.append(parsed.__dict__)

            crawldb.upsert_many(processed, ["article_url"])
            chan.output = []
            keep_going = bool(chan.queue)


if __name__ == "__main__":
    trio.run(main)
