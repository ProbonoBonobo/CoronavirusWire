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
import pickle
from src.coronaviruswire.pointAdaptor import (Point, adapt_point)
from psycopg2.extensions import adapt, register_adapter, AsIs
from url_normalize import url_normalize
from src.coronaviruswire.utils import load_csv
from munch import Munch
from dateutil.parser import parse as parse_timestamp
from lxml.html import fromstring as parse_html
from urllib.parse import urlparse
from newspaper import Article as NewsArticle
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
news_sources = load_news_sources("./lib/newspapers.tsv")

# i recommend dropping the moderation table before proceding, there are some small updates to the schema
create_moderation_table()

crawldb = db["moderationtable_v2"]

MAX_SOURCES = 100
MAX_ARTICLES_PER_SOURCE = 50
MAX_REQUESTS = 5
BUFFER_SIZE = 100

seen_urls  = set()
try:
    with open("seen.pkl", "rb") as f:
        seen_urls = pickle.load(f)
except:
    pass


class chan:
    queue = deque()
    output = deque()
    seen = set(seen_urls)


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
        "keywords": "keywords",
    }

    def __init__(self, url, dom, schema):
        register_adapter(Point, adapt_point)

        self._dom = dom
        self.schema = schema
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
        self.author = site
        # latitude = -1 * float(news_sources[site]["lat"].split("deg")[0])
        # longitude = float(news_sources[site]["long"].split("deg")[0])
        #
        # # latitude input is wrong, see pointAdaptor for more details
        # self.sourcelonglat = Point(longitude, latitude)
        if site in news_sources:
            self.sourceloc = news_sources[site]["loc"]
            self.author = news_sources[site]["name"]
        else:
            for k,v  in news_sources.items():
                if site in k or k in site:
                    self.sourceloc = v['loc']
                    self.author = v['name']
                    break
        self.sourcecountry = "us"
        self.article_url = url_normalize(url)

        self.article_id = str(uuid.uuid4())
        self.source_id = self.author

        self.raw_content = copy(self._articleBody)

        del self._dom
        del self.schema
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
            return format_text(text) if text else ""

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
        return format_text("\n".join(
            [node.text.strip() for node in self._dom.xpath("//h1") if node.text]
        ))

    @property
    def _articleBody(self):
        body = []
        extracted = ""

        if 'articleBody' in self.schema:
            extracted = self.schema['articleBody']
        for node in self._dom.xpath("//p"):
            if node and node.text_content():
                for line in node.text_content().strip().split("\n"):
                    txt = line.strip()
                    if txt and len(txt) > 10:
                        body.append(txt)
        fallback = " \n ".join(body)
        return format_text(list(sorted([extracted, fallback], key=len))[-1])



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
    try:
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=10, headers=default_headers)
            except:
                return
    except:
        return
    soup = BeautifulSoup(res.content, "xml")
    urls = soup.find_all("url")
    for i, url in enumerate(urls):
        if i > MAX_ARTICLES_PER_SOURCE:
            break
        text = url_normalize(url.find("loc").text.strip())
        print(f"url #{i} :: {text}")
        if text in chan.seen:
            continue
        chan.queue.append(text)
        chan.seen.add(text)
    # chan.queue = deque(random.sample(list(chan.queue), len(chan.queue)))


async def fetch_content(url):
    """Delegate function for fetching a content URL, which appends the response to the output channel"""
    try:
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, timeout=10, headers=default_headers)
            except:
                return
            print(f"Got response from {url}")
        chan.output.append((url, res.content))

    except:
        return



async def main():
    keep_going = True
    print(f"Loaded {len(news_sources)} sources")
    queue = list(flatten_list([row["sitemap_urls"] for row in news_sources.values()]))
    print(queue)
    if MAX_SOURCES and len(queue) >= MAX_SOURCES:
        queue = random.sample(queue, MAX_SOURCES)
    sitemap_urls = set(queue)
    chan.queue = deque(queue)
    while keep_going:

        print(f"Initializing nursery")
        async with trio.open_nursery() as nursery:
            for i in range(min(len(chan.queue), MAX_REQUESTS)):
                print(f"Processing item {i}")
                next_url = chan.queue.popleft()
                if next_url in sitemap_urls:
                    print(f"Starting {next_url}")
                    nursery.start_soon(fetch_sitemap, next_url)
                    print(f"Continuing")
                else:
                    nursery.start_soon(fetch_content, next_url)
                    print(f"Got url {next_url}")

        if len(chan.output) >= BUFFER_SIZE or not bool(chan.queue):
            # chan.queue = deque(random.sample(list(chan.queue), len(chan.queue)))
            processed = []
            for url, html in chan.output:
                chan.seen.add(url)
                parsed = NewsArticle(url)
                parsed.download(html)
                parsed.parse()
                parsed.nlp()
                city, state = None, None
                site = re.sub(r"(https?://|www\.)", "", url_normalize(urlparse(parsed.source_url).netloc))
                if site in news_sources:
                    sourceloc = news_sources[site]["loc"]
                    author = news_sources[site]["name"]
                    city, state = sourceloc.split(", ")
                else:
                    for k, v in news_sources.items():
                        if site in k or k in site:
                            sourceloc = v['loc']
                            author = v['name']
                            break
                # latitude = -1 * float(news_sources[site]["lat"].split("deg")[0])
                # longitude = float(news_sources[site]["long"].split("deg")[0])
                #
                # # latitude input is wrong, see pointAdaptor for more details
                # sourcelonglat = Point(longitude, latitude)
                # sourceloc = news_sources[site]["loc"]
                sourcecountry = "us"

                article_id = str(uuid.uuid4())
                _section = None
                _tag = None
                category = []
                keywords = [unidecode(kw) for kw in parsed.keywords]
                published = parsed.publish_date
                modified = parsed.publish_date
                description = parsed.summary

                try:
                    description = unidecode(parsed.meta_data['article']['description'])
                except:
                    pass

                try:
                    modified = parse_timestamp(parsed.meta_data['article']['modified'])
                except:
                    pass

                try:
                    alt_tags = re.findall(r"([^,;:]{4,})", parsed.meta_data['article']['news_keywords'])
                    keywords = list(keywords.union([unidecode(kw) for kw in alt_tags]))
                except:
                    pass


                try:
                    _section = unidecode(parsed.meta_data['article']['section'])
                    _tag = unidecode(parsed.meta_data['article']['tag'])
                except:
                    pass


                if _section or _tag:
                    category = [s for s in (_section, _tag) if s]
                if not published:
                    dom = parse_html(html)
                    metadata = extract_schemata(dom)
                    article = Article(url, dom, metadata)
                    published = article._datePublished
                    modified = article._dateModified


                row = {
                    "raw_content": unidecode(parsed.text),
                    "content": unidecode(parsed.text),
                    "title": unidecode(parsed.title),
                    "summary": unidecode(description),
                    "keywords": keywords,
                    "image_url": parsed.top_image,
                    "article_url": url,
                    "author": ', '.join(parsed.authors),
                    "category": category,
                    "source_id": site,
                     "sourceloc": sourceloc,
                    # "sourcelonglat": sourcelonglat,
                    "sourcecountry": sourcecountry,
                    "article_id": article_id,
                    "has_ner": False,
                    "has_geotags": False,
                    "has_coords": False,
                    "published_at": published,
                    "edited_at": modified,
                    "city": city,
                    "state": state

                }
                if not re.search(r"(covid|virus|hospital|pandemic|corona)", str(row), re.IGNORECASE):
                    print(f"No match for coronavirus in article: {url}")
                    continue

                else:
                    print(json.dumps(row, indent=4, default=str))
                    processed.append(row)

            print("upserting many...")
            crawldb.upsert_many(processed, ["article_url"])
            chan.output = []
            keep_going = bool(chan.queue)
            print(chan.queue)
            print(f"{len(chan.queue)} urls in the queue")
        with open("seen.pkl", "wb") as f:
            pickle.dump(chan.seen, f)


if __name__ == "__main__":
    trio.run(main)
    deduplicate_moderation_table(crawldb)
