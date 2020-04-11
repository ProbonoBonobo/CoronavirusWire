from src.coronaviruswire.common import patterns, default_headers, create_sitemaps_table, create_crawldb_table, create_moderation_table, db
from src.coronaviruswire.utils import parse_schemata, format_text, deduplicate_content, deduplicate_moderation_table, deduplicate_table
from url_normalize import url_normalize
from src.coronaviruswire.utils import load_csv
from munch import Munch
from dateutil.parser import parse as parse_timestamp
from urllib.parse import urlparse
import random
import datetime
import os
import weakref
from collections import deque
import httpx
from string import punctuation
from lxml.html import fromstring as parse_html
from bs4 import BeautifulSoup
import trio
import json
from unidecode import unidecode
from html import unescape
import re
from utils import format_text
from copy import deepcopy as copy
import uuid
from postgresConnection import PostgresConnection

# create_crawldb_table()
create_moderation_table()

create_sitemaps_table()

crawldb = db['moderationtable']
# crawldb = db['crawldb']

sitemapdb = db['sitemaps']
# crawldb.drop()
# sitemapdb.drop()
# create_crawldb_table()
# create_sitemaps_table()
import time
seen = set([row['url'] for row in crawldb])

googleCloudConn = PostgresConnection()

def get_text_chunks(node):
    def recursively_get_text(node):
        if not node:
            return ""
        elif hasattr(node, 'getchildren') and node.getchildren():
            text = [recursively_get_text(kid) for kid in node.getchildren()]
            text = [txt for txt in text if txt]
            return '\n'.join(text)
        elif isinstance(node, str):
            return node.strip()
        else:
            return node.text

    return recursively_get_text(node)


import sys


def flatten_list(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


class Article(Munch):
    """URL objects extracted from a sitemap contain usually only a small subset of
       attribute annotations specified by the <urlset> schema, so let's define a
       generic, database-friendly interface."""

    __seen__ = set(
        seen
    )  # not really useful currently, but can be initialized from a database table object

    def __init__(self, xml):
        url = xml.find("loc")
        lastmod = xml.find("lastmod")
        title = xml.find("news:title")
        description = xml.find("news:description")
        keywords = xml.find("news:keywords")
        publication_date = xml.find("news:publication_date")
        if not title:
            title = xml.find("video:title")
        if not description:
            description = xml.find("video:description")
        self.url = format_text(url_normalize(url.text.strip().lower()))
        self.html = ""
        self.tree = None
        parsed = urlparse(self.url)
        self.site = parsed.netloc
        self.path = parsed.path
        try:
            pardir = '/'.join(re.sub(r'(/)$', '', self.path).split("/")[:-2])
        except:
            pardir = "/"
        self.base_url = f"{parsed.scheme}://{parsed.netloc}{pardir}"
        self.lastmod = parse_timestamp(format_text(
            lastmod.text)) if lastmod else None
        self.headline = format_text(title.text.strip()) if title else ""
        self.keywords = [format_text(kw) for kw in keywords.text.split(",")
                         ] if keywords else []
        self.publication_date = format_text(
            publication_date.text) if publication_date else ""
        self.description = format_text(description.text) if description else ""
        self.xml = format_text(xml.__repr__())
        self.metadata = {"schemata": [], "errors": []}
        self.has_metadata = False
        self.seen = self.url in seen
        # seen.add(self.url)
        self.articlebody = ""
        self.visited = False

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, default=str)

    def parse(self):
        self.has_metadata = bool(self.metadata['schemata'])
        self.metadata_count = len(self.metadata['schemata'])
        self.visited = bool(self.html)
        for k, v in parse_schemata(self.__dict__).items():
            setattr(self, k, v)
        try:
            tree = parse_html(self.html, self.base_url)

            def find_one(selector):
                try:
                    return format_text(tree.xpath(selector)[0].text_content())
                except:
                    return ""

            if not self.headline:
                self.headline = find_one("//h1")
            if not self.articlebody:
                self.articlebody = '\n'.join([
                    format_text(node.text_content())
                    for node in tree.xpath("//p")
                ])
            print(self.articlebody)
        except Exception as e:
            print(e)
        # self.html = ""
        return self.__dict__


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %X")


class Crawler:
    """Superclass (non-abstract) used to accumulate its subclasses' return values and
       offer a more convenient alternative to calling those subclasses' crawl methods
       directly."""

    index = {}  # my kids write their outputs to this object
    children = weakref.WeakValueDictionary()  # to remember where my kids are
    chan = []  # shared memory buffer for accumulating asynchronous outputs

    max_requests = 20

    async def crawl_sitemaps(self, children=None):
        """Sequentially ask Crawler instances to crawl and parse a domain's sitemap URLs. This
            isn't reaaaally asynchronous, because the major bottleneck here is parsing the responses."""
        if not children:
            children = list(self.children.values())
        n_kids = len(children)
        accumulator = []
        for i, child in enumerate(children):
            domain = child.domain
            print(
                f"[ {get_timestamp()} ] Initiating sitemap crawl for domain {domain} ({i+1}/{n_kids})"
            )
            urls = await child.update_urls()
            print(
                f"[ {get_timestamp()} ] Completed sitemap crawl for domain {domain} ({i}/{n_kids}). Extracted {len(urls)} urls."
            )
            accumulator.extend(urls)
        print(
            f"[ {get_timestamp()} ] Sitemap crawl complete. Extracted {len(accumulator)} total urls."
        )
        return accumulator

    async def crawl_urls(self, urls):
        queue = deque(random.sample(
            urls, len(urls)))  # distribute the load between domains

        async def _fetch_async(url: Article, all_done=False):
            _url = url.url
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    print(f"[ {get_timestamp()} ] Requesting URL {_url}...")
                    url.lastcrawled = datetime.datetime.now()
                    response = await client.get(_url, headers=default_headers)
                    print(
                        f"[ {get_timestamp()} ] Received URL {_url} (response length: {len(response.content)} bytes)"
                    )
                    url.status_code = response.status_code
                    url.ok = url.status_code == '200'
                    url.visited = url.ok
                    url.html = response.content
                    url.length = len(response.content)

                    # Trio coroutines can't return values, so append
                    # the payload to a shared memory buffer instead
                    self.chan.append(url)
                    print(
                        f"[ {get_timestamp() } ] Appending {url['url']} payload to the buffer. (Current size: {len(self.chan)} items)"
                    )

            except Exception as e:
                print(f"{e.__class__.__name__} :: {e} (URL: {url})")

        async def _initiate_crawl(queue):
            """Schedule the HTTP get requests for each of the URLs, creating no more than
               `self.max_responses` pending response objects at a given time."""
            curr = 0
            n = len(queue)
            queue = deque([url for url in queue if url.url not in seen])
            print(f"[ {get_timestamp()} ] Found {n} relevant local URLs.")
            while True:
                async with trio.open_nursery() as nursery:
                    for i in range(self.max_requests):
                        try:
                            next_url = queue.popleft()
                        except:
                            print(
                                f"[ {get_timestamp()} ] Scheduling complete.")
                            break

                        seen.add(next_url.url)
                        print(
                            f"[ {get_timestamp()} ] Scheduling crawl of URL {next_url.url} ({curr}/{n})"
                        )
                        crawldb.upsert_many(self.chan)
                        self.chan = []
                        nursery.start_soon(_fetch_async, next_url)
                if not queue:
                    break
            return self.chan

        return await _initiate_crawl(queue)

    async def extract_schema_objects(self, responses):
        """Iterate through a collection of HTTP response objects, extract any
           embedded json objects from the DOM (possibly an empty list), load those
           data structures into memory, and append them to the response."""
        for response in responses:
            # try:
            html = response.html
            tree = parse_html(html)
            schemata = tree.xpath("//script[contains(@type, 'json')]/text()")
            jsonized = []
            errors = []
            for schema in schemata:
                try:
                    jsonized.append(json.loads(schema))
                except Exception as e:
                    serialized = [f"{e.__class__.__name__} :: {e}", schema]
                    errors.append(serialized)

            response.metadata = {"schemata": jsonized, "errors": errors}
            response.has_metadata = bool(jsonized)
            response.metadata_count = len(jsonized)

        # except Exception as e:
        #     print(e.__class__.__name__, e, response)
        #     response['metadata'] = {"schemata": [], "errors": }
        return responses

    async def crawl(self, *kids):
        """Crawl sitemaps, then urls, then finally extract schema objects from the
           HTTP responses."""
        fetched_urls = await self.crawl_sitemaps(kids)

        relevant_local_urls = [
            url for url in fetched_urls if url.url not in seen
        ]
        responses = await self.crawl_urls(relevant_local_urls)
        parsedList = [
            obj.parse() for obj in await self.extract_schema_objects(responses)
        ]
        # seen = set()
        # deduped = []
        # for url in fetched_urls:
        #     if url['url'] in seen:
        #         continue
        #     deduped.append(url)
        #     seen.add(url['url'])

        # Transform to GCP format
        print("TTTT 1")
        print(parsedList)

        newParsedList = []
        for parsed in parsedList:
            newArticle = {}

            newArticle['article_id'] = str(uuid.uuid4())
            if 'headline' in parsed:
                newArticle['title'] = parsed['headline']
            else:
                print("Warning: skipping due to missing headline")
                continue

            print("GGGG" + newArticle['title'])

            newArticle['author'] = parsed['site']
            newArticle['source_id'] = parsed['site']
            newArticle['article_url'] = parsed['url']

            if 'articleBody' in parsed:
                newArticle['content'] = parsed['articlebody']
            else:
                print("Warning: skipping due to missing articlebody")
                continue

            newArticle['category'] = parsed['keywords']
            newArticle['mod_status'] = 'pending'

            if 'publication_date' in parsed:
                newArticle['published_at'] = parsed['publication_date']
            else:
                print("Warning: skipping due to missing publication_date")
                continue

            newArticle['created_by'] = 'crawler'

            if 'city' in parsed:
                newArticle['city'] = parsed['city']

            if 'state' in parsed:
                newArticle['region'] = parsed['state']

            metadata = {}
            if 'metadata' in parsed:
                newArticle['metadata'] = parsed['metadata']

            print("AAAA Inserting new article " + article_id)

            newParsedList.append(newArticle)


        crawldb.upsert_many(newParsedList, ['article_url'])

        #  At this point I would usually write insert the parsed responses into a database,
        #  but to make the code a little easier to distribute, I'm just going to serialize them
        #  to a JSON file
        # with open(f"output.json", "w") as f:
        #     data = [{
        #         k: v
        #         for k, v in obj.items()
        #         if k in ("url", "lastmod", "title", "keywords", "description",
        #                  "is_local", "is_relevant", "status_code",
        #                  "content_length", "ok", "schemata")
        #     } for obj in parsed]
        #     json.dump({"responses": data}, f, indent=4, sort_keys=True)


class SitemapInfo(Crawler):
    """The primary purpose of this subclass is to allow sitemap crawlers to identify
       relevant URLs on a per-domain basis, potentially using domain-specific clues
       (url structure, publication date, descriptive summaries, etc.) to earmark URLs
       that are likely to contain relevant local news content. (News sites are really big,
       and it's not unusual for a sitemap to contain millions of URLs.)"""
    def __init__(
            self,
            city: str,
            state: str,
            loc: str,
            lat: str,
            long: str,
            domain: str,
            sitemap_urls: list,
            is_local: callable,
            is_relevant: callable = lambda url: bool(
                re.findall(
                    patterns['coronavirus'], '\n'.join(
                        [url.text, url.description, '\n'.join(url.keywords)])))
    ):
        super().__init__()
        super().children[domain] = self
        self.chan = deque()
        self.index[domain] = {}
        self.domain = domain
        self.urls = sitemap_urls
        self._is_local = is_local
        self._is_relevant = is_relevant

    async def update_urls(self):
        ok = await self.async_update()
        copied = [obj for obj in list(self.chan)]
        self.chan = deque()
        return copied

    async def async_update(self):
        responses = []

        async def _fetch(url):
            async with httpx.AsyncClient() as client:
                responses.append(await client.get(str(url),
                                                  timeout=120,
                                                  headers=default_headers))

        queue = deque(self.urls)
        async with trio.open_nursery() as nursery:
            for sitemap_url in self.urls:

                nursery.start_soon(_fetch, sitemap_url)

        for res in responses:
            #  extracting url metadata from the subtrees could potentially
            #  be expedited with a threadpool, but I'm afraid of mixing
            #  parallel and concurrent code in the same program :3
            soup = BeautifulSoup(res.content, 'xml')
            urls = soup.find_all("url")
            print(urls)
            for url in urls:
                obj = Article(url)
                # obj = dict(
                #     parsed.__dict__, **{
                #         "is_local": self.is_local(parsed),
                #         "is_relevant": self.is_relevant(parsed)
                #     })
                self.chan.append(obj)

                print(json.dumps(obj.__dict__, indent=4, default=str))

        return self.chan

    def is_local(self, url):
        return self._is_local(url)

    def is_relevant(self, url):
        return self._is_relevant(url)


def tokenize(txt):
    """Get rid of numbers, punctuation, and anything inside HTML tags"""
    txt = unidecode(unescape(txt))
    nums_and_xml_removed = re.sub(r"(<[^>]*>|\d+)", " ", txt)
    tokens = re.split(r"\b", nums_and_xml_removed)
    punctuation_removed = [
        token for token in tokens
        if token.strip() and not any(ch in set(['\n', ' ', *punctuation])
                                     for ch in token)
    ]
    return punctuation_removed


def load_sitemap_urls(fp="../../lib/newspapers.tsv"):
    fp = os.path.abspath(fp)
    news = load_csv(fp)
    loaded = []
    for row in list(news):
        resolved_urls = []
        for k, v in list(row.items()):
            if not v:
                continue
            elif k.startswith("sitemap_url_template"):
                resolved = datetime.datetime.now().strftime(v)
                resolved_urls.append(resolved)
            elif k.startswith("sitemap_url"):
                resolved_urls.append(v)
        print("rrresolved_urls")
        print(resolved_urls)
        print(row)
        url = url_normalize(row['url']).strip().lower()
        parsed = urlparse(url)
        row['url'] = url
        row['site'] = parsed.netloc
        row['sitemap_urls'] = resolved_urls
        loaded.append(row)
    return loaded


def initialize_crawlers():
    index = {}
    news = [row for row in load_sitemap_urls() if row['sitemap_urls']]
    # restrict to just the first 5 rows until we hammer out the glitches
    for row in random.sample(news, 2):
        index[row['name']] = SitemapInfo(row['city'],
                                         row['state'],
                                         row['loc'],
                                         row['lat'],
                                         row['long'],
                                         row['url'],
                                         row['sitemap_urls'],
                                         is_local=lambda xml: True,
                                         is_relevant=lambda xml: True)
    return index


# ==================================== CRAWLER OBJECTS =====================================
#
crawlers = initialize_crawlers()
# Washington Post
# wapo = SitemapInfo("washingtonpost.com", [
#     "https://www.washingtonpost.com/sitemaps/local.xml",
#     "https://www.washingtonpost.com/sitemaps/national.xml"
# ],
#                    is_local=lambda xml: xml.url.startswith(
#                        "https://www.washingtonpost.com/local"))
#
# # Los Angeles Times
# latimes = SitemapInfo(
#     "latimes.com", [
#         "https://www.latimes.com/news-sitemap-content.xml",
#         "https://www.latimes.com/news-sitemap-latest.xml",
#         datetime.datetime.now().strftime(
#             'https://www.latimes.com/sitemap-%Y%m.xml')
#     ],
#     is_local=lambda xml: xml.url.startswith(
#         "https://www.latimes.com/california") or any(
#             re.findall(patterns['la'], ' '.join(tokenize(xml.text)))))
#
# # KCRW (Los Angeles's NPR station)
# kcrw = SitemapInfo(
#     "kcrw.com", ["https://www.kcrw.com/sitemap-shows/news/sitemap-1.xml"],
#     is_local=lambda xml: any(
#         re.findall(
#             r"^https://www.kcrw.com/news/shows/(greater\-la|kcrw\-features)",
#             xml.url)))
#
# # KTLA
# ktla = SitemapInfo(
#     "ktla.com", [
#         datetime.datetime.now().strftime(
#             'https://ktla.com/sitemap.xml?yyyy=%Y&mm=%m&dd=%d')
#     ],
#     is_local=lambda xml: any([
#         re.findall("https://ktla.com/news/(local\-news|california)", xml.url,
#                    re.IGNORECASE),
#         re.findall(patterns['la'], ' '.join(tokenize(xml.text)))
#     ]))
#
# # ABC 7 Los Angeles
# abc7_la = SitemapInfo(
#     "abc7.com", ["https://abc7.com/sitemap/news.xml"],
#     is_local=lambda xml: any([
#         xml.url.startswith("https://abc7.com/community-events"),
#         re.findall(patterns['la'], ' '.join(tokenize(xml.text)))
#     ]))
#
# # Idaho Statesman
# idaho_statesman = SitemapInfo(
#     "idahostatesman.com",
#     ["https://www.idahostatesman.com/sitemap/googlenews/story.xml"],
#     is_local=lambda xml: xml.url.startswith(
#         "https://www.idahostatesman.com/news/local") or any(
#             'idaho' in kw.lower() for kw in xml.keywords) or any(
#                 re.findall(r'boise|idaho|treasure.{0,1}valley', xml.title, re.
#                            IGNORECASE)))
#
# # San Diego Union Tribune
# union_trib = SitemapInfo(
#     "sandiegouniontribune.com", [
#         "https://www.sandiegouniontribune.com/news-sitemap-content.xml",
#         "https://www.sandiegouniontribune.com/news-sitemap-latest.xml"
#     ],
#     is_local=lambda xml: True, is_relevant=lambda xml: True)

# is_local=lambda xml: xml.url.startswith(
#     "https://www.sandiegouniontribune.com/communities/"
# ) or xml.url.startswith(
#     "https://www.sandiegouniontribune.com/north-county-community-news"
# ) or xml.url.startswith(
#     "https://www.sandiegouniontribune.com/opinion/letters-to-the-editor")
# or any(
#     re.findall(
#         r"(county|san.{0,1}diego|north.{0,1}county|local|california)", '\n'
#         .join([xml.title, ', '.join(xml.keywords)]), re.IGNORECASE)))

if __name__ == '__main__':
    crawler = Crawler()
    trio.run(crawler.crawl)
    deduped = deduplicate_moderation_table(crawldb)
    for row in deduped:
        print(f"=================== Before:  ====================")
        print(row['before'])
        print(f"\n\n=================== After:  ====================")
        print(row['after'])

        print("===============================================")
