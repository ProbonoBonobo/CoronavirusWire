from src.coronaviruswire.common import patterns, default_headers
from src.coronaviruswire.utils import load_csv
import random
import datetime
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


def format_text(txt):
    """Go away, weird ASCII unicode transliterations"""
    return unidecode(unescape(txt.strip()))


class SitemapContentLink:
    """URL objects extracted from a sitemap contain usually only a small subset of
       attribute annotations specified by the <urlset> schema, so let's define a
       generic, database-friendly interface."""

    _seen = set(
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
        self.url = format_text(url.text) if url else ""
        self.lastmod = format_text(lastmod.text) if lastmod else ""
        self.title = format_text(title.text) if title else ""
        self.keywords = [format_text(kw) for kw in keywords.text.split(",")
                         ] if keywords else []
        self.publication_date = format_text(
            publication_date.text) if publication_date else ""
        self.description = format_text(description.text) if description else ""
        self.text = format_text(xml.__repr__())
        self.seen = self.url in self._seen
        self._seen.add(self.url)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %X")


class Crawler:
    """Superclass (non-abstract) used to accumulate its subclasses' return values and
       offer a more convenient alternative to calling those subclasses' crawl methods
       directly."""

    index = {}  # my kids write their outputs to this object
    children = weakref.WeakValueDictionary()  # to remember where my kids are
    chan = []  # shared memory buffer for accumulating asynchronous outputs

    max_requests = 25

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
        random.shuffle(urls)  # distribute the load between domains
        queue = deque(urls)
        capacitator = trio.CapacityLimiter(self.max_requests)

        async def _fetch_async(url, all_done=False):
            _url = url['url']
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    print(f"[ {get_timestamp()} ] Requesting URL {_url}...")
                    response = await client.get(_url, headers=default_headers)
                    print(
                        f"[ {get_timestamp()} ] Received URL {_url} (response length: {len(response.content)} bytes)"
                    )
                    url['response_code'] = response.status_code
                    url['content'] = response.content
                    url['content_length'] = len(response.content)

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
            print(f"[ {get_timestamp()} ] Found {n} relevant local URLs.")
            async with trio.open_nursery() as nursery:
                async with capacitator:
                    for next_url in queue:
                        print(
                            f"[ {get_timestamp()} ] Scheduling crawl of URL {next_url} ({curr}/{n})"
                        )
                        nursery.start_soon(_fetch_async, next_url)
            return self.chan

        return await _initiate_crawl(queue)

    async def extract_schema_objects(self, responses):
        """Iterate through a collection of HTTP response objects, extract any
           embedded json objects from the DOM (possibly an empty list), load those
           data structures into memory, and append them to the response."""
        for response in responses:
            try:
                html = response['content']
                tree = parse_html(html)
                schemata = tree.xpath(
                    "//script[contains(@type, 'json')]/text()")
                jsonized = []
                for schema in schemata:
                    try:
                        jsonized.append(json.loads(schema))
                    except json.decoder.JSONDecodeError as e:
                        print(
                            f"URL {response.url} contains a bad JSON object:")
                        print(schema)

                response['schemata'] = jsonized
            except Exception as e:
                print(e.__class__.__name__, e, response)
        return responses

    async def crawl(self, *kids):
        """Crawl sitemaps, then urls, then finally extract schema objects from the
           HTTP responses."""
        fetched_urls = await self.crawl_sitemaps(kids)

        relevant_local_urls = [
            url for url in fetched_urls
            if url['is_relevant'] and url['is_local']
        ]
        responses = await self.crawl_urls(relevant_local_urls)
        parsed = await self.extract_schema_objects(responses)

        #  At this point I would usually write insert the parsed responses into a database,
        #  but to make the code a little easier to distribute, I'm just going to serialize them
        #  to a JSON file
        with open(f"output.json", "w") as f:
            data = [{
                k: v
                for k, v in obj.items()
                if k in ("url", "lastmod", "title", "keywords", "description",
                         "is_local", "is_relevant", "status_code",
                         "content_length", "ok", "schemata")
            } for obj in parsed]
            json.dump({"responses": data}, f, indent=4, sort_keys=True)


class SitemapInfo(Crawler):
    """The primary purpose of this subclass is to allow sitemap crawlers to identify
       relevant URLs on a per-domain basis, potentially using domain-specific clues
       (url structure, publication date, descriptive summaries, etc.) to earmark URLs
       that are likely to contain relevant local news content. (News sites are really big,
       and it's not unusual for a sitemap to contain millions of URLs.)"""
    def __init__(
            self,
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
        copied = [obj for obj in self.chan]
        self.chan = deque()
        return copied

    async def async_update(self):
        responses = []

        async def _fetch(url):
            async with httpx.AsyncClient() as client:
                responses.append(await client.get(str(url),
                                                  timeout=60,
                                                  headers=default_headers))

        for sitemap_url in self.urls:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(_fetch, sitemap_url)

        for res in responses:
            #  extracting url metadata from the subtrees could potentially
            #  be expedited with a threadpool, but I'm afraid of mixing
            #  parallel and concurrent code in the same program :3
            soup = BeautifulSoup(res.content, 'xml')
            urls = soup.find_all("url")
            print(urls)
            for url in urls:
                parsed = SitemapContentLink(url)
                obj = dict(
                    parsed.__dict__, **{
                        "is_local": self.is_local(parsed),
                        "is_relevant": self.is_relevant(parsed)
                    })
                self.chan.append(obj)
                print(json.dumps(obj, indent=4))

        return sorted(self.chan, key=lambda parsed: parsed['lastmod'])

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


def load_sitemap_urls(fp="lib/newspapers.tsv"):
    news = load_csv(fp)
    loaded = []
    for row in news:
        resolved_urls = []
        for k, v in row.items():
            if not v:
                continue
            elif k.startswith("sitemap_url_template"):
                resolved = datetime.datetime.now().strftime(v)
                resolved_urls.append(resolved)
            elif k.startswith("sitemap_url"):
                resolved_urls.append(v)
        row['sitemap_urls'] = resolved_urls
        loaded.append(row)
    return loaded


def initialize_crawlers():
    index = {}
    news = load_sitemap_urls()
    # restrict to just the first 5 rows until we hammer out the glitches
    for row in news[0:5]:
        index[row['name']] = SitemapInfo(row['url'],
                                         row['sitemap_urls'],
                                         is_local=lambda xml: True,
                                         is_relevant=lambda xml: True)
    return index


# ==================================== CRAWLER OBJECTS =====================================

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
