from src.coronaviruswire.common import (
    patterns,
    default_headers,
    create_sitemaps_table,
    create_crawldb_table,
    create_moderation_table,
    db,
)
from flatdict import FlatDict
from src.coronaviruswire.utils import (
    parse_schemata,
    format_text,
    load_news_sources,
    deduplicate_content,
    deduplicate_moderation_table,
    deduplicate_table,
)
from pylev import levenshtein
import pickle
from gemeinsprache.utils import red, yellow, green, blue, magenta, cyan
from src.coronaviruswire.pointAdaptor import Point, adapt_point
from psycopg2.extensions import adapt, register_adapter, AsIs
from url_normalize import url_normalize
from src.coronaviruswire.utils import load_csv
from munch import Munch
from dateutil.parser import parse as parse_timestamp
from unicodedata import normalize
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
news_sources = load_news_sources("lib/newspapers.tsv")

# i recommend dropping the moderation table before proceding, there are some small updates to the schema
create_moderation_table()

crawldb = db["moderationtable_v2"]

MAX_SOURCES = 200
MAX_ARTICLES_PER_SOURCE = 50
MAX_REQUESTS = 4
BUFFER_SIZE = 50

seen_urls = set([row["article_url"] for row in crawldb])
try:
    with open("seen.pkl", "rb") as f:
        seen_urls.update(pickle.load(f))
except:
    pass


class chan:
    queue = deque()
    output = deque()
    seen = set(row['article_url'] for row in crawldb)


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

    def __init__(self, url, dom, schema, soup):
        register_adapter(Point, adapt_point)
        self._soup = soup
        self._dom = dom
        self.schema = schema
        self.article_url = url_normalize(url)
        for k, v in self.required_attrs.items():
            if k in schema and schema[k]:
                value = schema[k]
            else:
                # compute the value from this object's property methods
                value = getattr(self, f"_{k}")
            if isinstance(value, str):
                pass
            setattr(self, v, value)
        self.raw_content = copy(self.content)
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
            for k, v in news_sources.items():
                if site in k or k in site:
                    self.sourceloc = v["loc"]
                    self.author = v["name"]
                    break
        self.sourcecountry = "us"


        self.article_id = str(uuid.uuid4())
        self.source_id = self.author



        # del self._dom
        # del self.schema
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
        headline = []
        headline = self._soup.find("h1")
        if headline and hasattr(headline, 'text') and headline.text:
            return headline.text
        return ""


    @property
    def _articleBody(self):
        body = []
        extracted = ""

        if "articleBody" in self.schema:
            text = self.schema['articleBody']
            try:
                text = normalize("NFKD", unescape(text))
            except:
                pass
            
            try:
                text = text.encode("utf-8").decode("utf-8")
            except:
                pass
            if text:
                extracted = text.strip()
            
        for node in self._soup.find_all("p"):
            text = node.text
            if text and text.strip():
                body.append(text.strip())

        fallback = " \n ".join(body)
        selected = list(sorted([extracted, fallback], key=len))[-1]
        print(f"Selected text is:\n\n{selected}\n\n========================================================\nEnd of url: {self.article_url}")
        return selected


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
            err = TypeError(

                f"Weird type for keywords: {type(keywords).__class__.__name__} :: {keywords}"
            )
            print(err)
            return []

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


async def fetch_sitemap(sitemap_url):
    """Delegate function for sitemap urls, which parses the http response and adds new urls to the
       queue channel"""
    try:
        async with httpx.AsyncClient() as client:
            try:
                # print(magenta(f"[ fetch_sitemap ] ") + f":: Initiating request for sitemap: {sitemap_url}")
                res = await client.get(sitemap_url, timeout=20, headers=default_headers)
                # no_weird_characters = normalize('NFKD', unescape('NKFD', s.decode(res.encoding)))
                # maybe preemptively resolve escaped utf-8 characters to their unescaped equivalents?
                # hopefully_sanitized = no_weird_characters.encode('utf-8').decode('unicode-escape').encode('utf-8')


            except Exception as e:
                print(
                    magenta(f"[ fetch_sitemap ] ")
                    + red(
                        f":: Failed to fetch url: {sitemap_url}. {e.__class__.__name__} :: {e}"
                    )
                )
                return
    except Exception as e:
        print(
            magenta(f"[ fetch_sitemap ] ")
            + red(
                f":: Encountered an unknown exception while fetching url {sitemap_url}: {e.__class__.__name__} :: {e}"
            )
        )
        return
    soup = BeautifulSoup(res.content, from_encoding=res.encoding)
    elements = soup.findAll("loc", limit=MAX_ARTICLES_PER_SOURCE*2)
    urls = []
    for elem in elements:
        url = elem.text
        if url and url.strip() not in chan.seen:
            urls.append(url.strip())

    print(magenta("[ fetch_sitemap ] "), f":: Extracted {len(urls)} from sitemap: {sitemap_url}")
    chan.queue.extend(urls)
    # total = len(urls)
    # print(magenta(
    #     "[ fetch_sitemap ] ") + f":: Received {green(str(len(res.content)) + ' bytes')} and extracted {green(total)} {green('total urls')} from sitemap: {sitemap_url}")
    # found = 0
    # dups = 0
    # for url_string in urls:
    #     try:
    #         url_string = url_string.strip()
    #     except:
    #         url_string = url_string
    #     if found >= MAX_ARTICLES_PER_SOURCE:
    #         break
    #     # text = url_normalize(url.find("loc").text.strip())
    #     elif url_string in chan.seen:
    #         dups += 1
    #         continue
    #     else:
    #         found += 1
    #
    #         print(
    #             magenta("[ fetch_sitemap ]") + f" :: url #{found}: {url_string}"
    #         )
    #         chan.queue.append(url_string)

            #chan.seen.add(url_string)

    # chan.queue = deque(random.sample(list(chan.queue), len(chan.queue)))


async def fetch_content(url):
    """Delegate function for fetching a content URL, which appends the response to the output channel"""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            try:
                print(
                    f"{blue('[ fetch_content ]')} :: Initiating request for url: {url}"
                )
                res = await client.get(url, timeout=10, headers=default_headers)
                print(
                    f"{blue('[ fetch_content ]')} :: Fetched {green(len(res.content))} {green('bytes')} from url: {url}"
                )
                chan.seen.add(url)
            except Exception as e:
                print(
                    blue("[ fetch_content ]")
                    + red(
                        f":: Failed to fetch url: {url}. {e.__class__.__name__} :: {e}"
                    )
                )
                return

        chan.output.append((url, res.content))

    except Exception as e:
        print(
            blue("[ fetch_content ]")
            + red(
                f" :: Encountered an unknown exception while fetching url {url}: {e.__class__.__name__} :: {e}"
            )
        )
        return


async def main():
    keep_going = True
    print(f"{cyan('[ eventloop ]')} :: Loaded {len(news_sources)} sources")
    _l = list(flatten_list([row["sitemap_urls"] for row in news_sources.values()]))
    queue = random.sample(_l, len(_l))
    print(queue)
    if MAX_SOURCES and len(queue) >= MAX_SOURCES:
        queue = random.sample(queue, MAX_SOURCES)
    sitemap_urls = set(queue)
    chan.queue = deque(queue)
    iterations = 0
    while keep_going:
        iterations += 1
        print(
            cyan(
                f"[ eventloop ] :: Iteration #{iterations} is now starting. Initializing the nursery..."
            )
        )
        async with trio.open_nursery() as nursery:
            for i in range(min(len(chan.queue), MAX_REQUESTS)):
                # print(f"Processing item {i}")
                raw_url = chan.queue.popleft()
                if raw_url in sitemap_urls:
                    print(
                        cyan("[ eventloop ]")
                        + f" :: Scheduling {cyan('sitemap crawl')} for url: {raw_url}"
                    )
                    nursery.start_soon(fetch_sitemap, raw_url)
                    # sitemap payloads seem to be exceeding the maximum buffer size for asyncio operations...
                    # try requesting only 2 at a time?

                else:
                    print(
                        cyan("[ eventloop ]")
                        + f" :: Scheduling {magenta('content crawl')} for url: {raw_url}"
                    )
                    nursery.start_soon(fetch_content, raw_url)
                    # print(f"Got url {raw_url}")
        print(
            cyan(
                f"[ eventloop ] :: Iteration #{iterations} is now complete. There are now {len(chan.output)} HTTP responses in the output buffer."
            )
        )
        if len(chan.output) >= BUFFER_SIZE or not bool(chan.queue):
            # chan.queue = deque(random.sample(list(chan.queue), len(chan.queue)))
            print(cyan(f"[ eventloop ] :: Processing those responses now."))
            processed = []
            curr = 0
            for url, html in chan.output:
                curr += 1
                print(cyan(f"[ eventloop ] :: Processing url #{curr}: {url}"))
                dup_row = crawldb.find_one(article_url=url)
                # chan.seen.add(url)
                parsed = NewsArticle(url)
                parsed.download(html)
                parsed.parse()
                parsed.nlp()
                city, state = None, None
                site = re.sub(
                    r"(https?://|www\.)*",
                    "",
                    url_normalize(urlparse(parsed.source_url).netloc),
                )
                sourceloc = None
                if site in news_sources:
                    sourceloc = news_sources[site]["loc"]
                    author = news_sources[site]["name"]
                    try:
                        city, state = sourceloc.split(", ")
                    except Exception as e:
                        print(
                            red(
                                f"[ parser ] :: Invalid 'loc' value for site {site}: {sourceloc}. This value needs to have the form: (<city>, <state>)."
                            )
                        )
                        city = sourceloc

                else:
                    for k in sorted(list(news_sources.keys())):
                        print(f"    {yellow(k)}")
                    print(
                        yellow(
                            f"[ parser ] :: The current site '{site}' could not be found in the list of news sources above. I will attempt to find a partial match for this source, but you may wish to revise the load_news_sources procedure."
                        )
                    )
                    ok = False
                    for k, v in news_sources.items():
                        if site in k or k in site:
                            print(
                                green(
                                    f"[ parser ] :: Success! Key {k} appears to match site {site}. I will add that as an alias for {k} to expedite parsing those URLs in the future."
                                )
                            )
                            sourceloc = v["loc"]
                            author = v["name"]
                            news_sources[site] = v
                            ok = True
                            break
                    if not ok:
                        print(
                            red(
                                f"[ parser ] :: I couldn't find any partial matches for {site} either. Resorting to using levenshtein distance..."
                            )
                        )
                        ranked = list(
                            sorted(
                                [(levenshtein(site, k), k) for k in news_sources.keys()]
                            )
                        )
                        print(
                            f"[ parser ] :: I will use {cyan(ranked[0][1])} for {cyan(site)} because it has the smallest edit distance. If this is incorrect, you'll want to edit the load_news_sources procedure."
                        )
                        k = ranked[0][1]
                        v = news_sources[k]
                        sourceloc = v["loc"]
                        author = v["name"]
                        news_sources[site] = v
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
                keywords = set()
                article_metadata = FlatDict(dict(parsed.meta_data))
                published = parsed.publish_date
                modified = parsed.publish_date
                description = parsed.summary
                print(json.dumps(parsed.meta_data, indent=4, default=str))
                print(
                    f"====================== END OF METADATA FOR URL {url} =========================="
                )
                try:
                    for k, v in article_metadata.items():
                        if "description" in k:
                            description = unidecode(unescape(v))
                            print(
                                green(
                                    f"[ parser ] Found metadescription for article {url}:"
                                )
                            )
                            print(
                                f"============================== SUMMARY ===================================\n {blue(parsed.summary)}\n\n============================== METADESCRIPTION ==========================================\n{magenta(description)}\n\n"
                            )
                            print(
                                green(
                                    f"[ parser ] Using METADESCRIPTION instead for article {url}."
                                )
                            )
                            break

                except Exception as e:
                    print(
                        yellow(
                            f"[ parser ] No metadescription for article {url}. \n           Falling back to summary: \n\n {blue(description)}\n ==============================================================================="
                        )
                    )
                    pass

                try:
                    for k, v in article_metadata.items():
                        if "modified" in k and isinstance(v, str):
                            modified = parse_timestamp(
                                parsed.meta_data["article"]["modified"]
                            )
                            break
                        elif "modified" in k and isinstance(v, datetime.datetime):
                            modified = v
                            break
                except:
                    pass

                try:
                    # schema_types = ['blogpost', 'article', 'newsarticle']
                    # # parsed.meta_data = dict(parsed.meta_data)
                    # if parsed.meta_data and any(k in parsed.meta_data and parsed.meta_data[k] for k in schema_types):
                    #     metadata = {}
                    #     for k in schema_types:
                    #         if k in parsed.meta_data:
                    #             print(green(f"[ parser ] URL {url} metadata contains a {k} object:"))
                    #             print(green(json.dumps(parsed.meta_data[k], indent=4, default=str)))
                    #             metadata.update(parsed.meta_data[k])
                    #     print("============================================== AGGREGATED METADATA =========================================")
                    #     print(green(json.dumps(metadata, indent=4, default=str)))
                    #     print(green(f"[ parser ] Aggregated metadata for url {url} has keys:"))
                    #     for i, k in enumerate(metadata.keys()):
                    #         print(f"    {i}. {yellow(k)}")
                    ks = [
                        k
                        for k in article_metadata.keys()
                        if "keyword" in k.lower() or "tag" in k.lower()
                    ]

                    keyword_strings = ", ".join([str(article_metadata[k]) for k in ks])
                    alt_tags = list(
                        set(
                            [
                                kw.strip().replace("-", " ").title()
                                for kw in re.findall(r"([^,;:]{4,})", keyword_strings)
                            ]
                        )
                    )
                    print(
                        green(
                            f"[ parser ] Found {len(alt_tags)} additional news keywords from {len(ks)} keys in the metadata for article {url}:"
                        )
                    )

                    for i, tag in enumerate(alt_tags):
                        print(cyan(f"  {i}.  {tag}"))
                    print(f"[ parser ] Keyword attributes present in site {site}:")
                    for i, k in enumerate(ks):
                        print(cyan(f"   {i}. {k}"))

                    keywords = list(keywords.union([unidecode(kw) for kw in alt_tags]))
                # else:
                #     print(f"===================================================== METADATA ===============================================")
                #     print(yellow(json.dumps(parsed.meta_data, indent=4, default=str)))
                #     print(f"==============================================================================================================")
                #     print(yellow(f"[ parser ] Couldn't extract keywords for url {url} because site {site} does not appear to have a schema object. If the object above does contain useful keyword data, you may want to modify the parsing procedure"))
                except Exception as e:
                    print(
                        f"=================================================== METADATA =============================================="
                    )
                    try:
                        print(red(json.dumps(parsed.meta_data, indent=4, default=str)))
                    except:
                        print(red(parsed.meta_data))
                    print(
                        f"==========================================================================================================="
                    )
                    print(
                        red(
                            f"[ parser ] Encountered {e.__class__.__name__} while extracting keywords from url {url} : {e}"
                        )
                    )

                    pass

                category = []
                for k, v in article_metadata.items():
                    if "section" in k.lower():
                        category.append(v)

                # if not published:
                dom = parse_html(html)
                soup = BeautifulSoup(html, from_encoding="utf-8")
                metadata = extract_schemata(dom)
                article = Article(url, dom, metadata, soup)
                published = article._datePublished
                modified = article._dateModified

                row = {
                    "raw_content": unidecode(article.content),
                    "content": unidecode(article.content),
                    "title": unidecode(parsed.title),
                    "summary": unidecode(description),
                    "keywords": keywords,
                    "image_url": parsed.top_image,
                    "article_url": url,
                    "author": ", ".join(parsed.authors),
                    "category": category,
                    "source_id": site,
                    "metadata": json.loads(json.dumps(dict(article_metadata))),
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
                    "state": state,
                }
                if dup_row:
                    print(red(f"[ parser ] Url {url} appears to be a duplicate!"))
                    print(
                        "================================================== PARSED =========================================="
                    )
                    print(blue(json.dumps(row, indent=4, default=str)))
                    print(
                        "===================================================================================================="
                    )
                    print(
                        "================================================= DUPLICATE ========================================"
                    )
                    print(magenta(json.dumps(row, indent=4, default=str)))
                    print(
                        "===================================================================================================="
                    )
                    print(
                        red(
                            f"[ parser ] Not adding url {url}, as that would violate the uniqueness constraint."
                        )
                    )
                    continue
                if not re.search(
                    r"(covid|virus|hospital|pandemic|corona)", str(row), re.IGNORECASE
                ):
                    print(
                        yellow(
                            f"[ parser ] :: No match for coronavirus in article: {url}"
                        )
                    )
                    continue

                else:
                    print(json.dumps(row, indent=4, default=str))
                    print(
                        green(
                            f"[ parser ] Finished parsing {url}. {len(processed)} total rows are now in the buffer."
                        )
                    )
                    processed.append(row)

            print(green(f"[ eventloop ] Upserting {len(processed)} rows..."))
            crawldb.upsert_many(processed, ["article_url"])
            chan.output = []
            keep_going = bool(chan.queue)
            print(
                green(f"[ eventloop ] {len(chan.queue)} urls remaining in the queue.")
            )
            with open("seen.pkl", "wb") as f:
                try:
                    pickle.dump(chan.seen, f)
                except Exception as e:
                    bad = []

                    print(e.__class__.__name__, e)
                    pass

            # deduplicate_moderation_table(crawldb)


if __name__ == "__main__":
    deduplicate_moderation_table(crawldb)
    trio.run(main)
    deduplicate_moderation_table(crawldb)
