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
    normalize_state_name
)
from pylev import levenshtein
import pickle
from gemeinsprache.utils import red, yellow, green, blue, magenta, cyan
from src.coronaviruswire.pointAdaptor import Point, adapt_point
import sys
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
from morph import flatten
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
from flatdict import FlatDict, FlatterDict
import uuid
import nltk
import us
import os


nltk.download('punkt')
path_to_news = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lib/newspapers4.csv")
# this is a global variable because we need to reference its contents when building the database entry
news_sources = load_news_sources(path_to_news, delimiter=",")
# news_sources = [row for row in news_sources if 'washington' in str(row).lower()]

crawldb = db["moderationtable"]

MAX_SOURCES = 100
MAX_ARTICLES_PER_SOURCE = 10
MAX_REQUESTS = 5
BUFFER_SIZE = 100



tmp = []

# ======================== DANGER ZONE !!! =================================================
# changing this variable to True will drop the table. You have been warned!
DROP_TABLE = False

# ==========================================================================================

if DROP_TABLE:
    ans = ""
    while ans not in ("y", "n"):
        ans = input("You are about to drop the table. Are you sure you want to do that? [y/n]  ").strip()
    if ans == "n":
        print(f"Aborting. Please update variable `DROP_TABLE` in file crawler_v2.py.")
        sys.exit(1)


# i recommend dropping the moderation table before proceding, there are some small updates to the schema
create_moderation_table(drop_table=DROP_TABLE)

seen_urls = set([row["article_url"] for row in crawldb])
try:
    with open("seen.pkl", "rb") as f:
        seen_urls.update(pickle.load(f))
except:
    pass


class chan:
    queue = deque()
    output = deque()
    seen = seen_urls

tmp = []

def flatten_list(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item

def glob_metadata(dom):
     haystack = {'meta': {"attrs": []},
             'schema': []} 
     for obj in dom.xpath("//meta"):
         args = list(obj.attrib.values())
         if len(args) == 2:
             k,v = args
             haystack['meta'][k] = v
         elif len(args) > 2:
             v, *keys = list(reversed(args))
             for k in keys:
                 haystack['meta'][k] = v
         else:
             haystack['meta']['attrs'].extend(args)

     for obj in dom.xpath("//script"):
         try:
             txt = format_text(obj.text)
             for o in re.findall(r"(\{.+\})", txt, re.DOTALL|re.MULTILINE):

                 try:
                     o = json.loads(obj.text)
                     if o and '@type' in o and o['@type'] in ("Article", "NewsArticle", "VideoObject"):
                         haystack.update(o)
                     elif o:
                         haystack['schema'].append(o)

                 except json.decoder.JSONDecodeError as e:
                     print(f"Decode error: {e}")
                     print(obj.text)
                     continue
         except:
             continue

      
     haystack = flatten(haystack)
     needles = defaultdict(set) 
      
     for k,v in haystack.items(): 
         if any(common_upload_date_prefix in k.lower() for common_upload_date_prefix in ("uploaddate", "publi")):
             try:
                 parse_timestamp(v)
                 needles['published_at'].add(v)
             except:
                 continue
         elif 'modified' in k.lower():
             try:
                 parse_timestamp(v)
                 needles['updated_at'].add(v)
             except:
                 continue
         elif 'title' in k.lower(): 
             needles['title'].add(v) 
         elif 'section' in k.lower(): 
             needles['category'].add(v)
         elif 'author' in k.lower(): 
             needles['author'].add(v) 
         elif 'keyword' in k.lower() and v and isinstance(v, str):
             needles['keywords'].update([kw.strip() for kw in v.split(",")])
         elif 'tag' in k.lower() and v and isinstance(v, str):
             needles['keywords'].update([kw.strip() for kw in v.split(",")])
         elif 'description' in k.lower(): 
             needles['description'].add(v) 
         elif 'nlp' in k.lower(): 
             needles['nlp'].add((k,v))
         elif isinstance(v, str) and '2020' in v and not needles['published_at']:
             try:
                 to_datetime = parse_timestamp(v)
                 needles['published_at'].add(v)
                 needles['updated_at'].add(v)
             except Exception as e:
                 pass

     for k,v in needles.items():
         if len(list(v)) >= 1 and k in ('published_at', 'updated_at', 'author', 'title', 'description'):
             needles[k] = list(v)[0] 
         else: 
             needles[k] = list(v)

     haystack.update(needles) 
     return haystack 

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
        objects = []
        for meta in dom.xpath("//meta"):
            self.schema['meta'] = {}
            if meta.attrib and 'itemprop' in meta.attrib and 'content' in meta.attrib:
                self.schema['meta'][meta.attrib['itemprop']] = meta.attrib['content']
        for s in dom.xpath("//script[contains(@type,'application/ld+json')]"):
            try:
                objects.append(json.loads(s.text))
            except Exception as e:
                print(f"couldn't parse {s.text} :: {e}")
        objects.append(self.schema)
        self.schema = {k:v for k,v in FlatterDict(objects).items()}
        self.schema.update(schema)


        self.article_url = url_normalize(url)
        for k, v in self.required_attrs.items():
            # if k in self.schema and self.schema[k]:
            #     value = self.schema[k]
            # else:
                # compute the value from this object's property methods
            value = getattr(self, f"_{k}")

            setattr(self, v, value)
        print(json.dumps(self.schema, indent=4, default=str))
        self.raw_content = copy(self.content)
        site = re.sub(r"(https?://|www\.)", "", url_normalize(urlparse(url).netloc))
        self.author = site
        if site in news_sources:
            self.sourceloc = ', '.join([news_sources[site]["city"], news_sources[site]['state']])
            self.author = news_sources[site]["name"]
        else:
            for k, v in news_sources.items():
                if site in k or k in site:
                    self.sourceloc = ', '.join([news_sources[site]["city"], news_sources[site]['state']])
                    self.author = v["name"]
                    break
        self.sourcecountry = "us"


        self.article_id = str(uuid.uuid4())
        self.source_id = self.author




        # del self._dom
        # del self.schema
        if not isinstance(self.published_at, datetime.datetime):
            print(red("[ date ] "), red(f" :: NOT A DATE: {self.published_at}") )
            print(json.dumps(self.schema, indent=4, default=str))
            print("")
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
        dt = None
        if 'dateModified' in self.schema:
            print(f"[ date ] Got a legit modified date for url {self.article_url}: {self.schema['dateModified']}")
            dt = self.schema['dateModified']
        elif any('dateModified' in k for k in self.schema):
            for k, v in self.schema.items():
                if 'dateModified' in k:
                    print(f"[ date ] Found a needle in the haystack on key {k} with value {v} for url {self.article_url}")
                    dt = v
                    break
        else:
            for k,v in self.schema.items():
                try:
                    dt = parse_timestamp(v)
                    print(f"[ date ] Using {dt} extracted from key {k} ({v}) for {self.article_url}")
                    break
                except:
                    continue
        if not dt:
            dt = datetime.datetime.now()
        elif isinstance(dt, str):
            try:
                dt = parse_timestamp(dt)
            except Exception as e:
                print(f"[ date ] Error parsing dateModified timestamp {dt}: {e.__class__.__name__} :: {e}")
                dt = datetime.datetime.now()
        assert isinstance(dt, datetime.datetime), f"Not a date: {dt}"
        return dt

    @property
    def _datePublished(self):
        dt = None
        if 'datePublished' in self.schema:
            print(f"[ date ] Got a legit published date for url {self.article_url}: {self.schema['datePublished']}")
            dt = self.schema['datePublished']
        elif any('datePublished' in k for k in self.schema):
            for k, v in self.schema.items():
                if 'datePublished' in k:
                    print(f"[ date ] Found a needle in the haystack on key {k} with value {v} for url {self.article_url}")
                    dt = v
                    break
        else:
            for k,v in self.schema.items():
                try:
                    dt = parse_timestamp(v)
                    print(f"[ date ] Using {dt} extracted from key {k} ({v}) for {self.article_url}")
                    break
                except:
                    continue
        if not dt:
            dt = datetime.datetime.now()
        elif isinstance(dt, str):
            try:
                dt = parse_timestamp(dt)
            except Exception as e:
                print(f"[ date ] Error parsing datePublished timestamp {dt}: {e.__class__.__name__} :: {e}")
                dt = datetime.datetime.now()

        assert isinstance(dt, datetime.datetime), f"Not a date: {dt}"
        return dt

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
        # if len(body)<=10:
        #     for node in self._soup.find_all("div"):
        #         text = node.text
        #         if text and text.strip():
        #             body.append(text.strip())

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
    elements = soup.findAll("loc", limit=9999)
    urls = []
    for elem in elements:
        url = elem.text
        if url and url.strip() not in chan.seen:
            urls.append(url.strip())

    urls_length_before = len(urls)
    print(magenta("[ fetch_sitemap ] "), f":: Extracted {urls_length_before} from sitemap: {sitemap_url}")

    # Change to reverse chronological order, to get the most recent articles
    urls = list(reversed(urls))
    urls_length = min(len(urls), MAX_ARTICLES_PER_SOURCE)
    urls = urls[:urls_length]

    if urls_length_before != urls_length:
        print(blue("[ truncate_sitemap ] "), f":: Truncating to {urls_length} urls")

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
                if res.status_code == 200:
                    chan.seen.add(url)
            except Exception as e:
                print(
                    blue("[ fetch_content ]")
                    + red(
                        f":: Failed to fetch url: {url}. {e.__class__.__name__} :: {e}"
                    )
                )
                return
        mentions = re.findall(rb'(covid|virus|pandemic|infect)', res.content, re.IGNORECASE)
        if len(mentions) >= 5:
            print(blue("[ fetch_content ]"), green(f":: Queueing response from {url}. It contains {len(mentions)} of coronavirus."))
            chan.output.append((url, res.content))
        else:
            print(blue("[ fetch_content ]"), yellow(f":: Sending response from {url} to the dustbin, as it contains only {len(mentions)} mentions of coronavirus."))

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

                # chan.seen.add(url)
                parsed = NewsArticle(url)
                dom = parse_html(html)
                glob = glob_metadata(dom)
                soup = BeautifulSoup(html, from_encoding="utf-8")
                metadata = extract_schemata(dom)
                article = Article(url, dom, metadata, soup)

                article_metadata = FlatterDict(parsed.meta_data)
                article_metadata.update(article.schema)

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
                    sourceloc = ', '.join([news_sources[site]["city"], news_sources[site]['state']])
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
                            sourceloc = ', '.join([news_sources[site]["city"], news_sources[site]['state']])
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
                        sourceloc = ', '.join([v["city"], v['state']])
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


                published = parsed.publish_date

                modified = parsed.publish_date
                keywords = set()

                description = format_text(parsed.summary)
                print(json.dumps(parsed.meta_data, indent=4, default=str))
                print(
                    f"====================== END OF METADATA FOR URL {url} =========================="
                )
                try:
                    for k, v in article_metadata.items():
                        if "description" in k:
                            description = format_text(v)
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
                                v
                            )

                        elif "modified" in k and isinstance(v, datetime.datetime):
                            modified = v

                        if "published" in k and isinstance(v, str):
                            published = parse_timestamp(
                                v
                            )

                        elif "published" in k and isinstance(v, datetime.datetime):
                            published = v

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

                tmp.append(article)
                if not published:

                    published = article._datePublished
                    modified = article._dateModified

                title = unidecode(parsed.title)
                _keywords = set()
                _category = set()
                for kw in _keywords:
                    _keywords.update([format_text(_kw.strip()) for _kw in kw.split(",")])
                for cat in category:
                    _category.update([format_text(_cat.strip()) for _cat in cat.split(",")])
                if 'keywords' in glob:
                    _keywords.update(glob['keywords'])

                row = {
                    "raw_content": unidecode(article.content),
                    "content": unidecode(article.content),
                    "title": title,
                    "summary": description,
                    "keywords": list(_keywords),
                    "image_url": parsed.top_image,
                    "article_url": url,
                    "author": ", ".join(parsed.authors),
                    "category": list(_category),
                    "source_id": site,
                    "metadata": glob,
                    "sourceloc": sourceloc,
                    "sourcecity": city,
                    "sourcestate": normalize_state_name(state),
                    # "sourcelonglat": sourcelonglat,
                    "sourcecountry": sourcecountry,
                    "article_id": article_id,
                    "has_ner": False,
                    "has_geotags": False,
                    "has_coords": False,
                    # "published_at": glob['published_at'],
                    # "updated_at": glob['updated_at'],
                }
                try:
                    if glob['published_at']:
                        row['published_at'] = parse_timestamp(glob['published_at'])
                    if glob['updated_at']:
                        row['updated_at'] = parse_timestamp(glob['updated_at'])

                except Exception as e:
                    row['published_at'] = datetime.datetime(1960,1,1)
                    row['updated_at'] = datetime.datetime(1960,1,1)

                dup_row_url = crawldb.find_one(article_url=url)
                dup_row_title = crawldb.find_one(title=title)

                dup_row = (dup_row_url != None or dup_row_title != None)
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

                # r"(covid|virus)", f"{row['title']}\n{row['keywords']}\n{row['summary']}\n{row['content']}\n{row['metadata']}", re.IGNORECASE
                regexResults = re.findall(
                    r"(covid|virus|pandemic)", f"{row['title']}\n{row['summary']}\n{row['content']}", re.IGNORECASE
                )

                if not len(regexResults) >= 1:
                    print(
                        yellow(
                            f"[ parser ] :: No match for coronavirus in article: {url}"
                        )
                    )
                    continue

                # Article is good to be added
                print(json.dumps(row, indent=4, default=str))
                print(
                    green(
                        f"[ parser ] Finished parsing {url}. {len(processed)} total rows are now in the buffer."
                    )
                )
                if 'published_at' in row:
                    processed.append(row)

            print(green(f"[ eventloop ] Upserting {len(processed)} rows..."))
            crawldb.upsert_many(processed, ["article_url"])
            # tmp.extend(processed)
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
    # deduplicate_moderation_table(crawldb)
    trio.run(main)
    deduplicate_moderation_table(crawldb)
