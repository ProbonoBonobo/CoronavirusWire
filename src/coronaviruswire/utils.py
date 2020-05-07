import json
import csv
import random
import os
from hashlib import blake2b
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
import spacy
import termcolor
from gemeinsprache.utils import blue, red
import us

nlp = None


def normalize_state_name(state):

    stateCodeToNameDict = us.states.mapping('abbr', 'name')

    if len(state) > 3:
        return state

    state_lowercase = state.upper()
    full_name = stateCodeToNameDict[state_lowercase]
    if full_name:
        return full_name
    else:
        return state

def async_fetch(*urls, max_requests=25, headers=default_headers, timeout=60, **kwargs):
    if isinstance(urls, str):
        urls = [urls]
    urls = [url_normalize(url) for url in urls]
    chan = {}

    async def fetch():
        limit = trio.CapacityLimiter(max_requests)

        async def async_fetch(url, headers, timeout, **kwargs):
            async with httpx.AsyncClient() as client:
                chan[url] = await client.get(
                    url, headers=default_headers, timeout=timeout, **kwargs
                )

        async with trio.open_nursery() as nursery:
            for url in urls:
                async with limit:
                    nursery.start_soon(async_fetch, url, headers, timeout, **kwargs)

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


def iter_csv(fp="./lib/newspapers.csv", delimiter="\t"):
    with open(fp, "r") as f:
        f_csv = csv.DictReader(f, delimiter=delimiter)
        cols = f_csv.fieldnames
        for row in f_csv:
            yield dict(row.items())


def load_csv(fp="./lib/newspapers.tsv", delimiter="\t"):
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
def load_news_sources(fp="./lib/newspapers.tsv"):
    fp = os.path.abspath(fp)
    news = load_csv(fp)
    loaded = {}
    for row in list(news):
        resolved_urls = []
        for k, v in list(row.items()):
            if len(resolved_urls) > 10:
                break
            if not v:
                continue
            elif k.startswith("sitemap_url_template"):
                resolved = datetime.datetime.now().strftime(v)
                resolved_urls.append(resolved)
            elif k.startswith("sitemap_url"):
                resolved_urls.append(v)
        print("resolved_urls")
        print(resolved_urls)
        print(row)
        url = url_normalize(row['url']).strip().lower()
        parsed = urlparse(url)
        row['url'] = url
        row['site'] = re.sub(r"(https?://|www\.)", "", url_normalize(parsed.netloc))
        row['sitemap_urls'] = resolved_urls
        if row['sitemap_urls']:
            loaded[row['site']] = row
    return loaded

def load_news_sources(fp="./lib/newspapers.tsv"):
    fp = os.path.abspath(fp)
    news = load_csv(fp)
    loaded = {}
    for row in list(news):
        resolved_urls = []
        for k, v in list(row.items()):
            if len(resolved_urls) > 10:
                break
            if not v:
                continue
            elif k.startswith("sitemap_url_template"):
                resolved = datetime.datetime.now().strftime(v)
                resolved_urls.append(resolved)
            elif k.startswith("sitemap_url"):
                resolved_urls.append(v)
        print("resolved_urls")
        print(resolved_urls)
        print(row)
        url = url_normalize(row["url"]).strip().lower()
        parsed = urlparse(url)
        row["url"] = url
        row["site"] = re.sub(r"(https?://|www\.)", "", url_normalize(parsed.netloc))
        row["sitemap_urls"] = resolved_urls
        if row["sitemap_urls"]:
            loaded[row["site"]] = row
    return loaded


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
        # html_chars_padded = re.sub(r"(\S?)(&\w+\;)(\S?)", r"\1 \2 \3",
        #                            txt.strip())
        # padded = re.sub(r"(\S{4,})([\.\!\?]+|\-{3,})(\'?\"?)([A-Z])",
        #                 r"\1\2 \n \3", unidecode(unescape(html_chars_padded)))
        # no_camels = re.sub(r"([a-z]{2,})([A-Z]\w*)", r"\1. \2", padded)
        no_html_tags = re.sub(
            r"(\s\<[^\>]*\>\s*|\s{2,})", " ", unidecode(unescape(txt.strip()))
        ).strip()
        return no_html_tags
    else:
        return txt


def deg2dec(coord):
    patt = re.compile(
        r"(?P<deg>[\d\.]{0,3})°\s?(?P<m>[\d\.]{0,6})?′?\s?(?P<s>[\d\.]{0,8})?[^\w]?(?P<dir>[NWSE])"
    )
    lat, lon = [re.search(patt, c).groupdict() for c in coord]
    lat["m"] = float(lat["m"]) if lat["m"] else 0
    lon["m"] = float(lon["m"]) if lon["m"] else 0
    lat["s"] = float(lat["s"]) if lat["s"] else 0
    lon["s"] = float(lon["s"]) if lon["s"] else 0

    if lat["s"]:
        lat["m"] = lat["m"] + (lat["s"] / 60)
    if lon["s"]:
        lon["m"] = lon["m"] + (lon["s"] / 60)
    lat_m = lat["m"] / 60
    lon_m = lon["m"] / 60
    lat = (int(lat["deg"]) + lat_m) * [1, -1][int(lat["dir"] in "WS")]
    lon = (int(lon["deg"]) + lon_m) * [1, -1][int(lon["dir"] in "WS")]
    return lon, lat


def blake(thing):
    """Calculate a blake2b hash value for any valid object. If `thing` isn't a string, check to see if it has a __dict__
       representation that could be hashed instead (so different references to the same value will hash to the same
       value); otherwise, use its __repr__() value as a fallback."""
    thingstring = (
        thing
        if isinstance(thing, str)
        else repr(thing.__dict__)
        if hasattr(thing, "__dict__")
        else repr(thing)
    )
    return blake2b(thingstring.encode("utf-8")).hexdigest()


import inspect
from itertools import zip_longest


def serialize_call_args(f):
    argspec = inspect.getfullargspec(f)
    argnames = argspec.args
    n_argnames = len(argnames) if argnames else 0
    n_defaults = len(argspec.defaults) if argspec.defaults else 0
    print(n_argnames, n_defaults)
    default_vals = [None for i in range(n_argnames - n_defaults)]
    if n_defaults:
        default_vals += list(argspec.defaults)

    def wrapped(*args, **kwargs):
        argmap = {}
        for k, arg, default in zip_longest(argnames, args, default_vals):
            argmap[k] = arg if arg else kwargs[k] if k in kwargs else default
        return argmap

    return wrapped


def cache_queries(func):
    sym = func.__name__

    argmapper = serialize_call_args(func)

    def wrapped(*args, **kwargs):
        try:
            with open("geocache.json", "r") as f:
                cache = json.load(f)
        except:
            cache = {}
            with open("geocache.json", "w") as f:
                json.dump(cache, f)
        if sym not in cache:
            cache[sym] = {}
        call_args = argmapper(*args, **kwargs)
        hashable = blake(call_args)
        if "cache_override" in kwargs and kwargs["cache_override"]:
            print(f"Overriding cached query: {args}")
        elif hashable in cache[sym]:
            print(f"Reusing cached query: {hashable}")
            return cache[sym][hashable]["__output__"]
        else:
            print(f"Cache miss. Executing API query for: {hashable}")
        out = func(*args, **kwargs)
        call_args["__output__"] = out
        cache[sym][hashable] = call_args
        with open("geocache.json", "w") as f:
            json.dump(cache, f)
        return cache[sym][hashable]["__output__"]

    return wrapped


def initialize_kmedoids_model(
    path_to_points="./lib/us_metros_scraped_geocoords.tsv"
):

    import numpy as np
    from sklearn.cluster import KMeans

    coords = []
    labels = []
    for row in load_csv(path_to_points, delimiter="\t"):
        try:
            coords.append(list(deg2dec((row["latitude"], row["longitude"]))))
            labels.append(row["parent"])
        except Exception as e:
            print(e)
    arr = np.array([c for c in coords if c and len(c) == 2])
    k = 64
    from pyclustering.cluster.kmedoids import kmedoids

    # Load list of points for cluster analysis.

    # Set random initial medoids.
    initial_medoids = random.sample(range(0, len(arr)), k=k)
    # Create instance of K-Medoids algorithm.
    kmedoids_instance = kmedoids(arr, initial_medoids)
    # Run cluster analysis and obtain results.
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    # Show allocated clusters.
    print(clusters)
    return kmedoids_instance
    # Display clusters.
    # visualizer = cluster_visualizer()
    # visualizer.append_clusters(clusters, sample)
    # visualizer.show()


from munch import Munch


def cache_wikipedia_queries(func):
    sym = func.__name__

    argmapper = serialize_call_args(func)

    def wrapped(*args, **kwargs):
        try:
            with open("geocache.json", "r") as f:
                cache = json.load(f)
        except:
            cache = {}
            with open("geocache.json", "w") as f:
                json.dump(cache, f)
        if sym not in cache:
            cache[sym] = {}
        # print(cache['search'])
        print(f"Args: {args}, kwargs: {kwargs}")
        callargs = argmapper(*args, **kwargs)
        print(f"Call args: {callargs}")
        acc = {}
        args = []

        for query in callargs["places"]:
            print(f"Next query is: {query}")
            call_args = argmapper([query], **kwargs)
            hashable = blake(call_args)
            print(f"Hash: {hashable}")
            if "cache_override" in kwargs and kwargs["cache_override"]:
                print(f"Overriding cached query: {query}")
                args.append(query)
            elif hashable in cache[sym]:
                print(f"Reusing cached wikipedia search for: {query}")
                acc[query] = [Munch(v) for v in cache[sym][hashable]["__output__"]]
            else:
                print(f"Wikipedia cache miss: {query}")
                args.append(query)
        result = func(args)
        for query, v in result.items():
            call_args = argmapper([query], **kwargs)
            hashable = blake(call_args)
            call_args["__output__"] = v
            cache[sym][hashable] = call_args
            acc[query] = result

        with open("geocache.json", "w") as f:
            json.dump(cache, f)
        return acc

    return wrapped


from urllib.parse import quote_plus
from collections import defaultdict


@cache_wikipedia_queries
def search(places, top_n=2, cache_override=False):
    search_urls = [
        (
            place,
            f"https://en.wikipedia.org/w/index.php?cirrusUserTesting=control&sort=relevance&search={quote_plus(place)}&title=Special:Search&profile=advanced&fulltext=1&advancedSearch-current=%7B%7D&ns0=1",
        )
        for place in places
    ]

    queue = deque(search_urls)
    results = defaultdict(list)

    async def get_page(args):
        place, i, result = args
        # print(f"Fetching result {i} for search: {place} :: {result}")
        url = f"https://en.wikipedia.org/wiki/{result.replace(' ', '_')}"
        async with httpx.AsyncClient() as client:
            cats = []
            dec_coords = None
            lat = None
            lon = None
            res = await client.get(url)
            dom = fromstring(res.content)
            lat = dom.cssselect(".latitude")
            lng = dom.cssselect(".longitude")
            categories = dom.cssselect(".catlinks li a")
            links = dom.cssselect("#mw-content-text a")
            hrefs = []
            if lat and lng and lat[0].text_content() and lng[0].text_content():
                deg_coords = (lat[0].text_content(), lng[0].text_content())
                try:
                    dec_coords = deg2dec(deg_coords)
                    lat = dec_coords[0]
                    lon = dec_coords[1]
                except Exception as e:
                    # print(f"Failed to convert {deg_coords} to decimal!")
                    lat = None
                    lon = None
            if categories:
                cats = [cat.text_content() for cat in categories]
            if links:
                hrefs = [link.text_content() for link in links]
            results[place].append(
                Munch(
                    {
                        "query": place,
                        "result": result,
                        "rank": i,
                        "coords": dec_coords,
                        "lat": lat,
                        "lon": lon,
                        "categories": cats,
                        "links": hrefs,
                    }
                )
            )

    async def search_for(args):
        place, url = args
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            dom = fromstring(res.content)
            print(dom.xpath("//h1")[0].text_content())
            results = dom.xpath("//li[contains(@class, 'mw-search-result')]//a/@title")
            for i, result in enumerate(results):
                if i <= top_n:
                    # print(f"Queueing result {i} for search: {place} :: {result}")
                    queue.append((place, i, result))

    async def start_search():
        while queue:
            async with trio.open_nursery() as nursery:
                for i in range(len(queue)):
                    args = queue.popleft()
                    f = search_for if args in search_urls else get_page
                    nursery.start_soon(f, args)

    trio.run(start_search)
    _sorted = {k: sorted(v, key=lambda result: result.rank) for k, v in results.items()}

    return _sorted


def flatten_list(alist):
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                yield subitem
        else:
            yield item


def extract_entities(s):
    global nlp
    if not nlp:
        print(f"Loading spaCy NER model...")
        nlp = spacy.load("en_core_web_sm")
        print(f"Load complete.")

    ents = []
    doc = nlp(s)
    for sent in doc.sents:
        print(sent)
        for ent in sent.ents:
            if ent.label_ in (
                "ORG",
                "GPE",
                "LOC",
                "PERSON",
                "FAC",
                "NORP",
                "WORK_OF_ART",
                "EVENT",
                "PRODUCT",
                "LANGUAGE",
            ):
                print(f"        {blue(ent.label_):>16} {red(ent.string)}")
            if ent.label_ in ("ORG", "GPE", "FAC", "LOC", "NORP", "EVENT"):
                ents.append(ent.string)
            for neighbor in ent.lefts:
                if neighbor.string in ents:
                    ents.append("".join([ent.string, neighbor.string]))
    return ents


from pyproj import Geod


def calculate_bounding_box(point, km):
    lon, lat = point
    geod = Geod(ellps="WGS84")
    ne1, ne2, az1 = geod.fwd(lon, lat, az=45, dist=km / 2 * 1000, radians=False)
    sw1, se2, az2 = geod.fwd(lon, lat, az=az1, dist=km / 2 * 1000, radians=False)
    c1 = (ne1, ne2)
    c2 = (sw1, se2)
    return c1, c2


@cache_queries
def search_for_place(
    s=None,
    locationbias=None,
    radius=600,
    bounded=False,
    countrycodes="us",
    extratags=True,
    normalizecity=True,
    statecode=True,
    matchquality=True,
    addressdetails=True,
    polygon_geojson=False,
    reverse_lookup=False,
    latitude=None,
    longitude=None,
):
    import requests
    from urllib.parse import quote

    params = {
        "key": "4e4e7f8c29232f",
        "format": "json",
        "polygon_geojson": int(polygon_geojson),
        "countrycodes": countrycodes,
        "extratags": int(extratags),
        "normalizecity": int(normalizecity),
        "statecode": int(statecode),
        "matchquality": int(matchquality),
        "addressdetails": int(addressdetails),
        "q": s,
        "bounded": int(bounded),
    }
    if reverse_lookup:
        params = {
            "lat": latitude,
            "format": "json",
            "key": "4e4e7f8c29232f",
            "addressdetails": 1,
            "lon": longitude,
        }
    viewbox = ""
    if locationbias:
        try:
            print(f"Calculating viewbox for location: '{locationbias}'")
            res = search_for_place(locationbias, None)[0]
            print(f"Resolved {locationbias} to place: {res['display_name']}")
            c1, c2 = calculate_bounding_box(
                (float(res["lon"]), float(res["lat"])), radius
            )
            minlat, maxlat, minlon, maxlon = list(sorted([c1[1], c2[1], c2[0], c1[0]]))
            viewbox = ",".join([str(x) for x in [minlat, minlon, maxlat, maxlon]])
            params["viewbox"] = viewbox
            print(f"Using viewbox of size: {viewbox}")
        except Exception as e:
            print(
                f"{e.__class__.__name__} while fetching location '{locationbias}': {e}"
            )
    url = (
        f"https://us1.locationiq.com/v1/search.php"
        if not reverse_lookup
        else "https://us1.locationiq.com/v1/reverse.php"
    )
    res = requests.get(url, params=params)
    if not res.ok:
        print(json.dumps(params, indent=4))
        print(f"Response: {res.status_code}")
        return None
    # assert (
    #     res.ok
    # ), f"{res.content}\n\n=================================\nInvalid response from URL {url}. Got response code: {res.status_code}"
    return res.json()


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

    metadata = row["metadata"]
    objects = metadata["schemata"]

    typemap = {
        "articleBody": str,
        "headline": str,
        "description": str,
        "keywords": coll,
        "datePublished": dt,
        "dateModified": dt,
    }

    target_attributes = set(typemap.keys())
    for obj in flatten(objects):
        found_attrs = list(target_attributes.intersection(obj.keys()))
        tx = [typemap[k] for k in found_attrs]
        info = {
            attr: format_text(func(obj[attr])) for attr, func in zip(found_attrs, tx)
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
    duplicate_lines = len([line for line, count in counts.items() if count > max_count])
    sample_len = min(duplicate_lines, 20)
    sample = random.sample(list(counts.keys()), sample_len)
    print(f"Found {duplicate_lines} duplicate lines. Sampling {sample_len} lines:")
    for i, line in enumerate(sorted(sample, key=len)):
        print(f"  {i}. {line} ({counts[line]} occurrences)")

    for id, article in _formatted.items():
        uniques = [line for line in article if counts[line] <= max_count]
        deduplicated = "\n".join(uniques)
        output.append({"id": id, "before": index[id], "after": deduplicated})
        print(
            f"=================================================== BEFORE ================================================="
        )
        print(index[id])
        print(
            f"=================================================== AFTER =================================================="
        )
        print(deduplicated)

    return output


def deduplicate_moderation_table(tab):
    print(f"Indexing article contents...")
    updates = {
        row["article_id"]: row["raw_content"] for row in tab if row["raw_content"]
    }
    processed = [
        {"article_id": article["id"], "content": article["after"]}
        for article in deduplicate_content(updates)
    ]
    tab.update_many(processed, ["article_id"])
    return processed


def get_geocoords():
    import subprocess
    import requests
    from lxml.html import fromstring
    from src.coronaviruswire.common import default_headers

    ip = subprocess.check_output(
        "curl -s https://ipinfo.io/ip", shell=True, encoding="utf-8"
    )
    url = f"https://tools.keycdn.com/geo?host={ip}"
    res = requests.get(url, headers=default_headers)
    dom = fromstring(res.content)

    loc = dom.xpath('//*[@id="geoResult"]//dd')
    for node in loc:
        txt = node.text_content()
        print(txt)
        if "(lat)" in txt:
            match = [float(coord) for coord in re.findall(r"([\d\.\-]{3,})", txt)]
            lat, long = match
            print(f"Latitude: {lat}")
            print(f"Longitude: {long}")
            return {"lat": lat, "lng": long, "ok": True}
    else:
        return {"lat": None, "lng": None, "ok": False}


def deduplicate_table(tab):
    print(f"Indexing article contents...")
    updates = {row["id"]: row["articlebody"] for row in tab}
    processed = deduplicate_content(updates)

    return processed


if __name__ == "__main__":

    crawldb = db["moderationtable"]
    deduplicate_moderation_table(crawldb)
    # responses = async_fetch("msn.com", "yahoo.com", "nytimes.com",
    #                         "news.ycombinator.com")
    #
    # rows = [(row['headline'], row['url'], row['articlebody'])
    #         for row in crawldb]
    # headlines = [row[0] for row in rows]
    # articles = [row[-1] for row in rows]
    # cleaned = remove_cruft(articles)
    # for before, after in zip(articles, cleaned):
    #     print(f" ============= BEFORE ============== ")
    #     print(before)
    #     print(f" ============= AFTER =============== ")
    #     print(after)
    #     print("\n\n")
    # with open("./outputs/cleaned.json", "w") as f:
    #     out = []
    #     for before, after, headline in zip(articles, cleaned, headlines):
    #         obj = {"headline": headline, "before": before, "after": after}
    #         out.append(obj)
    #     json.dump({"output": out}, f, indent=4)
