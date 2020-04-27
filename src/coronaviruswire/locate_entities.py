from src.coronaviruswire.common import db
from shapely.geometry import Point
from collections import deque
from unidecode import unidecode
from more_itertools import grouper
from pylev import damerau_levenshtein
import os
import json
from urllib.parse import quote_plus
import httpx
from munch import Munch
import trio
from fuzzywuzzy import fuzz
import plotly.io as pio
import re
import ast

pio.renderers.default = "browser"
import plotly
from geopy.distance import geodesic
import subprocess
from math import log
import matplotlib.pyplot as plt
import numpy as np

import shapely.geometry as sg
from shapely.ops import cascaded_union, unary_union, polygonize
import shapely.affinity
from itertools import combinations

global my_coords

coords = {}
crawldb = db["moderationtable"]
k = copy(globals())
for s, val in k.items():
    if val == search_for_place:
        print(s)

try:
    with open("geocache.json", "r") as f:
        cache = json.load(f)
except:
    cache = {}


def cache_queries(func):
    sym = func.__name__
    cache[sym] = {}

    def wrapped(*args, **kwargs):
        hashable = (repr(args), repr(kwargs))
        if hashable in cache[sym]:
            print(f"Reusing cached query: {hashable}")
            return cache[sym][hashable]
        print(f"Cache miss. Executing API query for: {hashable}")
        result = func(*args, **kwargs)
        cache[sym][hashable] = result
        with open("geocache.json", "w") as f:
            json.dump(cache, f)
        return result

    return wrapped


def cache_wikipedia_queries(func):
    sym = func.__name__
    if not sym in cache:
        cache[sym] = {}

    def wrapped(queries):
        acc = {}
        args = []
        for query in queries:
            if query in cache[sym]:
                print(f"Reusing cached wikipedia search for: {query}")
                acc[query] = cache[sym][query]
            else:
                print(f"Wikipedia cache miss: {query}")
                args.append(query)
        result = func(queries)
        return dict(acc, **result)

    return wrapped


def resolve_entity(raw_string, bias=None):
    args = raw_string if not bias else [raw_string, bias]
    result = search(args)
    candidates = []
    if bias and bias in result and result[bias]:
        for rank, name, coords in result[bias]:
            c1 = coords
            break
        assert c1, "Location not found: {bias}"

    if raw_string in result and result[raw_string]:
        for v in result[raw_string]:
            rank, name, coords = v
            lat = coords[0] if coords else None
            lon = coords[1] if coords else None
            dist = math.inf if (not coords or not bias) else geodesic(c1, coords)
            edit_dist = damerau_levenshtein(raw_string, name)
            ratio = fuzz.ratio(raw_string, name)

            candidates.append(
                Munch(
                    {
                        "query": raw_string,
                        "bias": bias,
                        "result": name,
                        "rank": rank,
                        "geodist": dist,
                        "editdist": edit_dist,
                        "ratio": ratio,
                        "coords": coords,
                        "lat": lat,
                        "lon": lon,
                        "has_coords": bool(coords),
                    }
                )
            )

    _sorted = sorted(
        candidates, key=lambda result: (result.geodist, result.editdist, result.ratio)
    )
    return _sorted


class chan:
    output = []


def diag2poly(p1, p2):
    points = [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]
    return points


def diag2circle(p1, p2):
    ys = (p1[1], p2[1])
    xs = (p1[0], p2[0])
    ymin = min(ys)
    ymax = max(ys)
    xmax = max(xs)
    xmin = min(xs)
    dy = ymax - ymin
    dx = xmax - xmin
    cy = sum(ymin, ymax) / 2
    cx = sum(xs) / 2
    bott = (cx, cy + dy)
    side = (cx + dx, cy)
    dist2bottom = geodesic((cx, cy), (cx, cy + dy))
    dist2side = geodesic((cx, cy), (cx + dx, cy))
    return Munch(
        {
            "cx": cx,
            "cy": cy,
            "c": (cx, cy),
            "r": cy,
            "bottom": bott,
            "side": side,
            "rad_length_km": Munch(
                {"bottom": dist2bottom.kilometers, "side": dist2side.kilometers}
            ),
        }
    )


def similarity(a, b):
    ratio = fuzz.ratio(a, b)
    dist = damerau_levenshtein(a, b)
    diff_len = abs(len(a) - len(b))
    len_penalty = log(len(a) / (1 + diff_len))
    # penalty = 0.5 + 1/log(1 + dist)

    score = ratio
    return Munch(locals())


async def search_for_place_async(
    place_name, location=None, radius=300, self_coords=None
):
    print(f"Async func has self_coords: {my_coords}")
    assert my_coords and len(my_coords) == 2
    if location is None:
        center = None
    elif location in coords:
        center = coords[location][0]
    else:
        response = await search_for_place_async(location)
        center = response.center

    if center:
        lat, long = center
        location_bias = f"&locationbias=circle:{radius}@{lat},{long}"
    else:
        location_bias = ""

    query = quote_plus(place_name)
    s = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}{location_bias}&fields=formatted_address,name,geometry&inputtype=textquery&key=AIzaSyALE94yjbDhNRZbigm6xnaDnnSIe4Vlw00"
    async with httpx.AsyncClient() as client:
        res = await client.get(s)
        print(res.json())
    try:
        candidate = res.json()["candidates"][0]
    except Exception as e:
        print(e)
        print(res.json())
        return Munch({"ok": False, "center": None, "similarity": 0.0})
    c1 = list(candidate["geometry"]["location"].values())
    diag1 = [(v["lat"], v["lng"]) for k, v in candidate["geometry"]["viewport"].items()]
    box1 = diag2poly(*diag1)
    coords[place_name] = (c1, diag1, box1)
    lat, long = c1
    sim = similarity(place_name, candidate["name"])
    if center:
        dist = geodesic(c1, center).kilometers
        bias = {"lat": lat, "long": long, "radius": radius}

    else:

        dist = 0
        bias = {"lat": None, "long": None, "radius": None}
    import math

    distance_to_here = geodesic(c1, self_coords)
    print(f"{c1} is {distance_to_here.kilometers}km to {my_coords}")

    if distance_to_here and distance_to_here.kilometers < 500:
        penalty = 0.1  # math.log(dist.kilometers, distance_to_here.kilometers)
    else:
        penalty = 1
    final_score = sim.score

    obj = Munch(
        {
            "ok": True,
            "query": place_name,
            "center": c1,
            "lat": c1[0],
            "long": c1[1],
            "diag": diag1,
            "box": box1,
            "dist": dist,
            "name": unidecode(candidate["name"]),
            "address": candidate["formatted_address"],
            "similarity": sim,
            "edit_distance": sim.dist,
            "self_lat": self_coords[0],
            "self_long": self_coords[1],
            "proximity_to_self": distance_to_here.kilometers,
            "proximity_penalty": penalty,
            "score": final_score * penalty,
            "bias": bias,
        }
    )
    print(json.dumps(obj, indent=4))
    return obj


async def locate_all(article_id, entities, origin, my_coords):
    geo_ents = {}
    print(f"My coords are {my_coords}")

    async def locate_entity(entity, origin, my_coords):
        print(f"Value of my_coords now {my_coords}")
        ent = await search_for_place_async(entity, origin, 500, self_coords=my_coords)

        if ent.ok and ent.score >= 70 and ent.edit_distance < 8:
            geo_ents[entity] = ent

    queue = deque(entities)
    while queue:
        async with trio.open_nursery() as nursery:
            for i in range(5):
                try:
                    entity = queue.popleft()
                except:
                    break
                nursery.start_soon(locate_entity, entity, origin, my_coords)
    chan.output.append(
        {
            "article_id": article_id,
            "geotags": geo_ents,
            "has_geotags": True,
            "has_ner": True,
        }
    )
    return geo_ents


@cache_wikipedia_queries
def search(places):
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
        print(f"Fetching result {i} for search: {place} :: {result}")
        url = f"https://en.wikipedia.org/wiki/{result.replace(' ', '_')}"
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            dom = fromstring(res.content)
            lat = dom.cssselect(".latitude")
            lng = dom.cssselect(".longitude")
            if lat and lng:
                results[place].append(
                    (i, result, (lat[0].text_content(), lng[0].text_content()))
                )

    async def search_for(args):
        place, url = args
        async with httpx.AsyncClient() as client:
            res = await client.get(url)
            dom = fromstring(res.content)
            print(dom.xpath("//h1")[0].text_content())
            results = dom.xpath("//li[contains(@class, 'mw-search-result')]//a/@title")
            for i, result in enumerate(results):
                print(f"Queueing result {i} for search: {place} :: {result}")
                queue.append((place, i, result))

    async def start_search():
        while queue:
            async with trio.open_nursery() as nursery:
                for i in range(len(queue)):
                    args = queue.popleft()
                    f = search_for if args in search_urls else get_page
                    nursery.start_soon(f, args)

    trio.run(start_search)
    _sorted = {k: sorted(v) for k, v in results.items()}

    return _sorted


if __name__ == "__main__":
    from src.coronaviruswire.utils import get_geocoords

    loc = get_geocoords()
    assert loc["ok"]
    lat = loc["lat"]
    long = loc["lng"]
    my_coords = (lat, long)
    print(
        f"Couldn't find ip2geocoords.sh! You probably need to change your working directory to the project root."
    )

    for row in crawldb.find(has_geotags=False):
        if row["ner"] is None:
            continue
        article_id = row["article_id"]
        city = row["city"]
        country = row["country"]
        loc = ", ".join([city, country])
        try:
            ent_counts = row["ner"]

            ents = list(ent_counts.keys())
            trio.run(locate_all, article_id, ents, loc, my_coords)
            if chan.output and len(chan.output) > 10:
                crawldb.update_many(chan.output, ["article_id"])
                chan.output = []
        except Exception as e:
            pass
