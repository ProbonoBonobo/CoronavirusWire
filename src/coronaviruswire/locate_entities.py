from src.coronaviruswire.common import db
from shapely.geometry import Point
from collections import deque
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
pio.renderers.default = 'browser'
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
my_coords = (None, None)
coords = {}
crawldb = db['moderationtable']

class chan:
    output = []

def diag2poly(p1, p2):
    points = [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]
    return points


def similarity(a, b):
    ratio = fuzz.partial_ratio(a, b)
    dist = damerau_levenshtein(a, b)
    diff_len = abs(len(a) - len(b))
    len_penalty = log(len(a) / (1 + diff_len))
    #penalty = 0.5 + 1/log(1 + dist)

    score = ratio
    return Munch(locals())




async def search_for_place_async(place_name, location=None, radius=300):
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
    s = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}{location_bias}&fields=formatted_address,name,opening_hours,geometry&inputtype=textquery&key=AIzaSyALE94yjbDhNRZbigm6xnaDnnSIe4Vlw00"
    async with httpx.AsyncClient() as client:
        res = await client.get(s)
        print(res.json())
    try:
        candidate = res.json()['candidates'][0]
    except Exception as e:
        print(e)
        print(res.json())
        return Munch({"ok": False, "center": None, "similarity": 0.0})
    c1 = list(candidate['geometry']['location'].values())
    diag1 = [(v['lat'], v['lng'])
             for k, v in candidate['geometry']['viewport'].items()]
    box1 = diag2poly(*diag1)
    coords[place_name] = (c1, diag1, box1)
    sim = similarity(place_name, candidate['name'])
    if center:
        dist = geodesic(c1, center).kilometers
        bias = {"lat": lat, "long": long, "radius": radius}

    else:

        dist = 0
        bias = {"lat": None, "long": None, "radius": None}
    import math
    distance_to_here = geodesic(
        c1, my_coords)
    if dist and distance_to_here.kilometers < 500:
        penalty = 0.1  #math.log(dist.kilometers, distance_to_here.kilometers)
    else:
        penalty = 1
    final_score = sim.score

    obj = Munch({
        "ok": True,
        "query": place_name,
        "center": c1,
        "lat": c1[0],
        "long": c1[1],
        "diag": diag1,
        "box": box1,
        "dist": dist,
        "name": candidate['name'],
        "address": candidate['formatted_address'],
        "similarity": sim,
        "edit_distance": sim.dist,
        "self_lat": my_coords[0],
        "self_long": my_coords[1],
        "proximity_to_self": distance_to_here,
        "proximity_penalty": penalty,
        "score": final_score * penalty,
        "bias": bias
    })
    print(json.dumps(obj, indent=4))
    return obj


async def locate_all(article_id, entities, origin):
    geo_ents = {}
    async def locate_entity(entity, origin):
        ent = await search_for_place_async(entity, origin)

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
                nursery.start_soon(locate_entity, entity, origin)
    chan.output.append({"article_id": id, "geotags": geo_ents, "has_geotags": True, "has_ner": True})
    return geo_ents

if __name__ == '__main__':
    if os.path.isfile("ip2geocoords.sh"):
        out = subprocess.check_output("./ip2geocoords.sh")
        loc = [float(val) for val in re.split(r"\s", re.sub(r'(\"|\')', '', out.decode('utf-8'))) if val]
        lat, long = loc
        print(f"Lat: {lat}\nLong: {long}")

    for row in crawldb.find(has_geotags=False):
        if row['ner'] is None:
            continue
        article_id = row['article_id']
        city = row['city']
        country = row['country']
        loc = ', '.join([city, country])
        ent_counts = row['ner']
        ents = list(ent_counts.keys())
        trio.run(locate_all, article_id, ents, loc)
        if chan.output and len(chan.output) > 200:
            crawldb.update(chan.output, ['article_id'])
            chan.output = []

