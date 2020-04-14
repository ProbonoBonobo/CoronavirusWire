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

coords = {}
crawldb = db['moderationtable']

class chan:
    output = []

def diag2poly(p1, p2):
    points = [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]
    return points


def similarity(a, b):
    ratio = fuzz.ratio(a, b)
    dist = damerau_levenshtein(a, b)
    diff_len = abs(len(a) - len(b))
    len_penalty = log(len(a) / (1 + diff_len))
    #penalty = 0.5 + 1/log(1 + dist)

    score = ratio
    return Munch(locals())




async def search_for_place_async(place_name, location=None, radius=300, self_coords=None):
    print(f"Async func has self_coords: {my_coords}")
    assert(my_coords and len(my_coords) == 2)
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
    lat, long = c1
    sim = similarity(place_name, candidate['name'])
    if center:
        dist = geodesic(c1, center).kilometers
        bias = {"lat": lat, "long": long, "radius": radius}

    else:

        dist = 0
        bias = {"lat": None, "long": None, "radius": None}
    import math
    distance_to_here = geodesic(
        c1, self_coords)
    print(f"{c1} is {distance_to_here.kilometers}km to {my_coords}")

    if distance_to_here and distance_to_here.kilometers < 500:
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
        "name": unidecode(candidate['name']),
        "address": candidate['formatted_address'],
        "similarity": sim,
        "edit_distance": sim.dist,
        "self_lat": self_coords[0],
        "self_long": self_coords[1],
        "proximity_to_self": distance_to_here.kilometers,
        "proximity_penalty": penalty,
        "score": final_score * penalty,
        "bias": bias
    })
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
    chan.output.append({"article_id": article_id, "geotags": geo_ents, "has_geotags": True, "has_ner": True})
    return geo_ents

if __name__ == '__main__':
    from src.coronaviruswire.utils import get_geocoords
    loc = get_geocoords()
    assert loc['ok']
    lat = loc['lat']
    long = loc['lng']
    my_coords = (lat, long)
    print(f"Couldn't find ip2geocoords.sh! You probably need to change your working directory to the project root.")


    for row in crawldb.find(has_geotags=False):
        if row['ner'] is None:
            continue
        article_id = row['article_id']
        city = row['city']
        country = row['country']
        loc = ', '.join([city, country])
        try:
            ent_counts = row['ner']

            ents = list(ent_counts.keys())
            trio.run(locate_all, article_id, ents, loc, my_coords)
            if chan.output and len(chan.output) > 10:
                crawldb.update_many(chan.output, ['article_id'])
                chan.output = []
        except Exception as e:
            pass

