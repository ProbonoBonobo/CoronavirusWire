import plotly.graph_objects as go
from shapely.geometry import Point
from copy import copy
from shapely.ops import cascaded_union
from lxml.html import fromstring
from itertools import combinations
from collections import deque
from more_itertools import grouper
from shapely.ops import polygonize, cascaded_union
from shapely.geometry import Polygon, LineString
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from pylev import damerau_levenshtein
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import math
from urllib.parse import quote_plus
from unidecode import unidecode
import random
import re
import plotly.graph_objects as go
import httpx
from munch import Munch
import trio
from fuzzywuzzy import fuzz
import plotly.io as pio

pio.renderers.default = "browser"
import plotly
from geopy.distance import geodesic

from math import log
import matplotlib.pyplot as plt
import numpy as np

import shapely.geometry as sg
from shapely.ops import cascaded_union, unary_union, polygonize
import shapely.affinity
from itertools import combinations
from more_itertools import grouper
import plotly.io as pio
import json


def intersect_circles(shapes, radius=0.61):
    try:
        shapes = [shape.buffer(radius) for shape in shapes]
        listpoly = [
            a.intersection(b) for a, b in combinations(shapes, 2)
        ]  # list of intersections
        rings = [
            LineString(list(pol.exterior.coords)) for pol in listpoly
        ]  # list of rings

        union = unary_union(rings)

        result = [
            geom for geom in polygonize(union)
        ]  # list all intersection geometries

        multi = cascaded_union(
            result
        )  # Create a single geometry out of all intersections
        try:
            hulls = list(multi.exterior.coords)
            ys, xs = [arr.tolist() for arr in multi.exterior.xy]
            xs = [a for a, b, c, d, e, f, g, h in grouper(xs, 8)]
            ys = [a for a, b, c, d, e, f, g, h in grouper(ys, 8)]
            centroid = [list(reversed(list(multi.centroid.coords)[0]))]
        except:
            hulls = []
            xs, ys = [], []
            centroid = []

            for poly in multi:
                poly = Polygon(
                    [
                        list(sorted([x, y]))
                        for x, y in list(poly.convex_hull.exterior.coords)
                    ]
                )
                within = any(poly.within(other) for other in hulls)
                covers = any(poly.covers(other) for other in hulls)
                disjoint = not (any(poly.intersects(other) for other in hulls))
                if within:
                    print(f"Skipping {poly}")

                elif disjoint:
                    hulls.append(poly.convex_hull)
                else:
                    for i, hull in enumerate(hulls):
                        if poly.covers(hull):
                            print(f"replacing hull {i}")
                            hulls[i] = poly.convex_hull
                        if poly.intersects(hull):
                            print(f"merging hull {i}")
                            hulls[i] = poly.convex_hull.union(hull).convex_hull
                print(
                    f"{poly} is disjoint: {disjoint}; is within: {within}; covers: {covers}"
                )

                _ys, _xs = [arr.tolist() for arr in poly.exterior.xy]
                _xs = [a for a, b, c, d, e, f, g, h in grouper(_xs, 8)]
                _ys = [a for a, b, c, d, e, f, g, h in grouper(_ys, 8)]
                xs.extend(_xs)
                xs.extend([_xs[0], None])
                ys.extend(_ys)
                ys.extend([_ys[0], None])

                centroid.append(list(reversed(list(poly.centroid.coords[0]))))
            hulls = [
                Polygon(
                    list(
                        [
                            list(sorted([x, y]))
                            for x, y in list(poly.convex_hull.exterior.coords)
                        ]
                    )
                )
                for poly in hulls
            ]
    except Exception as e:
        print(e.__class__.__name__, e)
    return hulls, centroid


from pyproj import Geod


@cache_queries
def search_for_place(
    s,
    locationbias=None,
    radius=None,
    bounded=False,
    countrycodes="us",
    extratags=True,
    normalizecity=True,
    statecode=True,
    polygon_geojson=False,
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
        "q": s,
        "bounded": int(bounded),
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
    url = f"https://us1.locationiq.com/v1/search.php"
    res = requests.get(url, params=params)
    if not res.ok:
        print(json.dumps(params, indent=4))
        print(f"Response: {res.status_code}")
        pass
    assert (
        res.ok
    ), f"{res.content}\n\n=================================\nInvalid response from URL {url}. Got response code: {res.status_code}"
    return res.json()


def deg2dec(coord):
    patt = re.compile(
        r"(?P<deg>\d{2,3})°(?P<m>\d{0,2})[^\d]?(?P<s>\d{0,2}\.?\d{0,8})[^\w]?(?P<dir>[NWSE])"
    )
    lat, lon = [re.search(patt, c).groupdict() for c in coord]

    if lat["s"]:
        lat["m"] = int(lat["m"]) + (float(lat["s"]) / 60)
    if lon["s"]:
        lon["m"] = int(lon["m"]) + (float(lon["s"]) / 60)
    if lat["m"]:
        lat_m = int(lat["m"]) / 60
    else:
        lat_m = 0
    if lon["m"]:
        lon_m = int(lon["m"]) / 60
    else:
        lon_m = 0
    print(lon["dir"])
    lat = (int(lat["deg"]) + lat_m) * [1, -1][int(lat["dir"] in "WS")]
    lon = (int(lon["deg"]) + lon_m) * [1, -1][int(lon["dir"] in "WS")]
    return lat, lon


def resolve_entity(raw_string, bias=None):
    args = [raw_string] if not bias else [raw_string, bias]
    result = search(args)
    candidates = []
    if bias and bias in result and result[bias]:
        for v in result[bias]:
            c1 = v.coords
            break
        assert c1, f"Location not found: {bias}"

    if raw_string in result and result[raw_string]:
        for v in result[raw_string]:
            dist = (
                math.inf
                if (not v.coords or not bias)
                else geodesic(c1, v.coords).kilometers
            )
            edit_dist = damerau_levenshtein(raw_string.lower(), v.result.lower())
            ratio = fuzz.token_sort_ratio(raw_string, v.result)
            corefs = set()
            _ks = re.findall(r"([\w\s\-]{3,})", raw_string)
            for _k in _ks:
                for cat in v.categories:
                    if re.search(_k, cat, re.IGNORECASE):
                        corefs.add(cat)
            corefs = list(corefs)

            if ratio > 68 or edit_dist <= 10:
                try:
                    if dist is math.inf:
                        d = pow(2, 64)
                    else:
                        d = dist
                    score = log(1 + d, 1 + len(v.categories)) * -1
                except Exception as e:
                    print(e.__class__.__name__, e, dist, len(v.categories))
                    score = math.inf
            else:
                score = ratio
                score *= len(corefs) / 2 * -1
            if edit_dist <= 4:
                score = -math.inf

            index = (dist, edit_dist, ratio)
            candidates.append(
                Munch(
                    {
                        "query": raw_string,
                        "bias": bias,
                        "result": v.result,
                        "rank": v.rank,
                        "geodist": dist,
                        "editdist": edit_dist,
                        "corefs": list(corefs),
                        "ratio": ratio,
                        "coords": v.coords,
                        "index": index,
                        "score": score,
                        "lat": v.lat,
                        "lon": v.lon,
                        "categories": v.categories,
                        "has_coords": bool(v.coords),
                    }
                )
            )

    _sorted = sorted(candidates, key=lambda result: result.index)
    return _sorted


def tx_geojson(responses):
    from itertools import takewhile
    from shapely.geometry import Point, GeometryCollection, Polygon, box as Box

    transformed = []
    xs = []
    ys = []
    labels = []
    arr3d = []
    polygons = []
    colors = ["royalblue", "crimson", "lightseagreen", "orange"]
    my_traces = []
    buffer = 0.05
    for i, obj in enumerate(responses):
        color = colors[i % len(colors)]
        label = f"Name: {obj['display_name']}<br>Centroid: {obj['lat']},{obj['lon']}"
        points = []
        geojson = obj["geojson"]
        points = []
        print(json.dumps(obj))
        if geojson["type"] == "Polygon":
            for shape in geojson["coordinates"]:
                print(f"shape: {shape}")
                for point in shape:
                    print(f"Point: {point}")
                    points.append(point)
            poly = Polygon([point for point in points])
            buffered = poly.buffer(buffer)
        elif geojson["type"] == "Point":
            x1, x2, y1, y2 = [float(x) for x in obj["boundingbox"]]
            xy = sorted([x1, x2]) + sorted([y1, y2])
            minx, maxx, miny, maxy = xy

            box = Box(minx, miny, maxx, maxy)
            buffered = box.buffer(buffer)
        else:
            print(geojson)
            print(f"Weird type: {geojson['type']}")
            continue
        hull = list(buffered.convex_hull.exterior.coords)
        simplex = []

        for g in grouper(4, hull, fillvalue=hull[-1]):
            print(f"Next group is: {g}")
            print(f"Appending {g[0]}")
            simplex.append(list(reversed(g[0])))
        poly = Polygon(simplex)
        polygons.append(poly)
        coords = list(poly.exterior.coords)
        center = list(poly.centroid.coords)
        print(f"Coords are: {coords}")
        arr2d = []
        _xs = []
        _ys = []
        _lab = []
        for x, y in coords:
            x, y = list(sorted([x, y]))
            print(f"x:{x}, y:{y}")
            arr2d.append([x, y])
            _xs.append(x)
            _ys.append(y)
            _lab.append(label)
        arr3d.append(arr2d)

        labels.extend(_lab + [None])
        xs.extend(_xs + [None])
        ys.extend(_ys + [None])

        _tx = copy(obj)
        _tx["geojson"] = {"type": "Polygon", "coordinates": [arr2d]}

        trace = {
            "fill": "toself",
            "hoverinfo": "all",
            "mode": "lines+text",
            "text": _lab,
            "lon": _xs,
            "lat": _ys,
            "selected": {"marker": {"opacity": 1 / (len(responses) + 1) * i}},
            "name": obj["display_name"],
            "marker": {"size": 10, "color": color},
        }
        print(json.dumps(trace))
        transformed.append(_tx)
        my_traces.append(trace)
        print(my_traces)

    arr2d = list(flatten_list(arr3d))

    polygons = GeometryCollection(polygons)
    # area = {}
    # for trace, poly in zip(my_traces, polygons):
    #     area[trace['name']] = poly.area
    # my_traces = list(sorted(my_traces, key=lambda trace: area[trace['name']]))
    centroids = sorted(list(polygons.centroid.coords))
    centroid = (
        sum([x for x, y in centroids]) / (1 + len(centroids)),
        sum([y for x, y in centroids]) / (1 + len(centroids)),
    )
    for trace in my_traces:
        print(json.dumps(trace, indent=4))

    out = {
        "traces": my_traces,
        "centroids": centroid,
        "centroid": centroid,
        "lats": xs,
        "longs": ys,
        "geom": polygons,
        "labels": labels,
        "points": arr2d,
        "polygons": arr3d,
        "data": transformed,
    }
    return Munch(out)


def plot_traces(traces, center=(-98, 35), fig=None, layout=None):
    if not fig:
        fig = go.Figure(go.Scattermapbox())
    if not layout:
        x, y = center
        fig.update_layout(
            hoverdistance=50,
            hovermode="x unified",
            mapbox={
                "style": "carto-positron",
                "center": {"lat": y, "lon": x},
                "zoom": 2,
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            showlegend=True,
        )

    for trace in traces:
        fig.add_trace(go.Scattermapbox(**trace))
    fig.show()
    return fig


if __name__ == "__main__":
    # counts = {"the Broward Sheriff's Office": 4, "Lauderdale Lakes": 1, "Ford Fusion": 2, "Broward Sheriff's": 5, "Office Robbery Detective Marcos Ruiz-Toledo": 1, "Broward Crime Stoppers": 1, "browardcrimestoppers.org": 1}
    # counts = {"Baltimore": 1, "Wuhan": 1, "America": 1, "U.S.": 1, "Medicaid": 1, "SPARC Center": 1, "the Johns Hopkins Bloomberg School of Public Health": 1}
    from src.coronaviruswire.common import db
    import time
    from collections import Counter

    crawldb = db["moderationtable"]
    rows = random.sample(
        [
            row
            for row in crawldb.find(has_ner=True)
            if "houston" not in row["article_url"]
            and not "dallas" in row["article_url"]
            and len(list(row["ner"].keys())) > 3
        ],
        10,
    )
    for row in rows:
        if "houston" in row["article_url"] or "dallas" in row["article_url"]:
            continue
        counts = Counter(row["ner"])
        loc = row["city"]
        print(row["title"], row["article_url"])
        print(row["content"])
        ents = list(row["ner"].keys())
        try:
            city, state = loc.split(", ")
            #
            # counts[state] += 1
        except:
            city = loc
        counts[loc] += 1
        ents = list(row["ner"].keys())
        if len(ents) <= 6:
            continue
        ref = resolve_entity(loc)[0]
        ref_point = ref["coords"]

        responses = {}
        geo_ents = {}
        for ent in ents:
            resolved = resolve_entity(ent, loc)
            if resolved:
                res, *alts = resolved[0], resolved[1:]
                if res.geodist > 500:
                    continue
                print(f"Resolved {ent} to {res.result}")
                _ent = resolved[0].result
                counts[_ent] = counts[ent]
                try:
                    responses[ent] = search_for_place(_ent, loc)[0]
                    # res = None
                    # for response in _responses:
                    #     if response['matchquality']['matchtype'] == 'exact':
                    #         res = response
                    #         print(f"Exact match for {ent}: {response['display_name']}")
                    #         responses[ent] = [response]
                    #         break
                    # if not res:
                    #     print(f"Skipping {ent}")
                    #     continue
                    print(
                        f"Resolved {ent} to {res.result} => {responses[ent][0]['display_name']} (dist: {geodesic((responses[ent][0]['lat'], responses[ent][0]['lon']), ref_point)}"
                    )
                    geo_ents[ent] = tx_geojson(responses[ent])
                except Exception as e:
                    print(e.__class__.__name__, e)
                    print("break")
        # for ent, c in counts.items():
        #     for i in range(c):
        #         ents.append(ent)
        feeling_lucky = []
        geo = []
        for ent, response in geo_ents.items():
            if not response.traces:
                continue

            # # traces = sorted(response.traces, key=lambda trace: (fuzz.partial_ratio(trace['name'], ent), (trace['name'].split(",")[0], ent.split(",")[0])))
            # for i, trace in enumerate(traces):
            #     print(f"{i}. {trace}")
            feeling_lucky.append(response.traces[0])
            for count in range(counts[ent]):
                geo.append(response.geom[0])

        intersection = intersect_circles(geo, 0.23)
        coords, center = intersection

        lats = []
        lons = []
        for p in coords:

            if isinstance(p, Polygon):
                # lats.append(None)
                # lons.append(None)
                lats = []
                lons = []
                for _p in list(p.convex_hull.exterior.coords):
                    x, y = list(sorted(_p))
                    lats.append(x)
                    lons.append(y)
                    print(lats, lons)
                feeling_lucky.append(
                    {
                        "lat": lons,
                        "lon": lats,
                        "name": "intersection",
                        "fill": "toself",
                        "text": "intersection",
                        "mode": "lines+text",
                    }
                )

            elif isinstance(p, (list, tuple)):
                x, y = list(sorted(p))
                lats.append(x)
                lons.append(y)
        if lats:
            feeling_lucky.append(
                {
                    "lat": lons,
                    "lon": lats,
                    "name": f"intersection{random.randrange(0,100)}",
                    "fill": "toself",
                    "text": "intersection",
                    "mode": "lines+text",
                }
            )

        output = plot_traces(feeling_lucky)
