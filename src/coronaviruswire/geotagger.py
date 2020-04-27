from shapely.geometry import Point
from shapely.ops import cascaded_union
from itertools import combinations
from collections import deque
from more_itertools import grouper
from shapely.geometry import Polygon
from gemeinsprache.utils import red
from src.coronaviruswire.utils import extract_entities
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


plotly.io.orca.config.executable = "/home/kz/.nvm/versions/node/v13.1.0/bin/orca"
plotly.io.orca.config.mapbox_access_token = "sk.eyJ1IjoibmVvbmNvbnRyYWlscyIsImEiOiJjazhzazh5M3EwNzlnM21xZm9kam80OGhrIn0.59fAYtfIHZzI3lEtCfUWjA"
plotly.io.orca.config.save()
local = None


def similarity(a, b):
    ratio = fuzz.ratio(a, b)
    dist = damerau_levenshtein(a, b)
    diff_len = abs(len(a) - len(b))
    len_penalty = log(len(a) / (1 + diff_len))
    # penalty = 0.5 + 1/log(1 + dist)

    score = ratio
    return Munch(locals())


<<<<<<< Updated upstream
def intersect_circles(centers, radii):
    try:
        circles = [Point(*center).buffer(math.sqrt(radius/2/110)) for center, radius in zip(centers, radii)]
        listpoly = [a.intersection(b) for a, b in combinations(circles, 2)]  # list of intersections
        rings = [sg.LineString(list(pol.exterior.coords)) for pol in listpoly]  # list of rings
=======
def intersect_shapes(shapes, buffer_radius):
    try:
        shapes = [shape.buffer(buffer_radius) for shape in shapes]
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


def intersect_circles(centers, radii):
    try:
        circles = [Point(*center).buffer(1) for center, radius in zip(centers, radii)]
        listpoly = [
            a.intersection(b) for a, b in combinations(circles, 2)
        ]  # list of intersections
        rings = [
            sg.LineString(list(pol.exterior.coords)) for pol in listpoly
        ]  # list of rings
>>>>>>> Stashed changes

        union = unary_union(rings)

        result = [
            geom for geom in polygonize(union)
        ]  # list all intersection geometries

        multi = cascaded_union(
            result
        )  # Create a single geometry out of all intersections
        # fin = [c.difference(multi) for c in circles]  # Cut multi from circles and leave only outside geometries.
        #
        # result = result + fin  # add the outside geometries to the intersections geometries

        # # Plot settings:
        # plt.figure(figsize=(5, 5))
        # ax = plt.gca()
        #
        # name = 1
        #
        # for e in result:
        #     ax.add_patch(descartes.PolygonPatch(e,
        #                                         fc=np.random.rand(3),
        #                                         ec=None,
        #                                         alpha=0.5))
        #
        #     ax.text(e.centroid.x, e.centroid.y,
        #             '%s' % name, fontsize=9,
        #             bbox=dict(facecolor='orange', alpha=0.5),
        #             color='blue',
        #             horizontalalignment='center')
        #     name += 1

        # plt.xlim(-1.5, 2.5)
        # plt.ylim(-1.5, 2.5)
        # plt.show()

        #
        # intersection = cascaded_union(
        #     [a.intersection(b) for a, b in combinations(circles, 2)]
        # )
        #
        try:
            hull = list(multi.exterior.coords)
            ys, xs = [arr.tolist() for arr in multi.exterior.xy]
<<<<<<< Updated upstream
            xs = [a for a,b,c,d,e,f,g,h in grouper(xs, 8)]
            ys = [a for a,b,c,d,e,f,g,h in grouper(ys, 8)]
            centroid = [list(reversed(list(multi.centroid.coords)[0]))]
            xs.append(xs[0])
            ys.append(ys[0])
        except Exception as e:
            # out = input(f"{e.__class__.__name__}: {e}")
            # breakpoint()
=======
            xs = [a for a, b, c, d, e, f, g, h in grouper(xs, 8)]
            ys = [a for a, b, c, d, e, f, g, h in grouper(ys, 8)]
            centroid = [reversed(list(multi.centroid.coords)[0])]
        except:
>>>>>>> Stashed changes
            hull = []
            xs, ys = [], []
            centroid = []

            for poly in multi:
<<<<<<< Updated upstream
                coords = list(poly.exterior.coords)
                hull.extend(coords)
                hull.extend([coords[0], None])
                _ys, _xs = [arr.tolist() for arr in poly.exterior.xy]
                _xs = [a for a,b,c,d,e,f,g,h in grouper(_xs, 8)]
                _ys = [a for a,b,c,d,e,f,g,h in grouper(_ys, 8)]
=======
                hull.extend(poly.exterior.coords)
                hull.extend([poly.exterior.coords[0], None])
                _ys, _xs = [arr.tolist for arr in poly.exterior.xy]
                _xs = [a for a, b, c, d, e, f, g, h in grouper(_xs, 8)]
                _ys = [a for a, b, c, d, e, f, g, h in grouper(_ys, 8)]
>>>>>>> Stashed changes
                xs.extend(_xs)
                xs.extend([_xs[0], None])
                ys.extend(_ys)
                ys.extend([_ys[0], None])

                centroid.append(list(reversed(list(poly.centroid.coords[0]))))
    except Exception as e:
        print(f"Centers: {centers}\nRadii: {radii}")
        print(e)
        return None

    return Munch({"hulls": hull, "xs": xs, "ys": ys, "centroids": centroid})


<<<<<<< Updated upstream
def diag2poly(p1, p2):
    points = [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]
    return points


=======
>>>>>>> Stashed changes
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
        # print(res.json())
    try:
        candidate = res.json()["candidates"][0]
    except Exception as e:
        # print(e)
        # print(res.json())
        return Munch({"ok": False, "center": None, "similarity": 0.0})
    c1 = list(candidate["geometry"]["location"].values())
    diag1 = [(v["lat"], v["lng"]) for k, v in candidate["geometry"]["viewport"].items()]
    box1 = diag2poly(*diag1)
    coords[place_name] = (c1, diag1, box1)
    sim = similarity(place_name, candidate["name"])
    if center:
        dist = geodesic(c1, center).kilometers
        bias = {"lat": lat, "long": long, "radius": radius}

    else:

        dist = 0
        bias = {"lat": None, "long": None, "radius": None}
    import math

    distance_to_here = geodesic(c1, coords["North Park, San Diego, California, USA"][0])
    if dist and distance_to_here.kilometers < 500:
        penalty = 0.1  #math.log(dist.kilometers, distance_to_here.kilometers)
    else:
        penalty = 1
    final_score = sim.score

<<<<<<< Updated upstream
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
        "score": final_score * penalty,
        "bias": bias
    })
    # print(json.dumps(obj, indent=4))
=======
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
            "name": candidate["name"],
            "address": candidate["formatted_address"],
            "similarity": sim,
            "edit_distance": sim.dist,
            "score": final_score * penalty,
            "bias": bias,
        }
    )
    print(json.dumps(obj, indent=4))
>>>>>>> Stashed changes
    return obj


#
#
# def extract_entities(s):
#     days = r"((Mon(d|\s)|Tue|Wed(n|\b)|Thur|Fri|Sat|Sun(\b|d))[\w\.\,]*\s*\d*\s*)|((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\.]*\s*\d*\s)"
#     cleaned = [
#         re.sub(days, "", tok[0]) for tok in re.findall(
#             r"((D\.?C\.?)|(^\s*[A-Z\s]{5,})|([A-Z]([a-zA-Z]+|\.|\'|\-\,)+)+(\s[A-Z][a-zA-Z]+)+)|([A-Z]{1,})|([a-zA-Z][A-Z])[a-zA-Z]*[A-Z][a-z]*|^\s*([A-Z]{5,})\s*\-",
#             unidecode(s)) if tok[0] and len(tok[0]) > 4 and not "\n" in tok
#     ]
#     single_words = [
#         w for w in re.findall(r"[A-Z][a-z]{5,}", s) if s.count(w) >= 2
#     ]
#
#     compounds = [
#         tok.strip() for tok in cleaned
#         if '\n' not in tok and (len(tok) > 5 or tok in ("D.C.", "DC"))
#     ]
#
#     ents = single_words + compounds
#     for i, ent in enumerate(ents):
#         print(f"Entity #{i} :: {ent}")
#     return ents


async def locate_all(entities, origin):
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

    return geo_ents


async def prepare_geo_points(geo_ents, origin, counts):
    scale = 200
    traces = []
    points = []
    labels = []
    queries = []
    weights = []
    dists = []
    sizes = []

    try:
        origin = coords[origin][0]
    except KeyError:
        await search_for_place_async(origin)
        if origin not in coords:
            return {}
        origin = coords[origin][0]
    colors = ["royalblue", "crimson", "lightseagreen", "orange"]

    for k, v in geo_ents.items():
        if k not in counts:
            continue
        if not v.ok:
            continue
        # for i in range(counts[k]):
        #     if not v.ok:
        #         continue

        else:
            count = counts[k]
            color = colors[min(count, len(colors) - 1)]
            hyp = geodesic(*v.diag)
            radius = hyp.kilometers
            area = math.sqrt(float(math.pi * pow(hyp.kilometers, 2) / scale)) + 50
            sizes.append(radius)
            for i in range(counts[k]):
                points.append(v.center)
                sizes.append(radius)
            weights.append(hyp.km)
            labels.append(v.name)
            queries.append(k)
            dists.append(geodesic(tuple(v.center), origin))
            bubble_trace = go.Scattergeo(
                {
                    "locationmode": "USA-states",
                    "lon": [v.long],
                    "lat": [v.lat],
                    "name": v.name,
                    "type": "scattergeo",
                    "text": [
                        f"String: {v.query}<br>Entity: {v.name}<br>Address: {v.address}<br>Occurrences: {count}"
                    ],
                    "marker": {
                        "size": area,
                        "color": color,
                        "line_color": "rgb(40,40,40)",
                        "line_width": 0.5,
                        "sizemode": "area",
                    },
                }
            )
            traces.append(bubble_trace)
<<<<<<< Updated upstream
            # print(bubble_trace)
    bubble_traces = sorted(traces, key=lambda obj: obj['marker']['size'])
=======
            print(bubble_trace)
    bubble_traces = sorted(traces, key=lambda obj: obj["marker"]["size"])
>>>>>>> Stashed changes
    intersection = intersect_circles(points, sizes)

    for trace in traces:
        trace["marker"]["opacity"] = 0.5
    return Munch(
        {
            "points": points,
            "labels": labels,
            "queries": queries,
            "dists": dists,
            "weights": weights,
            "traces": bubble_traces,
            "intersection": intersection,
        }
    )


def cluster(points):
    global clusters
    clarans_instance = clarans(points, 4, 8, 12)
    (ticks, result) = timedcall(clarans_instance.process)
    print("Execution time : ", ticks, "\n")
    indices = clarans_instance.get_clusters()
    medoids = clarans_instance.get_medoids()
    print("Index of the points that are in a cluster : ", indices)
    print("The index of medoids that algorithm found to be best : ", medoids)
    clusters = []
    for clique in indices:
        cluster = []
        for i in clique:
            cluster.append(points[i])
        clusters.append(cluster)
    print(f"Clusters: {clusters}")

    return clusters


def get_convex_hull(clusters):
    polygons = []
    for points in clusters:
        # print(points)
        try:
            polygon = []
            for point in points:
                polygon.append(point)
            hull = list(Polygon(polygon).convex_hull.boundary.coords)
            polygons.append(hull)
        except Exception as e:
            print(e)
    return polygons


def plot_bubblemap(traces, row):
    global fig
    fig = None
    fig = go.Figure()
    for trace in traces:
        # print(trace)
        fig.add_trace(trace)
    fig.update_layout(
        title_text=f"Article #{row['id']}<br>{row['headline']}<br>({row['site']})",
        showlegend=True,
        geo=dict(scope="usa", landcolor="rgb(217, 217, 217)"),
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
    )

    # print(fig)
    # fig.show()
    return fig


def plot_intersection(map_data):
    global fig2
    if not map_data.intersection:
        return
    latitudes = map_data.intersection.xs
    longitudes = map_data.intersection.ys
    try:
        center_lat, center_long = map_data.intersection.centroids[0]
    except:
        print(red(f"\n\n\n\nNo center coords!\n\n"))
        print(json.dumps(map_data, indent=4, default=str))
    fig2 = go.Figure(
        go.Scattermapbox(mode="lines", fill="toself", lon=latitudes, lat=longitudes)
    )
    fig2.update_layout(
        mapbox={
            "style": "stamen-terrain",
            "center": {"lon": center_lat, "lat": center_long},
            "zoom": 4,
        },
        showlegend=False,
<<<<<<< Updated upstream
        margin={
            'l': 0,
            'r': 0,
            'b': 0,
            't': 0
        })
    # fig2.show()
=======
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
    )
    fig2.show()
>>>>>>> Stashed changes


def plot_clusters(latitudes, longitudes, origin):
    lat, long = origin
    print(f"Latitudes: {latitudes}")
    print(f"Longitudes: {longitudes}")
    global fig

    fig = go.Figure(
        go.Scattermapbox(mode="lines", fill="toself", lon=latitudes, lat=longitudes)
    )

    fig.update_layout(
        mapbox={
            "style": "stamen-terrain",
            "center": {"lon": lat, "lat": long},
            "zoom": 4,
        },
        showlegend=False,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
    )


def vectorize_coords(geometries):
    lat, long = [], []

    for shape in geometries:
        for p1, p2 in shape:
            long.append(p1)
            lat.append(p2)
        lat.append(None)
        long.append(None)
    return lat, long


async def process_row(rows):
    global geo_ents
    global map_data
    global clusters
    import os
    global fig
    global fig2
    global coords
    global geo_ents
    global map_data

    global polygons
<<<<<<< Updated upstream
=======
    fig = None
    local = await search_for_place_async("North Park, San Diego, California, USA")

    origin = f"{row['loc']}, USA"
    text = "\n\n".join(
        [
            str(x)
            for x in [
                row["headline"],
                row["description"],
                row["articlebody"],
                row["keywords"],
            ]
        ]
    )
    ents = extract_entities(text)
    counts = {ent: text.count(ent) for ent in ents}
    geo_ents = await locate_all(ents, origin)
    map_data = await prepare_geo_points(geo_ents, origin, counts)
    if not map_data or len(map_data.points) < 5:
        return
    # clusters = cluster(map_data.points)
    # polygons = get_convex_hull(clusters)
    # start_coords = list(reversed(coords[origin][0]))
    # latitude_vec, longitude_vec = vectorize_coords(polygons)
    # plot_clusters(latitude_vec, longitude_vec, start_coords)
    plot_bubblemap(map_data.traces, row)

    plot_intersection(map_data)


if __name__ == "__main__":
    from src.coronaviruswire.common import db

    crawldb = db["crawldb"]
    rows = random.sample(
        [row for row in crawldb.find() if "dallas" not in row["site"]], 1000
    )
>>>>>>> Stashed changes
    for row in rows:
        fig2 = None
        fig = None
        coords = {}
        geo_ents = {}
<<<<<<< Updated upstream
        map_data = Munch()
        fig = None
        local = await search_for_place_async(
            "North Park, San Diego, California, USA")

        origin = f"{row['loc']}, USA"
        text = '\n\n'.join([
            str(x) for x in [
                row['headline'], row['description'], row['articlebody'],
                row['keywords']
            ]
        ])
        ents = extract_entities(text)
        counts = {ent: text.count(ent) for ent in ents}
        geo_ents = await locate_all(ents, origin)
        map_data = await prepare_geo_points(geo_ents, origin, counts)
        if not map_data or len(map_data.points) < 5:
            return
        # clusters = cluster(map_data.points)
        # polygons = get_convex_hull(clusters)
        # start_coords = list(reversed(coords[origin][0]))
        # latitude_vec, longitude_vec = vectorize_coords(polygons)
        # plot_clusters(latitude_vec, longitude_vec, start_coords)
        plot_bubblemap(map_data.traces, row)

        plot_intersection(map_data)
        print(row['headline'])
        print(row['description'])
        print(row['articlebody'])
=======

        map_data = {}

        trio.run(process_row, row)
        print(row["headline"])
        print(row["description"])
        print(row["articlebody"])
>>>>>>> Stashed changes

        try:
            row["extracted_features"] = geo_ents
            row["map_data"] = map_data
            print(fig)

            # if fig:
            #     fig.show()
            #
            # if fig2 and map_data and map_data.intersection:
            #     fig2.show()
            # row['clusters'] = clusters
            dirname = f"/home/kz/projects/coronaviruswire/src/coronaviruswire/outputs/article_{row['id']}"
            os.makedirs(dirname)
            with open(
<<<<<<< Updated upstream
                    f"/home/kz/projects/coronaviruswire/src/coronaviruswire/outputs/article_{row['id']}/data.json",
                    "w") as f:
                json.dump(row, f, default=str)
            if fig:
                fig.write_image(
                    os.path.join(dirname, "bubble.png"),
                    width=1000,
                    height=618)
                fig.write_html(
                    os.path.join(dirname, "bubble.html")
                )
            if fig2:
                fig2.write_image(
                    os.path.join(dirname, "intersection.png"),
                    width=1000,
                    height=618
                )
                fig2.write_html(
                    os.path.join(dirname, "intersection.html"))

=======
                f"/home/kz/projects/coronaviruswire/src/coronaviruswire/outputs/article_{row['id']}.json",
                "w",
            ) as f:
                json.dump(row, f, default=str)
            # if fig:
            # fig.write_image(
            #     f"/home/kz/projects/coronaviruswire/src/coronaviruswire/outputs/article_{row['id']}.png",
            #     width=2000,
            #     height=1230)
            # fig.write_html(
            #     f"/home/kz/projects/coronaviruswire/src/coronaviruswire/outputs/article_{row['id']}.html"
            # )
            # fig.show()
>>>>>>> Stashed changes
        except Exception as e:
            print(e)
            print(f"No figure for row:")
            for k, v in row.items():
                print(k, v)
            print("ok")


if __name__ == '__main__':

    from src.coronaviruswire.common import db
    crawldb = db['crawldb']
    rows = random.sample(
        [row for row in crawldb.find() if 'dallas' not in row['site'] and len(row['articlebody']) > 300], 100)

    trio.run(process_row, rows)
