from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from collections import defaultdict
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
pio.renderers.default = 'browser'
import plotly
from geopy.distance import geodesic
fig = None
coords = {}
geo_ents = {}
from collections import deque
plotly.io.orca.config.executable = "/home/kz/.nvm/versions/node/v13.1.0/bin/orca"
plotly.io.orca.config.mapbox_access_token = 'sk.eyJ1IjoibmVvbmNvbnRyYWlscyIsImEiOiJjazhzazh5M3EwNzlnM21xZm9kam80OGhrIn0.59fAYtfIHZzI3lEtCfUWjA'
plotly.io.orca.config.save()


def diag2poly(p1, p2):
    points = [p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]
    return points


async def search_for_place_async(place_name, location=None, radius=300):
    if location is None:
        center = None
        radius = ""
    elif location in coords:
        center = coords[location][0]

    else:
        response = await search_for_place_async(location)
        center = response.center
    if center:
        lat, long = center
        radius = 1000
        radius = f"&locationbias=circle:{radius}@{lat},{long}"

    query = quote_plus(place_name)
    s = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}{radius}&fields=formatted_address,name,opening_hours,geometry&inputtype=textquery&key=AIzaSyALE94yjbDhNRZbigm6xnaDnnSIe4Vlw00"
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
    similarity = fuzz.partial_ratio(place_name, candidate['name'])

    if center:
        dist = geodesic(c1, center)
        bias = {"lat": lat, "long": long, "radius": radius}

    else:
        dist = 0
        bias = {"lat": None, "long": None, "radius": None}

    return Munch({
        "ok": True,
        "center": c1,
        "lat": c1[0],
        "long": c1[1],
        "diag": diag1,
        "box": box1,
        "dist": 0,
        "name": candidate['name'],
        "address": candidate['formatted_address'],
        "similarity": similarity,
        "bias": bias
    })


def extract_entities(s):
    days = r"((Mon(d|\s)|Tue|Wed(n|\b)|Thur|Fri|Sat|Sun(\b|d))[\w\.\,]*\s*\d*\s*)|((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\.]*\s*\d*\s)"
    cleaned = [
        re.sub(days, "", tok[0]) for tok in re.findall(
            r"((D\.?C\.?)|(^\s*[A-Z\s]{5,})|([A-Z]([a-zA-Z]+|\.|\'|\-\,)+)+(\s[A-Z][a-zA-Z]+)+)|([A-Z]{1,})|([a-zA-Z][A-Z])[a-zA-Z]*[A-Z][a-z]*|^\s*([A-Z]{5,})\s*\-",
            unidecode(s)) if tok[0] and len(tok[0]) > 4 and not "\n" in tok
    ]
    single_words = [
        w for w in re.findall(r"[A-Z][a-z]{5,}", s) if s.count(w) > 1
    ]

    compounds = [
        tok.strip() for tok in cleaned
        if '\n' not in tok and (len(tok) > 5 or tok in ("D.C.", "DC"))
    ]

    ents = single_words + compounds
    for i, ent in enumerate(ents):
        print(f"Entity #{i} :: {ent}")
    return ents


async def locate_all(entities, origin):
    async def locate_entity(entity, origin):
        ent = await search_for_place_async(entity, origin)

        if ent.similarity > 0.8 and re.search(r'(US|United States|U\.S\.)',
                                              ent.address):
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
    points = []
    labels = []
    queries = []
    dists = []
    try:
        origin = coords[origin][0]
    except KeyError:
        await search_for_place_async(origin)
        if origin not in coords:
            return {}
        origin = coords[origin][0]
    for k, v in geo_ents.items():
        if k not in counts:
            continue
        for i in range(counts[k]):
            if not v.ok:
                continue
            points.append(v.center)
            points.extend([list(coord) for coord in v.box])
            labels.append(v.name)
            queries.append(k)
            dists.append(geodesic(tuple(v.center), origin))
    return Munch({
        "points": points,
        "labels": labels,
        "queries": queries,
        "dists": dists
    })


def cluster(points):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(points)
    y_kmeans = kmeans.predict(points)
    clusters = defaultdict(list)
    for klazz, p in zip(y_kmeans, points):
        clusters[klazz].append(p)
    return [x[1] for x in sorted(clusters.items(), key=lambda item: item[0])]


def get_convex_hull(clusters):
    polygons = []
    for points in clusters:
        print(points)
        try:
            polygon = []
            for point in points:
                polygon.append(point)
            hull = list(Polygon(polygon).convex_hull.boundary.coords)
            polygons.append(hull)
        except Exception as e:
            print(e)
    return polygons


def plot_clusters(latitudes, longitudes, origin):
    lat, long = origin
    print(f"Latitudes: {latitudes}")
    print(f"Longitudes: {longitudes}")
    global fig
    fig = go.Figure(
        go.Scattermapbox(mode="lines",
                         fill="toself",
                         lon=latitudes,
                         lat=longitudes))

    fig.update_layout(mapbox={
        'style': "stamen-terrain",
        'center': {
            'lon': lat,
            'lat': long
        },
        'zoom': 4
    },
                      showlegend=False,
                      margin={
                          'l': 0,
                          'r': 0,
                          'b': 0,
                          't': 0
                      })


def vectorize_coords(geometries):
    lat, long = [], []

    for shape in geometries:
        for p1, p2 in shape:
            long.append(p1)
            lat.append(p2)
        lat.append(None)
        long.append(None)
    return lat, long


async def process_row(row):
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
    if not map_data:
        return
    clusters = cluster(map_data.points)
    polygons = get_convex_hull(clusters)
    start_coords = list(reversed(coords[origin][0]))
    latitude_vec, longitude_vec = vectorize_coords(polygons)
    plot_clusters(latitude_vec, longitude_vec, start_coords)


if __name__ == '__main__':
    from src.coronaviruswire.common import db
    crawldb = db['crawldb']
    rows = random.sample([row for row in crawldb.find()], 20)
    for row in rows:

        trio.run(process_row, row)
        print(row['headline'])
        print(row['description'])
        print(row['articlebody'])
        fig.show()
        fig.write_image(f"article_{row['id']}".replace(" ", " ") + ".png",
                        width=2000,
                        height=1230)
