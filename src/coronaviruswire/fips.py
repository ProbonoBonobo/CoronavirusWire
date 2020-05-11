from collections import Counter
from geopy.distance import geodesic
from src.coronaviruswire.utils import (
    search_for_place,
    search,
    calculate_bounding_box,
    normalize_state_name
)
from math import sqrt
import datetime as dt

def load_csv():
    """>>> counties["features"][0]
     {'type': 'Feature',
      'properties': {'GEO_ID': '0500000US01001',
      'STATE': '01',
      'COUNTY': '001',
      'NAME': 'Autauga',
      'LSAD': 'County',
      'CENSUSAREA': 594.436},
     'geometry': {'type': 'Polygon',
      'coordinates': [[[-86.496774, 32.344437],
        [-86.717897, 32.402814],
        [-86.814912, 32.340803],
        [-86.890581, 32.502974],
        [-86.917595, 32.664169],
        [-86.71339, 32.661732],
        [-86.714219, 32.705694],
        [-86.413116, 32.707386],
        [-86.411172, 32.409937],
        [-86.496774, 32.344437]]]},
     'id': '01001'}
 """
    with open("../../lib/us_geoindex.json", "r") as f:
        index = json.loads(f)
    return index


# =============================================  SCORING CRITERIA  ================================================
# Functions for assessing result quality. As a rule, these should have the signature
#     def pred(result: dict): -> bool
# When testing whether a result falls within an acceptable range (e.g., distance < n km), a function may instead
# *return* a function with this signature.


def distance_less_than(n_kilometers: float, reference_point: tuple):
    """Configurable predicate test for checking whether a result
       lies within a certain radius of a given (longitude, latitude)
       coordinate."""
    p1 = reference_point

    def apply_func(result):
        lat = float(result["lat"])
        lon = float(result["lon"])
        p2 = (lat, lon)
        distance = geodesic(p1, p2).kilometers
        return distance < n_kilometers

    return apply_func


def match_type_isinstance(tup):
    if not isinstance(tup, tuple):
        tup = tuple(list(tup))

    def apply_func(result):
        t = result["type"]
        return t in tup

    return apply_func


def match_class_isinstance(tup):
    if not isinstance(tup, tuple):
        tup = tuple(list(tup))

    def apply_func(result):
        c = result["class"]
        return c in tup

    return apply_func


def is_city(result):
    return (
        result["matchquality"]["matchcode"] in ("exact", "fallback")
        and result["matchquality"]["matchlevel"] == "city"
        and "hamlet" not in result["address"]
    )


def is_state(result):
    return (
        result["matchquality"]["matchcode"] in ("exact", "fallback")
        and result["matchquality"]["matchlevel"] == "state"
    )


def is_country(result):
    return (
        result["matchquality"]["matchcode"] in ("exact", "fallback")
        and result["matchquality"]["matchlevel"] == "country"
    )


def is_hamlet(result):
    return "hamlet" in result["address"]


def is_non_us_place(result):
    return (
        not is_country(result)
        and "country_code" in result["address"]
        and result["address"]["country_code"] != "us"
    )


def is_approximate_match_greater_than_km(km, reference_point):
    pred = distance_less_than(km, reference_point)

    def apply_func(result):
        return (
            not pred(result)
            and "matchcode" in result["matchquality"]
            and result["matchquality"]["matchcode"] == "approximate"
        )

    return apply_func


def is_outside_state(state):
    def apply_func(result):
        return (
            not is_state(result)
            and "state" in result["address"]
            and result["address"]["state"] != state
        )

    return apply_func


def is_exact_match(result):
    return (
        "matchcode" in result["matchquality"]
        and result["matchquality"]["matchcode"] == "exact"
    )


def name_contains_query(search):
    search = search.lower()

    def apply_func(result):
        return search in result["display_name"].lower()

    return apply_func


from fuzzywuzzy import fuzz


def score_results(query, reference_location, radius=300, **kwargs):
    import wikipedia

    scored = []
    ref = search_for_place(reference_location)[0]
    original_query = query
    state = reference_location.split(",")[-1]
    alt_queries = [f"{query}, {reference_location}", f"{query}, {state}", f"{query}"]
    all_queries = []

    alt_query = search(alt_queries, reference_location, radius=500)

    all_results = []
    for query, results in alt_query.items():
        for result in results:
            try:
                all_queries.append(result)
            except Exception as e:
                print(e.__class__.__name__, e, result)
    resolved_queries = [query]
    fallbacks = []
    for qs in set(resolved_queries):
        results = search_for_place(qs, reference_location, radius=radius, **kwargs)
        if not results:
            fallbacks.append(wikipedia.WikipediaPage(wikipedia.search(qs)[0]))
        all_results.extend(results)
    ref_lat = ref["lat"]
    ref_lon = ref["lon"]
    state = ref["address"]["state"]
    reference_point = (float(ref_lat), float(ref_lon))
    boost_factors = {
        match_type_isinstance(("suburb", "place", "townhall")): 5.0,
        match_type_isinstance(("way", "highway", "fuel")): 0.5,
        distance_less_than(300, reference_point): 1.5,
        match_class_isinstance(("office", "restaurant")): 0.25,
        is_country: 100.0,
        is_state: 50.0,
        is_city: 25.0,
        is_hamlet: 0.01,
        is_non_us_place: 0.01,
        is_exact_match: 100.0,
        is_approximate_match_greater_than_km(500, reference_point): 0.001,
        is_outside_state(state): 0.01,
        name_contains_query(query): 25,
    }
    for rank, result in enumerate(all_results):
        try:
            score = float(result["importance"]) / (1 + sqrt(rank))

            for pred, boost in boost_factors.items():
                if pred(result):
                    score *= boost
            result["score"] = score
            scored.append(result)
        except Exception as e:
            print(e.__class__.__name__, e)
            breakpoint()
    try:

        scored = sorted(scored, key=lambda x: x["score"])
        top_result = list(sorted(scored, key=lambda s: s["score"]))[-1]
    except Exception as e:
        top_result = None
    # top_result['string'] = original_query
    # top_result['ok'] = top_result['score'] >= 1
    return top_result, fallbacks

    refs = Counter()
    for result in new_results:
        if result["result"] and result["result"]["address"]:
            addr = result["result"]["address"]
            state = addr["state"]
            if not "county" in addr:
                county = None
            else:
                county = re.sub(r" County", "", addr["county"])

            try:
                if state != "Washington" and (
                    result["result"]["matchquality"]["matchcode"] != "exact"
                    or result["result"]["importance"] <= 0.25
                    or result["result"]["class"] == "office"
                ):
                    print(f"Ignoring this result")
                    continue
                elif state and county:
                    fips_code = county_fips[state][county]

                    refs[fips_code] += 1
                elif not "county" in addr and state == "Washington":
                    print(f"Got a state:{addr}")
                    for county, fips_code in county_fips[state].items():
                        refs[fips_code] += 0.33

                print(
                    result["string"],
                    result["result"]["display_name"],
                    county,
                    state,
                    fips,
                )

            except Exception as e:
                print(e)


from sklearn.cluster import KMeans


def kmeans(arr, n=2):
    kmeans = KMeans(n_clusters=n, max_iter=12).fit(arr)
    labels = kmeans.labels_
    return labels

def mark_as_traversed(crawldb, article_id):
    print(f"Marking unsavable article as fips processed: {article_id}")
    new_row = dict(
        article_id=article_id,
        fips_processed = True,
        country='us',
        updated_at = dt.datetime.utcnow().replace(microsecond=0),
        updated_by = 'fips',
        lang = 'en'
    )
    crawldb.update(new_row, ['article_id'])

if __name__ == "__main__":
    import time
    import pandas as pd
    from src.coronaviruswire.common import (db, database_name)
    from src.coronaviruswire.pointAdaptor import (Point, adapt_point, adapt_point_array)
    from psycopg2.extensions import adapt, register_adapter, AsIs
    import random
    import numpy as np

    from collections import defaultdict
    from src.coronaviruswire.utils import initialize_kmedoids_model, deg2dec
    from urllib.request import urlopen
    import plotly.graph_objects as go
    import json
    import requests


    # CONSTANTS
    LIMIT_ARTICLES = 50
    LIMIT_NERS = 30 # Truncate to 30 if more than 30
    UPPER_LIMIT_NERS = 100 # if greater than 90, don't even truncate, marks as unprocessable

    # Initialization
    register_adapter(Point, adapt_point)


    # this loads the county polygons for the plotly diagrams below
    countiesJSON = requests.get("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
    counties = countiesJSON.json()

    crawldb = db[database_name]

    # this loads the treemap of US States => US Counties => FIPS Codes
    with open("./lib/us_geoindex.json", "r") as f:
        fips_index = json.load(f)

    fips_index = fips_index["fips"]

    # a quirk of the SpaCY NER model is that the occasional foreign language article confuses the model to such an
    # extent that pretty much every word gets classified as an entity. Because the location tagging procedure is
    # time consuming (and potentially expensive, literal $$$ depending on the API), we will want to filter rows
    # containing an unusually large number of entities as a basic sanity check
    rows = [
        row for row in crawldb.find(has_ner=True, fips_processed=False, mod_status='approved')
    ]

    print(f"Found {len(rows)} unprocessed and approved articles!")

    if len(rows) == 0:
        # row for row in crawldb.find(has_ner=True, fips_processed=False, _limit=LIMIT_ARTICLES) if len(list(row["ner"].keys())) <= 30
        rows = [
            row for row in crawldb.find(has_ner=True, fips_processed=False)
        ]

        print(f"Approved articles already approved. Founding {len(rows)} unprocessed articles!")

    rows_length = min(len(rows), LIMIT_ARTICLES)
    rows = random.sample(rows, rows_length)
    rows = sorted(rows, key=lambda row: row['published_at'], reverse=True)

    # try_article = rows[0]
    # print(try_article)
    # print(f"AAA has_ner: {try_article['has_ner']}")
    # print(f"AAA has_ner type: {type(try_article['has_ner'])}")
    # print(f"AAA fips_processed: {try_article['fips_processed']}")
    # print(f"AAA mod_status: {try_article['mod_status']}")

    results = {}

    # the kmedoids model is trained from thousands of US regional features I scraped from wikipedia. I'm using it
    # to partition the US into 64 clusters, then label geospatial coordinates with the most likely cluster ID. Toward
    # the end, I collect the points and labels and throw away any singleton coordinates. (Intuition: meaningful
    # references to a place usually make multiple references to it. If we only extracted a single reference to a place,
    # it's probably either a passing reference (e.g., "...said UC Berkeley Professor, Christos Papadimitriou...") or
    # we resolved the string to an incorrect entity and/or assigned it the wrong coordinate location.
    model = initialize_kmedoids_model()
    import wikipedia

    for row in rows:

        article_id = row["article_id"]

        locations = []

        geo_results = {}
        arr = []
        labels = []
        entities = []

        sourceloc = row["sourceloc"]

        if sourceloc == None:
            print("Article has no sourceloc, discarding...")
            mark_as_traversed(crawldb, article_id)
            continue

        state = sourceloc.split(", ")[-1]

        ner_items_raw = row["ner"]
        if ner_items_raw == None:
            print("Article has no NERs, discarding...")
            mark_as_traversed(crawldb, article_id)
            continue

        len_ner_items_raw = len(ner_items_raw.keys())
        print(f"The current article has {len_ner_items_raw} unprocessed NERs")
        if len_ner_items_raw > UPPER_LIMIT_NERS:
            print(f"Article has way too many ners ({len_ner_items_raw}), discarding...")
            mark_as_traversed(crawldb, article_id)
            continue

        ner_items = {}
        ner_i = 0
        for (key, val) in ner_items_raw.items():
            ner_i += 1
            if ner_i > LIMIT_NERS:
                print(f'Note: Truncating number of NERs to only {LIMIT_NERS}')
                break

            ner_items[key] = val

        print(f'Processed ner_items count: {len(ner_items)}')

        if ner_items and len(ner_items.keys()) <= LIMIT_NERS:
            for ent, references in ner_items.items():
                coords = None
                wiki_results = None
                query = f"{ent}, {state}"
                page = []

                while not wiki_results:
                    # unfortunately even with postprocessing, locationIQ results are still pretty crazy. Wikipedia
                    # is better at resolving references to prominent landmarks, but the API I'm using isn't async
                    # so this takes a bit of time.

                    print(f"Searching for {ent}")

                    wiki_results = wikipedia.search(
                        query
                    )  # first search for the feature + state name...
                    if not wiki_results and query.endswith(
                        f", {state}"
                    ):  # if no results, just search for the feature...
                        query = query.replace(f", {state}", "")[0]
                    elif not wiki_results:
                        break
                if wiki_results:
                    for p in wiki_results[: min(2, len(wiki_results))]:
                        try:
                            page.append(wikipedia.WikipediaPage(p))
                        except Exception as e:
                            # this usually means one of the pages is a disambiguation page, which throws an exception
                            continue
                    coords = None
                    for p in page:
                        # for each wikipedia page, check to see if it has geocoordinates. If it does, then
                        # use that for the lat/long coordinates and use locationIQ to reverse lookup the
                        # state/county name
                        try:
                            if hasattr(p, "coordinates") and p.coordinates:

                                coords = [float(x) for x in p.coordinates]
                                print(f"Using {p.title} for {ent}")
                                try:

                                    reverse_geo = search_for_place(
                                        latitude=coords[0],
                                        longitude=coords[1],
                                        reverse_lookup=True,
                                    )
                                    if reverse_geo and "address" in reverse_geo:
                                        locations.append(reverse_geo["address"])
                                        arr.append(
                                            list([float(x) for x in sorted(coords)])
                                        )
                                        labels.append(p.title)
                                        entities.append(ent)
                                        break
                                except Exception as e:
                                    print(e, e.__class__, ent, p)
                                    pass
                        except Exception as e:
                            pass
                if not coords:
                    # If Wikipedia couldn't resolve it to a place, it's probably either a local org/business or
                    # not a real place. (NER model includes many false positives.) In case it's the former, we'll
                    # use locationIQ before giving up.
                    bounded = True
                    result = None
                    while not result:
                        # first exclude results outside the state bounding box, and see if that gives us any local
                        # results. If not, relax the `bounded` constraint, then search again. If still nothing,
                        # move on to the next feature.
                        result = search_for_place(ent, state, bounded=bounded)
                        if result:
                            # yay! we got a match. add this feature to the collection
                            result = result[0]
                            coords = [float(result["lon"]), float(result["lat"])]
                            arr.append(coords)
                            print(result)
                            labels.append(result["display_name"])
                            locations.append(result["address"])
                            entities.append(ent)
                        elif not bounded:
                            # welp. let's move on then
                            break
                        else:
                            # perhaps it is a non-local feature?
                            bounded = False

        # the following code is mostly for the purposes of generating Plotly choropleth figures. I highly recommend
        # eyeballing the outputs to make sure that they look reasonable before integrating the `tx_fips` object
        # with the database.

        mapbox_access_token = "pk.eyJ1IjoibmVvbmNvbnRyYWlscyIsImEiOiJjazhzazZxNmQwaG4xM2xtenB2YmZiaDQ5In0.CJhvMwotvbdJX4FhbyFCxA"

        if not arr:
            # Article has no fips after processing, move on to next one
            mark_as_traversed(crawldb, article_id)
            continue

        arr = np.array(arr)
        lat = arr[:, 1]
        lon = arr[:, 0]
        regions = model.predict(arr)
        by_region = defaultdict(list)
        results[row["article_url"]] = regions
        for ent, g in zip(labels, regions):
            print(f"Entity {ent} resolved to group: {g}")
            by_region[g].append(ent)
        filtered_lat = []
        filtered_lon = []
        fips_values = defaultdict(list)

        print("**************************************")
        print(entities)
        print("**************************************")

        db_specificity = "local"
        region_override = None
        db_list = []

        for entity, ent, g, lt, lng, address in zip(
            entities, labels, regions, lat, lon, locations
        ):
            if len(by_region[g]) >= 2:
                state = None
                county = None
                country = None
                filtered_lat.append(lt)
                filtered_lon.append(lng)
                fips = None

                if "state" in address:
                    state = address["state"]

                if "county" in address:
                    county = address["county"].replace(" County", "")
                if "country" in address:
                    country = address["country"]

                if state and county:
                    try:
                        fips = fips_index[state][county]
                        fips_values[fips].append(ent)
                    except:
                        print(f"No fips code for state {state}, county {county}")
                elif state:
                    db_specificity = 'regional'
                    region_override = state
                    try:
                        for county, fips_code in fips_index[state].items():
                            fips_values[fips_code].append(ent)
                    except:
                        print(f"No fips code for state {state}")

                # gather all data into db_list
                db_item = {
                    'entity': entity,
                    'label': ent,
                    'coord': Point(lng, lt),
                    'fips': str(fips),
                    'locref': len(ent),
                    'city': county,
                    'region': state
                }

                db_list.append(db_item)


        # **************************
        # Database Stuff
        if not db_list or len(db_list) == 0:
            mark_as_traversed(crawldb, article_id)
            continue

        # Sort by most to least entity references
        db_list.sort(key=lambda obj: obj['locref'], reverse=True)
        print(f"db_list: {db_list}")

        item0 = db_list[0]
        num_fips = len(db_list)
        if num_fips >= 5:
            db_specificity = 'regional'

        # Give moderator the override power
        moderator_specificity = row["specificity"]
        if moderator_specificity:
            print(f"Warning: Specificity already set by moderator: {moderator_specificity}")
            db_specificity = moderator_specificity

        db_state = item0['region']
        db_city = item0['city']
        db_longlat = item0['coord']

        db_coords_array = [item['coord'] for item in db_list]
        db_coords = adapt_point_array(db_coords_array)

        # db_coords = [item['coord'] for item in db_list]
        print("db_coords")
        print(db_coords)
        db_cities = [item['city'] for item in db_list]
        db_states = [item['region'] for item in db_list]
        db_labels = [item['label'] for item in db_list]
        db_entities = [item['entity'] for item in db_list]
        db_fips = [item['fips'] for item in db_list]
        db_locrefs = [item['locref'] for item in db_list]

        if not db_fips:
            db_fips = []

        new_row = dict(
            article_id=article_id,
            fips_processed = True,
            specificity=db_specificity,
            country='us',
            state = normalize_state_name(db_state),
            city = db_city,
            longlat = db_longlat,
            coords = db_coords,
            cities = db_cities,
            states = db_states,
            labels = db_labels,
            entities = db_entities,
            fips = db_fips,
            locrefs = db_locrefs,
            updated_at = dt.datetime.utcnow().replace(microsecond=0),
            updated_by = 'fips',
            lang = 'en'
        )

        print(f"SAVING TO DATABASE {article_id} for ({db_city}, {db_state})[{db_specificity}]")
        print(f"Article Title: {row['title']}")
        print(f"Article URL: {row['article_url']}")
        crawldb.update(new_row, ['article_id'])


        # ********************************************

        # tx_fips = [
        #     {"fips": str(k), "z_value": len(v), "references": ", ".join(v)}
        #     for k, v in fips_values.items()
        # ]
        # if not tx_fips:
        #     continue

        # display_map(tx_fips, counties)
        # print(json.dumps(geo_results, indent=4))


def display_map(tx_fips, counties):

    df = pd.DataFrame(tx_fips)
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=counties,
            locations=df.fips,
            z=df.z_value,
            text=df.references,
            colorscale="Viridis",
            zmin=0,
            zmax=6,
            marker_line_width=0,
        )
    )

    fig.update_layout(
        hovermode="closest",
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(lat=39, lon=-64),
            pitch=0,
            zoom=2,
        ),
    )

    fig.show()
