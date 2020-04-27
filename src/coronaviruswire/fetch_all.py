import json
import trio
import time
from urllib.parse import urljoin
from html import unescape
from gemeinsprache.utils import green
from src.coronaviruswire.common import default_headers, db
from lxml.html import fromstring
import random
from collections import deque
import httpx

xpath_selectors = {
    # "linked_pages": """//div[contains(@id,"mw-content-text")]//a""",
    # "link_hrefs": """//div[contains(@id,'mw-content-text')]//a""",
    # "linked_categories": """//div[contains(@id,"catlinks")]//a""",
    # "category_hrefs": """//div[contains(@id,"catlinks")]//a""",
    "geocoord_hrefs": """//a[contains(@href,"geohack.php")]""",
    "latitude": """//*[contains(@class,"latitude")]""",
    "longitude": """//*[contains(@class,"longitude")]""",
}

# lat = dom.cssselect(".latitude")
#             lng = dom.cssselect(".longitude")
#             categories = dom.cssselect(".catlinks li a")
#             links = dom.cssselect("#mw-content-text a")
# }
from collections import defaultdict


async def main(queue, limit=30):
    crawldb = db["us_metros2"]
    updates = []
    dups = defaultdict(list)
    seen = set()
    print(f"Initializing crawler....")

    async def fetch(name, url, pagetype, parent):
        seen.add(url)
        async with httpx.AsyncClient() as client:
            try:
                res = await client.get(url, headers=default_headers, timeout=10)
            except Exception as e:
                print(e.__class__.__name__, e, url)
                return
            print(f"Fetched {url}")
        dom = fromstring(res.content)
        if not res.status_code == 200:
            return
        stub = {
            "page": name,
            "url": url,
            "type": pagetype,
            "parent_url": parent,
            "page_hrefs": [
                f"https://en.wikipedia.org{link}"
                for link in dom.xpath("//div[contains(@id,'mw-content-text')]//a/@href")
            ],
            "page_links": [
                link.attrib["title"]
                for link in dom.xpath("//div[contains(@id,'mw-content-text')]//a")
                if link and hasattr(link, "attrib") and "title" in link.attrib
            ],
            "response": res.status_code,
            "ok": res.status_code == 200,
            "length": len(res.content),
            "category_hrefs": [
                f"https://en.wikipedia.org{link}"
                for link in dom.xpath("//div[contains(@id,'catlinks')]//a/@href")
            ],
            "category_links": [
                link.attrib["title"]
                for link in dom.xpath("//div[contains(@id,'catlinks')]//a")
                if link and hasattr(link, "attrib") and "title" in link.attrib
            ],
            "latitude": None,
            "longitude": None,
        }
        for colname, sel in xpath_selectors.items():
            values = []
            if colname in ("latitude", "longitude"):
                result = [node.text_content() for node in dom.xpath(sel)]
                if result:
                    result = result[0]
                    stub[colname] = result

        updates.append(stub)

        for url in stub["category_hrefs"] + stub["page_hrefs"]:
            # print(green(url))
            if url and url in seen:
                dups[url].append(name)
                # print(f"Page {url} is duplicated on: {len(dups[url])} pages")
            elif url and url not in seen and "Talk" not in url:
                name = url.split("/wiki/")[-1].replace("_", " ")
                queue.append((name, url, "link", parent))

    queue = deque(queue)

    while queue:
        async with trio.open_nursery() as nursery:
            for i in range(limit):
                next_url = None
                while next_url is None:
                    page_name, _next, page_type, parent = queue.popleft()

                    if (
                        _next not in seen
                        and "User" not in _next
                        and "Talk" not in _next
                    ):
                        seen.add(_next)
                        next_url = _next
                nursery.start_soon(fetch, page_name, next_url, page_type, parent)
        if len(updates) and len(updates) > 500:
            print(f"Updating database...")
            crawldb.upsert_many(
                [{k: v for k, v in update.items()} for update in updates], ["url"]
            )
            for item in updates:
                print(green(json.dumps(item, indent=4)))
            print(f"Inserted {len(updates)} items.")
            updates = []


if __name__ == "__main__":
    with open("../../lib/us_county_wikicats--nodups.json", "r") as f:
        data = json.load(f)
    queue = set()
    for k, cats in data.items():
        if "metro" not in k.lower():
            continue
        url = f'https://en.wikipedia.org/wiki/{k.replace(" ", "_")}'
        # queue.add((k, urljoin(f"https://en.wikipedia.org/wiki/", f"Category:{k.replace(' ', '_')}"), "category", k))
        queue.add(
            (
                k,
                urljoin(f"https://en.wikipedia.org/wiki/", k.replace(" ", "_")),
                "region",
                k,
            )
        )
        for c in cats:
            if "Talk" in c:
                continue
            queue.add(
                (
                    c,
                    urljoin("https://en.wikipedia.org/wiki/", c.replace(" ", "_")),
                    "feature",
                    k,
                )
            )
            # queue.add((c, urljoin(f"https://en.wikipedia.org/wiki/", f"Category:{c.replace(' ', '_')}"), "category", k))
        # if len(list(queue)) > 1000:
        #     continue
    q = set()
    for t in queue:
        name, url, t, parent = t
        if not url.startswith("http"):
            url = f"https://en.wikipedia.org/wiki/{url}"
        q.add((name, url, t, parent))
    print(q)
    start_time = time.perf_counter()
    trio.run(main, q)
    duration = time.perf_counter() - start_time
    print("Took {:.2f} seconds".format(duration))
