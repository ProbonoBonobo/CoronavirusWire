import csv
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


def async_fetch(*urls,
                max_requests=25,
                headers=default_headers,
                timeout=60,
                **kwargs):
    if isinstance(urls, str):
        urls = [urls]
    urls = [url_normalize(url) for url in urls]
    chan = {}

    async def fetch():
        limit = trio.CapacityLimiter(max_requests)

        async def async_fetch(url, headers, timeout, **kwargs):
            async with httpx.AsyncClient() as client:
                chan[url] = await client.get(url,
                                             headers=default_headers,
                                             timeout=timeout,
                                             **kwargs)

        async with trio.open_nursery() as nursery:
            for url in urls:
                async with limit:
                    nursery.start_soon(async_fetch, url, headers, timeout,
                                       **kwargs)

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


def iter_csv(fp="/home/kz/projects/coronaviruswire/lib/newspapers.csv",
             delimiter="\t"):
    with open(fp, 'r') as f:
        f_csv = csv.DictReader(f, delimiter='\t')
        cols = f_csv.fieldnames
        for row in f_csv:
            yield dict(row.items())


def load_csv(fp="/home/kz/projects/coronaviruswire/lib/newspapers.tsv",
             delimiter="\t"):
    return list(iter_csv(fp, delimiter))


if __name__ == '__main__':
    responses = async_fetch("msn.com", "yahoo.com", "nytimes.com",
                            "news.ycombinator.com")
    # responses = parse_html(responses)
    # doc = responses[0]
    # print([node for node in doc.tree.xpath("//a/@href")])
    # print(doc.__dict__)
    # print("ok")
