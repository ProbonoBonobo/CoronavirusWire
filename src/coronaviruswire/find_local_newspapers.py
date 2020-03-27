import sys
import json
from collections import defaultdict
from src.coronaviruswire.utils import async_fetch
from lxml.html import fromstring
from urllib.parse import urljoin
import re
from unidecode import unidecode

wiki_url = "https://en.wikipedia.org/wiki/List_of_newspapers_serving_cities_over_100,000_in_the_United_States"


def main():
    response = async_fetch(wiki_url)
    parsed = fromstring(response.content)
    root = parsed.xpath("/html/body/div[3]/div[3]/div[4]/div")[0]
    publications = []
    curr = defaultdict()
    for node in root.getchildren()[4:]:
        if node.tag == 'h2':
            if curr:
                publications.append(dict(curr))
                curr = defaultdict()
            curr['loc'] = re.sub(r'(^Greater | Bay Area.+$)', '', node.cssselect(".mw-headline")[0].text)
        elif node.tag == 'ul':
            curr['name'] = node.cssselect("a")[0].text
            curr['city_url'] = urljoin('https://en.wikipedia.org/wiki/', curr['loc'].replace(" ", "_"))
            curr['newspaper_url'] = urljoin('https://en.wikipedia.org/', node.xpath(".//a/@href")[0])
    responses = async_fetch([pub['city_url'] for pub in publications])
    output = []
    for data, res in zip(publications, responses):
      try:
        print(data)
        if not res.status_code == '200':
            print(res.url)

        tree = fromstring(res.content, 'https://en.wikipedia.org/wiki/')
        data['lat'] = unidecode(tree.cssselect(".latitude")[0].text)
        data['long'] = unidecode(tree.cssselect(".longitude")[0].text)
        data['city'], data['state'] = data['loc'].split(", ")
        data['info'] = {}

        meta = tree.xpath("//tr[contains(., 'Municipalities')]//following-sibling::tr")
        # meta.extend(tree.xpath("//tr[contains(., 'Neighborhoods')//following-sibling::tr"))
        for node in meta:
          try:

            vals = node.xpath(".//td/div/ul/li")

            v = [[child.text_content() for child in children][0] for children in [node.getchildren() for node in vals]]
            cats = node.xpath(".//th")[0]
            print(f"{cats.text_content()} : {v}")
            data['info'][cats.text_content()] = v

            # print(cats)
          except Exception as e:
            print(e.__class__.__name__, e)
            continue
        categories = [unidecode(node.xpath(".//th/a")[0].text) for node in meta]
        data['categories'] = categories
        print(json.dumps(data, indent=4))
      except Exception as e:
          print(e.__class__.__name__, e)
          print(sys.gettrace())
          print(f"Couldn't get {data}")
      output.append(data)
    for obj in publications:
        print(json.dumps(obj))
    return output



if __name__ == '__main__':
    out = main()
    with open('output.json', 'w') as f:
        json.dump({obj['name']: obj for obj in out}, f, indent=4, sort_keys=True)