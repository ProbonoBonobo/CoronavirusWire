import sys
import pickle
from munch import Munch
import json
from collections import defaultdict
from src.coronaviruswire.utils import async_fetch
from lxml.html import fromstring
from urllib.parse import urljoin
import re
from unidecode import unidecode
import time

wiki_url = "https://en.wikipedia.org/wiki/List_of_newspapers_serving_cities_over_100,000_in_the_United_States"


class WikiMetadata:
    """Inherit serializable interface from Munch"""
    flattened = defaultdict(list)
    index = defaultdict(list)
    reverse_index = defaultdict(list)

    def __init__(self, page_title, heading, subheading, group, items):
        self.page_title = page_title
        self.heading = heading
        self.subheading = subheading or group
        self.group = group
        self.items = items
        self.flattened[self.page_title].extend(self.items)
        self.flattened[self.page_title].append(self.page_title.split(",")[0])
        self.index[self.page_title].append(self.__dict__)
        self.flattened[self.page_title] = sorted(set(self.flattened))
        for item in self.items:
            if self.page_title not in self.reverse_index[item]:
                self.reverse_index[item].append(self.page_title)

    # def __dict__(self):
    #     return {self.group: {self.page_title: self.items}}
    # def __str__(self):
    #     return str(self.__dict__)
    # def __repr__(self):
    #     return self.__str__


def get_text_chunks(node):
    def recursively_get_text(node):
        if node.getchildren():
            text = [recursively_get_text(kid) for kid in node.getchildren()]
            text = [txt for txt in text if txt]
            return '\n'.join(text)

        else:
            return node.text

    return recursively_get_text(node).split("\n")


def parse_wiki_footer(city, state, tree):
    rows = tree.xpath("//tr")
    page_title = ', '.join([city, state])
    curr = {}
    coll = []
    title = ""
    group = ""
    heading = ""
    subheading = ""
    for row in rows:
        items = []
        for kid in row.getchildren():
            try:
                if 'class' not in kid.attrib:
                    continue
                tag, klass, text = kid.tag, kid.attrib['class'].split(" "), [
                    unidecode(chunk) for chunk in get_text_chunks(kid)
                ]
                is_heading = 'navbox-title' in klass
                is_subheading = 'navbox-abovebelow' in klass
                is_group_label = 'navbox-group' in klass
                is_group_vals = 'navbox-list' in klass
                if is_heading:
                    heading = ' '.join(
                        [chunk for chunk in text if chunk not in 'vte'])
                    subheading = ""
                    group = ""
                    items = []
                elif is_subheading:
                    subheading = ' '.join(text)
                elif is_group_label:
                    group = ' '.join(text)
                elif is_group_vals:
                    items = text
                    print(f"Group items are: {items}")
                    obj = WikiMetadata(page_title, heading, subheading, group,
                                       items)
                    print(json.dumps(obj.__dict__))
                    coll.append(obj)

            except Exception as e:
                pass
    return coll


def extract_publications_by_city(tree):
    nodes = tree.xpath("""//*[@id="mw-content-text"]/div""")[0][4:]
    groups = defaultdict(list)
    k = ""
    for node in nodes:
        if node.tag == 'h2':
            k = re.sub(r"\[edit\]", "", node.text_content()).strip()
            if not k.strip() or "References" in k:
                continue

            if 'Bay Area' in k:
                k = ', '.join(*re.findall(r"(.+) Bay Area \(([^\)]+)\)", k))

            else:
                k = re.sub("^(Greater |References)", "", k)
        else:
            try:
                groups[k] = {
                    node.text_content(): urljoin("https://en.wikipedia.org/",
                                                 node.attrib['href'])
                    for node in node.xpath(".//a")
                    if not "index.php" in node.attrib['href']
                }
            except Exception as e:
                print(e)
    pubs = []
    for k, v in groups.items():
        if not k.strip():
            continue
        try:
            city, state = k.split(", ")
        except ValueError:
            continue

        city_url = f"""https://en.wikipedia.org/wiki/{city.replace(" ", "_")},_{state.replace(" ", "_")}"""

        for name, url in v.items():
            pubs.append({
                "name": name,
                "loc": k,
                "city": city,
                "state": state,
                "news_wiki": url,
                "city_wiki": city_url
            })

    return pubs


def main():
    response = async_fetch(wiki_url)
    parsed = fromstring(response.content)
    publications = extract_publications_by_city(parsed)

    curr = defaultdict()

    city_pages = async_fetch(*[pub['city_wiki'] for pub in publications])
    news_pages = async_fetch(*[pub['news_wiki'] for pub in publications])
    output = []
    for data, city_wiki, news_wiki in zip(publications, city_pages,
                                          news_pages):
        if 'Los' in data['loc']:
            print(json.dumps(data))
        try:
            print(data)
            if not city_wiki.status_code == '200':
                print(city_wiki.url)

            city_tree = fromstring(city_wiki.content,
                                   'https://en.wikipedia.org/wiki/')
            news_tree = fromstring(news_wiki.content,
                                   'https://en.wikipedia.org/wiki/')
            print(
                city_tree.xpath("//h1")[0].text,
                news_tree.xpath("//h1")[0].text_content())

            data['lat'] = unidecode(city_tree.cssselect(".latitude")[0].text)
            data['long'] = unidecode(city_tree.cssselect(".longitude")[0].text)
            data['city'], data['state'] = data['loc'].split(", ")
            data['info'] = [
                wiki.__dict__ for wiki in parse_wiki_footer(
                    data['city'], data['state'], city_tree)
            ]
            try:
                data['url'] = unidecode(
                    news_tree.xpath("//tr//a[contains(.,'.com')]/@href")[0])
                print(f"Got url for {data['name']}: {data['url']}")
            except Exception as e:
                print(data)
                print(f"{e.__class__.__name__} :: {e}")
            output.append(data)
        except Exception as e:
            print(f"Couldn't parse {data['city_url']}")
    return output

    #
    #     meta = city_tree.xpath(
    #         "//tr[contains(., 'Municipalities')]")
    #     # meta.extend(tree.xpath("//tr[contains(., 'Neighborhoods')//following-sibling::tr"))
    #     for node in meta:
    #         try:
    #
    #             vals = node.xpath(".//td/div/ul/li")
    #             for val in vals:
    #                 while val.has_children
    #
    #             v = [[child.text_content() for child in children][0]
    #                  for children in [node.getchildren() for node in vals]]
    #             cats = node.xpath(".//th")[0]
    #             print(f"{cats.text_content()} : {v}")
    #             data['info'][cats.text_content()] = v
    #
    #             # print(cats)
    #         except Exception as e:
    #             print(e.__class__.__name__, e)
    #             continue
    #     categories = [
    #         unidecode(node.xpath(".//th/a")[0].text) for node in meta
    #     ]
    #     data['categories'] = categories
    #     print(json.dumps(data, indent=4))
    # except Exception as e:
    #     print(e.__class__.__name__, e)
    #     print(sys.gettrace())
    #     print(f"Couldn't get {data}")
    # output.append(data)
    # for obj in publications:
    #     print(json.dumps(obj))
    #     return output


if __name__ == '__main__':
    out = main()
    meta = []
    for result in out:
        for obj in result['info']:
            meta.append(obj)
    with open('meta.json', 'w') as f:
        json.dump(
            {
                "flat": WikiMetadata.flattened,
                "reverse_index": WikiMetadata.reverse_index,
                "index": WikiMetadata.index
            }, f)
    with open('output.pkl', 'wb') as f:
        pickle.dump(out, f)
