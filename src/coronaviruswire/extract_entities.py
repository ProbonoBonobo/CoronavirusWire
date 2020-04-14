from src.coronaviruswire.common import db
from src.coronaviruswire.utils import extract_entities
import spacy


nlp = spacy.load("en_core_web_sm")
crawldb = db['crawldb']

if __name__ == '__main__':
    updates = []
    for row in crawldb.find(has_ner=False):
        content = row['content']
        ents = extract_entities(content)
        unique_ents = set(ents)
        counts = {}
        for ent in ents:
            counts[ent] = content.count(ent)
        updates.append({"article_id": row['article_id'], "ner": counts})
        if updates and not len(updates) % 200:
            crawldb.update_many(updates, ['article_id'])
            updates = []
    crawldb.update_many(updates, ['article_id'])

