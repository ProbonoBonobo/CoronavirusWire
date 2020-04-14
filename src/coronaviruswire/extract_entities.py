from src.coronaviruswire.common import db
from src.coronaviruswire.utils import extract_entities,format_text
import spacy



nlp = spacy.load("en_core_web_sm")
crawldb = db['moderationtable']

if __name__ == '__main__':
    updates = []
    print(f"Extracting entities..")
    for row in [row for row in crawldb.find() if not row['has_ner']]:
        print(f"Updating {row}")
        content = format_text('\n'.join([str(attr) for attr in [row['title'], row['summary'], row['keywords'], row['content']] if attr]))
        ents = [ent.strip() for ent in extract_entities(content)]
        unique_ents = set(ents)
        counts = {}
        for ent in ents:
            counts[ent] = content.count(ent)
        updates.append({"article_id": row['article_id'], "ner": counts, "has_ner": True})
        if updates and not len(updates) % 10:
            crawldb.update_many(updates, ['article_id'])
            updates = []
    crawldb.update_many(updates, ['article_id'])

