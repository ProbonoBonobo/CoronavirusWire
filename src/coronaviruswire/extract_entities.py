from src.coronaviruswire.common import db
from src.coronaviruswire.utils import extract_entities_with_allennlp, format_text
import spacy

LIMIT_ARTICLES = 100

crawldb = db["moderationtable"]

if __name__ == "__main__":
    updates = []
    print(f"Extracting entities..")
    for row in [row for row in crawldb.find(ner=None, mod_status='approved', _limit=LIMIT_ARTICLES)]:
        print(f"Updating {row}")
        content = format_text(
            "\n".join(
                [
                    str(attr)
                    for attr in [
                        row["title"],
                        row["summary"],
                        row["content"],
                    ]
                    if attr
                ]
            )
        )
        ents = [[ent.strip().title(), ent.strip()][int(bool('\'' in ent))] for ent in extract_entities_with_allennlp(content)]
        unique_ents = set(ents)
        counts = {}
        for ent in ents:
            counts[ent] = content.lower().count(ent)
        # updates.append(
        #     {"article_id": row["article_id"], "ner": counts, "has_ner": True}
        # )
        crawldb.update(  {"article_id": row["article_id"], "ner": counts, "has_ner": True}, ['article_id'])
        # if updates and not len(updates) % 1000:
        #     crawldb.update_many(updates, ["article_id"])
        #     updates = []
    crawldb.update_many(updates, ["article_id"])
