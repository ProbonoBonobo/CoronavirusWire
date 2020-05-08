from src.coronaviruswire.common import (db, database_name)
from src.coronaviruswire.utils import extract_entities, format_text
import spacy

LIMIT_CYCLES = 20
LIMIT_ARTICLES = 100


nlp = spacy.load("en_core_web_sm")
crawldb = db[database_name]

if __name__ == "__main__":

    for x in range(LIMIT_CYCLES):
        updates = []
        print(f"Extracting entities..")
        rows = [
            row for row in crawldb.find(ner=None, mod_status='approved', _limit=LIMIT_ARTICLES)
        ]

        print(f"Found {len(rows)} rows that are approved to extract entities.")

        if len(rows) == 0:
            rows = [
                row for row in crawldb.find(has_ner=False, _limit=LIMIT_ARTICLES)
            ]

        print(f"All approved articles have extracted entities, Found other {len(rows)} rows to extract entities.")

        for row in rows:
            print(f"Updating {row}")
            content = format_text(
                "\n".join(
                    [
                        str(attr)
                        for attr in [
                            row["title"],
                            row["summary"],
                            row["keywords"],
                            row["content"],
                        ]
                        if attr
                    ]
                )
            )
            ents = [ent.strip() for ent in extract_entities(content)]
            unique_ents = set(ents)
            counts = {}
            for ent in ents:
                counts[ent] = content.count(ent)
            updates.append(
                {"article_id": row["article_id"], "ner": counts, "has_ner": True}
            )
            if updates and not len(updates) % 1000:
                crawldb.update_many(updates, ["article_id"])
                updates = []
        crawldb.update_many(updates, ["article_id"])
