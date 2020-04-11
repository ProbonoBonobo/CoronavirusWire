# coronaviruswire

A collection of Python modules I'm contributing to the CoronavirusWire project. These will mainly pertain to crawling, scraping, parsing, and aggregating news articles, but may employ some machine learning models in the near future to optimize data quality.

## instructions

```pip install pipenv
git clone https://github.com/ProbonoBonobo/CoronavirusWire.git
cd CoronavirusWire
pipenv install --skip-lock
python -m spacy download en_core_web_sm
pipenv run python src/coronaviruswire/crawler.py
```

This will launch several sitemap crawlers to populate the queue with URLs. Once that is finished, the crawler requests up to 25 urls simultaneously, accumulates the responses, and does a rudimentary parsing step before writing the results to a file. (This generates a lot of console output!)
