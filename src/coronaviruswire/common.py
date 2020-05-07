import re
import dataset
import psycopg2

# db_config = {
#     "user": "kz",
#     "password": "admin",
#     "host": "127.0.0.1",
#     "port": "5432",
#     "database": "cvwire",
# }
db_config = {"user": "postgres",
             "password": "admin",
             "host": "34.83.188.109",
             "port": "5432",
             "database": "postgres"}

db = dataset.connect(
    f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
conn = psycopg2.connect(**db_config)


def create_moderation_table(drop_table=False):

    if "moderationtable" in db.tables:
        if drop_table:
            tab = db['moderationtable']
            tab.drop()
        return


    create_table_query = """CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE TABLE moderationtable
    (ID          SERIAL NOT NULL,
    ARTICLE_ID   uuid DEFAULT uuid_generate_v4 (),

    TITLE        TEXT NOT NULL,
    AUTHOR       TEXT,
    SOURCE_ID    VARCHAR(255) NOT NULL,
    ARTICLE_URL  TEXT,
    IMAGE_URL    TEXT,
    RAW_CONTENT  TEXT,
    CONTENT      TEXT NOT NULL,
    SUMMARY      TEXT,
    CATEGORY     TEXT[],

    NER          JSON,
    GEOTAGS      JSON,

    HAS_NER      BOOLEAN DEFAULT FALSE,
    HAS_GEOTAGS  BOOLEAN DEFAULT FALSE,
    HAS_COORDS   BOOLEAN DEFAULT FALSE,

    POSITIVITY   INT,
    MOD_STATUS   VARCHAR(255) DEFAULT 'pending',
    BOOST_FACTOR FLOAT8,
    FEATURED     BOOL DEFAULT FALSE,

    SPECIFICITY  TEXT,
    COUNTRY      TEXT,
    SOURCECOUNTRY TEXT,
    REGION       TEXT,
    SOURCEREGION TEXT,
    STATE        TEXT,
    SOURCESTATE  TEXT,
    CITY         TEXT,
    SOURCECITY   TEXT,
    SOURCELOC    TEXT,
    LONGLAT      POINT,
    SOURCELONGLAT POINT,
    COORDS       POINT[],
    CITIES       TEXT[],
    STATES       TEXT[],
    REGIONS      TEXT[],
    LABELS       TEXT[],
    ENTITIES     TEXT[],
    FIPS         TEXT[],
    LOCREFS      Int[],

    PUBLISHED_AT TIMESTAMP NOT NULL,
    CREATED_AT   TIMESTAMP NOT NULL DEFAULT NOW(),
    CREATED_BY   TEXT,
    UPDATED_AT   TIMESTAMP DEFAULT NOW(),
    UPDATED_BY   TEXT,

    NUM_CLICKS   INT DEFAULT 0,
    METADATA     JSON,
    LANG         TEXT,
    PRIMARY KEY (ARTICLE_ID)
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
    conn.commit()


def create_crawldb_table():
    if "crawldb" in db.tables:
        return

    create_table_query = """CREATE TABLE crawldb
     (id SERIAL PRIMARY KEY,
      name text,
      site TEXT,
      url TEXT,
      path TEXT,
      visited boolean NOT NULL DEFAULT FALSE,
      lastmod timestamp,
      articlebody text,
      headline text,
      lastcrawled timestamp,
      metadata json,
      has_metadata boolean,
      html text,
      audience_tag text,
      is_relevant boolean,
      status_code int,
      city text,
      state text,
      loc text,
      lat text,
      long text,
      ok boolean DEFAULT FALSE,
      length int DEFAULT 0)"""
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
    conn.commit()


def create_sitemaps_table():
    if "sitemaps" in db.tables:
        return
    create_table_query = (
        "CREATE TABLE sitemaps\n"
        "     (id SERIAL PRIMARY KEY,\n"
        "      site TEXT,\n"
        "      url TEXT NOT NULL UNIQUE,\n"
        "      last_modified date DEFAULT CURRENT_DATE,\n"
        "      has_timestamp boolean,\n"
        "      needs_crawl boolean DEFAULT TRUE,\n"
        "      last_crawled date,\n"
        "      is_fresh boolean default TRUE,\n"
        "      fresh_content_urls int,\n"
        "      fresh_sitemap_urls int,\n"
        "      fresh_urls int,\n"
        "      is_sitemapindex boolean,\n"
        "      is_urlset boolean,\n"
        "      type text)"
    )
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
    conn.commit()


# these regex patterns help the sitemap crawler to identify local URLs that are relevant to
# coronavirus
patterns = {
    "la": re.compile(
        r"(\bca\b|\bl.?a.?\b|\bo.?c.?\b|\bsce\b|adelanto|anaheim|angels|arcadia|asuza|beach|bernardino|beverly|brea|burbank|california|canyon|chargers|chino|clippers|community|compton|county|covina|culver|desert|dodger|downey|downtown|ducks|fullerton|garcetti|getty|glendale|glendora|hockley|hollywood|inland.{0,1}empire|irvine|joshua.?tree|koreatown|ladwp|lakewood|lapd|lausd|lax|library|local|los.{0,1}angeles|malibu|metro|monrovia|montebello|monterey|newsom|officials|ontario|orange|palmdale|pasadena|pomona|rail|rams|rancho|regional|riverside|san.?pedro|santa.{0,1}monica|segundo|shelter|sheriff|sherman.?oaks|socal|state|stores|torrance|ucla|usc|valley|ventura|westlake|yorba)",
        re.IGNORECASE,
    ),
    "coronavirus": re.compile(
        r"(hospital|campus|corona|moratorium|emergency|open|resum|shut|restaurant|travel|clos|pandemic|layoff|furlough|death.?toll|recession|stimulus|medic|nurse|quarantine|shut.?down|covid|postpone|cancel|virus)",
        re.IGNORECASE,
    ),
}

# most sites simply won't respond unless you identify yourself via user-agent string
default_headers = {
    "user-agent": """Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36""",
    "dnt": "1",
    "cookie": "nyt-a=29482ninwfwe_efw;",
}
