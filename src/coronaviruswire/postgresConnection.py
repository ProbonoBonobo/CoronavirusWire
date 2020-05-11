import psycopg2
import os

class PostgresConnection:

    # Default table name is ModerationTable
    def __init__(self, connection, tableName="moderationtable"):
        connection = None
        cursor = None
        try:
            connection = psycopg2.connect(
                user="postgres",
                password=os.environ.get('MODERATION_PASSWORD'),
                port="5432",
                host="35.188.134.37",
                database="postgres",
            )
            cursor = connection.cursor()
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

        self.connection = connection
        self.cursor = cursor
        self.tableName = tableName

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    # Must call commit
    def commit(self):
        try:
            print("Commiting Record...")
            self.connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def executeQuery(self, query, records=None):
        try:
            self.cursor.execute(query, records)
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

        return self.cursor.fetchall()

    def insertNewArticle(
        self,
        article_id,
        title,
        author,
        source_id,
        article_url,
        content,
        mod_status,
        published_at,
        created_by,
    ):
        try:
            query = f"INSERT INTO {self.tableName} (ARTICLE_ID, TITLE, AUTHOR, SOURCE_ID, ARTICLE_URL, CONTENT, MOD_STATUS, PUBLISHED_AT, CREATED_BY, UPDATED_BY) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            record_to_insert = (
                article_id,
                title,
                author,
                source_id,
                article_url,
                content,
                mod_status,
                published_at,
                created_by,
                created_by,
            )
            self.cursor.execute(query, record_to_insert)

        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def insertNewArticleDetailed(
        self,
        article_id,
        title,
        author,
        source_id,
        article_url,
        content,
        category,
        mod_status,
        published_at,
        created_by,
        city,
        region,
        metadata,
    ):
        try:
            query = f"INSERT INTO {self.tableName} (ARTICLE_ID, TITLE, AUTHOR, SOURCE_ID, ARTICLE_URL, CONTENT, CATEGORY, MOD_STATUS, PUBLISHED_AT, CREATED_BY, UPDATED_BY, CITY, REGION, METADATA) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            record_to_insert = (
                article_id,
                title,
                author,
                source_id,
                article_url,
                content,
                category,
                mod_status,
                published_at,
                created_by,
                created_by,
                city,
                region,
                metadata,
            )
            self.cursor.execute(query, record_to_insert)

        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def fetchLatestStoredArticlePublishDateForSource(self, source_id):
        try:
            query = f"SELECT (PUBLISHED_AT) FROM {self.tableName} WHERE SOURCE_ID=%s ORDER BY PUBLISHED_AT DESC"
            self.cursor.execute(query, [source_id])
            print("fetched latest article publish date:")
            articleDates = self.cursor.fetchall()

            if articleDates and len(articleDates) > 0:
                return articleDates[0][0]
            else:
                return None

        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)
            return None

    def printAllArticles(self):
        try:
            postgres_query = f"select * from {self.tableName}"
            self.cursor.execute(postgres_query)
            print("Fetching Records...")
            print(self.cursor.fetchall())
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def rowCount(self):
        try:
            postgres_query = f"SELECT COUNT(ARTICLE_ID) FROM {self.tableName}"
            self.cursor.execute(postgres_query)
            print("Fetching Records...")
            print(self.cursor.fetchall())
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def forceDeleteAllArticles(self):
        try:
            query = f"DELETE FROM {self.tableName}"
            self.cursor.execute(query)
            print(f"Deleting all records... Query:{query}")
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)

    def deleteArticleByArticleId(self, article_id):
        try:
            query = f"DELETE FROM {self.tableName} WHERE ARTICLE_ID=(%s)"
            self.cursor.execute(query, article_id)
            print("Deleting Record...")
        except (Exception, psycopg2.Error) as error:
            print("Postgres Error!", error)


#
