import logging
import random
import requests
import time
from typing import Callable, Iterable

REQUESTS_PER_MIN = 30
RETRY_LIMIT = 5
REVIEWS_PER_PAGE = 25
QUERY_LIMIT = 1000
THROWOUT_CHANCE = 0.15

# REVIEW_THRESHOLD = 100
REVIEW_THRESHOLD = 5
REVIEW_WORD_THRESHOLD = 100
REVIEW_N_RATINGS_THRESHOLD = 10
REVIEW_USER_RATING_THRESHOLD = 7

MEDIA_POPULARITY_THRESHOLD = 100
MEDIA_N_THRESHOLD = 20000
# MEDIA_ID_LIMIT = 500000
MEDIA_ID_LIMIT = 5

QUERY_URL = "https://graphql.anilist.co"


logger = logging.getLogger(__name__)

# Here we define our query as a multi-line string
media_query = """
query ($id: Int) { # Define which variables will be used in the query (id)
    Media (id: $id, type: ANIME) {
        id
        title {
            english
            native
            romaji
        }
        meanScore
        genres
        description
        startDate {
            year
            month
            day
        }
        endDate {
            year
            month
            day 
        }
        isAdult
        isLicensed
        countryOfOrigin
        popularity
        rankings {
            context
            rank
        }
        studios {
            nodes {
                id
                name
            }
        }
    }
}

"""
review_query = """
query ($mediaId: Int, $page: Int = 1, $perPage: Int = 5) {
    Page(page: $page, perPage: $perPage) {
        pageInfo {
            currentPage
            hasNextPage
            perPage
        }
        reviews(mediaId: $mediaId) {
        id
        summary
        body
        rating
        ratingAmount
        score
        user {
            name
        }
        createdAt
        updatedAt
        siteUrl
        }
    }
}
"""


def attempt_query(limit: int, request: dict[str, any]) -> dict[str, any] | None:
    for _ in range(limit):
        response = requests.post(QUERY_URL, json=request)
        if "Retry-After" in response.headers:
            retry_after = int(response.headers["Retry-After"])
            logger.error(f"Rate limit reached. Waiting for {retry_after} seconds")
            time.sleep(retry_after)
            continue

        body = response.json()

        if "errors" not in body:
            return body

        for error in body["errors"]:
            logger.error(f'Received error status {error["status"]}: {error["message"]}')
            if error["status"] == 404:
                return None

    return None


def exhaust_pages(
    limit: int, query: dict[str, any], vars: dict[str, any], page_num: int
) -> Iterable[dict[str, any]]:
    while True:
        vars["page"] = page_num
        request = {"query": query, "variables": vars}

        response = attempt_query(limit, request)
        if response is None:
            logger.error("Failed to exhaust pages")
            break

        yield response["data"]["Page"]

        if not response["data"]["pageInfo"]["hasNextPage"]:
            break
        page_num += 1


def get_reviews(
    media_id: int, review_filter: Callable[[dict[str, any]], bool]
) -> Iterable[dict[str, any]]:
    vars = {"perPage": REVIEWS_PER_PAGE, "mediaId": media_id}
    count = 0
    threshold_reached = False
    for page in exhaust_pages(QUERY_LIMIT, review_query, vars, 1):
        for review in page["reviews"]:
            if review_filter(review):
                yield review
                count += 1
                if count >= REVIEW_THRESHOLD:
                    threshold_reached = True
                    break
        if threshold_reached:
            break


# TODO number of reviews per show threshold
# TODO throwout chance for reviews


def is_good_review(review: dict[str, any]) -> bool:
    # TODO add NLP strategy for judging writing quality
    if len(review["body"].split()) < REVIEW_WORD_THRESHOLD:
        return False

    # number of times a user rated the review
    if review["ratingAmount"] < REVIEW_N_RATINGS_THRESHOLD:
        return False

    # number of thumbs up - number of thumbs down
    if review["rating"] < REVIEW_USER_RATING_THRESHOLD:
        return False

    return True


def get_media(
    media_id: int, media_filter: Callable[[dict[str, any]], bool]
) -> dict[str, any] | None:
    query = {"query": media_query, "variables": {"id": media_id}}
    media = attempt_query(QUERY_LIMIT, query)
    if media is None or random.random() < THROWOUT_CHANCE:
        return None

    media = media["data"]["Media"]
    if not media_filter(media):
        return None
    return media


def is_good_media(media: dict[str, any]) -> bool:
    # media['data']['Media']['popularity']
    return media["popularity"] >= MEDIA_POPULARITY_THRESHOLD


def media_log(media: dict[str, any]) -> str:
    return f"id {media["id"]}: {media["title"]["english"]}"


def review_log(review: dict[str, any]) -> str:
    return f"review id {review["id"]}: by user {review["user"]["name"]}"


def main() -> None:
    for media_id in range(1, MEDIA_ID_LIMIT):
        media = get_media(media_id, is_good_media)
        if media is None:
            continue

        logger.info(media_log(media))
        reviews = get_reviews(media_id, is_good_review)
        for review in reviews:
            logger.info(review_log(review))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    logging.basicConfig(format="%(name)s:%(levelname)s:%(message)s")
    main()
