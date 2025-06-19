import logging
import random
import requests
import time
from typing import Callable, Iterable

RETRY_LIMIT = 5
QUERY_LIMIT = 1000

REVIEWS_PER_PAGE = 25
REVIEWS_PER_MEDIA_LIMIT = 3
REVIEW_WORD_THRESHOLD = 100
REVIEW_N_RATINGS_THRESHOLD = 10
REVIEW_USER_RATING_THRESHOLD = 7
REVIEW_THROWOUT_CHANCE = 0.5

MEDIA_PER_PAGE = 20
MEDIA_POPULARITY_THRESHOLD = 10000
MEDIA_N_THRESHOLD = 2000
MEDIA_ID_LIMIT = 20000
MEDIA_THROWOUT_CHANCE = 0.5
MEDIA_PAGE_THROWOUT_CHANCE = 0.25

QUERY_URL = "https://graphql.anilist.co"


logger = logging.getLogger(__name__)

# popularity_greater: 100
# TODO: add the media filters into the GraphQL. -- Need to incorporate popularity, most anime don't have reviews
media_query_pre = """
query ($page: Int = 1, $perPage: Int = 5) {
    Page(page: $page, perPage: $perPage) {
        pageInfo {
            currentPage
            hasNextPage
            perPage
        }
        media (
            type: ANIME
            popularity_greater: $POPULARITY
            sort: POPULARITY_DESC
        ) {
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
} 
"""
media_query = media_query_pre.replace("$POPULARITY", str(MEDIA_POPULARITY_THRESHOLD))
review_query = """
query ($mediaId: Int, $page: Int = 1, $perPage: Int = 5) {
    Page(page: $page, perPage: $perPage) {
        pageInfo {
            currentPage
            hasNextPage
            perPage
        }
        reviews(
            mediaId: $mediaId
            sort: RATING_DESC
        ) {
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
    limit: int,
    query: dict[str, any],
    vars: dict[str, any],
    page_num: int,
    page_throwout_chance: float = 0,
) -> Iterable[dict[str, any]]:
    while True:
        vars["page"] = page_num
        request = {"query": query, "variables": vars}

        response = attempt_query(limit, request)
        if response is None:
            logger.error("Failed to exhaust pages")
            break

        if random.random() >= page_throwout_chance:
            yield response["data"]["Page"]

        if not response["data"]["Page"]["pageInfo"]["hasNextPage"]:
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
                if count >= REVIEWS_PER_MEDIA_LIMIT:
                    threshold_reached = True
                    break
        if threshold_reached:
            break


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

    if random.random() < REVIEW_THROWOUT_CHANCE:
        return False

    return True


def get_media(
    media_filter: Callable[[dict[str, any]], bool],
) -> Iterable[dict[str, any]]:
    vars = {"perPage": MEDIA_PER_PAGE}
    threshold_reached = False
    count = 0
    for page in exhaust_pages(
        QUERY_LIMIT, media_query, vars, 1, MEDIA_PAGE_THROWOUT_CHANCE
    ):
        for media in page["media"]:
            if media_filter(media):
                yield media
                count += 1
                if count >= MEDIA_N_THRESHOLD:
                    threshold_reached = True
                    break

        if threshold_reached:
            break


def is_good_media(media: dict[str, any]) -> bool:
    rand = random.random()
    if rand < MEDIA_THROWOUT_CHANCE:
        return False
    return True


def media_log(media: dict[str, any]) -> str:
    return f"id {media['id']}: {media['title']['english']}"


def review_log(review: dict[str, any]) -> str:
    return f"\treview id {review['id']}: by user {review['user']['name']}"


def get_data() -> Iterable[tuple[dict[str, any], dict[str, any]]]:
    for media in get_media(is_good_media):
        logger.info(media_log(media))
        media_id = media["id"]
        reviews = get_reviews(media_id, is_good_review)
        for review in reviews:
            logger.info(review_log(review))
            yield (media, review)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    logging.basicConfig(format="%(name)s:%(levelname)s:%(message)s")
    for x in get_data():
        pass
