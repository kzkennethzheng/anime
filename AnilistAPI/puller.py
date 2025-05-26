import requests

base = "https://graphql.anilist.co"

# Here we define our query as a multi-line string
query = """
query ($id: Int) { # Define which variables will be used in the query (id)
  Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
    id
    title {
      english
      native
    }
  }
}
"""
query = """
query ($mediaId: Int, $page: Int = 1, $perPage: Int = 5) {
  Page(page: $page, perPage: $perPage) {
    reviews(mediaId: $mediaId) {
      id
      summary
      body
      rating
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

query = """
query ($id: Int, $page: Int, $perPage: Int, $search: String) {
    Page (page: $page, perPage: $perPage) {
        pageInfo {
            currentPage
            hasNextPage
            perPage
        }
        media (id: $id, search: $search) {
            id
            title {
                english
                romaji
            }
        }
    }
}
"""

# Define our query variables and values that will be used in the query request
# variables = {"mediaId": 15125, "page": 1, "perPage": 5}
variables = {"search": "Hero", "page": 2, "perPage": 3}

url = "https://graphql.anilist.co"

# Make the HTTP Api request
response = requests.post(url, json={"query": query, "variables": variables})
print(response.json())
