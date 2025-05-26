import kagglehub

datasets = [
    "azathoth42/myanimelist",
    "svanoo/myanimelist-dataset",
    "marlesson/myanimelist-dataset-animes-profiles-reviews"
    "andreuvallhernndez/myanimelist"
    "hernan4444/anime-recommendation-database-2020",
]

path = kagglehub.dataset_download("azathoth42/myanimelist")

print("Path to dataset files:", path)
