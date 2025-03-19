import json
import pandas as pd
import ast

with open("genre_map.json") as genre_map:
    artist_genres = json.load(genre_map)

t300k_df = pd.read_csv("t300k_stripped.csv")

genres = {
    "edm": [
        "edm",
        "house",
        "techno",
        "step",
        "dance pop",
        "big room",
        " bass",
        "moombah",
        "uk garage",
        "trance",
        "electro swing",
        "electronica",
        "phonk",
        "synthwave",
    ]
}


def bucket(row):
    # only for edm rn idk what to do with others
    row["Bucket"] = None
    if not isinstance(row["Artist URIs"], str):
        return row
    for artist_uri in ast.literal_eval(row["Artist URIs"]):
        for genre in genres["edm"]:
            if artist_uri not in artist_genres:
                continue
            for artist_genre in artist_genres[artist_uri]:
                if genre in artist_genre:
                    row["Bucket"] = "edm"
    return row


t300k_df = t300k_df.drop_duplicates(["Artists", "Title"])
t300k_df = t300k_df.apply(bucket, axis=1)

t300k_df[t300k_df["Bucket"].isnull()].to_csv("PEPEGA.csv")
t300k_df[~t300k_df["Bucket"].isnull()].to_csv("PEPEGA2.csv")

print(sorted({y for x in artist_genres.values() for y in x}))
