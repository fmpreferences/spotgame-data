import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import ast
import json
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

df = pd.read_csv("t300k_stripped.csv")

artists = set()

for i, row in df.iterrows():
    if not isinstance(row["Artist URIs"], str):
        continue
    artists.update(ast.literal_eval(row["Artist URIs"]))

genres = {}

artists_lst = list(artists)

for i in range(0, len(artists), 50):
    artists_sp = sp.artists(artists_lst[i : i + 50])
    for artist in artists_sp["artists"]:
        genres[artist["uri"]] = artist["genres"]


with open("genre_map.json", "w") as genre_f:
    json.dump(genres, genre_f)
