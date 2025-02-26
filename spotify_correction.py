import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())


def alnum_only(s):
    return re.sub("[^a-z0-9]", "", s.lower())


def get_real_song(row):
    name, artist, rank = row["Title"], row["Artist"], row["Occurrence"]
    reses = sp.search(f"{name} artist:{artist}", type="track")["tracks"]["items"]
    lst = []
    for res in reses:
        aname = alnum_only(name)
        if aname in alnum_only(res["name"]) and artist in [
            artist_item["name"] for artist_item in res["artists"]
        ]:
            lst.append(
                (
                    -res["popularity"],
                    res["album"]["release_date"][:4],
                    res["name"],
                    res["uri"],
                    [artist_item["name"] for artist_item in res["artists"]],
                    [artist_item["uri"] for artist_item in res["artists"]],
                )
            )
    try:
        _, release_date, track_name, uri, artist_names, artist_uris = sorted(lst)[rank]
    except IndexError:
        with open("POOPOO", "w") as POOPOO:
            POOPOO.write(str(reses))
        print(name, artist, rank)
        return None
    row["Title"] = track_name
    row["Artists"] = artist_names
    row["URI"] = uri
    row["Artist URIs"] = artist_uris
    row["Release Date"] = release_date
    return row.drop(["Artist and Title", "Artist", "Daily", "Occurrence"])


# df = pd.DataFrame()
tracks = pd.read_csv("t300k.csv")

step = 100

for i in range(0, len(tracks), step):
    while True:
        try:
            track_df = tracks[i : i + step]
            df = track_df.apply(get_real_song, axis=1)
            df.to_csv(
                "t300k_stripped.csv",
                mode="a",
                header=not os.path.exists("t300k_stripped.csv"),
                index=False,
            )
        except requests.exceptions.ReadTimeout:
            print("rate limit")
            time.sleep(5)
            continue
        break
