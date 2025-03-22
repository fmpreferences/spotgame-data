import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import sqlite3
import requests
import re
from dotenv import load_dotenv

load_dotenv()

BUCKET_MAP = {
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

DB_PATH = "somewhereonmyvps.sqlite3"
KWORB_PATH = "KWORB.csv"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

WHITELIST_EDM = pd.read_csv('edm_whitelist')['aid'].unique()


def pull_kworb(threshold):
    years = [
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
        2010,
        2005,
        2000,
        1990,
        1980,
        1970,
        1960,
        1950,
    ]

    df = pd.DataFrame()

    for year in years:
        pull = pd.read_html(f"https://kworb.net/spotify/songs_{year}.html")[0]
        df = pd.concat([df, pull[pull["Streams"] >= threshold]])

    df = df.drop_duplicates(["Artist and Title", "Streams"]).sort_values(
        "Streams", ascending=False
    )
    df["streams"] = df["Streams"]

    df["Occurrence"] = df.groupby("Artist and Title").cumcount()
    df[["Artist", "title"]] = df["Artist and Title"].str.split(" - ", n=1, expand=True)

    df.to_csv(KWORB_PATH, index=False)

    print(df)


def alnum_only(s):
    return re.sub("[^a-z0-9]", "", s.lower())


def get_real_song(row):
    name, artist, rank = row["title"], row["Artist"], row["Occurrence"]
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
                    int(res["album"]["release_date"][:4]),
                    res["name"],
                    res["id"],
                    [artist_item["name"] for artist_item in res["artists"]],
                    [artist_item["id"] for artist_item in res["artists"]],
                )
            )
    try:
        _, release_date, track_name, uri, artist_names, artist_uris = sorted(lst)[rank]
    except IndexError:
        print(name, artist, rank)
        return None
    row["title"] = track_name
    row["artists"] = artist_names
    row["id"] = uri
    row["artist_id"] = artist_uris
    row["date"] = release_date
    return row.drop(["Artist and Title", "Artist", "Streams", "Daily", "Occurrence"])


def process_artist_genres(artists_lst):
    dfs = []

    for i in range(0, len(artists_lst), 50):
        artists_sp = sp.artists(artists_lst[i : i + 50])
        for artist in artists_sp["artists"]:
            temp_genres = artist["genres"]
            if not temp_genres:
                if artist['id'] in WHITELIST_EDM:
                    temp_genres = ['edm']
                else:
                    temp_genres = [None]
            df2 = pd.DataFrame(
                {
                    "artist_name": artist["name"],
                    "artist_id": artist["id"],
                    "genre": temp_genres,
                }
            )
            dfs.append(df2)
    df = pd.concat(dfs)
    df = df.explode("genre")
    return df


def bucket(row):
    # only for edm rn idk what to do with others
    row["bucket"] = None
    for genre in BUCKET_MAP["edm"]:
        if genre in row["genre"]:
            row["bucket"] = "edm"
    return row.drop("genre")


def process_songs():
    tracks = pd.read_csv(KWORB_PATH)

    step = 250

    df = pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        for i in range(0, len(tracks), step):
            while True:
                try:
                    track_df = tracks[i : i + step]
                    df = pd.concat([df, track_df.apply(get_real_song, axis=1)])
                except requests.exceptions.ReadTimeout:
                    print("rate limit")
                    time.sleep(5)
                    continue
                break
            # break

        df = df.drop_duplicates("id").dropna()

        # print(df)
        song_artist_df = df[["id", "artist_id"]].explode("artist_id")
        all_artists = song_artist_df["artist_id"].unique()
        genre_df = process_artist_genres(all_artists)
        artist_df = genre_df.drop("genre", axis=1).drop_duplicates()
        genre_df = genre_df.drop("artist_name", axis=1).dropna()

        df = df.drop(["artists", "artist_id"], axis=1)
        df.to_sql("tracks", conn, if_exists="replace", index=False)
        song_artist_df.to_sql("track_artists", conn, if_exists="replace", index=False)

        bucket_df = (
            df[["id", "title"]]
            .merge(song_artist_df, "inner")
            .merge(genre_df, "inner")[["id", "genre"]]
        )
        bucket_df.apply(bucket, axis=1).dropna().drop_duplicates().to_sql(
            "buckets", conn, if_exists="replace", index=False
        )

        genre_df.to_sql("genres", conn, if_exists="replace", index=False)
        artist_df.to_sql("artists", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    pull_kworb(250000000)
    process_songs()
