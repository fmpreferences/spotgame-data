import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import sqlite3
import requests
import os
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

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())


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

    df = df.drop_duplicates(["Artist and Title"]).sort_values(
        "Streams", ascending=False
    )

    df["Occurrence"] = df.groupby("Artist and Title").cumcount()
    df[["Artist", "Title"]] = df["Artist and Title"].str.split(" - ", n=1, expand=True)

    df.to_csv("KWORB_PULL.csv", index=False)

    print(df)


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
    row["Title"] = track_name
    row["Artists"] = artist_names
    row["ID"] = uri
    row["ArtistID"] = artist_uris
    row["Date"] = release_date
    return row.drop(["Artist and Title", "Artist", "Daily", "Occurrence"])


def process_artist_genres(artists_lst, conn):
    dfs = []

    for i in range(0, len(artists_lst), 50):
        artists_sp = sp.artists(artists_lst[i : i + 50])
        for artist in artists_sp["artists"]:
            if artist["genres"]:
                df2 = pd.DataFrame(
                    {
                        "ArtistName": artist["name"],
                        "ArtistID": artist["id"],
                        "Genre": artist["genres"],
                    }
                )
                dfs.append(df2)
    df = pd.concat(dfs)
    df = df.explode("Genre")
    df.to_sql("genres", conn, if_exists='replace', index=False)
    return df

def bucket(row):
    # only for edm rn idk what to do with others
    row["Bucket"] = None
    for genre in BUCKET_MAP["edm"]:
        if genre in row['Genre']:
            row["Bucket"] = "edm"
    return row.drop('Genre')

def process_songs():
    tracks = pd.read_csv("t300k.csv")

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

        df = df.drop_duplicates("ID").dropna()

        # print(df)
        song_artist_df = df[["ID", "ArtistID"]].explode("ArtistID")
        all_artists = song_artist_df["ArtistID"].unique()
        genre_df = process_artist_genres(all_artists, conn)

        df = df.drop(["Artists", "ArtistID"], axis=1)
        df.to_sql(
            "tracks", conn, if_exists="replace", index=False
        )
        song_artist_df.to_sql("track_artists", conn, if_exists="replace", index=False)

        bucket_df = df[['ID', 'Title']].merge(song_artist_df, 'inner').merge(genre_df, 'inner')[['ID', 'Genre']]
        bucket_df.apply(bucket, axis=1).dropna().to_sql(
            'buckets',  conn, if_exists='replace', index=False
        )




if __name__ == "__main__":
    # pull_kworb(275000000)
    process_songs()
