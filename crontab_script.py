from types import NoneType
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import sqlite3
import requests
import re
from dotenv import load_dotenv
from collections import defaultdict
import json
import numpy as np

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

WHITELIST_EDM = pd.read_csv("edm_whitelist")["aid"].unique()


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


def clean_junk_words(s):
    return re.sub(
        r" \(feat\. .*?\)| \(with .*?\)| - (\d+ )?Remaster(ed)?( \d+)?", "", s
    )


SAMPLE = 0
TOT = 0


def get_real_song(row, recheck=defaultdict(list), searchres=defaultdict(list)):
    global SAMPLE
    global TOT
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
        lst = sorted(lst)
        first = lst[rank]
        for a, b, c, d, e, f in lst:
            if clean_junk_words(c).lower() != clean_junk_words(first[2]).lower() or set(
                f
            ) != set(first[-1]):
                if abs(a - first[0]) <= 12:
                    SAMPLE += 1
                    recheck[first[-1][0]].append(
                        (name, rank, first[3], row["Streams"], len(first[-1]))
                    )  # need the chosen id for quick replace
                    searchres[name] = lst
                else:
                    TOT += 1
                break
        else:
            TOT += 1
        _, release_date, track_name, uri, artist_names, artist_uris = lst[rank]
    except IndexError:
        print(name, artist, rank)
        return None
    return_df = pd.DataFrame(
        [
            {
                "id": uri,
                "streams": row["Streams"],
                "title": track_name,
                "date": release_date,
                "artists": a,
                "artist_id": b,
            }
            for a, b in zip(artist_names, artist_uris)
        ]
    )
    return return_df


def process_artist_genres(artists_lst):
    dfs = []

    for i in range(0, len(artists_lst), 50):
        artists_sp = sp.artists(artists_lst[i : i + 50])
        for artist in artists_sp["artists"]:
            temp_genres = artist["genres"]
            if not temp_genres:
                if artist["id"] in WHITELIST_EDM:
                    temp_genres = ["edm"]
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


REQS = 0


def resolve_tracklist(tracks, step):
    global REQS
    df = pd.DataFrame()
    recheck = defaultdict(list)
    searchres = defaultdict(list)
    for i in range(0, len(tracks), step):
        while True:
            try:
                track_df = tracks[i : i + step]
                df = pd.concat(
                    [
                        df,
                        pd.concat(
                            track_df.apply(
                                lambda x: get_real_song(x, recheck, searchres), axis=1
                            ).tolist(),
                            axis=0,
                            ignore_index=True,
                        ),
                    ]
                )
            except requests.exceptions.ReadTimeout:
                print("rate limit")
                time.sleep(5)
                continue
            break
        break
    print(df)
    reconstructed = []
    for artist, tracklists in recheck.items():
        temp_df = pd.read_html(f"https://kworb.net/spotify/artist/{artist}_songs.html")[
            1
        ]
        REQS += 1
        for track in tracklists:
            title, rank, chosen, streams, no_artists = track
            valids = np.nonzero(temp_df["Song Title"].str.contains(title))
            valid_rank = valids[0][rank]
            row = temp_df.iloc[valid_rank]
            newtitle = re.sub(r"^\*", "", row["Song Title"])
            newdate, newid, newartists, newartistids = 0, "", [], []
            for _, b, c, d, e, f in searchres[title]:
                if c == newtitle:  # EXACT
                    newdate, newid, newartists, newartistids = b, d, e, f
            if newid == chosen:
                continue
            df = df.drop(np.nonzero(df["id"] == chosen)[0][:no_artists].tolist())
            reconstructed.append(
                pd.DataFrame(
                    [
                        {
                            "streams": streams,
                            "title": newtitle,
                            "artists": a,
                            "id": newid + "%MOD",
                            "artist_id": b,
                            "date": newdate,
                        }
                        for a, b in zip(newartists, newartistids)
                    ]
                )
            )
            df = pd.concat([df] + reconstructed).reset_index(drop=True)
    df = df.sort_values("streams", ascending=False).reset_index(drop=True)

    return df.drop_duplicates(["id", "artist_id"]).dropna()


def bucket(row):
    # only for edm rn idk what to do with others
    row["bucket"] = None
    for genre in BUCKET_MAP["edm"]:
        if genre in row["genre"]:
            row["bucket"] = "edm"
    return row.drop("genre")


def process_songs(df):
    with sqlite3.connect(DB_PATH) as conn:
        song_artist_df = df[["id", "artist_id"]].drop_duplicates()
        all_artists = song_artist_df["artist_id"].unique()
        genre_df = process_artist_genres(all_artists)
        artist_df = genre_df.drop("genre", axis=1).drop_duplicates()
        genre_df = genre_df.drop("artist_name", axis=1).dropna()

        df = df.drop(["artists", "artist_id"], axis=1).drop_duplicates()
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
    # pull_kworb(250000000)
    df = resolve_tracklist(pd.read_csv(KWORB_PATH), 85)
    df.to_csv("RECHECKTEST.csv")
    process_songs(df)
    print(SAMPLE, TOT, REQS)
