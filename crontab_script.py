from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional
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

# order matters
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
    ],
    "latin": [
        "latin",
        "bolero",
        "reggaeton",
        "chilean",
        "salsa",
        "bachata",
        "merengue",
        "colombia",
        "mexican",
        "argentine",
        "norteño",
        "cuban",
        "mambo",
        "cumbia",
        "mariachi",
    ],
    "rock": ["rock", "punk", "metal"],
}

DB_PATH = "somewhereonmyvps.sqlite3"
CACHE_PATH = "searches_cache.sqlite3"
KWORB_PATH = "KWORB.csv"
KWORB_CACHE_PATH = 'kworb_cache'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

WHITELIST_EDM = pd.read_csv("edm_whitelist")["aid"].unique()


def create_dbs():
    with sqlite3.connect(CACHE_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE if not exists "cache" (
                "date"	TEXT,
                "title"	TEXT,
                "artist"	TEXT,
                "json_search"	TEXT,
                PRIMARY KEY("title","artist")
            )"""
        )
        cur.execute(
            '''CREATE TABLE "kworb_cache" (
                "date"	TEXT,
                "artist_id"	TEXT,
                PRIMARY KEY("artist_id")
            )'''
        )


def pull_kworb(threshold: int) -> None:
    """
    Pulls songs from all years above threshold.
    Saves in KWORB_PATH CSV file

    :param threshold: threshold to pull over
    """
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


def alnum_only(s):
    return re.sub("[^a-z0-9]", "", s.lower())


def clean_junk_words(s):
    return re.sub(
        r" \(feat\. .*?\)| \(with .*?\)| - (\d+ )?Remaster(ed)?( \d+)?", "", s
    )




def search_caches(title: str, artist: str, days: int) -> dict:
    """
    Checks CACHE_PATH if the given query has been searched from spotify within
    days d. If it has, return it. If not, call spotify api and save to cache for days d.

    :param title: title as displayed on kworb songs endpoint
    :param artist: artist as displayed on kworb songs endpoint
    :param days: days until value in cache is outdated
    :returns: the search result
    """
    s_title, s_artist = alnum_only(title), alnum_only(artist)
    with sqlite3.connect(CACHE_PATH) as conn:
        df = pd.read_sql(
            f"select * from cache where title='{s_title}' and artist='{s_artist}';",
            conn,
            parse_dates=True,
        )
        if not df.empty and np.all(
            df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
            < datetime.now()
        ):
            cur = conn.cursor()
            cur.execute(
                f"delete from cache where title='{s_title}' and artist='{s_artist}';"
            )
            cur.close()
        # after purging cache, reselect only updated
        df = pd.read_sql(
            f"select * from cache where title='{s_title}' and artist='{s_artist}';",
            conn,
            parse_dates=True,
        )
        if not df.empty:
            return json.loads(df["json_search"].unique().tolist()[0])
        reses = sp.search(f"{title} artist:{artist}", type="track")["tracks"]["items"]
        df = pd.DataFrame(
            [
                {
                    "title": s_title,
                    "artist": s_artist,
                    "date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
                    "json_search": json.dumps(reses),
                }
            ]
        )
        df.to_sql("cache", conn, if_exists="append", index=False)
        return reses

def kworb_caches(artist_id: str, days: int) -> Optional[pd.DataFrame] :
    """
    Checks CACHE_PATH if the given artist's page has been pulled from kworb in days d.
    If it has, return it. If not, pull the updated kworb page and save to cache for days d.

    :param artist_id: artist_id of the kworb page to pull
    :param days: days until value in cache is outdated
    :returns: the search result
    """
    with sqlite3.connect(CACHE_PATH) as conn:
        df = pd.read_sql(
            f"select * from cache where artist_id='{artist_id}';",
            conn,
            parse_dates=True,
        )
        if not df.empty and np.all(
            df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
            < datetime.now()
        ):
            cur = conn.cursor()
            cur.execute(
                f"delete from cache where artist_id='{artist_id}';"
            )
            cur.close()
        # after purging cache, reselect only updated
        df = pd.read_sql(
            f"select * from cache where artist_id='{artist_id}';",
            conn,
            parse_dates=True,
        )
        artist_endpoint = f"https://kworb.net/spotify/artist/{artist_id}_songs.html"
        artist_pth = Path(KWORB_CACHE_PATH, f'{artist_id}.csv')
        if not df.empty and artist_pth.exists():
            return pd.read_csv(artist_pth)
        try:
            temp_df = pd.read_html(artist_endpoint)[1]
            temp_df.to_csv(artist_pth)
            df = pd.DataFrame(
                [
                    {
                        "date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
                        "artist_id": artist_id
                    }
                ]
            )
            df.to_sql("cache", conn, if_exists="append", index=False)
            return temp_df
        except:
            print("Invalid Artist:")
            print(artist_id)
            return None


def bucket(row):
    # EDM first since i assume whatever has EDM on it is prolly edm
    # latin next to delete latin types of rock & pop
    row["bucket"] = None
    for broad, genre in BUCKET_MAP.items():
        for subgenre in genre:
            if subgenre in row["genre"]:
                row["bucket"] = broad
                return row.drop("genre")
    return row.drop("genre")


def result_from_spotify_search(title: str, artist: str, reses: dict) -> List[Any]:
    """
    Processes spotify search result reses and returns in a friendly format
    for determining most favored matches for the kworb title and artist

    :param title: title as displayed on kworb songs endpoint
    :param artist: artist as displayed on kworb songs endpoint
    :param reses: spotify api search result
    :returns: list of -pop, rdate, title, id, artist_list, artist_id_list
    sorted by best result first
    """
    lst = []
    for res in reses:
        aname = alnum_only(title)
        # order by popularity up and date down
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
    return sorted(lst)


def get_real_song(row, recheck=defaultdict(list), searchres=defaultdict(list)) -> pd.DataFrame:
    """
    Vectorized function which matches kworb to actual spotify song id. Returns a df which has
    the spotify id and a row for each individual artist, rdbms style

    :param row: row in DF
    :param recheck: dictionary for checking "sus" assignments
    :param searchres: outdated
    :returns: df which each row represents the id of the song and one artist on the song
    """
    title, artist, rank = row["title"], row["Artist"], row["Occurrence"]
    reses = search_caches(title, artist, 7)
    lst = result_from_spotify_search(title, artist, reses)

    # get all viable candidates from spotify search
    try:
        # filter candidates which might be too close
        first = lst[rank]
        for lpopularity, _, lname, _, lartists, lartistids in lst:
            # if the result with the closest popularity that is not the same exact song
            # (name/artists) is too close in popularity, add for processing
            # may break if there is any song with 3 editions under the threshold
            if clean_junk_words(lname).lower() == clean_junk_words(first[2]).lower():
                continue
            if set(lartistids) == set(first[-1]):
                continue
            # need the original cached search to get the "correct" one
            # other ones are just ease of access and or not accesible anymore
            if abs(lpopularity - first[0]) <= 12:
                for t_artist, t_artist_id in zip(lartists, lartistids):
                    if t_artist != artist:
                        continue
                    recheck[t_artist_id].append(
                        (title, rank, first[3], row["Streams"], artist, len(first[-1]))
                    )
                    break
                searchres[title] = lst
            break
        _, release_date, track_name, uri, artist_names, artist_uris = lst[rank]
    except IndexError:
        print(title, artist, rank)
        return None
    return_df = pd.DataFrame(
        [
            {
                "id": uri,
                "streams": row["Streams"],
                "title": track_name,
                "date": release_date,
                "artists": t_artist_name,
                "artist_id": t_artist_id,
            }
            for t_artist_name, t_artist_id in zip(artist_names, artist_uris)
        ]
    )
    return return_df


def process_artist_genres(artists_lst):
    """
    Takes a list of artists and associates them with the genres assigned to them
    by spotify. Returns df which associates artist IDs with their genres, rdbms style

    :param artists_lst: list of unique artists to process
    :returns: df of artist_name, artist_id, genre
    """
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
                [
                    {
                        "artist_name": artist["name"],
                        "artist_id": artist["id"],
                        "genre": genre,
                    }
                    for genre in temp_genres
                ]
            )
            dfs.append(df2)
    df = pd.concat(dfs)
    return df




def resolve_and_save_track_info(tracks: pd.DataFrame, step: int):
    """
    Takes a df of tracks from kworb and a step size (max how many to roll
    back when rate limited) and converts it to the associated track in the
    real spotify

    :param tracks: df of tracks pulled from kworb data
    :param step: how many tracks to process vectorizedly
    :returns: df of artist_name, artist_id, genre
    """
    df = pd.DataFrame()
    recheck = defaultdict(list)
    searchres = defaultdict(list)
    for i in range(0, len(tracks), step):
        # execution with "vectorized" search in increments of step
        # by way of df.apply
        while True:
            try:
                track_df = tracks[i : i + step]

                def get_real_song_dicts(x):
                    return get_real_song(x, recheck, searchres)

                # shitty jank shit that I had to do to even
                # coherently explode the artists
                exploded_df = pd.concat(
                    track_df.apply(get_real_song_dicts, axis=1).tolist(),
                    axis=0,
                    ignore_index=True,
                )
                df = pd.concat([df, exploded_df])
            except requests.exceptions.ReadTimeout:
                print("rate limit")
                time.sleep(5)
                continue
            break
        # comment this out to pull everything:
        # break

    # the sus stuff that was pulled in prev step is checked against kworb again
    # on the artists page instead
    df = df.reset_index(drop=True)
    reconstructed = []
    for artist, tracklists in recheck.items():
        for track in tracklists:
            # 1. we find the nth one that matches our query on the kworb page
            # 2. this one will have its exact title from kworb. we find the first thing from
            # our cached result that matches it
            temp_df = kworb_caches(artist, 3)
            if temp_df is None:
                continue
            title, rank, chosen, streams, og_artist, no_artists = track
            valids = np.nonzero(
                temp_df["Song Title"]
                .str.lower()
                .str.contains(title.lower(), regex=False)
            )
            try:
                valid_rank = valids[0][rank]
            except IndexError:
                print("IndexError at Reconstruction")
                print(title, rank, chosen, streams, no_artists)
                continue
            row = temp_df.iloc[valid_rank]
            newtitle = re.sub(r"^\*", "", row["Song Title"])
            newdate, newid, newartists, newartistids = 0, chosen, [], []
            reses = search_caches(title, og_artist, 7 )
            search_res = result_from_spotify_search(title, og_artist, reses)
            for _, s_date, s_title, s_id, s_artists, s_artist_id in search_res:
                if s_title.lower() == newtitle.lower() and artist in s_artist_id:
                    newdate, newid = s_date, s_id
                    newartists, newartistids = s_artists, s_artist_id
                    break
            if newid == chosen:
                continue
            # drop exactly the first amount of the song's id with its artists and renew it
            df = df.drop(np.nonzero(df["id"] == chosen)[0][:no_artists].tolist())
            reconstructed_df = pd.DataFrame(
                [
                    {
                        "streams": streams,
                        "title": newtitle,
                        "artists": t_artist,
                        "id": newid + "%MOD",
                        "artist_id": t_artist_id,
                        "date": newdate,
                    }
                    for t_artist, t_artist_id in zip(newartists, newartistids)
                ]
            )
            reconstructed.append(reconstructed_df)
            df = pd.concat([df] + reconstructed).reset_index(drop=True)
    df = df.sort_values("streams", ascending=False).reset_index(drop=True)
    df["id"] = df["id"].str.replace("%MOD", "")

    df = df.drop_duplicates(["id", "artist_id"]).dropna()

    with sqlite3.connect(DB_PATH) as conn:
        song_artist_df = df[["id", "artist_id"]]

        df = df.drop(["artists", "artist_id"], axis=1).drop_duplicates()
        df.to_sql("tracks", conn, if_exists="replace", index=False)
        song_artist_df.to_sql("track_artists", conn, if_exists="replace", index=False)


def process_genres():
    with sqlite3.connect(DB_PATH) as conn:
        track_artist_df = pd.read_sql("select * from track_artists;", conn)
        all_artists = track_artist_df["artist_id"].unique()
        genre_df = process_artist_genres(all_artists)
        artist_df = genre_df.drop("genre", axis=1).drop_duplicates()
        genre_df = genre_df.drop("artist_name", axis=1).dropna()
        genre_df.to_sql("genres", conn, if_exists="replace", index=False)
        artist_df.to_sql("artists", conn, if_exists="replace", index=False)


def process_buckets():
    with sqlite3.connect(DB_PATH) as conn:
        qry_stmt = (
            "select a.id, c.genre from tracks a join track_artists b on a.id = b.id"
            " join genres c on b.artist_id = c.artist_id;"
        )
        bucket_df = pd.read_sql(qry_stmt, conn)

        bucket_df.apply(bucket, axis=1).dropna().drop_duplicates().to_sql(
            "buckets", conn, if_exists="replace", index=False
        )


if __name__ == "__main__":
    # pull_kworb(250000000)
    # resolve_and_save_track_info(pd.read_csv(KWORB_PATH), 100)
    process_genres()
    process_buckets()
