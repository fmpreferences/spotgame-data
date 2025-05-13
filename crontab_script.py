import json
import re
import sqlite3
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import requests
import spotipy
from discogs import discogs_token
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials, urllibparse
import subprocess

load_dotenv()

# order matters
BUCKET_MAP = {
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
    "rnb": [
        "r&b",
        "rnb",
        "soul",
        "funk",
        "blues",
        "disco",
    ],
    "edm": [
        "edm",
        "house",
        "techno",
        "step",
        # "dance pop",
        "big room",
        " bass",
        "trance",
        "electro swing",
        "electronica",
        "phonk",
        "synthwave",
        "nu disco",
    ],
    "rock": ["rock", "punk", "metal"],
    "country": ["country", "bluegrass"],
    "pop": ["pop", "afrobeat", "ballad"],
    "hiphop": ["hip hop", "rap"],
}

DB_PATH = "somewhereonmyvps.sqlite3"
PROD_PATH = "prod.sqlite3"
CACHE_PATH = "searches_cache.sqlite3"
KWORB_PATH = "KWORB.csv"
KWORB_CACHE_PATH = "kworb_cache"

DISCOGS_PATH = "discogs_cache.sqlite3"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())


def create_dbs():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE "artists" (
                "artist_name"	TEXT,
                "artist_id"	TEXT,
                PRIMARY KEY("artist_id")
            )"""
        )
        cur.execute(
            """CREATE TABLE "tracks" (
                "id"	TEXT,
                "streams"	INTEGER,
                "title"	TEXT,
                "date"	INTEGER,
                "album_art"	TEXT,
                PRIMARY KEY("id")
            )
            """
        )
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
            """CREATE TABLE "kworb_cache" (
                "date"	TEXT,
                "artist_id"	TEXT,
                PRIMARY KEY("artist_id")
            )"""
        )
    with sqlite3.connect(DISCOGS_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE "cache" (
                "date"	TEXT,
                "artist"	TEXT,
                "latest_genres"	TEXT,
                "album"	TEXT,
                PRIMARY KEY("artist","album")
            )
            """
        )


def pull_kworb(threshold: int) -> None:
    """
    Pulls songs from all years above threshold.
    Saves in KWORB_PATH CSV file

    :param threshold: threshold to pull over
    """
    years = list(range(2010, datetime.now().year + 1)) + [
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
    """
    Removes flavor words that aren't relevant and clog up the button size
    """
    s = s.replace('"', "")
    s = re.sub(
        r" [\(\[][Ff]eat\. .*?[\)\]]| - [Ff]eaturing [a-z-A-Z 0-9]+| [\(\[]with .*?[\)\]]",
        "",
        s,
    )
    s = re.sub(
        r" - (\d+ )?Remaster(ed)?( \d+)?| [\(\[](\d+ )?Remaster(ed)?( \d+)?[\)\]]",
        "",
        s,
    )
    s = re.sub(r" \(.*? Vs\. .*?\)| - .*? vs [a-z-A-Z 0-9]+,", "", s)
    s = re.sub(r" [\[\(][Ff]rom .*?[\]\)]| - [Ff]rom .*?", "", s)
    return s


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
                    "date": (datetime.now() + timedelta(days=days)).strftime(
                        "%Y-%m-%d"
                    ),
                    "json_search": json.dumps(reses),
                }
            ]
        )
        df.to_sql("cache", conn, if_exists="append", index=False)
        return reses


def kworb_caches(artist_id: str, days: int) -> Optional[pd.DataFrame]:
    """
    Checks CACHE_PATH if the given artist's page has been pulled from kworb in days d.
    If it has, return it. If not, pull the updated kworb page and save to cache for days d.

    :param artist_id: artist_id of the kworb page to pull
    :param days: days until value in cache is outdated
    :returns: the search result
    """
    with sqlite3.connect(CACHE_PATH) as conn:
        df = pd.read_sql(
            f"select * from kworb_cache where artist_id='{artist_id}';",
            conn,
            parse_dates=True,
        )
        if not df.empty and np.all(
            df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
            < datetime.now()
        ):
            cur = conn.cursor()
            cur.execute(f"delete from kworb_cache where artist_id='{artist_id}';")
            cur.close()
        # after purging cache, reselect only updated
        df = pd.read_sql(
            f"select * from kworb_cache where artist_id='{artist_id}';",
            conn,
            parse_dates=True,
        )
        artist_endpoint = f"https://kworb.net/spotify/artist/{artist_id}_songs.html"
        artist_pth = Path(KWORB_CACHE_PATH, f"{artist_id}.csv")
        if not df.empty and artist_pth.exists():
            return pd.read_csv(artist_pth)
        try:
            temp_df = pd.read_html(artist_endpoint)[1]
            temp_df.to_csv(artist_pth)
            df = pd.DataFrame(
                [
                    {
                        "date": (datetime.now() + timedelta(days=days)).strftime(
                            "%Y-%m-%d"
                        ),
                        "artist_id": artist_id,
                    }
                ]
            )
            df.to_sql("kworb_cache", conn, if_exists="append", index=False)
            return temp_df
        except:
            print("Invalid Artist:")
            print(artist_id)
            return None


def process_albums(albums):
    """
    Takes a list of artists and associates them with the genres assigned to them
    by spotify. Returns df which associates artist IDs with their genres, rdbms style

    :param artists_lst: list of unique artists to process
    :returns: df of artist_name, artist_id, genre
    """
    dfs = []

    for i in range(0, len(albums), 20):
        albums_sp = sp.albums(albums[i : i + 20])
        for album in albums_sp["albums"]:
            df2 = pd.DataFrame(
                [
                    {
                        "artist_id": album["artists"][0]["id"],
                        "album_id": album["id"],
                        "album_name": album["name"],
                    }
                ]
            )
            dfs.append(df2)
    df = pd.concat(dfs)
    return df


def bucket(row):
    row["bucket"] = "other"
    for broad, genre in BUCKET_MAP.items():
        for subgenre in genre:
            if subgenre in row["genre"].lower():
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
    :returns: list of -pop, rdate, title, id, artist_list, artist_id_list,
    artist index, album cover
    sorted by best result first
    """
    lst = []
    for res in reses:
        aname = alnum_only(title)
        # order by popularity up and date down
        artist_names = [artist_item["name"] for artist_item in res["artists"]]
        if aname in alnum_only(res["name"]) and artist in artist_names:
            try:
                cover = res["album"]["images"][0]["url"]
            except IndexError:
                cover = "INVALID"
            rdate = None
            album_rdate = res["album"]["release_date"]
            if len(album_rdate) == 4:
                if album_rdate == "0000":
                    album_rdate = "0001"
                rdate = date(int(album_rdate), 12, 31)
            elif len(album_rdate) == 7:
                rdate = datetime.strptime(album_rdate, "%Y-%m")
                # shitty jank function to find last day of this month
                lastday = date(rdate.year, (rdate.month) % 12 + 1, 1) - timedelta(
                    days=1
                )
                rdate = date(
                    rdate.year,
                    rdate.month,
                    lastday.day,
                )
            else:
                rdate = datetime.strptime(album_rdate, "%Y-%m-%d").date()
            lst.append(
                (
                    -res["popularity"],
                    rdate,
                    res["name"],
                    res["id"],
                    [artist_item["name"] for artist_item in res["artists"]],
                    [artist_item["id"] for artist_item in res["artists"]],
                    list(range(len(res["artists"]))),
                    cover,
                    res["album"]["id"],
                )
            )
    return sorted(lst)


def get_real_song(row, recheck=defaultdict(list)) -> pd.DataFrame | None:
    """
    Vectorized function which matches kworb to actual spotify song id. Returns a df which has
    the spotify id and a row for each individual artist, rdbms style

    :param row: row in DF
    :param recheck: dictionary for checking "sus" assignments
    :param searchres: outdated
    :returns: df which each row represents the id of the song and one artist on the song,
    or None if IndexError
    """
    title, artist, rank = row["title"], row["Artist"], row["Occurrence"]
    reses = search_caches(title, artist, 6)
    lst = result_from_spotify_search(title, artist, reses)

    # get all viable candidates from spotify search
    try:
        # filter candidates which might be too close
        first = lst[rank]
        for lst_item in lst:
            # if the result with the closest popularity that is not the same exact song
            # (name/artists) is too close in popularity, add for processing
            # may break if there is any song with 3 editions under the threshold
            if (
                clean_junk_words(lst_item[2]).lower()
                == clean_junk_words(first[2]).lower()
            ):
                continue
            if set(lst_item[5]) == set(first[5]):
                continue

            # need the original cached search to get the "correct" one
            # other ones are just ease of access and or not accesible anymore
            if abs(lst_item[0] - first[0]) <= 15:
                for t_artist, t_artist_id in zip(lst_item[4], lst_item[5]):
                    if t_artist != artist:
                        continue
                    recheck[t_artist_id].append(
                        (title, rank, first[3], row["Streams"], artist, len(first[5]))
                    )
                    break
            break

        rel = lst[rank]
    except IndexError:
        print(title, artist, rank)
        return None
    return_df = pd.DataFrame(
        [
            {
                "id": rel[3],
                "streams": row["Streams"],
                "title": rel[2],
                "date": rel[1],
                "artists": t_artist_name,
                "artist_id": t_artist_id,
                "artist_index": t_artist_idx,
                "album_art": rel[7],
                "album_id": rel[8],
            }
            for t_artist_name, t_artist_id, t_artist_idx in zip(*rel[4:7])
        ]
    )
    return return_df


def search_discogs(artist, album, days, remove_parens):
    # album = legacy words, refers to regular title
    album = clean_junk_words(album)
    if remove_parens:
        album = re.sub(r" \(.*?\)| \[.*?\]| ?:.*?[Aa]nniversary.*?", "", album)
    artist = artist.replace('"', "")
    with sqlite3.connect(DISCOGS_PATH) as conn:
        df = pd.read_sql(
            f'select * from cache where artist="{artist}" and album="{album}";',
            conn,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
        )
        if not df.empty and np.all(df["date"] < datetime.now()):
            cur = conn.cursor()
            cur.execute(
                f'delete from cache where artist="{artist}" and album="{album}";'
            )
            cur.close()
        # after purging cache, reselect only updated
        df = pd.read_sql(
            f'select * from cache where artist="{artist}" and album="{album}";',
            conn,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
        )
        if df.empty:
            max_tries = 250
            for _ in range(max_tries):
                try:
                    album2, artist2 = urllibparse.quote(album), urllibparse.quote(
                        artist
                    )
                    r = requests.get(
                        f"https://api.discogs.com/database/search?q={album2}&type=release&artist={artist2}&token={discogs_token}"
                    )
                    chosen_res = None
                    print(
                        f"https://api.discogs.com/database/search?q={album2}&type=release&artist={artist2}&token={discogs_token}"
                    )
                    print(album, artist)
                    for res in r.json()["results"]:
                        res_title = res["title"].split(" - ")[-1]
                        res_title = clean_junk_words(res_title)
                        if remove_parens:
                            res_title = re.sub(
                                r" \(.*?\)| \[.*?\]| ?:.*?[Aa]nniversary.*?", "", album
                            )
                        if res_title.lower() == album.lower():
                            chosen_res = res
                            break
                    if chosen_res is None:
                        df = pd.DataFrame(
                            [
                                {
                                    "artist": artist,
                                    "date": (
                                        datetime.now() + timedelta(days=days)
                                    ).strftime("%Y-%m-%d"),
                                    "latest_genres": json.dumps([None]),
                                    "album": album,
                                }
                            ]
                        )
                        df.to_sql("cache", conn, if_exists="append", index=False)
                        return []
                    reses = chosen_res["style"]
                    if not reses:
                        reses = chosen_res["genre"]
                    if not reses:
                        reses = []
                    time.sleep(1.01)
                except KeyError:
                    time.sleep(2.11)
                    continue
                except json.JSONDecodeError:
                    print(f"{artist} Json Error!")
                    time.sleep(2.11)
                    return []
                except IndexError:
                    print(f"{artist} No result!")
                    time.sleep(1.01)
                    df = pd.DataFrame(
                        [
                            {
                                "artist": artist,
                                "date": (
                                    datetime.now() + timedelta(days=days)
                                ).strftime("%Y-%m-%d"),
                                "latest_genres": json.dumps([None]),
                                "album": album,
                            }
                        ]
                    )
                    df.to_sql("cache", conn, if_exists="append", index=False)
                    return []
                df = pd.DataFrame(
                    [
                        {
                            "artist": artist,
                            "date": (datetime.now() + timedelta(days=days)).strftime(
                                "%Y-%m-%d"
                            ),
                            "latest_genres": json.dumps(reses),
                            "album": album,
                        }
                    ]
                )
                try:
                    df.to_sql("cache", conn, if_exists="append", index=False)
                except sqlite3.IntegrityError:
                    print("artist & album failed integrity check!")
                return reses
        return json.loads(df["latest_genres"].tolist()[0])


def process_artists(artists_lst):
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
            genres = artist["genres"]
            # NEED THIS because artist wont show up in list otehrwise
            if not genres:
                genres = [None]
            df2 = pd.DataFrame(
                [
                    {
                        "artist_name": artist["name"],
                        "artist_id": artist["id"],
                        "genre": genre,
                    }
                    for genre in genres
                ]
            )
            dfs.append(df2)
        time.sleep(0.33)
    df = pd.concat(dfs)
    return df


def process_x_genres(artists_lst, x, xid, remove_parens=False):
    """
    Takes a list of artists and associates them with the genres assigned to them
    by spotify. Returns df which associates artist IDs with their genres, rdbms style

    :param artists_lst: list of unique artists to process
    :returns: df of artist_name, artist_id, genre
    """
    dfs = []
    for _, row in artists_lst.iterrows():
        reses = search_discogs(row["artist_name"], row[x], 69, remove_parens)
        dfs.append(pd.DataFrame([{xid: row[xid], "genre": genre} for genre in reses]))
    return pd.concat(dfs)


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
    for i in range(0, len(tracks), step):
        while True:
            try:
                track_df = tracks[i : i + step]

                def get_real_song_dicts(x):
                    return get_real_song(x, recheck)

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
                print("IndexError at reconstruction:")
                print(title, rank, chosen, streams, no_artists)
                continue
            row = temp_df.iloc[valid_rank]
            newtitle = re.sub(r"^\*", "", row["Song Title"])
            # date, title, id, artist_names, artist_ids, artist_indexes, albumart, album_id
            new_song_info = [date(1, 12, 31), newtitle, chosen, [], [], [], "", ""]

            reses = search_caches(title, og_artist, 7)
            search_res = result_from_spotify_search(title, og_artist, reses)
            for res in search_res:
                if res[2].lower() == newtitle.lower() and artist in res[5]:
                    new_song_info = res[1:]
                    break
            if new_song_info[2] == chosen:
                continue

            # drop exactly the first amount of the song's id with its artists and renew it
            df = df.drop(np.nonzero(df["id"] == chosen)[0][:no_artists].tolist())
            reconstructed_df = pd.DataFrame(
                [
                    {
                        "streams": streams,
                        "title": newtitle,
                        "artists": t_artist,
                        "id": f"{new_song_info[2]}%MOD",
                        "artist_id": t_artist_id,
                        "artist_index": t_artist_idx,
                        "date": new_song_info[0],
                        "album_art": new_song_info[6],
                        "album_id": new_song_info[7],
                    }
                    for t_artist, t_artist_id, t_artist_idx in zip(*new_song_info[3:6])
                ]
            )
            reconstructed.append(reconstructed_df)
            df = pd.concat([df] + reconstructed).reset_index(drop=True)
    df = df.sort_values("streams", ascending=False).reset_index(drop=True)
    df["id"] = df["id"].str.replace("%MOD", "")

    df = df.drop_duplicates(["id", "artist_id"]).dropna()
    df["title"] = df["title"].apply(clean_junk_words)

    with sqlite3.connect(DB_PATH) as conn:
        song_artist_df = df[["id", "artist_id", "artist_index"]]
        song_album_df = df[["id", "album_id"]]

        df = df.drop(
            ["artists", "artist_id", "artist_index", "album_id"], axis=1
        ).drop_duplicates()
        df.to_sql("tracks", conn, if_exists="replace", index=False)
        song_artist_df.to_sql("track_artists", conn, if_exists="replace", index=False)
        song_album_df.to_sql("track_albums", conn, if_exists="replace", index=False)


def process_spotify_stuff():
    """
    processes artist df and albums df which require spotify bulk api
    """
    with sqlite3.connect(DB_PATH) as conn:
        track_artist_df = pd.read_sql(
            "select distinct artist_id from track_artists;",
            conn,
        )
        artist_df = process_artists(track_artist_df["artist_id"])
        # id column first for auto script
        artist_df = artist_df.loc[:, ["artist_id", "artist_name", "genre"]]
        artist_df.drop("genre", axis=1).drop_duplicates().to_sql(
            "artists", conn, if_exists="replace", index=False
        )
        artist_df.drop("artist_name", axis=1).dropna().to_sql(
            "artists_genres", conn, if_exists="replace", index=False
        )
        track_album_df = pd.read_sql(
            "select distinct album_id from tracks a join track_albums b on a.id=b.id;",
            conn,
        )
        print(f"number of albums: {len(track_album_df['album_id'].unique())}")
        albums_df = process_albums(
            track_album_df["album_id"].unique()
        ).drop_duplicates()
        albums_df.to_sql("albums", conn, if_exists="replace", index=False)


def process_genres():
    with sqlite3.connect(DB_PATH) as conn:
        artist_album_df = pd.read_sql(
            "select distinct artist_name, album_id, album_name from artists a"
            " join albums b on a.artist_id = b.artist_id;",
            conn,
        )
        print(len(artist_album_df))
        album_genre_df = process_x_genres(
            artist_album_df, "album_name", "album_id", True
        )
        album_genre_df.dropna().to_sql(
            "album_genres", conn, if_exists="replace", index=False
        )

        artist_single_df = pd.read_sql(
            "select distinct title, artist_name, c.id from artists a join track_artists b on a.artist_id = b.artist_id"
            " join tracks c on b.id = c.id group  by c.id;",
            conn,
        )
        print(len(artist_single_df))
        single_genre_df = process_x_genres(artist_single_df, "title", "id")
        single_genre_df.dropna().to_sql(
            "track_genres", conn, if_exists="replace", index=False
        )


def process_buckets():
    with sqlite3.connect(DB_PATH) as conn, sqlite3.connect(PROD_PATH) as prod:
        new_artist_genre_df = pd.concat(
            [
                pd.read_sql(
                    "select a.artist_id, genre from artists a join track_artists b on a.artist_id = b.artist_id join track_genres c on b.id = c.id;",
                    conn,
                ),
                pd.read_sql(
                    "select a.artist_id, genre from artists a join albums b on a.artist_id = b.artist_id join album_genres c on b.album_id = c.album_id",
                    conn,
                ),
                pd.read_sql(
                    "select artist_id, genre from artists_genres;",
                    conn,
                ),
            ]
        ).apply(bucket, axis=1)

        new_artist_genre_df = new_artist_genre_df[
            new_artist_genre_df["bucket"] != "other"
        ]

        df_genres = (
            new_artist_genre_df.groupby(["artist_id", "bucket"]).size().reset_index()
        )
        df_artists = new_artist_genre_df.groupby(["artist_id"]).size().reset_index()

        df = df_genres.merge(df_artists, "left", ["artist_id"])
        df["Prop"] = df["0_x"] / df["0_y"]

        df_small = df[df["0_y"] <= 8]
        df_big = df[df["0_y"] > 8]

        df = pd.concat(
            [df_small[df_small["Prop"] >= 0.5], df_big[df_big["Prop"] > 0.3333]]
        )
        track_artists_df = pd.read_sql(
            "select * from track_artists;",
            conn,
        )
        df = df.merge(track_artists_df, "right", "artist_id")
        bucket_df = df[["id", "bucket"]].fillna("other").drop_duplicates()
        bucket_df.to_sql("buckets", conn, if_exists="replace", index=False)
        track_artists_df.to_sql("track_artists", prod, if_exists="replace", index=False)
        pd.read_sql(
            "select * from tracks;",
            conn,
        ).to_sql("tracks", prod, if_exists="replace", index=False)
        pd.read_sql(
            "select * from artists;",
            conn,
        ).to_sql("artists", prod, if_exists="replace", index=False)
        bucket_df.to_sql("buckets", prod, if_exists="replace", index=False)


def crontab_commit():
    if Path("spotgame.sql").exists():
        Path("spotgame.sql").rename("spotgamebak.sql")
    subprocess.run("sqlite3 prod.sqlite3 .dump > spotgame.sql", shell=True)
    with open("spotgame.sql") as s1, open("spotgamebak.sql") as s2, open(
        "spotdiff.sql", "w"
    ) as sd:
        l1, l2 = s1.readlines(), s2.readlines()
        removal = set(l2) - set(l1)
        addition = set(l1) - set(l2)
        removal = [remove for remove in removal if "INSERT" in remove]
        # print(removal)
        deletes = []
        for removing in removal:
            table = removing.split(" ", 3)[2]
            # print(table)
            if table in ("tracks", "buckets"):
                deletes.append(
                    f"delete from {table} where id={removing.split('(', 1)[1].split(',')[0]};\n"
                )
        ex = f"""
PRAGMA foreign_keys=OFF;
{''.join(deletes)}
{''.join(addition)}
        """
        sd.write(ex)


if __name__ == "__main__":
    pull_kworb(250000000)
    resolve_and_save_track_info(pd.read_csv(KWORB_PATH), 100)
    process_spotify_stuff()
    process_genres()
    process_buckets()
    crontab_commit()
