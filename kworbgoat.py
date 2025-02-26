import pandas as pd

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
    df = pd.concat([df, pull[pull["Streams"] >= 300000000]])


df = df.drop_duplicates(["Artist and Title"]).sort_values("Streams", ascending=False)

df["Occurrence"] = df.groupby("Artist and Title").cumcount()
df[["Artist", "Title"]] = df["Artist and Title"].str.split(" - ", n=1, expand=True)

df.to_csv("t300k.csv", index=False)

df_artist = df["Artist"].drop_duplicates()
df_artist.to_csv("t300k_artists.csv", index=False)

print(df)
