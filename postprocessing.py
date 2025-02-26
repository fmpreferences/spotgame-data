import json

with open("genre_map.json") as genre_map:
    goat = json.load(genre_map)

poop = set()

for k, v in goat.items():
    if "edm" in v:
        poop.update(v)

print(poop)
