"""Lista las competitions de StatsBomb open data con matches disponibles."""
import requests

r = requests.get("https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json", timeout=10)
data = r.json()
print(f"Total: {len(data)} competitions\n")
for c in data:
    if c.get("match_available"):
        comp_id = c["competition_id"]
        season_id = c["season_id"]
        country = c.get("country_name", "")
        comp = c["competition_name"]
        season = c["season_name"]
        n_matches = len(c["match_available"])
        print(f"  {comp_id:3d}/{season_id:4d}  {country:15s}  {comp:25s}  {season:10s}  ({n_matches} matches)")
