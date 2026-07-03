"""Diagnostico: encuentra predicciones incorrectas."""
from src.cli.wc2026_readme import predict_match, _load_data, _compute_as_of_from_csv
from src.data.injuries import load_injuries
from src.models.calibration import TemperatureScaler
import numpy as np
import pandas as pd

df, timeline, cache = _load_data()
calibrator = TemperatureScaler.load("data/processed/temperature_calibrator.json")
injuries = load_injuries()
as_of = _compute_as_of_from_csv(df)

wc = df[(df["date"] >= "2026-06-01") & (df["tournament"] == "FIFA World Cup")].copy()
wc = wc.dropna(subset=["home_goals", "away_goals"])

errores = []
aciertos_draw = 0
total_draw = 0
for _, m in wc.iterrows():
    pred = predict_match(df, timeline, cache, m["home_team"], m["away_team"],
                        str(m["date"])[:10], as_of=as_of, calibrator=calibrator, injuries=injuries)
    if pred.get("degraded"):
        continue
    hg, ag = int(m["home_goals"]), int(m["away_goals"])
    if hg > ag: outc = 0; actual = "H"
    elif hg < ag: outc = 2; actual = "A"
    else: outc = 1; actual = "D"; total_draw += 1
    pick = ["H","D","A"][np.argmax([pred["p_h"], pred["p_d"], pred["p_a"]])]
    if pick == actual and actual == "D":
        aciertos_draw += 1
    if pick != actual:
        errores.append({
            "date": str(m["date"])[:10],
            "match": f"{m['home_team']} {hg}-{ag} {m['away_team']}",
            "pred": f"H={pred['p_h']:.0%} D={pred['p_d']:.0%} A={pred['p_a']:.0%}",
            "pick": pick, "actual": actual,
        })

print(f"Total errores: {len(errores)} de {len(wc)}")
print(f"Draws totales: {total_draw}, draws acertados: {aciertos_draw}")
print()
for e in errores:
    print(f"  {e['date']} {e['match']:50s} | Pred: {e['pred']:30s} | Pick={e['pick']} Actual={e['actual']}")
