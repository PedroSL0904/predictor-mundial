"""Compara metricas predictor-mundial vs Oloraculo en partidos FT del WC 2026."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\dev\predictor-mundial")

df = pd.read_csv(r"C:\dev\predictor-mundial\compare_wc2026.csv")
ft = df[df.status == "FT"].copy()


def outcome_from_score(s: str) -> str:
    a, b = s.split("-")
    a, b = int(a), int(b)
    if a > b:
        return "H"
    if a < b:
        return "A"
    return "D"


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score para 3 outcomes."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), outcomes] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.log(np.maximum(probs[np.arange(len(probs)), outcomes], eps)).mean())


def sign_accuracy(probs: np.ndarray) -> tuple[int, int]:
    """Retorna (correct, total) contando si el top pick es el resultado real."""
    picks = np.argmax(probs, axis=1)  # 0=H, 1=D, 2=A
    return int((picks == ft_outcomes_int).sum()), len(picks)


ft["outcome"] = ft["result"].apply(outcome_from_score)
ft_outcomes_int = ft["outcome"].map({"H": 0, "D": 1, "A": 2}).values

olo_probs = ft[["olo_ph", "olo_pd", "olo_pa"]].fillna(1/3).values
my_probs = ft[["my_ph", "my_pd", "my_pa"]].fillna(1/3).values

# Filtrar partidos donde Oloraculo tiene probs (no "unavailable")
olo_valid = ft[ft.olo_ph.notna()]
olo_outcomes = olo_valid["outcome"].map({"H": 0, "D": 1, "A": 2}).values
olo_probs_valid = olo_valid[["olo_ph", "olo_pd", "olo_pa"]].values
my_probs_valid = olo_valid[["my_ph", "my_pd", "my_pa"]].values

print(f"Partidos FT con prediccion de Oloraculo: {len(olo_valid)}")
print()

olo_brier = brier(olo_probs_valid, olo_outcomes)
my_brier = brier(my_probs_valid, olo_outcomes)
olo_ll = log_loss(olo_probs_valid, olo_outcomes)
my_ll = log_loss(my_probs_valid, olo_outcomes)

olo_picks = np.argmax(olo_probs_valid, axis=1)
my_picks = np.argmax(my_probs_valid, axis=1)
olo_acc = (olo_picks == olo_outcomes).mean()
my_acc = (my_picks == olo_outcomes).mean()

print("=" * 60)
print("METRICAS EN PARTIDOS FT DEL WC 2026")
print("=" * 60)
print(f"{'Metrica':<20} {'Oloraculo':<15} {'Predictor-Mundial':<15}")
print("-" * 60)
print(f"{'Brier (lower=better)':<20} {olo_brier:<15.4f} {my_brier:<15.4f}")
print(f"{'Log loss (lower=better)':<20} {olo_ll:<15.4f} {my_ll:<15.4f}")
print(f"{'Sign accuracy':<20} {olo_acc:<15.1%} {my_acc:<15.1%}")
print()

# Detalle por partido
print("=" * 60)
print("DETALLE POR PARTIDO")
print("=" * 60)
print(f"{'Home':<20} {'Away':<20} {'Result':<8} {'Olo pick':<8} {'My pick':<8} {'Olo top%':<10} {'My top%':<10}")
print("-" * 90)
for _, r in olo_valid.iterrows():
    actual = r["outcome"]
    olo_top = "H" if r["olo_ph"] == max(r["olo_ph"], r["olo_pd"], r["olo_pa"]) else ("D" if r["olo_pd"] == max(r["olo_ph"], r["olo_pd"], r["olo_pa"]) else "A")
    my_top = "H" if r["my_ph"] == max(r["my_ph"], r["my_pd"], r["my_pa"]) else ("D" if r["my_pd"] == max(r["my_ph"], r["my_pd"], r["my_pa"]) else "A")
    olo_pct = max(r["olo_ph"], r["olo_pd"], r["olo_pa"])
    my_pct = max(r["my_ph"], r["my_pd"], r["my_pa"])
    mark_olo = "OK" if olo_top == actual else "X"
    mark_my = "OK" if my_top == actual else "X"
    print(f"{r['home']:<20} {r['away']:<20} {actual:<8} {olo_top+'('+mark_olo+')':<8} {my_top+'('+mark_my+')':<8} {olo_pct:<10.1%} {my_pct:<10.1%}")

# Aciertos de signo
print()
print(f"Oloraculo aciertos: {(olo_picks == olo_outcomes).sum()}/{len(olo_outcomes)}")
print(f"Predictor-Mundial aciertos: {(my_picks == olo_outcomes).sum()}/{len(olo_outcomes)}")
