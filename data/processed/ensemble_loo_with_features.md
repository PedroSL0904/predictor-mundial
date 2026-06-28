# Ensemble LOO: con vs sin features historicas

## Sin features (Sprint A4 original)
- Pesos optimos: P=0.00, BP=0.00, S=1.00
- Brier promedio LOO: 0.5817

## Con features (Sprint A4b)
- Pesos optimos: P=0.00, BP=1.00, S=0.00
- Brier promedio LOO: 0.5815

## Mejora
- Delta brier: -0.0002
- Mejora porcentual: +0.03%

## Conclusion
Las features mejoran el ensemble LOO.

Recomendacion: usar pesos = [0.00, 1.00, 0.00]
