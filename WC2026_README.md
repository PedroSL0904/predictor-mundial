# Predictor Mundial 2026

Predicciones para el Mundial 2026 generadas con el modelo Poisson + Dixon-Coles,
ponderado por Elo rolling, con ajustes de recent form, draw boost y elo gap inflation.

_Generado 2026-06-21 15:48 UTC._

**72 partidos de fase de grupos** | 36 jugados | 36 pendientes

## Métricas en partidos jugados

Partidos evaluados: **36**

| Métrica | Valor |
|---|---|
| Brier score (1X2) | **0.5757** |
| Log loss | **0.9698** |
| Sign accuracy | **58.3%** |
| Exact score accuracy | **16.7%** |

## Probabilidades del torneo (Monte Carlo)

Simulacion: 1000 corridas del torneo completo (fase de grupos + R32 + R16 + QF + SF + Final). Respetando los 36 partidos ya jugados. Tiempo: 11.3s.

Top 16 por probabilidad de campeon:

| Equipo | Campeon | Final | SF | QF | R16 | R32 |
|---|---:|---:|---:|---:|---:|---:|
| Argentina | 27.6% | 39.7% | 54.8% | 74.7% | 89.1% | 100.0% |
| Morocco | 13.1% | 21.6% | 36.0% | 53.3% | 68.0% | 100.0% |
| Belgium | 10.0% | 20.4% | 34.0% | 62.6% | 76.4% | 100.0% |
| Spain | 8.3% | 15.3% | 27.8% | 45.5% | 64.9% | 100.0% |
| England | 5.4% | 10.6% | 23.5% | 42.9% | 70.5% | 100.0% |
| Portugal | 4.8% | 12.2% | 24.4% | 47.0% | 74.0% | 100.0% |
| France | 3.9% | 7.7% | 18.7% | 39.7% | 77.2% | 100.0% |
| Germany | 3.9% | 8.4% | 19.4% | 41.9% | 73.1% | 100.0% |
| Japan | 3.2% | 6.6% | 15.4% | 28.4% | 54.0% | 100.0% |
| Colombia | 3.0% | 7.5% | 16.4% | 35.6% | 70.1% | 100.0% |
| Brazil | 2.5% | 7.0% | 14.8% | 27.2% | 46.0% | 100.0% |
| Algeria | 2.5% | 5.7% | 11.5% | 22.6% | 36.2% | 76.1% |
| Ivory Coast | 2.1% | 4.9% | 14.2% | 29.9% | 63.4% | 100.0% |
| Netherlands | 2.0% | 5.0% | 11.3% | 21.6% | 32.0% | 100.0% |
| Austria | 1.8% | 4.5% | 8.7% | 23.9% | 41.3% | 100.0% |
| Bosnia and Herzegovina | 1.3% | 2.1% | 4.8% | 9.5% | 17.7% | 34.8% |

## Llave completa de eliminatorias (500 simulaciones MC)

Para cada partido, mostramos la probabilidad de que cada equipo juegue (**Juega**) y gane (**Gana**) ese partido, agregado sobre 500 simulaciones.

*Nota: para partidos de R32 con un slot de 3rd (ej. 'Winner E vs 3rd {A,B,C,D,F}'), la columna 'Juega' puede no atribuirse al 3rd correcto por ambiguedad del bracket, pero la columna 'Gana' es exacta.*

### R32

**Partido 73** (Runner A vs Runner B)

| Equipo | Juega | Gana |
|---|---:|---:|
| South Korea | 67.2% | 30.4% |
| Switzerland | 54.2% | 29.4% |
| Canada | 45.8% | 26.4% |
| South Africa | 32.8% | 13.8% |

**Partido 74** (Winner E vs 3rd {A,B,C,D,F})

| Equipo | Juega | Gana |
|---|---:|---:|
| Germany | 100.0% | 72.2% |
| Bosnia and Herzegovina | 0.2% | 12.2% |
| South Korea | 0.2% | 9.8% |
| Scotland | 0.2% | 4.2% |
| Qatar | 0.2% | 1.6% |

**Partido 75** (Winner F vs Runner C)

| Equipo | Juega | Gana |
|---|---:|---:|
| Morocco | 100.0% | 67.8% |
| Netherlands | 100.0% | 32.2% |

**Partido 76** (Winner C vs Runner F)

| Equipo | Juega | Gana |
|---|---:|---:|
| Japan | 100.0% | 52.2% |
| Brazil | 100.0% | 47.8% |

**Partido 77** (Winner I vs 3rd {C,D,F,G,H})

| Equipo | Juega | Gana |
|---|---:|---:|
| France | 100.0% | 76.8% |
| Scotland | 0.2% | 8.2% |
| Paraguay | 0.2% | 7.4% |
| Australia | 0.2% | 5.2% |
| Turkey | 0.2% | 1.6% |
| Sweden | 0.2% | 0.8% |

**Partido 78** (Runner E vs Runner I)

| Equipo | Juega | Gana |
|---|---:|---:|
| Ivory Coast | 100.0% | 61.6% |
| Norway | 72.2% | 29.4% |
| Senegal | 27.8% | 9.0% |

**Partido 79** (Winner A vs 3rd {C,E,F,H,I})

| Equipo | Juega | Gana |
|---|---:|---:|
| Mexico | 100.0% | 69.2% |
| Scotland | 0.2% | 12.4% |
| Sweden | 0.2% | 9.2% |
| Ecuador | 0.2% | 9.0% |
| Saudi Arabia | 0.2% | 0.2% |

**Partido 80** (Winner L vs 3rd {E,H,I,J,K})

| Equipo | Juega | Gana |
|---|---:|---:|
| England | 100.0% | 70.2% |
| Portugal | 0.2% | 11.6% |
| Colombia | 0.2% | 6.6% |
| Congo DR | 0.2% | 4.2% |
| Uzbekistan | 0.2% | 4.0% |
| Senegal | 0.2% | 1.4% |
| Saudi Arabia | 0.2% | 0.8% |
| Ecuador | 0.2% | 0.6% |
| Uruguay | 0.2% | 0.2% |
| Cape Verde | 0.2% | 0.2% |
| Norway | 0.2% | 0.2% |

**Partido 81** (Winner D vs 3rd {B,E,F,I,J})

| Equipo | Juega | Gana |
|---|---:|---:|
| USA | 100.0% | 55.6% |
| Sweden | 0.2% | 18.4% |
| Senegal | 0.2% | 8.4% |
| Bosnia and Herzegovina | 0.2% | 5.4% |
| Norway | 0.2% | 4.4% |
| Ecuador | 0.2% | 2.6% |
| Qatar | 0.2% | 2.2% |
| Austria | 0.2% | 1.8% |
| Iraq | 0.2% | 1.0% |
| Algeria | 0.2% | 0.2% |

**Partido 82** (Winner G vs 3rd {A,E,H,I,J})

| Equipo | Juega | Gana |
|---|---:|---:|
| Belgium | 100.0% | 75.2% |
| Austria | 0.2% | 6.0% |
| Algeria | 0.2% | 5.2% |
| Senegal | 0.2% | 4.0% |
| Norway | 0.2% | 3.0% |
| Saudi Arabia | 0.2% | 2.8% |
| Cape Verde | 0.2% | 1.8% |
| Uruguay | 0.2% | 0.8% |
| South Korea | 0.2% | 0.6% |
| Ecuador | 0.2% | 0.4% |
| Iraq | 0.2% | 0.2% |

**Partido 83** (Runner K vs Runner L)

| Equipo | Juega | Gana |
|---|---:|---:|
| Croatia | 100.0% | 35.4% |
| Colombia | 39.0% | 27.4% |
| Portugal | 35.0% | 25.8% |
| Congo DR | 26.0% | 11.4% |

**Partido 84** (Winner H vs Runner J)

| Equipo | Juega | Gana |
|---|---:|---:|
| Spain | 100.0% | 65.6% |
| Austria | 56.6% | 17.0% |
| Algeria | 43.4% | 17.4% |

**Partido 85** (Winner B vs 3rd {E,F,G,I,J})

| Equipo | Juega | Gana |
|---|---:|---:|
| Canada | 54.2% | 21.8% |
| Switzerland | 45.8% | 22.8% |
| Egypt | 0.2% | 19.2% |
| Austria | 0.2% | 11.6% |
| Algeria | 0.2% | 8.8% |
| IR Iran | 0.2% | 3.8% |
| Sweden | 0.2% | 3.6% |
| Norway | 0.2% | 3.0% |
| Senegal | 0.2% | 2.6% |
| New Zealand | 0.2% | 2.0% |
| Iraq | 0.2% | 0.8% |

**Partido 86** (Winner J vs Runner H)

| Equipo | Juega | Gana |
|---|---:|---:|
| Argentina | 100.0% | 90.4% |
| Uruguay | 52.6% | 8.4% |
| Cape Verde | 38.0% | 0.8% |
| Saudi Arabia | 9.4% | 0.4% |

**Partido 87** (Winner K vs 3rd {D,E,I,J,L})

| Equipo | Juega | Gana |
|---|---:|---:|
| Colombia | 44.6% | 37.2% |
| Portugal | 43.2% | 36.4% |
| Congo DR | 12.2% | 5.6% |
| Ghana | 0.2% | 5.6% |
| Austria | 0.2% | 4.4% |
| Algeria | 0.2% | 3.4% |
| Paraguay | 0.2% | 2.6% |
| Australia | 0.2% | 2.0% |
| Turkey | 0.2% | 1.6% |
| Ecuador | 0.2% | 0.4% |
| Norway | 0.2% | 0.4% |
| Senegal | 0.2% | 0.4% |

**Partido 88** (Runner D vs Runner G)

| Equipo | Juega | Gana |
|---|---:|---:|
| Australia | 63.0% | 22.6% |
| Egypt | 49.2% | 27.8% |
| IR Iran | 47.2% | 36.0% |
| Paraguay | 37.0% | 12.6% |
| New Zealand | 3.6% | 1.0% |

### R16

**Partido 89** (Wo74 vs Wo77)

| Equipo | Juega | Gana |
|---|---:|---:|
| France | 76.8% | 42.8% |
| Germany | 72.2% | 40.4% |
| Scotland | 12.4% | 2.8% |
| Bosnia and Herzegovina | 12.2% | 6.4% |
| South Korea | 9.8% | 3.4% |
| Paraguay | 7.4% | 2.0% |
| Australia | 5.2% | 1.0% |
| Turkey | 1.6% | 0.6% |
| Qatar | 1.6% | 0.2% |
| Sweden | 0.8% | 0.4% |

**Partido 90** (Wo73 vs Wo75)

| Equipo | Juega | Gana |
|---|---:|---:|
| Morocco | 67.8% | 53.2% |
| Netherlands | 32.2% | 21.8% |
| South Korea | 30.4% | 6.6% |
| Switzerland | 29.4% | 10.0% |
| Canada | 26.4% | 7.0% |
| South Africa | 13.8% | 1.4% |

**Partido 91** (Wo76 vs Wo78)

| Equipo | Juega | Gana |
|---|---:|---:|
| Ivory Coast | 61.6% | 26.8% |
| Japan | 52.2% | 29.8% |
| Brazil | 47.8% | 30.4% |
| Norway | 29.4% | 10.4% |
| Senegal | 9.0% | 2.6% |

**Partido 92** (Wo79 vs Wo80)

| Equipo | Juega | Gana |
|---|---:|---:|
| England | 70.2% | 42.0% |
| Mexico | 69.2% | 33.0% |
| Scotland | 12.4% | 2.8% |
| Portugal | 11.6% | 7.8% |
| Ecuador | 9.6% | 4.2% |
| Sweden | 9.2% | 2.0% |
| Colombia | 6.6% | 3.2% |
| Congo DR | 4.2% | 2.2% |
| Uzbekistan | 4.0% | 1.6% |
| Senegal | 1.4% | 0.6% |
| Saudi Arabia | 1.0% | 0.2% |
| Uruguay | 0.2% | 0.2% |
| Norway | 0.2% | 0.2% |
| Cape Verde | 0.2% | 0.0% |

**Partido 93** (Wo83 vs Wo84)

| Equipo | Juega | Gana |
|---|---:|---:|
| Spain | 65.6% | 48.0% |
| Croatia | 35.4% | 7.6% |
| Colombia | 27.4% | 9.8% |
| Portugal | 25.8% | 12.0% |
| Algeria | 17.4% | 11.4% |
| Austria | 17.0% | 9.4% |
| Congo DR | 11.4% | 1.8% |

**Partido 94** (Wo81 vs Wo82)

| Equipo | Juega | Gana |
|---|---:|---:|
| Belgium | 75.2% | 62.0% |
| USA | 55.6% | 11.8% |
| Sweden | 18.4% | 3.6% |
| Senegal | 12.4% | 4.4% |
| Austria | 7.8% | 6.0% |
| Norway | 7.4% | 4.0% |
| Algeria | 5.4% | 3.2% |
| Bosnia and Herzegovina | 5.4% | 2.6% |
| Ecuador | 3.0% | 0.6% |
| Saudi Arabia | 2.8% | 0.6% |
| Qatar | 2.2% | 0.0% |
| Cape Verde | 1.8% | 0.4% |
| Iraq | 1.2% | 0.2% |
| Uruguay | 0.8% | 0.4% |
| South Korea | 0.6% | 0.2% |

**Partido 95** (Wo86 vs Wo88)

| Equipo | Juega | Gana |
|---|---:|---:|
| Argentina | 90.4% | 74.8% |
| IR Iran | 36.0% | 11.4% |
| Egypt | 27.8% | 6.2% |
| Australia | 22.6% | 2.8% |
| Paraguay | 12.6% | 0.4% |
| Uruguay | 8.4% | 4.2% |
| New Zealand | 1.0% | 0.0% |
| Cape Verde | 0.8% | 0.0% |
| Saudi Arabia | 0.4% | 0.2% |

**Partido 96** (Wo85 vs Wo87)

| Equipo | Juega | Gana |
|---|---:|---:|
| Colombia | 37.2% | 24.0% |
| Portugal | 36.4% | 25.8% |
| Switzerland | 22.8% | 8.2% |
| Canada | 21.8% | 6.4% |
| Egypt | 19.2% | 5.8% |
| Austria | 16.0% | 9.8% |
| Algeria | 12.2% | 7.4% |
| Congo DR | 5.6% | 2.8% |
| Ghana | 5.6% | 0.8% |
| IR Iran | 3.8% | 2.0% |
| Sweden | 3.6% | 1.2% |
| Norway | 3.4% | 1.2% |
| Senegal | 3.0% | 0.8% |
| Paraguay | 2.6% | 1.0% |
| Australia | 2.0% | 1.0% |
| New Zealand | 2.0% | 0.4% |
| Turkey | 1.6% | 0.8% |
| Iraq | 0.8% | 0.2% |
| Ecuador | 0.4% | 0.4% |

### QF

**Partido 97** (Wo89 vs Wo90)

| Equipo | Juega | Gana |
|---|---:|---:|
| Morocco | 53.2% | 36.8% |
| France | 42.8% | 20.4% |
| Germany | 40.4% | 17.0% |
| Netherlands | 21.8% | 11.8% |
| Switzerland | 10.0% | 3.4% |
| South Korea | 10.0% | 3.0% |
| Canada | 7.0% | 1.8% |
| Bosnia and Herzegovina | 6.4% | 3.0% |
| Scotland | 2.8% | 0.6% |
| Paraguay | 2.0% | 0.8% |
| South Africa | 1.4% | 0.8% |
| Australia | 1.0% | 0.6% |
| Turkey | 0.6% | 0.0% |
| Sweden | 0.4% | 0.0% |
| Qatar | 0.2% | 0.0% |

**Partido 98** (Wo93 vs Wo94)

| Equipo | Juega | Gana |
|---|---:|---:|
| Belgium | 62.0% | 33.8% |
| Spain | 48.0% | 30.4% |
| Austria | 15.4% | 6.2% |
| Algeria | 14.6% | 8.0% |
| Portugal | 12.0% | 6.8% |
| USA | 11.8% | 2.2% |
| Colombia | 9.8% | 4.4% |
| Croatia | 7.6% | 2.6% |
| Senegal | 4.4% | 1.0% |
| Norway | 4.0% | 0.8% |
| Sweden | 3.6% | 0.4% |
| Bosnia and Herzegovina | 2.6% | 1.6% |
| Congo DR | 1.8% | 0.8% |
| Ecuador | 0.6% | 0.4% |
| Saudi Arabia | 0.6% | 0.0% |
| Cape Verde | 0.4% | 0.2% |
| Uruguay | 0.4% | 0.0% |
| Iraq | 0.2% | 0.2% |
| South Korea | 0.2% | 0.2% |

**Partido 99** (Wo91 vs Wo92)

| Equipo | Juega | Gana |
|---|---:|---:|
| England | 42.0% | 23.0% |
| Mexico | 33.0% | 16.4% |
| Brazil | 30.4% | 16.4% |
| Japan | 29.8% | 16.2% |
| Ivory Coast | 26.8% | 12.4% |
| Norway | 10.6% | 3.4% |
| Portugal | 7.8% | 5.0% |
| Ecuador | 4.2% | 2.2% |
| Colombia | 3.2% | 2.0% |
| Senegal | 3.2% | 0.8% |
| Scotland | 2.8% | 0.6% |
| Congo DR | 2.2% | 0.6% |
| Sweden | 2.0% | 0.6% |
| Uzbekistan | 1.6% | 0.2% |
| Uruguay | 0.2% | 0.2% |
| Saudi Arabia | 0.2% | 0.0% |

**Partido 100** (Wo95 vs Wo96)

| Equipo | Juega | Gana |
|---|---:|---:|
| Argentina | 74.8% | 54.2% |
| Portugal | 25.8% | 12.4% |
| Colombia | 24.0% | 9.2% |
| IR Iran | 13.4% | 5.6% |
| Egypt | 12.0% | 3.8% |
| Austria | 9.8% | 3.4% |
| Switzerland | 8.2% | 1.8% |
| Algeria | 7.4% | 3.0% |
| Canada | 6.4% | 1.4% |
| Uruguay | 4.2% | 1.6% |
| Australia | 3.8% | 0.8% |
| Congo DR | 2.8% | 1.2% |
| Paraguay | 1.4% | 0.4% |
| Norway | 1.2% | 0.4% |
| Sweden | 1.2% | 0.2% |
| Senegal | 0.8% | 0.4% |
| Turkey | 0.8% | 0.2% |
| Ghana | 0.8% | 0.0% |
| Ecuador | 0.4% | 0.0% |
| New Zealand | 0.4% | 0.0% |
| Saudi Arabia | 0.2% | 0.0% |
| Iraq | 0.2% | 0.0% |

### SF

**Partido 101** (Wo97 vs Wo98)

| Equipo | Juega | Gana |
|---|---:|---:|
| Morocco | 36.8% | 21.8% |
| Belgium | 33.8% | 20.2% |
| Spain | 30.4% | 16.6% |
| France | 20.4% | 7.0% |
| Germany | 17.0% | 6.6% |
| Netherlands | 11.8% | 5.0% |
| Algeria | 8.0% | 4.6% |
| Portugal | 6.8% | 3.8% |
| Austria | 6.2% | 3.4% |
| Bosnia and Herzegovina | 4.6% | 2.4% |
| Colombia | 4.4% | 2.4% |
| Switzerland | 3.4% | 1.4% |
| South Korea | 3.2% | 0.2% |
| Croatia | 2.6% | 1.0% |
| USA | 2.2% | 0.8% |
| Canada | 1.8% | 0.6% |
| Senegal | 1.0% | 0.2% |
| Norway | 0.8% | 0.4% |
| Congo DR | 0.8% | 0.4% |
| Paraguay | 0.8% | 0.4% |
| South Africa | 0.8% | 0.0% |
| Scotland | 0.6% | 0.2% |
| Australia | 0.6% | 0.0% |
| Ecuador | 0.4% | 0.2% |
| Sweden | 0.4% | 0.0% |
| Cape Verde | 0.2% | 0.2% |
| Iraq | 0.2% | 0.2% |

**Partido 102** (Wo99 vs Wo100)

| Equipo | Juega | Gana |
|---|---:|---:|
| Argentina | 54.2% | 38.0% |
| England | 23.0% | 11.2% |
| Portugal | 17.4% | 7.2% |
| Brazil | 16.4% | 8.6% |
| Mexico | 16.4% | 5.6% |
| Japan | 16.2% | 8.8% |
| Ivory Coast | 12.4% | 5.2% |
| Colombia | 11.2% | 5.2% |
| IR Iran | 5.6% | 3.0% |
| Egypt | 3.8% | 1.2% |
| Norway | 3.8% | 0.2% |
| Austria | 3.4% | 2.0% |
| Algeria | 3.0% | 1.0% |
| Ecuador | 2.2% | 1.2% |
| Switzerland | 1.8% | 0.6% |
| Uruguay | 1.8% | 0.2% |
| Congo DR | 1.8% | 0.0% |
| Canada | 1.4% | 0.4% |
| Senegal | 1.2% | 0.0% |
| Australia | 0.8% | 0.2% |
| Sweden | 0.8% | 0.0% |
| Scotland | 0.6% | 0.0% |
| Paraguay | 0.4% | 0.2% |
| Turkey | 0.2% | 0.0% |
| Uzbekistan | 0.2% | 0.0% |

### Final

**Partido 104** (Wo101 vs Wo102)

| Equipo | Juega | Gana |
|---|---:|---:|
| Argentina | 38.0% | 27.2% |
| Morocco | 21.8% | 13.8% |
| Belgium | 20.2% | 9.4% |
| Spain | 16.6% | 8.6% |
| England | 11.2% | 6.2% |
| Portugal | 11.0% | 4.0% |
| Japan | 8.8% | 3.8% |
| Brazil | 8.6% | 3.2% |
| Colombia | 7.6% | 3.6% |
| France | 7.0% | 2.8% |
| Germany | 6.6% | 3.4% |
| Algeria | 5.6% | 2.6% |
| Mexico | 5.6% | 1.0% |
| Austria | 5.4% | 1.6% |
| Ivory Coast | 5.2% | 2.4% |
| Netherlands | 5.0% | 2.2% |
| IR Iran | 3.0% | 0.6% |
| Bosnia and Herzegovina | 2.4% | 1.2% |
| Switzerland | 2.0% | 0.6% |
| Ecuador | 1.4% | 0.2% |
| Egypt | 1.2% | 0.2% |
| Canada | 1.0% | 0.2% |
| Croatia | 1.0% | 0.2% |
| USA | 0.8% | 0.4% |
| Norway | 0.6% | 0.2% |
| Paraguay | 0.6% | 0.0% |
| Congo DR | 0.4% | 0.0% |
| Scotland | 0.2% | 0.2% |
| Cape Verde | 0.2% | 0.2% |
| Uruguay | 0.2% | 0.0% |
| South Korea | 0.2% | 0.0% |
| Senegal | 0.2% | 0.0% |
| Australia | 0.2% | 0.0% |
| Iraq | 0.2% | 0.0% |

## Grupos

### Group A

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Mexico vs South Africa | FT | **2-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 59% | 20% | 21% |
| South Korea vs Czechia | FT | **2-1**<br><sub>Pred: 2-2 -&gt; A (X)</sub> | 37% | 18% | 46% |
| Czechia vs South Africa | FT | **1-1**<br><sub>Pred: 1-1 -&gt; H (X)</sub> | 47% | 20% | 34% |
| Mexico vs South Korea | FT | **1-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 52% | 19% | 30% |
| Mexico vs Czechia | 2026-06-24 | 2-1 | 54% | 17% | 29% |
| South Africa vs South Korea | 2026-06-24 | 1-1 | 33% | 21% | 45% |

### Group B

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Canada vs Bosnia and Herzegovina | FT | **1-1**<br><sub>Pred: 1-1 -&gt; H (X)</sub> | 61% | 20% | 19% |
| Qatar vs Switzerland | FT | **1-1**<br><sub>Pred: 1-2 -&gt; A (X)</sub> | 14% | 14% | 72% |
| Switzerland vs Bosnia and Herzegovina | FT | **4-1**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 73% | 13% | 14% |
| Canada vs Qatar | FT | **6-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 61% | 20% | 19% |
| Canada vs Switzerland | 2026-06-24 | 1-1 | 35% | 22% | 44% |
| Bosnia and Herzegovina vs Qatar | 2026-06-24 | 1-1 | 34% | 24% | 43% |

### Group C

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Brazil vs Morocco | FT | **1-1**<br><sub>Pred: 1-1 -&gt; A (X)</sub> | 29% | 19% | 53% |
| Haiti vs Scotland | FT | **0-1**<br><sub>Pred: 1-2 -&gt; A (OK)</sub> | 33% | 17% | 50% |
| Scotland vs Morocco | FT | **0-1**<br><sub>Pred: 1-2 -&gt; A (OK)</sub> | 15% | 15% | 70% |
| Brazil vs Haiti | FT | **3-0**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 77% | 10% | 12% |
| Scotland vs Brazil | 2026-06-24 | 1-2 | 19% | 13% | 68% |
| Morocco vs Haiti | 2026-06-24 | 2-1 | 80% | 11% | 9% |

### Group D

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| USA vs Paraguay | FT | **4-1**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 39% | 24% | 38% |
| Australia vs Turkey | FT | **2-0**<br><sub>Pred: 1-1 -&gt; A (X)</sub> | 35% | 18% | 47% |
| USA vs Australia | FT | **2-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 41% | 22% | 37% |
| Turkey vs Paraguay | FT | **0-1**<br><sub>Pred: 1-1 -&gt; H (X)</sub> | 44% | 20% | 36% |
| USA vs Turkey | 2026-06-25 | 2-2 | 38% | 17% | 45% |
| Paraguay vs Australia | 2026-06-25 | 1-1 | 38% | 26% | 36% |

### Group E

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Germany vs Curacao | FT | **7-1**<br><sub>Pred: 4-1 -&gt; H (OK)</sub> | 91% | 5% | 5% |
| Ivory Coast vs Ecuador | FT | **1-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 46% | 23% | 31% |
| Germany vs Ivory Coast | FT | **2-1**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 45% | 18% | 36% |
| Ecuador vs Curacao | FT | **0-0**<br><sub>Pred: 2-1 -&gt; H (X)</sub> | 78% | 12% | 10% |
| Curacao vs Ivory Coast | 2026-06-25 | 1-3 | 7% | 7% | 86% |
| Ecuador vs Germany | 2026-06-25 | 1-1 | 32% | 20% | 49% |

### Group F

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Netherlands vs Japan | FT | **2-2**<br><sub>Pred: 1-1 -&gt; A (X)</sub> | 35% | 20% | 45% |
| Sweden vs Tunisia | FT | **5-1**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 48% | 19% | 33% |
| Netherlands vs Sweden | FT | **5-1**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 71% | 12% | 16% |
| Tunisia vs Japan | FT | **0-4**<br><sub>Pred: 0-2 -&gt; A (OK)</sub> | 10% | 13% | 77% |
| Japan vs Sweden | 2026-06-25 | 2-1 | 73% | 12% | 14% |
| Tunisia vs Netherlands | 2026-06-25 | 1-2 | 11% | 13% | 76% |

### Group G

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Belgium vs Egypt | FT | **1-1**<br><sub>Pred: 2-1 -&gt; H (X)</sub> | 62% | 17% | 21% |
| IR Iran vs New Zealand | FT | **2-2**<br><sub>Pred: 2-1 -&gt; H (X)</sub> | 79% | 11% | 9% |
| Belgium vs IR Iran | 2026-06-21 | 2-1 | 49% | 18% | 33% |
| New Zealand vs Egypt | 2026-06-21 | 1-1 | 18% | 19% | 63% |
| Egypt vs IR Iran | 2026-06-26 | 1-1 | 29% | 22% | 49% |
| New Zealand vs Belgium | 2026-06-26 | 1-3 | 5% | 7% | 88% |

### Group H

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Spain vs Cape Verde | FT | **0-0**<br><sub>Pred: 2-1 -&gt; H (X)</sub> | 82% | 10% | 8% |
| Saudi Arabia vs Uruguay | FT | **1-1**<br><sub>Pred: 1-1 -&gt; A (X)</sub> | 23% | 25% | 52% |
| Spain vs Saudi Arabia | 2026-06-21 | 2-0 | 78% | 13% | 9% |
| Uruguay vs Cape Verde | 2026-06-21 | 1-1 | 55% | 21% | 24% |
| Cape Verde vs Saudi Arabia | 2026-06-26 | 1-1 | 37% | 28% | 35% |
| Uruguay vs Spain | 2026-06-26 | 1-2 | 18% | 17% | 65% |

### Group I

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| France vs Senegal | FT | **3-1**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 61% | 16% | 23% |
| Iraq vs Norway | FT | **1-4**<br><sub>Pred: 1-1 -&gt; A (OK)</sub> | 18% | 17% | 65% |
| France vs Iraq | 2026-06-22 | 2-1 | 78% | 12% | 10% |
| Norway vs Senegal | 2026-06-22 | 1-1 | 46% | 19% | 35% |
| Norway vs France | 2026-06-26 | 1-2 | 29% | 16% | 55% |
| Senegal vs Iraq | 2026-06-26 | 1-1 | 58% | 21% | 22% |

### Group J

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Argentina vs Algeria | FT | **3-0**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 59% | 19% | 22% |
| Austria vs Jordan | FT | **3-1**<br><sub>Pred: 2-1 -&gt; H (OK)</sub> | 81% | 10% | 10% |
| Argentina vs Austria | 2026-06-22 | 2-1 | 64% | 17% | 19% |
| Jordan vs Algeria | 2026-06-22 | 1-2 | 8% | 10% | 82% |
| Algeria vs Austria | 2026-06-27 | 1-1 | 44% | 21% | 35% |
| Jordan vs Argentina | 2026-06-27 | 0-3 | 3% | 5% | 92% |

### Group K

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| Portugal vs Congo DR | FT | **1-1**<br><sub>Pred: 1-1 -&gt; H (X)</sub> | 66% | 18% | 16% |
| Uzbekistan vs Colombia | FT | **1-3**<br><sub>Pred: 1-2 -&gt; A (OK)</sub> | 16% | 17% | 68% |
| Portugal vs Uzbekistan | 2026-06-23 | 2-1 | 74% | 14% | 12% |
| Colombia vs Congo DR | 2026-06-23 | 1-1 | 59% | 21% | 20% |
| Colombia vs Portugal | 2026-06-27 | 1-1 | 34% | 19% | 47% |
| Congo DR vs Uzbekistan | 2026-06-27 | 1-1 | 44% | 26% | 30% |

### Group L

| Match | Status | Pick / Result | H | D | A |
|---|---|---|---:|---:|---:|
| England vs Croatia | FT | **4-2**<br><sub>Pred: 1-1 -&gt; H (OK)</sub> | 61% | 18% | 22% |
| Ghana vs Panama | FT | **1-0**<br><sub>Pred: 1-1 -&gt; A (X)</sub> | 32% | 18% | 50% |
| England vs Ghana | 2026-06-23 | 2-0 | 84% | 10% | 6% |
| Panama vs Croatia | 2026-06-23 | 1-2 | 20% | 14% | 66% |
| Panama vs England | 2026-06-27 | 1-2 | 7% | 9% | 84% |
| Croatia vs Ghana | 2026-06-27 | 2-1 | 71% | 14% | 15% |

<!-- predictor:snapshots:end -->
