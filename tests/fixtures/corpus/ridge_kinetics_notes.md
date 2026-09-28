---
doc_id: ridge_kinetics_notes
title: Kinetics notes for two-variable batch reactions
doi: 10.5555/fixture.notes.kinetics
material: substrate R
equipment: jacketed batch reactor
development_only: true
---
<!-- page: 1 -->
# Kinetics notes

## Temperature dependence

Increasing temperature increases the rate constant between 40 and 100 degC for substrate R.
The Arrhenius relation links the rate constant to the absolute temperature:

$$
k = A \exp\left(-\frac{E_a}{R T}\right)
$$

A higher rate constant increases yield when the reaction time is held fixed.

## Operating window

The jacket must satisfy T_max ≤ 120 °C and the hold time t_hold is 1–60 min.
Selectivity stays within 80–95 % when 40 degC < T < 100 degC; below 40 degC conversion is < 5 %.

<!-- page: 2 -->
## Conversion log

| run | temperature | time | conversion |
|---|---|---|---|
| [1] | [degC] | [min] | [%] |
| 1 | 40 | 10 | 12.5 |
| 2 | 60 | 10 | 31.0 |
| 3 | 80 | 10 | 58.2 |

## Procedure

1. Charge substrate R into the jacketed batch reactor.
2. Heat to the setpoint temperature at 2 degC/min.
3. Hold for the chosen time, then quench in ice water.
