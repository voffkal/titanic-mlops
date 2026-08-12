---
title: Titanic Survival
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.23.1
app_file: app.py
python_version: 3.12.12
pinned: false
license: mit
---

# Titanic Survival API

A six-model stack trained on the 891 labelled passengers of the Kaggle Titanic
dataset. Source, method and measurements:
**https://github.com/voffkal/titanic-mlops**

The artifact is downloaded from that repository's GitHub release at startup, so
what answers here is the same binary the leaderboard score belongs to, not a
rebuild of it.

Try the two women in the examples. Both travelled without a first-class cabin
between them, and the model separates them 91% against 10% — because it knows
what happened to the people they were travelling with.

**Kaggle public leaderboard 0.79665** — 333 of 418 passengers, against 320 for
the "all women survive" baseline. Cross-validated accuracy 0.8465 ±0.0184 over
25 folds.

The model reads the fate of a passenger's companions out of the training labels,
which makes it transductive: it answers partly by looking up relatives rather
than purely from the attributes in the request. That is a deliberate choice and
it is what the last release bought; the repository explains the trade-off.
