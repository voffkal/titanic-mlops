---
title: Titanic Survival API
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8002
pinned: false
license: mit
---

# Titanic Survival API

FastAPI service over a six-model stack trained on the Kaggle Titanic dataset.
Source, method and measurements: **https://github.com/voffkal/titanic-mlops**

Interactive docs are at [`/docs`](/docs).

```bash
curl -s $SPACE_URL/api/v1/predict -H 'content-type: application/json' -d '{"inputs":[
  {"Pclass":1,"Name":"Cumings, Mrs. John Bradley","Sex":"female","Age":38,
   "SibSp":1,"Parch":0,"Ticket":"PC 17599","Fare":71.2833,"Cabin":"C85","Embarked":"C"}
]}'
```

Field names are the Kaggle ones, so a row of `test.csv` posts unchanged.
`Pclass`, `Name`, `Sex`, `SibSp` and `Parch` are required; the rest are imputed.

**Kaggle public leaderboard 0.79665** — 333 of 418 passengers, against 320 for
the "all women survive" baseline. Cross-validated accuracy 0.8465 ±0.0184 over
25 folds.

The model reads the fate of a passenger's companions out of the training labels,
which makes it transductive: it answers partly by looking up relatives rather
than purely from the attributes in the request. That is a deliberate choice and
it is what the last release bought; the repository explains the trade-off.
