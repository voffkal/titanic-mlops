"""Gradio demo for the released Titanic model.

The artifact is downloaded from the GitHub release rather than rebuilt here, so
what answers on the Space is the same binary the leaderboard score belongs to.
"""

from __future__ import annotations

import os
import re
import traceback
import urllib.request
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import spaces

from titanic_model import __version__
from titanic_model.features import TitleFeature
from titanic_model.predict import load_model, make_prediction

RELEASE = f"https://github.com/voffkal/titanic-mlops/releases/download/model-v{__version__}"
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "/tmp/titanic-artifacts"))
ASSETS = [f"titanic_model-{__version__}.joblib", "input_schema.json", "metrics.json"]

# Passenger counts are stated, not derived: int(0.76555 * 418) truncates to 319
# and the demo spent a day claiming the baseline got one fewer right than it did.
TOTAL = 418
LB_SCORE = 0.79665
LB_CORRECT = 333
SEX_RULE_CORRECT = 320  # "all women survive"
TITLE = re.compile(r",\s*[^.]+\.")  # the API rejects a title-less name; so does this


# ZeroGPU refuses to start a Space that declares no @spaces.GPU function, and the
# free Gradio tier offers only ZeroGPU. This satisfies that check and is never
# called: the model is sklearn and runs on the CPU, so putting the decorator on
# the real prediction would queue for a GPU and spend the daily quota — which is
# consumed only inside decorated functions — to run a logistic regression.
@spaces.GPU(duration=1)
def _gpu_probe() -> str:
    return "the model runs on CPU"


def fetch_artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        target = ARTIFACTS / name
        if not target.exists():
            urllib.request.urlretrieve(f"{RELEASE}/{name}", target)
    return ARTIFACTS


def check_versions() -> None:
    """The API refuses to start on version skew; so does this, and for the same
    reason. A pandas that merely satisfies the range unpickles the artifact into
    different StringDtype semantics and fails on every prediction with a message
    that never mentions versions."""
    import importlib.metadata as md
    import json

    trained = json.loads((ARTIFACTS / "metrics.json").read_text())["libraries"]
    rows, skew = [], []
    for name, want in sorted(trained.items()):
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = "missing"
        rows.append(f"  {name:18s} trained {want:10s} running {got}")
        if got != want:
            skew.append(f"{name}: trained {want}, running {got}")

    print("library versions against the artifact's manifest:", *rows, sep="\n", flush=True)
    critical = [s for s in skew if s.split(":")[0] in {"pandas", "numpy", "scikit-learn"}]
    if critical:
        raise RuntimeError(
            "version skew on libraries that decide how the artifact unpickles: "
            + "; ".join(critical)
        )


@lru_cache(maxsize=1)
def group_lookup():
    """The fitted GroupSurvival out of the artifact, loaded once rather than per call."""
    # through the package, not joblib.load: the sha256 check happens there
    model = load_model(ARTIFACTS)
    return model.estimators_[0].named_steps["group_feats"]


def companions(record: dict) -> str:
    """What the model knows about this passenger's group, which is its whole edge."""
    fitted = group_lookup()
    frame = TitleFeature().fit_transform(pd.DataFrame([record]))
    out = fitted.transform(frame)

    if not out["group_known"].iloc[0]:
        return "No companion of this passenger appears in the training data, so the answer rests on the other features alone."
    rate = float(out["group_rate"].iloc[0])
    return f"Companions sharing this surname or ticket are known, and {rate:.0%} of the women and boys among them survived."


def predict(pclass, sex, name, sibsp, parch, age, fare, ticket, cabin, embarked):
    record = {
        "pclass": int(pclass),
        "name": (name or "").strip(),
        "sex": sex,
        "sibsp": int(sibsp or 0),   # a cleared Number field arrives as None
        "parch": int(parch or 0),
        "age": age or None,
        "fare": fare or None,
        "ticket": ticket.strip() or None,
        "cabin": cabin.strip() or None,
        "embarked": embarked or None,
    }
    if not TITLE.search(record["name"]):
        return "—", (
            "**Rejected:** expected the Kaggle format `Surname, Title. Given names`. "
            "The title is a feature, and without one the model answers out of distribution."
        )

    try:
        result = make_prediction([record], artifacts_dir=fetch_artifacts())
    except ValueError as exc:
        return "—", f"**Rejected:** {exc}"
    except Exception as exc:  # surfaced rather than raised: a toast saying "Error" is useless
        traceback.print_exc()
        return "—", f"**{type(exc).__name__}:** {exc}"

    p = result["probabilities"][0]
    verdict = "survives" if result["predictions"][0] else "does not survive"
    try:
        note = companions(record)
    except Exception as exc:  # the group lookup is a nicety, not the answer
        traceback.print_exc()
        note = f"(companion lookup unavailable: {type(exc).__name__})"
    return f"{p:.1%}", f"The model says this passenger **{verdict}**.\n\n{note}"


EXAMPLES = [
    [1, "female", "Cumings, Mrs. John Bradley", 1, 0, 38, 71.28, "PC 17599", "C85", "C"],
    [3, "male", "Braund, Mr. Owen Harris", 1, 0, 22, 7.25, "A/5 21171", "", "S"],
    [3, "female", "Sage, Miss. Dorothy", 8, 2, 14, 69.55, "CA. 2343", "", "S"],
    [2, "male", "Palsson, Master. Gosta", 3, 1, 2, 21.07, "349909", "", "S"],
]

with gr.Blocks(title="Titanic Survival") as demo:
    gr.Markdown(
        f"""
        # Would this passenger have survived?

        A six-model stack trained on the 891 labelled passengers of the Kaggle
        Titanic dataset. **Public leaderboard {LB_SCORE}** — {LB_CORRECT} of {TOTAL}
        passengers, against {SEX_RULE_CORRECT} for simply predicting that every woman
        lived. The whole model is worth {LB_CORRECT - SEX_RULE_CORRECT} passengers.

        Method, measurements and the arguments against it:
        [github.com/voffkal/titanic-mlops](https://github.com/voffkal/titanic-mlops)
        """
    )
    with gr.Row():
        with gr.Column():
            pclass = gr.Radio([1, 2, 3], value=3, label="Class")
            sex = gr.Radio(["female", "male"], value="female", label="Sex")
            name = gr.Textbox(
                "Sage, Miss. Dorothy",
                label="Name",
                info="Kaggle format 'Surname, Title. Given names' — the title is a feature and the surname finds relatives",
            )
            with gr.Row():
                sibsp = gr.Number(0, label="Siblings / spouse aboard", precision=0)
                parch = gr.Number(0, label="Parents / children aboard", precision=0)
            with gr.Accordion("Optional — imputed when left empty", open=False):
                age = gr.Number(label="Age")
                fare = gr.Number(label="Fare")
                ticket = gr.Textbox("", label="Ticket")
                cabin = gr.Textbox("", label="Cabin", info="deck letter A–G or T")
                embarked = gr.Radio(["S", "C", "Q"], value=None, label="Embarked")
            go = gr.Button("Predict", variant="primary")
        with gr.Column():
            prob = gr.Label(label="Probability of survival")
            note = gr.Markdown()

    fields = [pclass, sex, name, sibsp, parch, age, fare, ticket, cabin, embarked]
    go.click(predict, fields, [prob, note])
    gr.Examples(EXAMPLES, fields, label="Real passengers from the training set")

    gr.Markdown(
        """
        ---
        The model reads the fate of a passenger's companions out of the training
        labels, so it partly answers by looking relatives up rather than from the
        attributes alone. That is what the last release bought, and the repository
        argues both sides of it.
        """
    )

try:  # at import, so a failure lands in the Space log instead of on a click
    fetch_artifacts()
    print(f"artifacts ready in {ARTIFACTS}: {sorted(p.name for p in ARTIFACTS.iterdir())}", flush=True)
    check_versions()
except Exception:
    traceback.print_exc()
    raise

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
