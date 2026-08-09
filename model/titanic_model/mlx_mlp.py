from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

__all__ = ["TitanicMLP", "train_mlp", "MLXClassifier"]

Monitor = Literal["auc", "accuracy"]


class TitanicMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_classes: int = 2,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden)
        self.l2 = nn.Linear(hidden, hidden // 2)
        self.l3 = nn.Linear(hidden // 2, n_classes)
        self.dropout = nn.Dropout(p_drop)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.l1(x))
        x = self.dropout(x)
        x = nn.relu(self.l2(x))
        return self.l3(x)


def _loss_fn(model: TitanicMLP, x: mx.array, y: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def _val_score(model: TitanicMLP, x: mx.array, y: np.ndarray, monitor: Monitor) -> float:
    logits = model(x)
    if monitor == "auc":
        proba = np.asarray(mx.softmax(logits, axis=1))[:, 1]
        return float(roc_auc_score(y, proba))
    return float(np.mean(np.asarray(mx.argmax(logits, axis=1)) == y))


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 3e-3,
    batch_size: int = 32,
    hidden: int = 64,
    p_drop: float = 0.2,
    patience: int = 40,
    monitor: Monitor = "auc",
    verbose: int = 0,
) -> tuple[TitanicMLP, float, int]:

    mx.random.seed(seed)  # BEFORE the constructor: Linear init draws from here
    rng = np.random.default_rng(seed)  # local generator, not the global np.random

    Xtr = mx.array(np.asarray(X_train, dtype=np.float32))
    ytr = mx.array(np.asarray(y_train).astype(np.int32))
    Xva = mx.array(np.asarray(X_val, dtype=np.float32))
    yva = np.asarray(y_val).astype(np.int32)

    n = Xtr.shape[0]
    model = TitanicMLP(in_features=Xtr.shape[1], hidden=hidden, p_drop=p_drop)
    mx.eval(model.parameters())

    steps_per_epoch = max(1, n // batch_size)
    schedule = optim.cosine_decay(lr, epochs * steps_per_epoch)
    optimizer = optim.Adam(learning_rate=schedule)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    best_score, best_params, best_epoch, no_improve = -np.inf, None, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, n, batch_size):
            idx = mx.array(perm[i : i + batch_size])
            loss, grads = loss_and_grad(model, Xtr[idx], ytr[idx])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss)
            n_batches += 1

        model.eval()
        score = _val_score(model, Xva, yva, monitor)

        if score > best_score:
            best_score, best_epoch, no_improve = score, ep, 0
            best_params = mlx.utils.tree_map(lambda p: mx.array(p), model.parameters())
        else:
            no_improve += 1

        if verbose and ep % verbose == 0:
            # epoch mean loss, not the last mini-batch
            print(
                f"epoch {ep:3d} | loss {epoch_loss / n_batches:.4f} | "
                f"val_{monitor} {score:.4f} | best {best_score:.4f} (epoch {best_epoch})"
            )

        if no_improve >= patience:
            break

    if best_params is not None:
        model.update(best_params)
        mx.eval(model.parameters())

    return model, best_score, best_epoch


class MLXClassifier(ClassifierMixin, BaseEstimator):  # base order matters: otherwise roc_auc returns NaN
    """Splits the validation set for early stopping off ``train`` inside ``fit``,
    so under cross-validation the outer test fold is never touched.
    ``random_state`` drives the train/val split, weight init and batch shuffling.
    """

    def __init__(
        self,
        epochs: int = 300,
        lr: float = 3e-3,
        batch_size: int = 32,
        hidden: int = 64,
        p_drop: float = 0.2,
        patience: int = 40,
        val_size: float = 0.2,
        monitor: Monitor = "auc",
        random_state: int = 0,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden = hidden
        self.p_drop = p_drop
        self.patience = patience
        self.val_size = val_size
        self.monitor = monitor
        self.random_state = random_state

    def fit(self, X, y) -> MLXClassifier:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X = np.asarray(X, dtype=np.float32)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=self.val_size,
            stratify=y,
            random_state=self.random_state,
        )

        self.model_, self.best_score_, self.best_epoch_ = train_mlp(
            X_tr,
            y_tr,
            X_va,
            y_va,
            seed=self.random_state,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            hidden=self.hidden,
            p_drop=self.p_drop,
            patience=self.patience,
            monitor=self.monitor,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        logits = self.model_(mx.array(np.asarray(X, dtype=np.float32)))
        return np.asarray(mx.softmax(logits, axis=1))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
