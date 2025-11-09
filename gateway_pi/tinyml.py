#!/usr/bin/env python3
"""Ferramentas TinyML para o gateway Raspberry Pi.

Inclui:
- Construção de datasets a partir do SQLite local.
- Treino de um modelo simples em TensorFlow e conversão para `.tflite`.
- Inferência em tempo real com `tflite-runtime` (ou `tensorflow` fallback).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("iot.gateway.tinyml")


# ---------------------------------------------------------------------------
# Estruturas de configuração
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TinyMLRecommendationRule:
    limite_percent: float
    percent: int


@dataclass(slots=True)
class TinyMLRecommendationConfig:
    baseline_percent: int = 30
    thresholds: list[TinyMLRecommendationRule] = field(default_factory=list)

    def percent_for_prediction(self, predicted_percent: float) -> int:
        percent = self.baseline_percent
        for rule in sorted(self.thresholds, key=lambda r: r.limite_percent, reverse=True):
            if predicted_percent >= rule.limite_percent:
                percent = max(percent, rule.percent)
                break
        return int(percent)


@dataclass(slots=True)
class TinyMLDatasetConfig:
    ala_id: str
    capacidade_maxima: int
    dataset_path: Path
    horizon_minutes: int = 5
    flux_window_seconds: int = 120
    min_samples: int = 60


@dataclass(slots=True)
class TinyMLTrainingConfig:
    dataset_path: Path
    model_output_path: Path
    metrics_path: Path
    epochs: int = 120
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 0.005
    patience: int = 8


@dataclass(slots=True)
class TinyMLRuntimeConfig:
    enabled: bool
    ala_id: str
    dataset_path: Path
    model_path: Path
    metrics_path: Path
    horizon_minutes: int = 5
    flux_window_seconds: int = 120
    inference_interval_seconds: int = 60
    override_ttl_seconds: int = 180
    safety_quality_threshold: float = 3.5
    safety_full_percent: int = 100
    min_dataset_rows: int = 30
    recommendation: TinyMLRecommendationConfig = field(default_factory=TinyMLRecommendationConfig)


@dataclass(slots=True)
class TinyMLPrediction:
    timestamp_ms: int
    features: Dict[str, float]
    predicted_percent: float
    recommended_percent: int


# Ordem fixa dos atributos que alimentam o modelo.
FEATURE_NAMES: Sequence[str] = (
    "ocupacao_percent",
    "soma_lugares_percent",
    "ventoinha_percent",
    "qualidade_ar_tensao",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "flux_entrada",
    "flux_saida",
    "flux_liquido",
)
TARGET_NAME = "target_percent"


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _row_timestamp(row: sqlite3.Row, default_tz: timezone = timezone.utc) -> int:
    if row["timestamp_ms"] is not None:
        return int(row["timestamp_ms"])
    created_at = row["created_at"]
    if not created_at:
        raise ValueError("Linha SQLite sem timestamp_ms nem created_at")
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return int(dt.timestamp() * 1000)


def _clamp_percent(value: float) -> int:
    return int(max(0, min(100, round(value))))


@dataclass(slots=True)
class _Snapshot:
    timestamp_ms: int
    ocupacao_ala: int
    soma_lugares: int
    qualidade_ar_tensao: float
    ventoinha_percent: int

    def ocupacao_percent(self, capacidade_maxima: int) -> float:
        if capacidade_maxima <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.ocupacao_ala / capacidade_maxima) * 100.0))

    def soma_percent(self, capacidade_maxima: int) -> float:
        if capacidade_maxima <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.soma_lugares / capacidade_maxima) * 100.0))


@dataclass(slots=True)
class _AlaEvent:
    timestamp_ms: int
    tipo: str


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


class TinyMLDatasetBuilder:
    """Extrai amostras da base de dados local e gera um CSV para treino."""

    def __init__(self, conn: sqlite3.Connection, config: TinyMLDatasetConfig) -> None:
        self.conn = conn
        self.config = config
        if self.config.capacidade_maxima <= 0:
            raise ValueError("capacidade_maxima tem de ser > 0")

    def export(self) -> int:
        rows = self._build_rows()
        if not rows:
            LOGGER.warning("Sem linhas suficientes para o dataset TinyML.")
            return 0
        _ensure_path(self.config.dataset_path)
        with self.config.dataset_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp_ms", *FEATURE_NAMES, TARGET_NAME]
            writer.writerow(header)
            writer.writerows(rows)
        LOGGER.info(
            "Dataset TinyML exportado: %s (%s amostras, horizonte %s min, janela fluxo %s s)",
            self.config.dataset_path,
            len(rows),
            self.config.horizon_minutes,
            self.config.flux_window_seconds,
        )
        return len(rows)

    def _build_rows(self) -> list[list[float]]:
        snapshots = self._fetch_snapshots()
        if len(snapshots) < self.config.min_samples:
            LOGGER.warning(
                "Encontradas %s amostras, inferior ao mínimo (%s).",
                len(snapshots),
                self.config.min_samples,
            )
            return []
        events = self._fetch_events()
        horizon_ms = self.config.horizon_minutes * 60_000
        flux_window_ms = self.config.flux_window_seconds * 1_000
        timestamps = [snap.timestamp_ms for snap in snapshots]
        rows: list[list[float]] = []

        for idx, snap in enumerate(snapshots):
            target_ms = snap.timestamp_ms + horizon_ms
            target_idx = _bisect_left(timestamps, target_ms, lo=idx + 1)
            if target_idx >= len(snapshots):
                break  # sem futuro suficiente
            target_snap = snapshots[target_idx]
            features = self._build_features(snap, flux_window_ms, events)
            if not features:
                continue
            target_percent = target_snap.ocupacao_percent(self.config.capacidade_maxima)
            row = [snap.timestamp_ms, *[features[name] for name in FEATURE_NAMES], target_percent]
            rows.append(row)
        return rows

    def _fetch_snapshots(self) -> list[_Snapshot]:
        cursor = self.conn.execute(
            "SELECT timestamp_ms, created_at, ocupacao_ala, soma_lugares, qualidade_ar_tensao, ventoinha_percent "
            "FROM state_snapshots WHERE ala_id = ? ORDER BY created_at",
            (self.config.ala_id,),
        )
        snapshots: list[_Snapshot] = []
        for row in cursor.fetchall():
            ts_ms = _row_timestamp(row)
            snapshots.append(
                _Snapshot(
                    timestamp_ms=ts_ms,
                    ocupacao_ala=int(row["ocupacao_ala"] or 0),
                    soma_lugares=int(row["soma_lugares"] or 0),
                    qualidade_ar_tensao=float(row["qualidade_ar_tensao"] or 0.0),
                    ventoinha_percent=int(row["ventoinha_percent"] or 0),
                )
            )
        return snapshots

    def _fetch_events(self) -> list[_AlaEvent]:
        cursor = self.conn.execute(
            "SELECT evento, timestamp_ms, created_at "
            "FROM ala_events WHERE ala_id = ? ORDER BY created_at",
            (self.config.ala_id,),
        )
        events: list[_AlaEvent] = []
        for row in cursor.fetchall():
            evento = str(row["evento"] or "")
            if evento not in {"entrada", "saida"}:
                continue
            ts_ms = _row_timestamp(row)
            events.append(_AlaEvent(timestamp_ms=ts_ms, tipo=evento))
        return events

    def _build_features(
        self,
        snapshot: _Snapshot,
        flux_window_ms: int,
        events: list[_AlaEvent],
    ) -> Optional[Dict[str, float]]:
        window_start = snapshot.timestamp_ms - flux_window_ms
        fluxo_entrada = 0
        fluxo_saida = 0
        for evento in events:
            if evento.timestamp_ms > snapshot.timestamp_ms:
                break
            if evento.timestamp_ms < window_start:
                continue
            if evento.tipo == "entrada":
                fluxo_entrada += 1
            elif evento.tipo == "saida":
                fluxo_saida += 1

        dt = datetime.fromtimestamp(snapshot.timestamp_ms / 1000.0, tz=timezone.utc)
        hour_fraction = (dt.hour * 60 + dt.minute) / (24 * 60)
        hour_angle = 2 * math.pi * hour_fraction
        weekday_fraction = dt.weekday() / 7.0
        weekday_angle = 2 * math.pi * weekday_fraction

        features: Dict[str, float] = {
            "ocupacao_percent": snapshot.ocupacao_percent(self.config.capacidade_maxima),
            "soma_lugares_percent": snapshot.soma_percent(self.config.capacidade_maxima),
            "ventoinha_percent": float(snapshot.ventoinha_percent),
            "qualidade_ar_tensao": float(snapshot.qualidade_ar_tensao),
            "hour_sin": math.sin(hour_angle),
            "hour_cos": math.cos(hour_angle),
            "weekday_sin": math.sin(weekday_angle),
            "weekday_cos": math.cos(weekday_angle),
            "flux_entrada": float(fluxo_entrada),
            "flux_saida": float(fluxo_saida),
            "flux_liquido": float(fluxo_entrada - fluxo_saida),
        }
        return features


def _bisect_left(a: Sequence[int], x: int, lo: int = 0, hi: Optional[int] = None) -> int:
    if lo < 0:
        raise ValueError("lo deve ser não negativo")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Treino (TensorFlow)
# ---------------------------------------------------------------------------


class TinyMLTrainer:
    """Treina um modelo Keras simples e exporta para TensorFlow Lite."""

    def __init__(self, config: TinyMLTrainingConfig) -> None:
        self.config = config

    def train(self) -> Dict[str, float]:
        data = self._load_dataset()
        if not data:
            raise RuntimeError("Dataset vazio ou inexistente.")
        features = np.array([row["features"] for row in data], dtype=np.float32)
        targets = np.array([row["target"] for row in data], dtype=np.float32)

        import tensorflow as tf  # type: ignore

        tf.random.set_seed(42)
        np.random.seed(42)

        dataset_size = len(features)
        permutation = np.random.permutation(dataset_size)
        features = features[permutation]
        targets = targets[permutation]

        split_index = max(1, int(dataset_size * (1 - self.config.validation_split)))
        train_X = features[:split_index]
        train_y = targets[:split_index]
        val_X = features[split_index:]
        val_y = targets[split_index:]

        inputs = tf.keras.layers.Input(shape=(len(FEATURE_NAMES),))
        x = tf.keras.layers.Dense(32, activation="relu")(inputs)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        callbacks: list[tf.keras.callbacks.Callback] = []
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
            )
        )

        history = model.fit(
            train_X,
            train_y,
            validation_data=(val_X, val_y) if len(val_X) else None,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        metrics = {
            "samples": float(dataset_size),
            "train_loss": float(history.history["loss"][-1]),
            "train_mae": float(history.history["mae"][-1]),
        }
        if "val_loss" in history.history and len(history.history["val_loss"]) > 0:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
        if "val_mae" in history.history and len(history.history["val_mae"]) > 0:
            metrics["val_mae"] = float(history.history["val_mae"][-1])

        # Exportar modelo para ficheiro .tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        _ensure_path(self.config.model_output_path)
        self.config.model_output_path.write_bytes(tflite_model)

        _ensure_path(self.config.metrics_path)
        self.config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOGGER.info(
            "Modelo TinyML treinado (%s amostras). Métricas guardadas em %s",
            dataset_size,
            self.config.metrics_path,
        )
        return metrics

    def _load_dataset(self) -> list[Dict[str, np.ndarray]]:
        if not self.config.dataset_path.exists():
            LOGGER.error("Dataset %s não existe.", self.config.dataset_path)
            return []
        rows: list[Dict[str, np.ndarray]] = []
        with self.config.dataset_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line in reader:
                try:
                    features = np.array([float(line[name]) for name in FEATURE_NAMES], dtype=np.float32)
                    target = float(line[TARGET_NAME])
                except (KeyError, ValueError) as exc:
                    LOGGER.debug("Linha inválida no dataset (%s): %s", exc, line)
                    continue
                rows.append({"features": features, "target": target})
        return rows


# ---------------------------------------------------------------------------
# Inferência em tempo real
# ---------------------------------------------------------------------------


class TinyMLUnavailableError(RuntimeError):
    """Lançado quando o ambiente não suporta inferência TinyML."""


class TinyMLRuntime:
    """Executa previsões periódicas com base num modelo TFLite."""

    def __init__(self, config: TinyMLRuntimeConfig, conn: sqlite3.Connection) -> None:
        self.config = config
        self.conn = conn
        self._interpreter = self._load_interpreter(config.model_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._last_run = 0.0

    @property
    def available(self) -> bool:
        return self._interpreter is not None

    def should_run(self, now: Optional[float] = None) -> bool:
        if not self.available:
            return False
        now = now or time.time()
        return now - self._last_run >= max(1, self.config.inference_interval_seconds)

    def predict(
        self,
        *,
        timestamp_ms: int,
        ocupacao_ala: int,
        soma_lugares: int,
        capacidade_maxima: int,
        qualidade_ar_tensao: float,
        ventoinha_percent: int,
    ) -> Optional[TinyMLPrediction]:
        if not self.available:
            return None
        if capacidade_maxima <= 0:
            LOGGER.debug("Capacidade máxima inválida (<=0).")
            return None
        features = self._build_features(
            timestamp_ms=timestamp_ms,
            ocupacao_ala=ocupacao_ala,
            soma_lugares=soma_lugares,
            capacidade_maxima=capacidade_maxima,
            qualidade_ar_tensao=qualidade_ar_tensao,
            ventoinha_percent=ventoinha_percent,
        )
        if not features:
            return None

        input_array = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
        self._interpreter.set_tensor(self._input_details[0]["index"], input_array)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        predicted_percent = float(output[0][0])
        recommended_percent = _clamp_percent(
            self.config.recommendation.percent_for_prediction(predicted_percent)
        )
        self._last_run = time.time()
        return TinyMLPrediction(
            timestamp_ms=timestamp_ms,
            features=features,
            predicted_percent=predicted_percent,
            recommended_percent=recommended_percent,
        )

    def _load_interpreter(self, model_path: Path):
        if not model_path.exists():
            raise TinyMLUnavailableError(f"Modelo TinyML não encontrado: {model_path}")
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:  # pragma: no cover
            try:
                from tensorflow.lite import Interpreter  # type: ignore
            except ImportError as exc:
                raise TinyMLUnavailableError(
                    "tflite-runtime ou tensorflow.lite não disponíveis"
                ) from exc
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        LOGGER.info("Modelo TinyML carregado: %s", model_path)
        return interpreter

    def _build_features(
        self,
        *,
        timestamp_ms: int,
        ocupacao_ala: int,
        soma_lugares: int,
        capacidade_maxima: int,
        qualidade_ar_tensao: float,
        ventoinha_percent: int,
    ) -> Optional[Dict[str, float]]:
        cursor = self.conn.execute(
            "SELECT evento, timestamp_ms, created_at FROM ala_events "
            "WHERE ala_id = ? AND created_at >= ? ORDER BY created_at",
            (
                self.config.ala_id,
                datetime.fromtimestamp(
                    (timestamp_ms - self.config.flux_window_seconds * 1000) / 1000.0,
                    tz=timezone.utc,
                ).isoformat(),
            ),
        )
        fluxo_entrada = 0
        fluxo_saida = 0
        window_start = timestamp_ms - self.config.flux_window_seconds * 1000
        for row in cursor.fetchall():
            ts_ms = _row_timestamp(row)
            if ts_ms < window_start or ts_ms > timestamp_ms:
                continue
            evento = row["evento"]
            if evento == "entrada":
                fluxo_entrada += 1
            elif evento == "saida":
                fluxo_saida += 1

        ocupacao_percent = max(0.0, min(100.0, (ocupacao_ala / capacidade_maxima) * 100.0))
        soma_percent = max(0.0, min(100.0, (soma_lugares / capacidade_maxima) * 100.0))

        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        hour_fraction = (dt.hour * 60 + dt.minute) / (24 * 60)
        hour_angle = 2 * math.pi * hour_fraction
        weekday_fraction = dt.weekday() / 7.0
        weekday_angle = 2 * math.pi * weekday_fraction

        features: Dict[str, float] = {
            "ocupacao_percent": ocupacao_percent,
            "soma_lugares_percent": soma_percent,
            "ventoinha_percent": float(ventoinha_percent),
            "qualidade_ar_tensao": float(qualidade_ar_tensao),
            "hour_sin": math.sin(hour_angle),
            "hour_cos": math.cos(hour_angle),
            "weekday_sin": math.sin(weekday_angle),
            "weekday_cos": math.cos(weekday_angle),
            "flux_entrada": float(fluxo_entrada),
            "flux_saida": float(fluxo_saida),
            "flux_liquido": float(fluxo_entrada - fluxo_saida),
        }
        return features


__all__ = [
    "TinyMLDatasetBuilder",
    "TinyMLDatasetConfig",
    "TinyMLTrainer",
    "TinyMLTrainingConfig",
    "TinyMLRuntime",
    "TinyMLRuntimeConfig",
    "TinyMLPrediction",
    "TinyMLRecommendationConfig",
    "TinyMLRecommendationRule",
    "TinyMLUnavailableError",
    "FEATURE_NAMES",
    "TARGET_NAME",
]


