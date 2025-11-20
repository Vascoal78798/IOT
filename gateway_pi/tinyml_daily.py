#!/usr/bin/env python3
"""Pipeline TinyML diário – previsões por hora para o dia seguinte."""
from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("iot.gateway.tinyml.daily")

FEATURE_WEEKDAY_NAMES: Sequence[str] = [f"weekday_{i}" for i in range(7)]
FEATURE_STAT_NAMES: Sequence[str] = ["mean_occ", "max_occ", "min_occ", "std_occ"]
TARGET_NAMES: Sequence[str] = [f"target_{hour:02d}" for hour in range(24)]


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_created_at(row: sqlite3.Row) -> datetime:
    created_at = row["created_at"]
    if created_at:
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            pass
    ts_ms = row["timestamp_ms"]
    if ts_ms is None:
        raise ValueError("Linha sem created_at nem timestamp_ms")
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)


@dataclass
class DailyProfile:
    date: date
    weekday: int
    hourly_percent: List[float]  # tamanho 24, 0..100
    mean_percent: float
    max_percent: float
    min_percent: float
    std_percent: float


class DailyDatasetBuilder:
    """Gera dataset diário (features → ocupação por hora do dia seguinte)."""

    def __init__(self, conn: sqlite3.Connection, config, capacidade_maxima: int) -> None:
        self.conn = conn
        self.config = config
        self.capacidade_maxima = capacidade_maxima
        if self.capacidade_maxima <= 0:
            raise ValueError("capacidade_maxima deve ser > 0 para o dataset diário")

    def collect_profiles(self) -> List[DailyProfile]:
        cursor = self.conn.execute(
            "SELECT created_at, timestamp_ms, ocupacao_ala FROM state_snapshots "
            "WHERE ala_id = ? ORDER BY created_at",
            (self.config.ala_id,),
        )
        daily_map: Dict[date, Dict[int, List[float]]] = {}
        for row in cursor.fetchall():
            dt = _parse_created_at(row)
            day = dt.date()
            hour = dt.hour
            percent = 0.0
            if self.capacidade_maxima > 0:
                ocupacao = row["ocupacao_ala"] or 0
                percent = min(100.0, max(0.0, (ocupacao / self.capacidade_maxima) * 100.0))
            daily_map.setdefault(day, {}).setdefault(hour, []).append(percent)

        profiles: List[DailyProfile] = []
        for day in sorted(daily_map.keys()):
            hour_map = daily_map[day]
            all_values = [value for values in hour_map.values() for value in values]
            if len(all_values) < self.config.min_hours_per_day:
                LOGGER.debug("Dia %s ignorado (apenas %s amostras)", day, len(all_values))
                continue

            hourly: List[float] = []
            last_value = float(all_values[0]) if all_values else 0.0
            day_mean = float(sum(all_values) / len(all_values)) if all_values else 0.0
            for hour in range(24):
                values = hour_map.get(hour)
                if values:
                    value = float(sum(values) / len(values))
                    last_value = value
                else:
                    value = last_value if hour > 0 else day_mean
                hourly.append(max(0.0, min(100.0, value)))

            day_max = max(hourly) if hourly else 0.0
            day_min = min(hourly) if hourly else 0.0
            variance = (
                sum((value - day_mean) ** 2 for value in hourly) / len(hourly)
                if hourly
                else 0.0
            )
            day_std = math.sqrt(variance)
            profiles.append(
                DailyProfile(
                    date=day,
                    weekday=day.weekday(),
                    hourly_percent=hourly,
                    mean_percent=day_mean,
                    max_percent=day_max,
                    min_percent=day_min,
                    std_percent=day_std,
                )
            )
        return profiles

    def _build_features(self, profile: DailyProfile) -> List[float]:
        weekday_one_hot = [1.0 if profile.weekday == i else 0.0 for i in range(7)]
        stats = [
            profile.mean_percent / 100.0,
            profile.max_percent / 100.0,
            profile.min_percent / 100.0,
            profile.std_percent / 100.0,
        ]
        return weekday_one_hot + stats

    def export(self) -> int:
        profiles = self.collect_profiles()
        if len(profiles) < self.config.min_days:
            LOGGER.warning(
                "Apenas %s dias disponíveis (< min_days=%s). Dataset pode ficar fraco.",
                len(profiles),
                self.config.min_days,
            )
        rows: List[List[float]] = []
        for idx in range(len(profiles) - 1):
            today = profiles[idx]
            tomorrow = profiles[idx + 1]
            features = self._build_features(today)
            targets = [value / 100.0 for value in tomorrow.hourly_percent]
            rows.append([today.date.isoformat(), *features, *targets])

        if not rows:
            LOGGER.warning("Sem pares de dias consecutivos suficientes para o dataset diário.")
            return 0

        _ensure_path(self.config.dataset_path)
        with self.config.dataset_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["date", *FEATURE_WEEKDAY_NAMES, *FEATURE_STAT_NAMES, *TARGET_NAMES]
            writer.writerow(header)
            writer.writerows(rows)
        LOGGER.info("Dataset diário exportado para %s (%s amostras).", self.config.dataset_path, len(rows))
        return len(rows)


class DailyTrainer:
    """Treina modelo diário (features → 24 horas do dia seguinte)."""

    def __init__(self, config, epochs: int = 200, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.dataset_path.exists():
            LOGGER.error("Dataset diário %s não existe. Exporta antes de treinar.", self.config.dataset_path)
            return np.empty((0,)), np.empty((0,))
        features: List[List[float]] = []
        targets: List[List[float]] = []
        with self.config.dataset_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    feature_vector = [
                        float(row[name]) for name in FEATURE_WEEKDAY_NAMES + FEATURE_STAT_NAMES
                    ]
                    target_vector = [float(row[name]) for name in TARGET_NAMES]
                except (TypeError, ValueError) as exc:
                    LOGGER.debug("Linha inválida no dataset diário (%s): %s", exc, row)
                    continue
                features.append(feature_vector)
                targets.append(target_vector)
        if not features:
            return np.empty((0,)), np.empty((0,))
        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def train(self) -> Dict[str, float]:
        import tensorflow as tf  # type: ignore

        X, y = self._load_dataset()
        if X.size == 0 or y.size == 0:
            raise RuntimeError("Dataset diário vazio; não foi possível treinar.")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(len(TARGET_NAMES), activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        history = model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1 if len(X) > 10 else 0.0,
            verbose=0,
        )

        metrics = {
            "samples": float(len(X)),
            "train_loss": float(history.history["loss"][-1]),
            "train_mae": float(history.history["mae"][-1]),
        }
        if "val_loss" in history.history and history.history["val_loss"]:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
        if "val_mae" in history.history and history.history["val_mae"]:
            metrics["val_mae"] = float(history.history["val_mae"][-1])

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        _ensure_path(self.config.model_path)
        self.config.model_path.write_bytes(tflite_model)
        LOGGER.info("Modelo diário treinado e guardado em %s", self.config.model_path)

        _ensure_path(self.config.metrics_path)
        self.config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


class DailyForecaster:
    """Gera previsão para o dia seguinte e grava em JSON + SQLite."""

    def __init__(self, conn: sqlite3.Connection, config, capacidade_maxima: int) -> None:
        self.conn = conn
        self.config = config
        self.capacidade_maxima = capacidade_maxima
        self.builder = DailyDatasetBuilder(conn, config, capacidade_maxima)

    def forecast(self) -> Dict[str, float]:
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Modelo diário {self.config.model_path} não encontrado. Treina primeiro.")

        profiles = self.builder.collect_profiles()
        if len(profiles) < 1:
            raise RuntimeError("Sem dados suficientes para gerar previsão diária.")

        latest_profile = profiles[-1]
        feature_vector = np.array([self.builder._build_features(latest_profile)], dtype=np.float32)

        try:
            from tflite_runtime.interpreter import Interpreter as TFLInterpreter  # type: ignore
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter as TFLInterpreter  # type: ignore
            except ImportError:  # pragma: no cover
                from tensorflow.lite import Interpreter as TFLInterpreter  # type: ignore

        interpreter = TFLInterpreter(model_path=str(self.config.model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], feature_vector)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        forecast_percent = [max(0.0, min(100.0, float(value) * 100.0)) for value in output]
        forecast_date = latest_profile.date + timedelta(days=1)
        created_at = datetime.now(timezone.utc).isoformat()

        forecast_payload = {
            "ala": self.config.ala_id,
            "forecast_date": forecast_date.isoformat(),
            "hours": {f"{hour:02d}": percent for hour, percent in enumerate(forecast_percent)},
            "created_at": created_at,
        }

        _ensure_path(self.config.forecast_path)
        self.config.forecast_path.write_text(json.dumps(forecast_payload, indent=2), encoding="utf-8")
        LOGGER.info("Previsão diária gravada em %s", self.config.forecast_path)

        self.conn.execute(
            "DELETE FROM daily_forecasts WHERE ala_id = ? AND forecast_date = ?",
            (self.config.ala_id, forecast_date.isoformat()),
        )
        for hour, percent in enumerate(forecast_percent):
            self.conn.execute(
                "INSERT INTO daily_forecasts (ala_id, forecast_date, hour, percent, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (self.config.ala_id, forecast_date.isoformat(), hour, percent, created_at),
            )
        self.conn.commit()
        return forecast_payload


