#!/usr/bin/env python3
"""Gateway do Raspberry Pi

- Lê dados dos Arduinos (nó do lugar e nó da ala) através da porta série.
- Normaliza e guarda os eventos em SQLite.
- Mantém métricas em memória (ocupação, soma de lugares, qualidade do ar, ventoinha).
- (Opcional) Publica/recebe mensagens via MQTT conforme configuração.

Uso:
    python main.py --config config.yaml

Antes de correr, cria um ficheiro de configuração baseado em `config_example.yaml`.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import serial  # type: ignore
import yaml  # type: ignore

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:  # pragma: no cover - opcional
    mqtt = None  # type: ignore

LOGGER = logging.getLogger("iot.gateway")
if TYPE_CHECKING:
    from tinyml import (
        TinyMLRecommendationConfig as TinyMLRecommendationConfigType,
        TinyMLRecommendationRule as TinyMLRecommendationRuleType,
        TinyMLRuntime as TinyMLRuntimeType,
        TinyMLRuntimeConfig as TinyMLRuntimeConfigType,
        TinyMLUnavailableError as TinyMLUnavailableErrorType,
        TinyMLPrediction as TinyMLPredictionType,
    )
else:
    TinyMLRecommendationConfigType = Any
    TinyMLRecommendationRuleType = Any
    TinyMLRuntimeType = Any
    TinyMLRuntimeConfigType = Any
    TinyMLUnavailableErrorType = Exception
    TinyMLPredictionType = Any

MQTTClientType = Any
MQTTMessageType = Any

try:
    from tinyml import (  # type: ignore[attr-defined]
        TinyMLRecommendationConfig,
        TinyMLRecommendationRule,
        TinyMLRuntime,
        TinyMLRuntimeConfig,
        TinyMLUnavailableError,
        TinyMLPrediction,
    )
except ImportError as exc:  # pragma: no cover - TinyML é opcional
    LOGGER.warning("TinyML não disponível (%s). Funcionalidade desativada.", exc)
    TinyMLRecommendationConfig = None  # type: ignore[assignment]
    TinyMLRecommendationRule = None  # type: ignore[assignment]
    TinyMLRuntime = None  # type: ignore[assignment]
    TinyMLRuntimeConfig = None  # type: ignore[assignment]
    TinyMLUnavailableError = RuntimeError  # type: ignore[assignment]
    TinyMLPrediction = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class SerialDeviceConfig:
    path: str
    tipo: str  # "lugar" ou "ala"
    baudrate: int = 115200
    id_lugar: Optional[str] = None
    id_ala: Optional[str] = None


@dataclass
class MQTTConfig:
    enabled: bool
    broker: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    topics: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlaOccupancyRule:
    percent_min: float
    ventoinha_percent: int


@dataclass
class AlaSafetyConfig:
    default_percent: int = 30
    qualidade_ar_threshold: float = 3.5
    qualidade_ar_percent: int = 100
    alerta_sensor_percent: int = 100
    mismatch_delta: int = 3
    mismatch_duration_seconds: int = 5
    ack_timeout_seconds: int = 5
    occupancy_rules: list[AlaOccupancyRule] = field(default_factory=list)


@dataclass
class AlaConfig:
    capacidade_maxima: int
    soma_lugares_inicial: int = 0
    safety: Optional["AlaSafetyConfig"] = None


@dataclass
class GatewayConfig:
    sqlite_db_path: Path
    serial_devices: list[SerialDeviceConfig]
    alas: Dict[str, AlaConfig]
    mqtt: Optional[MQTTConfig] = None
    intervalos: Dict[str, int] = field(default_factory=dict)
    tinyml: Optional["TinyMLRuntimeConfigType"] = None


def load_config(path: Path) -> GatewayConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    serial_devices = [SerialDeviceConfig(**device) for device in raw.get("serial_devices", [])]

    alas_cfg: Dict[str, AlaConfig] = {}
    for ala_id, cfg in raw.get("alas", {}).items():
        alas_cfg[ala_id] = _parse_ala_config(cfg)

    mqtt_cfg = None
    raw_mqtt = raw.get("mqtt")
    if raw_mqtt:
        mqtt_cfg = MQTTConfig(**raw_mqtt)

    tinyml_cfg = None
    raw_tinyml = raw.get("tinyml")
    if raw_tinyml and TinyMLRuntimeConfig:
        tinyml_cfg = _parse_tinyml_config(raw_tinyml)

    return GatewayConfig(
        sqlite_db_path=Path(raw.get("sqlite_db_path", "iot_gateway.db")),
        serial_devices=serial_devices,
        alas=alas_cfg,
        mqtt=mqtt_cfg,
        intervalos=raw.get("intervalos", {}),
        tinyml=tinyml_cfg,
    )


def _parse_ala_config(raw: Dict[str, Any]) -> AlaConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração da ala inválida: {raw}")
    cfg = dict(raw)
    safety_raw = cfg.pop("safety", None)
    capacidade = cfg.get("capacidade_maxima")
    if capacidade is None:
        raise ValueError("Cada ala deve definir 'capacidade_maxima'.")
    soma_inicial = cfg.get("soma_lugares_inicial", 0)
    safety_cfg = _parse_ala_safety(safety_raw)
    return AlaConfig(
        capacidade_maxima=int(capacidade),
        soma_lugares_inicial=int(soma_inicial),
        safety=safety_cfg,
    )


def _parse_tinyml_config(raw: Dict[str, Any]) -> "TinyMLRuntimeConfigType":
    recommendation_raw = raw.get("recommendation", {}) if isinstance(raw, dict) else {}
    threshold_items = recommendation_raw.get("thresholds", []) if recommendation_raw else []
    thresholds: list["TinyMLRecommendationRuleType"] = []
    if TinyMLRecommendationRule:
        for item in threshold_items:
            if not isinstance(item, dict):
                continue
            limite_val = (
                item.get("limite_percent")
                or item.get("limite")
                or item.get("threshold")
            )
            percent_val = item.get("percent")
            if limite_val is None or percent_val is None:
                continue
            try:
                thresholds.append(TinyMLRecommendationRule(float(limite_val), int(percent_val)))
            except (TypeError, ValueError):
                LOGGER.warning("Threshold TinyML inválido: %s", item)
    baseline_percent = recommendation_raw.get("baseline_percent", 30) if recommendation_raw else 30
    recommendation_cfg = TinyMLRecommendationConfig(
        baseline_percent=int(baseline_percent),
        thresholds=thresholds,
    )
    return TinyMLRuntimeConfig(
        enabled=bool(raw.get("enabled", False)),
        ala_id=str(raw.get("ala_id", "A")),
        dataset_path=Path(raw.get("dataset_path", "data/tinyml_dataset.csv")),
        model_path=Path(raw.get("model_path", "models/tinyml_model.tflite")),
        metrics_path=Path(raw.get("metrics_path", "data/tinyml_metrics.json")),
        horizon_minutes=int(raw.get("horizon_minutes", 5)),
        flux_window_seconds=int(raw.get("flux_window_seconds", 120)),
        inference_interval_seconds=int(raw.get("inference_interval_seconds", 60)),
        override_ttl_seconds=int(raw.get("override_ttl_seconds", 180)),
        safety_quality_threshold=float(raw.get("safety_quality_threshold", 3.5)),
        safety_full_percent=int(raw.get("safety_full_percent", 100)),
        min_dataset_rows=int(raw.get("min_dataset_rows", 30)),
        recommendation=recommendation_cfg,
    )


def _parse_ala_safety(raw: Optional[Dict[str, Any]]) -> Optional[AlaSafetyConfig]:
    if not raw:
        return None
    occupancy_rules: list[AlaOccupancyRule] = []
    for item in raw.get("occupancy_rules", []):
        if not isinstance(item, dict):
            continue
        percent_min = (
            item.get("percent_min")
            or item.get("ocupacao_percent")
            or item.get("limite_percent")
            or item.get("ocupacao")
        )
        vent_percent = item.get("ventoinha_percent") or item.get("percent")
        if percent_min is None or vent_percent is None:
            continue
        try:
            occupancy_rules.append(
                AlaOccupancyRule(float(percent_min), int(vent_percent))
            )
        except (TypeError, ValueError):
            LOGGER.warning("Regra de ocupação inválida na segurança da ala: %s", item)
            continue
    occupancy_rules.sort(key=lambda r: r.percent_min, reverse=True)

    return AlaSafetyConfig(
        default_percent=int(raw.get("default_percent", 30)),
        qualidade_ar_threshold=float(raw.get("qualidade_ar_threshold", 3.5)),
        qualidade_ar_percent=int(raw.get("qualidade_ar_percent", 100)),
        alerta_sensor_percent=int(raw.get("alerta_sensor_percent", 100)),
        mismatch_delta=int(raw.get("mismatch_delta", 3)),
        mismatch_duration_seconds=int(raw.get("mismatch_duration_seconds", 5)),
        ack_timeout_seconds=int(raw.get("ack_timeout_seconds", 5)),
        occupancy_rules=occupancy_rules,
    )


# ---------------------------------------------------------------------------
# Base de dados SQLite
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS place_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lugar_id TEXT NOT NULL,
    estado TEXT NOT NULL,
    distancia REAL,
    referencia REAL,
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ala_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ala_id TEXT NOT NULL,
    evento TEXT NOT NULL,
    total INTEGER,
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS air_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ala_id TEXT NOT NULL,
    valor_bruto REAL,
    tensao REAL,
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fan_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ala_id TEXT NOT NULL,
    percent INTEGER,
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ala_id TEXT NOT NULL,
    ocupacao_ala INTEGER,
    soma_lugares INTEGER,
    qualidade_ar_bruto REAL,
    qualidade_ar_tensao REAL,
    ventoinha_percent INTEGER,
    alerta_sensor INTEGER,
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tinyml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ala_id TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    predicted_percent REAL NOT NULL,
    recommended_percent INTEGER NOT NULL,
    features TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _clamp_percent(value: float) -> int:
    return max(0, min(100, int(round(value))))


# ---------------------------------------------------------------------------
# Estado em memória
# ---------------------------------------------------------------------------

@dataclass
class AlaState:
    capacidade_maxima: int
    ocupacao_ala: int = 0
    soma_lugares: int = 0
    qualidade_ar_bruto: float = 0.0
    qualidade_ar_tensao: float = 0.0
    ventoinha_percent: int = 0
    alerta_sensor: bool = False
    safety: Optional["AlaSafetyConfig"] = None
    safety_target_percent: int = 0
    tinyml_target_percent: int = 0
    mismatch_since: Optional[float] = None
    ack_pendente: Optional[int] = None
    ack_warned: bool = False
    ultimo_comando_time: float = 0.0

    def percent_ocupacao(self) -> float:
        if self.capacidade_maxima <= 0:
            return 0.0
        return min(100.0, (self.ocupacao_ala / self.capacidade_maxima) * 100.0)


class StateTracker:
    def __init__(self, config: GatewayConfig) -> None:
        self.alas: Dict[str, AlaState] = {
            ala_id: AlaState(
                capacidade_maxima=cfg.capacidade_maxima,
                soma_lugares=cfg.soma_lugares_inicial,
                safety=cfg.safety,
                safety_target_percent=(cfg.safety.default_percent if cfg.safety else 0),
            )
            for ala_id, cfg in config.alas.items()
        }

    def ensure_ala(self, ala_id: str, capacidade_default: int = 10) -> AlaState:
        if ala_id not in self.alas:
            LOGGER.warning("Ala %s não definida na configuração, a criar com capacidade %s", ala_id, capacidade_default)
            self.alas[ala_id] = AlaState(capacidade_default)
        return self.alas[ala_id]


# ---------------------------------------------------------------------------
# Serial → Queue
# ---------------------------------------------------------------------------

@dataclass
class SerialMessage:
    device: SerialDeviceConfig
    raw_line: str


class SerialReader(threading.Thread):
    def __init__(self, device_cfg: SerialDeviceConfig, output_queue: "queue.Queue[SerialMessage]") -> None:
        super().__init__(daemon=True)
        self.device_cfg = device_cfg
        self.output_queue = output_queue
        self._stop_event = threading.Event()
        self.serial_handle: Optional[serial.Serial] = None
        self.write_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()
        if self.serial_handle and self.serial_handle.is_open:
            self.serial_handle.close()

    def send_line(self, line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        with self.write_lock:
            if not self.serial_handle or not self.serial_handle.is_open:
                LOGGER.warning("Porta %s não está aberta; não foi possível enviar comando.", self.device_cfg.path)
                return
            try:
                self.serial_handle.write(line.encode("utf-8"))
                self.serial_handle.flush()
            except serial.SerialException as exc:  # pragma: no cover - depende de hardware
                LOGGER.error("Erro ao enviar pela porta %s: %s", self.device_cfg.path, exc)

    def run(self) -> None:  # pragma: no cover (depende de hardware)
        while not self._stop_event.is_set():
            try:
                if not self.serial_handle or not self.serial_handle.is_open:
                    LOGGER.info("[Serial] Abrindo %s (%s)", self.device_cfg.path, self.device_cfg.tipo)
                    self.serial_handle = serial.Serial(
                        self.device_cfg.path,
                        self.device_cfg.baudrate,
                        timeout=1.0,
                    )
                line = self.serial_handle.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.output_queue.put(SerialMessage(self.device_cfg, line))
            except serial.SerialException as exc:
                LOGGER.error("Erro na porta %s: %s", self.device_cfg.path, exc)
                time.sleep(2)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Erro inesperado na leitura série (%s): %s", self.device_cfg.path, exc)
                time.sleep(2)


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------

class Gateway:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.conn = init_sqlite(config.sqlite_db_path)
        self.state_tracker = StateTracker(config)
        self.place_states: Dict[str, str] = {}
        self.serial_queue: "queue.Queue[SerialMessage]" = queue.Queue()
        self.serial_threads = [SerialReader(device, self.serial_queue) for device in config.serial_devices]
        self.ala_serial_reader: Optional[SerialReader] = next(
            (reader for reader in self.serial_threads if reader.device_cfg.tipo == "ala"), None
        )
        self.mqtt_client = self._init_mqtt(config.mqtt) if config.mqtt and config.mqtt.enabled else None
        self._sync_cloud_interval = max(0, config.intervalos.get("sync_cloud", 60))
        self._sync_batch_limit = config.intervalos.get("sync_batch_limit", 200)
        self._last_sync_time = time.time()
        self._last_synced_at = datetime.now(timezone.utc).isoformat()
        self.tinyml_runtime: Optional["TinyMLRuntimeType"] = None
        self._cloud_overrides: Dict[str, Tuple[int, Optional[float]]] = {}
        if config.tinyml and config.tinyml.enabled and TinyMLRuntime:
            self.state_tracker.ensure_ala(config.tinyml.ala_id)
            try:
                self.tinyml_runtime = TinyMLRuntime(config.tinyml, self.conn)
            except TinyMLUnavailableError as exc:
                LOGGER.warning("TinyML desativado: %s", exc)

    def start(self) -> None:
        for thread in self.serial_threads:
            thread.start()
        LOGGER.info("Gateway iniciado. A aguardar mensagens...")
        try:
            while True:
                self._maybe_sync_cloud()
                self._maybe_run_tinyml()
                self._check_pending_acks()
                try:
                    message = self.serial_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._handle_serial_message(message)
        except KeyboardInterrupt:
            LOGGER.info("A terminar...")
        finally:
            self.stop()

    def stop(self) -> None:
        for thread in self.serial_threads:
            thread.stop()
        if self.conn:
            self.conn.close()
        if self.mqtt_client:
            self.mqtt_client.disconnect()

    # ------------------------------------------------------------------
    # MQTT (opcional)
    # ------------------------------------------------------------------
    def _init_mqtt(self, cfg: MQTTConfig) -> Optional["MQTTClientType"]:  # pragma: no cover - depende de broker
        if mqtt is None:
            LOGGER.warning("paho-mqtt não está instalado; MQTT será ignorado")
            return None
        client = mqtt.Client()
        if cfg.username and cfg.password:
            client.username_pw_set(cfg.username, cfg.password)
        client.on_connect = self._on_mqtt_connect
        client.on_message = self._on_mqtt_message
        client.connect(cfg.broker, cfg.port)
        client.loop_start()
        return client

    def _publish_mqtt(self, topic_key: str, payload: Dict[str, Any]) -> None:
        if not self.mqtt_client or not self.config.mqtt:
            return
        topic = self.config.mqtt.topics.get(topic_key)
        if not topic:
            return
        self.mqtt_client.publish(topic, json.dumps(payload))

    # ------------------------------------------------------------------
    # Serial message handling
    # ------------------------------------------------------------------
    def _handle_serial_message(self, message: SerialMessage) -> None:
        try:
            data = json.loads(message.raw_line)
        except json.JSONDecodeError:
            LOGGER.warning("Mensagem inválida (%s): %s", message.device.path, message.raw_line)
            return

        if "id_lugar" in data:
            self._process_place_event(data, message.device)
        elif data.get("evento"):
            self._process_ala_event(data, message.device)
        else:
            LOGGER.debug("Mensagem desconhecida: %s", data)

    # Processamento do nó do lugar
    def _process_place_event(self, data: Dict[str, Any], device: SerialDeviceConfig) -> None:
        lugar_id = data.get("id_lugar")
        estado = data.get("estado_lugar")
        timestamp_ms = data.get("timestamp_ms")
        distancia = data.get("distancia_cm")
        referencia = data.get("referencia_cm")
        created_at = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT INTO place_events (lugar_id, estado, distancia, referencia, timestamp_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (lugar_id, estado, distancia, referencia, timestamp_ms, created_at),
        )
        self.conn.commit()

        ala_id = device.id_ala or (lugar_id.split("-")[0] if lugar_id else "?")
        ala_state = self.state_tracker.ensure_ala(ala_id)

        estado_anterior = self.place_states.get(lugar_id)
        if estado_anterior != estado:
            if estado == "ocupado":
                ala_state.soma_lugares += 1
            elif estado == "livre" and ala_state.soma_lugares > 0:
                if estado_anterior == "ocupado" or estado_anterior is None:
                    ala_state.soma_lugares -= 1
        self.place_states[lugar_id] = estado or "desconhecido"
        ala_state.soma_lugares = max(0, min(ala_state.soma_lugares, ala_state.capacidade_maxima))

        LOGGER.info("[%s] Lugar %s → %s (dist=%.1fcm)", device.path, lugar_id, estado, distancia or 0.0)
        snapshot = {
            "ala": ala_id,
            "capacidade": ala_state.capacidade_maxima,
            "percent_ocupacao": ala_state.percent_ocupacao(),
        }
        self._publish_mqtt("lugar", {"lugar": lugar_id, "estado": estado, "snapshot": snapshot})
        self._enforce_safety_rules(ala_id)

    # Processamento do nó da ala
    def _process_ala_event(self, data: Dict[str, Any], device: SerialDeviceConfig) -> None:
        evento = data.get("evento")
        ala_id = device.id_ala or "?"
        timestamp_ms = data.get("timestamp_ms")
        created_at = datetime.now(timezone.utc).isoformat()

        if evento in {"entrada", "saida"}:
            total = data.get("total")
            self.conn.execute(
                "INSERT INTO ala_events (ala_id, evento, total, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ala_id, evento, total, timestamp_ms, created_at),
            )
            self.conn.commit()
            ala_state = self.state_tracker.ensure_ala(ala_id)
            ala_state.ocupacao_ala = int(total or ala_state.ocupacao_ala)
            LOGGER.info("[%s] Ala %s %s → total=%s", device.path, ala_id, evento, total)
            self._enforce_safety_rules(ala_id)
            return

        if evento == "qualidade_ar":
            valor_bruto = data.get("valor_bruto")
            tensao = data.get("tensao_v")
            self.conn.execute(
                "INSERT INTO air_samples (ala_id, valor_bruto, tensao, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ala_id, valor_bruto, tensao, timestamp_ms, created_at),
            )
            self.conn.commit()
            ala_state = self.state_tracker.ensure_ala(ala_id)
            ala_state.qualidade_ar_bruto = float(valor_bruto or 0.0)
            ala_state.qualidade_ar_tensao = float(tensao or 0.0)
            LOGGER.debug("[%s] AR ala %s → %.1f (%.3f V)", device.path, ala_id, valor_bruto or 0.0, tensao or 0.0)
            self._enforce_safety_rules(ala_id)
            return

        if evento == "ventoinha":
            percent_raw = data.get("percent")
            try:
                percent = int(percent_raw)
            except (TypeError, ValueError):
                percent = None
            if percent is None:
                LOGGER.warning("Percentagem inválida recebida: %s", percent_raw)
                return
            self.conn.execute(
                "INSERT INTO fan_events (ala_id, percent, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?)",
                (ala_id, percent, timestamp_ms, created_at),
            )
            self.conn.commit()
            ala_state = self.state_tracker.ensure_ala(ala_id)
            ala_state.ventoinha_percent = percent
            if ala_state.ack_pendente == percent:
                ala_state.ack_pendente = None
                ala_state.ack_warned = False
            LOGGER.debug("[%s] Ventoinha ala %s → %s%%", device.path, ala_id, percent)
            self._enforce_safety_rules(ala_id)
            return

        if evento == "estado":
            ala_state = self.state_tracker.ensure_ala(ala_id)
            ala_state.ocupacao_ala = int(data.get("ocupacao_ala", ala_state.ocupacao_ala))
            ala_state.soma_lugares = int(data.get("soma_lugares", ala_state.soma_lugares))
            ala_state.qualidade_ar_bruto = float(data.get("qualidade_ar_bruto", ala_state.qualidade_ar_bruto))
            ala_state.qualidade_ar_tensao = float(data.get("qualidade_ar_tensao", ala_state.qualidade_ar_tensao))
            ala_state.ventoinha_percent = int(data.get("ventoinha_percent", ala_state.ventoinha_percent))
            ala_state.alerta_sensor = bool(data.get("alerta_sensor", ala_state.alerta_sensor))

            self.conn.execute(
                "INSERT INTO state_snapshots (ala_id, ocupacao_ala, soma_lugares, qualidade_ar_bruto,"
                " qualidade_ar_tensao, ventoinha_percent, alerta_sensor, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ala_id,
                    ala_state.ocupacao_ala,
                    ala_state.soma_lugares,
                    ala_state.qualidade_ar_bruto,
                    ala_state.qualidade_ar_tensao,
                    ala_state.ventoinha_percent,
                    int(ala_state.alerta_sensor),
                    timestamp_ms,
                    created_at,
                ),
            )
            self.conn.commit()

            resumo = {
                "ala": ala_id,
                "ocupacao": ala_state.ocupacao_ala,
                "soma_lugares": ala_state.soma_lugares,
                "capacidade": ala_state.capacidade_maxima,
                "percent_ocupacao": ala_state.percent_ocupacao(),
                "alerta_sensor": ala_state.alerta_sensor,
            }
            LOGGER.info("[%s] Estado ala %s → %s", device.path, ala_id, resumo)
            self._publish_mqtt("ala", resumo)
            self._enforce_safety_rules(ala_id)
            return

        LOGGER.debug("Evento da ala não tratado: %s", data)

    def _maybe_sync_cloud(self) -> None:
        if not self.mqtt_client or not self.config.mqtt:
            return
        if self._sync_cloud_interval <= 0:
            return
        now = time.time()
        if now - self._last_sync_time < self._sync_cloud_interval:
            return
        self._last_sync_time = now
        try:
            self._sync_with_cloud()
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Erro ao sincronizar com a cloud: %s", exc)

    def _sync_with_cloud(self) -> None:
        since_iso = self._last_synced_at
        place_rows, latest_place = self._fetch_table_rows("place_events", since_iso)
        ala_rows, latest_ala = self._fetch_table_rows("ala_events", since_iso)
        air_rows, latest_air = self._fetch_table_rows("air_samples", since_iso)
        fan_rows, latest_fan = self._fetch_table_rows("fan_events", since_iso)
        snapshot_rows, latest_state = self._fetch_table_rows("state_snapshots", since_iso)

        total = sum(len(rows) for rows in (place_rows, ala_rows, air_rows, fan_rows, snapshot_rows))
        if total == 0:
            LOGGER.debug("Sem novos dados para sincronizar com a cloud.")
            return

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "place_events": place_rows,
            "ala_events": ala_rows,
            "air_samples": air_rows,
            "fan_events": fan_rows,
            "state_snapshots": snapshot_rows,
            "summary": self._build_summary(),
        }
        self._publish_mqtt("cloud_out", payload)
        latest_candidates = [latest_place, latest_ala, latest_air, latest_fan, latest_state]
        latest_candidates = [value for value in latest_candidates if value is not None]
        if latest_candidates:
            self._last_synced_at = max(latest_candidates)
        LOGGER.info("Dados sincronizados com a cloud (%s registos)", total)

    def _enforce_safety_rules(self, ala_id: str) -> None:
        ala_state = self.state_tracker.ensure_ala(ala_id)
        safety = ala_state.safety
        if not safety:
            return
        percent = safety.default_percent
        ocupacao_percent = ala_state.percent_ocupacao()
        for rule in safety.occupancy_rules:
            if ocupacao_percent >= rule.percent_min:
                percent = max(percent, rule.ventoinha_percent)
                break
        if ala_state.qualidade_ar_tensao >= safety.qualidade_ar_threshold:
            percent = max(percent, safety.qualidade_ar_percent)

        delta = abs(ala_state.ocupacao_ala - ala_state.soma_lugares)
        now = time.time()
        if delta > safety.mismatch_delta:
            if ala_state.mismatch_since is None:
                ala_state.mismatch_since = now
            elif now - ala_state.mismatch_since >= safety.mismatch_duration_seconds:
                if not ala_state.alerta_sensor:
                    LOGGER.warning(
                        "[Segurança] Divergência persistente na ala %s (ocupacao=%s, soma_lugares=%s)",
                        ala_id,
                        ala_state.ocupacao_ala,
                        ala_state.soma_lugares,
                    )
                ala_state.alerta_sensor = True
                percent = max(percent, safety.alerta_sensor_percent)
        else:
            ala_state.mismatch_since = None

        ala_state.safety_target_percent = percent
        self._apply_control_for_ala(ala_id, reason="safety")

    def _compute_final_percent(self, ala_id: str) -> int:
        ala_state = self.state_tracker.ensure_ala(ala_id)
        candidates = [
            ala_state.safety_target_percent,
            ala_state.tinyml_target_percent,
        ]
        override = self._current_cloud_override(ala_id)
        if override is not None:
            candidates.append(override)
        # Filtra None e negativos
        usable = [value for value in candidates if value is not None]
        if not usable:
            return _clamp_percent(ala_state.ventoinha_percent)
        return _clamp_percent(max(usable))

    def _apply_control_for_ala(self, ala_id: str, reason: str, final_percent: Optional[int] = None) -> None:
        ala_state = self.state_tracker.ensure_ala(ala_id)
        percent = final_percent if final_percent is not None else self._compute_final_percent(ala_id)
        if ala_state.ack_pendente == percent:
            return
        if percent == ala_state.ventoinha_percent and ala_state.ack_pendente is None:
            return
        self._apply_ventoinha_percent(ala_id, percent, reason)

    def _apply_ventoinha_percent(self, ala_id: str, percent: int, reason: str) -> None:
        percent = _clamp_percent(percent)
        ala_state = self.state_tracker.ensure_ala(ala_id)
        if not self.ala_serial_reader:
            LOGGER.warning("Sem dispositivo série da ala para enviar comando (ala %s → %s%%)", ala_id, percent)
            return
        ala_state.ack_pendente = percent
        ala_state.ack_warned = False
        ala_state.ultimo_comando_time = time.time()
        try:
            self._send_command_to_ala({"percent": percent})
        except Exception:  # pragma: no cover
            ala_state.ack_pendente = None
            raise
        LOGGER.info("[Controlo:%s] Ventoinha ala %s → %s%%", reason, ala_id, percent)

    def _check_pending_acks(self) -> None:
        now = time.time()
        for ala_id, state in self.state_tracker.alas.items():
            if state.ack_pendente is None:
                continue
            safety = state.safety
            timeout = safety.ack_timeout_seconds if safety else 5
            if now - state.ultimo_comando_time > timeout and not state.ack_warned:
                LOGGER.warning(
                    "Sem ACK da ventoinha da ala %s (%s%%) após %ss.",
                    ala_id,
                    state.ack_pendente,
                    timeout,
                )
                state.ack_warned = True

    def _store_tinyml_prediction(self, ala_id: str, prediction: "TinyMLPredictionType", final_percent: int) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        features_to_store = dict(prediction.features)
        features_to_store["_tinyml_recommended"] = prediction.recommended_percent
        features_to_store["_final_percent"] = final_percent
        try:
            self.conn.execute(
                "INSERT INTO tinyml_predictions (ala_id, timestamp_ms, predicted_percent, recommended_percent,"
                " features, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ala_id,
                    prediction.timestamp_ms,
                    prediction.predicted_percent,
                    final_percent,
                    json.dumps(features_to_store),
                    created_at,
                ),
            )
            self.conn.commit()
        except sqlite3.DatabaseError as exc:
            LOGGER.debug("Não foi possível registar tinyml_predictions: %s", exc)

    def _current_cloud_override(self, ala_id: str) -> Optional[int]:
        override = self._cloud_overrides.get(ala_id)
        if not override:
            return None
        percent, expiry = override
        if expiry is None or time.time() <= expiry:
            return percent
        # Expirado
        del self._cloud_overrides[ala_id]
        return None

    def _register_cloud_override(self, ala_id: str, percent: int) -> None:
        ttl_seconds = 0
        if self.config.tinyml:
            ttl_seconds = self.config.tinyml.override_ttl_seconds
        expiry: Optional[float]
        if ttl_seconds <= 0:
            expiry = None
        else:
            expiry = time.time() + ttl_seconds
        self._cloud_overrides[ala_id] = (percent, expiry)

    def _maybe_run_tinyml(self) -> None:
        if not self.tinyml_runtime or not self.config.tinyml:
            return
        if not self.tinyml_runtime.should_run():
            return
        ala_id = self.config.tinyml.ala_id
        ala_state = self.state_tracker.ensure_ala(ala_id)
        capacidade = ala_state.capacidade_maxima
        if capacidade <= 0:
            LOGGER.debug("TinyML ignorado: capacidade inválida para ala %s", ala_id)
            return
        timestamp_ms = int(time.time() * 1000)
        prediction = self.tinyml_runtime.predict(
            timestamp_ms=timestamp_ms,
            ocupacao_ala=ala_state.ocupacao_ala,
            soma_lugares=ala_state.soma_lugares,
            capacidade_maxima=capacidade,
            qualidade_ar_tensao=ala_state.qualidade_ar_tensao,
            ventoinha_percent=ala_state.ventoinha_percent,
        )
        if not prediction:
            return
        ala_state.tinyml_target_percent = _clamp_percent(prediction.recommended_percent)
        final_percent = self._compute_final_percent(ala_id)
        self._store_tinyml_prediction(ala_id, prediction, final_percent)
        LOGGER.info(
            "[TinyML] Ala %s previsão=%.1f%% → recom=%s%% → final=%s%%",
            ala_id,
            prediction.predicted_percent,
            ala_state.tinyml_target_percent,
            final_percent,
        )
        self._apply_control_for_ala(ala_id, reason="tinyml", final_percent=final_percent)

    def _fetch_table_rows(self, table: str, since_iso: str, limit: Optional[int] = None) -> tuple[list[Dict[str, Any]], Optional[str]]:
        limit = limit or self._sync_batch_limit
        cursor = self.conn.execute(
            f"SELECT * FROM {table} WHERE created_at > ? ORDER BY created_at LIMIT ?",
            (since_iso, limit),
        )
        rows = cursor.fetchall()
        latest = None
        result: list[Dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            result.append(row_dict)
            created_at = row_dict.get("created_at")
            if created_at and (latest is None or created_at > latest):
                latest = created_at
        return result, latest

    def _build_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for ala_id, state in self.state_tracker.alas.items():
            summary[ala_id] = {
                "capacidade": state.capacidade_maxima,
                "ocupacao": state.ocupacao_ala,
                "soma_lugares": state.soma_lugares,
                "percent_ocupacao": state.percent_ocupacao(),
                "qualidade_ar_bruto": state.qualidade_ar_bruto,
                "qualidade_ar_tensao": state.qualidade_ar_tensao,
                "ventoinha_percent": state.ventoinha_percent,
                "alerta_sensor": state.alerta_sensor,
            }
        return summary

    def _handle_cloud_command(self, payload: Dict[str, Any]) -> None:
        ala_id = payload.get("ala") or payload.get("ala_id")
        if not ala_id:
            if self.state_tracker.alas:
                ala_id = next(iter(self.state_tracker.alas.keys()))
            else:
                LOGGER.warning("Comando cloud sem ala definida: %s", payload)
                return
        ala_state = self.state_tracker.ensure_ala(ala_id)
        updated = False
        final_percent_override: Optional[int] = None

        if "percent" in payload:
            try:
                percent = int(payload.get("percent", 0))
            except (TypeError, ValueError):
                LOGGER.warning("Percentagem inválida recebida da cloud: %s", payload.get("percent"))
            else:
                percent = _clamp_percent(percent)
                self._register_cloud_override(ala_id, percent)
                final_percent_override = self._compute_final_percent(ala_id)
                self._apply_control_for_ala(ala_id, reason="cloud", final_percent=final_percent_override)
                updated = True
                LOGGER.info("Comando cloud → ventoinha ala %s: %s%%", ala_id, percent)

        if "lugares_ocupados" in payload:
            try:
                soma = int(payload.get("lugares_ocupados", ala_state.soma_lugares))
            except (TypeError, ValueError):
                LOGGER.warning("Valor 'lugares_ocupados' inválido: %s", payload.get("lugares_ocupados"))
            else:
                ala_state.soma_lugares = max(0, min(soma, ala_state.capacidade_maxima))
                updated = True
                LOGGER.info("Atualização cloud → soma_lugares ala %s: %s", ala_id, ala_state.soma_lugares)
                self._enforce_safety_rules(ala_id)

        if not updated:
            LOGGER.debug("Comando cloud sem alterações aplicáveis: %s", payload)
            return

        timestamp_ms = payload.get("timestamp_ms")
        if not isinstance(timestamp_ms, int):
            timestamp_ms = int(time.time() * 1000)
        created_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO state_snapshots (ala_id, ocupacao_ala, soma_lugares, qualidade_ar_bruto,"
            " qualidade_ar_tensao, ventoinha_percent, alerta_sensor, timestamp_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ala_id,
                ala_state.ocupacao_ala,
                ala_state.soma_lugares,
                ala_state.qualidade_ar_bruto,
                ala_state.qualidade_ar_tensao,
                final_percent_override if final_percent_override is not None else ala_state.ventoinha_percent,
                int(ala_state.alerta_sensor),
                timestamp_ms,
                created_at,
            ),
        )
        self.conn.commit()
        resumo = self._build_summary().get(ala_id, {})
        self._publish_mqtt("ala", {"ala": ala_id, **resumo})

    def _send_command_to_ala(self, payload: Dict[str, Any]) -> None:
        if not self.ala_serial_reader:
            LOGGER.warning("Sem dispositivo série associado à ala; comando ignorado: %s", payload)
            return
        try:
            self.ala_serial_reader.send_line(json.dumps(payload))
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Erro ao enviar comando para a ala: %s", exc)

    def _on_mqtt_connect(self, client: "MQTTClientType", userdata: Any, flags: Dict[str, Any], rc: int) -> None:
        if rc == 0:
            LOGGER.info("Ligação MQTT estabelecida")
            if self.config.mqtt:
                topic = self.config.mqtt.topics.get("cloud_in")
                if topic:
                    client.subscribe(topic)
                    LOGGER.info("Subscrito ao tópico cloud_in: %s", topic)
        else:
            LOGGER.error("Falha na ligação MQTT (rc=%s)", rc)

    def _on_mqtt_message(self, client: "MQTTClientType", userdata: Any, msg: "MQTTMessageType") -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Mensagem MQTT inválida: %s", msg.payload)
            return
        LOGGER.info("Comando da cloud recebido (%s): %s", msg.topic, payload)
        self._handle_cloud_command(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gateway Raspberry Pi")
    parser.add_argument("--config", type=Path, required=True, help="Ficheiro YAML com a configuração")
    parser.add_argument("--log-level", default="INFO", help="Nível de logging (DEBUG, INFO, WARNING, ...)")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - ponto de entrada
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = load_config(args.config)
    gateway = Gateway(config)
    gateway.start()


if __name__ == "__main__":  # pragma: no cover
    main()
