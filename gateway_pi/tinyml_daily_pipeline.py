#!/usr/bin/env python3
"""CLI para o pipeline TinyML diário (previsões por hora do dia seguinte)."""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from main import GatewayConfig, init_sqlite, load_config
from tinyml_daily import DailyDatasetBuilder, DailyForecaster, DailyTrainer

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore

LOGGER = logging.getLogger("iot.gateway.tinyml.daily.cli")


def _require_daily_cfg(config: GatewayConfig) -> None:
    if not config.tinyml_daily:
        raise SystemExit("Configuração 'tinyml_daily' não encontrada no ficheiro YAML.")
    if config.tinyml_daily.enabled is False:
        LOGGER.warning("tinyml_daily.enabled está a false; a pipeline pode não ser executada automaticamente.")


def _resolve_ala(config: GatewayConfig, override: Optional[str]) -> str:
    if override:
        return override
    if config.tinyml_daily and config.tinyml_daily.ala_id:
        return config.tinyml_daily.ala_id
    if config.tinyml and config.tinyml.ala_id:
        return config.tinyml.ala_id
    if config.alas:
        return next(iter(config.alas.keys()))
    raise SystemExit("Nenhuma ala definida na configuração.")


def cmd_aggregate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _require_daily_cfg(config)
    ala_id = _resolve_ala(config, args.ala)
    capacidade = config.alas[ala_id].capacidade_maxima
    conn = init_sqlite(config.sqlite_db_path)
    try:
        builder = DailyDatasetBuilder(conn, config.tinyml_daily, capacidade)
        samples = builder.export()
        print(f"Dataset diário exportado ({samples} amostras) para {config.tinyml_daily.dataset_path}")
    finally:
        conn.close()


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _require_daily_cfg(config)
    trainer = DailyTrainer(
        config.tinyml_daily,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


def cmd_forecast(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _require_daily_cfg(config)
    ala_id = _resolve_ala(config, args.ala)
    capacidade = config.alas[ala_id].capacidade_maxima
    conn = init_sqlite(config.sqlite_db_path)
    try:
        forecaster = DailyForecaster(conn, config.tinyml_daily, capacidade)
        forecast = forecaster.forecast()
        print(json.dumps(forecast, indent=2))
        _publish_forecast(config, forecast)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline TinyML diário (previsão por hora do dia seguinte)")
    parser.add_argument("--config", type=Path, required=True, help="Ficheiro de configuração YAML.")
    parser.add_argument("--ala", type=str, help="ID da ala (override).")

    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="Agrega dados diários e exporta dataset CSV.")
    aggregate_parser.set_defaults(func=cmd_aggregate)

    train_parser = subparsers.add_parser("train", help="Treina o modelo diário e gera .tflite.")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.set_defaults(func=cmd_train)

    forecast_parser = subparsers.add_parser("forecast", help="Gera previsão das próximas 24h e grava no SQLite/JSON.")
    forecast_parser.set_defaults(func=cmd_forecast)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    args.func(args)


def _publish_forecast(config: GatewayConfig, payload: dict) -> None:
    """Envia a previsão diária para os tópicos MQTT configurados."""
    if mqtt is None:
        return

    topic_key = "daily_forecast"
    message = json.dumps(payload)

    # MQTT local
    if config.mqtt and config.mqtt.enabled:
        topic = config.mqtt.topics.get(topic_key)
        if topic:
            client = mqtt.Client()
            if config.mqtt.username or config.mqtt.password:
                client.username_pw_set(config.mqtt.username, config.mqtt.password)
            try:
                client.connect(config.mqtt.broker, config.mqtt.port)
                client.loop_start()
                info = client.publish(topic, message)
                info.wait_for_publish()
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Não foi possível publicar previsão no broker local: {exc}")
            finally:
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception:  # pragma: no cover
                    pass

    # MQTT cloud (TLS)
    if config.cloud and config.cloud.mqtt and config.cloud.mqtt.enabled:
        cfg = config.cloud.mqtt
        topic = cfg.topics.get(topic_key) if cfg.topics else None
        if topic and cfg.endpoint:
            client_id = cfg.client_id or "daily-forecaster"
            client = mqtt.Client(client_id=f"{client_id}-daily")
            if cfg.username or cfg.password:
                client.username_pw_set(cfg.username, cfg.password)
            if cfg.ca_cert or cfg.certfile or cfg.keyfile:
                try:
                    client.tls_set(
                        ca_certs=cfg.ca_cert,
                        certfile=cfg.certfile,
                        keyfile=cfg.keyfile,
                    )
                    client.tls_insecure_set(cfg.tls_insecure)
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] Falha ao configurar TLS para cloud MQTT: {exc}")
                    return
            try:
                client.connect(cfg.endpoint, cfg.port, keepalive=cfg.keepalive)
                client.loop_start()
                info = client.publish(topic, message)
                info.wait_for_publish()
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Não foi possível publicar previsão na cloud: {exc}")
            finally:
                try:
                    client.loop_stop()
                    client.disconnect()
                except Exception:  # pragma: no cover
                    pass


if __name__ == "__main__":
    main()
