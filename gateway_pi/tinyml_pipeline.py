#!/usr/bin/env python3
"""Pipeline TinyML para o gateway.

Comandos:
    export  - gera dataset CSV a partir do SQLite local.
    train   - (re)gera dataset e treina modelo Keras → TFLite.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from main import GatewayConfig, init_sqlite, load_config
from tinyml import TinyMLDatasetBuilder, TinyMLDatasetConfig, TinyMLRuntimeConfig, TinyMLTrainer, TinyMLTrainingConfig

LOGGER = logging.getLogger("iot.gateway.tinyml.pipeline")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _resolve_ala(cfg: GatewayConfig, override: Optional[str]) -> str:
    if override:
        if override not in cfg.alas:
            raise SystemExit(f"Ala '{override}' não existe na configuração.")
        return override
    if cfg.tinyml:
        return cfg.tinyml.ala_id
    if cfg.alas:
        return next(iter(cfg.alas.keys()))
    raise SystemExit("Configuração sem alas definidas.")


def _resolve_tinyml_cfg(cfg: GatewayConfig) -> Optional[TinyMLRuntimeConfig]:
    return cfg.tinyml


def _build_dataset_cfg(
    cfg: GatewayConfig,
    ala_id: str,
    dataset_path: Path,
    horizon_minutes: Optional[int],
    flux_window_seconds: Optional[int],
    min_samples: Optional[int],
) -> TinyMLDatasetConfig:
    ala_cfg = cfg.alas.get(ala_id)
    if not ala_cfg:
        raise SystemExit(f"Ala '{ala_id}' não encontrada na configuração.")
    runtime_cfg = _resolve_tinyml_cfg(cfg)
    return TinyMLDatasetConfig(
        ala_id=ala_id,
        capacidade_maxima=ala_cfg.capacidade_maxima,
        dataset_path=dataset_path,
        horizon_minutes=horizon_minutes or (runtime_cfg.horizon_minutes if runtime_cfg else 5),
        flux_window_seconds=flux_window_seconds or (runtime_cfg.flux_window_seconds if runtime_cfg else 120),
        min_samples=min_samples or (runtime_cfg.min_dataset_rows if runtime_cfg else 60),
    )


def _build_training_cfg(
    dataset_path: Path,
    model_output_path: Path,
    metrics_path: Path,
    runtime_cfg: Optional[TinyMLRuntimeConfig],
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    validation_split: Optional[float],
    patience: Optional[int],
) -> TinyMLTrainingConfig:
    return TinyMLTrainingConfig(
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        metrics_path=metrics_path,
        epochs=epochs or 120,
        batch_size=batch_size or 32,
        validation_split=validation_split or 0.2,
        learning_rate=learning_rate or 0.005,
        patience=patience or 8,
    )


def _export_dataset(conn: sqlite3.Connection, dataset_cfg: TinyMLDatasetConfig) -> int:
    builder = TinyMLDatasetBuilder(conn, dataset_cfg)
    return builder.export()


def _train_model(
    conn: sqlite3.Connection,
    cfg: GatewayConfig,
    ala_id: str,
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
    args: argparse.Namespace,
) -> None:
    runtime_cfg = _resolve_tinyml_cfg(cfg)
    dataset_cfg = _build_dataset_cfg(
        cfg=cfg,
        ala_id=ala_id,
        dataset_path=dataset_path,
        horizon_minutes=args.horizon_minutes,
        flux_window_seconds=args.flux_window_seconds,
        min_samples=args.min_samples,
    )
    if not args.skip_export:
        rows = _export_dataset(conn, dataset_cfg)
        if rows == 0:
            LOGGER.warning("Dataset vazio; treino pode falhar por falta de dados.")
    trainer = TinyMLTrainer(
        _build_training_cfg(
            dataset_path=dataset_path,
            model_output_path=model_path,
            metrics_path=metrics_path,
            runtime_cfg=runtime_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            patience=args.patience,
        )
    )
    metrics = trainer.train()
    LOGGER.info("Treino TinyML concluído. Métricas: %s", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline TinyML (dataset + treino)")
    parser.add_argument("--config", type=Path, required=True, help="Ficheiro YAML da configuração do gateway.")
    parser.add_argument("--ala", type=str, help="ID da ala (override).")
    parser.add_argument("--verbose", action="store_true", help="Ativa logging em nível DEBUG.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Exporta dataset para CSV.")
    export_parser.add_argument("--output", type=Path, help="Destino do dataset CSV.")
    export_parser.add_argument("--horizon-minutes", type=int)
    export_parser.add_argument("--flux-window-seconds", type=int)
    export_parser.add_argument("--min-samples", type=int)

    train_parser = subparsers.add_parser("train", help="Treina modelo TFLite.")
    train_parser.add_argument("--dataset", type=Path, help="Dataset CSV (default = tinyml.dataset_path).")
    train_parser.add_argument("--output", type=Path, help="Destino do modelo .tflite.")
    train_parser.add_argument("--metrics", type=Path, help="Ficheiro JSON para métricas.")
    train_parser.add_argument("--horizon-minutes", type=int)
    train_parser.add_argument("--flux-window-seconds", type=int)
    train_parser.add_argument("--min-samples", type=int)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--validation-split", type=float)
    train_parser.add_argument("--patience", type=int)
    train_parser.add_argument("--skip-export", action="store_true", help="Não reconstruir dataset antes do treino.")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)
    ala_id = _resolve_ala(cfg, args.ala)

    dataset_default = cfg.tinyml.dataset_path if cfg.tinyml else Path("data/tinyml_dataset.csv")
    model_default = cfg.tinyml.model_path if cfg.tinyml else Path("models/tinyml_model.tflite")
    metrics_default = cfg.tinyml.metrics_path if cfg.tinyml else Path("data/tinyml_metrics.json")

    if args.command == "export":
        dataset_path = args.output or dataset_default
    else:
        dataset_path = args.dataset or dataset_default
    model_path = getattr(args, "output", None) or model_default
    metrics_path = getattr(args, "metrics", None) or metrics_default

    conn = init_sqlite(cfg.sqlite_db_path)
    try:
        if args.command == "export":
            dataset_cfg = _build_dataset_cfg(
                cfg=cfg,
                ala_id=ala_id,
                dataset_path=dataset_path,
                horizon_minutes=args.horizon_minutes,
                flux_window_seconds=args.flux_window_seconds,
                min_samples=args.min_samples,
            )
            rows = _export_dataset(conn, dataset_cfg)
            LOGGER.info("Dataset gerado com %s linhas em %s", rows, dataset_path)
        elif args.command == "train":
            _train_model(
                conn=conn,
                cfg=cfg,
                ala_id=ala_id,
                dataset_path=dataset_path,
                model_path=model_path,
                metrics_path=metrics_path,
                args=args,
            )
        else:  # pragma: no cover - argparse garante comandos válidos
            parser.print_help()
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()


