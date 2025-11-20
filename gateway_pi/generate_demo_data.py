#!/usr/bin/env python3
"""Gera dados sintéticos para o dataset TinyML e para testes do gateway.

Uso típico no Raspberry Pi:

    source /home/pi/IOT/.venv/bin/activate
    python gateway_pi/generate_demo_data.py --config gateway_pi/config.yaml --minutes 360 --clear

Isto cria (~6 h) de registos nas tabelas SQLite (`state_snapshots`, `ala_events`,
`air_samples`, `fan_events`) com valores plausíveis de ocupação, fluxos e qualidade
do ar, permitindo treinar o modelo TinyML e testar a pipeline fim-a-fim.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from main import GatewayConfig, init_sqlite, load_config

TABLES = [
    "place_events",
    "ala_events",
    "air_samples",
    "fan_events",
    "state_snapshots",
    "tinyml_predictions",
]


def _choose_ala(config: GatewayConfig, ala: Optional[str]) -> str:
    if ala:
        return ala
    if config.tinyml and config.tinyml.ala_id:
        return config.tinyml.ala_id
    if config.alas:
        return next(iter(config.alas.keys()))
    raise ValueError("Nenhuma ala encontrada na configuração.")


def _ensure_random_seed(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)


def purge_tables(conn: sqlite3.Connection, ala_id: str) -> None:
    """Apaga registos da ala especificada (ajustando coluna conforme tabela)."""
    conn.execute("DELETE FROM state_snapshots WHERE ala_id = ?", (ala_id,))
    conn.execute("DELETE FROM ala_events WHERE ala_id = ?", (ala_id,))
    conn.execute("DELETE FROM air_samples WHERE ala_id = ?", (ala_id,))
    conn.execute("DELETE FROM fan_events WHERE ala_id = ?", (ala_id,))
    conn.execute("DELETE FROM tinyml_predictions WHERE ala_id = ?", (ala_id,))
    conn.execute("DELETE FROM daily_forecasts WHERE ala_id = ?", (ala_id,))
    # place_events usa lugar_id (ex: "A-01"), não ala_id
    conn.execute("DELETE FROM place_events WHERE lugar_id LIKE ?", (f"{ala_id}-%",))
    conn.commit()


def generate_data(
    conn: sqlite3.Connection,
    config: GatewayConfig,
    ala_id: str,
    minutes: int,
    start_ago_minutes: int,
) -> None:
    capacidade = config.alas[ala_id].capacidade_maxima
    if capacidade <= 0:
        raise ValueError("capacidade_maxima deve ser > 0 para gerar dados TinyML.")

    base_time = datetime.now(timezone.utc) - timedelta(minutes=start_ago_minutes)
    ocupacao = 0
    soma_lugares = 0
    alerta_sensor = False
    mismatch_since: Optional[datetime] = None
    diferenca_limite = config.alas[ala_id].safety.mismatch_delta if config.alas[ala_id].safety else 2
    mismatch_duracao = (
        config.alas[ala_id].safety.mismatch_duration_seconds if config.alas[ala_id].safety else 5
    )
    mismatch_duracao_td = timedelta(seconds=mismatch_duracao)

    for minute in range(minutes):
        ts = base_time + timedelta(minutes=minute)
        timestamp_ms = int(ts.timestamp() * 1000)
        created_at = ts.isoformat()

        # Variação da ocupação simulada
        step = random.choices([-1, 0, 1], weights=[0.25, 0.55, 0.20])[0]
        novo_valor = max(0, min(capacidade, ocupacao + step))
        if novo_valor > ocupacao:
            ocupacao = novo_valor
            conn.execute(
                "INSERT INTO ala_events (ala_id, evento, total, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ala_id, "entrada", ocupacao, timestamp_ms, created_at),
            )
        elif novo_valor < ocupacao:
            ocupacao = novo_valor
            conn.execute(
                "INSERT INTO ala_events (ala_id, evento, total, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ala_id, "saida", ocupacao, timestamp_ms, created_at),
            )

        # Fluxos recentes para features
        flux_entrada = random.randint(0, 2) if step > 0 else random.randint(0, 1)
        flux_saida = random.randint(0, 2) if step < 0 else random.randint(0, 1)

        # Soma de lugares com ligeira divergência
        soma_lugares = max(0, min(capacidade, ocupacao + random.choice([-1, 0, 0, 1])))

        # Qualidade do ar simulada (quanto mais ocupação, pior)
        qualidade_ar_bruto = max(200, min(900, int(220 + ocupacao * 8 + random.gauss(0, 20))))
        qualidade_ar_tensao = round((qualidade_ar_bruto / 1023.0) * 5.0, 3)
        conn.execute(
            "INSERT INTO air_samples (ala_id, valor_bruto, tensao, timestamp_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (ala_id, qualidade_ar_bruto, qualidade_ar_tensao, timestamp_ms, created_at),
        )

        # Percentagem recomendada simples (antes de TinyML)
        ventoinha_percent = max(35, min(100, int((ocupacao / capacidade) * 100)))
        if minute % 15 == 0 or step != 0:
            conn.execute(
                "INSERT INTO fan_events (ala_id, percent, timestamp_ms, created_at)"
                " VALUES (?, ?, ?, ?)",
                (ala_id, ventoinha_percent, timestamp_ms, created_at),
            )

        # Gestão de alerta sensor (divergência persistente)
        delta = abs(ocupacao - soma_lugares)
        if delta > diferenca_limite:
            if mismatch_since is None:
                mismatch_since = ts
            elif ts - mismatch_since >= mismatch_duracao_td:
                alerta_sensor = True
        else:
            mismatch_since = None
            if alerta_sensor:
                alerta_sensor = False

        conn.execute(
            "INSERT INTO state_snapshots (ala_id, ocupacao_ala, soma_lugares, qualidade_ar_bruto,"
            " qualidade_ar_tensao, ventoinha_percent, alerta_sensor, timestamp_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ala_id,
                ocupacao,
                soma_lugares,
                qualidade_ar_bruto,
                qualidade_ar_tensao,
                ventoinha_percent,
                int(alerta_sensor),
                timestamp_ms,
                created_at,
            ),
        )

        # Atualiza métricas de fluxo nos eventos (opcional)
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerador de dados sintéticos para TinyML.")
    parser.add_argument("--config", type=Path, required=True, help="Ficheiro de configuração YAML.")
    parser.add_argument("--ala", type=str, help="ID da ala (override).")
    parser.add_argument("--minutes", type=int, default=360, help="Quantidade de minutos a gerar (default: 360).")
    parser.add_argument(
        "--start-ago-minutes",
        type=int,
        default=0,
        help="Começar a geração 'n' minutos atrás (default: 0 = começa agora).",
    )
    parser.add_argument("--seed", type=int, help="Seed opcional para random.")
    parser.add_argument("--clear", action="store_true", help="Limpa dados existentes dessa ala antes de gerar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_random_seed(args.seed)

    config = load_config(args.config)
    ala_id = _choose_ala(config, args.ala)
    conn = init_sqlite(config.sqlite_db_path)
    try:
        if args.clear:
            purge_tables(conn, ala_id)
            print(f"[INFO] Dados anteriores da ala {ala_id} removidos.")
        generate_data(conn, config, ala_id, args.minutes, args.start_ago_minutes or args.minutes)
        print(f"[OK] Geradas {args.minutes} amostras sintéticas para a ala {ala_id}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


