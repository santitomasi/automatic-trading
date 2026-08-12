"""
=============================================================================
 BACKTEST DE SCORE - ¿Las señales 4/4 rinden mas que las 3/4?
=============================================================================
 Responde dos preguntas con 7 años de historia (2018-2024):

   1. ¿El expectancy de las señales 4/4 es mayor que el de las 3/4?
      (Es la hipotesis detras del riesgo dinamico del paper trader v3.)
   2. ¿Cuanto habria rendido el riesgo dinamico {3/4->1%, 4/4->2%}
      comparado con el 1% fijo?

 Usa la regla LARGA intacta de screener.py, con costes, igual que el
 backtest v3. Solo añade el registro del score en cada operacion.
=============================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from screener import enrich, evaluate, CONFIG


BTS_CONFIG = {
    **CONFIG,
    "start_equity": 10000.0,
    "spread_pct": 0.02,
    "commission_pct": 0.01,
    "slippage_pct": 0.02,
    "start": "2018-01-01",
    "end": "2024-12-31",
}


def cost_fraction(cfg):
    return (cfg["spread_pct"] + cfg["commission_pct"] + cfg["slippage_pct"]) / 100.0


# =============================================================================
# SIMULACIÓN (larga, con costes, registrando el score de cada entrada)
# =============================================================================

def backtest_with_score(df, cfg):
    trades = []
    pos = None
    start = cfg["ema_trend"]
    cost = cost_fraction(cfg)

    for i in range(start, len(df)):
        bar = df.iloc[i]

        if pos is not None:
            hit_sl = bar["Low"] <= pos["sl"]
            hit_tp = bar["High"] >= pos["tp"]
            exit_price = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else None)

            if exit_price is not None:
                risk = pos["entry"] - pos["sl"]
                eff_entry = pos["entry"] * (1 + cost)
                eff_exit  = exit_price * (1 - cost)
                r = (eff_exit - eff_entry) / risk
                trades.append({"score": pos["score"], "R": round(r, 3)})
                pos = None

        if pos is None:
            res = evaluate(df.iloc[: i + 1], cfg)
            if res and res["is_signal"]:
                pos = {"entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
                       "score": res["score"]}
    return trades


# =============================================================================
# ANÁLISIS POR SCORE
# =============================================================================

def group_stats(trades_df):
    """Metricas por grupo de score."""
    out = {}
    for score, g in trades_df.groupby("score"):
        wins = g[g["R"] > 0]
        losses = g[g["R"] <= 0]
        gw, gl = wins["R"].sum(), abs(losses["R"].sum())
        out[int(score)] = {
            "n": len(g),
            "win_rate": round(len(wins) / len(g) * 100, 1),
            "expectancy": round(g["R"].mean(), 3),
            "pf": round(gw / gl, 2) if gl > 0 else float("inf"),
        }
    return out


def simulate_sizing(trades_df, risk_map, start_equity):
    """Curva de equity aplicando un mapa score->riesgo, en orden cronologico."""
    equity = start_equity
    peak, max_dd = equity, 0
    for _, t in trades_df.iterrows():
        frac = risk_map.get(int(t["score"]), 0.01)
        equity += equity * frac * t["R"]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
    ret = (equity - start_equity) / start_equity * 100
    return round(ret, 1), round(max_dd, 1)


# =============================================================================
# ORQUESTADOR
# =============================================================================

def run(cfg=BTS_CONFIG):
    print(f"Backtest de score: {len(cfg['tickers'])} tickers, {cfg['start']} a {cfg['end']}...\n")

    all_trades = []
    for t in cfg["tickers"]:
        try:
            d = yf.Ticker(t).history(start=cfg["start"], end=cfg["end"], interval=cfg["interval"])
            if d.empty:
                continue
        except Exception as e:
            print(f"  [!] {t}: {e}")
            continue
        d = enrich(d, cfg)
        all_trades += backtest_with_score(d, cfg)

    if not all_trades:
        print("Sin operaciones. Revisa tickers/fechas.")
        return

    df = pd.DataFrame(all_trades)
    stats = group_stats(df)

    print("=" * 66)
    print("  RESULTADO: SEÑALES 3/4 vs 4/4  (larga, con costes, 2018-2024)")
    print("=" * 66)
    print(f"\n  {'Score':<8} {'Ops':>6} {'Acierto':>9} {'Expectancy':>11} {'ProfFactor':>11}")
    print("  " + "-" * 50)
    for score in sorted(stats):
        s = stats[score]
        print(f"  {score}/4{'':<4} {s['n']:>6} {s['win_rate']:>8}% {s['expectancy']:>11} {s['pf']:>11}")

    # Veredicto sobre la hipotesis
    print("\n  " + "-" * 50)
    if 3 in stats and 4 in stats:
        e3, e4 = stats[3]["expectancy"], stats[4]["expectancy"]
        n4 = stats[4]["n"]
        if n4 < 30:
            print(f"  OJO: solo {n4} operaciones 4/4 -> muestra chica, conclusion fragil.")
        if e4 > e3 * 1.15:
            print(f"  VEREDICTO: las 4/4 SI rinden mas ({e4} vs {e3}).")
            print("  El riesgo dinamico tiene sustento historico.")
        elif e4 < e3 * 0.85:
            print(f"  VEREDICTO: las 4/4 rinden PEOR ({e4} vs {e3}).")
            print("  El riesgo dinamico NO tiene sustento -> conviene revertir a 1% fijo.")
        else:
            print(f"  VEREDICTO: sin diferencia clara ({e4} vs {e3}).")
            print("  El riesgo dinamico ni suma ni resta edge -> simplicidad (1% fijo) gana.")

    # Comparacion de sizing
    print("\n  SIZING SOBRE LAS MISMAS OPERACIONES:")
    flat_ret, flat_dd = simulate_sizing(df, {3: 0.01, 4: 0.01}, cfg["start_equity"])
    dyn_ret, dyn_dd = simulate_sizing(df, {3: 0.01, 4: 0.02}, cfg["start_equity"])
    print(f"    1% fijo:              retorno {flat_ret:>8}%  | max DD {flat_dd}%")
    print(f"    dinamico (1%/2%):     retorno {dyn_ret:>8}%  | max DD {dyn_dd}%")
    print("    (Secuencial, sin tope de cartera: compara sizing, no predice retornos.)")
    print("=" * 66 + "\n")
    return stats


if __name__ == "__main__":
    run()
