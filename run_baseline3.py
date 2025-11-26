# -*- coding: utf-8 -*-
"""
Baseline 3 (Pairwise + BT) — Robust version with:
- A/B randomization and strict parsing
- Anchor–anchor calibration edges
- Pair logs CSV for self-check
- Multiple BT fits (ILSR, MM if available)
- Two mappings (Linear vs Isotonic) + Borda fallback on collapse
- QWK computed per method and saved via your save_results()
"""

import time
import re
import random
import csv
import os
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import choix
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from common import (
    target_df, train_df, essay_col, score_col, id_col,
    call_llm, save_results, PROVIDER, HOLISTIC_RUBRIC_FULL
)

# -----------------------------
# Utility: Quadratic Weighted Kappa
# -----------------------------
def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], min_rating=1, max_rating=6) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cats = np.arange(min_rating, max_rating + 1)
    n_cat = len(cats)
    # confusion matrix O
    O = np.zeros((n_cat, n_cat), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t - min_rating, p - min_rating] += 1.0

    # histograms
    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    E = np.outer(hist_t, hist_p) / max(1.0, O.sum())

    # weights
    W = np.zeros((n_cat, n_cat), dtype=float)
    for i in range(n_cat):
        for j in range(n_cat):
            W[i, j] = ((i - j) ** 2) / ((n_cat - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum() if (W * E).sum() > 0 else 1.0
    return 1.0 - num / den

# -----------------------------
# Strict parser for "Essay A"/"Essay B"
# -----------------------------
PAT_AB = re.compile(r'^\s*(Essay\s+[AB])\s*$', flags=re.I | re.M)

def parse_winner(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = PAT_AB.search(raw.strip())
    if not m:
        return None
    ans = m.group(1).strip().upper()  # 'ESSAY A' or 'ESSAY B'
    # Normalize to "Essay A"/"Essay B"
    return "Essay A" if "A" in ans else "Essay B"

# -----------------------------
# BT helpers
# -----------------------------
def fit_bt_params_ilsr(n_items: int, pairs: List[Tuple[int, int]], alpha: float = 0.01) -> Optional[np.ndarray]:
    try:
        return choix.ilsr_pairwise(n_items, pairs, alpha=alpha)
    except Exception as e:
        print(f"[BT][ILSR] Failed: {e}")
        return None

def fit_bt_params_mm(n_items: int, pairs: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    # mm_pairwise may not be available in all choix versions
    try:
        mm = getattr(choix, "mm_pairwise", None)
        if mm is None:
            print("[BT][MM] choix.mm_pairwise not available; skipping.")
            return None
        return mm(n_items, pairs)
    except Exception as e:
        print(f"[BT][MM] Failed: {e}")
        return None

def linear_mapping(anc_x: np.ndarray, anc_y: np.ndarray):
    reg = LinearRegression().fit(anc_x.reshape(-1, 1), anc_y)
    def f(z):
        return float(reg.predict(np.array([[float(z)]]))[0])
    return f

def isotonic_mapping(anc_x: np.ndarray, anc_y: np.ndarray):
    # De-duplicate / jitter equal x to avoid iso errors
    order = np.argsort(anc_x)
    anc_x_sorted = anc_x[order].astype(float).copy()
    anc_y_sorted = anc_y[order].astype(float).copy()
    eps = 1e-8
    for i in range(1, len(anc_x_sorted)):
        if abs(anc_x_sorted[i] - anc_x_sorted[i - 1]) < eps:
            anc_x_sorted[i] = anc_x_sorted[i - 1] + eps
    iso = IsotonicRegression(y_min=1.0, y_max=6.0, increasing=True, out_of_bounds='clip')
    iso.fit(anc_x_sorted, anc_y_sorted)
    def f(z):
        return float(iso.predict([float(z)])[0])
    return f

def clamp_round_1_6(x: float) -> int:
    return int(max(1, min(6, round(x))))

# -----------------------------
# Save pairwise logs for self-check
# -----------------------------
def save_pairs_csv(pairs_rows: List[Dict], tag: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_Pairs_{ts}.csv")
    df = pd.DataFrame(pairs_rows)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[PAIRS] Saved pairwise log -> {path}")

# -----------------------------
# Main runner
# -----------------------------
def run():
    print(f"\n🚀 Starting Baseline 3 (Pairwise + Robust BT)...")
    print(f"⚠️ CAUTION: This baseline may trigger many API calls.")

    # 1) Prepare single anchor per score in [1..6] (full text)
    anchors = {}
    for s in range(1, 7):
        cand = train_df[train_df[score_col] == s]
        if not cand.empty:
            anchors[s] = {
                'id': cand.iloc[0][id_col],
                'text': str(cand.iloc[0][essay_col]),
                'score': int(s),
            }
        else:
            anchors[s] = {'id': f"mock_{s}", 'text': "Mock Full Text", 'score': int(s)}

    # Global item list (targets + anchors)
    all_ids = list(target_df[id_col].unique()) + [a['id'] for a in anchors.values()]
    all_ids = list(dict.fromkeys(all_ids))  # keep order, drop dup
    id_map = {uid: i for i, uid in enumerate(all_ids)}
    inv_id = {v: k for k, v in id_map.items()}

    # Stats and logs
    comparisons: List[Tuple[int, int]] = []  # (winner_idx, loser_idx)
    pairs_rows: List[Dict] = []
    token_sum_per_target: Dict[str, int] = defaultdict(int)
    latency_sum_per_target: Dict[str, float] = defaultdict(float)
    parsed_A = 0
    parsed_B = 0
    parsed_none = 0

    # 2) Collect pairwise comparisons (Target vs each Anchor), with A/B randomization
    print("Collecting target–anchor comparisons with A/B randomization + strict parsing...")
    for idx, row in target_df.iterrows():
        t_id = row[id_col]
        t_essay = str(row[essay_col])

        essay_start = time.time()
        essay_tokens = 0

        for s, anchor in anchors.items():
            a_id = anchor['id']
            if t_id == a_id:
                continue

            swap = (random.random() < 0.5)
            if not swap:
                A_txt, B_txt = t_essay, anchor['text']
                A_uid, B_uid = t_id, a_id
            else:
                A_txt, B_txt = anchor['text'], t_essay
                A_uid, B_uid = a_id, t_id

            prompt = f"""System: Compare based on Official Rubric.

### RUBRIC
{HOLISTIC_RUBRIC_FULL}

### ESSAY A
{A_txt}

### ESSAY B
{B_txt}

### TASK
Output EXACTLY one token: "Essay A" or "Essay B"."""
            st = time.time()
            raw_out, in_tok = call_llm(prompt, max_tokens=4)
            lt = time.time() - st
            essay_tokens += (in_tok or 0)

            parsed = parse_winner(raw_out or "")
            if parsed == "Essay A":
                win_uid, lose_uid = A_uid, B_uid
                parsed_A += 1
            elif parsed == "Essay B":
                win_uid, lose_uid = B_uid, A_uid
                parsed_B += 1
            else:
                win_uid, lose_uid = None, None
                parsed_none += 1

            # Log one row for self-check
            pairs_rows.append({
                "target_id": t_id,
                "anchor_score": s,
                "anchor_id": a_id,
                "swap_AB": int(swap),  # 0: target=A, 1: target=B
                "A_uid": A_uid, "B_uid": B_uid,
                "winner_uid": win_uid, "loser_uid": lose_uid,
                "parsed_label": (parsed or "INVALID"),
                "input_tokens": int(in_tok or 0),
                "latency": round(lt, 3),
            })

            if parsed in ("Essay A", "Essay B"):
                comparisons.append((id_map[win_uid], id_map[lose_uid]))

            # polite rate-limit
            if PROVIDER != "MOCK":
                time.sleep(0.2)

        latency_sum_per_target[t_id] += time.time() - essay_start
        token_sum_per_target[t_id] += essay_tokens

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(target_df)} targets...")

    # 3) Add anchor–anchor calibration (AB/BA each once)
    print("Adding anchor–anchor calibration edges...")
    anchor_pairs = [(1,2),(2,3),(3,4),(4,5),(5,6)]
    for s1, s2 in anchor_pairs:
        for swap in [0, 1]:
            if swap == 0:
                A_txt, B_txt = anchors[s1]['text'], anchors[s2]['text']
                A_uid, B_uid = anchors[s1]['id'], anchors[s2]['id']
            else:
                A_txt, B_txt = anchors[s2]['text'], anchors[s1]['text']
                A_uid, B_uid = anchors[s2]['id'], anchors[s1]['id']

            prompt = f"""System: Compare based on Official Rubric.

### RUBRIC
{HOLISTIC_RUBRIC_FULL}

### ESSAY A
{A_txt}

### ESSAY B
{B_txt}

### TASK
Output EXACTLY one token: "Essay A" or "Essay B"."""
            st = time.time()
            raw_out, in_tok = call_llm(prompt, max_tokens=4)
            lt = time.time() - st

            parsed = parse_winner(raw_out or "")
            if parsed == "Essay A":
                win_uid, lose_uid = A_uid, B_uid
                parsed_A += 1
            elif parsed == "Essay B":
                win_uid, lose_uid = B_uid, A_uid
                parsed_B += 1
            else:
                win_uid, lose_uid = None, None
                parsed_none += 1

            pairs_rows.append({
                "target_id": None,
                "anchor_score": f"{s1}vs{s2}",
                "anchor_id": None,
                "swap_AB": int(swap),
                "A_uid": A_uid, "B_uid": B_uid,
                "winner_uid": win_uid, "loser_uid": lose_uid,
                "parsed_label": (parsed or "INVALID"),
                "input_tokens": int(in_tok or 0),
                "latency": round(lt, 3),
            })

            if parsed in ("Essay A", "Essay B"):
                comparisons.append((id_map[win_uid], id_map[lose_uid]))

            if PROVIDER != "MOCK":
                time.sleep(0.2)

    # 4) Save pair logs CSV (self-check file)
    save_pairs_csv(pairs_rows, tag="Baseline3_Pairwise_FullContext")

    # Quick console self-check
    print(f"[SELF-CHECK] Parsed A: {parsed_A}, Parsed B: {parsed_B}, INVALID: {parsed_none}")
    print(f"[SELF-CHECK] Total comparisons kept: {len(comparisons)}")

    if not comparisons:
        print("No valid comparisons collected; aborting BT.")
        return

    # 5) Fit BT parameters by multiple methods
    n_items = len(all_ids)
    bt_methods = {}

    print("[BT] Fitting ILSR(alpha=0.01)...")
    params_ilsr = fit_bt_params_ilsr(n_items, comparisons, alpha=0.01)
    if params_ilsr is not None:
        bt_methods["BT_ILSR"] = params_ilsr

    print("[BT] Fitting ILSR(alpha=0.001)...")
    params_ilsr2 = fit_bt_params_ilsr(n_items, comparisons, alpha=0.001)
    if params_ilsr2 is not None:
        bt_methods["BT_ILSR_A1E-3"] = params_ilsr2

    print("[BT] Fitting MM (if available)...")
    params_mm = fit_bt_params_mm(n_items, comparisons)
    if params_mm is not None:
        bt_methods["BT_MM"] = params_mm

    if not bt_methods:
        print("All BT fits failed; aborting.")
        return

    # 6) Prepare anchor latent → score mappings, run predictions, compute QWK, and save
    anchor_ids = [anchors[s]['id'] for s in range(1, 7)]
    anchor_scores = np.array([s for s in range(1, 7)], dtype=float)

    # Build ground-truth vectors
    gt_scores = [int(row[score_col]) for _, row in target_df.iterrows()]
    tgt_ids_in_order = [row[id_col] for _, row in target_df.iterrows()]

    def method_predict_and_save(method_name: str, params: np.ndarray):
        # collect anchor latents
        anc_x = np.array([params[id_map[uid]] for uid in anchor_ids], dtype=float)
        std_x = float(np.std(anc_x))
        print(f"[{method_name}] Anchor latent std = {std_x:.6e}")

        predictions_linear = []
        predictions_iso = []
        predictions_borda = []

        # If collapsed -> Borda fallback
        collapsed = (std_x < 1e-6)

        # Create mappings
        if not collapsed:
            f_lin = linear_mapping(anc_x, anchor_scores)
            f_iso = isotonic_mapping(anc_x, anchor_scores)

        # Also prepare a simple Borda-like count for fallback/diagnostics
        # wins_by_target counts wins over anchors only
        wins_by_target = Counter()
        anchor_set = set(anchor_ids)
        for w_idx, l_idx in comparisons:
            w_uid, l_uid = inv_id[w_idx], inv_id[l_idx]
            if (w_uid in tgt_ids_in_order) and (l_uid in anchor_set):
                wins_by_target[w_uid] += 1

        for uid in tgt_ids_in_order:
            z = float(params[id_map[uid]])
            if collapsed:
                pred_lin = pred_iso = None
                pred_borda = clamp_round_1_6(1 + wins_by_target.get(uid, 0))
            else:
                pred_lin = clamp_round_1_6(f_lin(z))
                pred_iso = clamp_round_1_6(f_iso(z))
                pred_borda = clamp_round_1_6(1 + wins_by_target.get(uid, 0))

            predictions_linear.append(pred_lin if pred_lin is not None else pred_borda)
            predictions_iso.append(pred_iso if pred_iso is not None else pred_borda)
            predictions_borda.append(pred_borda)

        # Compute QWK
        qwk_lin = quadratic_weighted_kappa(gt_scores, predictions_linear, 1, 6)
        qwk_iso = quadratic_weighted_kappa(gt_scores, predictions_iso, 1, 6)
        qwk_bor = quadratic_weighted_kappa(gt_scores, predictions_borda, 1, 6)
        print(f"[{method_name}] QWK-linear = {qwk_lin:.4f}, QWK-isotonic = {qwk_iso:.4f}, QWK-borda = {qwk_bor:.4f}")

        # Save three result files via your save_results()
        def pack_results(pred_list, suffix):
            results = []
            for uid, gt, pred in zip(tgt_ids_in_order, gt_scores, pred_list):
                results.append({
                    'essay_id': uid,
                    'human_score': gt,
                    'pred_score': pred,
                    'input_tokens': int(token_sum_per_target.get(uid, 0)),
                    'latency': round(float(latency_sum_per_target.get(uid, 0.0)), 2),
                })
            save_results(results, f"Baseline3_{method_name}_{suffix}")

        pack_results(predictions_linear, "Linear")
        pack_results(predictions_iso, "Isotonic")
        pack_results(predictions_borda, "Borda")

    for mname, mparams in bt_methods.items():
        method_predict_and_save(mname, mparams)

if __name__ == "__main__":
    run()
