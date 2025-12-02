
# -*- coding: utf-8 -*-
"""
run_baseline2.py (MTS with Dynamic Rubric → Dev Mapping → Isotonic)
- Step 0: 从官方 Holistic rubric 动态生成三维度(Dev/Org/Lang)的 1–6 锚点(JSON 缓存)；
          优先读取 /mnt/data/asap_scoring_rubric.docx，失败则回退到 common.HOLISTIC_RUBRIC_FULL。
- Step 1: 用该“生成版标准”对 dev split 打分，拟合 (D,O,L)->human 的线性映射；
- Step 2: 用 Isotonic 对线性输出做单调校准；
- Step 3: 在 target 集上输出三种预测（avg / linear / isotonic），并各自保存报告；
- 兼容原 save_results 与 MOCK/GEMINI/OPENAI 调用；保留原始简单平均输出。
- 新增：
  * API Key 注入：--openai-key / --google-key（或环境变量 OPENAI_API_KEY / GOOGLE_API_KEY）
  * provider / model 覆写：--provider {MOCK,OPENAI,GEMINI}，--model <name>
  * test mode：--test N 仅在前 N 条数据上快速跑通流程
"""

import os
import re
import time
import json
import argparse
import hashlib
import numpy as np
import pandas as pd

# 可选读取 docx（若环境无该包，将自动回退）
try:
    import docx  # python-docx
except Exception:
    docx = None

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score

# ========== 不覆盖外部已设置的环境变量；也不要默认塞“假 key” ==========
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GOOGLE_API_KEY", "")

# 之后再导入 common，便于在运行前通过 CLI 覆写其 PROVIDER/MODEL_NAME
import common as C  # 统一通过 C.* 访问：数据、列名、call_llm/save_results、HOLISTIC_RUBRIC_FULL 等

# ================= 调参 =================
DEV_LIMIT = 150          # 从训练集中抽多少篇作为 dev split 去拟合 (D,O,L)->human
FALLBACK_SCORE = 3       # 解析失败时的容错默认分（Trait 1-6）
CACHE_PATH = "generated_mts_rubric.json"

# ============== 动态 Rubric 生成（失败回退到内置） ==============
DEFAULT_MTS_RUBRIC = {
    "development": {
        "1": "No viable claim; little/no relevant evidence; reasoning absent.",
        "2": "Vague/seriously limited claim; insufficient or inappropriate evidence; weak reasoning.",
        "3": "Developing claim; thin or loosely connected evidence; uneven reasoning.",
        "4": "Plausible claim; adequate but sometimes generic evidence; competent reasoning.",
        "5": "Clear claim; specific and mostly relevant evidence; strong reasoning with minor lapses.",
        "6": "Insightful claim; precise, well-integrated evidence; strong and nuanced reasoning."
    },
    "organization": {
        "1": "Disorganized/unfocused; disjointed sequence; no sustained coherence.",
        "2": "Poorly organized; serious problems with coherence/focus; unclear sequencing.",
        "3": "Limited/inconsistent structure; lapses in coherence; weak/abrupt transitions.",
        "4": "Recognizable structure; workable sequencing; occasional coherence lapses.",
        "5": "Clearly organized and focused; coherent development; minor lapses.",
        "6": "Strong global structure; purposeful paragraphing; smooth progression and clear focus."
    },
    "language": {
        "1": "Fundamental/pervasive errors; severe sentence problems; meaning often impeded.",
        "2": "Very limited vocabulary or incorrect word choice; frequent sentence problems; many errors.",
        "3": "Sometimes weak/imprecise word choice; sentence problems; noticeable error accumulation.",
        "4": "Adequate control; some sentence variety; errors present but meaning generally clear.",
        "5": "Appropriate vocabulary; noticeable sentence variety; few distracting errors.",
        "6": "Skillful, precise language; apt vocabulary; meaningful sentence variety; very few errors."
    }
}

def _normalize_json_str(s: str) -> str:
    """尽可能从 LLM 文本中提取 JSON。"""
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    m = re.search(r"```json\s*([\s\S]*?)```", s, re.I)
    if m:
        cand = m.group(1).strip()
        try:
            obj = json.loads(cand)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}$", s.strip())
    if m:
        cand = m.group(0)
        try:
            obj = json.loads(cand)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        cand = s[first:last + 1]
        try:
            obj = json.loads(cand)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass

    raise ValueError("No valid JSON found in model output.")

def _hash_obj(obj) -> str:
    norm = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:10]

def _read_official_rubric_text() -> str:
    """
    优先读取你上传的 /mnt/data/asap_scoring_rubric.docx；
    若不可用，则回退到 C.HOLISTIC_RUBRIC_FULL。
    """
    p = "/mnt/data/asap_scoring_rubric.docx"
    if docx is not None and os.path.exists(p):
        try:
            d = docx.Document(p)
            return "\n".join([para.text for para in d.paragraphs if para.text.strip()])
        except Exception:
            pass
    return getattr(C, "HOLISTIC_RUBRIC_FULL", "")

def generate_trait_rubric(
    holistic_text: str,
    cache_path: str = CACHE_PATH,
    force_regen: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1200
):
    """
    让 LLM 把官方 Holistic rubric 转写为三维 rubric JSON。
    失败则回退 DEFAULT_MTS_RUBRIC。
    """
    if (not force_regen) and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for k in ("development", "organization", "language"):
                assert k in cached and all(str(i) in cached[k] for i in range(1, 7))
            print(f"[Rubric] Loaded cached rubric from {cache_path}")
            return cached
        except Exception:
            print("[Rubric] Cache exists but invalid, will regenerate...")

    schema_example = """
{
  "development": {"1": "...", "2": "...", "3": "...", "4": "...", "5": "...", "6": "..."},
  "organization": {"1": "...", "2": "...", "3": "...", "4": "...", "5": "...", "6": "..."},
  "language":    {"1": "...", "2": "...", "3": "...", "4": "...", "5": "...", "6": "..."}
}
""".strip()

    prompt = f"""You are an expert writing-assessment designer.
Convert the OFFICIAL HOLISTIC RUBRIC into a three-trait, 1–6 anchored rubric
with crisp, decisionable descriptions for each level.

Return STRICT JSON ONLY (no commentary), schema:
{schema_example}

Rules:
- Each level must be 1–2 sentences, observable from text.
- Keep concise. NO extra keys. NO markdown.

=== OFFICIAL HOLISTIC RUBRIC ===
{holistic_text}
"""
    raw, *_ = C.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
    try:
        norm = _normalize_json_str(raw)
        obj = json.loads(norm)
        for k in ("development", "organization", "language"):
            if k not in obj:
                raise ValueError(f"Missing key: {k}")
            for i in range(1, 7):
                if str(i) not in obj[k]:
                    raise ValueError(f"Missing level '{i}' in {k}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"[Rubric] Generated & cached -> {cache_path}")
        return obj
    except Exception as e:
        print(f"[Rubric] ❌ Generation failed: {e}. Fallback to DEFAULT_MTS_RUBRIC.")
        return DEFAULT_MTS_RUBRIC

def rubric_json_to_prompt_text(rj) -> str:
    def block(name_show: str, key: str) -> str:
        lines = [f"{name_show} (1–6):"]
        for i in range(1, 7):
            desc = rj.get(key, {}).get(str(i), "").strip()
            lines.append(f"- Score {i}: {desc}")
        return "\n".join(lines)

    dev = block("TRAIT 1: DEVELOPMENT (Ideas, Evidence, Critical Thinking)", "development")
    org = block("TRAIT 2: ORGANIZATION (Structure, Coherence, Focus)", "organization")
    lng = block("TRAIT 3: LANGUAGE (Vocabulary, Sentence Control, Mechanics)", "language")
    return dev + "\n\n" + org + "\n\n" + lng

def parse_mts_triplet(raw_text: str):
    """从 LLM 文本里解析 Development/Organization/Language 三个 1-6 分。"""
    def _one(key):
        m = re.search(rf"{key}\s*:\s*([1-6])", raw_text, re.I)
        return int(m.group(1)) if m else FALLBACK_SCORE

    d, o, l = _one("Development"), _one("Organization"), _one("Language")
    d = max(1, min(6, d))
    o = max(1, min(6, o))
    l = max(1, min(6, l))
    return d, o, l

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, round(float(x)))))

def fit_dev_mapping(dev_df: pd.DataFrame, rubric_text: str, rubric_id: str):
    """
    用动态 rubric 对 dev split 打分，拟合：
      - LinearRegression: y ~ [D,O,L]
      - IsotonicRegression: y ~ f(y_linear)
    返回 (lin, iso, score_min, score_max, dev_rows, dev_report)
    """
    X, y = [], []
    rows = []

    for _, r in dev_df.iterrows():
        prompt = f"""System: Expert MTS Scorer.

### RUBRIC (Generated 1–6 anchors)
{rubric_text}

### TARGET ESSAY
{r[C.essay_col]}

### TASK
Score 3 traits independently (1-6). Format EXACTLY as:
Development: [Score]
Organization: [Score]
Language: [Score]
"""
        start_t = time.time()
        out = C.call_llm(prompt, max_tokens=300)

        if isinstance(out, (list, tuple)) and len(out) >= 1:
            raw_out = out[0]
            in_tok = out[1] if len(out) >= 2 else 0
        else:
            raw_out, in_tok = str(out), 0

        latency = time.time() - start_t
        d, o, l = parse_mts_triplet(raw_out)

        human_score = int(r[C.score_col])
        rows.append({
            "essay_id": r[C.id_col],
            "human": human_score,
            "d": d, "o": o, "l": l,
            "rubric_id": rubric_id,
            "input_tokens": in_tok,
            "latency": round(latency, 2),
            "raw_output": raw_out[:2000]
        })
        X.append([d, o, l])
        y.append(human_score)

        if getattr(C, "PROVIDER", "MOCK") != "MOCK":
            time.sleep(1)

    if len(X) < 5:
        raise RuntimeError("Dev split too small for fitting. Increase DEV_LIMIT or check data.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    score_min = int(np.min(y))
    score_max = int(np.max(y))

    lin = LinearRegression().fit(X, y)
    y_hat_lin = lin.predict(X)

    iso = IsotonicRegression(
        y_min=float(score_min),
        y_max=float(score_max),
        increasing=True,
        out_of_bounds='clip'
    )
    iso.fit(y_hat_lin, y)

    y_true = y.astype(int)
    y_pred_avg = np.clip(np.rint(np.mean(X, axis=1)), 1, 6).astype(int)  # legacy: 1-6 average
    y_pred_lin = np.clip(np.rint(y_hat_lin), score_min, score_max).astype(int)
    y_pred_iso = np.clip(np.rint(iso.predict(y_hat_lin)), score_min, score_max).astype(int)

    qwk_avg = cohen_kappa_score(y_true, y_pred_avg, weights='quadratic')
    qwk_lin = cohen_kappa_score(y_true, y_pred_lin, weights='quadratic')
    qwk_iso = cohen_kappa_score(y_true, y_pred_iso, weights='quadratic')

    dev_report = {
        "n_dev": int(len(X)),
        "score_min": score_min,
        "score_max": score_max,
        "linear_coef": [float(c) for c in lin.coef_.tolist()],
        "linear_intercept": float(lin.intercept_),
        "qwk_dev_avg": float(qwk_avg),
        "qwk_dev_linear": float(qwk_lin),
        "qwk_dev_iso": float(qwk_iso),
    }
    return lin, iso, score_min, score_max, rows, dev_report

def score_targets_and_export(lin, iso, rubric_text: str, rubric_id: str, target_df: pd.DataFrame, score_min: int, score_max: int):
    results_legacy, results_linear, results_isotonic, rows_all = [], [], [], []

    for idx, row in target_df.iterrows():
        prompt = f"""System: Expert MTS Scorer.

### RUBRIC (Generated 1–6 anchors)
{rubric_text}

### TARGET ESSAY
{row[C.essay_col]}

### TASK
Score 3 traits independently (1-6). Format EXACTLY as:
Development: [Score]
Organization: [Score]
Language: [Score]
"""
        start_t = time.time()
        out = C.call_llm(prompt, max_tokens=300)

        if isinstance(out, (list, tuple)) and len(out) >= 1:
            raw_out = out[0]
            in_tok = out[1] if len(out) >= 2 else 0
        else:
            raw_out, in_tok = str(out), 0

        latency = time.time() - start_t
        d, o, l = parse_mts_triplet(raw_out)

        avg_pred = (d + o + l) / 3.0
        final_avg = clamp_int(avg_pred, 1, 6)

        lin_pred = float(lin.predict(np.array([[d, o, l]], dtype=float))[0])
        iso_pred = float(iso.predict([lin_pred])[0])

        final_lin = clamp_int(lin_pred, score_min, score_max)
        final_iso = clamp_int(iso_pred, score_min, score_max)

        human_score = int(row[C.score_col])

        results_legacy.append({
            'essay_id': row[C.id_col],
            'human_score': human_score,
            'pred_score': final_avg,
            'score_dev': d, 'score_org': o, 'score_lang': l,
            'rubric_id': rubric_id,
            'input_tokens': in_tok,
            'latency': round(latency, 2),
            'raw_output': raw_out[:2000]
        })
        results_linear.append({
            'essay_id': row[C.id_col],
            'human_score': human_score,
            'pred_score': final_lin,
            'score_dev': d, 'score_org': o, 'score_lang': l,
            'rubric_id': rubric_id,
            'input_tokens': in_tok,
            'latency': round(latency, 2),
            'raw_output': raw_out[:2000]
        })
        results_isotonic.append({
            'essay_id': row[C.id_col],
            'human_score': human_score,
            'pred_score': final_iso,
            'score_dev': d, 'score_org': o, 'score_lang': l,
            'rubric_id': rubric_id,
            'input_tokens': in_tok,
            'latency': round(latency, 2),
            'raw_output': raw_out[:2000]
        })
        rows_all.append({
            'essay_id': row[C.id_col],
            'human_score': human_score,
            'd': d, 'o': o, 'l': l,
            'pred_avg': final_avg,
            'pred_linear': final_lin,
            'pred_isotonic': final_iso,
            'rubric_id': rubric_id,
            'input_tokens': in_tok,
            'latency': round(latency, 2),
            'raw_output': raw_out[:2000]
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(target_df)}...")
        if getattr(C, "PROVIDER", "MOCK") != "MOCK":
            time.sleep(1)

    C.save_results(results_legacy, "Baseline2_MTS_FullContext")
    C.save_results(results_linear, "Baseline2_MTS_FullContext_CALIB_Linear")
    C.save_results(results_isotonic, "Baseline2_MTS_FullContext_CALIB_Isotonic")

    ts = time.strftime('%Y%m%d_%H%M%S')
    prefix = "MOCK_" if getattr(C, "PROVIDER", "MOCK") == "MOCK" else ""
    all_csv = f"{prefix}Baseline2_MTS_FullContext_ALL_{ts}.csv"
    pd.DataFrame(rows_all).to_csv(all_csv, index=False)
    print(f"[All-in-One] Saved -> {all_csv}")

    df_all = pd.DataFrame(rows_all)
    y_true = df_all['human_score'].astype(int).to_numpy()
    qwk_avg = cohen_kappa_score(y_true, df_all['pred_avg'].astype(int).to_numpy(), weights='quadratic')
    qwk_lin = cohen_kappa_score(y_true, df_all['pred_linear'].astype(int).to_numpy(), weights='quadratic')
    qwk_iso = cohen_kappa_score(y_true, df_all['pred_isotonic'].astype(int).to_numpy(), weights='quadratic')
    print(f"[QWK on target] avg(legacy 1-6)={qwk_avg:.4f} | linear={qwk_lin:.4f} | isotonic={qwk_iso:.4f}")

def run(force_regen: bool = False, test_limit: int = 0):
    print(f"\n🚀 Baseline 2 (Dynamic Rubric → Dev Mapping), provider={getattr(C, 'PROVIDER', 'MOCK')}, model={getattr(C, 'MODEL_NAME', 'N/A')}")

    official_text = _read_official_rubric_text()
    rubric_obj = generate_trait_rubric(
        holistic_text=official_text,
        cache_path=CACHE_PATH,
        force_regen=force_regen,
        temperature=0.0,
        max_tokens=1200
    )
    rubric_id = _hash_obj(rubric_obj)
    rubric_text = rubric_json_to_prompt_text(rubric_obj)
    print(f"[Rubric] rubric_id={rubric_id} (source={'DOCX' if official_text != getattr(C,'HOLISTIC_RUBRIC_FULL','') else 'common.HOLISTIC_RUBRIC_FULL'})")

    if len(C.train_df) == 0:
        raise RuntimeError("train_df is empty; please check data loading in common.py")

    if test_limit and test_limit > 0:
        train_use = C.train_df.head(test_limit).copy()
        target_use = C.target_df.head(test_limit).copy()
        print(f"[TEST MODE] Using first {test_limit} items for train/target quick run.")
    else:
        train_use = C.train_df
        target_use = C.target_df

    dev_df = train_use.sample(n=min(DEV_LIMIT, len(train_use)), random_state=42) if len(train_use) > DEV_LIMIT else train_use.copy()
    print(f"Fitting on dev split: n={len(dev_df)} (from {'subset' if test_limit else 'train set'})")

    lin_model, iso_model, score_min, score_max, dev_rows, dev_report = fit_dev_mapping(dev_df, rubric_text, rubric_id)

    ts = time.strftime('%Y%m%d_%H%M%S')
    prefix = "MOCK_" if getattr(C, "PROVIDER", "MOCK") == "MOCK" else ""
    dev_csv = f"{prefix}Baseline2_MTS_DevSplit_{ts}.csv"
    pd.DataFrame(dev_rows).to_csv(dev_csv, index=False)
    with open(f"{prefix}Baseline2_MTS_DevFitReport_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(dev_report, f, indent=2, ensure_ascii=False)

    print(f"[Dev-Fit] CSV -> {dev_csv}")
    print(f"[Dev-Fit] Report -> {prefix}Baseline2_MTS_DevFitReport_{ts}.json")
    print(f"[Dev-Fit] human score range on dev: [{score_min}, {score_max}]")
    print(f"[Dev-Fit] Linear coef={dev_report['linear_coef']}, intercept={dev_report['linear_intercept']:.4f}")
    print(f"[Dev-Fit] QWK(dev): avg(legacy 1-6)={dev_report['qwk_dev_avg']:.4f} | linear={dev_report['qwk_dev_linear']:.4f} | isotonic={dev_report['qwk_dev_iso']:.4f}")

    score_targets_and_export(lin_model, iso_model, rubric_text, rubric_id, target_use, score_min, score_max)

def parse_args():
    p = argparse.ArgumentParser(description="Baseline2: Dynamic rubric → (D,O,L) → Linear → Isotonic")
    p.add_argument("--force-regen", action="store_true", help="不走缓存，强制重新生成三维度锚点")
    p.add_argument("--test", type=int, default=0, help="TEST MODE：仅用前 N 条 train/target 快速跑通")
    p.add_argument("--provider", type=str, default=None, choices=["MOCK", "OPENAI", "GEMINI"], help="覆写 common.PROVIDER")
    p.add_argument("--model", type=str, default=None, help="覆写 common.MODEL_NAME（若 common 提供该变量）")
    p.add_argument("--openai-key", type=str, default=None, help="覆盖 OPENAI_API_KEY")
    p.add_argument("--google-key", type=str, default=None, help="覆盖 GOOGLE_API_KEY")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key.strip()
    if args.google_key:
        os.environ["GOOGLE_API_KEY"] = args.google_key.strip()

    if args.provider is not None:
        try:
            C.PROVIDER = args.provider
        except Exception:
            pass
    if args.model is not None and hasattr(C, "MODEL_NAME"):
        try:
            C.MODEL_NAME = args.model
        except Exception:
            pass

    run(force_regen=args.force_regen, test_limit=args.test)
