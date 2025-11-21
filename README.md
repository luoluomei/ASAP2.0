# ASAP 2.0 Evaluator Alignment Baselines

This repository contains the implementation of three LLM-based automated essay scoring (AES) baselines for the **ASAP 2.0 dataset**. The goal of this project is to benchmark different prompting strategies—**Holistic Scoring**, **Multi-Trait Scoring**, and **Pairwise Comparison**—to evaluate their alignment with human judgments (Ground Truth).

## Task Overview for Team Members

Your primary task is to execute these baselines on the ASAP 2.0 **Test Set** to generate performance metrics (QWK, Accuracy, etc.) and cost analysis.

1.  **Configure your Environment:** Set up the API provider you intend to use.  
2.  **Run Baselines 1, 2, and 3:** Execute the scripts to perform inference on the full dataset.  
3.  **Collect Results:** Submit the generated `.csv` data logs and `.txt` summary reports.

---

## Configuration & API Setup

All configuration settings, including data loading and API wrappers, are centralized in `common.py`. **You do not need to modify the individual baseline scripts.**

### 1. Select Your Provider

Open `common.py` and locate the configuration section at the top:

```python
# common.py

# MODE SELECTION
# Options: "MOCK" (Testing), "GEMINI", "OPENAI"
PROVIDER = "MOCK" 

# MODEL CONFIGURATION
# Gemini: "gemini-2.0-flash" (Recommended for speed/cost)
# OpenAI: "gpt-4o"
MODEL_NAME = "gemini-2.0-flash"
```

MOCK: Use this first to verify the code runs without errors. It uses zero tokens and returns random dummy data.

GEMINI / OPENAI: Switch to these for actual experiments.

### 2. Set API Keys

Ensure your API keys are accessible. You can set them as environment variables or (temporarily) paste them into common.py:

Python

```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_KEY_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
```

### 3. Adding Custom API Providers (Optional)

If you are using a different model provider (e.g., Anthropic/Claude, local LLMs via vLLM/Ollama), you must modify the call_llm function in common.py:

- Add your provider name to the PROVIDER check.  
- Implement the API call logic.  
- Ensure the function returns a tuple: (text_output, input_token_count).

Python

```python
# Example extension in common.py
elif PROVIDER == "CLAUDE":
    # Your custom implementation here
    response = anthropic_client.messages.create(...)
    text_out = response.content[0].text
    tokens_in = response.usage.input_tokens
```

---

# Baseline Descriptions & Prompt Structures

## Baseline 1: Holistic Scoring (Few-Shot)

**Methodology:** The model acts as an expert scorer and assigns a single integer score (1–6) based on the official ASAP 2.0 holistic rubric and provided reference examples.

**Prompt Structure (Exact Match):**
- **System:** Expert role definition.
- **OFFICIAL RUBRIC:** The complete text of the ASAP 2.0 holistic rubric.
- **REFERENCE EXAMPLES:** Few-shot examples from the training set (Score 1-6).
- **TARGET ESSAY:** The full text of the essay to be scored.
- **TASK:** Instruction to assign a holistic integer score.

**Output:** A single integer (1-6).

---

## Baseline 2: Multi-Trait Scoring (Zero-Shot)

**Methodology:** Uses Chain-of-Thought (CoT) decomposition. The model evaluates the essay on three distinct dimensions (Development, Organization, Language) independently before calculating the final score. This uses a detailed breakdown of the rubric and does not use few-shot examples to save context window space for the detailed criteria.

**Prompt Structure (Exact Match):**
- **System:** Expert MTS Scorer role.
- **RUBRIC:** Detailed criteria for Development, Organization, and Language (Score 1 & 6 anchors).
- **TARGET ESSAY:** The full text of the essay to be scored.
- **TASK:** Instruction to score 3 traits independently in a specific format.

**Output:** Structured text containing:
- `Development: [Score]`
- `Organization: [Score]`
- `Language: [Score]`  
  *(Final prediction is the rounded mean of these three.)*

---

## Baseline 3: Pairwise Comparison (Bradley-Terry Model)

**Methodology:** The model performs pairwise comparisons between the target essay and "Anchor Essays" (representative essays for scores 1-6 from the training set). Rankings are aggregated using the Bradley-Terry statistical model to predict the final score.

**Prompt Structure (Exact Match):**
- **System:** Comparative task definition.
- **RUBRIC:** The complete Official ASAP 2.0 holistic rubric.
- **ESSAY A:** The full text of the target essay (no truncation).
- **ESSAY B:** The full text of the anchor essay (no truncation).
- **TASK:** "Which essay better demonstrates the mastery described in the rubric?"

**Output:** `"Essay A"` or `"Essay B"`.

**Note:** This baseline is computationally expensive (N * 6 API calls per essay).

---

## Output & Deliverables

Each execution will generate two files with timestamps (e.g., 20251121_103000):

**BaselineX_..._Data_TIMESTAMP.csv:**

- Contains row-level details: essay_id, human_score, pred_score, input_tokens, latency, and raw_output.  
- For Baseline 2: Includes sub-scores (score_dev, score_org, etc.).  
- For Baseline 3: Includes bt_latent (the latent ability score from the BT model).

**BaselineX_..._Report_TIMESTAMP.txt:**

- A summary file containing the Quadratic Weighted Kappa (QWK), Exact Accuracy, and efficiency metrics (avg tokens/latency).

Please submit both the CSV and TXT files for analysis.
