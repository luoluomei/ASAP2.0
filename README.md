ASAP 2.0 Evaluator Alignment Baselines

This repository contains the implementation of three LLM-based automated essay scoring (AES) baselines for the ASAP 2.0 dataset. The goal of this project is to benchmark different prompting strategies—Holistic Scoring, Multi-Trait Scoring, and Pairwise Comparison—to evaluate their alignment with human judgments (Ground Truth).

Task Overview for Team Members

Your primary task is to execute these baselines on the ASAP 2.0 Test Set to generate performance metrics (QWK, Accuracy, etc.) and cost analysis.

Configure your Environment: Set up the API provider you intend to use.

Run Baselines 1, 2, and 3: Execute the scripts to perform inference on the full dataset.

Collect Results: Submit the generated .csv data logs and .txt summary reports.

Configuration & API Setup

All configuration settings, including data loading and API wrappers, are centralized in common.py. You do not need to modify the individual baseline scripts.

1. Select Your Provider

Open common.py and locate the configuration section at the top:

# common.py

# MODE SELECTION
# Options: "MOCK" (Testing), "GEMINI", "OPENAI"
PROVIDER = "MOCK" 

# MODEL CONFIGURATION
# Gemini: "gemini-2.0-flash" (Recommended for speed/cost)
# OpenAI: "gpt-4o"
MODEL_NAME = "gemini-2.0-flash"


MOCK: Use this first to verify the code runs without errors. It uses zero tokens and returns random dummy data.

GEMINI / OPENAI: Switch to these for actual experiments.

2. Set API Keys

Ensure your API keys are accessible. You can set them as environment variables or (temporarily) paste them into common.py:

Python

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_KEY_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")

3. Adding Custom API Providers (Optional)

If you are using a different model provider (e.g., Anthropic/Claude, local LLMs via vLLM/Ollama), you must modify the call_llm function in common.py:

Add your provider name to the PROVIDER check.

Implement the API call logic.

Ensure the function returns a tuple: (text_output, input_token_count).

Python

# Example extension in common.py
elif PROVIDER == "CLAUDE":
    # Your custom implementation here
    response = anthropic_client.messages.create(...)
    text_out = response.content[0].text
    tokens_in = response.usage.input_tokens

Baseline Descriptions
Baseline 1: Holistic Scoring (Direct Assessment)

Methodology: The model acts as an expert scorer and assigns a single integer score (1–6) based on the official ASAP 2.0 holistic rubric.

Prompt Structure:

System: Role definition.

Context: The complete Official Rubric + Few-Shot Reference Examples.

Input: The full target essay.

Task: Assign a single integer score.

Output: A single integer (1-6).

Baseline 2: Multi-Trait Scoring (MTS)

Methodology: Uses Chain-of-Thought (CoT) decomposition. The model evaluates the essay on three distinct dimensions before calculating the final score. This helps reduce hallucination and improves explainability.

Prompt Structure:

Context: Detailed criteria for Development, Organization, and Language.

Input: The full target essay.

Task: Output scores for all three traits in a structured format.

Output: Three sub-scores (e.g., Dev: 4, Org: 3, Lang: 4). The final prediction is the rounded mean of these traits.

Baseline 3: Pairwise Comparison (Bradley-Terry Model)

Methodology: Instead of absolute scoring, the model performs pairwise comparisons between the target essay and "Anchor Essays" (representative essays for scores 1-6 from the training set). The rankings are aggregated using the Bradley-Terry statistical model to predict the final score.

Prompt Structure:

Context: Official Rubric.

Comparison: [Essay A] vs [Essay B].

Task: "Which essay demonstrates better mastery? Output Essay A or Essay B."

Output: A binary preference label.

Note: This baseline is computationally expensive (N * 6 API calls). Please ensure you have sufficient quota/credits before running on the full test set.

Execution Instructions

First, install the required dependencies:

Bash

pip install google-generativeai pandas scikit-learn openai choix


Run the experiments in order:

Bash

# 1. Run Holistic Baseline
python run_baseline1.py

# 2. Run Multi-Trait Baseline
python run_baseline2.py

# 3. Run Pairwise Baseline (Warning: Long execution time)
python run_baseline3.py

Output & Deliverables

Each execution will generate two files with timestamps (e.g., 20251121_103000):

BaselineX_..._Data_TIMESTAMP.csv:

Contains row-level details: essay_id, human_score, pred_score, input_tokens, latency, and raw_output.

For Baseline 2: Includes sub-scores (score_dev, score_org, etc.).

For Baseline 3: Includes bt_latent (the latent ability score from the BT model).

BaselineX_..._Report_TIMESTAMP.txt:

A summary file containing the Quadratic Weighted Kappa (QWK), Exact Accuracy, and efficiency metrics (avg tokens/latency).

Please submit both the CSV and TXT files for analysis.
