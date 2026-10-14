import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import pandas as pd
from config import RAW_DATA, PROCESSED_DATA, CASE_BANK, FAISS_INDEX, CASE_META
from data.preprocess import preprocess
from memory.case_bank import build_case_bank
from memory.retriever import CaseRetriever, build_index
from models.risk_model import train_risk_model
from models.yield_model import train_yield_model
from agent.agent import MementoAgriAgent

def build():
    df = preprocess(RAW_DATA, PROCESSED_DATA)
    print(f"Processed {len(df)} rows -> {PROCESSED_DATA}")
    return df

def build_memory():
    df = pd.read_csv(PROCESSED_DATA)
    cases = build_case_bank(df, CASE_BANK)
    n = build_index(cases, FAISS_INDEX, CASE_META)
    print(f"Indexed {n} cases.")

def demo():
    df = pd.read_csv(PROCESSED_DATA)
    retriever = CaseRetriever.load(FAISS_INDEX, CASE_META)
    agent = MementoAgriAgent(retriever)
    row = df.iloc[-1]
    result = agent.advise(
        "My crop is facing unpredictable rainfall and heat. What should I do?",
        row
    )
    print(result["advisory"])
    print("\nRetrieved cases:", [c["case_id"] for c in result["cases"]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build", "train-risk", "train-yield", "build-memory", "demo"]
    )
    args = parser.parse_args()

    if args.command == "build":
        build()
    elif args.command == "train-risk":
        df = pd.read_csv(PROCESSED_DATA)
        print(json.dumps(train_risk_model(df), indent=2, default=str))
    elif args.command == "train-yield":
        df = pd.read_csv(PROCESSED_DATA)
        print(json.dumps(train_yield_model(df), indent=2))
    elif args.command == "build-memory":
        build_memory()
    elif args.command == "demo":
        demo()

# Detect if executed via Streamlit (e.g. Streamlit Cloud default main file setting)
try:
    import streamlit.runtime
    _RUNNING_IN_STREAMLIT = streamlit.runtime.exists()
except Exception:
    _RUNNING_IN_STREAMLIT = False

if _RUNNING_IN_STREAMLIT:
    import runpy
    from pathlib import Path
    _app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    runpy.run_path(str(_app_path), run_name="__main__")

if __name__ == "__main__":
    if not _RUNNING_IN_STREAMLIT:
        main()

