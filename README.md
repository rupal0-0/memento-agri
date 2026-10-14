# Memento-Agri

Outcome-Aware Memory-Augmented LLM Agent for Personalized Climate-Resilient Agricultural Advisory.

This research prototype implements:
- tabular climate/agriculture preprocessing
- climate-risk feature engineering
- ML baselines for climate-risk classification and yield prediction
- case-bank construction
- FAISS semantic retrieval with outcome-aware reranking
- Memento-inspired planner/executor agent
- optional OpenAI LLM integration (works in deterministic mock mode without an API key)
- Streamlit demo
- evaluation/ablation scripts

## Important research note
The historical ICRISAT-style data provides observations and outcomes, not verified historical intervention labels. This code therefore does NOT fabricate historical recommendations. Candidate interventions are generated from an explicit rule layer and evaluated separately.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Put your data in `data/raw/master_dataset.csv`. The loader accepts flexible column aliases; see `config.py` and `data/sample_master_dataset.csv`.

Then:

```bash
python main.py build
python main.py train-risk
python main.py train-yield
python main.py build-memory
python main.py demo
streamlit run app/streamlit_app.py
```

For an LLM-backed run:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
python main.py demo
```

Without an API key, the system uses a deterministic local advisory generator so the whole pipeline remains runnable.

## Expected canonical columns

`district, state, year, crop, season, crop_area, crop_production, crop_yield, rainfall, normal_rainfall, tmax, tmin, ppt, aet, pet, def, runoff, wind_speed, irrigation_percentage, soil_type`

Only a subset is required for every experiment. Missing optional variables are handled where possible.

## Research experiments

```bash
python experiments/run_ablation.py
python experiments/run_memory_growth.py
python experiments/run_temporal_split.py
```

Outputs go to `outputs/`.

## Disclaimer

This is a research prototype, not a production agricultural advisory service. Recommendations should be validated by qualified agricultural experts and local government advisories before real-world use.
