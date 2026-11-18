# Section 8: Academic Integrity & AI-Assistance Disclosure

## 8.1 Statement of Authenticity & Originality

In accordance with the PES University UE24CS352A Assessment and Academic Integrity Policy, all work submitted in this repository represents the original, independent work of this team:
- All data exploration, spatial indexing, candidate generation, feature design, model training, hyperparameter optimisation, HMM sequential decoding, and API/UI deployments were developed specifically for this project.
- No model weights, code, parameters, or precomputed predictions were shared with or obtained from other participating teams.
- Ground truth from the evaluation set was strictly quarantined and never used to fit models, select features, or tune hyperparameters.

---

## 8.2 AI-Assistance Disclosure

In transparency and compliance with competition guidelines, AI assistance (specifically Google Antigravity) was utilized during the development of this project for the following permitted tasks:
1. **Scaffolding and Syntax Acceleration:** Assisting in formulating boilerplate code for FastAPI endpoints (`api/main.py`), Folium HTML legend injection (`pathsnap/viz.py`), and Streamlit state management (`ui/app.py`).
2. **Exploratory Data Analysis Automation:** Generating exploratory plotting scripts and statistical aggregation routines for `notebooks/p4_eda.ipynb` across all 90 trip files.
3. **Documentation and Markdown Structuring:** Assisting with technical documentation formatting, LaTeX mathematical equation rendering, and markdown report drafting.

All architectural decisions, algorithmic designs (spatial candidate lookup, feature engineering, loss formulation, Viterbi log-likelihood recurrence, and distributed client-server separation) were directed, verified, and validated by the team members.
