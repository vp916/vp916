<!-- ─────────────────────────────────────────────────────────────
     varun.prajapati — a model that's still training
     every visual in this README is hand-built SVG in /assets
───────────────────────────────────────────────────────────── -->

<p align="center">
  <img src="assets/hero.svg" width="100%" alt="Terminal: python train.py --model varun-prajapati --epochs ∞. CSE at MSU Baroda plus BS Data Science at IIT Madras. Still training." />
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/vp916/"><img src="https://img.shields.io/badge/linkedin-vp916-0d1117?style=flat-square&logo=linkedin&logoColor=3fb950&labelColor=0d1117&color=161b22" alt="LinkedIn" /></a>
  <a href="mailto:vnp785@gmail.com"><img src="https://img.shields.io/badge/email-vnp785%40gmail.com-0d1117?style=flat-square&logo=gmail&logoColor=3fb950&labelColor=0d1117&color=161b22" alt="Email" /></a>
  <img src="https://komarev.com/ghpvc/?username=vp916&style=flat-square&color=238636&label=inference+calls" alt="Profile views" />
</p>

```python
class Varun(Engineer):
    loss_fn   = "the gap between a notebook and a shipped product"
    optimizer = "read the paper → break the baseline → ship the API"
    weights   = {"deep_learning": 0.45, "classical_ml": 0.25, "full_stack": 0.30}

    def forward(self, problem):
        model = self.train(problem, from_scratch=True)   # u-net++ dice 0.9796, no pretrained weights
        api   = FastAPI(model)                            # served as a microservice
        return React(api)                                 # …and someone actually clicks it
```

<br/>

## ⟢ `model.summary()`

<p align="center">
  <img src="assets/stack.svg" width="100%" alt="Stack drawn as a neural network. Input: Python, TypeScript, C++, Java, SQL. Learning: PyTorch, scikit-learn, XGBoost/LightGBM/CatBoost, OpenCV, NumPy/Pandas. Serving: FastAPI, Node/Express, PostgreSQL, MongoDB, Docker. Output: React apps, ML microservices, competition-grade models." />
</p>

<br/>

## ⟢ `experiments/` — click a run to open the repo

<p align="center">
  <a href="https://github.com/vp916/chest-xray-disease-classification"><img src="assets/card-xray.svg" width="49%" alt="Chest X-ray Classification: 20 classes, DenseNet-201, focal loss, per-class thresholds. Test score −5.22 → −5.04." /></a>
  <a href="https://github.com/vp916/mechanical-parts-segmentation"><img src="assets/card-seg.svg" width="49%" alt="Mechanical Parts Segmentation: U-Net++, trained from scratch, TTA ensemble. Dice 0.9796." /></a>
  <a href="https://github.com/vp916/plant-leaf-super-resolution"><img src="assets/card-sr.svg" width="49%" alt="Plant Leaf Super-Resolution: 4× SRGAN, 32 → 128 px. MAE 16.77." /></a>
  <a href="https://github.com/vp916/heavy_equipment_price_prediction"><img src="assets/card-price.svg" width="49%" alt="Heavy Equipment Pricing: stacked XGBoost, LightGBM, CatBoost on 138K rows. RMSLE 0.1928." /></a>
</p>

<br/>

## ⟢ `deploy/` — where the models meet users

**AI Transaction Analytics Platform** — a full ML product, not a notebook:

```mermaid
flowchart LR
    U([👤 user]) --> R[React + TypeScript<br/>dashboard]
    R -- JWT --> N[Node · Express API]
    N <--> M[(MongoDB)]
    N -- REST --> F{{FastAPI ML service}}
    F --> A[Isolation Forest<br/>anomaly detection]
    F --> S[spend<br/>forecasting]
    F --> T[TF-IDF<br/>auto-categorisation]
    classDef ml fill:#0d1117,stroke:#3fb950,color:#e6edf3
    classDef web fill:#0d1117,stroke:#58a6ff,color:#e6edf3
    class F,A,S,T ml
    class R,N,M web
```

**AI Mental Health Support Platform** — journaling, self-assessments and an OpenAI-powered chat on React · Express · MongoDB with role-based access control.

<br/>

## ⟢ `git log --oneline --graph achievements`

```diff
+ * 3a91f2e  (HEAD → main)  🥉 2nd Runner-Up · Adani Innovation Mindstorm 2026
+ |                         built the Isolation Forest + LSTM pipeline for ICS anomaly detection
+ * 7c04d1b  🌱 Open Source Contributor · GirlScript Summer of Code 2026
+ |                         issues, features & PRs through reviewed GitHub workflows
+ * e52b8a0  🏁 Top 40 Finalist · Odoo × MSU FootPrints'25 Hackathon
+ * 1f6c3d9  🤝 Junior Training & Placement Coordinator · FTE, MSU Baroda
+ |                         bridging recruitment drives between students and recruiters
  * 0000000  init: hello, world
```

<br/>

<details>
<summary><b>⟢ <code>tensorboard --logdir=github</code></b> &nbsp;<sub>(click to expand stats)</sub></summary>
<br/>
<p align="center">
  <img height="165" src="https://github-stats-extended.vercel.app/api?username=vp916&show_icons=true&theme=github_dark&hide_border=true&include_all_commits=true&title_color=3FB950&icon_color=3FB950&bg_color=0d1117" alt="GitHub stats" />
  <img height="165" src="https://streak-stats.demolab.com?user=vp916&theme=github-dark-blue&hide_border=true&ring=3FB950&fire=3FB950&currStreakLabel=3FB950&background=0d1117" alt="GitHub streak" />
</p>
</details>

<br/>

<p align="center">
  <sub><code>early_stopping(patience=∞)</code> · open an issue, send a PR, or just say hi — the gradient's always open.</sub>
</p>
