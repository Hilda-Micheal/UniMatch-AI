# Feature Engineering Report — Task 5

## 1. Derived Features Created

**University Master dataset (53527 rows):**
- `rank_tier` — Tier 1 (rank ≤ 10) / Tier 2 (11-25) / Tier 3 (26-50) / Tier 4 (> 50) from `avg_rank_all3`; 'Unranked' for universities not in the rankings file (5695 of 53527 rows are ranked)
- `admit_chance_index` — weighted composite (0-1) of applicant CGPA%, GRE-Q/V, TOEFL, research & industry experience
- `split` — train / val / test assignment **by applicant** (34207 / 8579 / 10741 rows): every applicant's applications fall in one part only, so the test set measures performance on unseen applicants
- `fit_score` — inverse of the applicant's average z-distance from the admitted-cohort mean at their target university (1 = closest fit); cohort statistics computed from the training split only to avoid target leakage
- `affordability_index_proxy` — **proxy only**: no tuition/fee/country column exists in the source data, so this is built from institution scale (`total_students`), international-student share, and rank, for ranked universities only (NaN otherwise, excluded from the ML feature set) — it is NOT a true cost/affordability measure and should be replaced with real fee/country data if available

**Graduate Admission dataset (500 rows):**
- `admit_chance_index` — weighted composite (0-1) of GRE, TOEFL, CGPA, SOP, LOR, research; kept as a reference column only, **not** a model feature (it would let the correlation filter drop the raw attributes it is built from)
- `split` — stratified 60 / 20 / 20 train / val / test split on the Task 6 label (`chance_of_admit >= 0.7`)
- `university_rating` (1-5) used directly as the ordinal rank tier

## 2. Encoding
- One-hot: `program`, `rank_tier`, `univname` (54 universities); graduate dataset needs no encoding
- `department` / `specialization` (1486 / 3619 free-text levels): one-hot for levels with ≥ 200 training rows (27 / 9 columns; rarer levels share an all-zero "other" encoding) plus frequency encoding
- Vocabularies and frequencies are computed on training rows only and saved to `category_encodings.json` for the application
- Label (integer) encoding is no longer used: arbitrary alphabetical codes carry no order, which only tree models can work around
- No `country` column exists in the source data, so no country encoding was performed — see note in Section 1 of the notebook for how to add one later

## 3. Correlation Analysis
- Full correlation matrices computed for both datasets (see `correlation_matrix_university.csv`, `correlation_matrix_graduate.csv`)
- Top correlate with `admit` (university dataset): profile_distance (r=-0.080)
- Top correlate with `chance_of_admit` (graduate dataset): admit_chance_index

## 4. Feature Importance
- Random Forest + mutual information computed for both datasets
- Top 5 features (university, by RF importance): ['cgpa_pct', 'univ_University_of_Texas_Dallas', 'fit_score', 'admit_chance_index', 'profile_distance']
- Top 5 features (graduate, by RF importance): ['cgpa', 'gre_score', 'toefl_score', 'lor', 'sop']
- Importance is computed on training rows only for both datasets

## 5. Feature Selection
- Rule: smallest top-ranked set covering 95% of cumulative RF importance (minimum 5), then drop any feature with |corr| > 0.90 to an already-selected one. For the university dataset the rule runs on the numeric features; the one-hot groups are always kept whole
- University dataset: 105 of 108 features retained
- Graduate dataset: 5 of 7 features retained

## Known Limitations (be upfront about these in the report / viva)
- No tuition/fee or country column in the merged dataset → Affordability Index is a resource-availability proxy, not real cost
- Only a small share of applicant rows match a ranked university, so ranking-based features carry limited signal
- `department`/`specialization` are extremely high-cardinality free-text fields — only levels with ≥ 200 training rows get their own column; cleaning/merging the free text (or a text embedding) could capture more
