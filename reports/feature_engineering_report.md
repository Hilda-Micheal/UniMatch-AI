# Feature Engineering Report — Task 5

## 1. Derived Features Created

**University Master dataset (53527 rows):**
- `rank_tier` — Tier 1 (rank ≤ 10) / Tier 2 (11-25) / Tier 3 (26-50) / Tier 4 (> 50) from `avg_rank_all3`; 'Unranked' for universities not in the rankings file (5695 of 53527 rows are ranked)
- `admit_chance_index` — weighted composite (0-1) of applicant CGPA%, GRE-Q/V, TOEFL, research & industry experience
- `fit_score` — inverse of the applicant's average z-distance from the admitted-cohort mean at their target university (1 = closest fit); cohort statistics computed from the training split only to avoid target leakage
- `affordability_index_proxy` — **proxy only**: no tuition/fee/country column exists in the source data, so this is built from institution scale (`total_students`), international-student share, and rank, for ranked universities only (NaN otherwise, excluded from the ML feature set) — it is NOT a true cost/affordability measure and should be replaced with real fee/country data if available

**Graduate Admission dataset (500 rows):**
- `admit_chance_index` — weighted composite (0-1) of GRE, TOEFL, CGPA, SOP, LOR, research, independent of the true `chance_of_admit` label (used as a sanity-check feature, correlates with but does not duplicate the label)
- `university_rating` (1-5) used directly as the ordinal rank tier

## 2. Encoding
- One-hot: `program`, `rank_tier` (university dataset); graduate dataset needs no encoding
- Label + frequency encoding: `department`, `specialization`, `univname` (high cardinality: 1486, 3619, 54 unique levels respectively)
- No `country` column exists in the source data, so no country encoding was performed — see note in Section 1 of the notebook for how to add one later

## 3. Correlation Analysis
- Full correlation matrices computed for both datasets (see `correlation_matrix_university.csv`, `correlation_matrix_graduate.csv`)
- Top correlate with `admit` (university dataset): univname_freq (r=0.112)
- Top correlate with `chance_of_admit` (graduate dataset): admit_chance_index

## 4. Feature Importance
- Random Forest + mutual information computed for both datasets
- Top 5 features (university, by RF importance): ['univname_freq', 'univname_label', 'cgpa_pct', 'admit_chance_index', 'profile_distance']
- Top 5 features (graduate, by RF importance): ['admit_chance_index', 'cgpa', 'gre_score', 'toefl_score', 'sop']

## 5. Feature Selection
- Rule: smallest top-ranked set covering 95% of cumulative RF importance (minimum 5), then drop any feature with |corr| > 0.90 to an already-selected one. Importance for the university dataset was measured on the training split only
- University dataset: 15 of 28 features retained
- Graduate dataset: 3 of 8 features retained

## Known Limitations (be upfront about these in the report / viva)
- No tuition/fee or country column in the merged dataset → Affordability Index is a resource-availability proxy, not real cost
- Only a small share of applicant rows match a ranked university, so ranking-based features carry limited signal
- `department`/`specialization` are extremely high-cardinality free-text fields — label/frequency encoding was used instead of one-hot to keep the matrix usable; this may lose some information a more careful text-embedding approach could capture
