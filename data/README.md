# Data

```
data/
├── README.md
└── raw/
    ├── Admission_Predict_Ver1.1.csv
    ├── university_recommendation.csv
    ├── college_admissions.csv
    └── World_University_Rankings_2026.csv
```

Load Notebooks with relative paths, e.g. `pd.read_csv("../data/raw/Admission_Predict_Ver1.1.csv")`.

> Processed feature matrices produced from these files are in `outputs/task5_outputs/`, so Tasks 6-8 can run without the raw data.

---

## Summary

| File | Rows | Cols | Size | What it is |
|------|-----:|-----:|-----:|------------|
| `Admission_Predict_Ver1.1.csv` | 500 | 9 | 16 KB | Graduate applicant profiles with admission probability |
| `university_recommendation.csv` | 53,644 | 26 | 10.9 MB | Applicant profiles with admit/reject outcome per university |
| `college_admissions.csv` | 1,534 | 108 | 773 KB | US institution-level admissions, enrollment, tuition and graduation statistics |
| `World_University_Rankings_2026.csv` | 57 | 30 | 12 KB | Compiled QS / THE / ARWU rankings and indicators |

---

## 1. `Admission_Predict_Ver1.1.csv`

Graduate admission applicant records (the widely used *Graduate Admissions* dataset).

| Column | Description |
|--------|-------------|
| `Serial No.` | Row ID |
| `GRE Score` | GRE score (out of 340) |
| `TOEFL Score` | TOEFL score (out of 120) |
| `University Rating` | Rating of the target university, 1-5 |
| `SOP` | Statement of Purpose strength, 1-5 |
| `LOR ` | Letter of Recommendation strength, 1-5 |
| `CGPA` | Undergraduate CGPA (out of 10) |
| `Research` | Research experience (0 = no, 1 = yes) |
| `Chance of Admit ` | Probability of admission, 0-1 (**target**) |

Note: `LOR ` and `Chance of Admit ` have a **trailing space** in the original header. Strip column names after loading: `df.columns = df.columns.str.strip()`.

## 2. `university_recommendation.csv`

Self-reported applicant profiles and outcomes, scraped from a graduate-admissions forum (profile links point to edulix.com).

| Group | Columns |
|-------|---------|
| Outcome | `admit` (1 = admitted, 0 = not) (**target**) |
| Target | `univName`, `program` (MS / PhD / ...), `department`, `major`, `specialization`, `termAndYear` |
| Test scores | `greV`, `greQ`, `greA`, `toeflScore`, `toeflEssay`, `gmatV`, `gmatQ`, `gmatA` |
| Academics | `cgpa`, `cgpaScale`, `topperCgpa`, `ugCollege` |
| Experience | `researchExp`, `industryExp`, `internExp`, `journalPubs`, `confPubs` |
| Identifiers | `userName`, `userProfileLink` |

Things to know:
- Roughly 52% admits / 48% rejects across 54 universities. Most rows are MS applicants.
- Many fields are missing (e.g. GMAT, TOEFL essay) and `cgpaScale` varies (4, 10, 100), so CGPA must be normalised before use.
- `userName` and `userProfileLink` identify individual people. **Drop them in Task 4** and avoid publishing them (see Privacy below).

## 3. `college_admissions.csv`

US institution-level statistics (1,534 institutions, 108 columns; fees refer to 2010-2014).

| Group | Example columns |
|-------|-----------------|
| Admissions | `Applicants total`, `Admissions total`, `Enrolled total`, `Percent admitted - total`, `Admissions yield - total` |
| Test scores | SAT Critical Reading / Math / Writing and ACT Composite, 25th and 75th percentiles |
| Cost | `Tuition and fees, 2010-11` to `2013-14`, total price for in-state / out-of-state students |
| Institution | `Name`, `State abbreviation`, `Geographic region`, `Control of institution`, `Sector of institution`, `Carnegie Classification 2010: Basic` |
| Enrollment | Total / undergraduate / graduate, full and part time, with demographic percentages |
| Outcomes | Graduation rates (4, 5, 6 years), financial-aid percentages, endowment per FTE |

## 4. `World_University_Rankings_2026.csv`

A compiled table of 57 universities across three ranking systems.

| Group | Columns |
|-------|---------|
| Identity | `university`, `country`, `region`, `university_type` (Public / Private), `founded`, `university_age_2026` |
| Ranks | `qs_rank_2026`, `the_rank_2026`, `arwu_rank_2025`, `avg_rank_all3`, `ranked_in_all_3` |
| Scores | `qs_score`, `the_score`, `arwu_score` |
| QS indicators | `qs_academic_rep`, `qs_employer_rep`, `qs_faculty_student`, `qs_citations`, `qs_intl_faculty`, `qs_intl_students` |
| THE indicators | `the_teaching`, `the_research_env`, `the_research_quality`, `the_industry`, `the_intl_outlook` |
| Profile | `total_students`, `intl_students_pct`, `nobel_laureates`, `fields_medalists` |
| Provenance | `data_sources` (QS Jun-2025, THE Oct-2025, ARWU Aug-2025) |



---

## Sources 


| File | Original source / link | 
|------|------------------------|
| `Admission_Predict_Ver1.1.csv` | https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fmohansacharya%2Fgraduate-admissions |
| `university_recommendation.csv` | https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnitishabharathi%2Funiversity-recommendation | 
| `college_admissions.csv` | https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fsamsonqian%2Fcollege-admissions | 
| `World_University_Rankings_2026.csv` | https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Falitaqishah%2Fworld-university-rankings-2026-qs-and-the-and-arwu |


