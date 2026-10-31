# Business case: Employee Attrition Predictor

**Flag teams where people are likely to leave, so managers talk to them sooner.**

## Who buys this

The buyer is an HR or people-analytics lead at a mid-sized employer. The users are HR business partners, who look at risk by team, and the data team that maintains the model.

## The problem and what it costs

Gallup estimates that replacing an employee costs **one-half to two times the employee's annual salary**, and that **52% of voluntarily exiting employees** say their manager or organisation could have done something to keep them ([Gallup, 2019](https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx)). An early signal helps only if it leads to a conversation, which is the same point Gallup makes.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Training with a record | A training job logs three logistic-regression variants to MLflow, with parameters, metrics (ROC AUC, precision, recall, F1) and the model signature |
| Controlled promotion | Each run registers a version of `employee-attrition`. The best gets the `champion` alias, and the API serves `models:/employee-attrition@champion`. |
| Scoring | `POST /predict` returns a probability, a yes/no and a red/amber/green risk band |
| Exploration | A React app with sliders and presets shows how risk changes with satisfaction, workload, tenure and pay band |
| Data | The public Kaggle HR Analytics dataset of about 15,000 employees, downloaded at training time |

## Use-case scenarios

All of these organisations are fictional.

1. **A regional logistics firm's HR team** reviews aggregated risk by depot each quarter and plans stay interviews where risk is high.
2. **A software consultancy** sees that high monthly hours plus no promotion in five years drives risk, and changes its project-staffing policy.
3. **A people-analytics team** uses the MLflow registry to show auditors which model version produced which score.

## ROI: a worked model (illustrative)

These figures are **illustrative assumptions** unless a source is linked.

| Input | Value | Basis |
|---|---|---|
| Headcount | 500 | Assumption |
| Annual voluntary turnover | 15% (75 leavers) | Assumption |
| Average salary | US$60,000 | Assumption |
| Replacement cost | 0.5 × salary, US$30,000 | Low end of the Gallup range ([Gallup](https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx)) |
| Leavers retained because of earlier conversations | 5% of 75, about 4 | Assumption |

Avoided cost: 4 × $30,000 = **US$120,000 a year**, using the low end of the replacement range. That is before the cost of HR time spent on the conversations, which should be counted against it.

## Path to production

1. Retrain on your own HRIS data, with legal and works-council review of which fields may be used.
2. Report risk at team level by default, not named individuals.
3. Check calibration and fairness across groups before use, and record each check in MLflow.
4. Move MLflow from SQLite to a managed database and object store, and add access control.

## Risks and limits

- Scoring individuals for attrition raises privacy, fairness and employment-law questions. It needs a clear policy on who sees what.
- A risk score must never feed a disciplinary or promotion decision.
- The training data is a public teaching dataset, so the model shows the method, not your workforce.

**Case strength:** **Moderate.** Turnover cost is well sourced and the MLOps around the model is solid. The model itself is a teaching example, and the ethical bar for using it on real people is high.

## Sources

- Gallup (2019), This Fixable Problem Costs U.S. Businesses $1 Trillion: https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx
- Kaggle HR Analytics dataset (mirror used by the repo): https://github.com/ryankarlos/Human-Resource-Analytics-Kaggle-Dataset
