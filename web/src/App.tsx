import { useEffect, useState } from "react";

type Emp = {
  satisfaction_level: number; last_evaluation: number; number_project: number; average_montly_hours: number;
  time_spend_company: number; Work_accident: number; promotion_last_5years: number; dept: string; salary: string;
};
type Pred = { attrition_probability: number; will_leave: boolean; risk: string };
type Info = { name: string; alias: string; version: string; run_id: string; params: Record<string, string>; metrics: Record<string, number> };

const DEPTS = ["sales", "technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting", "hr", "management"];
const PRESETS: Record<string, Emp> = {
  "Burned-out high performer": { satisfaction_level: 0.11, last_evaluation: 0.88, number_project: 7, average_montly_hours: 284, time_spend_company: 4, Work_accident: 0, promotion_last_5years: 0, dept: "technical", salary: "low" },
  "Under-utilised newcomer": { satisfaction_level: 0.38, last_evaluation: 0.5, number_project: 2, average_montly_hours: 140, time_spend_company: 3, Work_accident: 0, promotion_last_5years: 0, dept: "sales", salary: "low" },
  "Engaged veteran": { satisfaction_level: 0.82, last_evaluation: 0.76, number_project: 4, average_montly_hours: 205, time_spend_company: 6, Work_accident: 1, promotion_last_5years: 1, dept: "management", salary: "high" },
};

function Slider({ label, value, min, max, step, fmt, onChange }: { label: string; value: number; min: number; max: number; step: number; fmt?: (v: number) => string; onChange: (v: number) => void }) {
  return (
    <label className="field">
      <span>{label}<b>{fmt ? fmt(value) : value}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </label>
  );
}

export default function App() {
  const [emp, setEmp] = useState<Emp>(PRESETS["Burned-out high performer"]);
  const [pred, setPred] = useState<Pred | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [err, setErr] = useState("");
  const set = <K extends keyof Emp>(k: K) => (v: Emp[K]) => setEmp((e) => ({ ...e, [k]: v }));

  useEffect(() => { fetch("/api/model").then((r) => r.json()).then(setInfo).catch(() => setErr("API offline")); }, []);
  useEffect(() => {
    const t = setTimeout(() => {
      fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ employees: [emp] }) })
        .then((r) => r.json()).then((d) => { setPred(d.predictions[0]); setErr(""); }).catch((e) => setErr(String(e)));
    }, 150);
    return () => clearTimeout(t);
  }, [emp]);

  const pct = pred ? Math.round(pred.attrition_probability * 100) : 0;
  return (
    <div className="page">
      <header>
        <h1>🧭 Employee Attrition Predictor</h1>
        <p>Adjust an employee profile and see the model's estimate that they'll leave, served from the MLflow model registry.</p>
      </header>
      <div className="presets">
        {Object.keys(PRESETS).map((p) => <button key={p} className="ghost" onClick={() => setEmp(PRESETS[p])}>{p}</button>)}
      </div>
      <div className="grid">
        <section className="card">
          <h2>Employee profile</h2>
          <Slider label="Satisfaction level" value={emp.satisfaction_level} min={0} max={1} step={0.01} fmt={(v) => v.toFixed(2)} onChange={set("satisfaction_level")} />
          <Slider label="Last evaluation" value={emp.last_evaluation} min={0} max={1} step={0.01} fmt={(v) => v.toFixed(2)} onChange={set("last_evaluation")} />
          <Slider label="Number of projects" value={emp.number_project} min={1} max={8} step={1} onChange={set("number_project")} />
          <Slider label="Average monthly hours" value={emp.average_montly_hours} min={80} max={320} step={1} onChange={set("average_montly_hours")} />
          <Slider label="Years at company" value={emp.time_spend_company} min={1} max={10} step={1} onChange={set("time_spend_company")} />
          <div className="row2">
            <label className="field"><span>Department</span>
              <select value={emp.dept} onChange={(e) => set("dept")(e.target.value)}>{DEPTS.map((d) => <option key={d}>{d}</option>)}</select></label>
            <label className="field"><span>Salary band</span>
              <select value={emp.salary} onChange={(e) => set("salary")(e.target.value)}>{["low", "medium", "high"].map((d) => <option key={d}>{d}</option>)}</select></label>
          </div>
          <div className="row2">
            <label className="check"><input type="checkbox" checked={!!emp.Work_accident} onChange={(e) => set("Work_accident")(+e.target.checked)} /> Had a work accident</label>
            <label className="check"><input type="checkbox" checked={!!emp.promotion_last_5years} onChange={(e) => set("promotion_last_5years")(+e.target.checked)} /> Promoted in last 5 years</label>
          </div>
        </section>
        <div className="side">
          <section className={`card result ${pred?.risk ?? ""}`}>
            <h2>Attrition risk</h2>
            <div className="big">{pct}%</div>
            <div className="meter"><div style={{ width: `${pct}%` }} /></div>
            <p className="verdict">{pred ? (pred.will_leave ? "Likely to leave" : "Likely to stay") + ` · ${pred.risk} risk` : "…"}</p>
            {err && <p className="err">{err}</p>}
          </section>
          {info && (
            <section className="card">
              <h2>Serving model</h2>
              <p className="mono">models:/{info.name}@{info.alias} → v{info.version}</p>
              <table><tbody>
                {Object.entries(info.params).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
                {Object.entries(info.metrics).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v.toFixed(3)}</td></tr>)}
              </tbody></table>
              <a href="http://localhost:5050" target="_blank">Open MLflow UI ↗</a>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
