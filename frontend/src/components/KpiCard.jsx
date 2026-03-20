export default function KpiCard({ title, value, extra }) {
  return (
    <div className="card kpi-card">
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{value}</div>
      {extra ? <div className="kpi-extra">{extra}</div> : null}
    </div>
  )
}