export default function PageHeader({ title, description }) {
  return (
    <div className="page-header">
      <h2>{title}</h2>
      {description ? <p>{description}</p> : null}
    </div>
  )
}