import { getEventTypeLabel } from '../utils/display'

export default function EventTable({ data = [] }) {
  return (
    <div className="card table-card">
      <table className="data-table">
        <thead>
          <tr>
            <th>事件日期</th>
            <th>事件类型</th>
            <th>来源</th>
            <th>标题</th>
            <th>数量</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => (
            <tr key={`${item.event_date}-${index}`}>
              <td>{item.event_date}</td>
              <td>{getEventTypeLabel(item.event_type)}</td>
              <td>{item.event_source}</td>
              <td>{item.event_title}</td>
              <td>{item.event_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}