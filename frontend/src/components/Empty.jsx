export default function Empty({ text = '暂无数据' }) {
  return <div className="status-box empty">{text}</div>
}