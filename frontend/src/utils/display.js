export const EVENT_TYPE_MAP = {
  1: '定期报告',
  2: '事项公告',
  3: '分红派息',
  4: '经营/交易事项',
  5: '人事变动',
  6: '公司治理',
  7: '风险提示'
}

export function getEventTypeLabel(value) {
  const key = Number(value)
  return EVENT_TYPE_MAP[key] || `类型 ${value ?? '--'}`
}

export function formatNumber(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(digits)
}

export function formatPercent(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${(num * 100).toFixed(digits)}%`
}

export function formatPercentPlain(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${num.toFixed(digits)}%`
}

export function formatInteger(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return Math.round(num).toLocaleString('zh-CN')
}

export function formatMetric(value, digits = 4) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(digits)
}

export function formatBoolResult(value) {
  if (value === true) return '预测正确'
  if (value === false) return '预测错误'
  return '暂无真实结果'
}