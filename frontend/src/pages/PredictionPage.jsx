import { useEffect, useMemo, useState } from 'react'
import {
  getPredictOptions,
  predictByEvent
} from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'
import {
  formatBoolResult,
  formatMetric,
  formatNumber,
  formatPercent,
  getEventTypeLabel
} from '../utils/display'

function formatFeature(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(6)
}

export default function PredictionPage() {
  const [options, setOptions] = useState([])
  const [selectedDate, setSelectedDate] = useState('')
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function initPage() {
      try {
        const res = await getPredictOptions()
        const list = res.data || []
        setOptions(list)
        if (list.length > 0) {
          setSelectedDate(list[0].event_date)
        }
      } catch (err) {
        console.error(err)
        setError('预测页初始化失败，请检查后端接口。')
      } finally {
        setLoading(false)
      }
    }

    initPage()
  }, [])

  const selectedEvent = useMemo(() => {
    return options.find((item) => item.event_date === selectedDate) || null
  }, [options, selectedDate])

  async function handlePredict() {
    if (!selectedEvent) return

    setSubmitting(true)
    setError('')

    try {
      const res = await predictByEvent({
        event_date: selectedEvent.event_date,
        event_type: selectedEvent.event_type,
        event_title: selectedEvent.event_title
      })
      setResult(res.data)
    } catch (err) {
      console.error(err)
      setResult(null)

      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        '真实预测失败，请检查模型文件和后端日志。'

      setError(String(detail))
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return <Loading text="预测页加载中..." />
  }

  return (
    <div className="prediction-page-wrap">
      <PageHeader
        title="预测结果"
        description="当前版本使用真实 LSTM 模型，对历史事件库中的已有事件进行推理，并提供真实值回看。"
      />

      <div className="prediction-layout section-gap">
        <div className="card prediction-main-panel">
          <div className="card-title">选择预测事件</div>

          <div className="form-box">
            <div className="form-item">
              <label>事件日期</label>
              <select
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
              >
                {options.map((item) => (
                  <option key={item.event_date} value={item.event_date}>
                    {item.event_date} - {item.event_title}
                  </option>
                ))}
              </select>
            </div>

            {selectedEvent ? (
              <div className="selected-event-info">
                <div><strong>事件日期：</strong>{selectedEvent.event_date}</div>
                <div><strong>事件类型：</strong>{getEventTypeLabel(selectedEvent.event_type)}</div>
                <div><strong>事件来源：</strong>{selectedEvent.event_source}</div>
                <div><strong>事件标题：</strong>{selectedEvent.event_title}</div>
                <div><strong>事件数量：</strong>{selectedEvent.event_count}</div>
              </div>
            ) : (
              <Empty text="暂无可选事件" />
            )}

            <div className="predict-note-box">
              当前真实预测依赖事件窗口行情特征，因此该版本支持对历史事件库中的已有事件做真实推理，并回看真实结果。
            </div>

            <button
              className="primary-btn"
              onClick={handlePredict}
              disabled={!selectedEvent || submitting}
            >
              {submitting ? '真实预测中...' : '运行真实预测'}
            </button>
          </div>
        </div>

        <div className="card prediction-output-panel">
          <div className="card-title">预测输出</div>

          {error ? <div className="status-box empty">{error}</div> : null}

          {!result ? (
            <Empty text="尚未发起真实预测" />
          ) : (
            <div className="prediction-output-box">
              <div className="prediction-label-row">
                <span className="prediction-label-title">预测方向</span>
                <span
                  className={`prediction-label-badge ${
                    result.prediction_label === '上涨' ? 'up' : 'down'
                  }`}
                >
                  {result.prediction_label}
                </span>
              </div>

              <div className="grid grid-3 section-gap-small">
                <KpiCard
                  title="预测概率"
                  value={formatPercent(result.probability)}
                  extra="sigmoid 输出"
                />
                <KpiCard
                  title="预测置信度"
                  value={formatPercent(result.confidence)}
                  extra="max(p, 1-p)"
                />
                <KpiCard
                  title="模型名称"
                  value={result.model_name || '--'}
                  extra="当前真实推理模型"
                />
              </div>

              <div className="selected-event-info section-gap">
                <div><strong>事件日期：</strong>{result.event_date}</div>
                <div><strong>事件类型：</strong>{getEventTypeLabel(result.event_type)}</div>
                <div><strong>事件来源：</strong>{result.event_source}</div>
                <div><strong>事件标题：</strong>{result.event_title}</div>
                <div><strong>使用序列日期：</strong>{(result.used_sequence_dates || []).join('、')}</div>
              </div>

              <div className="section-gap">
                <div className="card-title">真实值回看</div>

                {!result.actual_is_available ? (
                  <Empty text="该事件暂时没有足够的未来交易日，无法回看真实结果。" />
                ) : (
                  <>
                    <div className="grid grid-3 section-gap-small">
                      <KpiCard
                        title="真实方向"
                        value={result.actual_label || '--'}
                        extra={`基于事件后 ${result.actual_horizon_days ?? 3} 个交易日`}
                      />
                      <KpiCard
                        title="真实收益率"
                        value={formatPercent(result.actual_post_return)}
                        extra={`${result.event_date} → ${result.actual_future_date || '--'}`}
                      />
                      <KpiCard
                        title="预测结果校验"
                        value={formatBoolResult(result.actual_is_correct)}
                        extra="预测方向 vs 真实方向"
                      />
                    </div>

                    <div className="selected-event-info section-gap">
                      <div><strong>事件日收盘价：</strong>{formatNumber(result.actual_event_close)}</div>
                      <div><strong>未来收盘价：</strong>{formatNumber(result.actual_future_close)}</div>
                      <div><strong>未来日期：</strong>{result.actual_future_date || '--'}</div>
                    </div>
                  </>
                )}
              </div>

              <div className="section-gap">
                <div className="card-title">模型输入特征序列</div>

                {!result.feature_sequence?.length ? (
                  <Empty text="暂无特征序列数据" />
                ) : (
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>日期</th>
                        <th>日内收益率</th>
                        <th>振幅</th>
                        <th>成交量变化</th>
                        <th>收盘价变化</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.feature_sequence.map((item, index) => (
                        <tr key={`${item.date}-${index}`}>
                          <td>{item.date}</td>
                          <td>{formatFeature(item.intraday_return)}</td>
                          <td>{formatFeature(item.amplitude)}</td>
                          <td>{formatFeature(item.volume_change_vs_prev)}</td>
                          <td>{formatFeature(item.close_change_vs_prev_close)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}