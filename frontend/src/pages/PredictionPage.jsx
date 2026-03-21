import { useEffect, useMemo, useState } from 'react'
import {
  getPredictOptions,
  getSimulateContext,
  predictByEvent,
  predictSimulate
} from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'
import {
  formatBoolResult,
  formatInteger,
  formatNumber,
  formatPercent,
  getEventTypeLabel
} from '../utils/display'

const EVENT_TYPE_OPTIONS = [1, 2, 3, 4, 5, 6, 7]

const MODEL_VERSION_OPTIONS = [
  {
    value: 'simulation_v1',
    label: 'Simulation V1（Pre-event Logistic Regression）'
  },
  {
    value: 'hybrid_lstm',
    label: 'Hybrid LSTM（Pre-event）'
  }
]

function formatFeature(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(6)
}

function getModelVersionLabel(version) {
  const matched = MODEL_VERSION_OPTIONS.find((item) => item.value === version)
  return matched ? matched.label : version || '--'
}

export default function PredictionPage() {
  const [mode, setMode] = useState('history')

  // 历史事件预测
  const [options, setOptions] = useState([])
  const [selectedDate, setSelectedDate] = useState('')
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historySubmitting, setHistorySubmitting] = useState(false)
  const [historyResult, setHistoryResult] = useState(null)
  const [historyError, setHistoryError] = useState('')

  // 新事件模拟预测
  const [simForm, setSimForm] = useState({
    event_date: '',
    event_type: 3,
    event_source: 'manual',
    event_title: '',
    model_version: 'simulation_v1'
  })
  const [simContextLoading, setSimContextLoading] = useState(false)
  const [simContext, setSimContext] = useState(null)
  const [simSubmitting, setSimSubmitting] = useState(false)
  const [simResult, setSimResult] = useState(null)
  const [simError, setSimError] = useState('')

  useEffect(() => {
    async function initHistoryMode() {
      try {
        const res = await getPredictOptions()
        const list = res.data || []
        setOptions(list)

        if (list.length > 0) {
          setSelectedDate(list[0].event_date)
        }
      } catch (err) {
        console.error(err)
        setHistoryError('预测页初始化失败，请检查后端接口。')
      } finally {
        setHistoryLoading(false)
      }
    }

    initHistoryMode()
  }, [])

  useEffect(() => {
    setHistoryResult(null)
    setHistoryError('')
  }, [selectedDate])

  useEffect(() => {
    setSimResult(null)
    setSimError('')
  }, [
    simForm.event_date,
    simForm.event_type,
    simForm.event_source,
    simForm.event_title,
    simForm.model_version
  ])

  useEffect(() => {
    async function loadSimContext() {
      if (!simForm.event_date) {
        setSimContext(null)
        return
      }

      setSimContextLoading(true)
      try {
        const res = await getSimulateContext(simForm.event_date)
        setSimContext(res.data || null)
      } catch (err) {
        console.error(err)
        setSimContext(null)
        setSimError(
          err?.response?.data?.detail ||
            err?.message ||
            '模拟上下文获取失败，请检查日期是否合理。'
        )
      } finally {
        setSimContextLoading(false)
      }
    }

    if (mode === 'simulate') {
      loadSimContext()
    }
  }, [mode, simForm.event_date])

  const selectedEvent = useMemo(() => {
    return options.find((item) => item.event_date === selectedDate) || null
  }, [options, selectedDate])

  const simModelNote = useMemo(() => {
    if (simForm.model_version === 'hybrid_lstm') {
      return '当前使用 Hybrid LSTM（时序特征 + 静态特征融合），更适合做新事件模拟预测升级版实验。'
    }
    return '当前使用 Simulation V1（Pre-event Logistic Regression），适合作为新事件模拟预测基线模型。'
  }, [simForm.model_version])

  function handleModeChange(nextMode) {
    setMode(nextMode)
  }

  function handleSimInputChange(e) {
    const { name, value } = e.target
    setSimForm((prev) => ({
      ...prev,
      [name]: name === 'event_type' ? Number(value) : value
    }))
  }

  async function handleHistoryPredict() {
    if (!selectedEvent) return

    setHistorySubmitting(true)
    setHistoryError('')

    try {
      const res = await predictByEvent({
        event_date: selectedEvent.event_date,
        event_type: selectedEvent.event_type,
        event_title: selectedEvent.event_title
      })
      setHistoryResult(res.data)
    } catch (err) {
      console.error(err)
      setHistoryResult(null)

      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        '真实预测失败，请检查模型文件和后端日志。'

      setHistoryError(String(detail))
    } finally {
      setHistorySubmitting(false)
    }
  }

  async function handleSimPredict() {
    if (!simForm.event_date || !simForm.event_title) {
      setSimError('请填写事件日期和事件标题。')
      return
    }

    setSimSubmitting(true)
    setSimError('')

    try {
      const res = await predictSimulate(simForm)
      setSimResult(res.data)
    } catch (err) {
      console.error(err)
      setSimResult(null)

      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        '新事件模拟预测失败，请检查后端日志。'

      setSimError(String(detail))
    } finally {
      setSimSubmitting(false)
    }
  }

  if (historyLoading) {
    return <Loading text="预测页加载中..." />
  }

  return (
    <div className="prediction-page-wrap">
      <PageHeader
        title="预测结果"
        description="支持历史事件真实预测与新事件模拟预测两种模式。"
      />

      <div className="predict-mode-switch section-gap">
        <button
          className={`mode-btn ${mode === 'history' ? 'active' : ''}`}
          onClick={() => handleModeChange('history')}
        >
          历史事件预测
        </button>
        <button
          className={`mode-btn ${mode === 'simulate' ? 'active' : ''}`}
          onClick={() => handleModeChange('simulate')}
        >
          新事件模拟预测
        </button>
      </div>

      {mode === 'history' ? (
        <div className="prediction-layout section-gap">
          <div className="card prediction-main-panel">
            <div className="card-title">选择历史事件</div>

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
                当前真实预测依赖历史事件窗口特征，因此该模式适用于对历史事件进行真实推理与结果回看。
              </div>

              <button
                className="primary-btn"
                onClick={handleHistoryPredict}
                disabled={!selectedEvent || historySubmitting}
              >
                {historySubmitting ? '真实预测中...' : '运行真实预测'}
              </button>
            </div>
          </div>

          <div className="card prediction-output-panel">
            <div className="card-title">预测输出</div>

            {historyError ? <div className="status-box empty">{historyError}</div> : null}

            {!historyResult ? (
              <Empty text="尚未发起真实预测" />
            ) : (
              <div className="prediction-output-box">
                <div className="prediction-label-row">
                  <span className="prediction-label-title">预测方向</span>
                  <span
                    className={`prediction-label-badge ${
                      historyResult.prediction_label === '上涨' ? 'up' : 'down'
                    }`}
                  >
                    {historyResult.prediction_label}
                  </span>
                </div>

                <div className="grid grid-3 section-gap-small">
                  <KpiCard
                    title="预测概率"
                    value={formatPercent(historyResult.probability)}
                    extra="sigmoid 输出"
                  />
                  <KpiCard
                    title="预测置信度"
                    value={formatPercent(historyResult.confidence)}
                    extra="max(p, 1-p)"
                  />
                  <KpiCard
                    title="模型名称"
                    value={historyResult.model_name || '--'}
                    extra="当前真实推理模型"
                  />
                </div>

                <div className="selected-event-info section-gap">
                  <div><strong>事件日期：</strong>{historyResult.event_date}</div>
                  <div><strong>事件类型：</strong>{getEventTypeLabel(historyResult.event_type)}</div>
                  <div><strong>事件来源：</strong>{historyResult.event_source}</div>
                  <div><strong>事件标题：</strong>{historyResult.event_title}</div>
                  <div><strong>使用序列日期：</strong>{(historyResult.used_sequence_dates || []).join('、')}</div>
                </div>

                <div className="section-gap">
                  <div className="card-title">真实值回看</div>

                  {!historyResult.actual_is_available ? (
                    <Empty text="该事件暂时没有足够的未来交易日，无法回看真实结果。" />
                  ) : (
                    <>
                      <div className="grid grid-3 section-gap-small">
                        <KpiCard
                          title="真实方向"
                          value={historyResult.actual_label || '--'}
                          extra={`基于事件后 ${historyResult.actual_horizon_days ?? 3} 个交易日`}
                        />
                        <KpiCard
                          title="真实收益率"
                          value={formatPercent(historyResult.actual_post_return)}
                          extra={`${historyResult.event_date} → ${historyResult.actual_future_date || '--'}`}
                        />
                        <KpiCard
                          title="预测结果校验"
                          value={formatBoolResult(historyResult.actual_is_correct)}
                          extra="预测方向 vs 真实方向"
                        />
                      </div>

                      <div className="selected-event-info section-gap">
                        <div><strong>事件日收盘价：</strong>{formatNumber(historyResult.actual_event_close)}</div>
                        <div><strong>未来收盘价：</strong>{formatNumber(historyResult.actual_future_close)}</div>
                        <div><strong>未来日期：</strong>{historyResult.actual_future_date || '--'}</div>
                      </div>
                    </>
                  )}
                </div>

                <div className="section-gap">
                  <div className="card-title">模型输入特征序列</div>

                  {!historyResult.feature_sequence?.length ? (
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
                        {historyResult.feature_sequence.map((item, index) => (
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
      ) : (
        <div className="prediction-layout section-gap">
          <div className="card prediction-main-panel">
            <div className="card-title">输入新事件</div>

            <div className="form-box">
              <div className="form-item">
                <label>模型版本</label>
                <select
                  name="model_version"
                  value={simForm.model_version}
                  onChange={handleSimInputChange}
                >
                  {MODEL_VERSION_OPTIONS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-item">
                <label>事件日期</label>
                <input
                  type="date"
                  name="event_date"
                  value={simForm.event_date}
                  onChange={handleSimInputChange}
                />
              </div>

              <div className="form-item">
                <label>事件类型</label>
                <select
                  name="event_type"
                  value={simForm.event_type}
                  onChange={handleSimInputChange}
                >
                  {EVENT_TYPE_OPTIONS.map((item) => (
                    <option key={item} value={item}>
                      {getEventTypeLabel(item)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-item">
                <label>事件来源</label>
                <select
                  name="event_source"
                  value={simForm.event_source}
                  onChange={handleSimInputChange}
                >
                  <option value="manual">manual</option>
                  <option value="sina_bulletin">sina_bulletin</option>
                  <option value="eastmoney_notice">eastmoney_notice</option>
                </select>
              </div>

              <div className="form-item">
                <label>事件标题</label>
                <textarea
                  name="event_title"
                  rows="4"
                  value={simForm.event_title}
                  onChange={handleSimInputChange}
                  placeholder="请输入新事件标题，例如：贵州茅台拟实施特别分红方案"
                />
              </div>

              <div className="predict-note-box">
                模拟预测严格只使用事件发生前最近两个交易日上下文，不使用事件日及未来信息。
              </div>

              <div className="predict-note-box">
                {simModelNote}
              </div>

              <button
                className="primary-btn"
                onClick={handleSimPredict}
                disabled={!simForm.event_date || !simForm.event_title || simSubmitting}
              >
                {simSubmitting ? '模拟预测中...' : '运行新事件模拟预测'}
              </button>
            </div>
          </div>

          <div className="card prediction-output-panel">
            <div className="card-title">模拟预测输出</div>

            {simError ? <div className="status-box empty">{simError}</div> : null}

            {!simResult ? (
              <>
                <div className="card-title">预测上下文</div>
                {simContextLoading ? (
                  <Loading text="模拟上下文加载中..." />
                ) : !simContext ? (
                  <Empty text="请选择一个事件日期以获取模拟上下文" />
                ) : (
                  <div className="selected-event-info">
                    <div><strong>模拟模式：</strong>{simContext.simulation_mode}</div>
                    <div><strong>上下文日期：</strong>{(simContext.context_dates || []).join('、')}</div>
                    <div><strong>当前模型：</strong>{getModelVersionLabel(simForm.model_version)}</div>
                    <div><strong>说明：</strong>{simContext.note}</div>
                  </div>
                )}

                {simContext?.market_sequence?.length ? (
                  <div className="section-gap">
                    <div className="card-title">事件前市场上下文</div>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>日期</th>
                          <th>开盘价</th>
                          <th>收盘价</th>
                          <th>最高价</th>
                          <th>最低价</th>
                          <th>成交量</th>
                        </tr>
                      </thead>
                      <tbody>
                        {simContext.market_sequence.map((item, index) => (
                          <tr key={`${item.date}-${index}`}>
                            <td>{item.date}</td>
                            <td>{formatNumber(item.open)}</td>
                            <td>{formatNumber(item.close)}</td>
                            <td>{formatNumber(item.high)}</td>
                            <td>{formatNumber(item.low)}</td>
                            <td>{formatInteger(item.volume)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="prediction-output-box">
                <div className="prediction-label-row">
                  <span className="prediction-label-title">模拟预测方向</span>
                  <span
                    className={`prediction-label-badge ${
                      simResult.prediction_label === '上涨' ? 'up' : 'down'
                    }`}
                  >
                    {simResult.prediction_label}
                  </span>
                </div>

                <div className="grid grid-4 section-gap-small">
                  <KpiCard
                    title="预测概率"
                    value={formatPercent(simResult.probability)}
                    extra="模拟预测概率"
                  />
                  <KpiCard
                    title="预测置信度"
                    value={formatPercent(simResult.confidence)}
                    extra="max(p, 1-p)"
                  />
                  <KpiCard
                    title="模型名称"
                    value={simResult.model_name || '--'}
                    extra="当前推理模型"
                  />
                  <KpiCard
                    title="模型版本"
                    value={getModelVersionLabel(simResult.model_version || simForm.model_version)}
                    extra="前端选择 / 后端回传"
                  />
                </div>

                <div className="selected-event-info section-gap">
                  <div><strong>事件日期：</strong>{simResult.event_date}</div>
                  <div><strong>事件类型：</strong>{getEventTypeLabel(simResult.event_type)}</div>
                  <div><strong>事件来源：</strong>{simResult.event_source}</div>
                  <div><strong>事件标题：</strong>{simResult.event_title}</div>
                  <div><strong>上下文日期：</strong>{(simResult.context_dates || []).join('、')}</div>
                  <div><strong>模拟模式：</strong>{simResult.simulation_mode || '--'}</div>
                  <div><strong>说明：</strong>{simResult.note}</div>
                </div>

                <div className="section-gap">
                  <div className="card-title">标题特征</div>
                  <div className="grid grid-4 section-gap-small">
                    <KpiCard
                      title="标题长度"
                      value={simResult.text_features?.title_length ?? '--'}
                      extra="字符数"
                    />
                    <KpiCard
                      title="利好关键词数"
                      value={simResult.text_features?.positive_keyword_count ?? '--'}
                      extra="规则匹配"
                    />
                    <KpiCard
                      title="利空关键词数"
                      value={simResult.text_features?.negative_keyword_count ?? '--'}
                      extra="规则匹配"
                    />
                    <KpiCard
                      title="情感分数"
                      value={simResult.text_features?.sentiment_score ?? '--'}
                      extra="利好数 - 利空数"
                    />
                  </div>
                </div>

                <div className="section-gap">
                  <div className="card-title">事件前市场上下文</div>
                  {!simResult.market_sequence?.length ? (
                    <Empty text="暂无市场上下文数据" />
                  ) : (
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>日期</th>
                          <th>开盘价</th>
                          <th>收盘价</th>
                          <th>日内收益率</th>
                          <th>振幅</th>
                          <th>成交量变化</th>
                          <th>收盘价变化</th>
                        </tr>
                      </thead>
                      <tbody>
                        {simResult.market_sequence.map((item, index) => (
                          <tr key={`${item.date}-${index}`}>
                            <td>{item.date}</td>
                            <td>{formatNumber(item.open)}</td>
                            <td>{formatNumber(item.close)}</td>
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
      )}
    </div>
  )
}