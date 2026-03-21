import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'
import {
  getBacktestDrawdown,
  getBacktestEquity,
  getBacktestLatest,
  getBacktestTrades,
  runBacktest
} from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'
import {
  formatInteger,
  formatNumber
} from '../utils/display'

const MODEL_OPTIONS = [
  { value: 'simulation_v1', label: 'Simulation V1' },
  { value: 'hybrid_lstm', label: 'Hybrid LSTM' }
]

function formatMaybePercent(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${(num * 100).toFixed(2)}%`
}

function formatMaybeNumber(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(digits)
}

function getModelLabel(modelName) {
  const found = MODEL_OPTIONS.find((item) => item.value === modelName)
  return found ? found.label : modelName || '--'
}

function getExitReasonLabel(reason) {
  const map = {
    hold_days: '到期平仓',
    stop_loss: '止损',
    take_profit: '止盈',
    manual: '手动/其他'
  }
  return map[reason] || reason || '--'
}

function getEffectiveStartIndex(rows, valueKey) {
  if (!rows.length) return -1

  const firstValue = Number(rows[0][valueKey])
  for (let i = 1; i < rows.length; i += 1) {
    const currentValue = Number(rows[i][valueKey])
    if (currentValue !== firstValue) {
      return i
    }
  }

  return -1
}

function buildAutoConclusion(summary, params) {
  if (!summary) return '暂无回测结论，请先运行一次回测。'

  const totalReturn = Number(summary.total_return)
  const maxDrawdown = Number(summary.max_drawdown)
  const sharpe = Number(summary.sharpe_ratio)
  const tradeCount = Number(summary.trade_count)
  const winRate = Number(summary.win_rate)
  const targetPercent = Number(params?.target_percent)

  let styleText = '策略整体表现中性。'
  if (!Number.isNaN(totalReturn) && !Number.isNaN(maxDrawdown)) {
    if (totalReturn > 0 && maxDrawdown > -0.05) {
      styleText = '策略呈现小幅盈利且回撤较低，整体偏保守。'
    } else if (totalReturn > 0 && maxDrawdown <= -0.05) {
      styleText = '策略实现了正收益，但回撤相对明显。'
    } else if (totalReturn <= 0 && maxDrawdown > -0.03) {
      styleText = '策略收益偏弱，但风险控制相对平稳。'
    } else {
      styleText = '策略当前收益与风险表现均不理想，仍需继续优化。'
    }
  }

  let tradeText = '交易频率正常。'
  if (!Number.isNaN(tradeCount)) {
    if (tradeCount <= 5) {
      tradeText = '当前交易次数较少，策略触发较为谨慎。'
    } else if (tradeCount <= 15) {
      tradeText = '当前交易次数适中，属于事件驱动策略较常见的触发频率。'
    } else {
      tradeText = '当前交易次数较多，需关注信号是否过于频繁。'
    }
  }

  let qualityText = '胜率表现一般。'
  if (!Number.isNaN(winRate) && !Number.isNaN(sharpe)) {
    if (winRate >= 0.6 && sharpe >= 0.5) {
      qualityText = '胜率和风险调整后收益表现较好。'
    } else if (winRate >= 0.5) {
      qualityText = '胜率尚可，但风险收益比还有提升空间。'
    } else {
      qualityText = '胜率偏低，说明当前模型信号转化为交易收益的稳定性不足。'
    }
  }

  const holdText = params?.hold_days
    ? `本次采用固定持有 ${params.hold_days} 天`
    : '本次采用固定持有策略'

  const thresholdText =
    params?.prob_threshold !== null &&
    params?.prob_threshold !== undefined &&
    params?.prob_threshold !== ''
      ? `，手动阈值设为 ${formatMaybeNumber(params.prob_threshold, 2)}`
      : '，使用模型自带阈值'

  const positionText = !Number.isNaN(targetPercent)
    ? `，目标仓位为 ${formatMaybePercent(targetPercent)}`
    : ''

  return `${styleText}${tradeText}${qualityText}${holdText}${thresholdText}${positionText}。`
}

function BacktestLineChart({
  title,
  rows,
  yKey,
  yName,
  valueFormatter = (value) => String(value)
}) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return
    const chart = echarts.init(chartRef.current)

    if (!rows?.length) {
      chart.clear()
      chart.setOption({
        title: {
          text: title,
          left: 'center',
          top: 10,
          textStyle: { fontSize: 15, fontWeight: 'bold' }
        }
      })
      return () => chart.dispose()
    }

    const xData = rows.map((item) => item.date)
    const yData = rows.map((item) => {
      const value = Number(item[yKey])
      return Number.isNaN(value) ? null : value
    })

    chart.setOption({
      title: {
        text: title,
        left: 'center',
        top: 10,
        textStyle: { fontSize: 15, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          const p = params?.[0]
          if (!p) return ''
          return `${p.axisValue}<br/>${yName}：${valueFormatter(p.value)}`
        }
      },
      grid: {
        left: 70,
        right: 24,
        top: 56,
        bottom: 42
      },
      xAxis: {
        type: 'category',
        data: xData
      },
      yAxis: {
        type: 'value',
        name: yName,
        scale: true
      },
      series: [
        {
          type: 'line',
          smooth: true,
          symbol: 'none',
          areaStyle: {},
          data: yData
        }
      ]
    })

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [rows, title, yKey, yName, valueFormatter])

  return <div className="backtest-chart" ref={chartRef} />
}

function BacktestMiniBarChart({
  title,
  data,
  yName = 'Count'
}) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return
    const chart = echarts.init(chartRef.current)

    if (!data?.length) {
      chart.clear()
      chart.setOption({
        title: {
          text: title,
          left: 'center',
          top: 10,
          textStyle: { fontSize: 15, fontWeight: 'bold' }
        }
      })
      return () => chart.dispose()
    }

    chart.setOption({
      title: {
        text: title,
        left: 'center',
        top: 10,
        textStyle: { fontSize: 15, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: 50,
        right: 24,
        top: 56,
        bottom: 42
      },
      xAxis: {
        type: 'category',
        data: data.map((item) => item.name)
      },
      yAxis: {
        type: 'value',
        name: yName
      },
      series: [
        {
          type: 'bar',
          data: data.map((item) => item.value),
          barMaxWidth: 56,
          label: {
            show: true,
            position: 'top'
          }
        }
      ]
    })

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [title, data, yName])

  return <div className="backtest-mini-chart" ref={chartRef} />
}

function BacktestMiniPieChart({
  title,
  data
}) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return
    const chart = echarts.init(chartRef.current)

    if (!data?.length) {
      chart.clear()
      chart.setOption({
        title: {
          text: title,
          left: 'center',
          top: 10,
          textStyle: { fontSize: 15, fontWeight: 'bold' }
        }
      })
      return () => chart.dispose()
    }

    chart.setOption({
      title: {
        text: title,
        left: 'center',
        top: 10,
        textStyle: { fontSize: 15, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)'
      },
      legend: {
        bottom: 0
      },
      series: [
        {
          type: 'pie',
          radius: ['42%', '68%'],
          center: ['50%', '52%'],
          data,
          label: {
            formatter: '{b}\n{c}'
          }
        }
      ]
    })

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [title, data])

  return <div className="backtest-mini-chart" ref={chartRef} />
}

export default function BacktestPage() {
  const [form, setForm] = useState({
    model_name: 'simulation_v1',
    initial_cash: 100000,
    commission: 0.001,
    hold_days: 3,
    prob_threshold: '',
    stop_loss_pct: 0.03,
    take_profit_pct: 0.05,
    target_percent: 0.95
  })

  const [pageLoading, setPageLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const [latestReport, setLatestReport] = useState(null)
  const [equityRows, setEquityRows] = useState([])
  const [drawdownRows, setDrawdownRows] = useState([])
  const [tradeRows, setTradeRows] = useState([])

  const summary = latestReport?.report?.summary || latestReport?.summary || null
  const latestParams = latestReport?.latest_meta?.params || latestReport?.report?.params || null

  const latestModelName = useMemo(() => {
    return latestReport?.model_name || form.model_name
  }, [latestReport, form.model_name])

  const normalizedEquityRows = useMemo(() => {
    const initialCash = Number(summary?.initial_cash)
    if (!equityRows.length || Number.isNaN(initialCash) || initialCash === 0) return []

    return equityRows.map((item) => {
      const portfolioValue = Number(item.portfolio_value)
      const nav = portfolioValue / initialCash
      const cumulativeReturn = (portfolioValue - initialCash) / initialCash

      return {
        ...item,
        nav,
        cumulative_return: cumulativeReturn
      }
    })
  }, [equityRows, summary])

  const effectiveEquityRows = useMemo(() => {
    if (!normalizedEquityRows.length) return []

    const startIndex = getEffectiveStartIndex(normalizedEquityRows, 'portfolio_value')

    if (startIndex === -1) {
      return normalizedEquityRows.slice(-120)
    }

    return normalizedEquityRows.slice(Math.max(0, startIndex - 5))
  }, [normalizedEquityRows])

  const effectiveDrawdownRows = useMemo(() => {
    if (!drawdownRows.length || !effectiveEquityRows.length) return []

    const firstDate = effectiveEquityRows[0]?.date
    if (!firstDate) return drawdownRows.slice(-120)

    const filtered = drawdownRows.filter((row) => row.date >= firstDate)
    return filtered.length ? filtered : drawdownRows.slice(-120)
  }, [drawdownRows, effectiveEquityRows])

  const autoConclusion = useMemo(() => {
    return buildAutoConclusion(summary, latestParams)
  }, [summary, latestParams])

  const exitReasonChartData = useMemo(() => {
    if (!tradeRows.length) return []

    const counter = {
    '到期平仓': 0,
    '止损': 0,
    '止盈': 0,
    '手动/其他': 0
    }

    tradeRows.forEach((row) => {
      const label = getExitReasonLabel(row.exit_reason)
      counter[label] = (counter[label] || 0) + 1
    })

    return Object.entries(counter)
      .map(([name, value]) => ({ name, value }))
      .filter((item) => item.value > 0)
  }, [tradeRows])

  const profitLossChartData = useMemo(() => {
    if (!tradeRows.length) return []

    let winCount = 0
    let lossCount = 0

    tradeRows.forEach((row) => {
      const pnl = Number(row.pnl_comm)
      if (Number.isNaN(pnl)) return
      if (pnl > 0) {
        winCount += 1
      } else {
        lossCount += 1
      }
    })

    return [
      { name: '盈利交易', value: winCount },
      { name: '亏损交易', value: lossCount }
    ]
  }, [tradeRows])

  async function loadBacktestData(modelName) {
    const [latestRes, equityRes, drawdownRes, tradesRes] = await Promise.all([
      getBacktestLatest(modelName),
      getBacktestEquity(modelName),
      getBacktestDrawdown(modelName),
      getBacktestTrades(modelName, 200)
    ])

    setLatestReport(latestRes.data || null)
    setEquityRows(equityRes.data?.rows || [])
    setDrawdownRows(drawdownRes.data?.rows || [])
    setTradeRows(tradesRes.data?.rows || [])
  }

  useEffect(() => {
    async function initPage() {
      try {
        await loadBacktestData(form.model_name)
      } catch (err) {
        console.error(err)
      } finally {
        setPageLoading(false)
      }
    }

    initPage()
  }, [])

  async function handleModelQuickSwitch(nextModel) {
    setForm((prev) => ({ ...prev, model_name: nextModel }))
    setError('')

    try {
      setPageLoading(true)
      await loadBacktestData(nextModel)
    } catch (err) {
      console.error(err)
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          '读取回测结果失败，请先运行一次对应模型回测。'
      )
    } finally {
      setPageLoading(false)
    }
  }

  function handleChange(e) {
    const { name, value } = e.target
    setForm((prev) => ({
      ...prev,
      [name]:
        [
          'initial_cash',
          'commission',
          'hold_days',
          'stop_loss_pct',
          'take_profit_pct',
          'target_percent'
        ].includes(name)
          ? Number(value)
          : value
    }))
  }

  async function handleRunBacktest() {
    setSubmitting(true)
    setError('')

    try {
      const payload = {
        model_name: form.model_name,
        initial_cash: Number(form.initial_cash),
        commission: Number(form.commission),
        hold_days: Number(form.hold_days),
        stop_loss_pct: Number(form.stop_loss_pct),
        take_profit_pct: Number(form.take_profit_pct),
        target_percent: Number(form.target_percent)
      }

      if (form.prob_threshold !== '' && form.prob_threshold !== null) {
        payload.prob_threshold = Number(form.prob_threshold)
      }

      await runBacktest(payload)
      await loadBacktestData(form.model_name)
    } catch (err) {
      console.error(err)
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          '运行回测失败，请检查后端日志。'
      )
    } finally {
      setSubmitting(false)
    }
  }

  if (pageLoading) {
    return <Loading text="回测页加载中..." />
  }

  return (
    <div>
      <PageHeader
        title="策略回测"
        description="基于模型预测信号，使用 Backtrader 完成事件驱动策略回测，并输出风险收益分析结果。"
      />

      <div className="section-gap">
        <div className="compare-section-title">回测参数设置</div>
      </div>

      <div className="prediction-layout section-gap">
        <div className="card prediction-main-panel">
          <div className="card-title">运行回测</div>

          <div className="form-box">
            <div className="form-item">
              <label>模型版本</label>
              <select
                name="model_name"
                value={form.model_name}
                onChange={(e) => {
                  handleChange(e)
                  handleModelQuickSwitch(e.target.value)
                }}
              >
                {MODEL_OPTIONS.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-item">
              <label>初始资金</label>
              <input
                type="number"
                name="initial_cash"
                value={form.initial_cash}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>手续费率</label>
              <input
                type="number"
                step="0.0001"
                name="commission"
                value={form.commission}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>固定持仓天数</label>
              <input
                type="number"
                name="hold_days"
                value={form.hold_days}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>目标仓位比例</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                name="target_percent"
                value={form.target_percent}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>手动覆盖阈值（可选）</label>
              <input
                type="number"
                step="0.01"
                name="prob_threshold"
                value={form.prob_threshold}
                onChange={(e) =>
                  setForm((prev) => ({
                    ...prev,
                    prob_threshold: e.target.value
                  }))
                }
                placeholder="留空则使用模型/信号自带阈值"
              />
            </div>

            <div className="form-item">
              <label>止损比例</label>
              <input
                type="number"
                step="0.01"
                name="stop_loss_pct"
                value={form.stop_loss_pct}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>止盈比例</label>
              <input
                type="number"
                step="0.01"
                name="take_profit_pct"
                value={form.take_profit_pct}
                onChange={handleChange}
              />
            </div>

            <div className="predict-note-box">
              当前策略为事件驱动单票 long-only 回测：模型给出看多信号时，在下一交易日开盘买入，并按固定持仓天数、目标仓位和止盈止损规则执行策略。
            </div>

            {error ? <div className="status-box empty">{error}</div> : null}

            <button
              className="primary-btn"
              onClick={handleRunBacktest}
              disabled={submitting}
            >
              {submitting ? '回测运行中...' : '运行回测'}
            </button>
          </div>
        </div>

        <div className="card prediction-output-panel">
          <div className="card-title">回测结果概览</div>

          {!summary ? (
            <Empty text="暂无回测结果，请先运行一次回测。" />
          ) : (
            <>
              <div className="selected-event-info">
                <div><strong>当前模型：</strong>{getModelLabel(latestModelName)}</div>
                <div><strong>策略名称：</strong>{latestReport?.report?.strategy_name || '--'}</div>
                <div><strong>模型标识：</strong>{latestReport?.report?.model_name || latestModelName}</div>
              </div>

              <div className="section-gap">
                <div className="card-title">本次回测配置摘要</div>
                <div className="selected-event-info">
                  <div><strong>持仓天数：</strong>{latestParams?.hold_days ?? '--'}</div>
                  <div><strong>目标仓位：</strong>{formatMaybePercent(latestParams?.target_percent)}</div>
                  <div>
                    <strong>概率阈值：</strong>
                    {latestParams?.prob_threshold !== null &&
                    latestParams?.prob_threshold !== undefined
                      ? formatMaybeNumber(latestParams.prob_threshold, 2)
                      : '使用模型自带阈值'}
                  </div>
                  <div><strong>止损比例：</strong>{formatMaybePercent(latestParams?.stop_loss_pct)}</div>
                  <div><strong>止盈比例：</strong>{formatMaybePercent(latestParams?.take_profit_pct)}</div>
                  <div><strong>手续费率：</strong>{formatMaybeNumber(latestParams?.commission, 4)}</div>
                  <div><strong>交易次数：</strong>{formatInteger(summary.trade_count)}</div>
                </div>
              </div>

              <div className="section-gap">
                <div className="card-title">核心指标</div>
                <div className="grid grid-3 section-gap-small">
                  <KpiCard
                    title="累计收益率"
                    value={formatMaybePercent(summary.total_return)}
                    extra="策略总收益表现"
                  />
                  <KpiCard
                    title="年化收益率"
                    value={formatMaybePercent(summary.annual_return)}
                    extra="按 252 个交易日折算"
                  />
                  <KpiCard
                    title="最大回撤"
                    value={formatMaybePercent(summary.max_drawdown)}
                    extra="风险控制核心指标"
                  />
                  <KpiCard
                    title="夏普比率"
                    value={formatMaybeNumber(summary.sharpe_ratio, 4)}
                    extra="风险调整后收益"
                  />
                  <KpiCard
                    title="胜率"
                    value={formatMaybePercent(summary.win_rate)}
                    extra="盈利交易占比"
                  />
                  <KpiCard
                    title="交易次数"
                    value={formatInteger(summary.trade_count)}
                    extra="完成交易总数"
                  />
                </div>
              </div>

              <div className="section-gap">
                <div className="card-title">扩展指标</div>
                <div className="grid grid-3 section-gap-small">
                  <KpiCard
                    title="初始资金"
                    value={formatNumber(summary.initial_cash)}
                    extra="回测起始资金"
                  />
                  <KpiCard
                    title="最终资金"
                    value={formatNumber(summary.final_value)}
                    extra="回测结束资金"
                  />
                  <KpiCard
                    title="平均盈利"
                    value={formatMaybePercent(summary.avg_win)}
                    extra="盈利交易平均收益率"
                  />
                  <KpiCard
                    title="平均亏损"
                    value={formatMaybePercent(summary.avg_loss)}
                    extra="亏损交易平均收益率"
                  />
                  <KpiCard
                    title="平均持仓天数"
                    value={formatMaybeNumber(summary.avg_hold_days, 2)}
                    extra="单笔交易平均 bar 数"
                  />
                  <KpiCard
                    title="Profit Factor"
                    value={formatMaybeNumber(summary.profit_factor, 4)}
                    extra="总盈利 / 总亏损绝对值"
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">自动结论</div>
      </div>

      <div className="card">
        <div className="compare-note-box">
          {autoConclusion}
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">净值与回撤曲线</div>
      </div>

      <div className="compare-note-box" style={{ marginBottom: 16 }}>
        图表当前仅展示资金首次发生变化后的有效回测区间，已隐藏前期长时间空仓且资金不变的区间。
      </div>

      <div className="grid grid-2 section-gap">
        <div className="card">
          <div className="card-title">净值曲线</div>
          {!effectiveEquityRows.length ? (
            <Empty text="暂无净值曲线数据" />
          ) : (
            <BacktestLineChart
              title="净值曲线"
              rows={effectiveEquityRows}
              yKey="nav"
              yName="NAV"
              valueFormatter={(value) => formatMaybeNumber(value, 4)}
            />
          )}
        </div>

        <div className="card">
          <div className="card-title">回撤曲线</div>
          {!effectiveDrawdownRows.length ? (
            <Empty text="暂无回撤曲线数据" />
          ) : (
            <BacktestLineChart
              title="回撤曲线"
              rows={effectiveDrawdownRows}
              yKey="drawdown_pct"
              yName="Drawdown"
              valueFormatter={(value) => formatMaybePercent(value)}
            />
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">交易结构分析</div>
      </div>

      <div className="grid grid-2 section-gap">
        <div className="card">
          <div className="card-title">退出原因分布</div>
          {!exitReasonChartData.length ? (
            <Empty text="暂无退出原因统计" />
          ) : (
            <BacktestMiniPieChart
              title="退出原因分布"
              data={exitReasonChartData}
            />
          )}
        </div>

        <div className="card">
          <div className="card-title">盈利 / 亏损交易数量对比</div>
          {!profitLossChartData.length ? (
            <Empty text="暂无盈亏交易统计" />
          ) : (
            <BacktestMiniBarChart
              title="盈利 / 亏损交易数量对比"
              data={profitLossChartData}
              yName="Trade Count"
            />
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">交易记录</div>
      </div>

      <div className="card table-card">
        {!tradeRows.length ? (
          <Empty text="暂无交易记录" />
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>信号日期</th>
                <th>开仓日期</th>
                <th>平仓日期</th>
                <th>开仓价</th>
                <th>平仓价</th>
                <th>收益率</th>
                <th>收益金额</th>
                <th>持仓天数</th>
                <th>退出原因</th>
                <th>预测概率</th>
                <th>阈值</th>
                <th>目标仓位</th>
                <th>事件标题</th>
              </tr>
            </thead>
            <tbody>
              {tradeRows.map((row, index) => (
                <tr key={`${row.entry_date}-${row.exit_date}-${index}`}>
                  <td>{row.signal_date || '--'}</td>
                  <td>{row.entry_date || '--'}</td>
                  <td>{row.exit_date || '--'}</td>
                  <td>{formatNumber(row.entry_price)}</td>
                  <td>{formatNumber(row.exit_price)}</td>
                  <td>{formatMaybePercent(row.return_pct)}</td>
                  <td>{formatNumber(row.pnl_comm)}</td>
                  <td>{formatMaybeNumber(row.hold_days, 0)}</td>
                  <td>{getExitReasonLabel(row.exit_reason)}</td>
                  <td>{formatMaybePercent(row.signal_prob)}</td>
                  <td>{formatMaybeNumber(row.signal_threshold, 2)}</td>
                  <td>{formatMaybePercent(row.target_percent)}</td>
                  <td>{row.event_title || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}