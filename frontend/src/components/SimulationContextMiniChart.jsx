import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'
import Empty from './Empty'

const METRIC_OPTIONS = [
  {
    key: 'intraday_return',
    label: '日内收益率',
    t2Key: 't_minus_2_intraday_return',
    t1Key: 't_minus_1_intraday_return',
    digits: 6
  },
  {
    key: 'amplitude',
    label: '振幅',
    t2Key: 't_minus_2_amplitude',
    t1Key: 't_minus_1_amplitude',
    digits: 6
  },
  {
    key: 'volume_change',
    label: '成交量变化',
    t2Key: 't_minus_2_volume_change_vs_prev',
    t1Key: 't_minus_1_volume_change_vs_prev',
    digits: 6
  },
  {
    key: 'close_change',
    label: '收盘价变化',
    t2Key: 't_minus_2_close_change_vs_prev_close',
    t1Key: 't_minus_1_close_change_vs_prev_close',
    digits: 6
  }
]

function toNumber(value) {
  const num = Number(value)
  return Number.isNaN(num) ? null : num
}

export default function SimulationContextMiniChart({ row }) {
  const chartRef = useRef(null)
  const [metricKey, setMetricKey] = useState('intraday_return')

  const activeMetric = useMemo(() => {
    return METRIC_OPTIONS.find((item) => item.key === metricKey) || METRIC_OPTIONS[0]
  }, [metricKey])

  const chartData = useMemo(() => {
    if (!row) return null

    const xData = [row.t_minus_2_date || 'T-2', row.t_minus_1_date || 'T-1']
    const yData = [
      toNumber(row[activeMetric.t2Key]),
      toNumber(row[activeMetric.t1Key])
    ]

    if (yData.every((v) => v === null)) return null

    return {
      xData,
      yData
    }
  }, [row, activeMetric])

  useEffect(() => {
    if (!chartRef.current || !chartData) return

    const chart = echarts.init(chartRef.current)

    const option = {
      title: {
        text: `${activeMetric.label}（T-2 / T-1）`,
        left: 'center',
        textStyle: {
          fontSize: 15,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params) => {
          const p = params?.[0]
          if (!p) return ''
          return `${p.axisValue}<br/>${activeMetric.label}：${Number(p.value).toFixed(activeMetric.digits)}`
        }
      },
      grid: {
        left: 60,
        right: 24,
        top: 56,
        bottom: 42
      },
      xAxis: {
        type: 'category',
        data: chartData.xData
      },
      yAxis: {
        type: 'value',
        name: activeMetric.label
      },
      series: [
        {
          type: 'line',
          data: chartData.yData,
          smooth: true,
          symbol: 'circle',
          symbolSize: 10,
          label: {
            show: true,
            position: 'top',
            formatter: ({ value }) => {
              const num = Number(value)
              if (Number.isNaN(num)) return '--'
              return num.toFixed(activeMetric.digits)
            }
          }
        }
      ]
    }

    chart.setOption(option)

    const handleResize = () => chart.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.dispose()
    }
  }, [chartData, activeMetric])

  if (!row) {
    return <Empty text="暂无上下文图表数据" />
  }

  return (
    <div>
      <div className="mini-metric-switch">
        {METRIC_OPTIONS.map((item) => (
          <button
            key={item.key}
            className={`mini-metric-btn ${metricKey === item.key ? 'active' : ''}`}
            onClick={() => setMetricKey(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="mini-context-chart" ref={chartRef} />
    </div>
  )
}