import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export default function ModelBarChart({ data = [] }) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return

    const chart = echarts.init(chartRef.current)

    const option = {
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: 50,
        right: 30,
        top: 50,
        bottom: 80
      },
      xAxis: {
        type: 'category',
        data: data.map((item) => item.model),
        axisLabel: {
          interval: 0,
          rotate: 20
        }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: {
          formatter: (value) => value.toFixed(1)
        }
      },
      series: [
        {
          name: 'Accuracy',
          type: 'bar',
          data: data.map((item) => item.accuracy),
          barWidth: 40,
          itemStyle: {
            borderRadius: [6, 6, 0, 0]
          },
          label: {
            show: true,
            position: 'top',
            formatter: ({ value }) => Number(value).toFixed(4)
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
  }, [data])

  return <div className="chart-box" ref={chartRef} />
}