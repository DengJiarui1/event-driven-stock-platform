import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export default function MetricBarChart({
  title = '',
  data = [],
  yAxisName = '',
  valueFormatter,
}) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return

    const chart = echarts.init(chartRef.current)

    const option = {
      title: {
        text: title,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: 50,
        right: 30,
        top: 60,
        bottom: 60
      },
      xAxis: {
        type: 'category',
        data: data.map((item) => item.name),
        axisLabel: {
          interval: 0,
          rotate: 15
        }
      },
      yAxis: {
        type: 'value',
        name: yAxisName
      },
      series: [
        {
          type: 'bar',
          data: data.map((item) => item.value),
          barWidth: 40,
          label: {
            show: true,
            position: 'top',
            formatter: ({ value }) => {
              if (typeof valueFormatter === 'function') {
                return valueFormatter(value)
              }
              return Number(value).toFixed(4)
            }
          },
          itemStyle: {
            borderRadius: [6, 6, 0, 0]
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
  }, [data, title, yAxisName, valueFormatter])

  return <div className="chart-box" ref={chartRef} />
}