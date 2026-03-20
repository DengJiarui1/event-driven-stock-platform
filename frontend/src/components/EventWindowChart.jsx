import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export default function EventWindowChart({ data = [], eventDate = '' }) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current || !data.length) return

    const chart = echarts.init(chartRef.current)

    const dates = data.map((item) => item.date)
    const closePrices = data.map((item) => Number(item.close))
    const volumes = data.map((item) => Number(item.volume))
    const eventIndex = dates.findIndex((d) => d === eventDate)

    const option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        top: 10,
        data: ['收盘价', '成交量']
      },
      grid: [
        {
          left: 60,
          right: 40,
          top: 60,
          height: '42%'
        },
        {
          left: 60,
          right: 40,
          top: '66%',
          height: '18%'
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          boundaryGap: false,
          axisLabel: {
            rotate: 20
          }
        },
        {
          type: 'category',
          gridIndex: 1,
          data: dates,
          axisLabel: {
            rotate: 20
          }
        }
      ],
      yAxis: [
        {
          type: 'value',
          name: '收盘价',
          scale: true,
          splitLine: {
            lineStyle: {
              type: 'dashed'
            }
          }
        },
        {
          type: 'value',
          gridIndex: 1,
          name: '成交量',
          splitLine: {
            show: false
          }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1]
        },
        {
          type: 'slider',
          xAxisIndex: [0, 1],
          bottom: 8
        }
      ],
      series: [
        {
          name: '收盘价',
          type: 'line',
          smooth: true,
          data: closePrices,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: {
            width: 3
          },
          areaStyle: {
            opacity: 0.08
          },
          markLine:
            eventIndex !== -1
              ? {
                symbol: ['none', 'none'],
                label: {
                  formatter: '事件日',
                  position: 'insideEndTop',
                  color: '#dc2626',
                  fontWeight: 'bold'
                },
                lineStyle: {
                  color: '#dc2626',
                  type: 'dashed',
                  width: 2
                },
                data: [{ xAxis: eventDate }]
              }
              : undefined,
          markPoint:
            eventIndex !== -1
              ? {
                symbolSize: 56,
                itemStyle: {
                  color: '#dc2626'
                },
                label: {
                  formatter: '事件日',
                  color: '#fff',
                  fontWeight: 'bold'
                },
                data: [
                  {
                    coord: [eventDate, closePrices[eventIndex]],
                    value: closePrices[eventIndex]
                  }
                ]
              }
              : undefined
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes,
          barWidth: 28
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
  }, [data, eventDate])

  if (!data.length) {
    return <div className="status-box empty">暂无事件窗口数据</div>
  }

  return <div ref={chartRef} className="event-window-chart event-window-chart-large" />
}