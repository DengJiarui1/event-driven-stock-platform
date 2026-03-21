import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export default function ConfusionMatrixChart({
  matrix = [[0, 0], [0, 0]],
  xLabels = ['预测下跌', '预测上涨'],
  yLabels = ['实际下跌', '实际上涨'],
  title = '混淆矩阵'
}) {
  const chartRef = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return

    const chart = echarts.init(chartRef.current)

    const safeMatrix =
      Array.isArray(matrix) && matrix.length === 2 && Array.isArray(matrix[0]) && Array.isArray(matrix[1])
        ? matrix
        : [[0, 0], [0, 0]]

    const values = [
      [0, 0, Number(safeMatrix[0][0] || 0)], // TN
      [1, 0, Number(safeMatrix[0][1] || 0)], // FP
      [0, 1, Number(safeMatrix[1][0] || 0)], // FN
      [1, 1, Number(safeMatrix[1][1] || 0)]  // TP
    ]

    const maxValue = Math.max(...values.map((item) => item[2]), 1)

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
        position: 'top',
        formatter: (params) => {
          const x = xLabels[params.value[0]]
          const y = yLabels[params.value[1]]
          const v = params.value[2]
          return `${y} / ${x}<br/>数量：${v}`
        }
      },
      grid: {
        top: 60,
        bottom: 40,
        left: 70,
        right: 20
      },
      xAxis: {
        type: 'category',
        data: xLabels,
        splitArea: {
          show: true
        }
      },
      yAxis: {
        type: 'category',
        data: yLabels,
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: 0,
        max: maxValue,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0
      },
      series: [
        {
          name: 'Confusion Matrix',
          type: 'heatmap',
          data: values,
          label: {
            show: true,
            fontSize: 16,
            fontWeight: 'bold'
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.25)'
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
  }, [matrix, title, xLabels, yLabels])

  return <div className="chart-box confusion-chart-box" ref={chartRef} />
}