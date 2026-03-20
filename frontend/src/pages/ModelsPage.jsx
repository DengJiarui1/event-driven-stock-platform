import { useEffect, useMemo, useState } from 'react'
import { getLatestInformerResult, getModelComparison } from '../api/stockApi'
import ModelBarChart from '../components/ModelBarChart'
import MetricBarChart from '../components/MetricBarChart'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'
import { formatMetric, formatPercent } from '../utils/display'

export default function ModelsPage() {
  const [classificationModels, setClassificationModels] = useState([])
  const [informerLatest, setInformerLatest] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [clsRes, informerRes] = await Promise.allSettled([
          getModelComparison(),
          getLatestInformerResult()
        ])

        if (clsRes.status === 'fulfilled') {
          setClassificationModels(clsRes.value.data || [])
        }

        if (informerRes.status === 'fulfilled') {
          setInformerLatest(informerRes.value.data || null)
        }
      } catch (error) {
        console.error('获取模型对比数据失败：', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const bestClassifier = useMemo(() => {
    if (!classificationModels.length) return null
    return [...classificationModels].sort((a, b) => Number(b.accuracy) - Number(a.accuracy))[0]
  }, [classificationModels])

  const informerMetrics = useMemo(() => {
    if (!informerLatest?.result?.metrics) return []
    const metrics = informerLatest.result.metrics
    return [
      { name: 'MAE', value: Number(metrics.mae) },
      { name: 'MSE', value: Number(metrics.mse) },
      { name: 'RMSE', value: Number(metrics.rmse) },
      { name: 'MAPE', value: Number(metrics.mape) }
    ].filter((item) => !Number.isNaN(item.value))
  }, [informerLatest])

  if (loading) {
    return <Loading text="模型对比数据加载中..." />
  }

  return (
    <div>
      <PageHeader
        title="模型对比"
        description="本页将分类模型与时序回归模型分开展示，并补充结论说明。"
      />

      <div className="section-gap">
        <div className="compare-section-title">分类模型对比</div>
      </div>

      <div className="grid grid-3 section-gap-small">
        <KpiCard title="分类模型数量" value={classificationModels.length} extra="纳入 Accuracy 对比" />
        <KpiCard
          title="最佳分类模型"
          value={bestClassifier ? bestClassifier.model : '--'}
          extra={bestClassifier ? `Accuracy: ${formatPercent(bestClassifier.accuracy)}` : '暂无数据'}
        />
        <KpiCard title="评价指标" value="Accuracy" extra="适用于涨跌方向分类任务" />
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">分类模型准确率柱状图</div>
          {classificationModels.length ? (
            <ModelBarChart data={classificationModels} />
          ) : (
            <Empty text="暂无分类模型数据" />
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card table-card">
          <div className="card-title">分类模型结果表</div>
          {!classificationModels.length ? (
            <Empty text="暂无分类模型结果" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>模型名称</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {classificationModels.map((item, index) => (
                  <tr key={`${item.model}-${index}`}>
                    <td>{item.model}</td>
                    <td>{formatPercent(item.accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">分类模型结论说明</div>
          <div className="compare-note-box">
            当前实验中，<strong>{bestClassifier?.model || '暂无结果'}</strong> 在分类任务上表现最好，
            表明在小样本、事件驱动的短期涨跌判断场景下，简单模型可能比复杂深度模型更稳定。
          </div>
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">时序回归模型对比</div>
      </div>

      <div className="grid grid-4 section-gap-small">
        <KpiCard title="回归模型" value="Informer" extra="Return-only 扩展实验" />
        <KpiCard title="最新任务状态" value={informerLatest ? informerLatest.status : '--'} extra="来自 Informer 实验中心" />
        <KpiCard title="输入形式" value="单变量收益率" extra="target = return" />
        <KpiCard title="评价指标" value="MAE / MSE / RMSE / MAPE" extra="适用于时序回归任务" />
      </div>

      <div className="section-gap">
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">Informer 回归误差指标柱状图</div>
            {informerMetrics.length ? (
              <MetricBarChart
                title="Informer 回归指标"
                data={informerMetrics}
                yAxisName="Metric Value"
                valueFormatter={(v) => Number(v).toFixed(4)}
              />
            ) : (
              <Empty text="暂无 Informer 成功结果" />
            )}
          </div>

          <div className="card">
            <div className="card-title">Informer 实验说明</div>
            {informerLatest?.result ? (
              <div className="selected-event-info">
                <div><strong>任务ID：</strong>{informerLatest.result.job_id}</div>
                <div><strong>输入文件：</strong>{informerLatest.result.input_csv}</div>
                <div><strong>结果目录：</strong>{informerLatest.result.result_dir}</div>
                <div><strong>pred_shape：</strong>{JSON.stringify(informerLatest.result.pred_shape)}</div>
                <div><strong>true_shape：</strong>{JSON.stringify(informerLatest.result.true_shape)}</div>
                <div><strong>seq_len：</strong>{informerLatest.result.config?.seq_len}</div>
                <div><strong>label_len：</strong>{informerLatest.result.config?.label_len}</div>
                <div><strong>pred_len：</strong>{informerLatest.result.config?.pred_len}</div>
              </div>
            ) : (
              <Empty text="暂无 Informer 说明信息" />
            )}
          </div>
        </div>
      </div>

      <div className="section-gap">
        <div className="card table-card">
          <div className="card-title">Informer 回归指标表</div>
          {!informerMetrics.length ? (
            <Empty text="暂无 Informer 指标结果" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>模型名称</th>
                  <th>任务类型</th>
                  <th>MAE</th>
                  <th>MSE</th>
                  <th>RMSE</th>
                  <th>MAPE</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Informer (Return-only)</td>
                  <td>时序回归</td>
                  <td>{formatMetric(informerLatest?.result?.metrics?.mae)}</td>
                  <td>{formatMetric(informerLatest?.result?.metrics?.mse)}</td>
                  <td>{formatMetric(informerLatest?.result?.metrics?.rmse)}</td>
                  <td>{formatMetric(informerLatest?.result?.metrics?.mape)}</td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">时序回归模型结论说明</div>
          <div className="compare-note-box">
            Informer 当前作为单变量收益率预测扩展实验使用。虽然其具备长序列建模能力，但在股票短期波动场景下，
            仅依赖收益率序列时预测结果更平滑、对突发波动的刻画能力有限，这进一步说明引入事件信息是必要的。
          </div>
        </div>
      </div>
    </div>
  )
}