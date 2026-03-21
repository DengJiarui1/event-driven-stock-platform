import { useEffect, useMemo, useState } from 'react'
import {
  getLatestInformerResult,
  getModelComparison,
  getSimulationModelReport,
  getSimulationTestPredictions,
  getSimHybridReport,
  getSimHybridTestPredictions
} from '../api/stockApi'
import ModelBarChart from '../components/ModelBarChart'
import MetricBarChart from '../components/MetricBarChart'
import ConfusionMatrixChart from '../components/ConfusionMatrixChart'
import SimulationSampleDrawer from '../components/SimulationSampleDrawer'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'
import {
  formatMetric,
  formatPercent,
  getEventTypeLabel
} from '../utils/display'

function formatProb(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${(num * 100).toFixed(2)}%`
}

function formatPlainNumber(value, digits = 4) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(digits)
}

function formatSigned(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${num >= 0 ? '+' : ''}${num.toFixed(digits)}`
}

function getDefaultSimulationRow(payload) {
  if (!payload?.rows?.length) return null
  const wrongRow = payload.rows.find((row) => row.is_correct === false)
  return wrongRow || payload.rows[0]
}

function getMetricNumber(report, key) {
  const value = Number(report?.metrics?.[key])
  return Number.isNaN(value) ? null : value
}

function getModelWinner(simV1, hybrid, key) {
  const a = getMetricNumber(simV1, key)
  const b = getMetricNumber(hybrid, key)

  if (a == null && b == null) return '--'
  if (a != null && b == null) return 'Simulation V1'
  if (a == null && b != null) return 'Hybrid LSTM'
  if (a === b) return '两者相同'
  return a > b ? 'Simulation V1' : 'Hybrid LSTM'
}

export default function ModelsPage() {
  const [classificationModels, setClassificationModels] = useState([])
  const [informerLatest, setInformerLatest] = useState(null)
  const [simulationReport, setSimulationReport] = useState(null)
  const [simHybridReport, setSimHybridReport] = useState(null)
  const [simulationPredictions, setSimulationPredictions] = useState(null)
  const [showOnlyError, setShowOnlyError] = useState(false)
  const [selectedSimulationRow, setSelectedSimulationRow] = useState(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [predictionTableLoading, setPredictionTableLoading] = useState(false)
  const [simHybridPredictions, setSimHybridPredictions] = useState(null)
  const [showOnlyHybridError, setShowOnlyHybridError] = useState(false)
  const [selectedHybridRow, setSelectedHybridRow] = useState(null)
  const [hybridDrawerOpen, setHybridDrawerOpen] = useState(false)
  const [hybridPredictionTableLoading, setHybridPredictionTableLoading] = useState(false)



  useEffect(() => {
    async function fetchData() {
      try {
        const [clsRes, informerRes, simRes, predRes, hybridRes, hybridPredRes] = await Promise.allSettled([
          getModelComparison(),
          getLatestInformerResult(),
          getSimulationModelReport(),
          getSimulationTestPredictions({ limit: 30, only_error: false }),
          getSimHybridReport(),
          getSimHybridTestPredictions({ limit: 30, only_error: false })
        ])

        if (hybridPredRes.status === 'fulfilled') {
          const data = hybridPredRes.value.data || null
          setSimHybridPredictions(data)
          setSelectedHybridRow(getDefaultSimulationRow(data))
        }

        if (clsRes.status === 'fulfilled') {
          setClassificationModels(clsRes.value.data || [])
        }

        if (informerRes.status === 'fulfilled') {
          setInformerLatest(informerRes.value.data || null)
        }

        if (simRes.status === 'fulfilled') {
          setSimulationReport(simRes.value.data || null)
        }

        if (predRes.status === 'fulfilled') {
          const data = predRes.value.data || null
          setSimulationPredictions(data)
          setSelectedSimulationRow(getDefaultSimulationRow(data))
        }

        if (hybridRes.status === 'fulfilled') {
          setSimHybridReport(hybridRes.value.data || null)
        }
      } catch (error) {
        console.error('获取模型对比数据失败：', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  useEffect(() => {
    async function fetchPredictionRows() {
      try {
        setPredictionTableLoading(true)
        const res = await getSimulationTestPredictions({
          limit: 30,
          only_error: showOnlyError
        })
        const data = res.data || null
        setSimulationPredictions(data)
        setSelectedSimulationRow(getDefaultSimulationRow(data))
      } catch (error) {
        console.error('获取 Simulation V1 测试明细失败：', error)
      } finally {
        setPredictionTableLoading(false)
      }
    }

    fetchPredictionRows()
  }, [showOnlyError])

  useEffect(() => {
    async function fetchHybridPredictionRows() {
      try {
        setHybridPredictionTableLoading(true)
        const res = await getSimHybridTestPredictions({
          limit: 30,
          only_error: showOnlyHybridError
        })
        const data = res.data || null
        setSimHybridPredictions(data)
        setSelectedHybridRow(getDefaultSimulationRow(data))
      } catch (error) {
        console.error('获取 Hybrid LSTM 测试明细失败：', error)
      } finally {
        setHybridPredictionTableLoading(false)
      }
    }

    fetchHybridPredictionRows()
  }, [showOnlyHybridError])

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

  const simulationMetrics = useMemo(() => {
    if (!simulationReport?.metrics) return []
    const metrics = simulationReport.metrics
    return [
      { name: 'Accuracy', value: Number(metrics.accuracy) },
      { name: 'Precision', value: Number(metrics.precision) },
      { name: 'Recall', value: Number(metrics.recall) },
      { name: 'F1', value: Number(metrics.f1) }
    ].filter((item) => !Number.isNaN(item.value))
  }, [simulationReport])

  const simHybridMetrics = useMemo(() => {
    if (!simHybridReport?.metrics) return []
    const metrics = simHybridReport.metrics
    return [
      { name: 'Accuracy', value: Number(metrics.accuracy) },
      { name: 'Precision', value: Number(metrics.precision) },
      { name: 'Recall', value: Number(metrics.recall) },
      { name: 'F1', value: Number(metrics.f1) }
    ].filter((item) => !Number.isNaN(item.value))
  }, [simHybridReport])

  const simulationMatrix = useMemo(() => {
    if (!simulationReport?.confusion_matrix) return [[0, 0], [0, 0]]
    return simulationReport.confusion_matrix
  }, [simulationReport])

  const simHybridMatrix = useMemo(() => {
    if (!simHybridReport?.confusion_matrix) return [[0, 0], [0, 0]]
    return simHybridReport.confusion_matrix
  }, [simHybridReport])

  const comparisonWinnerAccuracy = useMemo(
    () => getModelWinner(simulationReport, simHybridReport, 'accuracy'),
    [simulationReport, simHybridReport]
  )

  const comparisonWinnerF1 = useMemo(
    () => getModelWinner(simulationReport, simHybridReport, 'f1'),
    [simulationReport, simHybridReport]
  )

  const sentimentDistributionMetrics = useMemo(() => {
    const sentiment = simulationPredictions?.error_analysis?.sentiment_analysis
    if (!sentiment) return []
    return [
      { name: '利好倾向误判', value: Number(sentiment.wrong_positive_count || 0) },
      { name: '中性标题误判', value: Number(sentiment.wrong_neutral_count || 0) },
      { name: '利空倾向误判', value: Number(sentiment.wrong_negative_count || 0) }
    ]
  }, [simulationPredictions])

  const hybridSentimentDistributionMetrics = useMemo(() => {
    const sentiment = simHybridPredictions?.error_analysis?.sentiment_analysis
    if (!sentiment) return []
    return [
      { name: '利好倾向误判', value: Number(sentiment.wrong_positive_count || 0) },
      { name: '中性标题误判', value: Number(sentiment.wrong_neutral_count || 0) },
      { name: '利空倾向误判', value: Number(sentiment.wrong_negative_count || 0) }
    ]
  }, [simHybridPredictions])

  if (loading) {
    return <Loading text="模型对比数据加载中..." />
  }

  return (
    <div>
      <PageHeader
        title="模型对比"
        description="本页将分类模型、时序回归模型与新事件模拟预测模型分开展示，避免不同任务评价指标混用。"
      />

      {/* 分类模型对比 */}
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

      {/* 时序回归模型对比 */}
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

      {/* 新事件模拟预测模型对比 */}
      <div className="section-gap">
        <div className="compare-section-title">新事件模拟预测模型对比</div>
      </div>

      <div className="grid grid-4 section-gap-small">
        <KpiCard
          title="参与对比模型"
          value="2"
          extra="Simulation V1 / Hybrid LSTM"
        />
        <KpiCard
          title="Accuracy 更优"
          value={comparisonWinnerAccuracy}
          extra="新事件模拟预测任务"
        />
        <KpiCard
          title="F1 更优"
          value={comparisonWinnerF1}
          extra="分类综合指标"
        />
        <KpiCard
          title="任务特点"
          value="Pre-event"
          extra="只使用事件前可获得信息"
        />
      </div>

      <div className="section-gap">
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">Simulation V1 指标</div>
            <div className="grid grid-2 section-gap-small">
              <KpiCard
                title="Accuracy"
                value={simulationReport?.metrics?.accuracy != null ? formatPercent(simulationReport.metrics.accuracy) : '--'}
                extra="基线模型"
              />
              <KpiCard
                title="F1-score"
                value={simulationReport?.metrics?.f1 != null ? formatPercent(simulationReport.metrics.f1) : '--'}
                extra="分类综合指标"
              />
            </div>

            {simulationMetrics.length ? (
              <MetricBarChart
                title="Simulation V1 分类指标"
                data={simulationMetrics}
                yAxisName="Metric Value"
                valueFormatter={(v) => `${(Number(v) * 100).toFixed(2)}%`}
              />
            ) : (
              <Empty text="暂无 Simulation V1 结果" />
            )}
          </div>

          <div className="card">
            <div className="card-title">Hybrid LSTM 指标</div>
            <div className="grid grid-2 section-gap-small">
              <KpiCard
                title="Accuracy"
                value={simHybridReport?.metrics?.accuracy != null ? formatPercent(simHybridReport.metrics.accuracy) : '--'}
                extra="升级模型"
              />
              <KpiCard
                title="F1-score"
                value={simHybridReport?.metrics?.f1 != null ? formatPercent(simHybridReport.metrics.f1) : '--'}
                extra="分类综合指标"
              />
              <KpiCard
                title="最佳阈值"
                value={simHybridReport?.best_threshold != null ? formatPlainNumber(simHybridReport.best_threshold, 2) : '--'}
                extra="验证集阈值扫描"
              />
              <KpiCard
                title="pos_weight"
                value={simHybridReport?.pos_weight != null ? formatPlainNumber(simHybridReport.pos_weight, 4) : '--'}
                extra="BCEWithLogitsLoss 类别权重"
              />
            </div>

            {simHybridMetrics.length ? (
              <MetricBarChart
                title="Hybrid LSTM 分类指标"
                data={simHybridMetrics}
                yAxisName="Metric Value"
                valueFormatter={(v) => `${(Number(v) * 100).toFixed(2)}%`}
              />
            ) : (
              <Empty text="暂无 Hybrid LSTM 结果" />
            )}
          </div>
        </div>
      </div>

      <div className="section-gap">
        <div className="card table-card">
          <div className="card-title">新事件模拟预测模型对比表</div>
          {(!simulationReport?.metrics && !simHybridReport?.metrics) ? (
            <Empty text="暂无新事件模拟预测模型对比结果" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>模型名称</th>
                  <th>任务类型</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>最佳阈值</th>
                  <th>pos_weight</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{simulationReport?.model_name || 'Simulation V1'}</td>
                  <td>新事件模拟预测</td>
                  <td>{simulationReport?.metrics ? formatPercent(simulationReport.metrics.accuracy) : '--'}</td>
                  <td>{simulationReport?.metrics ? formatPercent(simulationReport.metrics.precision) : '--'}</td>
                  <td>{simulationReport?.metrics ? formatPercent(simulationReport.metrics.recall) : '--'}</td>
                  <td>{simulationReport?.metrics ? formatPercent(simulationReport.metrics.f1) : '--'}</td>
                  <td>--</td>
                  <td>--</td>
                </tr>
                <tr>
                  <td>{simHybridReport?.model_name || 'Hybrid LSTM'}</td>
                  <td>新事件模拟预测</td>
                  <td>{simHybridReport?.metrics ? formatPercent(simHybridReport.metrics.accuracy) : '--'}</td>
                  <td>{simHybridReport?.metrics ? formatPercent(simHybridReport.metrics.precision) : '--'}</td>
                  <td>{simHybridReport?.metrics ? formatPercent(simHybridReport.metrics.recall) : '--'}</td>
                  <td>{simHybridReport?.metrics ? formatPercent(simHybridReport.metrics.f1) : '--'}</td>
                  <td>{simHybridReport?.best_threshold != null ? formatPlainNumber(simHybridReport.best_threshold, 2) : '--'}</td>
                  <td>{simHybridReport?.pos_weight != null ? formatPlainNumber(simHybridReport.pos_weight, 4) : '--'}</td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="grid grid-2">
          <div className="card">
            <div className="card-title">Simulation V1 混淆矩阵</div>
            {simulationReport?.confusion_matrix ? (
              <ConfusionMatrixChart
                matrix={simulationMatrix}
                xLabels={['预测下跌', '预测上涨']}
                yLabels={['实际下跌', '实际上涨']}
                title="Simulation V1 Confusion Matrix"
              />
            ) : (
              <Empty text="暂无 Simulation V1 混淆矩阵" />
            )}
          </div>

          <div className="card">
            <div className="card-title">Hybrid LSTM 混淆矩阵</div>
            {simHybridReport?.confusion_matrix ? (
              <ConfusionMatrixChart
                matrix={simHybridMatrix}
                xLabels={['预测下跌', '预测上涨']}
                yLabels={['实际下跌', '实际上涨']}
                title="Hybrid LSTM Confusion Matrix"
              />
            ) : (
              <Empty text="暂无 Hybrid LSTM 混淆矩阵" />
            )}
          </div>
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">新事件模拟预测模型结论说明</div>
          <div className="compare-note-box">
            Simulation V1 作为基线模型，更容易快速构建并解释；Hybrid LSTM 则将事件前市场时序特征与事件静态特征进行融合，
            更符合事件驱动深度模型的研究方向。当前对比结果中，Accuracy 更优模型为
            <strong> {comparisonWinnerAccuracy} </strong>，
            F1 更优模型为
            <strong> {comparisonWinnerF1} </strong>。
            当前 Hybrid LSTM 使用的最佳阈值为
            <strong> {simHybridReport?.best_threshold != null ? ` ${formatPlainNumber(simHybridReport.best_threshold, 2)} ` : ' -- '} </strong>，
            训练时使用的 pos_weight 为
            <strong> {simHybridReport?.pos_weight != null ? ` ${formatPlainNumber(simHybridReport.pos_weight, 4)} ` : ' -- '} </strong>。
            这表明该模型已经引入阈值扫描与类别权重优化，不再固定使用默认 0.50 阈值。
          </div>
        </div>
      </div>

      {/* 以下保留你原来的 Simulation V1 详细分析区 */}
      <div className="section-gap">
        <div className="compare-section-title">Simulation V1 详细分析</div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">误判样本自动分析</div>

          {!simulationPredictions?.error_analysis ? (
            <Empty text="暂无误判样本分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判为上涨(FP)"
                  value={simulationPredictions.error_analysis.false_positive_count ?? '--'}
                  extra="实际下跌，预测上涨"
                />
                <KpiCard
                  title="误判为下跌(FN)"
                  value={simulationPredictions.error_analysis.false_negative_count ?? '--'}
                  extra="实际上涨，预测下跌"
                />
                <KpiCard
                  title="低置信度误判"
                  value={simulationPredictions.error_analysis.low_confidence_wrong_count ?? '--'}
                  extra="confidence < 0.60"
                />
                <KpiCard
                  title="高置信度误判"
                  value={simulationPredictions.error_analysis.high_confidence_wrong_count ?? '--'}
                  extra="confidence ≥ 0.60"
                />
              </div>

              <div className="section-gap">
                <div className="compare-note-box">
                  {simulationPredictions.error_analysis.summary_text || '暂无自动结论。'}
                </div>
              </div>

              <div className="section-gap">
                <div className="card-title">误判事件类型分布</div>

                {!simulationPredictions.error_analysis.top_error_event_types?.length ? (
                  <Empty text="暂无误判事件类型统计" />
                ) : (
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>事件类型</th>
                        <th>误判数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {simulationPredictions.error_analysis.top_error_event_types.map((item, index) => (
                        <tr key={`${item.event_type_raw}-${index}`}>
                          <td>{getEventTypeLabel(item.event_type_raw)}</td>
                          <td>{item.error_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">误判样本标题关键词分析</div>

          {!simulationPredictions?.error_analysis?.keyword_analysis ? (
            <Empty text="暂无误判样本标题关键词分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判样本总数"
                  value={simulationPredictions.error_analysis.keyword_analysis.total_wrong_titles ?? '--'}
                  extra="参与标题分析"
                />
                <KpiCard
                  title="命中关键词样本"
                  value={simulationPredictions.error_analysis.keyword_analysis.titles_with_keyword_count ?? '--'}
                  extra="标题中出现预设关键词"
                />
                <KpiCard
                  title="未命中关键词样本"
                  value={simulationPredictions.error_analysis.keyword_analysis.no_keyword_match_count ?? '--'}
                  extra="标题未出现预设关键词"
                />
                <KpiCard
                  title="分析方式"
                  value="规则关键词"
                  extra="标题命中统计"
                />
              </div>

              <div className="section-gap">
                <div className="compare-note-box">
                  {simulationPredictions.error_analysis.keyword_analysis.summary_text || '暂无关键词分析结论。'}
                </div>
              </div>

              <div className="section-gap">
                <div className="card-title">高频误判关键词</div>

                {!simulationPredictions.error_analysis.keyword_analysis.top_keywords?.length ? (
                  <Empty text="暂无高频关键词统计" />
                ) : (
                  <div className="keyword-chip-list">
                    {simulationPredictions.error_analysis.keyword_analysis.top_keywords.map((item, index) => (
                      <span className="keyword-chip" key={`${item.keyword}-${index}`}>
                        {item.keyword}（{item.count}）
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">标题情感得分分析</div>

          {!simulationPredictions?.error_analysis?.sentiment_analysis ? (
            <Empty text="暂无标题情感得分分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判样本平均情感分"
                  value={formatSigned(simulationPredictions.error_analysis.sentiment_analysis.avg_wrong_sentiment)}
                  extra="错误样本标题平均 sentiment_score"
                />
                <KpiCard
                  title="正确样本平均情感分"
                  value={formatSigned(simulationPredictions.error_analysis.sentiment_analysis.avg_correct_sentiment)}
                  extra="正确样本标题平均 sentiment_score"
                />
                <KpiCard
                  title="主导误判情感"
                  value={simulationPredictions.error_analysis.sentiment_analysis.dominant_wrong_sentiment_label || '--'}
                  extra="误判标题主要倾向"
                />
                <KpiCard
                  title="情感分差"
                  value={formatSigned(simulationPredictions.error_analysis.sentiment_analysis.sentiment_gap)}
                  extra="误判样本 - 正确样本"
                />
              </div>

              <div className="section-gap">
                <div className="grid grid-2">
                  <div className="card">
                    <div className="card-title">误判样本情感分布</div>
                    {sentimentDistributionMetrics.length ? (
                      <MetricBarChart
                        title="错误样本情感分布"
                        data={sentimentDistributionMetrics}
                        yAxisName="Count"
                        valueFormatter={(v) => Number(v).toFixed(0)}
                      />
                    ) : (
                      <Empty text="暂无情感分布统计" />
                    )}
                  </div>

                  <div className="card">
                    <div className="card-title">情感分析结论</div>
                    <div className="compare-note-box">
                      {simulationPredictions.error_analysis.sentiment_analysis.summary_text || '暂无情感分析结论。'}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">Simulation V1 测试集明细统计</div>
          {!simulationPredictions ? (
            <Empty text="暂无测试集明细统计" />
          ) : (
            <div className="grid grid-4 section-gap-small">
              <KpiCard title="测试样本总数" value={simulationPredictions.total_rows} extra="完整测试集规模" />
              <KpiCard title="预测正确数" value={simulationPredictions.correct_rows} extra="y_true == y_pred" />
              <KpiCard title="预测错误数" value={simulationPredictions.wrong_rows} extra="y_true != y_pred" />
              <KpiCard
                title="当前筛选"
                value={showOnlyError ? '仅误判样本' : '全部样本'}
                extra="可切换查看"
              />
            </div>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="table-toolbar">
          <div className="card-title">Simulation V1 测试集明细表</div>
          <button
            className={`mode-btn ${showOnlyError ? 'active' : ''}`}
            onClick={() => setShowOnlyError((prev) => !prev)}
          >
            {showOnlyError ? '查看全部样本' : '仅看误判样本'}
          </button>
        </div>

        <div className="card table-card">
          {predictionTableLoading ? (
            <Loading text="测试集明细加载中..." />
          ) : !simulationPredictions?.rows?.length ? (
            <Empty text="暂无 Simulation V1 测试集明细" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>事件日期</th>
                  <th>事件类型</th>
                  <th>事件标题</th>
                  <th>真实标签</th>
                  <th>预测标签</th>
                  <th>预测概率(上涨)</th>
                  <th>是否正确</th>
                </tr>
              </thead>
              <tbody>
                {simulationPredictions.rows.map((row, index) => {
                  const isActive =
                    selectedSimulationRow &&
                    row.event_date === selectedSimulationRow.event_date &&
                    row.event_title === selectedSimulationRow.event_title &&
                    row.t_minus_1_date === selectedSimulationRow.t_minus_1_date

                  return (
                    <tr
                      key={`${row.event_date}-${index}`}
                      className={`clickable-row ${isActive ? 'active-row' : ''}`}
                      onClick={() => {
                        setSelectedSimulationRow(row)
                        setDrawerOpen(true)
                      }}
                    >
                      <td>{row.event_date}</td>
                      <td>{getEventTypeLabel(row.event_type_raw)}</td>
                      <td>{row.event_title}</td>
                      <td>{row.actual_label || '--'}</td>
                      <td>{row.pred_label || '--'}</td>
                      <td>{row.pred_prob_up != null ? formatProb(row.pred_prob_up) : '--'}</td>
                      <td>
                        <span className={`truth-badge ${row.is_correct ? 'correct' : 'wrong'}`}>
                          {row.is_correct ? '正确' : '错误'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">Simulation V1 详细分析结论</div>
          <div className="compare-note-box">
            当前页面保留了 Simulation V1 的详细误判分析模块，包括误判分布、关键词分析、情感分析、测试集明细与单样本抽屉诊断，
            便于对基线模型的误差来源进行更深入观察。
          </div>
        </div>
      </div>

      <div className="section-gap">
        <div className="compare-section-title">Hybrid LSTM 详细分析</div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">误判样本自动分析</div>

          {!simHybridPredictions?.error_analysis ? (
            <Empty text="暂无 Hybrid LSTM 误判样本分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判为上涨(FP)"
                  value={simHybridPredictions.error_analysis.false_positive_count ?? '--'}
                  extra="实际下跌，预测上涨"
                />
                <KpiCard
                  title="误判为下跌(FN)"
                  value={simHybridPredictions.error_analysis.false_negative_count ?? '--'}
                  extra="实际上涨，预测下跌"
                />
                <KpiCard
                  title="低置信度误判"
                  value={simHybridPredictions.error_analysis.low_confidence_wrong_count ?? '--'}
                  extra="confidence < 0.60"
                />
                <KpiCard
                  title="高置信度误判"
                  value={simHybridPredictions.error_analysis.high_confidence_wrong_count ?? '--'}
                  extra="confidence ≥ 0.60"
                />
              </div>

              <div className="section-gap">
                <div className="compare-note-box">
                  {simHybridPredictions.error_analysis.summary_text || '暂无自动结论。'}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">标题关键词分析</div>

          {!simHybridPredictions?.error_analysis?.keyword_analysis ? (
            <Empty text="暂无 Hybrid LSTM 标题关键词分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判样本总数"
                  value={simHybridPredictions.error_analysis.keyword_analysis.total_wrong_titles ?? '--'}
                  extra="参与标题分析"
                />
                <KpiCard
                  title="命中关键词样本"
                  value={simHybridPredictions.error_analysis.keyword_analysis.titles_with_keyword_count ?? '--'}
                  extra="标题中出现预设关键词"
                />
                <KpiCard
                  title="未命中关键词样本"
                  value={simHybridPredictions.error_analysis.keyword_analysis.no_keyword_match_count ?? '--'}
                  extra="标题未出现预设关键词"
                />
                <KpiCard
                  title="分析方式"
                  value="规则关键词"
                  extra="标题命中统计"
                />
              </div>

              <div className="section-gap">
                <div className="compare-note-box">
                  {simHybridPredictions.error_analysis.keyword_analysis.summary_text || '暂无关键词分析结论。'}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">标题情感得分分析</div>

          {!simHybridPredictions?.error_analysis?.sentiment_analysis ? (
            <Empty text="暂无 Hybrid LSTM 情感分析结果" />
          ) : (
            <>
              <div className="grid grid-4 section-gap-small">
                <KpiCard
                  title="误判样本平均情感分"
                  value={formatSigned(simHybridPredictions.error_analysis.sentiment_analysis.avg_wrong_sentiment)}
                  extra="错误样本标题平均 sentiment_score"
                />
                <KpiCard
                  title="正确样本平均情感分"
                  value={formatSigned(simHybridPredictions.error_analysis.sentiment_analysis.avg_correct_sentiment)}
                  extra="正确样本标题平均 sentiment_score"
                />
                <KpiCard
                  title="主导误判情感"
                  value={simHybridPredictions.error_analysis.sentiment_analysis.dominant_wrong_sentiment_label || '--'}
                  extra="误判标题主要倾向"
                />
                <KpiCard
                  title="情感分差"
                  value={formatSigned(simHybridPredictions.error_analysis.sentiment_analysis.sentiment_gap)}
                  extra="误判样本 - 正确样本"
                />
              </div>

              <div className="section-gap">
                <div className="grid grid-2">
                  <div className="card">
                    <div className="card-title">误判样本情感分布</div>
                    {hybridSentimentDistributionMetrics.length ? (
                      <MetricBarChart
                        title="Hybrid LSTM 错误样本情感分布"
                        data={hybridSentimentDistributionMetrics}
                        yAxisName="Count"
                        valueFormatter={(v) => Number(v).toFixed(0)}
                      />
                    ) : (
                      <Empty text="暂无情感分布统计" />
                    )}
                  </div>

                  <div className="card">
                    <div className="card-title">情感分析结论</div>
                    <div className="compare-note-box">
                      {simHybridPredictions.error_analysis.sentiment_analysis.summary_text || '暂无情感分析结论。'}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">Hybrid LSTM 测试集明细统计</div>
          {!simHybridPredictions ? (
            <Empty text="暂无 Hybrid LSTM 测试集明细统计" />
          ) : (
            <div className="grid grid-4 section-gap-small">
              <KpiCard title="测试样本总数" value={simHybridPredictions.total_rows} extra="完整测试集规模" />
              <KpiCard title="预测正确数" value={simHybridPredictions.correct_rows} extra="y_true == y_pred" />
              <KpiCard title="预测错误数" value={simHybridPredictions.wrong_rows} extra="y_true != y_pred" />
              <KpiCard
                title="当前筛选"
                value={showOnlyHybridError ? '仅误判样本' : '全部样本'}
                extra="可切换查看"
              />
            </div>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="table-toolbar">
          <div className="card-title">Hybrid LSTM 测试集明细表</div>
          <button
            className={`mode-btn ${showOnlyHybridError ? 'active' : ''}`}
            onClick={() => setShowOnlyHybridError((prev) => !prev)}
          >
            {showOnlyHybridError ? '查看全部样本' : '仅看误判样本'}
          </button>
        </div>

        <div className="card table-card">
          {hybridPredictionTableLoading ? (
            <Loading text="Hybrid LSTM 测试集明细加载中..." />
          ) : !simHybridPredictions?.rows?.length ? (
            <Empty text="暂无 Hybrid LSTM 测试集明细" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>事件日期</th>
                  <th>事件类型</th>
                  <th>事件标题</th>
                  <th>真实标签</th>
                  <th>预测标签</th>
                  <th>预测概率(上涨)</th>
                  <th>是否正确</th>
                </tr>
              </thead>
              <tbody>
                {simHybridPredictions.rows.map((row, index) => {
                  const isActive =
                    selectedHybridRow &&
                    row.event_date === selectedHybridRow.event_date &&
                    row.event_title === selectedHybridRow.event_title &&
                    row.t_minus_1_date === selectedHybridRow.t_minus_1_date

                  return (
                    <tr
                      key={`${row.event_date}-${index}`}
                      className={`clickable-row ${isActive ? 'active-row' : ''}`}
                      onClick={() => {
                        setSelectedHybridRow(row)
                        setHybridDrawerOpen(true)
                      }}
                    >
                      <td>{row.event_date}</td>
                      <td>{getEventTypeLabel(row.event_type_raw)}</td>
                      <td>{row.event_title}</td>
                      <td>{row.actual_label || '--'}</td>
                      <td>{row.pred_label || '--'}</td>
                      <td>{row.pred_prob_up != null ? formatProb(row.pred_prob_up) : '--'}</td>
                      <td>
                        <span className={`truth-badge ${row.is_correct ? 'correct' : 'wrong'}`}>
                          {row.is_correct ? '正确' : '错误'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">Hybrid LSTM 详细分析结论</div>
          <div className="compare-note-box">
            当前页面已将 Hybrid LSTM 接入详细误差分析链路，包括误判分布、标题关键词分析、情感分析、
            测试集明细与单样本抽屉诊断。这样可以与 Simulation V1 形成完整对照，进一步观察升级模型的误判模式与偏置来源。
          </div>
        </div>
      </div>

      <SimulationSampleDrawer
        open={drawerOpen}
        row={selectedSimulationRow}
        onClose={() => setDrawerOpen(false)}
        drawerTitle="Simulation V1 样本详情"
        drawerSubtitle="查看基线模型在当前测试样本上的上下文、标题特征与诊断结果"
        modelFlavor="simulation_v1"
      />

      <SimulationSampleDrawer
        open={hybridDrawerOpen}
        row={selectedHybridRow}
        onClose={() => setHybridDrawerOpen(false)}
        drawerTitle="Hybrid LSTM 样本详情"
        drawerSubtitle="查看融合模型在当前测试样本上的时序上下文、静态特征与诊断结果"
        modelFlavor="hybrid_lstm"
      />
    </div>
  )
}