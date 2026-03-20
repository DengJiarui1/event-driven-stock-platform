import { useEffect, useMemo, useState } from 'react'
import { getEvents, getLatestInformerResult, getLatestPrediction, getModelComparison } from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import KpiCard from '../components/KpiCard'
import ModelBarChart from '../components/ModelBarChart'
import EventTable from '../components/EventTable'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import { formatMetric, formatPercent } from '../utils/display'

export default function Dashboard() {
  const [events, setEvents] = useState([])
  const [models, setModels] = useState([])
  const [latestPrediction, setLatestPrediction] = useState(null)
  const [latestInformer, setLatestInformer] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [eventsRes, modelsRes, predictionRes, informerRes] = await Promise.allSettled([
          getEvents(),
          getModelComparison(),
          getLatestPrediction(),
          getLatestInformerResult()
        ])

        if (eventsRes.status === 'fulfilled') setEvents(eventsRes.value.data || [])
        if (modelsRes.status === 'fulfilled') setModels(modelsRes.value.data || [])
        if (predictionRes.status === 'fulfilled') setLatestPrediction(predictionRes.value.data || null)
        if (informerRes.status === 'fulfilled') setLatestInformer(informerRes.value.data || null)
      } catch (error) {
        console.error(error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const bestModel = useMemo(() => {
    if (!models.length) return null
    return [...models].sort((a, b) => Number(b.accuracy) - Number(a.accuracy))[0]
  }, [models])

  if (loading) {
    return <Loading text="首页数据加载中..." />
  }

  return (
    <div>
      <PageHeader
        title="首页仪表盘"
        description="展示平台概览、最新预测、关键指标和当前实验结论。"
      />

      <div className="grid grid-4">
        <KpiCard title="股票名称" value="贵州茅台（600519）" extra="当前实验对象" />
        <KpiCard title="事件总数" value={events.length || 0} extra="基于新闻/公告构建" />
        <KpiCard
          title="最佳分类模型"
          value={bestModel ? bestModel.model : '--'}
          extra={bestModel ? `Accuracy: ${formatPercent(bestModel.accuracy)}` : '暂无数据'}
        />
        <KpiCard
          title="最新预测"
          value={latestPrediction?.prediction_label ?? '--'}
          extra={
            latestPrediction?.probability != null
              ? `概率：${formatPercent(latestPrediction.probability)}`
              : '后端未提供时显示为空'
          }
        />
      </div>

      <div className="grid grid-2 section-gap">
        <div className="card">
          <div className="card-title">系统结论说明</div>
          <div className="compare-note-box">
            当前样本条件下，分类任务中表现最稳定的是
            <strong> {bestModel?.model || '暂无结果'} </strong>，
            指标为 <strong>{bestModel ? formatPercent(bestModel.accuracy) : '--'}</strong>。
            说明在事件样本规模较小的场景下，简单模型仍具备较强稳定性。
          </div>
        </div>

        <div className="card">
          <div className="card-title">Informer 扩展实验结论</div>
          <div className="compare-note-box">
            Informer 当前作为单变量收益率预测扩展实验使用。
            {latestInformer?.result?.metrics ? (
              <>
                最新结果中 RMSE 为 <strong>{formatMetric(latestInformer.result.metrics.rmse)}</strong>，
                模型整体预测更平滑，说明仅依赖历史收益率较难刻画短期突发波动，因此引入事件信息是必要的。
              </>
            ) : (
              <>当前尚无 Informer 成功实验结果。</>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-2 section-gap">
        <div className="card">
          <div className="card-title">模型准确率概览</div>
          {models.length ? <ModelBarChart data={models} /> : <Empty text="暂无模型对比数据" />}
        </div>

        <div className="card">
          <div className="card-title">最近一次预测</div>
          {latestPrediction ? (
            <div className="prediction-card">
              <div><strong>事件日期：</strong>{latestPrediction.event_date}</div>
              <div><strong>事件标题：</strong>{latestPrediction.event_title}</div>
              <div><strong>预测结果：</strong>{latestPrediction.prediction_label}</div>
              <div><strong>预测概率：</strong>{formatPercent(latestPrediction.probability)}</div>
              <div><strong>真实回看：</strong>{latestPrediction.actual_label || '暂无'}</div>
            </div>
          ) : (
            <Empty text="后端暂未提供 latest-prediction 接口数据" />
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card-title">最新事件列表</div>
        {events.length ? (
          <EventTable data={events.slice(0, 8)} />
        ) : (
          <Empty text="暂无事件数据" />
        )}
      </div>
    </div>
  )
}