import { useEffect, useMemo, useState } from 'react'
import {
  createInformerJob,
  getInformerDefaultInput,
  getInformerJob,
  getLatestInformerResult
} from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import KpiCard from '../components/KpiCard'

function formatMetric(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(6)
}

export default function InformerPage() {
  const [form, setForm] = useState({
    data_path: '',
    target: 'return',
    seq_len: 60,
    label_len: 30,
    pred_len: 5,
    train_epochs: 6,
    batch_size: 16,
    patience: 3,
    learning_rate: 0.0001,
    d_model: 128,
    n_heads: 4,
    e_layers: 2,
    d_layers: 1,
    d_ff: 256,
    dropout: 0.05,
    run_name: 'maotai_return_only'
  })

  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [jobId, setJobId] = useState('')
  const [jobDetail, setJobDetail] = useState(null)
  const [latestJob, setLatestJob] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    async function initPage() {
      try {
        const [inputRes, latestRes] = await Promise.all([
          getInformerDefaultInput(),
          getLatestInformerResult()
        ])

        const defaultInput = inputRes.data?.data_path || ''
        setForm((prev) => ({
          ...prev,
          data_path: defaultInput
        }))

        if (latestRes.data) {
          setLatestJob(latestRes.data)
        }
      } catch (err) {
        console.error(err)
        setError('Informer 页面初始化失败，请检查后端接口。')
      } finally {
        setLoading(false)
      }
    }

    initPage()
  }, [])

  useEffect(() => {
    if (!jobId) return

    let timer = null

    async function pollJob() {
      try {
        const res = await getInformerJob(jobId)
        const detail = res.data
        setJobDetail(detail)

        if (detail.status === 'success') {
          setLatestJob(detail)
          return
        }

        if (detail.status === 'failed') {
          return
        }

        timer = setTimeout(pollJob, 3000)
      } catch (err) {
        console.error(err)
        timer = setTimeout(pollJob, 3000)
      }
    }

    pollJob()

    return () => {
      if (timer) clearTimeout(timer)
    }
  }, [jobId])

  function handleChange(e) {
    const { name, value } = e.target
    setForm((prev) => ({
      ...prev,
      [name]: value
    }))
  }

  async function handleRun() {
    setSubmitting(true)
    setError('')
    setJobDetail(null)

    try {
      const res = await createInformerJob({
        ...form,
        seq_len: Number(form.seq_len),
        label_len: Number(form.label_len),
        pred_len: Number(form.pred_len),
        train_epochs: Number(form.train_epochs),
        batch_size: Number(form.batch_size),
        patience: Number(form.patience),
        learning_rate: Number(form.learning_rate),
        d_model: Number(form.d_model),
        n_heads: Number(form.n_heads),
        e_layers: Number(form.e_layers),
        d_layers: Number(form.d_layers),
        d_ff: Number(form.d_ff),
        dropout: Number(form.dropout),
      })

      setJobId(res.data.job_id)
      setJobDetail(res.data)
    } catch (err) {
      console.error(err)
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          'Informer 实验启动失败。'
      )
    } finally {
      setSubmitting(false)
    }
  }

  const displayJob = useMemo(() => {
    return jobDetail?.status === 'success' ? jobDetail : latestJob
  }, [jobDetail, latestJob])

  if (loading) {
    return <Loading text="Informer 页面加载中..." />
  }

  const metrics = displayJob?.result?.metrics || {}
  const chartUrl = displayJob?.result?.chart_url || ''
  const logUrl = jobDetail?.log_url || displayJob?.log_url || ''

  return (
    <div>
      <PageHeader
        title="Informer 实验中心"
        description="该模块用于工程化运行 Informer 收益率预测实验，并展示预测结果与误差指标。"
      />

      <div className="grid grid-2 section-gap">
        <div className="card">
          <div className="card-title">实验参数配置</div>

          {error ? <div className="status-box empty">{error}</div> : null}

          <div className="form-grid">
            <div className="form-item">
              <label>输入数据文件</label>
              <input
                name="data_path"
                value={form.data_path}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>预测目标</label>
              <input
                name="target"
                value={form.target}
                onChange={handleChange}
              />
            </div>

            <div className="form-item">
              <label>seq_len</label>
              <input name="seq_len" value={form.seq_len} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>label_len</label>
              <input name="label_len" value={form.label_len} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>pred_len</label>
              <input name="pred_len" value={form.pred_len} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>train_epochs</label>
              <input name="train_epochs" value={form.train_epochs} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>batch_size</label>
              <input name="batch_size" value={form.batch_size} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>patience</label>
              <input name="patience" value={form.patience} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>learning_rate</label>
              <input name="learning_rate" value={form.learning_rate} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>d_model</label>
              <input name="d_model" value={form.d_model} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>n_heads</label>
              <input name="n_heads" value={form.n_heads} onChange={handleChange} />
            </div>

            <div className="form-item">
              <label>d_ff</label>
              <input name="d_ff" value={form.d_ff} onChange={handleChange} />
            </div>
          </div>

          <div className="predict-note-box section-gap">
            当前工程化接入的是 “单变量收益率 Informer 实验”。它主要用于扩展实验与对比分析，不替代你现在的事件驱动 LSTM 主流程。
          </div>

          <div className="section-gap">
            <button
              className="primary-btn"
              onClick={handleRun}
              disabled={submitting}
            >
              {submitting ? '已提交任务...' : '运行 Informer 实验'}
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">任务状态</div>

          {!jobDetail ? (
            <Empty text="尚未提交新任务" />
          ) : (
            <div className="selected-event-info">
              <div><strong>任务ID：</strong>{jobDetail.job_id}</div>
              <div><strong>状态：</strong>{jobDetail.status}</div>
              <div><strong>创建时间：</strong>{jobDetail.created_at}</div>
              <div><strong>开始时间：</strong>{jobDetail.started_at || '--'}</div>
              <div><strong>结束时间：</strong>{jobDetail.finished_at || '--'}</div>
              {jobDetail.error ? <div><strong>错误信息：</strong>{jobDetail.error}</div> : null}
              {logUrl ? (
                <div>
                  <strong>运行日志：</strong>
                  <a href={logUrl} target="_blank" rel="noreferrer">查看日志</a>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card-title">最新可用结果</div>
      </div>

      {!displayJob?.result ? (
        <Empty text="暂无 Informer 成功结果" />
      ) : (
        <>
          <div className="grid grid-4 section-gap">
            <KpiCard title="MAE" value={formatMetric(metrics.mae)} extra="平均绝对误差" />
            <KpiCard title="MSE" value={formatMetric(metrics.mse)} extra="均方误差" />
            <KpiCard title="RMSE" value={formatMetric(metrics.rmse)} extra="均方根误差" />
            <KpiCard title="MAPE" value={formatMetric(metrics.mape)} extra="平均绝对百分比误差" />
          </div>

          <div className="grid grid-2 section-gap">
            <div className="card">
              <div className="card-title">实验信息</div>
              <div className="selected-event-info">
                <div><strong>任务ID：</strong>{displayJob.result.job_id}</div>
                <div><strong>输入文件：</strong>{displayJob.result.input_csv}</div>
                <div><strong>结果目录：</strong>{displayJob.result.result_dir}</div>
                <div><strong>pred_shape：</strong>{JSON.stringify(displayJob.result.pred_shape)}</div>
                <div><strong>true_shape：</strong>{JSON.stringify(displayJob.result.true_shape)}</div>
              </div>
            </div>

            <div className="card">
              <div className="card-title">实验参数</div>
              <div className="selected-event-info">
                <div><strong>target：</strong>{displayJob.result.config?.target}</div>
                <div><strong>seq_len：</strong>{displayJob.result.config?.seq_len}</div>
                <div><strong>label_len：</strong>{displayJob.result.config?.label_len}</div>
                <div><strong>pred_len：</strong>{displayJob.result.config?.pred_len}</div>
                <div><strong>train_epochs：</strong>{displayJob.result.config?.train_epochs}</div>
                <div><strong>d_model：</strong>{displayJob.result.config?.d_model}</div>
              </div>
            </div>
          </div>

          <div className="section-gap">
            <div className="card">
              <div className="card-title">预测对比图</div>
              {chartUrl ? (
                <img
                  src={chartUrl}
                  alt="Informer prediction compare"
                  className="informer-result-image"
                />
              ) : (
                <Empty text="暂无预测图" />
              )}
            </div>
          </div>

          <div className="section-gap">
            <div className="card table-card">
              <div className="card-title">结果预览</div>
              {!displayJob.result.preview ? (
                <Empty text="暂无结果预览" />
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>序号</th>
                      <th>Pred</th>
                      <th>True</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(displayJob.result.preview.pred_head || []).map((v, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>{Number(v).toFixed(6)}</td>
                        <td>{Number(displayJob.result.preview.true_head?.[i] ?? 0).toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}