import Empty from './Empty'
import KpiCard from './KpiCard'
import SimulationContextMiniChart from './SimulationContextMiniChart'
import { formatMetric, getEventTypeLabel } from '../utils/display'

function formatProb(value) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${(num * 100).toFixed(2)}%`
}

function formatSigned(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${num >= 0 ? '+' : ''}${num.toFixed(digits)}`
}

function getSentimentLabel(score) {
  const num = Number(score)
  if (Number.isNaN(num)) return '未知'
  if (num > 0) return '利好倾向'
  if (num < 0) return '利空倾向'
  return '中性倾向'
}

function buildRuleTags(row) {
  if (!row) return []

  const tags = []

  const sentimentScore = Number(row.sentiment_score)
  const posCount = Number(row.positive_keyword_count || 0)
  const negCount = Number(row.negative_keyword_count || 0)
  const probUp = Number(row.pred_prob_up)
  const t1VolumeChange = Number(row.t_minus_1_volume_change_vs_prev)
  const t1Amplitude = Number(row.t_minus_1_amplitude)
  const t1CloseChange = Number(row.t_minus_1_close_change_vs_prev_close)

  if (!Number.isNaN(sentimentScore)) {
    if (sentimentScore > 0) {
      tags.push({ text: '标题偏利好', type: 'positive' })
    } else if (sentimentScore < 0) {
      tags.push({ text: '标题偏利空', type: 'negative' })
    } else {
      tags.push({ text: '标题中性', type: 'neutral' })
    }
  }

  if (posCount + negCount === 0) {
    tags.push({ text: '未命中关键词', type: 'neutral' })
  } else if (posCount + negCount >= 2) {
    tags.push({ text: '关键词较丰富', type: 'info' })
  } else {
    tags.push({ text: '关键词较少', type: 'info' })
  }

  if (!Number.isNaN(t1VolumeChange)) {
    if (t1VolumeChange >= 0.2) {
      tags.push({ text: '前一日放量', type: 'info' })
    } else if (t1VolumeChange <= -0.2) {
      tags.push({ text: '前一日缩量', type: 'neutral' })
    }
  }

  if (!Number.isNaN(t1Amplitude)) {
    if (t1Amplitude >= 0.03) {
      tags.push({ text: '前一日波动较大', type: 'warning' })
    } else if (t1Amplitude <= 0.01) {
      tags.push({ text: '前一日波动平稳', type: 'neutral' })
    }
  }

  if (!Number.isNaN(t1CloseChange)) {
    if (t1CloseChange > 0) {
      tags.push({ text: '前一日收涨', type: 'positive' })
    } else if (t1CloseChange < 0) {
      tags.push({ text: '前一日收跌', type: 'negative' })
    }
  }

  if (!Number.isNaN(probUp)) {
    if (probUp >= 0.5) {
      tags.push({ text: '模型偏向上涨', type: 'positive' })
    } else {
      tags.push({ text: '模型偏向下跌', type: 'negative' })
    }

    const confidence = Math.max(probUp, 1 - probUp)
    if (row.is_correct === false) {
      if (confidence >= 0.7) {
        tags.push({ text: '高置信度误判', type: 'danger' })
      } else if (confidence < 0.6) {
        tags.push({ text: '低置信度误判', type: 'warning' })
      }
    } else if (confidence >= 0.7) {
      tags.push({ text: '高置信度正确', type: 'positive' })
    }
  }

  return tags
}

function buildDiagnosticSummary(row, modelFlavor = 'simulation_v1') {
  if (!row) return '暂无样本诊断结论。'

  const sentimentScore = Number(row.sentiment_score)
  const posCount = Number(row.positive_keyword_count || 0)
  const negCount = Number(row.negative_keyword_count || 0)
  const probUp = Number(row.pred_prob_up)
  const t1VolumeChange = Number(row.t_minus_1_volume_change_vs_prev)
  const t1Amplitude = Number(row.t_minus_1_amplitude)
  const t1CloseChange = Number(row.t_minus_1_close_change_vs_prev_close)

  const titlePart = Number.isNaN(sentimentScore)
    ? '标题情绪未知'
    : sentimentScore > 0
      ? '标题整体偏利好'
      : sentimentScore < 0
        ? '标题整体偏利空'
        : '标题整体偏中性'

  const keywordPart =
    posCount + negCount === 0
      ? '，且未命中预设关键词'
      : `，命中 ${posCount + negCount} 个关键词`

  const marketParts = []
  if (!Number.isNaN(t1VolumeChange)) {
    if (t1VolumeChange >= 0.2) marketParts.push('前一日放量')
    else if (t1VolumeChange <= -0.2) marketParts.push('前一日缩量')
  }
  if (!Number.isNaN(t1Amplitude)) {
    if (t1Amplitude >= 0.03) marketParts.push('波动较大')
    else if (t1Amplitude <= 0.01) marketParts.push('波动平稳')
  }
  if (!Number.isNaN(t1CloseChange)) {
    if (t1CloseChange > 0) marketParts.push('收盘走强')
    else if (t1CloseChange < 0) marketParts.push('收盘走弱')
  }

  const marketPart =
    marketParts.length > 0 ? `；市场上下文表现为${marketParts.join('、')}` : ''

  let modelPart = ''
  if (!Number.isNaN(probUp)) {
    const direction = probUp >= 0.5 ? '上涨' : '下跌'
    const confidence = Math.max(probUp, 1 - probUp)

    if (modelFlavor === 'hybrid_lstm') {
      if (row.is_correct === false) {
        if (confidence >= 0.7) {
          modelPart = `；Hybrid LSTM 在融合时序特征与标题静态特征后仍以较高置信度偏向${direction}，说明该样本存在较强的融合误判倾向`
        } else if (confidence < 0.6) {
          modelPart = `；Hybrid LSTM 虽偏向${direction}，但置信度较低，说明该样本在时序与静态特征融合后仍存在较强不确定性`
        } else {
          modelPart = `；Hybrid LSTM 融合判断偏向${direction}，但最终未能正确识别该样本`
        }
      } else if (confidence >= 0.7) {
        modelPart = `；Hybrid LSTM 在融合时序与静态特征后以较高置信度正确判断为${direction}`
      } else {
        modelPart = `；Hybrid LSTM 融合判断后最终正确识别为${direction}`
      }
    } else {
      if (row.is_correct === false) {
        if (confidence >= 0.7) {
          modelPart = `；Simulation V1 以较高置信度偏向${direction}，但最终发生误判`
        } else if (confidence < 0.6) {
          modelPart = `；Simulation V1 虽偏向${direction}，但置信度较低，说明该样本本身存在较强不确定性`
        } else {
          modelPart = `；Simulation V1 偏向${direction}，但未能正确识别该样本`
        }
      } else if (confidence >= 0.7) {
        modelPart = `；Simulation V1 以较高置信度正确判断为${direction}`
      } else {
        modelPart = `；Simulation V1 最终正确判断为${direction}`
      }
    }
  }

  return `${titlePart}${keywordPart}${marketPart}${modelPart}。`
}

function buildExplanationText(row, modelFlavor = 'simulation_v1') {
  if (!row) return '暂无样本解释。'

  const modelName =
    modelFlavor === 'hybrid_lstm'
      ? 'Hybrid LSTM'
      : 'Simulation V1'

  const modelExtra =
    modelFlavor === 'hybrid_lstm'
      ? '该模型会同时综合事件前两日市场时序特征与标题静态特征，因此更强调“上下文 + 事件特征”的联合影响。'
      : '该模型主要基于事件前规则特征与表格化上下文进行判断，更适合作为新事件模拟预测的基线模型。'

  return `当前样本标题中，利好关键词 ${row.positive_keyword_count ?? 0} 个，利空关键词 ${row.negative_keyword_count ?? 0} 个，综合情感得分为 ${formatSigned(row.sentiment_score)}。${modelName} 给出的上涨概率为 ${row.pred_prob_up != null ? formatProb(row.pred_prob_up) : '--'}。${modelExtra}`
}

export default function SimulationSampleDrawer({
  open,
  row,
  onClose,
  drawerTitle = 'Simulation V1 样本详情',
  drawerSubtitle = '查看当前测试样本的上下文、标题特征与小图表',
  modelFlavor = 'simulation_v1'
}) {
  if (!open) return null

  const ruleTags = buildRuleTags(row)
  const diagnosticSummary = buildDiagnosticSummary(row, modelFlavor)
  const explanationText = buildExplanationText(row, modelFlavor)

  return (
    <>
      <div className="drawer-mask" onClick={onClose} />
      <div className={`drawer-panel ${open ? 'open' : ''}`}>
        <div className="drawer-header">
          <div>
            <div className="drawer-title">{drawerTitle}</div>
            <div className="drawer-subtitle">{drawerSubtitle}</div>
          </div>
          <button className="drawer-close-btn" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="drawer-body">
          {!row ? (
            <Empty text="暂无可展示的样本详情" />
          ) : (
            <>
              <div className="drawer-diagnosis-box">
                <div className="drawer-diagnosis-title">自动诊断结论</div>
                <div className="drawer-diagnosis-text">{diagnosticSummary}</div>
              </div>

              <div className="grid grid-2 section-gap-small">
                <KpiCard
                  title="事件日期"
                  value={row.event_date || '--'}
                  extra="当前选中样本"
                />
                <KpiCard
                  title="事件类型"
                  value={getEventTypeLabel(row.event_type_raw)}
                  extra="事件分类"
                />
              </div>

              <div className="grid grid-2 section-gap-small">
                <KpiCard
                  title="真实标签"
                  value={row.actual_label || '--'}
                  extra="测试集真实结果"
                />
                <KpiCard
                  title="预测标签"
                  value={row.pred_label || '--'}
                  extra={row.is_correct ? '预测正确' : '预测错误'}
                />
              </div>

              <div className="grid grid-2 section-gap-small">
                <KpiCard
                  title="预测上涨概率"
                  value={row.pred_prob_up != null ? formatProb(row.pred_prob_up) : '--'}
                  extra="pred_prob_up"
                />
                <KpiCard
                  title="标题情感"
                  value={getSentimentLabel(row.sentiment_score)}
                  extra={`score=${formatSigned(row.sentiment_score)}`}
                />
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">规则解释标签</div>
                {!ruleTags.length ? (
                  <Empty text="暂无规则解释标签" />
                ) : (
                  <div className="rule-tag-list">
                    {ruleTags.map((tag, index) => (
                      <span
                        key={`${tag.text}-${index}`}
                        className={`rule-tag ${tag.type || 'neutral'}`}
                      >
                        {tag.text}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">事件标题</div>
                <div className="selected-event-info">
                  <div>{row.event_title || '--'}</div>
                </div>
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">事件前市场上下文图表</div>
                <SimulationContextMiniChart row={row} />
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">事件前市场上下文表</div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>日期</th>
                      <th>日内收益率</th>
                      <th>振幅</th>
                      <th>成交量变化</th>
                      <th>收盘价变化</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{row.t_minus_2_date || '--'}</td>
                      <td>{formatMetric(row.t_minus_2_intraday_return, 6)}</td>
                      <td>{formatMetric(row.t_minus_2_amplitude, 6)}</td>
                      <td>{formatMetric(row.t_minus_2_volume_change_vs_prev, 6)}</td>
                      <td>{formatMetric(row.t_minus_2_close_change_vs_prev_close, 6)}</td>
                    </tr>
                    <tr>
                      <td>{row.t_minus_1_date || '--'}</td>
                      <td>{formatMetric(row.t_minus_1_intraday_return, 6)}</td>
                      <td>{formatMetric(row.t_minus_1_amplitude, 6)}</td>
                      <td>{formatMetric(row.t_minus_1_volume_change_vs_prev, 6)}</td>
                      <td>{formatMetric(row.t_minus_1_close_change_vs_prev_close, 6)}</td>
                    </tr>
                  </tbody>
                </table>

                <div className="section-gap detail-note-box">
                  该上下文严格使用事件发生前两个交易日数据，不包含事件日及未来信息。
                </div>
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">标题特征</div>
                <div className="selected-event-info">
                  <div><strong>标题长度：</strong>{row.title_length ?? '--'}</div>
                  <div><strong>利好关键词数：</strong>{row.positive_keyword_count ?? '--'}</div>
                  <div><strong>利空关键词数：</strong>{row.negative_keyword_count ?? '--'}</div>
                  <div><strong>情感得分：</strong>{formatSigned(row.sentiment_score)}</div>
                  <div><strong>情感标签：</strong>{getSentimentLabel(row.sentiment_score)}</div>
                </div>
              </div>

              <div className="drawer-card section-gap">
                <div className="card-title">样本解释</div>
                <div className="compare-note-box">
                  {explanationText}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  )
}