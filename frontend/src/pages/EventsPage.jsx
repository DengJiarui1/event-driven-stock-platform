import { useEffect, useMemo, useState } from 'react'
import {
  getEventWindowDates,
  getEventWindows,
  getEvents
} from '../api/stockApi'
import PageHeader from '../components/PageHeader'
import Loading from '../components/Loading'
import Empty from '../components/Empty'
import EventWindowChart from '../components/EventWindowChart'
import KpiCard from '../components/KpiCard'
import { formatInteger, formatNumber, formatPercentPlain, getEventTypeLabel } from '../utils/display'

// function formatNumber(value, digits = 2) {
//   const num = Number(value)
//   if (Number.isNaN(num)) return '--'
//   return num.toFixed(digits)
// }

// function formatInteger(value) {
//   const num = Number(value)
//   if (Number.isNaN(num)) return '--'
//   return num.toLocaleString('zh-CN')
// }

function formatPercent(value, digits = 2) {
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${num.toFixed(digits)}%`
}

export default function EventsPage() {
  const [eventDates, setEventDates] = useState([])
  const [selectedDate, setSelectedDate] = useState('')
  const [windowData, setWindowData] = useState([])
  const [events, setEvents] = useState([])
  const [pageLoading, setPageLoading] = useState(true)
  const [windowLoading, setWindowLoading] = useState(false)

  useEffect(() => {
    async function initPage() {
      try {
        const [datesRes, eventsRes] = await Promise.all([
          getEventWindowDates(),
          getEvents()
        ])

        const dates = datesRes.data || []
        const eventList = eventsRes.data || []

        setEventDates(dates)
        setEvents(eventList)

        if (dates.length > 0) {
          setSelectedDate(dates[0])
        }
      } catch (error) {
        console.error('初始化事件分析页失败：', error)
      } finally {
        setPageLoading(false)
      }
    }

    initPage()
  }, [])

  useEffect(() => {
    async function fetchWindowData() {
      if (!selectedDate) {
        setWindowData([])
        return
      }

      setWindowLoading(true)
      try {
        const res = await getEventWindows(selectedDate)
        setWindowData(res.data || [])
      } catch (error) {
        console.error('获取事件窗口数据失败：', error)
        setWindowData([])
      } finally {
        setWindowLoading(false)
      }
    }

    fetchWindowData()
  }, [selectedDate])

  const selectedEventInfo = useMemo(() => {
    if (!selectedDate || !events.length) return null
    return events.find((item) => item.event_date === selectedDate) || null
  }, [selectedDate, events])

  const metrics = useMemo(() => {
    if (!windowData.length) {
      return {
        windowReturn: null,
        eventDayReturn: null,
        avgVolume: null,
        amplitude: null,
        eventDayClose: null,
        eventDayRow: null
      }
    }

    const firstRow = windowData[0]
    const lastRow = windowData[windowData.length - 1]
    const eventDayRow =
      windowData.find((item) => item.date === selectedDate) || windowData[Math.floor(windowData.length / 2)]

    const eventIndex = windowData.findIndex((item) => item.date === selectedDate)
    const prevRow = eventIndex > 0 ? windowData[eventIndex - 1] : null

    const firstClose = Number(firstRow.close)
    const lastClose = Number(lastRow.close)
    const eventClose = Number(eventDayRow.close)
    const prevClose = prevRow ? Number(prevRow.close) : Number(eventDayRow.open)

    const windowReturn =
      firstClose ? ((lastClose - firstClose) / firstClose) * 100 : null

    const eventDayReturn =
      prevClose ? ((eventClose - prevClose) / prevClose) * 100 : null

    const volumes = windowData.map((item) => Number(item.volume)).filter((v) => !Number.isNaN(v))
    const avgVolume = volumes.length
      ? volumes.reduce((sum, cur) => sum + cur, 0) / volumes.length
      : null

    const highs = windowData.map((item) => Number(item.high)).filter((v) => !Number.isNaN(v))
    const lows = windowData.map((item) => Number(item.low)).filter((v) => !Number.isNaN(v))
    const maxHigh = highs.length ? Math.max(...highs) : null
    const minLow = lows.length ? Math.min(...lows) : null

    const amplitude =
      maxHigh && minLow ? ((maxHigh - minLow) / minLow) * 100 : null

    return {
      windowReturn,
      eventDayReturn,
      avgVolume,
      amplitude,
      eventDayClose: eventClose,
      eventDayRow
    }
  }, [windowData, selectedDate])

  if (pageLoading) {
    return <Loading text="事件分析页加载中..." />
  }

  return (
    <div>
      <PageHeader
        title="事件分析"
        description="选择某个事件日期后，查看该事件窗口内的股价走势、成交量变化和关键统计指标。"
      />

      <div className="card section-gap">
        <div className="card-title">事件选择</div>

        <div className="event-filter-box">
          <div className="event-filter-item">
            <label>事件日期</label>
            <select
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            >
              {eventDates.map((date) => (
                <option key={date} value={date}>
                  {date}
                </option>
              ))}
            </select>
          </div>
        </div>

        {selectedEventInfo ? (
          <div className="selected-event-info">
            <div><strong>事件日期：</strong>{selectedEventInfo.event_date}</div>
            <div><strong>事件类型：</strong>{getEventTypeLabel(selectedEventInfo.event_type)}</div>
            <div><strong>事件来源：</strong>{selectedEventInfo.event_source}</div>
            <div><strong>事件标题：</strong>{selectedEventInfo.event_title}</div>
            <div><strong>事件数量：</strong>{selectedEventInfo.event_count}</div>
          </div>
        ) : (
          <Empty text="未找到对应事件信息" />
        )}
      </div>

      <div className="grid grid-5 section-gap">
        <KpiCard
          title="事件日收盘价"
          value={formatNumber(metrics.eventDayClose)}
          extra="选中事件日对应收盘价"
        />
        <KpiCard
          title="事件日涨跌幅"
          value={formatPercent(metrics.eventDayReturn)}
          extra="相对前一交易日变化"
        />
        <KpiCard
          title="窗口收益率"
          value={formatPercent(metrics.windowReturn)}
          extra="窗口首日到末日收盘价变化"
        />
        <KpiCard
          title="平均成交量"
          // value={metrics.avgVolume != null ? formatNumber(metrics.avgVolume, 0) : '--'}
          value={metrics.avgVolume != null ? formatInteger(metrics.avgVolume) : '--'}
          extra="事件窗口内日均成交量"
        />
        <KpiCard
          title="窗口振幅"
          value={formatPercent(metrics.amplitude)}
          extra="窗口最高价与最低价波动"
        />
      </div>

      <div className="section-gap">
        <div className="card">
          <div className="card-title">事件窗口走势图</div>
          {windowLoading ? (
            <Loading text="事件窗口图加载中..." />
          ) : (
            <EventWindowChart data={windowData} eventDate={selectedDate} />
          )}
        </div>
      </div>

      <div className="section-gap">
        <div className="card table-card">
          <div className="card-title">事件窗口明细</div>

          {!windowData.length ? (
            <Empty text="暂无事件窗口明细数据" />
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>日期</th>
                  <th>开盘价</th>
                  <th>收盘价</th>
                  <th>最高价</th>
                  <th>最低价</th>
                  <th>成交量</th>
                  <th>事件日期</th>
                  <th>事件类型</th>
                </tr>
              </thead>
              <tbody>
                {windowData.map((item, index) => {
                  const isEventDay = item.date === selectedDate
                  return (
                    <tr
                      key={`${item.date}-${index}`}
                      className={isEventDay ? 'event-day-row' : ''}
                    >
                      <td>
                        {item.date}
                        {isEventDay ? <span className="event-day-badge">事件日</span> : null}
                      </td>
                      <td>{formatNumber(item.open)}</td>
                      <td>{formatNumber(item.close)}</td>
                      <td>{formatNumber(item.high)}</td>
                      <td>{formatNumber(item.low)}</td>
                      <td>{formatInteger(item.volume)}</td>
                      <td>{item.event_date}</td>
                      <td>{getEventTypeLabel(item.event_type)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}