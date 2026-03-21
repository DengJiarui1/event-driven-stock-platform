import http from './http'

export const getModelComparison = () => http.get('/api/model-comparison')
export const getEvents = () => http.get('/api/events')
export const getLatestPrediction = () => http.get('/api/latest-prediction')

// 事件窗口相关
export const getEventWindowDates = () => http.get('/api/event-window-dates')
export const getEventWindows = (eventDate) =>
  http.get('/api/event-windows', {
    params: {
      event_date: eventDate
    }
  })

// 历史事件真实预测
export const getPredictOptions = () => http.get('/api/predict-options')
export const predictByEventDate = (eventDate) =>
  http.get('/api/predict-by-date', {
    params: {
      event_date: eventDate
    }
  })
export const predictByEvent = (payload) => http.post('/api/predict', payload)

// 新事件模拟预测
export const getSimulateContext = (eventDate) =>
  http.get('/api/simulate/context', {
    params: {
      event_date: eventDate
    }
  })

export const predictSimulate = (payload) => http.post('/api/predict-simulate', payload)

// Informer 实验相关
export const getInformerDefaultInput = () => http.get('/api/informer/default-input')
export const createInformerJob = (payload) => http.post('/api/informer/jobs', payload)
export const getInformerJobs = () => http.get('/api/informer/jobs')
export const getInformerJob = (jobId) => http.get(`/api/informer/jobs/${jobId}`)
export const getLatestInformerResult = () => http.get('/api/informer/latest')


export const getSimulationModelReport = () => http.get('/api/simulation-model-report')
export const getSimulationTestPredictions = (params = {}) =>
  http.get('/api/simulation-test-predictions', { params })
export const getSimHybridReport = () => http.get('/api/sim-hybrid-report')
export const getSimHybridTestPredictions = (params = {}) =>
  http.get('/api/sim-hybrid-test-predictions', { params })