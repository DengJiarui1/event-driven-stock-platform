import { Navigate, Route, Routes } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import Dashboard from './pages/Dashboard'
import EventsPage from './pages/EventsPage'
import PredictionPage from './pages/PredictionPage'
import ModelsPage from './pages/ModelsPage'
import InformerPage from './pages/InformerPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="events" element={<EventsPage />} />
        <Route path="prediction" element={<PredictionPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="informer" element={<InformerPage />} />
      </Route>
    </Routes>
  )
}