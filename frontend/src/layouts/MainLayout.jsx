import { Link, Outlet, useLocation } from 'react-router-dom'

const menus = [
  { path: '/dashboard', label: '首页' },
  { path: '/events', label: '事件分析' },
  { path: '/prediction', label: '预测结果' },
  { path: '/models', label: '模型对比' },
  { path: '/backtest', label: '策略回测' },
  { path: '/informer', label: 'Informer实验' },
  
]

export default function MainLayout() {
  const location = useLocation()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="logo-box">
          <div className="logo-title">股票事件平台</div>
          <div className="logo-subtitle">Event-driven Stock Analysis</div>
        </div>

        <nav className="menu">
          {menus.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`menu-item ${location.pathname === item.path ? 'active' : ''}`}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      <div className="main-panel">
        <header className="topbar">
          <div className="topbar-content">
            <div className="topbar-text">
              <h1 className="topbar-title">基于事件驱动的股票短期价格波动数据分析平台</h1>
              <p className="topbar-subtitle">React + FastAPI + PyTorch</p>
            </div>
          </div>
        </header>

        <main className="page-container">
          <Outlet />
        </main>
      </div>
    </div>
  )
}