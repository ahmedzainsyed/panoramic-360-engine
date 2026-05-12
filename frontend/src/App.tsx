import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAppStore } from './store';
import { Dashboard } from './pages/Dashboard';
import { PanoramaView } from './pages/PanoramaView';
import { UploadPage } from './pages/UploadPage';
import { AnalyticsPage } from './pages/AnalyticsPage';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 2, staleTime: 30_000 } }
});

function Sidebar() {
  const { sidebarOpen, toggleSidebar, userEmail, logout, alerts } = useAppStore();
  const unread = alerts.filter(a => !a.acknowledged).length;

  const navItems = [
    { to: '/', label: 'Dashboard', icon: '🏠', exact: true },
    { to: '/viewer', label: '360° Viewer', icon: '🌐' },
    { to: '/upload', label: 'Upload', icon: '⬆️' },
    { to: '/analytics', label: 'Analytics', icon: '📊' },
  ];

  return (
    <aside className={`bg-gray-900 border-r border-gray-800 flex flex-col transition-all duration-200 ${sidebarOpen ? 'w-56' : 'w-14'}`}>
      {/* Logo */}
      <div className="flex items-center gap-3 p-4 border-b border-gray-800">
        <span className="text-2xl flex-shrink-0">🌐</span>
        {sidebarOpen && <span className="text-white font-bold text-sm leading-tight">360° Site<br/>Intelligence</span>}
      </div>

      {/* Nav */}
      <nav className="flex-1 p-2 space-y-1">
        {navItems.map(({ to, label, icon }) => (
          <NavLink key={to} to={to} end={to === '/'}
            className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              isActive ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}>
            <span className="text-lg flex-shrink-0">{icon}</span>
            {sidebarOpen && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Alerts badge */}
      {unread > 0 && (
        <div className="px-2 pb-2">
          <div className="bg-red-900/40 border border-red-800 rounded-lg px-3 py-2 text-xs text-red-300 flex items-center gap-2">
            <span className="animate-pulse">🔴</span>
            {sidebarOpen && `${unread} alert${unread > 1 ? 's' : ''}`}
          </div>
        </div>
      )}

      {/* User + toggle */}
      <div className="p-2 border-t border-gray-800 space-y-1">
        {sidebarOpen && userEmail && (
          <div className="px-3 py-2 text-xs text-gray-500 truncate">{userEmail}</div>
        )}
        <button onClick={toggleSidebar}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 text-sm">
          <span className="flex-shrink-0">{sidebarOpen ? '◀' : '▶'}</span>
          {sidebarOpen && <span>Collapse</span>}
        </button>
        {sidebarOpen && (
          <button onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-gray-400 hover:text-red-400 hover:bg-gray-800 text-sm">
            <span>🚪</span><span>Logout</span>
          </button>
        )}
      </div>
    </aside>
  );
}

function LoginPage() {
  const { setToken } = useAppStore();
  const [email, setEmail] = React.useState('demo@site.com');
  const [password, setPassword] = React.useState('demo123');
  const [loading, setLoading] = React.useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username: email, password }),
      });
      if (res.ok) {
        const data = await res.json();
        setToken(data.access_token, email);
      }
    } catch {
      // For demo, set a mock token
      setToken('demo-token', email);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 w-96">
        <div className="text-center mb-6">
          <div className="text-5xl mb-3">🌐</div>
          <h1 className="text-white text-xl font-bold">360° Site Intelligence</h1>
          <p className="text-gray-400 text-sm mt-1">Construction Safety Platform</p>
        </div>
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="text-gray-400 text-xs block mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <div>
            <label className="text-gray-400 text-xs block mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-blue-500" />
          </div>
          <button type="submit" disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2.5 text-sm font-medium transition-colors disabled:opacity-50">
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}

function AppLayout() {
  return (
    <div className="flex h-screen bg-gray-950 overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/viewer" element={<PanoramaView />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  const { token } = useAppStore();
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Toaster position="top-right" toastOptions={{
          style: { background: '#1f2937', color: '#f9fafb', border: '1px solid #374151' }
        }} />
        {token ? <AppLayout /> : <LoginPage />}
      </BrowserRouter>
    </QueryClientProvider>
  );
}
