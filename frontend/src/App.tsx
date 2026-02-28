import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { GoogleOAuthProvider } from '@react-oauth/google'
import { useStore } from './store'
import { api } from './api'
import LandingPage from './components/LandingPage'
import MainLayout from './components/MainLayout'
import ChatInterface from './components/ChatInterface'
import SettingsPage from './components/SettingsPage'
import LegalPages from './components/LegalPages'
import ProtectedRoute from './components/ProtectedRoute'

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''

function AuthSplash() {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      width: '100%',
      background: 'var(--bg-0)',
      gap: 16,
    }}>
      <span style={{
        fontSize: 24,
        fontWeight: 700,
        color: 'var(--text-1)',
        letterSpacing: '-0.02em',
      }}>
        Themis
      </span>
      <div style={{
        width: 24,
        height: 24,
        border: '3px solid var(--border)',
        borderTopColor: 'var(--accent)',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
      }} />
    </div>
  )
}

function RootRedirect() {
  const isAuthenticated = useStore((s) => s.isAuthenticated)
  if (isAuthenticated) return <Navigate to="/chat" replace />
  return <LandingPage />
}

export default function App() {
  const authLoading = useStore((s) => s.authLoading)
  const setAuthLoading = useStore((s) => s.setAuthLoading)
  const logout = useStore((s) => s.logout)

  useEffect(() => {
    const jwt = localStorage.getItem('jwt')
    if (!jwt) {
      setAuthLoading(false)
      return
    }
    api.auth.me()
      .then(() => setAuthLoading(false))
      .catch(() => { logout(); setAuthLoading(false) })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (authLoading) {
    return <AuthSplash />
  }

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <Routes>
        <Route path="/" element={<RootRedirect />} />
        <Route path="/legal/:tab" element={<LegalPages />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<MainLayout />}>
            <Route path="/chat" element={<ChatInterface />} />
            <Route path="/chat/:conversationId" element={<ChatInterface />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </GoogleOAuthProvider>
  )
}
