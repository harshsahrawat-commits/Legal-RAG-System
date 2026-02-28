import { Navigate, Outlet } from 'react-router-dom'
import { useStore } from '../store'

export default function ProtectedRoute() {
  const isAuthenticated = useStore((s) => s.isAuthenticated)

  if (!isAuthenticated) {
    return <Navigate to="/" replace />
  }

  return <Outlet />
}
