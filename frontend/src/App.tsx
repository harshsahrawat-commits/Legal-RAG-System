import { GoogleOAuthProvider } from '@react-oauth/google'
import { useStore } from './store'
import LoginScreen from './components/LoginScreen'
import MainLayout from './components/MainLayout'

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''

export default function App() {
  const isAuthenticated = useStore((s) => s.isAuthenticated)

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      {isAuthenticated ? <MainLayout /> : <LoginScreen />}
    </GoogleOAuthProvider>
  )
}
