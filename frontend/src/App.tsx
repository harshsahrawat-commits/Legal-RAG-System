import { useStore } from './store'
import LoginScreen from './components/LoginScreen'
import MainLayout from './components/MainLayout'

export default function App() {
  const isAuthenticated = useStore((s) => s.isAuthenticated)

  return isAuthenticated ? <MainLayout /> : <LoginScreen />
}
