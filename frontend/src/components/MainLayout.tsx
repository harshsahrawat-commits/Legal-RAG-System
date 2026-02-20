import { useEffect } from 'react'
import Header from './Header'
import Sidebar from './Sidebar'
import ChatInterface from './ChatInterface'
import SettingsPage from './SettingsPage'
import { useStore } from '../store'
import { api } from '../api'

export default function MainLayout() {
  const settingsOpen = useStore((s) => s.settingsOpen)
  const setFamilies = useStore((s) => s.setFamilies)

  // Load families on mount
  useEffect(() => {
    api.families.list()
      .then(({ data }) => setFamilies(data))
      .catch(() => {})
  }, [setFamilies])

  return (
    <div style={styles.layout}>
      <Sidebar />
      <div style={styles.main}>
        <Header />
        {settingsOpen ? <SettingsPage /> : <ChatInterface />}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  layout: {
    display: 'flex',
    height: '100%',
    width: '100%',
    overflow: 'hidden',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    minWidth: 0,
  },
}
