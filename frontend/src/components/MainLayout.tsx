import Header from './Header'
import Sidebar from './Sidebar'
import ChatInterface from './ChatInterface'
import SourcePanel from './SourcePanel'
import SettingsPage from './SettingsPage'
import { useStore } from '../store'

export default function MainLayout() {
  const settingsOpen = useStore((s) => s.settingsOpen)

  return (
    <div style={styles.layout}>
      <Sidebar />
      <div style={styles.main}>
        <Header />
        {settingsOpen ? <SettingsPage /> : <ChatInterface />}
      </div>
      {!settingsOpen && <SourcePanel />}
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
