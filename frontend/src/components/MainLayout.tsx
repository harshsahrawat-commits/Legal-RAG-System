import { useState } from 'react'
import Header from './Header'
import Sidebar from './Sidebar'
import ChatInterface from './ChatInterface'
import SourcePanel from './SourcePanel'
import ConfigModal from './ConfigModal'

export default function MainLayout() {
  const [configOpen, setConfigOpen] = useState(false)

  return (
    <div style={styles.layout}>
      <Sidebar />
      <div style={styles.main}>
        <Header onSettingsClick={() => setConfigOpen(true)} />
        <ChatInterface />
      </div>
      <SourcePanel />
      <ConfigModal open={configOpen} onClose={() => setConfigOpen(false)} />
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
