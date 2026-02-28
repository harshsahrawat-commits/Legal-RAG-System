import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import Header from './Header'
import Sidebar from './Sidebar'
import { useStore } from '../store'
import { api } from '../api'

export default function MainLayout() {
  const setFamilies = useStore((s) => s.setFamilies)
  const pruneStaleFamilies = useStore((s) => s.pruneStaleFamilies)

  // Load families on mount and prune stale family toggle IDs
  useEffect(() => {
    api.families.list()
      .then(({ data }) => {
        setFamilies(data)
        pruneStaleFamilies(data.map((f) => f.id))
      })
      .catch(() => {})
  }, [setFamilies, pruneStaleFamilies])

  return (
    <div style={styles.layout}>
      <Sidebar />
      <div style={styles.main}>
        <Header />
        <Outlet />
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
