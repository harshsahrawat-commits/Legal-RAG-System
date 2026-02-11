import { create } from 'zustand'
import type { SourceInfo } from './types'

interface AppState {
  apiKey: string | null
  isAuthenticated: boolean
  setApiKey: (key: string) => void
  logout: () => void

  sourcePanelOpen: boolean
  selectedSourceIndex: number
  currentSources: SourceInfo[]
  openSourcePanel: (index: number, sources: SourceInfo[]) => void
  closeSourcePanel: () => void
  navigateSource: (direction: 'prev' | 'next') => void
}

export const useStore = create<AppState>((set, get) => ({
  apiKey: localStorage.getItem('apiKey'),
  isAuthenticated: !!localStorage.getItem('apiKey'),

  setApiKey: (key: string) => {
    localStorage.setItem('apiKey', key)
    set({ apiKey: key, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('apiKey')
    set({ apiKey: null, isAuthenticated: false })
  },

  sourcePanelOpen: false,
  selectedSourceIndex: 0,
  currentSources: [],

  openSourcePanel: (index: number, sources: SourceInfo[]) => {
    set({ sourcePanelOpen: true, selectedSourceIndex: index, currentSources: sources })
  },

  closeSourcePanel: () => {
    set({ sourcePanelOpen: false })
  },

  navigateSource: (direction: 'prev' | 'next') => {
    const { selectedSourceIndex, currentSources } = get()
    if (direction === 'prev' && selectedSourceIndex > 0) {
      set({ selectedSourceIndex: selectedSourceIndex - 1 })
    } else if (direction === 'next' && selectedSourceIndex < currentSources.length - 1) {
      set({ selectedSourceIndex: selectedSourceIndex + 1 })
    }
  },
}))
