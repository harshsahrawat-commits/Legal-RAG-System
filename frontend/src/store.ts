import { create } from 'zustand'
import type { SourceInfo, ChatMessage } from './types'

interface AppState {
  apiKey: string | null
  isAuthenticated: boolean
  setApiKey: (key: string) => void
  logout: () => void

  messages: ChatMessage[]
  setMessages: (msgs: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void
  clearMessages: () => void

  selectedDocumentId: string | null
  setSelectedDocumentId: (id: string | null) => void

  sourcePanelOpen: boolean
  selectedSourceIndex: number
  currentSources: SourceInfo[]
  openSourcePanel: (index: number, sources: SourceInfo[]) => void
  closeSourcePanel: () => void
  navigateSource: (direction: 'prev' | 'next') => void
}

function loadMessages(): ChatMessage[] {
  try {
    const raw = sessionStorage.getItem('chatMessages')
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveMessages(msgs: ChatMessage[]) {
  try {
    sessionStorage.setItem('chatMessages', JSON.stringify(msgs))
  } catch { /* quota exceeded */ }
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
    sessionStorage.removeItem('chatMessages')
    set({ apiKey: null, isAuthenticated: false, messages: [] })
  },

  messages: loadMessages(),
  setMessages: (msgs) => {
    const newMsgs = typeof msgs === 'function' ? msgs(get().messages) : msgs
    saveMessages(newMsgs)
    set({ messages: newMsgs })
  },
  clearMessages: () => {
    sessionStorage.removeItem('chatMessages')
    set({ messages: [] })
  },

  selectedDocumentId: null,
  setSelectedDocumentId: (id) => set({ selectedDocumentId: id }),

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
