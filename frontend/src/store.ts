import { create } from 'zustand'
import type { SourceInfo, SourceToggles, ChatMessage, UserInfo, Conversation } from './types'

interface AppState {
  // Auth (JWT-based)
  jwt: string | null
  user: UserInfo | null
  isAuthenticated: boolean
  setAuth: (token: string, user: UserInfo) => void
  logout: () => void

  // Legacy API key support (kept for backward compat)
  apiKey: string | null
  setApiKey: (key: string) => void

  // Messages (current conversation)
  messages: ChatMessage[]
  setMessages: (msgs: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => void
  clearMessages: () => void

  // Conversations
  conversations: Conversation[]
  setConversations: (convos: Conversation[]) => void
  activeConversationId: string | null
  setActiveConversationId: (id: string | null) => void

  selectedDocumentId: string | null
  setSelectedDocumentId: (id: string | null) => void

  sourceToggles: SourceToggles
  setSourceToggle: (source: keyof SourceToggles, enabled: boolean) => void

  sourcePanelOpen: boolean
  selectedSourceIndex: number
  currentSources: SourceInfo[]
  openSourcePanel: (index: number, sources: SourceInfo[]) => void
  closeSourcePanel: () => void
  navigateSource: (direction: 'prev' | 'next') => void

  // Settings page visibility
  settingsOpen: boolean
  setSettingsOpen: (open: boolean) => void
}

function loadUser(): UserInfo | null {
  try {
    const raw = localStorage.getItem('user')
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export const useStore = create<AppState>((set, get) => ({
  // Auth — check JWT first, fall back to legacy apiKey
  jwt: localStorage.getItem('jwt'),
  user: loadUser(),
  isAuthenticated: !!(localStorage.getItem('jwt') || localStorage.getItem('apiKey')),

  apiKey: localStorage.getItem('apiKey'),

  setAuth: (token: string, user: UserInfo) => {
    localStorage.setItem('jwt', token)
    localStorage.setItem('user', JSON.stringify(user))
    // Clear legacy key if present
    localStorage.removeItem('apiKey')
    set({ jwt: token, user, apiKey: null, isAuthenticated: true })
  },

  setApiKey: (key: string) => {
    localStorage.setItem('apiKey', key)
    set({ apiKey: key, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('jwt')
    localStorage.removeItem('user')
    localStorage.removeItem('apiKey')
    sessionStorage.removeItem('chatMessages')
    set({
      jwt: null,
      user: null,
      apiKey: null,
      isAuthenticated: false,
      messages: [],
      conversations: [],
      activeConversationId: null,
    })
  },

  messages: [],
  setMessages: (msgs) => {
    const newMsgs = typeof msgs === 'function' ? msgs(get().messages) : msgs
    set({ messages: newMsgs })
  },
  clearMessages: () => {
    set({ messages: [] })
  },

  conversations: [],
  setConversations: (convos) => set({ conversations: convos }),
  activeConversationId: null,
  setActiveConversationId: (id) => set({ activeConversationId: id }),

  selectedDocumentId: null,
  setSelectedDocumentId: (id) => set({ selectedDocumentId: id }),

  sourceToggles: { cylaw: true, hudoc: true, eurlex: true },
  setSourceToggle: (source, enabled) => set((state) => ({
    sourceToggles: { ...state.sourceToggles, [source]: enabled },
  })),

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

  settingsOpen: false,
  setSettingsOpen: (open) => set({ settingsOpen: open }),
}))
