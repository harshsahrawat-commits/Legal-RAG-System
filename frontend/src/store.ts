import { create } from 'zustand'
import type { SourceToggles, ChatMessage, UserInfo, Conversation, DocumentFamily } from './types'

interface AppState {
  // Auth (Google OAuth + JWT)
  jwt: string | null
  user: UserInfo | null
  isAuthenticated: boolean
  setAuth: (token: string, user: UserInfo) => void
  logout: () => void

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
  setSourceToggle: (source: keyof Omit<SourceToggles, 'families'>, enabled: boolean) => void
  setFamilyToggle: (familyId: string, enabled: boolean) => void

  // Document families
  families: DocumentFamily[]
  setFamilies: (families: DocumentFamily[]) => void

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
  jwt: localStorage.getItem('jwt'),
  user: loadUser(),
  isAuthenticated: !!localStorage.getItem('jwt'),

  setAuth: (token: string, user: UserInfo) => {
    localStorage.setItem('jwt', token)
    localStorage.setItem('user', JSON.stringify(user))
    set({ jwt: token, user, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('jwt')
    localStorage.removeItem('user')
    sessionStorage.removeItem('chatMessages')
    set({
      jwt: null,
      user: null,
      isAuthenticated: false,
      messages: [],
      conversations: [],
      activeConversationId: null,
      families: [],
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

  sourceToggles: { cylaw: true, hudoc: true, eurlex: true, families: [] },
  setSourceToggle: (source, enabled) => set((state) => ({
    sourceToggles: { ...state.sourceToggles, [source]: enabled },
  })),
  setFamilyToggle: (familyId, enabled) => set((state) => {
    const current = state.sourceToggles.families
    if (enabled) {
      return { sourceToggles: { ...state.sourceToggles, families: [...current, familyId] } }
    }
    return { sourceToggles: { ...state.sourceToggles, families: current.filter((id) => id !== familyId) } }
  }),

  families: [],
  setFamilies: (families) => set({ families }),

  settingsOpen: false,
  setSettingsOpen: (open) => set({ settingsOpen: open }),
}))
