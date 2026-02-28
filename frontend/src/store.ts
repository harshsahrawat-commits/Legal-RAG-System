import { create } from 'zustand'
import type { SourceToggles, ChatMessage, UserInfo, Conversation, DocumentFamily, MetadataFilters } from './types'

const STORAGE_KEYS = {
  SOURCE_TOGGLES: 'sourceToggles',
  RESEARCH_MODE: 'researchMode',
  RESEARCH_FILTERS: 'researchFilters',
} as const

function loadSourceToggles(): SourceToggles {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.SOURCE_TOGGLES)
    if (raw) {
      const p = JSON.parse(raw)
      return {
        cylaw: typeof p.cylaw === 'boolean' ? p.cylaw : true,
        hudoc: typeof p.hudoc === 'boolean' ? p.hudoc : true,
        eurlex: typeof p.eurlex === 'boolean' ? p.eurlex : true,
        families: Array.isArray(p.families) ? p.families : [],
      }
    }
  } catch { /* ignore */ }
  return { cylaw: true, hudoc: true, eurlex: true, families: [] }
}

function loadResearchMode(): boolean {
  try { return localStorage.getItem(STORAGE_KEYS.RESEARCH_MODE) === 'true' } catch { return false }
}

function loadResearchFilters(): MetadataFilters {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.RESEARCH_FILTERS)
    return raw ? JSON.parse(raw) : {}
  } catch { return {} }
}

interface AppState {
  // Auth (Google OAuth + JWT)
  jwt: string | null
  user: UserInfo | null
  isAuthenticated: boolean
  authLoading: boolean
  setAuthLoading: (loading: boolean) => void
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

  sourceToggles: SourceToggles
  setSourceToggle: (source: keyof Omit<SourceToggles, 'families'>, enabled: boolean) => void
  setFamilyToggle: (familyId: string, enabled: boolean) => void
  pruneStaleFamilies: (validIds: string[]) => void

  // Document families
  families: DocumentFamily[]
  setFamilies: (families: DocumentFamily[]) => void

  // Research mode
  researchMode: boolean
  researchFilters: MetadataFilters
  setResearchMode: (enabled: boolean) => void
  setResearchFilter: <K extends keyof MetadataFilters>(key: K, value: MetadataFilters[K]) => void
  resetResearchFilters: () => void

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
  authLoading: !!localStorage.getItem('jwt'),
  setAuthLoading: (loading) => set({ authLoading: loading }),

  setAuth: (token: string, user: UserInfo) => {
    localStorage.setItem('jwt', token)
    localStorage.setItem('user', JSON.stringify(user))
    set({ jwt: token, user, isAuthenticated: true })
  },

  logout: () => {
    localStorage.removeItem('jwt')
    localStorage.removeItem('user')
    sessionStorage.removeItem('chatMessages')
    localStorage.removeItem(STORAGE_KEYS.SOURCE_TOGGLES)
    localStorage.removeItem(STORAGE_KEYS.RESEARCH_MODE)
    localStorage.removeItem(STORAGE_KEYS.RESEARCH_FILTERS)
    set({
      jwt: null,
      user: null,
      isAuthenticated: false,
      messages: [],
      conversations: [],
      activeConversationId: null,
      families: [],
      sourceToggles: { cylaw: true, hudoc: true, eurlex: true, families: [] },
      researchMode: false,
      researchFilters: {},
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

  sourceToggles: loadSourceToggles(),
  setSourceToggle: (source, enabled) => set((state) => {
    const newToggles = { ...state.sourceToggles, [source]: enabled }
    localStorage.setItem(STORAGE_KEYS.SOURCE_TOGGLES, JSON.stringify(newToggles))
    return { sourceToggles: newToggles }
  }),
  setFamilyToggle: (familyId, enabled) => set((state) => {
    const current = state.sourceToggles.families
    const newFamilies = enabled
      ? [...current, familyId]
      : current.filter((id) => id !== familyId)
    const newToggles = { ...state.sourceToggles, families: newFamilies }
    localStorage.setItem(STORAGE_KEYS.SOURCE_TOGGLES, JSON.stringify(newToggles))
    return { sourceToggles: newToggles }
  }),
  pruneStaleFamilies: (validIds) => set((state) => {
    const pruned = state.sourceToggles.families.filter((id) => validIds.includes(id))
    if (pruned.length === state.sourceToggles.families.length) return state
    const newToggles = { ...state.sourceToggles, families: pruned }
    localStorage.setItem(STORAGE_KEYS.SOURCE_TOGGLES, JSON.stringify(newToggles))
    return { sourceToggles: newToggles }
  }),

  families: [],
  setFamilies: (families) => set({ families }),

  researchMode: loadResearchMode(),
  researchFilters: loadResearchFilters(),
  setResearchMode: (enabled) => {
    localStorage.setItem(STORAGE_KEYS.RESEARCH_MODE, String(enabled))
    set({ researchMode: enabled })
  },
  setResearchFilter: (key, value) => set((state) => {
    const newFilters = { ...state.researchFilters, [key]: value }
    localStorage.setItem(STORAGE_KEYS.RESEARCH_FILTERS, JSON.stringify(newFilters))
    return { researchFilters: newFilters }
  }),
  resetResearchFilters: () => {
    localStorage.removeItem(STORAGE_KEYS.RESEARCH_FILTERS)
    set({ researchFilters: {} })
  },
}))
