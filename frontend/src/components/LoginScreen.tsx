import { useState } from 'react'
import { Scale, ArrowRight, Loader2 } from 'lucide-react'
import { GoogleLogin, type CredentialResponse } from '@react-oauth/google'
import { api } from '../api'
import { useStore } from '../store'

export default function LoginScreen() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const setAuth = useStore((s) => s.setAuth)

  // Legacy API key state
  const [showApiKey, setShowApiKey] = useState(false)
  const [key, setKey] = useState('')
  const [keyLoading, setKeyLoading] = useState(false)
  const setApiKey = useStore((s) => s.setApiKey)

  const handleGoogleSuccess = async (response: CredentialResponse) => {
    if (!response.credential) {
      setError('Google sign-in failed. Please try again.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const { data } = await api.auth.google(response.credential)
      setAuth(data.token, data.user)
    } catch {
      setError('Authentication failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleApiKeySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!key.trim()) return

    setKeyLoading(true)
    setError('')

    try {
      await api.validateKey(key.trim())
      setApiKey(key.trim())
    } catch {
      setError('Invalid API key. Please check and try again.')
    } finally {
      setKeyLoading(false)
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.logo}>
          <Scale size={32} color="var(--accent)" />
          <h1 style={styles.title}>Legal RAG</h1>
        </div>
        <p style={styles.subtitle}>AI-powered legal document analysis with precise citations</p>

        {loading ? (
          <div style={styles.loadingWrap}>
            <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
            <span style={{ color: 'var(--text-2)', fontSize: 14 }}>Signing in...</span>
          </div>
        ) : (
          <div style={styles.googleWrap}>
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('Google sign-in failed. Please try again.')}
              theme="filled_black"
              size="large"
              width="320"
              text="signin_with"
            />
          </div>
        )}

        {error && <p style={styles.error}>{error}</p>}

        {/* Legacy API key option */}
        <div style={styles.divider}>
          <span style={styles.dividerLine} />
          <button
            onClick={() => setShowApiKey(!showApiKey)}
            style={styles.dividerText}
          >
            {showApiKey ? 'Hide' : 'Use API key instead'}
          </button>
          <span style={styles.dividerLine} />
        </div>

        {showApiKey && (
          <form onSubmit={handleApiKeySubmit} style={styles.form}>
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="lrag_xxxxxxxxxxxxxxxxxx"
              style={styles.input}
            />
            <button type="submit" disabled={keyLoading || !key.trim()} style={styles.button}>
              {keyLoading ? (
                <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <>Sign In <ArrowRight size={16} /></>
              )}
            </button>
          </form>
        )}

        <p style={styles.footer}>Cyprus Law &middot; ECHR &middot; EU Law &middot; Cited Answers</p>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-0)',
  },
  card: {
    width: '100%',
    maxWidth: 400,
    padding: 40,
    background: 'var(--bg-1)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-lg)',
    textAlign: 'center' as const,
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: '-0.02em',
  },
  subtitle: {
    color: 'var(--text-2)',
    fontSize: 14,
    marginBottom: 32,
  },
  googleWrap: {
    display: 'flex',
    justifyContent: 'center',
    marginBottom: 8,
  },
  loadingWrap: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    padding: 16,
  },
  error: {
    color: 'var(--danger)',
    fontSize: 13,
    margin: '8px 0 0',
  },
  divider: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    margin: '20px 0',
  },
  dividerLine: {
    flex: 1,
    height: 1,
    background: 'var(--border)',
  },
  dividerText: {
    fontSize: 12,
    color: 'var(--text-3)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
  },
  form: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 12,
    textAlign: 'left' as const,
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    fontSize: 15,
    background: 'var(--bg-2)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: '12px 24px',
    background: 'var(--accent)',
    color: '#000',
    fontWeight: 600,
    fontSize: 15,
    borderRadius: 'var(--radius-sm)',
    border: 'none',
    cursor: 'pointer',
    transition: 'background var(--transition)',
  },
  footer: {
    marginTop: 24,
    fontSize: 12,
    color: 'var(--text-3)',
  },
}
