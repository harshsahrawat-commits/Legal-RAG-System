import { useState } from 'react'
import { Scale, Loader2 } from 'lucide-react'
import { GoogleLogin, type CredentialResponse } from '@react-oauth/google'
import { api } from '../api'
import { useStore } from '../store'

export default function LoginScreen() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const setAuth = useStore((s) => s.setAuth)

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

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.logo}>
          <Scale size={32} color="var(--accent)" style={{ filter: 'drop-shadow(0 0 12px var(--accent-glow))' }} />
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
              text="continue_with"
            />
          </div>
        )}

        {error && <p style={styles.error}>{error}</p>}

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
    background: 'var(--glass-bg)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    borderRadius: 'var(--radius-lg)',
    border: '1px solid var(--glass-border)',
    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.6), 0 0 60px var(--accent-glow)',
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
  footer: {
    marginTop: 24,
    fontSize: 12,
    color: 'var(--text-3)',
  },
}
