import { useState, useEffect } from 'react'
import { Scale, Search, FileCheck, Globe, ArrowRight, ChevronRight, Loader2 } from 'lucide-react'
import { GoogleLogin, type CredentialResponse } from '@react-oauth/google'
import { useStore } from '../store'
import { api } from '../api'

export default function LandingPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)
  const setAuth = useStore((s) => s.setAuth)

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

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

  const scrollToCta = () => {
    document.getElementById('cta-section')?.scrollIntoView({ behavior: 'smooth' })
  }

  /* ---------- Navbar ---------- */
  const navbar: React.CSSProperties = {
    position: 'sticky',
    top: 0,
    height: 64,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 32px',
    background: 'var(--bg-1)',
    borderBottom: '1px solid var(--border)',
    zIndex: 100,
  }

  const navBrand: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  }

  const navWordmark: React.CSSProperties = {
    fontSize: 20,
    fontWeight: 700,
    color: 'var(--text-1)',
    letterSpacing: '-0.02em',
  }

  /* ---------- Hero ---------- */
  const heroSection: React.CSSProperties = {
    background: 'var(--bg-0)',
    padding: isMobile ? '48px 20px' : '80px 32px',
  }

  const heroInner: React.CSSProperties = {
    maxWidth: 1200,
    margin: '0 auto',
    display: 'flex',
    flexDirection: isMobile ? 'column' : 'row',
    alignItems: 'center',
    gap: isMobile ? 48 : 64,
  }

  const heroLeft: React.CSSProperties = {
    flex: isMobile ? 'none' : '0 0 58%',
    width: isMobile ? '100%' : undefined,
  }

  const heroH1: React.CSSProperties = {
    fontSize: isMobile ? 32 : 48,
    fontWeight: 700,
    lineHeight: 1.15,
    color: 'var(--text-1)',
    letterSpacing: '-0.025em',
    marginBottom: 20,
  }

  const heroSub: React.CSSProperties = {
    fontSize: isMobile ? 16 : 18,
    lineHeight: 1.65,
    color: 'var(--text-2)',
    marginBottom: 32,
    maxWidth: 540,
  }

  const ctaBtn: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    padding: '14px 28px',
    background: 'var(--accent)',
    color: '#fff',
    fontSize: 16,
    fontWeight: 600,
    borderRadius: 'var(--radius-sm)',
    border: 'none',
    cursor: 'pointer',
    transition: 'background var(--transition)',
  }

  const heroRight: React.CSSProperties = {
    flex: isMobile ? 'none' : '0 0 38%',
    width: isMobile ? '100%' : undefined,
    display: 'flex',
    justifyContent: 'center',
    position: 'relative',
    minHeight: 320,
  }

  /* Floating cards visual */
  const cardVisualBase: React.CSSProperties = {
    position: 'absolute',
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: '16px 20px',
    boxShadow: 'var(--shadow-md)',
    width: 220,
  }

  const badgeBase: React.CSSProperties = {
    display: 'inline-block',
    fontSize: 11,
    fontWeight: 600,
    fontFamily: 'var(--font-mono)',
    padding: '3px 8px',
    borderRadius: 4,
    marginBottom: 8,
  }

  const lineShim: React.CSSProperties = {
    height: 8,
    borderRadius: 4,
    background: 'var(--bg-0)',
    marginTop: 8,
  }

  /* ---------- Features Section ---------- */
  const sectionWhite: React.CSSProperties = {
    background: 'var(--bg-1)',
    padding: isMobile ? '56px 20px' : '80px 32px',
  }

  const sectionGray: React.CSSProperties = {
    background: 'var(--bg-0)',
    padding: isMobile ? '56px 20px' : '80px 32px',
  }

  const sectionTitle: React.CSSProperties = {
    fontSize: isMobile ? 24 : 32,
    fontWeight: 700,
    color: 'var(--text-1)',
    textAlign: 'center',
    marginBottom: 12,
    letterSpacing: '-0.02em',
  }

  const sectionSubtitle: React.CSSProperties = {
    fontSize: 16,
    color: 'var(--text-2)',
    textAlign: 'center',
    marginBottom: 48,
    maxWidth: 600,
    marginLeft: 'auto',
    marginRight: 'auto',
  }

  const grid3: React.CSSProperties = {
    maxWidth: 1200,
    margin: '0 auto',
    display: 'grid',
    gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)',
    gap: 24,
  }

  const featureCard: React.CSSProperties = {
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: 32,
    transition: 'box-shadow var(--transition), transform var(--transition)',
    cursor: 'default',
  }

  const iconCircle: React.CSSProperties = {
    width: 48,
    height: 48,
    borderRadius: '50%',
    background: 'var(--accent-dim)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  }

  const cardTitle: React.CSSProperties = {
    fontSize: 18,
    fontWeight: 600,
    color: 'var(--text-1)',
    marginBottom: 8,
  }

  const cardDesc: React.CSSProperties = {
    fontSize: 14,
    lineHeight: 1.65,
    color: 'var(--text-2)',
  }

  /* ---------- How It Works ---------- */
  const stepsRow: React.CSSProperties = {
    maxWidth: 900,
    margin: '0 auto',
    display: 'flex',
    flexDirection: isMobile ? 'column' : 'row',
    alignItems: isMobile ? 'flex-start' : 'flex-start',
    gap: isMobile ? 40 : 0,
    position: 'relative',
  }

  const stepItem: React.CSSProperties = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: isMobile ? 'flex-start' : 'center',
    textAlign: isMobile ? 'left' : 'center',
    position: 'relative',
    paddingLeft: isMobile ? 60 : 0,
  }

  const stepCircle: React.CSSProperties = {
    width: 40,
    height: 40,
    borderRadius: '50%',
    background: 'var(--accent)',
    color: '#fff',
    fontSize: 14,
    fontWeight: 700,
    fontFamily: 'var(--font-mono)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    flexShrink: 0,
    position: isMobile ? 'absolute' : 'relative',
    left: isMobile ? 0 : undefined,
    top: isMobile ? 0 : undefined,
    zIndex: 2,
  }

  const stepTitle: React.CSSProperties = {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text-1)',
    marginBottom: 8,
  }

  const stepDesc: React.CSSProperties = {
    fontSize: 14,
    lineHeight: 1.6,
    color: 'var(--text-2)',
    maxWidth: 260,
    margin: isMobile ? undefined : '0 auto',
  }

  /* ---------- Source Cards ---------- */
  const sourceCard: React.CSSProperties = {
    background: 'var(--bg-1)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: 28,
    display: 'flex',
    alignItems: 'flex-start',
    gap: 16,
  }

  const sourceDot: (bg: string) => React.CSSProperties = (bg) => ({
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: bg,
    flexShrink: 0,
    marginTop: 5,
  })

  /* ---------- Stats ---------- */
  const statsBar: React.CSSProperties = {
    background: 'var(--bg-0)',
    padding: isMobile ? '40px 20px' : '48px 32px',
  }

  const statsInner: React.CSSProperties = {
    maxWidth: 800,
    margin: '0 auto',
    display: 'flex',
    flexDirection: isMobile ? 'column' : 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: isMobile ? 24 : 64,
  }

  const statItem: React.CSSProperties = {
    textAlign: 'center',
  }

  const statNumber: React.CSSProperties = {
    fontFamily: 'var(--font-mono)',
    fontSize: 28,
    fontWeight: 700,
    color: 'var(--accent)',
    lineHeight: 1.2,
  }

  const statLabel: React.CSSProperties = {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: 'var(--text-3)',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    marginTop: 4,
  }

  /* ---------- CTA Section ---------- */
  const ctaSection: React.CSSProperties = {
    background: 'var(--bg-1)',
    padding: isMobile ? '56px 20px' : '80px 32px',
    textAlign: 'center',
  }

  const ctaHeadline: React.CSSProperties = {
    fontSize: isMobile ? 24 : 32,
    fontWeight: 700,
    color: 'var(--text-1)',
    marginBottom: 24,
    letterSpacing: '-0.02em',
  }

  const ctaSmall: React.CSSProperties = {
    fontSize: 13,
    color: 'var(--text-3)',
    marginTop: 16,
  }

  /* ---------- Footer ---------- */
  const footer: React.CSSProperties = {
    background: 'var(--bg-0)',
    padding: '24px 32px',
    textAlign: 'center',
    borderTop: '1px solid var(--border)',
  }

  const footerText: React.CSSProperties = {
    fontSize: 13,
    color: 'var(--text-3)',
  }

  /* ---------- Error display ---------- */
  const errorStyle: React.CSSProperties = {
    color: 'var(--danger)',
    fontSize: 13,
    marginTop: 8,
    textAlign: 'center',
  }

  /* ---------- Google login wrapper ---------- */
  const googleWrap: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 44,
  }

  /* ---------- Connecting line for steps (desktop) ---------- */
  const connectingLine: React.CSSProperties = {
    position: 'absolute',
    top: 20,
    left: '16.67%',
    right: '16.67%',
    height: 2,
    background: 'var(--border)',
    zIndex: 1,
  }

  /* ---------- Connecting line for steps (mobile vertical) ---------- */
  const connectingLineVertical: React.CSSProperties = {
    position: 'absolute',
    left: 19,
    top: 40,
    bottom: 0,
    width: 2,
    background: 'var(--border)',
    zIndex: 1,
  }

  const renderGoogleLogin = () => {
    if (loading) {
      return (
        <div style={googleWrap}>
          <Loader2 size={22} color="var(--accent)" style={{ animation: 'spin 1s linear infinite' }} />
          <span style={{ color: 'var(--text-2)', fontSize: 14, marginLeft: 8 }}>Signing in...</span>
        </div>
      )
    }
    return (
      <div style={googleWrap}>
        <GoogleLogin
          onSuccess={handleGoogleSuccess}
          onError={() => setError('Google sign-in failed. Please try again.')}
          theme="outline"
          size="large"
          text="signin_with"
        />
      </div>
    )
  }

  const features = [
    { icon: <Search size={22} color="var(--accent)" />, title: 'Multi-Source Legal Search', desc: 'Search across CyLaw, HUDOC (ECHR), and EUR-Lex databases simultaneously. One query, three legal systems.' },
    { icon: <FileCheck size={22} color="var(--accent)" />, title: 'AI Answers with Citations', desc: 'Every answer cites exact sources with document references. No hallucinations \u2014 only verified legal information.' },
    { icon: <Globe size={22} color="var(--accent)" />, title: 'Multilingual Support', desc: 'Ask questions in English or Greek. Themis understands both and searches across all sources regardless of language.' },
  ]

  const steps = [
    { num: '01', title: 'Ask Your Question', desc: 'Type any legal question in English or Greek. From simple lookups to complex legal analysis.' },
    { num: '02', title: 'Intelligent Search', desc: 'Themis searches across multiple legal databases, finding the most relevant documents and precedents.' },
    { num: '03', title: 'Cited Answer', desc: 'Receive a comprehensive answer with inline citations. Click any citation to view the original source.' },
  ]

  const sources = [
    { name: 'CyLaw', desc: 'Cyprus legislation, court decisions, and legal provisions', dotColor: 'var(--badge-cylaw-fg)', badgeBg: 'var(--badge-cylaw-bg)', badgeFg: 'var(--badge-cylaw-fg)' },
    { name: 'HUDOC (ECHR)', desc: 'European Court of Human Rights case law and judgments', dotColor: 'var(--badge-echr-fg)', badgeBg: 'var(--badge-echr-bg)', badgeFg: 'var(--badge-echr-fg)' },
    { name: 'EUR-Lex', desc: 'EU treaties, legislation, directives, and regulations', dotColor: 'var(--badge-eurlex-fg)', badgeBg: 'var(--badge-eurlex-bg)', badgeFg: 'var(--badge-eurlex-fg)' },
  ]

  return (
    <div style={{ minHeight: '100%', background: 'var(--bg-0)' }}>
      {/* ==================== NAVBAR ==================== */}
      <nav style={navbar}>
        <div style={navBrand}>
          <Scale size={24} color="var(--accent)" />
          <span style={navWordmark}>Themis</span>
        </div>
        <div>
          {renderGoogleLogin()}
          {error && <p style={{ ...errorStyle, marginTop: 4 }}>{error}</p>}
        </div>
      </nav>

      {/* ==================== HERO ==================== */}
      <section style={heroSection}>
        <div style={heroInner}>
          <div style={heroLeft}>
            <h1 style={heroH1}>AI-powered legal research for Cyprus &amp; the EU</h1>
            <p style={heroSub}>
              Search across Cyprus Law, European Court of Human Rights, and EU legislation.
              Get precise answers with exact source citations.
            </p>
            <button
              style={ctaBtn}
              onClick={scrollToCta}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-hover)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--accent)')}
            >
              Get Started &mdash; Free
              <ArrowRight size={18} />
            </button>
          </div>

          {/* Floating document cards visual */}
          <div style={heroRight}>
            {/* Card 1 - CyLaw */}
            <div style={{
              ...cardVisualBase,
              top: 0,
              left: isMobile ? '5%' : 0,
              transform: 'rotate(-3deg)',
              zIndex: 3,
            }}>
              <span style={{ ...badgeBase, background: 'var(--badge-cylaw-bg)', color: 'var(--badge-cylaw-fg)' }}>CyLaw</span>
              <div style={lineShim} />
              <div style={{ ...lineShim, width: '75%' }} />
              <div style={{ ...lineShim, width: '85%' }} />
            </div>

            {/* Card 2 - HUDOC */}
            <div style={{
              ...cardVisualBase,
              top: 80,
              left: isMobile ? '20%' : 60,
              transform: 'rotate(2deg)',
              zIndex: 2,
            }}>
              <span style={{ ...badgeBase, background: 'var(--badge-echr-bg)', color: 'var(--badge-echr-fg)' }}>HUDOC</span>
              <div style={lineShim} />
              <div style={{ ...lineShim, width: '60%' }} />
              <div style={{ ...lineShim, width: '90%' }} />
            </div>

            {/* Card 3 - EUR-Lex */}
            <div style={{
              ...cardVisualBase,
              top: 170,
              left: isMobile ? '10%' : 20,
              transform: 'rotate(-1deg)',
              zIndex: 1,
            }}>
              <span style={{ ...badgeBase, background: 'var(--badge-eurlex-bg)', color: 'var(--badge-eurlex-fg)' }}>EUR-Lex</span>
              <div style={lineShim} />
              <div style={{ ...lineShim, width: '80%' }} />
              <div style={{ ...lineShim, width: '55%' }} />
            </div>
          </div>
        </div>
      </section>

      {/* ==================== FEATURES ==================== */}
      <section style={sectionWhite}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <h2 style={sectionTitle}>Everything you need for legal research</h2>
          <p style={sectionSubtitle}>Themis combines AI-powered search with verified legal sources to deliver accurate, cited answers.</p>
          <div style={grid3}>
            {features.map((f) => (
              <div
                key={f.title}
                style={featureCard}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = 'var(--shadow-md)'
                  e.currentTarget.style.transform = 'translateY(-2px)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = 'none'
                  e.currentTarget.style.transform = 'translateY(0)'
                }}
              >
                <div style={iconCircle}>{f.icon}</div>
                <h3 style={cardTitle}>{f.title}</h3>
                <p style={cardDesc}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ==================== HOW IT WORKS ==================== */}
      <section style={sectionGray}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <h2 style={sectionTitle}>How it works</h2>
          <p style={sectionSubtitle}>Three simple steps from question to cited answer.</p>
          <div style={stepsRow}>
            {/* Connecting line */}
            {!isMobile && <div style={connectingLine} />}
            {isMobile && <div style={connectingLineVertical} />}

            {steps.map((s, i) => (
              <div key={s.num} style={stepItem}>
                <div style={stepCircle}>{s.num}</div>
                <h3 style={stepTitle}>{s.title}</h3>
                <p style={stepDesc}>{s.desc}</p>
                {!isMobile && i < steps.length - 1 && (
                  <ChevronRight
                    size={20}
                    color="var(--text-3)"
                    style={{
                      position: 'absolute',
                      right: -12,
                      top: 10,
                      zIndex: 3,
                    }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ==================== SOURCE DATABASES ==================== */}
      <section style={sectionWhite}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <h2 style={sectionTitle}>Source databases</h2>
          <p style={sectionSubtitle}>Authoritative legal sources from Cyprus and the European Union.</p>
          <div style={grid3}>
            {sources.map((s) => (
              <div key={s.name} style={sourceCard}>
                <div style={sourceDot(s.dotColor)} />
                <div>
                  <span style={{
                    ...badgeBase,
                    background: s.badgeBg,
                    color: s.badgeFg,
                    marginBottom: 10,
                  }}>
                    {s.name}
                  </span>
                  <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--text-2)', marginTop: 4 }}>{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ==================== STATS BAR ==================== */}
      <section style={statsBar}>
        <div style={statsInner}>
          <div style={statItem}>
            <div style={statNumber}>3</div>
            <div style={statLabel}>Legal Databases</div>
          </div>
          <div style={statItem}>
            <div style={statNumber}>2</div>
            <div style={statLabel}>Languages</div>
          </div>
          <div style={statItem}>
            <div style={statNumber}>&bull;</div>
            <div style={statLabel}>Cited Answers</div>
          </div>
        </div>
      </section>

      {/* ==================== CTA FOOTER ==================== */}
      <section id="cta-section" style={ctaSection}>
        <h2 style={ctaHeadline}>Start researching with Themis</h2>
        {renderGoogleLogin()}
        {error && <p style={errorStyle}>{error}</p>}
        <p style={ctaSmall}>Free to use &middot; No credit card required</p>
      </section>

      {/* ==================== FOOTER ==================== */}
      <footer style={footer}>
        <p style={footerText}>&copy; 2026 Themis &middot; AI-powered legal research</p>
      </footer>
    </div>
  )
}
