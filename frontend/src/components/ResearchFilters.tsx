import { useStore } from '../store'

const DOCUMENT_TYPES = [
  { value: '', label: 'All Types' },
  { value: 'case_law', label: 'Case Law' },
  { value: 'statute', label: 'Legislation' },
  { value: 'regulation', label: 'Regulations' },
]

const JURISDICTIONS = [
  { value: '', label: 'All' },
  { value: 'Cyprus', label: 'Cyprus' },
  { value: 'ECHR', label: 'ECHR' },
  { value: 'EU', label: 'EU' },
]

const COURT_LEVELS = [
  'Supreme',
  'Constitutional',
  'Appeal',
  'Administrative',
  'First Instance',
]

const OUTCOMES = [
  { value: '', label: 'Any Outcome' },
  { value: 'violation_found_true', label: 'Violation Found' },
  { value: 'violation_found_false', label: 'No Violation' },
  { value: 'annulment_granted_true', label: 'Annulment Granted' },
  { value: 'annulment_granted_false', label: 'Appeal Dismissed' },
]

export default function ResearchFilters() {
  const filters = useStore((s) => s.researchFilters)
  const setFilter = useStore((s) => s.setResearchFilter)

  const handleCourtToggle = (level: string) => {
    const current = filters.court_levels || []
    if (current.includes(level)) {
      setFilter('court_levels', current.filter((l) => l !== level))
    } else {
      setFilter('court_levels', [...current, level])
    }
  }

  const handleOutcomeChange = (value: string) => {
    if (!value) {
      setFilter('outcome', null)
      return
    }
    const parts = value.match(/^(.+)_(true|false)$/)
    if (parts) {
      setFilter('outcome', { [parts[1]]: parts[2] === 'true' })
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.headerLabel}>Research Filters</span>
      </div>

      <div style={styles.grid}>
        {/* Document Type */}
        <div style={styles.field}>
          <label style={styles.label}>Document Type</label>
          <select
            style={styles.select}
            value={filters.document_type || ''}
            onChange={(e) => setFilter('document_type', e.target.value || null)}
          >
            {DOCUMENT_TYPES.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </div>

        {/* Jurisdiction */}
        <div style={styles.field}>
          <label style={styles.label}>Jurisdiction</label>
          <select
            style={styles.select}
            value={filters.jurisdiction || ''}
            onChange={(e) => setFilter('jurisdiction', e.target.value || null)}
          >
            {JURISDICTIONS.map((j) => (
              <option key={j.value} value={j.value}>{j.label}</option>
            ))}
          </select>
        </div>

        {/* Court Level */}
        <div style={styles.field}>
          <label style={styles.label}>Court Level</label>
          <div style={styles.checkboxGroup}>
            {COURT_LEVELS.map((level) => (
              <label key={level} style={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={(filters.court_levels || []).includes(level)}
                  onChange={() => handleCourtToggle(level)}
                  style={styles.checkbox}
                />
                {level}
              </label>
            ))}
          </div>
        </div>

        {/* Outcome */}
        <div style={styles.field}>
          <label style={styles.label}>Outcome</label>
          <select
            style={styles.select}
            value={
              filters.outcome
                ? Object.entries(filters.outcome).map(([k, v]) => `${k}_${v}`)[0] || ''
                : ''
            }
            onChange={(e) => handleOutcomeChange(e.target.value)}
          >
            {OUTCOMES.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '12px 24px',
    borderTop: '1px solid var(--border)',
    background: '#FAFAF8',
    flexShrink: 0,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  headerLabel: {
    fontSize: 11,
    fontWeight: 600,
    fontFamily: 'var(--font-mono)',
    textTransform: 'uppercase' as const,
    letterSpacing: '1.5px',
    color: 'var(--text-3)',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
    gap: 12,
  },
  field: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 4,
  },
  label: {
    fontSize: 11,
    fontWeight: 500,
    color: 'var(--text-2)',
  },
  select: {
    padding: '6px 10px',
    fontSize: 13,
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm, 6px)',
    background: '#FFFFFF',
    color: 'var(--text-1)',
    outline: 'none',
    cursor: 'pointer',
  },
  checkboxGroup: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: 6,
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 12,
    color: 'var(--text-1)',
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
  },
  checkbox: {
    accentColor: 'var(--accent)',
  },
}
