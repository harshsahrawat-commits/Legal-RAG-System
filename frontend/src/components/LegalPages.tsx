import { useState, useEffect } from 'react'
import { ArrowLeft, Scale, FileText, Shield, Users, AlertTriangle } from 'lucide-react'
import { useStore } from '../store'

const LAST_UPDATED = 'February 27, 2026'
const COMPANY = '[Company Name]'
const CONTACT_EMAIL = '[contact@example.com]'
const DPO_EMAIL = '[dpo@example.com]'

type TabId = 'terms' | 'privacy' | 'gdpr' | 'liability'

interface Tab {
  id: TabId
  label: string
  icon: React.ReactNode
}

const TABS: Tab[] = [
  { id: 'terms', label: 'Terms of Service', icon: <FileText size={15} /> },
  { id: 'privacy', label: 'Privacy Policy', icon: <Shield size={15} /> },
  { id: 'gdpr', label: 'GDPR Compliance', icon: <Users size={15} /> },
  { id: 'liability', label: 'Limitation of Liability', icon: <AlertTriangle size={15} /> },
]

export default function LegalPages() {
  const activeLegalTab = useStore((s) => s.activeLegalTab) as TabId
  const setActiveLegalTab = useStore((s) => s.setActiveLegalTab)
  const setLegalPageOpen = useStore((s) => s.setLegalPageOpen)
  const isAuthenticated = useStore((s) => s.isAuthenticated)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const handleBack = () => {
    setLegalPageOpen(false)
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <button onClick={handleBack} style={styles.backBtn}>
            <ArrowLeft size={18} />
            <span>{isAuthenticated ? 'Back to chat' : 'Back'}</span>
          </button>
          <div style={styles.headerBrand}>
            <Scale size={18} color="var(--accent)" />
            <span style={styles.headerTitle}>Legal</span>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav style={{
        ...styles.tabNav,
        flexDirection: isMobile ? 'column' : 'row',
        gap: isMobile ? 4 : 0,
        padding: isMobile ? '12px 16px' : '0 32px',
      }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveLegalTab(tab.id)}
            style={{
              ...styles.tab,
              ...(activeLegalTab === tab.id ? styles.tabActive : {}),
              justifyContent: isMobile ? 'flex-start' : 'center',
              borderRadius: isMobile ? 'var(--radius-sm)' : 0,
              borderBottom: isMobile ? 'none' : activeLegalTab === tab.id ? '2px solid var(--accent)' : '2px solid transparent',
              background: isMobile && activeLegalTab === tab.id ? 'var(--accent-dim)' : 'transparent',
            }}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>

      {/* Content */}
      <div style={styles.content}>
        <div style={styles.contentInner}>
          {activeLegalTab === 'terms' && <TermsOfService />}
          {activeLegalTab === 'privacy' && <PrivacyPolicy />}
          {activeLegalTab === 'gdpr' && <GDPRCompliance />}
          {activeLegalTab === 'liability' && <LimitationOfLiability />}
        </div>
      </div>
    </div>
  )
}

/* ==================== TERMS OF SERVICE ==================== */
function TermsOfService() {
  return (
    <article style={styles.article}>
      <h1 style={styles.h1}>Terms of Service</h1>
      <p style={styles.meta}>Last updated: {LAST_UPDATED}</p>

      <Section title="1. Introduction">
        <p>
          Welcome to Themis (&quot;the Platform&quot;), a legal research assistance tool operated by {COMPANY}
          (&quot;we&quot;, &quot;us&quot;, or &quot;our&quot;). By accessing or using the Platform, you agree to be
          bound by these Terms of Service (&quot;Terms&quot;). If you do not agree to these Terms, you must not
          use the Platform.
        </p>
      </Section>

      <Section title="2. Service Description">
        <p>
          Themis is an AI-powered legal research assistance tool that enables users to search across multiple
          legal databases -- including CyLaw (Cyprus legislation and case law), HUDOC (European Court of Human Rights
          case law), and EUR-Lex (European Union legislation) -- and receive AI-generated summaries with source
          citations.
        </p>
        <p style={styles.important}>
          The Platform is a legal research assistance tool. It is NOT a legal advice provider. The outputs generated
          by the Platform do not constitute legal advice, legal opinion, or legal representation. Users must not
          rely on the Platform's outputs as a substitute for professional legal counsel.
        </p>
      </Section>

      <Section title="3. User Accounts">
        <p>
          To access the Platform, you must authenticate using a Google account. By creating an account, you agree to:
        </p>
        <ul style={styles.ul}>
          <li>Provide accurate and current information</li>
          <li>Maintain the confidentiality of your account credentials</li>
          <li>Accept responsibility for all activities that occur under your account</li>
          <li>Notify us immediately of any unauthorized use of your account</li>
        </ul>
      </Section>

      <Section title="4. Acceptable Use">
        <p>You agree to use the Platform solely for lawful legal research purposes. You must NOT:</p>
        <ul style={styles.ul}>
          <li>Rely on the Platform's outputs for legal decisions without independent verification by a qualified legal professional</li>
          <li>Present AI-generated outputs as formal legal advice or legal opinion to third parties</li>
          <li>Use the Platform to engage in unauthorized practice of law</li>
          <li>Attempt to reverse-engineer, decompile, or extract the underlying models or algorithms</li>
          <li>Use automated scripts, bots, or scrapers to access the Platform beyond the provided API</li>
          <li>Upload documents containing malicious code or content that violates applicable laws</li>
          <li>Circumvent rate limits, authentication mechanisms, or other security measures</li>
          <li>Use the Platform in any manner that could damage, disable, or impair our infrastructure</li>
        </ul>
      </Section>

      <Section title="5. Intellectual Property">
        <p>
          The Platform's software, design, and proprietary algorithms are the intellectual property of {COMPANY}.
          Legal documents retrieved from CyLaw, HUDOC, and EUR-Lex remain subject to the terms and licensing
          of their respective publishers. Documents uploaded by users remain the property of the uploading user;
          by uploading, you grant us a limited, non-exclusive license to process and index the content solely
          for the purpose of providing the Platform's services.
        </p>
      </Section>

      <Section title="6. Account Suspension and Termination">
        <p>
          We reserve the right to suspend or terminate your account at our sole discretion if you:
        </p>
        <ul style={styles.ul}>
          <li>Violate these Terms or any applicable laws</li>
          <li>Engage in abusive or excessive use that degrades the Platform for other users</li>
          <li>Fail to pay applicable fees, if any</li>
        </ul>
        <p>
          Upon termination, your right to access the Platform ceases immediately. We may, at our discretion,
          retain or delete your data in accordance with our Privacy Policy and applicable data protection laws.
        </p>
      </Section>

      <Section title="7. Modifications to the Service">
        <p>
          We reserve the right to modify, suspend, or discontinue the Platform (or any part thereof) at any
          time, with or without notice. We shall not be liable to you or any third party for any modification,
          suspension, or discontinuation of the Platform.
        </p>
      </Section>

      <Section title="8. Governing Law and Dispute Resolution">
        <p>
          These Terms are governed by and construed in accordance with the laws of the Republic of Cyprus.
          Any disputes arising from or relating to these Terms or your use of the Platform shall be subject
          to the exclusive jurisdiction of the courts of Nicosia, Republic of Cyprus.
        </p>
        <p>
          Before initiating any legal proceedings, the parties agree to attempt to resolve disputes through
          good-faith negotiation. Either party may initiate arbitration under the rules of the Cyprus
          Arbitration and Mediation Centre if negotiation is unsuccessful.
        </p>
      </Section>

      <Section title="9. Severability">
        <p>
          If any provision of these Terms is held to be unenforceable, the remaining provisions shall
          continue in full force and effect.
        </p>
      </Section>

      <Section title="10. Contact">
        <p>
          For questions regarding these Terms of Service, contact us at: {CONTACT_EMAIL}
        </p>
      </Section>
    </article>
  )
}

/* ==================== PRIVACY POLICY ==================== */
function PrivacyPolicy() {
  return (
    <article style={styles.article}>
      <h1 style={styles.h1}>Privacy Policy</h1>
      <p style={styles.meta}>Last updated: {LAST_UPDATED}</p>

      <Section title="1. Introduction">
        <p>
          This Privacy Policy describes how {COMPANY} (&quot;we&quot;, &quot;us&quot;, or &quot;our&quot;)
          collects, uses, stores, and protects your personal data when you use the Themis legal research
          platform (&quot;the Platform&quot;). We are committed to protecting your privacy in accordance
          with the General Data Protection Regulation (GDPR), the Cyprus Processing of Personal Data
          (Protection of the Individual) Law of 2001 (as amended), and all applicable data protection legislation.
        </p>
      </Section>

      <Section title="2. Data Controller">
        <p>
          The data controller for your personal data is:
        </p>
        <div style={styles.infoBox}>
          <p><strong>{COMPANY}</strong></p>
          <p>[Registered Address]</p>
          <p>Nicosia, Republic of Cyprus</p>
          <p>Email: {CONTACT_EMAIL}</p>
        </div>
      </Section>

      <Section title="3. Data We Collect">
        <p>We collect and process the following categories of personal data:</p>

        <h4 style={styles.h4}>3.1 Account Information</h4>
        <ul style={styles.ul}>
          <li>Name (as provided by Google OAuth)</li>
          <li>Email address</li>
          <li>Profile picture URL</li>
          <li>Google account identifier</li>
        </ul>

        <h4 style={styles.h4}>3.2 Usage Data</h4>
        <ul style={styles.ul}>
          <li>Search queries submitted to the Platform</li>
          <li>Conversation history (queries and AI-generated responses)</li>
          <li>Search filter preferences and settings</li>
          <li>Timestamps and session information</li>
        </ul>

        <h4 style={styles.h4}>3.3 Uploaded Documents</h4>
        <ul style={styles.ul}>
          <li>PDF documents uploaded by users for custom research collections</li>
          <li>Extracted text content and metadata from uploaded documents</li>
          <li>Document embeddings (vector representations used for search)</li>
        </ul>

        <h4 style={styles.h4}>3.4 Technical Data</h4>
        <ul style={styles.ul}>
          <li>IP address and browser information</li>
          <li>API usage metrics (request counts, response times)</li>
        </ul>
      </Section>

      <Section title="4. Purpose and Legal Basis for Processing">
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Purpose</th>
              <th style={styles.th}>Legal Basis</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={styles.td}>Providing the legal research service</td>
              <td style={styles.td}>Performance of a contract (Art. 6(1)(b) GDPR)</td>
            </tr>
            <tr>
              <td style={styles.td}>User authentication via Google OAuth</td>
              <td style={styles.td}>Performance of a contract (Art. 6(1)(b) GDPR)</td>
            </tr>
            <tr>
              <td style={styles.td}>Storing conversation history</td>
              <td style={styles.td}>Legitimate interest (Art. 6(1)(f) GDPR)</td>
            </tr>
            <tr>
              <td style={styles.td}>Processing uploaded documents</td>
              <td style={styles.td}>Consent (Art. 6(1)(a) GDPR)</td>
            </tr>
            <tr>
              <td style={styles.td}>Service improvement and analytics</td>
              <td style={styles.td}>Legitimate interest (Art. 6(1)(f) GDPR)</td>
            </tr>
            <tr>
              <td style={styles.td}>Security and abuse prevention</td>
              <td style={styles.td}>Legitimate interest (Art. 6(1)(f) GDPR)</td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section title="5. Third-Party Data Processors">
        <p>
          We use the following third-party service providers to process data on our behalf.
          All processors are bound by data processing agreements in compliance with GDPR Article 28.
        </p>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Processor</th>
              <th style={styles.th}>Purpose</th>
              <th style={styles.th}>Data Processed</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={styles.td}>NVIDIA (NIM API)</td>
              <td style={styles.td}>AI language model inference for answer generation and query enhancement</td>
              <td style={styles.td}>Search queries, document context snippets</td>
            </tr>
            <tr>
              <td style={styles.td}>Voyage AI</td>
              <td style={styles.td}>Generating document and query embeddings for semantic search</td>
              <td style={styles.td}>Document text, search queries</td>
            </tr>
            <tr>
              <td style={styles.td}>Cohere / ZeroEntropy</td>
              <td style={styles.td}>Search result reranking for relevance optimization</td>
              <td style={styles.td}>Search queries, document snippets</td>
            </tr>
            <tr>
              <td style={styles.td}>Google</td>
              <td style={styles.td}>User authentication (OAuth 2.0)</td>
              <td style={styles.td}>Authentication tokens, basic profile information</td>
            </tr>
            <tr>
              <td style={styles.td}>Neon (PostgreSQL)</td>
              <td style={styles.td}>Database hosting and storage</td>
              <td style={styles.td}>All stored user data, documents, conversations</td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section title="6. Data Retention">
        <ul style={styles.ul}>
          <li><strong>Account data:</strong> Retained for the duration of your active account, and deleted within 30 days of account deletion request</li>
          <li><strong>Conversation history:</strong> Retained for the duration of your active account; users may delete individual conversations at any time</li>
          <li><strong>Uploaded documents:</strong> Retained until deleted by the user or upon account deletion</li>
          <li><strong>Query cache:</strong> Automatically purged after 24 hours</li>
          <li><strong>Technical logs:</strong> Retained for up to 90 days for security and debugging purposes</li>
        </ul>
      </Section>

      <Section title="7. Your Rights Under GDPR">
        <p>As a data subject, you have the following rights:</p>
        <ul style={styles.ul}>
          <li><strong>Right of access</strong> (Art. 15) -- Request a copy of all personal data we hold about you</li>
          <li><strong>Right to rectification</strong> (Art. 16) -- Request correction of inaccurate personal data</li>
          <li><strong>Right to erasure</strong> (Art. 17) -- Request deletion of your personal data</li>
          <li><strong>Right to restriction</strong> (Art. 18) -- Request restricted processing of your data</li>
          <li><strong>Right to data portability</strong> (Art. 20) -- Receive your data in a structured, machine-readable format</li>
          <li><strong>Right to object</strong> (Art. 21) -- Object to processing based on legitimate interest</li>
          <li><strong>Right to withdraw consent</strong> (Art. 7(3)) -- Withdraw consent at any time where processing is based on consent</li>
        </ul>
        <p>
          To exercise any of these rights, contact us at {CONTACT_EMAIL}. We will respond within 30 days
          as required by GDPR.
        </p>
      </Section>

      <Section title="8. Cookies and Local Storage">
        <p>
          The Platform uses browser local storage (not cookies) to maintain your session. Specifically:
        </p>
        <ul style={styles.ul}>
          <li><strong>JWT token:</strong> Stored in localStorage for session authentication</li>
          <li><strong>User profile data:</strong> Stored in localStorage for display purposes</li>
          <li><strong>Preferences:</strong> Search source toggles and settings stored in application state</li>
        </ul>
        <p>
          We do not use third-party tracking cookies. Google OAuth may set its own cookies during the
          authentication flow, governed by Google's Privacy Policy.
        </p>
      </Section>

      <Section title="9. Data Security">
        <p>
          We implement appropriate technical and organizational measures to protect your personal data,
          including:
        </p>
        <ul style={styles.ul}>
          <li>Encrypted connections (TLS/HTTPS) for all data in transit</li>
          <li>Row-Level Security (RLS) in PostgreSQL for strict multi-tenant data isolation</li>
          <li>JWT-based authentication with secure token handling</li>
          <li>Rate limiting to prevent abuse (60 requests per minute per API key)</li>
          <li>CORS restrictions to authorized origins only</li>
        </ul>
      </Section>

      <Section title="10. Contact">
        <p>
          For privacy-related inquiries or to exercise your data protection rights:
        </p>
        <div style={styles.infoBox}>
          <p><strong>Data Protection Officer</strong></p>
          <p>{COMPANY}</p>
          <p>Email: {DPO_EMAIL}</p>
        </div>
      </Section>
    </article>
  )
}

/* ==================== GDPR COMPLIANCE ==================== */
function GDPRCompliance() {
  return (
    <article style={styles.article}>
      <h1 style={styles.h1}>GDPR Compliance Notice</h1>
      <p style={styles.meta}>Last updated: {LAST_UPDATED}</p>

      <Section title="1. Data Controller Identification">
        <div style={styles.infoBox}>
          <p><strong>Data Controller:</strong> {COMPANY}</p>
          <p><strong>Registered Address:</strong> [Registered Address], Nicosia, Republic of Cyprus</p>
          <p><strong>Registration Number:</strong> [Registration Number]</p>
          <p><strong>Contact Email:</strong> {CONTACT_EMAIL}</p>
          <p><strong>Data Protection Officer:</strong> {DPO_EMAIL}</p>
        </div>
      </Section>

      <Section title="2. Legal Basis for Processing">
        <p>We process personal data under the following legal bases as defined in Article 6 of the GDPR:</p>
        <ul style={styles.ul}>
          <li>
            <strong>Consent (Art. 6(1)(a)):</strong> For processing uploaded documents and optional analytics.
            Consent may be withdrawn at any time without affecting the lawfulness of processing prior to withdrawal.
          </li>
          <li>
            <strong>Performance of a Contract (Art. 6(1)(b)):</strong> For providing the legal research service,
            user authentication, and maintaining your account.
          </li>
          <li>
            <strong>Legitimate Interest (Art. 6(1)(f)):</strong> For service improvement, security monitoring,
            fraud prevention, and maintaining conversation history for user convenience. We have conducted a
            legitimate interest assessment for each of these purposes.
          </li>
        </ul>
      </Section>

      <Section title="3. Categories of Personal Data Processed">
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Category</th>
              <th style={styles.th}>Examples</th>
              <th style={styles.th}>Retention Period</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={styles.td}>Identity data</td>
              <td style={styles.td}>Name, email, Google profile ID</td>
              <td style={styles.td}>Duration of account + 30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Usage data</td>
              <td style={styles.td}>Search queries, conversation history</td>
              <td style={styles.td}>Duration of account</td>
            </tr>
            <tr>
              <td style={styles.td}>Content data</td>
              <td style={styles.td}>Uploaded documents, extracted text</td>
              <td style={styles.td}>Until user deletion</td>
            </tr>
            <tr>
              <td style={styles.td}>Technical data</td>
              <td style={styles.td}>IP address, API usage logs</td>
              <td style={styles.td}>90 days</td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section title="4. Cross-Border Data Transfers">
        <p>
          Your personal data may be transferred outside the European Economic Area (EEA) when processed
          by the following third-party service providers:
        </p>
        <ul style={styles.ul}>
          <li><strong>NVIDIA NIM API</strong> -- AI model inference servers (United States)</li>
          <li><strong>Voyage AI</strong> -- Embedding generation (United States)</li>
          <li><strong>Cohere / ZeroEntropy</strong> -- Search result reranking (United States / Canada)</li>
        </ul>
        <p>
          For all cross-border transfers, we ensure appropriate safeguards are in place as required by
          Chapter V of the GDPR, including:
        </p>
        <ul style={styles.ul}>
          <li>Standard Contractual Clauses (SCCs) approved by the European Commission (Art. 46(2)(c))</li>
          <li>Data processing agreements with all sub-processors (Art. 28)</li>
          <li>Assessment of the legal framework of the recipient country</li>
          <li>Supplementary technical measures including encryption in transit and data minimization</li>
        </ul>
      </Section>

      <Section title="5. Data Subject Rights">
        <p>
          Under the GDPR, you have the following rights regarding your personal data. To exercise any of
          these rights, contact our Data Protection Officer at {DPO_EMAIL}.
        </p>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Right</th>
              <th style={styles.th}>Description</th>
              <th style={styles.th}>Response Time</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={styles.td}>Access (Art. 15)</td>
              <td style={styles.td}>Obtain a copy of all personal data we process about you</td>
              <td style={styles.td}>30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Rectification (Art. 16)</td>
              <td style={styles.td}>Correct inaccurate or incomplete personal data</td>
              <td style={styles.td}>30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Erasure (Art. 17)</td>
              <td style={styles.td}>Request deletion of your personal data (&quot;right to be forgotten&quot;)</td>
              <td style={styles.td}>30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Restriction (Art. 18)</td>
              <td style={styles.td}>Restrict processing of your personal data</td>
              <td style={styles.td}>30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Portability (Art. 20)</td>
              <td style={styles.td}>Receive your data in a structured, machine-readable format</td>
              <td style={styles.td}>30 days</td>
            </tr>
            <tr>
              <td style={styles.td}>Objection (Art. 21)</td>
              <td style={styles.td}>Object to processing based on legitimate interest</td>
              <td style={styles.td}>30 days</td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section title="6. Data Breach Procedures">
        <p>
          In the event of a personal data breach, {COMPANY} will:
        </p>
        <ul style={styles.ul}>
          <li>Notify the <strong>Cyprus Commissioner for the Protection of Personal Data</strong> within 72 hours of becoming aware of the breach (Art. 33)</li>
          <li>Notify affected data subjects without undue delay where the breach is likely to result in a high risk to their rights and freedoms (Art. 34)</li>
          <li>Document all breaches, including facts, effects, and remedial actions taken</li>
          <li>Conduct a post-breach review and implement measures to prevent recurrence</li>
        </ul>
      </Section>

      <Section title="7. Data Protection Impact Assessments">
        <p>
          We conduct Data Protection Impact Assessments (DPIAs) as required by Article 35 of the GDPR
          for processing operations that are likely to result in a high risk to data subjects. This
          includes our use of AI models to process legal queries and generate responses.
        </p>
      </Section>

      <Section title="8. Supervisory Authority">
        <p>
          If you believe that our processing of your personal data infringes the GDPR, you have the
          right to lodge a complaint with:
        </p>
        <div style={styles.infoBox}>
          <p><strong>Commissioner for the Protection of Personal Data</strong></p>
          <p>(Epitropos Prostasias Prosopikon Dedomenon)</p>
          <p>1, Iasonos Street, 2nd Floor</p>
          <p>1082 Nicosia, Cyprus</p>
          <p>Phone: +357 22818456</p>
          <p>Fax: +357 22304565</p>
          <p>Email: commissioner@dataprotection.gov.cy</p>
        </div>
      </Section>

      <Section title="9. Contact the Data Protection Officer">
        <p>
          For any questions about this GDPR Compliance Notice or to exercise your data subject rights:
        </p>
        <div style={styles.infoBox}>
          <p><strong>Data Protection Officer</strong></p>
          <p>{COMPANY}</p>
          <p>Email: {DPO_EMAIL}</p>
        </div>
      </Section>
    </article>
  )
}

/* ==================== LIMITATION OF LIABILITY ==================== */
function LimitationOfLiability() {
  return (
    <article style={styles.article}>
      <h1 style={styles.h1}>Limitation of Liability</h1>
      <p style={styles.meta}>Last updated: {LAST_UPDATED}</p>

      <Section title="1. Nature of the Service">
        <p style={styles.important}>
          Themis is a legal research assistance tool. It is NOT a law firm, does NOT provide legal advice,
          and does NOT establish an attorney-client relationship. The Platform uses artificial intelligence
          to search legal databases and generate research summaries, which may contain errors, omissions,
          or outdated information.
        </p>
        <p>
          All outputs generated by the Platform -- including but not limited to AI-generated answers,
          summaries, citations, and document analyses -- are provided solely as research aids.
          They should be independently verified by a qualified legal professional before being used
          for any legal purpose.
        </p>
      </Section>

      <Section title="2. No Guarantee of Accuracy or Completeness">
        <p>
          {COMPANY} makes no warranties, express or implied, regarding:
        </p>
        <ul style={styles.ul}>
          <li>The accuracy, completeness, or currency of AI-generated answers or summaries</li>
          <li>The completeness of search results across CyLaw, HUDOC, or EUR-Lex databases</li>
          <li>The correctness of citations, case references, or legislative references</li>
          <li>The applicability of any legal information to specific facts or circumstances</li>
          <li>The availability or uninterrupted operation of the Platform</li>
        </ul>
        <p>
          Legal databases are maintained by third-party authorities (CyLaw by the Cyprus Bar Association,
          HUDOC by the Council of Europe, EUR-Lex by the EU Publications Office). We do not control
          the content, completeness, or accuracy of these source databases.
        </p>
      </Section>

      <Section title="3. Assumption of Risk">
        <p>
          By using the Platform, you expressly acknowledge and agree that:
        </p>
        <ul style={styles.ul}>
          <li>You use the Platform at your own risk</li>
          <li>You are solely responsible for any decisions made based on the Platform's outputs</li>
          <li>You will not rely on the Platform's outputs without independent verification by a qualified legal professional</li>
          <li>AI-generated content may be inaccurate, incomplete, or misleading</li>
          <li>The Platform is not a substitute for professional legal advice, and use of the Platform does not create any professional-client relationship</li>
        </ul>
      </Section>

      <Section title="4. Limitation of Liability">
        <p>
          To the maximum extent permitted by the laws of the Republic of Cyprus and applicable EU law:
        </p>
        <ul style={styles.ul}>
          <li>
            {COMPANY} shall not be liable for any <strong>indirect, incidental, special, consequential,
            or punitive damages</strong>, including but not limited to loss of profits, data, business
            opportunities, or goodwill, arising from your use of or inability to use the Platform.
          </li>
          <li>
            {COMPANY}'s total aggregate liability for any claims arising from your use of the Platform
            shall not exceed the total amount paid by you to {COMPANY} in the twelve (12) months
            preceding the event giving rise to the claim, or one hundred euros (EUR 100), whichever is greater.
          </li>
          <li>
            {COMPANY} shall not be liable for any loss or damage arising from your reliance on
            AI-generated outputs without independent professional verification.
          </li>
        </ul>
      </Section>

      <Section title="5. Indemnification">
        <p>
          You agree to indemnify, defend, and hold harmless {COMPANY}, its officers, directors, employees,
          agents, and affiliates from and against any and all claims, damages, losses, liabilities, costs,
          and expenses (including reasonable attorneys' fees) arising from or relating to:
        </p>
        <ul style={styles.ul}>
          <li>Your use of or reliance on the Platform's outputs</li>
          <li>Your violation of these Terms or any applicable law</li>
          <li>Your infringement of any third-party rights</li>
          <li>Any content you upload to the Platform</li>
        </ul>
      </Section>

      <Section title="6. Force Majeure">
        <p>
          {COMPANY} shall not be liable for any failure or delay in performing its obligations under
          these Terms where such failure or delay results from circumstances beyond our reasonable
          control, including but not limited to:
        </p>
        <ul style={styles.ul}>
          <li>Natural disasters, epidemics, or pandemics</li>
          <li>Acts of war, terrorism, or civil unrest</li>
          <li>Government actions, sanctions, or regulatory changes</li>
          <li>Failures of third-party service providers (including database operators, cloud infrastructure, or AI model providers)</li>
          <li>Internet or telecommunications failures</li>
          <li>Cyberattacks or security breaches beyond our reasonable control</li>
        </ul>
      </Section>

      <Section title="7. Jurisdictional Limitations">
        <p>
          Some jurisdictions do not allow the exclusion or limitation of certain damages. In such
          jurisdictions, {COMPANY}'s liability shall be limited to the maximum extent permitted by
          applicable law. Nothing in this notice excludes or limits liability that cannot be excluded
          or limited under applicable Cyprus or EU law, including liability for death or personal
          injury caused by negligence, or for fraud or fraudulent misrepresentation.
        </p>
      </Section>

      <Section title="8. Governing Law">
        <p>
          This Limitation of Liability notice is governed by and construed in accordance with the laws
          of the Republic of Cyprus. Any disputes shall be subject to the exclusive jurisdiction of
          the courts of Nicosia, Republic of Cyprus.
        </p>
      </Section>

      <Section title="9. Contact">
        <p>
          For questions regarding this Limitation of Liability notice, contact us at: {CONTACT_EMAIL}
        </p>
      </Section>
    </article>
  )
}

/* ==================== SHARED COMPONENTS ==================== */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={styles.section}>
      <h2 style={styles.h2}>{title}</h2>
      {children}
    </section>
  )
}

/* ==================== STYLES ==================== */
const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    background: 'var(--bg-0)',
  },
  header: {
    height: 56,
    borderBottom: '1px solid var(--border)',
    background: '#FFFFFF',
    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
  },
  headerInner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    padding: '0 24px',
  },
  backBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--accent)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    padding: 0,
  },
  headerBrand: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    fontSize: 15,
    fontWeight: 600,
    color: 'var(--text-1)',
  },
  tabNav: {
    display: 'flex',
    background: '#FFFFFF',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
  },
  tab: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '12px 20px',
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--text-2)',
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    cursor: 'pointer',
    transition: 'color var(--transition), border-color var(--transition)',
    whiteSpace: 'nowrap',
  },
  tabActive: {
    color: 'var(--accent)',
    fontWeight: 600,
  },
  content: {
    flex: 1,
    overflowY: 'auto',
  },
  contentInner: {
    maxWidth: 800,
    margin: '0 auto',
    padding: '32px 24px 64px',
  },
  article: {},
  h1: {
    fontSize: 26,
    fontWeight: 700,
    color: 'var(--text-1)',
    letterSpacing: '-0.02em',
    marginBottom: 4,
  },
  meta: {
    fontSize: 13,
    color: 'var(--text-3)',
    marginBottom: 32,
  },
  section: {
    marginBottom: 28,
  },
  h2: {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text-1)',
    marginBottom: 12,
  },
  h4: {
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--text-1)',
    marginTop: 16,
    marginBottom: 8,
  },
  ul: {
    paddingLeft: 24,
    marginTop: 8,
    marginBottom: 12,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    fontSize: 14,
    lineHeight: 1.7,
    color: 'var(--text-2)',
  },
  important: {
    fontSize: 14,
    lineHeight: 1.7,
    color: 'var(--text-1)',
    fontWeight: 500,
    background: 'var(--accent-dim)',
    padding: '12px 16px',
    borderRadius: 'var(--radius-sm)',
    borderLeft: '3px solid var(--accent)',
    marginTop: 8,
    marginBottom: 12,
  },
  infoBox: {
    background: 'var(--bg-0)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding: '16px 20px',
    marginTop: 8,
    marginBottom: 12,
    fontSize: 14,
    lineHeight: 1.8,
    color: 'var(--text-2)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: 12,
    marginBottom: 12,
    fontSize: 13,
  },
  th: {
    textAlign: 'left',
    padding: '10px 14px',
    background: 'var(--bg-0)',
    borderBottom: '2px solid var(--border)',
    fontWeight: 600,
    color: 'var(--text-1)',
    fontSize: 12,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  td: {
    padding: '10px 14px',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-2)',
    lineHeight: 1.6,
    verticalAlign: 'top',
  },
}
