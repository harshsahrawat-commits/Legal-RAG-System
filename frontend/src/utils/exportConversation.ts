import type { ChatMessage } from '../types'

/**
 * Convert basic markdown to HTML for export.
 * Handles: headings, bold, italic, lists, blockquotes, citation badges, links.
 */
function markdownToHTML(text: string): string {
  const lines = text.split('\n')
  const html: string[] = []
  let inList = false
  let inBlockquote = false

  for (const line of lines) {
    let l = line

    // Close blockquote if we exit one
    if (inBlockquote && !l.startsWith('>')) {
      html.push('</blockquote>')
      inBlockquote = false
    }

    // Close list if line is not a list item
    if (inList && !l.match(/^\s*[-*]\s/)) {
      html.push('</ul>')
      inList = false
    }

    // Headings
    if (l.startsWith('#### ')) {
      html.push(`<h4>${inlineFormat(l.slice(5))}</h4>`)
      continue
    }
    if (l.startsWith('### ')) {
      html.push(`<h3>${inlineFormat(l.slice(4))}</h3>`)
      continue
    }
    if (l.startsWith('## ')) {
      html.push(`<h2>${inlineFormat(l.slice(3))}</h2>`)
      continue
    }
    if (l.startsWith('# ')) {
      html.push(`<h1>${inlineFormat(l.slice(2))}</h1>`)
      continue
    }

    // Blockquotes
    if (l.startsWith('> ')) {
      if (!inBlockquote) {
        html.push('<blockquote>')
        inBlockquote = true
      }
      html.push(`<p>${inlineFormat(l.slice(2))}</p>`)
      continue
    }

    // Unordered list items
    const listMatch = l.match(/^\s*[-*]\s(.+)/)
    if (listMatch) {
      if (!inList) {
        html.push('<ul>')
        inList = true
      }
      html.push(`<li>${inlineFormat(listMatch[1])}</li>`)
      continue
    }

    // Numbered list items
    const numMatch = l.match(/^\s*(\d+)\.\s(.+)/)
    if (numMatch) {
      if (!inList) {
        html.push('<ul>')
        inList = true
      }
      html.push(`<li>${inlineFormat(numMatch[2])}</li>`)
      continue
    }

    // Empty line
    if (l.trim() === '') {
      html.push('<br/>')
      continue
    }

    // Regular paragraph
    html.push(`<p>${inlineFormat(l)}</p>`)
  }

  if (inList) html.push('</ul>')
  if (inBlockquote) html.push('</blockquote>')

  return html.join('\n')
}

/** Inline formatting: bold, italic, citation badges, links */
function inlineFormat(text: string): string {
  let s = text
  // Escape HTML
  s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Italic
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>')
  // Citation badges [N]
  s = s.replace(/\[(\d+)\]/g, '<span class="cite-badge">[$1]</span>')
  return s
}

function escapeHTML(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function originLabel(origin?: string | null): string {
  if (!origin) return 'Unknown'
  const map: Record<string, string> = { cylaw: 'CyLaw', hudoc: 'HUDOC', eurlex: 'EUR-Lex', user: 'My Docs', session: 'Upload' }
  return map[origin] || origin
}

export function generateExportHTML(title: string, messages: ChatMessage[]): string {
  const now = new Date()
  const dateStr = now.toLocaleDateString('en-GB', {
    year: 'numeric', month: 'long', day: 'numeric',
  })
  const timeStr = now.toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit',
  })

  // Build exchanges
  let exchangesHTML = ''
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]

    if (msg.role === 'user') {
      exchangesHTML += `
        <div class="exchange">
          <div class="question">
            <div class="q-label">Q:</div>
            <div class="q-text">${escapeHTML(msg.content)}</div>
          </div>`
    }

    if (msg.role === 'assistant') {
      exchangesHTML += `
          <div class="answer">
            <div class="answer-body">${markdownToHTML(msg.content)}</div>`

      // Sources table
      if (msg.sources && msg.sources.length > 0) {
        exchangesHTML += `
            <div class="sources-section">
              <div class="sources-title">Sources</div>
              <table class="sources-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Origin</th>
                    <th>Title</th>
                    <th>Relevance</th>
                    <th>Citation</th>
                  </tr>
                </thead>
                <tbody>`

        msg.sources.forEach((src, idx) => {
          const relevance = src.relevance_score != null
            ? `${Math.round(src.relevance_score * 100)}%`
            : '—'
          exchangesHTML += `
                  <tr>
                    <td>${idx + 1}</td>
                    <td><span class="origin-badge origin-${src.source_origin || 'unknown'}">${originLabel(src.source_origin)}</span></td>
                    <td>${escapeHTML(src.document_title || '—')}</td>
                    <td>${relevance}</td>
                    <td>${escapeHTML(src.short_citation || '—')}</td>
                  </tr>`
        })

        exchangesHTML += `
                </tbody>
              </table>
            </div>`
      }

      // Latency
      if (msg.latency_ms) {
        exchangesHTML += `
            <div class="latency">Response time: ${(msg.latency_ms / 1000).toFixed(1)}s</div>`
      }

      exchangesHTML += `
          </div>
        </div>`
    }
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Themis — ${escapeHTML(title)}</title>
<style>
  @page {
    margin: 2cm;
    size: A4;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: Georgia, 'Times New Roman', serif;
    color: #1a1a1a;
    line-height: 1.7;
    max-width: 800px;
    margin: 0 auto;
    padding: 40px 24px;
    background: #fff;
  }

  h1, h2, h3, h4 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    line-height: 1.3;
    margin-top: 16px;
    margin-bottom: 8px;
  }

  h1 { font-size: 22px; }
  h2 { font-size: 18px; }
  h3 { font-size: 16px; }
  h4 { font-size: 14px; }

  p { margin-bottom: 8px; font-size: 14px; }
  ul { padding-left: 24px; margin-bottom: 8px; }
  li { font-size: 14px; margin-bottom: 4px; }
  blockquote {
    border-left: 3px solid #999;
    padding-left: 16px;
    margin: 8px 0;
    color: #555;
  }

  strong { font-weight: 700; }
  em { font-style: italic; }

  .header {
    text-align: center;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 20px;
    margin-bottom: 32px;
  }
  .header .brand {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #666;
    margin-bottom: 8px;
  }
  .header .title {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 4px;
  }
  .header .meta {
    font-size: 13px;
    color: #888;
  }

  .exchange {
    margin-bottom: 32px;
    page-break-inside: avoid;
  }

  .question {
    display: flex;
    gap: 12px;
    background: #f5f5f2;
    border-left: 4px solid #2563eb;
    padding: 16px 20px;
    margin-bottom: 16px;
    border-radius: 0 6px 6px 0;
  }
  .q-label {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-weight: 700;
    font-size: 14px;
    color: #2563eb;
    flex-shrink: 0;
  }
  .q-text {
    font-size: 14px;
    font-weight: 600;
    color: #1a1a1a;
  }

  .answer {
    padding: 0 0 0 20px;
  }
  .answer-body {
    margin-bottom: 12px;
  }

  .cite-badge {
    display: inline-block;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #2563eb;
    background: #eef2ff;
    padding: 1px 5px;
    border-radius: 3px;
    vertical-align: super;
  }

  .sources-section {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #e5e5e5;
  }
  .sources-title {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #888;
    margin-bottom: 8px;
  }
  .sources-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  }
  .sources-table th {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 2px solid #ddd;
    font-weight: 600;
    color: #555;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .sources-table td {
    padding: 6px 8px;
    border-bottom: 1px solid #eee;
    color: #333;
    vertical-align: top;
  }

  .origin-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .origin-cylaw { background: #e0f2fe; color: #0369a1; }
  .origin-hudoc { background: #fef3c7; color: #92400e; }
  .origin-eurlex { background: #dbeafe; color: #1e40af; }
  .origin-user { background: #f0e6f5; color: #6e3a8c; }
  .origin-session { background: #f5ede6; color: #8c5a3a; }
  .origin-unknown { background: #f3f4f6; color: #6b7280; }

  .latency {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 11px;
    color: #aaa;
    margin-top: 8px;
  }

  .disclaimer {
    margin-top: 48px;
    padding-top: 20px;
    border-top: 2px solid #1a1a1a;
    font-size: 12px;
    color: #888;
    line-height: 1.6;
  }
  .disclaimer strong {
    color: #666;
  }

  @media print {
    body { padding: 0; max-width: none; }
    .exchange { page-break-inside: avoid; }
    .header { page-break-after: avoid; }
    .disclaimer { page-break-before: avoid; }
  }
</style>
</head>
<body>
  <div class="header">
    <div class="brand">Themis &mdash; Legal Research Report</div>
    <div class="title">${escapeHTML(title)}</div>
    <div class="meta">Exported on ${dateStr} at ${timeStr}</div>
  </div>

  ${exchangesHTML}

  <div class="disclaimer">
    <strong>Legal Disclaimer:</strong> This document was generated by Themis, an AI-powered legal research
    assistance tool. The content herein does not constitute legal advice, legal opinion, or legal
    representation. AI-generated answers may contain errors, omissions, or outdated information. All
    information should be independently verified by a qualified legal professional before being relied
    upon for any legal purpose. Use of this document does not establish an attorney-client relationship.
  </div>
</body>
</html>`
}
