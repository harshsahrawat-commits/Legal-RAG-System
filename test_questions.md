# Legal RAG — Test Questions

Comprehensive test suite covering every aspect of the retrieval and answer pipeline.
Each question targets a specific dimension. Run with all sources enabled unless noted.

---

## 1. Query Classification

Tests that the classifier routes queries to the correct pipeline tier.

### Simple (<=4 words, no question words → skips all enhancement)

| # | Query | Expected |
|---|-------|----------|
| 1.1 | `GDPR Article 17` | Simple: no expansion, no HyDE, no multi-query |
| 1.2 | `right to privacy` | Simple: fast response, minimal latency |
| 1.3 | `asylum seeker rights` | Simple: 3 words, direct vector+keyword search |
| 1.4 | `CAP154` | Simple: Cypriot statute reference, direct lookup |

### Factual (Who/What/When/Where → expansion only)

| # | Query | Expected |
|---|-------|----------|
| 1.5 | `What is the right to be forgotten under GDPR?` | Factual: query expansion runs, no HyDE |
| 1.6 | `Who can file a complaint with the ECHR?` | Factual: expansion adds legal synonyms |
| 1.7 | `When did the General Data Protection Regulation come into force?` | Factual: date-specific, expansion enriches |
| 1.8 | `List the grounds for asylum under EU law` | Factual: "list" trigger word |

### Analytical (Explain/Compare/Why → full enhancement pipeline)

| # | Query | Expected |
|---|-------|----------|
| 1.9 | `Explain the proportionality test applied by the ECHR in freedom of expression cases` | Analytical: expansion + HyDE + multi-query |
| 1.10 | `Compare the data protection frameworks of Cyprus and the EU` | Analytical: "compare" trigger, cross-source |
| 1.11 | `Why did the ECHR find a violation of Article 8 in surveillance cases?` | Analytical: "why" trigger, full pipeline |
| 1.12 | `What are the implications of the CJEU Schrems II ruling for data transfers?` | Analytical: "implications" keyword |

### Standard (Legal references, default fallback → expansion + multi-query)

| # | Query | Expected |
|---|-------|----------|
| 1.13 | `Article 6 of the European Convention on Human Rights fair trial guarantees` | Standard: "article" trigger |
| 1.14 | `Obligations of data controllers under Section 5 of the Cyprus Data Protection Law` | Standard: "section" trigger |
| 1.15 | `How does the principle of subsidiarity apply in EU legislative competence?` | Standard: default fallback |

---

## 2. Source Types

Tests that each source database returns correct results with proper metadata.

### CyLaw (Cypriot Legislation)

| # | Query | Verify |
|---|-------|--------|
| 2.1 | `What are the penalties for tax evasion under Cyprus law?` | Sources have `source_origin: cylaw`, CyLaw URLs work |
| 2.2 | `Employment termination notice period in Cyprus` | Origin badge shows "CyLaw", View button links to cylaw.org |
| 2.3 | `Ο περί Εταιρειών Νόμος` | Greek query returns CyLaw results with correct titles |

### HUDOC (ECHR Case Law)

| # | Query | Verify |
|---|-------|--------|
| 2.4 | `ECHR case law on inhuman or degrading treatment Article 3` | Sources have `source_origin: hudoc`, HUDOC URLs work |
| 2.5 | `Right to a fair trial violations in criminal proceedings` | Origin badge shows "ECHR", external_url points to hudoc.echr.coe.int |
| 2.6 | `Margin of appreciation doctrine in religious freedom cases` | Multiple HUDOC sources with distinct case names |

### EUR-Lex (EU Legislation)

| # | Query | Verify |
|---|-------|--------|
| 2.7 | `GDPR data breach notification requirements` | Sources have `source_origin: eurlex`, EUR-Lex URLs work |
| 2.8 | `EU Consumer Rights Directive withdrawal period` | Origin badge shows "EU Law", external_url points to eur-lex.europa.eu |
| 2.9 | `Regulation 2016/679 lawful basis for processing` | CELEX-based URL generation |

### Cross-Source (answer draws from multiple databases)

| # | Query | Verify |
|---|-------|--------|
| 2.10 | `How does Cyprus implement GDPR data protection requirements?` | Mix of CyLaw + EUR-Lex sources, answer cites both |
| 2.11 | `Compare ECHR privacy rights with EU data protection law` | Mix of HUDOC + EUR-Lex, analytical pipeline |
| 2.12 | `What protections exist for asylum seekers under Cypriot and EU law?` | CyLaw + EUR-Lex + possibly HUDOC |

---

## 3. Source Toggles

Test with specific sources disabled. Toggle via the filter button in the input bar.

| # | Toggle State | Query | Verify |
|---|-------------|-------|--------|
| 3.1 | CyLaw OFF, HUDOC ON, EUR-Lex ON | `Data protection rights` | No CyLaw sources in results |
| 3.2 | CyLaw ON, HUDOC OFF, EUR-Lex OFF | `Right to privacy` | Only CyLaw sources returned |
| 3.3 | CyLaw OFF, HUDOC OFF, EUR-Lex ON | `Consumer protection regulations` | Only EUR-Lex sources |
| 3.4 | All OFF | `Any query` | Error message: "No sources are enabled..." |
| 3.5 | All ON (default) | `Fair trial rights` | Mix of all three source types |

---

## 4. Language Support

### English Queries

| # | Query | Verify |
|---|-------|--------|
| 4.1 | `What constitutes unfair dismissal under Cyprus employment law?` | English answer, proper legal prose |
| 4.2 | `Explain the exhaustion of domestic remedies requirement before the ECHR` | English analytical answer |

### Greek Queries

| # | Query | Verify |
|---|-------|--------|
| 4.3 | `Ποια είναι τα δικαιώματα του κατηγορουμένου σύμφωνα με το κυπριακό δίκαιο;` | Greek answer with [N] citations |
| 4.4 | `Εξηγήστε την αρχή της αναλογικότητας στο ευρωπαϊκό δίκαιο` | Greek analytical — should be capped to "standard" (no HyDE) |
| 4.5 | `Τι είναι το GDPR;` | Greek factual, simple question |

### Cross-Lingual Retrieval

| # | Query | Verify |
|---|-------|--------|
| 4.6 | `What does Cypriot law say about company registration?` | English query retrieves Greek CyLaw docs via cross-lingual translation |
| 4.7 | `Ποιες είναι οι απαιτήσεις του GDPR για τη γνωστοποίηση παραβίασης δεδομένων;` | Greek query retrieves English EUR-Lex docs |
| 4.8 | `Explain Article 8 ECHR case law on surveillance` (CyLaw ON only) | English query, CyLaw sources in Greek — triggers EN→EL translation |

---

## 5. Answer Format & LLM Output

Verify the updated prompts produce clean answers without legacy formatting.

| # | Query | Verify |
|---|-------|--------|
| 5.1 | `What are the key principles of GDPR?` | NO `---` separator in answer |
| 5.2 | `Explain the right to asylum under EU law` | NO "Supporting Evidence:" section |
| 5.3 | `Who can file a case at the ECHR?` | NO "Answer:" or "Summary:" header at start |
| 5.4 | `Compare ECHR and CJEU approaches to privacy` | NO asterisks (**bold**) in answer |
| 5.5 | `Τι προβλέπει ο νόμος για την προστασία δεδομένων;` | NO "Υποστηρικτικά Στοιχεία:" section (Greek) |
| 5.6 | Any query with multiple sources | Every claim has [N] citation, multi-paragraph flowing text |
| 5.7 | `What is the capital of Mars?` | "The provided documents do not contain this information." |

---

## 6. Inline Sources UI

Test the new collapsible source display below each assistant message.

| # | Action | Verify |
|---|--------|--------|
| 6.1 | Ask any question that returns sources | Sources section appears below bubble, COLLAPSED by default |
| 6.2 | Click the "Sources . N" toggle | Section expands, all sources listed |
| 6.3 | Check each source row | Shows: [N] number, title (linked), origin badge, short citation, relevance bar, View button |
| 6.4 | Check origin badges | CyLaw = cyan, ECHR = purple, EU Law = amber |
| 6.5 | Click a [N] citation badge in the answer text | Sources auto-expand, scroll to that source, highlight it |
| 6.6 | Click a different [N] citation | Active highlight moves to the new source |
| 6.7 | Click "View" button on a CyLaw source | Opens cylaw.org URL in new tab |
| 6.8 | Click "View" button on a HUDOC source | Opens hudoc.echr.coe.int URL in new tab |
| 6.9 | Click "View" button on a EUR-Lex source | Opens eur-lex.europa.eu URL in new tab |
| 6.10 | Click source title link | Same external URL opens in new tab |
| 6.11 | Collapse sources, ask another question | New message also has its own collapsed sources section |
| 6.12 | Hover over [N] citation badge | Tooltip shows source origin label, title, content preview |
| 6.13 | Verify NO side panel appears | Old SourcePanel is fully removed, no slide-out drawer |

---

## 7. Streaming & Progressive Rendering

| # | Action | Verify |
|---|--------|--------|
| 7.1 | Submit a query | Thinking indicator appears with rotating status messages |
| 7.2 | Wait for sources SSE event | "N sources found" badge appears in thinking UI |
| 7.3 | Wait for first token | Thinking fades out, answer text starts streaming in |
| 7.4 | Wait for done event | Latency shown (e.g., "12.3s"), source count in meta row |
| 7.5 | Submit during existing stream | Should be blocked (send button disabled while loading) |

---

## 8. Conversation System

| # | Action | Verify |
|---|--------|--------|
| 8.1 | Ask first question without selecting a conversation | New conversation auto-created, appears in sidebar |
| 8.2 | Check sidebar after first query | Conversation title = first query truncated to 80 chars |
| 8.3 | Ask follow-up question | Same conversation_id used, messages appended |
| 8.4 | Click a different conversation in sidebar | Messages load from that conversation |
| 8.5 | Click "New Chat" button | Messages cleared, new conversation starts on next query |
| 8.6 | Refresh the page | Conversations persist in sidebar, last active loads |

---

## 9. Document Management

| # | Action | Verify |
|---|--------|--------|
| 9.1 | Upload a PDF via chat (+) button | File attached to current conversation as session doc |
| 9.2 | Ask a question about the uploaded PDF | Results include chunks from the uploaded document |
| 9.3 | Upload a PDF via Settings > Documents | Document persisted in a family |
| 9.4 | Toggle a family on in source filters | Family docs included in search results |
| 9.5 | Delete a document in Settings | Document removed, no longer appears in search |

---

## 10. Edge Cases & Error Handling

| # | Query / Action | Verify |
|---|---------------|--------|
| 10.1 | `asdfghjkl zxcvbnm` | No relevant results found message, no crash |
| 10.2 | Very long query (500+ chars of legal text) | Handled gracefully, possibly truncated |
| 10.3 | Query with special characters: `Article 6(1) — "fair trial"` | Parsed correctly, no regex errors |
| 10.4 | Query with only numbers: `2016/679` | Returns GDPR regulation results |
| 10.5 | Mixed language query: `What is the Ο περί Εταιρειών Νόμος?` | Language detection handles mixed content |
| 10.6 | Rapid-fire 10 queries in succession | Rate limiter kicks in at 60/min, returns 429 |
| 10.7 | Submit empty query | Send button disabled, nothing happens |
| 10.8 | Network disconnect during streaming | Error event displayed in chat |

---

## 11. Caching Behavior

| # | Action | Verify |
|---|--------|--------|
| 11.1 | Ask the same question twice | Second response noticeably faster (cache hit) |
| 11.2 | Ask same question with different source toggles | Different results (cache key includes toggles) |
| 11.3 | Upload a new document, repeat a cached query | Cache invalidated, fresh results include new doc |
| 11.4 | Rephrase a question slightly | May still hit semantic cache (cosine >= 0.92) |

---

## 12. Performance Benchmarks

Rough latency expectations. Measure with browser DevTools or the latency shown in the UI.

| # | Query Type | Expected Latency Range |
|---|-----------|----------------------|
| 12.1 | Simple query (cache miss) | 3–8s |
| 12.2 | Factual query (expansion only) | 5–12s |
| 12.3 | Standard query (expansion + multi-query) | 8–18s |
| 12.4 | Analytical query (full pipeline) | 12–25s |
| 12.5 | Any query (cache hit) | < 2s |
| 12.6 | Cross-lingual query | +3–5s over base type |

---

## 13. Citation Accuracy

| # | Action | Verify |
|---|--------|--------|
| 13.1 | Read answer text, note [1] claim | Expand sources, source [1] content supports the claim |
| 13.2 | Check [N] for N > source count | Badge shows as invalid (greyed out, not clickable) |
| 13.3 | Answer with [1, 2] combined citation | Both badges render, both sources are relevant |
| 13.4 | Answer with range [3-5] | Expands to three separate badges: [3] [4] [5] |
| 13.5 | Verify every claim cites a source | No unsupported assertions in the answer |

---

## Quick Smoke Test (5 questions, covers all dimensions)

Run these 5 in order for a fast end-to-end validation:

1. **Simple + CyLaw**: `Cyprus company law` → fast, CyLaw sources, cyan badges
2. **Factual + HUDOC**: `What is the margin of appreciation doctrine?` → ECHR sources, purple badges
3. **Analytical + EUR-Lex**: `Explain the legal basis for GDPR extraterritorial application` → EU Law sources, amber badges, full pipeline latency
4. **Greek + Cross-lingual**: `Ποια είναι τα δικαιώματα του υποκειμένου δεδομένων;` → Greek answer, may pull from EUR-Lex via translation
5. **Cross-source**: `How does Cyprus implement EU asylum directives in light of ECHR case law?` → All three source types, multi-paragraph answer, no Supporting Evidence section
