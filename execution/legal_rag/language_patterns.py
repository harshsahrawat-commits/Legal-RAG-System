"""
Multilingual Pattern Definitions for Legal RAG

All regex patterns, prompt templates, and UI labels organized by language.
Modules import from here instead of defining patterns inline.
"""

import re

# =============================================================================
# Document Type Detection Patterns
# =============================================================================

DOCTYPE_PATTERNS = {
    "en": {
        "contract": [
            r"(?i)agreement|contract|terms and conditions|parties agree",
            r"(?i)witnesseth|whereas|now therefore",
            r"(?i)executed as of|effective date",
        ],
        "statute": [
            r"(?i)enacted by|legislature|public law",
            r"(?i)be it enacted|section \d+\.",
            r"(?i)code of|statutes at large",
        ],
        "case_law": [
            r"(?i)plaintiff|defendant|court|judge",
            r"(?i)opinion|held|judgment|ruling",
            r"(?i)appeal|appellant|appellee",
        ],
        "regulation": [
            r"(?i)regulation|rule|agency|federal register",
            r"(?i)promulgated|pursuant to",
            r"(?i)cfr|code of federal regulations",
        ],
    },
    "el": {
        "contract": [
            r"(?i)συμφωνία|σύμβαση|όροι και προϋποθέσεις|τα μέρη συμφωνούν",
            r"(?i)συμφωνήθηκαν|ενώ|λαμβάνοντας υπόψη",
            r"(?i)ημερομηνία έναρξης|υπεγράφη",
        ],
        "statute": [
            r"(?i)ψηφίστηκε|νομοθεσία|δημόσιος νόμος",
            r"(?i)νόμος|διάταγμα|προεδρικό διάταγμα",
            r"(?i)κώδικας|φεκ|εφημερίδα της κυβερνήσεως",
        ],
        "case_law": [
            r"(?i)ενάγων|εναγόμενος|δικαστήριο|δικαστής",
            r"(?i)απόφαση|κρίθηκε|γνωμοδότηση",
            r"(?i)έφεση|εκκαλών|εφεσίβλητος",
        ],
        "regulation": [
            r"(?i)κανονισμός|κανόνας|αρχή|ρυθμιστική",
            r"(?i)εκδόθηκε|σύμφωνα με",
            r"(?i)υπουργική απόφαση",
        ],
    },
}

# =============================================================================
# Section Detection Patterns (for document structure parsing)
# =============================================================================

SECTION_PATTERNS = {
    "en": {
        "part": re.compile(r"^(?:PART|Part)\s+([IVXLCDM]+|\d+)[.:\s]", re.MULTILINE),
        "chapter": re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)[.:\s]", re.MULTILINE),
        "section": re.compile(r"^(?:SECTION|Section|§)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
        "article": re.compile(r"^(?:ARTICLE|Article)\s+(\d+|[IVXLCDM]+)[.:\s]", re.MULTILINE),
        "clause": re.compile(r"^(?:Clause|\d+\.)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
    },
    "el": {
        "part": re.compile(r"^(?:ΜΕΡΟΣ|Μέρος)\s+([IVXLCDM]+|\d+|[Α-Ω])[.:\s]", re.MULTILINE),
        "chapter": re.compile(r"^(?:ΚΕΦΑΛΑΙΟ|Κεφάλαιο)\s+(\d+|[IVXLCDM]+|[Α-Ω])[.:\s]", re.MULTILINE),
        "section": re.compile(r"^(?:ΤΜΗΜΑ|Τμήμα)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
        "article": re.compile(r"^(?:ΑΡΘΡΟ|Άρθρο)\s+(\d+|[IVXLCDM]+)[.:\s]", re.MULTILINE),
        "clause": re.compile(r"^(?:Ρήτρα|Όρος)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
    },
}

# =============================================================================
# Chunker Section Markers (for chunk boundary detection)
# =============================================================================

SECTION_MARKERS = {
    "en": [
        r"^(?:ARTICLE|Article)\s+[IVXLCDM\d]+",
        r"^(?:SECTION|Section|§)\s*\d+",
        r"^(?:PART|Part)\s+[IVXLCDM\d]+",
        r"^(?:CHAPTER|Chapter)\s+\d+",
        r"^\d+\.\s+[A-Z]",
        r"^\([a-z]\)\s+",
        r"^\(\d+\)\s+",
    ],
    "el": [
        r"^(?:ΑΡΘΡΟ|Άρθρο)\s+[IVXLCDM\d]+",
        r"^(?:ΤΜΗΜΑ|Τμήμα)\s*\d+",
        r"^(?:ΜΕΡΟΣ|Μέρος)\s+[IVXLCDM\d]+",
        r"^(?:ΚΕΦΑΛΑΙΟ|Κεφάλαιο)\s+\d+",
        r"^\d+\.\s+[Α-Ω]",
        r"^\([α-ω]\)\s+",
        r"^\(\d+\)\s+",
    ],
}

# =============================================================================
# Legal Cross-Reference Patterns (for chunker metadata extraction)
# =============================================================================

REFERENCE_PATTERNS = {
    "en": [
        r"(?:Section|§)\s*\d+(?:\.\d+)*",
        r"(?:Article)\s+[IVXLCDM]+",
        r"(?:Clause)\s+\d+(?:\.\d+)*",
        r"(?:paragraph|para\.)\s*\([a-z]\)",
    ],
    "el": [
        r"(?:Άρθρο|Αρθ\.)\s*\d+(?:\.\d+)*",
        r"(?:Τμήμα)\s+[IVXLCDM\d]+",
        r"(?:Ρήτρα|Όρος)\s+\d+(?:\.\d+)*",
        r"(?:παράγραφος|παρ\.)\s*\([α-ω]\)",
    ],
}

# =============================================================================
# Definition Markers (for extracting defined terms from chunks)
# =============================================================================

DEFINITION_MARKERS = {
    "en": [
        r'"([A-Z][A-Za-z\s]+)"',
        r"'([A-Z][A-Za-z\s]+)'",
        r'"([A-Z][A-Za-z\s]+)"\s+means',
        r"(?:defined as|shall mean|refers to)\s+['\"]([^'\"]+)['\"]",
    ],
    "el": [
        r'"([Α-Ω][Α-Ωα-ωάέήίόύώΆΈΉΊΌΎΏ\s]+)"',
        r"«([Α-Ω][Α-Ωα-ωάέήίόύώΆΈΉΊΌΎΏ\s]+)»",
        r'"([Α-Ω][Α-Ωα-ωάέήίόύώΆΈΉΊΌΎΏ\s]+)"\s+σημαίνει',
        r"(?:ορίζεται ως|σημαίνει|αναφέρεται σε)\s+[«\"]([^»\"]+)[»\"]",
    ],
}

# =============================================================================
# Date Patterns (for document metadata extraction)
# =============================================================================

DATE_PATTERNS = {
    "en": {
        "patterns": [
            r'(?i)effective\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)dated\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)this\s+(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+\w+,?\s+\d{4})',
        ],
        "formats": ["%B %d, %Y", "%B %d %Y", "%d day of %B, %Y", "%d day of %B %Y"],
    },
    "el": {
        "patterns": [
            r'(?i)ημερομηνία\s+(\d{1,2}\s+\w+\s+\d{4})',
            r'(?i)υπεγράφη\s+(?:στις\s+)?(\d{1,2}\s+\w+\s+\d{4})',
            r'(?i)(\d{1,2}\s+\w+\s+\d{4})',
        ],
        "month_map": {
            "ιανουαρίου": "01", "φεβρουαρίου": "02", "μαρτίου": "03",
            "απριλίου": "04", "μαΐου": "05", "ιουνίου": "06",
            "ιουλίου": "07", "αυγούστου": "08", "σεπτεμβρίου": "09",
            "οκτωβρίου": "10", "νοεμβρίου": "11", "δεκεμβρίου": "12",
        },
    },
}

# =============================================================================
# Party Extraction Patterns (for contract metadata)
# =============================================================================

PARTY_PATTERNS = {
    "en": [
        r'(?i)(?:between|by and between)\s+([A-Z][A-Za-z\s,\.]+?)(?:\s+\("|\s+and\s+)',
        r'(?i)"([A-Z][A-Za-z]+)"\s*(?:,\s*)?(?:a|an|the)',
        r'(?i)(?:hereinafter|referred to as)\s+"([A-Z][A-Za-z]+)"',
    ],
    "el": [
        r'(?i)(?:μεταξύ|ανάμεσα σε)\s+(.+?)(?:\s+\(«|\s+και\s+)',
        r'(?i)«([Α-Ω][Α-Ωα-ωάέήίόύώ\s]+)»\s*(?:,\s*)?(?:η|ο|το)',
        r'(?i)(?:εφεξής|αναφερόμεν[ηος] ως)\s+«([^»]+)»',
    ],
}

# =============================================================================
# Jurisdiction Patterns (for document metadata)
# =============================================================================

JURISDICTION_PATTERNS = {
    "en": [
        r"(?i)state of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:court|district|circuit)",
        r"(?i)laws of (?:the state of )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?i)governed by (?:the laws of )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ],
    "el": [
        r"(?i)δίκαιο\s+(?:της\s+)?(.+?)(?:\.|,|\s+και)",
        r"(?i)νομοθεσία\s+(?:της\s+)?(.+?)(?:\.|,)",
        r"(?i)(?:δικαστήριο|πρωτοδικείο|εφετείο)\s+(.+?)(?:\.|,)",
        r"(?i)αρμόδι[οα]\s+(?:δικαστήρι[οα]\s+)?(?:της\s+)?(.+?)(?:\.|,)",
    ],
}

# =============================================================================
# Query Classification Patterns (for retriever adaptive pipeline)
# =============================================================================

QUERY_CLASSIFICATION = {
    "en": {
        "analytical": [
            r'^(explain|analyze|compare|contrast|evaluate|assess)\b',
            r'^why\s+',
            r'\b(implications?|consequences?|impact|effect)\b',
            r'\b(relationship|difference|similarity)\s+(between|among)\b',
            r'\b(pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b',
        ],
        "factual": [
            r'^(what|who|when|where)\s+(is|are|was|were)\b',
            r'^(what|who)\s+(did|does|do)\b',
            r'^(list|name|identify)\s+',
            r'\b(how\s+many|how\s+much)\b',
            r'^what\s+are\s+the\b',
        ],
        "legal_reference": r'section|article|clause|paragraph|¶',
        "question_starters": ['what', 'who', 'when', 'where', 'how', 'why', 'which', 'list', 'explain'],
    },
    "el": {
        "analytical": [
            r'^(εξηγήστε|αναλύστε|συγκρίνετε|αξιολογήστε)\b',
            r'^γιατί\s+',
            r'\b(επιπτώσεις|συνέπειες|αντίκτυπο|αποτέλεσμα)\b',
            r'\b(σχέση|διαφορά|ομοιότητα)\s+(μεταξύ|ανάμεσα)\b',
            r'\b(πλεονεκτήματα\s+και\s+μειονεκτήματα)\b',
        ],
        "factual": [
            r'^(τι|ποιος|πότε|πού)\s+(είναι|ήταν)\b',
            r'^(τι|ποιος)\s+(έκανε|κάνει)\b',
            r'^(αναφέρατε|ονομάστε|προσδιορίστε)\s+',
            r'\b(πόσ[οαε]ι?|πόσ[οαε])\b',
            r'^ποια\s+είναι\s+τα\b',
        ],
        "legal_reference": r'άρθρο|τμήμα|ρήτρα|παράγραφος|¶',
        "question_starters": ['τι', 'ποιος', 'πότε', 'πού', 'πώς', 'γιατί', 'ποιο', 'αναφέρατε', 'εξηγήστε'],
    },
}

# =============================================================================
# Paragraph Reference Patterns (for retriever paragraph extraction)
# =============================================================================

PARAGRAPH_REFERENCE_PATTERNS = {
    "en": [
        r'\bparagraph\s*#?\s*(\d+)\b',
        r'\bpara\.?\s*#?\s*(\d+)\b',
        r'¶\s*(\d+)',
        r'\bp\.\s*(\d+)\b',
    ],
    "el": [
        r'\bπαράγραφος\s*#?\s*(\d+)\b',
        r'\bπαρ\.?\s*#?\s*(\d+)\b',
        r'¶\s*(\d+)',
    ],
}

# =============================================================================
# Citation Section Patterns (for citation extractor)
# =============================================================================

CITATION_SECTION_PATTERNS = {
    "en": [
        (r"Article[_\s]+([IVXLCDM]+|\d+)", "Article {}"),
        (r"Section[_\s]+(\d+(?:\.\d+)*)", "Section {}"),
        (r"Clause[_\s]+(\d+(?:\.\d+)*)", "Clause {}"),
        (r"Part[_\s]+([IVXLCDM]+|\d+)", "Part {}"),
        (r"Chapter[_\s]+(\d+)", "Chapter {}"),
    ],
    "el": [
        (r"Άρθρο[_\s]+([IVXLCDM]+|\d+)", "Άρθρο {}"),
        (r"Τμήμα[_\s]+(\d+(?:\.\d+)*)", "Τμήμα {}"),
        (r"Ρήτρα[_\s]+(\d+(?:\.\d+)*)", "Ρήτρα {}"),
        (r"Μέρος[_\s]+([IVXLCDM]+|\d+)", "Μέρος {}"),
        (r"Κεφάλαιο[_\s]+(\d+)", "Κεφάλαιο {}"),
    ],
}

# =============================================================================
# UI Labels and Formatting
# =============================================================================

LABELS = {
    "en": {
        "document_summary": "DOCUMENT SUMMARY:",
        "sources": "Sources:",
        "page_single": "p.",
        "page_range": "pp.",
        "page_na": "Page N/A",
    },
    "el": {
        "document_summary": "ΠΕΡΙΛΗΨΗ ΕΓΓΡΑΦΟΥ:",
        "sources": "Πηγές:",
        "page_single": "σελ.",
        "page_range": "σσ.",
        "page_na": "Σελ. Μ/Δ",
    },
}

# =============================================================================
# LLM Prompt Templates
# =============================================================================

LLM_PROMPTS = {
    "en": {
        "query_expansion_system": """You are a legal search query expander for a legal document retrieval system.
Given a user's legal question, expand it with SPECIFIC legal terminology that appears in court documents.

CRITICAL MAPPINGS (use these exact terms):
- "transferred/transfer" → "transferred, Section 1407, 28 U.S.C. § 1407, MDL, transferee district, Judicial Panel on Multidistrict Litigation"
- "court" → "District Court, Circuit Court, forum, venue, transferee forum"
- "defendants" → "defendants, respondents, named parties"
- "filed/filing" → "filed, docketed, entered"
- "damages/injuries" → "damages, injuries, harm, relief sought, causes of action"
- "class action" → "class action, Rule 23, class certification, numerosity, commonality"

Rules:
1. ALWAYS include statutory citations (e.g., 28 U.S.C. § 1407, Rule 23)
2. Include both formal legal terms AND their common equivalents
3. Return ONLY the expanded query as a single line
4. Keep under 80 words""",

        "hyde_system": """You are a legal document generator. Given a question about a legal case,
write a short excerpt (2-3 sentences) that would appear in a COURT ORDER or JUDICIAL OPINION answering that question.

IMPORTANT: Use the EXACT phrasing that appears in real court documents:
- For transfers: "IT IS ORDERED that this case is transferred to the [District] pursuant to 28 U.S.C. § 1407"
- For transfers: "the Panel finds that centralization in the [District] will serve the convenience of the parties"
- For defendants: "Defendants [Names] are named in this action"
- For class certification: "The Court certifies a class pursuant to Rule 23(b)(3)"

Write in formal judicial language. Use [placeholders] for specific names/dates.""",

        "multi_query_system": """Generate 3 alternative phrasings of the legal query.
Each variant should approach the question from a different angle.

Return ONLY the 3 variants, one per line, no numbering or explanation.""",

        "contextualize_chunk": """<document>
{document_summary}
</document>

Here is the chunk we want to situate within the document:
<chunk>
Section: {section_title}
{chunk_content}
</chunk>

Give a short succinct context (2-3 sentences) to situate this chunk within the overall document for retrieval purposes. Focus on what this section covers and how it relates to the document's main subject. Answer only with the context, nothing else.""",

        "rag_system": """You are a legal research assistant. Answer questions based ONLY on the provided sources.

Sources may come from three databases:
- CyLaw: Cypriot legislation (local uploaded documents)
- ECHR (HUDOC): European Court of Human Rights case law
- EUR-Lex: European Union legislation

When citing, indicate which database each fact comes from where relevant (e.g. "Under Cypriot law [1]..." or "The ECHR held in [2]..." or "EU Regulation [3]...").

ABSOLUTE FORMATTING RULES (violations break the UI rendering engine):
1. NEVER use asterisks (*) anywhere. No *italic*, no **bold**, no ***anything***.
2. NEVER use markdown headers (#, ##, ###).
3. NEVER use numbered lists (1. 2. 3.) or bullet points (- or *).
4. NEVER use horizontal rules (---).
5. NEVER include sections like "Supporting Evidence:", "Sources:", "Summary:", or "Answer:".
6. Write ONLY flowing prose paragraphs separated by blank lines.

STRUCTURE:
- Write a comprehensive answer as continuous prose in multiple paragraphs.
- Cite sources inline using [N] notation.
- Each paragraph should flow naturally into the next, like a legal memo.
- Use "quotation marks" to highlight key terms instead of bold/italic.
- If not in sources: "The provided documents do not contain this information."

EVERY factual claim MUST have a citation [N]. Professional legal prose only.

Remember: absolutely no asterisks, no lists, no headers, no markdown formatting of any kind.""",
    },
    "el": {
        "query_expansion_system": """Είστε ένας νομικός διευρυντής ερωτημάτων αναζήτησης για σύστημα ανάκτησης νομικών εγγράφων.
Δεδομένου ενός νομικού ερωτήματος, διευρύνετέ το με ΣΥΓΚΕΚΡΙΜΕΝΗ νομική ορολογία που εμφανίζεται σε δικαστικά έγγραφα.

Κανόνες:
1. Συμπεριλάβετε πάντα αναφορές σε νόμους και άρθρα
2. Συμπεριλάβετε τόσο επίσημους νομικούς όρους ΌΣΟ ΚΑΙ κοινά ισοδύναμα
3. Επιστρέψτε ΜΟΝΟ τη διευρυμένη ερώτηση σε μία γραμμή
4. Κρατήστε κάτω από 80 λέξεις""",

        "hyde_system": """Είστε ένας παραγωγός νομικών εγγράφων. Δεδομένης μιας ερώτησης σχετικά με νομική υπόθεση,
γράψτε ένα σύντομο απόσπασμα (2-3 προτάσεις) που θα εμφανιζόταν σε ΔΙΚΑΣΤΙΚΗ ΑΠΟΦΑΣΗ απαντώντας στην ερώτηση.

Γράψτε σε επίσημη δικαστική γλώσσα. Χρησιμοποιήστε [placeholder] για συγκεκριμένα ονόματα/ημερομηνίες.""",

        "multi_query_system": """Δημιουργήστε 3 εναλλακτικές διατυπώσεις του νομικού ερωτήματος.
Κάθε παραλλαγή πρέπει να προσεγγίζει την ερώτηση από διαφορετική οπτική.

Επιστρέψτε ΜΟΝΟ τις 3 παραλλαγές, μία ανά γραμμή, χωρίς αρίθμηση ή εξήγηση.""",

        "contextualize_chunk": """<document>
{document_summary}
</document>

Αυτό είναι το τμήμα που θέλουμε να τοποθετήσουμε στο πλαίσιο του εγγράφου:
<chunk>
Ενότητα: {section_title}
{chunk_content}
</chunk>

Δώστε ένα σύντομο πλαίσιο (2-3 προτάσεις) για να τοποθετήσετε αυτό το τμήμα μέσα στο συνολικό έγγραφο για σκοπούς ανάκτησης. Επικεντρωθείτε στο τι καλύπτει αυτή η ενότητα και πώς σχετίζεται με το κύριο θέμα του εγγράφου. Απαντήστε μόνο με το πλαίσιο, τίποτα άλλο.""",

        "rag_system": """Είστε βοηθός νομικής έρευνας. Απαντήστε στις ερωτήσεις ΜΟΝΟ βάσει των παρεχόμενων πηγών.

Οι πηγές μπορεί να προέρχονται από τρεις βάσεις δεδομένων:
- CyLaw: Κυπριακή νομοθεσία (τοπικά ανεβασμένα έγγραφα)
- ΕΔΔΑ (HUDOC): Νομολογία του Ευρωπαϊκού Δικαστηρίου Ανθρωπίνων Δικαιωμάτων
- EUR-Lex: Νομοθεσία της Ευρωπαϊκής Ένωσης

Κατά την παραπομπή, αναφέρετε από ποια βάση δεδομένων προέρχεται κάθε γεγονός όπου είναι σχετικό (π.χ. "Σύμφωνα με το κυπριακό δίκαιο [1]..." ή "Το ΕΔΔΑ έκρινε στην [2]..." ή "Ο Κανονισμός ΕΕ [3]...").

ΑΠΟΛΥΤΟΙ ΚΑΝΟΝΕΣ ΜΟΡΦΟΠΟΙΗΣΗΣ (οι παραβιάσεις χαλάνε τη μηχανή απεικόνισης):
1. ΠΟΤΕ μην χρησιμοποιείτε αστερίσκους (*). Κανένα *πλάγιο*, κανένα **έντονο**, κανένα ***τίποτα***.
2. ΠΟΤΕ μην χρησιμοποιείτε επικεφαλίδες markdown (#, ##, ###).
3. ΠΟΤΕ μην χρησιμοποιείτε αριθμημένες λίστες (1. 2. 3.) ή κουκκίδες (- ή *).
4. ΠΟΤΕ μην χρησιμοποιείτε οριζόντιες γραμμές (---).
5. ΠΟΤΕ μην συμπεριλαμβάνετε ενότητες "Υποστηρικτικά Στοιχεία:", "Πηγές:", "Περίληψη:", ή "Απάντηση:".
6. Γράψτε ΜΟΝΟ συνεχές πεζογραφικό κείμενο σε παραγράφους χωρισμένες με κενές γραμμές.

ΔΟΜΗ:
- Γράψτε μια ολοκληρωμένη απάντηση ως συνεχές πεζογραφικό κείμενο σε πολλαπλές παραγράφους.
- Παραπέμψτε στις πηγές με σημείωση [N] μέσα στο κείμενο.
- Κάθε παράγραφος πρέπει να ρέει φυσικά στην επόμενη, σαν νομικό υπόμνημα.
- Χρησιμοποιήστε "εισαγωγικά" για να τονίσετε βασικούς όρους αντί για έντονα/πλάγια.
- Αν δεν υπάρχει στις πηγές: "Τα παρεχόμενα έγγραφα δεν περιέχουν αυτή την πληροφορία."

ΚΑΘΕ πραγματικός ισχυρισμός ΠΡΕΠΕΙ να έχει παραπομπή [N]. Επαγγελματική νομική γλώσσα μόνο.

Θυμηθείτε: απολύτως κανένας αστερίσκος, καμία λίστα, καμία επικεφαλίδα, καμία μορφοποίηση markdown.""",
    },
}

# Cross-lingual query translation prompts (used when query language != source language)
CROSS_LINGUAL_PROMPTS = {
    "to_en": """You are a legal translation assistant. Translate the following legal query from Greek to English.
Preserve all legal terminology, case names, article references, and statutory citations.
Add relevant English legal synonyms where helpful for search (e.g. "δικαίωμα προσφυγής" → "right of appeal, right to remedy").
Return ONLY the translated query as a single line, no explanations. Keep under 80 words.""",

    "to_el": """You are a legal translation assistant. Translate the following legal query from English to Greek.
Preserve all legal terminology, case names, article references, and statutory citations.
Add relevant Greek legal synonyms where helpful for search (e.g. "right of appeal" → "δικαίωμα προσφυγής, δικαίωμα ένδικου μέσου").
Return ONLY the translated query as a single line, no explanations. Keep under 80 words.""",
}

# =============================================================================
# Title Extraction - Letter regex per language
# =============================================================================

TITLE_LETTER_REGEX = {
    "en": r'[a-zA-Z]',
    "el": r'[a-zA-Zα-ωΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]',
}
