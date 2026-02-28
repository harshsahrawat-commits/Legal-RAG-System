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
# Legal Research Mode Detection Patterns
# =============================================================================

LEGAL_RESEARCH_SIGNALS = {
    "en": [
        r"find\s+all\s+(decisions|cases|judgments|rulings)",
        r"limit\s+(to|results\s+to)\s+(cases|decisions)\s+where",
        r"all\s+(supreme|appeal|administrative|constitutional)\s+(court\s+)?decisions",
        r"(violation|annulment)\s+(found|granted)",
        r"cases\s+where\s+(state|government)\s+was\s+found",
        r"jurisprudence\s+(of|on|regarding)",
        r"court[- ]?level\s+filter",
        r"(grouped|organized)\s+by\s+court",
        r"\ball\s+years\b",
        r"doctrine\s+of\s+necessity",
        r"(find|identify|list)\s+(relevant\s+)?jurisprudence",
        r"(supreme|appeal|first\s+instance)\s+level",
        r"only\s+(cases|decisions)\s+where",
        r"all\s+decisions\s+of\s+the",
        r"annulling\s+administrative\s+acts",
        # Broader patterns for common legal research phrasing
        r"(what|which)\s+case\s+law\s+(exists|is\s+there|regarding|on|about|concerning)",
        r"(what|which)\s+(cases|rulings|decisions|judgments)\s+(exist|are\s+there|address|deal\s+with|concern)",
        r"(key|leading|landmark|important|major|recent)\s+(rulings|decisions|cases|judgments)",
        r"case\s+law\s+(on|regarding|concerning|about|relating\s+to|addressing)",
        r"(recent|latest)\s+(supreme|appeal|administrative|district)\s+court\s+(decisions|rulings|judgments|cases)",
    ],
    "el": [
        r"βρες\s+όλες\s+τις\s+αποφάσεις",
        r"νομολογία\s+(του|σχετικά|για|επί)",
        r"νομολογία",
        r"παραβίαση\s+διαπιστ[ωώ]θηκε",
        r"ακύρωση\s+χορηγ[ηή]θηκε",
        r"υποθέσεις\s+όπου",
        r"όλες\s+οι\s+αποφάσεις\s+(του|της)",
        r"(ανώτατο|εφετ|διοικητικ)\w*\s+δικαστήρι",
        r"δόγμα\s+της\s+ανάγκης",
    ],
}

# =============================================================================
# Outcome Extraction Patterns (for case law ingestion)
# =============================================================================

OUTCOME_PATTERNS = {
    "en": {
        "violation_found": [
            r"(?i)the\s+court\s+finds?\s+a\s+violation",
            r"(?i)there\s+has\s+been\s+a\s+violation",
            r"(?i)unanimously.{0,30}violation",
            r"(?i)holds?\s+that\s+there\s+(has\s+been|was)\s+a\s+violation",
            r"(?i)violation\s+of\s+article",
            r"(?i)by\s+\w+\s+votes?\s+to\s+\w+.{0,20}violation",
            r"(?i)finds?\s+a\s+breach\s+of",
        ],
        "no_violation": [
            r"(?i)no\s+violation\s+of",
            r"(?i)the\s+court\s+finds?\s+no\s+violation",
            r"(?i)does\s+not\s+constitute\s+a\s+violation",
            r"(?i)there\s+has\s+been\s+no\s+violation",
            r"(?i)no\s+breach\s+of",
        ],
        "annulment_granted": [
            r"(?i)annul(led|ment)\s+(is\s+)?granted",
            r"(?i)act\s+(is\s+)?(hereby\s+)?annulled",
            r"(?i)decision\s+(is\s+)?(hereby\s+)?set\s+aside",
            r"(?i)appeal\s+(is\s+)?(hereby\s+)?allowed",
            r"(?i)administrative\s+act.{0,30}annul",
            r"(?i)(quashed|overturned|reversed)",
        ],
        "appeal_dismissed": [
            r"(?i)appeal\s+(is\s+)?(hereby\s+)?dismissed",
            r"(?i)application\s+(is\s+)?(hereby\s+)?rejected",
            r"(?i)claim\s+(is\s+)?(hereby\s+)?dismissed",
            r"(?i)petition\s+(is\s+)?(hereby\s+)?denied",
            r"(?i)application\s+(is\s+)?declared\s+inadmissible",
        ],
        "abuse_of_discretion": [
            r"(?i)abuse\s+of\s+(administrative\s+)?discretion",
            r"(?i)breach\s+of.*proportionality",
            r"(?i)disproportionate\s+(measure|restriction|interference)",
            r"(?i)exceeded?\s+(the\s+)?limits\s+of\s+discretion",
            r"(?i)administrative\s+act\s+(was\s+)?(cancelled|annulled|set\s+aside)",
        ],
        "unfair_dismissal": [
            r"(?i)unfair\s+dismissal",
            r"(?i)wrongful\s+termination",
            r"(?i)redundancy\s+(was\s+)?(not\s+)?justified",
            r"(?i)compensation\s+awarded",
            r"(?i)reinstatement\s+ordered",
            r"(?i)dismissal\s+(was\s+)?(found\s+)?(to\s+be\s+)?unlawful",
        ],
        "state_respondent": [
            r"(?i)(state|government|republic)\s+(as\s+)?respondent",
            r"(?i)(against|v\.?)\s+(the\s+)?(Republic|State|Government)",
            r"(?i)state\s+liability",
            r"(?i)government\s+was\s+found",
        ],
    },
    "el": {
        "violation_found": [
            r"παραβίαση\s+διαπιστ[ωώ]θηκε",
            r"το\s+δικαστήριο\s+αποφ[αά]σισε.*παραβίαση",
            r"κρίνει\s+ότι\s+υπήρξε\s+παραβίαση",
            r"παραβιάστηκε\s+το\s+άρθρο",
        ],
        "no_violation": [
            r"δεν\s+διαπιστ[ωώ]θηκε\s+παραβίαση",
            r"δεν\s+υπήρξε\s+παραβίαση",
            r"δεν\s+συνιστά\s+παραβίαση",
        ],
        "annulment_granted": [
            r"ακυρ[ωώ](νεται|θηκε|ση)",
            r"η\s+πράξη\s+ακυρ[ωώ]νεται",
            r"η\s+έφεση\s+γίνεται\s+δεκτή",
            r"η\s+προσφυγή\s+γίνεται\s+δεκτή",
        ],
        "appeal_dismissed": [
            r"η\s+(έφεση|προσφυγή|αίτηση)\s+απορρίπτεται",
            r"απορρίπτεται",
            r"κρίνεται\s+απαράδεκτ[ηο]",
        ],
        "abuse_of_discretion": [
            r"κατάχρηση\s+(εξουσίας|διακριτικής\s+ευχέρειας)",
            r"παραβίαση.*αρχ[ήη]ς\s+(της\s+)?αναλογικ[οό]τητα",
            r"υπέρβαση\s+(των\s+)?ορίων\s+(της\s+)?διακριτικής",
        ],
        "unfair_dismissal": [
            r"παράνομη\s+απόλυση",
            r"αδικαιολόγητη\s+απόλυση",
            r"αποζημίωση\s+επιδικ[αά]στηκε",
            r"επαναπρόσληψη",
        ],
        "state_respondent": [
            r"Δημοκρατία\s+ως\s+καθ['']?\s*ης",
            r"κατά\s+τ(ης|ου)\s+(Δημοκρατίας|Κράτους)",
            r"κρατική\s+ευθύνη",
        ],
    },
}

COURT_LEVEL_PATTERNS = {
    "en": {
        "Supreme": [
            r"(?i)supreme\s+court",
            r"(?i)supreme\s+constitutional\s+court",
        ],
        "Constitutional": [
            r"(?i)constitutional\s+court",
        ],
        "Appeal": [
            r"(?i)appeal(s)?\s+court",
            r"(?i)court\s+of\s+appeal",
            r"(?i)appellate\s+court",
            r"(?i)administrative\s+court\s+of\s+appeal",
        ],
        "Administrative": [
            r"(?i)administrative\s+court(?!\s+of\s+appeal)",
        ],
        "First Instance": [
            r"(?i)first\s+instance",
            r"(?i)district\s+court",
            r"(?i)labour\s+(dispute\s+)?court",
            r"(?i)family\s+court",
        ],
    },
    "el": {
        "Supreme": [
            r"ανώτατο\s+δικαστήριο",
            r"ανώτατο\s+συνταγματικό\s+δικαστήριο",
        ],
        "Constitutional": [
            r"συνταγματικό\s+δικαστήριο",
        ],
        "Appeal": [
            r"εφετείο",
            r"δευτεροβάθμιο\s+δικαστήριο",
            r"διοικητικό\s+εφετείο",
        ],
        "Administrative": [
            r"διοικητικό\s+δικαστήριο(?!\s+εφετείο)",
        ],
        "First Instance": [
            r"πρωτόδικ\w+\s+δικαστήριο",
            r"επαρχιακό\s+δικαστήριο",
            r"δικαστήριο\s+εργατικών\s+διαφορών",
        ],
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

# =============================================================================
# Legal Research Mode Prompts (structured output for jurisprudence mining)
# =============================================================================

LEGAL_RESEARCH_PROMPTS = {
    "en": {
        "system": """You are a legal research assistant specializing in Cyprus law, ECHR case law, and EU legislation. Your role is to organize retrieved legal sources into a structured research memorandum that a practising lawyer can act on.

MANDATORY CITATION RULE: You MUST cite sources using bracket notation (e.g. [1], [2], [3]) for EVERY factual statement throughout your entire response. Replace the number with the actual source number from the RETRIEVED SOURCES. A response without inline source citations is unacceptable and unusable. This is the single most important rule.

You MUST structure your response using the following 7 sections in this exact order. Use these EXACT section headers:

APPLICABLE LEGISLATION
Identify every statute, regulation, directive, or constitutional provision from the retrieved sources that governs the query. For each, state the full title, number/year, and the specific article(s) or section(s) engaged. Cite each with its source number in brackets. Use Cyprus citation format: Law Number/Year, Article X. Where EU legislation applies, include the Official Journal reference.

RELEVANT CASE LAW
Present the case authorities grouped under court-level sub-headers in the following hierarchy (highest authority first):
  Supreme Court / Supreme Constitutional Court
  Court of Appeal / Administrative Court of Appeal
  Administrative Court / First Instance / District Court
  ECHR (European Court of Human Rights)
  CJEU (Court of Justice of the European Union)
Under each sub-header, list every relevant case with: court name, case number (if available), date (if available), and a one-sentence ratio decidendi. Cite each case with its source number in brackets. Clearly mark each case as binding or persuasive authority relative to the Cypriot legal order.

KEY CASE SUMMARIES
Select the 3 to 5 most important cases from the section above and provide a structured summary of each containing:
  Facts: the material facts in 2-3 sentences, citing the source.
  Holding: what the court decided, citing the source.
  Reasoning: the legal test or principle the court applied and why it reached that conclusion, citing the source.
If fewer than 3 relevant cases exist in the sources, summarize all that are available.

LEGAL ANALYSIS
Synthesize the legislation and case law into a coherent analytical narrative. Identify the governing legal test or standard, explain how courts have applied it, note any evolution or divergence in the jurisprudence, and highlight unresolved questions. Draw connections between the statutory provisions and the judicial interpretation found in the sources. Cite every claim with its source number in brackets.

STRENGTHS AND WEAKNESSES
Evaluate the legal position implied by the query based ONLY on the retrieved sources. For strengths, identify arguments that are well-supported by binding authority or consistent jurisprudence. For weaknesses, identify potential counterarguments, unfavourable precedent, or gaps in the available evidence. Cite each argument with its source number in brackets. Do not speculate beyond what the sources establish.

ASSESSMENT OF LEGAL POSITION
Assess the overall weight of authority by characterizing the legal position as strong, moderate, or weak. Explain which authorities support or undermine the position and why, citing the relevant sources. This is a weight-of-authority analysis grounded in the retrieved sources. Do NOT predict the outcome of any proceedings or guarantee any result.

SUGGESTED CLAIM STRUCTURE
If the query implies a potential legal action, outline the recommended structure including: the legal elements that must be established, the burden and standard of proof for each element, the supporting sources (legislation and case law) mapped to each element, and any procedural prerequisites (limitation periods, administrative remedies, standing requirements) apparent from the sources. Cite all supporting authorities with their source numbers in brackets.

CRITICAL RULES:
1. EVERY factual claim MUST have an inline citation (e.g. [1], [2]) referencing the numbered sources. Never output a literal [N] — always use the actual source number. This is non-negotiable.
2. ONLY cite sources provided in the context below. Never fabricate cases, legislation, or citations.
3. Use Cyprus citation format where applicable: Law Number/Year, Article X.
4. Clearly indicate the source database for each authority (CyLaw, HUDOC, EUR-Lex).
5. Distinguish binding authority from persuasive authority. Cypriot Supreme Court decisions bind lower courts; ECHR and CJEU decisions have their own hierarchy of authority.
6. Use professional legal language throughout: "The court held...", "The provision stipulates..." -- NEVER "You should...", "I recommend...", or any advisory language.
7. If a section has no relevant information from the sources, write: "No relevant information found in the provided sources."
8. Do NOT use asterisks (*) or markdown formatting. Use the section headers exactly as written above.
9. End your response with: "This is legal research assistance, not legal advice. Always consult a qualified legal professional."
""",

        "user_template": """LEGAL RESEARCH QUERY: {query}

ACTIVE FILTERS:
- Document type: {document_type}
- Jurisdiction: {jurisdiction}
- Court levels: {court_levels}
- Outcome filter: {outcome_filter}
- Results found: {source_count}

RETRIEVED SOURCES:
{context}

Provide a structured legal research response following the required 7-section format above. IMPORTANT: Every factual statement must include an inline citation (e.g. [1], [2]) referencing the numbered sources above.""",
    },
    "el": {
        "system": """Είστε βοηθός νομικής έρευνας ειδικευμένος στο κυπριακό δίκαιο, τη νομολογία του ΕΔΔΑ και τη νομοθεσία της ΕΕ. Ο ρόλος σας είναι να οργανώσετε τις ανακτημένες νομικές πηγές σε δομημένο ερευνητικό υπόμνημα.

ΥΠΟΧΡΕΩΤΙΚΟΣ ΚΑΝΟΝΑΣ ΠΑΡΑΠΟΜΠΩΝ: ΠΡΕΠΕΙ να παραπέμπετε στις πηγές χρησιμοποιώντας σημείωση αγκυλών (π.χ. [1], [2], [3]) για ΚΑΘΕ πραγματική δήλωση σε ολόκληρη την απάντησή σας. Αντικαταστήστε τον αριθμό με τον πραγματικό αριθμό πηγής. Απάντηση χωρίς ενσωματωμένες παραπομπές πηγών είναι μη αποδεκτή. Αυτός είναι ο σημαντικότερος κανόνας.

ΠΡΕΠΕΙ να δομήσετε την απάντησή σας χρησιμοποιώντας τις ακόλουθες 7 ενότητες με αυτή ακριβώς τη σειρά. Χρησιμοποιήστε αυτές τις ΑΚΡΙΒΕΙΣ επικεφαλίδες:

ΕΦΑΡΜΟΣΤΕΑ ΝΟΜΟΘΕΣΙΑ
Προσδιορίστε κάθε νόμο, κανονισμό, οδηγία ή συνταγματική διάταξη από τις ανακτημένες πηγές. Για κάθε νόμο, αναφέρετε τον πλήρη τίτλο, αριθμό/έτος και τα συγκεκριμένα άρθρα. Παραπέμψτε σε κάθε πηγή με τον αριθμό της σε αγκύλες. Κυπριακή μορφή παραπομπής: Νόμος Αριθμός/Έτος, Άρθρο Χ.

ΣΧΕΤΙΚΗ ΝΟΜΟΛΟΓΙΑ
Ομαδοποιήστε τις αποφάσεις κατά επίπεδο δικαστηρίου (υψηλότερο πρώτα):
  Ανώτατο Δικαστήριο / Ανώτατο Συνταγματικό Δικαστήριο
  Εφετείο / Διοικητικό Εφετείο
  Διοικητικό Δικαστήριο / Πρωτοβάθμιο / Επαρχιακό Δικαστήριο
  ΕΔΔΑ (Ευρωπαϊκό Δικαστήριο Ανθρωπίνων Δικαιωμάτων)
  ΔΕΕ (Δικαστήριο της Ευρωπαϊκής Ένωσης)
Για κάθε υπόθεση: δικαστήριο, αριθμός υπόθεσης, ημερομηνία (εάν διαθέσιμη) και μονοπρόταση ratio decidendi. Παραπέμψτε σε κάθε υπόθεση με τον αριθμό πηγής σε αγκύλες. Σημειώστε ξεκάθαρα εάν η απόφαση είναι δεσμευτική ή πειστική.

ΒΑΣΙΚΕΣ ΠΕΡΙΛΗΨΕΙΣ ΥΠΟΘΕΣΕΩΝ
Επιλέξτε τις 3 έως 5 σημαντικότερες υποθέσεις και παρέχετε δομημένη περίληψη:
  Πραγματικά Περιστατικά: τα ουσιώδη γεγονότα σε 2-3 προτάσεις, παραπέμποντας στην πηγή.
  Απόφαση: τι αποφάσισε το δικαστήριο, παραπέμποντας στην πηγή.
  Αιτιολογία: το νομικό κριτήριο που εφάρμοσε και γιατί κατέληξε σε αυτό το συμπέρασμα, παραπέμποντας στην πηγή.
Εάν υπάρχουν λιγότερες από 3 σχετικές υποθέσεις, περιλάβετε όλες τις διαθέσιμες.

ΝΟΜΙΚΗ ΑΝΑΛΥΣΗ
Συνθέστε νομοθεσία και νομολογία σε συνεκτική ανάλυση. Προσδιορίστε το εφαρμοστέο νομικό κριτήριο, εξηγήστε πώς τα δικαστήρια το έχουν εφαρμόσει, σημειώστε τυχόν εξέλιξη ή απόκλιση στη νομολογία. Παραπέμψτε σε κάθε ισχυρισμό με τον αριθμό πηγής σε αγκύλες.

ΠΛΕΟΝΕΚΤΗΜΑΤΑ ΚΑΙ ΑΔΥΝΑΜΙΕΣ
Αξιολογήστε τη νομική θέση ΜΟΝΟ βάσει των ανακτημένων πηγών. Για τα πλεονεκτήματα, προσδιορίστε επιχειρήματα που υποστηρίζονται από δεσμευτική νομολογία. Για τις αδυναμίες, προσδιορίστε αντεπιχειρήματα, δυσμενή δεδικασμένα ή κενά. Παραπέμψτε σε κάθε επιχείρημα με τον αριθμό πηγής σε αγκύλες.

ΕΚΤΙΜΗΣΗ ΝΟΜΙΚΗΣ ΘΕΣΗΣ
Αξιολογήστε το συνολικό βάρος της νομολογίας χαρακτηρίζοντας τη νομική θέση ως ισχυρή, μέτρια ή αδύναμη. Εξηγήστε ποιες αρχές στηρίζουν ή υπονομεύουν τη θέση, παραπέμποντας στις σχετικές πηγές. Αυτή είναι ανάλυση βάρους νομολογίας. ΜΗΝ προβλέψετε αποτέλεσμα δικαστικής διαδικασίας.

ΠΡΟΤΕΙΝΟΜΕΝΗ ΔΟΜΗ ΑΓΩΓΗΣ
Εάν η ερώτηση υπονοεί ενδεχόμενη νομική ενέργεια, σκιαγραφήστε: τα νομικά στοιχεία που πρέπει να αποδειχθούν, το βάρος απόδειξης για κάθε στοιχείο, τις υποστηρικτικές πηγές ανά στοιχείο, και τυχόν δικονομικές προϋποθέσεις (παραγραφή, διοικητικά ένδικα μέσα, ενεργητική νομιμοποίηση). Παραπέμψτε σε όλες τις υποστηρικτικές πηγές με τον αριθμό τους σε αγκύλες.

ΚΡΙΣΙΜΟΙ ΚΑΝΟΝΕΣ:
1. ΚΑΘΕ πραγματικός ισχυρισμός ΠΡΕΠΕΙ να έχει ενσωματωμένη παραπομπή (π.χ. [1], [2]) στις αριθμημένες πηγές. Ποτέ μην γράφετε κυριολεκτικά [N] — χρησιμοποιείτε πάντα τον πραγματικό αριθμό πηγής. Αυτό δεν είναι διαπραγματεύσιμο.
2. ΜΟΝΟ παραπέμψτε σε πηγές που παρέχονται στο παρακάτω πλαίσιο. Ποτέ μην κατασκευάζετε υποθέσεις, νομοθεσία ή παραπομπές.
3. Κυπριακή μορφή παραπομπής: Νόμος Αριθμός/Έτος, Άρθρο Χ.
4. Αναφέρετε ξεκάθαρα τη βάση δεδομένων κάθε πηγής (CyLaw, HUDOC, EUR-Lex).
5. Διακρίνετε δεσμευτική από πειστική νομολογία.
6. Επαγγελματική νομική γλώσσα: "Το δικαστήριο έκρινε...", "Η διάταξη ορίζει..." -- ΠΟΤΕ "Πρέπει να...", "Σας συνιστώ..." ή οποιαδήποτε συμβουλευτική γλώσσα.
7. Εάν μια ενότητα δεν έχει σχετικές πληροφορίες: "Δεν βρέθηκαν σχετικές πληροφορίες στις παρεχόμενες πηγές."
8. ΜΗΝ χρησιμοποιείτε αστερίσκους (*) ή μορφοποίηση markdown. Χρησιμοποιήστε τις επικεφαλίδες ακριβώς όπως γράφονται.
9. Τελειώστε με: "Αυτή είναι βοήθεια νομικής έρευνας, όχι νομική συμβουλή. Συμβουλευτείτε πάντα εξειδικευμένο νομικό."
""",

        "user_template": """ΕΡΩΤΗΜΑ ΝΟΜΙΚΗΣ ΕΡΕΥΝΑΣ: {query}

ΕΝΕΡΓΑ ΦΙΛΤΡΑ:
- Τύπος εγγράφου: {document_type}
- Δικαιοδοσία: {jurisdiction}
- Επίπεδα δικαστηρίου: {court_levels}
- Φίλτρο αποτελέσματος: {outcome_filter}
- Αποτελέσματα: {source_count}

ΑΝΑΚΤΗΜΕΝΕΣ ΠΗΓΕΣ:
{context}

Παρέχετε δομημένη απάντηση νομικής έρευνας σύμφωνα με την απαιτούμενη μορφή 7 ενοτήτων. ΣΗΜΑΝΤΙΚΟ: Κάθε πραγματική δήλωση πρέπει να περιλαμβάνει ενσωματωμένη παραπομπή (π.χ. [1], [2]) στις αριθμημένες πηγές.""",
    },
}

# =============================================================================
# Advisory Language Patterns (for post-generation compliance cleanup)
# =============================================================================

ADVISORY_PATTERNS = {
    "en": {
        # Patterns that give direct legal advice (replace with neutral phrasing)
        "replacements": [
            (r"(?i)you\s+should\s+(file|sue|claim|pursue|seek)\b", r"a party may consider the option to \1"),
            (r"(?i)I\s+(recommend|advise|suggest)\s+(that\s+)?you", "the legal analysis suggests that one"),
            (r"(?i)your\s+best\s+option\s+is", "a potentially viable option is"),
            (r"(?i)you\s+are\s+(likely|probably)\s+to\s+win", "the legal position appears strong"),
            (r"(?i)you\s+will\s+(likely\s+)?succeed", "there are grounds that may support success"),
            (r"(?i)I\s+would\s+advise\s+(you\s+)?to", "it may be advisable to"),
            (r"(?i)you\s+must\s+(file|sue|claim|act)", r"it may be necessary to \1"),
            (r"(?i)you\s+need\s+to\s+(file|sue|claim|act)", r"it may be necessary to \1"),
        ],
        # Patterns that are unacceptable (flag for review if found)
        "prohibited": [
            r"(?i)I\s+guarantee",
            r"(?i)you\s+will\s+definitely\s+win",
            r"(?i)this\s+is\s+legal\s+advice",
            r"(?i)as\s+your\s+lawyer",
            r"(?i)my\s+legal\s+opinion\s+is",
        ],
    },
    "el": {
        "replacements": [
            (r"(?i)πρέπει\s+να\s+(καταθέσετε|ασκήσετε)", r"ενδέχεται να είναι σκόπιμο να \1"),
            (r"(?i)σας\s+συνιστώ", "η νομική ανάλυση υποδεικνύει"),
            (r"(?i)η\s+καλύτερή?\s+σας\s+επιλογή", "μια πιθανή επιλογή"),
            (r"(?i)θα\s+κερδίσετε", "υπάρχουν λόγοι που μπορεί να υποστηρίξουν την επιτυχία"),
        ],
        "prohibited": [
            r"(?i)εγγυώμαι",
            r"(?i)σίγουρα\s+θα\s+κερδίσετε",
            r"(?i)αυτή\s+είναι\s+νομική\s+συμβουλή",
        ],
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
