"""Tests for language_patterns.py - verify all patterns match expected text."""

import re
import pytest

from execution.legal_rag.language_patterns import (
    DOCTYPE_PATTERNS,
    SECTION_PATTERNS,
    SECTION_MARKERS,
    REFERENCE_PATTERNS,
    DEFINITION_MARKERS,
    PARTY_PATTERNS,
    JURISDICTION_PATTERNS,
    QUERY_CLASSIFICATION,
    PARAGRAPH_REFERENCE_PATTERNS,
    CITATION_SECTION_PATTERNS,
    LABELS,
    LLM_PROMPTS,
    TITLE_LETTER_REGEX,
    DATE_PATTERNS,
)


# =============================================================================
# English pattern tests (ensure existing patterns still work)
# =============================================================================

class TestEnglishDoctypePatterns:
    def test_contract_patterns(self):
        text = "This Agreement is entered into between the parties"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["en"]["contract"])

    def test_statute_patterns(self):
        text = "Be it enacted by the legislature"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["en"]["statute"])

    def test_case_law_patterns(self):
        text = "The plaintiff filed a motion before the court"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["en"]["case_law"])

    def test_regulation_patterns(self):
        text = "This regulation was promulgated pursuant to the CFR"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["en"]["regulation"])


class TestEnglishSectionPatterns:
    def test_article_pattern(self):
        text = "ARTICLE IV - TERMINATION\nSome content"
        assert SECTION_PATTERNS["en"]["article"].search(text)

    def test_section_pattern(self):
        text = "SECTION 3.1 Payment Terms"
        assert SECTION_PATTERNS["en"]["section"].search(text)

    def test_part_pattern(self):
        text = "PART II - OBLIGATIONS"
        assert SECTION_PATTERNS["en"]["part"].search(text)


class TestEnglishQueryClassification:
    def test_analytical_pattern(self):
        patterns = QUERY_CLASSIFICATION["en"]["analytical"]
        assert any(re.search(p, "explain the termination clause") for p in patterns)
        assert any(re.search(p, "compare the two contracts") for p in patterns)
        assert any(re.search(p, "why was the case dismissed") for p in patterns)

    def test_factual_pattern(self):
        patterns = QUERY_CLASSIFICATION["en"]["factual"]
        assert any(re.search(p, "what is the termination clause") for p in patterns)
        assert any(re.search(p, "who are the parties") for p in patterns)

    def test_legal_reference_pattern(self):
        pattern = QUERY_CLASSIFICATION["en"]["legal_reference"]
        assert re.search(pattern, "what does section 4.2 say")
        assert re.search(pattern, "find article about damages")


class TestEnglishParagraphReference:
    def test_paragraph_word(self):
        patterns = PARAGRAPH_REFERENCE_PATTERNS["en"]
        text = "paragraph 28"
        assert any(re.search(p, text) for p in patterns)

    def test_para_abbreviation(self):
        patterns = PARAGRAPH_REFERENCE_PATTERNS["en"]
        text = "para. 15"
        assert any(re.search(p, text) for p in patterns)

    def test_pilcrow_symbol(self):
        patterns = PARAGRAPH_REFERENCE_PATTERNS["en"]
        text = "¶28"
        assert any(re.search(p, text) for p in patterns)


# =============================================================================
# Greek pattern tests
# =============================================================================

class TestGreekDoctypePatterns:
    def test_contract_patterns(self):
        text = "Η παρούσα σύμβαση υπεγράφη μεταξύ των μερών"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["el"]["contract"])

    def test_statute_patterns(self):
        text = "Ψηφίστηκε ο νόμος 4412/2016"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["el"]["statute"])

    def test_case_law_patterns(self):
        text = "Ο ενάγων κατέθεσε αγωγή στο δικαστήριο"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["el"]["case_law"])

    def test_regulation_patterns(self):
        text = "Ο κανονισμός εκδόθηκε σύμφωνα με τον νόμο"
        assert any(re.search(p, text) for p in DOCTYPE_PATTERNS["el"]["regulation"])


class TestGreekSectionPatterns:
    def test_article_pattern(self):
        text = "ΑΡΘΡΟ 5 - Υποχρεώσεις"
        assert SECTION_PATTERNS["el"]["article"].search(text)

    def test_chapter_pattern(self):
        text = "ΚΕΦΑΛΑΙΟ Α - Γενικές Διατάξεις"
        assert SECTION_PATTERNS["el"]["chapter"].search(text)

    def test_part_pattern(self):
        text = "ΜΕΡΟΣ II - Ειδικοί Όροι"
        assert SECTION_PATTERNS["el"]["part"].search(text)


class TestGreekQueryClassification:
    def test_analytical_pattern(self):
        patterns = QUERY_CLASSIFICATION["el"]["analytical"]
        assert any(re.search(p, "εξηγήστε τον όρο καταγγελίας") for p in patterns)
        assert any(re.search(p, "συγκρίνετε τις δύο συμβάσεις") for p in patterns)
        assert any(re.search(p, "γιατί απορρίφθηκε η αγωγή") for p in patterns)

    def test_factual_pattern(self):
        patterns = QUERY_CLASSIFICATION["el"]["factual"]
        assert any(re.search(p, "τι είναι η ρήτρα καταγγελίας") for p in patterns)
        assert any(re.search(p, "ποιος είναι ο ενάγων") for p in patterns)

    def test_legal_reference_pattern(self):
        pattern = QUERY_CLASSIFICATION["el"]["legal_reference"]
        assert re.search(pattern, "τι λέει το άρθρο 5")
        assert re.search(pattern, "βρείτε την παράγραφος 3")


class TestGreekParagraphReference:
    def test_paragraph_word_greek(self):
        patterns = PARAGRAPH_REFERENCE_PATTERNS["el"]
        text = "παράγραφος 28"
        assert any(re.search(p, text) for p in patterns)

    def test_para_abbreviation_greek(self):
        patterns = PARAGRAPH_REFERENCE_PATTERNS["el"]
        text = "παρ. 15"
        assert any(re.search(p, text) for p in patterns)


class TestGreekReferencePatterns:
    def test_article_reference(self):
        patterns = REFERENCE_PATTERNS["el"]
        text = "σύμφωνα με το Άρθρο 12"
        assert any(re.search(p, text) for p in patterns)

    def test_paragraph_reference(self):
        patterns = REFERENCE_PATTERNS["el"]
        text = "βλ. παράγραφος (α)"
        assert any(re.search(p, text) for p in patterns)


class TestGreekDefinitionMarkers:
    def test_quoted_definition(self):
        patterns = DEFINITION_MARKERS["el"]
        text = '"Εργοδότης" σημαίνει η εταιρεία'
        assert any(re.findall(p, text) for p in patterns)

    def test_guillemet_definition(self):
        patterns = DEFINITION_MARKERS["el"]
        text = "«Εργοδότης» σημαίνει"
        assert any(re.findall(p, text) for p in patterns)


# =============================================================================
# Shared / cross-language tests
# =============================================================================

class TestLabels:
    def test_english_labels(self):
        assert LABELS["en"]["document_summary"] == "DOCUMENT SUMMARY:"
        assert LABELS["en"]["sources"] == "Sources:"
        assert LABELS["en"]["page_single"] == "p."

    def test_greek_labels(self):
        assert LABELS["el"]["document_summary"] == "ΠΕΡΙΛΗΨΗ ΕΓΓΡΑΦΟΥ:"
        assert LABELS["el"]["sources"] == "Πηγές:"
        assert LABELS["el"]["page_single"] == "σελ."


class TestLLMPrompts:
    def test_english_prompts_exist(self):
        assert "query_expansion_system" in LLM_PROMPTS["en"]
        assert "hyde_system" in LLM_PROMPTS["en"]
        assert "multi_query_system" in LLM_PROMPTS["en"]
        assert "contextualize_chunk" in LLM_PROMPTS["en"]
        assert "rag_system" in LLM_PROMPTS["en"]

    def test_greek_prompts_exist(self):
        assert "query_expansion_system" in LLM_PROMPTS["el"]
        assert "hyde_system" in LLM_PROMPTS["el"]
        assert "multi_query_system" in LLM_PROMPTS["el"]
        assert "contextualize_chunk" in LLM_PROMPTS["el"]
        assert "rag_system" in LLM_PROMPTS["el"]

    def test_contextualize_chunk_has_placeholders(self):
        template = LLM_PROMPTS["en"]["contextualize_chunk"]
        assert "{document_summary}" in template
        assert "{section_title}" in template
        assert "{chunk_content}" in template

        template_el = LLM_PROMPTS["el"]["contextualize_chunk"]
        assert "{document_summary}" in template_el
        assert "{section_title}" in template_el
        assert "{chunk_content}" in template_el


class TestTitleLetterRegex:
    def test_english_letters(self):
        assert re.findall(TITLE_LETTER_REGEX["en"], "Hello World") == list("HelloWorld")

    def test_greek_letters(self):
        found = re.findall(TITLE_LETTER_REGEX["el"], "Σύμβαση ABC")
        assert "Σ" in found
        assert "A" in found


class TestCitationSectionPatterns:
    def test_english_patterns(self):
        patterns = CITATION_SECTION_PATTERNS["en"]
        for regex, template in patterns:
            # Ensure they compile
            re.compile(regex, re.IGNORECASE)

    def test_greek_patterns(self):
        patterns = CITATION_SECTION_PATTERNS["el"]
        for regex, template in patterns:
            re.compile(regex, re.IGNORECASE)

    def test_english_article_match(self):
        patterns = CITATION_SECTION_PATTERNS["en"]
        path = "Document/Article_IV"
        for regex, template in patterns:
            match = re.search(regex, path, re.IGNORECASE)
            if match:
                assert template.format(match.group(1)) == "Article IV"
                return
        pytest.fail("No English article pattern matched")

    def test_greek_article_match(self):
        patterns = CITATION_SECTION_PATTERNS["el"]
        path = "Έγγραφο/Άρθρο_5"
        for regex, template in patterns:
            match = re.search(regex, path, re.IGNORECASE)
            if match:
                assert template.format(match.group(1)) == "Άρθρο 5"
                return
        pytest.fail("No Greek article pattern matched")


class TestDatePatterns:
    def test_english_date_patterns_exist(self):
        assert "patterns" in DATE_PATTERNS["en"]
        assert "formats" in DATE_PATTERNS["en"]

    def test_greek_date_patterns_exist(self):
        assert "patterns" in DATE_PATTERNS["el"]
        assert "month_map" in DATE_PATTERNS["el"]
        assert len(DATE_PATTERNS["el"]["month_map"]) == 12
