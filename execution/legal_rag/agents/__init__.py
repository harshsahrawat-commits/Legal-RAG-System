"""
Legal RAG Agents

Agentic components for intelligent legal document querying:
- Query Understanding Agent: Analyzes and decomposes queries
- Retrieval Planning Agent: Plans search strategy
- Citation Verification Agent: Validates sources
- Response Synthesis Agent: Generates cited responses
- HudocAgent: Searches ECHR case law on HUDOC
- EurLexAgent: Searches EU legislation on EUR-Lex

These agents work together to provide high-precision legal answers.
"""

from .hudoc_agent import HudocAgent, HudocResult
from .eurlex_agent import EurLexAgent, EurLexResult
from .result_converter import hudoc_to_source_info, eurlex_to_source_info

__all__ = [
    "HudocAgent",
    "HudocResult",
    "EurLexAgent",
    "EurLexResult",
    "hudoc_to_source_info",
    "eurlex_to_source_info",
]
