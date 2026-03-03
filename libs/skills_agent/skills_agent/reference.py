"""
Reference Manager — intelligent chunking and search for skill reference files.

Problem: Reference files (API docs, knowledge bases) can be very large.
Loading 50KB of API docs when you only need one endpoint wastes tokens.

Solution: ReferenceManager chunks references by markdown sections and provides
keyword/section search to retrieve only the relevant portion.

Indexing strategy (no external dependencies required):
  1. Split by markdown headings (##, ###) into sections
  2. Build inverted keyword index per section
  3. Support exact section lookup, keyword search, and LLM-assisted search

Usage:
    ref_mgr = ReferenceManager()
    ref_mgr.index_skill_references(skill)

    # Exact section
    text = ref_mgr.get_section("api_doc.md", "POST /users")

    # Keyword search (returns top-k most relevant sections)
    results = ref_mgr.search("authentication token refresh", top_k=3)

    # Get all section headings (for LLM to choose from)
    toc = ref_mgr.get_toc("api_doc.md")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RefSection:
    """A single section extracted from a reference file."""
    file_name: str          # e.g. "api_doc.md"
    heading: str            # e.g. "POST /users"
    level: int              # heading level (1=H1, 2=H2, etc.)
    content: str            # full text of this section (heading + body)
    char_count: int = 0     # length of content
    keywords: set[str] = field(default_factory=set)  # extracted keywords

    def __post_init__(self):
        self.char_count = len(self.content)


@dataclass
class SearchResult:
    """A search result with relevance score."""
    section: RefSection
    score: float
    matched_terms: list[str] = field(default_factory=list)


class ReferenceManager:
    """Manages chunked reference files for token-efficient retrieval.

    Features:
      - Markdown section-based chunking (split by ## / ### headings)
      - Keyword index for fast search (no ML dependencies)
      - Table-of-contents generation for LLM navigation
      - Exact section retrieval by heading
      - Configurable max chunk size
    """

    def __init__(self, max_section_chars: int = 3000):
        """
        Args:
            max_section_chars: If a section exceeds this, split into sub-chunks.
        """
        self._max_section_chars = max_section_chars
        # file_name → list of sections
        self._sections: dict[str, list[RefSection]] = {}
        # keyword → list of (file_name, section_index)
        self._keyword_index: dict[str, list[tuple[str, int]]] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, file_name: str, content: str) -> int:
        """Index a single reference file by splitting into sections.

        Returns number of sections created.
        """
        sections = self._split_into_sections(file_name, content)
        self._sections[file_name] = sections

        # Build keyword index
        for i, sec in enumerate(sections):
            sec.keywords = self._extract_keywords(sec.heading + " " + sec.content)
            for kw in sec.keywords:
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = []
                self._keyword_index[kw].append((file_name, i))

        logger.info(
            "Indexed reference '%s': %d sections, %d total chars",
            file_name, len(sections), sum(s.char_count for s in sections),
        )
        return len(sections)

    def index_skill_references(self, skill: Any) -> int:
        """Index all reference files from a Skill object.

        Args:
            skill: A Skill instance with reference_files dict.

        Returns:
            Total number of sections indexed.
        """
        total = 0
        for name, content in skill.reference_files.items():
            total += self.index_file(name, content)
        logger.info(
            "Indexed %d reference files for skill '%s' → %d sections",
            len(skill.reference_files), skill.id, total,
        )
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_toc(self, file_name: str | None = None) -> str:
        """Get table of contents (headings + char counts) for reference navigation.

        Args:
            file_name: Specific file, or None for all files.

        Returns:
            Formatted TOC string for LLM consumption.
        """
        files = [file_name] if file_name else list(self._sections.keys())
        lines = []
        for fname in files:
            if fname not in self._sections:
                continue
            lines.append(f"📄 {fname}:")
            for i, sec in enumerate(self._sections[fname]):
                indent = "  " * sec.level
                lines.append(f"{indent}[{i}] {sec.heading} ({sec.char_count} chars)")
        return "\n".join(lines) if lines else "No references indexed."

    def get_section(self, file_name: str, heading: str) -> str | None:
        """Get a specific section by exact or partial heading match.

        Args:
            file_name: Reference file name.
            heading: Full or partial heading to match.

        Returns:
            Section content, or None if not found.
        """
        if file_name not in self._sections:
            return None
        heading_lower = heading.lower()
        for sec in self._sections[file_name]:
            if heading_lower in sec.heading.lower():
                logger.debug("get_section: matched '%s' → '%s' (%d chars)",
                             heading, sec.heading, sec.char_count)
                return sec.content
        return None

    def get_section_by_index(self, file_name: str, index: int) -> str | None:
        """Get section by its index in the TOC."""
        secs = self._sections.get(file_name, [])
        if 0 <= index < len(secs):
            return secs[index].content
        return None

    def search(self, query: str, top_k: int = 3, file_name: str | None = None) -> list[SearchResult]:
        """Search references by keyword matching.

        Uses TF-based scoring: sections with more query term matches rank higher.

        Args:
            query: Search query (natural language or keywords).
            top_k: Maximum results to return.
            file_name: Restrict to a specific file.

        Returns:
            List of SearchResult sorted by relevance.
        """
        query_terms = self._extract_keywords(query)
        if not query_terms:
            return []

        # Score each section
        scored: list[SearchResult] = []
        for fname, sections in self._sections.items():
            if file_name and fname != file_name:
                continue
            for sec in sections:
                matched = query_terms & sec.keywords
                if matched:
                    # Score: matched terms / total query terms, weighted by match count
                    score = len(matched) / len(query_terms) * (1 + len(matched))
                    scored.append(SearchResult(
                        section=sec,
                        score=score,
                        matched_terms=sorted(matched),
                    ))

        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:top_k]

        if results:
            logger.info(
                "Reference search '%s': %d results (top score=%.2f, terms=%s)",
                query[:50], len(results), results[0].score, results[0].matched_terms[:5],
            )
        else:
            logger.debug("Reference search '%s': no results", query[:50])

        return results

    def search_text(self, query: str, top_k: int = 3, file_name: str | None = None) -> str:
        """Search and return formatted text results (for LLM consumption).

        Returns concatenated section contents, or a "not found" message.
        """
        results = self.search(query, top_k, file_name)
        if not results:
            return f"No relevant sections found for: {query}"
        parts = []
        for r in results:
            parts.append(f"--- [{r.section.file_name}] {r.section.heading} (score={r.score:.2f}) ---")
            parts.append(r.section.content)
            parts.append("")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return indexing statistics."""
        total_sections = sum(len(secs) for secs in self._sections.values())
        total_chars = sum(
            s.char_count for secs in self._sections.values() for s in secs
        )
        return {
            "files": len(self._sections),
            "sections": total_sections,
            "total_chars": total_chars,
            "unique_keywords": len(self._keyword_index),
        }

    def clear(self):
        """Clear all indexed data."""
        self._sections.clear()
        self._keyword_index.clear()

    # ------------------------------------------------------------------
    # Internal: chunking
    # ------------------------------------------------------------------

    def _split_into_sections(self, file_name: str, content: str) -> list[RefSection]:
        """Split markdown content into sections by headings."""
        sections: list[RefSection] = []

        # Split by headings (# through ####)
        heading_re = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
        matches = list(heading_re.finditer(content))

        if not matches:
            # No headings: treat entire file as one section
            sec = RefSection(
                file_name=file_name,
                heading=file_name,
                level=1,
                content=content.strip(),
            )
            if sec.char_count > self._max_section_chars:
                return self._split_large_section(sec)
            return [sec] if sec.content else []

        # Add content before first heading if substantial
        if matches[0].start() > 50:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.append(RefSection(
                    file_name=file_name,
                    heading="(preamble)",
                    level=0,
                    content=preamble,
                ))

        # Extract each section
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end].strip()

            sec = RefSection(
                file_name=file_name,
                heading=heading,
                level=level,
                content=text,
            )
            if sec.char_count > self._max_section_chars:
                sections.extend(self._split_large_section(sec))
            else:
                sections.append(sec)

        return sections

    def _split_large_section(self, section: RefSection) -> list[RefSection]:
        """Split an oversized section into smaller chunks."""
        chunks = []
        lines = section.content.split("\n")
        current_chunk: list[str] = []
        current_size = 0
        part = 1

        for line in lines:
            line_len = len(line) + 1
            if current_size + line_len > self._max_section_chars and current_chunk:
                chunks.append(RefSection(
                    file_name=section.file_name,
                    heading=f"{section.heading} (part {part})",
                    level=section.level,
                    content="\n".join(current_chunk),
                ))
                current_chunk = [line]
                current_size = line_len
                part += 1
            else:
                current_chunk.append(line)
                current_size += line_len

        if current_chunk:
            chunks.append(RefSection(
                file_name=section.file_name,
                heading=f"{section.heading} (part {part})" if part > 1 else section.heading,
                level=section.level,
                content="\n".join(current_chunk),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Internal: keyword extraction
    # ------------------------------------------------------------------

    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "because", "but", "and", "or", "if",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "me", "my", "your", "his", "her", "our", "their",
        "的", "了", "在", "是", "和", "与", "或", "及", "等", "对", "从",
        "到", "为", "以", "被", "将", "把", "让", "给", "向", "往",
    })

    @classmethod
    def _extract_keywords(cls, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        # Normalize: lowercase, split on non-alphanumeric (keep CJK)
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        # Filter stopwords and short tokens
        return {t for t in tokens if t not in cls._STOP_WORDS and len(t) > 1}
