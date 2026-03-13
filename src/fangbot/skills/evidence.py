"""Evidence retrieval orchestrator — structured guideline extraction and conflict detection."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    """Type of evidence source."""

    GUIDELINE = "guideline"
    LANDMARK_TRIAL = "landmark_trial"
    HOSPITAL_PROTOCOL = "hospital_protocol"


class EvidenceStrength(str, Enum):
    """Strength of evidence / recommendation grade."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    EXPERT_OPINION = "expert_opinion"


class EvidenceCitation(BaseModel):
    """A single evidence citation extracted from a tool result."""

    recommendation: str
    source: EvidenceSource
    doi: str | None = None
    pmid: str | None = None
    strength: EvidenceStrength | None = None
    guideline_id: str | None = None
    section: str | None = None
    organization: str | None = None


class GuidelineReference(BaseModel):
    """A guideline that was consulted during the session."""

    guideline_id: str
    title: str
    organization: str | None = None
    year: int | None = None
    sections_consulted: list[str] = []


class EvidenceConflict(BaseModel):
    """A detected conflict between two or more evidence sources."""

    topic: str
    citations: list[EvidenceCitation] = Field(min_length=2)
    description: str


class EvidenceExtractor:
    """Extracts structured evidence metadata from MCP tool result text."""

    # Regex patterns for evidence markers
    _DOI_PATTERN = re.compile(r"(?:DOI:\s*|doi:\s*)(10\.\d{4,}/[^\s,]+)")
    _PMID_PATTERN = re.compile(r"(?:PMID:\s*)(\d{6,})")
    _GUIDELINE_ID_PATTERN = re.compile(
        r"(?:Guideline ID:\s*|^\d+\.\s*\[)([a-z0-9-]+)\]?", re.MULTILINE
    )
    _TITLE_PATTERN = re.compile(r"Title:\s*(.+?)(?:\n|$)")
    _ORG_PATTERN = re.compile(r"Organization:\s*(.+?)(?:\n|$)")
    _YEAR_PATTERN = re.compile(r"Year:\s*(\d{4})")
    _SECTION_PATTERN = re.compile(r"Section:\s*(.+?)(?:\n|$)")
    _RECOMMENDATION_PATTERN = re.compile(r"Recommendation:\s*(.+?)(?:\n|$)")
    _STRENGTH_PATTERN = re.compile(r"Strength:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    _SEARCH_ENTRY_PATTERN = re.compile(
        r"\d+\.\s*\[([a-z0-9-]+)\]\s*(.+?)(?:\s*—\s*DOI:\s*(10\.\d{4,}/[^\s,]+))?$",
        re.MULTILINE,
    )

    def extract_citations(self, text: str, tool_name: str) -> list[EvidenceCitation]:
        """Extract evidence citations from a tool result string."""
        if not text or not text.strip():
            return []

        dois = self._DOI_PATTERN.findall(text)
        pmids = self._PMID_PATTERN.findall(text)

        if not dois and not pmids:
            return []

        source = self._classify_source(text)
        strength = self._extract_strength(text)
        recommendation = self._extract_recommendation(text)
        section_match = self._SECTION_PATTERN.search(text)
        org_match = self._ORG_PATTERN.search(text)
        guideline_id_match = self._GUIDELINE_ID_PATTERN.search(text)

        citations: list[EvidenceCitation] = []

        # Build one citation per DOI found
        for i, doi in enumerate(dois):
            citations.append(
                EvidenceCitation(
                    doi=doi,
                    pmid=pmids[i] if i < len(pmids) else None,
                    source=source,
                    strength=strength,
                    recommendation=recommendation or f"See DOI: {doi}",
                    section=section_match.group(1).strip() if section_match else None,
                    organization=org_match.group(1).strip() if org_match else None,
                    guideline_id=guideline_id_match.group(1) if guideline_id_match else None,
                )
            )

        # Handle PMIDs without DOIs
        for i, pmid in enumerate(pmids):
            if i >= len(dois):
                citations.append(
                    EvidenceCitation(
                        pmid=pmid,
                        source=source,
                        strength=strength,
                        recommendation=recommendation or f"See PMID: {pmid}",
                        section=section_match.group(1).strip() if section_match else None,
                        organization=org_match.group(1).strip() if org_match else None,
                    )
                )

        return citations

    def extract_guideline_reference(self, text: str) -> GuidelineReference | None:
        """Extract a structured guideline reference from tool result text."""
        id_match = self._GUIDELINE_ID_PATTERN.search(text)
        if not id_match:
            return None

        title_match = self._TITLE_PATTERN.search(text)
        org_match = self._ORG_PATTERN.search(text)
        year_match = self._YEAR_PATTERN.search(text)
        section_match = self._SECTION_PATTERN.search(text)

        return GuidelineReference(
            guideline_id=id_match.group(1),
            title=title_match.group(1).strip() if title_match else id_match.group(1),
            organization=org_match.group(1).strip() if org_match else None,
            year=int(year_match.group(1)) if year_match else None,
            sections_consulted=[section_match.group(1).strip()] if section_match else [],
        )

    def extract_search_results(self, text: str) -> list[GuidelineReference]:
        """Extract guideline references from search_guidelines results."""
        refs: list[GuidelineReference] = []
        for match in self._SEARCH_ENTRY_PATTERN.finditer(text):
            guideline_id = match.group(1)
            title = match.group(2).strip()
            refs.append(
                GuidelineReference(
                    guideline_id=guideline_id,
                    title=title,
                )
            )
        return refs

    def _classify_source(self, text: str) -> EvidenceSource:
        """Classify the evidence source type from text content."""
        lower = text.lower()
        if "landmark trial" in lower or "trial:" in lower or "rct" in lower:
            return EvidenceSource.LANDMARK_TRIAL
        if "hospital protocol" in lower or "institutional" in lower:
            return EvidenceSource.HOSPITAL_PROTOCOL
        return EvidenceSource.GUIDELINE

    def _extract_strength(self, text: str) -> EvidenceStrength | None:
        """Extract recommendation strength from text."""
        match = self._STRENGTH_PATTERN.search(text)
        if not match:
            return None
        strength_text = match.group(1).lower()
        if "strong" in strength_text or "class i" in strength_text:
            return EvidenceStrength.STRONG
        if "moderate" in strength_text or "class ii" in strength_text:
            return EvidenceStrength.MODERATE
        if "weak" in strength_text or "class iii" in strength_text:
            return EvidenceStrength.WEAK
        if "expert" in strength_text:
            return EvidenceStrength.EXPERT_OPINION
        return None

    def _extract_recommendation(self, text: str) -> str | None:
        """Extract the recommendation text."""
        match = self._RECOMMENDATION_PATTERN.search(text)
        return match.group(1).strip() if match else None


# Guideline tool names for filtering
GUIDELINE_TOOLS = {"search_guidelines", "retrieve_guideline"}


class EvidenceTracker:
    """Tracks evidence citations and guidelines consulted during a session."""

    def __init__(self) -> None:
        self._extractor = EvidenceExtractor()
        self._citations: list[EvidenceCitation] = []
        self._guidelines: list[GuidelineReference] = []
        self._conflicts: list[EvidenceConflict] = []

    @property
    def citations(self) -> list[EvidenceCitation]:
        return list(self._citations)

    @property
    def guidelines(self) -> list[GuidelineReference]:
        return list(self._guidelines)

    @property
    def conflicts(self) -> list[EvidenceConflict]:
        return list(self._conflicts)

    @property
    def has_conflicts(self) -> bool:
        return len(self._conflicts) > 0

    def add_citation(self, citation: EvidenceCitation) -> None:
        self._citations.append(citation)

    def add_guideline(self, ref: GuidelineReference) -> None:
        """Add a guideline, merging sections if already tracked."""
        for existing in self._guidelines:
            if existing.guideline_id == ref.guideline_id:
                for section in ref.sections_consulted:
                    if section not in existing.sections_consulted:
                        existing.sections_consulted.append(section)
                return
        self._guidelines.append(ref)

    def add_conflict(
        self,
        topic: str,
        citations: list[EvidenceCitation],
        description: str,
    ) -> None:
        self._conflicts.append(
            EvidenceConflict(topic=topic, citations=citations, description=description)
        )

    def process_tool_result(self, tool_name: str, result_text: str) -> None:
        """Process an MCP tool result, extracting evidence if it's a guideline tool."""
        if tool_name not in GUIDELINE_TOOLS:
            return

        if tool_name == "search_guidelines":
            refs = self._extractor.extract_search_results(result_text)
            for ref in refs:
                self.add_guideline(ref)
        elif tool_name == "retrieve_guideline":
            citations = self._extractor.extract_citations(result_text, tool_name)
            for c in citations:
                self.add_citation(c)
            ref = self._extractor.extract_guideline_reference(result_text)
            if ref:
                self.add_guideline(ref)

    def summary(self) -> str:
        """Generate a text summary of all evidence consulted."""
        if not self._citations and not self._guidelines:
            return ""

        parts: list[str] = []

        if self._guidelines:
            parts.append("Guidelines consulted:")
            for g in self._guidelines:
                line = f"- {g.title}"
                if g.organization:
                    line += f" ({g.organization})"
                if g.sections_consulted:
                    line += f" — Sections: {', '.join(g.sections_consulted)}"
                parts.append(line)

        if self._citations:
            parts.append("\nEvidence citations:")
            for c in self._citations:
                line = f"- {c.recommendation}"
                if c.organization:
                    line += f" [{c.organization}]"
                if c.doi:
                    line += f" (DOI: {c.doi})"
                if c.pmid:
                    line += f" (PMID: {c.pmid})"
                parts.append(line)

        if self._conflicts:
            parts.append("\n⚠ Evidence conflicts detected:")
            for conf in self._conflicts:
                parts.append(f"- {conf.topic}: {conf.description}")

        return "\n".join(parts)
