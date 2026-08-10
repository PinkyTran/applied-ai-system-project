"""Retrieval over the local pet-care knowledge base.

This is the "R" in PawPal+'s RAG layer. It is deliberately dependency-free:
scoring is a hand-rolled TF-IDF over a dozen markdown files, so the same query
always returns the same chunks on any machine with no model download, no vector
database, and no network call. That makes the retrieval step reproducible and
testable on its own, separately from anything the language model does.

Typical use:

    kb = KnowledgeBase.load()
    chunks = kb.search(build_query(pet, owner), top_k=4)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Words carrying no retrieval signal. Kept small on purpose: an aggressive list
# would strip terms like "old" or "long" that genuinely discriminate here.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how in is it its of on or that
    the this to was were will with your you my me i not no do does than then
    when which who whom what while about into over under more most less need
    needs needed should must can could would per each every they them their
    """.split()
)

# Multiplier applied to terms in a chunk's title and tags. A chunk explicitly
# tagged "senior" should beat one that merely says "senior" once in prose.
HEADER_WEIGHT = 3

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase content words, crudely singularised.

    Stripping a trailing "s" is not real stemming, but it makes "cats" match
    "cat" and "walks" match "walk", which is the only inflection that matters
    for this vocabulary. Words of 3 letters or fewer keep their "s" so that
    genuinely short words are not mangled.

    Args:
        text: raw text to tokenize.

    Returns:
        A list of normalised terms, stopwords removed.
    """
    terms = []
    for word in _TOKEN_RE.findall(text.lower()):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word in STOPWORDS or len(word) < 2:
            continue
        terms.append(word)
    return terms


@dataclass
class Chunk:
    """One retrievable document from the knowledge base."""

    id: str  # filename stem, e.g. "senior-pets" — used for citations
    title: str
    tags: list[str]
    text: str
    term_counts: Counter[str] = field(default_factory=Counter, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Chunk:
        """Parse one markdown file into a Chunk.

        The expected shape is a "# Title" first line, one or more "tags:" lines,
        then free prose. Anything missing degrades gracefully: a file with no
        title falls back to its filename, and a file with no tags simply gets
        no header boost.

        Args:
            path: the markdown file to parse.

        Returns:
            A Chunk with its term counts already computed.
        """
        raw = path.read_text(encoding="utf-8")
        title = ""
        tags: list[str] = []
        body_lines: list[str] = []

        for line in raw.splitlines():
            stripped = line.strip()
            if not title and stripped.startswith("# "):
                title = stripped[2:].strip()
            elif stripped.lower().startswith("tags:"):
                tags.extend(t.strip() for t in stripped[5:].split(",") if t.strip())
            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()
        title = title or path.stem.replace("-", " ")

        # Header terms are counted HEADER_WEIGHT times so they dominate scoring.
        counts = Counter(tokenize(body))
        counts.update(tokenize(f"{title} {' '.join(tags)}") * HEADER_WEIGHT)

        return cls(id=path.stem, title=title, tags=tags, text=body, term_counts=counts)

    def citation(self) -> str:
        """Return the human-readable source label used in the UI and logs."""
        return f"{self.title} ({self.id})"


class KnowledgeBase:
    """An in-memory TF-IDF index over the knowledge/ directory."""

    def __init__(self, chunks: list[Chunk]):
        """Build the index and precompute inverse document frequencies."""
        self.chunks = chunks
        self._idf = self._compute_idf(chunks)

    @classmethod
    def load(cls, directory: Path | str = KNOWLEDGE_DIR) -> KnowledgeBase:
        """Load every .md file in a directory into a KnowledgeBase.

        Args:
            directory: folder holding the markdown chunks.

        Returns:
            A ready-to-query KnowledgeBase.

        Raises:
            FileNotFoundError: if the directory is missing or holds no .md files.
                Failing loudly here is deliberate — a silently empty knowledge
                base would make the AI answer from memory instead of sources.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Knowledge directory not found: {directory}")

        paths = sorted(directory.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No .md knowledge files found in {directory}")

        chunks = [Chunk.from_file(p) for p in paths]
        logger.info("Loaded %d knowledge chunks from %s", len(chunks), directory)
        return cls(chunks)

    @staticmethod
    def _compute_idf(chunks: list[Chunk]) -> dict[str, float]:
        """Return each term's inverse document frequency across the corpus.

        Smoothed so a term appearing in every chunk scores above zero rather
        than being discarded outright.
        """
        n = len(chunks)
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            doc_freq.update(chunk.term_counts.keys())
        return {term: math.log((n + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}

    def score(self, query_terms: list[str], chunk: Chunk) -> float:
        """Return the TF-IDF similarity between a tokenized query and one chunk.

        Document term frequency is damped with 1 + log(tf) so a chunk that
        repeats one word many times cannot crowd out a chunk matching several
        query terms. Query-side frequency is NOT damped: repeating a word in the
        query is how build_query() says "this term matters most", so three
        mentions of "cat" really should count three times. The result is
        length-normalised so long chunks are not favoured for sheer wordcount.

        Args:
            query_terms: output of tokenize() on the query, duplicates kept.
            chunk: the candidate chunk.

        Returns:
            A non-negative score; 0.0 means no overlap at all.
        """
        if not chunk.term_counts:
            return 0.0
        total = 0.0
        for term, query_tf in Counter(query_terms).items():
            tf = chunk.term_counts.get(term, 0)
            if tf:
                total += query_tf * (1 + math.log(tf)) * self._idf.get(term, 1.0)
        return total / math.sqrt(sum(chunk.term_counts.values()))

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[Chunk]:
        """Return the top_k chunks most relevant to a query, best first.

        Chunks scoring at or below min_score are dropped, so an off-topic query
        returns fewer results (or none) rather than padding the list with
        irrelevant sources the model would then be tempted to cite.

        Args:
            query: free text, e.g. "senior cat 12 years grooming".
            top_k: maximum number of chunks to return.
            min_score: exclusive lower bound on the similarity score.

        Returns:
            A list of Chunks, ordered most relevant first.
        """
        terms = tokenize(query)
        if not terms:
            return []

        scored = [(self.score(terms, c), c) for c in self.chunks]
        # Sort by score desc, then id asc so ties resolve deterministically.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))

        hits = [chunk for score, chunk in scored[:top_k] if score > min_score]
        logger.info(
            "Retrieval query=%r -> %s",
            query,
            [(c.id, round(s, 3)) for s, c in scored[:top_k] if s > min_score],
        )
        return hits


def age_band(species: str, age: int | None) -> str:
    """Describe a pet's life stage in words the knowledge base actually uses.

    The chunks are written in terms of "puppy", "kitten" and "senior" rather
    than raw numbers, so turning the age into the matching vocabulary is what
    makes retrieval fire on the right documents. Thresholds follow
    knowledge/senior-pets.md: dogs are senior at 7, cats at 10.

    Args:
        species: "dog", "cat", or anything else.
        age: age in years, or None if unknown.

    Returns:
        A descriptive phrase, or "" when the age is unknown.
    """
    if age is None:
        return ""
    species = (species or "").lower()
    if age < 1:
        return "puppy young" if species == "dog" else "kitten young"
    if species == "dog" and age >= 7:
        return "senior old elderly arthritis mobility"
    if species == "cat" and age >= 10:
        return "senior old elderly arthritis mobility"
    return "adult"


def build_query(pet, owner=None) -> str:
    """Turn a Pet (and optionally its Owner) into a retrieval query string.

    Accepts any object exposing name/species/breed/age, so tests can pass a
    lightweight stub instead of constructing a full Pet.

    Terms are repeated to weight them: species matters most (a cat must not be
    handed dog guidance), then breed, then life stage. score() counts query-side
    repetition, so this is the weighting mechanism rather than decoration.

    Args:
        pet: the pet to build a query for.
        owner: optional owner; a tight time budget pulls in the triage chunk.

    Returns:
        A query string ready for KnowledgeBase.search().
    """
    species = (getattr(pet, "species", "") or "").strip()
    breed = (getattr(pet, "breed", "") or "").strip()

    # The daily-care topics differ by species, and naming them is what pulls in
    # the toileting chunk: without "litter" a cat query never reaches
    # cat-litter.md, and without "walk" a dog query never reaches dog-walking.md.
    topics = {
        "dog": "walk potty feeding grooming play",
        "cat": "litter box feeding grooming play",
    }.get(species.lower(), "feeding grooming play routine")

    parts = [
        " ".join([species] * 4),
        " ".join([breed] * 2),
        age_band(species, getattr(pet, "age", None)),
        topics,
    ]
    if owner is not None and getattr(owner, "available_minutes", 999) < 60:
        parts.append("short on time busy budget triage priority minimum")
    return " ".join(p for p in parts if p.strip()).strip()
