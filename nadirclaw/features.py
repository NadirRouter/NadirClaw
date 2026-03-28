"""Multi-dimensional structural feature extractor for NadirClaw.

Extracts 25+ named features from OpenAI-format messages across six
dimensions, producing both a human-readable dict and a normalised float
vector suitable for ML input.  Ported from Horizen's EnhancedBERT
analyzer with additional signals from NadirClaw's routing intelligence.

Feature dimensions
------------------
1. **Pattern matching** (weight 0.30)
   Regex-based scoring against curated simple / medium / complex prompt
   pattern libraries.  Captures the *shape* of the request (e.g. "write
   a function" vs "design a system").

2. **Linguistic complexity** (weight 0.25)
   Sentence length, vocabulary sophistication (presence of words like
   "paradigm", "methodology"), and technical jargon density.

3. **Domain complexity** (weight 0.20)
   Keyword matching across mathematics, programming, science, business,
   and legal domains.  Each domain carries a base complexity score that
   increases with additional keyword hits.

4. **Structural features** (weight 0.15)
   Surface-level metrics: word count, question marks, numbered /
   bulleted lists, code blocks, inline code, URLs, and JSON structures.

5. **Intent detection** (weight 0.10)
   Creative / analysis / problem-solving / teaching indicators derived
   from imperative verbs and intent phrases.

6. **Routing signals** (weight 0.00 in complexity score, but included in
   the feature vector for downstream ML)
   Agentic score, reasoning detection, vision content, token estimate,
   tool count, and message count -- sourced from ``nadirclaw.routing``.

Performance
-----------
All extraction is pure regex + string ops -- no ML inference.  Typical
latency is <0.5 ms on a modern CPU.

Dependencies: stdlib only (``re``, ``math``).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pre-compiled pattern libraries (module-level for zero per-call cost)
# ---------------------------------------------------------------------------

# Simple task indicators (low complexity, score ~0.2)
_SIMPLE_PATTERNS = {
    "basic_math": [
        re.compile(r"\d+\s*[+\-*/]\s*\d+"),
        re.compile(r"what is \d+"),
        re.compile(r"calculate \d+"),
    ],
    "simple_definitions": [
        re.compile(r"what is (?:a|an|the)?\s*\w+"),
        re.compile(r"define \w+"),
    ],
    "basic_questions": [
        re.compile(r"how do you"),
        re.compile(r"can you"),
        re.compile(r"what does"),
    ],
    "single_word": [
        re.compile(r"^\w+\?*$"),
        re.compile(r"^explain \w+$"),
    ],
}

# Medium complexity indicators (score ~0.5)
_MEDIUM_PATTERNS = {
    "code_simple": [
        re.compile(r"write (?:a|an)?\s*function"),
        re.compile(r"how to \w+ in python"),
        re.compile(r"create (?:a|an)?\s*\w+"),
    ],
    "explanations": [
        re.compile(r"explain how"),
        re.compile(r"describe the"),
        re.compile(r"what are the steps"),
    ],
    "comparisons": [
        re.compile(r"difference between"),
        re.compile(r"compare \w+ and \w+"),
        re.compile(r"vs\.?"),
    ],
    "tutorials": [
        re.compile(r"how to (?:\w+\s+){2,}"),
        re.compile(r"step by step"),
        re.compile(r"guide to"),
    ],
}

# Complex task indicators (score ~0.8)
_COMPLEX_PATTERNS = {
    "advanced_code": [
        re.compile(r"implement (?:a|an)?\s*\w+(?:\s+\w+)+"),
        re.compile(r"design (?:a|an)?\s*system"),
        re.compile(r"architecture"),
    ],
    "analysis": [
        re.compile(r"analyze"),
        re.compile(r"evaluate"),
        re.compile(r"critique"),
        re.compile(r"assess"),
    ],
    "research": [
        re.compile(r"research"),
        re.compile(r"investigate"),
        re.compile(r"comprehensive"),
        re.compile(r"detailed analysis"),
    ],
    "creative": [
        re.compile(r"write (?:a|an)?\s*(?:story|essay|article)"),
        re.compile(r"creative"),
        re.compile(r"generate"),
    ],
    "multi_part": [
        re.compile(r"first.*then.*finally"),
        re.compile(r"multiple"),
        re.compile(r"several"),
    ],
    "reasoning": [
        re.compile(r"reasoning"),
        re.compile(r"logic"),
        re.compile(r"proof"),
        re.compile(r"demonstrate"),
    ],
}

# Domain keyword sets with base complexity scores
_DOMAIN_COMPLEXITY: Dict[str, Dict[str, Any]] = {
    "mathematics": {
        "keywords": [
            "calculus", "algebra", "theorem", "proof", "equation",
            "integral", "derivative", "eigenvalue", "matrix",
        ],
        "base_score": 0.6,
    },
    "programming": {
        "keywords": [
            "algorithm", "data structure", "optimization", "refactor",
            "concurrency", "distributed", "microservice", "api",
        ],
        "base_score": 0.5,
    },
    "science": {
        "keywords": [
            "quantum", "molecular", "scientific", "research",
            "hypothesis", "experiment", "empirical", "genome",
        ],
        "base_score": 0.7,
    },
    "business": {
        "keywords": [
            "strategy", "analysis", "report", "presentation",
            "stakeholder", "revenue", "roi", "kpi",
        ],
        "base_score": 0.5,
    },
    "legal": {
        "keywords": [
            "statute", "regulation", "compliance", "liability",
            "jurisdiction", "precedent", "contractual", "tort",
        ],
        "base_score": 0.65,
    },
}

# Vocabulary sophistication markers
_SOPHISTICATED_WORDS = frozenset([
    "sophisticated", "comprehensive", "elaborate", "intricate", "nuanced",
    "methodology", "implementation", "optimization", "paradigm", "framework",
    "holistic", "multifaceted", "juxtapose", "synthesize", "extrapolate",
])

# Technical jargon markers
_TECHNICAL_TERMS = frozenset([
    "algorithm", "architecture", "infrastructure", "scalability",
    "optimization", "refactoring", "deployment", "integration",
    "configuration", "specification", "serialization", "idempotent",
    "polymorphism", "abstraction", "encapsulation",
])

# Intent indicator sets
_CREATIVE_INDICATORS = frozenset([
    "write a story", "create", "design", "invent", "imagine",
    "brainstorm", "compose", "draft",
])
_ANALYSIS_INDICATORS = frozenset([
    "analyze", "compare", "evaluate", "assess", "critique",
    "review", "audit", "benchmark",
])
_PROBLEM_INDICATORS = frozenset([
    "solve", "fix", "debug", "optimize", "improve",
    "troubleshoot", "diagnose", "resolve",
])
_TEACHING_INDICATORS = frozenset([
    "explain", "teach", "show how", "demonstrate",
    "walk me through", "help me understand", "tutor",
])

# Structural regex helpers
_CODE_BLOCK_RE = re.compile(r"```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_URL_RE = re.compile(r"https?://\S+")
_JSON_RE = re.compile(r"\{[^{}]*\}")
_LIST_MARKERS = ("1.", "2.", "a)", "b)", "\u2022", "- ", "* ")

# Agentic system-prompt keywords (mirrors routing.py)
_AGENTIC_SYSTEM_KEYWORDS = re.compile(
    r"\b("
    r"you are an? (?:ai |coding |software )?agent"
    r"|execute (?:commands?|tools?|code|tasks?)"
    r"|you (?:can|have access to|may) (?:use |call |run |execute )?"
    r"(?:tools?|functions?|commands?)"
    r"|tool[ _]?(?:use|call|execution)"
    r"|multi[- ]?step"
    r"|(?:read|write|edit|create|delete) files?"
    r"|run (?:commands?|shell|bash|terminal)"
    r"|code execution"
    r"|file (?:system|access)"
    r"|web ?search"
    r"|browser"
    r"|autonomous"
    r")\b",
    re.IGNORECASE,
)

# Reasoning markers (mirrors routing.py)
_REASONING_MARKERS = re.compile(
    r"\b("
    r"step[- ]by[- ]step"
    r"|think (?:through|carefully|deeply|about)"
    r"|chain[- ]of[- ]thought"
    r"|let'?s? reason"
    r"|reason(?:ing)? (?:about|through)"
    r"|prove (?:that|this|the)"
    r"|formal (?:proof|verification)"
    r"|mathematical(?:ly)? (?:prove|show|derive)"
    r"|derive (?:the|a|an)"
    r"|analyze the (?:tradeoffs?|trade-offs?|implications?|consequences?)"
    r"|compare and contrast"
    r"|what are the (?:pros? and cons?|advantages? and disadvantages?)"
    r"|evaluate (?:the|whether|if)"
    r"|critically (?:analyze|assess|examine)"
    r"|explain (?:why|how|the reasoning)"
    r"|work through"
    r"|break (?:this|it) down"
    r"|logical(?:ly)? (?:deduce|infer|conclude)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(messages: List[Any], system_prompt: str = "") -> Tuple[str, str]:
    """Extract user text and system text from OpenAI-format messages.

    Handles both dict messages and objects with ``role`` / ``content``
    attributes.

    Returns:
        (user_text, system_text) where each is the concatenated content
        for that role.
    """
    user_parts: List[str] = []
    system_parts: List[str] = [system_prompt] if system_prompt else []

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "")

        # content may be a list of parts (vision messages)
        if isinstance(content, list):
            text_bits = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_bits.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_bits.append(part)
            content = " ".join(text_bits)
        elif not isinstance(content, str):
            content = str(content) if content else ""

        if role == "user":
            user_parts.append(content)
        elif role in ("system", "developer"):
            system_parts.append(content)

    return " ".join(user_parts), " ".join(system_parts)


def _count_images(messages: List[Any]) -> int:
    """Count image_url / image parts across all messages."""
    count = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
        else:
            content = getattr(m, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("image_url", "image"):
                count += 1
    return count


def _count_tools(messages: List[Any]) -> int:
    """Count messages with role 'tool'."""
    count = 0
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        if role == "tool":
            count += 1
    return count


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class StructuralFeatureExtractor:
    """Extracts 25+ structural features from prompts for ML routing.

    All extraction is pure regex and string operations -- no ML inference,
    no external dependencies beyond ``re`` and ``math``.  Typical latency
    is well under 1 ms.

    The extractor accepts OpenAI-format messages (list of dicts with
    ``role`` / ``content`` keys) and an optional explicit system prompt.

    Two output formats are provided:

    * ``extract()`` -- returns a dict of named features with human-readable
      keys, suitable for logging and explainability.
    * ``extract_vector()`` -- returns a flat ``list[float]`` in a fixed
      order, normalised to [0, 1], suitable for feeding into an ML model.

    Feature dimensions are documented at the module level.
    """

    # Ordered list of feature names produced by extract_vector().
    # The order is stable across versions -- new features are appended.
    VECTOR_KEYS: List[str] = [
        # Pattern matching (dim 1)
        "pattern_score",
        "simple_pattern_hits",
        "medium_pattern_hits",
        "complex_pattern_hits",
        # Linguistic complexity (dim 2)
        "linguistic_score",
        "avg_sentence_length_norm",
        "sophisticated_word_count_norm",
        "technical_term_count_norm",
        # Domain complexity (dim 3)
        "domain_score",
        "domain_math",
        "domain_programming",
        "domain_science",
        "domain_business",
        "domain_legal",
        # Structural features (dim 4)
        "structure_score",
        "word_count_norm",
        "question_count_norm",
        "has_lists",
        "code_block_count_norm",
        "inline_code_count_norm",
        "url_count_norm",
        "json_structure_count_norm",
        # Intent detection (dim 5)
        "intent_score",
        "intent_creative",
        "intent_analysis",
        "intent_problem_solving",
        "intent_teaching",
        # Routing signals (dim 6)
        "agentic_score",
        "reasoning_marker_count_norm",
        "has_images",
        "token_estimate_norm",
        "tool_message_count_norm",
        "message_count_norm",
    ]

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        messages: List[Any],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Extract named features from an OpenAI-format message list.

        Args:
            messages: List of message dicts (``{"role": ..., "content": ...}``).
            system_prompt: Optional explicit system prompt text.  If a
                system message also appears in *messages*, both are used.

        Returns:
            A dict mapping feature names to their values.  Numeric features
            are floats; boolean features are ``0.0`` or ``1.0``.
        """
        user_text, system_text = _extract_text(messages, system_prompt)
        combined = f"{system_text} {user_text}".strip()
        combined_lower = combined.lower()

        features: Dict[str, Any] = {}

        # --- Dimension 1: Pattern matching ---
        p_score, p_simple, p_medium, p_complex = self._analyze_patterns(combined_lower)
        features["pattern_score"] = p_score
        features["simple_pattern_hits"] = p_simple
        features["medium_pattern_hits"] = p_medium
        features["complex_pattern_hits"] = p_complex

        # --- Dimension 2: Linguistic complexity ---
        l_score, avg_sl, soph_ct, tech_ct = self._analyze_linguistic(combined, combined_lower)
        features["linguistic_score"] = l_score
        features["avg_sentence_length"] = avg_sl
        features["sophisticated_word_count"] = soph_ct
        features["technical_term_count"] = tech_ct

        # --- Dimension 3: Domain complexity ---
        d_score, domain_hits = self._analyze_domain(combined_lower)
        features["domain_score"] = d_score
        features["domain_math"] = domain_hits.get("mathematics", 0)
        features["domain_programming"] = domain_hits.get("programming", 0)
        features["domain_science"] = domain_hits.get("science", 0)
        features["domain_business"] = domain_hits.get("business", 0)
        features["domain_legal"] = domain_hits.get("legal", 0)

        # --- Dimension 4: Structural features ---
        (
            s_score, word_count, question_count, has_lists,
            code_blocks, inline_codes, urls, json_structs,
        ) = self._analyze_structure(combined)
        features["structure_score"] = s_score
        features["word_count"] = word_count
        features["question_count"] = question_count
        features["has_lists"] = has_lists
        features["code_block_count"] = code_blocks
        features["inline_code_count"] = inline_codes
        features["url_count"] = urls
        features["json_structure_count"] = json_structs

        # --- Dimension 5: Intent detection ---
        i_score, i_creative, i_analysis, i_problem, i_teaching = (
            self._analyze_intent(combined_lower)
        )
        features["intent_score"] = i_score
        features["intent_creative"] = i_creative
        features["intent_analysis"] = i_analysis
        features["intent_problem_solving"] = i_problem
        features["intent_teaching"] = i_teaching

        # --- Dimension 6: Routing signals ---
        features["agentic_score"] = self._compute_agentic_score(
            messages, system_text,
        )
        features["reasoning_marker_count"] = len(
            _REASONING_MARKERS.findall(combined)
        )
        features["has_images"] = float(_count_images(messages) > 0)
        features["image_count"] = _count_images(messages)
        features["token_estimate"] = _estimate_tokens(combined)
        features["tool_message_count"] = _count_tools(messages)
        features["message_count"] = len(messages)

        return features

    def extract_vector(
        self,
        messages: List[Any],
        system_prompt: str = "",
    ) -> List[float]:
        """Extract a normalised float vector for ML input.

        The vector has a fixed length equal to ``len(VECTOR_KEYS)`` and
        every element is in [0, 1].  The order matches ``VECTOR_KEYS``.

        Args:
            messages: OpenAI-format message list.
            system_prompt: Optional system prompt text.

        Returns:
            A list of floats, one per feature, all in [0, 1].
        """
        raw = self.extract(messages, system_prompt)
        return self._normalise_to_vector(raw)

    # ------------------------------------------------------------------ #
    # Dimension 1 — Pattern matching                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_patterns(text_lower: str) -> Tuple[float, int, int, int]:
        """Score the prompt against simple / medium / complex pattern sets.

        Returns:
            (max_score, simple_hits, medium_hits, complex_hits)
        """
        max_score = 0.0
        simple_hits = 0
        medium_hits = 0
        complex_hits = 0

        for patterns in _SIMPLE_PATTERNS.values():
            for p in patterns:
                if p.search(text_lower):
                    simple_hits += 1
                    max_score = max(max_score, 0.2)

        for patterns in _MEDIUM_PATTERNS.values():
            for p in patterns:
                if p.search(text_lower):
                    medium_hits += 1
                    max_score = max(max_score, 0.5)

        for patterns in _COMPLEX_PATTERNS.values():
            for p in patterns:
                if p.search(text_lower):
                    complex_hits += 1
                    max_score = max(max_score, 0.8)

        return max_score, simple_hits, medium_hits, complex_hits

    # ------------------------------------------------------------------ #
    # Dimension 2 — Linguistic complexity                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_linguistic(
        text: str, text_lower: str,
    ) -> Tuple[float, float, int, int]:
        """Measure sentence length, vocabulary sophistication, and jargon.

        Returns:
            (score, avg_sentence_length, sophisticated_count, technical_count)
        """
        score = 0.0

        # Sentence complexity (count sentence-ending punctuation)
        sentence_ends = text.count(".") + text.count("!") + text.count("?")
        words = text.split()
        word_count = len(words)
        avg_sentence_length = word_count / max(sentence_ends, 1)

        if avg_sentence_length > 20:
            score += 0.2
        elif avg_sentence_length > 15:
            score += 0.1

        # Vocabulary sophistication
        sophisticated_count = sum(
            1 for w in _SOPHISTICATED_WORDS if w in text_lower
        )
        score += min(sophisticated_count * 0.1, 0.3)

        # Technical jargon density
        technical_count = sum(
            1 for t in _TECHNICAL_TERMS if t in text_lower
        )
        score += min(technical_count * 0.15, 0.4)

        return min(score, 1.0), avg_sentence_length, sophisticated_count, technical_count

    # ------------------------------------------------------------------ #
    # Dimension 3 — Domain complexity                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_domain(text_lower: str) -> Tuple[float, Dict[str, int]]:
        """Detect domain-specific keywords and compute domain complexity.

        Returns:
            (max_domain_score, {domain_name: keyword_hit_count})
        """
        max_score = 0.0
        hits: Dict[str, int] = {}

        for domain, cfg in _DOMAIN_COMPLEXITY.items():
            kw_count = sum(1 for kw in cfg["keywords"] if kw in text_lower)
            hits[domain] = kw_count
            if kw_count > 0:
                domain_score = cfg["base_score"] + (kw_count - 1) * 0.1
                max_score = max(max_score, min(domain_score, 1.0))

        return max_score, hits

    # ------------------------------------------------------------------ #
    # Dimension 4 — Structural features                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_structure(
        text: str,
    ) -> Tuple[float, int, int, float, int, int, int, int]:
        """Measure structural complexity: length, questions, lists, code, etc.

        Returns:
            (score, word_count, question_count, has_lists,
             code_block_count, inline_code_count, url_count, json_count)
        """
        score = 0.0
        word_count = len(text.split())

        # Length-based complexity
        if word_count > 100:
            score += 0.4
        elif word_count > 50:
            score += 0.25
        elif word_count > 20:
            score += 0.1
        elif word_count < 5:
            score += 0.05

        # Question marks
        question_count = text.count("?")
        if question_count > 2:
            score += 0.2
        elif question_count > 1:
            score += 0.1

        # Lists and enumerations
        has_lists = 1.0 if any(m in text for m in _LIST_MARKERS) else 0.0
        if has_lists:
            score += 0.15

        # Code blocks (triple backtick pairs)
        code_block_count = len(_CODE_BLOCK_RE.findall(text)) // 2
        # Inline code
        inline_code_count = len(_INLINE_CODE_RE.findall(text))
        if code_block_count > 0 or inline_code_count > 0:
            score += 0.2

        # URLs
        url_count = len(_URL_RE.findall(text))
        if url_count > 0:
            score += 0.05

        # JSON-like structures
        json_count = len(_JSON_RE.findall(text))
        if json_count > 0:
            score += 0.1

        return (
            min(score, 1.0),
            word_count,
            question_count,
            has_lists,
            code_block_count,
            inline_code_count,
            url_count,
            json_count,
        )

    # ------------------------------------------------------------------ #
    # Dimension 5 — Intent detection                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _analyze_intent(
        text_lower: str,
    ) -> Tuple[float, float, float, float, float]:
        """Detect the intent class(es) of the prompt.

        Returns:
            (score, creative, analysis, problem_solving, teaching)
            Each sub-indicator is 0.0 or 1.0.
        """
        score = 0.0
        creative = 0.0
        analysis = 0.0
        problem = 0.0
        teaching = 0.0

        if any(ind in text_lower for ind in _CREATIVE_INDICATORS):
            score += 0.6
            creative = 1.0

        if any(ind in text_lower for ind in _ANALYSIS_INDICATORS):
            score += 0.7
            analysis = 1.0

        if any(ind in text_lower for ind in _PROBLEM_INDICATORS):
            score += 0.5
            problem = 1.0

        if any(ind in text_lower for ind in _TEACHING_INDICATORS):
            score += 0.3
            teaching = 1.0

        return min(score, 1.0), creative, analysis, problem, teaching

    # ------------------------------------------------------------------ #
    # Dimension 6 — Routing signals                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_agentic_score(
        messages: List[Any],
        system_text: str,
    ) -> float:
        """Compute a lightweight agentic confidence score.

        Mirrors the logic in ``nadirclaw.routing.detect_agentic`` but
        operates on raw dicts so the feature extractor stays dependency-
        free.
        """
        score = 0.0

        # Tool-role messages
        tool_msgs = 0
        msg_count = len(messages)
        for m in messages:
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            if role == "tool":
                tool_msgs += 1
        if tool_msgs >= 1:
            score += 0.30

        # Assistant->tool cycles
        roles = []
        for m in messages:
            roles.append(
                m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            )
        cycles = 0
        i = 0
        while i < len(roles) - 1:
            if roles[i] == "assistant" and roles[i + 1] == "tool":
                cycles += 1
                i += 2
            else:
                i += 1
        if cycles >= 2:
            score += 0.20
        elif cycles == 1:
            score += 0.10

        # Long system prompt
        if len(system_text) > 500:
            score += 0.10

        # Agentic keywords in system prompt
        if system_text and _AGENTIC_SYSTEM_KEYWORDS.search(system_text):
            score += 0.20

        # Deep conversation
        if msg_count > 10:
            score += 0.10

        return min(score, 1.0)

    # ------------------------------------------------------------------ #
    # Vector normalisation                                                #
    # ------------------------------------------------------------------ #

    def _normalise_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Map raw features to a [0, 1] normalised vector.

        Uses sigmoid squashing for unbounded counts and direct passthrough
        for features already in [0, 1].
        """

        def _sigmoid_norm(x: float, midpoint: float = 5.0) -> float:
            """Soft normalisation via sigmoid: maps [0, inf) -> [0, 1)."""
            return 1.0 / (1.0 + math.exp(-((x - midpoint) / midpoint)))

        def _linear_norm(x: float, max_val: float) -> float:
            """Linear normalisation clipped to [0, 1]."""
            return _clamp(x / max_val)

        vec: List[float] = []
        for key in self.VECTOR_KEYS:
            if key == "pattern_score":
                vec.append(float(features["pattern_score"]))
            elif key == "simple_pattern_hits":
                vec.append(_linear_norm(features["simple_pattern_hits"], 10.0))
            elif key == "medium_pattern_hits":
                vec.append(_linear_norm(features["medium_pattern_hits"], 10.0))
            elif key == "complex_pattern_hits":
                vec.append(_linear_norm(features["complex_pattern_hits"], 10.0))
            elif key == "linguistic_score":
                vec.append(float(features["linguistic_score"]))
            elif key == "avg_sentence_length_norm":
                vec.append(_sigmoid_norm(features["avg_sentence_length"], 15.0))
            elif key == "sophisticated_word_count_norm":
                vec.append(_linear_norm(features["sophisticated_word_count"], 8.0))
            elif key == "technical_term_count_norm":
                vec.append(_linear_norm(features["technical_term_count"], 8.0))
            elif key == "domain_score":
                vec.append(float(features["domain_score"]))
            elif key == "domain_math":
                vec.append(_linear_norm(features["domain_math"], 5.0))
            elif key == "domain_programming":
                vec.append(_linear_norm(features["domain_programming"], 5.0))
            elif key == "domain_science":
                vec.append(_linear_norm(features["domain_science"], 5.0))
            elif key == "domain_business":
                vec.append(_linear_norm(features["domain_business"], 5.0))
            elif key == "domain_legal":
                vec.append(_linear_norm(features["domain_legal"], 5.0))
            elif key == "structure_score":
                vec.append(float(features["structure_score"]))
            elif key == "word_count_norm":
                vec.append(_sigmoid_norm(features["word_count"], 50.0))
            elif key == "question_count_norm":
                vec.append(_linear_norm(features["question_count"], 10.0))
            elif key == "has_lists":
                vec.append(float(features["has_lists"]))
            elif key == "code_block_count_norm":
                vec.append(_linear_norm(features["code_block_count"], 5.0))
            elif key == "inline_code_count_norm":
                vec.append(_linear_norm(features["inline_code_count"], 10.0))
            elif key == "url_count_norm":
                vec.append(_linear_norm(features["url_count"], 5.0))
            elif key == "json_structure_count_norm":
                vec.append(_linear_norm(features["json_structure_count"], 5.0))
            elif key == "intent_score":
                vec.append(float(features["intent_score"]))
            elif key == "intent_creative":
                vec.append(float(features["intent_creative"]))
            elif key == "intent_analysis":
                vec.append(float(features["intent_analysis"]))
            elif key == "intent_problem_solving":
                vec.append(float(features["intent_problem_solving"]))
            elif key == "intent_teaching":
                vec.append(float(features["intent_teaching"]))
            elif key == "agentic_score":
                vec.append(float(features["agentic_score"]))
            elif key == "reasoning_marker_count_norm":
                vec.append(_linear_norm(features["reasoning_marker_count"], 5.0))
            elif key == "has_images":
                vec.append(float(features["has_images"]))
            elif key == "token_estimate_norm":
                vec.append(_sigmoid_norm(features["token_estimate"], 500.0))
            elif key == "tool_message_count_norm":
                vec.append(_linear_norm(features["tool_message_count"], 10.0))
            elif key == "message_count_norm":
                vec.append(_sigmoid_norm(features["message_count"], 5.0))
            else:
                vec.append(0.0)

        return vec


# ---------------------------------------------------------------------------
# Standalone complexity scoring
# ---------------------------------------------------------------------------

# Dimension weights for the weighted complexity score.
# These match Horizen's EnhancedBERT analyzer.
_DIMENSION_WEIGHTS = {
    "patterns": 0.30,
    "linguistic": 0.25,
    "domain": 0.20,
    "structure": 0.15,
    "intent": 0.10,
}


def compute_complexity_score(features: Dict[str, Any]) -> float:
    """Produce a weighted 0-1 complexity score from structural features.

    This uses the same five-dimension weighted formula as Horizen's
    EnhancedBERT analyzer, followed by a calibration curve that prevents
    scores from clustering around 0.5.

    The routing signals dimension (agentic, reasoning, vision, etc.) is
    intentionally excluded -- those are used by the routing layer, not
    the complexity scorer.

    Args:
        features: Feature dict as returned by
            ``StructuralFeatureExtractor.extract()``.

    Returns:
        A float in [0, 1] where 0 is trivially simple and 1 is maximally
        complex.
    """
    raw = (
        features.get("pattern_score", 0.0) * _DIMENSION_WEIGHTS["patterns"]
        + features.get("linguistic_score", 0.0) * _DIMENSION_WEIGHTS["linguistic"]
        + features.get("domain_score", 0.0) * _DIMENSION_WEIGHTS["domain"]
        + features.get("structure_score", 0.0) * _DIMENSION_WEIGHTS["structure"]
        + features.get("intent_score", 0.0) * _DIMENSION_WEIGHTS["intent"]
    )

    return _calibrate_score(raw)


def _calibrate_score(raw_score: float) -> float:
    """Apply a calibration curve to spread the score distribution.

    Mirrors Horizen's ``_calibrate_score``:
    - Low range  [0.0, 0.3) is compressed  -> [0.0, 0.21)
    - Mid range  [0.3, 0.7] is linearly mapped -> [0.21, 0.70]
    - High range (0.7, 1.0] is expanded   -> (0.70, 1.0]

    This prevents scores from bunching in the middle and provides
    better discrimination at the tier boundaries.
    """
    if raw_score < 0.3:
        return raw_score * 0.7
    elif raw_score > 0.7:
        return _clamp(0.7 + (raw_score - 0.7) * 1.5)
    else:
        return 0.21 + (raw_score - 0.3) * 1.225
