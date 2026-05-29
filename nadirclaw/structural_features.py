"""Structural feature extractor for trained classifier.

Ported from NadirClaw's features.py — pure regex + string ops, no ML dependencies.
Extracts 33 normalized features across 6 dimensions for the GradientBoosting classifier.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pre-compiled pattern libraries
# ---------------------------------------------------------------------------

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

_SOPHISTICATED_WORDS = frozenset([
    "sophisticated", "comprehensive", "elaborate", "intricate", "nuanced",
    "methodology", "implementation", "optimization", "paradigm", "framework",
    "holistic", "multifaceted", "juxtapose", "synthesize", "extrapolate",
])

_TECHNICAL_TERMS = frozenset([
    "algorithm", "architecture", "infrastructure", "scalability",
    "optimization", "refactoring", "deployment", "integration",
    "configuration", "specification", "serialization", "idempotent",
    "polymorphism", "abstraction", "encapsulation",
])

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

_CODE_BLOCK_RE = re.compile(r"```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_URL_RE = re.compile(r"https?://\S+")
_JSON_RE = re.compile(r"\{[^{}]*\}")
_LIST_MARKERS = ("1.", "2.", "a)", "b)", "\u2022", "- ", "* ")

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
    user_parts: List[str] = []
    system_parts: List[str] = [system_prompt] if system_prompt else []

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            role = getattr(m, "role", "")
            content = getattr(m, "content", "")

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
    count = 0
    for m in messages:
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        if role == "tool":
            count += 1
    return count


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class StructuralFeatureExtractor:
    """Extracts 33 structural features from prompts for ML routing.

    Pure regex and string operations — no ML, no external deps beyond re and math.
    """

    VECTOR_KEYS: List[str] = [
        "pattern_score",
        "simple_pattern_hits",
        "medium_pattern_hits",
        "complex_pattern_hits",
        "linguistic_score",
        "avg_sentence_length_norm",
        "sophisticated_word_count_norm",
        "technical_term_count_norm",
        "domain_score",
        "domain_math",
        "domain_programming",
        "domain_science",
        "domain_business",
        "domain_legal",
        "structure_score",
        "word_count_norm",
        "question_count_norm",
        "has_lists",
        "code_block_count_norm",
        "inline_code_count_norm",
        "url_count_norm",
        "json_structure_count_norm",
        "intent_score",
        "intent_creative",
        "intent_analysis",
        "intent_problem_solving",
        "intent_teaching",
        "agentic_score",
        "reasoning_marker_count_norm",
        "has_images",
        "token_estimate_norm",
        "tool_message_count_norm",
        "message_count_norm",
    ]

    def extract(self, messages: List[Any], system_prompt: str = "") -> Dict[str, Any]:
        user_text, system_text = _extract_text(messages, system_prompt)
        combined = f"{system_text} {user_text}".strip()
        combined_lower = combined.lower()

        features: Dict[str, Any] = {}

        p_score, p_simple, p_medium, p_complex = self._analyze_patterns(combined_lower)
        features["pattern_score"] = p_score
        features["simple_pattern_hits"] = p_simple
        features["medium_pattern_hits"] = p_medium
        features["complex_pattern_hits"] = p_complex

        l_score, avg_sl, soph_ct, tech_ct = self._analyze_linguistic(combined, combined_lower)
        features["linguistic_score"] = l_score
        features["avg_sentence_length"] = avg_sl
        features["sophisticated_word_count"] = soph_ct
        features["technical_term_count"] = tech_ct

        d_score, domain_hits = self._analyze_domain(combined_lower)
        features["domain_score"] = d_score
        features["domain_math"] = domain_hits.get("mathematics", 0)
        features["domain_programming"] = domain_hits.get("programming", 0)
        features["domain_science"] = domain_hits.get("science", 0)
        features["domain_business"] = domain_hits.get("business", 0)
        features["domain_legal"] = domain_hits.get("legal", 0)

        (s_score, word_count, question_count, has_lists,
         code_blocks, inline_codes, urls, json_structs) = self._analyze_structure(combined)
        features["structure_score"] = s_score
        features["word_count"] = word_count
        features["question_count"] = question_count
        features["has_lists"] = has_lists
        features["code_block_count"] = code_blocks
        features["inline_code_count"] = inline_codes
        features["url_count"] = urls
        features["json_structure_count"] = json_structs

        i_score, i_creative, i_analysis, i_problem, i_teaching = self._analyze_intent(combined_lower)
        features["intent_score"] = i_score
        features["intent_creative"] = i_creative
        features["intent_analysis"] = i_analysis
        features["intent_problem_solving"] = i_problem
        features["intent_teaching"] = i_teaching

        features["agentic_score"] = self._compute_agentic_score(messages, system_text)
        features["reasoning_marker_count"] = len(_REASONING_MARKERS.findall(combined))
        features["has_images"] = float(_count_images(messages) > 0)
        features["image_count"] = _count_images(messages)
        features["token_estimate"] = _estimate_tokens(combined)
        features["tool_message_count"] = _count_tools(messages)
        features["message_count"] = len(messages)

        return features

    def extract_vector(self, messages: List[Any], system_prompt: str = "") -> List[float]:
        raw = self.extract(messages, system_prompt)
        return self._normalise_to_vector(raw)

    @staticmethod
    def _analyze_patterns(text_lower: str) -> Tuple[float, int, int, int]:
        max_score = 0.0
        simple_hits = medium_hits = complex_hits = 0

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

    @staticmethod
    def _analyze_linguistic(text: str, text_lower: str) -> Tuple[float, float, int, int]:
        score = 0.0
        sentence_ends = text.count(".") + text.count("!") + text.count("?")
        words = text.split()
        word_count = len(words)
        avg_sentence_length = word_count / max(sentence_ends, 1)

        if avg_sentence_length > 20:
            score += 0.2
        elif avg_sentence_length > 15:
            score += 0.1

        sophisticated_count = sum(1 for w in _SOPHISTICATED_WORDS if w in text_lower)
        score += min(sophisticated_count * 0.1, 0.3)

        technical_count = sum(1 for t in _TECHNICAL_TERMS if t in text_lower)
        score += min(technical_count * 0.15, 0.4)

        return min(score, 1.0), avg_sentence_length, sophisticated_count, technical_count

    @staticmethod
    def _analyze_domain(text_lower: str) -> Tuple[float, Dict[str, int]]:
        max_score = 0.0
        hits: Dict[str, int] = {}

        for domain, cfg in _DOMAIN_COMPLEXITY.items():
            kw_count = sum(1 for kw in cfg["keywords"] if kw in text_lower)
            hits[domain] = kw_count
            if kw_count > 0:
                domain_score = cfg["base_score"] + (kw_count - 1) * 0.1
                max_score = max(max_score, min(domain_score, 1.0))

        return max_score, hits

    @staticmethod
    def _analyze_structure(text: str) -> Tuple[float, int, int, float, int, int, int, int]:
        score = 0.0
        word_count = len(text.split())

        if word_count > 100:
            score += 0.4
        elif word_count > 50:
            score += 0.25
        elif word_count > 20:
            score += 0.1
        elif word_count < 5:
            score += 0.05

        question_count = text.count("?")
        if question_count > 2:
            score += 0.2
        elif question_count > 1:
            score += 0.1

        has_lists = 1.0 if any(m in text for m in _LIST_MARKERS) else 0.0
        if has_lists:
            score += 0.15

        code_block_count = len(_CODE_BLOCK_RE.findall(text)) // 2
        inline_code_count = len(_INLINE_CODE_RE.findall(text))
        if code_block_count > 0 or inline_code_count > 0:
            score += 0.2

        url_count = len(_URL_RE.findall(text))
        if url_count > 0:
            score += 0.05

        json_count = len(_JSON_RE.findall(text))
        if json_count > 0:
            score += 0.1

        return (min(score, 1.0), word_count, question_count, has_lists,
                code_block_count, inline_code_count, url_count, json_count)

    @staticmethod
    def _analyze_intent(text_lower: str) -> Tuple[float, float, float, float, float]:
        score = 0.0
        creative = analysis = problem = teaching = 0.0

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

    @staticmethod
    def _compute_agentic_score(messages: List[Any], system_text: str) -> float:
        score = 0.0
        tool_msgs = 0
        msg_count = len(messages)
        for m in messages:
            role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
            if role == "tool":
                tool_msgs += 1
        if tool_msgs >= 1:
            score += 0.30

        roles = []
        for m in messages:
            roles.append(m.get("role", "") if isinstance(m, dict) else getattr(m, "role", ""))
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

        if len(system_text) > 500:
            score += 0.10
        if system_text and _AGENTIC_SYSTEM_KEYWORDS.search(system_text):
            score += 0.20
        if msg_count > 10:
            score += 0.10

        return min(score, 1.0)

    def _normalise_to_vector(self, features: Dict[str, Any]) -> List[float]:
        def _sigmoid_norm(x: float, midpoint: float = 5.0) -> float:
            return 1.0 / (1.0 + math.exp(-((x - midpoint) / midpoint)))

        def _linear_norm(x: float, max_val: float) -> float:
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
