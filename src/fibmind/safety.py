"""Write-time safety scan.

Long-term memory is read back into future prompts, so anything that gets in
is replayed forever: a leaked key stays leaked, an injected instruction keeps
steering every later session. The scan runs inside admission, before the
duplicate check, and is deliberately conservative — patterns, not a model.

Three families:

- **secrets**: API keys, bearer tokens, private-key blocks, cloud credentials;
- **prompt injection**: text addressed to the model rather than about the work;
- **hidden text**: zero-width and bidi-override code points that render as
  nothing but survive copy-paste into a prompt.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._~+/=-]{24,}\b")),
    (
        "assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret)"
            r"\s*[:=]\s*['\"]?(?!\s*(?:<|\$\{|\$[A-Z_]|\*{3,}|xxx|redacted|placeholder|your[_-]))[A-Za-z0-9._~+/=-]{12,}"
        ),
    ),
    ("connection_string", re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s@/]{6,}@")),
)

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+|the\s+|your\s+|any\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules?)")),
    ("role_override", re.compile(r"(?i)\byou are now\b|\bact as (?:an? )?(?:unrestricted|jailbroken|dan)\b|\bnew system prompt\b")),
    ("system_tag", re.compile(r"(?i)</?\s*(?:system|assistant)\s*>|\[\s*(?:SYSTEM|INST)\s*\]|<\|(?:im_start|system|endoftext)\|>")),
    ("tool_coercion", re.compile(r"(?i)\b(?:always|must|you should)\s+(?:call|run|execute|invoke)\s+(?:the\s+)?(?:[\w-]+\s+){0,2}(?:tool|command|function)s?\b.*\b(?:without|before)\s+(?:asking|confirm)")),
    ("exfiltration", re.compile(r"(?i)\b(?:send|post|upload|exfiltrate)\b.{0,40}\b(?:api[_ -]?keys?|secrets?|credentials?|\.env|passwords?)\b.{0,40}\bto\b")),
    ("cjk_ignore_instructions", re.compile(r"忽略(?:之前|以上|上面|所有)?(?:的)?(?:指令|指示|提示|规则)|你现在是|忘记(?:之前|上面)的")),
)

# Zero-width and bidi-control code points. Legitimate text almost never needs
# them; injected text uses them to hide.
HIDDEN_CODEPOINTS = {
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # zero width space/joiners, LRM/RLM
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embeddings / overrides
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,  # word joiner, invisible operators
    0x2066, 0x2067, 0x2068, 0x2069,  # bidi isolates
    0xFEFF,  # BOM used mid-text
}
HIDDEN_ALLOWANCE = 0  # any occurrence is a finding
MAX_TAG_CHARS = 120


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    family: str  # secret | injection | hidden
    kind: str
    field: str  # title | content | tag
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return {"family": self.family, "kind": self.kind, "field": self.field, "excerpt": self.excerpt}


@dataclass(frozen=True, slots=True)
class SafetyReport:
    findings: tuple[SafetyFinding, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return bool(self.findings)

    def reason(self) -> str:
        if not self.findings:
            return "clean"
        families = sorted({item.family for item in self.findings})
        kinds = sorted({item.kind for item in self.findings})
        return f"safety: {', '.join(families)} ({', '.join(kinds)})"

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "reason": self.reason(), "findings": [f.to_dict() for f in self.findings]}


def scan_text(text: str, field_name: str) -> list[SafetyFinding]:
    findings: list[SafetyFinding] = []
    if not text:
        return findings
    for kind, pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(SafetyFinding("secret", kind, field_name, _redact(match.group(0))))
    for kind, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(SafetyFinding("injection", kind, field_name, _excerpt(match.group(0))))
    hidden = sorted({ord(ch) for ch in text if ord(ch) in HIDDEN_CODEPOINTS or unicodedata.category(ch) == "Cf" and ord(ch) not in (0x200D,)})
    if len(hidden) > HIDDEN_ALLOWANCE:
        findings.append(SafetyFinding("hidden", "invisible_codepoints", field_name, " ".join(f"U+{cp:04X}" for cp in hidden[:6])))
    return findings


def scan_candidate(title: str, content: str, tags: tuple[str, ...] = ()) -> SafetyReport:
    findings = scan_text(title, "title") + scan_text(content, "content")
    for tag in tags:
        findings += scan_text(tag[:MAX_TAG_CHARS], "tag")
    return SafetyReport(tuple(findings))


def _redact(secret: str) -> str:
    if len(secret) <= 12:
        return "*" * len(secret)
    return f"{secret[:6]}…{secret[-4:]}"


def _excerpt(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"
