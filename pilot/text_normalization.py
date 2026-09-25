#!/usr/bin/env python3
"""Deterministic, conservative text normalization for the business ER task.

This module is the preprocessing / data-quality layer.  It never decides that two
records are the same entity: it produces reproducible, *reversible* normalized
views of ``business_name`` and ``business_address`` plus explicit missing-value
and quality flags, and leaves every merge decision to the matching model.

Design contract
---------------
* Originals are always preserved (``*_raw`` columns).
* Normalization is deterministic and idempotent: no randomness, no locale
  dependent calls, no dependence on dictionary iteration order, and
  ``normalize(normalize(x)) == normalize(x)`` for every generated field.
* Nothing is invented.  Missing stays missing (null plus a boolean flag); no
  imputation, no substitution, no country whitelist.
* A normalization that could merge two different businesses is never applied to
  a comparison key.  Order-insensitive signatures and stop-word removal exist
  for *blocking keys only* and are marked as such by name.

Measured rationale (2026-09-25 streaming profile of the challenge TSVs) is in
``pilot/DATASET_INVENTORY.md``; the same evidence that motivates each rule is
documented on the rule-bearing constants below.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

NORMALIZER_VERSION = "1.0.0"

SOURCE_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "business_name_raw",
    "business_name_display",
    "business_name_key",
    "business_name_signature",
    "business_name_legal_forms",
    "business_name_trade_name",
    "business_name_web_token",
    "business_name_is_missing",
    "business_name_flags",
    "business_address_raw",
    "business_address_display",
    "business_address_key",
    "business_address_signature",
    "business_address_blocking_tokens",
    "business_address_numbers",
    "business_address_postal_codes",
    "business_address_is_missing",
    "business_address_flags",
    "country_raw",
    "country_key",
    "country_is_missing",
    "country_flags",
    "qa_flags",
)

# ---------------------------------------------------------------------------
# Unicode, whitespace and control-character rules  (C1, C2, A1)
# ---------------------------------------------------------------------------

# C1/A1: drop C0/C1 controls other than plain whitespace, which is folded to a
# single space below.  Guards the TSV contract: a surviving tab or newline would
# corrupt a ``\t``-separated, unquoted output row.
CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
# ``\s`` is Unicode aware for str patterns, so NBSP (U+00A0), narrow no-break
# space (U+202F) and ideographic space (U+3000) are covered.
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
DIGIT_RE = re.compile(r"\d+", flags=re.UNICODE)
# A2: canonical spacing for separators.  ``/`` is preserved (19 1/2 must not
# become 19 1, 2) and only the space runs around it are collapsed.
SEPARATOR_SPACE_RE = re.compile(r"\s*([,;|])\s*")
# ``/`` never gains a space: "19 1/2" must stay "19 1/2" and "G-3 / 571" tidies
# to "G-3/571".  Inserting a space after the slash corrupted fractions.
SLASH_SPACE_RE = re.compile(r"\s*/\s*")
# A3: add a space after a letter-dot only, so ``Opp.HOTEL`` tidies up while
# ``1.5`` and ``No.204`` are left numerically intact.
LETTER_DOT_RE = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")
TRAILING_SPACE_RE = re.compile(r"\s+([,;./|])")

# C5: trailing web segment of a name.  Only explicit markers are split so a
# legitimate token such as ``Sunrise.in`` is not silently discarded.
WEB_MARKER_RE = re.compile(r"(?:\|\s*|\bhttps?://\s*|\bwww\.)", flags=re.IGNORECASE)
# The candidate tail must actually look like a web reference, otherwise a name
# such as ``Chain | Store`` would lose real content.
URLISH_RE = re.compile(
    r"(?:https?://|www\.|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*"
    r"\.(?:com|net|org|co|io|us|uk|fr|in|edu|gov)\b)",
    flags=re.IGNORECASE,
)
# C6: trade-name marker.  ``Dba`` on its own is left alone (it would empty the
# core) and is flagged instead.
DBA_SPLIT_RE = re.compile(r"\s*\(?\s*\bdba\b\s*\)?\s*[:.\-]?\s*", flags=re.IGNORECASE)

# ---------------------------------------------------------------------------
# Missing-value rules  (M1-M4)
# ---------------------------------------------------------------------------

# M1/M2: full-field sentinels only (never a substring match), compared after
# NFKC + casefold + trim.  Measured on the challenge TSVs: no such tokens exist
# in business_name or business_address apart from two ``na`` company names in
# 120k rows of test_source3, so the list stays deliberately small.
COMMON_SENTINELS: frozenset = frozenset(
    {
        "",
        "n/a",
        "n.a.",
        "na.",
        "nan",
        "null",
        "none",
        "nil",
        "unknown",
        "unk",
        "not available",
        "not applicable",
        "missing",
        "empty",
        "n.d.",
        "-",
        "--",
        "---",
        "?",
        "??",
    }
)
# M2: ``na``/``not found`` are only treated as missing for addresses.  A company
# literally named "NA" is implausible but the cost of a false "missing" flag is
# zero, whereas the cost of inventing a name is not, so the name side stays
# conservative and flags the value instead.
ADDRESS_ONLY_SENTINELS: frozenset = frozenset({"na", "n.a", "not found", "nil"})

MISSING_FLAG = "missing_value"
SENTINEL_FLAG = "sentinel_value"
NON_ALNUM_FLAG = "non_alphanumeric_value"

# ---------------------------------------------------------------------------
# Company-name rules  (C8-C10)
# ---------------------------------------------------------------------------

# C8: single-letter runs are reassembled before anything else so ``L.L.C.``
# becomes ``llc`` instead of the token triple ``l l c``.
LEGAL_FORM_LETTER_FORMS: frozenset = frozenset(
    {
        "ab", "ag", "as", "bv", "co", "epe", "gmbh", "inc", "kg", "kk", "lda",
        "llda", "llc", "llp", "lp", "ltd", "oy", "plc", "pllc", "sa", "sarl",
        "sas", "srl",
    }
)

# C9: explicit alias table.  Tokens absent from this table are never treated as
# legal forms.  Ambiguous short tokens are excluded on purpose and the edge rule
# below additionally requires a length of three or more, so "Vishal As",
# "Spa & Co" and "SA" keep their tokens: a name that loses its only content
# token is worse than a name that keeps a legal form.
LEGAL_FORM_ALIASES: Dict[str, str] = {
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "pllc": "plc",
    "plc": "plc",
    "corp": "corp",
    "inc": "inc",
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasl",
    "eurl": "eurl",
    "sci": "sci",
    "snc": "snc",
    "sca": "sca",
    "gmbh": "gmbh",
    "ag": "ag",
    "nv": "nv",
    "bv": "bv",
    "oy": "oy",
    "ab": "ab",
    "kg": "kg",
    "kft": "kft",
    "pte": "pte",
    "ll": "ll",
    "lda": "lda",
    "llda": "llda",
    "srl": "srl",
    # Non-Latin legal forms observed in the India records of this dataset.
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "एलएलपी": "llp",
    "பிரைவேட்": "pvt",
    "லிமிடெட்": "ltd",
}

# C9: legal forms recognised only as a contiguous run of trailing tokens, e.g.
# "Private Limited" / "SARL" / "EURL".  An interior occurrence is never removed,
# which protects distinguishing names such as "Limited Edition Cafe".
LEGAL_FORM_RUNS: Tuple[Tuple[str, ...], ...] = (
    ("private", "limited"),
    ("pvt", "ltd"),
    ("private", "ltd"),
    ("l", "l", "c"),
    ("i", "n", "c"),
    ("l", "l", "p"),
    ("p", "l", "c"),
    ("s", "a", "r", "l"),
    ("e", "u", "r", "l"),
    ("s", "a", "s", "u"),
)

# C9a: Source 3 prepends legal forms ("LLC Hernandez Colonial Redwood",
# "Pvt. EFS Print Ventures Ltd."), so they are stripped from the leading edge
# too - but only when the name still has this many non-legal content tokens.
# Without the guard, "Limited Edition Cafe" would collapse onto "Edition Cafe",
# which is precisely the kind of invented merge this module must not create.
LEADING_FORM_MIN_CONTENT_TOKENS = 3

# "spa" is deliberately absent from the alias table: S1-854849282 is the business
# "Spa & Co", so stripping "spa" would empty a real name.

# C4/C11: dropped from the name key and from name blocking tokens.
NAME_STOP_TOKENS: frozenset = frozenset({"the", "and"})

# C11: dropped from name *blocking tokens* only.  These carry no discriminating
# power, so excluding them stops huge generic blocks such as "india" or
# "group" while leaving the name key and signature intact.
NAME_BLOCKING_STOP_TOKENS: frozenset = frozenset(
    {
        "and", "the", "company", "corporation", "incorporated", "limited",
        "private", "holdings", "holding", "international", "group", "india",
        "indian", "united", "states", "usa", "america", "of",
    }
)

# ---------------------------------------------------------------------------
# Address rules  (A4-A8)
# ---------------------------------------------------------------------------

# A4/A5: multi-word US state names are replaced before tokenization, longest
# phrase first, in the key only.
US_STATES: Dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
_US_STATE_PHRASES: Tuple[Tuple[str, str], ...] = tuple(
    sorted(US_STATES.items(), key=lambda item: (-len(item[0]), item[0]))
)

# A4: US street types, unit designators and directionals, key only.
ADDRESS_TOKEN_ALIASES: Dict[str, str] = {
    "street": "st", "str": "st", "strt": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "aven": "ave", "av": "ave", "ave.": "ave",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd",
    "drive": "dr", "drv": "dr", "cir": "cir", "circle": "cir",
    "lane": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl",
    "plaza": "plz", "plz": "plz", "squareplaza": "plz", "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl",
    "way": "way", "apartment": "apt", "apartments": "apt", "appt": "apt", "apt": "apt",
    "suite": "ste", "ste": "ste", "floor": "fl", "flr": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg", "bldgno": "bldg", "buildingno": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    # India forms observed in the India records.
    "house": "hno", "hno": "hno", "number": "no", "no": "no", "plot": "plot",
    "kh": "kh", "door": "door", "shop": "shop", "sector": "sec", "section": "sec",
    "sec": "sec", "district": "dist", "dist": "dist", "layout": "layout",
    "colony": "colony", "society": "soc", "soc": "soc", "nagar": "nagar",
    "marg": "marg", "roadno": "rd",
    # France forms observed in the test records.
    "rue": "rue", "avenue": "ave", "boulevard": "blvd", "allee": "allee",
    "chemin": "chemin", "impasse": "impasse", "quai": "quai", "place": "pl",
    "zone": "zi", "industrielle": "zi", "cedex": "cedex", "bis": "bis",
}

# A7: dropped from the address *blocking token set* only.  Landmark and
# municipal boilerplate must not generate blocks of their own.
ADDRESS_BLOCKING_STOP_TOKENS: frozenset = frozenset(
    {
        "the", "and", "at", "of", "in", "near", "nr", "opposite", "opp", "beside",
        "behind", "front", "india", "indian", "united", "states", "america",
        "france", "french", "sector", "sec", "district", "dist", "layout",
        "complex", "building", "bldg", "floor", "fl", "first", "second", "third",
        "fourth", "fifth", "ground", "co", "c/o", "s/o", "c/o:-", "via",
    }
)

# A6: US ZIP / ZIP+4, India PIN, France postal code.
US_ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")
INDIA_PIN_RE = re.compile(r"^[1-9]\d{5}$")
FRANCE_POSTAL_RE = re.compile(r"^\d{5}$")

FLAG_LEGAL_FORM_ONLY = "key_is_legal_form_only"
FLAG_LEGAL_FORMS_STRIPPED = "legal_forms_stripped"
FLAG_WEB_TOKEN = "web_token_extracted"
FLAG_TRADE_NAME = "trade_name_extracted"
FLAG_REPEAT_TOKEN = "consecutive_repeat_collapsed"
FLAG_ALL_CAPS = "all_caps"
FLAG_NON_ASCII = "non_ascii"
FLAG_LEADING_JUNK = "leading_non_alphanumeric"
FLAG_TRAILING_JUNK = "trailing_non_alphanumeric"
FLAG_EMPTY_KEY = "empty_key"
FLAG_NO_DIGITS = "no_digits"
FLAG_HAS_DIGITS = "has_digits"
FLAG_STATE_EXPANDED = "state_name_expanded"
FLAG_SENTINEL = SENTINEL_FLAG
FLAG_MISSING = MISSING_FLAG


# ---------------------------------------------------------------------------
# Low-level text operations
# ---------------------------------------------------------------------------


def clean_text(value: Optional[str]) -> str:
    """Return NFKC-normalized text with controls removed and spaces collapsed.

    Case is preserved: casing is source-specific evidence, so it is folded only
    inside the derived keys (C3).
    """
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = CONTROL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def fold_case_and_marks(text: str) -> str:
    """Case-fold and strip Latin accents while preserving other scripts' marks.

    Devanagari and Tamil vowel signs are combining marks, so removing every mark
    would destroy roughly 15% of the names in this dataset.  Only marks that
    decorate an ASCII/Latin base character are dropped.  Mirrors
    ``pilot.er_common.ascii_fold`` so keys stay consistent with the frozen model.
    """
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    output: List[str] = []
    for character in decomposed:
        is_mark = unicodedata.category(character).startswith("M")
        latin_base = bool(output) and ("A" <= output[-1].upper() <= "Z")
        if is_mark and latin_base:
            continue
        output.append(character)
    return "".join(output)


def tokenize(text: str) -> List[str]:
    """Split on non-alphanumerics, keeping combining marks attached to a base."""
    folded = fold_case_and_marks(text)
    tokens: List[str] = []
    current: List[str] = []
    for character in folded:
        if character.isalnum() or (unicodedata.category(character).startswith("M") and current):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def collapse_consecutive_repeats(tokens: Sequence[str]) -> Tuple[List[str], int]:
    """Collapse a token immediately repeated (``l l c`` stays, ``a a`` -> ``a``).

    Only *consecutive* duplicates are collapsed and only one repetition is
    removed per run, so genuinely repeated content is never lost (C7).
    """
    collapsed: List[str] = []
    removed = 0
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            removed += 1
            continue
        collapsed.append(token)
    return collapsed, removed


def extract_numbers(text: str) -> List[str]:
    """Return digit runs in order, with a zero-stripped comparison variant."""
    return [match.group(0) for match in DIGIT_RE.finditer(fold_case_and_marks(text))]


def numbers_comparison_key(numbers: Sequence[str]) -> List[str]:
    """Zero-stripped numbers for equality tests; stored values are untouched.

    ``00567`` and ``567`` compare equal here, but neither source value is
    rewritten, so an India PIN that legitimately starts with ``0`` is not lost.
    """
    return [number.lstrip("0") or "0" for number in numbers]


def postal_codes(numbers: Sequence[str]) -> List[str]:
    """Select plausible postal codes for US, India and France, order preserved."""
    selected: List[str] = []
    for number in numbers:
        compact = number.replace("-", "")
        if US_ZIP_RE.match(number) or FRANCE_POSTAL_RE.match(compact):
            selected.append(number)
        elif INDIA_PIN_RE.match(number):
            selected.append(number)
    return selected


def is_all_caps(value: str) -> bool:
    """True when the value has letters and no lowercase letters."""
    return any(character.isalpha() for character in value) and value == value.upper()


def has_non_ascii(value: str) -> bool:
    return any(ord(character) > 127 for character in value)


# The challenge files are unquoted, tab-separated, so a value containing a tab,
# newline or double quote would either split one row into two or make an unquoted
# reader fail.  Real names do contain quotes, so the writer is configured with an
# escape character and the reader below inverts it.  With QUOTE_NONE, csv escapes
# the delimiter, the quote character, the escape character itself and every
# character of the line terminator, which is exactly the scheme implemented by
# :func:`unescape_tsv`.  The backslash entry must stay first.
_TSV_ESCAPES = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r", '"': '\\"'}


def escape_tsv(value: object) -> str:
    """Escape backslashes, tabs, newlines and quotes so a value stays on one row.

    Provided for callers that must serialize a single normalized field themselves;
    :func:`write_normalized_tsv` lets the csv writer do the same work.
    """
    text = "" if value is None else str(value)
    for character, replacement in _TSV_ESCAPES.items():
        text = text.replace(character, replacement)
    return text


def unescape_tsv(value: str) -> str:
    """Inverse of the escaping applied by :func:`write_normalized_tsv`."""
    output: List[str] = []
    index = 0
    length = len(value)
    simple = {"t": "\t", "n": "\n", "r": "\r", '"': '"', "\\": "\\"}
    while index < length:
        character = value[index]
        if character == "\\" and index + 1 < length and value[index + 1] in simple:
            output.append(simple[value[index + 1]])
            index += 2
            continue
        output.append(character)
        index += 1
    return "".join(output)


def signature(tokens: Sequence[str]) -> str:
    """Order-insensitive key for *blocking only*.

    Measured on train_source1, 26% of rows share a name signature with another
    row and generic clusters such as ``meridian`` or ``summit`` contain dozens of
    unrelated businesses.  This value must never be used as a merge rule.
    """
    return " ".join(sorted(tokens))


# ---------------------------------------------------------------------------
# Missing-value detection  (M1-M4)
# ---------------------------------------------------------------------------


def sentinel_kind(value: Optional[str], field_name: str) -> Optional[str]:
    """Return ``"missing"``/``"sentinel"``/``None`` for a raw field value."""
    cleaned = clean_text(value)
    if not cleaned:
        return "missing"
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    sentinels = COMMON_SENTINELS
    if field_name == "business_address":
        sentinels = COMMON_SENTINELS | ADDRESS_ONLY_SENTINELS
    if folded in sentinels:
        if folded == "":
            return "missing"
        if not TOKEN_RE.search(cleaned):
            return NON_ALNUM_FLAG
        return SENTINEL_FLAG
    if not TOKEN_RE.search(cleaned):
        return NON_ALNUM_FLAG
    return None


def is_missing(value: Optional[str], field_name: str) -> bool:
    """True when the value carries no information (M1/M2)."""
    return sentinel_kind(value, field_name) is not None


# ---------------------------------------------------------------------------
# Company-name normalization  (C1-C12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyNormalization:
    """Normalized views of one company name; every part is reversible."""

    raw: str
    display: str
    key: str
    signature: str
    legal_forms: Tuple[str, ...]
    trade_name: str
    web_token: str
    blocking_tokens: Tuple[str, ...]
    is_missing: bool
    flags: Tuple[str, ...]


def _split_web_segment(text: str) -> Tuple[str, str]:
    """Split a trailing web/domain segment off a name (C5).

    Only a tail that itself looks like a web reference is removed, and a name
    that is *entirely* a domain (``wilfordhancock.com``) is kept intact because
    the domain is the business name there, not metadata.
    """
    candidate = text.strip()
    if not candidate:
        return "", ""
    for match in WEB_MARKER_RE.finditer(candidate):
        if match.start() == 0:
            continue
        tail = candidate[match.start() :].strip(" |")
        if tail and URLISH_RE.search(tail):
            return candidate[: match.start()].strip(" |,"), tail
    return candidate, ""


def _split_trade_name(text: str) -> Tuple[str, str]:
    """Split a ``DBA`` trade-name clause off a name (C6)."""
    for match in DBA_SPLIT_RE.finditer(text):
        head = text[: match.start()].strip()
        tail = re.sub(r"^(?:of|the)\s+", "", text[match.end() :].strip(), flags=re.IGNORECASE)
        if head and tail:
            return head, tail
    return text, ""


def _merge_letter_legal_forms(tokens: Sequence[str]) -> List[str]:
    """Reassemble ``l l c`` -> ``llc`` before anything else (C8).

    Runs of two or three single-character tokens are merged only when the result
    is a known legal form, so ``A B C`` stays three tokens.
    """
    merged: List[str] = []
    index = 0
    length = len(tokens)
    while index < length:
        for size in (3, 2):
            window = tokens[index : index + size]
            if len(window) == size and all(len(token) == 1 for token in window):
                candidate = "".join(window)
                if candidate in LEGAL_FORM_LETTER_FORMS:
                    merged.append(candidate)
                    index += size
                    break
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def _strip_edge_legal_forms(
    tokens: Sequence[str], strip_legal_forms: bool
) -> Tuple[List[str], List[str]]:
    """Remove legal forms from both edges only, with an alias table (C9/C9a).

    Trailing forms are always removed.  A leading form is removed only when at
    least ``LEADING_FORM_MIN_CONTENT_TOKENS`` non-legal tokens survive, which
    keeps "Limited Edition Cafe" intact while still matching
    "LLC Hernandez Colonial Redwood" to "Hernandez Colonial Redwood".
    """
    if not strip_legal_forms:
        return list(tokens), []
    working = list(tokens)
    found: List[str] = []

    def content_count(candidate: Sequence[str]) -> int:
        return sum(1 for token in candidate if token not in LEGAL_FORM_ALIASES)

    while working:
        run = next((run for run in LEGAL_FORM_RUNS if tuple(working[-len(run) :]) == run), None)
        if run is not None:
            found.append(LEGAL_FORM_ALIASES.get(run[-1], run[-1]))
            del working[-len(run) :]
            continue
        if working[-1] in LEGAL_FORM_ALIASES and len(working[-1]) > 2:
            found.append(LEGAL_FORM_ALIASES[working[-1]])
            working.pop()
            continue
        break

    while working:
        run = next((run for run in LEGAL_FORM_RUNS if tuple(working[: len(run)]) == run), None)
        if run is not None:
            remaining = working[len(run) :]
            if content_count(remaining) >= LEADING_FORM_MIN_CONTENT_TOKENS:
                found.append(LEGAL_FORM_ALIASES.get(run[0], run[0]))
                del working[: len(run)]
                continue
            break
        if working[0] in LEGAL_FORM_ALIASES and len(working[0]) > 2:
            remaining = working[1:]
            if content_count(remaining) >= LEADING_FORM_MIN_CONTENT_TOKENS:
                found.append(LEGAL_FORM_ALIASES[working[0]])
                working.pop(0)
                continue
            break
        break

    return working, found


def normalize_company(raw: Optional[str], *, strip_legal_forms: bool = True) -> CompanyNormalization:
    """Normalize one company name deterministically (C1-C12).

    ``display`` keeps the original case and word order, ``key`` is the folded
    comparison form, and ``signature`` is blocking-only.
    """
    original = "" if raw is None else str(raw)
    flags: List[str] = []
    kind = sentinel_kind(original, "business_name")
    is_missing_value = kind is not None
    if kind is not None:
        flags.append(FLAG_MISSING if kind == "missing" else kind)

    display_source = clean_text(original)
    if is_missing_value:
        return CompanyNormalization(
            raw=original,
            display="",
            key="",
            signature="",
            legal_forms=(),
            trade_name="",
            web_token="",
            blocking_tokens=(),
            is_missing=True,
            flags=tuple(flags),
        )

    if is_all_caps(display_source):
        flags.append(FLAG_ALL_CAPS)
    if has_non_ascii(display_source):
        flags.append(FLAG_NON_ASCII)
    if display_source and not display_source[0].isalnum():
        flags.append(FLAG_LEADING_JUNK)
    if display_source and not display_source[-1].isalnum():
        flags.append(FLAG_TRAILING_JUNK)

    name_part, web_token = _split_web_segment(display_source)
    if web_token:
        flags.append(FLAG_WEB_TOKEN)
    name_part, trade_name = _split_trade_name(name_part)
    if trade_name:
        flags.append(FLAG_TRADE_NAME)

    display = name_part
    key_text = fold_case_and_marks(display)
    # C4: a standalone ampersand becomes "and" (later dropped as a stop word so
    # "Lee and Lawson" == "Lee & Lawson"), while an intra-word ampersand is
    # deleted so "AT&T" -> "att".
    key_text = re.sub(r"(?:(?<=\s)|(?<=^))\s*&\s*(?=\s)", " and ", key_text)
    key_text = key_text.replace("&", "")
    key_text = re.sub(r"['\u2019\u02bc]", "", key_text)
    tokens = _merge_letter_legal_forms(tokenize(key_text))
    tokens, repeats = collapse_consecutive_repeats(tokens)
    if repeats:
        flags.append(FLAG_REPEAT_TOKEN)
    tokens = [token for token in tokens if token not in NAME_STOP_TOKENS]

    stripped, legal_forms = _strip_edge_legal_forms(tokens, strip_legal_forms)
    if legal_forms:
        flags.append(FLAG_LEGAL_FORMS_STRIPPED)
    if not stripped:
        flags.append(FLAG_LEGAL_FORM_ONLY)
        stripped = list(tokens)
        legal_forms = []
    key_tokens = stripped
    if key_tokens and all(
        token in LEGAL_FORM_ALIASES or len(token) <= 2 for token in key_tokens
    ):
        flags.append(FLAG_LEGAL_FORM_ONLY)
    if not key_tokens:
        flags.append(FLAG_EMPTY_KEY)

    blocking_tokens = tuple(
        token for token in key_tokens if token not in NAME_BLOCKING_STOP_TOKENS and len(token) >= 3
    )

    return CompanyNormalization(
        raw=original,
        display=display,
        key=" ".join(key_tokens),
        signature=signature(key_tokens),
        legal_forms=tuple(legal_forms),
        trade_name=trade_name,
        web_token=web_token,
        blocking_tokens=blocking_tokens,
        is_missing=False,
        flags=tuple(dict.fromkeys(flags)),
    )


# ---------------------------------------------------------------------------
# Address normalization  (A1-A9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddressNormalization:
    """Normalized views of one address; every part is reversible."""

    raw: str
    display: str
    key: str
    signature: str
    blocking_tokens: Tuple[str, ...]
    numbers: Tuple[str, ...]
    postal_codes: Tuple[str, ...]
    is_missing: bool
    flags: Tuple[str, ...]


def _tidy_separators(text: str) -> str:
    """Canonical spacing around separators; ``/`` is preserved (A2/A3)."""
    tidied = SEPARATOR_SPACE_RE.sub(r"\1 ", text)
    tidied = SLASH_SPACE_RE.sub("/", tidied)
    tidied = LETTER_DOT_RE.sub(". ", tidied)
    tidied = TRAILING_SPACE_RE.sub(r"\1", tidied)
    tidied = SPACE_RE.sub(" ", tidied)
    return tidied.strip(" ,")


def normalize_address(raw: Optional[str]) -> AddressNormalization:
    """Normalize one address deterministically (A1-A9)."""
    original = "" if raw is None else str(raw)
    flags: List[str] = []
    kind = sentinel_kind(original, "business_address")
    is_missing_value = kind is not None
    if kind is not None:
        flags.append(FLAG_MISSING if kind == "missing" else kind)

    display_source = clean_text(original)
    if is_missing_value:
        return AddressNormalization(
            raw=original,
            display="",
            key="",
            signature="",
            blocking_tokens=(),
            numbers=(),
            postal_codes=(),
            is_missing=True,
            flags=tuple(flags),
        )

    if is_all_caps(display_source):
        flags.append(FLAG_ALL_CAPS)
    if has_non_ascii(display_source):
        flags.append(FLAG_NON_ASCII)
    if DIGIT_RE.search(display_source):
        flags.append(FLAG_HAS_DIGITS)
    else:
        flags.append(FLAG_NO_DIGITS)

    display = _tidy_separators(display_source)
    folded = fold_case_and_marks(display)
    for phrase, abbreviation in _US_STATE_PHRASES:
        pattern = re.compile(rf"(?<![^\W_]){re.escape(phrase)}(?![^\W_])")
        replaced, count = pattern.subn(abbreviation, folded)
        if count:
            folded = replaced
            flags.append(FLAG_STATE_EXPANDED)
    tokens = tokenize(folded)
    tokens, repeats = collapse_consecutive_repeats(tokens)
    if repeats:
        flags.append(FLAG_REPEAT_TOKEN)
    tokens = [ADDRESS_TOKEN_ALIASES.get(token, token) for token in tokens]
    numbers = extract_numbers(display)
    key_tokens = [token for token in tokens if token]
    if not key_tokens:
        flags.append(FLAG_EMPTY_KEY)

    blocking_tokens = tuple(
        token
        for token in key_tokens
        if token not in ADDRESS_BLOCKING_STOP_TOKENS and len(token) >= 2
    )

    return AddressNormalization(
        raw=original,
        display=display,
        key=" ".join(key_tokens),
        signature=signature(key_tokens),
        blocking_tokens=blocking_tokens,
        numbers=tuple(numbers),
        postal_codes=tuple(postal_codes(numbers)),
        is_missing=False,
        flags=tuple(dict.fromkeys(flags)),
    )


# ---------------------------------------------------------------------------
# Country  (open set, no whitelist)
# ---------------------------------------------------------------------------


def normalize_country(raw: Optional[str]) -> Tuple[str, bool, Tuple[str, ...]]:
    """Return ``(key, is_missing, flags)`` treating country as an open string.

    The test set contains ``France``, which never appears in training, so no
    whitelist, aliasing or filtering is applied.  US variants are *not* merged
    either: that decision belongs to the matching stage.
    """
    original = "" if raw is None else str(raw)
    kind = sentinel_kind(original, "country")
    if kind is not None:
        return "", True, (FLAG_MISSING if kind == "missing" else kind,)
    cleaned = clean_text(original)
    return unicodedata.normalize("NFKC", cleaned).casefold(), False, ()


# ---------------------------------------------------------------------------
# Record and stream APIs
# ---------------------------------------------------------------------------


def normalize_record(
    record: Mapping[str, object], *, strip_legal_forms: bool = True
) -> Dict[str, object]:
    """Normalize one source row into the output column contract.

    Originals are preserved under ``*_raw`` and every judgement call is exposed
    through ``*_flags`` / ``qa_flags`` rather than applied silently.
    """
    company = normalize_company(record.get("business_name"), strip_legal_forms=strip_legal_forms)
    address = normalize_address(record.get("business_address"))
    country_key, country_missing, country_flags = normalize_country(record.get("country"))
    qa_flags = tuple(dict.fromkeys(company.flags + address.flags + country_flags))
    return {
        "entity_id": "" if record.get("entity_id") is None else str(record.get("entity_id")),
        "business_name_raw": company.raw,
        "business_name_display": company.display,
        "business_name_key": company.key,
        "business_name_signature": company.signature,
        "business_name_legal_forms": ",".join(company.legal_forms),
        "business_name_trade_name": company.trade_name,
        "business_name_web_token": company.web_token,
        "business_name_is_missing": company.is_missing,
        "business_name_flags": ",".join(company.flags),
        "business_address_raw": address.raw,
        "business_address_display": address.display,
        "business_address_key": address.key,
        "business_address_signature": address.signature,
        "business_address_blocking_tokens": " ".join(address.blocking_tokens),
        "business_address_numbers": ",".join(address.numbers),
        "business_address_postal_codes": ",".join(address.postal_codes),
        "business_address_is_missing": address.is_missing,
        "business_address_flags": ",".join(address.flags),
        "country_raw": "" if record.get("country") is None else str(record.get("country")),
        "country_key": country_key,
        "country_is_missing": country_missing,
        "country_flags": ",".join(country_flags),
        "qa_flags": ",".join(qa_flags),
    }


def iter_source_rows(path: Path) -> Iterator[Dict[str, str]]:
    """Stream a challenge source TSV as dictionaries, validating the schema."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(
                f"Unexpected schema in {path}: {reader.fieldnames!r}; expected {SOURCE_COLUMNS!r}"
            )
        for row in reader:
            yield row


def normalize_stream(
    rows: Iterable[Mapping[str, object]], *, strip_legal_forms: bool = True
) -> Iterator[Dict[str, object]]:
    """Lazily normalize an iterable of source rows."""
    for row in rows:
        yield normalize_record(row, strip_legal_forms=strip_legal_forms)


# ---------------------------------------------------------------------------
# Data-quality accumulation
# ---------------------------------------------------------------------------


@dataclass
class CollisionTracker:
    """Bounded duplicate/collision profiler for normalized keys.

    5M-row sources cannot be profiled exactly in memory, so key frequencies are
    capped; ``overflow_keys`` records how many distinct keys were seen beyond
    the cap so the report is never mistaken for a complete census.
    """

    max_keys: int = 2_000_000
    name_counts: Counter = field(default_factory=Counter)
    address_counts: Counter = field(default_factory=Counter)
    name_variants: Dict[str, Set[str]] = field(default_factory=dict)
    overflow_keys: int = 0

    def add(self, normalized: Mapping[str, object]) -> None:
        name_key = str(normalized.get("business_name_key", ""))
        address_key = str(normalized.get("business_address_key", ""))
        if name_key:
            if name_key in self.name_counts or len(self.name_counts) < self.max_keys:
                self.name_counts[name_key] += 1
                variants = self.name_variants.setdefault(name_key, set())
                if len(variants) < 8:
                    variants.add(str(normalized.get("business_name_raw", "")))
            else:
                self.overflow_keys += 1
        if address_key:
            if address_key in self.address_counts or len(self.address_counts) < self.max_keys:
                self.address_counts[address_key] += 1
            else:
                self.overflow_keys += 1

    def summary(self) -> Dict[str, object]:
        name_shared = sum(count for count in self.name_counts.values() if count > 1)
        address_shared = sum(count for count in self.address_counts.values() if count > 1)
        variant_keys = sum(1 for variants in self.name_variants.values() if len(variants) > 1)
        return {
            "tracked_name_keys": len(self.name_counts),
            "tracked_address_keys": len(self.address_counts),
            "name_rows_sharing_a_key": name_shared,
            "address_rows_sharing_a_key": address_shared,
            "name_keys_with_multiple_raw_spellings": variant_keys,
            "overflow_keys": self.overflow_keys,
            "capped": self.overflow_keys > 0,
        }


@dataclass
class DataQualityAccumulator:
    """Streaming missing-value and flag profiler for one or more source files."""

    normalizer_version: str = NORMALIZER_VERSION
    rows: int = 0
    countries: Counter = field(default_factory=Counter)
    flags: Counter = field(default_factory=Counter)
    missing: Counter = field(default_factory=Counter)
    legal_form_only_rows: int = 0
    empty_name_key_rows: int = 0
    empty_address_key_rows: int = 0
    files: List[str] = field(default_factory=list)
    collisions: CollisionTracker = field(default_factory=CollisionTracker)

    def observe(self, normalized: Mapping[str, object]) -> None:
        self.rows += 1
        country = str(normalized.get("country_key", ""))
        if country:
            self.countries[country] += 1
        for column in ("business_name", "business_address", "country"):
            if normalized.get(f"{column}_is_missing"):
                self.missing[column] += 1
        for flag in str(normalized.get("qa_flags", "")).split(","):
            if flag:
                self.flags[flag] += 1
        if FLAG_LEGAL_FORM_ONLY in str(normalized.get("business_name_flags", "")):
            self.legal_form_only_rows += 1
        if not str(normalized.get("business_name_key", "")):
            self.empty_name_key_rows += 1
        if not str(normalized.get("business_address_key", "")):
            self.empty_address_key_rows += 1
        self.collisions.add(normalized)

    def report(self) -> Dict[str, object]:
        rows = self.rows or 1
        return {
            "normalizer_version": self.normalizer_version,
            "files": self.files,
            "rows": self.rows,
            "missing": {
                column: {"rows": count, "rate": count / rows}
                for column, count in sorted(self.missing.items())
            },
            "flags": dict(self.flags.most_common()),
            "countries": dict(self.countries.most_common(20)),
            "company_key_is_legal_form_only_rows": self.legal_form_only_rows,
            "empty_company_key_rows": self.empty_name_key_rows,
            "empty_address_key_rows": self.empty_address_key_rows,
            "duplicates": self.collisions.summary(),
            "notes": [
                "Signatures and blocking tokens are recall devices, never merge rules.",
                "Missing values are preserved as null; nothing is imputed.",
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_normalized_tsv(
    rows: Iterable[Mapping[str, object]], destination: Path
) -> int:
    """Write normalized rows in the challenge TSV layout; returns the count.

    The writer is explicit rather than csv-based on purpose.  ``csv`` with
    ``QUOTE_NONE`` refuses a value containing the delimiter or a quote, and its
    ``escapechar`` emits a literal backslash *followed by the real character*, so
    an escaped tab would still be a tab.  Escaping here guarantees the invariant
    that no field contains a tab or a newline, which keeps every row intact in an
    unquoted TSV.  Pair with :func:`read_normalized_tsv` to recover the originals.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(OUTPUT_COLUMNS) + "\n")
        for row in rows:
            handle.write(
                "\t".join(escape_tsv(row.get(column, "")) for column in OUTPUT_COLUMNS) + "\n"
            )
            written += 1
    return written


def read_normalized_tsv(path: Path) -> Iterator[Dict[str, str]]:
    """Read a file written by :func:`write_normalized_tsv` back losslessly."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r").split("\t")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").rstrip("\r").split("\t")
            yield {
                column: unescape_tsv(value)
                for column, value in zip(header, fields)
            }


def _limit(rows: Iterable[Mapping[str, object]], count: int) -> Iterator[Mapping[str, object]]:
    if count <= 0:
        yield from rows
        return
    for index, row in enumerate(rows):
        if index >= count:
            return
        yield row


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Normalize source TSVs and emit a data-quality report."""
    parser = argparse.ArgumentParser(
        description="Normalize challenge source TSVs (company, address, missing values)."
    )
    parser.add_argument("--input", nargs="+", required=True, help="source TSV file(s)")
    parser.add_argument("--output-dir", required=True, help="directory for normalized TSVs")
    parser.add_argument("--qa-json", help="path for the data-quality JSON report")
    parser.add_argument("--limit", type=int, default=0, help="rows per file (0 = all)")
    parser.add_argument(
        "--keep-legal-forms",
        action="store_true",
        help="keep legal forms in the company key instead of stripping them from the edges",
    )
    parser.add_argument(
        "--max-collision-keys",
        type=int,
        default=2_000_000,
        help="cap on distinct keys tracked for duplicate profiling",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    accumulator = DataQualityAccumulator()
    accumulator.collisions = CollisionTracker(max_keys=args.max_collision_keys)
    output_dir = Path(args.output_dir)
    for source in args.input:
        source_path = Path(source)
        accumulator.files.append(source_path.name)

        def observed(rows: Iterable[Mapping[str, object]]) -> Iterator[Dict[str, object]]:
            for row in rows:
                accumulator.observe(row)
                yield row

        written = write_normalized_tsv(
            observed(
                normalize_stream(
                    _limit(iter_source_rows(source_path), args.limit),
                    strip_legal_forms=not args.keep_legal_forms,
                )
            ),
            output_dir / source_path.name,
        )
        print(f"{source_path.name}: {written} rows -> {output_dir / source_path.name}", file=sys.stderr)

    report = accumulator.report()
    if args.qa_json:
        report_path = Path(args.qa_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
