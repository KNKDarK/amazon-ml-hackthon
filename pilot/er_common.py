#!/usr/bin/env python3
"""Shared, dependency-light utilities for the 10K Source-1 pilot.

The module deliberately avoids loading full source tables and never constructs a
cross-source similarity matrix.  All full-file operations are line-streamed.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - the pilot environment has NumPy
    raise RuntimeError("NumPy is required for the compact logistic classifier") from exc


# ---------------------------------------------------------------------------
# Streaming and deterministic sampling
# ---------------------------------------------------------------------------

CSV_DIALECT = {"delimiter": "\t", "quoting": csv.QUOTE_NONE}


def iter_tsv(path: Path) -> Iterator[Dict[str, str]]:
    """Stream a strict four-column TSV as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
        expected = ["entity_id", "business_name", "business_address", "country"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames!r}")
        for row in reader:
            yield row


def stable_u64(text: str, salt: str = "") -> int:
    payload = (salt + "\0" + text).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big")


def reservoir_sample_rows(path: Path, size: int, seed: int = 20260925) -> List[Dict[str, str]]:
    """Uniform sample without replacing the full Source-1 file in memory."""
    import random

    rng = random.Random(seed)
    reservoir: List[Dict[str, str]] = []
    for index, row in enumerate(iter_tsv(path)):
        if index < size:
            reservoir.append(row)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < size:
                reservoir[replacement] = row
    if len(reservoir) != size:
        raise ValueError(f"Requested {size} rows, but {path} contains only {len(reservoir)}")
    return reservoir


# ---------------------------------------------------------------------------
# Unicode / business / address normalization
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
DIGIT_RE = re.compile(r"\d+", flags=re.UNICODE)

LEGAL_TERMS = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc", "incorporated",
    "limited", "llc", "llp", "lp", "ltd", "plc", "private", "pte", "pty", "pvt", "sarl",
    "sa", "sl", "srl", "ll", "lda", "llda", "oy", "ab", "as", "kk", "kft", "spa", "sas",
}

# Removed from name keys so legal-form differences do not block true matches.
NAME_STOP = LEGAL_TERMS | {
    "company", "corporation", "incorporated", "limited", "private", "holdings", "holding",
    "international", "group", "india", "indian", "united", "states", "usa", "america",
}

ADDRESS_WORD_EQUIVALENTS = {
    # US street/address forms.
    "street": "st", "str": "st", "rd": "rd", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "boul": "blvd", "drive": "dr", "drv": "dr", "lane": "ln",
    "court": "ct", "circle": "cir", "place": "pl", "plaza": "plz", "plz": "plz",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "way": "way",
    "square": "sq", "squareplaza": "plz", "apartment": "apt", "appt": "apt", "suite": "ste",
    "ste": "ste", "floor": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    # Common India forms.
    "house": "hno", "hno": "hno", "number": "no", "near": "nr", "opposite": "opp",
    "sector": "sec", "section": "sec", "district": "dist", "layout": "layout",
    "complex": "complex", "apartments": "apt", "buildingno": "bldg",
}

ADDRESS_STOP = {
    "the", "and", "at", "of", "near", "nr", "opposite", "opp", "beside", "behind",
    "india", "indian", "united", "states", "america", "france", "french", "sector",
    "sec", "district", "dist", "layout", "complex", "building", "bldg", "floor", "fl",
    "first", "second", "third", "fourth", "ground",
}

US_STATES = {
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
}


def ascii_fold(text: str) -> str:
    """Case-fold and strip Latin accents while preserving non-Lingual marks.

    Removing every Unicode mark corrupts Indic scripts because vowel signs are
    category ``Mc`` rather than canonical combining class ``Mn``.  Marks are
    therefore removed only when they decorate an ASCII/Latin base character;
    all marks belonging to other scripts remain part of the token.
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


def raw_tokens(text: str) -> List[str]:
    """Unicode tokenizer that keeps combining marks attached to their base."""
    folded = ascii_fold(text)
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


def normalize_country(text: str) -> str:
    value = ascii_fold(text).strip()
    return "india" if value == "india" else "us" if value in {"us", "usa", "united states"} else value


def normalize_name_tokens(text: str) -> List[str]:
    tokens = raw_tokens(text)
    # Strip one or more trailing legal terms, but retain meaningful original tokens.
    while tokens and tokens[-1] in LEGAL_TERMS:
        tokens.pop()
    return [token for token in tokens if token and token not in {"the", "and"}]


def normalize_address_tokens(text: str) -> List[str]:
    # Replace multiword US state names first.
    folded = ascii_fold(text)
    folded = folded.replace(",", " ").replace("/", " ")
    for full, abbreviation in sorted(US_STATES.items(), key=lambda item: -len(item[0])):
        folded = re.sub(rf"\b{re.escape(full)}\b", abbreviation, folded)
    tokens = raw_tokens(folded)
    normalized: List[str] = []
    for token in tokens:
        token = ADDRESS_WORD_EQUIVALENTS.get(token, token)
        if token:
            normalized.append(token)
    return normalized


def name_signature(text: str) -> str:
    """Order-insensitive normalized name used by exact/prefix blocking."""
    return " ".join(sorted(normalize_name_tokens(text)))


def address_signature(text: str) -> str:
    """Order-insensitive normalized address used by exact/token blocking."""
    return " ".join(sorted(normalize_address_tokens(text)))


def canonical_target_fields(name: str, address: str, country: str) -> Tuple[str, str, str]:
    """Return canonicalized target fields used by candidate generation."""
    return name_signature(name), address_signature(address), normalize_country(country)


def numbers(text: str) -> List[str]:
    return [token.lstrip("0") or "0" for token in DIGIT_RE.findall(ascii_fold(text))]


def soundex(text: str) -> str:
    """Soundex for ASCII words; deterministic Unicode prefix fallback otherwise."""
    letters = [ch for ch in ascii_fold(text) if "a" <= ch <= "z"]
    if not letters:
        compact = "".join(raw_tokens(text))
        return ("u" + compact[:5]) if compact else ""
    first = letters[0].upper()
    digits: List[str] = []
    previous = first
    for ch in letters[1:]:
        if ch in "aeiou":
            code = "0"
        elif ch == "h":
            code = previous if previous in "0123456789" else "0"
        elif ch in "bfpv":
            code = "1"
        elif ch in "cgjkqsxz":
            code = "2"
        elif ch in "dt":
            code = "3"
        elif ch == "l":
            code = "4"
        elif ch in "mn":
            code = "5"
        elif ch == "r":
            code = "6"
        else:
            code = "0"
        if code != previous:
            digits.append(code)
        previous = code
        if len(digits) == 3:
            break
    return first + ("".join(digits).ljust(3, "0"))


# Block scheme IDs; persisted as a bit mask in candidate_pairs pilot artifacts.
BLOCK_SCHEMES = [
    "name_exact",
    "name_prefix",
    "name_token",
    "name_soundex",
    "address_numeric",
    "address_house_postal",
    "address_token",
    "address_bigram",
]
BLOCK_BIT = {name: 1 << index for index, name in enumerate(BLOCK_SCHEMES)}


def blocking_keys(name: str, address: str, country: str) -> Set[str]:
    """Generate complementary normalized name/address keys; never a dense comparison."""
    country_key = normalize_country(country)
    ntokens = normalize_name_tokens(name)
    atokens = normalize_address_tokens(address)
    nsig = " ".join(sorted(ntokens))
    asig = " ".join(sorted(atokens))
    keys: Set[str] = set()

    if nsig:
        keys.add(f"1|{country_key}|nfull|{nsig}")
        if len(nsig) >= 4:
            keys.add(f"2|{country_key}|npre|{nsig[:8]}")
        for token in set(ntokens):
            if len(token) >= 3 and token not in NAME_STOP:
                keys.add(f"3|{country_key}|ntok|{token}")
        for token in set(ntokens):
            sound = soundex(token)
            if sound and sound != "U":
                keys.add(f"4|{country_key}|nsnd|{sound}")

    nums = numbers(address)
    significant_nums = [num for num in nums if len(num) >= 3]
    for num in set(significant_nums):
        keys.add(f"5|{country_key}|anum|{num}")
    if nums and significant_nums:
        first_num = nums[0]
        # Longest number is generally a PIN/ZIP; alternatives make this less order-sensitive.
        for postal in set(significant_nums):
            keys.add(f"6|{country_key}|ahp|{first_num}|{postal}")

    if asig:
        for token in set(atokens):
            if len(token) >= 3 and token not in ADDRESS_STOP:
                keys.add(f"7|{country_key}|atok|{token}")
        filtered = [token for token in atokens if len(token) >= 3 and token not in ADDRESS_STOP]
        for left, right in zip(filtered, filtered[1:]):
            keys.add(f"8|{country_key}|abg|{left}|{right}")
    return keys


# ---------------------------------------------------------------------------
# Pairwise string features
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "bias",
    "same_country",
    "name_exact",
    "name_prefix_fraction",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_containment",
    "name_char3_dice",
    "name_char4_dice",
    "name_length_ratio",
    "name_soundex_overlap",
    "address_exact",
    "address_prefix_fraction",
    "address_token_jaccard",
    "address_token_dice",
    "address_token_containment",
    "address_char3_dice",
    "address_char4_dice",
    "address_length_ratio",
    "house_number_equal",
    "postal_equal",
    "numeric_overlap_fraction",
    "name_address_mean",
    "name_address_min",
    "name_address_product",
    "name_high_address_low",
    "name_low_address_high",
    "both_exact",
    "name_missing",
    "address_missing",
    "token_count_difference",
]


@dataclass(slots=True)
class PairText:
    name: str
    address: str
    country: str
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    name3: frozenset[str]
    name4: frozenset[str]
    address3: frozenset[str]
    address4: frozenset[str]
    numbers: Tuple[str, ...]
    soundexes: frozenset[str]

    @classmethod
    def make(cls, name: str, address: str, country: str) -> "PairText":
        ntokens = tuple(normalize_name_tokens(name))
        atokens = tuple(normalize_address_tokens(address))
        name_compact = " ".join(ntokens)
        address_compact = " ".join(atokens)
        return cls(
            name=name_compact,
            address=address_compact,
            country=normalize_country(country),
            name_tokens=frozenset(ntokens),
            address_tokens=frozenset(atokens),
            name3=frozenset(name_compact[i : i + 3] for i in range(max(0, len(name_compact) - 2))),
            name4=frozenset(name_compact[i : i + 4] for i in range(max(0, len(name_compact) - 3))),
            address3=frozenset(address_compact[i : i + 3] for i in range(max(0, len(address_compact) - 2))),
            address4=frozenset(address_compact[i : i + 4] for i in range(max(0, len(address_compact) - 3))),
            numbers=tuple(numbers(address)),
            soundexes=frozenset(filter(None, (soundex(token) for token in ntokens))),
        )


def set_similarity(left: frozenset[str], right: frozenset[str]) -> Tuple[float, float, float]:
    if not left and not right:
        return 1.0, 1.0, 1.0
    union = len(left | right)
    if union == 0:
        return 0.0, 0.0, 0.0
    intersection = len(left & right)
    smaller = min(len(left), len(right))
    return (
        intersection / union,
        (2.0 * intersection) / (len(left) + len(right)) if left and right else 0.0,
        intersection / smaller if smaller else 0.0,
    )


def dice(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2.0 * len(left & right) / (len(left) + len(right))


def prefix_fraction(left: str, right: str) -> float:
    shorter = min(len(left), len(right))
    if shorter == 0:
        return 0.0
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count / shorter


def length_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 0.0


def pair_features(left: PairText, right: PairText) -> np.ndarray:
    nj, nd, nc = set_similarity(left.name_tokens, right.name_tokens)
    aj, ad, ac = set_similarity(left.address_tokens, right.address_tokens)
    n3 = dice(left.name3, right.name3)
    n4 = dice(left.name4, right.name4)
    a3 = dice(left.address3, right.address3)
    a4 = dice(left.address4, right.address4)
    name_exact = float(bool(left.name) and left.name == right.name)
    address_exact = float(bool(left.address) and left.address == right.address)
    left_nums = set(left.numbers)
    right_nums = set(right.numbers)
    numeric_union = len(left_nums | right_nums)
    numeric_overlap = len(left_nums & right_nums) / numeric_union if numeric_union else 0.0
    left_long = [n for n in left.numbers if len(n) >= 3]
    right_long = [n for n in right.numbers if len(n) >= 3]
    postal_equal = float(bool(left_long and right_long and set(left_long) & set(right_long)))
    house_equal = float(bool(left.numbers and right.numbers and left.numbers[0] == right.numbers[0]))
    soundex_overlap = (
        len(left.soundexes & right.soundexes) / min(len(left.soundexes), len(right.soundexes))
        if left.soundexes and right.soundexes else 0.0
    )
    mean_score = (nj + aj) / 2.0
    min_score = min(nj, aj)
    product = nj * aj
    values = [
        1.0,
        float(left.country == right.country and bool(left.country)),
        name_exact,
        prefix_fraction(left.name, right.name),
        nj, nd, nc, n3, n4,
        length_ratio(left.name, right.name),
        soundex_overlap,
        address_exact,
        prefix_fraction(left.address, right.address),
        aj, ad, ac, a3, a4,
        length_ratio(left.address, right.address),
        house_equal,
        postal_equal,
        numeric_overlap,
        mean_score,
        min_score,
        product,
        float(nj >= 0.75 and aj < 0.35),
        float(nj < 0.35 and aj >= 0.75),
        name_exact * address_exact,
        float(not left.name or not right.name),
        float(not left.address or not right.address),
        float(abs(len(left.name_tokens) - len(right.name_tokens))),
    ]
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Resource monitoring and evaluation
# ---------------------------------------------------------------------------

def available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


@dataclass
class MemoryMonitor:
    interval: float = 0.25
    peak_rss: int = field(default_factory=current_rss_bytes)
    min_available: int = field(default_factory=available_memory_bytes)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def sample(self) -> None:
        self.peak_rss = max(self.peak_rss, current_rss_bytes())
        available = available_memory_bytes()
        if available:
            self.min_available = min(self.min_available, available)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> "MemoryMonitor":
        self.sample()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def as_dict(self) -> Dict[str, float]:
        return {
            "peak_process_rss_bytes": self.peak_rss,
            "peak_process_rss_mib": self.peak_rss / (1024 ** 2),
            "minimum_system_mem_available_bytes": self.min_available,
            "minimum_system_mem_available_mib": self.min_available / (1024 ** 2),
        }


def adapt_batch(batch_size: int, minimum_batch: int = 1000) -> int:
    available = available_memory_bytes()
    if available and available < 2 * 1024 ** 3:
        return max(minimum_batch, batch_size // 2)
    return batch_size


def f_beta(y_true: Sequence[int], y_pred: Sequence[int], beta: float = 0.5) -> float:
    tp = sum(t & p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(query_ids: Sequence[int], truth: Mapping[int, Set[str]],
              predictions: Mapping[int, Sequence[str]]) -> Dict[str, float]:
    scores: List[float] = []
    for qid in query_ids:
        actual = truth.get(qid, set())
        predicted = set(predictions.get(qid, ()))
        if not actual:
            scores.append(1.0 if not predicted else 0.0)
            continue
        tp = len(actual & predicted)
        fp = len(predicted - actual)
        fn = len(actual - predicted)
        if tp == 0:
            scores.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            scores.append(1.25 * precision * recall / (0.25 * precision + recall)
                          if precision + recall else 0.0)
    total_tp = sum(len(truth.get(q, set()) & set(predictions.get(q, ()))) for q in query_ids)
    total_pred = sum(len(predictions.get(q, ())) for q in query_ids)
    total_true = sum(len(truth.get(q, set())) for q in query_ids)
    return {
        "macro_f05": sum(scores) / len(scores) if scores else 0.0,
        "micro_precision": total_tp / total_pred if total_pred else 1.0,
        "micro_recall": total_tp / total_true if total_true else 1.0,
        "tp": total_tp,
        "fp": total_pred - total_tp,
        "fn": total_true - total_tp,
    }


def percentile(values: Sequence[int | float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))
