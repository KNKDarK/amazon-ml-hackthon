"""Rule-level tests for pilot.text_normalization.

Every expectation here traces back to a measured property of the challenge TSVs
or to one of the documented normalization rules (C1-C12, A1-A9, M1-M4).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot.text_normalization import (  # noqa: E402
    NORMALIZER_VERSION,
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    CollisionTracker,
    DataQualityAccumulator,
    normalize_address,
    normalize_company,
    normalize_country,
    normalize_record,
    normalize_stream,
    numbers_comparison_key,
)
from pilot import text_normalization as tn  # noqa: E402

REAL_ROWS = [
    {
        "entity_id": "S1-925783039",
        "business_name": "Orelee's Barbershop",
        "business_address": "1795 Westchester Drive, High Point, NC",
        "country": "US",
    },
    {
        "entity_id": "S2-584977605",
        "business_name": "SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com",
        "business_address": "H.NO 204 C ROAD HOSHIARPUR, PUNJAB, Punjab",
        "country": "India",
    },
    {
        "entity_id": "S2-166376419",
        "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "country": "India",
    },
    {
        "entity_id": "S3-859268022",
        "business_name": "International South Consultants Private Ltd",
        "business_address": "",
        "country": "India",
    },
    {
        "entity_id": "S3-202863386",
        "business_name": "wilfordhancock.com",
        "business_address": "Mack Rd, Haltom City, Texas",
        "country": "US",
    },
    {
        "entity_id": "S1-156285671",
        "business_name": "<< Team Ecole",
        "business_address": "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
        "country": "France",
    },
    {
        "entity_id": "S3-578159284",
        "business_name": "LLC Hernandez Colonial Redwood",
        "business_address": "2260- Housecreek Trail, Unit 407, Raleigh, North Carolina",
        "country": "US",
    },
    {
        "entity_id": "S1-692436420",
        "business_name": "Ab & Co",
        "business_address": "  12  Main\tStreet ,  High Point ,\nNorth Carolina ",
        "country": "US",
    },
    {
        "entity_id": "S2-166389524",
        "business_name": "SA",
        "business_address": "n/a",
        "country": "India",
    },
]


class MissingValueRuleTests(unittest.TestCase):
    def test_empty_and_whitespace_only_are_missing(self) -> None:
        for value in ("", "   ", "\t\n", None):
            for field_name in ("business_name", "business_address"):
                with self.subTest(value=value, field=field_name):
                    self.assertTrue(tn.is_missing(value, field_name))

    def test_sentinels_are_missing_but_never_substrings(self) -> None:
        for value in ("N/A", "n.a.", "NaN", "NULL", "None", "unknown", "n.d.", "-", "--", "??"):
            with self.subTest(value=value):
                self.assertTrue(tn.is_missing(value, "business_name"))
        self.assertFalse(tn.is_missing("N/A Foods", "business_name"))
        self.assertFalse(tn.is_missing("12 Unknown Road", "business_address"))
        self.assertFalse(tn.is_missing("Nullable Foods", "business_address"))

    def test_na_is_missing_for_addresses_only(self) -> None:
        self.assertTrue(tn.is_missing("NA", "business_address"))
        self.assertFalse(tn.is_missing("NA", "business_name"))
        self.assertFalse(tn.is_missing("Narayana Foods", "business_name"))

    def test_non_alphanumeric_values_are_flagged_distinctly(self) -> None:
        self.assertEqual(tn.sentinel_kind("---", "business_address"), tn.NON_ALNUM_FLAG)
        self.assertEqual(tn.sentinel_kind("N/A", "business_address"), tn.SENTINEL_FLAG)
        self.assertEqual(tn.sentinel_kind("", "business_name"), "missing")

    def test_missing_is_preserved_and_never_imputed(self) -> None:
        company = normalize_company("")
        self.assertTrue(company.is_missing)
        self.assertEqual(company.key, "")
        self.assertEqual(company.display, "")
        self.assertIn(tn.MISSING_FLAG, company.flags)
        record = normalize_record(REAL_ROWS[3])
        self.assertEqual(record["business_address_raw"], "")
        self.assertEqual(record["business_address_display"], "")
        self.assertTrue(record["business_address_is_missing"])
        self.assertEqual(record["business_name_key"], "international south consultants")

    def test_country_is_an_open_string(self) -> None:
        self.assertEqual(normalize_country("France"), ("france", False, ()))
        self.assertEqual(normalize_country(" US "), ("us", False, ()))
        self.assertEqual(normalize_country("FR"), ("fr", False, ()))
        self.assertEqual(normalize_country("Atlantis"), ("atlantis", False, ()))
        self.assertEqual(normalize_country("-")[1], True)


class CompanyNameRuleTests(unittest.TestCase):
    def test_case_and_punctuation_variants_share_a_key(self) -> None:
        variants = [
            "ACME FOODS LLC",
            "Acme Foods, LLC",
            "  acme   foods llc  ",
            "Acme Foods LLC",
        ]
        keys = {normalize_company(value).key for value in variants}
        self.assertEqual(keys, {"acme foods"})

    def test_display_preserves_case_and_raw_is_untouched(self) -> None:
        result = normalize_company("  ACME  Foods, Inc. ")
        self.assertEqual(result.display, "ACME Foods, Inc.")
        self.assertEqual(result.raw, "  ACME  Foods, Inc. ")
        self.assertEqual(result.key, "acme foods")
        self.assertEqual(result.legal_forms, ("inc",))
        self.assertEqual(normalize_company("ACME FOODS LLC").key, "acme foods")
        self.assertIn(tn.FLAG_ALL_CAPS, normalize_company("ACME FOODS LLC").flags)

    def test_control_characters_and_newlines_are_removed(self) -> None:
        result = normalize_company("Acme\x00 Foods\tLLC\n")
        self.assertEqual(result.key, "acme foods")
        self.assertNotIn("\t", result.display)
        self.assertNotIn("\n", result.display)

    def test_ampersand_and_apostrophe_handling(self) -> None:
        self.assertEqual(normalize_company("Lee & Lawson").key, "lee lawson")
        self.assertEqual(normalize_company("Lee and Lawson").key, "lee lawson")
        self.assertEqual(normalize_company("O'Reilly & Sons").key, "oreilly sons")
        self.assertEqual(normalize_company("AT&T Wireless").key, "att wireless")

    def test_legal_forms_are_stripped_only_from_the_edges(self) -> None:
        self.assertEqual(normalize_company("Acme Foods Private Limited").key, "acme foods")
        self.assertEqual(normalize_company("Acme Foods Pvt. Ltd.").key, "acme foods")
        self.assertEqual(normalize_company("Acme Foods Corporation").key, "acme foods")
        self.assertEqual(normalize_company("Acme Foods Company").key, "acme foods")
        self.assertEqual(
            normalize_company("Limited Edition Cafe").key,
            "limited edition cafe",
            "a short generic name must not lose its leading token",
        )
        self.assertEqual(normalize_company("LLC Hernandez Colonial Redwood").key, "hernandez colonial redwood")
        self.assertEqual(normalize_company("Pvt. EFS Print Ventures Ltd.").key, "efs print ventures")
        self.assertEqual(
            normalize_company("Private Shivam Projects Limited").key,
            "private shivam projects",
            "below the content-token guard the leading form is deliberately kept",
        )

    def test_dotted_letter_legal_forms_are_reassembled(self) -> None:
        self.assertEqual(normalize_company("Meridian L.L.C.").key, "meridian")
        self.assertEqual(normalize_company("Meridian I.N.C.").key, "meridian")
        self.assertEqual(normalize_company("Olszewski Holding Company LLC LLC").key, "olszewski holding")

    def test_legal_form_only_names_keep_their_tokens_and_are_flagged(self) -> None:
        for value, expected in (("Ab & Co", "ab co"), ("SA", "sa"), ("As", "as")):
            with self.subTest(value=value):
                result = normalize_company(value)
                self.assertEqual(result.key, expected)
                self.assertIn(tn.FLAG_LEGAL_FORM_ONLY, result.flags)

    def test_ambiguous_short_tokens_are_not_stripped(self) -> None:
        self.assertEqual(normalize_company("Vishal As").key, "vishal as")
        self.assertEqual(normalize_company("Spa & Co").key, "spa co")

    def test_consecutive_repeats_collapse_once(self) -> None:
        result = normalize_company("Secure Secure Carolina")
        self.assertEqual(result.key, "secure carolina")
        self.assertIn(tn.FLAG_REPEAT_TOKEN, result.flags)
        self.assertEqual(normalize_company("A A A Cafe").key, "a cafe")

    def test_web_segment_is_split_only_when_it_looks_like_a_web_reference(self) -> None:
        with_web = normalize_company("WEST COURIERS | www.westltd.com")
        self.assertEqual(with_web.key, "west couriers")
        self.assertEqual(with_web.web_token, "www.westltd.com")
        self.assertIn(tn.FLAG_WEB_TOKEN, with_web.flags)
        self.assertEqual(
            normalize_company("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com").key,
            "shivshakti vidyalaya overseas",
        )
        self.assertEqual(normalize_company("heassociates.com").web_token, "")
        self.assertEqual(normalize_company("heassociates.com").key, "heassociates com")
        self.assertEqual(normalize_company("Chain | Store").key, "chain store")
        self.assertEqual(normalize_company("Chain | Store").web_token, "")

    def test_trade_name_is_split_and_preserved(self) -> None:
        result = normalize_company("John Smith Dba of Denning")
        self.assertEqual(result.key, "john smith")
        self.assertEqual(result.trade_name, "Denning")
        self.assertIn(tn.FLAG_TRADE_NAME, result.flags)
        self.assertEqual(normalize_company("DBA").trade_name, "")

    def test_indic_tokens_keep_their_vowel_signs(self) -> None:
        result = normalize_company("राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(result.key, "राम मार्केटिंग")
        self.assertTrue(
            any(unicodedata.category(ch).startswith("M") for ch in result.key),
            "Devanagari vowel signs must survive folding",
        )
        self.assertEqual(normalize_company("आदित्य प्रॉपर्टीज एलएलपी").key, "आदित्य प्रॉपर्टीज")

    def test_digits_are_preserved(self) -> None:
        self.assertEqual(normalize_company("24HR Towing").key, "24hr towing")
        self.assertEqual(normalize_company("U 8 Optimal Recruiting").key, "u 8 optimal recruiting")

    def test_signature_is_order_insensitive_and_flagged_as_blocking_only(self) -> None:
        self.assertEqual(
            normalize_company("Prime Money").signature,
            normalize_company("Money Prime Private Limited").signature,
        )
        self.assertNotEqual(normalize_company("Prime Money").key, normalize_company("Money Prime").key)


class AddressRuleTests(unittest.TestCase):
    def test_case_and_abbreviation_variants_share_a_key(self) -> None:
        left = normalize_address("105 ELM ST, MORGANTON, NC")
        right = normalize_address("105 Elm Street, Morganton, NC")
        self.assertEqual(left.key, right.key)
        self.assertIn("st", left.key.split())

    def test_state_names_expand_to_abbreviations(self) -> None:
        result = normalize_address("1795 Westchester Drive, High Point, North Carolina")
        self.assertIn("nc", result.key.split())
        self.assertNotIn("carolina", result.key.split())
        self.assertIn(tn.FLAG_STATE_EXPANDED, result.flags)

    def test_slashes_and_fractions_survive(self) -> None:
        result = normalize_address("19 1/2 STARDUST TRAIL, GREENSBORO, NC")
        self.assertIn("1/2", result.display)
        self.assertEqual(result.display.count(","), 2)
        self.assertEqual(normalize_address("G-3/571, GULMOHAR COLONY").display, "G-3/571, GULMOHAR COLONY")

    def test_decimals_are_not_split(self) -> None:
        self.assertIn("1.5", normalize_address("1.5 Miles Road, Austin, TX").display)

    def test_repeats_and_landmarks(self) -> None:
        result = normalize_address("294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI")
        self.assertEqual(result.key.split().count("unit"), 1)
        landmark = normalize_address("46/1106, Dhanalakshmi Road, Near Amani Auditorium, Kannur, Kerala")
        self.assertNotIn("near", landmark.blocking_tokens)
        self.assertIn("dhanalakshmi", landmark.blocking_tokens)

    def test_numbers_and_postal_codes_are_extracted_without_rewriting(self) -> None:
        result = normalize_address("Door No 183, 41St Cross, Bengaluru Urban, Bangalore, 560001")
        self.assertIn("560001", result.postal_codes)
        self.assertIn("183", result.numbers)
        self.assertIn("560001", result.display)
        self.assertEqual(numbers_comparison_key(["00567"]), numbers_comparison_key(["567"]))
        us = normalize_address("17560 Ellis Road, Tahlequah, OK 74464")
        self.assertIn("74464", us.postal_codes)

    def test_addresses_without_digits_are_flagged(self) -> None:
        self.assertIn(tn.FLAG_NO_DIGITS, normalize_address("Lille").flags)
        self.assertIn(tn.FLAG_HAS_DIGITS, normalize_address("12 Rue Lafayette").flags)

    def test_french_address_handling(self) -> None:
        result = normalize_address("1 Allée des Lilas, Mérignac, Nouvelle-Aquitaine")
        self.assertIn("allee", result.key.split())
        self.assertIn(tn.FLAG_NON_ASCII, result.flags)
        self.assertEqual(result.postal_codes, ())
        self.assertIn("rue", normalize_address("20 Rue Parmentier, Dunkerque").key.split())

    def test_signature_is_order_insensitive(self) -> None:
        self.assertEqual(
            normalize_address("High Point, NC, 100 Main Street").signature,
            normalize_address("100 Main Street, High Point, NC").signature,
        )


class RecordContractTests(unittest.TestCase):
    def test_originals_and_flags_are_always_present(self) -> None:
        for row in REAL_ROWS:
            with self.subTest(entity_id=row["entity_id"]):
                record = normalize_record(row)
                self.assertEqual(set(record), set(OUTPUT_COLUMNS))
                self.assertEqual(record["business_name_raw"], row["business_name"])
                self.assertEqual(record["business_address_raw"], row["business_address"])
                self.assertEqual(record["country_raw"], row["country"])
                for column, value in record.items():
                    if isinstance(value, str) and not column.endswith("_raw"):
                        self.assertNotIn("\t", value, column)
                        self.assertNotIn("\n", value, column)
                        self.assertNotIn("\r", value, column)

    def test_normalization_is_idempotent_and_deterministic(self) -> None:
        for row in REAL_ROWS:
            with self.subTest(entity_id=row["entity_id"]):
                first = normalize_record(row)
                second = normalize_record(
                    {
                        "entity_id": row["entity_id"],
                        "business_name": first["business_name_display"],
                        "business_address": first["business_address_display"],
                        "country": row["country"],
                    }
                )
                for column in (
                    "business_name_display",
                    "business_name_key",
                    "business_name_signature",
                    "business_address_display",
                    "business_address_key",
                    "business_address_signature",
                    "country_key",
                ):
                    self.assertEqual(first[column], second[column], column)
                self.assertEqual(first, normalize_record(row))

    def test_strip_legal_forms_can_be_disabled(self) -> None:
        self.assertEqual(normalize_company("Acme Foods LLC").key, "acme foods")
        kept = normalize_company("Acme Foods LLC", strip_legal_forms=False)
        self.assertEqual(kept.key, "acme foods llc")
        self.assertEqual(kept.legal_forms, ())

    def test_schema_validation_rejects_wrong_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.tsv"
            bad.write_text("entity_id\tbusiness_name\nS1-1\tAcme\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(tn.iter_source_rows(bad))

    def test_stream_api_and_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source1.tsv"
            header = "\t".join(SOURCE_COLUMNS)
            body = "\n".join(
                "\t".join(
                    (
                        row["entity_id"],
                        row["business_name"],
                        tn.clean_text(row["business_address"]),
                        row["country"],
                    )
                )
                for row in REAL_ROWS
            )
            source.write_text(f"{header}\n{body}\n", encoding="utf-8")
            output_dir = Path(directory) / "normalized"
            report_path = Path(directory) / "qa.json"
            exit_code = tn.main(
                [
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--qa-json",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            written = (output_dir / "source1.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(written[0].split("\t"), list(OUTPUT_COLUMNS))
            self.assertEqual(len(written), len(REAL_ROWS) + 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["rows"], len(REAL_ROWS))
            self.assertEqual(report["normalizer_version"], NORMALIZER_VERSION)
            self.assertEqual(report["missing"]["business_address"]["rows"], 2)
            self.assertEqual(report["countries"], {"us": 4, "india": 4, "france": 1})

    def test_raw_values_with_control_characters_stay_on_one_row(self) -> None:
        row = normalize_record(REAL_ROWS[7])
        row["business_name_raw"] = 'Quote " and backslash \\ and tab \t mix'
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "normalized.tsv"
            written = tn.write_normalized_tsv([row], destination)
            self.assertEqual(written, 1)
            self.assertEqual(len(destination.read_text(encoding="utf-8").splitlines()), 2)
            recovered = list(tn.read_normalized_tsv(destination))
            self.assertEqual(len(recovered), 1)
            self.assertEqual(
                recovered[0]["business_address_raw"],
                REAL_ROWS[7]["business_address"],
                "escaping must be lossless",
            )
            self.assertEqual(
                recovered[0]["business_name_raw"],
                'Quote " and backslash \\ and tab \t mix',
            )
            for column in OUTPUT_COLUMNS:
                self.assertIn(column, recovered[0])


class DataQualityReportTests(unittest.TestCase):
    def test_duplicate_and_inconsistent_representations_are_profiled(self) -> None:
        accumulator = DataQualityAccumulator()
        rows = [
            {"entity_id": "S1-1", "business_name": "Meridian Inc", "business_address": "1 A St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "MERIDIAN LLC", "business_address": "1 A Street", "country": "US"},
            {"entity_id": "S1-3", "business_name": "Meridian", "business_address": "9 Z Rd", "country": "India"},
            {"entity_id": "S1-4", "business_name": "N/A", "business_address": "-", "country": "US"},
        ]
        for row in normalize_stream(rows):
            accumulator.observe(row)
        report = accumulator.report()
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["missing"]["business_name"]["rows"], 1)
        self.assertEqual(report["missing"]["business_address"]["rows"], 1)
        self.assertEqual(report["duplicates"]["name_rows_sharing_a_key"], 3)
        self.assertEqual(report["duplicates"]["name_keys_with_multiple_raw_spellings"], 1)
        self.assertEqual(report["countries"], {"us": 3, "india": 1})
        self.assertEqual(
            report["flags"][tn.SENTINEL_FLAG] + report["flags"][tn.NON_ALNUM_FLAG],
            2,
        )

    def test_collision_tracker_is_bounded(self) -> None:
        tracker = CollisionTracker(max_keys=1)
        tracker.add({"business_name_key": "acme", "business_name_raw": "Acme"})
        tracker.add({"business_name_key": "beta", "business_name_raw": "Beta"})
        summary = tracker.summary()
        self.assertEqual(summary["tracked_name_keys"], 1)
        self.assertEqual(summary["overflow_keys"], 1)
        self.assertTrue(summary["capped"])


class FrozenModelConsistencyTests(unittest.TestCase):
    def test_keys_align_with_the_frozen_model_tokenizer(self) -> None:
        try:
            from pilot.er_common import normalize_name_tokens
        except RuntimeError:  # pragma: no cover - NumPy is a declared dependency
            self.skipTest("NumPy unavailable")
        for value in ("Acme Foods, Inc.", "Meridian LLC", "Lee and Lawson", "Summit Corp"):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize_company(value).key.split(),
                    normalize_name_tokens(value),
                )


if __name__ == "__main__":
    unittest.main()
