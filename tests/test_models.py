"""Confirmed drafts contain encodable text before reaching persistence."""

from datetime import datetime
import unittest

from models import DeadlineDraft, MAX_SUBJECT_LENGTH, clean_field


NOW = datetime(2026, 10, 11, 12)


class FieldTests(unittest.TestCase):
    def test_lone_surrogates_raise_a_useful_validation_error(self):
        for value in (chr(0xD800), "Текст " + chr(0xDFFF)):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "Unicode"):
                clean_field(value, "Предмет", MAX_SUBJECT_LENGTH)

    def test_unicode_text_and_emoji_remain_supported(self):
        value = clean_field("  Теория\nвероятностей 🧮  ", "Предмет", MAX_SUBJECT_LENGTH)
        self.assertEqual(value, "Теория вероятностей 🧮")
        value.encode("utf-8")


class DraftTests(unittest.TestCase):
    def test_invalid_unicode_is_rejected_in_either_manual_field(self):
        for field in ("subject", "task"):
            with self.subTest(field=field):
                data = {"subject": "Базы данных", "task": "Лаба 3", "due_date": "2026-10-16 18:00"}
                data[field] += chr(0xD800)
                with self.assertRaisesRegex(ValueError, "Unicode"):
                    DeadlineDraft.from_dict(data, now=NOW)

    def test_valid_non_ascii_draft_preserves_confirmation_key(self):
        data = {"subject": "Базы данных 📚", "task": "Лабораторная работа №3",
                "due_date": "2026-10-16 18:00", "creation_key": "confirmation-1"}
        draft = DeadlineDraft.from_dict(data, now=NOW)
        self.assertEqual(draft.creation_key, "confirmation-1")
        self.assertEqual(draft.task, data["task"])


if __name__ == "__main__":
    unittest.main()
