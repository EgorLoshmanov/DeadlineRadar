"""Validate demonstration CLI arguments before opening configuration or SQLite."""

from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from seed import main, positive_user_id


class SeedArgumentsTests(unittest.TestCase):
    def test_invalid_user_ids_exit_cleanly_before_touching_database(self):
        for value in ("0", "-1", str(2**63), "9" * 30, "not-a-number"):
            with self.subTest(value=value):
                diagnostics = io.StringIO()
                with patch("seed.Database") as database, patch("seed.Settings.from_env") as settings, \
                        redirect_stderr(diagnostics), self.assertRaises(SystemExit) as raised:
                    main(["--user-id", value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("Telegram ID", diagnostics.getvalue())
                self.assertNotIn("Traceback", diagnostics.getvalue())
                database.assert_not_called()
                settings.assert_not_called()

    def test_normal_and_largest_supported_user_ids_are_accepted(self):
        for value in (1, 1001, 1234567890, 2**63 - 1):
            with self.subTest(value=value):
                self.assertEqual(positive_user_id(str(value)), value)


if __name__ == "__main__":
    unittest.main()
