"""Configuration checks use isolated environment variables and temporary files."""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from config import MAX_LLM_TIMEOUT, ROOT, Settings, load_env, semester_start


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.env_file = Path(temporary.name) / ".env"

    def write_env(self, value):
        self.env_file.write_text(value, encoding="utf-8")

    def test_missing_file_is_optional(self):
        load_env(self.env_file)
        self.assertEqual(dict(os.environ), {})

    def test_process_values_have_priority_including_empty_values(self):
        os.environ.update(BOT_TOKEN="process-token", LLM_MODEL="")
        self.write_env("BOT_TOKEN=file-token\nLLM_MODEL=file-model\nDATABASE_PATH=demo.db\n")
        load_env(self.env_file)
        self.assertEqual(os.environ["BOT_TOKEN"], "process-token")
        self.assertEqual(os.environ["LLM_MODEL"], "")
        self.assertEqual(os.environ["DATABASE_PATH"], "demo.db")

    def test_quotes_export_and_inline_comments(self):
        self.write_env(
            "# Comment\n\n"
            "export BOT_TOKEN = 'test token'\n"
            'LLM_MODEL="model#literal"\n'
            "DATABASE_PATH=demo.db # local storage\n"
            "EMPTY=\n"
        )
        load_env(self.env_file)
        self.assertEqual(os.environ["BOT_TOKEN"], "test token")
        self.assertEqual(os.environ["LLM_MODEL"], "model#literal")
        self.assertEqual(os.environ["DATABASE_PATH"], "demo.db")
        self.assertEqual(os.environ["EMPTY"], "")

    def test_invalid_lines_report_line_number(self):
        for invalid in ("missing-equals", "BAD KEY=value", "9KEY=value", "=value"):
            with self.subTest(invalid=invalid):
                self.write_env("# first line\n" + invalid + "\n")
                with self.assertRaisesRegex(ValueError, "строка 2"):
                    load_env(self.env_file)

    def test_unclosed_quoted_value_is_rejected(self):
        for value in ('MODEL="unclosed', "MODEL='unclosed", 'MODEL="mismatched\''):
            with self.subTest(value=value):
                self.write_env(value)
                with self.assertRaisesRegex(ValueError, "Незакрытая кавычка"):
                    load_env(self.env_file)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.env_file = Path(temporary.name) / ".env"
        self.env_file.write_text("", encoding="utf-8")

    def settings(self, *, require_token=False):
        return Settings.from_env(env_file=self.env_file, require_token=require_token)

    def test_defaults_allow_offline_startup_check(self):
        value = self.settings()
        self.assertEqual(value.bot_token, "")
        self.assertEqual(value.database_path, ROOT / "deadlines.db")
        self.assertEqual(value.timezone, "Asia/Novosibirsk")
        self.assertEqual(value.llm_timeout, MAX_LLM_TIMEOUT)
        self.assertEqual(value.scheduler_interval, 30)
        self.assertEqual(value.llm_api_key, "")
        self.assertEqual(value.llm_model, "")
        self.assertEqual(value.llm_provider, "anthropic")
        self.assertEqual(value.llm_api_url, "https://api.anthropic.com/v1/messages")
        self.assertIsNone(value.llm_ca_bundle)
        self.assertIsNone(value.llm_chat_template_kwargs)

    def test_relative_database_is_resolved_from_project_root(self):
        os.environ["DATABASE_PATH"] = "data/students.db"
        self.assertEqual(self.settings().database_path, ROOT / "data" / "students.db")

    def test_absolute_database_path_is_preserved(self):
        database = self.env_file.parent / "students.db"
        os.environ["DATABASE_PATH"] = str(database)
        self.assertEqual(self.settings().database_path, database)

    def test_required_token_is_validated_and_trimmed(self):
        for token in ("", "  "):
            with self.subTest(token=token):
                os.environ["BOT_TOKEN"] = token
                with self.assertRaisesRegex(ValueError, "BOT_TOKEN"):
                    self.settings(require_token=True)
        os.environ["BOT_TOKEN"] = "  example-token  "
        self.assertEqual(self.settings(require_token=True).bot_token, "example-token")

    def test_http_api_endpoint_is_rejected(self):
        for url in ("http://example.org/messages", "file:///tmp/messages", ""):
            with self.subTest(url=url):
                os.environ["LLM_API_URL"] = url
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    self.settings()
        os.environ["LLM_API_URL"] = "https://example.org/messages"
        self.assertEqual(self.settings().llm_api_url, "https://example.org/messages")

    def test_invalid_timezone_is_rejected(self):
        os.environ["APP_TIMEZONE"] = "Unknown/NoSuchCity"
        with self.assertRaisesRegex(ValueError, "APP_TIMEZONE"):
            self.settings()

    def test_malformed_https_endpoints_fail_configuration_for_every_provider(self):
        endpoints = (
            "https://", "https:///messages", "https://:443/messages", "https://[broken/messages",
            "https://example.org:bad/messages", "https://example.org:65536/messages",
            "https://example.org:0/messages", "https://exa mple.org/messages",
            "https://example.org/path with space", "https://exam\nple.org/messages",
            "https://user:fixture-password@example.org/messages", "https://@example.org/messages",
        )
        for provider in ("anthropic", "gigachat", "openai"):
            os.environ["LLM_PROVIDER"] = provider
            for endpoint in endpoints:
                with self.subTest(provider=provider, endpoint=endpoint):
                    os.environ["LLM_API_URL"] = endpoint
                    with self.assertRaisesRegex(ValueError, "LLM_API_URL") as raised:
                        self.settings()
                    self.assertNotIn("fixture-password", str(raised.exception))

    def test_valid_custom_https_endpoints_keep_ports_queries_and_ipv6(self):
        for endpoint in ("https://example.org:8443/api/messages?version=2",
                         "https://[::1]:9443/custom/messages", "https://example.org/path%20encoded"):
            with self.subTest(endpoint=endpoint):
                os.environ["LLM_API_URL"] = f" {endpoint} "
                self.assertEqual(self.settings().llm_api_url, endpoint)

    def test_now_is_naive_in_configured_application_timezone(self):
        os.environ["APP_TIMEZONE"] = "Asia/Novosibirsk"
        settings = self.settings()
        local_time = datetime(2026, 10, 12, 0, 15, tzinfo=ZoneInfo(settings.timezone))
        with patch("config.datetime") as clock:
            clock.now.return_value = local_time
            current = settings.now()
        self.assertEqual(clock.now.call_args.args[0].key, "Asia/Novosibirsk")
        self.assertEqual(current, datetime(2026, 10, 12, 0, 15))
        self.assertIsNone(current.tzinfo)

    def test_valid_numeric_boundaries(self):
        for timeout, interval in (("0.1", "5"), ("5", "300")):
            with self.subTest(timeout=timeout, interval=interval):
                os.environ.update(LLM_TIMEOUT=timeout, SCHEDULER_INTERVAL=interval)
                value = self.settings()
                self.assertEqual(value.llm_timeout, min(float(timeout), MAX_LLM_TIMEOUT))
                self.assertEqual(value.scheduler_interval, int(interval))

    def test_timeout_rejects_invalid_and_nonfinite_values(self):
        for value in ("0", "-1", "5.1", "NaN", "inf", "-inf", "not-a-number"):
            with self.subTest(value=value):
                os.environ["LLM_TIMEOUT"] = value
                with self.assertRaises(ValueError):
                    self.settings()

    def test_scheduler_interval_rejects_out_of_range_and_nonintegers(self):
        for value in ("4", "301", "-1", "5.5", "NaN", "not-a-number"):
            with self.subTest(value=value):
                os.environ["SCHEDULER_INTERVAL"] = value
                with self.assertRaises(ValueError):
                    self.settings()

    def test_log_level_is_normalized_and_validated(self):
        os.environ["LOG_LEVEL"] = "warning"
        self.assertEqual(self.settings().log_level, "WARNING")
        os.environ["LOG_LEVEL"] = "SILENT"
        with self.assertRaisesRegex(ValueError, "LOG_LEVEL"):
            self.settings()

    def test_environment_file_values_and_process_priority_reach_settings(self):
        self.env_file.write_text("BOT_TOKEN=file\nAPP_TIMEZONE=UTC\nLLM_MODEL=model\n", encoding="utf-8")
        os.environ["BOT_TOKEN"] = "process"
        value = self.settings(require_token=True)
        self.assertEqual(value.bot_token, "process")
        self.assertEqual(value.timezone, "UTC")
        self.assertEqual(value.llm_model, "model")

    def test_gigachat_defaults_supply_model_and_endpoint_without_enabling_empty_key(self):
        os.environ["LLM_PROVIDER"] = "gigachat"
        value = self.settings()
        self.assertEqual(value.llm_provider, "gigachat")
        self.assertEqual(value.llm_model, "GigaChat-2")
        self.assertEqual(value.llm_api_url, "https://api.giga.chat/v1/chat/completions")
        self.assertEqual(value.llm_api_key, "")

    def test_provider_is_normalized_and_unknown_values_are_rejected(self):
        for value, expected in ((" GigaChat ", "gigachat"), (" OpenAI ", "openai"),
                                (" Anthropic ", "anthropic")):
            with self.subTest(value=value):
                os.environ["LLM_PROVIDER"] = value
                self.assertEqual(self.settings().llm_provider, expected)
        for value in ("", "other", "claude"):
            with self.subTest(value=value):
                os.environ["LLM_PROVIDER"] = value
                with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                    self.settings()

    def test_gigachat_accepts_explicit_model_endpoint_and_authorization_key(self):
        os.environ.update(
            LLM_PROVIDER="gigachat", LLM_MODEL=" GigaChat-2-Pro ",
            LLM_API_URL="https://example.org/chat/completions", LLM_API_KEY=" authorization-key ",
        )
        value = self.settings()
        self.assertEqual(value.llm_model, "GigaChat-2-Pro")
        self.assertEqual(value.llm_api_url, "https://example.org/chat/completions")
        self.assertEqual(value.llm_api_key, "authorization-key")

    def test_explicit_empty_model_is_preserved(self):
        for provider in ("gigachat", "anthropic", "openai"):
            with self.subTest(provider=provider):
                os.environ.update(LLM_PROVIDER=provider, LLM_MODEL="", LLM_API_KEY="")
                self.assertEqual(self.settings().llm_model, "")
                self.assertEqual(self.settings().llm_api_key, "")

    def test_openai_defaults_supply_nsu_model_and_endpoint_without_requiring_key(self):
        os.environ["LLM_PROVIDER"] = "openai"
        value = self.settings()
        self.assertEqual(value.llm_provider, "openai")
        self.assertEqual(value.llm_model, "Qwen3.8-27B")
        self.assertEqual(value.llm_api_url, "https://deepcode.ci.nsu.ru/api/chat/completions")
        self.assertEqual(value.llm_api_key, "")
        self.assertIsNone(value.llm_ca_bundle)
        self.assertEqual(value.llm_timeout, 5.0)

    def test_chat_template_kwargs_are_optional_and_blank_values_disable_them(self):
        for raw in ("", "  \t "):
            with self.subTest(raw=raw):
                os.environ["LLM_CHAT_TEMPLATE_KWARGS"] = raw
                self.assertIsNone(self.settings().llm_chat_template_kwargs)

    def test_chat_template_kwargs_preserve_only_supported_boolean_values(self):
        for raw, expected in (
            ('{"enable_thinking": false}', {"enable_thinking": False}),
            (' {"thinking": true} ', {"thinking": True}),
            ('{"thinking": false, "enable_thinking": true}', {"thinking": False, "enable_thinking": True}),
            ("{}", {}),
        ):
            with self.subTest(raw=raw):
                os.environ["LLM_CHAT_TEMPLATE_KWARGS"] = raw
                self.assertEqual(self.settings().llm_chat_template_kwargs, expected)

    def test_invalid_chat_template_kwargs_never_reveal_the_configured_value(self):
        expected = "LLM_CHAT_TEMPLATE_KWARGS должен быть JSON-объектом с ключами thinking/enable_thinking и логическими значениями"
        for raw in (
            "private-config-secret", '{"thinking":false,', "[]", "null", "true", "42",
            '"private-config-secret"', '{"private-config-secret":false}', '{"thinking":0}',
            '{"thinking":1}', '{"thinking":"private-config-secret"}', '{"thinking":null}',
            '{"enable_thinking":[]}', '{"thinking":false,"extra":"private-config-secret"}',
            '{"thinking":NaN}',
        ):
            with self.subTest(raw=raw):
                os.environ["LLM_CHAT_TEMPLATE_KWARGS"] = raw
                with self.assertRaises(ValueError) as raised:
                    self.settings()
                self.assertEqual(str(raised.exception), expected)
                self.assertTrue(raised.exception.__suppress_context__)

    def test_chat_template_kwargs_load_from_file_and_process_values_take_priority(self):
        self.env_file.write_text(
            'LLM_PROVIDER=openai\nLLM_CHAT_TEMPLATE_KWARGS=\'{"enable_thinking":false}\'\n',
            encoding="utf-8",
        )
        self.assertEqual(self.settings().llm_chat_template_kwargs, {"enable_thinking": False})
        os.environ["LLM_CHAT_TEMPLATE_KWARGS"] = '{"thinking":true}'
        self.assertEqual(self.settings().llm_chat_template_kwargs, {"thinking": True})
        os.environ["LLM_CHAT_TEMPLATE_KWARGS"] = ""
        self.assertIsNone(self.settings().llm_chat_template_kwargs)

    def test_openai_accepts_explicit_model_endpoint_and_api_key(self):
        os.environ.update(
            LLM_PROVIDER="openai", LLM_MODEL=" Qwen3.8-27B ",
            LLM_API_URL=" https://example.org/v1/chat/completions ", LLM_API_KEY=" example-key ",
        )
        value = self.settings()
        self.assertEqual(value.llm_model, "Qwen3.8-27B")
        self.assertEqual(value.llm_api_url, "https://example.org/v1/chat/completions")
        self.assertEqual(value.llm_api_key, "example-key")

    def test_openai_keeps_process_overrides_and_explicit_empty_credentials(self):
        self.env_file.write_text(
            "LLM_PROVIDER=openai\nLLM_MODEL=file-model\nLLM_API_KEY=file-key\nLLM_TIMEOUT=5\n",
            encoding="utf-8",
        )
        os.environ.update(LLM_MODEL="Qwen3.8-27B", LLM_API_KEY="")
        value = self.settings()
        self.assertEqual(value.llm_provider, "openai")
        self.assertEqual(value.llm_model, "Qwen3.8-27B")
        self.assertEqual(value.llm_api_key, "")
        self.assertEqual(value.llm_timeout, 5.0)

    def test_openai_rejects_insecure_or_empty_endpoint(self):
        os.environ["LLM_PROVIDER"] = "openai"
        for endpoint in ("http://example.org/chat/completions", "", "file:///tmp/response"):
            with self.subTest(endpoint=endpoint):
                os.environ["LLM_API_URL"] = endpoint
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    self.settings()

    def test_legacy_anthropic_configuration_keeps_provider_and_custom_model(self):
        self.env_file.write_text("LLM_MODEL=existing-model\nLLM_API_URL=https://example.org/messages\n", encoding="utf-8")
        value = self.settings()
        self.assertEqual(value.llm_provider, "anthropic")
        self.assertEqual(value.llm_model, "existing-model")
        self.assertEqual(value.llm_api_url, "https://example.org/messages")

    def test_timeout_values_up_to_five_seconds_are_preserved(self):
        for value in ("2.5", "3", "4.5", "5"):
            with self.subTest(value=value):
                os.environ["LLM_TIMEOUT"] = value
                self.assertEqual(self.settings().llm_timeout, float(value))
        os.environ["LLM_TIMEOUT"] = "1.25"
        self.assertEqual(self.settings().llm_timeout, 1.25)

    def test_relative_ca_bundle_is_resolved_from_application_root(self):
        temporary_root = self.env_file.parent
        bundle = temporary_root / "certs" / "local.pem"
        bundle.parent.mkdir()
        bundle.write_text("test PEM fixture", encoding="utf-8")
        os.environ["LLM_CA_BUNDLE"] = "certs/local.pem"
        with patch("config.ROOT", temporary_root):
            value = self.settings()
        self.assertEqual(value.llm_ca_bundle, bundle)

    def test_absolute_ca_bundle_is_preserved(self):
        bundle = self.env_file.parent / "local.pem"
        bundle.write_text("test PEM fixture", encoding="utf-8")
        os.environ["LLM_CA_BUNDLE"] = str(bundle)
        self.assertEqual(self.settings().llm_ca_bundle, bundle)

    def test_ca_bundle_must_be_an_existing_file(self):
        for path in (self.env_file.parent / "missing.pem", self.env_file.parent):
            with self.subTest(path=path):
                os.environ["LLM_CA_BUNDLE"] = str(path)
                with self.assertRaisesRegex(ValueError, "LLM_CA_BUNDLE"):
                    self.settings()
        os.environ["LLM_CA_BUNDLE"] = "  "
        self.assertIsNone(self.settings().llm_ca_bundle)

    def test_appended_provider_fields_keep_previous_positional_settings_compatible(self):
        value = Settings("bot-token", Path("example.db"), "UTC", "key", "model", "https://example.org/messages", 1.5, 45, "DEBUG")
        self.assertEqual(value.llm_timeout, 1.5)
        self.assertEqual(value.scheduler_interval, 45)
        self.assertEqual(value.log_level, "DEBUG")
        self.assertEqual(value.llm_provider, "anthropic")
        self.assertIsNone(value.llm_ca_bundle)
        self.assertIsNone(value.llm_chat_template_kwargs)


class SemesterTests(unittest.TestCase):
    def test_fall_semester_starts_on_september_first(self):
        for current in (datetime(2026, 9, 1), datetime(2026, 10, 11), datetime(2026, 12, 31, 23, 59)):
            with self.subTest(current=current):
                self.assertEqual(semester_start(current), datetime(2026, 9, 1))

    def test_spring_semester_starts_on_february_first(self):
        for current in (datetime(2026, 2, 1), datetime(2026, 5, 20), datetime(2026, 8, 31, 23, 59)):
            with self.subTest(current=current):
                self.assertEqual(semester_start(current), datetime(2026, 2, 1))

    def test_january_belongs_to_previous_fall_semester(self):
        self.assertEqual(semester_start(datetime(2027, 1, 31, 23, 59)), datetime(2026, 9, 1))


if __name__ == "__main__":
    unittest.main()
