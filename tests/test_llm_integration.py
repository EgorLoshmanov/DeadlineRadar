"""Model routing and isolated CLI checks use a simulated GigaChat transport."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import ssl
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from config import ROOT, Settings
from main import check_llm, create_parser, main
from parser import ParserEngine


NOW = datetime(2026, 10, 11, 12, 0)
RESULT = {"subject": "Базы данных", "task": "Лабораторная работа №3",
          "due_date": "2026-10-16 18:00", "confidence": 0.95}


class Response:
    def __init__(self, value):
        self.raw = json.dumps(value, ensure_ascii=False).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, maximum):
        return self.raw[:maximum]


def responses():
    return [Response({"access_token": "temporary-secret", "expires_at": int((time.time() + 1800) * 1000)}),
            Response({"choices": [{"message": {"content": json.dumps(RESULT)}}]})]


class GigaChatParserTests(unittest.IsolatedAsyncioTestCase):
    def parser(self, **overrides):
        return ParserEngine(api_key="authorization-secret", model="GigaChat-2", provider="gigachat", **overrides)

    async def test_real_client_routing_and_validation_preserve_confirmation_fields(self):
        parser = self.parser(ca_bundle=ROOT / "certs" / "russian_trusted_root_ca.pem")
        with patch("llm.urlopen", side_effect=responses()) as opener:
            result = await parser.parse("Лаба 3 по БД до пятницы в 18:00", NOW)
        self.assertEqual(result["parsing_method"], "llm")
        self.assertEqual(result["due_date"], RESULT["due_date"])
        self.assertEqual(opener.call_count, 2)
        generation = opener.call_args.args[0]
        payload = json.loads(generation.data)
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["untrusted_message"], "Лаба 3 по БД до пятницы в 18:00")
        self.assertEqual(user["current_local_weekday"], "воскресенье")
        self.assertEqual(user["verified_due_date"], "2026-10-16 18:00")
        context = opener.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_calendar_context_does_not_guess_missing_or_ambiguous_dates(self):
        for text in ("Лаба по БД", "Лаба по БД до 15 числа", "Лаба до пятницы или субботы",
                     "Лаба к первой паре завтра", "Лаба по БД 31 ноября",
                     "Лаба по БД не 15 октября, а 16 октября"):
            with self.subTest(text=text):
                _, content = ParserEngine._messages(text, NOW)
                self.assertNotIn("verified_due_date", json.loads(content))

    async def test_corrected_date_reaches_gigachat_and_its_valid_response_is_primary(self):
        text = "Лаба 3 по БД не 15 октября, а 16 октября в 18:00"
        parser = self.parser()
        with patch("llm.urlopen", side_effect=responses()) as opener:
            result = await parser.parse(text, NOW)
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(result["parsing_method"], "llm")
        self.assertEqual(result["due_date"], RESULT["due_date"])
        payload = json.loads(opener.call_args.args[0].data)
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["untrusted_message"], text)
        self.assertNotIn("verified_due_date", user)
        self.assertEqual(user["required_assignment_numbers"], ["3"])

    async def test_temporal_negatives_and_corrections_survive_real_client_routing(self):
        cases = (
            ("Лаба 3 по БД завтра не в 18:00, а в 19:00", "2026-10-12 19:00",
             "llm", "2026-10-12 19:00"),
            ("Лаба 3 по БД до 15 октября, но не 16 октября", "2026-10-16 23:59",
             "regex_fallback", "2026-10-15 23:59"),
            ("Лаба 3 по БД до пятницы 15 октября", "2026-10-15 23:59",
             "regex_fallback", None),
        )
        for text, model_due, method, expected_due in cases:
            with self.subTest(text=text):
                parser = self.parser()
                fixture = responses()
                answer = dict(RESULT, due_date=model_due)
                fixture[1] = Response({"choices": [{"message": {"content": json.dumps(answer)}}]})
                with patch("llm.urlopen", side_effect=fixture) as opener:
                    result = await parser.parse(text, NOW)
                self.assertEqual(opener.call_count, 2)
                self.assertEqual(result["parsing_method"], method)
                self.assertEqual(result["due_date"], expected_due)

    async def test_unsafe_date_is_rejected_after_gigachat_has_been_called(self):
        for expression in (
            "до прошлой пятницы", "до пятницы на этой неделе", "до 15 числа", "31 ноября",
            "завтра в 25:00", "завтра утром", "к первой паре завтра", "завтра в 18:00 МСК",
        ):
            with self.subTest(expression=expression):
                parser = self.parser()
                with patch("llm.urlopen", side_effect=responses()) as opener:
                    result = await parser.parse("Лаба 3 по БД " + expression, NOW)
                self.assertEqual(opener.call_count, 2)
                self.assertEqual(result["status"], "manual_input_required")
                self.assertIsNone(result["due_date"])
                self.assertEqual(parser.last_remote_error, "response")

    async def test_malformed_or_uncertain_gigachat_reply_keeps_correction_manual(self):
        for answer in ("not JSON", json.dumps(dict(RESULT, confidence=0.2)),
                       json.dumps(dict(RESULT, due_date="2026-10-15 18:00"))):
            with self.subTest(answer=answer):
                parser = self.parser()
                fixture = responses()
                fixture[1] = Response({"choices": [{"message": {"content": answer}}]})
                with patch("llm.urlopen", side_effect=fixture) as opener:
                    result = await parser.parse("Лаба 3 по БД не 15 октября, а 16 октября в 18:00", NOW)
                self.assertEqual(opener.call_count, 2)
                self.assertEqual(result["status"], "manual_input_required")
                self.assertIsNone(result["due_date"])
                self.assertEqual(parser.last_remote_error, "response")

    def test_calendar_context_handles_week_boundaries_and_explicit_clock(self):
        monday = datetime(2026, 10, 12, 10, 30)
        for text, expected in (
            ("Лаба до ближайшей пятницы", "2026-10-16 23:59"),
            ("Лаба до следующей пятницы", "2026-10-23 23:59"),
            ("Лаба завтра в 6 вечера", "2026-10-13 18:00"),
            ("Лаба 15 октября", "2026-10-15 23:59"),
        ):
            with self.subTest(text=text):
                _, content = ParserEngine._messages(text, monday)
                self.assertEqual(json.loads(content)["verified_due_date"], expected)

    async def test_missing_key_or_explicit_empty_model_stays_offline(self):
        for key, model in (("", "GigaChat-2"), ("authorization-secret", "")):
            with self.subTest(key_present=bool(key), model=model):
                parser = ParserEngine(api_key=key, model=model, provider="gigachat")
                with patch("llm.urlopen", side_effect=AssertionError("Unexpected network request")):
                    result = await parser.parse("Лаба по БД завтра", NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")

    async def test_http_and_tls_failures_fall_back_with_safe_diagnostic_categories(self):
        for error, reason in (
            (HTTPError("https://private.example/secret", 401, "private-secret", {}, io.BytesIO(b"secret")), "auth"),
            (HTTPError("https://private.example/secret", 429, "private-secret", {}, io.BytesIO(b"secret")), "quota"),
            (URLError(ssl.SSLCertVerificationError("private-secret")), "tls"),
        ):
            with self.subTest(reason=reason):
                parser = self.parser()
                with patch("llm.urlopen", side_effect=error):
                    result = await parser.parse("Лаба по БД завтра", NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(parser.last_remote_error, reason)

    async def test_untrusted_remote_date_or_overlong_fields_are_not_accepted(self):
        for changes in ({"due_date": "2026-10-17 18:00"}, {"subject": "я" * 121}, {"task": "я" * 501}):
            with self.subTest(changes=next(iter(changes))):
                parser = self.parser()
                fake = dict(RESULT, **changes)
                fixture = responses()
                fixture[1] = Response({"choices": [{"message": {"content": json.dumps(fake)}}]})
                with patch("llm.urlopen", side_effect=fixture):
                    result = await parser.parse("Лаба 3 по БД до пятницы в 18:00", NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(result["due_date"], RESULT["due_date"])

    async def test_shared_budget_returns_before_slow_oauth_can_start_generation(self):
        parser = self.parser(timeout=0.02)
        token_response = responses()[0]

        def slow_oauth(*_, **__):
            time.sleep(0.08)
            return token_response

        started = time.monotonic()
        with patch("llm.urlopen", side_effect=slow_oauth) as opener:
            result = await parser.parse("Лаба 3 по БД до пятницы в 18:00", NOW)
            self.assertLess(time.monotonic() - started, 0.07)
            self.assertEqual(parser.last_remote_error, "timeout")
            self.assertEqual(result["parsing_method"], "regex_fallback")
            await asyncio.sleep(0.10)
            self.assertEqual(opener.call_count, 1)


class ModelCheckTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, **overrides):
        return Settings(llm_api_key="authorization-secret", llm_model="GigaChat-2",
                        llm_provider="gigachat", llm_api_url="https://api.giga.chat/v1/chat/completions",
                        llm_ca_bundle=ROOT / "certs" / "russian_trusted_root_ca.pem", **overrides)

    def test_parser_receives_chat_template_kwargs_and_legacy_settings_remain_supported(self):
        current = Settings(llm_provider="openai", llm_model="Qwen3.8-27B",
                           llm_chat_template_kwargs={"enable_thinking": False})
        legacy = SimpleNamespace(llm_api_key="", llm_model="", llm_timeout=2.5,
                                 llm_api_url="https://example.org/messages")
        for settings, expected in ((current, {"enable_thinking": False}), (legacy, None)):
            with self.subTest(legacy=settings is legacy), patch("main.ParserEngine") as parser:
                create_parser(settings)
                self.assertEqual(parser.call_args.kwargs["chat_template_kwargs"], expected)

    async def test_empty_key_never_opens_network_telegram_or_database(self):
        settings = Settings(llm_provider="gigachat", llm_model="GigaChat-2")
        output = io.StringIO()
        with patch("main.Database", side_effect=AssertionError("Unexpected DB access")), \
                patch("llm.urlopen", side_effect=AssertionError("Unexpected network request")), redirect_stdout(output):
            self.assertEqual(await check_llm(settings), 1)
        self.assertIn("LLM не подтверждён", output.getvalue())

    async def test_success_requires_valid_remote_result_without_database(self):
        output = io.StringIO()
        with patch("llm.urlopen", side_effect=responses()), patch("config.datetime") as clock, \
                patch("main.Database", side_effect=AssertionError("Unexpected DB access")), redirect_stdout(output):
            clock.now.return_value = NOW
            self.assertEqual(await check_llm(self.settings()), 0)
        self.assertIn("Проверка LLM пройдена: gigachat", output.getvalue())
        self.assertNotIn("authorization-secret", output.getvalue())
        self.assertNotIn("temporary-secret", output.getvalue())

    async def test_rejected_key_reports_failure_without_remote_secrets(self):
        output = io.StringIO()
        error = HTTPError("https://private.example/key", 401, "authorization-secret", {},
                          io.BytesIO(b"temporary-secret"))
        with patch("llm.urlopen", side_effect=error), redirect_stdout(output):
            self.assertEqual(await check_llm(self.settings()), 1)
        self.assertIn("сервис отклонил ключ", output.getvalue())
        for secret in ("authorization-secret", "temporary-secret", "private.example"):
            self.assertNotIn(secret, output.getvalue())

    def test_cli_model_check_requires_no_bot_token_and_avoids_database(self):
        settings = Settings(llm_provider="gigachat", llm_model="GigaChat-2")
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--check-llm"]), \
                patch("main.Settings.from_env", return_value=settings) as configuration, \
                patch("main.logging.basicConfig"), \
                patch("main.Database", side_effect=AssertionError("Unexpected DB access")), redirect_stdout(output):
            self.assertEqual(main(), 1)
        configuration.assert_called_once_with(require_token=False)
        self.assertIn("LLM не подтверждён", output.getvalue())

    async def test_rejected_model_or_parameters_fail_safely_without_database_or_telegram(self):
        settings = Settings(llm_provider="openai", llm_api_key="authorization-secret",
                            llm_model="configured-model", llm_api_url="https://private.example/api/chat/completions")
        for status in (400, 404, 405, 413, 415, 422):
            with self.subTest(status=status):
                output, errors = io.StringIO(), io.StringIO()
                body = io.BytesIO(b"temporary-secret")
                error = HTTPError("https://private.example/key?token=query-secret", status,
                                  "authorization-secret", {"Location": "https://secret-location.example"}, body)
                with patch("config.datetime") as clock, \
                        patch("llm.urlopen", side_effect=error) as opener, \
                        patch("main.Database", side_effect=AssertionError("Unexpected DB access")), \
                        patch("main.run", side_effect=AssertionError("Unexpected Telegram startup")), \
                        redirect_stdout(output), redirect_stderr(errors):
                    clock.now.return_value = NOW
                    self.assertEqual(await check_llm(settings), 1)
                self.assertEqual(opener.call_count, 1)
                self.assertTrue(body.closed)
                diagnostic = output.getvalue() + errors.getvalue()
                self.assertIn("LLM не подтверждён", diagnostic)
                self.assertIn("API отклонил модель или параметры запроса", diagnostic)
                self.assertIn("LLM_MODEL и LLM_API_URL", diagnostic)
                self.assertIn("Использован локальный разбор", diagnostic)
                for secret in ("authorization-secret", "temporary-secret", "private.example",
                               "query-secret", "secret-location.example"):
                    self.assertNotIn(secret, diagnostic)


if __name__ == "__main__":
    unittest.main()
