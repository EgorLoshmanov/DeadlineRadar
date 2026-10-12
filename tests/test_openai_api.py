"""NSU-compatible transport and parser checks without network or working data."""

import asyncio
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import ssl
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from config import Settings
from llm import (LLMError, MAX_RESPONSE_BYTES, OPENAI_CHAT_URL, OPENAI_DEFAULT_MODEL,
                 OpenAICompatibleClient)
from main import check_llm
from parser import ParserEngine


NOW = datetime(2026, 10, 11, 12, 0)
TEXT = "Лаба 3 по БД до пятницы в 18:00"
RESULT = {"subject": "Базы данных", "task": "Лабораторная работа №3",
          "due_date": "2026-10-16 18:00", "confidence": 0.95}


class Response:
    def __init__(self, value):
        self.raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, maximum):
        self.read_sizes.append(maximum)
        return self.raw[:maximum]


def completion(value=None, *, finish_reason="stop"):
    return Response({"choices": [{"message": {"content": json.dumps(RESULT if value is None else value)},
                                  "finish_reason": finish_reason}]})


class CompatibleTransportTests(unittest.TestCase):
    def test_direct_bearer_requests_never_exchange_oauth_tokens(self):
        client = OpenAICompatibleClient(" nsu-secret ")
        with patch("llm.urlopen", side_effect=[completion(), completion()]) as opener:
            self.assertEqual(client.request("rules", "сообщение"), RESULT)
            self.assertEqual(client.request("rules", "another message"), RESULT)
        self.assertEqual(opener.call_count, 2)
        for call in opener.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, OPENAI_CHAT_URL)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer nsu-secret")
            self.assertEqual(request.get_header("User-agent"), "DeadlineRadar/1.0")
            self.assertIsNone(request.get_header("X-api-key"))
            self.assertIsNone(request.get_header("Rquid"))
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], OPENAI_DEFAULT_MODEL)
            self.assertEqual(payload["messages"][0], {"role": "system", "content": "rules"})
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["max_tokens"], 512)
            self.assertLessEqual(call.kwargs["timeout"], 5.0)
            self.assertTrue(call.kwargs["context"].check_hostname)
            self.assertEqual(call.kwargs["context"].verify_mode, ssl.CERT_REQUIRED)

    def test_qwen_and_custom_endpoint_are_not_rewritten(self):
        client = OpenAICompatibleClient("key", model="Qwen3.8-27B",
                                        api_url="https://example.org/custom/chat/completions")
        with patch("llm.urlopen", return_value=completion()) as opener:
            self.assertEqual(client.request("rules", "text"), RESULT)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.org/custom/chat/completions")
        self.assertEqual(json.loads(request.data)["model"], "Qwen3.8-27B")

    def test_chat_template_controls_are_optional_top_level_fields(self):
        for kwargs in (None, {}, {"enable_thinking": False}, {"thinking": False}):
            with self.subTest(kwargs=kwargs):
                client = OpenAICompatibleClient("key", chat_template_kwargs=kwargs)
                with patch("llm.urlopen", return_value=completion()) as opener:
                    client.request("rules", "text")
                payload = json.loads(opener.call_args.args[0].data)
                self.assertNotIn("extra_body", payload)
                if kwargs:
                    self.assertEqual(payload["chat_template_kwargs"], kwargs)
                else:
                    self.assertNotIn("chat_template_kwargs", payload)

    def test_template_controls_are_copied_and_reject_untrusted_values(self):
        controls = {"enable_thinking": False}
        client = OpenAICompatibleClient("key", chat_template_kwargs=controls)
        controls["enable_thinking"] = True
        self.assertEqual(client.chat_template_kwargs, {"enable_thinking": False})
        for invalid in ([], "private value", {"enable_thinking": 0}, {"thinking": "false"},
                        {"Authorization": "nsu-secret"}):
            with self.subTest(invalid=invalid), patch("llm.urlopen") as opener:
                with self.assertRaises(ValueError) as raised:
                    OpenAICompatibleClient("key", chat_template_kwargs=invalid)
                self.assertNotIn("nsu-secret", str(raised.exception))
                self.assertNotIn("private value", str(raised.exception))
                opener.assert_not_called()

    def test_reasoning_controls_are_optional_and_validated_before_network_access(self):
        for effort in (None, "none", "low", "medium", "high", "default"):
            with self.subTest(effort=effort):
                client = OpenAICompatibleClient("key", reasoning_effort=effort)
                with patch("llm.urlopen", return_value=completion()) as opener:
                    client.request("rules", "text")
                payload = json.loads(opener.call_args.args[0].data)
                if effort is None:
                    self.assertNotIn("reasoning_effort", payload)
                else:
                    self.assertEqual(payload["reasoning_effort"], effort)
        for invalid in (True, [], {}, "private-config-secret", ""):
            with self.subTest(invalid=invalid), patch("llm.urlopen") as opener:
                with self.assertRaises(ValueError) as raised:
                    OpenAICompatibleClient("key", reasoning_effort=invalid)
                self.assertNotIn("private-config-secret", str(raised.exception))
                opener.assert_not_called()

    def test_http_errors_are_categorized_sanitized_and_never_retried(self):
        for code, reason in ((401, "auth"), (403, "auth"), (429, "quota"), (500, "network"),
                             (400, "request"), (404, "request"), (405, "request"),
                             (413, "request"), (415, "request"), (422, "request")):
            with self.subTest(code=code):
                error = HTTPError(OPENAI_CHAT_URL, code, "nsu-secret", {}, io.BytesIO(b"private response"))
                with patch("llm.urlopen", side_effect=error) as opener:
                    with self.assertRaises(LLMError) as raised:
                        OpenAICompatibleClient("nsu-secret").request("private rules", "private text")
                self.assertEqual(raised.exception.reason, reason)
                self.assertNotIn("secret", str(raised.exception))
                self.assertNotIn("private", str(raised.exception))
                self.assertEqual(opener.call_count, 1)

    def test_network_and_certificate_failures_have_safe_diagnostics(self):
        for error, reason in ((URLError("nsu-secret"), "network"),
                              (ssl.SSLCertVerificationError("nsu-secret"), "tls")):
            with self.subTest(reason=reason), patch("llm.urlopen", side_effect=error):
                with self.assertRaises(LLMError) as raised:
                    OpenAICompatibleClient("nsu-secret").request("rules", "text")
                self.assertEqual(raised.exception.reason, reason)
                self.assertNotIn("nsu-secret", str(raised.exception))

    def test_wrapped_socket_timeout_preserves_timeout_diagnostics(self):
        error = URLError(TimeoutError("nsu-secret and request text"))
        with patch("llm.urlopen", side_effect=error):
            with self.assertRaises(TimeoutError) as raised:
                OpenAICompatibleClient("nsu-secret").request("rules", "text")
        self.assertNotIn("nsu-secret", str(raised.exception))
        self.assertNotIn("request text", str(raised.exception))

    def test_incomplete_and_invalid_results_are_rejected(self):
        cases = [
            Response({"choices": []}),
            Response({"choices": [{"message": {"content": None}}]}),
            Response({"choices": [{"message": {"content": "```json\n{}\n```"}}]}),
            Response({"choices": [{"message": {"content": '{"confidence":NaN}'}}]}),
            Response({"choices": [{"message": {"content": '{"task":"a","task":"b"}'}}]}),
            Response({"choices": [{"message": {"content": "[" * 2000 + "]" * 2000}}]}),
            completion(finish_reason="length"),
            completion(finish_reason="tool_calls"),
            Response(b"<html>Login page</html>"),
        ]
        for response in cases:
            with self.subTest(raw=response.raw[:100]), patch("llm.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    OpenAICompatibleClient("key").request("rules", "text")

    def test_response_read_is_bounded(self):
        response = Response(b"x" * (MAX_RESPONSE_BYTES + 100))
        with patch("llm.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "размер"):
                OpenAICompatibleClient("key").request("rules", "text")
        self.assertEqual(response.read_sizes, [MAX_RESPONSE_BYTES + 1])

    def test_expired_worker_cannot_start_a_late_request(self):
        with patch("llm.time.monotonic", return_value=100), patch("llm.urlopen") as opener:
            with self.assertRaises(TimeoutError):
                OpenAICompatibleClient("key").request("rules", "text", deadline=99)
        opener.assert_not_called()

    def test_missing_or_unsafe_configuration_never_opens_network(self):
        with patch("llm.urlopen") as opener:
            with self.assertRaises(ValueError):
                OpenAICompatibleClient("").request("rules", "text")
            for values in ({"api_key": "key\nsecret"}, {"api_url": "http://example.org"},
                           {"api_url": "https://user:password@example.org"}, {"model": ""}):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    OpenAICompatibleClient(**dict({"api_key": "key"}, **values))
        opener.assert_not_called()

    def test_malformed_endpoints_are_rejected_before_opening_network(self):
        endpoints = ("https://", "https:///messages", "https://[broken/messages",
                     "https://example.org:bad/messages", "https://example.org:65536/messages",
                     "https://example.org:0/messages", "https://exa mple.org/messages",
                     "https://example.org/path with space", "https://exam\nple.org/messages",
                     "https://@example.org/messages")
        with patch("llm.urlopen") as opener:
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "HTTPS"):
                    OpenAICompatibleClient("fixture-key", api_url=endpoint)
        opener.assert_not_called()


class CompatibleParserTests(unittest.IsolatedAsyncioTestCase):
    def parser(self, **changes):
        options = {"api_key": "nsu-secret", "model": OPENAI_DEFAULT_MODEL, "provider": "openai"}
        options.update(changes)
        return ParserEngine(**options)

    async def test_groq_request_identifies_app_and_disables_reasoning(self):
        endpoint = "https://api.groq.com/openai/v1/chat/completions"

        def respond(request, **kwargs):
            if request.get_header("User-agent") != "DeadlineRadar/1.0":
                raise HTTPError(endpoint, 403, "Forbidden", {}, io.BytesIO(b"error code: 1010"))
            self.assertEqual(request.full_url, endpoint)
            self.assertEqual(request.get_header("Authorization"), "Bearer fixture-groq-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "qwen/qwen3.8-27b")
            self.assertEqual(payload["reasoning_effort"], "none")
            self.assertNotIn("chat_template_kwargs", payload)
            return completion()

        with patch("llm.urlopen", side_effect=respond) as opener:
            result = await self.parser(api_key="fixture-groq-key", model="qwen/qwen3.8-27b",
                                       api_url=endpoint, reasoning_effort="none").parse(TEXT, NOW)
        self.assertEqual(result["parsing_method"], "llm")
        self.assertEqual(result["due_date"], RESULT["due_date"])
        opener.assert_called_once()

    async def test_remote_result_uses_calendar_hints_and_existing_validation(self):
        with patch("llm.urlopen", return_value=completion()) as opener:
            result = await self.parser(chat_template_kwargs={"enable_thinking": False}).parse(TEXT, NOW)
        self.assertEqual(result["parsing_method"], "llm")
        self.assertEqual(result["due_date"], RESULT["due_date"])
        self.assertEqual(opener.call_count, 1)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, OPENAI_CHAT_URL)
        payload = json.loads(request.data)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        context = json.loads(payload["messages"][1]["content"])
        self.assertEqual(context["untrusted_message"], TEXT)
        self.assertEqual(context["verified_due_date"], RESULT["due_date"])
        self.assertEqual(context["required_assignment_numbers"], ["3"])

    async def test_relative_deadlines_keep_valid_model_result_and_calendar_hints(self):
        cases = (
            ("Через 5 часов нужно скинуть код по Питону", NOW, "2026-10-11 17:00"),
            ("Через 5 часов нужно скинуть код по Питону", datetime(2026, 10, 11, 23, 40),
             "2026-10-12 04:40"),
            ("Через 30 минут нужно скинуть код по Python", datetime(2026, 12, 31, 23, 45),
             "2027-01-01 00:15"),
            ("Через 2 часа 15 минут нужно скинуть код по Питону", NOW, "2026-10-11 14:15"),
        )
        for text, now, due in cases:
            with self.subTest(text=text, now=now):
                answer = dict(RESULT, subject="Питон", task="Скинуть код", due_date=due)
                with patch("llm.urlopen", return_value=completion(answer)) as opener:
                    result = await self.parser(chat_template_kwargs={"enable_thinking": False}).parse(text, now)
                self.assertEqual(result["parsing_method"], "llm")
                self.assertEqual(result["subject"], "Python")
                self.assertEqual(result["due_date"], due)
                self.assertEqual(opener.call_count, 1)
                payload = json.loads(opener.call_args.args[0].data)
                context = json.loads(payload["messages"][1]["content"])
                self.assertEqual(context["verified_due_date"], due)
                self.assertNotIn("required_assignment_numbers", context)

    async def test_relative_arithmetic_cannot_be_overridden_by_model_confidence(self):
        text = "Через 5 часов нужно скинуть код по Питону"
        for due in ("2026-10-11 23:59", "2026-10-12 17:00"):
            with self.subTest(due=due):
                answer = dict(RESULT, subject="Python", task="Скинуть код", due_date=due, confidence=1.0)
                with patch("llm.urlopen", return_value=completion(answer)):
                    parser = self.parser()
                    result = await parser.parse(text, NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(result["due_date"], "2026-10-11 17:00")
                self.assertEqual(result["subject"], "Python")
                self.assertEqual(parser.last_remote_error, "response")

    async def test_relative_deadlines_still_work_when_model_is_unavailable(self):
        with patch("llm.urlopen", side_effect=URLError("offline")):
            result = await self.parser().parse("Через 5 часов нужно скинуть код по Питону", NOW)
        self.assertEqual(result["parsing_method"], "regex_fallback")
        self.assertEqual(result["due_date"], "2026-10-11 17:00")
        self.assertEqual(result["subject"], "Python")

    async def test_conflicting_relative_deadlines_require_confirmation(self):
        for text in (
            "Через 5 часов или через 6 часов скинуть код по Питону",
            "Сегодня через 36 часов скинуть код по Питону",
            "Не через 5 часов скинуть код по Питону",
        ):
            with self.subTest(text=text):
                answer = dict(RESULT, subject="Python", task="Скинуть код", due_date="2026-10-11 17:00")
                with patch("llm.urlopen", return_value=completion(answer)):
                    result = await self.parser().parse(text, NOW)
                self.assertEqual(result["status"], "manual_input_required")
                self.assertIsNone(result["due_date"])

    async def test_unsafe_model_results_fall_back_to_the_correct_local_deadline(self):
        for changes in ({"due_date": "2026-10-15 18:00"}, {"task": "Лабораторная работа"},
                        {"confidence": 0.1}, {"subject": "x" * 200}, {"task": "x" * 1000}):
            with self.subTest(changes=changes), patch("llm.urlopen", return_value=completion(dict(RESULT, **changes))):
                parser = self.parser()
                result = await parser.parse(TEXT, NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(result["due_date"], RESULT["due_date"])
                self.assertEqual(parser.last_remote_error, "response")

    async def test_transport_failure_keeps_local_parsing_available(self):
        for error, reason in ((HTTPError(OPENAI_CHAT_URL, 401, "secret", {}, io.BytesIO()), "auth"),
                              (HTTPError(OPENAI_CHAT_URL, 429, "secret", {}, io.BytesIO()), "quota"),
                              (HTTPError(OPENAI_CHAT_URL, 400, "secret", {}, io.BytesIO()), "request"),
                              (URLError("private"), "network"), (TimeoutError(), "timeout")):
            with self.subTest(reason=reason), patch("llm.urlopen", side_effect=error):
                parser = self.parser()
                result = await parser.parse(TEXT, NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(result["due_date"], RESULT["due_date"])
                self.assertEqual(parser.last_remote_error, reason)

    async def test_empty_key_model_and_disabled_mode_never_use_network(self):
        for options in ({"api_key": ""}, {"model": ""}, {"use_llm": False}):
            with self.subTest(options=options), patch("llm.urlopen") as opener:
                result = await self.parser(**options).parse(TEXT, NOW)
                self.assertEqual(result["parsing_method"], "regex_fallback")
                opener.assert_not_called()

    async def test_timed_out_request_keeps_lock_until_worker_finishes(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def slow_response(*_, **__):
            entered.set()
            try:
                if not release.wait(1):
                    raise AssertionError("Worker was not released")
                return completion()
            finally:
                finished.set()

        parser = self.parser(timeout=0.05)
        with patch("llm.urlopen", side_effect=slow_response) as opener:
            try:
                first = await parser.parse(TEXT, NOW)
                self.assertTrue(entered.is_set())
                self.assertEqual(first["parsing_method"], "regex_fallback")
                self.assertEqual(parser.last_remote_error, "timeout")
                second = await parser.parse(TEXT, NOW)
                self.assertEqual(second["parsing_method"], "regex_fallback")
                self.assertEqual(parser.last_remote_error, "busy")
                self.assertEqual(opener.call_count, 1)
            finally:
                release.set()
                await asyncio.to_thread(finished.wait, 1)

    async def test_cli_check_requires_a_valid_remote_result_without_touching_database(self):
        settings = Settings(llm_api_key="nsu-secret", llm_provider="openai", llm_model=OPENAI_DEFAULT_MODEL,
                            llm_api_url=OPENAI_CHAT_URL)
        for response, expected_exit in ((completion(), 0), (completion(dict(RESULT, confidence=0.1)), 1)):
            with self.subTest(expected_exit=expected_exit):
                output = io.StringIO()
                with patch("config.datetime") as clock, patch("llm.urlopen", return_value=response), \
                        patch("main.Database", side_effect=AssertionError("Unexpected DB access")), redirect_stdout(output):
                    clock.now.return_value = NOW
                    self.assertEqual(await check_llm(settings), expected_exit)
                self.assertNotIn("nsu-secret", output.getvalue())
                self.assertIn("Проверка LLM пройдена: openai" if expected_exit == 0 else "LLM не подтверждён",
                              output.getvalue())


if __name__ == "__main__":
    unittest.main()
