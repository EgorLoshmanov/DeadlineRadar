"""GigaChat transport checks use fake HTTP responses and deterministic clocks."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import ssl
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs
from uuid import UUID

from llm import CHAT_URL, GigaChatClient, MAX_RESPONSE_BYTES, OAUTH_URL


WALL_TIME = 1_790_000_000.0
VALID_RESULT = {"subject": "Базы данных", "task": "Лабораторная работа №3",
                "due_date": "2026-10-16 18:00", "confidence": 0.95}


class Response:
    def __init__(self, value):
        self.body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, size):
        self.read_sizes.append(size)
        return self.body[:size]


def oauth(token="access-token", seconds=1800):
    return Response({"access_token": token, "expires_at": int((WALL_TIME + seconds) * 1000)})


def chat(content=None):
    value = VALID_RESULT if content is None else content
    return Response({"choices": [{"message": {"role": "assistant", "content": json.dumps(value)}}]})


class GigaChatClientTests(unittest.TestCase):
    def setUp(self):
        wall_clock = patch("llm.time.time", return_value=WALL_TIME)
        self.wall_clock = wall_clock.start()
        self.addCleanup(wall_clock.stop)
        self.monotonic_value = 100.0
        monotonic_clock = patch("llm.time.monotonic", side_effect=lambda: self.monotonic_value)
        monotonic_clock.start()
        self.addCleanup(monotonic_clock.stop)
        self.context = ssl.create_default_context()
        context_factory = patch("llm.ssl.create_default_context", return_value=self.context)
        self.context_factory = context_factory.start()
        self.addCleanup(context_factory.stop)
        self.client = GigaChatClient("authorization-key")

    def test_oauth_and_generation_request_shapes(self):
        with patch("llm.urlopen", side_effect=[oauth(), chat()]) as opener:
            result = self.client.request("system instructions", "сообщение пользователя")
        self.assertEqual(result, VALID_RESULT)
        authorization, generation = opener.call_args_list
        token_request = authorization.args[0]
        self.assertEqual(token_request.full_url, OAUTH_URL)
        self.assertEqual(token_request.get_method(), "POST")
        self.assertEqual(parse_qs(token_request.data.decode()), {"scope": ["GIGACHAT_API_PERS"]})
        self.assertEqual(token_request.get_header("Authorization"), "Basic authorization-key")
        self.assertEqual(token_request.get_header("Content-type"), "application/x-www-form-urlencoded")
        self.assertEqual(UUID(token_request.get_header("Rquid")).version, 4)
        request = generation.args[0]
        self.assertEqual(request.full_url, CHAT_URL)
        self.assertEqual(request.get_header("Authorization"), "Bearer access-token")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "GigaChat-2")
        self.assertEqual(body["messages"], [{"role": "system", "content": "system instructions"},
                                            {"role": "user", "content": "сообщение пользователя"}])
        self.assertFalse(body["stream"])
        for call in opener.call_args_list:
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 2.5)
            self.assertIs(call.kwargs["context"], self.context)

    def test_cached_token_skips_second_oauth_request(self):
        with patch("llm.urlopen", side_effect=[oauth(), chat(), chat()]) as opener:
            self.client.request("system", "first")
            self.client.request("system", "second")
        self.assertEqual([call.args[0].full_url for call in opener.call_args_list],
                         [OAUTH_URL, CHAT_URL, CHAT_URL])

    def test_token_refreshes_thirty_seconds_before_expiry(self):
        with patch("llm.urlopen", side_effect=[oauth("first", 60), chat(), chat(), oauth("second"), chat()]) as opener:
            self.client.request("system", "first")
            self.wall_clock.return_value = WALL_TIME + 29
            self.client.request("system", "cached")
            self.wall_clock.return_value = WALL_TIME + 30
            self.client.request("system", "refreshed")
        self.assertEqual([call.args[0].full_url for call in opener.call_args_list],
                         [OAUTH_URL, CHAT_URL, CHAT_URL, OAUTH_URL, CHAT_URL])
        self.assertEqual(opener.call_args.args[0].get_header("Authorization"), "Bearer second")

    def test_invalid_oauth_schema_does_not_start_generation(self):
        for value in (
            {}, {"access_token": "t"}, {"access_token": "", "expires_at": int((WALL_TIME + 1800) * 1000)},
            {"access_token": [], "expires_at": int((WALL_TIME + 1800) * 1000)},
            {"access_token": "t\nsecret", "expires_at": int((WALL_TIME + 1800) * 1000)},
            {"access_token": "t\x00secret", "expires_at": int((WALL_TIME + 1800) * 1000)},
            {"access_token": "t", "expires_at": str(int((WALL_TIME + 1800) * 1000))},
            {"access_token": "t", "expires_at": int(WALL_TIME * 1000)},
            {"access_token": "t", "expires_at": True},
            {"access_token": "t", "expires_at": 10**400},
        ):
            with self.subTest(value=value):
                client = GigaChatClient("key")
                with patch("llm.urlopen", return_value=Response(value)) as opener:
                    with self.assertRaisesRegex(ValueError, "авторизации"):
                        client.request("system", "message")
                self.assertEqual(opener.call_count, 1)
                self.assertEqual(client._access_token, "")

    def test_generation_401_invalidates_token_without_retry(self):
        error = HTTPError(CHAT_URL, 401, "secret details", {}, io.BytesIO(b"secret body"))
        with patch("llm.urlopen", side_effect=[oauth(), error, oauth("new-token"), chat()]) as opener:
            with self.assertRaisesRegex(ValueError, "HTTP 401") as raised:
                self.client.request("system", "first")
            self.assertNotIn("secret", str(raised.exception))
            self.assertEqual(opener.call_count, 2)
            self.assertEqual(self.client._access_token, "")
            self.client.request("system", "second")
        self.assertEqual([call.args[0].full_url for call in opener.call_args_list],
                         [OAUTH_URL, CHAT_URL, OAUTH_URL, CHAT_URL])

    def test_rate_limit_and_authorization_errors_do_not_retry(self):
        for stage, code in (("chat", 429), ("oauth", 401), ("chat", 500)):
            with self.subTest(stage=stage, code=code):
                client = GigaChatClient("key")
                endpoint = CHAT_URL if stage == "chat" else OAUTH_URL
                error = HTTPError(endpoint, code, "private response text", {}, io.BytesIO(b"private body"))
                responses = [oauth(), error] if stage == "chat" else [error]
                with patch("llm.urlopen", side_effect=responses) as opener:
                    with self.assertRaises(ValueError) as raised:
                        client.request("private prompt", "private message")
                self.assertEqual(opener.call_count, len(responses))
                self.assertNotIn("private", str(raised.exception))
                self.assertIn(f"HTTP {code}", str(raised.exception))

    def test_network_and_tls_errors_are_sanitized(self):
        for error in (URLError("authorization-key and message"),
                      ssl.SSLCertVerificationError("authorization-key and message"),
                      OSError("authorization-key and message")):
            with self.subTest(error=type(error).__name__):
                with patch("llm.urlopen", side_effect=error):
                    with self.assertRaises(ValueError) as raised:
                        self.client.request("system", "message")
                self.assertNotIn("authorization-key", str(raised.exception))
                self.assertNotIn("message", str(raised.exception))

    def test_malformed_choices_and_model_json_are_rejected(self):
        for value in (
            {}, {"choices": []}, {"choices": {}}, {"choices": [None]},
            {"choices": [{"message": None}]}, {"choices": [{"message": {"content": {}}}]},
            {"choices": [{"message": {"content": "[]"}}]},
            {"choices": [{"message": {"content": "not JSON"}}]},
            {"choices": [{"message": {"content": '```json\n{"value":1}\n```'}}]},
            {"choices": [{"message": {"content": '{"value":NaN}'}}]},
            {"choices": [{"message": {"content": '{"value":1,"value":2}'}}]},
        ):
            with self.subTest(value=value):
                client = GigaChatClient("key")
                with patch("llm.urlopen", side_effect=[oauth(), Response(value)]):
                    with self.assertRaises(ValueError):
                        client.request("system", "message")

    def test_malformed_transport_json_is_rejected(self):
        for raw in (b"not-json", b"[]", b'{"token": NaN}', b'\xff', b'{"a":1,"a":2}'):
            with self.subTest(raw=raw):
                client = GigaChatClient("key")
                with patch("llm.urlopen", return_value=Response(raw)):
                    with self.assertRaises(ValueError):
                        client.request("system", "message")

    def test_response_size_is_limited_for_both_endpoints(self):
        for stage in ("oauth", "chat"):
            with self.subTest(stage=stage):
                client = GigaChatClient("key")
                oversized = Response(b"x" * (MAX_RESPONSE_BYTES + 100))
                responses = [oversized] if stage == "oauth" else [oauth(), oversized]
                with patch("llm.urlopen", side_effect=responses):
                    with self.assertRaisesRegex(ValueError, "размер"):
                        client.request("system", "message")
                self.assertEqual(oversized.read_sizes, [MAX_RESPONSE_BYTES + 1])

    def test_one_budget_is_shared_between_oauth_and_generation(self):
        def response(request, **kwargs):
            if request.full_url == OAUTH_URL:
                self.assertAlmostEqual(kwargs["timeout"], 2.5)
                self.monotonic_value += 1.75
                return oauth()
            self.assertAlmostEqual(kwargs["timeout"], 0.75)
            return chat()

        with patch("llm.urlopen", side_effect=response) as opener:
            self.assertEqual(self.client.request("system", "message"), VALID_RESULT)
        self.assertEqual(opener.call_count, 2)

    def test_expired_budget_after_oauth_prevents_late_generation(self):
        def late_token(*_, **kwargs):
            self.monotonic_value += 2.6
            return oauth()

        with patch("llm.urlopen", side_effect=late_token) as opener:
            with self.assertRaises(TimeoutError):
                self.client.request("system", "message")
        self.assertEqual(opener.call_count, 1)

    def test_external_deadline_accounts_for_executor_delay(self):
        with patch("llm.urlopen") as opener:
            with self.assertRaises(TimeoutError):
                self.client.request("system", "message", deadline=99.0)
        opener.assert_not_called()
        self.context_factory.assert_not_called()

    def test_external_deadline_can_shorten_but_not_extend_own_budget(self):
        for deadline, expected in ((100.25, 0.25), (200.0, 2.5)):
            with self.subTest(deadline=deadline):
                client = GigaChatClient("key")
                with patch("llm.urlopen", side_effect=[oauth(), chat()]) as opener:
                    client.request("system", "message", deadline=deadline)
                self.assertAlmostEqual(opener.call_args_list[0].kwargs["timeout"], expected)

    def test_tls_initialization_consumes_the_same_budget(self):
        def slow_context():
            self.monotonic_value += 2.6
            return self.context

        self.context_factory.side_effect = slow_context
        with patch("llm.urlopen") as opener:
            with self.assertRaises(TimeoutError):
                self.client.request("system", "message")
        opener.assert_not_called()

    def test_token_and_generation_share_one_nonblocking_lock(self):
        for blocked_stage in ("oauth", "chat"):
            with self.subTest(stage=blocked_stage):
                client = GigaChatClient("key")
                entered = threading.Event()
                release = threading.Event()

                def response(request, **kwargs):
                    stage = "oauth" if request.full_url == OAUTH_URL else "chat"
                    if stage == blocked_stage:
                        entered.set()
                        if not release.wait(2):
                            raise AssertionError("Test worker was not released")
                    return oauth() if stage == "oauth" else chat()

                with patch("llm.urlopen", side_effect=response) as opener:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(client.request, "system", "first")
                        self.assertTrue(entered.wait(1))
                        try:
                            calls_before = opener.call_count
                            with self.assertRaisesRegex(ValueError, "занят"):
                                client.request("system", "second")
                            self.assertEqual(opener.call_count, calls_before)
                        finally:
                            release.set()
                        self.assertEqual(future.result(timeout=2), VALID_RESULT)
                    self.assertEqual(client.request("system", "after completion"), VALID_RESULT)

    def test_tls_context_is_lazy_verified_and_accepts_an_application_ca(self):
        bundle = Path("certificates/test-root.pem")
        client = GigaChatClient("key", ca_bundle=bundle)
        self.context_factory.assert_not_called()
        self.assertTrue(self.context.check_hostname)
        self.assertEqual(self.context.verify_mode, ssl.CERT_REQUIRED)
        with patch.object(self.context, "load_verify_locations") as load_ca:
            with patch("llm.urlopen", side_effect=[oauth(), chat(), chat()]) as opener:
                client.request("system", "first")
                client.request("system", "second")
        load_ca.assert_called_once_with(cafile=str(bundle))
        self.context_factory.assert_called_once_with()
        for call in opener.call_args_list:
            self.assertIs(call.kwargs["context"], self.context)
        self.assertTrue(self.context.check_hostname)
        self.assertEqual(self.context.verify_mode, ssl.CERT_REQUIRED)

    def test_missing_key_and_invalid_configuration_do_not_open_connections(self):
        with patch("llm.urlopen") as opener:
            with self.assertRaises(ValueError):
                GigaChatClient("").request("system", "message")
        opener.assert_not_called()
        self.context_factory.assert_not_called()
        for settings in ({"timeout": 0}, {"timeout": float("nan")}, {"timeout": True},
                         {"api_url": "http://example.invalid"}, {"model": ""}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                GigaChatClient("key", **settings)
        for key in ("key\nsecret", "key\x00secret", "key\x7fsecret"):
            with self.subTest(key=repr(key)), self.assertRaises(ValueError) as raised:
                GigaChatClient(key)
            self.assertNotIn("secret", str(raised.exception))
        self.assertEqual(GigaChatClient("key", timeout=10).timeout, 2.5)


if __name__ == "__main__":
    unittest.main()
