"""Credential-bearing requests reject redirects without opening a real socket."""

from email.message import Message
import io
import ssl
import unittest
from unittest.mock import Mock, patch
from urllib.request import HTTPSHandler, Request
from urllib.response import addinfourl

from llm import LLMError, OpenAICompatibleClient, _NoRedirect, urlopen


class RedirectTests(unittest.TestCase):
    def test_handler_never_constructs_a_redirected_credential_request(self):
        request = Request("https://api.example/chat/completions", data=b"{}",
                          headers={"Authorization": "Bearer test-key"})
        handler = _NoRedirect()
        for code in (301, 302, 303, 307, 308):
            for target in ("https://api.example/other", "https://other.example/receiver",
                           "http://other.example/receiver"):
                with self.subTest(code=code, target=target):
                    self.assertIsNone(handler.redirect_request(request, None, code, "Redirect", {}, target))

    def test_wrapper_preserves_tls_context_and_timeout_without_installing_global_opener(self):
        context = ssl.create_default_context()
        request = Request("https://api.example/chat/completions", data=b"{}")
        https_handler = Mock()
        response = Mock()
        opener = Mock()
        opener.open.return_value = response
        with patch("llm.HTTPSHandler", return_value=https_handler) as tls_factory, \
                patch("llm.build_opener", return_value=opener) as builder, \
                patch("urllib.request.install_opener") as global_install:
            self.assertIs(urlopen(request, timeout=0.75, context=context), response)
        tls_factory.assert_called_once_with(context=context)
        builder.assert_called_once()
        self.assertIs(builder.call_args.args[0], https_handler)
        self.assertIsInstance(builder.call_args.args[1], _NoRedirect)
        opener.open.assert_called_once_with(request, timeout=0.75)
        global_install.assert_not_called()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_real_opener_rejects_redirect_and_client_sanitizes_failure(self):
        # Exercise the actual opener/HTTPErrorProcessor/redirect-handler chain.
        # Only HTTPS transport is replaced; there are no socket or API calls.
        class RedirectingHTTPS(HTTPSHandler):
            def __init__(self, code, target):
                super().__init__()
                self.code, self.target = code, target
                self.requests = []
                self.body = io.BytesIO(b"private-server-error-body")

            def https_open(self, request):
                self.requests.append(request.full_url)
                headers = Message()
                headers["Location"] = self.target
                response = addinfourl(self.body, headers, request.full_url, self.code)
                response.msg = "Redirect with private-server-error-body"
                return response

        for code in (301, 302, 303, 307, 308):
            for target in ("https://api.example/other", "https://other.example/receiver",
                           "http://other.example/receiver"):
                with self.subTest(code=code, target=target):
                    transport = RedirectingHTTPS(code, target)
                    client = OpenAICompatibleClient("test-key", api_url="https://api.example/chat/completions")
                    with patch("llm.HTTPSHandler", return_value=transport), \
                            patch("urllib.request.HTTPHandler.http_open",
                                  side_effect=AssertionError("Redirect reached HTTP transport")), \
                            self.assertRaises(LLMError) as caught:
                        client.request("system", "test message")
                    self.assertEqual(caught.exception.reason, "network")
                    self.assertNotIn("private-server-error-body", str(caught.exception))
                    self.assertNotIn("test-key", str(caught.exception))
                    self.assertEqual(transport.requests, ["https://api.example/chat/completions"])
                    self.assertTrue(transport.body.closed)
                    self.assertTrue(client._request_lock.acquire(blocking=False))
                    client._request_lock.release()


if __name__ == "__main__":
    unittest.main()
