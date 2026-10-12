"""Exercise the probe process boundary with real children and no network."""

from contextlib import redirect_stdout
import io
import json
import re
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from config import Settings
from main import check_llm_bounded, main


REAL_POPEN = subprocess.Popen


class BoundedProbeTests(unittest.TestCase):
    def settings(self, **overrides):
        return Settings(llm_api_key="fixture-private-key", llm_model="fixture-model",
                        llm_provider="openai", llm_api_url="https://example.invalid/chat/completions",
                        bot_token="fixture-bot-token", **overrides)

    def invoke(self, code, *, timeout=0.5):
        children = []
        commands = []

        def spawn(command, **kwargs):
            commands.append(command)
            child = REAL_POPEN([sys.executable, "-c", code], **kwargs)
            children.append(child)
            return child

        output = io.StringIO()
        started = time.monotonic()
        with patch("main.subprocess.Popen", side_effect=spawn), redirect_stdout(output):
            result = check_llm_bounded(self.settings(llm_timeout=timeout))
        return result, output.getvalue(), children, commands, time.monotonic() - started

    def test_valid_result_and_credentials_cross_only_the_private_pipe(self):
        code = (
            "import json,sys,os; data=json.load(sys.stdin); "
            "assert data['llm_api_key']=='fixture-private-key'; "
            "assert 'bot_token' not in data and 'database_path' not in data; "
            "assert not any(key in os.environ for key in ('BOT_TOKEN','LLM_API_KEY','DATABASE_PATH')); "
            "print(json.dumps({'reason':None,'elapsed':0.01}))"
        )
        with patch.dict("os.environ", {"BOT_TOKEN": "fixture-bot-token", "LLM_API_KEY": "fixture-private-key",
                                       "DATABASE_PATH": "fixture-work.db"}):
            result, output, children, commands, _ = self.invoke(code)
        self.assertEqual(result, 0)
        self.assertIn("Проверка LLM пройдена: openai", output)
        self.assertEqual(children[0].returncode, 0)
        for secret in ("fixture-private-key", "fixture-bot-token"):
            self.assertNotIn(secret, output)
            self.assertNotIn(secret, repr(commands))

    def test_stalled_dns_or_http_process_is_killed_and_reaped(self):
        code = "import json,sys,time; json.load(sys.stdin); time.sleep(30)"
        result, output, children, _, elapsed = self.invoke(code, timeout=0.2)
        self.assertEqual(result, 1)
        self.assertIn("модель не ответила", output)
        self.assertIn("Проверка остановлена", output)
        self.assertLess(elapsed, 2)
        self.assertIsNotNone(children[0].poll())

    def test_fallback_result_exits_without_waiting_for_stalled_http_thread(self):
        # Exercise the actual worker: an uninterruptible HTTP/DNS call must
        # neither hold up asyncio.run nor keep the child process alive.
        code = """
import time
import main
from parser import ParserEngine
def slow(*args, **kwargs):
    time.sleep(30)
    raise TimeoutError()
def parser(settings):
    value = ParserEngine('fixture-private-key', 'fixture-model', timeout=0.02)
    value._request_llm = slow
    return value
main.create_parser = parser
main._llm_probe_worker()
"""
        result, output, children, _, elapsed = self.invoke(code, timeout=1.0)
        self.assertEqual(result, 1)
        self.assertIn("модель не ответила", output)
        reported = float(re.search(r"\(([\d.]+) с\)", output).group(1))
        self.assertLess(reported, 0.2)
        self.assertLess(elapsed, 0.8)
        self.assertEqual(children[0].returncode, 0)

    def test_invalid_child_output_and_tracebacks_are_not_forwarded(self):
        invalid = ["fixture-private-key", json.dumps({"reason": "fixture-private-key", "elapsed": 0.1}),
                   json.dumps({"reason": None, "elapsed": float("nan")})]
        for value in invalid:
            with self.subTest(value=value):
                code = f"import sys; sys.stdin.read(); print({value!r}); print('fixture-bot-token',file=sys.stderr)"
                result, output, children, _, _ = self.invoke(code)
                self.assertEqual(result, 1)
                self.assertIn("LLM не подтверждён", output)
                self.assertNotIn("fixture-private-key", output)
                self.assertNotIn("fixture-bot-token", output)
                self.assertIsNotNone(children[0].poll())

    def test_interrupted_probe_does_not_leave_child_running(self):
        child = REAL_POPEN([sys.executable, "-c", "import time; time.sleep(30)"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True)
        communicate = child.communicate
        try:
            with patch("main.subprocess.Popen", return_value=child), \
                    patch.object(child, "communicate", side_effect=[KeyboardInterrupt, ("", None)]) as ipc:
                with self.assertRaises(KeyboardInterrupt):
                    check_llm_bounded(self.settings())
                self.assertEqual(ipc.call_count, 2)
            child.wait(timeout=2)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:
                child.kill()
            communicate()

    def test_empty_credentials_do_not_spawn_and_cli_dispatch_uses_bounded_probe(self):
        output = io.StringIO()
        with patch("main.subprocess.Popen") as spawn, redirect_stdout(output):
            self.assertEqual(check_llm_bounded(Settings()), 1)
            spawn.assert_not_called()
        with patch("sys.argv", ["main.py", "--check-llm"]), \
                patch("main.Settings.from_env", return_value=self.settings()) as config, \
                patch("main.logging.basicConfig"), \
                patch("main.check_llm_bounded", return_value=0) as probe, \
                patch("main.Database", side_effect=AssertionError("Unexpected DB access")), \
                patch("main.run", side_effect=AssertionError("Unexpected Telegram startup")):
            self.assertEqual(main(), 0)
            config.assert_called_once_with(require_token=False)
            probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
