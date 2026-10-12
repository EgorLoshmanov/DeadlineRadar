"""Blocking model requests stay bounded and cannot delay process shutdown."""

import asyncio
from datetime import datetime
import threading
import time
import unittest
from unittest.mock import Mock, patch

from llm import LLMError, RequestWorker
from parser import ParserEngine


class RequestWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_results_errors_and_slot_reuse(self):
        worker = RequestWorker()
        self.assertEqual(await worker.run(lambda value: value + 1, 4), 5)
        with self.assertRaisesRegex(ValueError, "fixture"):
            await worker.run(Mock(side_effect=ValueError("fixture")))
        self.assertEqual(await worker.run(lambda: "recovered"), "recovered")

    async def test_parser_timeout_keeps_one_worker_until_request_finishes(self):
        parser = ParserEngine("fixture-key", "fixture-model", timeout=0.03)
        release = threading.Event()
        captured = []

        def blocked(*args, **kwargs):
            captured.append(threading.current_thread())
            release.wait(2)
            raise TimeoutError("fixture")

        try:
            with patch.object(parser, "_request_llm", side_effect=blocked) as request:
                result = await parser.parse("Лаба по БД завтра", datetime(2026, 10, 12, 12))
                self.assertEqual(result["parsing_method"], "regex_fallback")
                self.assertEqual(parser.last_remote_error, "timeout")
                self.assertTrue(captured[0].daemon)
                started = time.monotonic()
                for _ in range(10):
                    result = await parser.parse("Лаба по БД завтра", datetime(2026, 10, 12, 12))
                    self.assertEqual(result["parsing_method"], "regex_fallback")
                    self.assertEqual(parser.last_remote_error, "busy")
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(request.call_count, 1)
                release.set()
                captured[0].join(timeout=1)
                self.assertFalse(captured[0].is_alive())
            await asyncio.sleep(0)  # The abandoned future must ignore its late error.
            with patch.object(parser, "_request_llm", side_effect=TimeoutError) as request:
                await parser.parse("Лаба по БД завтра", datetime(2026, 10, 12, 12))
                request.assert_called_once()
                self.assertEqual(parser.last_remote_error, "timeout")
        finally:
            release.set()
            for thread in captured:
                thread.join(timeout=1)

    async def test_cancellation_retains_busy_slot_and_ignores_late_success(self):
        worker = RequestWorker()
        release = threading.Event()
        entered = threading.Event()
        captured = []

        def blocked():
            captured.append(threading.current_thread())
            entered.set()
            release.wait(2)
            return "late"

        task = asyncio.create_task(worker.run(blocked))
        try:
            await asyncio.sleep(0.01)
            self.assertTrue(entered.is_set())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(LLMError) as raised:
                await worker.run(lambda: "overlap")
            self.assertEqual(raised.exception.reason, "busy")
        finally:
            release.set()
            for thread in captured:
                thread.join(timeout=1)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertEqual(await worker.run(lambda: "next"), "next")

    async def test_failed_thread_start_releases_slot(self):
        worker = RequestWorker()
        with patch("llm.threading.Thread.start", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                await worker.run(lambda: "not started")
        self.assertEqual(await worker.run(lambda: "recovered"), "recovered")


class ClosedLoopTests(unittest.TestCase):
    def test_late_request_after_loop_close_does_not_raise_in_thread(self):
        loop = asyncio.new_event_loop()
        worker = RequestWorker()
        release = threading.Event()
        captured = []

        def blocked():
            captured.append(threading.current_thread())
            release.wait(2)
            return "late"

        try:
            with self.assertRaises(asyncio.TimeoutError):
                loop.run_until_complete(asyncio.wait_for(worker.run(blocked), 0.03))
            loop.close()
            with patch("threading.excepthook") as unexpected:
                release.set()
                captured[0].join(timeout=1)
                self.assertFalse(captured[0].is_alive())
                unexpected.assert_not_called()
        finally:
            release.set()
            for thread in captured:
                thread.join(timeout=1)
            if not loop.is_closed():
                loop.close()


if __name__ == "__main__":
    unittest.main()
