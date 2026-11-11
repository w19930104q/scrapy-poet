"""Merging of concurrent, identical in-flight provider calls.

This module implements the opt-in mechanism that allows
:class:`~scrapy_poet.page_input_providers.PageObjectInputProvider` subclasses
to share a single execution of their ``__call__`` method between concurrent
callers that ask for the same data.

Only *in-flight* work is merged: as soon as a call finishes (successfully or
not), its state is dropped, so nothing here acts as a result or error cache.
"""

import asyncio
from collections.abc import Awaitable, Callable, Hashable
from typing import Any


class _InflightCall:
    __slots__ = ("task", "waiters")

    def __init__(self, task: "asyncio.Task[Any]") -> None:
        self.task = task
        self.waiters = 0


class InflightCallMerger:
    """Merge concurrent calls with the same key into a single execution.

    The first caller for a key starts the work; concurrent callers with the
    same key await the same result instead of running the work again.

    * Cancelling one waiter does not affect the remaining ones.
    * If all waiters are cancelled, the underlying work is cancelled and its
      state is released.
    * Finished work (successful or failed) is forgotten immediately, so
      results and errors are never cached: a later call with the same key
      runs the work again.
    """

    def __init__(self) -> None:
        self._inflight: dict[Hashable, _InflightCall] = {}

    def __len__(self) -> int:
        return len(self._inflight)

    async def merge(self, key: Hashable, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Return the result of ``factory()``, joining an identical in-flight
        call if there is one."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No asyncio event loop in this thread (e.g. a non-asyncio
            # Twisted reactor): merging is not possible, run the work
            # directly instead.
            return await factory()
        entry = self._inflight.get(key)
        if entry is not None and entry.task.done():
            # Finished work is never joined, only in-flight work is merged.
            entry = None
        if entry is None:
            entry = _InflightCall(asyncio.ensure_future(factory()))
            self._inflight[key] = entry

            def _discard(
                task: "asyncio.Task[Any]",
                *,
                _key: Hashable = key,
                _entry: _InflightCall = entry,
            ) -> None:
                if self._inflight.get(_key) is _entry:
                    del self._inflight[_key]
                if not task.cancelled():
                    # Retrieve the exception, if any, so that it is not
                    # reported as "never retrieved" in the edge case where no
                    # waiter remained to observe it.
                    task.exception()

            entry.task.add_done_callback(_discard)
        entry.waiters += 1
        try:
            # shield() so that cancelling this waiter does not cancel the
            # shared work other waiters depend on.
            return await asyncio.shield(entry.task)
        finally:
            entry.waiters -= 1
            if entry.waiters == 0 and not entry.task.done():
                # All waiters were cancelled: release the in-flight work.
                entry.task.cancel()
                if self._inflight.get(key) is entry:
                    del self._inflight[key]

    def discard_all(self) -> None:
        """Cancel all in-flight work and release its state.

        Meant to be called when the crawl is stopping, so that no provider
        work or state outlives it.
        """
        entries = list(self._inflight.values())
        self._inflight.clear()
        for entry in entries:
            if not entry.task.done():
                entry.task.cancel()
