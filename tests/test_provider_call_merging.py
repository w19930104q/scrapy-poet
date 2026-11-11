"""Tests for the optional merging of concurrent provider calls, enabled
through the ``merge_concurrent_calls`` provider class attribute."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated, Any

import attr
import pytest
from andi.typeutils import strip_annotated
from scrapy import Request, Spider, signals
from scrapy.utils.defer import deferred_f_from_coro_f, maybe_deferred_to_future
from web_poet.annotated import AnnotatedInstance
from web_poet.serialization import SerializedLeafData, register_serialization

from scrapy_poet import DummyResponse, InjectionMiddleware, PageObjectInputProvider
from scrapy_poet.injection import (
    Injector,
    get_injector_for_testing,
    get_response_for_testing,
)
from scrapy_poet.utils.testing import make_crawler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# The merging of concurrent provider calls is based on asyncio, so it only
# works when the AsyncioSelectorReactor is used.
pytestmark = pytest.mark.skipif(
    os.environ.get("REACTOR") != "asyncio",
    reason="Using asyncio will only work if the AsyncioSelectorReactor is used.",
)


@attr.define
class Token:
    value: str


def _serialize_token(o: Token) -> SerializedLeafData:
    return {"txt": o.value.encode()}


def _deserialize_token(cls: type[Token], data: SerializedLeafData) -> Token:
    return cls(data["txt"].decode())


register_serialization(_serialize_token, _deserialize_token)


class ControlledProvider(PageObjectInputProvider):
    """Provides :class:`Token`, blocking until released, and records what
    happens to it so that tests can assert the real number of calls."""

    provided_classes = {Token}
    name = "token"
    merge_concurrent_calls = True

    def __init__(self, injector: Injector) -> None:
        super().__init__(injector)
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.exception: Exception | None = None

    def is_provided(self, type_: Any) -> bool:
        return super().is_provided(strip_annotated(type_))

    async def __call__(self, to_provide: set[Any], request: Request) -> list[Any]:
        self.calls += 1
        self.started.set()
        call_number = self.calls
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.exception is not None:
            raise self.exception
        result = []
        for cls in to_provide:
            obj: Any = Token(f"{request.url}#{call_number}")
            if metadata := getattr(cls, "__metadata__", None):
                obj = AnnotatedInstance(obj, metadata)
            result.append(obj)
        return result


class UnmergedProvider(ControlledProvider):
    """Same as :class:`ControlledProvider` but without call merging."""

    name = "unmerged_token"
    merge_concurrent_calls = False


def make_injector(
    provider_cls: type[ControlledProvider] = ControlledProvider,
    settings: dict[str, Any] | None = None,
) -> tuple[Injector, ControlledProvider]:
    injector = get_injector_for_testing({provider_cls: 1}, settings)
    provider = next(p for p in injector.providers if isinstance(p, provider_cls))
    return injector, provider


def token_callback(response: DummyResponse, token: Token) -> None:
    pass


def get_response(
    meta: dict[str, Any] | None = None, callback: Any = token_callback
) -> Any:
    response = get_response_for_testing(callback, meta=meta)
    assert response.request
    return response


async def run_and_release(
    injector: Injector, provider: ControlledProvider, *responses: Any
) -> list[Any]:
    """Run build_callback_dependencies() for the given responses concurrently,
    releasing the provider once it has been called."""
    tasks = [
        asyncio.ensure_future(injector.build_callback_dependencies(r.request, r))
        for r in responses
    ]
    await provider.started.wait()
    provider.release.set()
    results: list[Any] = await asyncio.gather(*tasks)
    return results


@deferred_f_from_coro_f
async def test_merge_concurrent_calls() -> None:
    """Concurrent identical calls are merged into a single provider call."""
    injector, provider = make_injector()
    # The page_params dict key order does not matter.
    response1 = get_response(meta={"page_params": {"a": 1, "b": 2}})
    response2 = get_response(meta={"page_params": {"b": 2, "a": 1}})

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    # The provider was really called only once and both callers share the
    # same result instances.
    assert provider.calls == 1
    assert kwargs1["token"] == kwargs2["token"] == Token("http://example.com/#1")
    assert kwargs1["token"] is kwargs2["token"]
    # The in-flight state is released once the call is done.
    assert injector._in_flight_provider_calls == {}

    stats = injector.crawler.stats.get_stats()
    # The injected dependencies stats keep counting every logical call.
    assert stats["poet/injector/tests.test_provider_call_merging.Token"] == 2
    assert stats["poet/providers/merged_calls"] == 1


@deferred_f_from_coro_f
async def test_no_merge_when_not_enabled() -> None:
    """Providers that do not opt in keep the previous behavior."""
    injector, provider = make_injector(UnmergedProvider)
    response1 = get_response()
    response2 = get_response()

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    assert provider.calls == 2
    assert kwargs1["token"] == Token("http://example.com/#1")
    assert kwargs2["token"] == Token("http://example.com/#2")
    assert kwargs1["token"] is not kwargs2["token"]
    assert "poet/providers/merged_calls" not in injector.crawler.stats.get_stats()


@deferred_f_from_coro_f
async def test_no_result_caching() -> None:
    """Only in-flight calls are merged: sequential identical calls are
    executed again."""
    injector, provider = make_injector()
    response1 = get_response()
    response2 = get_response()

    (kwargs1,) = await run_and_release(injector, provider, response1)
    provider.started.clear()
    provider.release.clear()
    (kwargs2,) = await run_and_release(injector, provider, response2)

    assert provider.calls == 2
    assert kwargs1["token"] == Token("http://example.com/#1")
    assert kwargs2["token"] == Token("http://example.com/#2")


@deferred_f_from_coro_f
async def test_failure_is_not_cached() -> None:
    """All waiters of a failed merged call get the exception, and calls made
    after the failure execute the provider again."""
    injector, provider = make_injector()
    provider.exception = ValueError("boom")
    response1 = get_response()
    response2 = get_response()

    task1 = asyncio.ensure_future(
        injector.build_callback_dependencies(response1.request, response1)
    )
    task2 = asyncio.ensure_future(
        injector.build_callback_dependencies(response2.request, response2)
    )
    await provider.started.wait()
    provider.release.set()
    results = await asyncio.gather(task1, task2, return_exceptions=True)

    assert all(isinstance(result, ValueError) for result in results)
    assert provider.calls == 1
    # The in-flight state is released after the failure.
    assert injector._in_flight_provider_calls == {}

    # A later call retries the provider and succeeds.
    provider.exception = None
    provider.started.clear()
    provider.release.clear()
    (kwargs,) = await run_and_release(injector, provider, response1)
    assert provider.calls == 2
    assert kwargs["token"] == Token("http://example.com/#2")


@deferred_f_from_coro_f
async def test_single_waiter_cancellation() -> None:
    """Cancelling one waiter does not affect the remaining ones."""
    injector, provider = make_injector()
    response1 = get_response()
    response2 = get_response()

    task1 = asyncio.ensure_future(
        injector.build_callback_dependencies(response1.request, response1)
    )
    task2 = asyncio.ensure_future(
        injector.build_callback_dependencies(response2.request, response2)
    )
    await provider.started.wait()
    task1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task1

    # The other waiter still gets the result of the single real call.
    provider.release.set()
    kwargs2 = await task2
    assert provider.calls == 1
    assert kwargs2["token"] == Token("http://example.com/#1")
    assert injector._in_flight_provider_calls == {}


@deferred_f_from_coro_f
async def test_all_waiters_cancelled() -> None:
    """When all waiters are cancelled, the in-flight call is cancelled and
    its state is released, so later calls execute the provider again."""
    injector, provider = make_injector()
    response1 = get_response()
    response2 = get_response()

    task1 = asyncio.ensure_future(
        injector.build_callback_dependencies(response1.request, response1)
    )
    task2 = asyncio.ensure_future(
        injector.build_callback_dependencies(response2.request, response2)
    )
    await provider.started.wait()
    task1.cancel()
    task2.cancel()
    await asyncio.gather(task1, task2, return_exceptions=True)

    # The in-flight state is released and the underlying call is cancelled.
    assert injector._in_flight_provider_calls == {}
    await asyncio.wait_for(provider.cancelled.wait(), 1)

    # A later call executes the provider again.
    provider.started.clear()
    provider.release.clear()
    (kwargs,) = await run_and_release(injector, provider, response1)
    assert provider.calls == 2
    assert kwargs["token"] == Token("http://example.com/#2")


@deferred_f_from_coro_f
async def test_spider_closed_releases_in_flight_calls() -> None:
    """Stopping the crawl releases the in-flight state."""
    injector, provider = make_injector()
    response = get_response()

    task = asyncio.ensure_future(
        injector.build_callback_dependencies(response.request, response)
    )
    await provider.started.wait()
    injector.crawler.signals.send_catch_log(
        signals.spider_closed, spider=injector.crawler.spider
    )

    assert injector._in_flight_provider_calls == {}
    await asyncio.wait_for(provider.cancelled.wait(), 1)
    with pytest.raises(asyncio.CancelledError):
        await task


@deferred_f_from_coro_f
async def test_page_params_isolation() -> None:
    """Calls with different page_params are not merged."""
    injector, provider = make_injector()
    response1 = get_response(meta={"page_params": {"a": 1}})
    response2 = get_response(meta={"page_params": {"a": 2}})

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    assert provider.calls == 2
    assert kwargs1["token"] is not kwargs2["token"]


@deferred_f_from_coro_f
async def test_unserializable_page_params_are_not_merged() -> None:
    """Calls whose page_params cannot be stably encoded are not merged."""
    injector, provider = make_injector()
    response1 = get_response(meta={"page_params": {"a": object()}})
    response2 = get_response(meta={"page_params": {"a": object()}})

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    assert provider.calls == 2
    assert kwargs1["token"] is not kwargs2["token"]


@deferred_f_from_coro_f
async def test_provided_types_are_part_of_the_key() -> None:
    """Calls for different types, including different Annotated metadata,
    are not merged, while identical Annotated dependencies are."""

    def callback_a(response: DummyResponse, token: Annotated[Token, "a"]) -> None:
        pass

    def callback_b(response: DummyResponse, token: Annotated[Token, "b"]) -> None:
        pass

    injector, provider = make_injector()
    response1 = get_response(callback=callback_a)
    response2 = get_response(callback=callback_b)

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    # Different Annotated metadata: not merged.
    assert provider.calls == 2
    assert kwargs1["token"] is not kwargs2["token"]

    provider.started.clear()
    provider.release.clear()
    response3 = get_response(callback=callback_a)
    response4 = get_response(callback=callback_a)

    kwargs3, kwargs4 = await run_and_release(injector, provider, response3, response4)

    # Identical Annotated dependencies: merged.
    assert provider.calls == 3
    assert kwargs3["token"] is kwargs4["token"]


@deferred_f_from_coro_f
async def test_type_order_does_not_matter() -> None:
    """The order of the types to provide does not matter."""

    def callback1(response: DummyResponse, a: Token, b: Annotated[Token, "x"]) -> None:
        pass

    def callback2(response: DummyResponse, b: Annotated[Token, "x"], a: Token) -> None:
        pass

    injector, provider = make_injector()
    response1 = get_response(callback=callback1)
    response2 = get_response(callback=callback2)

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    assert provider.calls == 1
    assert kwargs1["a"] is kwargs2["a"]
    assert kwargs1["b"] is kwargs2["b"]


@deferred_f_from_coro_f
async def test_cross_crawler_isolation() -> None:
    """Calls from different crawlers are never merged."""
    injector1, provider1 = make_injector()
    injector2, provider2 = make_injector()
    response1 = get_response()
    response2 = get_response()

    task1 = asyncio.ensure_future(
        injector1.build_callback_dependencies(response1.request, response1)
    )
    task2 = asyncio.ensure_future(
        injector2.build_callback_dependencies(response2.request, response2)
    )
    await provider1.started.wait()
    await provider2.started.wait()
    provider1.release.set()
    provider2.release.set()
    kwargs1, kwargs2 = await asyncio.gather(task1, task2)

    # Each crawler executed its own provider call.
    assert provider1.calls == 1
    assert provider2.calls == 1
    assert kwargs1["token"] is not kwargs2["token"]


@deferred_f_from_coro_f
async def test_persistent_cache_compatibility(tmp_path: Path) -> None:
    """Call merging does not interfere with the persistent cache."""
    settings = {"SCRAPY_POET_CACHE": str(tmp_path / "cache")}
    injector, provider = make_injector(settings=settings)
    response1 = get_response()
    response2 = get_response()

    kwargs1, kwargs2 = await run_and_release(injector, provider, response1, response2)

    assert provider.calls == 1
    assert kwargs1["token"] == kwargs2["token"] == Token("http://example.com/#1")
    stats = injector.crawler.stats.get_stats()
    assert stats["poet/cache/miss"] == 2

    # A new crawl over the same cache does not call the provider at all.
    injector2, provider2 = make_injector(settings=settings)
    response3 = get_response()
    kwargs3 = await injector2.build_callback_dependencies(response3.request, response3)
    assert provider2.calls == 0
    assert kwargs3["token"] == Token("http://example.com/#1")
    stats2 = injector2.crawler.stats.get_stats()
    assert stats2["poet/cache/hit"] == 1


class MergeSpider(Spider):
    """Makes 2 identical requests whose callbacks need a Token."""

    name = "merge_spider"
    custom_settings = {
        "SCRAPY_POET_PROVIDERS": {ControlledProvider: 1},
    }

    async def start(self) -> AsyncIterator[Any]:
        for _ in range(2):
            yield Request("http://example.com/", callback=self.parse, dont_filter=True)  # type: ignore[arg-type]

    def parse(self, response: DummyResponse, token: Token) -> Any:  # type: ignore[override]
        yield {"token": token.value}


def _get_injector(crawler: Any) -> Injector:
    assert crawler.engine
    for middleware in crawler.engine.downloader.middleware.middlewares:
        if isinstance(middleware, InjectionMiddleware):
            return middleware.injector
    raise RuntimeError("InjectionMiddleware not found")


@deferred_f_from_coro_f
async def test_crawl(settings: dict[str, Any]) -> None:
    """End-to-end test: a crawl with 2 identical concurrent requests only
    executes the provider once, and both items are collected."""
    crawler = make_crawler(MergeSpider, settings)
    crawl_deferred = crawler.crawl()
    provider: ControlledProvider = next(
        p for p in _get_injector(crawler).providers if isinstance(p, ControlledProvider)
    )
    await provider.started.wait()
    await asyncio.sleep(0.1)  # let the 2nd request join the in-flight call
    provider.release.set()
    await maybe_deferred_to_future(crawl_deferred)

    assert provider.calls == 1
    assert crawler.spider.collected_items == [  # type: ignore[union-attr]
        {"token": "http://example.com/#1"},
        {"token": "http://example.com/#1"},
    ]
