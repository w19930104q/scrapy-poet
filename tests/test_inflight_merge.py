"""Tests for the opt-in merging of concurrent in-flight provider calls."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from typing import Annotated, Any

import attr
import pytest
import scrapy
from andi.typeutils import strip_annotated
from scrapy import Request, signals
from scrapy.utils.defer import deferred_f_from_coro_f, maybe_deferred_to_future
from scrapy.utils.reactor import is_asyncio_reactor_installed
from web_poet.annotated import AnnotatedInstance
from web_poet.serialization import register_serialization
from web_poet.utils import get_fq_class_name

from scrapy_poet import DummyResponse
from scrapy_poet.injection import (
    Injector,
    get_injector_for_testing,
    get_response_for_testing,
)
from scrapy_poet.page_input_providers import PageObjectInputProvider
from scrapy_poet.utils.testing import make_crawler

pytestmark = pytest.mark.skipif(
    not is_asyncio_reactor_installed(),
    reason="concurrent call merging requires the asyncio reactor",
)


@attr.s(auto_attribs=True)
class Token:
    value: int


@attr.s(auto_attribs=True)
class Tagged:
    value: int


class ControlledProvider(PageObjectInputProvider):
    """Async provider whose real executions can be counted, gated and made
    to fail on demand."""

    provided_classes = {Token}
    merge_concurrent_calls = True

    def __init__(self, injector: Injector) -> None:
        super().__init__(injector)
        self.calls = 0
        self.cancelled = 0
        self.started = asyncio.Event()
        self.cancelled_event = asyncio.Event()
        self.gate = asyncio.Event()
        self.failures = 0

    async def __call__(self, to_provide: set[Callable[..., Any]]) -> list[Any]:
        self.calls += 1
        call_no = self.calls
        self.started.set()
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            self.cancelled_event.set()
            raise
        if self.failures:
            self.failures -= 1
            raise ValueError(f"planned failure in call {call_no}")
        return [Token(call_no)]


class OptedOutProvider(ControlledProvider):
    merge_concurrent_calls = False


def token_callback(response: DummyResponse, token: Token) -> None:
    pass


def get_provider(
    injector: Injector, cls: type[ControlledProvider]
) -> ControlledProvider:
    return next(p for p in injector.providers if isinstance(p, cls))


def make_call(
    injector: Injector, meta: dict[str, Any] | None = None, url: str | None = None
) -> Coroutine[Any, Any, dict[Callable[..., Any], Any]]:
    """Return a coroutine building the provider instances for a request."""
    response = get_response_for_testing(token_callback)
    if url is None:
        assert response.request
        request = response.request
        if meta:
            request.meta.update(meta)
    else:
        request = Request(url, callback=token_callback, meta=meta or {})  # type: ignore[arg-type]
    plan = injector.build_plan(request)
    return injector.build_instances_from_providers(request, response, plan)


@deferred_f_from_coro_f
async def test_concurrent_calls_are_merged() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector))
    await asyncio.sleep(0)
    assert provider.calls == 1

    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    # A single real execution, whose result is shared by both callers.
    assert provider.calls == 1
    assert instances1[Token] is instances2[Token]
    assert instances1[Token].value == 1
    # The in-flight state is released once the call finishes.
    assert len(injector.inflight_call_merger) == 0
    # Injection stats keep per-call semantics.
    stats = injector.crawler.stats
    assert stats
    assert stats.get_value(f"poet/injector/{get_fq_class_name(Token)}") == 2


@deferred_f_from_coro_f
async def test_opted_out_provider_is_not_merged() -> None:
    injector = get_injector_for_testing({OptedOutProvider: 1})
    provider = get_provider(injector, OptedOutProvider)

    task1 = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector))
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    assert provider.calls == 2
    assert instances1[Token] is not instances2[Token]
    assert {instances1[Token].value, instances2[Token].value} == {1, 2}


@deferred_f_from_coro_f
async def test_failed_calls_are_not_cached() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)
    provider.failures = 1

    task1 = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector))
    await asyncio.sleep(0)
    provider.gate.set()
    results = await asyncio.gather(task1, task2, return_exceptions=True)

    # Both callers see the failure of the single real execution.
    assert provider.calls == 1
    assert all(isinstance(result, ValueError) for result in results)
    # The failure is not cached: the in-flight state is released and a later
    # call invokes the provider again.
    assert len(injector.inflight_call_merger) == 0
    instances = await make_call(injector)
    assert provider.calls == 2
    assert instances[Token].value == 2


@deferred_f_from_coro_f
async def test_single_waiter_cancellation_does_not_affect_others() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector))
    await asyncio.sleep(0)

    task1.cancel()
    results1 = await asyncio.gather(task1, return_exceptions=True)
    assert isinstance(results1[0], asyncio.CancelledError)
    # The shared work is still running for the remaining waiter.
    assert provider.cancelled == 0
    assert len(injector.inflight_call_merger) == 1

    provider.gate.set()
    instances2 = await task2
    assert provider.calls == 1
    assert instances2[Token].value == 1
    assert len(injector.inflight_call_merger) == 0


@deferred_f_from_coro_f
async def test_cancelling_all_waiters_releases_inflight_state() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector))
    await asyncio.sleep(0)

    task1.cancel()
    task2.cancel()
    results = await asyncio.gather(task1, task2, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)

    # The underlying provider work is cancelled and its state released.
    await asyncio.wait_for(provider.cancelled_event.wait(), timeout=1)
    assert provider.cancelled == 1
    assert len(injector.inflight_call_merger) == 0

    # A later call invokes the provider again.
    provider.gate.set()
    instances = await make_call(injector)
    assert provider.calls == 2
    assert instances[Token].value == 2


@deferred_f_from_coro_f
async def test_spider_closed_releases_inflight_state() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task = asyncio.ensure_future(make_call(injector))
    await provider.started.wait()
    assert len(injector.inflight_call_merger) == 1

    assert injector.crawler.spider
    injector.crawler.signals.send_catch_log(
        signal=signals.spider_closed, spider=injector.crawler.spider
    )
    assert len(injector.inflight_call_merger) == 0
    await asyncio.wait_for(provider.cancelled_event.wait(), timeout=1)
    assert provider.cancelled == 1
    results = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)


@deferred_f_from_coro_f
async def test_page_params_key_order_is_irrelevant() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(
        make_call(injector, meta={"page_params": {"a": 1, "b": 2}})
    )
    await provider.started.wait()
    task2 = asyncio.ensure_future(
        make_call(injector, meta={"page_params": {"b": 2, "a": 1}})
    )
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    assert provider.calls == 1
    assert instances1[Token] is instances2[Token]


@deferred_f_from_coro_f
async def test_different_page_params_are_not_merged() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector, meta={"page_params": {"a": 1}}))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector, meta={"page_params": {"a": 2}}))
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    assert provider.calls == 2
    assert instances1[Token] is not instances2[Token]


@deferred_f_from_coro_f
async def test_unencodable_page_params_are_not_merged() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(
        make_call(injector, meta={"page_params": {"x": object()}})
    )
    await provider.started.wait()
    task2 = asyncio.ensure_future(
        make_call(injector, meta={"page_params": {"x": object()}})
    )
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    # No merging, but the calls work as usual.
    assert provider.calls == 2
    assert instances1[Token] is not instances2[Token]


@deferred_f_from_coro_f
async def test_different_urls_are_not_merged() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector, url="http://example.com/1"))
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_call(injector, url="http://example.com/2"))
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    assert provider.calls == 2
    assert instances1[Token] is not instances2[Token]


class AnnotatedProvider(ControlledProvider):
    provided_classes = {Token, Tagged}

    def is_provided(self, type_: Any) -> bool:
        return super().is_provided(strip_annotated(type_))

    async def __call__(self, to_provide: set[Callable[..., Any]]) -> list[Any]:
        self.calls += 1
        self.started.set()
        await self.gate.wait()
        result: list[Any] = []
        for cls in to_provide:
            metadata = getattr(cls, "__metadata__", None)
            if metadata:
                result.append(AnnotatedInstance(Tagged(self.calls), metadata))
            else:
                result.append(Token(self.calls))
        return result


def annotated_callback(
    response: DummyResponse, token: Token, tagged: Annotated[Tagged, "m"]
) -> None:
    pass


@deferred_f_from_coro_f
async def test_annotated_types_are_merged() -> None:
    injector = get_injector_for_testing({AnnotatedProvider: 1})
    provider = get_provider(injector, AnnotatedProvider)

    def make_annotated_call() -> Coroutine[Any, Any, dict[Callable[..., Any], Any]]:
        response = get_response_for_testing(annotated_callback)
        assert response.request
        plan = injector.build_plan(response.request)
        return injector.build_instances_from_providers(response.request, response, plan)

    task1 = asyncio.ensure_future(make_annotated_call())
    await provider.started.wait()
    task2 = asyncio.ensure_future(make_annotated_call())
    await asyncio.sleep(0)
    provider.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    assert provider.calls == 1
    annotated_cls = Annotated[Tagged, "m"]
    assert instances1.keys() == instances2.keys() == {Token, annotated_cls}
    assert instances1[Token] is instances2[Token]
    assert instances1[annotated_cls] is instances2[annotated_cls]
    assert instances1[annotated_cls].value == 1


@deferred_f_from_coro_f
async def test_calls_are_not_merged_across_crawlers() -> None:
    injector1 = get_injector_for_testing({ControlledProvider: 1})
    injector2 = get_injector_for_testing({ControlledProvider: 1})
    provider1 = get_provider(injector1, ControlledProvider)
    provider2 = get_provider(injector2, ControlledProvider)

    task1 = asyncio.ensure_future(make_call(injector1))
    await provider1.started.wait()
    task2 = asyncio.ensure_future(make_call(injector2))
    await provider2.started.wait()
    provider1.gate.set()
    provider2.gate.set()
    instances1, instances2 = await asyncio.gather(task1, task2)

    # Each crawler ran its own real call.
    assert provider1.calls == 1
    assert provider2.calls == 1
    assert instances1[Token] is not instances2[Token]


def test_merge_key() -> None:
    injector = get_injector_for_testing({ControlledProvider: 1})
    provider = get_provider(injector, ControlledProvider)
    key = injector._provider_merge_key

    # The type set (including Annotated metadata) and the page_params dict
    # are order-insensitive.
    request1 = Request("http://example.com", meta={"page_params": {"a": 1, "b": 2}})
    request2 = Request("http://example.com", meta={"page_params": {"b": 2, "a": 1}})
    assert key(provider, request1, {Token, Annotated[Tagged, "m"]}) == key(
        provider, request2, {Annotated[Tagged, "m"], Token}
    )

    # Different page_params or types produce different keys.
    request3 = Request("http://example.com", meta={"page_params": {"a": 1}})
    assert key(provider, request1, {Token}) != key(provider, request3, {Token})
    assert key(provider, request1, {Token}) != key(provider, request1, {Token, Tagged})
    # Different Annotated metadata produces different keys.
    assert key(provider, request1, {Annotated[Tagged, "m"]}) != key(
        provider, request1, {Annotated[Tagged, "other"]}
    )

    # Unencodable page_params cannot be stably identified: no merge key.
    request4 = Request("http://example.com", meta={"page_params": {"x": object()}})
    assert key(provider, request4, {Token}) is None
    # Unhashable Annotated metadata cannot be stably identified either.
    assert key(provider, request1, [Annotated[Token, {}]]) is None  # type: ignore[arg-type]


class CrawlProvider(PageObjectInputProvider):
    """Opted-in provider used by the end-to-end crawl tests."""

    provided_classes = {Token}
    merge_concurrent_calls = True
    name = "token"
    calls = 0

    async def __call__(self, to_provide: set[Callable[..., Any]]) -> list[Any]:
        type(self).calls += 1
        await asyncio.sleep(0.05)
        return [Token(type(self).calls)]


def _serialize_token(obj: Token) -> dict[str, bytes]:
    return {"value": str(obj.value).encode()}


def _deserialize_token(cls: type[Token], data: dict[str, bytes]) -> Token:
    return cls(int(data["value"].decode()))


register_serialization(_serialize_token, _deserialize_token)


class MergeSpider(scrapy.Spider):
    """Issues several identical requests whose callbacks only need Token."""

    name = "merge_spider"
    custom_settings = {
        "SCRAPY_POET_PROVIDERS": {CrawlProvider: 1},
    }
    request_count = 5

    async def start(self) -> AsyncIterator[Any]:
        for _ in range(self.request_count):
            yield scrapy.Request(
                "http://example.com/",
                callback=self.parse,  # type: ignore[arg-type]
                dont_filter=True,
            )

    def parse(self, response: DummyResponse, token: Token) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        yield {"token": token.value}


@deferred_f_from_coro_f
async def test_crawl_merges_concurrent_calls(settings: dict[str, Any]) -> None:
    CrawlProvider.calls = 0
    crawler = make_crawler(MergeSpider, settings)
    await maybe_deferred_to_future(crawler.crawl())

    # 5 identical concurrent requests triggered a single real provider call,
    # and every callback got the shared result.
    assert CrawlProvider.calls == 1
    assert crawler.spider
    assert crawler.spider.collected_items == [{"token": 1}] * 5  # type: ignore[attr-defined]
    # Injection stats keep per-request semantics.
    assert crawler.stats
    stats = crawler.stats.get_stats()
    assert stats[f"poet/injector/{get_fq_class_name(Token)}"] == 5


@deferred_f_from_coro_f
async def test_crawl_merging_is_compatible_with_persistent_cache(
    settings: dict[str, Any], tmp_path: Any
) -> None:
    CrawlProvider.calls = 0
    settings["SCRAPY_POET_CACHE"] = str(tmp_path / "cache")

    crawler = make_crawler(MergeSpider, settings)
    await maybe_deferred_to_future(crawler.crawl())
    assert CrawlProvider.calls == 1
    assert crawler.spider
    assert crawler.spider.collected_items == [{"token": 1}] * 5  # type: ignore[attr-defined]

    # A second crawl is served from the persistent cache without any
    # additional provider call.
    crawler2 = make_crawler(MergeSpider, settings)
    await maybe_deferred_to_future(crawler2.crawl())
    assert CrawlProvider.calls == 1
    assert crawler2.spider
    assert crawler2.spider.collected_items == [{"token": 1}] * 5  # type: ignore[attr-defined]
    assert crawler2.stats
    assert crawler2.stats.get_value("poet/cache/hit") == 5
