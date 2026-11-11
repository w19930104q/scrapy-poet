.. _stats:

=====
Stats
=====

scrapy-poet tracks :external+web-poet:ref:`web-poet stats <stats>` as part of
:ref:`Scrapy stats <topics-stats>`, prefixed with ``poet/stats/``.

Injector stats
==============

The injector produces some stats also. These are:

* cache stats have the ``poet/cache`` prefix.
* injected dependencies stats have the ``poet/injector`` prefix.
* provider stats have the ``poet/providers`` prefix. Currently there is only
  ``poet/providers/merged_calls``, which counts how many provider calls were
  merged into an identical in-flight one by providers with
  :ref:`call merging <provider-call-merging>` enabled.
