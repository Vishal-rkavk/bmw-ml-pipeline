"""Central metrics: every step pushes its run metrics to the Prometheus
Pushgateway (batch jobs push; services get scraped).

A Pushgateway push REPLACES the whole group (job + grouping labels), so
all metrics of one step must go out in a single call — separate pushes
to the same group would delete each other. `push` exists exactly for
that: scalars and labeled series are built into one registry and sent
once.

Pushes are best-effort: if the gateway is down, a warning is printed
once and the pipeline continues — monitoring must never be a hard
dependency of the data pipeline.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from pipeline import config


def push(job: str, values: dict[str, float] | None = None,
         series: dict[str, tuple[str, dict[str, float]]] | None = None,
         grouping: dict[str, str] | None = None) -> None:
    """Push a job's metrics to the gateway in one atomic request.

    values: {metric_name: value} -> plain gauges.
    series: {metric_name: (label_name, {label_value: value})} -> labeled
            gauges (e.g. one metric across train/eval or ok/error).
    grouping: Pushgateway grouping key; different keys are kept as
              separate series.

    A Pushgateway push replaces the group, hence one call per step: if a
    step sends its scalars and its labeled series in two pushes, the
    second push silently deletes the first one's metrics.
    """
    registry = CollectorRegistry()
    for name, value in (values or {}).items():
        Gauge(name, name, registry=registry).set(value)
    for name, (label, labeled) in (series or {}).items():
        gauge = Gauge(name, name, [label], registry=registry)
        for label_value, value in labeled.items():
            gauge.labels(**{label: label_value}).set(value)
    _push(job, registry, grouping)


def _push(job: str, registry: CollectorRegistry,
          grouping: dict[str, str] | None) -> None:
    """Deliver to the gateway; never raise into the pipeline."""
    if not config.METRICS_ENABLED:
        return
    global _warned
    try:
        push_to_gateway(config.PUSHGATEWAY_URL, job=job, registry=registry,
                        grouping_key=grouping or {})
        _warned = False
    except Exception as exc:  # noqa: BLE001 - monitoring is best-effort
        if not _warned:  # warn once per process, not on every push
            print(f"[warn] metrics push failed ({exc}); "
                  f"will retry silently until the gateway is reachable")
            _warned = True


_warned = False
