"""Central metrics for all pipeline steps.

Every step pushes its run metrics to the Prometheus Pushgateway at the
end of a run (batch jobs push; long-lived services get scraped — that
is why a gateway fits a pipeline better than an exporter endpoint).

All pushes are best-effort: if the gateway is down, the step prints a
warning and continues. Monitoring must never be a hard dependency of
the data pipeline itself.

Two helpers:
- push_metrics: one or more plain gauge values, e.g. a duration.
- push_series: one metric split across a label, e.g. document counts
  per status (Prometheus convention: dashboards aggregate by label).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from pipeline import config


def push_metrics(job: str, values: dict[str, float],
                 grouping: dict[str, str] | None = None) -> None:
    """Push plain gauge values: {metric_name: value}.

    grouping becomes the Pushgateway grouping key: pushes with different
    keys are kept as separate series (training uses this to accumulate
    a per-step loss curve).
    """
    registry = CollectorRegistry()
    for name, value in values.items():
        Gauge(name, name, registry=registry).set(value)
    _push(job, registry, grouping)


def push_series(job: str, name: str, label: str,
                values: dict[str, float],
                grouping: dict[str, str] | None = None) -> None:
    """Push one metric across label values: {label_value: value}."""
    registry = CollectorRegistry()
    gauge = Gauge(name, name, [label], registry=registry)
    for label_value, value in values.items():
        gauge.labels(**{label: label_value}).set(value)
    _push(job, registry, grouping)


def _push(job: str, registry: CollectorRegistry,
          grouping: dict[str, str] | None) -> None:
    """Deliver to the gateway; never raise into the pipeline."""
    try:
        push_to_gateway(config.PUSHGATEWAY_URL, job=job, registry=registry,
                        grouping_key=grouping or {})
    except Exception as exc:  # noqa: BLE001 - monitoring is best-effort
        print(f"[warn] metrics push failed ({exc}); continuing without it")
