"""
Reusable execution stats: execution time, tokens consumed, energy (CodeCarbon).
Use for all execution paths (embeddings, clustering, labeling, KNN, workflow, evaluation).
"""
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")
logger = logging.getLogger("uvicorn.error")

# Sample power every 1s so short runs (e.g. clustering) get at least one measurement.
# Default 10–15s often yields 0 energy for runs under that length.
_MEASURE_POWER_SECS = 1


def _get(data: Any, key: str, default: Any = None) -> Any:
    """Get value from dict or object (getattr)."""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _read_energy_from_tracker(tracker: Any) -> tuple[float, float]:
    """Extract energy_consumed (kWh) and emissions (kg CO2eq) from CodeCarbon tracker."""
    energy_kwh = 0.0
    emissions_kg = 0.0
    try:
        tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        energy_kwh = float(
            _get(data, "energy_consumed") or _get(data, "total_energy_consumed_kwh") or 0.0
        )
        emissions_kg = float(_get(data, "emissions") or 0.0)
        if energy_kwh == 0.0 and data is not None:
            for k in ("cpu_energy", "gpu_energy", "ram_energy"):
                v = _get(data, k)
                if v is not None:
                    try:
                        energy_kwh += float(v)
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logger.warning("CodeCarbon: failed to read final_emissions_data: %s", e)
    return energy_kwh, emissions_kg


def run_with_stats(
    fn: Callable[[], T],
    token_handler: Optional[Any] = None,
) -> tuple[T, Dict[str, Any]]:
    """
    Run fn() and collect execution_time_seconds, tokens_consumed, energy_consumed_kwh, emissions_kg_co2eq.
    Pass token_handler when fn involves LLM calls. For embedding-only flows, routers can augment
    tokens_consumed from the result (e.g. when using OpenAI embeddings).
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        EmissionsTracker = None

    tracker = None
    if EmissionsTracker is not None:
        try:
            tracker = EmissionsTracker(
                measure_power_secs=_MEASURE_POWER_SECS,
                save_to_file=False,
                log_level="warning",
            )
            tracker.start()
        except Exception as e:
            logger.warning("CodeCarbon: EmissionsTracker failed to start: %s", e)
            tracker = None

    start = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed = time.perf_counter() - start

    energy_consumed_kwh = 0.0
    emissions_kg_co2 = 0.0
    if tracker is not None:
        energy_consumed_kwh, emissions_kg_co2 = _read_energy_from_tracker(tracker)

    tokens = 0
    if token_handler is not None and hasattr(token_handler, "tokens_consumed"):
        tokens = token_handler.tokens_consumed

    stats = {
        "execution_time_seconds": round(elapsed, 3),
        "tokens_consumed": tokens,
        "energy_consumed_kwh": round(energy_consumed_kwh, 6),
        "emissions_kg_co2eq": round(emissions_kg_co2, 6),
    }
    return result, stats
