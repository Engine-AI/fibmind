"""Fibonacci-based layer policy for FibMind."""

from __future__ import annotations

from dataclasses import dataclass


def fibonacci_numbers(max_value: int, include_duplicate_one: bool = False) -> list[int]:
    """Return Fibonacci numbers less than or equal to ``max_value``.

    The default sequence starts at 1, 2, 3... to avoid the duplicated 1 that
    complicates capacity/index semantics. Set ``include_duplicate_one`` when
    the canonical 1, 1, 2... sequence is explicitly needed.
    """
    if max_value < 1:
        return []

    seed = [1, 1] if include_duplicate_one else [1, 2]
    values = [value for value in seed if value <= max_value]
    if len(values) < 2:
        return values
    while True:
        next_value = values[-1] + values[-2]
        if next_value > max_value:
            break
        values.append(next_value)
    return values


def fib_capacities(layers: tuple[str, ...], start: int = 21) -> dict[str, int]:
    """Assign consecutive Fibonacci capacities to memory layers."""
    if start < 1:
        raise ValueError("start must be positive")
    if not layers:
        return {}

    capacities: list[int] = []
    a, b = 1, 2
    while len(capacities) < len(layers):
        if a >= start:
            capacities.append(a)
        a, b = b, a + b
    return dict(zip(layers, capacities, strict=True))


@dataclass(frozen=True, slots=True)
class FibonacciLayerPolicy:
    """Capacity limits and promotion order for memory layers."""

    capacities: dict[str, int]
    order: tuple[str, ...]

    def capacity_for(self, layer: str) -> int:
        if layer not in self.capacities:
            raise KeyError(f"Unknown layer: {layer}")
        return self.capacities[layer]

    def next_layer(self, layer: str) -> str | None:
        try:
            index = self.order.index(layer)
        except ValueError as exc:
            raise KeyError(f"Unknown layer: {layer}") from exc
        if index + 1 >= len(self.order):
            return None
        return self.order[index + 1]


DEFAULT_LAYER_POLICY = FibonacciLayerPolicy(
    order=("raw", "compressed", "summary", "long_term"),
    capacities=fib_capacities(("raw", "compressed", "summary", "long_term"), start=21),
)
