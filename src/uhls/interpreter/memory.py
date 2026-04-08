"""Array-backed memory for canonical µhLS interpreters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _is_sequence_like(value: Any) -> bool:
    """Return whether ``value`` should be treated as array contents.

    Args:
        value: Candidate initializer object.

    Returns:
        ``True`` for non-string sequences that can seed array storage.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


@dataclass
class ArraySlot:
    """Storage for one explicit array object."""

    data: list[int]
    element_type: Any | None = None


@dataclass
class ArrayMemory:
    """Non-aliasing array memory model."""

    arrays: dict[str, ArraySlot] = field(default_factory=dict)

    def bind(self, name: str, values: Sequence[int], element_type: Any | None = None) -> None:
        """Bind an array name to explicit initial contents.

        Args:
            name: Symbolic array name used by IR ``load`` and ``store`` ops.
            values: Initial element contents.
            element_type: Optional element type metadata for width normalization.
        """
        self.arrays[name] = ArraySlot(data=[int(value) for value in values], element_type=element_type)

    def allocate(
        self,
        name: str,
        size: int,
        *,
        element_type: Any | None = None,
        fill: int = 0,
    ) -> None:
        """Allocate a zero- or fill-initialized array object.

        Args:
            name: Symbolic array name.
            size: Number of elements to allocate.
            element_type: Optional element type metadata.
            fill: Initial value written to every allocated slot.
        """
        if size < 0:
            raise ValueError(f"array '{name}' size must be non-negative")
        self.arrays[name] = ArraySlot(data=[int(fill)] * size, element_type=element_type)

    def alias(self, name: str, target: str) -> None:
        """Bind ``name`` to the same storage slot as ``target``."""
        self.arrays[name] = self._slot(target)

    def initialize(self, arrays: Mapping[str, Any] | None) -> None:
        """Populate memory from a flexible mapping of array initializers.

        Args:
            arrays: Mapping from array names to either explicit contents,
                allocation sizes, or dictionaries describing ``data``/``size``
                plus optional element-type metadata.
        """
        if not arrays:
            return

        for name, spec in arrays.items():
            if isinstance(spec, Mapping):
                element_type = spec.get("element_type", spec.get("type"))
                if "data" in spec:
                    self.bind(name, spec["data"], element_type=element_type)
                    continue
                if "values" in spec:
                    self.bind(name, spec["values"], element_type=element_type)
                    continue
                if "size" in spec:
                    fill = int(spec.get("fill", 0))
                    self.allocate(name, int(spec["size"]), element_type=element_type, fill=fill)
                    continue
                if "alias" in spec:
                    self.alias(name, str(spec["alias"]))
                    continue
                raise ValueError(f"unsupported array initializer for '{name}': {spec!r}")

            if isinstance(spec, int):
                self.allocate(name, spec)
                continue

            if _is_sequence_like(spec):
                self.bind(name, spec)
                continue

            raise ValueError(f"unsupported array initializer for '{name}': {spec!r}")

    def has(self, name: str) -> bool:
        """Check whether an array object named ``name`` exists."""
        return name in self.arrays

    def element_type(self, name: str) -> Any | None:
        """Return the remembered element type metadata for ``name``."""
        return self._slot(name).element_type

    def load(self, name: str, index: int) -> int:
        """Read one array element.

        Args:
            name: Array symbol to read from.
            index: Element index within the array.
        """
        slot = self._slot(name)
        return slot.data[self._checked_index(name, index, len(slot.data))]

    def store(self, name: str, index: int, value: int) -> None:
        """Write one array element.

        Args:
            name: Array symbol to write to.
            index: Element index within the array.
            value: Integer value to store.
        """
        slot = self._slot(name)
        slot.data[self._checked_index(name, index, len(slot.data))] = int(value)

    def overwrite(self, name: str, values: Sequence[int]) -> None:
        """Replace one existing array object's contents in place."""
        slot = self._slot(name)
        slot.data[:] = [int(value) for value in values]

    def snapshot(self) -> dict[str, list[int]]:
        """Return a deep-enough copy of the current array contents."""
        return {name: slot.data[:] for name, slot in self.arrays.items()}

    def _slot(self, name: str) -> ArraySlot:
        """Fetch the storage record for ``name`` or raise a descriptive error."""
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"unknown array '{name}'") from exc

    @staticmethod
    def _checked_index(name: str, index: int, length: int) -> int:
        """Validate an array index and return it as an ``int``.

        Args:
            name: Array name used in the error message.
            index: Candidate element index.
            length: Array length used for bounds checking.
        """
        int_index = int(index)
        if int_index < 0 or int_index >= length:
            raise IndexError(f"array index out of bounds for '{name}': {int_index}")
        return int_index
