from sat_net.datablock import DataBlock


class DictBuffer:
    """Models an order-less buffer."""

    def __init__(self, capacity: float):
        self._size = 0.0
        self.capacity = capacity
        self._data: dict[int, DataBlock] = {}

    def add(self, data_block: DataBlock)->bool:
        """Add a DataBlock to the buffer."""
        if self.get_remaining_capacity() < data_block.size:
            return False
        self._data[data_block.id] = data_block
        self._size += data_block.size
        return True

    def remove(self, block_id: int) -> bool:
        """Remove DataBlock from buffer."""
        if block_id not in self._data:
            return False

        data_block = self._data.pop(block_id)
        self._size -= data_block.size
        return True

    def get_remaining_capacity(self) -> float:
        """Get the remaining capacity of the buffer."""
        return self.capacity - self._size

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._data) == 0

    def get_data_size(self) -> float:
        """Return the size of the buffer."""
        return self._size

    def get_load_factor(self) -> float:
        """Return the load of the buffer."""
        return self._size / self.capacity

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data.values())

    def __getitem__(self, index: int) -> DataBlock:
        return list(self._data.values())[index]

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._data
