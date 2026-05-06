# Temporal decay

`TemporalHypergrid` applies exponential decay before each `update()` call:

$$H_t \leftarrow \lambda \cdot H_{t-1} + \Delta_t$$

where $\lambda \in (0, 1]$ is the `decay` factor and $\Delta_t$ is the new data batch.

This is equivalent to an exponential moving average over batches, giving more weight to recent data.
The effective half-life in batches is $\log(0.5) / \log(\lambda)$ — e.g. with $\lambda = 0.99$ and
batches of 1000 points, the half-life is ~69 batches (69 000 points).

Snapshots are saved every `snapshot_interval` points and compared with `evolution(method)` to track drift over time.
