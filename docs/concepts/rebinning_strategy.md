# Rebinning strategy

`rebin_to(target_edges)` reprojects a histogram from its original grid onto a new set of edges,
preserving the total mass as closely as possible.

## Algorithm

For each non-empty source bin $b$ with index $i$ and mass $m_i$:

1. **Compute the bin centre** in data space:
$c_d = \frac{e_d[i_d] + e_d[i_d+1]}{2} \quad \text{for each dimension } d$

2. **Find the target bin** that contains $c$:
$j_d = \text{searchsorted}(\text{target\_edges}_d, c_d) - 1$
Clipped to $[0, n_\text{bins} - 1]$ so boundary centroids never fall out of range.

3. **Accumulate** mass into target bin $j$:
$Q(j) \mathrel{+}= m_i$

## Properties

- **Mass conservation**: $\sum Q = \sum P$ exactly (every source bin's mass lands somewhere in the target)
- **Many-to-one mapping**: multiple source bins can map to the same target bin (coarser grid); each source bin maps to exactly one target bin
- **Loss-free coarsening**: going from fine to coarse loses no mass; going from coarse to fine produces block-uniform distributions (centroids land in a single fine bin)
- **Returns a `dict`** `{tuple_index: float}` — not a grid object — so it can be used with any downstream code without coupling to a specific backend

## Limitations

The centroid-mapping approach is a heuristic: it does not account for the shape of the distribution within each bin. For grids with very different resolutions, consider fitting a new grid directly from raw data rather than rebinning.
