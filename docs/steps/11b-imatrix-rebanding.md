# Step 11b — Imatrix re-banding

← [10 Build imatrix](./10-build-imatrix.md) · [Index](./README.md) · Next: [11 Reference logits](./11-cache-reference-logits.md) →

## Goal

Re-cut each role's depth groups on **measured imatrix importance** instead of
fixed layer thirds. Step 04 groups tensors as `role@depth` using geometric
early/middle/late bands; this pass moves the boundaries onto importance cliffs
while keeping the same band count per role (3 by default).

## Command

```bash
odg reband --model functiongemma:latest
```

`odg run` executes this step automatically after step 10 (imatrix). Use
`--bands-per-role N` to change the number of bands and `--grouping per-tensor`
to skip banding.

## How it works

- Reads `tensor_catalog.json` (from catalog / weight features / activation
  features) and `imatrix_proxy.json` (per-tensor importance written by every
  imatrix path).
- Per role, orders layers and runs an exact 1-D Fisher-Jenks segmentation to
  minimize within-band importance variance.
- A role's cuts are adopted only when they reduce the sum of squares by at
  least 10%; otherwise the existing groups are kept.
- Group membership, group aggregates, depth labels, and `catalog_sha256` are
  rewritten. Tensors without a layer index (embeddings and other globals) keep
  their existing groups.

## Outputs

```text
steps/11b_reband/
  tensor_catalog.json    # rebanded catalog consumed by later steps
  reband_report.json     # boundaries, roles rebanded/skipped, score source
  output.json            # summary payload
  status.json
  log.txt
```

## Done when

- [x] Rebanded catalog is written and downstream steps use it
- [x] Boundaries move only when the importance reduction clears 10%
- [x] Global tensors and roles without layer structure keep their groups
- [x] Real imatrix statistics are preferred; proxy scores are an explicit fallback
