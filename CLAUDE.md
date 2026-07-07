# origami_comms

Extending Origami's analytical GEMM cost model to support communication primitives,
enabling cost prediction for concurrent comm+gemm and producer-consumer overlap patterns.

## Project Goal

Model the cost of:
1. **Concurrent comm+gemm** — GEMM and collective running simultaneously on partitioned CUs
2. **Comm-producer → gemm-consumer** — collective completes tiles that feed a dependent GEMM
3. **Gemm-producer → comm-consumer** — GEMM tiles complete and are immediately communicated

Applications: codesign of RCCL and hipBLASLt so they can launch with mutual awareness.
Test vehicle: TritonBLAS/Triton-Distributed gemm+comm overlap examples.

## Wiki

This project maintains an LLM wiki under `wiki/` following the Karpathy LLM wiki pattern.

### Structure

```
wiki/
  index.md          # Content catalog — updated on every ingest
  log.md            # Append-only chronological record
  pages/            # LLM-owned markdown files (concepts, entities, analyses)
  sources/          # Immutable input documents (papers, notes, code snapshots)
```

### Operations

- **Ingest**: Drop a source into `wiki/sources/`, then update relevant pages and `index.md`
- **Query**: Answer questions by reading the index, synthesizing from pages, and filing valuable answers back
- **Lint**: Check for contradictions, orphan pages, missing cross-links, stale claims

### Conventions

- Pages use `[[page-name]]` wiki-links for cross-references
- Each page has a one-line summary at the top after the title
- Pages are factual and dense — no filler, no hedging
- Update `wiki/log.md` with a dated entry on every substantive change
- Keep `wiki/index.md` under 200 lines

## Key Upstream Repos

- **Origami/tritonBLAS**: `ROCm/rocm-libraries` → `shared/origami/`
- **hipBLASLt**: `ROCm/hipBLASLt` (consumes Origami for kernel selection)
- **RCCL**: `ROCm/rccl` (ROCm Communication Collectives Library)
- **Triton-Distributed**: `ByteDance-Seed/Triton-distributed` (gemm+comm overlap framework)

## Key Papers

- tritonBLAS (arXiv:2512.04226) — analytical GEMM cost model
- C3 (arXiv:2412.14335) — concurrent compute-communication on MI300X
- Fused Computation-Collectives (arXiv:2305.06942) — GPU-initiated comm during GEMM
- DMA Collectives (arXiv:2511.06605) — DMA-based communication offload
