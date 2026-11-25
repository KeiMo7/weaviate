# Quantization targets, evaluation and online switches

This package is the collection-scoped control plane for recall-target-driven
quantization.

## Guaranteed domain semantics

- `Target.Validate` validates `minRecall`, `tolerance` and profile aliases.
- `Resolve` is the only target-to-form mapping point. It returns codec, width
  and rescore depth with a persisted `PolicyVersion`.
- `FormObservation` represents the form read from written encoding data.
  `CompareDeclaredForm` explicitly reports declaration/effective drift.
- `Evaluate` computes deterministic recall against a full-precision searcher.
  Reports include sample count, `k`, strategy, seed and duration. Failed and
  canceled evaluations have nil `Recall`; they never report fake zero recall.
- `Manager` persists switch phases in `Store`, enforces one active switch per
  collection, aggregates shard/replica units, rolls back failed runs and resumes
  nonterminal records after restart without repeating completed units.
- `ReferenceBackend` is an executable semantic adapter: readers use one
  immutable form snapshot, cross-shard mixed-form barriers return retryable
  `ErrSwitchUnavailable`, and writes during a switch are encoded into both the
  serving and future forms.
- Prometheus instruments are in `metrics.go`; HTTP management DTOs/routes are
  in `handlers/`.

## Tests

```bash
go test -race ./wl/quantization/...
go test -tags integrationTest -race -run TestQuantizationLocalShardRecallIntegration \
  -count=1 ./adapters/repos/db
```

The db integration test exercises a real local shard, deterministic raw-vector
scanning, HNSW search, SQ activation and a write issued while compression is in
progress.

## Production adapter boundary

The state machine and semantic tests are storage-independent. A production
Weaviate adapter must implement `Backend` using a shadow vector index/build,
lease-drained atomic slot promotion and cleanup of old encoded artifacts. The
current HNSW `UpdateUserConfig` path can enable first-time compression but does
not itself rebuild from one codec to another or back to uncompressed. It now
returns `hnsw.ErrCompressionRebuildRequired` instead of silently accepting that
echo, pinned by
`adapters/repos/db/vector/hnsw/config_update_form_guard_test.go`. An adapter
must not report `PhaseCompleted` until its data-plane swap and observation probe
confirm the destination form.
