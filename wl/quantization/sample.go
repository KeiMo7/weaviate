//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| | \ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_|  \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package quantization

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"sort"
)

const (
	// StrategySeededReservoir is a deterministic Algorithm-R reservoir using a
	// SplitMix64 stream. Equal seed + equal input order yields equal samples.
	StrategySeededReservoir = "seeded-reservoir"
	// StrategyStableHash selects the lowest FNV-1a(seed,id) hashes. It is
	// independent of input order and is the default for managed evaluations.
	StrategyStableHash = "stable-hash"
)

// DefaultLimits bounds expensive evaluation/switch work. Zero Limits is not
// usable; call DefaultLimits or populate every field.
type Limits struct {
	MaxSampleSize        int `json:"maxSampleSize"`
	DefaultSampleSize    int `json:"defaultSampleSize"`
	MaxK                 int `json:"maxK"`
	DefaultK             int `json:"defaultK"`
	MaxEvaluationWorkers int `json:"maxEvaluationWorkers"`
	MaxSwitchUnitsPerRun int `json:"maxSwitchUnitsPerRun"`
	MaxBatchVectors      int `json:"maxBatchVectors"`
}

// DefaultLimits returns conservative repository-safe bounds.
func DefaultLimits() Limits {
	return Limits{
		MaxSampleSize:        1000,
		DefaultSampleSize:    100,
		MaxK:                 100,
		DefaultK:             10,
		MaxEvaluationWorkers: 4,
		MaxSwitchUnitsPerRun: 1024,
		MaxBatchVectors:      1000,
	}
}

func (l Limits) withDefaults() Limits {
	d := DefaultLimits()
	if l.MaxSampleSize <= 0 {
		l.MaxSampleSize = d.MaxSampleSize
	}
	if l.DefaultSampleSize <= 0 {
		l.DefaultSampleSize = d.DefaultSampleSize
	}
	if l.MaxK <= 0 {
		l.MaxK = d.MaxK
	}
	if l.DefaultK <= 0 {
		l.DefaultK = d.DefaultK
	}
	if l.MaxEvaluationWorkers <= 0 {
		l.MaxEvaluationWorkers = d.MaxEvaluationWorkers
	}
	if l.MaxSwitchUnitsPerRun <= 0 {
		l.MaxSwitchUnitsPerRun = d.MaxSwitchUnitsPerRun
	}
	if l.MaxBatchVectors <= 0 {
		l.MaxBatchVectors = d.MaxBatchVectors
	}
	if l.DefaultSampleSize > l.MaxSampleSize {
		l.DefaultSampleSize = l.MaxSampleSize
	}
	if l.DefaultK > l.MaxK {
		l.DefaultK = l.MaxK
	}
	return l
}

// Query is a sampled evaluation query. ID makes stable-hash sampling and
// per-query recall output deterministic; Vector is never retained beyond an
// evaluation batch.
type Query struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"-"`
}

// SampleOptions controls reproducible query sampling.
type SampleOptions struct {
	SampleSize int    `json:"sampleSize"`
	K          int    `json:"k"`
	Seed       int64  `json:"seed"`
	Strategy   string `json:"strategy"`
}

func (o SampleOptions) withDefaults(l Limits, scopeSeed int64) SampleOptions {
	l = l.withDefaults()
	if o.SampleSize <= 0 {
		o.SampleSize = l.DefaultSampleSize
	}
	if o.SampleSize > l.MaxSampleSize {
		o.SampleSize = l.MaxSampleSize
	}
	if o.K <= 0 {
		o.K = l.DefaultK
	}
	if o.K > l.MaxK {
		o.K = l.MaxK
	}
	if o.Strategy == "" {
		o.Strategy = StrategyStableHash
	}
	if o.Seed == 0 {
		// Scope seed keeps different collections/vectors from choosing the same
		// ID ranks while remaining reproducible. Explicit Seed always wins.
		o.Seed = scopeSeed
	}
	return o
}

func (o SampleOptions) validate() error {
	if o.SampleSize <= 0 {
		return fmt.Errorf("%w: sampleSize must be positive", ErrInvalidTarget)
	}
	if o.K <= 0 {
		return fmt.Errorf("%w: k must be positive", ErrInvalidTarget)
	}
	switch o.Strategy {
	case StrategyStableHash, StrategySeededReservoir:
	default:
		return fmt.Errorf("%w: unknown sampling strategy %q", ErrInvalidTarget, o.Strategy)
	}
	return nil
}

// ScopeSeed derives a stable int64 seed from collection and vector.
func ScopeSeed(collection, vector string) int64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte("weaviate-quantization-eval/v1/"))
	_, _ = h.Write([]byte(collection))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(vector))
	return int64(h.Sum64())
}

func stableHash(seed int64, id string) uint64 {
	h := fnv.New64a()
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(seed))
	_, _ = h.Write(b[:])
	_, _ = h.Write([]byte(id))
	return h.Sum64()
}

// SelectSamples returns a deterministic subset. The returned order is sorted
// by query ID so concurrent result aggregation and JSON output are stable.
func SelectSamples(queries []Query, opts SampleOptions) []Query {
	if len(queries) == 0 {
		return nil
	}
	if opts.SampleSize <= 0 || opts.SampleSize >= len(queries) {
		out := append([]Query(nil), queries...)
		sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
		return out
	}

	var out []Query
	switch opts.Strategy {
	case StrategySeededReservoir:
		out = reservoirSample(queries, opts.SampleSize, opts.Seed)
	default:
		type ranked struct {
			q    Query
			hash uint64
		}
		rankedQueries := make([]ranked, len(queries))
		for i, q := range queries {
			rankedQueries[i] = ranked{q: q, hash: stableHash(opts.Seed, q.ID)}
		}
		sort.SliceStable(rankedQueries, func(i, j int) bool {
			if rankedQueries[i].hash != rankedQueries[j].hash {
				return rankedQueries[i].hash < rankedQueries[j].hash
			}
			return rankedQueries[i].q.ID < rankedQueries[j].q.ID
		})
		out = make([]Query, opts.SampleSize)
		for i := 0; i < opts.SampleSize; i++ {
			out[i] = rankedQueries[i].q
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// splitMix64 is a deterministic pseudo-random stream.
type splitMix64 struct{ state uint64 }

func (r *splitMix64) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func reservoirSample(queries []Query, n int, seed int64) []Query {
	rng := splitMix64{state: uint64(seed)}
	out := append([]Query(nil), queries[:n]...)
	for i := n; i < len(queries); i++ {
		// Random integer in [0,i].
		j := int(rng.next() % uint64(i+1))
		if j < n {
			out[j] = queries[i]
		}
	}
	return out
}
