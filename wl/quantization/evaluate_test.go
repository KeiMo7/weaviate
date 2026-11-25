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
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelectSamplesDeterministic(t *testing.T) {
	queries := make([]Query, 50)
	for i := range queries {
		queries[i] = Query{ID: idN(i), Vector: []float32{float32(i)}}
	}
	opts := SampleOptions{SampleSize: 12, Seed: 42, Strategy: StrategyStableHash}

	first := SelectSamples(queries, opts)
	second := SelectSamples(queries, opts)
	require.Len(t, first, 12)
	assert.Equal(t, queryIDs(first), queryIDs(second))

	// Input order must not change stable-hash samples.
	reversed := make([]Query, len(queries))
	for i := range queries {
		reversed[len(queries)-1-i] = queries[i]
	}
	assert.Equal(t, queryIDs(first), queryIDs(SelectSamples(reversed, opts)))

	reservoir := SampleOptions{SampleSize: 12, Seed: 7, Strategy: StrategySeededReservoir}
	r1 := SelectSamples(queries, reservoir)
	r2 := SelectSamples(queries, reservoir)
	assert.Equal(t, queryIDs(r1), queryIDs(r2))
	require.Len(t, r1, 12)
}

func TestEvaluationSuccessAndNoZeroOnFailure(t *testing.T) {
	vectors := [][]float32{
		{1, 0}, {0, 1}, {1, 1}, {0, 0}, {.9, .1},
	}
	ids := []string{"a", "b", "c", "d", "e"}
	batcher := &sliceBatcher{ids: ids, vectors: vectors}
	full := NewStreamingExactSearcher(batcher, SquaredL2Distance, 2)
	current := &staticSearcher{results: map[string][]string{
		"q1": {"a", "e"},
	}}
	req := EvaluationRequest{
		Collection: "C", Vector: "v",
		SampleQueries: []Query{{ID: "q1", Vector: []float32{1, 0}}},
		Current:       current, Full: full,
		Observed: FormObservation{Codec: CodecNone, Bits: 32},
	}
	report, err := Evaluate(context.Background(), req, DefaultLimits())
	require.NoError(t, err)
	require.NotNil(t, report.Recall)
	assert.Equal(t, EvalStatusSucceeded, report.Status)
	assert.Equal(t, 1, report.SampleCount)
	assert.Equal(t, 10, report.K) // default K
	assert.Greater(t, report.ResultCount, 0)
	assert.GreaterOrEqual(t, report.Took, time.Duration(0))
}

func TestEvaluationFailureHasNilRecall(t *testing.T) {
	req := EvaluationRequest{
		Collection: "C", Vector: "v",
		SampleQueries: []Query{{ID: "q1", Vector: []float32{1}}},
		Current:       &errorSearcher{}, Full: &fakeFullSearcher{},
	}
	report, err := Evaluate(context.Background(), req, DefaultLimits())
	require.Error(t, err)
	require.NotNil(t, report)
	assert.Equal(t, EvalStatusFailed, report.Status)
	assert.Nil(t, report.Recall)
	assert.Contains(t, report.Error, "boom")
}

func TestEvaluationCanceledIsDistinguishable(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := EvaluationRequest{
		Collection: "C", Vector: "v",
		SampleQueries: []Query{{ID: "q1", Vector: []float32{1}}},
		Current:       &blockingSearcher{}, Full: &fakeFullSearcher{},
	}
	report, err := Evaluate(ctx, req, DefaultLimits())
	require.ErrorIs(t, err, ErrEvaluationCanceled)
	assert.Equal(t, EvalStatusCanceled, report.Status)
	assert.Nil(t, report.Recall)
}

type sliceBatcher struct {
	ids     []string
	vectors [][]float32
}

func (b *sliceBatcher) ScanRawVectors(_ context.Context, size int, fn func(VectorBatch) error) error {
	for start := 0; start < len(b.ids); start += size {
		end := min(start+size, len(b.ids))
		if err := fn(VectorBatch{IDs: b.ids[start:end], Vectors: b.vectors[start:end]}); err != nil {
			return err
		}
	}
	return nil
}

type staticSearcher struct{ results map[string][]string }

func (s *staticSearcher) SearchQuantized(_ context.Context, q []float32, k int) ([]string, error) {
	// Tests use one query ID indirectly? Evaluate does not pass ID. Use vector key
	// impossible here; return first configured result for simple tests.
	for _, ids := range s.results {
		if k > 0 && len(ids) > k {
			return ids[:k], nil
		}
		return ids, nil
	}
	return nil, nil
}

type errorSearcher struct{}

func (errorSearcher) SearchQuantized(context.Context, []float32, int) ([]string, error) {
	return nil, errors.New("boom")
}

type blockingSearcher struct{}

func (blockingSearcher) SearchQuantized(ctx context.Context, _ []float32, _ int) ([]string, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

type fakeFullSearcher struct{}

func (fakeFullSearcher) SearchFullPrecision(context.Context, []float32, int) ([]string, error) {
	return []string{"a"}, nil
}

func idN(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "id-000"
	}
	var b []byte
	for n := i; n > 0; n /= 10 {
		b = append([]byte{digits[n%10]}, b...)
	}
	return "id-" + string(b)
}

func queryIDs(qs []Query) []string {
	out := make([]string, len(qs))
	for i := range qs {
		out[i] = qs[i].ID
	}
	return out
}
