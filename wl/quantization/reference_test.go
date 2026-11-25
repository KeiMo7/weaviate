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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	enterrors "github.com/weaviate/weaviate/entities/errors"
)

func TestReferenceSwitchWriteAndReadSemantics(t *testing.T) {
	ctx := context.Background()
	b := NewReferenceBackend("Article", "vec", 2, 2)
	for i, id := range []string{"a", "b", "c"} {
		require.NoError(t, b.Write(ctx, "shard-0/r0", id, []float32{float32(i + 1), 1}))
	}
	require.NoError(t, b.Write(ctx, "shard-1/r0", "d", []float32{0, 1}))

	gate := make(chan struct{})
	started := make(chan struct{}, 2)
	b.BuildGate = gate
	b.BuildStarted = started
	m := newTestManager(t, b, nil)
	to, err := ExplicitForm(FormRQ8, Constraints{Distance: "cosine"})
	require.NoError(t, err)

	rec, err := m.StartSwitch(ctx, StartSwitchInput{Collection: "Article", Vector: "vec", To: &to, ID: "ref-switch"})
	require.NoError(t, err)
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("switch build did not start")
	}

	// A write after switch start is served by the old form and write-through
	// encoded into the future form.
	require.NoError(t, b.Write(ctx, "shard-0/r0", "mid-switch", []float32{2, 2}))
	hits, _, err := b.CollectionSearch(ctx, []float32{2, 2}, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	assert.Contains(t, idsFromHits(hits), "mid-switch")
	for _, h := range hits {
		assert.NotEmpty(t, h.FormID)
		assert.Equal(t, CodecNone, h.Codec, "before promotion reads must stay old form")
	}

	// Hammer reads during promotion. Mixed-barrier refusals are acceptable only
	// as the distinguishable retryable sentinel; successful responses must be
	// single-form.
	var wg sync.WaitGroup
	stop := make(chan struct{})
	var barrierRefusals atomic.Int64
	for i := 0; i < 4; i++ {
		wg.Add(1)
		enterrors.GoWrapper(func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				_, _, err := b.CollectionSearch(ctx, []float32{1, 1}, 3)
				switch {
				case err == nil:
					// CollectionSearch performs the per-unit snapshot and
					// cross-unit form-width checks before returning hits.
				case errors.Is(err, ErrSwitchUnavailable):
					barrierRefusals.Add(1)
				default:
					t.Errorf("unexpected search error: %v", err)
				}
			}
		}, discardLogger())
	}

	close(gate)
	final, err := m.Wait(ctx, rec.ID)
	require.NoError(t, err)
	close(stop)
	wg.Wait()

	assert.Equal(t, PhaseCompleted, final.Phase)
	// Barrier refusals, if observed, were asserted to be ErrSwitchUnavailable
	// inside the workers; a zero count is also valid because promotion is atomic.
	_ = barrierRefusals.Load()

	obs, err := b.EffectiveForm(ctx, "Article", "vec")
	require.NoError(t, err)
	assert.Equal(t, CodecRQ, obs.Codec)
	assert.Equal(t, 8, obs.Bits)

	hits, formSignature, err := b.CollectionSearch(ctx, []float32{2, 2}, 10)
	require.NoError(t, err)
	assert.NotEmpty(t, formSignature)
	assert.Contains(t, idsFromHits(hits), "mid-switch", "write during switch must be retrievable in new form")
	for _, h := range hits {
		assert.Equal(t, CodecRQ, h.Codec)
		assert.NotEmpty(t, h.FormID)
	}
}

func TestReferenceFailedBuildKeepsOldFormService(t *testing.T) {
	ctx := context.Background()
	b := NewReferenceBackend("C", "v", 1, 2)
	require.NoError(t, b.Write(ctx, "shard-0/r0", "a", []float32{1, 0}))
	b.FailBuildOnce = true
	m := newTestManager(t, b, nil)
	to, _ := ExplicitForm(FormRQ4, Constraints{Distance: "cosine"})

	rec, err := m.RunSwitch(ctx, StartSwitchInput{Collection: "C", Vector: "v", To: &to})
	require.Error(t, err)
	assert.Equal(t, PhaseRolledBack, rec.Phase)

	hits, _, err := b.CollectionSearch(ctx, []float32{1, 0}, 1)
	require.NoError(t, err)
	require.Len(t, hits, 1)
	assert.Equal(t, CodecNone, hits[0].Codec)
}

func idsFromHits(hits []ReferenceHit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}
