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

//go:build integrationTest

package db

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/additional"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/storobj"
	enthnsw "github.com/weaviate/weaviate/entities/vectorindex/hnsw"
	"github.com/weaviate/weaviate/wl/quantization"
)

type shardQuantizedSearcher struct {
	shard  ShardLike
	target string
}

func (s *shardQuantizedSearcher) SearchQuantized(ctx context.Context, q []float32, k int) ([]string, error) {
	idx, release, ok := s.shard.AcquireVectorIndex(s.target)
	if !ok {
		return nil, fmt.Errorf("vector index %q is not available", s.target)
	}
	defer release()
	ids, _, err := idx.SearchByVector(ctx, q, k, nil)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = strconv.FormatUint(id, 10)
	}
	return out, nil
}

type shardRawVectorBatcher struct {
	shard  *Shard
	target string
}

func (b *shardRawVectorBatcher) ScanRawVectors(ctx context.Context, batchSize int, fn func(quantization.VectorBatch) error) error {
	if batchSize <= 0 {
		batchSize = 256
	}
	batch := quantization.VectorBatch{}
	flush := func() error {
		if len(batch.IDs) == 0 {
			return nil
		}
		if err := fn(batch); err != nil {
			return err
		}
		batch = quantization.VectorBatch{}
		return nil
	}
	err := b.shard.iterateOnLSMVectors(ctx, 0, b.target, func(id uint64, v []float32) error {
		if len(batch.IDs) >= batchSize {
			if err := flush(); err != nil {
				return err
			}
		}
		batch.IDs = append(batch.IDs, strconv.FormatUint(id, 10))
		batch.Vectors = append(batch.Vectors, append([]float32(nil), v...))
		return nil
	})
	if err != nil {
		return err
	}
	return flush()
}

// TestQuantizationLocalShardRecallIntegration evaluates recall against a real
// local shard before and after enabling SQ, and writes during compression.
func TestQuantizationLocalShardRecallIntegration(t *testing.T) {
	ctx := context.Background()
	className := "QuantizationRecallLocalShard"
	cfg := enthnsw.NewDefaultUserConfig()
	cfg.EF = 256
	cfg.DynamicEFMin = 128
	cfg.DynamicEFMax = 512

	shardLike, idx := testShardWithSettings(t, ctx, &models.Class{Class: className}, cfg, false, false)
	t.Cleanup(func() {
		_ = idx.drop()
		_ = os.RemoveAll(idx.Config.RootPath)
	})
	shard := underlyingShard(t, shardLike)

	const n = 160
	const dims = 16
	rng := rand.New(rand.NewSource(42))
	vectors := make([][]float32, n)
	objs := make([]*storobj.Object, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dims)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vectors[i] = v
		obj := testObject(className)
		obj.Vector = v
		objs[i] = obj
	}
	errs := shardLike.PutObjectBatch(ctx, objs)
	for _, err := range errs {
		require.NoError(t, err)
	}
	waitForVectorQueue(t, shardLike, "")

	queries := make([]quantization.Query, 20)
	for i := range queries {
		queries[i] = quantization.Query{ID: fmt.Sprintf("q-%02d", i), Vector: vectors[i*4]}
	}
	limits := quantization.DefaultLimits()
	limits.DefaultK = 10

	evalReq := quantization.EvaluationRequest{
		Collection:    className,
		Vector:        "",
		Options:       quantization.SampleOptions{SampleSize: len(queries), K: 10, Strategy: quantization.StrategyStableHash, Seed: 42},
		SampleQueries: queries,
		Current:       &shardQuantizedSearcher{shard: shardLike, target: ""},
		Full: quantization.NewStreamingExactSearcher(
			&shardRawVectorBatcher{shard: shard, target: ""}, quantization.CosineDistance, 64),
		Logger: discardQuantTestLogger(),
	}
	baseline, err := quantization.Evaluate(ctx, evalReq, limits)
	require.NoError(t, err)
	require.NotNil(t, baseline.Recall)
	assert.GreaterOrEqual(t, *baseline.Recall, .9, "uncompressed HNSW baseline should be near exact")

	// Enable 8-bit SQ with deep rescoring. This is the existing local-shard
	// online compression path; the quantization manager's bidirectional state
	// machine is covered exhaustively in wl/quantization tests.
	compressedCfg := cfg
	compressedCfg.SQ.Enabled = true
	compressedCfg.SQ.RescoreLimit = 128
	compressedCfg.SQ.RescoreFactor = 2
	require.NoError(t, shardLike.UpdateVectorIndexConfig(ctx, compressedCfg))

	// Write while compression is in progress and ensure it remains searchable.
	mid := testObject(className)
	mid.Vector = []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	require.NoError(t, shardLike.PutObject(ctx, mid))

	vidx, release, ok := shardLike.AcquireVectorIndex("")
	require.True(t, ok)
	require.Eventually(t, vidx.Compressed, 60*time.Second, 100*time.Millisecond)
	release()
	waitForVectorQueue(t, shardLike, "")

	compressedReport, err := quantization.Evaluate(ctx, evalReq, limits)
	require.NoError(t, err)
	require.NotNil(t, compressedReport.Recall)
	assert.GreaterOrEqual(t, *compressedReport.Recall, .9,
		"SQ8 with rescore 128 should meet a high recall target on this fixture")

	hits, _, err := shardLike.ObjectVectorSearch(ctx, []models.Vector{mid.Vector}, []string{""}, 0, 5,
		nil, nil, nil, additional.Properties{}, nil, nil)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	assert.Equal(t, mid.ID(), hits[0].ID())
}

func discardQuantTestLogger() logrus.FieldLogger {
	l := logrus.New()
	l.SetOutput(io.Discard)
	return l
}
