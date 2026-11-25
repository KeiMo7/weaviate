//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| | \ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_|  \_/ |_|\__,_|_| \___/
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package quantization

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/vectorindex/hnsw"
)

func TestHNSWConfigRoundTripEquivalentForm(t *testing.T) {
	base := hnsw.NewDefaultUserConfig()
	form, err := ExplicitForm(FormRQ4, Constraints{Distance: "cosine", IndexType: IndexHNSW})
	require.NoError(t, err)

	cfg := HNSWConfigForForm(base, form)
	assert.True(t, cfg.RQ.Enabled)
	assert.EqualValues(t, 4, cfg.RQ.Bits)
	assert.False(t, cfg.BQ.Enabled)
	assert.False(t, cfg.SQ.Enabled)
	assert.False(t, cfg.PQ.Enabled)
	assert.Equal(t, base.MaxConnections, cfg.MaxConnections, "immutable HNSW field must be preserved")

	got := FormFromHNSWConfig(cfg)
	assert.Equal(t, FormRQ4, got.Form)
	assert.Equal(t, CodecRQ, got.Codec)
	assert.Equal(t, 4, got.Bits)
}

func TestHNSWConfigForUncompressedDisablesAllCodecs(t *testing.T) {
	base := hnsw.NewDefaultUserConfig()
	base.BQ.Enabled = true
	form, err := ExplicitForm(FormUncompressed, Constraints{})
	require.NoError(t, err)
	cfg := HNSWConfigForForm(base, form)
	assert.False(t, cfg.BQ.Enabled)
	assert.False(t, cfg.RQ.Enabled)
	assert.False(t, cfg.SQ.Enabled)
	assert.False(t, cfg.PQ.Enabled)
	assert.Equal(t, FormUncompressed, FormFromHNSWConfig(cfg).Form)
}
