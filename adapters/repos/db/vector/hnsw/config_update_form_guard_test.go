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

package hnsw

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	ent "github.com/weaviate/weaviate/entities/vectorindex/hnsw"
)

// TestUpdateUserConfigRejectsSilentFormChange pins the echo-vs-data bug: once
// the live index is compressed, accepting a disabled/different codec config
// without rebuilding would make schema readback claim a form search no longer
// uses. The managed quantization switch is the only supported path across that
// boundary.
func TestUpdateUserConfigRejectsSilentFormChange(t *testing.T) {
	tests := []struct {
		name    string
		current ent.UserConfig
		updated ent.UserConfig
	}{
		{
			name:    "disable compression",
			current: ent.UserConfig{BQ: ent.BQConfig{Enabled: true}},
			updated: ent.UserConfig{},
		},
		{
			name:    "change bq to rq8",
			current: ent.UserConfig{BQ: ent.BQConfig{Enabled: true}},
			updated: ent.UserConfig{RQ: ent.RQConfig{Enabled: true, Bits: 8}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &hnsw{compressActionLock: &sync.RWMutex{}}
			h.compressed.Store(true)
			h.bqConfig = tt.current.BQ
			h.rqConfig = tt.current.RQ
			h.sqConfig = tt.current.SQ
			h.pqConfig = tt.current.PQ

			called := false
			err := h.UpdateUserConfig(tt.updated, func() { called = true })
			require.ErrorIs(t, err, ErrCompressionRebuildRequired)
			require.True(t, called, "callback must still settle callers waiting on config apply")
		})
	}
}
