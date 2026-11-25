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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTargetValidate(t *testing.T) {
	tests := []struct {
		name    string
		target  Target
		wantErr bool
	}{
		{name: "balanced explicit", target: Target{MinRecall: .95, Tolerance: .02}},
		{name: "alias fills values", target: Target{Alias: AliasBalanced}},
		{name: "alias matching explicit", target: Target{Alias: AliasBalanced, MinRecall: .95, Tolerance: .02}},
		{name: "lossless hard one", target: Target{MinRecall: 1}},
		{name: "nil", target: Target{}, wantErr: true},
		{name: "zero recall", target: Target{MinRecall: 0}, wantErr: true},
		{name: "negative recall", target: Target{MinRecall: -.1}, wantErr: true},
		{name: "recall above one", target: Target{MinRecall: 1.1}, wantErr: true},
		{name: "negative tolerance", target: Target{MinRecall: .9, Tolerance: -.1}, wantErr: true},
		{name: "tolerance swallows target", target: Target{MinRecall: .5, Tolerance: .5}, wantErr: true},
		{name: "hard one with tolerance", target: Target{MinRecall: 1, Tolerance: .01}, wantErr: true},
		{name: "unknown alias", target: Target{Alias: "does-not-exist"}, wantErr: true},
		{name: "alias mismatch recall", target: Target{Alias: AliasBalanced, MinRecall: .8}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.target.Validate()
			if tt.wantErr {
				require.Error(t, err)
				assert.True(t, errors.Is(err, ErrInvalidTarget), "error %v must wrap ErrInvalidTarget", err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestResolveDeterministicAndStable(t *testing.T) {
	constraints := Constraints{Dimensions: 128, Distance: "cosine", IndexType: IndexHNSW}
	target := Target{MinRecall: .95, Tolerance: .02}

	first, err := Resolve(target, constraints)
	require.NoError(t, err)
	second, err := Resolve(target, constraints)
	require.NoError(t, err)
	assert.Equal(t, first, second)
	assert.Equal(t, PolicyVersion, first.PolicyVersion)
	assert.Equal(t, CodecRQ, first.Codec)
	assert.Equal(t, 8, first.Bits)
	assert.Greater(t, first.RescoreLimit, 0)

	// Replaying after restart uses only the persisted policy version and inputs;
	// there is no node-local randomness in this mapping.
	third, err := Resolve(Target{Alias: AliasBalanced}, constraints)
	require.NoError(t, err)
	assert.Equal(t, first.Codec, third.Codec)
	assert.Equal(t, first.Bits, third.Bits)
}

func TestResolveTable(t *testing.T) {
	cos := Constraints{Dimensions: 64, Distance: "cosine"}
	ham := Constraints{Dimensions: 64, Distance: "hamming"}
	flat := Constraints{Dimensions: 64, Distance: "cosine", IndexType: IndexFlat}

	tests := []struct {
		name      string
		minRecall float64
		c         Constraints
		wantCodec string
		wantBits  int
		wantErrIs error
	}{
		{name: "exact is uncompressed", minRecall: 1, c: cos, wantCodec: CodecNone, wantBits: 32},
		{name: "ultra uses eight bits", minRecall: .99, c: cos, wantCodec: CodecRQ, wantBits: 8},
		{name: "balanced uses eight bits", minRecall: .96, c: cos, wantCodec: CodecRQ, wantBits: 8},
		{name: "compact uses four bits", minRecall: .91, c: cos, wantCodec: CodecRQ, wantBits: 4},
		{name: "economy uses one bit", minRecall: .82, c: cos, wantCodec: CodecRQ, wantBits: 1},
		{name: "hamming falls back to BQ", minRecall: .91, c: ham, wantCodec: CodecBQ, wantBits: 1},
		{name: "flat high uses RQ8", minRecall: .97, c: flat, wantCodec: CodecRQ, wantBits: 8},
		{name: "flat compact uses legal BQ", minRecall: .91, c: flat, wantCodec: CodecBQ, wantBits: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Resolve(Target{MinRecall: tt.minRecall}, tt.c)
			if tt.wantErrIs != nil {
				require.ErrorIs(t, err, tt.wantErrIs)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantCodec, got.Codec)
			assert.Equal(t, tt.wantBits, got.Bits)
			assert.NotEmpty(t, got.Rationale)
		})
	}
}

func TestExplicitFormBidirectional(t *testing.T) {
	for _, form := range []Form{FormUncompressed, FormBQ1, FormRQ1, FormRQ4, FormRQ8, FormSQ8, FormPQ} {
		t.Run(string(form), func(t *testing.T) {
			c := Constraints{Dimensions: 128, Distance: "cosine"}
			if form == FormSQ8 || form == FormPQ {
				c.IndexType = IndexHNSW
			}
			got, err := ExplicitForm(form, c)
			require.NoError(t, err)
			assert.Equal(t, form, got.Form)
			assert.Equal(t, PolicyVersion, got.PolicyVersion)
		})
	}
}

func TestExplicitFormRejectsUnsupportedFlatRQ4(t *testing.T) {
	_, err := ExplicitForm(FormRQ4, Constraints{IndexType: IndexFlat, Distance: "cosine"})
	require.ErrorIs(t, err, ErrUnsupportedCombination)
}

func TestEquivalentFormLegacySemantics(t *testing.T) {
	assert.Equal(t, FormUncompressed, EquivalentForm("none", 0))
	assert.Equal(t, FormBQ1, EquivalentForm("bq", 0))
	assert.Equal(t, FormSQ8, EquivalentForm("sq", 0))
	assert.Equal(t, FormPQ, EquivalentForm("pq", 0))
	assert.Equal(t, FormRQ8, EquivalentForm("rq", 8))
	assert.Equal(t, FormRQ4, EquivalentForm("rq", 4))
	assert.Equal(t, FormRQ1, EquivalentForm("rq", 1))
}
