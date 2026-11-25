//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//   \_/\_/ \___|\__,_|  \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package quantization

import (
	"github.com/weaviate/weaviate/entities/vectorindex/flat"
	"github.com/weaviate/weaviate/entities/vectorindex/hnsw"
)

// HNSWConfigForForm returns a copy of base with exactly one compression block
// enabled, configured according to f. Immutable HNSW fields are preserved by
// copying base first.
func HNSWConfigForForm(base hnsw.UserConfig, f FormSpec) hnsw.UserConfig {
	out := base
	out.PQ.Enabled = false
	out.BQ.Enabled = false
	out.SQ.Enabled = false
	out.RQ.Enabled = false

	rescore := f.RescoreLimit
	factor := f.RescoreFactor
	switch f.Codec {
	case CodecPQ:
		out.PQ.Enabled = true
		out.PQ.BitCompression = f.Bits == 1
		out.PQ.Segments = f.PQSegments
		if f.PQCentroids > 0 {
			out.PQ.Centroids = f.PQCentroids
		}
		if f.TrainingLimit > 0 {
			out.PQ.TrainingLimit = f.TrainingLimit
		}
		out.PQ.RescoreLimit = rescore
		out.PQ.RescoreFactor = factor
	case CodecBQ:
		out.BQ.Enabled = true
		out.BQ.RescoreLimit = rescore
		out.BQ.RescoreFactor = factor
	case CodecSQ:
		out.SQ.Enabled = true
		if f.TrainingLimit > 0 {
			out.SQ.TrainingLimit = f.TrainingLimit
		}
		out.SQ.RescoreLimit = rescore
		out.SQ.RescoreFactor = factor
	case CodecRQ:
		out.RQ.Enabled = true
		out.RQ.Bits = int16(f.Bits)
		out.RQ.Centering = f.Centering
		if f.TrainingLimit > 0 {
			out.RQ.TrainingLimit = f.TrainingLimit
		}
		out.RQ.RescoreLimit = rescore
		out.RQ.RescoreFactor = factor
	case CodecNone:
		// all compression blocks remain disabled
	}
	return out
}

// FormFromHNSWConfig parses the equivalent form from an existing HNSW config.
// It gives legacy compression settings equivalent semantics without requiring a
// declared recall target.
func FormFromHNSWConfig(cfg hnsw.UserConfig) FormSpec {
	switch {
	case cfg.PQ.Enabled:
		return FormSpec{
			PolicyVersion: PolicyVersion, Form: FormPQ, Codec: CodecPQ, Bits: 8,
			PQSegments: cfg.PQ.Segments, PQCentroids: cfg.PQ.Centroids,
			RescoreLimit: cfg.PQ.RescoreLimit, RescoreFactor: cfg.PQ.RescoreFactor,
			TrainingLimit: cfg.PQ.TrainingLimit,
		}
	case cfg.BQ.Enabled:
		return FormSpec{
			PolicyVersion: PolicyVersion, Form: FormBQ1, Codec: CodecBQ, Bits: 1,
			RescoreLimit: cfg.BQ.RescoreLimit, RescoreFactor: cfg.BQ.RescoreFactor,
		}
	case cfg.SQ.Enabled:
		return FormSpec{
			PolicyVersion: PolicyVersion, Form: FormSQ8, Codec: CodecSQ, Bits: 8,
			RescoreLimit: cfg.SQ.RescoreLimit, RescoreFactor: cfg.SQ.RescoreFactor,
			TrainingLimit: cfg.SQ.TrainingLimit,
		}
	case cfg.RQ.Enabled:
		bits := int(cfg.RQ.Bits)
		return FormSpec{
			PolicyVersion: PolicyVersion, Form: EquivalentForm(CodecRQ, bits), Codec: CodecRQ, Bits: bits,
			Centering: cfg.RQ.Centering, RescoreLimit: cfg.RQ.RescoreLimit,
			RescoreFactor: cfg.RQ.RescoreFactor, TrainingLimit: cfg.RQ.TrainingLimit,
		}
	default:
		return FormSpec{PolicyVersion: PolicyVersion, Form: FormUncompressed, Codec: CodecNone, Bits: 32}
	}
}

// FlatConfigForForm returns a flat config with the supported BQ/RQ form set.
// Unsupported forms are left uncompressed and reported by the caller through
// ExplicitForm/Resolve validation.
func FlatConfigForForm(base flat.UserConfig, f FormSpec) flat.UserConfig {
	out := base
	out.BQ.Enabled = false
	out.PQ.Enabled = false
	out.SQ.Enabled = false
	out.RQ.Enabled = false
	switch f.Codec {
	case CodecBQ:
		out.BQ.Enabled = true
		out.BQ.RescoreLimit = f.RescoreLimit
		out.BQ.RescoreFactor = f.RescoreFactor
	case CodecRQ:
		out.RQ.Enabled = true
		out.RQ.Bits = f.Bits
		out.RQ.RescoreLimit = f.RescoreLimit
		out.RQ.RescoreFactor = f.RescoreFactor
	case CodecNone:
		// no compression
	}
	return out
}
