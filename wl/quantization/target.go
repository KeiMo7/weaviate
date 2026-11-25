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

// Package quantization provides collection-level recall targets, deterministic
// target-to-form resolution, recall evaluation, online form switching and
// observability.
//
// It is Weaviate-licensed (wl/LICENSE-WEAVIATE), unlike the BSD-3-Clause code
// outside wl/.
package quantization

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// PolicyVersion is persisted in every resolved form and switch record. Bump it
// only when the target-to-form table changes; records created by older builds
// remain readable and never silently reinterpreted.
const PolicyVersion = "recall-form-v1"

// Codec names match the compression names already used by schema and usage
// reporting.
const (
	CodecNone = "none"
	CodecPQ   = "pq"
	CodecBQ   = "bq"
	CodecSQ   = "sq"
	CodecRQ   = "rq"
)

// Index types accepted in ResolveConstraints. Empty is treated as hnsw.
const (
	IndexHNSW    = "hnsw"
	IndexFlat    = "flat"
	IndexDynamic = "dynamic"
)

// Form is the concrete encoded shape selected for a target.
type Form string

const (
	FormUncompressed Form = "none"
	FormBQ1          Form = "bq1"
	FormRQ1          Form = "rq1"
	FormRQ4          Form = "rq4"
	FormRQ8          Form = "rq8"
	FormSQ8          Form = "sq8"
	FormPQ           Form = "pq"
)

// Profile aliases are stable operator-facing档位 names.
const (
	AliasLossless   = "lossless"
	AliasFull       = "full-precision"
	AliasUltra      = "ultra"
	AliasHigh       = "high-recall"
	AliasBalanced   = "balanced"
	AliasCompact    = "compact"
	AliasEconomy    = "economy"
	AliasAggressive = "aggressive"
)

var (
	// ErrInvalidTarget marks every rejected create/update target combination.
	ErrInvalidTarget = errors.New("invalid quantization recall target")
	// ErrUnsupportedCombination marks a target that is legal in isolation but
	// cannot be represented for the collection's immutable constraints.
	ErrUnsupportedCombination = errors.New("unsupported quantization combination")
	// ErrActiveSwitch exists because a collection may have only one switch.
	ErrActiveSwitch = errors.New("quantization form switch already active")
	// ErrNoActiveSwitch is returned when a terminal mutation is requested for
	// an unknown or already terminal switch.
	ErrNoActiveSwitch = errors.New("no active quantization form switch")
	// ErrIllegalTransition pins the durable state machine's transition table.
	ErrIllegalTransition = errors.New("illegal quantization switch phase transition")
	// ErrEvaluationFailed is the distinguishable evaluation failure state.
	ErrEvaluationFailed = errors.New("recall evaluation failed")
	// ErrEvaluationCanceled is the distinguishable cancellation state.
	ErrEvaluationCanceled = errors.New("recall evaluation canceled")
	// ErrSwitchUnavailable is a retryable, overload-independent read refusal.
	ErrSwitchUnavailable = errors.New("quantization switch barrier: form temporarily unavailable")
	// ErrCollectionGone means the collection was deleted or shard unloaded.
	ErrCollectionGone = errors.New("collection or shard no longer exists")
)

// Target is the declarative accuracy objective for one vector.
//
// MinRecall is the required lower bound in [0,1]. Tolerance is an absolute
// evaluation confidence/measurement slack in [0,1): an evaluation passes when
// recall >= MinRecall-Tolerance. It never changes the form selected from
// MinRecall. Alias may either fill omitted fields or assert that the explicit
// values are the named profile.
type Target struct {
	MinRecall float64 `json:"minRecall"`
	Tolerance float64 `json:"tolerance,omitempty"`
	Alias     string  `json:"alias,omitempty"`
}

// RecallTarget is an explanatory alias for callers that prefer the full name.
type RecallTarget = Target

// Profile describes a named target档位.
type Profile struct {
	Alias       string  `json:"alias"`
	MinRecall   float64 `json:"minRecall"`
	Tolerance   float64 `json:"tolerance"`
	Description string  `json:"description"`
}

// Profiles returns the built-in aliases in recall order.
func Profiles() []Profile {
	return []Profile{
		{AliasLossless, 1, 0, "exact/full-precision vectors; no compression"},
		{AliasFull, 1, 0, "alias of lossless"},
		{AliasUltra, 0.99, 0.005, "near-lossless 8-bit encoding with extra rescore depth"},
		{AliasHigh, 0.97, 0.01, "high-recall 8-bit encoding"},
		{AliasBalanced, 0.95, 0.02, "balanced 8/4-bit encoding and rescore depth"},
		{AliasCompact, 0.90, 0.03, "compact 4-bit rotational encoding"},
		{AliasEconomy, 0.85, 0.05, "1-bit encoding with deep rescoring"},
		{AliasAggressive, 0.80, 0.05, "maximum compression with the deepest rescore budget"},
	}
}

func profileByName(alias string) (Profile, bool) {
	alias = strings.ToLower(strings.TrimSpace(alias))
	for _, p := range Profiles() {
		if p.Alias == alias {
			return p, true
		}
	}
	return Profile{}, false
}

// Validate normalizes and checks a target. A zero-valued target is invalid:
// "undeclared" is represented by absence of a Target pointer, not Target{}.
func (t *Target) Validate() error {
	if t == nil {
		return fmt.Errorf("%w: nil target", ErrInvalidTarget)
	}
	if t.Alias != "" {
		p, ok := profileByName(t.Alias)
		if !ok {
			return fmt.Errorf("%w: unknown alias %q", ErrInvalidTarget, t.Alias)
		}
		// Explicit zero values mean "not supplied" when an alias is present.
		if t.MinRecall == 0 {
			t.MinRecall = p.MinRecall
		}
		if t.Tolerance == 0 {
			t.Tolerance = p.Tolerance
		}
		if math.Abs(t.MinRecall-p.MinRecall) > 1e-12 {
			return fmt.Errorf("%w: alias %q requires minRecall %.3f, got %.6f",
				ErrInvalidTarget, p.Alias, p.MinRecall, t.MinRecall)
		}
		if math.Abs(t.Tolerance-p.Tolerance) > 1e-12 {
			return fmt.Errorf("%w: alias %q requires tolerance %.3f, got %.6f",
				ErrInvalidTarget, p.Alias, p.Tolerance, t.Tolerance)
		}
		t.Alias = p.Alias
	}

	if math.IsNaN(t.MinRecall) || math.IsInf(t.MinRecall, 0) {
		return fmt.Errorf("%w: minRecall must be finite", ErrInvalidTarget)
	}
	if t.MinRecall <= 0 || t.MinRecall > 1 {
		return fmt.Errorf("%w: minRecall must be in (0,1], got %v", ErrInvalidTarget, t.MinRecall)
	}
	if math.IsNaN(t.Tolerance) || math.IsInf(t.Tolerance, 0) {
		return fmt.Errorf("%w: tolerance must be finite", ErrInvalidTarget)
	}
	if t.Tolerance < 0 || t.Tolerance >= 1 {
		return fmt.Errorf("%w: tolerance must be in [0,1), got %v", ErrInvalidTarget, t.Tolerance)
	}
	if t.Tolerance >= t.MinRecall {
		return fmt.Errorf("%w: tolerance %v must be smaller than minRecall %v",
			ErrInvalidTarget, t.Tolerance, t.MinRecall)
	}
	if t.MinRecall == 1 && t.Tolerance != 0 {
		return fmt.Errorf("%w: minRecall=1 is a hard exact target and requires tolerance=0",
			ErrInvalidTarget)
	}
	return nil
}

// Normalize validates and returns a value copy with alias defaults filled.
func (t Target) Normalize() (Target, error) {
	if err := t.Validate(); err != nil {
		return Target{}, err
	}
	return t, nil
}

// PassingFloor is MinRecall-Tolerance: the lowest measured recall that still
// passes an evaluation.
func (t Target) PassingFloor() float64 {
	return t.MinRecall - t.Tolerance
}

// Constraints are the immutable collection/vector facts the deterministic
// mapper is allowed to use. Node-local runtime state is deliberately absent.
type Constraints struct {
	Dimensions int    `json:"dimensions,omitempty"`
	Distance   string `json:"distance,omitempty"`
	IndexType  string `json:"indexType,omitempty"`
	// VectorCount is optional and only used to avoid selecting PQ before there
	// are enough training vectors. Zero means unknown; mapping remains stable.
	VectorCount int `json:"vectorCount,omitempty"`
}

// FormSpec is the single canonical answer of target -> encoding/width/rescore.
type FormSpec struct {
	PolicyVersion string  `json:"policyVersion"`
	Form          Form    `json:"form"`
	Codec         string  `json:"codec"`
	Bits          int     `json:"bits"`
	RescoreLimit  int     `json:"rescoreLimit"`
	RescoreFactor float64 `json:"rescoreFactor"`
	// PQ-specific fields. Segments zero means Weaviate's automatic adaptation.
	PQSegments    int  `json:"pqSegments,omitempty"`
	PQCentroids   int  `json:"pqCentroids,omitempty"`
	Centering     bool `json:"centering,omitempty"`
	TrainingLimit int  `json:"trainingLimit,omitempty"`
	// EstimatedCompressionRatio uses payload bits only; actual on-disk ratio is
	// separately measured and reported as ObservedForm.CompressionRatio.
	EstimatedCompressionRatio float64 `json:"estimatedCompressionRatio"`
	Rationale                 string  `json:"rationale"`
}

// Resolve maps a target and immutable constraints to exactly one form. This is
// the only target-to-form decision point; persistence and restart replay the
// returned PolicyVersion rather than consulting a node-local heuristic.
func Resolve(target Target, constraints Constraints) (FormSpec, error) {
	n, err := target.Normalize()
	if err != nil {
		return FormSpec{}, err
	}
	c, err := normalizeConstraints(constraints)
	if err != nil {
		return FormSpec{}, err
	}

	spec := FormSpec{PolicyVersion: PolicyVersion, RescoreFactor: 2}
	useRQ := rqAllowed(c.Distance)

	switch {
	case n.MinRecall >= 1:
		spec.Form, spec.Codec, spec.Bits = FormUncompressed, CodecNone, 32
		spec.RescoreFactor = 0
		spec.EstimatedCompressionRatio = 1
		spec.Rationale = "hard exact-recall target requires full-precision vectors"
	case n.MinRecall >= 0.985:
		if c.IndexType == IndexFlat {
			// Flat supports RQ8 but not SQ.
			spec = rqSpec(8, 128, 2, false)
		} else if useRQ {
			spec = rqSpec(8, 96, 2, false)
		} else {
			spec = sqSpec(128)
		}
		spec.Rationale = "near-lossless target selects 8-bit encoding with elevated rescore depth"
	case n.MinRecall >= 0.95:
		switch {
		case c.IndexType == IndexFlat:
			spec = rqSpec(8, 96, 2, false)
		case useRQ:
			spec = rqSpec(8, 64, 2, false)
		default:
			spec = sqSpec(96)
		}
		spec.Rationale = "high-recall target selects 8-bit encoding and moderate rescoring"
	case n.MinRecall >= 0.92:
		if useRQ && c.IndexType != IndexFlat {
			spec = rqSpec(4, 192, 3, true)
		} else if c.IndexType == IndexFlat {
			spec = rqSpec(1, 512, 4, true)
		} else {
			spec = sqSpec(128)
		}
		spec.Rationale = "balanced target trades roughly 8x payload size for deeper rescoring"
	case n.MinRecall >= 0.88:
		switch {
		case c.IndexType == IndexFlat:
			spec = bqSpec(768)
		case useRQ:
			spec = rqSpec(4, 320, 3, true)
		default:
			spec = bqSpec(512)
		}
		spec.Rationale = "compact target selects a width supported by the index type with deep rescoring"
	default:
		spec = bqSpec(768)
		if useRQ {
			spec = rqSpec(1, 768, 4, true)
		}
		spec.Rationale = "economy target selects 1-bit encoding with maximum bounded rescore depth"
	}

	if err := validateFormForConstraints(spec, c); err != nil {
		return FormSpec{}, err
	}
	return spec, nil
}

func normalizeConstraints(c Constraints) (Constraints, error) {
	if c.Dimensions < 0 {
		return c, fmt.Errorf("%w: dimensions must be non-negative", ErrUnsupportedCombination)
	}
	if c.VectorCount < 0 {
		return c, fmt.Errorf("%w: vectorCount must be non-negative", ErrUnsupportedCombination)
	}
	c.Distance = strings.ToLower(strings.TrimSpace(c.Distance))
	if c.Distance == "" {
		c.Distance = "cosine"
	}
	c.IndexType = strings.ToLower(strings.TrimSpace(c.IndexType))
	if c.IndexType == "" {
		c.IndexType = IndexHNSW
	}
	switch c.IndexType {
	case IndexHNSW, IndexFlat, IndexDynamic:
	default:
		return c, fmt.Errorf("%w: unknown index type %q", ErrUnsupportedCombination, c.IndexType)
	}
	return c, nil
}

func rqAllowed(distance string) bool {
	switch distance {
	case "cosine", "dot", "l2-squared", "l2":
		return true
	default:
		return false
	}
}

func rqSpec(bits, rescore int, factor float64, centering bool) FormSpec {
	f := FormRQ8
	switch bits {
	case 4:
		f = FormRQ4
	case 1:
		f = FormRQ1
	}
	ratio := 4.0
	if bits > 0 {
		ratio = 32 / float64(bits)
	}
	return FormSpec{
		PolicyVersion:             PolicyVersion,
		Form:                      f,
		Codec:                     CodecRQ,
		Bits:                      bits,
		RescoreLimit:              rescore,
		RescoreFactor:             factor,
		Centering:                 centering,
		TrainingLimit:             10000,
		EstimatedCompressionRatio: ratio,
	}
}

func sqSpec(rescore int) FormSpec {
	return FormSpec{
		PolicyVersion:             PolicyVersion,
		Form:                      FormSQ8,
		Codec:                     CodecSQ,
		Bits:                      8,
		RescoreLimit:              rescore,
		RescoreFactor:             2,
		TrainingLimit:             100000,
		EstimatedCompressionRatio: 4,
	}
}

func bqSpec(rescore int) FormSpec {
	return FormSpec{
		PolicyVersion:             PolicyVersion,
		Form:                      FormBQ1,
		Codec:                     CodecBQ,
		Bits:                      1,
		RescoreLimit:              rescore,
		RescoreFactor:             4,
		EstimatedCompressionRatio: 32,
	}
}

func validateFormForConstraints(s FormSpec, c Constraints) error {
	if c.IndexType == IndexFlat {
		switch s.Codec {
		case CodecPQ, CodecSQ:
			return fmt.Errorf("%w: flat indexes support BQ and RQ only, cannot select %s",
				ErrUnsupportedCombination, s.Form)
		}
		if s.Codec == CodecRQ && s.Bits == 4 {
			return fmt.Errorf("%w: flat RQ supports 1 or 8 bits, not 4", ErrUnsupportedCombination)
		}
	}
	if !rqAllowed(c.Distance) && s.Codec == CodecRQ {
		return fmt.Errorf("%w: RQ does not support distance %q", ErrUnsupportedCombination, c.Distance)
	}
	if c.Dimensions > 0 && s.Codec == CodecPQ {
		if c.Dimensions < 2 {
			return fmt.Errorf("%w: PQ requires at least 2 dimensions", ErrUnsupportedCombination)
		}
		if s.PQSegments > 0 && c.Dimensions%s.PQSegments != 0 {
			return fmt.Errorf("%w: PQ segments %d must divide dimensions %d",
				ErrUnsupportedCombination, s.PQSegments, c.Dimensions)
		}
	}
	return nil
}

// ExplicitForm resolves a named operator form without consulting the target
// table. It is used for bidirectional switches and equivalence parsing of
// pre-existing compression configuration.
func ExplicitForm(form Form, constraints Constraints) (FormSpec, error) {
	c, err := normalizeConstraints(constraints)
	if err != nil {
		return FormSpec{}, err
	}
	var s FormSpec
	switch form {
	case FormUncompressed:
		s = FormSpec{PolicyVersion: PolicyVersion, Form: form, Codec: CodecNone, Bits: 32, EstimatedCompressionRatio: 1}
	case FormBQ1:
		s = bqSpec(512)
	case FormRQ1:
		s = rqSpec(1, 512, 4, true)
	case FormRQ4:
		s = rqSpec(4, 256, 3, true)
	case FormRQ8:
		s = rqSpec(8, 64, 2, false)
	case FormSQ8:
		s = sqSpec(64)
	case FormPQ:
		s = FormSpec{PolicyVersion: PolicyVersion, Form: FormPQ, Codec: CodecPQ, Bits: 8, RescoreLimit: 256, RescoreFactor: 2, PQCentroids: 256, TrainingLimit: 100000, EstimatedCompressionRatio: 16}
	default:
		return FormSpec{}, fmt.Errorf("%w: unknown form %q", ErrUnsupportedCombination, form)
	}
	if err := validateFormForConstraints(s, c); err != nil {
		return FormSpec{}, err
	}
	return s, nil
}

// EquivalentForm identifies the effective form represented by existing
// quantization booleans/widths. It gives legacy configs equivalent semantics
// without requiring a recall target.
func EquivalentForm(codec string, bits int) Form {
	switch strings.ToLower(strings.TrimSpace(codec)) {
	case "", CodecNone:
		return FormUncompressed
	case CodecBQ:
		return FormBQ1
	case CodecSQ:
		return FormSQ8
	case CodecPQ:
		return FormPQ
	case CodecRQ:
		switch bits {
		case 1:
			return FormRQ1
		case 4:
			return FormRQ4
		default:
			return FormRQ8
		}
	default:
		return Form(codec)
	}
}

// SortedProfileAliases is useful for deterministic docs/API output.
func SortedProfileAliases() []string {
	p := Profiles()
	out := make([]string, 0, len(p))
	for _, profile := range p {
		out = append(out, profile.Alias)
	}
	sort.Strings(out)
	return out
}
