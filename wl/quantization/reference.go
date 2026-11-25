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
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// ReferenceBackend is an executable semantic model of per-unit single-form
// reads, shadow build, write-through catch-up and atomic promotion. It is not
// the production HNSW adapter; it pins the concurrency and recovery semantics
// that adapter must preserve.
type ReferenceBackend struct {
	collection string
	vector     string
	dims       int
	batchSize  int

	mu    sync.RWMutex
	units map[string]*referenceUnit
	order []UnitInfo

	// BuildGate, when non-nil, is waited on at the start of BuildUnit. It lets
	// tests write while a switch is definitively in progress.
	BuildGate    chan struct{}
	BuildStarted chan struct{}
	// FailBuildOnce makes the next BuildUnit fail exactly once.
	FailBuildOnce bool
}

type referenceUnit struct {
	mu      sync.RWMutex
	ids     []string
	raw     map[string][]float32
	current *immutableForm
	old     *immutableForm
	shadow  *immutableForm
}

type immutableForm struct {
	id    string
	spec  FormSpec
	codes map[string][]float32
}

// ReferenceHit is a search hit tagged with the exact encoding snapshot used.
type ReferenceHit struct {
	ID     string
	Dist   float64
	FormID string
	Codec  string
}

// NewReferenceBackend creates unitCount units with an uncompressed current form.
func NewReferenceBackend(collection, vector string, unitCount, dims int) *ReferenceBackend {
	if unitCount <= 0 {
		unitCount = 1
	}
	b := &ReferenceBackend{
		collection: collection,
		vector:     vector,
		dims:       dims,
		batchSize:  16,
		units:      map[string]*referenceUnit{},
	}
	now := time.Now()
	for i := 0; i < unitCount; i++ {
		u := UnitInfo{ID: fmt.Sprintf("shard-%d/r0", i), Shard: fmt.Sprintf("shard-%d", i), Replica: "r0", Node: "n0"}
		b.order = append(b.order, u)
		b.units[u.ID] = &referenceUnit{
			raw: map[string][]float32{},
			current: &immutableForm{
				id:    fmt.Sprintf("initial-none-%d", i),
				spec:  FormSpec{Codec: CodecNone, Bits: 32, Form: FormUncompressed},
				codes: map[string][]float32{},
			},
		}
	}
	_ = now
	return b
}

func (b *ReferenceBackend) ListUnits(context.Context, string) ([]UnitInfo, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return append([]UnitInfo(nil), b.order...), nil
}

func (b *ReferenceBackend) unit(id string) (*referenceUnit, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	u, ok := b.units[id]
	if !ok {
		return nil, fmt.Errorf("%w: unit %s", ErrCollectionGone, id)
	}
	return u, nil
}

// Write inserts/updates a raw vector. Writes after BuildUnit has created the
// shadow are encoded into both the serving current form and the future form, so
// they cannot become unreadable black holes.
func (b *ReferenceBackend) Write(_ context.Context, unitID, id string, v []float32) error {
	u, err := b.unit(unitID)
	if err != nil {
		return err
	}
	cp := append([]float32(nil), v...)
	u.mu.Lock()
	defer u.mu.Unlock()
	if _, exists := u.raw[id]; !exists {
		u.ids = append(u.ids, id)
	}
	u.raw[id] = cp
	if u.current != nil {
		u.current.codes[id] = encodeVector(cp, u.current.spec)
	}
	if u.shadow != nil {
		u.shadow.codes[id] = encodeVector(cp, u.shadow.spec)
	}
	return nil
}

func (b *ReferenceBackend) EffectiveForm(_ context.Context, _, _ string) (FormObservation, error) {
	infos, _ := b.ListUnits(context.Background(), b.collection)
	if len(infos) == 0 {
		return FormObservation{}, ErrCollectionGone
	}

	var spec FormSpec
	var codec string
	var bits int
	var count, encodedBytes, rawBytes int64
	for i, info := range infos {
		u, err := b.unit(info.ID)
		if err != nil {
			return FormObservation{}, err
		}
		u.mu.RLock()
		if u.current == nil {
			u.mu.RUnlock()
			continue
		}
		if i == 0 || codec == "" {
			spec = u.current.spec
			codec = u.current.spec.Codec
			bits = u.current.spec.Bits
		}
		unitCount := int64(len(u.current.codes))
		count += unitCount
		rawBytes += unitCount * int64(b.dims*4)
		unitBits := u.current.spec.Bits
		if unitBits <= 0 {
			unitBits = 32
		}
		if unitBits >= 8 {
			encodedBytes += unitCount * int64(b.dims*unitBits/8)
		} else {
			encodedBytes += unitCount * int64((b.dims*unitBits+7)/8)
		}
		u.mu.RUnlock()
	}
	ratio := 1.0
	if encodedBytes > 0 && rawBytes > 0 {
		ratio = float64(rawBytes) / float64(encodedBytes)
	} else if spec.EstimatedCompressionRatio > 0 {
		ratio = spec.EstimatedCompressionRatio
	}
	return FormObservation{
		Collection: b.collection, Vector: b.vector, UnitID: "all-units",
		Form: spec, Codec: codec, Bits: bits,
		EncodedVectorCount: count, EncodedBytes: encodedBytes,
		RawBytes: rawBytes, CompressionRatio: ratio,
		EncodingStateRecorded: true, ObservedAt: time.Now(), Source: "reference-encoded-data",
	}, nil
}

func (b *ReferenceBackend) BuildUnit(ctx context.Context, req BuildRequest, report ProgressFunc) error {
	if b.BuildStarted != nil {
		select {
		case b.BuildStarted <- struct{}{}:
		default:
		}
	}
	if b.BuildGate != nil {
		select {
		case <-b.BuildGate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	b.mu.Lock()
	if b.FailBuildOnce {
		b.FailBuildOnce = false
		b.mu.Unlock()
		return fmt.Errorf("training failed for collection=%q vector=%q stage=train: injected disk error", req.Collection, req.Vector)
	}
	b.mu.Unlock()

	u, err := b.unit(req.Unit.ID)
	if err != nil {
		return err
	}
	u.mu.Lock()
	if u.shadow == nil {
		u.shadow = &immutableForm{
			id:    req.SwitchID + ":" + req.Unit.ID + ":to",
			spec:  req.To,
			codes: map[string][]float32{},
		}
	}
	total := int64(len(u.ids))
	u.mu.Unlock()

	for start := 0; ; start += b.batchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		u.mu.RLock()
		if start >= len(u.ids) {
			processed := int64(len(u.ids))
			u.mu.RUnlock()
			if report != nil {
				report(processed, total, "complete")
			}
			return nil
		}
		end := min(start+b.batchSize, len(u.ids))
		ids := append([]string(nil), u.ids[start:end]...)
		batch := make([][]float32, len(ids))
		for i, id := range ids {
			batch[i] = u.raw[id]
		}
		shadow := u.shadow
		u.mu.RUnlock()

		for i, id := range ids {
			code := encodeVector(batch[i], req.To)
			u.mu.Lock()
			// A concurrent rollback may have discarded this shadow.
			if u.shadow == shadow {
				u.shadow.codes[id] = code
			}
			u.mu.Unlock()
		}
		if report != nil {
			report(int64(end), total, fmt.Sprintf("batch-%d", end))
		}
	}
}

func (b *ReferenceBackend) SwapUnit(_ context.Context, req UnitRequest) error {
	u, err := b.unit(req.Unit.ID)
	if err != nil {
		return err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.shadow == nil {
		return fmt.Errorf("swap unit %s: shadow form missing", req.Unit.ID)
	}
	u.old = u.current
	u.current = u.shadow
	u.shadow = nil
	return nil
}

func (b *ReferenceBackend) RollbackUnit(_ context.Context, req UnitRequest) error {
	u, err := b.unit(req.Unit.ID)
	if err != nil {
		return err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	u.shadow = nil
	// Current was never replaced before a successful SwapUnit. If swap failed
	// after replacing one unit, old lets rollback restore it.
	if u.old != nil && u.current != u.old {
		u.current = u.old
	}
	u.old = nil
	return nil
}

func (b *ReferenceBackend) CleanupUnit(_ context.Context, req UnitRequest) error {
	u, err := b.unit(req.Unit.ID)
	if err != nil {
		return err
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	u.old = nil
	u.shadow = nil
	return nil
}

// SearchUnit searches one unit and tags every hit with one immutable form ID.
func (b *ReferenceBackend) SearchUnit(_ context.Context, unitID string, q []float32, k int) ([]ReferenceHit, error) {
	u, err := b.unit(unitID)
	if err != nil {
		return nil, err
	}
	u.mu.RLock()
	form := u.current
	u.mu.RUnlock()
	if form == nil {
		return nil, ErrSwitchUnavailable
	}
	hits := make([]ReferenceHit, 0, len(form.codes))
	for id, code := range form.codes {
		hits = append(hits, ReferenceHit{ID: id, Dist: SquaredL2Distance(q, code), FormID: form.id, Codec: form.spec.Codec})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Dist < hits[j].Dist })
	if k > 0 && len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
}

// CollectionSearch fans out across units. It refuses with ErrSwitchUnavailable
// when units do not all advertise the same codec/width generation, preventing a
// query from blending old and new encodings across shards.
func (b *ReferenceBackend) CollectionSearch(ctx context.Context, q []float32, k int) ([]ReferenceHit, string, error) {
	infos, err := b.ListUnits(ctx, b.collection)
	if err != nil {
		return nil, "", err
	}
	formSignatures := map[string]struct{}{}
	var all []ReferenceHit
	for _, info := range infos {
		u, err := b.unit(info.ID)
		if err != nil {
			return nil, "", err
		}
		u.mu.RLock()
		advertisedID := ""
		signature := ""
		if u.current != nil {
			advertisedID = u.current.id
			signature = u.current.spec.Codec + ":" + fmt.Sprint(u.current.spec.Bits)
		}
		u.mu.RUnlock()
		formSignatures[signature] = struct{}{}

		hits, err := b.SearchUnit(ctx, info.ID, q, k)
		if err != nil {
			return nil, "", err
		}
		for _, h := range hits {
			if h.FormID != advertisedID {
				return nil, "", fmt.Errorf("%w: search crossed form snapshots", ErrSwitchUnavailable)
			}
		}
		all = append(all, hits...)
	}
	if len(formSignatures) != 1 {
		return nil, "", fmt.Errorf("%w: units advertise %d form widths during barrier", ErrSwitchUnavailable, len(formSignatures))
	}
	signature := ""
	for s := range formSignatures {
		signature = s
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Dist < all[j].Dist })
	if k > 0 && len(all) > k*len(infos) {
		all = all[:k*len(infos)]
	}
	return all, signature, nil
}

func encodeVector(v []float32, spec FormSpec) []float32 {
	if len(v) == 0 || spec.Codec == CodecNone || spec.Bits >= 32 || spec.Bits == 0 {
		return append([]float32(nil), v...)
	}
	out := make([]float32, len(v))
	switch spec.Bits {
	case 1:
		for i, x := range v {
			if math.Signbit(float64(x)) {
				out[i] = -1
			} else {
				out[i] = 1
			}
		}
	case 4:
		const levels = 7
		for i, x := range v {
			out[i] = float32(math.Round(float64(x)*levels) / levels)
		}
	default:
		const levels = 127
		for i, x := range v {
			out[i] = float32(math.Round(float64(x)*levels) / levels)
		}
	}
	return out
}
