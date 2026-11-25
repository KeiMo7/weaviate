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
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeBackend struct {
	mu sync.Mutex

	units    []UnitInfo
	form     FormObservation
	listErr  error
	buildErr error
	swapErr  error
	rbErr    error

	buildCalls    []string
	swapCalls     []string
	rollbackCalls []string
	cleanupCalls  []string
	checkpoints   map[string]string
	buildHook     func(req BuildRequest, report ProgressFunc) error
}

func newFakeBackend() *fakeBackend {
	return &fakeBackend{
		units: []UnitInfo{
			{ID: "shard-0/r0", Shard: "shard-0", Replica: "r0", Node: "n0"},
			{ID: "shard-1/r0", Shard: "shard-1", Replica: "r0", Node: "n1"},
		},
		form:        FormObservation{Codec: CodecNone, Bits: 32, Form: FormSpec{Codec: CodecNone, Bits: 32}},
		checkpoints: map[string]string{},
	}
}

func (b *fakeBackend) ListUnits(context.Context, string) ([]UnitInfo, error) {
	if b.listErr != nil {
		return nil, b.listErr
	}
	return b.units, nil
}

func (b *fakeBackend) EffectiveForm(context.Context, string, string) (FormObservation, error) {
	return b.form, nil
}

func (b *fakeBackend) BuildUnit(_ context.Context, req BuildRequest, report ProgressFunc) error {
	b.mu.Lock()
	b.buildCalls = append(b.buildCalls, req.Unit.ID)
	if cp := b.checkpoints[req.Unit.ID]; cp != "" {
		req.Checkpoint = cp
	}
	hook := b.buildHook
	err := b.buildErr
	b.mu.Unlock()
	if report != nil {
		report(5, 10, "half")
	}
	if hook != nil {
		if err := hook(req, report); err != nil {
			return err
		}
	}
	if err != nil {
		return err
	}
	if report != nil {
		report(10, 10, "done")
	}
	return nil
}

func (b *fakeBackend) SwapUnit(_ context.Context, req UnitRequest) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.swapCalls = append(b.swapCalls, req.Unit.ID)
	return b.swapErr
}

func (b *fakeBackend) RollbackUnit(_ context.Context, req UnitRequest) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.rollbackCalls = append(b.rollbackCalls, req.Unit.ID)
	return b.rbErr
}

func (b *fakeBackend) CleanupUnit(_ context.Context, req UnitRequest) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cleanupCalls = append(b.cleanupCalls, req.Unit.ID)
	return nil
}

func newTestManager(t *testing.T, b Backend, s Store) *Manager {
	t.Helper()
	if s == nil {
		s = NewInMemoryStore()
	}
	return NewManager(ManagerConfig{
		Store: s, Backend: b, Metrics: NewMetrics(nil),
		Logger: discardLogger(), Now: func() time.Time { return time.Unix(1700000000, 0) },
	})
}

func TestManagerSuccessfulSwitchAggregatesUnits(t *testing.T) {
	b := newFakeBackend()
	m := newTestManager(t, b, nil)
	to, err := ExplicitForm(FormRQ8, Constraints{Distance: "cosine"})
	require.NoError(t, err)

	r, err := m.RunSwitch(context.Background(), StartSwitchInput{
		Collection: "Article", Vector: "title", To: &to, ID: "sw-1",
	})
	require.NoError(t, err)
	assert.Equal(t, PhaseCompleted, r.Phase)
	assert.Len(t, b.buildCalls, 2)
	assert.Len(t, b.swapCalls, 2)
	assert.Len(t, b.cleanupCalls, 2)
	assert.Empty(t, b.rollbackCalls)

	stored, err := m.GetSwitch("sw-1")
	require.NoError(t, err)
	assert.Equal(t, PhaseCompleted, stored.Phase)
	p := stored.Progress()
	assert.Equal(t, 2, p.UnitsTotal)
	assert.Equal(t, 2, p.UnitsComplete)
	assert.InDelta(t, 1, p.FractionComplete, 0.001)
}

func TestManagerOneActiveSwitchPerCollection(t *testing.T) {
	b := newFakeBackend()
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	b.buildHook = func(req BuildRequest, _ ProgressFunc) error {
		if req.Unit.ID == b.units[0].ID {
			started <- struct{}{}
			<-release
		}
		return nil
	}
	m := newTestManager(t, b, nil)
	to, _ := ExplicitForm(FormBQ1, Constraints{})

	first, err := m.StartSwitch(context.Background(), StartSwitchInput{Collection: "C", Vector: "v", To: &to})
	require.NoError(t, err)
	<-started

	_, err = m.RunSwitch(context.Background(), StartSwitchInput{Collection: "C", Vector: "v2", To: &to})
	require.ErrorIs(t, err, ErrActiveSwitch)

	close(release)
	final, err := m.Wait(context.Background(), first.ID)
	require.NoError(t, err)
	assert.Equal(t, PhaseCompleted, final.Phase)
}

func TestManagerBuildFailureRollsBackAndOldFormRemains(t *testing.T) {
	b := newFakeBackend()
	b.buildErr = errors.New("simulated disk write failure: no space left")
	m := newTestManager(t, b, nil)
	to, _ := ExplicitForm(FormRQ4, Constraints{Distance: "cosine"})

	r, err := m.RunSwitch(context.Background(), StartSwitchInput{Collection: "C", Vector: "v", To: &to, ID: "sw-fail"})
	require.Error(t, err)
	require.NotNil(t, r)
	assert.Equal(t, PhaseRolledBack, r.Phase)
	assert.NotEmpty(t, r.LastError)
	assert.Len(t, b.rollbackCalls, 2, "every unit must be rolled back")
	assert.Len(t, b.cleanupCalls, 2)
	assert.Equal(t, CodecNone, r.From.Codec)
}

func TestManagerCollectionGoneAbortsAndDoesNotResurrect(t *testing.T) {
	b := newFakeBackend()
	b.buildErr = ErrCollectionGone
	store := NewInMemoryStore()
	m := newTestManager(t, b, store)
	to, _ := ExplicitForm(FormBQ1, Constraints{})

	r, err := m.RunSwitch(context.Background(), StartSwitchInput{Collection: "Deleted", Vector: "v", To: &to})
	require.Error(t, err)
	assert.Equal(t, PhaseAborted, r.Phase)

	rec, err := store.GetSwitch(r.ID)
	require.NoError(t, err)
	assert.True(t, rec.Phase.Terminal())

	resumed, err := m.Recover(context.Background())
	require.NoError(t, err)
	assert.Empty(t, resumed, "terminal aborted job must not restart")
}

func TestFileStoreRecoverSkipsCompletedUnits(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(filepath.Join(dir, "quant"))
	require.NoError(t, err)
	now := time.Unix(1700000100, 0)
	to, _ := ExplicitForm(FormSQ8, Constraints{})
	rec := &SwitchRecord{
		ID: "sw-restart", Collection: "C", Vector: "v", From: FormSpec{Codec: CodecNone}, To: to,
		Phase: PhaseBuilding, CreatedAt: now, UpdatedAt: now,
		Units: map[string]*UnitState{
			"u1": {UnitID: "u1", Phase: PhaseCompleted, Checkpoint: "cp-complete"},
			"u2": {UnitID: "u2", Phase: PhaseBuilding, Checkpoint: "cp-half"},
		},
	}
	require.NoError(t, store.SaveSwitch(rec))

	b := newFakeBackend()
	b.units = []UnitInfo{{ID: "u1", Shard: "s1"}, {ID: "u2", Shard: "s2"}}
	m := newTestManager(t, b, store)
	resumed, err := m.Recover(context.Background())
	require.NoError(t, err)
	require.Len(t, resumed, 1)

	final, err := m.Wait(context.Background(), "sw-restart")
	require.NoError(t, err)
	assert.Equal(t, PhaseCompleted, final.Phase)
	b.mu.Lock()
	defer b.mu.Unlock()
	assert.Equal(t, []string{"u2"}, b.buildCalls, "completed unit must not be rebuilt")
	assert.ElementsMatch(t, []string{"u1", "u2"}, b.swapCalls)
}

func TestManagerRejectsDurableActiveRecordWithoutInMemoryState(t *testing.T) {
	store := NewInMemoryStore()
	to, _ := ExplicitForm(FormBQ1, Constraints{})
	now := time.Unix(1700000200, 0)
	require.NoError(t, store.SaveSwitch(&SwitchRecord{
		ID: "durable-active", Collection: "C", Vector: "v", To: to,
		Phase: PhaseBuilding, CreatedAt: now, UpdatedAt: now,
		Units: map[string]*UnitState{"u": {UnitID: "u", Phase: PhaseBuilding}},
	}))
	b := newFakeBackend()
	m := newTestManager(t, b, store)

	_, err := m.RunSwitch(context.Background(), StartSwitchInput{Collection: "C", Vector: "v2", To: &to})
	require.ErrorIs(t, err, ErrActiveSwitch)
}

func TestCancelSwitch(t *testing.T) {
	b := newFakeBackend()
	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	b.buildHook = func(req BuildRequest, _ ProgressFunc) error {
		if req.Unit.ID == b.units[0].ID {
			entered <- struct{}{}
			<-release
			return context.Canceled
		}
		return nil
	}
	m := newTestManager(t, b, nil)
	to, _ := ExplicitForm(FormBQ1, Constraints{})
	rec, err := m.StartSwitch(context.Background(), StartSwitchInput{Collection: "C", Vector: "v", To: &to})
	require.NoError(t, err)
	<-entered
	require.NoError(t, m.Cancel(rec.ID))
	close(release)
	final, err := m.Wait(context.Background(), rec.ID)
	require.NoError(t, err)
	assert.Equal(t, PhaseCanceled, final.Phase)
}
