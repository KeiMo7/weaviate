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
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	enterrors "github.com/weaviate/weaviate/entities/errors"
)

// UnitInfo identifies one shard or replica that must switch.
type UnitInfo struct {
	ID      string `json:"id"`
	Shard   string `json:"shard"`
	Replica string `json:"replica,omitempty"`
	Node    string `json:"node,omitempty"`
}

// BuildRequest is passed to a UnitSwitchBackend implementation.
type BuildRequest struct {
	SwitchID   string
	Collection string
	Vector     string
	From       FormSpec
	To         FormSpec
	Unit       UnitInfo
	Checkpoint string
}

// UnitRequest identifies one unit during swap/rollback/cleanup.
type UnitRequest struct {
	SwitchID   string
	Collection string
	Vector     string
	From       FormSpec
	To         FormSpec
	Unit       UnitInfo
	Checkpoint string
}

// ProgressFunc persists incremental progress. Implementations should not scan
// user data merely to produce a percentage.
type ProgressFunc func(processed, totalHint int64, checkpoint string)

// Backend is the storage/vector-index port. The repository adapter owns the
// atomic shadow-build/catch-up/swap mechanics; Manager owns durable state,
// exclusivity, aggregation and failure policy.
type Backend interface {
	ListUnits(ctx context.Context, collection string) ([]UnitInfo, error)
	EffectiveForm(ctx context.Context, collection, vector string) (FormObservation, error)
	BuildUnit(ctx context.Context, req BuildRequest, report ProgressFunc) error
	SwapUnit(ctx context.Context, req UnitRequest) error
	RollbackUnit(ctx context.Context, req UnitRequest) error
	CleanupUnit(ctx context.Context, req UnitRequest) error
}

// ManagerConfig constructs a Manager.
type ManagerConfig struct {
	Store   Store
	Backend Backend
	Metrics *Metrics
	Limits  Limits
	Logger  logrus.FieldLogger
	Now     func() time.Time
	NewID   func() string
}

// Manager runs at most one switch per collection.
type Manager struct {
	store   Store
	backend Backend
	metrics *Metrics
	limits  Limits
	logger  logrus.FieldLogger
	now     func() time.Time
	newID   func() string

	mu      sync.Mutex
	active  map[string]string
	cancels map[string]context.CancelFunc
	dones   map[string]chan struct{}

	// recordMu serializes read-modify-write updates to one stored record from
	// concurrent unit workers. It never surrounds backend I/O.
	recordMu sync.Mutex
}

// NewManager wires a manager.
func NewManager(cfg ManagerConfig) *Manager {
	if cfg.Store == nil {
		cfg.Store = NewInMemoryStore()
	}
	if cfg.Metrics == nil {
		cfg.Metrics = NewMetrics(nil)
	}
	if cfg.Logger == nil {
		l := logrus.New()
		l.SetLevel(logrus.WarnLevel)
		cfg.Logger = l
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.NewID == nil {
		cfg.NewID = uuid.NewString
	}
	return &Manager{
		store:   cfg.Store,
		backend: cfg.Backend,
		metrics: cfg.Metrics,
		limits:  cfg.Limits.withDefaults(),
		logger:  cfg.Logger.WithField("component", "quantization_switch_manager"),
		now:     cfg.Now,
		newID:   cfg.NewID,
		active:  map[string]string{},
		cancels: map[string]context.CancelFunc{},
		dones:   map[string]chan struct{}{},
	}
}

// StartSwitchInput starts a bidirectional form switch. Target and To are
// mutually exclusive; Target resolves through the single policy table.
type StartSwitchInput struct {
	Collection  string
	Vector      string
	Target      *Target
	To          *FormSpec
	Constraints Constraints
	ID          string
}

// StartSwitch creates the durable record and runs the switch in background.
func (m *Manager) StartSwitch(parent context.Context, in StartSwitchInput) (*SwitchRecord, error) {
	rec, err := m.prepare(in)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.WithoutCancel(parent))
	done := make(chan struct{})
	m.mu.Lock()
	m.cancels[rec.ID] = cancel
	m.dones[rec.ID] = done
	m.mu.Unlock()

	m.metrics.SwitchesStarted.WithLabelValues(rec.Collection, rec.Vector, string(rec.To.Form)).Inc()
	m.metrics.SwitchActive.WithLabelValues(rec.Collection, rec.Vector).Set(1)
	enterrors.GoWrapper(func() {
		defer close(done)
		defer cancel()
		m.run(ctx, rec.ID)
	}, m.logger)
	return m.GetSwitch(rec.ID)
}

// RunSwitch runs synchronously; it is useful for tests and CLI jobs.
func (m *Manager) RunSwitch(ctx context.Context, in StartSwitchInput) (*SwitchRecord, error) {
	rec, err := m.prepare(in)
	if err != nil {
		return nil, err
	}
	done := make(chan struct{})
	m.mu.Lock()
	m.dones[rec.ID] = done
	m.mu.Unlock()
	defer close(done)
	return m.run(ctx, rec.ID)
}

func (m *Manager) prepare(in StartSwitchInput) (*SwitchRecord, error) {
	if in.Collection == "" {
		return nil, fmt.Errorf("%w: collection is required", ErrInvalidTarget)
	}
	if m.backend == nil {
		return nil, errors.New("quantization manager requires a Backend")
	}

	var to FormSpec
	var target *Target
	if in.Target != nil && in.To != nil {
		return nil, fmt.Errorf("%w: Target and To are mutually exclusive", ErrInvalidTarget)
	}
	if in.Target != nil {
		resolved, err := Resolve(*in.Target, in.Constraints)
		if err != nil {
			return nil, err
		}
		to = resolved
		t := *in.Target
		target = &t
	} else if in.To != nil {
		to = *in.To
		if to.PolicyVersion == "" {
			to.PolicyVersion = PolicyVersion
		}
		if to.Codec == "" {
			return nil, fmt.Errorf("%w: To.codec is required", ErrInvalidTarget)
		}
	} else {
		return nil, fmt.Errorf("%w: either Target or To is required", ErrInvalidTarget)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	fromObs, err := m.backend.EffectiveForm(ctx, in.Collection, in.Vector)
	if err != nil && !errors.Is(err, ErrCollectionGone) {
		return nil, fmt.Errorf("observe effective form: %w", err)
	}
	from := fromObs.Form
	if from.Codec == "" {
		from = FormSpec{PolicyVersion: PolicyVersion, Form: FormUncompressed, Codec: CodecNone, Bits: 32}
	}
	units, err := m.backend.ListUnits(ctx, in.Collection)
	if err != nil {
		return nil, fmt.Errorf("list switch units: %w", err)
	}
	if len(units) == 0 {
		return nil, fmt.Errorf("%w: collection %q has no shard/replica units", ErrCollectionGone, in.Collection)
	}
	if len(units) > m.limits.MaxSwitchUnitsPerRun {
		return nil, fmt.Errorf("switch units %d exceed limit %d", len(units), m.limits.MaxSwitchUnitsPerRun)
	}

	existing, listErr := m.store.ListSwitches()
	if listErr != nil {
		return nil, fmt.Errorf("inspect existing quantization switches: %w", listErr)
	}
	m.mu.Lock()
	if id, ok := m.active[in.Collection]; ok {
		m.mu.Unlock()
		return nil, fmt.Errorf("%w: collection %q already has switch %s", ErrActiveSwitch, in.Collection, id)
	}
	for _, r := range existing {
		if r.Collection == in.Collection && !r.Phase.Terminal() {
			m.mu.Unlock()
			return nil, fmt.Errorf("%w: collection %q already has durable switch %s in phase %s",
				ErrActiveSwitch, in.Collection, r.ID, r.Phase)
		}
	}
	id := in.ID
	if id == "" {
		id = m.newID()
	}
	now := m.now()
	rec := &SwitchRecord{
		ID:             id,
		Collection:     in.Collection,
		Vector:         in.Vector,
		DeclaredTarget: target,
		From:           from,
		To:             to,
		Phase:          PhasePending,
		Units:          map[string]*UnitState{},
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	for _, u := range units {
		rec.Units[u.ID] = &UnitState{
			UnitID:    u.ID,
			Shard:     u.Shard,
			Replica:   u.Replica,
			Node:      u.Node,
			Phase:     PhasePending,
			UpdatedAt: now,
		}
	}
	m.active[in.Collection] = id
	m.mu.Unlock()

	if err := m.store.SaveSwitch(rec); err != nil {
		m.mu.Lock()
		delete(m.active, in.Collection)
		m.mu.Unlock()
		return nil, err
	}
	return rec, nil
}

// GetSwitch returns a fresh stored record.
func (m *Manager) GetSwitch(id string) (*SwitchRecord, error) {
	return m.store.GetSwitch(id)
}

// ListSwitches lists all persisted records.
func (m *Manager) ListSwitches() ([]*SwitchRecord, error) {
	return m.store.ListSwitches()
}

// Progress returns aggregated progress.
func (m *Manager) Progress(id string) (Progress, error) {
	r, err := m.store.GetSwitch(id)
	if err != nil {
		return Progress{}, err
	}
	return r.Progress(), nil
}

// Cancel requests cancellation. Swapping is an atomic barrier and cannot be
// interrupted once entered; callers receive ErrIllegalTransition in that case.
func (m *Manager) Cancel(id string) error {
	r, err := m.store.GetSwitch(id)
	if err != nil {
		return err
	}
	if r.Phase == PhaseSwapping || r.Phase.Terminal() {
		return fmt.Errorf("%w: cannot cancel switch %s in phase %s", ErrIllegalTransition, id, r.Phase)
	}
	m.mu.Lock()
	cancel := m.cancels[id]
	m.mu.Unlock()
	if cancel == nil {
		return ErrNoActiveSwitch
	}
	cancel()
	return nil
}

// Wait blocks until the background run finishes or ctx ends.
func (m *Manager) Wait(ctx context.Context, id string) (*SwitchRecord, error) {
	m.mu.Lock()
	done := m.dones[id]
	m.mu.Unlock()
	if done == nil {
		return m.store.GetSwitch(id)
	}
	select {
	case <-done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return m.store.GetSwitch(id)
}

// Recover reloads nonterminal records and continues them. Completed units are
// skipped because their durable UnitState is PhaseCompleted.
func (m *Manager) Recover(ctx context.Context) ([]*SwitchRecord, error) {
	records, err := m.store.ListSwitches()
	if err != nil {
		return nil, err
	}
	var resumed []*SwitchRecord
	for _, r := range records {
		if r.Phase.Terminal() {
			continue
		}
		// Collection deletion wins over stale on-disk work.
		_, listErr := m.backend.ListUnits(ctx, r.Collection)
		if listErr != nil && errors.Is(listErr, ErrCollectionGone) {
			_ = m.transitionAndSave(r, PhaseAborted, r.LastError)
			m.releaseActive(r.Collection, r.ID)
			continue
		}
		m.mu.Lock()
		m.active[r.Collection] = r.ID
		runCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
		done := make(chan struct{})
		m.cancels[r.ID] = cancel
		m.dones[r.ID] = done
		m.mu.Unlock()
		resumed = append(resumed, r)
		m.metrics.SwitchesRecovered.WithLabelValues(r.Collection, r.Vector, string(r.To.Form)).Inc()
		enterrors.GoWrapper(func() {
			defer close(done)
			defer cancel()
			m.run(runCtx, r.ID)
		}, m.logger)
	}
	return resumed, nil
}

func (m *Manager) run(ctx context.Context, id string) (*SwitchRecord, error) {
	r, err := m.store.GetSwitch(id)
	if err != nil {
		return nil, err
	}
	startPhase := r.Phase
	if startPhase == PhasePending {
		if err := m.transitionAndSave(r, PhasePlanning, ""); err != nil {
			return m.finalizeError(r, err)
		}
	}

	units := r.SortedUnits()
	infos := make([]UnitInfo, 0, len(units))
	for _, u := range units {
		infos = append(infos, UnitInfo{ID: u.UnitID, Shard: u.Shard, Replica: u.Replica, Node: u.Node})
	}

	if r.Phase == PhasePlanning || r.Phase == PhasePending {
		if err := m.transitionAndSave(r, PhaseBuilding, ""); err != nil {
			return m.finalizeError(r, err)
		}
	}

	buildErr := m.runUnits(ctx, r, infos, PhaseBuilding, func(unit UnitInfo, checkpoint string, report ProgressFunc) error {
		return m.backend.BuildUnit(ctx, BuildRequest{
			SwitchID: id, Collection: r.Collection, Vector: r.Vector,
			From: r.From, To: r.To, Unit: unit, Checkpoint: checkpoint,
		}, report)
	})
	if buildErr != nil {
		return m.rollback(ctx, r, infos, buildErr)
	}
	if err := ctx.Err(); err != nil {
		return m.rollback(ctx, r, infos, fmt.Errorf("%w: %w", ErrEvaluationCanceled, err))
	}

	if err := m.transitionAndSave(r, PhaseSwapping, ""); err != nil {
		return m.finalizeError(r, err)
	}
	swapErr := m.runUnits(ctx, r, infos, PhaseSwapping, func(unit UnitInfo, checkpoint string, _ ProgressFunc) error {
		return m.backend.SwapUnit(ctx, UnitRequest{
			SwitchID: id, Collection: r.Collection, Vector: r.Vector,
			From: r.From, To: r.To, Unit: unit, Checkpoint: checkpoint,
		})
	})
	if swapErr != nil {
		return m.rollback(ctx, r, infos, swapErr)
	}

	// Release old/temporary encoded resources. A cleanup failure is explicit in
	// the record and metrics; the form switch itself is durably committed.
	for _, u := range infos {
		if err := m.backend.CleanupUnit(ctx, UnitRequest{
			SwitchID: id, Collection: r.Collection, Vector: r.Vector,
			From: r.From, To: r.To, Unit: u,
		}); err != nil && !errors.Is(err, ErrCollectionGone) {
			r, _ = m.store.GetSwitch(id)
			r.LastError = fmt.Sprintf("cleanup unit %s: %v", u.ID, err)
			r.FailureCount++
			_ = m.store.SaveSwitch(r)
			m.metrics.SwitchFailures.WithLabelValues(r.Collection, r.Vector, "cleanup").Inc()
			m.logger.Errorf("quantization cleanup failed switch=%s unit=%s: %v", id, u.ID, err)
		}
	}

	r, err = m.store.GetSwitch(id)
	if err != nil {
		return nil, err
	}
	if err := m.transitionAndSave(r, PhaseCompleted, ""); err != nil {
		return m.finalizeError(r, err)
	}
	m.metrics.SwitchesCompleted.WithLabelValues(r.Collection, r.Vector, string(r.To.Form)).Inc()
	m.metrics.SwitchActive.WithLabelValues(r.Collection, r.Vector).Set(0)
	m.releaseActive(r.Collection, id)
	return r, nil
}

type unitStageFunc func(unit UnitInfo, checkpoint string, report ProgressFunc) error

func (m *Manager) runUnits(ctx context.Context, r *SwitchRecord, infos []UnitInfo, phase Phase, fn unitStageFunc) error {
	workers := m.limits.MaxEvaluationWorkers
	if workers <= 0 || workers > len(infos) {
		workers = len(infos)
	}
	if workers <= 0 {
		workers = 1
	}
	jobs := make(chan UnitInfo)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	setErr := func(err error) {
		mu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		mu.Unlock()
	}

	for w := 0; w < workers; w++ {
		wg.Add(1)
		enterrors.GoWrapper(func() {
			defer wg.Done()
			for u := range jobs {
				if ctx.Err() != nil {
					setErr(ctx.Err())
					continue
				}
				var checkpoint string
				skip := false
				if _, err := m.mutateRecord(r.ID, func(latest *SwitchRecord) {
					us := latest.Units[u.ID]
					if us != nil && us.Phase == PhaseCompleted && phase != PhaseSwapping {
						skip = true
						return
					}
					if us == nil {
						us = &UnitState{UnitID: u.ID, Shard: u.Shard, Replica: u.Replica, Node: u.Node}
						latest.Units[u.ID] = us
					}
					checkpoint = us.Checkpoint
					us.Phase = phase
					us.Attempts++
					us.UpdatedAt = m.now()
				}); err != nil {
					setErr(err)
					continue
				}
				if skip {
					continue
				}

				unitID := u.ID
				report := func(processed, total int64, checkpoint string) {
					latest, err := m.mutateRecord(r.ID, func(latest *SwitchRecord) {
						st := latest.Units[unitID]
						if st == nil {
							st = &UnitState{UnitID: unitID}
							latest.Units[unitID] = st
						}
						st.Processed = processed
						st.TotalHint = total
						if checkpoint != "" {
							st.Checkpoint = checkpoint
						}
						st.UpdatedAt = m.now()
					})
					if err != nil {
						return
					}
					m.metrics.RecordProgress(latest.Progress())
				}
				runErr := fn(u, checkpoint, report)
				latest, mutateErr := m.mutateRecord(r.ID, func(latest *SwitchRecord) {
					st := latest.Units[u.ID]
					if st == nil {
						st = &UnitState{UnitID: u.ID}
						latest.Units[u.ID] = st
					}
					if runErr != nil {
						st.Phase = PhaseFailed
						st.LastError = runErr.Error()
						latest.FailureCount++
						latest.LastError = runErr.Error()
						return
					}
					st.Phase = PhaseCompleted
					st.LastError = ""
					st.UpdatedAt = m.now()
				})
				if mutateErr != nil {
					setErr(mutateErr)
					continue
				}
				if runErr != nil {
					m.metrics.SwitchFailures.WithLabelValues(latest.Collection, latest.Vector, string(phase)).Inc()
					setErr(fmt.Errorf("unit %s: %w", u.ID, runErr))
				}
			}
		}, m.logger)
	}

feed:
	for _, u := range infos {
		select {
		case jobs <- u:
		case <-ctx.Done():
			break feed
		}
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}

func (m *Manager) rollback(ctx context.Context, r *SwitchRecord, infos []UnitInfo, cause error) (*SwitchRecord, error) {
	latest, _ := m.store.GetSwitch(r.ID)
	if latest != nil {
		r = latest
	}
	gone := errors.Is(cause, ErrCollectionGone)
	if errors.Is(cause, context.Canceled) {
		_ = m.transitionAndSave(r, PhaseCanceled, cause.Error())
		m.metrics.SwitchesCanceled.WithLabelValues(r.Collection, r.Vector, string(r.To.Form)).Inc()
		m.metrics.SwitchActive.WithLabelValues(r.Collection, r.Vector).Set(0)
		m.releaseActive(r.Collection, r.ID)
		return r, cause
	}
	if gone {
		if err := m.transitionAndSave(r, PhaseAborted, cause.Error()); err != nil {
			return m.finalizeError(r, err)
		}
		_ = m.transitionAndSave(r, PhaseAborted, cause.Error())
		for _, u := range infos {
			_ = m.backend.CleanupUnit(context.WithoutCancel(ctx), UnitRequest{
				SwitchID: r.ID, Collection: r.Collection, Vector: r.Vector, From: r.From, To: r.To, Unit: u,
			})
		}
		m.releaseActive(r.Collection, r.ID)
		return r, cause
	}

	if err := m.transitionAndSave(r, PhaseRollingBack, cause.Error()); err != nil {
		return m.finalizeError(r, err)
	}
	var rbErr error
	for _, u := range infos {
		if err := m.backend.RollbackUnit(context.WithoutCancel(ctx), UnitRequest{
			SwitchID: r.ID, Collection: r.Collection, Vector: r.Vector, From: r.From, To: r.To, Unit: u,
		}); err != nil && rbErr == nil {
			rbErr = fmt.Errorf("rollback unit %s: %w", u.ID, err)
		}
		_ = m.backend.CleanupUnit(context.WithoutCancel(ctx), UnitRequest{
			SwitchID: r.ID, Collection: r.Collection, Vector: r.Vector, From: r.From, To: r.To, Unit: u,
		})
	}
	if rbErr != nil {
		_ = m.transitionAndSave(r, PhaseFailed, errors.Join(cause, rbErr).Error())
		m.releaseActive(r.Collection, r.ID)
		return r, errors.Join(cause, rbErr)
	}
	if err := m.transitionAndSave(r, PhaseRolledBack, cause.Error()); err != nil {
		return m.finalizeError(r, err)
	}
	m.metrics.SwitchesRolledBack.WithLabelValues(r.Collection, r.Vector, string(r.To.Form)).Inc()
	m.metrics.SwitchActive.WithLabelValues(r.Collection, r.Vector).Set(0)
	m.releaseActive(r.Collection, r.ID)
	return r, cause
}

func (m *Manager) mutateRecord(id string, fn func(*SwitchRecord)) (*SwitchRecord, error) {
	m.recordMu.Lock()
	defer m.recordMu.Unlock()
	r, err := m.store.GetSwitch(id)
	if err != nil {
		return nil, err
	}
	fn(r)
	r.UpdatedAt = m.now()
	if err := m.store.SaveSwitch(r); err != nil {
		return nil, err
	}
	return r, nil
}

func (m *Manager) transitionAndSave(r *SwitchRecord, to Phase, errMsg string) error {
	if err := r.Transition(to, m.now()); err != nil {
		return err
	}
	if errMsg != "" {
		r.LastError = errMsg
	}
	return m.store.SaveSwitch(r)
}

func (m *Manager) finalizeError(r *SwitchRecord, cause error) (*SwitchRecord, error) {
	latest, _ := m.store.GetSwitch(r.ID)
	if latest != nil {
		r = latest
	}
	_ = m.transitionAndSave(r, PhaseFailed, cause.Error())
	m.metrics.SwitchFailures.WithLabelValues(r.Collection, r.Vector, "state").Inc()
	m.metrics.SwitchActive.WithLabelValues(r.Collection, r.Vector).Set(0)
	m.releaseActive(r.Collection, r.ID)
	return r, cause
}

func (m *Manager) releaseActive(collection, id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if cur := m.active[collection]; cur == id {
		delete(m.active, collection)
	}
}
