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
)

// ErrNotFound marks a missing declaration/evaluation/switch.
var ErrNotFound = errors.New("quantization resource not found")

// Declaration is the collection/vector-scoped recall target. The resolved form
// is stored as well as returned so restarts read the same policy version.
type Declaration struct {
	Collection  string      `json:"collection"`
	Vector      string      `json:"vector"`
	Target      Target      `json:"target"`
	Constraints Constraints `json:"constraints"`
	Resolved    FormSpec    `json:"resolved"`
	CreatedAt   time.Time   `json:"createdAt"`
	UpdatedAt   time.Time   `json:"updatedAt"`
}

// DeclarationStore persists declared targets.
type DeclarationStore interface {
	SaveDeclaration(*Declaration) error
	GetDeclaration(collection, vector string) (*Declaration, error)
	ListDeclarations() ([]*Declaration, error)
}

// InMemoryDeclarationStore is the default declaration store.
type InMemoryDeclarationStore struct {
	mu      sync.RWMutex
	records map[string]*Declaration
	now     func() time.Time
}

func NewInMemoryDeclarationStore() *InMemoryDeclarationStore {
	return &InMemoryDeclarationStore{records: map[string]*Declaration{}, now: time.Now}
}

func declarationKey(collection, vector string) string {
	return collection + "\x00" + vector
}

func (s *InMemoryDeclarationStore) SaveDeclaration(d *Declaration) error {
	if d == nil || d.Collection == "" {
		return errors.New("declaration requires collection")
	}
	if err := d.Target.Validate(); err != nil {
		return err
	}
	if d.Resolved.Codec == "" {
		resolved, err := Resolve(d.Target, d.Constraints)
		if err != nil {
			return err
		}
		d.Resolved = resolved
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *d
	s.records[declarationKey(d.Collection, d.Vector)] = &cp
	return nil
}

func (s *InMemoryDeclarationStore) GetDeclaration(collection, vector string) (*Declaration, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	d, ok := s.records[declarationKey(collection, vector)]
	if !ok {
		return nil, fmt.Errorf("%w: target for collection=%q vector=%q", ErrNotFound, collection, vector)
	}
	cp := *d
	return &cp, nil
}

func (s *InMemoryDeclarationStore) ListDeclarations() ([]*Declaration, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Declaration, 0, len(s.records))
	for _, d := range s.records {
		cp := *d
		out = append(out, &cp)
	}
	return out, nil
}

// Status is the management-plane read model. It deliberately separates the
// declared/echoed form from the form observed in encoded data.
type Status struct {
	Collection   string          `json:"collection"`
	Vector       string          `json:"vector"`
	Declaration  *Declaration    `json:"declaration,omitempty"`
	Observed     FormObservation `json:"observed"`
	Drift        Drift           `json:"drift"`
	ActiveSwitch *SwitchRecord   `json:"activeSwitch,omitempty"`
	Progress     *Progress       `json:"progress,omitempty"`
	Metrics      StatusVolume    `json:"metrics"`
}

// StatusVolume reports measured encoding volume/compression.
type StatusVolume struct {
	EncodedBytes     int64   `json:"encodedBytes"`
	RawBytes         int64   `json:"rawBytes"`
	CompressionRatio float64 `json:"compressionRatio"`
}

// Service is the management facade used by HTTP handlers.
type Service struct {
	Declarations DeclarationStore
	Manager      *Manager
	Backend      Backend
	Evaluations  *EvaluationRegistry
	Metrics      *Metrics
}

// NewService fills optional in-memory stores and metrics.
func NewService(decls DeclarationStore, manager *Manager, backend Backend, metrics *Metrics) *Service {
	if decls == nil {
		decls = NewInMemoryDeclarationStore()
	}
	if metrics == nil {
		metrics = NewMetrics(nil)
	}
	if manager == nil {
		manager = NewManager(ManagerConfig{Backend: backend, Metrics: metrics})
	}
	return &Service{
		Declarations: decls,
		Manager:      manager,
		Backend:      backend,
		Evaluations:  NewEvaluationRegistry(nil, metrics),
		Metrics:      metrics,
	}
}

// WithEvaluationRunner attaches the storage adapter used for evaluations.
func (s *Service) WithEvaluationRunner(runner EvaluationRunner) *Service {
	s.Evaluations = NewEvaluationRegistry(runner, s.Metrics)
	return s
}

// DeclareTarget validates, resolves exactly once, and persists a target.
func (s *Service) DeclareTarget(_ context.Context, collection, vector string, target Target, constraints Constraints) (*Declaration, error) {
	normalized, err := target.Normalize()
	if err != nil {
		return nil, err
	}
	resolved, err := Resolve(normalized, constraints)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	existing, err := s.Declarations.GetDeclaration(collection, vector)
	if err == nil && existing != nil {
		existing.Target = normalized
		existing.Constraints = constraints
		existing.Resolved = resolved
		existing.UpdatedAt = now
		if err := s.Declarations.SaveDeclaration(existing); err != nil {
			return nil, err
		}
		s.Metrics.DeclaredTargetRecall.WithLabelValues(collection, vector, normalized.Alias).Set(normalized.MinRecall)
		return existing, nil
	}
	d := &Declaration{
		Collection: collection, Vector: vector, Target: normalized, Constraints: constraints,
		Resolved: resolved, CreatedAt: now, UpdatedAt: now,
	}
	if err := s.Declarations.SaveDeclaration(d); err != nil {
		return nil, err
	}
	s.Metrics.DeclaredTargetRecall.WithLabelValues(collection, vector, normalized.Alias).Set(normalized.MinRecall)
	return d, nil
}

// Status returns declared-vs-effective drift and the active switch.
func (s *Service) Status(ctx context.Context, collection, vector string) (*Status, error) {
	out := &Status{Collection: collection, Vector: vector}
	if d, err := s.Declarations.GetDeclaration(collection, vector); err == nil {
		out.Declaration = d
	}
	if s.Backend != nil {
		obs, err := s.Backend.EffectiveForm(ctx, collection, vector)
		if err != nil && !errors.Is(err, ErrCollectionGone) {
			return nil, err
		}
		out.Observed = obs
		out.Metrics = StatusVolume{obs.EncodedBytes, obs.RawBytes, obs.CompressionRatio}
		s.Metrics.RecordObservedForm(obs)
		if out.Declaration != nil {
			out.Drift = CompareDeclaredForm(out.Declaration.Resolved, obs)
		}
	}
	switches, err := s.Manager.ListSwitches()
	if err == nil {
		for _, r := range switches {
			if r.Collection == collection && (vector == "" || r.Vector == vector) && !r.Phase.Terminal() {
				cp := r
				out.ActiveSwitch = cp
				p := r.Progress()
				out.Progress = &p
				s.Metrics.RecordProgress(p)
				break
			}
		}
	}
	return out, nil
}
