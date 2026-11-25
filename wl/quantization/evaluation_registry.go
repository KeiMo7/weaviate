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
	"sync"

	"github.com/google/uuid"
)

// EvaluationRunner is the storage-adapter boundary. A production implementation
// loads deterministic query samples and current/full-precision searchers, then
// calls the package Evaluate function.
type EvaluationRunner interface {
	RunEvaluation(ctx context.Context, collection, vector string, opts SampleOptions) (*EvaluationReport, error)
}

// EvaluationRunnerFunc adapts a function.
type EvaluationRunnerFunc func(ctx context.Context, collection, vector string, opts SampleOptions) (*EvaluationReport, error)

func (f EvaluationRunnerFunc) RunEvaluation(ctx context.Context, collection, vector string, opts SampleOptions) (*EvaluationReport, error) {
	return f(ctx, collection, vector, opts)
}

// EvaluationRegistry retains recent evaluation reports for management reads. It
// never stores vectors, only per-query IDs and recall measurements.
type EvaluationRegistry struct {
	mu      sync.RWMutex
	reports map[string]*EvaluationReport
	runner  EvaluationRunner
	metrics *Metrics
}

func NewEvaluationRegistry(runner EvaluationRunner, metrics *Metrics) *EvaluationRegistry {
	if metrics == nil {
		metrics = NewMetrics(nil)
	}
	return &EvaluationRegistry{reports: map[string]*EvaluationReport{}, runner: runner, metrics: metrics}
}

// Evaluate runs an evaluation and stores its report. Failed and canceled
// reports are retained with nil Recall so status is distinguishable.
func (r *EvaluationRegistry) Evaluate(ctx context.Context, collection, vector string, opts SampleOptions) (*EvaluationReport, error) {
	if r.runner == nil {
		return nil, ErrNotFound
	}
	report, err := r.runner.RunEvaluation(ctx, collection, vector, opts)
	if report == nil {
		report = &EvaluationReport{Collection: collection, Vector: vector, Status: EvalStatusFailed}
		if err != nil {
			report.Error = err.Error()
		}
	}
	if report.ID == "" {
		report.ID = uuid.NewString()
	}
	if report.Collection == "" {
		report.Collection = collection
	}
	if report.Vector == "" {
		report.Vector = vector
	}
	r.mu.Lock()
	r.reports[report.ID] = report
	r.mu.Unlock()
	r.metrics.RecordEvaluation(report)
	return report, err
}

func (r *EvaluationRegistry) Get(id string) (*EvaluationReport, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	report, ok := r.reports[id]
	if !ok {
		return nil, ErrNotFound
	}
	return report, nil
}

func (r *EvaluationRegistry) List() []*EvaluationReport {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*EvaluationReport, 0, len(r.reports))
	for _, v := range r.reports {
		out = append(out, v)
	}
	return out
}
