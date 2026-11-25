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
	"container/heap"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	enterrors "github.com/weaviate/weaviate/entities/errors"
)

const (
	EvalStatusSucceeded = "succeeded"
	EvalStatusFailed    = "failed"
	EvalStatusCanceled  = "canceled"
)

// Searcher runs the form under evaluation and returns ordered object IDs.
type Searcher interface {
	SearchQuantized(ctx context.Context, query []float32, k int) ([]string, error)
}

// FullPrecisionSearcher computes exact results from uncompressed vectors.
// Implementations must stream rather than loading an entire collection when
// possible; NewStreamingExactSearcher is the standard adapter.
type FullPrecisionSearcher interface {
	SearchFullPrecision(ctx context.Context, query []float32, k int) ([]string, error)
}

// EvaluationRequest is a self-contained recall evaluation. Managed callers use
// EvaluationService to populate Queries and searchers from a shard adapter.
type EvaluationRequest struct {
	Collection string        `json:"collection"`
	Vector     string        `json:"vector"`
	Options    SampleOptions `json:"options"`
	// Queries is the full candidate query pool. SelectSamples performs the
	// deterministic sampling. Managed adapters may instead set SampleQueries.
	Queries       []Query               `json:"-"`
	SampleQueries []Query               `json:"-"`
	Current       Searcher              `json:"-"`
	Full          FullPrecisionSearcher `json:"-"`
	// Logger is optional; a discard logger is used when nil.
	Logger logrus.FieldLogger `json:"-"`
	// Observed is the effective form being measured. It is used for labels and
	// drift explanation, never as proof that the encoded data exists.
	Observed FormObservation `json:"observed"`
}

// QueryRecall explains one sampled query.
type QueryRecall struct {
	QueryID          string   `json:"queryId"`
	Recall           float64  `json:"recall"`
	Intersection     int      `json:"intersection"`
	Denominator      int      `json:"denominator"`
	ReturnedCount    int      `json:"returnedCount"`
	GroundTruthCount int      `json:"groundTruthCount"`
	CurrentIDs       []string `json:"currentIds,omitempty"`
	TruthIDs         []string `json:"truthIds,omitempty"`
	Error            string   `json:"error,omitempty"`
}

// EvaluationReport is the only success output. Recall is nil for failed or
// canceled evaluations so JSON/management clients cannot mistake status=failed
// plus recall=0 for a measured zero.
type EvaluationReport struct {
	ID           string          `json:"id"`
	Collection   string          `json:"collection"`
	Vector       string          `json:"vector"`
	Status       string          `json:"status"`
	Form         FormSpec        `json:"form"`
	Observed     FormObservation `json:"observed,omitempty"`
	Recall       *float64        `json:"recall,omitempty"`
	SampleCount  int             `json:"sampleCount"`
	K            int             `json:"k"`
	ResultCount  int             `json:"resultCount"`
	Strategy     string          `json:"strategy"`
	Seed         int64           `json:"seed"`
	Took         time.Duration   `json:"took"`
	TookMillis   int64           `json:"tookMillis"`
	PerQuery     []QueryRecall   `json:"perQuery,omitempty"`
	Error        string          `json:"error,omitempty"`
	FailureCount int64           `json:"failureCount,omitempty"`
	StartedAt    time.Time       `json:"startedAt"`
	FinishedAt   time.Time       `json:"finishedAt"`
}

// EvaluationError carries collection/vector/stage context for failures.
type EvaluationError struct {
	Collection string
	Vector     string
	Stage      string
	Err        error
}

func (e *EvaluationError) Error() string {
	return fmt.Sprintf("quantization evaluation %s collection=%q vector=%q: %v",
		e.Stage, e.Collection, e.Vector, e.Err)
}

func (e *EvaluationError) Unwrap() error { return e.Err }

// Evaluate computes mean recall@k of Current against Full.
//
// It enforces the configured sample/k caps and bounded search concurrency. It
// does not itself retain vectors after the call except in the returned report.
func Evaluate(ctx context.Context, req EvaluationRequest, limits Limits) (*EvaluationReport, error) {
	limits = limits.withDefaults()
	started := time.Now()
	report := &EvaluationReport{
		Collection: req.Collection,
		Vector:     req.Vector,
		Observed:   req.Observed,
		Form:       req.Observed.Form,
		StartedAt:  started,
	}

	opts := req.Options.withDefaults(limits, ScopeSeed(req.Collection, req.Vector))
	if err := opts.validate(); err != nil {
		return failEvaluation(report, started, err), err
	}
	if opts.SampleSize > limits.MaxSampleSize {
		err := fmt.Errorf("%w: sampleSize %d exceeds limit %d", ErrInvalidTarget, opts.SampleSize, limits.MaxSampleSize)
		return failEvaluation(report, started, err), err
	}
	if opts.K > limits.MaxK {
		err := fmt.Errorf("%w: k %d exceeds limit %d", ErrInvalidTarget, opts.K, limits.MaxK)
		return failEvaluation(report, started, err), err
	}
	if req.Current == nil || req.Full == nil {
		err := fmt.Errorf("%w: current and full-precision searchers are required", ErrEvaluationFailed)
		wrapped := &EvaluationError{req.Collection, req.Vector, "setup", err}
		return failEvaluation(report, started, wrapped), wrapped
	}

	samples := req.SampleQueries
	if len(samples) == 0 {
		samples = SelectSamples(req.Queries, opts)
	}
	if len(samples) == 0 {
		err := fmt.Errorf("%w: no queries available for sampling", ErrEvaluationFailed)
		wrapped := &EvaluationError{req.Collection, req.Vector, "sample", err}
		return failEvaluation(report, started, wrapped), wrapped
	}
	if len(samples) > opts.SampleSize {
		samples = SelectSamples(samples, opts)
	}

	report.SampleCount = len(samples)
	report.K = opts.K
	report.Strategy = opts.Strategy
	report.Seed = opts.Seed

	workers := limits.MaxEvaluationWorkers
	if workers > len(samples) {
		workers = len(samples)
	}
	if workers <= 0 {
		workers = 1
	}

	logger := req.Logger
	if logger == nil {
		discard := logrus.New()
		discard.SetOutput(io.Discard)
		logger = discard
	}

	type result struct {
		qr  QueryRecall
		err error
	}
	results := make([]result, len(samples))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		enterrors.GoWrapper(func() {
			defer wg.Done()
			for i := range jobs {
				q := samples[i]
				qr := QueryRecall{QueryID: q.ID}
				current, err := req.Current.SearchQuantized(ctx, q.Vector, opts.K)
				if err != nil {
					results[i] = result{qr: qr, err: fmt.Errorf("current search query %q: %w", q.ID, err)}
					continue
				}
				truth, err := req.Full.SearchFullPrecision(ctx, q.Vector, opts.K)
				if err != nil {
					results[i] = result{qr: qr, err: fmt.Errorf("full-precision search query %q: %w", q.ID, err)}
					continue
				}
				recall, intersection, denominator := overlapRecall(current, truth, opts.K)
				qr.Recall = recall
				qr.Intersection = intersection
				qr.Denominator = denominator
				qr.ReturnedCount = len(current)
				qr.GroundTruthCount = len(truth)
				qr.CurrentIDs = current
				qr.TruthIDs = truth
				results[i] = result{qr: qr}
			}
		}, logger)
	}

	canceled := false
feed:
	for i := range samples {
		select {
		case jobs <- i:
		case <-ctx.Done():
			canceled = true
			break feed
		}
	}
	close(jobs)
	wg.Wait()

	if canceled || ctx.Err() != nil {
		err := ctx.Err()
		wrapped := fmt.Errorf("%w: %w", ErrEvaluationCanceled, err)
		return failEvaluation(report, started, wrapped), wrapped
	}

	perQuery := make([]QueryRecall, 0, len(results))
	sum := 0.0
	resultCount := 0
	for _, r := range results {
		if r.err != nil {
			wrapped := &EvaluationError{req.Collection, req.Vector, "search", r.err}
			return failEvaluation(report, started, wrapped), wrapped
		}
		perQuery = append(perQuery, r.qr)
		sum += r.qr.Recall
		resultCount += r.qr.ReturnedCount
	}
	mean := sum / float64(len(results))
	report.Status = EvalStatusSucceeded
	report.Recall = &mean
	report.PerQuery = perQuery
	report.ResultCount = resultCount
	report.FinishedAt = time.Now()
	report.Took = report.FinishedAt.Sub(started)
	report.TookMillis = report.Took.Milliseconds()
	return report, nil
}

func failEvaluation(report *EvaluationReport, started time.Time, err error) *EvaluationReport {
	report.Status = EvalStatusFailed
	if errors.Is(err, context.Canceled) || errors.Is(err, ErrEvaluationCanceled) {
		report.Status = EvalStatusCanceled
	}
	report.Error = err.Error()
	report.FinishedAt = time.Now()
	report.Took = report.FinishedAt.Sub(started)
	report.TookMillis = report.Took.Milliseconds()
	return report
}

// overlapRecall computes |a ∩ b| / min(k, |b|). IDs are treated as an
// unordered set because recall@k measures membership, not rank.
func overlapRecall(a, b []string, k int) (float64, int, int) {
	truth := make(map[string]struct{}, len(b))
	for _, id := range b {
		truth[id] = struct{}{}
	}
	intersection := 0
	seen := make(map[string]struct{}, len(a))
	for _, id := range a {
		if _, ok := truth[id]; ok {
			if _, dup := seen[id]; !dup {
				intersection++
				seen[id] = struct{}{}
			}
		}
	}
	den := k
	if len(b) < den {
		den = len(b)
	}
	if den <= 0 {
		return 0, intersection, 0
	}
	return float64(intersection) / float64(den), intersection, den
}

// VectorBatch is a bounded raw-vector scan batch.
type VectorBatch struct {
	IDs     []string
	Vectors [][]float32
}

// RawVectorBatcher streams full-precision vectors. Implementations must not
// materialize the whole collection.
type RawVectorBatcher interface {
	ScanRawVectors(ctx context.Context, batchSize int, fn func(VectorBatch) error) error
}

// DistanceFunc returns a distance where smaller means closer.
type DistanceFunc func(a, b []float32) float64

// SquaredL2Distance is the standard Euclidean squared distance.
func SquaredL2Distance(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	if len(a) != len(b) {
		sum += math.Inf(1)
	}
	return sum
}

// NegativeDotDistance ranks dot product by smaller negative dot.
func NegativeDotDistance(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot float64
	for i := 0; i < n; i++ {
		dot += float64(a[i]) * float64(b[i])
	}
	return -dot
}

// CosineDistance computes 1-cosine; callers needing exact production semantics
// should supply the same distance function configured on the collection.
func CosineDistance(a, b []float32) float64 {
	var dot, na, nb float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 1
	}
	return 1 - dot/(math.Sqrt(na)*math.Sqrt(nb))
}

type neighbor struct {
	id   string
	dist float64
}

type maxHeap []neighbor

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(neighbor)) }
func (h *maxHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

type streamingExact struct {
	batcher   RawVectorBatcher
	distance  DistanceFunc
	batchSize int
}

// NewStreamingExactSearcher adapts a raw vector scanner. SearchFullPrecision
// scans in bounded batches and keeps only k candidates for this query.
func NewStreamingExactSearcher(b RawVectorBatcher, distance DistanceFunc, batchSize int) FullPrecisionSearcher {
	if distance == nil {
		distance = CosineDistance
	}
	if batchSize <= 0 {
		batchSize = DefaultLimits().MaxBatchVectors
	}
	return &streamingExact{batcher: b, distance: distance, batchSize: batchSize}
}

func (s *streamingExact) SearchFullPrecision(ctx context.Context, query []float32, k int) ([]string, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	h := &maxHeap{}
	heap.Init(h)
	err := s.batcher.ScanRawVectors(ctx, s.batchSize, func(batch VectorBatch) error {
		for i, v := range batch.Vectors {
			if err := ctx.Err(); err != nil {
				return err
			}
			d := s.distance(query, v)
			if h.Len() < k {
				heap.Push(h, neighbor{id: batch.IDs[i], dist: d})
			} else if d < (*h)[0].dist {
				(*h)[0] = neighbor{id: batch.IDs[i], dist: d}
				heap.Fix(h, 0)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := make([]string, h.Len())
	// Heap is max-farthest; pop yields farthest first. Reverse for nearest first.
	for i := len(out) - 1; i >= 0; i-- {
		n := heap.Pop(h).(neighbor)
		out[i] = n.id
	}
	return out, nil
}

// SortedIDs is a test/helper to normalize ID sets.
func SortedIDs(ids []string) []string {
	out := append([]string(nil), ids...)
	sort.Strings(out)
	return out
}
