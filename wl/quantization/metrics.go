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
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics contains quantization target, evaluation and switch instruments.
// Labels are bounded to collection and vector (plus low-cardinality status,
// codec and phase); shard/replica detail belongs in the management API.
type Metrics struct {
	DeclaredTargetRecall *prometheus.GaugeVec
	FormInfo             *prometheus.GaugeVec
	FormEncodedBytes     *prometheus.GaugeVec
	FormRawBytes         *prometheus.GaugeVec
	FormCompressionRatio *prometheus.GaugeVec

	EvaluationsTotal   *prometheus.CounterVec
	EvaluationRecall   *prometheus.GaugeVec
	EvaluationSamples  *prometheus.HistogramVec
	EvaluationDuration *prometheus.HistogramVec
	EvaluationFailures *prometheus.CounterVec

	SwitchesStarted     *prometheus.CounterVec
	SwitchesCompleted   *prometheus.CounterVec
	SwitchesCanceled    *prometheus.CounterVec
	SwitchesRolledBack  *prometheus.CounterVec
	SwitchesRecovered   *prometheus.CounterVec
	SwitchFailures      *prometheus.CounterVec
	SwitchActive        *prometheus.GaugeVec
	SwitchProgress      *prometheus.GaugeVec
	SwitchUnitsTotal    *prometheus.GaugeVec
	SwitchUnitsComplete *prometheus.GaugeVec
}

// NewMetrics registers metrics with reg. When reg is nil a private registry
// is used, so unit tests never double-register with Prometheus' default.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	if reg == nil {
		reg = prometheus.NewRegistry()
	}
	r := promauto.With(reg)
	labelsCV := []string{"collection", "vector", "codec"}
	switchLabels := []string{"collection", "vector", "form"}

	return &Metrics{
		DeclaredTargetRecall: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_target_min_recall",
			Help: "Declared minimum recall target for a collection vector.",
		}, []string{"collection", "vector", "alias"}),
		FormInfo: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_effective_form_info",
			Help: "Effective encoding observed from written data. Value is always 1; codec and bits are labels.",
		}, []string{"collection", "vector", "codec", "bits", "form"}),
		FormEncodedBytes: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_encoded_bytes",
			Help: "Measured bytes occupied by encoded vectors for the collection/vector, aggregated across units.",
		}, labelsCV),
		FormRawBytes: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_raw_vector_bytes",
			Help: "Measured full-precision vector bytes retained for evaluation and rescoring.",
		}, labelsCV),
		FormCompressionRatio: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_compression_ratio",
			Help: "Measured raw bytes divided by encoded bytes for the effective form.",
		}, labelsCV),

		EvaluationsTotal: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_evaluations_total",
			Help: "Recall evaluations by terminal status.",
		}, []string{"collection", "vector", "status", "strategy"}),
		EvaluationRecall: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_evaluation_last_recall",
			Help: "Last successful measured recall@k by sample strategy and k.",
		}, []string{"collection", "vector", "strategy", "k"}),
		EvaluationSamples: r.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "weaviate_quantization_evaluation_samples",
			Help:    "Number of queries included in a recall evaluation.",
			Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
		}, []string{"collection", "vector", "strategy"}),
		EvaluationDuration: r.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "weaviate_quantization_evaluation_duration_seconds",
			Help:    "Recall evaluation duration.",
			Buckets: prometheus.DefBuckets,
		}, []string{"collection", "vector", "status"}),
		EvaluationFailures: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_evaluation_failures_total",
			Help: "Failed recall evaluations by failure class.",
		}, []string{"collection", "vector", "class"}),

		SwitchesStarted: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switches_started_total",
			Help: "Quantization form switches started, by destination form.",
		}, switchLabels),
		SwitchesCompleted: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switches_completed_total",
			Help: "Quantization form switches that completed and promoted the new form.",
		}, switchLabels),
		SwitchesCanceled: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switches_canceled_total",
			Help: "Quantization form switches canceled before the atomic swap barrier.",
		}, switchLabels),
		SwitchesRolledBack: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switches_rolled_back_total",
			Help: "Quantization form switches that failed and restored the old form.",
		}, switchLabels),
		SwitchesRecovered: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switches_recovered_total",
			Help: "Nonterminal quantization switch records resumed at process startup.",
		}, switchLabels),
		SwitchFailures: r.NewCounterVec(prometheus.CounterOpts{
			Name: "weaviate_quantization_switch_failures_total",
			Help: "Quantization switch unit/state failures by phase.",
		}, []string{"collection", "vector", "phase"}),
		SwitchActive: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_switch_active",
			Help: "Whether a form switch is active for the collection/vector.",
		}, []string{"collection", "vector"}),
		SwitchProgress: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_switch_progress_fraction",
			Help: "Aggregated switch progress fraction from durable unit checkpoints, not a full scan.",
		}, []string{"collection", "vector"}),
		SwitchUnitsTotal: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_switch_units_total",
			Help: "Total shard/replica units in the switch.",
		}, []string{"collection", "vector"}),
		SwitchUnitsComplete: r.NewGaugeVec(prometheus.GaugeOpts{
			Name: "weaviate_quantization_switch_units_complete",
			Help: "Completed shard/replica units in the switch.",
		}, []string{"collection", "vector"}),
	}
}

// RecordEvaluation updates evaluation metrics from a report.
func (m *Metrics) RecordEvaluation(r *EvaluationReport) {
	if m == nil || r == nil {
		return
	}
	m.EvaluationsTotal.WithLabelValues(r.Collection, r.Vector, r.Status, r.Strategy).Inc()
	m.EvaluationDuration.WithLabelValues(r.Collection, r.Vector, r.Status).Observe(r.Took.Seconds())
	if r.SampleCount > 0 {
		m.EvaluationSamples.WithLabelValues(r.Collection, r.Vector, r.Strategy).Observe(float64(r.SampleCount))
	}
	if r.Status == EvalStatusSucceeded && r.Recall != nil {
		m.EvaluationRecall.WithLabelValues(r.Collection, r.Vector, r.Strategy, itoa(r.K)).Set(*r.Recall)
	}
	if r.Status == EvalStatusFailed {
		m.EvaluationFailures.WithLabelValues(r.Collection, r.Vector, "failed").Inc()
	}
	if r.Status == EvalStatusCanceled {
		m.EvaluationFailures.WithLabelValues(r.Collection, r.Vector, "canceled").Inc()
	}
}

// RecordObservedForm updates effective-form and volume metrics.
func (m *Metrics) RecordObservedForm(o FormObservation) {
	if m == nil {
		return
	}
	codec := o.Codec
	if codec == "" {
		codec = o.Form.Codec
	}
	bits := o.Bits
	if bits == 0 {
		bits = o.Form.Bits
	}
	form := string(o.Form.Form)
	if form == "" {
		form = string(EquivalentForm(codec, bits))
	}
	m.FormInfo.Reset()
	m.FormInfo.WithLabelValues(o.Collection, o.Vector, codec, itoa(bits), form).Set(1)
	m.FormEncodedBytes.WithLabelValues(o.Collection, o.Vector, codec).Set(float64(o.EncodedBytes))
	m.FormRawBytes.WithLabelValues(o.Collection, o.Vector, codec).Set(float64(o.RawBytes))
	ratio := o.CompressionRatio
	if ratio == 0 && o.RawBytes > 0 && o.EncodedBytes > 0 {
		ratio = float64(o.RawBytes) / float64(o.EncodedBytes)
	}
	m.FormCompressionRatio.WithLabelValues(o.Collection, o.Vector, codec).Set(ratio)
}

// RecordProgress updates switch progress gauges.
func (m *Metrics) RecordProgress(p Progress) {
	if m == nil {
		return
	}
	m.SwitchProgress.WithLabelValues(p.Collection, p.Vector).Set(p.FractionComplete)
	m.SwitchUnitsTotal.WithLabelValues(p.Collection, p.Vector).Set(float64(p.UnitsTotal))
	m.SwitchUnitsComplete.WithLabelValues(p.Collection, p.Vector).Set(float64(p.UnitsComplete))
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
