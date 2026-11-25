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
	"fmt"
	"sort"
	"time"
)

// Phase is the durable switch state-machine phase.
type Phase string

const (
	PhasePending     Phase = "pending"
	PhasePlanning    Phase = "planning"
	PhaseTraining    Phase = "training"
	PhaseBuilding    Phase = "building"
	PhaseCatchingUp  Phase = "catching_up"
	PhaseSwapping    Phase = "swapping"
	PhaseCompleted   Phase = "completed"
	PhaseRollingBack Phase = "rolling_back"
	PhaseRolledBack  Phase = "rolled_back"
	PhaseFailed      Phase = "failed"
	PhaseCanceled    Phase = "canceled"
	PhaseAborted     Phase = "aborted"
)

// Terminal phases never transition again and are safe to retain for status.
func (p Phase) Terminal() bool {
	switch p {
	case PhaseCompleted, PhaseRolledBack, PhaseFailed, PhaseCanceled, PhaseAborted:
		return true
	default:
		return false
	}
}

// UnitState is one shard/replica's durable progress.
type UnitState struct {
	UnitID     string    `json:"unitId"`
	Shard      string    `json:"shard"`
	Replica    string    `json:"replica,omitempty"`
	Node       string    `json:"node,omitempty"`
	Phase      Phase     `json:"phase"`
	Checkpoint string    `json:"checkpoint,omitempty"`
	Processed  int64     `json:"processed"`
	TotalHint  int64     `json:"totalHint,omitempty"`
	Attempts   int       `json:"attempts"`
	LastError  string    `json:"lastError,omitempty"`
	UpdatedAt  time.Time `json:"updatedAt"`
}

// SwitchRecord is the disk-decidable phase/form/progress record.
type SwitchRecord struct {
	ID             string                `json:"id"`
	Collection     string                `json:"collection"`
	Vector         string                `json:"vector"`
	DeclaredTarget *Target               `json:"declaredTarget,omitempty"`
	From           FormSpec              `json:"from"`
	To             FormSpec              `json:"to"`
	Phase          Phase                 `json:"phase"`
	Units          map[string]*UnitState `json:"units"`
	FailureCount   int64                 `json:"failureCount"`
	LastError      string                `json:"lastError,omitempty"`
	CreatedAt      time.Time             `json:"createdAt"`
	UpdatedAt      time.Time             `json:"updatedAt"`
	StartedAt      *time.Time            `json:"startedAt,omitempty"`
	FinishedAt     *time.Time            `json:"finishedAt,omitempty"`
}

// Progress is the aggregated management/metrics view.
type Progress struct {
	SwitchID         string  `json:"switchId"`
	Collection       string  `json:"collection"`
	Vector           string  `json:"vector"`
	Phase            Phase   `json:"phase"`
	UnitsTotal       int     `json:"unitsTotal"`
	UnitsComplete    int     `json:"unitsComplete"`
	UnitsFailed      int     `json:"unitsFailed"`
	Processed        int64   `json:"processed"`
	TotalHint        int64   `json:"totalHint,omitempty"`
	FractionComplete float64 `json:"fractionComplete"`
	FailureCount     int64   `json:"failureCount"`
}

// SortedUnits returns unit states in stable ID order.
func (r *SwitchRecord) SortedUnits() []*UnitState {
	out := make([]*UnitState, 0, len(r.Units))
	for _, u := range r.Units {
		out = append(out, u)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].UnitID < out[j].UnitID })
	return out
}

// Progress aggregates all shard/replica units without scanning user data.
func (r *SwitchRecord) Progress() Progress {
	p := Progress{
		SwitchID:     r.ID,
		Collection:   r.Collection,
		Vector:       r.Vector,
		Phase:        r.Phase,
		UnitsTotal:   len(r.Units),
		FailureCount: r.FailureCount,
	}
	var weight float64
	for _, u := range r.Units {
		p.Processed += u.Processed
		p.TotalHint += u.TotalHint
		switch {
		case u.Phase == PhaseCompleted:
			p.UnitsComplete++
			weight++
		case u.Phase == PhaseFailed || u.LastError != "":
			p.UnitsFailed++
		case u.TotalHint > 0:
			weight += float64(u.Processed) / float64(u.TotalHint)
		}
	}
	if p.UnitsTotal > 0 {
		if p.TotalHint > 0 && p.Processed > 0 {
			p.FractionComplete = float64(p.Processed) / float64(p.TotalHint)
			if p.FractionComplete > 1 {
				p.FractionComplete = 1
			}
		} else {
			p.FractionComplete = weight / float64(p.UnitsTotal)
		}
	}
	if r.Phase == PhaseCompleted {
		p.FractionComplete = 1
		p.UnitsComplete = p.UnitsTotal
	}
	return p
}

// allowedTransitions is the one authoritative state-machine edge table.
var allowedTransitions = map[Phase]map[Phase]bool{
	PhasePending: {
		PhasePlanning: true, PhaseTraining: true, PhaseBuilding: true,
		PhaseCanceled: true, PhaseFailed: true, PhaseAborted: true,
	},
	PhasePlanning: {
		PhaseTraining: true, PhaseBuilding: true, PhaseRollingBack: true,
		PhaseFailed: true, PhaseCanceled: true, PhaseAborted: true,
	},
	PhaseTraining: {
		PhaseBuilding: true, PhaseRollingBack: true, PhaseFailed: true,
		PhaseCanceled: true, PhaseAborted: true,
	},
	PhaseBuilding: {
		PhaseCatchingUp: true, PhaseSwapping: true, PhaseRollingBack: true,
		PhaseFailed: true, PhaseCanceled: true, PhaseAborted: true,
	},
	PhaseCatchingUp: {
		PhaseSwapping: true, PhaseBuilding: true, PhaseRollingBack: true,
		PhaseFailed: true, PhaseCanceled: true, PhaseAborted: true,
	},
	PhaseSwapping: {
		PhaseCompleted: true, PhaseRollingBack: true, PhaseFailed: true,
		PhaseAborted: true,
	},
	PhaseRollingBack: {
		PhaseRolledBack: true, PhaseFailed: true, PhaseAborted: true,
	},
}

// Transition validates and applies a phase transition.
func (r *SwitchRecord) Transition(to Phase, now time.Time) error {
	if r.Phase == to {
		return nil
	}
	next, ok := allowedTransitions[r.Phase]
	if !ok || !next[to] {
		return fmt.Errorf("%w: %s -> %s (switch %s collection %q vector %q)",
			ErrIllegalTransition, r.Phase, to, r.ID, r.Collection, r.Vector)
	}
	r.Phase = to
	r.UpdatedAt = now
	switch to {
	case PhasePending, PhasePlanning, PhaseCatchingUp:
		// These phases neither start the expensive unit work nor terminate it.
	case PhaseTraining, PhaseBuilding, PhaseSwapping, PhaseRollingBack:
		if r.StartedAt == nil {
			t := now
			r.StartedAt = &t
		}
	case PhaseCompleted, PhaseRolledBack, PhaseFailed, PhaseCanceled, PhaseAborted:
		t := now
		r.FinishedAt = &t
	}
	return nil
}

// FormObservation is read from written encoding data/stats, not schema echo.
type FormObservation struct {
	Collection            string    `json:"collection"`
	Vector                string    `json:"vector"`
	UnitID                string    `json:"unitId,omitempty"`
	Form                  FormSpec  `json:"form"`
	Codec                 string    `json:"codec"`
	Bits                  int       `json:"bits"`
	EncodedVectorCount    int64     `json:"encodedVectorCount"`
	EncodedBytes          int64     `json:"encodedBytes"`
	RawBytes              int64     `json:"rawBytes"`
	CompressionRatio      float64   `json:"compressionRatio"`
	EncodingStateRecorded bool      `json:"encodingStateRecorded"`
	ObservedAt            time.Time `json:"observedAt"`
	Source                string    `json:"source"`
}

// Drift explains declared-vs-effective configuration without claiming success.
type Drift struct {
	InSync      bool     `json:"inSync"`
	Differences []string `json:"differences,omitempty"`
}

// CompareDeclaredForm compares a resolved declaration with observed encoding.
func CompareDeclaredForm(declared FormSpec, observed FormObservation) Drift {
	var diffs []string
	if observed.Codec != "" && observed.Codec != declared.Codec {
		diffs = append(diffs, fmt.Sprintf("codec: declared %s, encoded-data reports %s", declared.Codec, observed.Codec))
	}
	if observed.Bits != 0 && observed.Bits != declared.Bits {
		diffs = append(diffs, fmt.Sprintf("bits: declared %d, encoded-data reports %d", declared.Bits, observed.Bits))
	}
	if observed.Form.PolicyVersion != "" && observed.Form.PolicyVersion != PolicyVersion {
		diffs = append(diffs, fmt.Sprintf("policyVersion: current %s, data record %s", PolicyVersion, observed.Form.PolicyVersion))
	}
	return Drift{InSync: len(diffs) == 0, Differences: diffs}
}
