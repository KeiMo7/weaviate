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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSwitchStateMachineLegalAndIllegalTransitions(t *testing.T) {
	now := time.Unix(100, 0)
	r := &SwitchRecord{Phase: PhasePending}

	require.NoError(t, r.Transition(PhasePlanning, now))
	require.NoError(t, r.Transition(PhaseBuilding, now.Add(time.Second)))
	require.NoError(t, r.Transition(PhaseCatchingUp, now.Add(2*time.Second)))
	require.NoError(t, r.Transition(PhaseSwapping, now.Add(3*time.Second)))
	require.NoError(t, r.Transition(PhaseCompleted, now.Add(4*time.Second)))
	assert.True(t, r.Phase.Terminal())
	require.ErrorIs(t, r.Transition(PhaseBuilding, now), ErrIllegalTransition)
}

func TestSwitchStateMachineRejectsSkippingSwapToComplete(t *testing.T) {
	r := &SwitchRecord{Phase: PhaseBuilding}
	err := r.Transition(PhaseCompleted, time.Now())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrIllegalTransition))
	assert.Equal(t, PhaseBuilding, r.Phase)
}

func TestSwitchStateMachineRollbackPath(t *testing.T) {
	r := &SwitchRecord{Phase: PhaseSwapping}
	require.NoError(t, r.Transition(PhaseRollingBack, time.Now()))
	require.NoError(t, r.Transition(PhaseRolledBack, time.Now()))
	require.ErrorIs(t, r.Transition(PhaseCompleted, time.Now()), ErrIllegalTransition)
}

func TestProgressAggregation(t *testing.T) {
	r := &SwitchRecord{
		Phase: PhaseBuilding,
		Units: map[string]*UnitState{
			"a": {UnitID: "a", Phase: PhaseCompleted, Processed: 10, TotalHint: 10},
			"b": {UnitID: "b", Phase: PhaseBuilding, Processed: 5, TotalHint: 10},
			"c": {UnitID: "c", Phase: PhasePending},
		},
	}
	p := r.Progress()
	assert.Equal(t, 3, p.UnitsTotal)
	assert.Equal(t, 1, p.UnitsComplete)
	assert.InDelta(t, .75, p.FractionComplete, .001) // 15 processed / 20 known hint
}

func TestCompareDeclaredFormDrift(t *testing.T) {
	declared, err := Resolve(Target{MinRecall: .95}, Constraints{Distance: "cosine"})
	require.NoError(t, err)

	inSync := CompareDeclaredForm(declared, FormObservation{Codec: CodecRQ, Bits: 8})
	assert.True(t, inSync.InSync)

	drift := CompareDeclaredForm(declared, FormObservation{Codec: CodecBQ, Bits: 1})
	assert.False(t, drift.InSync)
	require.Len(t, drift.Differences, 2)
}
