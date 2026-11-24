//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package classification

import (
	"testing"

	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/search"
	libvectorizer "github.com/weaviate/weaviate/usecases/vectorizer"
)

// vector positions relative to source [1,0,0]:
//
//	[1,0,0]     distance 0
//	[1,0.01,0]  distance ~0.000025 (very close)
//	[1,1,0]     distance ~0.146
//	[0,1,0]     distance 0.5
//	[0,0,1]     distance 0.5
//	[-1,0,0]    distance 1 (maximal)
var aggregatorSourceVector = []float32{1, 0, 0}

func neighborWithLabels(vector []float32, prop string, beacons ...string) search.Result {
	refs := models.MultipleRef{}
	for _, beacon := range beacons {
		refs = append(refs, &models.SingleRef{Beacon: strfmt.URI(beacon)})
	}
	return search.Result{
		ClassName: "Article",
		Vector:    vector,
		Schema: map[string]interface{}{
			prop: refs,
		},
	}
}

func mustDistance(t *testing.T, a, b []float32) float32 {
	t.Helper()
	d, err := libvectorizer.NormalizedDistance(a, b)
	require.Nil(t, err)
	return d
}

const (
	beaconA = "weaviate://localhost/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	beaconB = "weaviate://localhost/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
	beaconC = "weaviate://localhost/cccccccc-cccc-cccc-cccc-cccccccccccc"
)

func TestKnnAggregator_LegacySingleLabelSemantics(t *testing.T) {
	// pins the existing single-label output (counts and distance statistics)
	// of the unweighted aggregation
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
	}

	res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
		Aggregate(3, []string{"exactCategory"}, VotingRules{})
	require.Nil(t, err)
	require.Len(t, res, 1)

	dClose := mustDistance(t, aggregatorSourceVector, []float32{1, 0, 0})
	dMid := mustDistance(t, aggregatorSourceVector, []float32{0, 1, 0})
	dFar := mustDistance(t, aggregatorSourceVector, []float32{0, 0, 1})

	ref := res[0]
	assert.Equal(t, "exactCategory", ref.Property)
	assert.Equal(t, strfmt.URI(beaconA), ref.Beacon)
	assert.Equal(t, 2, ref.WinningCount)
	assert.Equal(t, 3, ref.OverallCount)
	assert.Equal(t, 1, ref.LosingCount)
	assert.False(t, ref.Abstained)
	assert.Nil(t, ref.Winners, "legacy requests must not carry multi-label results")

	assert.Equal(t, dClose, ref.Distances.ClosestWinningDistance)
	assert.Equal(t, (dClose+dMid)/2, ref.Distances.MeanWinningDistance)
	assert.Equal(t, dClose, ref.Distances.ClosestOverallDistance)
	require.NotNil(t, ref.Distances.MeanLosingDistance)
	assert.Equal(t, dFar, *ref.Distances.MeanLosingDistance)
	require.NotNil(t, ref.Distances.ClosestLosingDistance)
	assert.Equal(t, dFar, *ref.Distances.ClosestLosingDistance)
}

func TestKnnAggregator_TieBreakIsDeterministic(t *testing.T) {
	// equal vote counts for both labels: the closer label must win, every
	// time, regardless of map iteration order
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
	}

	first, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
		Aggregate(4, []string{"exactCategory"}, VotingRules{})
	require.Nil(t, err)
	require.Len(t, first, 1)
	assert.Equal(t, strfmt.URI(beaconA), first[0].Beacon,
		"on a vote tie the label with the smaller mean distance must win")
	assert.Equal(t, 2, first[0].WinningCount)
	assert.Equal(t, 4, first[0].OverallCount)

	for i := 0; i < 25; i++ {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(4, []string{"exactCategory"}, VotingRules{})
		require.Nil(t, err)
		assert.Equal(t, first, res, "run %d: identical input must produce identical output", i)
	}
}

func TestKnnAggregator_ExactTieBreaksByBeacon(t *testing.T) {
	// identical counts AND identical distances: the lexicographically
	// smaller beacon wins, so the outcome is still deterministic
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconA),
	}

	for i := 0; i < 25; i++ {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(2, []string{"exactCategory"}, VotingRules{})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Equal(t, strfmt.URI(beaconA), res[0].Beacon)
	}
}

func TestKnnAggregator_DistanceWeightedVoting(t *testing.T) {
	// a long-tail label whose only sample is the closest neighbor must beat
	// a majority label whose samples are all far away - but only when
	// distance weighting is enabled
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0.01, 0}, "exactCategory", beaconA), // long-tail, very close
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
		neighborWithLabels([]float32{-1, 0, 0}, "exactCategory", beaconB),
	}

	t.Run("unweighted majority vote keeps legacy semantics", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Equal(t, strfmt.URI(beaconB), res[0].Beacon)
		assert.Equal(t, 5, res[0].WinningCount)
		assert.Equal(t, 6, res[0].OverallCount)
	})

	t.Run("weighted vote lets the close minority win", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{DistanceWeighted: true})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Equal(t, strfmt.URI(beaconA), res[0].Beacon,
			"the closest neighbor's label must outvote the distant majority")
		assert.Equal(t, 1, res[0].WinningCount, "vote counts are unaffected by weighting")
		assert.Equal(t, 6, res[0].OverallCount)
	})

	t.Run("weights are monotonically decreasing in distance", func(t *testing.T) {
		assert.Greater(t, distanceWeight(0), distanceWeight(0.001))
		assert.Greater(t, distanceWeight(0.001), distanceWeight(0.5))
		assert.Greater(t, distanceWeight(0.5), distanceWeight(1))
	})
}

func TestKnnAggregator_MultiLabel(t *testing.T) {
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconC),
	}
	// unweighted scores: A=3/6=0.5, B=2/6=0.333, C=1/6=0.167

	t.Run("assigns up to maxLabels labels in a deterministic order", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{MaxLabels: 2})
		require.Nil(t, err)
		require.Len(t, res, 1)

		ref := res[0]
		require.Len(t, ref.Winners, 2)
		assert.Equal(t, strfmt.URI(beaconA), ref.Winners[0].Beacon)
		assert.Equal(t, strfmt.URI(beaconB), ref.Winners[1].Beacon)
		assert.InDelta(t, 0.5, ref.Winners[0].Score, 1e-6)
		assert.InDelta(t, 2.0/6.0, ref.Winners[1].Score, 1e-6)
		assert.Equal(t, 3, ref.Winners[0].Count)
		assert.Equal(t, 2, ref.Winners[1].Count)

		// the legacy fields keep describing the top label
		assert.Equal(t, strfmt.URI(beaconA), ref.Beacon)
		assert.Equal(t, 3, ref.WinningCount)
		assert.Equal(t, 6, ref.OverallCount)
		assert.Equal(t, 3, ref.LosingCount)
	})

	t.Run("maxLabels caps the number of labels", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{MaxLabels: 1})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Nil(t, res[0].Winners, "maxLabels=1 is the legacy single-label path")
		assert.Equal(t, strfmt.URI(beaconA), res[0].Beacon)
	})

	t.Run("relative score factor drops labels that lag the top label", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{MaxLabels: 3, MinRelativeScore: 0.6})
		require.Nil(t, err)
		require.Len(t, res, 1)
		// threshold: 0.6 * 0.5 = 0.3 -> B (0.333) qualifies, C (0.167) does not
		require.Len(t, res[0].Winners, 2)
		assert.Equal(t, strfmt.URI(beaconA), res[0].Winners[0].Beacon)
		assert.Equal(t, strfmt.URI(beaconB), res[0].Winners[1].Beacon)
	})
}

func TestKnnAggregator_Abstention(t *testing.T) {
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
	}
	// scores: A=0.5, B=0.5

	t.Run("abstains when no label reaches the confidence threshold", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{MinScore: 0.6})
		require.Nil(t, err, "abstention is a normal outcome, not an error")
		require.Len(t, res, 1)
		assert.True(t, res[0].Abstained)
		assert.Empty(t, res[0].Beacon)
		assert.Nil(t, res[0].Winners)
		assert.Equal(t, 6, res[0].OverallCount)
	})

	t.Run("assigns the label when the threshold is reached", func(t *testing.T) {
		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(6, []string{"exactCategory"}, VotingRules{MinScore: 0.5})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.False(t, res[0].Abstained)
		assert.NotEmpty(t, res[0].Beacon)
	})
}

func TestKnnAggregator_FewerLabeledSamplesThanNeighbors(t *testing.T) {
	// some neighbors do not carry the label property at all: they simply do
	// not vote, the remaining samples still produce a result
	withDescription := func(res search.Result) search.Result {
		res.Schema.(map[string]interface{})["description"] = "some text"
		return res
	}
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		withDescription(search.Result{
			ClassName: "Article",
			Vector:    []float32{0, 1, 0},
			Schema:    map[string]interface{}{"description": "no label here"},
		}),
		withDescription(search.Result{
			ClassName: "Article",
			Vector:    []float32{0, 0, 1},
			Schema:    map[string]interface{}{"description": "no label here either"},
		}),
	}

	res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
		Aggregate(3, []string{"exactCategory"}, VotingRules{})
	require.Nil(t, err)
	require.Len(t, res, 1)
	assert.Equal(t, strfmt.URI(beaconA), res[0].Beacon)
	assert.Equal(t, 1, res[0].WinningCount)
	assert.Equal(t, 1, res[0].OverallCount, "unlabeled neighbors must not be counted")

	t.Run("no labeled samples at all is a missing-training-data error", func(t *testing.T) {
		unlabeled := search.Results{
			{
				ClassName: "Article",
				Vector:    []float32{0, 1, 0},
				Schema:    map[string]interface{}{"description": "no label"},
			},
		}
		_, err := NewKnnAggregator(unlabeled, aggregatorSourceVector).
			Aggregate(1, []string{"exactCategory"}, VotingRules{})
		require.NotNil(t, err)
		assert.Equal(t, CauseMissingTrainingData, CauseOf(err))
	})

	t.Run("empty neighbor set is a missing-training-data error", func(t *testing.T) {
		_, err := NewKnnAggregator(search.Results{}, aggregatorSourceVector).
			Aggregate(3, []string{"exactCategory"}, VotingRules{})
		require.NotNil(t, err)
		assert.Equal(t, CauseMissingTrainingData, CauseOf(err))
	})
}

func TestKnnAggregator_MultiLabelSamples(t *testing.T) {
	t.Run("a sample carrying several labels votes for each of them", func(t *testing.T) {
		neighbors := search.Results{
			neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA, beaconB),
			neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconA),
		}

		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(2, []string{"exactCategory"}, VotingRules{})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Equal(t, strfmt.URI(beaconA), res[0].Beacon)
		assert.Equal(t, 2, res[0].WinningCount)
		assert.Equal(t, 3, res[0].OverallCount, "the multi-label sample contributes two votes")
	})

	t.Run("the same label repeated on one sample votes only once", func(t *testing.T) {
		neighbors := search.Results{
			neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA, beaconA),
			neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		}

		res, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(2, []string{"exactCategory"}, VotingRules{})
		require.Nil(t, err)
		require.Len(t, res, 1)
		assert.Equal(t, 2, res[0].OverallCount,
			"a duplicated label on a single sample must be deduplicated")
		assert.Equal(t, 1, res[0].WinningCount)
	})
}

func TestKnnAggregator_VectorDimensionMismatch(t *testing.T) {
	neighbors := search.Results{
		neighborWithLabels([]float32{1, 0}, "exactCategory", beaconA), // 2 dims, source has 3
	}

	_, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
		Aggregate(1, []string{"exactCategory"}, VotingRules{})
	require.NotNil(t, err)
	assert.Equal(t, CauseVectorMismatch, CauseOf(err))
	assert.Contains(t, err.Error(), "different dimensions")
}

func TestKnnAggregator_DanglingReference(t *testing.T) {
	t.Run("nil reference entry", func(t *testing.T) {
		neighbors := search.Results{
			{
				ClassName: "Article",
				Vector:    []float32{1, 0, 0},
				Schema: map[string]interface{}{
					"exactCategory": models.MultipleRef{nil},
				},
			},
		}
		_, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(1, []string{"exactCategory"}, VotingRules{})
		require.NotNil(t, err)
		assert.Equal(t, CauseDanglingReference, CauseOf(err))
	})

	t.Run("empty beacon", func(t *testing.T) {
		neighbors := search.Results{
			neighborWithLabels([]float32{1, 0, 0}, "exactCategory", ""),
		}
		_, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
			Aggregate(1, []string{"exactCategory"}, VotingRules{})
		require.NotNil(t, err)
		assert.Equal(t, CauseDanglingReference, CauseOf(err))
	})
}

func TestKnnAggregator_MalformedLabelProperty(t *testing.T) {
	neighbors := search.Results{
		{
			ClassName: "Article",
			Vector:    []float32{1, 0, 0},
			Schema: map[string]interface{}{
				"exactCategory": "not-a-reference",
			},
		},
	}
	_, err := NewKnnAggregator(neighbors, aggregatorSourceVector).
		Aggregate(1, []string{"exactCategory"}, VotingRules{})
	require.NotNil(t, err)
	assert.Equal(t, CauseMalformedLabel, CauseOf(err))
}

func TestKnnAggregator_DeterministicOutput(t *testing.T) {
	buildNeighbors := func(description string) search.Results {
		neighbors := search.Results{
			neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
			neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconB),
			neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconA),
			neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconB),
			neighborWithLabels([]float32{1, 0.01, 0}, "exactCategory", beaconC),
		}
		for i := range neighbors {
			neighbors[i].Schema.(map[string]interface{})["description"] = description
			neighbors[i].Schema.(map[string]interface{})["mainCategory"] = models.MultipleRef{&models.SingleRef{Beacon: strfmt.URI(beaconC)}}
		}
		return neighbors
	}

	properties := []string{"mainCategory", "exactCategory"}
	rules := VotingRules{DistanceWeighted: true, MaxLabels: 2}

	first, err := NewKnnAggregator(buildNeighbors("run one"), aggregatorSourceVector).
		Aggregate(5, properties, rules)
	require.Nil(t, err)
	require.Len(t, first, 2)

	t.Run("repeated runs produce identical output", func(t *testing.T) {
		for i := 0; i < 25; i++ {
			res, err := NewKnnAggregator(buildNeighbors("run one"), aggregatorSourceVector).
				Aggregate(5, properties, rules)
			require.Nil(t, err)
			assert.Equal(t, first, res, "run %d must be identical", i)
		}
	})

	t.Run("unrelated field changes do not affect the result", func(t *testing.T) {
		res, err := NewKnnAggregator(buildNeighbors("completely different text"), aggregatorSourceVector).
			Aggregate(5, properties, rules)
		require.Nil(t, err)
		assert.Equal(t, first, res)
	})

	t.Run("output order follows the requested properties", func(t *testing.T) {
		assert.Equal(t, "mainCategory", first[0].Property)
		assert.Equal(t, "exactCategory", first[1].Property)
	})
}
