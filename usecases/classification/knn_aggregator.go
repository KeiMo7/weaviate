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
	"fmt"
	"math"
	"sort"

	"github.com/go-openapi/strfmt"
	"github.com/pkg/errors"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/search"
	libvectorizer "github.com/weaviate/weaviate/usecases/vectorizer"
)

// distanceWeightEpsilon keeps the weight of an exact match (distance 0)
// finite. It is part of the weighting rule, do not change independently of
// distanceWeight.
const distanceWeightEpsilon = 1e-6

// distanceWeight maps a normalized distance d in [0, 1] to a vote weight.
//
// The rule is: weight(d) = 1 / (d + 1e-6)
//
// It is strictly monotonically decreasing in d (the closer a neighbor, the
// heavier its vote) and fully deterministic: the same distance always yields
// the same weight.
func distanceWeight(d float32) float64 {
	return 1 / (float64(d) + distanceWeightEpsilon)
}

// VotingRules control how the labels of the k neighbors are aggregated into
// the label(s) assigned to an item. The zero value reproduces the legacy
// behavior exactly: unweighted majority voting, exactly one label per
// property, no abstention.
type VotingRules struct {
	// DistanceWeighted weights every neighbor's vote by distanceWeight(d)
	// instead of 1, so close neighbors outvote distant majorities.
	DistanceWeighted bool

	// MaxLabels caps how many labels are assigned per property. Values < 1
	// are treated as 1 (legacy single-label behavior).
	MaxLabels int32

	// MinScore is the minimal normalized score (label weight / total weight,
	// in [0, 1]) a label needs to be assigned. If no label reaches it, the
	// property is abstained - a normal outcome, not an error. 0 disables.
	MinScore float64

	// MinRelativeScore requires a label's normalized score to be at least
	// this fraction of the top label's normalized score. 0 disables.
	MinRelativeScore float64
}

func (r VotingRules) withDefaults() VotingRules {
	if r.MaxLabels < 1 {
		r.MaxLabels = 1
	}
	return r
}

// NeighborLabel is a single assigned label of a multi-label aggregation. The
// slice NeighborRef.Winners is ordered by score (desc), then by mean distance
// (asc), then by beacon (asc) and is therefore fully deterministic.
type NeighborLabel struct {
	Beacon strfmt.URI

	// Score is the label's normalized score in (0, 1]: its share of the
	// total vote weight.
	Score float64

	// Count is the number of neighbor votes for this label.
	Count int

	Distances NeighborRefDistances
}

// KnnAggregator aggregates the reference properties of the k nearest
// neighbors into classification labels. The result depends only on the set of
// neighbors and their distances, never on their order: identical input always
// produces identical output, including the order of the output slice.
type KnnAggregator struct {
	input        search.Results
	sourceVector []float32
}

func NewKnnAggregator(input search.Results, sourceVector []float32) *KnnAggregator {
	return &KnnAggregator{input: input, sourceVector: sourceVector}
}

func (a *KnnAggregator) Aggregate(k int, properties []string, rules VotingRules) ([]NeighborRef, error) {
	if len(a.input) == 0 {
		return nil, &ClassifyError{
			Cause: CauseMissingTrainingData,
			Err:   fmt.Errorf("no neighbors found to aggregate labels from (k=%d)", k),
		}
	}

	votes, err := a.extractVotes(properties)
	if err != nil {
		return nil, errors.Wrap(err, "aggregate: extract votes from neighbors")
	}

	return a.aggregateVotes(votes, properties, rules.withDefaults())
}

// extractVotes collects, per property and per distinct label beacon, the
// distances of the samples voting for it. A sample carrying several labels
// votes once for each of them; a label repeated on the same sample votes only
// once. Samples that do not carry a property at all are skipped for that
// property.
func (a *KnnAggregator) extractVotes(properties []string) (neighborVotes, error) {
	votes := neighborVotes{}
	for i, elem := range a.input {
		schemaMap, ok := elem.Schema.(map[string]interface{})
		if !ok {
			return nil, &ClassifyError{
				Cause: CauseMalformedLabel,
				Err:   fmt.Errorf("expected element[%d].Schema to be map, got: %T", i, elem.Schema),
			}
		}

		distance := float32(0)
		distanceComputed := false
		for _, prop := range properties {
			refProp, ok := schemaMap[prop]
			if !ok {
				// this sample carries no label for this property
				continue
			}

			refTyped, ok := refProp.(models.MultipleRef)
			if !ok {
				return nil, &ClassifyError{
					Cause: CauseMalformedLabel,
					Err:   fmt.Errorf("expected element[%d].Schema.%s to be models.MultipleRef, got: %T", i, prop, refProp),
				}
			}

			if len(refTyped) == 0 {
				continue
			}

			if !distanceComputed {
				d, err := libvectorizer.NormalizedDistance(a.sourceVector, elem.Vector)
				if err != nil {
					return nil, &ClassifyError{
						Cause: CauseVectorMismatch,
						Err:   errors.Wrap(err, "calculate distance between source and candidate"),
					}
				}
				distance = d
				distanceComputed = true
			}

			seen := map[string]struct{}{}
			for _, ref := range refTyped {
				if ref == nil || ref.Beacon == "" {
					return nil, &ClassifyError{
						Cause: CauseDanglingReference,
						Err: fmt.Errorf("element[%d].Schema.%s contains a reference without a "+
							"target, the labeled object may have been deleted", i, prop),
					}
				}
				beacon := ref.Beacon.String()
				if _, dup := seen[beacon]; dup {
					continue
				}
				seen[beacon] = struct{}{}

				propVotes := votes[prop]
				if propVotes == nil {
					propVotes = beaconVotes{}
					votes[prop] = propVotes
				}
				propVotes[beacon] = append(propVotes[beacon], distance)
			}
		}
	}

	return votes, nil
}

func (a *KnnAggregator) aggregateVotes(votes neighborVotes, properties []string,
	rules VotingRules,
) ([]NeighborRef, error) {
	// the output follows the order of the requested properties, so it is
	// stable across runs
	out := make([]NeighborRef, 0, len(properties))
	for _, propName := range properties {
		beacons := votes[propName]
		if len(beacons) == 0 {
			return nil, &ClassifyError{
				Cause: CauseMissingTrainingData,
				Err:   fmt.Errorf("no neighbor carries a label for property %q", propName),
			}
		}
		out = append(out, aggregateProperty(propName, beacons, rules))
	}

	return out, nil
}

// labelScore accumulates the votes of a single label (beacon).
type labelScore struct {
	beacon       string
	count        int
	weight       float64
	score        float64
	distances    []float32
	meanDistance float32
}

func aggregateProperty(prop string, beacons beaconVotes, rules VotingRules) NeighborRef {
	// iterate beacons in sorted order, so that float summation and ranking
	// are independent of map iteration order
	names := make([]string, 0, len(beacons))
	for beacon := range beacons {
		names = append(names, beacon)
	}
	sort.Strings(names)

	labels := make([]labelScore, 0, len(names))
	totalCount := 0
	totalWeight := 0.
	for _, name := range names {
		distances := beacons[name]
		weight := 0.
		for _, d := range distances {
			if rules.DistanceWeighted {
				weight += distanceWeight(d)
			} else {
				weight++
			}
		}
		totalCount += len(distances)
		totalWeight += weight
		labels = append(labels, labelScore{
			beacon:       name,
			count:        len(distances),
			weight:       weight,
			distances:    distances,
			meanDistance: meanDistance(distances),
		})
	}
	for i := range labels {
		labels[i].score = labels[i].weight / totalWeight
	}

	// rank by score (desc), break ties by mean distance (asc) and finally by
	// beacon (asc) - the last comparison is on a unique key, so the order is
	// total and deterministic
	sort.SliceStable(labels, func(i, j int) bool {
		if labels[i].score != labels[j].score {
			return labels[i].score > labels[j].score
		}
		if labels[i].meanDistance != labels[j].meanDistance {
			return labels[i].meanDistance < labels[j].meanDistance
		}
		return labels[i].beacon < labels[j].beacon
	})

	selected := make([]labelScore, 0, rules.MaxLabels)
	if len(labels) > 0 {
		topScore := labels[0].score
		for _, label := range labels {
			if rules.MinScore > 0 && label.score < rules.MinScore {
				continue
			}
			if rules.MinRelativeScore > 0 && label.score < rules.MinRelativeScore*topScore {
				continue
			}
			selected = append(selected, label)
			if int32(len(selected)) >= rules.MaxLabels {
				break
			}
		}
	}

	ref := NeighborRef{Property: prop, OverallCount: totalCount}
	if len(selected) == 0 {
		// abstention: no label reached the confidence threshold
		ref.Abstained = true
		return ref
	}

	winner := selected[0]
	ref.Beacon = strfmt.URI(winner.beacon)
	ref.WinningCount = winner.count
	ref.LosingCount = totalCount - winner.count
	ref.Distances = labelDistances(beacons, winner.beacon)

	if rules.MaxLabels > 1 {
		ref.Winners = make([]NeighborLabel, len(selected))
		for i, label := range selected {
			ref.Winners[i] = NeighborLabel{
				Beacon:    strfmt.URI(label.beacon),
				Score:     label.score,
				Count:     label.count,
				Distances: labelDistances(beacons, label.beacon),
			}
		}
	}

	return ref
}

// labelDistances computes the distance statistics of the winning label
// against all losing ones, preserving the legacy field semantics.
func labelDistances(beacons beaconVotes, winner string) NeighborRefDistances {
	out := NeighborRefDistances{}

	var winningDistances []float32
	var losingDistances []float32

	// iterate in sorted order, so the appended losing distances do not depend
	// on map iteration order
	names := make([]string, 0, len(beacons))
	for beacon := range beacons {
		names = append(names, beacon)
	}
	sort.Strings(names)

	for _, beacon := range names {
		if beacon == winner {
			winningDistances = beacons[beacon]
		} else {
			losingDistances = append(losingDistances, beacons[beacon]...)
		}
	}

	if len(losingDistances) > 0 {
		mean := meanDistance(losingDistances)
		out.MeanLosingDistance = &mean

		closest := minDistance(losingDistances)
		out.ClosestLosingDistance = &closest
	}

	out.ClosestOverallDistance = minDistance(append(winningDistances, losingDistances...))
	out.ClosestWinningDistance = minDistance(winningDistances)
	out.MeanWinningDistance = meanDistance(winningDistances)

	return out
}

type neighborVotes map[string]beaconVotes

// beaconVotes maps a label beacon to the distances of the samples voting for
// it, in the order the neighbors were returned by the vector search.
type beaconVotes map[string][]float32

func meanDistance(in []float32) float32 {
	sum := float32(0)
	for _, v := range in {
		sum += v
	}

	return sum / float32(len(in))
}

func minDistance(in []float32) float32 {
	min := float32(math.MaxFloat32)
	for _, dist := range in {
		if dist < min {
			min = dist
		}
	}

	return min
}
