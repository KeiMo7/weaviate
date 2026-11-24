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
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/entities/additional"
	"github.com/weaviate/weaviate/entities/dto"
	libfilters "github.com/weaviate/weaviate/entities/filters"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/search"
	"github.com/weaviate/weaviate/usecases/objects"
)

// realAggregatorRepos runs the actual KnnAggregator over a fixed neighbor set,
// so the classifyItemUsingKNN path can be tested end to end without a
// database.
type realAggregatorRepo struct {
	neighbors search.Results
}

func (r *realAggregatorRepo) GetUnclassified(ctx context.Context, class string,
	properties []string, propertiesToReturn []string, filter *libfilters.LocalFilter,
) ([]search.Result, error) {
	return nil, nil
}

func (r *realAggregatorRepo) AggregateNeighbors(ctx context.Context, vector []float32,
	class string, properties []string, k int,
	filter *libfilters.LocalFilter, rules VotingRules,
) ([]NeighborRef, error) {
	return NewKnnAggregator(r.neighbors, vector).Aggregate(k, properties, rules)
}

func (r *realAggregatorRepo) VectorSearch(ctx context.Context, params dto.GetParams,
	targetVectors []string, searchVectors []models.Vector,
) ([]search.Result, error) {
	return nil, nil
}

func (r *realAggregatorRepo) ZeroShotSearch(ctx context.Context, vector []float32,
	class string, properties []string, filter *libfilters.LocalFilter,
) ([]search.Result, error) {
	return nil, nil
}

func (r *realAggregatorRepo) BatchPutObjects(ctx context.Context, objects objects.BatchObjects,
	repl *additional.ReplicationProperties, schemaVersion uint64,
) (objects.BatchObjects, error) {
	return objects, nil
}

type capturingWriter struct {
	items []search.Result
}

func (w *capturingWriter) Start() {}

func (w *capturingWriter) Store(item search.Result) error {
	w.items = append(w.items, item)
	return nil
}

func (w *capturingWriter) Stop() WriterResults { return nil }

func knnTestItem() search.Result {
	return search.Result{
		ID:        "75ba35af-6a08-40ae-b442-3bec69b355f9",
		ClassName: "Article",
		Vector:    []float32{1, 0, 0},
		Schema: map[string]interface{}{
			"description": "an article about politics and food",
		},
	}
}

func knnTestNeighbors() search.Results {
	// unweighted scores: A=3/6, B=2/6, C=1/6
	return search.Results{
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 0, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconA),
		neighborWithLabels([]float32{1, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 1, 0}, "exactCategory", beaconB),
		neighborWithLabels([]float32{0, 0, 1}, "exactCategory", beaconC),
	}
}

func ptInt32(i int32) *int32 { return &i }

func knnParams(settings *ParamsKNN) models.Classification {
	return models.Classification{
		ID:                 "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
		Class:              "Article",
		ClassifyProperties: []string{"exactCategory"},
		Settings:           settings,
	}
}

func TestClassifyItemUsingKNN_MultiLabelWrite(t *testing.T) {
	repo := &realAggregatorRepo{neighbors: knnTestNeighbors()}
	classifier := &Classifier{vectorRepo: repo}
	writer := &capturingWriter{}

	params := knnParams(&ParamsKNN{K: ptInt32(6), MaxLabels: ptInt32(2)})
	err := classifier.classifyItemUsingKNN(knnTestItem(), 0, params, classificationFilters{}, writer)
	require.Nil(t, err)
	require.Len(t, writer.items, 1)

	schemaMap := writer.items[0].Schema.(map[string]interface{})
	refs, ok := schemaMap["exactCategory"].(models.MultipleRef)
	require.True(t, ok, "exactCategory must be a reference list")
	require.Len(t, refs, 2, "both labels above the cutoff must be written")

	assert.Equal(t, beaconA, refs[0].Beacon.String())
	assert.Equal(t, beaconB, refs[1].Beacon.String())

	require.NotNil(t, refs[0].Classification)
	assert.Equal(t, int64(3), refs[0].Classification.WinningCount)
	assert.Equal(t, int64(6), refs[0].Classification.OverallCount)
	require.NotNil(t, refs[1].Classification)
	assert.Equal(t, int64(2), refs[1].Classification.WinningCount)
	assert.Equal(t, int64(6), refs[1].Classification.OverallCount)

	meta, ok := writer.items[0].AdditionalProperties["classification"].(additional.Classification)
	require.True(t, ok)
	assert.Equal(t, []string{"exactCategory"}, meta.ClassifiedFields)
}

func TestClassifyItemUsingKNN_Abstention(t *testing.T) {
	repo := &realAggregatorRepo{neighbors: knnTestNeighbors()}
	classifier := &Classifier{vectorRepo: repo}
	writer := &capturingWriter{}

	// top score is 0.5, the threshold of 0.6 is not reached
	params := knnParams(&ParamsKNN{K: ptInt32(6), MinScore: ptFloat64(0.6)})
	err := classifier.classifyItemUsingKNN(knnTestItem(), 0, params, classificationFilters{}, writer)
	require.Nil(t, err, "abstention is a normal outcome, not an error")
	require.Len(t, writer.items, 1, "the item is still stored, just without a label")

	schemaMap := writer.items[0].Schema.(map[string]interface{})
	_, ok := schemaMap["exactCategory"]
	assert.False(t, ok, "no label may be written when abstaining")

	meta, ok := writer.items[0].AdditionalProperties["classification"].(additional.Classification)
	require.True(t, ok)
	assert.Empty(t, meta.ClassifiedFields)
}

func TestClassifyItemUsingKNN_RepeatedRunsAreIdentical(t *testing.T) {
	settings := &ParamsKNN{K: ptInt32(6), MaxLabels: ptInt32(2), DistanceWeighted: true}

	runOnce := func() search.Result {
		repo := &realAggregatorRepo{neighbors: knnTestNeighbors()}
		classifier := &Classifier{vectorRepo: repo}
		writer := &capturingWriter{}
		err := classifier.classifyItemUsingKNN(knnTestItem(), 0, knnParams(settings),
			classificationFilters{}, writer)
		require.Nil(t, err)
		require.Len(t, writer.items, 1)
		return writer.items[0]
	}

	first := runOnce()
	second := runOnce()
	assert.Equal(t, first.Schema, second.Schema,
		"the same batch executed twice must produce identical labels")

	t.Run("re-classifying the same object does not stack references", func(t *testing.T) {
		repo := &realAggregatorRepo{neighbors: knnTestNeighbors()}
		classifier := &Classifier{vectorRepo: repo}
		writer := &capturingWriter{}

		item := knnTestItem()
		require.Nil(t, classifier.classifyItemUsingKNN(item, 0, knnParams(settings),
			classificationFilters{}, writer))
		require.Nil(t, classifier.classifyItemUsingKNN(writer.items[0], 0, knnParams(settings),
			classificationFilters{}, writer))
		require.Len(t, writer.items, 2)

		refs := writer.items[1].Schema.(map[string]interface{})["exactCategory"].(models.MultipleRef)
		assert.Len(t, refs, 2, "re-running must replace, not duplicate or stack references")
		assert.Equal(t, writer.items[0].Schema, writer.items[1].Schema)
	})
}

func TestClassifyItemUsingKNN_ItemFailures(t *testing.T) {
	params := knnParams(&ParamsKNN{K: ptInt32(1)})

	t.Run("vector dimension mismatch", func(t *testing.T) {
		repo := &realAggregatorRepo{neighbors: search.Results{
			neighborWithLabels([]float32{1, 0}, "exactCategory", beaconA),
		}}
		classifier := &Classifier{vectorRepo: repo}
		writer := &capturingWriter{}

		err := classifier.classifyItemUsingKNN(knnTestItem(), 0, params, classificationFilters{}, writer)
		require.NotNil(t, err)
		assert.Equal(t, CauseVectorMismatch, CauseOf(err))
		assert.Empty(t, writer.items, "a failed item must not write a half-classified object")
	})

	t.Run("label pointing at a deleted object", func(t *testing.T) {
		repo := &realAggregatorRepo{neighbors: search.Results{
			neighborWithLabels([]float32{1, 0, 0}, "exactCategory", ""),
		}}
		classifier := &Classifier{vectorRepo: repo}
		writer := &capturingWriter{}

		err := classifier.classifyItemUsingKNN(knnTestItem(), 0, params, classificationFilters{}, writer)
		require.NotNil(t, err)
		assert.Equal(t, CauseDanglingReference, CauseOf(err))
		assert.Empty(t, writer.items)
	})

	t.Run("no training data", func(t *testing.T) {
		repo := &realAggregatorRepo{neighbors: search.Results{}}
		classifier := &Classifier{vectorRepo: repo}
		writer := &capturingWriter{}

		err := classifier.classifyItemUsingKNN(knnTestItem(), 0, params, classificationFilters{}, writer)
		require.NotNil(t, err)
		assert.Equal(t, CauseMissingTrainingData, CauseOf(err))
		assert.Empty(t, writer.items)
	})
}

func TestRunWorkers_FailureCausesAreCounted(t *testing.T) {
	classifyFn := func(item search.Result, itemIndex int,
		params models.Classification, filters Filters, writer Writer,
	) error {
		switch item.ID {
		case "00000000-0000-0000-0000-000000000001":
			return &ClassifyError{Cause: CauseVectorMismatch, Err: errors.New("dimensions differ")}
		case "00000000-0000-0000-0000-000000000002":
			return &ClassifyError{Cause: CauseDanglingReference, Err: errors.New("target gone")}
		default:
			return writer.Store(item)
		}
	}

	items := []search.Result{
		{
			ID: "00000000-0000-0000-0000-000000000001", ClassName: "Article",
			Vector: []float32{1, 0, 0}, Schema: map[string]interface{}{},
		},
		{
			ID: "00000000-0000-0000-0000-000000000002", ClassName: "Article",
			Vector: []float32{1, 0, 0}, Schema: map[string]interface{}{},
		},
		{
			ID: "00000000-0000-0000-0000-000000000003", ClassName: "Article",
			Vector: []float32{1, 0, 0}, Schema: map[string]interface{}{},
		},
	}

	workers := newRunWorkers(1, classifyFn, models.Classification{}, classificationFilters{},
		newFakeVectorRepoKNN(nil, nil), newNullLogger())
	workers.addJobs(items)
	res := workers.work(context.Background())

	assert.Equal(t, int64(1), res.successCount, "one item classifies fine")
	assert.Equal(t, int64(2), res.errorCount, "two items fail, the batch is not interrupted")
	require.NotNil(t, res.err)
	assert.Contains(t, res.err.Error(),
		"item failure causes: danglingReference=1, vectorDimensionMismatch=1")
	assert.NotContains(t, res.err.Error(), "data save error",
		"classification failures must not be reported as save failures")
}

type failingBatchRepo struct {
	*fakeVectorRepoKNN
}

func (r *failingBatchRepo) BatchPutObjects(ctx context.Context, objs objects.BatchObjects,
	repl *additional.ReplicationProperties, schemaVersion uint64,
) (objects.BatchObjects, error) {
	for i := range objs {
		objs[i].Err = errors.New("save failed")
	}
	return objs, nil
}

func TestRunWorkers_SaveFailuresAreReported(t *testing.T) {
	classifyFn := func(item search.Result, itemIndex int,
		params models.Classification, filters Filters, writer Writer,
	) error {
		return writer.Store(item)
	}

	items := []search.Result{
		{
			ID: "00000000-0000-0000-0000-000000000001", ClassName: "Article",
			Vector: []float32{1, 0, 0}, Schema: map[string]interface{}{},
		},
	}

	repo := &failingBatchRepo{newFakeVectorRepoKNN(nil, nil)}
	workers := newRunWorkers(1, classifyFn, models.Classification{}, classificationFilters{},
		repo, newNullLogger())
	workers.addJobs(items)
	res := workers.work(context.Background())

	require.NotNil(t, res.err)
	assert.Contains(t, res.err.Error(), "data save error")
}

func TestParseKNNSettings_NewRules(t *testing.T) {
	classifier := &Classifier{}

	t.Run("defaults keep the legacy behavior", func(t *testing.T) {
		params := models.Classification{}
		require.Nil(t, classifier.parseKNNSettings(&params))
		settings := params.Settings.(*ParamsKNN)
		assert.Equal(t, int32(3), *settings.K)
		assert.Equal(t, int32(1), *settings.MaxLabels)
		assert.False(t, settings.DistanceWeighted)
		assert.Nil(t, settings.MinScore)
		assert.Nil(t, settings.MinRelativeScore)
		assert.Equal(t, VotingRules{MaxLabels: 1}, settings.VotingRules())
	})

	t.Run("valid new settings are parsed", func(t *testing.T) {
		params := models.Classification{Settings: map[string]interface{}{
			"k":                json.Number("5"),
			"maxLabels":        json.Number("2"),
			"distanceWeighted": true,
			"minScore":         json.Number("0.4"),
			"minRelativeScore": json.Number("0.7"),
		}}
		require.Nil(t, classifier.parseKNNSettings(&params))
		settings := params.Settings.(*ParamsKNN)
		assert.Equal(t, int32(5), *settings.K)
		assert.Equal(t, int32(2), *settings.MaxLabels)
		assert.True(t, settings.DistanceWeighted)
		assert.Equal(t, 0.4, *settings.MinScore)
		assert.Equal(t, 0.7, *settings.MinRelativeScore)
		assert.Equal(t, VotingRules{
			DistanceWeighted: true,
			MaxLabels:        2,
			MinScore:         0.4,
			MinRelativeScore: 0.7,
		}, settings.VotingRules())
	})

	t.Run("invalid settings are rejected", func(t *testing.T) {
		for name, settings := range map[string]map[string]interface{}{
			"maxLabels below 1":           {"maxLabels": json.Number("0")},
			"minScore above 1":            {"minScore": json.Number("1.5")},
			"minScore below 0":            {"minScore": json.Number("-0.1")},
			"minRelativeScore above 1":    {"minRelativeScore": json.Number("2")},
			"distanceWeighted not a bool": {"distanceWeighted": "yes"},
			"maxLabels not a number":      {"maxLabels": "two"},
			"minScore not a number":       {"minScore": true},
		} {
			t.Run(name, func(t *testing.T) {
				params := models.Classification{Settings: settings}
				assert.NotNil(t, classifier.parseKNNSettings(&params))
			})
		}
	})
}
