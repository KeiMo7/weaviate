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
	"fmt"
	"time"

	"github.com/go-openapi/strfmt"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"github.com/weaviate/weaviate/adapters/handlers/rest/filterext"
	"github.com/weaviate/weaviate/cluster/schema/local"
	"github.com/weaviate/weaviate/entities/additional"
	"github.com/weaviate/weaviate/entities/dto"
	enterrors "github.com/weaviate/weaviate/entities/errors"
	libfilters "github.com/weaviate/weaviate/entities/filters"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/modulecapabilities"
	"github.com/weaviate/weaviate/entities/search"
	"github.com/weaviate/weaviate/usecases/auth/authorization"
	"github.com/weaviate/weaviate/usecases/objects"
	"github.com/weaviate/weaviate/usecases/schema/namespacing"
	libvectorizer "github.com/weaviate/weaviate/usecases/vectorizer"
)

type classificationFilters struct {
	source      *libfilters.LocalFilter
	target      *libfilters.LocalFilter
	trainingSet *libfilters.LocalFilter
}

func (f classificationFilters) Source() *libfilters.LocalFilter {
	return f.source
}

func (f classificationFilters) Target() *libfilters.LocalFilter {
	return f.target
}

func (f classificationFilters) TrainingSet() *libfilters.LocalFilter {
	return f.trainingSet
}

type distancer func(a, b []float32) (float32, error)

type Classifier struct {
	schemaGetter          local.ClassReader
	repo                  Repo
	vectorRepo            vectorRepo
	vectorClassSearchRepo modulecapabilities.VectorClassSearchRepo
	authorizer            authorization.Authorizer
	distancer             distancer
	modulesProvider       ModulesProvider
	logger                logrus.FieldLogger
}

type ModulesProvider interface {
	ParseClassifierSettings(name string,
		params *models.Classification) error
	GetClassificationFn(className, name string,
		params modulecapabilities.ClassifyParams) (modulecapabilities.ClassifyItemFn, error)
}

func New(sg local.ClassReader, cr Repo, vr vectorRepo, authorizer authorization.Authorizer,
	logger logrus.FieldLogger, modulesProvider ModulesProvider,
) *Classifier {
	return &Classifier{
		logger:                logger,
		schemaGetter:          sg,
		repo:                  cr,
		vectorRepo:            vr,
		authorizer:            authorizer,
		distancer:             libvectorizer.NormalizedDistance,
		vectorClassSearchRepo: newVectorClassSearchRepo(vr),
		modulesProvider:       modulesProvider,
	}
}

// Repo to manage classification state, should be consistent, not used to store
// actual data object vectors, see VectorRepo
type Repo interface {
	Put(ctx context.Context, classification models.Classification) error
	Get(ctx context.Context, id strfmt.UUID) (*models.Classification, error)
}

type VectorRepo interface {
	GetUnclassified(ctx context.Context, class string,
		properties []string, propertiesToReturn []string, filter *libfilters.LocalFilter) ([]search.Result, error)
	AggregateNeighbors(ctx context.Context, vector []float32,
		class string, properties []string, k int,
		filter *libfilters.LocalFilter, rules VotingRules) ([]NeighborRef, error)
	VectorSearch(ctx context.Context, params dto.GetParams, targetVectors []string, searchVectors []models.Vector) ([]search.Result, error)
	ZeroShotSearch(ctx context.Context, vector []float32,
		class string, properties []string,
		filter *libfilters.LocalFilter) ([]search.Result, error)
}

type vectorRepo interface {
	VectorRepo
	BatchPutObjects(ctx context.Context, objects objects.BatchObjects,
		repl *additional.ReplicationProperties, schemaVersion uint64) (objects.BatchObjects, error)
}

// NeighborRef is the result of an aggregation of the ref properties of k
// neighbors
type NeighborRef struct {
	// Property indicates which property was aggregated
	Property string

	// The beacon of the most common (kNN) reference
	Beacon strfmt.URI

	OverallCount int
	WinningCount int
	LosingCount  int

	Distances NeighborRefDistances

	// Winners holds all assigned labels when multi-label classification is
	// enabled (VotingRules.MaxLabels > 1), ordered by score (desc), then
	// mean distance (asc), then beacon (asc). It is nil for legacy
	// single-label requests, where only Beacon is set.
	Winners []NeighborLabel

	// Abstained is true when no label reached the configured confidence
	// threshold (VotingRules.MinScore). Abstention is a normal outcome: the
	// property is simply left unassigned.
	Abstained bool
}

func (c *Classifier) classGetterWithAuthzFunc(ctx context.Context, principal *models.Principal) func(string) (*models.Class, error) {
	authorizedCollections := map[string]*models.Class{}
	return func(name string) (*models.Class, error) {
		class, ok := authorizedCollections[name]
		if !ok {
			if err := c.authorizer.Authorize(ctx, principal, authorization.READ, authorization.Collections(name)...); err != nil {
				return nil, err
			}
			class = c.schemaGetter.ReadOnlyClass(name)
			authorizedCollections[name] = class
		}
		if class == nil {
			return nil, fmt.Errorf("could not find class %s in schema", name)
		}
		return class, nil
	}
}

func (c *Classifier) Schedule(ctx context.Context, principal *models.Principal, params models.Classification) (*models.Classification, error) {
	err := c.authorizer.Authorize(ctx, principal, authorization.UPDATE, authorization.CollectionsMetadata(params.Class)...)
	if err != nil {
		return nil, err
	}

	err = c.parseAndSetDefaults(&params)
	if err != nil {
		return nil, err
	}

	err = NewValidator(c.classGetterWithAuthzFunc(ctx, principal), params).Do()
	if err != nil {
		return nil, err
	}

	if err := c.assignNewID(&params); err != nil {
		return nil, fmt.Errorf("classification: assign id: %w", err)
	}

	params.Status = models.ClassificationStatusRunning
	params.Meta = &models.ClassificationMeta{
		Started: strfmt.DateTime(time.Now()),
	}

	if err := c.repo.Put(ctx, params); err != nil {
		return nil, fmt.Errorf("classification: put: %w", err)
	}

	// asynchronously trigger the classification
	filters, err := c.extractFilters(ctx, principal, params)
	if err != nil {
		return nil, err
	}

	enterrors.GoWrapper(func() { c.run(params, filters) }, c.logger)

	return &params, nil
}

func (c *Classifier) extractFilters(ctx context.Context, principal *models.Principal, params models.Classification) (Filters, error) {
	if params.Filters == nil {
		return classificationFilters{}, nil
	}

	// Classification is not exercised on namespace-enabled clusters, so the
	// nested-path qualification in filterext.Parse is hard-wired off here.
	qualifier := namespacing.Disabled

	source, err := filterext.Parse(params.Filters.SourceWhere, params.Class, qualifier, principal)
	if err != nil {
		return classificationFilters{}, fmt.Errorf("field 'sourceWhere': %w", err)
	}

	trainingSet, err := filterext.Parse(params.Filters.TrainingSetWhere, params.Class, qualifier, principal)
	if err != nil {
		return classificationFilters{}, fmt.Errorf("field 'trainingSetWhere': %w", err)
	}

	target, err := filterext.Parse(params.Filters.TargetWhere, params.Class, qualifier, principal)
	if err != nil {
		return classificationFilters{}, fmt.Errorf("field 'targetWhere': %w", err)
	}

	filters := classificationFilters{
		source:      source,
		trainingSet: trainingSet,
		target:      target,
	}

	if err = c.validateFilters(ctx, principal, &params, &filters); err != nil {
		return nil, err
	}

	return filters, nil
}

func (c *Classifier) validateFilters(ctx context.Context, principal *models.Principal, params *models.Classification, filters *classificationFilters) (err error) {
	if params.Type == TypeKNN {
		if err = c.validateFilter(ctx, principal, filters.Source()); err != nil {
			return fmt.Errorf("invalid sourceWhere: %w", err)
		}
		if err = c.validateFilter(ctx, principal, filters.TrainingSet()); err != nil {
			return fmt.Errorf("invalid trainingSetWhere: %w", err)
		}
	}

	if params.Type == TypeContextual || params.Type == TypeZeroShot {
		if err = c.validateFilter(ctx, principal, filters.Source()); err != nil {
			return fmt.Errorf("invalid sourceWhere: %w", err)
		}
		if err = c.validateFilter(ctx, principal, filters.Target()); err != nil {
			return fmt.Errorf("invalid targetWhere: %w", err)
		}
	}

	return err
}

func (c *Classifier) validateFilter(ctx context.Context, principal *models.Principal, filter *libfilters.LocalFilter) error {
	if filter == nil {
		return nil
	}
	f := func(name string) (*models.Class, error) {
		if err := c.authorizer.Authorize(ctx, principal, authorization.READ, authorization.CollectionsMetadata(name)...); err != nil {
			return nil, err
		}
		class := c.schemaGetter.ReadOnlyClass(name)
		if class == nil {
			return nil, fmt.Errorf("could not find class %s in schema", name)
		}

		return class, nil
	}
	return libfilters.ValidateFilters(f, filter)
}

func (c *Classifier) assignNewID(params *models.Classification) error {
	id, err := uuid.NewRandom()
	if err != nil {
		return err
	}

	params.ID = strfmt.UUID(id.String())
	return nil
}

func (c *Classifier) Get(ctx context.Context, principal *models.Principal, id strfmt.UUID) (*models.Classification, error) {
	classification, err := c.repo.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	if classification == nil {
		return nil, nil
	}
	if err := c.authorizer.Authorize(ctx, principal, authorization.READ, authorization.CollectionsMetadata(classification.Class)...); err != nil {
		return nil, err
	}
	return classification, nil
}

func (c *Classifier) parseAndSetDefaults(params *models.Classification) error {
	if params.Type == "" {
		defaultType := "knn"
		params.Type = defaultType
	}

	if params.Type == "knn" {
		if err := c.parseKNNSettings(params); err != nil {
			return errors.Wrapf(err, "parse knn specific settings")
		}
		return nil
	}

	if c.modulesProvider != nil {
		if err := c.modulesProvider.ParseClassifierSettings(params.Type, params); err != nil {
			return errors.Wrapf(err, "parse %s specific settings", params.Type)
		}
		return nil
	}

	return nil
}

func (c *Classifier) parseKNNSettings(params *models.Classification) error {
	raw := params.Settings
	settings := &ParamsKNN{}
	if raw == nil {
		settings.SetDefaults()
		params.Settings = settings
		return nil
	}

	asMap, ok := raw.(map[string]interface{})
	if !ok {
		return errors.Errorf("settings must be an object got %T", raw)
	}

	v, err := extractNumberFromMap(asMap, "k")
	if err != nil {
		return err
	}
	settings.K = v

	maxLabels, err := extractNumberFromMap(asMap, "maxLabels")
	if err != nil {
		return err
	}
	settings.MaxLabels = maxLabels

	distanceWeighted, err := extractBoolFromMap(asMap, "distanceWeighted")
	if err != nil {
		return err
	}
	settings.DistanceWeighted = distanceWeighted

	minScore, err := extractFloatFromMap(asMap, "minScore")
	if err != nil {
		return err
	}
	settings.MinScore = minScore

	minRelativeScore, err := extractFloatFromMap(asMap, "minRelativeScore")
	if err != nil {
		return err
	}
	settings.MinRelativeScore = minRelativeScore

	settings.SetDefaults()
	if err := settings.Validate(); err != nil {
		return err
	}
	params.Settings = settings

	return nil
}

type ParamsKNN struct {
	K *int32 `json:"k"`

	// DistanceWeighted enables distance-weighted voting, see
	// VotingRules.DistanceWeighted. Defaults to false (legacy majority vote).
	DistanceWeighted bool `json:"distanceWeighted,omitempty"`

	// MaxLabels caps the number of labels assigned per property. Defaults
	// to 1 (legacy single-label behavior).
	MaxLabels *int32 `json:"maxLabels,omitempty"`

	// MinScore is the minimal normalized score a label needs to be assigned,
	// see VotingRules.MinScore. Defaults to 0 (no abstention).
	MinScore *float64 `json:"minScore,omitempty"`

	// MinRelativeScore requires a label's score to be at least this fraction
	// of the top label's score, see VotingRules.MinRelativeScore. Defaults
	// to 0 (disabled).
	MinRelativeScore *float64 `json:"minRelativeScore,omitempty"`
}

func (params *ParamsKNN) SetDefaults() {
	if params.K == nil {
		defaultK := int32(3)
		params.K = &defaultK
	}
	if params.MaxLabels == nil {
		defaultMaxLabels := int32(1)
		params.MaxLabels = &defaultMaxLabels
	}
}

func (params *ParamsKNN) Validate() error {
	if *params.MaxLabels < 1 {
		return errors.Errorf("settings.maxLabels must be >= 1, got %d", *params.MaxLabels)
	}
	if params.MinScore != nil && (*params.MinScore < 0 || *params.MinScore > 1) {
		return errors.Errorf("settings.minScore must be in [0, 1], got %f", *params.MinScore)
	}
	if params.MinRelativeScore != nil && (*params.MinRelativeScore < 0 || *params.MinRelativeScore > 1) {
		return errors.Errorf("settings.minRelativeScore must be in [0, 1], got %f", *params.MinRelativeScore)
	}
	return nil
}

// VotingRules maps the user-facing settings to the rules the neighbor
// aggregation is executed with.
func (params *ParamsKNN) VotingRules() VotingRules {
	rules := VotingRules{
		DistanceWeighted: params.DistanceWeighted,
	}
	if params.MaxLabels != nil {
		rules.MaxLabels = *params.MaxLabels
	}
	if params.MinScore != nil {
		rules.MinScore = *params.MinScore
	}
	if params.MinRelativeScore != nil {
		rules.MinRelativeScore = *params.MinRelativeScore
	}
	return rules
}

func extractNumberFromMap(in map[string]interface{}, field string) (*int32, error) {
	unparsed, present := in[field]
	if present {
		parsed, ok := unparsed.(json.Number)
		if !ok {
			return nil, errors.Errorf("settings.%s must be number, got %T",
				field, unparsed)
		}

		asInt64, err := parsed.Int64()
		if err != nil {
			return nil, errors.Wrapf(err, "settings.%s", field)
		}

		asInt32 := int32(asInt64)
		return &asInt32, nil
	}

	return nil, nil
}

func extractFloatFromMap(in map[string]interface{}, field string) (*float64, error) {
	unparsed, present := in[field]
	if !present {
		return nil, nil
	}

	parsed, ok := unparsed.(json.Number)
	if !ok {
		return nil, errors.Errorf("settings.%s must be number, got %T",
			field, unparsed)
	}

	asFloat64, err := parsed.Float64()
	if err != nil {
		return nil, errors.Wrapf(err, "settings.%s", field)
	}

	return &asFloat64, nil
}

func extractBoolFromMap(in map[string]interface{}, field string) (bool, error) {
	unparsed, present := in[field]
	if !present {
		return false, nil
	}

	parsed, ok := unparsed.(bool)
	if !ok {
		return false, errors.Errorf("settings.%s must be bool, got %T",
			field, unparsed)
	}

	return parsed, nil
}
