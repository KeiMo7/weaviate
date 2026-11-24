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

package db

import (
	"context"

	"github.com/pkg/errors"
	"github.com/weaviate/weaviate/entities/additional"
	"github.com/weaviate/weaviate/entities/dto"
	"github.com/weaviate/weaviate/entities/filters"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/schema"
	"github.com/weaviate/weaviate/entities/search"
	"github.com/weaviate/weaviate/usecases/classification"
)

// TODO: why is this logic in the persistence package? This is business-logic,
// move out of here!
func (db *DB) GetUnclassified(ctx context.Context, className string,
	properties []string, propsToReturn []string, filter *filters.LocalFilter,
) ([]search.Result, error) {
	propsToReturnTmp := append(properties, propsToReturn...)
	props := make(search.SelectProperties, len(propsToReturnTmp))
	for i, prop := range propsToReturnTmp {
		props[i] = search.SelectProperty{Name: prop}
	}
	mergedFilter := mergeUserFilterWithRefCountFilter(filter, className, properties,
		filters.OperatorEqual, 0)
	res, err := db.Search(ctx, dto.GetParams{
		ClassName: className,
		Filters:   mergedFilter,
		Pagination: &filters.Pagination{
			Limit: 10000, // TODO: gh-1219 increase
		},
		AdditionalProperties: additional.Properties{
			Classification: true,
			Vector:         true,
			ModuleParams: map[string]interface{}{
				"interpretation": true,
			},
		},
		Properties: props,
	})

	return res, err
}

// TODO: why is this logic in the persistence package? This is business-logic,
// move out of here!
func (db *DB) ZeroShotSearch(ctx context.Context, vector []float32,
	class string, properties []string,
	filter *filters.LocalFilter,
) ([]search.Result, error) {
	props := make(search.SelectProperties, len(properties))
	for i, prop := range properties {
		props[i] = search.SelectProperty{Name: prop}
	}

	res, err := db.VectorSearch(ctx, dto.GetParams{
		ClassName: class,
		Pagination: &filters.Pagination{
			Limit: 1,
		},
		Filters: filter,
		AdditionalProperties: additional.Properties{
			Vector: true,
		},
		Properties: props,
	}, []string{""}, []models.Vector{vector})

	return res, err
}

// TODO: why is this logic in the persistence package? This is business-logic,
// move out of here!
func (db *DB) AggregateNeighbors(ctx context.Context, vector []float32,
	class string, properties []string, k int,
	filter *filters.LocalFilter, rules classification.VotingRules,
) ([]classification.NeighborRef, error) {
	props := make(search.SelectProperties, len(properties))
	for i, prop := range properties {
		props[i] = search.SelectProperty{Name: prop}
	}
	mergedFilter := mergeUserFilterWithRefCountFilter(filter, class, properties,
		filters.OperatorGreaterThan, 0)
	res, err := db.VectorSearch(ctx, dto.GetParams{
		ClassName: class,
		Pagination: &filters.Pagination{
			Limit: k,
		},
		Filters: mergedFilter,
		AdditionalProperties: additional.Properties{
			Vector: true,
		},
		Properties: props,
	}, []string{""}, []models.Vector{vector})
	if err != nil {
		return nil, errors.Wrap(err, "aggregate neighbors: search neighbors")
	}

	return classification.NewKnnAggregator(res, vector).Aggregate(k, properties, rules)
}

func mergeUserFilterWithRefCountFilter(userFilter *filters.LocalFilter, className string,
	properties []string, op filters.Operator, refCount int,
) *filters.LocalFilter {
	countFilters := make([]filters.Clause, len(properties))
	for i, prop := range properties {
		countFilters[i] = filters.Clause{
			Operator: op,
			Value: &filters.Value{
				Type:  schema.DataTypeInt,
				Value: refCount,
			},
			On: &filters.Path{
				Class:    schema.ClassName(className),
				Property: schema.PropertyName(prop),
			},
		}
	}

	var countRootClause filters.Clause
	if len(countFilters) == 1 {
		countRootClause = countFilters[0]
	} else {
		countRootClause = filters.Clause{
			Operands: countFilters,
			Operator: filters.OperatorAnd,
		}
	}

	rootFilter := &filters.LocalFilter{}
	if userFilter == nil {
		rootFilter.Root = &countRootClause
	} else {
		rootFilter.Root = &filters.Clause{
			Operator: filters.OperatorAnd, // so we can AND the refcount requirements and whatever custom filters, the user has
			Operands: []filters.Clause{*userFilter.Root, countRootClause},
		}
	}

	return rootFilter
}
