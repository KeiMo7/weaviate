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
	"sort"
	"strings"
	"sync"

	"github.com/pkg/errors"
)

// FailureCause categorizes why a single item could not be classified, so a
// batch run can report not just how many items failed, but why.
type FailureCause string

const (
	// CauseMissingTrainingData: no (labeled) neighbors were found to learn from.
	CauseMissingTrainingData FailureCause = "missingTrainingData"

	// CauseVectorMismatch: the item's vector and a sample's vector have
	// different dimensions.
	CauseVectorMismatch FailureCause = "vectorDimensionMismatch"

	// CauseDanglingReference: a sample's label points at an object that no
	// longer exists.
	CauseDanglingReference FailureCause = "danglingReference"

	// CauseMalformedLabel: a label property has an unexpected type.
	CauseMalformedLabel FailureCause = "malformedLabelProperty"

	// CauseOther: anything not covered above.
	CauseOther FailureCause = "other"
)

// ClassifyError is an item-level classification failure with a machine
// readable Cause. It is returned before anything is written, so a failed item
// never leaves a half-classified object behind.
type ClassifyError struct {
	Cause FailureCause
	Err   error
}

func (e *ClassifyError) Error() string {
	return fmt.Sprintf("%v: %v", e.Cause, e.Err)
}

func (e *ClassifyError) Unwrap() error {
	return e.Err
}

// CauseOf extracts the FailureCause of an error, defaulting to CauseOther.
func CauseOf(err error) FailureCause {
	var classifyErr *ClassifyError
	if errors.As(err, &classifyErr) {
		return classifyErr.Cause
	}
	return CauseOther
}

// failureCauseCounter counts item failures per cause, safe for concurrent
// use by the classification workers.
type failureCauseCounter struct {
	mutex  sync.Mutex
	counts map[FailureCause]int64
}

func newFailureCauseCounter() *failureCauseCounter {
	return &failureCauseCounter{counts: map[FailureCause]int64{}}
}

func (c *failureCauseCounter) add(cause FailureCause) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.counts[cause]++
}

// summary renders the counts in a stable (sorted) order, e.g.
// "danglingReference=1, vectorDimensionMismatch=2"
func (c *failureCauseCounter) summary() string {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	causes := make([]string, 0, len(c.counts))
	for cause := range c.counts {
		causes = append(causes, string(cause))
	}
	sort.Strings(causes)

	parts := make([]string, 0, len(causes))
	for _, cause := range causes {
		parts = append(parts, fmt.Sprintf("%s=%d", cause, c.counts[FailureCause(cause)]))
	}
	return strings.Join(parts, ", ")
}
