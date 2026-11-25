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

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/wl/quantization"
)

func newTestMux(t *testing.T, b quantization.Backend) (*http.ServeMux, *quantization.Service) {
	t.Helper()
	svc := quantization.NewService(nil, nil, b, quantization.NewMetrics(nil))
	mux := http.NewServeMux()
	SetupHandlers(mux, svc, nil)
	return mux, svc
}

func TestTargetPutGetAndInvalid(t *testing.T) {
	mux, _ := newTestMux(t, quantization.NewReferenceBackend("Article", "vec", 1, 4))

	body := bytes.NewBufferString(`{"target":{"alias":"balanced"},"constraints":{"dimensions":128,"distance":"cosine"}}`)
	req := httptest.NewRequest(http.MethodPut, pathPrefix+"/collections/Article/vectors/vec/target", body)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var decl quantization.Declaration
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &decl))
	assert.Equal(t, .95, decl.Target.MinRecall)
	assert.Equal(t, quantization.CodecRQ, decl.Resolved.Codec)

	getReq := httptest.NewRequest(http.MethodGet, pathPrefix+"/collections/Article/vectors/vec/target", nil)
	getRec := httptest.NewRecorder()
	mux.ServeHTTP(getRec, getReq)
	require.Equal(t, http.StatusOK, getRec.Code)

	bad := httptest.NewRequest(http.MethodPut, pathPrefix+"/collections/Article/vectors/vec/target",
		bytes.NewBufferString(`{"target":{"minRecall":2}}`))
	badRec := httptest.NewRecorder()
	mux.ServeHTTP(badRec, bad)
	assert.Equal(t, http.StatusUnprocessableEntity, badRec.Code)
}

func TestSwitchWaitEndpoint(t *testing.T) {
	b := quantization.NewReferenceBackend("Article", "vec", 1, 2)
	require.NoError(t, b.Write(context.Background(), "shard-0/r0", "a", []float32{1, 0}))
	mux, _ := newTestMux(t, b)

	body := bytes.NewBufferString(`{"wait":true,"to":{"form":"rq8","codec":"rq","bits":8,"rescoreLimit":64}}`)
	req := httptest.NewRequest(http.MethodPost, pathPrefix+"/collections/Article/vectors/vec/switches", body)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	var sw quantization.SwitchRecord
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &sw))
	assert.Equal(t, quantization.PhaseCompleted, sw.Phase)
}

func TestDomainErrorSwitchUnavailableIsRetryable503(t *testing.T) {
	h := &Handler{}
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	rec := httptest.NewRecorder()
	h.writeDomainError(rec, req, quantization.ErrSwitchUnavailable)
	assert.Equal(t, http.StatusServiceUnavailable, rec.Code)
	assert.Equal(t, "1", rec.Header().Get("Retry-After"))
	assert.Contains(t, rec.Body.String(), "quantization switch barrier")
}

func TestEvaluationEndpointPreservesFailedStatus(t *testing.T) {
	svc := quantization.NewService(nil, nil, nil, quantization.NewMetrics(nil)).
		WithEvaluationRunner(quantization.EvaluationRunnerFunc(func(context.Context, string, string, quantization.SampleOptions) (*quantization.EvaluationReport, error) {
			return &quantization.EvaluationReport{Status: quantization.EvalStatusFailed, Error: "training samples insufficient"}, quantization.ErrEvaluationFailed
		}))
	mux := http.NewServeMux()
	SetupHandlers(mux, svc, nil)

	req := httptest.NewRequest(http.MethodPost, pathPrefix+"/collections/C/vectors/v/evaluations", bytes.NewBufferString(`{}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusBadGateway, rec.Code)
	var report quantization.EvaluationReport
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &report))
	assert.Equal(t, quantization.EvalStatusFailed, report.Status)
	assert.Nil(t, report.Recall)

	getReq := httptest.NewRequest(http.MethodGet, pathPrefix+"/evaluations/"+report.ID, nil)
	getRec := httptest.NewRecorder()
	mux.ServeHTTP(getRec, getReq)
	assert.Equal(t, http.StatusOK, getRec.Code)
}
