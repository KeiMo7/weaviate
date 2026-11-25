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

// Package handlers serves quantization target, evaluation and switch management.
package handlers

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/sirupsen/logrus"

	"github.com/weaviate/weaviate/wl/quantization"
)

const pathPrefix = "/debug/quantization/v1"

// Handler holds HTTP dependencies.
type Handler struct {
	service *quantization.Service
	logger  logrus.FieldLogger
}

// SetupHandlers registers quantization management routes. The endpoints are
// additive and live under /debug until the OpenAPI product API is generated.
func SetupHandlers(mux *http.ServeMux, service *quantization.Service, logger logrus.FieldLogger) *Handler {
	if logger == nil {
		l := logrus.New()
		l.SetLevel(logrus.WarnLevel)
		logger = l
	}
	h := &Handler{service: service, logger: logger.WithField("handler", "quantization")}
	mux.HandleFunc("PUT "+pathPrefix+"/collections/{collection}/vectors/{vector}/target", h.putTarget)
	mux.HandleFunc("GET "+pathPrefix+"/collections/{collection}/vectors/{vector}/target", h.getTarget)
	mux.HandleFunc("GET "+pathPrefix+"/collections/{collection}/vectors/{vector}/status", h.status)
	mux.HandleFunc("POST "+pathPrefix+"/collections/{collection}/vectors/{vector}/evaluations", h.evaluate)
	mux.HandleFunc("GET "+pathPrefix+"/evaluations/{id}", h.getEvaluation)
	mux.HandleFunc("POST "+pathPrefix+"/collections/{collection}/vectors/{vector}/switches", h.startSwitch)
	mux.HandleFunc("GET "+pathPrefix+"/switches/{id}", h.getSwitch)
	mux.HandleFunc("POST "+pathPrefix+"/switches/{id}/cancel", h.cancelSwitch)
	return h
}

type targetRequest struct {
	Target      quantization.Target      `json:"target"`
	Constraints quantization.Constraints `json:"constraints"`
}

type switchRequest struct {
	Target      *quantization.Target     `json:"target,omitempty"`
	To          *quantization.FormSpec   `json:"to,omitempty"`
	Constraints quantization.Constraints `json:"constraints"`
	ID          string                   `json:"id,omitempty"`
	Wait        bool                     `json:"wait,omitempty"`
}

func (h *Handler) putTarget(w http.ResponseWriter, r *http.Request) {
	collection := r.PathValue("collection")
	vector := r.PathValue("vector")
	var req targetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	decl, err := h.service.DeclareTarget(r.Context(), collection, vector, req.Target, req.Constraints)
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, decl)
}

func (h *Handler) getTarget(w http.ResponseWriter, r *http.Request) {
	decl, err := h.service.Declarations.GetDeclaration(r.PathValue("collection"), r.PathValue("vector"))
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, decl)
}

func (h *Handler) status(w http.ResponseWriter, r *http.Request) {
	st, err := h.service.Status(r.Context(), r.PathValue("collection"), r.PathValue("vector"))
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (h *Handler) evaluate(w http.ResponseWriter, r *http.Request) {
	var opts quantization.SampleOptions
	if r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&opts); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
			return
		}
	}
	report, err := h.service.Evaluations.Evaluate(r.Context(), r.PathValue("collection"), r.PathValue("vector"), opts)
	if report != nil {
		// The report is the resource; its status/recall=null distinguish failure.
		status := http.StatusOK
		if err != nil && errors.Is(err, quantization.ErrEvaluationCanceled) {
			status = 499
		} else if err != nil {
			status = http.StatusBadGateway
		}
		writeJSON(w, status, report)
		return
	}
	h.writeDomainError(w, r, err)
}

func (h *Handler) getEvaluation(w http.ResponseWriter, r *http.Request) {
	report, err := h.service.Evaluations.Get(r.PathValue("id"))
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, report)
}

func (h *Handler) startSwitch(w http.ResponseWriter, r *http.Request) {
	var req switchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	in := quantization.StartSwitchInput{
		Collection:  r.PathValue("collection"),
		Vector:      r.PathValue("vector"),
		Target:      req.Target,
		To:          req.To,
		Constraints: req.Constraints,
		ID:          req.ID,
	}
	if req.Wait {
		rec, err := h.service.Manager.RunSwitch(r.Context(), in)
		if err != nil {
			h.writeSwitchErrorWithRecord(w, r, rec, err)
			return
		}
		writeJSON(w, http.StatusOK, rec)
		return
	}
	rec, err := h.service.Manager.StartSwitch(r.Context(), in)
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, rec)
}

func (h *Handler) getSwitch(w http.ResponseWriter, r *http.Request) {
	rec, err := h.service.Manager.GetSwitch(r.PathValue("id"))
	if err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, rec)
}

func (h *Handler) cancelSwitch(w http.ResponseWriter, r *http.Request) {
	if err := h.service.Manager.Cancel(r.PathValue("id")); err != nil {
		h.writeDomainError(w, r, err)
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "canceling", "id": r.PathValue("id")})
}

func (h *Handler) writeSwitchErrorWithRecord(w http.ResponseWriter, r *http.Request, rec *quantization.SwitchRecord, err error) {
	if rec != nil {
		code := http.StatusInternalServerError
		switch rec.Phase {
		case quantization.PhaseRolledBack:
			code = http.StatusConflict
		case quantization.PhaseCanceled:
			code = 499
		case quantization.PhaseAborted:
			code = http.StatusGone
		default:
			// Pending/running/failed-without-rollback all remain 500 here.
		}
		writeJSON(w, code, map[string]any{"error": err.Error(), "record": rec})
		return
	}
	h.writeDomainError(w, r, err)
}

func (h *Handler) writeDomainError(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, quantization.ErrInvalidTarget), errors.Is(err, quantization.ErrUnsupportedCombination):
		writeError(w, http.StatusUnprocessableEntity, err.Error())
	case errors.Is(err, quantization.ErrActiveSwitch):
		writeError(w, http.StatusConflict, err.Error())
	case errors.Is(err, quantization.ErrNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, quantization.ErrCollectionGone):
		writeError(w, http.StatusGone, err.Error())
	case errors.Is(err, quantization.ErrSwitchUnavailable):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, err.Error())
	case errors.Is(err, quantization.ErrIllegalTransition):
		writeError(w, http.StatusConflict, err.Error())
	default:
		h.logger.WithField("path", r.URL.Path).Errorf("quantization management request failed: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

type errorBody struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(errorBody{Error: msg})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
