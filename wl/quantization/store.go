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
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// Store persists switch records so startup can determine the phase and form
// without consulting in-process state.
type Store interface {
	SaveSwitch(*SwitchRecord) error
	GetSwitch(id string) (*SwitchRecord, error)
	ListSwitches() ([]*SwitchRecord, error)
}

// FileStore stores one JSON record per switch under Root.
type FileStore struct {
	root string
}

// NewFileStore creates root if needed.
func NewFileStore(root string) (*FileStore, error) {
	if strings.TrimSpace(root) == "" {
		return nil, errors.New("quantization state directory is empty")
	}
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, fmt.Errorf("create quantization state dir %q: %w", root, err)
	}
	return &FileStore{root: root}, nil
}

func (s *FileStore) path(id string) string {
	return filepath.Join(s.root, sanitizeID(id)+".json")
}

// SaveSwitch atomically writes the record (temp file + fsync + rename).
func (s *FileStore) SaveSwitch(r *SwitchRecord) error {
	if r == nil || r.ID == "" {
		return errors.New("cannot save nil or ID-less switch record")
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal switch %s: %w", r.ID, err)
	}
	final := s.path(r.ID)
	tmp := final + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("create switch state temp file: %w", err)
	}
	if _, err := f.Write(b); err != nil {
		_ = f.Close()
		return fmt.Errorf("write switch state: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return fmt.Errorf("fsync switch state: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close switch state: %w", err)
	}
	if err := os.Rename(tmp, final); err != nil {
		return fmt.Errorf("rename switch state: %w", err)
	}
	dir, err := os.Open(s.root)
	if err == nil {
		_ = dir.Sync()
		_ = dir.Close()
	}
	return nil
}

// GetSwitch reads one record.
func (s *FileStore) GetSwitch(id string) (*SwitchRecord, error) {
	b, err := os.ReadFile(s.path(id))
	if err != nil {
		return nil, err
	}
	var r SwitchRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("parse switch state %s: %w", id, err)
	}
	return &r, nil
}

// ListSwitches returns records sorted by creation time then ID.
func (s *FileStore) ListSwitches() ([]*SwitchRecord, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return nil, err
	}
	out := make([]*SwitchRecord, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(s.root, e.Name()))
		if err != nil {
			return nil, err
		}
		var r SwitchRecord
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("parse %s: %w", e.Name(), err)
		}
		out = append(out, &r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID < out[j].ID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	return out, nil
}

// InMemoryStore is convenient for tests and single-process deployments that
// do not require crash recovery.
type InMemoryStore struct {
	mu      sync.RWMutex
	records map[string]*SwitchRecord
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{records: map[string]*SwitchRecord{}}
}

func (s *InMemoryStore) SaveSwitch(r *SwitchRecord) error {
	if r == nil || r.ID == "" {
		return errors.New("cannot save nil or ID-less switch record")
	}
	cp := cloneSwitch(r)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records[r.ID] = &cp
	return nil
}

func (s *InMemoryStore) GetSwitch(id string) (*SwitchRecord, error) {
	s.mu.RLock()
	r, ok := s.records[id]
	s.mu.RUnlock()
	if !ok {
		return nil, fs.ErrNotExist
	}
	cp := cloneSwitch(r)
	return &cp, nil
}

func (s *InMemoryStore) ListSwitches() ([]*SwitchRecord, error) {
	s.mu.RLock()
	ids := make([]string, 0, len(s.records))
	for id := range s.records {
		ids = append(ids, id)
	}
	s.mu.RUnlock()
	out := make([]*SwitchRecord, 0, len(ids))
	for _, id := range ids {
		r, err := s.GetSwitch(id)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.Before(out[j].CreatedAt) })
	return out, nil
}

func cloneSwitch(r *SwitchRecord) SwitchRecord {
	cp := *r
	cp.Units = make(map[string]*UnitState, len(r.Units))
	for k, u := range r.Units {
		ucp := *u
		cp.Units[k] = &ucp
	}
	return cp
}

func sanitizeID(id string) string {
	id = strings.ReplaceAll(id, "/", "_")
	id = strings.ReplaceAll(id, "\\", "_")
	id = strings.ReplaceAll(id, "..", "_")
	return id
}
