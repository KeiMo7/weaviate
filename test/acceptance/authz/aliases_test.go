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

package authz

import (
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/client/schema"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/test/helper"
	"github.com/weaviate/weaviate/test/helper/sample-schema/articles"
	"github.com/weaviate/weaviate/usecases/auth/authorization"
)

func TestAuthzAliases(t *testing.T) {
	adminKey := "admin-key"

	customUser := "custom-user"
	customKey := "custom-key"
	limitedUser := "limited-user"
	limitedKey := "limited-key"
	customRole := "custom"
	onlyAliasesStartingWithAliasRole := "only-aliases"

	_, down := composeUpShared(t)
	defer down()

	clsA := articles.ArticlesClass()

	helper.CreateClassAuth(t, articles.ParagraphsClass(), adminKey)
	helper.CreateClassAuth(t, clsA, adminKey)
	helper.CreateObjectsBatchAuth(t, []*models.Object{articles.NewArticle().WithTitle("article1").Object()}, adminKey)

	alias1 := "AliasParagraph1"
	alias2 := "AliasParagraph2"
	alias3 := "AliasParagraph3"
	otherAlias1 := "OtherAliasForParagraph1"

	for _, alias := range []string{alias1, alias2, alias3, otherAlias1} {
		helper.CreateAliasAuth(t, &models.Alias{Alias: alias, Class: articles.ParagraphsClass().Class}, adminKey)
	}

	// make custom role with read_aliases to read all aliases
	helper.CreateRole(t, adminKey, &models.Role{Name: &customRole, Permissions: []*models.Permission{
		helper.NewAliasesPermission().WithAction(authorization.ReadAliases).Permission(),
	}})

	// create a role that is able to list aliases starting with Alias*
	helper.CreateRole(t, adminKey, &models.Role{Name: &onlyAliasesStartingWithAliasRole, Permissions: []*models.Permission{
		helper.NewAliasesPermission().WithAction(authorization.ReadAliases).WithAlias("Alias*").Permission(),
	}})

	t.Run("fail to get aliaes without minimal read_aliases", func(t *testing.T) {
		resp, err := helper.Client(t).Schema.AliasesGet(schema.NewAliasesGetParams(), helper.CreateAuth(customKey))
		require.Nil(t, err)
		require.NotNil(t, resp)
		require.NotNil(t, resp.Payload)
		assert.Len(t, resp.Payload.Aliases, 0)
	})

	t.Run("assign custom role to custom user", func(t *testing.T) {
		helper.AssignRoleToUser(t, adminKey, customRole, customUser)
	})

	t.Run("get minimal aliases with read_aliases", func(t *testing.T) {
		resp, err := helper.Client(t).Schema.AliasesGet(schema.NewAliasesGetParams(), helper.CreateAuth(customKey))
		require.Nil(t, err)
		require.Len(t, resp.Payload.Aliases, 4)
	})

	t.Run("assign only aliases role to limited user and try to get aliases", func(t *testing.T) {
		helper.AssignRoleToUser(t, adminKey, onlyAliasesStartingWithAliasRole, limitedUser)
		resp, err := helper.Client(t).Schema.AliasesGet(schema.NewAliasesGetParams(), helper.CreateAuth(limitedKey))
		require.Nil(t, err)
		require.Len(t, resp.Payload.Aliases, 3)
	})

	t.Run("to get one alias with onlyAliasesStartingWithAliasRole role", func(t *testing.T) {
		params := schema.NewAliasesGetAliasParams().WithAliasName(alias1)
		resp, err := helper.Client(t).Schema.AliasesGetAlias(params, helper.CreateAuth(limitedKey))
		require.Nil(t, err)
		require.NotNil(t, resp.Payload)
	})
}

// TestAuthzAliasesHideForbiddenAlias pins that a caller without alias
// permissions cannot tell an existing alias from a missing one.
func TestAuthzAliasesHideForbiddenAlias(t *testing.T) {
	adminKey := "admin-key"
	noPermKey := "custom-key"
	readerUser := "read-user"
	readerKey := "read-key"
	readerRole := "alias-reader"

	compose, down := composeUpShared(t)
	defer down()

	target := articles.ParagraphsClass()
	helper.CreateClassAuth(t, target, adminKey)
	existing := "PublicView"
	missing := "Payroll"
	helper.CreateAliasAuth(t, &models.Alias{Alias: existing, Class: target.Class}, adminKey)

	do := func(t *testing.T, method, path, key string) (int, string) {
		t.Helper()
		req, err := http.NewRequest(method, fmt.Sprintf("http://%s/v1%s", compose.GetWeaviate().URI(), path), nil)
		require.NoError(t, err)
		req.Header.Set("Authorization", "Bearer "+key)
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		require.NoError(t, err)
		return resp.StatusCode, string(body)
	}

	tests := []struct {
		name         string
		method       string
		existingPath string
		missingPath  string
		wantStatus   int
	}{
		{
			name:         "get alias",
			method:       http.MethodGet,
			existingPath: "/aliases/" + existing,
			missingPath:  "/aliases/" + missing,
			wantStatus:   http.StatusNotFound,
		},
		{
			name:         "delete alias",
			method:       http.MethodDelete,
			existingPath: "/aliases/" + existing,
			missingPath:  "/aliases/" + missing,
			wantStatus:   http.StatusNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			existingStatus, existingBody := do(t, tt.method, tt.existingPath, noPermKey)
			missingStatus, missingBody := do(t, tt.method, tt.missingPath, noPermKey)

			assert.Equal(t, tt.wantStatus, missingStatus)
			assert.Equal(t, missingStatus, existingStatus)
			assert.Equal(t, missingBody, existingBody)
			assert.NotContains(t, existingBody, target.Class)
			helper.GetAliasWithAuthz(t, existing, helper.CreateAuth(adminKey))
		})
	}

	scopedReaders := []struct {
		name string
		user string
		key  string
		perm *models.Permission
	}{
		{
			name: "reader scoped to other aliases",
			user: "limited-user",
			key:  "limited-key",
			perm: helper.NewAliasesPermission().WithAction(authorization.ReadAliases).WithAlias("Other*").Permission(),
		},
		{
			name: "reader scoped to another collection",
			user: "write-user",
			key:  "write-key",
			perm: helper.NewAliasesPermission().WithAction(authorization.ReadAliases).WithCollection("OtherCollection").Permission(),
		},
	}
	for _, sr := range scopedReaders {
		t.Run(sr.name, func(t *testing.T) {
			role := "scoped-" + sr.user
			helper.CreateRole(t, adminKey, &models.Role{Name: &role, Permissions: []*models.Permission{sr.perm}})
			helper.AssignRoleToUser(t, adminKey, role, sr.user)

			for _, method := range []string{http.MethodGet, http.MethodDelete} {
				existingStatus, existingBody := do(t, method, "/aliases/"+existing, sr.key)
				missingStatus, missingBody := do(t, method, "/aliases/"+missing, sr.key)

				assert.Equal(t, http.StatusNotFound, missingStatus, method)
				assert.Equal(t, missingStatus, existingStatus, method)
				assert.Equal(t, missingBody, existingBody, method)
			}
			helper.GetAliasWithAuthz(t, existing, helper.CreateAuth(adminKey))
		})
	}

	t.Run("delete by a caller who may read the alias is forbidden", func(t *testing.T) {
		helper.CreateRole(t, adminKey, &models.Role{Name: &readerRole, Permissions: []*models.Permission{
			helper.NewAliasesPermission().WithAction(authorization.ReadAliases).Permission(),
		}})
		helper.AssignRoleToUser(t, adminKey, readerRole, readerUser)

		status, _ := do(t, http.MethodDelete, "/aliases/"+existing, readerKey)
		assert.Equal(t, http.StatusForbidden, status)
		status, _ = do(t, http.MethodDelete, "/aliases/"+missing, readerKey)
		assert.Equal(t, http.StatusNotFound, status)
		helper.GetAliasWithAuthz(t, existing, helper.CreateAuth(adminKey))
	})
}
