// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package internal_test

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/variant"
	"github.com/apache/iceberg-go/table/internal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseShreddingPaths(t *testing.T) {
	t.Run("empty string yields disabled config", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("")
		require.NoError(t, err)
		require.NotNil(t, cfg)
		assert.True(t, cfg.Empty())
		assert.Nil(t, cfg.ForColumn("anything"))
	})

	t.Run("happy path two columns three paths", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("payload:$.lat:double, payload:$.lng:double, attrs:$.tier:int")
		require.NoError(t, err)
		require.False(t, cfg.Empty())

		payload := cfg.ForColumn("payload")
		require.NotNil(t, payload)
		require.Len(t, payload.Paths, 2)
		assert.Equal(t, []string{"lat"}, payload.Paths[0].Segments)
		assert.Equal(t, arrow.PrimitiveTypes.Float64, payload.Paths[0].Type)
		assert.Equal(t, "$.lng", payload.Paths[1].Path())

		attrs := cfg.ForColumn("attrs")
		require.NotNil(t, attrs)
		require.Len(t, attrs.Paths, 1)
		assert.Equal(t, []string{"tier"}, attrs.Paths[0].Segments)
		assert.Equal(t, arrow.PrimitiveTypes.Int32, attrs.Paths[0].Type)
	})

	t.Run("accepts nested path", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("payload:$.user.email:string")
		require.NoError(t, err)
		p := cfg.ForColumn("payload")
		require.NotNil(t, p)
		require.Len(t, p.Paths, 1)
		assert.Equal(t, []string{"user", "email"}, p.Paths[0].Segments)
	})

	t.Run("rejects unknown type", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.x:decimal")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported type")
	})

	t.Run("rejects duplicate path on same column", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.x:int, c:$.x:long")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "duplicate path")
	})

	t.Run("rejects path missing dollar prefix", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:foo:int")
		require.Error(t, err)
	})

	t.Run("rejects array indexing", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.foo[0]:int")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "array indexing")
	})

	t.Run("rejects empty segment", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.a..b:int")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty segment")
	})

	t.Run("rejects prefix conflict (leaf shadows interior)", func(t *testing.T) {
		// $.user as a leaf then $.user.email would force user to be
		// both a primitive shred target AND an interior object.
		_, err := internal.ParseShreddingPaths("c:$.user:string, c:$.user.email:string")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "would shred fields under already-leaf path")
	})

	t.Run("rejects prefix conflict (interior shadowed by leaf)", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.user.email:string, c:$.user:string")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "is a prefix of already-declared path")
	})

	t.Run("allows independent sub-leaves under shared interior", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("c:$.user.email:string, c:$.user.id:long")
		require.NoError(t, err)
		require.NotNil(t, cfg.ForColumn("c"))
		assert.Len(t, cfg.ForColumn("c").Paths, 2)
	})
}

func mustVariant(t *testing.T, jsonText string) variant.Value {
	t.Helper()
	v, err := variant.ParseJSON(jsonText, false)
	require.NoError(t, err)

	return v
}

func TestShredVariantTopLevel(t *testing.T) {
	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Segments: []string{"lat"}, Type: arrow.PrimitiveTypes.Float64},
			{Segments: []string{"lng"}, Type: arrow.PrimitiveTypes.Float64},
			{Segments: []string{"tier"}, Type: arrow.PrimitiveTypes.Int32},
		},
	}

	t.Run("all paths matched leaves no residual", func(t *testing.T) {
		v := mustVariant(t, `{"lat": 1.5, "lng": 2.5, "tier": 3}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Nil(t, res.Root.ResidualValue,
			"residual must be null when every field is shredded")
		require.Len(t, res.Root.Fields, 3)
		assert.Equal(t, float64(1.5), res.Root.Fields[0].Typed)
		assert.Equal(t, float64(2.5), res.Root.Fields[1].Typed)
		assert.Equal(t, int32(3), res.Root.Fields[2].Typed)
		for i, f := range res.Root.Fields {
			assert.True(t, f.Present, "field %d should be present", i)
			assert.Nil(t, f.Residual, "field %d should not have residual storage", i)
		}
	})

	t.Run("unshredded fields land in residual", func(t *testing.T) {
		v := mustVariant(t, `{"lat": 1.5, "extra": "hello", "tier": 7}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.ResidualValue,
			"residual must carry the unshredded fields")

		// Reconstruct and confirm only "extra" survived.
		residual, err := variant.New(res.Metadata, res.Root.ResidualValue)
		require.NoError(t, err)
		obj, ok := residual.Value().(variant.ObjectValue)
		require.True(t, ok, "residual should be an object")
		assert.Equal(t, uint32(1), obj.NumElements())
		field, err := obj.ValueByKey("extra")
		require.NoError(t, err)
		assert.Equal(t, "hello", field.Value.Value())

		assert.Equal(t, float64(1.5), res.Root.Fields[0].Typed)
		assert.False(t, res.Root.Fields[1].Present, "lng missing in input -> Present=false")
		assert.Equal(t, int32(7), res.Root.Fields[2].Typed)
	})

	t.Run("type mismatch routes to per-path residual", func(t *testing.T) {
		// "lat" declared double but the variant value is a string.
		v := mustVariant(t, `{"lat": "not-a-number"}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		f := res.Root.Fields[0]
		assert.True(t, f.Present, "path present in input even though type mismatched")
		assert.Nil(t, f.Typed)
		require.NotNil(t, f.Residual, "type mismatch must store raw variant bytes in residual")

		assert.False(t, res.Root.Fields[1].Present)
		assert.False(t, res.Root.Fields[2].Present)
		assert.Nil(t, res.Root.ResidualValue,
			"lat was diverted to its per-path residual; no outer residual")
	})

	t.Run("non-object input preserved as residual", func(t *testing.T) {
		v := mustVariant(t, `"top-level string"`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Equal(t, v.Bytes(), res.Root.ResidualValue,
			"non-object inputs round-trip as residual unchanged")
		for i, f := range res.Root.Fields {
			assert.False(t, f.Present, "no shredding applies to non-object: field %d", i)
		}
	})

	t.Run("nil schema errors", func(t *testing.T) {
		v := mustVariant(t, `{}`)
		_, err := internal.ShredVariant(v, nil)
		require.Error(t, err)
	})
}

func TestShredVariantNested(t *testing.T) {
	// Two leaves under "user" plus an unrelated top-level "lat".
	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Segments: []string{"lat"}, Type: arrow.PrimitiveTypes.Float64},
			{Segments: []string{"user", "email"}, Type: arrow.BinaryTypes.String},
			{Segments: []string{"user", "id"}, Type: arrow.PrimitiveTypes.Int64},
		},
	}

	t.Run("typed value type is a nested struct", func(t *testing.T) {
		dt := internal.TypedValueArrowType(schema)
		st, ok := dt.(*arrow.StructType)
		require.True(t, ok)
		require.Equal(t, 2, st.NumFields(), "two top-level paths: lat and user")
		assert.Equal(t, "lat", st.Field(0).Name)
		assert.Equal(t, arrow.PrimitiveTypes.Float64, st.Field(0).Type)
		assert.Equal(t, "user", st.Field(1).Name)
		userStruct, ok := st.Field(1).Type.(*arrow.StructType)
		require.True(t, ok)
		require.Equal(t, 2, userStruct.NumFields())
		assert.Equal(t, "email", userStruct.Field(0).Name)
		assert.Equal(t, arrow.BinaryTypes.String, userStruct.Field(0).Type)
		assert.Equal(t, "id", userStruct.Field(1).Name)
		assert.Equal(t, arrow.PrimitiveTypes.Int64, userStruct.Field(1).Type)
	})

	t.Run("nested shred plus outer residual", func(t *testing.T) {
		v := mustVariant(t, `{
			"lat": 10.5,
			"user": {"email": "a@b.com", "id": 42, "name": "Alice"},
			"misc": "kept-in-outer-residual"
		}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		// Outer level: misc stays in outer residual; lat + user
		// fully consumed by typed_value paths.
		require.NotNil(t, res.Root.ResidualValue)
		outerResidual, err := variant.New(res.Metadata, res.Root.ResidualValue)
		require.NoError(t, err)
		outerObj := outerResidual.Value().(variant.ObjectValue)
		require.Equal(t, uint32(1), outerObj.NumElements())
		miscField, err := outerObj.ValueByKey("misc")
		require.NoError(t, err)
		assert.Equal(t, "kept-in-outer-residual", miscField.Value.Value())

		// Root.Fields[0] is lat -> primitive shred match.
		assert.Equal(t, float64(10.5), res.Root.Fields[0].Typed)

		// Root.Fields[1] is user -> interior object node.
		userField := res.Root.Fields[1]
		require.True(t, userField.Present)
		require.NotNil(t, userField.Object,
			"user is an interior object node so Object must be set")
		assert.Nil(t, userField.Typed)

		// user.email shredded, user.id shredded, user.name -> residual.
		assert.Equal(t, "a@b.com", userField.Object.Fields[0].Typed)
		assert.Equal(t, int64(42), userField.Object.Fields[1].Typed)

		require.NotNil(t, userField.Object.ResidualValue,
			"user.name was not shredded -> goes into user's residual")
		userResidual, err := variant.New(res.Metadata, userField.Object.ResidualValue)
		require.NoError(t, err)
		userResidualObj := userResidual.Value().(variant.ObjectValue)
		require.Equal(t, uint32(1), userResidualObj.NumElements())
		name, err := userResidualObj.ValueByKey("name")
		require.NoError(t, err)
		assert.Equal(t, "Alice", name.Value.Value())
	})

	t.Run("interior path with non-object payload routes to value", func(t *testing.T) {
		v := mustVariant(t, `{"user": "not-an-object"}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		userField := res.Root.Fields[1]
		require.True(t, userField.Present)
		assert.Nil(t, userField.Object,
			"non-object payload at user path leaves Object unset")
		require.NotNil(t, userField.Residual,
			"non-object payload must round-trip via per-path residual bytes")
	})

	t.Run("missing nested object means user absent at this level", func(t *testing.T) {
		v := mustVariant(t, `{"lat": 1.0}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.True(t, res.Root.Fields[0].Present, "lat present")
		assert.False(t, res.Root.Fields[1].Present, "user absent in source")
	})
}

func TestTypedValueArrowType(t *testing.T) {
	schema := &internal.ShreddingSchema{Paths: []internal.ShreddingPath{
		{Segments: []string{"lat"}, Type: arrow.PrimitiveTypes.Float64},
		{Segments: []string{"tier"}, Type: arrow.PrimitiveTypes.Int32},
	}}
	dt := internal.TypedValueArrowType(schema)
	require.NotNil(t, dt)
	st, ok := dt.(*arrow.StructType)
	require.True(t, ok)
	require.Equal(t, 2, st.NumFields())
	assert.Equal(t, "lat", st.Field(0).Name)
	assert.Equal(t, arrow.PrimitiveTypes.Float64, st.Field(0).Type)
	assert.Equal(t, "tier", st.Field(1).Name)
	assert.Equal(t, arrow.PrimitiveTypes.Int32, st.Field(1).Type)

	assert.Nil(t, internal.TypedValueArrowType(nil))
	assert.Nil(t, internal.TypedValueArrowType(&internal.ShreddingSchema{}))
}

func TestBuildShreddedVariantArrayTopLevel(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Segments: []string{"lat"}, Type: arrow.PrimitiveTypes.Float64},
			{Segments: []string{"lng"}, Type: arrow.PrimitiveTypes.Float64},
		},
	}

	rows := []variant.Value{
		mustVariant(t, `{"lat": 10.5, "lng": -20.25}`),
		mustVariant(t, `{"lat": 0.0, "extra": "metadata"}`),
		{}, // row 2 marked null below
	}

	bareArr := buildBareVariantArrayForTest(t, mem, rows, []bool{false, false, true})
	defer bareArr.Release()

	out, err := internal.BuildShreddedVariantArray(bareArr, schema, mem)
	require.NoError(t, err)
	defer out.Release()

	require.Equal(t, 3, out.Len())
	assert.True(t, out.IsNull(2))

	storageOut := out.(array.ExtensionArray).Storage().(*array.Struct)
	require.Equal(t, 3, storageOut.NumField())

	valueArr := storageOut.Field(1).(*array.Binary)
	typedArr := storageOut.Field(2).(*array.Struct)
	assert.True(t, valueArr.IsNull(0))
	require.False(t, typedArr.IsNull(0))

	latStruct := typedArr.Field(0).(*array.Struct)
	latTyped := latStruct.Field(1).(*array.Float64)
	require.False(t, latTyped.IsNull(0))
	assert.InDelta(t, 10.5, latTyped.Value(0), 1e-9)

	lngTyped := typedArr.Field(1).(*array.Struct).Field(1).(*array.Float64)
	require.False(t, lngTyped.IsNull(0))
	assert.InDelta(t, -20.25, lngTyped.Value(0), 1e-9)

	// Row 1: residual carries "extra".
	require.False(t, valueArr.IsNull(1))
	residual, err := variant.New(storageOut.Field(0).(*array.Binary).Value(1), valueArr.Value(1))
	require.NoError(t, err)
	obj := residual.Value().(variant.ObjectValue)
	field, err := obj.ValueByKey("extra")
	require.NoError(t, err)
	assert.Equal(t, "metadata", field.Value.Value())
}

func TestBuildShreddedVariantArrayNested(t *testing.T) {
	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Segments: []string{"user", "email"}, Type: arrow.BinaryTypes.String},
			{Segments: []string{"user", "id"}, Type: arrow.PrimitiveTypes.Int64},
		},
	}

	rows := []variant.Value{
		mustVariant(t, `{"user": {"email": "a@b.com", "id": 42, "name": "Alice"}}`),
	}
	bareArr := buildBareVariantArrayForTest(t, mem, rows, []bool{false})
	defer bareArr.Release()

	out, err := internal.BuildShreddedVariantArray(bareArr, schema, mem)
	require.NoError(t, err)
	defer out.Release()

	storage := out.(array.ExtensionArray).Storage().(*array.Struct)
	typedArr := storage.Field(2).(*array.Struct)

	// typed_value.user is itself a {value, typed_value}; user's
	// typed_value is the nested struct {email, id}.
	userPair := typedArr.Field(0).(*array.Struct)
	require.False(t, userPair.IsNull(0))

	userValue := userPair.Field(0).(*array.Binary)
	require.False(t, userValue.IsNull(0),
		"user.value carries the residual containing the unshredded 'name' field")
	userResidual, err := variant.New(storage.Field(0).(*array.Binary).Value(0), userValue.Value(0))
	require.NoError(t, err)
	userObj := userResidual.Value().(variant.ObjectValue)
	require.Equal(t, uint32(1), userObj.NumElements())
	name, err := userObj.ValueByKey("name")
	require.NoError(t, err)
	assert.Equal(t, "Alice", name.Value.Value())

	userTyped := userPair.Field(1).(*array.Struct)
	emailTyped := userTyped.Field(0).(*array.Struct).Field(1).(*array.String)
	assert.Equal(t, "a@b.com", emailTyped.Value(0))
	idTyped := userTyped.Field(1).(*array.Struct).Field(1).(*array.Int64)
	assert.Equal(t, int64(42), idTyped.Value(0))
}

// buildBareVariantArrayForTest constructs an unshredded VariantArray
// from the provided rows. nulls[i] true means row i is null in the
// resulting array (rows[i] is ignored).
func buildBareVariantArrayForTest(t *testing.T, mem memory.Allocator, rows []variant.Value, nulls []bool) *extensions.VariantArray {
	t.Helper()
	require.Equal(t, len(rows), len(nulls))

	ext := extensions.NewDefaultVariantType()
	structType := ext.StorageType().(*arrow.StructType)

	sb := array.NewStructBuilder(mem, structType)
	defer sb.Release()
	metaB := sb.FieldBuilder(0).(*array.BinaryBuilder)
	valueB := sb.FieldBuilder(1).(*array.BinaryBuilder)
	for i, v := range rows {
		if nulls[i] {
			sb.AppendNull()
			metaB.AppendNull()
			valueB.AppendNull()

			continue
		}
		sb.Append(true)
		metaB.Append(v.Metadata().Bytes())
		valueB.Append(v.Bytes())
	}
	storage := sb.NewArray()
	defer storage.Release()

	return array.NewExtensionArrayWithStorage(ext, storage).(*extensions.VariantArray)
}
