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
		assert.Equal(t, "lat", payload.Paths[0].Field)
		assert.Equal(t, arrow.PrimitiveTypes.Float64, payload.Paths[0].Type)
		assert.Equal(t, "lng", payload.Paths[1].Field)

		attrs := cfg.ForColumn("attrs")
		require.NotNil(t, attrs)
		require.Len(t, attrs.Paths, 1)
		assert.Equal(t, "tier", attrs.Paths[0].Field)
		assert.Equal(t, arrow.PrimitiveTypes.Int32, attrs.Paths[0].Type)
	})

	t.Run("rejects nested path", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.a.b:string")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "nested or array paths not supported")
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
}

func mustVariant(t *testing.T, jsonText string) variant.Value {
	t.Helper()
	v, err := variant.ParseJSON(jsonText, false)
	require.NoError(t, err)

	return v
}

func TestShredVariant(t *testing.T) {
	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Field: "lat", Type: arrow.PrimitiveTypes.Float64},
			{Field: "lng", Type: arrow.PrimitiveTypes.Float64},
			{Field: "tier", Type: arrow.PrimitiveTypes.Int32},
		},
	}

	t.Run("all paths matched leaves no residual", func(t *testing.T) {
		v := mustVariant(t, `{"lat": 1.5, "lng": 2.5, "tier": 3}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Nil(t, res.ResidualValue, "residual must be null when every field is shredded")
		require.Len(t, res.Fields, 3)
		assert.Equal(t, float64(1.5), res.Fields[0].Typed)
		assert.Equal(t, float64(2.5), res.Fields[1].Typed)
		assert.Equal(t, int32(3), res.Fields[2].Typed)
		for i, f := range res.Fields {
			assert.True(t, f.Present, "field %d should be present", i)
			assert.Nil(t, f.Residual, "field %d should not have residual storage", i)
		}
	})

	t.Run("unshredded fields land in residual", func(t *testing.T) {
		v := mustVariant(t, `{"lat": 1.5, "extra": "hello", "tier": 7}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.ResidualValue, "residual must carry the unshredded fields")

		// Reconstruct and confirm only "extra" survived.
		residual, err := variant.New(res.Metadata, res.ResidualValue)
		require.NoError(t, err)
		obj, ok := residual.Value().(variant.ObjectValue)
		require.True(t, ok, "residual should be an object")
		assert.Equal(t, uint32(1), obj.NumElements())
		field, err := obj.ValueByKey("extra")
		require.NoError(t, err)
		assert.Equal(t, "hello", field.Value.Value())

		assert.Equal(t, float64(1.5), res.Fields[0].Typed)
		assert.False(t, res.Fields[1].Present, "lng missing in input -> Present=false")
		assert.Equal(t, int32(7), res.Fields[2].Typed)
	})

	t.Run("type mismatch routes to per-path residual", func(t *testing.T) {
		// "lat" declared double but the variant value is a string.
		v := mustVariant(t, `{"lat": "not-a-number"}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		f := res.Fields[0]
		assert.True(t, f.Present, "path is present in input even though type mismatched")
		assert.Nil(t, f.Typed)
		require.NotNil(t, f.Residual, "type mismatch must store raw variant bytes in residual")

		// The other declared paths weren't present at all.
		assert.False(t, res.Fields[1].Present)
		assert.False(t, res.Fields[2].Present)
		// No non-shredded fields, but lat was diverted to its per-path
		// residual so the outer residual stays nil.
		assert.Nil(t, res.ResidualValue)
	})

	t.Run("non-object input preserved as residual", func(t *testing.T) {
		v := mustVariant(t, `"top-level string"`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Equal(t, v.Bytes(), res.ResidualValue,
			"non-object inputs should round-trip as residual unchanged")
		for i, f := range res.Fields {
			assert.False(t, f.Present, "no shredding applies to non-object: field %d", i)
		}
	})

	t.Run("nil schema errors", func(t *testing.T) {
		v := mustVariant(t, `{}`)
		_, err := internal.ShredVariant(v, nil)
		require.Error(t, err)
	})
}

func TestTypedValueArrowType(t *testing.T) {
	schema := &internal.ShreddingSchema{Paths: []internal.ShreddingPath{
		{Field: "lat", Type: arrow.PrimitiveTypes.Float64},
		{Field: "tier", Type: arrow.PrimitiveTypes.Int32},
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

func TestBuildShreddedVariantArrayRoundTrip(t *testing.T) {
	alloc := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer alloc.AssertSize(t, 0)

	schema := &internal.ShreddingSchema{
		Column: "payload",
		Paths: []internal.ShreddingPath{
			{Field: "lat", Type: arrow.PrimitiveTypes.Float64},
			{Field: "lng", Type: arrow.PrimitiveTypes.Float64},
		},
	}

	// Build a bare variant array with three rows:
	//   row 0: both lat+lng present -> fully shredded, residual null
	//   row 1: only lat present, plus an extra unshredded field
	//   row 2: null variant
	rows := []variant.Value{
		mustVariant(t, `{"lat": 10.5, "lng": -20.25}`),
		mustVariant(t, `{"lat": 0.0, "extra": "metadata"}`),
		{}, // placeholder; we mark this row null below
	}

	bareType := extensions.NewDefaultVariantType()
	structType := bareType.StorageType().(*arrow.StructType)
	builder := array.NewStructBuilder(alloc, structType)
	defer builder.Release()
	metaB := builder.FieldBuilder(0).(*array.BinaryBuilder)
	valueB := builder.FieldBuilder(1).(*array.BinaryBuilder)
	for i, v := range rows {
		if i == 2 {
			builder.AppendNull()

			continue
		}
		builder.Append(true)
		metaB.Append(v.Metadata().Bytes())
		valueB.Append(v.Bytes())
	}
	storage := builder.NewArray()
	defer storage.Release()
	bareArr := array.NewExtensionArrayWithStorage(bareType, storage).(*extensions.VariantArray)
	defer bareArr.Release()

	out, err := internal.BuildShreddedVariantArray(bareArr, schema, alloc)
	require.NoError(t, err)
	defer out.Release()

	require.Equal(t, 3, out.Len())
	assert.True(t, out.IsNull(2), "null input row should produce null output row")

	storageOut := out.(array.ExtensionArray).Storage().(*array.Struct)
	require.Equal(t, 3, storageOut.NumField())

	// Row 0: lat+lng shredded, outer value should be null, typed_value populated.
	valueArr := storageOut.Field(1).(*array.Binary)
	typedArr := storageOut.Field(2).(*array.Struct)
	assert.True(t, valueArr.IsNull(0), "row 0 fully-shredded: outer value should be null")
	require.False(t, typedArr.IsNull(0))

	latStruct := typedArr.Field(0).(*array.Struct)
	latTyped := latStruct.Field(1).(*array.Float64)
	require.False(t, latTyped.IsNull(0))
	assert.InDelta(t, 10.5, latTyped.Value(0), 1e-9)

	lngTyped := typedArr.Field(1).(*array.Struct).Field(1).(*array.Float64)
	require.False(t, lngTyped.IsNull(0))
	assert.InDelta(t, -20.25, lngTyped.Value(0), 1e-9)

	// Row 1: outer value carries the unshredded "extra" field.
	require.False(t, valueArr.IsNull(1), "row 1 has unshredded fields: outer value must be non-null")
	residual, err := variant.New(storageOut.Field(0).(*array.Binary).Value(1), valueArr.Value(1))
	require.NoError(t, err)
	obj := residual.Value().(variant.ObjectValue)
	field, err := obj.ValueByKey("extra")
	require.NoError(t, err)
	assert.Equal(t, "metadata", field.Value.Value())

	// lng was missing in row 1 -> per-path entry null.
	lngPerPath := typedArr.Field(1).(*array.Struct)
	assert.True(t, lngPerPath.IsNull(1), "row 1 lng missing -> per-path null")
}
