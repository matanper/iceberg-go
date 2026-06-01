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
	"github.com/google/uuid"
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

	t.Run("accepts top-level primitive shredding", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("c:$:double")
		require.NoError(t, err)
		p := cfg.ForColumn("c")
		require.NotNil(t, p)
		require.Len(t, p.Paths, 1)
		assert.Empty(t, p.Paths[0].Segments)
		assert.Equal(t, arrow.PrimitiveTypes.Float64, p.Paths[0].Type)
	})

	t.Run("rejects top-level alongside another path", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$:double, c:$.foo:int")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "conflicts")
	})

	t.Run("accepts array element shredding", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("c:$.tags[]:string")
		require.NoError(t, err)
		p := cfg.ForColumn("c")
		require.NotNil(t, p)
		require.Len(t, p.Paths, 1)
		assert.Equal(t, []string{"tags", "[]"}, p.Paths[0].Segments)
	})

	t.Run("accepts nested array elements", func(t *testing.T) {
		cfg, err := internal.ParseShreddingPaths("c:$.events[].timestamp:long")
		require.NoError(t, err)
		p := cfg.ForColumn("c")
		require.NotNil(t, p)
		require.Len(t, p.Paths, 1)
		assert.Equal(t, []string{"events", "[]", "timestamp"}, p.Paths[0].Segments)
	})

	t.Run("rejects positional array indexing", func(t *testing.T) {
		_, err := internal.ParseShreddingPaths("c:$.foo[0]:int")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "positional array indexing not supported")
	})
}

func TestShredVariantTopLevelPrimitive(t *testing.T) {
	// Schema declares the whole variant column as a shredded double.
	schema := &internal.ShreddingSchema{
		Column: "value",
		Paths:  []internal.ShreddingPath{{Segments: nil, Type: arrow.PrimitiveTypes.Float64}},
	}

	t.Run("matching primitive lands in typed", func(t *testing.T) {
		v, err := variant.Of[float64](3.14159)
		require.NoError(t, err)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.True(t, res.Root.Present)
		assert.InDelta(t, 3.14159, res.Root.Typed.(float64), 1e-9)
		assert.Nil(t, res.Root.Residual)
		assert.Nil(t, res.Root.Object)
	})

	t.Run("type mismatch routes to residual", func(t *testing.T) {
		v := mustVariant(t, `"hello"`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.True(t, res.Root.Present)
		assert.Nil(t, res.Root.Typed)
		assert.Equal(t, v.Bytes(), res.Root.Residual)
	})
}

func TestShredVariantArray(t *testing.T) {
	t.Run("array of primitives", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Column: "tags",
			Paths:  []internal.ShreddingPath{{Segments: []string{"[]"}, Type: arrow.BinaryTypes.String}},
		}
		v := mustVariant(t, `["alpha", "beta", "gamma"]`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Array)
		require.Len(t, res.Root.Array.Elements, 3)
		assert.Equal(t, "alpha", res.Root.Array.Elements[0].Typed)
		assert.Equal(t, "beta", res.Root.Array.Elements[1].Typed)
		assert.Equal(t, "gamma", res.Root.Array.Elements[2].Typed)
	})

	t.Run("array element type mismatch routes to per-element residual", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: []string{"[]"}, Type: arrow.PrimitiveTypes.Int64}},
		}
		v := mustVariant(t, `[1, "two", 3]`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Array)
		require.Len(t, res.Root.Array.Elements, 3)
		assert.EqualValues(t, 1, res.Root.Array.Elements[0].Typed)
		assert.Nil(t, res.Root.Array.Elements[1].Typed)
		assert.NotNil(t, res.Root.Array.Elements[1].Residual,
			"non-int element should land in per-path residual bytes")
		assert.EqualValues(t, 3, res.Root.Array.Elements[2].Typed)
	})

	t.Run("nested array elements (events[].timestamp)", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{
				{Segments: []string{"events", "[]", "timestamp"}, Type: arrow.PrimitiveTypes.Int64},
			},
		}
		v := mustVariant(t, `{"events": [{"timestamp": 1, "kind": "click"}, {"timestamp": 2}]}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Object)
		eventsField := res.Root.Object.Fields[0]
		require.NotNil(t, eventsField.Array)
		require.Len(t, eventsField.Array.Elements, 2)

		// Element 0: timestamp shredded, kind goes into the
		// element's residual sub-object.
		first := eventsField.Array.Elements[0]
		require.NotNil(t, first.Object)
		assert.EqualValues(t, 1, first.Object.Fields[0].Typed)
		require.NotNil(t, first.Object.ResidualValue,
			"unshredded 'kind' must land in the element's residual")

		// Element 1: only timestamp, no residual.
		second := eventsField.Array.Elements[1]
		require.NotNil(t, second.Object)
		assert.EqualValues(t, 2, second.Object.Fields[0].Typed)
		assert.Nil(t, second.Object.ResidualValue)
	})

	t.Run("non-array payload routes to residual", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: []string{"[]"}, Type: arrow.BinaryTypes.String}},
		}
		v := mustVariant(t, `"not-an-array"`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Nil(t, res.Root.Array)
		assert.Equal(t, v.Bytes(), res.Root.Residual)
		assert.True(t, res.Root.Present)
	})

	t.Run("empty array yields empty typed list", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: []string{"[]"}, Type: arrow.BinaryTypes.String}},
		}
		var b variant.Builder
		start, offsets := b.Offset(), []int{}
		require.NoError(t, b.FinishArray(start, offsets))
		v, err := b.Build()
		require.NoError(t, err)

		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Array,
			"empty array still gets an Array result (not residual)")
		assert.Empty(t, res.Root.Array.Elements,
			"typed_value list has zero elements")
		assert.True(t, res.Root.Present)
	})

	t.Run("array of variant nulls -> per-element residual", func(t *testing.T) {
		// Each element is a variant null primitive, not the array's
		// absence. Declared element type is string, so each null
		// fails extractTyped and lands in the element's Residual.
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: []string{"[]"}, Type: arrow.BinaryTypes.String}},
		}
		var b variant.Builder
		start, offsets := b.Offset(), []int{}
		offsets = append(offsets, b.NextElement(start))
		require.NoError(t, b.AppendNull())
		offsets = append(offsets, b.NextElement(start))
		require.NoError(t, b.AppendNull())
		require.NoError(t, b.FinishArray(start, offsets))
		v, err := b.Build()
		require.NoError(t, err)

		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Array)
		require.Len(t, res.Root.Array.Elements, 2)
		for i, elem := range res.Root.Array.Elements {
			assert.True(t, elem.Present, "element %d still present even when null", i)
			assert.Nil(t, elem.Typed)
			assert.NotNil(t, elem.Residual,
				"variant null at element %d must round-trip via residual bytes", i)
		}
	})
}

func TestShredVariantLeafTypeExtensions(t *testing.T) {
	t.Run("date payload extracts to arrow.Date32", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: nil, Type: arrow.FixedWidthTypes.Date32}},
		}
		var b variant.Builder
		want := arrow.Date32(19_876) // arbitrary
		require.NoError(t, b.AppendDate(want))
		v, err := b.Build()
		require.NoError(t, err)

		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.True(t, res.Root.Present)
		assert.Equal(t, want, res.Root.Typed)
	})

	t.Run("uuid payload extracts to uuid.UUID", func(t *testing.T) {
		schema := &internal.ShreddingSchema{
			Paths: []internal.ShreddingPath{{Segments: nil, Type: extensions.NewUUIDType()}},
		}
		want := uuid.MustParse("550e8400-e29b-41d4-a716-446655440000")
		var b variant.Builder
		require.NoError(t, b.AppendUUID(want))
		v, err := b.Build()
		require.NoError(t, err)

		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.True(t, res.Root.Present)
		assert.Equal(t, want, res.Root.Typed)
	})
}

func TestShredVariantVariantNullAtShreddedPath(t *testing.T) {
	schema := &internal.ShreddingSchema{
		Paths: []internal.ShreddingPath{
			{Segments: []string{"lat"}, Type: arrow.PrimitiveTypes.Float64},
		},
	}

	v := objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
		*fields = append(*fields, b.NextField(start, "lat"))
		require.NoError(t, b.AppendNull())
	})
	res, err := internal.ShredVariant(v, schema)
	require.NoError(t, err)

	lat := res.Root.Object.Fields[0]
	assert.True(t, lat.Present,
		"variant null payload still counts as present in the source object")
	assert.Nil(t, lat.Typed,
		"variant null does not satisfy a numeric leaf type")
	require.NotNil(t, lat.Residual,
		"variant null lands in per-path residual so a reader can recover it")
	assert.Nil(t, res.Root.Object.ResidualValue,
		"lat was fully consumed by the per-path slot; no outer residual")
}

func mustVariant(t *testing.T, jsonText string) variant.Value {
	t.Helper()
	v, err := variant.ParseJSON(jsonText, false)
	require.NoError(t, err)

	return v
}

// objBuilder runs fn against a variant builder pre-set up to write an
// object at the current offset, returning the finalized value. Used
// when tests need precise control over the on-wire variant type of a
// field — JSON parsing chooses the smallest matching primitive
// (e.g. it turns 1.5 into a Decimal4, not a Float64), which would
// fail to match shredding declarations of type "double".
func objBuilder(t *testing.T, fn func(b *variant.Builder, start int, fields *[]variant.FieldEntry)) variant.Value {
	t.Helper()
	var b variant.Builder
	start, fields := b.Offset(), []variant.FieldEntry{}
	fn(&b, start, &fields)
	require.NoError(t, b.FinishObject(start, fields))
	v, err := b.Build()
	require.NoError(t, err)

	return v
}

func mustField[T any](t *testing.T, b *variant.Builder, start int, fields *[]variant.FieldEntry, key string, appendFn func(*variant.Builder, T) error, value T) {
	t.Helper()
	*fields = append(*fields, b.NextField(start, key))
	require.NoError(t, appendFn(b, value))
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
		v := objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
			mustField(t, b, start, fields, "lat", (*variant.Builder).AppendFloat64, 1.5)
			mustField(t, b, start, fields, "lng", (*variant.Builder).AppendFloat64, 2.5)
			mustField(t, b, start, fields, "tier", (*variant.Builder).AppendInt, int64(3))
		})
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		assert.Nil(t, res.Root.Object.ResidualValue,
			"residual must be null when every field is shredded")
		require.Len(t, res.Root.Object.Fields, 3)
		assert.Equal(t, float64(1.5), res.Root.Object.Fields[0].Typed)
		assert.Equal(t, float64(2.5), res.Root.Object.Fields[1].Typed)
		assert.Equal(t, int32(3), res.Root.Object.Fields[2].Typed)
		for i, f := range res.Root.Object.Fields {
			assert.True(t, f.Present, "field %d should be present", i)
			assert.Nil(t, f.Residual, "field %d should not have residual storage", i)
		}
	})

	t.Run("unshredded fields land in residual", func(t *testing.T) {
		v := objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
			mustField(t, b, start, fields, "lat", (*variant.Builder).AppendFloat64, 1.5)
			mustField(t, b, start, fields, "extra", (*variant.Builder).AppendString, "hello")
			mustField(t, b, start, fields, "tier", (*variant.Builder).AppendInt, int64(7))
		})
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		require.NotNil(t, res.Root.Object.ResidualValue,
			"residual must carry the unshredded fields")

		// Reconstruct and confirm only "extra" survived.
		residual, err := variant.New(res.Metadata, res.Root.Object.ResidualValue)
		require.NoError(t, err)
		obj, ok := residual.Value().(variant.ObjectValue)
		require.True(t, ok, "residual should be an object")
		assert.Equal(t, uint32(1), obj.NumElements())
		field, err := obj.ValueByKey("extra")
		require.NoError(t, err)
		assert.Equal(t, "hello", field.Value.Value())

		assert.Equal(t, float64(1.5), res.Root.Object.Fields[0].Typed)
		assert.False(t, res.Root.Object.Fields[1].Present, "lng missing in input -> Present=false")
		assert.Equal(t, int32(7), res.Root.Object.Fields[2].Typed)
	})

	t.Run("type mismatch routes to per-path residual", func(t *testing.T) {
		// "lat" declared double but the variant value is a string.
		v := mustVariant(t, `{"lat": "not-a-number"}`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		f := res.Root.Object.Fields[0]
		assert.True(t, f.Present, "path present in input even though type mismatched")
		assert.Nil(t, f.Typed)
		require.NotNil(t, f.Residual, "type mismatch must store raw variant bytes in residual")

		assert.False(t, res.Root.Object.Fields[1].Present)
		assert.False(t, res.Root.Object.Fields[2].Present)
		assert.Nil(t, res.Root.Object.ResidualValue,
			"lat was diverted to its per-path residual; no outer residual")
	})

	t.Run("non-object input preserved as residual", func(t *testing.T) {
		v := mustVariant(t, `"top-level string"`)
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)
		// Root.Object is nil because the variant was not an object;
		// raw bytes flow into Root.Residual instead so the writer
		// stamps the outer `value` column and nulls typed_value.
		assert.Nil(t, res.Root.Object)
		assert.Equal(t, v.Bytes(), res.Root.Residual,
			"non-object inputs round-trip as Root.Residual unchanged")
		assert.True(t, res.Root.Present)
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
		v := objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
			mustField(t, b, start, fields, "lat", (*variant.Builder).AppendFloat64, 10.5)
			*fields = append(*fields, b.NextField(start, "user"))
			userStart, userFields := b.Offset(), []variant.FieldEntry{}
			mustField(t, b, userStart, &userFields, "email", (*variant.Builder).AppendString, "a@b.com")
			mustField(t, b, userStart, &userFields, "id", (*variant.Builder).AppendInt, int64(42))
			mustField(t, b, userStart, &userFields, "name", (*variant.Builder).AppendString, "Alice")
			require.NoError(t, b.FinishObject(userStart, userFields))
			mustField(t, b, start, fields, "misc", (*variant.Builder).AppendString, "kept-in-outer-residual")
		})
		res, err := internal.ShredVariant(v, schema)
		require.NoError(t, err)

		// Outer level: misc stays in outer residual; lat + user
		// fully consumed by typed_value paths.
		require.NotNil(t, res.Root.Object.ResidualValue)
		outerResidual, err := variant.New(res.Metadata, res.Root.Object.ResidualValue)
		require.NoError(t, err)
		outerObj := outerResidual.Value().(variant.ObjectValue)
		require.Equal(t, uint32(1), outerObj.NumElements())
		miscField, err := outerObj.ValueByKey("misc")
		require.NoError(t, err)
		assert.Equal(t, "kept-in-outer-residual", miscField.Value.Value())

		// Root.Fields[0] is lat -> primitive shred match.
		assert.Equal(t, float64(10.5), res.Root.Object.Fields[0].Typed)

		// Root.Fields[1] is user -> interior object node.
		userField := res.Root.Object.Fields[1]
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

		userField := res.Root.Object.Fields[1]
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
		assert.True(t, res.Root.Object.Fields[0].Present, "lat present")
		assert.False(t, res.Root.Object.Fields[1].Present, "user absent in source")
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
		objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
			mustField(t, b, start, fields, "lat", (*variant.Builder).AppendFloat64, 10.5)
			mustField(t, b, start, fields, "lng", (*variant.Builder).AppendFloat64, -20.25)
		}),
		objBuilder(t, func(b *variant.Builder, start int, fields *[]variant.FieldEntry) {
			mustField(t, b, start, fields, "lat", (*variant.Builder).AppendFloat64, 0.0)
			mustField(t, b, start, fields, "extra", (*variant.Builder).AppendString, "metadata")
		}),
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
