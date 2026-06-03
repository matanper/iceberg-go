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
	"context"
	"path/filepath"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/variant"
	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/io"
	"github.com/apache/iceberg-go/table"
	"github.com/apache/iceberg-go/table/internal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseShreddingPaths(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
		err  string
	}{
		{"empty", "", nil, ""},
		{"whitespace only", "   ", nil, ""},
		{"single", "$.foo", []string{"$.foo"}, ""},
		{"multi", "$.foo, $.bar.baz ,$.x", []string{"$.foo", "$.bar.baz", "$.x"}, ""},
		{"drops blanks", "$.foo, , $.bar", []string{"$.foo", "$.bar"}, ""},
		{"rejects non-rooted", "foo", nil, "must begin with '$'"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := internal.ParseShreddingPaths(tc.in)
			if tc.err != "" {
				require.ErrorContains(t, err, tc.err)

				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestBuildShreddingSchema_RootPrimitive(t *testing.T) {
	s, err := internal.BuildShreddingSchema([]string{"$"}, []arrow.DataType{arrow.PrimitiveTypes.Int64})
	require.NoError(t, err)
	require.False(t, s.IsEmpty())

	vt := s.VariantType()
	require.NotNil(t, vt)
	require.Equal(t, arrow.PrimitiveTypes.Int64.ID(), vt.TypedValue().Type.ID())
}

func TestBuildShreddingSchema_RejectsRootMixed(t *testing.T) {
	_, err := internal.BuildShreddingSchema(
		[]string{"$", "$.x"},
		[]arrow.DataType{arrow.PrimitiveTypes.Int64, arrow.BinaryTypes.String},
	)
	require.ErrorContains(t, err, "root path '$' cannot be combined")
}

func TestBuildShreddingSchema_ObjectFields(t *testing.T) {
	s, err := internal.BuildShreddingSchema(
		[]string{"$.event_type", "$.count"},
		[]arrow.DataType{arrow.BinaryTypes.String, arrow.PrimitiveTypes.Int64},
	)
	require.NoError(t, err)
	require.False(t, s.IsEmpty())

	tv := s.VariantType().TypedValue().Type
	st, ok := tv.(*arrow.StructType)
	require.True(t, ok, "typed_value should be a struct, got %T", tv)
	require.Equal(t, 2, st.NumFields())
	// Each child must be wrapped in {value, typed_value}.
	for i := 0; i < st.NumFields(); i++ {
		f := st.Field(i)
		child, ok := f.Type.(*arrow.StructType)
		require.True(t, ok, "child %s must be a struct, got %T", f.Name, f.Type)
		_, hasValue := child.FieldIdx("value")
		_, hasTyped := child.FieldIdx("typed_value")
		assert.True(t, hasValue, "child %s missing 'value'", f.Name)
		assert.True(t, hasTyped, "child %s missing 'typed_value'", f.Name)
	}
}

func TestBuildShreddingSchema_ConflictingPaths(t *testing.T) {
	_, err := internal.BuildShreddingSchema(
		[]string{"$.a", "$.a.b"},
		[]arrow.DataType{arrow.BinaryTypes.String, arrow.PrimitiveTypes.Int64},
	)
	require.ErrorContains(t, err, "conflicting paths share prefix")
}

func TestShredVariant_RootPrimitive(t *testing.T) {
	schema, err := internal.BuildShreddingSchema(
		[]string{"$"}, []arrow.DataType{arrow.PrimitiveTypes.Int64})
	require.NoError(t, err)

	val := mustBuildVariant(t, int64(42))
	typed, residual, err := internal.ShredVariant(val, schema)
	require.NoError(t, err)
	assert.Nil(t, residual, "primitive that matches root schema has no residual")
	assert.EqualValues(t, 42, typed)
}

func TestShredVariant_TypeMismatchFallsThroughToResidual(t *testing.T) {
	schema, err := internal.BuildShreddingSchema(
		[]string{"$"}, []arrow.DataType{arrow.PrimitiveTypes.Int64})
	require.NoError(t, err)

	val := mustBuildVariant(t, "not an int")
	typed, residual, err := internal.ShredVariant(val, schema)
	require.NoError(t, err)
	assert.Nil(t, typed, "type mismatch should not populate typed")
	assert.NotNil(t, residual, "type mismatch should land in residual")
}

func TestShredVariant_ObjectShreds_DropsExtra(t *testing.T) {
	schema, err := internal.BuildShreddingSchema(
		[]string{"$.event_type", "$.count"},
		[]arrow.DataType{arrow.BinaryTypes.String, arrow.PrimitiveTypes.Int64},
	)
	require.NoError(t, err)

	val := mustBuildVariant(t, map[string]any{
		"event_type": "click",
		"count":      int64(7),
		"extra":      "ignored",
	})
	typed, residual, err := internal.ShredVariant(val, schema)
	require.NoError(t, err)

	m, ok := typed.(map[string]any)
	require.True(t, ok, "typed should be map[string]any for object schema, got %T", typed)
	assert.Equal(t, "click", m["event_type"])
	assert.EqualValues(t, 7, m["count"])

	require.NotNil(t, residual, "extra field should land in residual bytes")
}

func TestShredVariantArray_PreservesLength(t *testing.T) {
	schema, err := internal.BuildShreddingSchema(
		[]string{"$.event_type"}, []arrow.DataType{arrow.BinaryTypes.String})
	require.NoError(t, err)

	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	bldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer bldr.Release()
	bldr.Append(mustBuildVariant(t, map[string]any{"event_type": "click"}))
	bldr.Append(mustBuildVariant(t, map[string]any{"event_type": "view"}))
	bldr.AppendNull()
	bldr.Append(mustBuildVariant(t, map[string]any{"event_type": "scroll", "extra": int64(1)}))
	src := bldr.NewArray().(*extensions.VariantArray)
	defer src.Release()

	dst, err := internal.ShredVariantArray(src, schema, mem)
	require.NoError(t, err)
	defer dst.Release()

	require.True(t, dst.IsShredded())
	require.Equal(t, src.Len(), dst.Len())

	// Reassemble round-trip: every reassembled row must equal the source row.
	reass, err := internal.ReassembleShreddedVariantColumn(dst, mem)
	require.NoError(t, err)
	defer reass.Release()
	require.Equal(t, src.Len(), reass.Len())

	for i := 0; i < src.Len(); i++ {
		if src.IsNull(i) {
			assert.True(t, reass.IsNull(i))

			continue
		}
		want, err := src.Value(i)
		require.NoError(t, err)
		got, err := reass.Value(i)
		require.NoError(t, err)
		assertVariantStructurallyEqual(t, want, got)
	}
}

// TestParquetWriter_ShredsVariantAndRoundTrips wires the public writer path
// (NewFileWriter → Write → Close) and proves three things:
//  1. With `write.variant.shredding-paths` set on WriteFileInfo, the produced
//     Parquet file carries a shredded variant column (Shredded() != nil).
//  2. The scanner-facing read path still sees a non-shredded VariantArray —
//     reassembly is invisible to consumers (existing reader test contract).
//  3. The round-tripped values equal the originals.
func TestParquetWriter_ShredsVariantAndRoundTrips(t *testing.T) {
	ctx := context.Background()
	fm := internal.GetFileFormat(iceberg.ParquetFile)

	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	icesc := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "payload", Type: iceberg.VariantType{}},
	)
	arrowSchema, err := table.SchemaToArrowSchema(icesc, nil, true, false)
	require.NoError(t, err)

	idBldr := array.NewInt64Builder(mem)
	defer idBldr.Release()
	payBldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer payBldr.Release()

	source := []variant.Value{
		mustBuildVariant(t, map[string]any{
			"event_type": "click", "count": int64(7), "extra": "drop-me",
		}),
		mustBuildVariant(t, map[string]any{
			"event_type": "view", "count": int64(11),
		}),
		mustBuildVariant(t, map[string]any{
			// event_type missing — shredded field should be null, rest stays.
			"count": int64(3), "note": "no event_type here",
		}),
	}
	for i, v := range source {
		idBldr.Append(int64(i + 1))
		payBldr.Append(v)
	}

	idArr := idBldr.NewInt64Array()
	defer idArr.Release()
	payArr := payBldr.NewArray()
	defer payArr.Release()

	rec := array.NewRecordBatch(arrowSchema, []arrow.Array{idArr, payArr}, int64(len(source)))
	defer rec.Release()

	dir := t.TempDir()
	path := filepath.Join(dir, "shredded.parquet")
	_, err = fm.WriteDataFile(ctx, io.LocalFS{}, nil, internal.WriteFileInfo{
		FileSchema:            icesc,
		FileName:              path,
		Spec:                  iceberg.PartitionSpec{},
		WriteProps:            []parquet.WriterProperty{},
		StatsCols:             shredStatsCols(icesc),
		VariantShreddingPaths: []string{"$.event_type", "$.count"},
	}, []arrow.RecordBatch{rec})
	require.NoError(t, err)

	// (1) The on-disk variant column carries a typed_value layout.
	rawArr := openVariantArray(t, path)
	defer rawArr.Release()
	require.True(t, rawArr.IsShredded(), "writer should emit a shredded variant column")
	require.NotNil(t, rawArr.Shredded(), "Shredded() should expose the typed_value array")

	// (2) The scanner path still sees a non-shredded VariantArray.
	rdr, err := fm.Open(ctx, io.LocalFS{}, path)
	require.NoError(t, err)
	defer rdr.Close()

	recs, err := rdr.GetRecords(ctx, allColumnIndices(t, path), nil)
	require.NoError(t, err)
	defer recs.Release()

	require.True(t, recs.Next())
	batch := recs.RecordBatch()
	gotVarArr := lastVariantColumn(t, batch)
	require.False(t, gotVarArr.IsShredded(),
		"scanner-facing variant column should be reassembled to non-shredded")

	// (3) Each row round-trips to a value structurally equal to the source.
	require.EqualValues(t, len(source), batch.NumRows())
	for i, want := range source {
		got, err := gotVarArr.Value(i)
		require.NoError(t, err, "row %d", i)
		assertVariantStructurallyEqual(t, want, got)
	}
}

// TestParquetWriter_NoShreddingPaths_LeavesColumnUnshredded checks the
// default posture: with no `write.variant.shredding-paths` set, the writer
// must not shred even when the column contains shred-able fields. Mirrors
// Java's "never shred unless configured" posture.
func TestParquetWriter_NoShreddingPaths_LeavesColumnUnshredded(t *testing.T) {
	ctx := context.Background()
	fm := internal.GetFileFormat(iceberg.ParquetFile)

	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	icesc := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "payload", Type: iceberg.VariantType{}},
	)
	arrowSchema, err := table.SchemaToArrowSchema(icesc, nil, true, false)
	require.NoError(t, err)

	payBldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer payBldr.Release()
	payBldr.Append(mustBuildVariant(t, map[string]any{"event_type": "click"}))
	payArr := payBldr.NewArray()
	defer payArr.Release()

	rec := array.NewRecordBatch(arrowSchema, []arrow.Array{payArr}, 1)
	defer rec.Release()

	dir := t.TempDir()
	path := filepath.Join(dir, "unshredded.parquet")
	_, err = fm.WriteDataFile(ctx, io.LocalFS{}, nil, internal.WriteFileInfo{
		FileSchema: icesc,
		FileName:   path,
		Spec:       iceberg.PartitionSpec{},
		WriteProps: []parquet.WriterProperty{},
		StatsCols:  shredStatsCols(icesc),
		// No VariantShreddingPaths.
	}, []arrow.RecordBatch{rec})
	require.NoError(t, err)

	rawArr := openVariantArray(t, path)
	defer rawArr.Release()
	assert.False(t, rawArr.IsShredded(),
		"variant column should remain non-shredded when no paths are configured")
}

// TestCrossClientFixture_IsReadable verifies the committed fixture under
// testdata/shredded_variant_write/ — the artifact for cross-client (Java,
// pyiceberg) read tests — still parses cleanly and contains the documented
// values. If this test fails, the fixture has drifted from its README and
// must be regenerated via `go run ./cmd/gen` from that directory.
func TestCrossClientFixture_IsReadable(t *testing.T) {
	ctx := context.Background()
	fm := internal.GetFileFormat(iceberg.ParquetFile)

	path := filepath.Join("testdata", "shredded_variant_write", "events.parquet")

	// (1) The on-disk variant column should be shredded.
	rawArr := openVariantArray(t, path)
	defer rawArr.Release()
	require.True(t, rawArr.IsShredded(), "committed fixture must be shredded")

	// (2) Scanner-facing view reassembles to the source values.
	rdr, err := fm.Open(ctx, io.LocalFS{}, path)
	require.NoError(t, err)
	defer rdr.Close()

	recs, err := rdr.GetRecords(ctx, allColumnIndices(t, path), nil)
	require.NoError(t, err)
	defer recs.Release()

	require.True(t, recs.Next())
	batch := recs.RecordBatch()
	require.EqualValues(t, 3, batch.NumRows())

	varArr := lastVariantColumn(t, batch)
	require.False(t, varArr.IsShredded(), "reader should hide shredding from scanner")

	expected := []map[string]any{
		{"event_type": "click", "count": int64(7), "extra": "drop-me"},
		{"event_type": "view", "count": int64(11)},
		{"count": int64(3), "note": "no event_type here"},
	}
	for i, exp := range expected {
		got, err := varArr.Value(i)
		require.NoError(t, err, "row %d", i)
		obj, ok := got.Value().(variant.ObjectValue)
		require.True(t, ok, "row %d: expected object", i)
		require.EqualValues(t, len(exp), obj.NumElements(), "row %d field count", i)
		for key, want := range exp {
			entry, err := obj.ValueByKey(key)
			require.NoError(t, err, "row %d key %q", i, key)
			gotVal := entry.Value.Value()
			if wantInt, ok := want.(int64); ok {
				gi := toInt64(gotVal)
				require.NotNil(t, gi, "row %d key %q: expected integer", i, key)
				assert.Equal(t, wantInt, *gi)

				continue
			}
			assert.Equal(t, want, gotVal, "row %d key %q", i, key)
		}
	}
}

// shredStatsCols builds a minimal StatsCols map keyed by field ID, suitable
// for tests that go through WriteDataFile. Variant columns are set to
// MetricModeNone (the stats pipeline ignores variant sub-columns regardless,
// per the spec — they have no Iceberg field IDs).
func shredStatsCols(sc *iceberg.Schema) map[int]internal.StatisticsCollector {
	out := map[int]internal.StatisticsCollector{}
	for _, f := range sc.Fields() {
		mode := internal.MetricsMode{Typ: internal.MetricModeFull}
		if _, isVariant := f.Type.(iceberg.VariantType); isVariant {
			mode = internal.MetricsMode{Typ: internal.MetricModeNone}
		}
		var prim iceberg.PrimitiveType
		if p, ok := f.Type.(iceberg.PrimitiveType); ok {
			prim = p
		}
		out[f.ID] = internal.StatisticsCollector{
			FieldID:    f.ID,
			Mode:       mode,
			ColName:    f.Name,
			IcebergTyp: prim,
		}
	}

	return out
}

func mustBuildVariant(t *testing.T, v any) variant.Value {
	t.Helper()
	var b variant.Builder
	require.NoError(t, b.Append(v))
	out, err := b.Build()
	require.NoError(t, err)

	return out
}

func TestParseShreddingSchema(t *testing.T) {
	t.Run("empty spec returns empty schema", func(t *testing.T) {
		s, err := internal.ParseShreddingSchema("")
		require.NoError(t, err)
		assert.True(t, s.IsEmpty())
	})

	t.Run("whitespace only returns empty schema", func(t *testing.T) {
		s, err := internal.ParseShreddingSchema("   ")
		require.NoError(t, err)
		assert.True(t, s.IsEmpty())
	})

	t.Run("single typed path", func(t *testing.T) {
		s, err := internal.ParseShreddingSchema("$.foo:long")
		require.NoError(t, err)
		require.False(t, s.IsEmpty())
		tv := s.VariantType().TypedValue().Type
		st, ok := tv.(*arrow.StructType)
		require.True(t, ok, "typed_value should be a struct for object shredding")
		require.Equal(t, 1, st.NumFields())
		assert.Equal(t, "foo", st.Field(0).Name)
	})

	t.Run("multiple paths with mixed types", func(t *testing.T) {
		s, err := internal.ParseShreddingSchema(" $.a:string , $.b:int , $.c:boolean ")
		require.NoError(t, err)
		require.False(t, s.IsEmpty())
		assert.ElementsMatch(t, []string{"$.a", "$.b", "$.c"}, s.Paths())
	})

	t.Run("iceberg type aliases", func(t *testing.T) {
		// 'integer' alias for int, 'bigint' alias for long, 'bool' for boolean.
		_, err := internal.ParseShreddingSchema("$.a:integer,$.b:bigint,$.c:bool")
		require.NoError(t, err)
	})

	t.Run("nested path", func(t *testing.T) {
		s, err := internal.ParseShreddingSchema("$.src.ip:string")
		require.NoError(t, err)
		require.False(t, s.IsEmpty())
	})

	t.Run("rejects unknown type", func(t *testing.T) {
		_, err := internal.ParseShreddingSchema("$.foo:bignumber")
		require.ErrorContains(t, err, "unsupported variant shredding type")
	})

	t.Run("rejects malformed entry without separator", func(t *testing.T) {
		_, err := internal.ParseShreddingSchema("$.foo")
		require.ErrorContains(t, err, "<path>:<type>")
	})

	t.Run("rejects empty type after colon", func(t *testing.T) {
		_, err := internal.ParseShreddingSchema("$.foo:")
		require.ErrorContains(t, err, "<path>:<type>")
	})

	t.Run("rejects bad path", func(t *testing.T) {
		_, err := internal.ParseShreddingSchema("foo:long")
		require.ErrorContains(t, err, "must begin with '$'")
	})

	t.Run("rejects conflicting paths", func(t *testing.T) {
		_, err := internal.ParseShreddingSchema("$.a:string,$.a.b:long")
		require.ErrorContains(t, err, "conflicting paths")
	})
}

// TestParquetWriter_DeclaredSchemaShredsEvenWhenFirstRowMissingPath
// exercises the central reason for the declared-schema property: a path
// declared in WriteVariantShreddingSchemaKey must be shredded even when
// the first sample row doesn't carry that field. With the inference path
// (WriteVariantShreddingPathsKey) the same input would drop the path
// from the file's layout for everyone.
func TestParquetWriter_DeclaredSchemaShredsEvenWhenFirstRowMissingPath(t *testing.T) {
	ctx := context.Background()
	fm := internal.GetFileFormat(iceberg.ParquetFile)

	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	icesc := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "payload", Type: iceberg.VariantType{}},
	)
	arrowSchema, err := table.SchemaToArrowSchema(icesc, nil, true, false)
	require.NoError(t, err)

	payBldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer payBldr.Release()
	// Row 0 deliberately omits "count" and "event_type" so the inference
	// path would skip both. Later rows have them.
	source := []variant.Value{
		mustBuildVariant(t, map[string]any{"unrelated": "x"}),
		mustBuildVariant(t, map[string]any{"event_type": "click", "count": int64(7)}),
		mustBuildVariant(t, map[string]any{"event_type": "view", "count": int64(11)}),
	}
	for _, v := range source {
		payBldr.Append(v)
	}
	payArr := payBldr.NewArray()
	defer payArr.Release()

	rec := array.NewRecordBatch(arrowSchema, []arrow.Array{payArr}, int64(len(source)))
	defer rec.Release()

	declared, err := internal.ParseShreddingSchema("$.event_type:string,$.count:long")
	require.NoError(t, err)
	require.False(t, declared.IsEmpty())

	dir := t.TempDir()
	path := filepath.Join(dir, "declared.parquet")
	_, err = fm.WriteDataFile(ctx, io.LocalFS{}, nil, internal.WriteFileInfo{
		FileSchema:             icesc,
		FileName:               path,
		Spec:                   iceberg.PartitionSpec{},
		WriteProps:             []parquet.WriterProperty{},
		StatsCols:              shredStatsCols(icesc),
		VariantShreddingSchema: declared,
	}, []arrow.RecordBatch{rec})
	require.NoError(t, err)

	// On-disk: shredded with both declared typed_value sub-columns present.
	rawArr := openVariantArray(t, path)
	defer rawArr.Release()
	require.True(t, rawArr.IsShredded(), "declared schema should produce a shredded layout")

	tv := rawArr.Shredded().DataType().(*arrow.StructType)
	require.Equal(t, 2, tv.NumFields(), "two declared fields → two typed_value sub-columns")
	names := []string{tv.Field(0).Name, tv.Field(1).Name}
	assert.ElementsMatch(t, []string{"event_type", "count"}, names,
		"declared layout is independent of which fields the sample row contained")

	// Round-trip: reassembled values match the source.
	rdr, err := fm.Open(ctx, io.LocalFS{}, path)
	require.NoError(t, err)
	defer rdr.Close()
	recs, err := rdr.GetRecords(ctx, allColumnIndices(t, path), nil)
	require.NoError(t, err)
	defer recs.Release()
	require.True(t, recs.Next())
	batch := recs.RecordBatch()
	gotVarArr := lastVariantColumn(t, batch)
	for i, want := range source {
		got, err := gotVarArr.Value(i)
		require.NoError(t, err, "row %d", i)
		assertVariantStructurallyEqual(t, want, got)
	}
}

// TestParquetWriter_DeclaredSchemaWinsOverPaths confirms that when both
// properties are set, the declared schema is used (and the inferred paths
// are ignored). This is the documented precedence.
func TestParquetWriter_DeclaredSchemaWinsOverPaths(t *testing.T) {
	ctx := context.Background()
	fm := internal.GetFileFormat(iceberg.ParquetFile)

	mem := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer mem.AssertSize(t, 0)

	icesc := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "payload", Type: iceberg.VariantType{}},
	)
	arrowSchema, err := table.SchemaToArrowSchema(icesc, nil, true, false)
	require.NoError(t, err)

	payBldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer payBldr.Release()
	payBldr.Append(mustBuildVariant(t, map[string]any{"a": int64(1), "b": "x"}))
	payArr := payBldr.NewArray()
	defer payArr.Release()
	rec := array.NewRecordBatch(arrowSchema, []arrow.Array{payArr}, 1)
	defer rec.Release()

	// Declared schema picks only `a`; inferred paths would also pick `b`.
	declared, err := internal.ParseShreddingSchema("$.a:long")
	require.NoError(t, err)

	dir := t.TempDir()
	path := filepath.Join(dir, "precedence.parquet")
	_, err = fm.WriteDataFile(ctx, io.LocalFS{}, nil, internal.WriteFileInfo{
		FileSchema:             icesc,
		FileName:               path,
		Spec:                   iceberg.PartitionSpec{},
		WriteProps:             []parquet.WriterProperty{},
		StatsCols:              shredStatsCols(icesc),
		VariantShreddingSchema: declared,
		VariantShreddingPaths:  []string{"$.a", "$.b"},
	}, []arrow.RecordBatch{rec})
	require.NoError(t, err)

	rawArr := openVariantArray(t, path)
	defer rawArr.Release()
	require.True(t, rawArr.IsShredded())
	tv := rawArr.Shredded().DataType().(*arrow.StructType)
	require.Equal(t, 1, tv.NumFields(),
		"declared schema (1 field) must win over inferred paths (2 fields)")
	assert.Equal(t, "a", tv.Field(0).Name)
}
