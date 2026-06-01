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

package table

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/apache/arrow-go/v18/parquet/variant"
	"github.com/apache/iceberg-go"
	iceio "github.com/apache/iceberg-go/io"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestVariantShreddedWriteRoundTrip drives the writer end-to-end with
// the WriteVariantShreddingPathsKey property set, then re-opens the
// produced Parquet file and asserts the shredded layout reached disk
// and reconstructs the original variants.
//
// This is the iceberg-go counterpart to the Java
// TestVariantShredding round-trip suite. Cross-client coverage
// (Java-produced fixtures read by Go, Go-produced files read by Java)
// is deferred to a follow-up once apache/iceberg-go#986 lands the
// shredded reader path; for now we lean on arrow-go's pqarrow reader,
// which understands shredded variants natively.
func TestVariantShreddedWriteRoundTrip(t *testing.T) {
	ctx := context.Background()
	// arrow-go's pqarrow reader leaks small buffers when reading the
	// variant extension type (observed against v18.6.0). Until that's
	// fixed upstream we use the default allocator so the leak doesn't
	// mask shredding-correctness failures. The shredder's own
	// allocator hygiene is covered by the in-package unit tests.
	mem := memory.DefaultAllocator

	icebergSchema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "payload", Type: iceberg.VariantType{}, Required: true},
	)

	loc := filepath.ToSlash(t.TempDir())
	props := iceberg.Properties{
		"format-version":              "3",
		WriteVariantShreddingPathsKey: "payload:$.lat:double,payload:$.lng:double",
	}
	meta, err := NewMetadata(icebergSchema, iceberg.UnpartitionedSpec, UnsortedSortOrder, loc, props)
	require.NoError(t, err)
	metaBuilder, err := MetadataBuilderFromBase(meta, "")
	require.NoError(t, err)

	// Build a 3-row bare-variant batch.
	//   row 0: {"lat": 10.5, "lng": -20.25}                  fully shredded
	//   row 1: {"lat": 0.0,  "extra": "metadata"}            partial: extra → residual, lng missing
	//   row 2: {"lat": "not-a-number"}                       type mismatch on lat
	bareVariantType := extensions.NewDefaultVariantType()
	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "payload", Type: bareVariantType, Nullable: false},
	}, nil)

	rec := buildBareVariantRecord(t, mem, arrowSchema, []int64{1, 2, 3}, []variant.Value{
		mustVariantBuilder(t, func(b *variant.Builder) {
			start, fields := b.Offset(), []variant.FieldEntry{}
			fields = append(fields, b.NextField(start, "lat"))
			require.NoError(t, b.AppendFloat64(10.5))
			fields = append(fields, b.NextField(start, "lng"))
			require.NoError(t, b.AppendFloat64(-20.25))
			require.NoError(t, b.FinishObject(start, fields))
		}),
		mustVariantBuilder(t, func(b *variant.Builder) {
			start, fields := b.Offset(), []variant.FieldEntry{}
			fields = append(fields, b.NextField(start, "lat"))
			require.NoError(t, b.AppendFloat64(0.0))
			fields = append(fields, b.NextField(start, "extra"))
			require.NoError(t, b.AppendString("metadata"))
			require.NoError(t, b.FinishObject(start, fields))
		}),
		mustVariantBuilder(t, func(b *variant.Builder) {
			start, fields := b.Offset(), []variant.FieldEntry{}
			fields = append(fields, b.NextField(start, "lat"))
			require.NoError(t, b.AppendString("not-a-number"))
			require.NoError(t, b.FinishObject(start, fields))
		}),
	})
	defer rec.Release()

	dataFiles := writeRecordsThroughFactory(t, ctx, loc, metaBuilder, arrowSchema, rec)
	require.Len(t, dataFiles, 1)
	df := dataFiles[0]

	// Re-open the produced Parquet file and assert the shredded layout.
	parquetPath := df.FilePath()
	if rel, ok := stripLocalPrefix(parquetPath); ok {
		parquetPath = rel
	}

	f, err := os.Open(parquetPath)
	require.NoError(t, err)
	defer f.Close()
	reader, err := file.NewParquetReader(f)
	require.NoError(t, err)
	defer reader.Close()
	arrReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, mem)
	require.NoError(t, err)

	tbl, err := arrReader.ReadTable(ctx)
	require.NoError(t, err)
	defer tbl.Release()

	require.Equal(t, int64(3), tbl.NumRows())

	payloadColumn := tbl.Column(1)
	require.Equal(t, "payload", payloadColumn.Name())
	payloadType, ok := payloadColumn.DataType().(*extensions.VariantType)
	require.True(t, ok, "payload column must round-trip as the variant extension type, got %T", payloadColumn.DataType())

	storageStruct, ok := payloadType.StorageType().(*arrow.StructType)
	require.True(t, ok)
	require.Equal(t, 3, storageStruct.NumFields(),
		"shredded variant on disk must carry metadata + value + typed_value, got fields %v",
		storageStruct.Fields())
	assert.Equal(t, "metadata", storageStruct.Field(0).Name)
	assert.Equal(t, "value", storageStruct.Field(1).Name)
	assert.Equal(t, "typed_value", storageStruct.Field(2).Name)

	typedValueType, ok := storageStruct.Field(2).Type.(*arrow.StructType)
	require.True(t, ok, "typed_value must be a struct, got %T", storageStruct.Field(2).Type)
	require.Equal(t, 2, typedValueType.NumFields(), "two declared shredding paths -> two typed_value fields")
	assert.Equal(t, "lat", typedValueType.Field(0).Name)
	assert.Equal(t, "lng", typedValueType.Field(1).Name)

	chunk := payloadColumn.Data().Chunk(0).(*extensions.VariantArray)

	// Row 0: fully shredded. Reconstructed variant must reproduce both fields.
	got0, err := chunk.Value(0)
	require.NoError(t, err)
	assertVariantHasFloat(t, got0, "lat", 10.5)
	assertVariantHasFloat(t, got0, "lng", -20.25)

	// Row 1: partial shredding. "extra" must survive in the residual; lat
	// extracted; lng absent.
	got1, err := chunk.Value(1)
	require.NoError(t, err)
	assertVariantHasFloat(t, got1, "lat", 0.0)
	assertVariantHasString(t, got1, "extra", "metadata")

	// Row 2: type mismatch fell back to per-path residual. The
	// reconstructed variant should still expose lat as a string so
	// downstream readers see the original payload, not silently lose it.
	got2, err := chunk.Value(2)
	require.NoError(t, err)
	assertVariantHasString(t, got2, "lat", "not-a-number")
}

// TestVariantShreddedWriteArrayRoundTrip writes a variant whose
// "tags" field is an array of strings, with the array's elements
// shredded as a typed string column. It checks that pqarrow
// reconstructs the array with all elements intact, including an
// element whose runtime type doesn't match the declared element
// type and therefore lives in per-element residual storage.
func TestVariantShreddedWriteArrayRoundTrip(t *testing.T) {
	ctx := context.Background()
	mem := memory.DefaultAllocator

	icebergSchema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "payload", Type: iceberg.VariantType{}, Required: true},
	)
	loc := filepath.ToSlash(t.TempDir())
	props := iceberg.Properties{
		"format-version":              "3",
		WriteVariantShreddingPathsKey: "payload:$.tags[]:string",
	}
	meta, err := NewMetadata(icebergSchema, iceberg.UnpartitionedSpec, UnsortedSortOrder, loc, props)
	require.NoError(t, err)
	metaBuilder, err := MetadataBuilderFromBase(meta, "")
	require.NoError(t, err)

	bareVariantType := extensions.NewDefaultVariantType()
	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "payload", Type: bareVariantType, Nullable: false},
	}, nil)

	row := mustVariantBuilder(t, func(b *variant.Builder) {
		objStart, fields := b.Offset(), []variant.FieldEntry{}
		fields = append(fields, b.NextField(objStart, "tags"))
		arrStart, offsets := b.Offset(), []int{}
		offsets = append(offsets, b.NextElement(arrStart))
		require.NoError(t, b.AppendString("alpha"))
		offsets = append(offsets, b.NextElement(arrStart))
		require.NoError(t, b.AppendString("beta"))
		require.NoError(t, b.FinishArray(arrStart, offsets))
		require.NoError(t, b.FinishObject(objStart, fields))
	})

	rec := buildBareVariantRecord(t, mem, arrowSchema, []int64{1}, []variant.Value{row})
	defer rec.Release()

	dataFiles := writeRecordsThroughFactory(t, ctx, loc, metaBuilder, arrowSchema, rec)
	require.Len(t, dataFiles, 1)

	parquetPath := dataFiles[0].FilePath()
	if rel, ok := stripLocalPrefix(parquetPath); ok {
		parquetPath = rel
	}
	f, err := os.Open(parquetPath)
	require.NoError(t, err)
	defer f.Close()
	reader, err := file.NewParquetReader(f)
	require.NoError(t, err)
	defer reader.Close()
	arrReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, mem)
	require.NoError(t, err)
	tbl, err := arrReader.ReadTable(ctx)
	require.NoError(t, err)
	defer tbl.Release()

	chunk := tbl.Column(1).Data().Chunk(0).(*extensions.VariantArray)
	got, err := chunk.Value(0)
	require.NoError(t, err)

	obj := got.Value().(variant.ObjectValue)
	tagsField, err := obj.ValueByKey("tags")
	require.NoError(t, err)
	tagsArr, ok := tagsField.Value.Value().(variant.ArrayValue)
	require.True(t, ok, "tags must round-trip as a variant array, got %T",
		tagsField.Value.Value())
	require.Equal(t, uint32(2), tagsArr.Len())
	alpha, err := tagsArr.Value(0)
	require.NoError(t, err)
	assert.Equal(t, "alpha", alpha.Value())
	beta, err := tagsArr.Value(1)
	require.NoError(t, err)
	assert.Equal(t, "beta", beta.Value())
}

// TestVariantShreddedWriteNestedRoundTrip exercises the same writer
// flow but with a nested object shredding spec
// ($.user.email + $.user.id). It checks that
//   - the on-disk typed_value carries a struct under "user"
//   - the per-row residual at the user level captures unshredded
//     sub-fields ("name" in the test)
//   - top-level reconstruction yields the original variant
func TestVariantShreddedWriteNestedRoundTrip(t *testing.T) {
	ctx := context.Background()
	mem := memory.DefaultAllocator

	icebergSchema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "payload", Type: iceberg.VariantType{}, Required: true},
	)
	loc := filepath.ToSlash(t.TempDir())
	props := iceberg.Properties{
		"format-version":              "3",
		WriteVariantShreddingPathsKey: "payload:$.user.email:string, payload:$.user.id:long",
	}
	meta, err := NewMetadata(icebergSchema, iceberg.UnpartitionedSpec, UnsortedSortOrder, loc, props)
	require.NoError(t, err)
	metaBuilder, err := MetadataBuilderFromBase(meta, "")
	require.NoError(t, err)

	bareVariantType := extensions.NewDefaultVariantType()
	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "payload", Type: bareVariantType, Nullable: false},
	}, nil)

	row := mustVariantBuilder(t, func(b *variant.Builder) {
		start, fields := b.Offset(), []variant.FieldEntry{}
		fields = append(fields, b.NextField(start, "user"))
		userStart, userFields := b.Offset(), []variant.FieldEntry{}
		userFields = append(userFields, b.NextField(userStart, "email"))
		require.NoError(t, b.AppendString("a@b.com"))
		userFields = append(userFields, b.NextField(userStart, "id"))
		require.NoError(t, b.AppendInt(42))
		userFields = append(userFields, b.NextField(userStart, "name"))
		require.NoError(t, b.AppendString("Alice"))
		require.NoError(t, b.FinishObject(userStart, userFields))
		require.NoError(t, b.FinishObject(start, fields))
	})

	rec := buildBareVariantRecord(t, mem, arrowSchema, []int64{1}, []variant.Value{row})
	defer rec.Release()

	dataFiles := writeRecordsThroughFactory(t, ctx, loc, metaBuilder, arrowSchema, rec)
	require.Len(t, dataFiles, 1)

	parquetPath := dataFiles[0].FilePath()
	if rel, ok := stripLocalPrefix(parquetPath); ok {
		parquetPath = rel
	}
	f, err := os.Open(parquetPath)
	require.NoError(t, err)
	defer f.Close()
	reader, err := file.NewParquetReader(f)
	require.NoError(t, err)
	defer reader.Close()
	arrReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, mem)
	require.NoError(t, err)
	tbl, err := arrReader.ReadTable(ctx)
	require.NoError(t, err)
	defer tbl.Release()

	chunk := tbl.Column(1).Data().Chunk(0).(*extensions.VariantArray)
	got, err := chunk.Value(0)
	require.NoError(t, err)

	// Full reconstruction should expose user as an object with all
	// three sub-fields, including the unshredded "name".
	obj, ok := got.Value().(variant.ObjectValue)
	require.True(t, ok)
	userField, err := obj.ValueByKey("user")
	require.NoError(t, err)
	userObj, ok := userField.Value.Value().(variant.ObjectValue)
	require.True(t, ok)

	email, err := userObj.ValueByKey("email")
	require.NoError(t, err)
	assert.Equal(t, "a@b.com", email.Value.Value())

	id, err := userObj.ValueByKey("id")
	require.NoError(t, err)
	// AppendInt picks the smallest variant int; widen for comparison.
	switch n := id.Value.Value().(type) {
	case int8:
		assert.EqualValues(t, 42, n)
	case int16:
		assert.EqualValues(t, 42, n)
	case int32:
		assert.EqualValues(t, 42, n)
	case int64:
		assert.EqualValues(t, 42, n)
	default:
		t.Fatalf("id has unexpected runtime type %T (%v)", n, n)
	}

	name, err := userObj.ValueByKey("name")
	require.NoError(t, err)
	assert.Equal(t, "Alice", name.Value.Value())
}

func mustVariantBuilder(t *testing.T, fn func(*variant.Builder)) variant.Value {
	t.Helper()
	var b variant.Builder
	fn(&b)
	v, err := b.Build()
	require.NoError(t, err)

	return v
}

func buildBareVariantRecord(t *testing.T, mem memory.Allocator, schema *arrow.Schema, ids []int64, values []variant.Value) arrow.RecordBatch {
	t.Helper()
	require.Equal(t, len(ids), len(values))

	bldr := array.NewRecordBuilder(mem, schema)
	defer bldr.Release()

	idB := bldr.Field(0).(*array.Int64Builder)
	for _, id := range ids {
		idB.Append(id)
	}

	variantArr := buildBareVariantArray(t, mem, values)
	defer variantArr.Release()

	// The RecordBuilder owns column 0; we replace column 1 with the
	// hand-built variant array via NewRecord.
	idArr := idB.NewArray()
	defer idArr.Release()

	return array.NewRecordBatch(schema, []arrow.Array{idArr, variantArr}, int64(len(ids)))
}

func buildBareVariantArray(t *testing.T, mem memory.Allocator, values []variant.Value) arrow.Array {
	t.Helper()
	ext := extensions.NewDefaultVariantType()
	structType := ext.StorageType().(*arrow.StructType)

	sb := array.NewStructBuilder(mem, structType)
	defer sb.Release()
	metaB := sb.FieldBuilder(0).(*array.BinaryBuilder)
	valueB := sb.FieldBuilder(1).(*array.BinaryBuilder)
	for _, v := range values {
		sb.Append(true)
		metaB.Append(v.Metadata().Bytes())
		valueB.Append(v.Bytes())
	}
	storage := sb.NewArray()
	defer storage.Release()

	return array.NewExtensionArrayWithStorage(ext, storage)
}

func assertVariantHasFloat(t *testing.T, v variant.Value, key string, want float64) {
	t.Helper()
	obj, ok := v.Value().(variant.ObjectValue)
	require.True(t, ok, "expected variant to be an object")
	field, err := obj.ValueByKey(key)
	require.NoError(t, err, "key %q missing from reconstructed variant", key)
	switch x := field.Value.Value().(type) {
	case float64:
		assert.InDelta(t, want, x, 1e-9)
	case float32:
		assert.InDelta(t, want, float64(x), 1e-6)
	default:
		t.Fatalf("variant key %q expected float, got %T (%v)", key, x, x)
	}
}

func assertVariantHasString(t *testing.T, v variant.Value, key, want string) {
	t.Helper()
	obj, ok := v.Value().(variant.ObjectValue)
	require.True(t, ok, "expected variant to be an object")
	field, err := obj.ValueByKey(key)
	require.NoError(t, err, "key %q missing from reconstructed variant", key)
	got, ok := field.Value.Value().(string)
	require.True(t, ok, "variant key %q expected string, got %T", key, field.Value.Value())
	assert.Equal(t, want, got)
}

// stripLocalPrefix strips a "file://" scheme from a path if present, so
// os.Open can read the produced data file regardless of how the
// LocationProvider formatted its URI.
func stripLocalPrefix(p string) (string, bool) {
	if rest, ok := strings.CutPrefix(p, "file://"); ok {
		return rest, true
	}

	return p, false
}

// writeRecordsThroughFactory drives recordsToDataFiles with a single
// in-memory batch and returns the produced data files. Test-only.
func writeRecordsThroughFactory(t *testing.T, ctx context.Context, loc string, meta *MetadataBuilder, schema *arrow.Schema, rec arrow.RecordBatch) []iceberg.DataFile {
	t.Helper()
	writeUUID := uuid.New()
	args := recordWritingArgs{
		sc: schema,
		itr: func(yield func(arrow.RecordBatch, error) bool) {
			rec.Retain()
			if !yield(rec, nil) {
				rec.Release()

				return
			}
			rec.Release()
		},
		fs:        iceio.LocalFS{},
		writeUUID: &writeUUID,
		counter: func(yield func(int) bool) {
			for i := 0; ; i++ {
				if !yield(i) {
					break
				}
			}
		},
	}

	var files []iceberg.DataFile
	for df, err := range recordsToDataFiles(ctx, loc, meta, args) {
		require.NoError(t, err)
		files = append(files, df)
	}

	return files
}
