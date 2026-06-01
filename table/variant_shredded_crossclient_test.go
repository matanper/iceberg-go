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
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
	"github.com/apache/arrow-go/v18/parquet/variant"
	"github.com/apache/iceberg-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestShreddedVariantReadsApacheParquetTestingFixtures verifies that
// arrow-go's pqarrow reader (the same reader iceberg-go uses
// transitively) correctly reconstructs each Java-produced shredded
// variant from the canonical apache/parquet-testing fixtures, and
// that the reconstruction matches the reference variant bytes
// committed alongside each Parquet file.
//
// This is the read-side half of the cross-client coverage requested
// by apache/iceberg-go#987. The write-side half
// (TestShreddedVariantWriteRoundTripsObjectFixture below) closes the
// loop by writing a similar variant through iceberg-go's shredded
// writer and asserting the output reads back to the same value via
// the same pqarrow reader path that the fixtures themselves prove
// out.
func TestShreddedVariantReadsApacheParquetTestingFixtures(t *testing.T) {
	cases := []struct {
		name        string
		parquetFile string
		variantFile string
	}{
		{"boolean primitive (case-004)", "case-004.parquet", "case-004_row-0.variant.bin"},
		{"int8 primitive (case-006)", "case-006.parquet", "case-006_row-0.variant.bin"},
		{"int64 primitive (case-012)", "case-012.parquet", "case-012_row-0.variant.bin"},
		{"object with null + empty string (case-046)", "case-046.parquet", "case-046_row-0.variant.bin"},
	}

	ctx := context.Background()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fixture := readShreddedFixture(t,
				filepath.Join("internal", "testdata", "shredded_variant", tc.parquetFile))
			defer fixture.Release()

			expected := readVariantBin(t,
				filepath.Join("internal", "testdata", "shredded_variant", tc.variantFile))

			got, err := fixture.Value(0)
			require.NoError(t, err)

			assertVariantStructurallyEqual(t, expected, got)
			_ = ctx
		})
	}
}

// TestShreddedVariantWriteRoundTripsObjectFixture takes case-046's
// canonical variant ({a: null, b: ""}), shreds it through
// iceberg-go's writer with $.a:string and $.b:string, reads the
// produced Parquet back via pqarrow, and asserts the reconstructed
// variant equals the original. This proves the write-side path
// produces Parquet that is compatible with the same readers Java and
// pyiceberg rely on.
func TestShreddedVariantWriteRoundTripsObjectFixture(t *testing.T) {
	ctx := context.Background()
	mem := memory.DefaultAllocator

	source := readVariantBin(t,
		filepath.Join("internal", "testdata", "shredded_variant", "case-046_row-0.variant.bin"))

	icebergSchema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "var", Type: iceberg.VariantType{}, Required: true},
	)
	loc := filepath.ToSlash(t.TempDir())
	props := iceberg.Properties{
		"format-version":              "3",
		WriteVariantShreddingPathsKey: "var:$.a:string, var:$.b:string",
	}
	meta, err := NewMetadata(icebergSchema, iceberg.UnpartitionedSpec, UnsortedSortOrder, loc, props)
	require.NoError(t, err)
	metaBuilder, err := MetadataBuilderFromBase(meta, "")
	require.NoError(t, err)

	bareVariantType := extensions.NewDefaultVariantType()
	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int64, Nullable: false},
		{Name: "var", Type: bareVariantType, Nullable: false},
	}, nil)

	rec := buildBareVariantRecord(t, mem, arrowSchema, []int64{1}, []variant.Value{source})
	defer rec.Release()

	dataFiles := writeRecordsThroughFactory(t, ctx, loc, metaBuilder, arrowSchema, rec)
	require.Len(t, dataFiles, 1)

	produced := openParquet(t, ctx, mem, dataFiles[0].FilePath())
	defer produced.Release()

	got, err := produced.Value(0)
	require.NoError(t, err)

	assertVariantStructurallyEqual(t, source, got)
}

// readShreddedFixture opens a vendored .parquet fixture and returns
// the variant column's first chunk. The caller is responsible for
// Release().
func readShreddedFixture(t *testing.T, path string) *extensions.VariantArray {
	t.Helper()

	return openParquet(t, context.Background(), memory.DefaultAllocator, path)
}

// openParquet opens a Parquet file and returns the variant column's
// first chunk as a *extensions.VariantArray. Path may be a relative
// path or a file:// URI as returned by a LocationProvider.
func openParquet(t *testing.T, ctx context.Context, mem memory.Allocator, path string) *extensions.VariantArray {
	t.Helper()
	if rel, ok := stripLocalPrefix(path); ok {
		path = rel
	}
	f, err := os.Open(path)
	require.NoError(t, err, "open %s", path)
	t.Cleanup(func() { _ = f.Close() })

	reader, err := file.NewParquetReader(f)
	require.NoError(t, err)
	t.Cleanup(func() { _ = reader.Close() })

	arrReader, err := pqarrow.NewFileReader(reader, pqarrow.ArrowReadProperties{}, mem)
	require.NoError(t, err)

	tbl, err := arrReader.ReadTable(ctx)
	require.NoError(t, err)
	t.Cleanup(tbl.Release)

	// In the apache/parquet-testing fixtures the variant column is
	// named "var"; in our own end-to-end test it's "payload". Pick
	// the last column either way — both fixtures have a leading
	// "id" column.
	col := tbl.Column(int(tbl.NumCols()) - 1)
	chunk := col.Data().Chunk(0)
	v, ok := chunk.(*extensions.VariantArray)
	require.True(t, ok, "expected VariantArray, got %T", chunk)
	v.Retain()

	return v
}

// readVariantBin parses a .variant.bin (concatenated metadata||value
// per apache/parquet-testing's encoding) into a variant.Value. The
// split logic mirrors arrow-go's pqarrow variant_test.go:117-130 so
// the fixture format stays in lockstep with the upstream test data.
func readVariantBin(t *testing.T, path string) variant.Value {
	t.Helper()
	data, err := os.ReadFile(path)
	require.NoError(t, err, "read %s", path)
	require.NotEmpty(t, data)

	hdr := data[0]
	offsetSize := int(1 + ((hdr & 0b11000000) >> 6))
	dictSize := int(readVariantUnsigned(data[1 : 1+offsetSize]))
	offsetListOffset := 1 + offsetSize
	dataOffset := offsetListOffset + ((1 + dictSize) * offsetSize)

	idx := offsetListOffset + (offsetSize * dictSize)
	endOffset := dataOffset + int(readVariantUnsigned(data[idx:idx+offsetSize]))
	v, err := variant.New(data[:endOffset], data[endOffset:])
	require.NoError(t, err, "parse %s", path)

	return v
}

// readVariantUnsigned mirrors arrow-go's internal little-endian
// unsigned read used by the variant metadata parser. Lengths in
// variant metadata are 1-4 bytes wide depending on the dictionary's
// offsetSize field; we zero-extend to fit the largest.
func readVariantUnsigned(b []byte) uint64 {
	var buf [8]byte
	copy(buf[:], b)

	return binary.LittleEndian.Uint64(buf[:])
}

// assertVariantStructurallyEqual compares two variant values by
// walking their object/array shape and comparing primitive payloads.
// Mirrors arrow-go's assertVariantEqual to stay consistent with the
// upstream test data semantics — variant.Bytes comparisons are
// fragile (re-encoded values can differ even when semantically
// equal), so we walk the structure instead.
func assertVariantStructurallyEqual(t *testing.T, expected, actual variant.Value) {
	t.Helper()
	switch expected.BasicType() {
	case variant.BasicObject:
		exp := expected.Value().(variant.ObjectValue)
		act, ok := actual.Value().(variant.ObjectValue)
		require.True(t, ok, "expected object, got %T", actual.Value())
		require.Equal(t, exp.NumElements(), act.NumElements())
		for i := range exp.NumElements() {
			expField, err := exp.FieldAt(i)
			require.NoError(t, err)
			actField, err := act.ValueByKey(expField.Key)
			require.NoError(t, err, "key %q missing in actual", expField.Key)
			assertVariantStructurallyEqual(t, expField.Value, actField.Value)
		}
	case variant.BasicArray:
		exp := expected.Value().(variant.ArrayValue)
		act, ok := actual.Value().(variant.ArrayValue)
		require.True(t, ok, "expected array")
		require.Equal(t, exp.Len(), act.Len())
		for i := range exp.Len() {
			expVal, err := exp.Value(i)
			require.NoError(t, err)
			actVal, err := act.Value(i)
			require.NoError(t, err)
			assertVariantStructurallyEqual(t, expVal, actVal)
		}
	default:
		assert.Equal(t, expected.Value(), actual.Value(),
			"primitive mismatch: expected %v (%T), got %v (%T)",
			expected.Value(), expected.Value(), actual.Value(), actual.Value())
	}
}
