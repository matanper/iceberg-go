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

package internal

import (
	"errors"
	"fmt"
	"strings"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/decimal"
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/variant"
	"github.com/apache/iceberg-go"
)

// ShreddingPath declares one variant subfield to extract into its own
// Parquet column. v1 supports only top-level object access ($.fieldName);
// nested ($.a.b) and array ($.foo[0]) paths return an error from the parser.
type ShreddingPath struct {
	// Field is the object key (the part of "$.foo" after the dot).
	Field string
	// Type is the Arrow primitive type the writer extracts when the
	// variant payload at this path matches. Mismatched payloads fall
	// back to the per-path `value` (residual) column.
	Type arrow.DataType
}

// ShreddingSchema is the parsed shredding spec for one variant column.
// Paths preserve declaration order; the order is mirrored in the Arrow
// typed_value struct field order produced by TypedValueArrowType.
type ShreddingSchema struct {
	// Column is the iceberg field name of the variant column this spec
	// applies to.
	Column string
	Paths  []ShreddingPath
}

// ShreddingConfig collects per-column ShreddingSchema entries parsed from
// the write.variant.shredding-paths table property. Lookup by column name
// uses ForColumn.
type ShreddingConfig struct {
	byColumn map[string]*ShreddingSchema
}

// ForColumn returns the ShreddingSchema for the given iceberg field name,
// or nil if the column is not configured for shredding.
func (c *ShreddingConfig) ForColumn(name string) *ShreddingSchema {
	if c == nil {
		return nil
	}

	return c.byColumn[name]
}

// Empty reports whether no columns are configured.
func (c *ShreddingConfig) Empty() bool { return c == nil || len(c.byColumn) == 0 }

// Columns returns the iceberg field names that have shredding configured.
// Order is not stable.
func (c *ShreddingConfig) Columns() []string {
	if c == nil {
		return nil
	}
	out := make([]string, 0, len(c.byColumn))
	for k := range c.byColumn {
		out = append(out, k)
	}

	return out
}

// shreddingTypeAliases maps the type names accepted in the shredding-paths
// property to their Arrow DataType. The names match Iceberg primitive
// names (boolean, int, long, float, double, string, binary) so the
// property is portable across Java / pyiceberg / iceberg-go.
var shreddingTypeAliases = map[string]arrow.DataType{
	"boolean": arrow.FixedWidthTypes.Boolean,
	"int":     arrow.PrimitiveTypes.Int32,
	"long":    arrow.PrimitiveTypes.Int64,
	"float":   arrow.PrimitiveTypes.Float32,
	"double":  arrow.PrimitiveTypes.Float64,
	"string":  arrow.BinaryTypes.String,
	"binary":  arrow.BinaryTypes.Binary,
}

// ParseShreddingPaths parses a write.variant.shredding-paths property
// value. The format is a comma-separated list of <column>:<path>:<type>
// triples; whitespace around entries is ignored. An empty string yields
// an empty config (shredding disabled).
//
// Example: "payload:$.lat:double, payload:$.lng:double, attrs:$.tier:int".
func ParseShreddingPaths(s string) (*ShreddingConfig, error) {
	cfg := &ShreddingConfig{byColumn: map[string]*ShreddingSchema{}}
	s = strings.TrimSpace(s)
	if s == "" {
		return cfg, nil
	}

	for _, raw := range strings.Split(s, ",") {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		parts := strings.Split(entry, ":")
		if len(parts) != 3 {
			return nil, fmt.Errorf("variant shredding: expected <column>:<path>:<type>, got %q", entry)
		}
		column := strings.TrimSpace(parts[0])
		path := strings.TrimSpace(parts[1])
		typeName := strings.TrimSpace(parts[2])

		field, err := parseTopLevelPath(path)
		if err != nil {
			return nil, fmt.Errorf("variant shredding: %w (entry %q)", err, entry)
		}
		dt, ok := shreddingTypeAliases[strings.ToLower(typeName)]
		if !ok {
			return nil, fmt.Errorf("variant shredding: unsupported type %q (entry %q)", typeName, entry)
		}

		sc, ok := cfg.byColumn[column]
		if !ok {
			sc = &ShreddingSchema{Column: column}
			cfg.byColumn[column] = sc
		}
		for _, existing := range sc.Paths {
			if existing.Field == field {
				return nil, fmt.Errorf("variant shredding: duplicate path %q on column %q", path, column)
			}
		}
		sc.Paths = append(sc.Paths, ShreddingPath{Field: field, Type: dt})
	}

	return cfg, nil
}

// parseTopLevelPath returns the object key for a "$.field" path or an
// error if the path uses nested access or array indexing (out of scope
// for v1).
func parseTopLevelPath(path string) (string, error) {
	if !strings.HasPrefix(path, "$.") {
		return "", fmt.Errorf("path must start with %q, got %q", "$.", path)
	}
	rest := path[2:]
	if rest == "" {
		return "", fmt.Errorf("empty field name in path %q", path)
	}
	if strings.ContainsAny(rest, ".[") {
		return "", fmt.Errorf("nested or array paths not supported in v1, got %q", path)
	}

	return rest, nil
}

// TypedValueArrowType returns the Arrow data type used for the variant's
// typed_value subfield given a shredding spec. The shape matches what
// extensions.NewShreddedVariantType expects when called with this type.
//
// For v1 (top-level primitives) the result is a struct of one field per
// declared path, named after the JSON-path field name and typed with the
// declared primitive. arrow-go wraps each field in the {value, typed_value}
// shape required by the Parquet Variant Shredding spec.
func TypedValueArrowType(schema *ShreddingSchema) arrow.DataType {
	if schema == nil || len(schema.Paths) == 0 {
		return nil
	}
	fields := make([]arrow.Field, len(schema.Paths))
	for i, p := range schema.Paths {
		fields[i] = arrow.Field{Name: p.Field, Type: p.Type}
	}

	return arrow.StructOf(fields...)
}

// ShreddedField holds one path's per-row result.
type ShreddedField struct {
	// Typed holds the extracted primitive value when the variant at this
	// path matched the declared type. Untyped to mirror variant.Value.Value().
	// nil when no typed value was produced for this row.
	Typed any
	// Residual carries the raw variant bytes at this path when the variant
	// existed but did not match the declared type. nil otherwise.
	// (Per the Parquet spec, exactly one of Typed/Residual is non-nil
	// when the path is present; both are nil when the path is missing.)
	Residual []byte
	// Present reports whether the path existed in the input variant at
	// all. False means the row's variant did not have this field; the
	// writer encodes that as both `value` and `typed_value` null.
	Present bool
}

// ShredResult is the per-row output of ShredVariant.
type ShredResult struct {
	// Metadata is the variant metadata bytes (shared with the input;
	// the dictionary is unchanged so consumers can re-key the residual).
	Metadata []byte
	// ResidualValue holds the variant value bytes after extracting the
	// shredded paths. nil when nothing remains (the input was an object
	// whose every field was shredded — the writer encodes that as null).
	ResidualValue []byte
	// Fields is the per-path result, in the same order as the schema's
	// Paths slice. Always allocated to len(schema.Paths) so callers can
	// index by position without bounds checking.
	Fields []ShreddedField
}

// ShredVariant extracts the configured paths from a variant value,
// returning the typed values and the residual variant.
//
// v1 only supports top-level object shredding: if the input is not an
// Object, every Fields entry is marked Present=false and ResidualValue
// is the input bytes verbatim (the writer will emit a null typed_value).
// This matches the Parquet spec: when typed_value can't apply, fall
// back to value.
func ShredVariant(v variant.Value, schema *ShreddingSchema) (ShredResult, error) {
	if schema == nil {
		return ShredResult{}, errors.New("variant shredding: nil schema")
	}

	res := ShredResult{
		Metadata: v.Metadata().Bytes(),
		Fields:   make([]ShreddedField, len(schema.Paths)),
	}

	if v.Type() != variant.Object {
		res.ResidualValue = v.Bytes()

		return res, nil
	}

	obj, ok := v.Value().(variant.ObjectValue)
	if !ok {
		// Defensive: variant.Type() said Object but the cast failed.
		// Treat as non-object — preserve the value as residual.
		res.ResidualValue = v.Bytes()

		return res, nil
	}

	// Index shredded fields by name for O(1) lookup as we iterate the
	// source object.
	shreddedByField := make(map[string]int, len(schema.Paths))
	for i, p := range schema.Paths {
		shreddedByField[p.Field] = i
	}

	// First pass: split each source field into either a typed bucket or
	// the residual bucket.
	type residualField struct {
		key   string
		bytes []byte
	}
	var residualFields []residualField
	matched := 0
	for key, fieldVal := range obj.Values() {
		shredIdx, isShredded := shreddedByField[key]
		if !isShredded {
			residualFields = append(residualFields, residualField{key: key, bytes: fieldVal.Bytes()})

			continue
		}
		matched++
		typed, ok := extractTyped(fieldVal, schema.Paths[shredIdx].Type)
		if ok {
			res.Fields[shredIdx] = ShreddedField{Typed: typed, Present: true}
		} else {
			// Path is present but the variant's actual type doesn't match
			// the declared shredding type. Per the spec, store the raw
			// variant bytes in the per-path `value` (residual) column.
			res.Fields[shredIdx] = ShreddedField{
				Residual: fieldVal.Bytes(),
				Present:  true,
			}
		}
	}

	// Second pass: build the residual variant value if anything remains.
	// When every source field was shredded, leave ResidualValue nil so
	// the writer encodes the outer `value` column as null — that's how
	// the spec signals "no residual data."
	if len(residualFields) == 0 && matched > 0 {
		return res, nil
	}
	if len(residualFields) == 0 {
		// Defensive: matched==0 but no residual fields. Means the input
		// was an empty object; preserve it as residual so reconstruction
		// roundtrips.
		res.ResidualValue = v.Bytes()

		return res, nil
	}

	b := variant.NewBuilderFromMeta(v.Metadata())
	start := b.Offset()
	entries := make([]variant.FieldEntry, 0, len(residualFields))
	for _, rf := range residualFields {
		entries = append(entries, b.NextField(start, rf.key))
		if err := b.UnsafeAppendEncoded(rf.bytes); err != nil {
			return ShredResult{}, fmt.Errorf("variant shredding: append residual field %q: %w", rf.key, err)
		}
	}
	if err := b.FinishObject(start, entries); err != nil {
		return ShredResult{}, fmt.Errorf("variant shredding: finish residual object: %w", err)
	}
	built, err := b.Build()
	if err != nil {
		return ShredResult{}, fmt.Errorf("variant shredding: build residual: %w", err)
	}
	res.ResidualValue = built.Bytes()

	return res, nil
}

// extractTyped tries to read v's payload as the declared Arrow type.
// Returns (value, true) on a successful match and (nil, false) when the
// variant's runtime type is incompatible (e.g. declared double but the
// payload is a string).
//
// The variant binary format stores integers in the smallest primitive
// that fits (Int8/Int16/Int32/Int64) and may parse JSON-encoded decimals
// as DecimalValue. We perform lossless widening (Int8→Int32, Int8→Int64,
// Float32→Float64, Decimal→Float64) so a user-friendly declaration like
// "double" or "long" matches the payload regardless of which on-wire
// type the producer chose. Lossy conversions (Float64→Float32,
// Int64→Int32 of a value that overflows) intentionally fall through to
// the residual path so no precision is silently dropped.
func extractTyped(v variant.Value, want arrow.DataType) (any, bool) {
	raw := v.Value()
	switch want.ID() {
	case arrow.BOOL:
		if b, ok := raw.(bool); ok {
			return b, true
		}
	case arrow.INT32:
		switch n := raw.(type) {
		case int8:
			return int32(n), true
		case int16:
			return int32(n), true
		case int32:
			return n, true
		}
	case arrow.INT64:
		switch n := raw.(type) {
		case int8:
			return int64(n), true
		case int16:
			return int64(n), true
		case int32:
			return int64(n), true
		case int64:
			return n, true
		}
	case arrow.FLOAT32:
		if f, ok := raw.(float32); ok {
			return f, true
		}
	case arrow.FLOAT64:
		switch f := raw.(type) {
		case float32:
			return float64(f), true
		case float64:
			return f, true
		}
		if d, ok := decimalAsFloat64(raw); ok {
			return d, true
		}
	case arrow.STRING:
		if s, ok := raw.(string); ok {
			return s, true
		}
	case arrow.BINARY:
		if bs, ok := raw.([]byte); ok {
			return bs, true
		}
	}

	return nil, false
}

// decimalAsFloat64 converts a variant DecimalValue payload (any of the
// 32/64/128-bit precisions) to float64. Returns (0, false) when raw is
// not a DecimalValue. Note: 128-bit values exceeding float64's 53-bit
// significand will round; callers concerned about precision should
// declare a decimal-typed shredding column instead (v1 has none, so
// such values just fall back to residual storage).
func decimalAsFloat64(raw any) (float64, bool) {
	switch d := raw.(type) {
	case variant.DecimalValue[decimal.Decimal32]:
		return d.Value.ToFloat64(int32(d.Scale)), true
	case variant.DecimalValue[decimal.Decimal64]:
		return d.Value.ToFloat64(int32(d.Scale)), true
	case variant.DecimalValue[decimal.Decimal128]:
		return d.Value.ToFloat64(int32(d.Scale)), true
	}

	return 0, false
}

// BuildShreddedVariantArray transforms a bare variant array (with
// metadata+value storage) into a shredded variant array per the Parquet
// Variant Shredding spec. The output array's storage type is
// extensions.NewShreddedVariantType(TypedValueArrowType(schema)).
//
// Each row is shredded independently via ShredVariant; nil input rows
// (where the validity bit is unset) propagate to a null output row.
func BuildShreddedVariantArray(in *extensions.VariantArray, schema *ShreddingSchema, alloc memory.Allocator) (arrow.Array, error) {
	if alloc == nil {
		alloc = memory.DefaultAllocator
	}
	typedType := TypedValueArrowType(schema)
	if typedType == nil {
		return nil, errors.New("variant shredding: empty schema")
	}
	outType := extensions.NewShreddedVariantType(typedType)
	storageType := outType.StorageType().(*arrow.StructType)

	builder := array.NewStructBuilder(alloc, storageType)
	defer builder.Release()

	metaBuilder := builder.FieldBuilder(0).(*array.BinaryBuilder)
	valueBuilder := builder.FieldBuilder(1).(*array.BinaryBuilder)
	typedBuilder := builder.FieldBuilder(2).(*array.StructBuilder)

	perPathBuilders := make([]*array.StructBuilder, len(schema.Paths))
	for i := range schema.Paths {
		perPathBuilders[i] = typedBuilder.FieldBuilder(i).(*array.StructBuilder)
	}

	for row := 0; row < in.Len(); row++ {
		if in.IsNull(row) {
			// StructBuilder.AppendNull only flips the validity bit on the
			// parent; child builders stay at their current length. Advance
			// every leaf to keep the columns aligned with the struct.
			builder.AppendNull()
			metaBuilder.AppendNull()
			valueBuilder.AppendNull()
			typedBuilder.AppendNull()
			for _, pb := range perPathBuilders {
				pb.AppendNull()
				pb.FieldBuilder(0).AppendNull()
				pb.FieldBuilder(1).AppendNull()
			}

			continue
		}

		val, err := in.Value(row)
		if err != nil {
			return nil, fmt.Errorf("variant shredding: read row %d: %w", row, err)
		}
		shred, err := ShredVariant(val, schema)
		if err != nil {
			return nil, fmt.Errorf("variant shredding: row %d: %w", row, err)
		}

		builder.Append(true)
		metaBuilder.Append(shred.Metadata)
		if shred.ResidualValue != nil {
			valueBuilder.Append(shred.ResidualValue)
		} else {
			valueBuilder.AppendNull()
		}

		typedBuilder.Append(true)
		for i, field := range shred.Fields {
			pb := perPathBuilders[i]
			perPathValue := pb.FieldBuilder(0).(*array.BinaryBuilder)
			perPathTyped := pb.FieldBuilder(1)
			if !field.Present {
				// Path absent in this row: null at every level so the
				// reader sees "no shredded value, no residual."
				pb.AppendNull()
				perPathValue.AppendNull()
				perPathTyped.AppendNull()

				continue
			}
			pb.Append(true)
			if field.Residual != nil {
				perPathValue.Append(field.Residual)
				perPathTyped.AppendNull()
			} else {
				perPathValue.AppendNull()
				if err := appendTypedValue(perPathTyped, field.Typed); err != nil {
					return nil, fmt.Errorf("variant shredding: row %d path %q: %w", row, schema.Paths[i].Field, err)
				}
			}
		}
	}

	storage := builder.NewArray()
	defer storage.Release()

	out := array.NewExtensionArrayWithStorage(outType, storage)

	return out, nil
}

// ApplyShreddingToSchema returns a copy of s with variant columns named
// in cfg upgraded to a shredded VariantType. Top-level fields only in v1.
// When cfg is empty or no column matches, s is returned unchanged.
func ApplyShreddingToSchema(s *iceberg.Schema, cfg *ShreddingConfig) *iceberg.Schema {
	if cfg.Empty() {
		return s
	}
	changed := false
	fields := make([]iceberg.NestedField, s.NumFields())
	for i, f := range s.Fields() {
		fields[i] = f
		if _, isVariant := f.Type.(iceberg.VariantType); !isVariant {
			continue
		}
		schema := cfg.ForColumn(f.Name)
		if schema == nil {
			continue
		}
		typedType := TypedValueArrowType(schema)
		if typedType == nil {
			continue
		}
		fields[i].Type = iceberg.NewShreddedVariantType(typedType)
		changed = true
	}
	if !changed {
		return s
	}

	return iceberg.NewSchemaWithIdentifiers(s.ID, s.IdentifierFieldIDs, fields...)
}

// ShredRecordBatch returns a new RecordBatch where every variant column
// listed in cfg has been transformed from its bare metadata+value
// representation into the shredded representation prescribed by the
// schema. Columns not listed in cfg are passed through verbatim.
//
// The output Arrow schema matches the input column-for-column except
// shredded columns adopt the typed_value-bearing variant extension type.
// Use the Arrow schema returned by table.SchemaToArrowSchema(applied) on
// the shredded iceberg schema to construct a writer that consumes this
// batch.
func ShredRecordBatch(rec arrow.RecordBatch, cfg *ShreddingConfig, alloc memory.Allocator) (arrow.RecordBatch, error) {
	if cfg.Empty() {
		rec.Retain()

		return rec, nil
	}
	if alloc == nil {
		alloc = memory.DefaultAllocator
	}

	inSchema := rec.Schema()
	cols := make([]arrow.Array, rec.NumCols())
	outFields := make([]arrow.Field, rec.NumCols())
	for i, f := range inSchema.Fields() {
		outFields[i] = f
		col := rec.Column(int(i))
		shreddingSpec := cfg.ForColumn(f.Name)
		if shreddingSpec == nil {
			col.Retain()
			cols[i] = col

			continue
		}
		variantArr, ok := col.(*extensions.VariantArray)
		if !ok {
			// Column is named in the shredding config but isn't a variant
			// array. Pass through — schema check downstream will surface
			// the misconfiguration (or it's a delete-file write where
			// shredding doesn't apply).
			col.Retain()
			cols[i] = col

			continue
		}
		shredded, err := BuildShreddedVariantArray(variantArr, shreddingSpec, alloc)
		if err != nil {
			for j := range i {
				cols[j].Release()
			}

			return nil, fmt.Errorf("variant shredding: column %q: %w", f.Name, err)
		}
		cols[i] = shredded
		outFields[i] = arrow.Field{Name: f.Name, Type: shredded.DataType(), Nullable: f.Nullable, Metadata: f.Metadata}
	}

	meta := inSchema.Metadata()
	outSchema := arrow.NewSchema(outFields, &meta)
	out := array.NewRecordBatch(outSchema, cols, rec.NumRows())
	// NewRecord retains each column; release our local refs.
	for _, c := range cols {
		c.Release()
	}

	return out, nil
}

func appendTypedValue(b array.Builder, v any) error {
	switch tb := b.(type) {
	case *array.BooleanBuilder:
		tb.Append(v.(bool))
	case *array.Int32Builder:
		tb.Append(v.(int32))
	case *array.Int64Builder:
		tb.Append(v.(int64))
	case *array.Float32Builder:
		tb.Append(v.(float32))
	case *array.Float64Builder:
		tb.Append(v.(float64))
	case *array.StringBuilder:
		tb.Append(v.(string))
	case *array.BinaryBuilder:
		tb.Append(v.([]byte))
	default:
		return fmt.Errorf("unsupported typed_value builder: %T", b)
	}

	return nil
}
