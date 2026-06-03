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
	"github.com/apache/arrow-go/v18/arrow/extensions"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet/variant"
)

// WriteVariantShreddingPathsKey is the table property carrying the
// comma-separated list of JSON-path expressions (rooted at "$") that select
// which variant fields to shred. The Arrow leaf type for each path is
// inferred per file from the first non-null variant value, which is fragile
// for heterogeneous schemas: a path absent from that one sample falls back
// to the residual `value` column for the entire file.
//
// Empty (the default) means no shredding. When both this property and
// [WriteVariantShreddingSchemaKey] are set, the declared schema wins.
const WriteVariantShreddingPathsKey = "write.variant.shredding-paths"

// WriteVariantShreddingSchemaKey is the table property carrying a fully
// declared shredding schema as comma-separated `<path>:<iceberg-type>`
// entries, e.g. `$.event_type:string,$.count:long,$.src.ip:string`.
//
// Unlike [WriteVariantShreddingPathsKey], this skips per-file type inference
// entirely: every file the writer produces uses the declared layout, so the
// typed_value sub-columns and their leaf types are stable across the whole
// table. Use this when the application already knows the schema it wants
// shredded (e.g. a fixed event schema) and wants reliable cross-file pruning
// and predicate pushdown.
//
// Supported type names follow Iceberg primitive naming: `boolean`, `int`,
// `long`, `float`, `double`, `string`, `binary`, `date`, `timestamp`,
// `timestamptz`. Decimal and other types can be added as needed.
//
// Single global property — the same declared schema applies to every variant
// column in the table. Per-column scoping is a future extension.
const WriteVariantShreddingSchemaKey = "write.variant.shredding-schema"

// ShreddingSchema describes the typed-column layout for one variant column.
// Build via InferShreddingSchema (paths + a sample value), or BuildShreddingSchema
// when the Arrow types are already known.
//
// A zero ShreddingSchema means "do not shred" — callers should leave the
// column as a non-shredded *extensions.VariantType.
type ShreddingSchema struct {
	// paths are the original $.path expressions, kept for diagnostics.
	paths []string
	// variantType is the arrow-go shredded VariantType (3-field storage with
	// typed_value populated per the Parquet variant shredding spec).
	variantType *extensions.VariantType
}

// IsEmpty reports whether the schema configures any shredding at all.
func (s ShreddingSchema) IsEmpty() bool { return s.variantType == nil }

// Paths returns the original $.path expressions (for diagnostics).
func (s ShreddingSchema) Paths() []string { return s.paths }

// VariantType returns the arrow-go *extensions.VariantType used to materialize
// shredded columns. Nil when the schema is empty.
func (s ShreddingSchema) VariantType() *extensions.VariantType { return s.variantType }

// ParseShreddingPaths splits a property value (comma-separated) into trimmed
// path strings. Whitespace-only entries are dropped; an empty/blank spec
// returns nil. Each path must start with "$".
func ParseShreddingPaths(spec string) ([]string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, nil
	}
	parts := strings.Split(spec, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if !strings.HasPrefix(p, "$") {
			return nil, fmt.Errorf("variant shredding path %q must begin with '$'", p)
		}
		out = append(out, p)
	}

	return out, nil
}

// ParseShreddingSchema parses a [WriteVariantShreddingSchemaKey] property
// value into a fully-resolved ShreddingSchema. The format is comma-separated
// `<path>:<iceberg-type>` entries; an empty spec returns an empty schema
// (IsEmpty == true).
//
// Example:
//
//	$.event_type:string, $.count:long, $.src.ip:string
//
// Unknown type names, malformed paths, prefix conflicts (e.g. `$.a` and
// `$.a.b`), or root `$` mixed with field paths all surface as errors so
// misconfiguration fails fast at table-property reconcile time.
func ParseShreddingSchema(spec string) (ShreddingSchema, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return ShreddingSchema{}, nil
	}
	parts := strings.Split(spec, ",")
	paths := make([]string, 0, len(parts))
	types := make([]arrow.DataType, 0, len(parts))
	for _, raw := range parts {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		// Split on the LAST ':' so paths that one day allow ':' don't trip us.
		idx := strings.LastIndex(entry, ":")
		if idx <= 0 || idx == len(entry)-1 {
			return ShreddingSchema{}, fmt.Errorf(
				"variant shredding schema entry %q must be of the form '<path>:<type>'", entry)
		}
		path := strings.TrimSpace(entry[:idx])
		typeName := strings.TrimSpace(entry[idx+1:])
		dt, err := icebergTypeNameToArrow(typeName)
		if err != nil {
			return ShreddingSchema{}, fmt.Errorf(
				"variant shredding schema entry %q: %w", entry, err)
		}
		paths = append(paths, path)
		types = append(types, dt)
	}
	if len(paths) == 0 {
		return ShreddingSchema{}, nil
	}

	return BuildShreddingSchema(paths, types)
}

// icebergTypeNameToArrow maps an Iceberg primitive type name to the Arrow
// leaf type the Parquet shredded typed_value column should carry.
//
// Only types that have a meaningful variant primitive correspondence are
// supported (the same coercion rules arrow-go's VariantBuilder applies).
// Returning an error keeps misconfiguration from silently producing a layout
// the writer can't actually populate.
func icebergTypeNameToArrow(name string) (arrow.DataType, error) {
	switch strings.ToLower(name) {
	case "boolean", "bool":
		return arrow.FixedWidthTypes.Boolean, nil
	case "int8":
		return arrow.PrimitiveTypes.Int8, nil
	case "int16", "short":
		return arrow.PrimitiveTypes.Int16, nil
	case "int", "int32", "integer":
		return arrow.PrimitiveTypes.Int32, nil
	case "long", "int64", "bigint":
		return arrow.PrimitiveTypes.Int64, nil
	case "float", "float32":
		return arrow.PrimitiveTypes.Float32, nil
	case "double", "float64":
		return arrow.PrimitiveTypes.Float64, nil
	case "string", "varchar":
		return arrow.BinaryTypes.String, nil
	case "binary":
		return arrow.BinaryTypes.Binary, nil
	case "date":
		return arrow.FixedWidthTypes.Date32, nil
	case "timestamp":
		return &arrow.TimestampType{Unit: arrow.Microsecond}, nil
	case "timestamptz":
		return &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "UTC"}, nil
	default:
		return nil, fmt.Errorf("unsupported variant shredding type %q", name)
	}
}

// splitPath parses a JSON path like "$", "$.foo", "$.foo.bar" into its
// dot-separated field segments. The leading "$" is dropped; "$" alone
// returns an empty slice (root primitive shredding).
//
// Bracketed field names (e.g. `$['weird.key']`) are not yet supported.
func splitPath(p string) ([]string, error) {
	if !strings.HasPrefix(p, "$") {
		return nil, fmt.Errorf("variant shredding path %q must begin with '$'", p)
	}
	rest := p[1:]
	if rest == "" {
		return nil, nil
	}
	if !strings.HasPrefix(rest, ".") {
		return nil, fmt.Errorf("variant shredding path %q must use dot notation (got %q)", p, rest)
	}
	rest = rest[1:]
	if rest == "" {
		return nil, fmt.Errorf("variant shredding path %q is malformed", p)
	}
	segments := strings.Split(rest, ".")
	for _, s := range segments {
		if s == "" {
			return nil, fmt.Errorf("variant shredding path %q has empty segment", p)
		}
		if strings.ContainsAny(s, "[]'\"") {
			return nil, fmt.Errorf("variant shredding path %q: bracketed segments not yet supported", p)
		}
	}

	return segments, nil
}

// BuildShreddingSchema constructs a schema from already-resolved field types.
// paths and leafTypes must be the same length; each leafTypes[i] is the Arrow
// type the typed_value column should carry for paths[i].
//
// All paths must reference distinct fields. A single root path "$" must be the
// only entry (root primitive shredding cannot be mixed with field shredding).
func BuildShreddingSchema(paths []string, leafTypes []arrow.DataType) (ShreddingSchema, error) {
	if len(paths) != len(leafTypes) {
		return ShreddingSchema{}, fmt.Errorf(
			"variant shredding: %d paths but %d leaf types", len(paths), len(leafTypes))
	}
	if len(paths) == 0 {
		return ShreddingSchema{}, nil
	}

	resolved := make([]shredEntry, 0, len(paths))
	rootOnly := false
	for i, p := range paths {
		segs, err := splitPath(p)
		if err != nil {
			return ShreddingSchema{}, err
		}
		if len(segs) == 0 {
			rootOnly = true
		}
		resolved = append(resolved, shredEntry{segments: segs, leaf: leafTypes[i]})
	}
	if rootOnly {
		if len(resolved) != 1 {
			return ShreddingSchema{}, errors.New(
				"variant shredding: root path '$' cannot be combined with other paths")
		}
		vt := extensions.NewShreddedVariantType(resolved[0].leaf)

		return ShreddingSchema{paths: append([]string(nil), paths...), variantType: vt}, nil
	}

	dt, err := buildShreddedStruct(resolved)
	if err != nil {
		return ShreddingSchema{}, err
	}
	vt := extensions.NewShreddedVariantType(dt)

	return ShreddingSchema{paths: append([]string(nil), paths...), variantType: vt}, nil
}

// shredEntry pairs a path's remaining segments with its leaf Arrow type.
type shredEntry struct {
	segments []string
	leaf     arrow.DataType
}

// buildShreddedStruct assembles an arrow.StructType whose fields mirror the
// shredding paths. Conflicting paths (e.g. "$.a" and "$.a.b") return an error.
func buildShreddedStruct(entries []shredEntry) (arrow.DataType, error) {
	type group struct {
		direct *arrow.DataType // non-nil if this segment is itself a leaf
		nested []shredEntry
	}
	order := make([]string, 0)
	groups := make(map[string]*group)
	for _, e := range entries {
		key := e.segments[0]
		g, ok := groups[key]
		if !ok {
			g = &group{}
			groups[key] = g
			order = append(order, key)
		}
		if len(e.segments) == 1 {
			if g.direct != nil || len(g.nested) > 0 {
				return nil, fmt.Errorf(
					"variant shredding: conflicting paths share prefix %q", key)
			}
			leaf := e.leaf
			g.direct = &leaf
		} else {
			if g.direct != nil {
				return nil, fmt.Errorf(
					"variant shredding: conflicting paths share prefix %q", key)
			}
			g.nested = append(g.nested, shredEntry{segments: e.segments[1:], leaf: e.leaf})
		}
	}

	fields := make([]arrow.Field, 0, len(order))
	for _, name := range order {
		g := groups[name]
		if g.direct != nil {
			fields = append(fields, arrow.Field{Name: name, Type: *g.direct})

			continue
		}
		child, err := buildShreddedStruct(g.nested)
		if err != nil {
			return nil, err
		}
		fields = append(fields, arrow.Field{Name: name, Type: child})
	}

	return arrow.StructOf(fields...), nil
}

// InferShreddingSchema picks an Arrow leaf type for each path by walking the
// given variant.Value, then defers to BuildShreddingSchema. Paths that don't
// resolve to a primitive in the sample (missing, null, or wrong shape) are
// dropped from the resulting schema — those rows will fall through to the
// residual value column at write time.
//
// A nil schema (IsEmpty) is returned when no path matches the sample.
//
// TODO: this is a single-sample inference. Java's VariantShreddingAnalyzer
// (parquet/.../VariantShreddingAnalyzer.java) instead walks a buffered window
// of rows, builds a path tree, prunes paths below a frequency threshold,
// caps the kept set, and picks the most-common type per field with integer/
// decimal widening + tie-break priority. Porting that analyzer is the
// follow-up that brings the inference path to parity with Java. Today,
// callers needing stable cross-file layouts should pre-declare types via
// WriteVariantShreddingSchemaKey instead of relying on this function.
func InferShreddingSchema(paths []string, sample variant.Value) (ShreddingSchema, error) {
	if len(paths) == 0 {
		return ShreddingSchema{}, nil
	}
	keptPaths := make([]string, 0, len(paths))
	keptTypes := make([]arrow.DataType, 0, len(paths))
	for _, p := range paths {
		segs, err := splitPath(p)
		if err != nil {
			return ShreddingSchema{}, err
		}
		v, ok := resolvePath(sample, segs)
		if !ok {
			continue
		}
		dt, ok := arrowLeafFromVariant(v)
		if !ok {
			continue
		}
		keptPaths = append(keptPaths, p)
		keptTypes = append(keptTypes, dt)
	}
	if len(keptPaths) == 0 {
		return ShreddingSchema{}, nil
	}

	return BuildShreddingSchema(keptPaths, keptTypes)
}

// resolvePath walks segs against v, returning the value at that path. Returns
// ok=false when any segment is missing or v isn't an object along the way.
// An empty segs slice resolves to v itself.
func resolvePath(v variant.Value, segs []string) (variant.Value, bool) {
	cur := v
	for _, s := range segs {
		if cur.BasicType() != variant.BasicObject {
			return variant.Value{}, false
		}
		obj, ok := cur.Value().(variant.ObjectValue)
		if !ok {
			return variant.Value{}, false
		}
		child, err := obj.ValueByKey(s)
		if err != nil {
			return variant.Value{}, false
		}
		cur = child.Value
	}

	return cur, true
}

// arrowLeafFromVariant maps a variant primitive value to its natural Arrow
// type. Returns ok=false for non-primitive values (objects, arrays, null) and
// for variant types arrow-go's VariantBuilder can't shred to a leaf.
func arrowLeafFromVariant(v variant.Value) (arrow.DataType, bool) {
	switch v.Type() {
	case variant.Bool:
		return arrow.FixedWidthTypes.Boolean, true
	case variant.Int8:
		return arrow.PrimitiveTypes.Int8, true
	case variant.Int16:
		return arrow.PrimitiveTypes.Int16, true
	case variant.Int32:
		return arrow.PrimitiveTypes.Int32, true
	case variant.Int64:
		return arrow.PrimitiveTypes.Int64, true
	case variant.Float:
		return arrow.PrimitiveTypes.Float32, true
	case variant.Double:
		return arrow.PrimitiveTypes.Float64, true
	case variant.String:
		return arrow.BinaryTypes.String, true
	case variant.Binary:
		return arrow.BinaryTypes.Binary, true
	case variant.Date:
		return arrow.FixedWidthTypes.Date32, true
	case variant.TimestampMicros:
		return &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "UTC"}, true
	case variant.TimestampMicrosNTZ:
		return &arrow.TimestampType{Unit: arrow.Microsecond}, true
	}

	return nil, false
}

// ShredVariant shreds a single variant value against schema. Internally it
// uses arrow-go's VariantBuilder — the same implementation arrow-go uses when
// constructing shredded VariantArrays — so the result matches the Parquet
// Variant shredding spec byte-for-byte.
//
// Returns:
//   - typed:    the typed_value content for the row, as Go-native data. For
//     root primitive shredding this is the primitive itself; for
//     object shredding it is a map[string]any keyed by field name.
//     Nil when nothing in the value matched the schema.
//   - residual: the bytes for the "value" column (the unshredded fields plus
//     anything that couldn't be coerced to a typed column). Nil
//     when value matched the schema exactly.
//
// The schema's typed_value Arrow type drives the coercion: the value is
// considered "shredded" for a leaf only when its variant primitive type
// matches the leaf's Arrow type per arrow-go's coercion rules. Mismatches
// fall back to the residual.
func ShredVariant(value variant.Value, schema ShreddingSchema) (typed any, residual []byte, err error) {
	if schema.IsEmpty() {
		return nil, value.Bytes(), nil
	}

	defer func() {
		if r := recover(); r != nil {
			typed, residual = nil, nil
			err = fmt.Errorf("variant shredder panicked: %v", r)
		}
	}()

	mem := memory.DefaultAllocator
	bldr := extensions.NewVariantBuilder(mem, schema.variantType)
	defer bldr.Release()

	bldr.Append(value)
	arr := bldr.NewArray().(*extensions.VariantArray)
	defer arr.Release()

	storage := arr.Storage().(*array.Struct)
	vt := schema.variantType
	st := vt.StorageType().(*arrow.StructType)
	valIdx, _ := st.FieldIdx("value")
	typedIdx, _ := st.FieldIdx("typed_value")

	valCol := storage.Field(valIdx).(*array.Binary)
	if !valCol.IsNull(0) {
		residual = append([]byte(nil), valCol.Value(0)...)
	}

	typedCol := storage.Field(typedIdx)
	if !typedCol.IsNull(0) {
		typed = goValueFromArray(typedCol, 0)
	}

	return typed, residual, nil
}

// goValueFromArray returns a Go-native representation of arr[i] for arrays
// produced by VariantBuilder's typed_value branch. Recursively unpacks the
// {value, typed_value} structs that the shredded spec uses to represent
// objects and arrays.
func goValueFromArray(arr arrow.Array, i int) any {
	if arr.IsNull(i) {
		return nil
	}
	switch a := arr.(type) {
	case *array.Boolean:
		return a.Value(i)
	case *array.Int8:
		return a.Value(i)
	case *array.Int16:
		return a.Value(i)
	case *array.Int32:
		return a.Value(i)
	case *array.Int64:
		return a.Value(i)
	case *array.Float32:
		return a.Value(i)
	case *array.Float64:
		return a.Value(i)
	case *array.String:
		return a.Value(i)
	case *array.Binary:
		return append([]byte(nil), a.Value(i)...)
	case *array.Date32:
		return a.Value(i)
	case *array.Timestamp:
		return a.Value(i)
	case *array.Struct:
		st := a.DataType().(*arrow.StructType)
		out := make(map[string]any, st.NumFields())
		for f := 0; f < st.NumFields(); f++ {
			field := st.Field(f)
			child := a.Field(f)
			if field.Type.ID() == arrow.STRUCT {
				if cs, ok := child.(*array.Struct); ok {
					out[field.Name] = unwrapShreddedField(cs, i)

					continue
				}
			}
			out[field.Name] = goValueFromArray(child, i)
		}

		return out
	}

	return nil
}

// unwrapShreddedField returns the user-visible value for one {value,
// typed_value} child of a shredded object. typed_value wins when present.
func unwrapShreddedField(s *array.Struct, i int) any {
	if s.IsNull(i) {
		return nil
	}
	st := s.DataType().(*arrow.StructType)
	typedIdx, hasTyped := st.FieldIdx("typed_value")
	valueIdx, hasValue := st.FieldIdx("value")
	if hasTyped {
		typedCol := s.Field(typedIdx)
		if !typedCol.IsNull(i) {
			return goValueFromArray(typedCol, i)
		}
	}
	if hasValue {
		valCol := s.Field(valueIdx)
		if !valCol.IsNull(i) {
			// Return the residual bytes as []byte so callers can see "field
			// exists, but didn't shred to a typed column."
			if b, ok := valCol.(*array.Binary); ok {
				return append([]byte(nil), b.Value(i)...)
			}
		}
	}

	return nil
}

// ShredVariantArray rebuilds a non-shredded *extensions.VariantArray as a
// shredded one with the given schema. Each element is fed through arrow-go's
// VariantBuilder so the produced array can be written directly into the
// Parquet variant shredded layout.
//
// When schema.IsEmpty(), arr is returned unchanged (with a Retain so callers
// can use a uniform Release pattern).
func ShredVariantArray(arr *extensions.VariantArray, schema ShreddingSchema, mem memory.Allocator) (*extensions.VariantArray, error) {
	if schema.IsEmpty() {
		arr.Retain()

		return arr, nil
	}
	if mem == nil {
		mem = memory.DefaultAllocator
	}

	bldr := extensions.NewVariantBuilder(mem, schema.variantType)
	defer bldr.Release()
	bldr.Reserve(arr.Len())

	for i := 0; i < arr.Len(); i++ {
		if arr.IsNull(i) {
			bldr.AppendNull()

			continue
		}
		v, err := arr.Value(i)
		if err != nil {
			return nil, fmt.Errorf("variant shredder reading row %d: %w", i, err)
		}
		bldr.Append(v)
	}

	out := bldr.NewArray().(*extensions.VariantArray)

	return out, nil
}
