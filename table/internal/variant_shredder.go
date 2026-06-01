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

// ShreddingPath declares one path-to-leaf in a shredding spec.
//
// Segments are the JSON-path field names walked from the variant root
// down to the leaf. "$.user.email" parses to ["user", "email"]; the
// top-level path "$.lat" parses to ["lat"]. Array index access ($.foo[0])
// and root-as-array shredding are not supported yet — the parser
// surfaces them as explicit errors so callers don't silently get a
// no-op.
type ShreddingPath struct {
	Segments []string
	// Type is the Arrow primitive type extracted at the leaf when the
	// variant payload matches. Mismatched payloads (including variant
	// null) fall back to per-path `value` residual storage so the
	// reader can still see the original bytes.
	Type arrow.DataType
}

// Path returns the dotted "$.a.b" form of the declaration.
func (p ShreddingPath) Path() string {
	return "$." + strings.Join(p.Segments, ".")
}

// ShreddingSchema is the parsed shredding spec for one variant column.
// Paths preserve declaration order at each level of the resulting tree;
// the order is mirrored in the Arrow typed_value struct returned by
// TypedValueArrowType.
type ShreddingSchema struct {
	Column string
	Paths  []ShreddingPath

	tree *pathTree
}

// Tree returns the cached pathTree representation, building it on
// first call. Returns nil for schemas with no declared paths.
func (s *ShreddingSchema) Tree() *pathTree {
	if s == nil {
		return nil
	}
	if s.tree == nil && len(s.Paths) > 0 {
		s.tree = buildPathTree(s.Paths)
	}

	return s.tree
}

// ShreddingConfig collects per-column ShreddingSchema entries parsed
// from the write.variant.shredding-paths table property. Lookup by
// column name uses ForColumn.
type ShreddingConfig struct {
	byColumn map[string]*ShreddingSchema
}

// ForColumn returns the ShreddingSchema for the given iceberg field
// name, or nil if the column has no shredding configured.
func (c *ShreddingConfig) ForColumn(name string) *ShreddingSchema {
	if c == nil {
		return nil
	}

	return c.byColumn[name]
}

// Empty reports whether no columns are configured.
func (c *ShreddingConfig) Empty() bool { return c == nil || len(c.byColumn) == 0 }

// Columns returns the iceberg field names that have shredding
// configured. Order is not stable.
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

// shreddingTypeAliases maps the type names accepted in the
// shredding-paths property to their Arrow DataType. The names match
// Iceberg primitive names (boolean, int, long, float, double, string,
// binary) so the property is portable across Java / pyiceberg /
// iceberg-go.
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
// Example: "payload:$.lat:double, payload:$.user.email:string".
//
// Conflicts and ambiguities are rejected at parse time:
//   - Two paths on the same column where one is a strict prefix of the
//     other (e.g. "$.user" and "$.user.email") would force the writer
//     to both extract user as a typed value AND shred sub-fields of
//     user — incompatible per the Parquet spec.
//   - Duplicate paths.
//   - Empty segments ("$..").
//   - Array indexing ("$.foo[0]").
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

		segments, err := parsePathSegments(path)
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
		if err := checkPathConflict(sc.Paths, segments); err != nil {
			return nil, fmt.Errorf("variant shredding: %w on column %q (entry %q)", err, column, entry)
		}
		sc.Paths = append(sc.Paths, ShreddingPath{Segments: segments, Type: dt})
	}

	return cfg, nil
}

// parsePathSegments splits "$.a.b.c" into ["a", "b", "c"]. Returns an
// error for paths missing the "$." prefix, paths with empty segments,
// or paths that use array indexing.
func parsePathSegments(path string) ([]string, error) {
	if !strings.HasPrefix(path, "$.") {
		return nil, fmt.Errorf("path must start with %q, got %q", "$.", path)
	}
	rest := path[2:]
	if rest == "" {
		return nil, fmt.Errorf("empty path %q", path)
	}
	if strings.ContainsAny(rest, "[]") {
		return nil, fmt.Errorf("array indexing not supported in shredding paths, got %q", path)
	}
	segments := strings.Split(rest, ".")
	for _, seg := range segments {
		if seg == "" {
			return nil, fmt.Errorf("empty segment in path %q", path)
		}
	}

	return segments, nil
}

// checkPathConflict rejects a new path that would be a strict prefix
// of, or strictly extend, an existing path on the same column.
//
// Example collisions:
//
//	existing "$.user"          new "$.user.email"   -> reject
//	existing "$.user.email"    new "$.user"         -> reject
//	existing "$.user.email"    new "$.user.email"   -> reject (duplicate)
//
// Independent paths under a common interior key are fine:
//
//	"$.user.email" + "$.user.id" -> both allowed (independent leaves).
func checkPathConflict(existing []ShreddingPath, candidate []string) error {
	for _, p := range existing {
		switch {
		case segmentsEqual(p.Segments, candidate):
			return fmt.Errorf("duplicate path %q", "$."+strings.Join(candidate, "."))
		case isStrictPrefix(p.Segments, candidate):
			return fmt.Errorf("path %q would shred fields under already-leaf path %q",
				"$."+strings.Join(candidate, "."), "$."+strings.Join(p.Segments, "."))
		case isStrictPrefix(candidate, p.Segments):
			return fmt.Errorf("path %q is a prefix of already-declared path %q",
				"$."+strings.Join(candidate, "."), "$."+strings.Join(p.Segments, "."))
		}
	}

	return nil
}

func segmentsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// isStrictPrefix reports whether a is a strict (proper) prefix of b.
func isStrictPrefix(a, b []string) bool {
	if len(a) >= len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// pathTree is the recursive representation of a ShreddingSchema. At
// every node exactly one of leafType or children is set:
//   - leafType != nil  -> primitive shred target. No children.
//   - children != nil  -> interior object node. typed_value at this
//     level is a struct of the children's typed_value types.
//
// childOrder preserves the user-declared order so the resulting Arrow
// struct fields are deterministic.
type pathTree struct {
	leafType   arrow.DataType
	children   map[string]*pathTree
	childOrder []string
}

func buildPathTree(paths []ShreddingPath) *pathTree {
	root := &pathTree{children: map[string]*pathTree{}}
	for _, p := range paths {
		insertPath(root, p.Segments, p.Type)
	}

	return root
}

func insertPath(node *pathTree, segments []string, leafType arrow.DataType) {
	if len(segments) == 0 {
		// Should never happen — parser rejects empty paths — but
		// guard anyway so a programming error doesn't silently
		// produce a broken tree.
		panic("variant shredding: empty path in tree insert")
	}
	head, tail := segments[0], segments[1:]
	child, ok := node.children[head]
	if !ok {
		child = &pathTree{}
		if node.children == nil {
			node.children = map[string]*pathTree{}
		}
		node.children[head] = child
		node.childOrder = append(node.childOrder, head)
	}
	if len(tail) == 0 {
		child.leafType = leafType

		return
	}
	if child.children == nil {
		child.children = map[string]*pathTree{}
	}
	insertPath(child, tail, leafType)
}

// TypedValueArrowType returns the "logical" typed_value Arrow type for
// a shredding schema. Pass the result to
// extensions.NewShreddedVariantType to get the full shredded variant
// Arrow type (arrow-go wraps each field in the {value, typed_value}
// shape required by the Parquet spec).
//
// Returns nil for empty schemas.
func TypedValueArrowType(schema *ShreddingSchema) arrow.DataType {
	if schema == nil || len(schema.Paths) == 0 {
		return nil
	}

	return treeArrowType(schema.Tree())
}

func treeArrowType(node *pathTree) arrow.DataType {
	if node.leafType != nil {
		return node.leafType
	}
	fields := make([]arrow.Field, 0, len(node.childOrder))
	for _, name := range node.childOrder {
		child := node.children[name]
		fields = append(fields, arrow.Field{Name: name, Type: treeArrowType(child)})
	}

	return arrow.StructOf(fields...)
}

// ShreddedField is the per-row, per-path result of ShredVariant. At any
// given path, at most one of {Typed, Residual, Object} is meaningful;
// Present says whether the path was present in the input variant at
// all.
//
// Leaf node, declared type matched:    {Typed: extracted, Present: true}
// Leaf node, type mismatch / null:     {Residual: bytes,  Present: true}
// Leaf or interior node, absent:       {Present: false}
// Interior node, input is an object:   {Object: <sub-result>, Present: true}
// Interior node, input is non-object:  {Residual: bytes,  Present: true}
type ShreddedField struct {
	Typed    any
	Residual []byte
	Object   *ShreddedObject
	Present  bool
}

// ShreddedObject is the per-row result at an interior object node.
// Fields are ordered to match the pathTree's childOrder.
// ResidualValue holds the variant bytes of unshredded sub-fields at
// this level (nil when nothing remains).
type ShreddedObject struct {
	Fields        []ShreddedField
	ResidualValue []byte
}

// ShredResult is the per-row output of ShredVariant for a single
// variant column. Root.Fields aligns with the root pathTree's child
// order, and Root.ResidualValue is the top-level residual variant
// bytes (or the verbatim input bytes when the input is not an object).
type ShredResult struct {
	Metadata []byte
	Root     ShreddedObject
}

// ShredVariant extracts the configured paths from a variant value,
// returning the typed values and residual at every level of the
// shredding tree. The schema's tree is built and cached on the
// schema on first call.
//
// The Parquet spec rules implemented here:
//
//  1. Either `value` (residual) or `typed_value` is non-null at a
//     leaf, never both.
//  2. At an object node, BOTH `value` and `typed_value` may be
//     populated: `typed_value` carries the declared shredded
//     sub-fields, `value` carries any sub-fields not declared.
//  3. When the variant at an interior node is *not* an object, the
//     typed_value sub-struct is null and the raw variant bytes go
//     into `value`.
//  4. Type mismatch at a leaf (declared double, payload is a string
//     or null) is not an error — the raw variant bytes are stored in
//     `value` and `typed_value` is left null, so a reader can still
//     surface the original payload.
func ShredVariant(v variant.Value, schema *ShreddingSchema) (ShredResult, error) {
	if schema == nil {
		return ShredResult{}, errors.New("variant shredding: nil schema")
	}
	tree := schema.Tree()
	if tree == nil {
		return ShredResult{}, errors.New("variant shredding: empty schema")
	}

	res := ShredResult{Metadata: v.Metadata().Bytes()}
	obj, err := shredObject(v, tree)
	if err != nil {
		return ShredResult{}, err
	}
	res.Root = obj

	return res, nil
}

// shredObject processes a variant value against an object-shaped
// pathTree node and returns the per-row result. If the variant is not
// an Object, every declared child is reported absent and the verbatim
// input bytes become the residual at this level.
func shredObject(v variant.Value, node *pathTree) (ShreddedObject, error) {
	out := ShreddedObject{Fields: make([]ShreddedField, len(node.childOrder))}
	if v.Type() != variant.Object {
		out.ResidualValue = v.Bytes()

		return out, nil
	}
	objValue, ok := v.Value().(variant.ObjectValue)
	if !ok {
		// Defensive — should not happen given variant.Type() == Object.
		out.ResidualValue = v.Bytes()

		return out, nil
	}

	// Index source object's fields by key; track what got matched so
	// we can compute the residual from what's left.
	type residualField struct {
		key   string
		bytes []byte
	}
	var residualFields []residualField
	matched := 0
	for key, fieldVal := range objValue.Values() {
		idx, isShredded := childIndex(node, key)
		if !isShredded {
			residualFields = append(residualFields, residualField{key: key, bytes: fieldVal.Bytes()})

			continue
		}
		matched++
		shredded, err := shredAt(fieldVal, node.children[node.childOrder[idx]])
		if err != nil {
			return ShreddedObject{}, fmt.Errorf("at %q: %w", key, err)
		}
		out.Fields[idx] = shredded
	}

	// Residual rule: when every input field was claimed by a shredded
	// path AND something matched, leave ResidualValue nil so the
	// writer encodes the outer `value` column as null. Empty objects
	// (matched == 0, no residual fields) get the verbatim input bytes
	// so reconstruction round-trips correctly.
	if len(residualFields) == 0 {
		if matched == 0 {
			out.ResidualValue = v.Bytes()
		}

		return out, nil
	}

	b := variant.NewBuilderFromMeta(v.Metadata())
	start := b.Offset()
	entries := make([]variant.FieldEntry, 0, len(residualFields))
	for _, rf := range residualFields {
		entries = append(entries, b.NextField(start, rf.key))
		if err := b.UnsafeAppendEncoded(rf.bytes); err != nil {
			return ShreddedObject{}, fmt.Errorf("append residual field %q: %w", rf.key, err)
		}
	}
	if err := b.FinishObject(start, entries); err != nil {
		return ShreddedObject{}, fmt.Errorf("finish residual object: %w", err)
	}
	built, err := b.Build()
	if err != nil {
		return ShreddedObject{}, fmt.Errorf("build residual: %w", err)
	}
	out.ResidualValue = built.Bytes()

	return out, nil
}

// shredAt processes a variant value against any pathTree node (leaf
// or interior). Always returns Present: true — the caller filters out
// absent paths before invoking this.
//
// At an interior node, a non-object payload short-circuits to per-path
// residual storage (Object left nil) so the writer encodes the entire
// typed_value sub-tree as null while preserving the raw variant bytes
// in `value`. Only when the payload is actually an object do we
// recurse and surface a ShreddedObject.
func shredAt(v variant.Value, node *pathTree) (ShreddedField, error) {
	if node.leafType != nil {
		typed, ok := extractTyped(v, node.leafType)
		if ok {
			return ShreddedField{Typed: typed, Present: true}, nil
		}

		return ShreddedField{Residual: v.Bytes(), Present: true}, nil
	}
	if v.Type() != variant.Object {
		return ShreddedField{Residual: v.Bytes(), Present: true}, nil
	}
	obj, err := shredObject(v, node)
	if err != nil {
		return ShreddedField{}, err
	}

	return ShreddedField{Object: &obj, Present: true}, nil
}

// childIndex returns the index of `key` in node.childOrder, or
// (-1, false) when the key is not a shredded child at this level.
func childIndex(node *pathTree, key string) (int, bool) {
	for i, n := range node.childOrder {
		if n == key {
			return i, true
		}
	}

	return -1, false
}

// extractTyped tries to read v's payload as the declared Arrow type.
// Returns (value, true) on a successful match and (nil, false) when
// the variant's runtime type is incompatible (e.g. declared double
// but payload is a string).
//
// The variant binary format stores integers in the smallest primitive
// that fits (Int8/Int16/Int32/Int64) and may parse JSON-encoded
// decimals as DecimalValue. We perform lossless widening
// (Int8→Int32, Int8→Int64, Float32→Float64, Decimal→Float64) so a
// user-friendly declaration like "double" or "long" matches the
// payload regardless of which on-wire type the producer chose. Lossy
// conversions (Float64→Float32, Int64→Int32 of an overflow value)
// intentionally fall through to the residual path so no precision is
// silently dropped.
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

// decimalAsFloat64 converts a variant DecimalValue payload (any of
// the 32/64/128-bit precisions) to float64. Returns (0, false) when
// raw is not a DecimalValue. Note: 128-bit values exceeding float64's
// 53-bit significand will round; callers concerned about precision
// should declare a decimal-typed shredding column (v1 has none, so
// such values fall back to residual storage).
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

// ShreddingConfigFromSchema reverse-engineers a ShreddingConfig from a
// post-ApplyShreddingToSchema iceberg schema, so the Parquet writer
// can drive shredding from FileSchema alone — no need to also plumb
// the table property through. Returns an empty config when no variant
// column carries a shredding spec.
//
// The mapping is the inverse of TypedValueArrowType: the field's
// shredded spec (an arrow.DataType) is walked, and every leaf primitive
// becomes a ShreddingPath whose segments are the dotted path of the
// containing struct fields.
func ShreddingConfigFromSchema(s *iceberg.Schema) *ShreddingConfig {
	cfg := &ShreddingConfig{byColumn: map[string]*ShreddingSchema{}}
	if s == nil {
		return cfg
	}
	for _, f := range s.Fields() {
		vt, ok := f.Type.(iceberg.VariantType)
		if !ok {
			continue
		}
		shredded := vt.Shredded()
		if shredded == nil {
			continue
		}
		st, ok := shredded.(*arrow.StructType)
		if !ok {
			continue
		}
		sc := &ShreddingSchema{Column: f.Name}
		collectShredPaths(st, nil, sc)
		if len(sc.Paths) > 0 {
			cfg.byColumn[f.Name] = sc
		}
	}

	return cfg
}

func collectShredPaths(st *arrow.StructType, prefix []string, out *ShreddingSchema) {
	for i := range st.NumFields() {
		field := st.Field(i)
		segments := append(append([]string(nil), prefix...), field.Name)
		switch t := field.Type.(type) {
		case *arrow.StructType:
			collectShredPaths(t, segments, out)
		default:
			out.Paths = append(out.Paths, ShreddingPath{Segments: segments, Type: field.Type})
		}
	}
}

// ApplyShreddingToSchema returns a copy of s with variant columns
// named in cfg upgraded to a shredded VariantType. Top-level fields
// only — nested variants (inside struct/list/map) are left untouched
// in v1.
//
// Returns s unchanged when cfg is empty or no column matches.
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

// ShredRecordBatch returns a new RecordBatch where every variant
// column listed in cfg has been transformed from its bare
// metadata+value representation into the shredded representation
// prescribed by the schema. Columns not listed in cfg are passed
// through verbatim.
//
// The output Arrow schema matches the input column-for-column except
// shredded columns adopt the typed_value-bearing variant extension
// type. Use the Arrow schema produced by SchemaToArrowSchema on the
// post-ApplyShreddingToSchema iceberg schema to construct a writer
// that consumes this batch.
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
		col := rec.Column(i)
		shreddingSpec := cfg.ForColumn(f.Name)
		if shreddingSpec == nil {
			col.Retain()
			cols[i] = col

			continue
		}
		variantArr, ok := col.(*extensions.VariantArray)
		if !ok {
			// Column is named in the shredding config but isn't a
			// variant array. Pass through — schema check downstream
			// will surface the misconfiguration (or this is a
			// delete-file write where shredding doesn't apply).
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
	// NewRecordBatch retains each column; release our local refs.
	for _, c := range cols {
		c.Release()
	}

	return out, nil
}

// BuildShreddedVariantArray transforms a bare variant array (with
// metadata+value storage) into a shredded variant array per the
// Parquet Variant Shredding spec. The output array's storage type is
// extensions.NewShreddedVariantType(TypedValueArrowType(schema)).
//
// Each row is shredded independently via ShredVariant; null input
// rows (validity bit unset) propagate to null output rows.
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

	tree := schema.Tree()

	for row := range in.Len() {
		if in.IsNull(row) {
			// StructBuilder.AppendNull only flips the validity bit
			// on the parent; child builders stay at their current
			// length. Advance every leaf to keep the columns
			// aligned with the struct.
			builder.AppendNull()
			metaBuilder.AppendNull()
			valueBuilder.AppendNull()
			appendNullObjectStruct(typedBuilder, tree)

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
		if shred.Root.ResidualValue != nil {
			valueBuilder.Append(shred.Root.ResidualValue)
		} else {
			valueBuilder.AppendNull()
		}
		if err := appendObjectStruct(typedBuilder, tree, shred.Root); err != nil {
			return nil, fmt.Errorf("variant shredding: row %d typed_value: %w", row, err)
		}
	}

	storage := builder.NewArray()
	defer storage.Release()

	return array.NewExtensionArrayWithStorage(outType, storage), nil
}

// appendObjectStruct writes one row of an interior object node's
// typed_value into the corresponding struct builder. Each child is a
// {value, typed_value} pair per the spec.
func appendObjectStruct(sb *array.StructBuilder, node *pathTree, obj ShreddedObject) error {
	sb.Append(true)
	for i, name := range node.childOrder {
		child := node.children[name]
		pairBuilder := sb.FieldBuilder(i).(*array.StructBuilder)
		valueB := pairBuilder.FieldBuilder(0).(*array.BinaryBuilder)
		typedB := pairBuilder.FieldBuilder(1)
		f := obj.Fields[i]

		if !f.Present {
			pairBuilder.AppendNull()
			valueB.AppendNull()
			appendTypedNull(typedB, child)

			continue
		}
		pairBuilder.Append(true)

		switch {
		case child.leafType != nil && f.Typed != nil:
			valueB.AppendNull()
			if err := appendTypedValue(typedB, f.Typed); err != nil {
				return fmt.Errorf("path %q: %w", name, err)
			}
		case child.leafType != nil:
			// Leaf, type mismatch -> raw bytes in `value`.
			valueB.Append(f.Residual)
			appendTypedNull(typedB, child)
		case f.Object != nil:
			// Interior, input was an object -> recurse into sub-tree.
			if f.Object.ResidualValue != nil {
				valueB.Append(f.Object.ResidualValue)
			} else {
				valueB.AppendNull()
			}
			if err := appendObjectStruct(typedB.(*array.StructBuilder), child, *f.Object); err != nil {
				return fmt.Errorf("path %q: %w", name, err)
			}
		default:
			// Interior, input was not an object -> raw bytes in
			// `value`, typed_value null for the entire sub-tree.
			valueB.Append(f.Residual)
			appendTypedNull(typedB, child)
		}
	}

	return nil
}

// appendNullObjectStruct appends a null for the typed_value struct at
// any interior node, recursively nulling every leaf and intermediate
// struct so the column lengths stay aligned.
func appendNullObjectStruct(sb *array.StructBuilder, node *pathTree) {
	sb.AppendNull()
	for i, name := range node.childOrder {
		child := node.children[name]
		pairBuilder := sb.FieldBuilder(i).(*array.StructBuilder)
		valueB := pairBuilder.FieldBuilder(0).(*array.BinaryBuilder)
		typedB := pairBuilder.FieldBuilder(1)
		pairBuilder.AppendNull()
		valueB.AppendNull()
		appendTypedNull(typedB, child)
	}
}

// appendTypedNull appends a null to typedB, the builder for one
// child's `typed_value` field. For interior nodes that's a recursive
// null-fill; for leaves it's a simple AppendNull on the primitive
// builder.
func appendTypedNull(typedB array.Builder, node *pathTree) {
	if node.leafType != nil {
		typedB.AppendNull()

		return
	}
	appendNullObjectStruct(typedB.(*array.StructBuilder), node)
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
