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
// Segments are the navigation steps walked from the variant root down
// to the leaf. Field names appear verbatim; the "[]" token denotes
// "every element of the surrounding array." So:
//
//	"$.user.email"   -> ["user", "email"]
//	"$.tags[]"       -> ["tags", "[]"]
//	"$.events[].ts"  -> ["events", "[]", "ts"]
//	"$"              -> []           // top-level: the variant itself is the leaf
//
// Positional indexing ($.foo[0]) is rejected at parse time — per the
// Parquet shredding spec, individual array positions are not separately
// addressable; the whole list is shredded as a single typed_value
// column with one element shape.
type ShreddingPath struct {
	Segments []string
	// Type is the Arrow primitive type extracted at the leaf when the
	// variant payload matches. Mismatched payloads (including variant
	// null) fall back to per-path `value` residual storage so the
	// reader can still see the original bytes.
	Type arrow.DataType
}

// Path returns the dotted "$.a.b[]" form of the declaration.
func (p ShreddingPath) Path() string {
	return pathString(p.Segments)
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
// Examples:
//
//	"payload:$.lat:double, payload:$.user.email:string"
//	"tags:$.tags[]:string"     // shred every element as a string
//	"value:$:double"            // top-level: the variant itself is the typed value
//
// Conflicts and ambiguities are rejected at parse time:
//   - Top-level shredding (path "$") is exclusive — it cannot coexist
//     with any other path on the same column.
//   - Two paths on the same column where one is a strict prefix of the
//     other (e.g. "$.user" and "$.user.email") would force the writer
//     to both extract user as a typed value AND shred sub-fields of
//     user — incompatible per the Parquet spec.
//   - Duplicate paths.
//   - Empty segments ("$..").
//   - Positional array indexing ("$.foo[0]") — use "[]" instead.
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

// arrayMarker is the segment token used inside a parsed path to
// denote "step into each element of the surrounding array". Path
// "$.tags[].name" parses to segments ["tags", "[]", "name"]. The
// marker is also used as the single child key under an array
// pathTree node so the tree shape uniformly represents arrays.
const arrayMarker = "[]"

// parsePathSegments converts a "$.a.b[].c" path string into the
// segment slice ["a", "b", "[]", "c"]. The lone "$" path is allowed
// and yields an empty slice, denoting top-level shredding (the
// variant payload itself is the shred target). Returns an error for
// paths missing the "$" prefix, paths with empty field names, or
// positional array indexing (e.g. "[0]") which the spec does not
// support.
func parsePathSegments(path string) ([]string, error) {
	if path == "$" {
		return []string{}, nil
	}
	if !strings.HasPrefix(path, "$.") {
		return nil, fmt.Errorf("path must start with %q, got %q", "$.", path)
	}
	rest := path[2:]
	if rest == "" {
		return nil, fmt.Errorf("empty path %q", path)
	}
	segments := make([]string, 0)
	for rest != "" {
		// Pull off the field name up to the next . or [.
		boundary := strings.IndexAny(rest, ".[")
		if boundary == -1 {
			segments = append(segments, rest)

			break
		}
		if boundary == 0 {
			return nil, fmt.Errorf("empty segment in path %q", path)
		}
		segments = append(segments, rest[:boundary])
		switch rest[boundary] {
		case '.':
			rest = rest[boundary+1:]
			if rest == "" {
				return nil, fmt.Errorf("trailing dot in path %q", path)
			}
		case '[':
			if !strings.HasPrefix(rest[boundary:], "[]") {
				return nil, fmt.Errorf("positional array indexing not supported (use []) in path %q", path)
			}
			segments = append(segments, arrayMarker)
			rest = rest[boundary+2:]
			if rest == "" {
				break
			}
			if rest[0] != '.' && rest[0] != '[' {
				return nil, fmt.Errorf("expected '.' or '[' after %q in path %q", arrayMarker, path)
			}
			if rest[0] == '.' {
				rest = rest[1:]
				if rest == "" {
					return nil, fmt.Errorf("trailing dot in path %q", path)
				}
			}
		}
	}
	for _, seg := range segments {
		if seg == "" {
			return nil, fmt.Errorf("empty segment in path %q", path)
		}
	}

	return segments, nil
}

// checkPathConflict rejects a new path that would collide with an
// existing one on the same column. Collisions include duplicate
// paths and prefix relationships that would force the writer to
// treat the same node as both a leaf and an interior.
//
// Top-level shredding (empty segments) is exclusive: it cannot
// coexist with any other path on the column.
//
// Example collisions:
//
//	existing "$.user"          new "$.user.email"   -> reject
//	existing "$.user.email"    new "$.user"         -> reject
//	existing "$"               new anything         -> reject
//
// Independent paths under a common interior key are fine:
//
//	"$.user.email" + "$.user.id" -> both allowed (independent leaves).
func checkPathConflict(existing []ShreddingPath, candidate []string) error {
	candidatePath := pathString(candidate)
	for _, p := range existing {
		existingPath := pathString(p.Segments)
		switch {
		case segmentsEqual(p.Segments, candidate):
			return fmt.Errorf("duplicate path %q", candidatePath)
		case len(p.Segments) == 0:
			return fmt.Errorf("path %q conflicts with top-level shredding %q",
				candidatePath, existingPath)
		case len(candidate) == 0:
			return fmt.Errorf("top-level path %q conflicts with already-declared path %q",
				candidatePath, existingPath)
		case isStrictPrefix(p.Segments, candidate):
			return fmt.Errorf("path %q would shred fields under already-leaf path %q",
				candidatePath, existingPath)
		case isStrictPrefix(candidate, p.Segments):
			return fmt.Errorf("path %q is a prefix of already-declared path %q",
				candidatePath, existingPath)
		}
	}

	return nil
}

// pathString rebuilds the "$.a[].b" form from a parsed segments
// slice for use in error messages.
func pathString(segments []string) string {
	if len(segments) == 0 {
		return "$"
	}
	var b strings.Builder
	b.WriteByte('$')
	for _, seg := range segments {
		if seg == arrayMarker {
			b.WriteString("[]")
		} else {
			b.WriteByte('.')
			b.WriteString(seg)
		}
	}

	return b.String()
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

// pathTree is the recursive representation of a ShreddingSchema. A
// node is exactly one of:
//   - leaf:   leafType != nil. The variant at this position is shred
//     as a primitive of the declared type.
//   - struct: children != nil, no arrayMarker child. The variant at
//     this position is shred as an object whose declared sub-keys
//     each carry their own pathTree.
//   - array:  children has exactly one entry under the arrayMarker
//     key. The variant at this position is shred as a list whose
//     element shape is the marker's subtree.
//
// childOrder preserves the user-declared order so the resulting Arrow
// struct fields are deterministic.
type pathTree struct {
	leafType   arrow.DataType
	children   map[string]*pathTree
	childOrder []string
}

func (n *pathTree) isLeaf() bool { return n != nil && n.leafType != nil }
func (n *pathTree) isArray() bool {
	return n != nil && len(n.childOrder) == 1 && n.childOrder[0] == arrayMarker
}

func (n *pathTree) isStruct() bool {
	return n != nil && !n.isLeaf() && !n.isArray() && len(n.childOrder) > 0
}

// arrayElem returns the element subtree of an array node, or nil if
// the node is not an array.
func (n *pathTree) arrayElem() *pathTree {
	if !n.isArray() {
		return nil
	}

	return n.children[arrayMarker]
}

func buildPathTree(paths []ShreddingPath) *pathTree {
	root := &pathTree{}
	for _, p := range paths {
		insertPath(root, p.Segments, p.Type)
	}

	return root
}

func insertPath(node *pathTree, segments []string, leafType arrow.DataType) {
	if len(segments) == 0 {
		// Top-level / leaf insertion.
		node.leafType = leafType

		return
	}
	head, tail := segments[0], segments[1:]
	if node.children == nil {
		node.children = map[string]*pathTree{}
	}
	child, ok := node.children[head]
	if !ok {
		child = &pathTree{}
		node.children[head] = child
		node.childOrder = append(node.childOrder, head)
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
	switch {
	case node.isLeaf():
		return node.leafType
	case node.isArray():
		return arrow.ListOf(treeArrowType(node.arrayElem()))
	}
	fields := make([]arrow.Field, 0, len(node.childOrder))
	for _, name := range node.childOrder {
		child := node.children[name]
		fields = append(fields, arrow.Field{Name: name, Type: treeArrowType(child)})
	}

	return arrow.StructOf(fields...)
}

// ShreddedField is the per-row, per-path result of ShredVariant. At
// any given node at most one of {Typed, Residual, Object, Array} is
// meaningful; Present says whether the path was present in the
// surrounding context at all.
//
//	Leaf,     type matched:    {Typed: <extracted>,    Present: true}
//	Leaf,     type mismatch:   {Residual: <raw>,       Present: true}
//	Struct,   input is object: {Object: <sub-result>,  Present: true}
//	Struct,   non-object:      {Residual: <raw>,       Present: true}
//	Array,    input is array:  {Array: <sub-result>,   Present: true}
//	Array,    non-array:       {Residual: <raw>,       Present: true}
//	Absent (any kind):         {Present: false}
type ShreddedField struct {
	Typed    any
	Residual []byte
	Object   *ShreddedObject
	Array    *ShreddedArray
	Present  bool
}

// ShreddedObject is the per-row result at a struct node. Fields are
// ordered to match the pathTree's childOrder. ResidualValue holds
// the variant bytes of unshredded sub-fields at this level (nil
// when nothing remains).
type ShreddedObject struct {
	Fields        []ShreddedField
	ResidualValue []byte
}

// ShreddedArray is the per-row result at an array node. Elements
// holds one ShreddedField per array element, in source order.
// Variant arrays don't have a per-position residual the way objects
// do (you can't drop elements without shifting indices), so when an
// element's variant value doesn't match the declared element shape
// the element's own ShreddedField carries the raw bytes in Residual.
type ShreddedArray struct {
	Elements []ShreddedField
}

// ShredResult is the per-row output of ShredVariant for a single
// variant column. Root represents the entire variant; what's set on
// Root depends on the schema's top-level node kind (leaf for
// top-level primitive shredding, Object for object shredding, Array
// for top-level array shredding).
type ShredResult struct {
	Metadata []byte
	Root     ShreddedField
}

// ShredVariant extracts the configured paths from a variant value,
// returning the typed values and residual at every level of the
// shredding tree. The schema's tree is built and cached on the
// schema on first call.
//
// The Parquet spec rules implemented here:
//
//  1. At a leaf, either `value` (residual) or `typed_value` is
//     non-null, never both.
//  2. At a struct node, BOTH `value` and `typed_value` may be
//     populated: `typed_value` carries the declared shredded
//     sub-fields, `value` carries any sub-fields not declared.
//  3. At an array node, `typed_value` is a Parquet 3-level LIST and
//     `value` is null when the payload is an array; non-array
//     payloads fall back to per-path residual storage.
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

	root, err := shredAt(v, tree)
	if err != nil {
		return ShredResult{}, err
	}

	return ShredResult{Metadata: v.Metadata().Bytes(), Root: root}, nil
}

// shredObject processes a variant value against a struct-shaped
// pathTree node and returns the per-row result. If the variant is
// not an Object, every declared child is reported absent and the
// verbatim input bytes become the residual at this level.
func shredObject(v variant.Value, node *pathTree) (ShreddedObject, error) {
	out := ShreddedObject{Fields: make([]ShreddedField, len(node.childOrder))}
	if v.Type() != variant.Object {
		out.ResidualValue = v.Bytes()

		return out, nil
	}
	objValue, ok := v.Value().(variant.ObjectValue)
	if !ok {
		out.ResidualValue = v.Bytes()

		return out, nil
	}

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

	if len(residualFields) == 0 {
		if matched == 0 {
			// Empty object input: round-trip the verbatim bytes so
			// reconstruction yields an empty object rather than nil.
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

// shredArray processes a variant value against an array-shaped
// pathTree node. The element shape (a single child under the array
// marker) is applied to every variant array element. Non-array
// payloads short-circuit to per-path residual storage at the caller.
func shredArray(v variant.Value, node *pathTree) (ShreddedArray, error) {
	arrValue, ok := v.Value().(variant.ArrayValue)
	if !ok {
		// Variant.Type() said Array but the cast failed; defensive.
		return ShreddedArray{}, fmt.Errorf("variant shredding: array node received non-array payload (type %v)", v.Type())
	}
	elemNode := node.arrayElem()
	out := ShreddedArray{Elements: make([]ShreddedField, 0, arrValue.Len())}
	for elem := range arrValue.Values() {
		f, err := shredAt(elem, elemNode)
		if err != nil {
			return ShreddedArray{}, fmt.Errorf("element %d: %w", len(out.Elements), err)
		}
		out.Elements = append(out.Elements, f)
	}

	return out, nil
}

// shredAt processes a variant value against any pathTree node (leaf,
// struct, or array). Always returns Present: true — callers filter
// out absent paths before invoking this.
//
// When the payload kind doesn't match the node kind (e.g. a struct
// node receives a non-object) the function short-circuits to
// per-path residual storage so the writer encodes the entire
// typed_value sub-tree as null while preserving the raw variant
// bytes in `value`.
func shredAt(v variant.Value, node *pathTree) (ShreddedField, error) {
	switch {
	case node.isLeaf():
		typed, ok := extractTyped(v, node.leafType)
		if ok {
			return ShreddedField{Typed: typed, Present: true}, nil
		}

		return ShreddedField{Residual: v.Bytes(), Present: true}, nil
	case node.isArray():
		if v.Type() != variant.Array {
			return ShreddedField{Residual: v.Bytes(), Present: true}, nil
		}
		arr, err := shredArray(v, node)
		if err != nil {
			return ShreddedField{}, err
		}

		return ShreddedField{Array: &arr, Present: true}, nil
	default:
		if v.Type() != variant.Object {
			return ShreddedField{Residual: v.Bytes(), Present: true}, nil
		}
		obj, err := shredObject(v, node)
		if err != nil {
			return ShreddedField{}, err
		}

		return ShreddedField{Object: &obj, Present: true}, nil
	}
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
		sc := &ShreddingSchema{Column: f.Name}
		collectShredPaths(shredded, nil, sc)
		if len(sc.Paths) > 0 {
			cfg.byColumn[f.Name] = sc
		}
	}

	return cfg
}

func collectShredPaths(dt arrow.DataType, prefix []string, out *ShreddingSchema) {
	switch t := dt.(type) {
	case *arrow.StructType:
		for i := range t.NumFields() {
			field := t.Field(i)
			collectShredPaths(field.Type,
				append(append([]string(nil), prefix...), field.Name), out)
		}
	case arrow.ListLikeType:
		collectShredPaths(t.Elem(),
			append(append([]string(nil), prefix...), arrayMarker), out)
	default:
		out.Paths = append(out.Paths, ShreddingPath{Segments: append([]string(nil), prefix...), Type: dt})
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
	typedBuilder := builder.FieldBuilder(2)

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
			appendNullForNode(typedBuilder, tree)

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
		outerResidual := topLevelResidual(shred.Root)
		if outerResidual != nil {
			valueBuilder.Append(outerResidual)
		} else {
			valueBuilder.AppendNull()
		}
		if err := appendShreddedField(typedBuilder, tree, shred.Root); err != nil {
			return nil, fmt.Errorf("variant shredding: row %d typed_value: %w", row, err)
		}
	}

	storage := builder.NewArray()
	defer storage.Release()

	return array.NewExtensionArrayWithStorage(outType, storage), nil
}

// topLevelResidual extracts the residual bytes that belong on the
// outer variant's `value` column. For object-rooted schemas this is
// the sub-object residual; for leaf or array roots it's the
// type-mismatch residual on the root field; nil otherwise.
func topLevelResidual(root ShreddedField) []byte {
	switch {
	case root.Object != nil:
		return root.Object.ResidualValue
	case root.Residual != nil:
		return root.Residual
	}

	return nil
}

// appendShreddedField writes one row's typed_value into b according
// to node's kind. Caller has already appended validity to the outer
// struct (or is appending into a leaf/array slot).
func appendShreddedField(b array.Builder, node *pathTree, f ShreddedField) error {
	switch {
	case node.isLeaf():
		if f.Present && f.Typed != nil {
			return appendTypedValue(b, f.Typed)
		}
		b.AppendNull()

		return nil
	case node.isArray():
		lb := b.(*array.ListBuilder)
		if !f.Present || f.Array == nil {
			lb.AppendNull()

			return nil
		}
		lb.Append(true)
		elemPair := lb.ValueBuilder().(*array.StructBuilder)
		for _, elem := range f.Array.Elements {
			if err := appendValueTypedPair(elemPair, node.arrayElem(), elem); err != nil {
				return err
			}
		}

		return nil
	}
	// Struct node.
	sb := b.(*array.StructBuilder)
	if !f.Present || f.Object == nil {
		sb.AppendNull()
		for i, name := range node.childOrder {
			pair := sb.FieldBuilder(i).(*array.StructBuilder)
			child := node.children[name]
			pair.AppendNull()
			pair.FieldBuilder(0).(*array.BinaryBuilder).AppendNull()
			appendNullForNode(pair.FieldBuilder(1), child)
		}

		return nil
	}
	sb.Append(true)
	for i, name := range node.childOrder {
		child := node.children[name]
		pair := sb.FieldBuilder(i).(*array.StructBuilder)
		if err := appendValueTypedPair(pair, child, f.Object.Fields[i]); err != nil {
			return fmt.Errorf("path %q: %w", name, err)
		}
	}

	return nil
}

// appendValueTypedPair writes one {value, typed_value} pair — the
// per-field wrapping arrow-go's createShreddedField generates for
// every shredded sub-position. The pair is itself a non-nullable
// struct, so we always Append(true); per-field absence is encoded by
// nulling both the inner value and typed_value.
func appendValueTypedPair(pair *array.StructBuilder, node *pathTree, f ShreddedField) error {
	pair.Append(true)
	valueB := pair.FieldBuilder(0).(*array.BinaryBuilder)
	typedB := pair.FieldBuilder(1)

	if !f.Present {
		valueB.AppendNull()
		appendNullForNode(typedB, node)

		return nil
	}
	switch {
	case node.isLeaf():
		if f.Typed != nil {
			valueB.AppendNull()

			return appendTypedValue(typedB, f.Typed)
		}
		valueB.Append(f.Residual)
		typedB.AppendNull()

		return nil
	case node.isArray():
		if f.Array == nil {
			// Array node, non-array payload.
			valueB.Append(f.Residual)
			appendNullForNode(typedB, node)

			return nil
		}
		valueB.AppendNull()

		return appendShreddedField(typedB, node, f)
	}
	// Struct child.
	if f.Object == nil {
		valueB.Append(f.Residual)
		appendNullForNode(typedB, node)

		return nil
	}
	if f.Object.ResidualValue != nil {
		valueB.Append(f.Object.ResidualValue)
	} else {
		valueB.AppendNull()
	}

	return appendShreddedField(typedB, node, f)
}

// appendNullForNode appends a null at every leaf builder underneath
// node so column lengths stay aligned with the parent. For struct
// and array nodes this recurses through the entire sub-tree.
func appendNullForNode(b array.Builder, node *pathTree) {
	switch {
	case node.isLeaf():
		b.AppendNull()
	case node.isArray():
		// A null list leaves child builders untouched.
		b.(*array.ListBuilder).AppendNull()
	default:
		sb := b.(*array.StructBuilder)
		sb.AppendNull()
		for i, name := range node.childOrder {
			child := node.children[name]
			pair := sb.FieldBuilder(i).(*array.StructBuilder)
			pair.AppendNull()
			pair.FieldBuilder(0).(*array.BinaryBuilder).AppendNull()
			appendNullForNode(pair.FieldBuilder(1), child)
		}
	}
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
