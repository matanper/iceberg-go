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

// gen writes the cross-client variant shredding fixture
// (events.parquet) under table/internal/testdata/shredded_variant_write/.
//
// The fixture mirrors what an iceberg-go writer produces under the
// `write.variant.shredding-paths=$.event_type,$.count` table property. It
// exists so Java / pyiceberg readers can verify they decode it back to the
// source variant values without round-tripping through iceberg-go itself.
//
// Run with `go run ./cmd/gen` from this directory.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

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
)

func main() {
	var out string
	flag.StringVar(&out, "out", "events.parquet", "output Parquet path")
	flag.Parse()

	if err := run(out); err != nil {
		log.Fatalf("gen: %v", err)
	}
}

func run(out string) error {
	ctx := context.Background()
	mem := memory.DefaultAllocator

	icesc := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "id", Type: iceberg.PrimitiveTypes.Int64, Required: true},
		iceberg.NestedField{ID: 2, Name: "payload", Type: iceberg.VariantType{}},
	)
	arrowSchema, err := table.SchemaToArrowSchema(icesc, nil, true, false)
	if err != nil {
		return err
	}

	idBldr := array.NewInt64Builder(mem)
	defer idBldr.Release()
	payBldr := extensions.NewVariantBuilder(mem, extensions.NewDefaultVariantType())
	defer payBldr.Release()

	rows := []map[string]any{
		{"event_type": "click", "count": int64(7), "extra": "drop-me"},
		{"event_type": "view", "count": int64(11)},
		{"count": int64(3), "note": "no event_type here"},
	}
	for i, row := range rows {
		idBldr.Append(int64(i + 1))
		var vb variant.Builder
		if err := vb.Append(row); err != nil {
			return fmt.Errorf("build variant row %d: %w", i, err)
		}
		v, err := vb.Build()
		if err != nil {
			return fmt.Errorf("finalize variant row %d: %w", i, err)
		}
		payBldr.Append(v)
	}

	idArr := idBldr.NewInt64Array()
	defer idArr.Release()
	payArr := payBldr.NewArray()
	defer payArr.Release()

	rec := array.NewRecordBatch(arrowSchema, []arrow.Array{idArr, payArr}, int64(len(rows)))
	defer rec.Release()

	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}

	fm := internal.GetFileFormat(iceberg.ParquetFile)
	_, err = fm.WriteDataFile(ctx, io.LocalFS{}, nil, internal.WriteFileInfo{
		FileSchema: icesc,
		FileName:   out,
		Spec:       iceberg.PartitionSpec{},
		WriteProps: []parquet.WriterProperty{},
		StatsCols: map[int]internal.StatisticsCollector{
			1: {FieldID: 1, Mode: internal.MetricsMode{Typ: internal.MetricModeFull}, ColName: "id", IcebergTyp: iceberg.PrimitiveTypes.Int64},
			2: {FieldID: 2, Mode: internal.MetricsMode{Typ: internal.MetricModeNone}, ColName: "payload"},
		},
		VariantShreddingPaths: []string{"$.event_type", "$.count"},
	}, []arrow.RecordBatch{rec})
	if err != nil {
		return err
	}

	fmt.Printf("wrote %s\n", out)

	return nil
}
