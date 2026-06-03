<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Cross-Client Shredded Variant Writes

This directory contains a Parquet fixture produced by the iceberg-go variant
shredding writer. It exists so Java / pyiceberg can read what iceberg-go wrote
and confirm the on-disk shredded variant layout is interoperable.

The companion read-side fixtures live in `../shredded_variant/` (sourced from
the `apache/parquet-testing` repository). Together they prove the iceberg-go
shredding implementation is symmetric: it reads Java-produced files and
produces files Java can read.

## Fixture

`events.parquet` carries a single record batch with two columns:

| Field ID | Name      | Iceberg type | Notes                                                     |
| ---------| --------- | ------------ | --------------------------------------------------------- |
| 1        | `id`      | `long`       | row identifier (`1`, `2`, `3`)                            |
| 2        | `payload` | `variant`    | shredded on `$.event_type` (string) and `$.count` (long)  |

Source variant values (one per row):

```jsonc
// row 0
{ "event_type": "click", "count": 7, "extra": "drop-me" }
// row 1
{ "event_type": "view",  "count": 11 }
// row 2
// note: `event_type` missing — typed_value for that field is null,
// rest stays in the residual `value` column
{ "count": 3, "note": "no event_type here" }
```

Writer configuration (matches the Java property name):

```
write.variant.shredding-paths = $.event_type,$.count
```

## Regenerating

Run the standalone generator from this directory:

```sh
go run ./cmd/gen -out events.parquet
```

## Cross-Client Verification

The iceberg-go round-trip is exercised in
`table/internal/variant_shredder_test.go` (`TestParquetWriter_ShredsVariantAndRoundTrips`).
For Java and pyiceberg verification:

### pyiceberg / pyarrow

```python
import pyarrow.parquet as pq
import json

tbl = pq.read_table("events.parquet")
# The "payload" extension column reassembles back to the source variants.
for row in tbl.to_pylist():
    print(row["id"], row["payload"])
```

Expected stdout (after JSON normalization of `payload`):

```
1 {"event_type": "click", "count": 7, "extra": "drop-me"}
2 {"event_type": "view",  "count": 11}
3 {"count": 3, "note": "no event_type here"}
```

### Java (iceberg-core)

```java
try (CloseableIterable<Record> records = Parquet.read(Files.localInput("events.parquet"))
        .project(SCHEMA)
        .createReaderFunc(fileSchema -> GenericParquetReaders.buildReader(SCHEMA, fileSchema))
        .build()) {
    for (Record r : records) {
        VariantValue v = r.get(1, VariantValue.class);
        System.out.println(r.get(0) + " " + v);
    }
}
```

The shredded `typed_value` struct must contain `event_type` (string) and
`count` (long); missing fields (row 2's `event_type`) appear as null in the
typed column, with all surviving non-shredded keys in the residual `value`
column.
