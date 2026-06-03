# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Cross-client validator for the variant-shredded Parquet files iceberg-go
writes. Runs inside the spark-iceberg container.

Reads the file at --data-file from MinIO using pyarrow directly — independent
of the iceberg-go reader path — and asserts:

  - row count matches --expected-rows
  - the column physical layout matches the Parquet Variant shredding spec:
      payload.metadata               (binary, the variant dict)
      payload.value                  (binary, the residual untyped fields)
      payload.typed_value.<f>.value         per shredded path f
      payload.typed_value.<f>.typed_value   per shredded path f

The point is: an independent Parquet reader sees the iceberg-go output as a
spec-conformant shredded variant. We deliberately do not load the table via
pyiceberg: pyiceberg 0.11 still rejects `variant` field types in the REST
TableResponse pydantic model, so the catalog-layer cross-client validation
remains gated on a newer pyiceberg release. That's a known-pending gap; the
file-layer check below is what proves cross-client interop today.

Prints `OK` on success and exits non-zero on the first failure.
"""

import argparse
import os
import sys
from urllib.parse import urlparse

import pyarrow.parquet as pq
import s3fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        required=True,
        help="s3://bucket/path/to/file.parquet — exact file iceberg-go committed",
    )
    parser.add_argument("--variant-column", required=True, help="Top-level variant column name")
    parser.add_argument(
        "--shredded-paths",
        required=True,
        help="Comma-separated list of $.field paths expected to be shredded",
    )
    parser.add_argument(
        "--expected-rows", type=int, required=True, help="Row count to assert"
    )
    args = parser.parse_args()

    fs = s3fs.S3FileSystem(
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT", "http://minio:9000"),
        anon=False,
        key="admin",
        secret="password",
        client_kwargs={"region_name": "us-east-1"},
    )

    s3_path = urlparse(args.data_file)
    key = f"{s3_path.netloc}{s3_path.path}"
    print(f"pyarrow: reading {args.data_file}")

    with fs.open(key, "rb") as f:
        pqfile = pq.ParquetFile(f)
        num_rows = pqfile.metadata.num_rows
        leaves = arrow_leaf_paths(pqfile.schema_arrow)

    if num_rows != args.expected_rows:
        die(f"row count mismatch: parquet has {num_rows}, expected {args.expected_rows}")
    print(f"pyarrow: rows={num_rows}")

    required = [
        f"{args.variant_column}.metadata",
        f"{args.variant_column}.value",
    ]
    for path in args.shredded_paths.split(","):
        path = path.strip()
        if not path or not path.startswith("$"):
            continue
        if path == "$":
            required.append(f"{args.variant_column}.typed_value")
        else:
            field = path[2:]  # strip leading "$."
            required.append(f"{args.variant_column}.typed_value.{field}.value")
            required.append(f"{args.variant_column}.typed_value.{field}.typed_value")

    missing = [p for p in required if p not in leaves]
    if missing:
        die(
            f"shredded layout missing columns: {missing}\n"
            f"available leaves: {leaves}"
        )
    print(f"pyarrow: shredded leaves present: {required}")

    print("OK")


def arrow_leaf_paths(schema):
    """Return the dot-separated leaf paths under an arrow.Schema.

    The Parquet ParquetSchema API (.num_columns / .column(i)) has shifted
    across pyarrow versions; walking the arrow schema gives stable output
    that still matches the dot-joined Parquet column paths.
    """
    out = []
    for field in schema:
        _walk_field(field, "", out)
    return out


def _walk_field(field, prefix, out):
    path = f"{prefix}{field.name}" if prefix else field.name
    t = field.type
    # arrow.StructType exposes .num_fields + .field(i)
    if hasattr(t, "num_fields") and t.num_fields > 0:
        for i in range(t.num_fields):
            _walk_field(t.field(i), path + ".", out)
    else:
        out.append(path)


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
