# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM tabulario/spark-iceberg:3.5.5_1.8.1

RUN pip3 install -q ipython
# Upgrade pyiceberg + pyarrow past what the base image bakes in. The base ships
# pyiceberg 0.7 which predates the Iceberg v3 variant type; 0.11 reads the
# spec'd shredded variant Parquet layout iceberg-go writes. pyarrow ≥20 lets
# scripts walk the Parquet schema for cross-client structural assertions.
RUN pip3 install -q --upgrade "pyiceberg[s3fs,hive]>=0.11" "pyarrow>=20"

COPY provision.py .
COPY validation.py .
COPY validation_shredded_variant.py .
COPY hadoop_validation.py .

ENTRYPOINT ["./entrypoint.sh"]
CMD ["notebook"]
