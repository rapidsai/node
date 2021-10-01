// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import DataTable from './data-table/data-table';
import DataMetrics from './data-metrics/data-metrics';

export default function DataRow() {
  return (
    <Row>
      <Col xs={12} sm={8} md={8} lg={8}>
        <DataTable />
      </Col>
      <Col>
        <DataMetrics />
      </Col>
    </Row>
  )
}
