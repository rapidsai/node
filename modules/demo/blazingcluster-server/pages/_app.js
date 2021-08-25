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

import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css';

import { Container, Navbar, Nav, Row, Col, FormControl } from 'react-bootstrap';
import { QueryBuilder } from '../components/querybuilder';
import Button from '@material-ui/core/Button';

export default function Home() {
  return (
    <div>
      <Navbar bg="dark" variant="dark">
        <Container>
          <Navbar.Brand className={"navbar"}>node-rapids ┆ Blazing Cluster Server Demo</Navbar.Brand>
          <Nav>
            <Nav.Link href="https://github.com/rapidsai/node">node-rapids github</Nav.Link>
          </Nav>
        </Container>
      </Navbar>
      <Container style={{ paddingTop: 10 }}>
        <Row className={"justify-content-center"}>
          <Col lg={8} md={8} sm={8} className={"customCol"}>
            <QueryBuilder />
          </Col>
          <Col className={"customCol"} lg md sm xs={12}>
            <Button variant="contained" className={"queryButton"}>Run Query</Button>
            <FormControl style={{ marginTop: 20 }} rows="4" as="textarea" disabled={true} value={"Waiting for query result..."} aria-label="SQL result" />
          </Col>
        </Row>
      </Container>
    </div >
  )
}
