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

import DemoCard from "../components/demo-card/demo-card";
import { Container, Row, Col, Jumbotron } from 'react-bootstrap';

export default function Home() {

  const demos = [
    {
      title: 'Graph',
      description: 'This is the description of the Graph demo.',
      href: '/demo/graph'
    },
    {
      title: 'Point Cloud',
      description: 'This is the description of the Point Cloud demo.',
      href: '/demo/point-cloud'
    }
  ];

  return (
    <Jumbotron>
      <Container>
        <Row className={"justify-content-center"}>
          {
            demos.map((demo) => (
              <Col xs={12} md={6} lg={4} className={"mb-4"} key={demo['title']}>
                <DemoCard title={demo['title']} description={demo['description']} href={demo['href']} />
              </Col>
            ))
          }
        </Row>
      </Container>
    </Jumbotron>
  )
}
