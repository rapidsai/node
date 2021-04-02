import Head from 'next/head'
import Layout from '../components/layout'
import { Row, Col, Card, CardGroup, Button } from 'react-bootstrap';
import Link from 'next/link'
import Image from 'next/image'

export default function Home() {
  return (
    <Layout title="Node-rapids Demos" displayReset="false">
      <Head>
        <title></title>
      </Head>
      <section >
        <p className="h4 text-white text-center">A collection of client server demos powered by node-rapids for compute and streaming from the server, to a react client. </p>

        <CardGroup>
          <Card variant="secondary" text="light">
            <Card.Img variant="top" src="/images/uber.png" style={{ "height": "350px" }} />
            <Card.Body className="text-center">
              <Card.Title>Uber Movement Dashboard</Card.Title>
              <Card.Text>
                <Link href="/dashboard/uber">
                  <Button variant="primary">Start Dashboard </Button>
                </Link>
              </Card.Text>
            </Card.Body>
          </Card>
          <Card variant="secondary" text="light">
            <Card.Img variant="top" src="/images/mortgage.png" style={{ "height": "350px" }} />
            <Card.Body className="text-center">
              <Card.Title>Fannie Mae Mortgage Dashboard</Card.Title>
              <Card.Text>
                <Link href="/dashboard/mortgage">
                  <Button variant="primary">Start Dashboard </Button>
                </Link>
              </Card.Text>
            </Card.Body>
          </Card>
        </CardGroup>
      </section >

      <footer className="text-center" style={{ "position": "fixed", "bottom": 50, "width": "100%", "text-align": "center" }}>
        <Row>
          <Col><a href="https://github.com/rapidsai/node-rapids/" target="_blank"> node-rapids github </a>{' '}</Col>
          <Col><a href="https://github.com/rapidsai/" target="_blank"> Explore other RAPIDS projects </a> </Col>
          <Col><a href="https://nextjs.org/learn" target="_blank">Next.js</a></Col>
        </Row>
      </footer>
    </Layout >
  )
}
