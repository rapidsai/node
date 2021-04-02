import Navbar from 'react-bootstrap/Navbar';
import Button from 'react-bootstrap/Button';
import Nav from 'react-bootstrap/Nav';
import Image from 'next/image';
import 'bootstrap/dist/css/bootstrap.min.css';
import styles from './navbar.module.css';

export default function CustomNavbar({ title, resetall }) {
    return (
        <Navbar bg="dark" variant="dark" className={styles.navbar}>
            <Image src="/images/rapids.png" className={styles.logo}
                width={80} height={40} border-radius={10}
            ></Image>
            <Navbar.Brand href="#"> <h2>{title}</h2></Navbar.Brand>
            <Nav className="mr-auto"></Nav>
            <div>
                <Button
                    variant="primary" className="mr-sm-2"
                >Show Console</Button>
                <Button
                    variant="primary" onClick={resetall}
                >Reset all</Button>
            </div>
        </Navbar>
    )
}
