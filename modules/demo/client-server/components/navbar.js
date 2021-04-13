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

import Navbar from 'react-bootstrap/Navbar';
import Button from 'react-bootstrap/Button';
import Nav from 'react-bootstrap/Nav';
import Image from 'next/image';
import 'bootstrap/dist/css/bootstrap.min.css';
import styles from './navbar.module.css';
import Link from 'next/link'

export default function CustomNavbar({ title, resetall, displayReset }) {
    return (
        <Navbar bg="dark" variant="dark" className={styles.navbar}>
            <Link href="/">
                <a><Image src="/images/rapids.png" className={styles.logo}
                    width={80} height={40} border-radius={10}
                ></Image></a>
            </Link>

            <Navbar.Brand href="#"> <h2>{title}</h2></Navbar.Brand>
            <Nav className="mr-auto"></Nav>
            <div>
                {displayReset != "false" &&
                    <Button variant="primary" onClick={resetall}>Reset all</Button>
                }
            </div>
        </Navbar >
    )
}
