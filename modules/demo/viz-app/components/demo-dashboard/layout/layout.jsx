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

import styles from './layout.module.css'
import CustomNavbar from '../navbar/navbar'

export default function Layout({ title, children, resetall, displayReset, isLoading }) {
  return (
    <>
      <CustomNavbar title={title} resetall={resetall} displayReset={displayReset} isLoading={isLoading}></CustomNavbar>
      <div className={styles.container}>
        {children}
      </div>
    </>
  )
}
