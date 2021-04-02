import styles from './layout.module.css'
import CustomNavbar from './navbar'

export default function Layout({ title, children, resetall, displayReset }) {
  return (
    <>
      <CustomNavbar title={title} resetall={resetall} displayReset={displayReset}></CustomNavbar>
      <div className={styles.container}>
        {children}
      </div>
    </>
  )
}
