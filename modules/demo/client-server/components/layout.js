import styles from './layout.module.css'
import CustomNavbar from './navbar'

export default function Layout({ title, children, resetall }) {
  return (
    <>
      <CustomNavbar title={title} resetall={resetall}></CustomNavbar>
      <div className={styles.container}>
        {children}
      </div>
    </>
  )
}
