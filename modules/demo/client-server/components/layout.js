import styles from './layout.module.css'
import CustomNavbar from './navbar'

export default function Layout({ title, children }) {
  return (
      <>
    <CustomNavbar title={title}></CustomNavbar>
    <div className={styles.container}>
      {children}
    </div>
    </>
  )
}
