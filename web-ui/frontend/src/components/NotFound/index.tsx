import { Button, Result } from 'antd'
import React from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import type { NotFoundPropsType } from './type'

const NotFound: React.FC<NotFoundPropsType> = (props) => {
  const { t } = useTranslation()
  const {
    status = '404',
    title = t('notFound.title'),
    subTitle = t('notFound.subtitle'),
    extra = (
      <Button type="primary">
        <Link to="/">{t('notFound.backHome')}</Link>
      </Button>
    )
  } = props
  return (
    <>
      <Result status={status} title={title} subTitle={subTitle} extra={extra} />
    </>
  )
}

export default NotFound
