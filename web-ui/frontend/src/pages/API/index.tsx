import React, { useState } from 'react'
import { Card, Alert, Spin } from 'antd'
import { SERVER_URL } from '../../utils'
import { useTranslation } from 'react-i18next'

const APIPage = () => {
    const { t } = useTranslation()
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(false)

    const handleLoad = () => {
        setLoading(false)
    }

    const handleError = () => {
        setLoading(false)
        setError(true)
    }

    return (
        <div className='h-screen'
        >
            {error && (
                <Alert
                    message={t('api.load_failed')}
                    description={`${t('api.ensure_backend')} ${SERVER_URL}`}
                    type="error"
                    showIcon
                    style={{ margin: 16 }}
                />
            )}

            {loading && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%'
                }}>
                    <Spin size="large" />
                </div>
            )}

            <iframe
                src={`${SERVER_URL}/docs`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    display: error ? 'none' : 'block'
                }}
                onLoad={handleLoad}
                onError={handleError}
                title={t('api.documentation_title')}
            />
        </div>
    )
}

export default APIPage