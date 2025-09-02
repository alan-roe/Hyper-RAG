import React, { useEffect, useState } from 'react';
import { Select, Typography, Space, Button, message, Spin, Modal, Input, Form } from 'antd';
import type { SizeType } from 'antd/es/config-provider/SizeContext';
import { observer } from 'mobx-react';
import { DatabaseOutlined, ReloadOutlined, PlusOutlined } from '@ant-design/icons';
import { storeGlobalUser } from '../../store/globalUser';
import { useTranslation } from 'react-i18next';
import { SERVER_URL } from '../../utils';

const { Text } = Typography;
const { Option } = Select;

interface DatabaseSelectorProps {
  /** 显示模式：选择器/按钮组/紧凑模式 */
  mode?: 'select' | 'buttons' | 'compact';
  /** 是否显示当前数据库信息 */
  showCurrent?: boolean;
  /** 是否显示刷新按钮 */
  showRefresh?: boolean;
  /** 选择器占位文本 */
  placeholder?: string;
  /** 自定义样式 */
  style?: React.CSSProperties;
  /** 组件大小 */
  size?: SizeType;
  /** 数据库变更回调 */
  onChange?: (value: string) => void;
  /** 是否禁用 */
  disabled?: boolean;
}

/**
 * 数据库选择组件
 */
const DatabaseSelector: React.FC<DatabaseSelectorProps> = ({
    mode = 'select',
    showCurrent = true,
    showRefresh = false,
    placeholder,
    style = {},
    size = 'middle',
    onChange,
    disabled = false
}) => {
    const { t } = useTranslation();
    const [createModalVisible, setCreateModalVisible] = useState(false);
    const [createForm] = Form.useForm();
    const [creating, setCreating] = useState(false);
    
    // 如果没有提供placeholder，使用默认的国际化文本
    const defaultPlaceholder = placeholder || t('database.select_database_placeholder');

    // 初始化数据库列表
    useEffect(() => {
        if (!storeGlobalUser.selectedDatabase) {
            storeGlobalUser.restoreSelectedDatabase();
        }
        if (storeGlobalUser.availableDatabases.length === 0) {
            storeGlobalUser.loadDatabases();
        }
    }, []);

    // 处理数据库变更
    const handleDatabaseChange = (value) => {
        storeGlobalUser.setSelectedDatabase(value);
        onChange?.(value);
    };

    // 刷新数据库列表
    const handleRefresh = async () => {
        try {
            await storeGlobalUser.loadDatabases();
            message.success(t('database.refresh_success'));
        } catch (error) {
            message.error(t('database.refresh_failed'));
        }
    };

    // 创建新数据库
    const handleCreateDatabase = async () => {
        try {
            const values = await createForm.validateFields();
            setCreating(true);
            
            const response = await fetch(`${SERVER_URL}/create-database`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: values.name }),
            });
            
            const data = await response.json();
            
            if (data.success) {
                message.success(data.message);
                setCreateModalVisible(false);
                createForm.resetFields();
                
                // 刷新数据库列表并选择新创建的数据库
                await storeGlobalUser.loadDatabases();
                storeGlobalUser.setSelectedDatabase(data.database);
                onChange?.(data.database);
            } else {
                message.error(data.message);
            }
        } catch (error) {
            message.error(t('database.create_failed'));
        } finally {
            setCreating(false);
        }
    };

    // 选择器模式
    const renderSelectMode = () => (
        <Space size="middle" style={style}>
            <Select
                value={storeGlobalUser.selectedDatabase}
                onChange={handleDatabaseChange}
                style={{ minWidth: 250 }}
                placeholder={placeholder}
                size={size}
                disabled={disabled}
                loading={storeGlobalUser.availableDatabases.length === 0}
                dropdownRender={(menu) => (
                    <div>
                        {menu}
                        <div style={{ padding: '8px', borderTop: '1px solid #f0f0f0' }}>
                            <Space direction="vertical" style={{ width: '100%' }}>
                                <Button
                                    type="text"
                                    size="small"
                                    icon={<PlusOutlined />}
                                    onClick={() => setCreateModalVisible(true)}
                                    style={{ width: '100%' }}
                                >
                                    {t('database.create_new')}
                                </Button>
                                {showRefresh && (
                                    <Button
                                        type="text"
                                        size="small"
                                        icon={<ReloadOutlined />}
                                        onClick={handleRefresh}
                                        style={{ width: '100%' }}
                                    >
                                        {t('database.refresh_list')}
                                    </Button>
                                )}
                            </Space>
                        </div>
                    </div>
                )}
            >
                {storeGlobalUser.availableDatabases.map((db) => (
                    <Option key={db.name} value={db.name}>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center' }}>
                                <DatabaseOutlined style={{ marginRight: 6, color: '#1890ff' }} />
                                {db.description}
                            </div>
                        </div>
                    </Option>
                ))}
            </Select>

            {showRefresh && (
                <Button
                    type="text"
                    size={size}
                    icon={<ReloadOutlined />}
                    onClick={handleRefresh}
                    disabled={disabled}
                />
            )}
        </Space>
    );

    // 按钮组模式
    const renderButtonsMode = () => (
        <Space size="small" style={style}>
            {storeGlobalUser.availableDatabases.map((db) => (
                <Button
                    key={db.name}
                    size={size}
                    type={storeGlobalUser.selectedDatabase === db.name ? 'primary' : 'default'}
                    onClick={() => handleDatabaseChange(db.name)}
                    disabled={disabled}
                    icon={<DatabaseOutlined />}
                    title={db.description}
                    style={{
                        borderColor: storeGlobalUser.selectedDatabase === db.name ? '#1890ff' : undefined,
                        borderRadius: '0.5rem'
                    }}
                    className='py-5 px-3'
                >
                    {db.description.replace(t('database.hypergraph_suffix'), '')}
                </Button>
            ))}
            {showRefresh && (
                <Button
                    type="text"
                    size={size}
                    icon={<ReloadOutlined />}
                    onClick={handleRefresh}
                    disabled={disabled}
                />
            )}
        </Space>
    );

    // 紧凑模式
    const renderCompactMode = () => (
        <Space size="small" style={style}>
            <Select
                value={storeGlobalUser.selectedDatabase}
                onChange={handleDatabaseChange}
                style={{ minWidth: 180 }}
                size={size}
                disabled={disabled}
                // bordered={false}
                placeholder={placeholder}
            >
                {storeGlobalUser.availableDatabases.map((db) => (
                    <Option key={db.name} value={db.name} title={db.description}>
                        <DatabaseOutlined style={{ marginRight: 6, color: '#1890ff' }} />
                        {db.description}
                    </Option>
                ))}
            </Select>
        </Space>
    );

    // 加载状态
    if (storeGlobalUser.availableDatabases.length === 0) {
        return (
            <Space style={style}>
                <Spin size="small" />
                <Text type="secondary">{t('database.loading_databases')}</Text>
            </Space>
        );
    }

    // 根据模式渲染不同的UI
    const renderContent = () => {
        switch (mode) {
            case 'buttons':
                return renderButtonsMode();
            case 'compact':
                return renderCompactMode();
            case 'select':
            default:
                return renderSelectMode();
        }
    };

    return (
        <>
            {renderContent()}
            
            <Modal
                title={t('database.create_new')}
                visible={createModalVisible}
                onOk={handleCreateDatabase}
                onCancel={() => {
                    setCreateModalVisible(false);
                    createForm.resetFields();
                }}
                confirmLoading={creating}
                okText={t('common.create')}
                cancelText={t('common.cancel')}
            >
                <Form form={createForm} layout="vertical">
                    <Form.Item
                        name="name"
                        label={t('database.name')}
                        rules={[
                            { required: true, message: t('database.name_required') },
                            { pattern: /^[a-zA-Z0-9_-]+$/, message: t('database.name_pattern') }
                        ]}
                    >
                        <Input 
                            placeholder={t('database.name_placeholder')}
                            addonAfter=".hgdb"
                        />
                    </Form.Item>
                </Form>
            </Modal>
        </>
    );
};

export default observer(DatabaseSelector);