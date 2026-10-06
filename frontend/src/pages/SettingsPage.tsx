/** 设置页：读写 GET/PUT /api/config，分 ASR / 翻译 / 界面三组。
 *
 * api_key 字段：显示 masked 值，留空表示不修改；
 * api_key_resolved=true 时显示绿色提示「已通过环境变量配置」。
 * 翻译组「测试连接」：POST /api/config/test，携带当前表单内容（未保存也能测，
 * api_key 留空沿用磁盘配置）。「系统检查」：POST /api/preflight，逐项展示
 * ok/warn/fail。
 */
import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Select,
  Space,
  Tabs,
  Tag,
  Typography,
  message,
} from "antd";
import { CheckCircleOutlined, MedicineBoxOutlined } from "@ant-design/icons";
import {
  api,
  type ConfigPayload,
  type ConfigTestResult,
  type PreflightCheck,
  type PreflightReport,
} from "../api";

const CHECK_TAG: Record<PreflightCheck["status"], { color: string; label: string }> = {
  ok: { color: "success", label: "通过" },
  warn: { color: "warning", label: "警告" },
  fail: { color: "error", label: "失败" },
};

export default function SettingsPage() {
  const [config, setConfig] = useState<ConfigPayload | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ConfigTestResult | null>(null);
  const [preflightOpen, setPreflightOpen] = useState(false);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightReport, setPreflightReport] = useState<PreflightReport | null>(null);
  const [form] = Form.useForm();

  const load = useCallback(async () => {
    try {
      const cfg = await api.getConfig();
      setConfig(cfg);
      form.setFieldsValue({
        asr: cfg.asr,
        translate: { ...cfg.translate, api_key: "" },
        ui: cfg.ui,
      });
    } catch (err) {
      message.error(`加载配置失败：${(err as Error).message}`);
    }
  }, [form]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async (values: {
    asr: ConfigPayload["asr"];
    translate: ConfigPayload["translate"];
    ui: ConfigPayload["ui"];
  }) => {
    setSaving(true);
    try {
      // api_key 留空 = 不修改（后端约定空串 / mask 值跳过）
      const translate: Record<string, unknown> = { ...values.translate };
      delete translate.api_key_resolved;
      if (!translate.api_key) translate.api_key = "";
      const updated = await api.updateConfig({
        asr: values.asr,
        translate: translate as unknown as ConfigPayload["translate"],
        ui: values.ui,
      });
      setConfig(updated);
      form.setFieldsValue({ translate: { api_key: "" } });
      message.success("配置已保存");
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  /** 测试连接：携带当前表单内容（未保存也能测），api_key 留空沿用磁盘配置 */
  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const values = { ...form.getFieldValue("translate") } as Record<string, unknown>;
      delete values.api_key_resolved;
      if (!values.api_key) values.api_key = "";
      setTestResult(await api.testConfig(values));
    } catch (err) {
      // 请求本身失败（如配置非法 400）也按失败结果展示
      setTestResult({
        ok: false,
        latency_ms: 0,
        response_preview: "",
        error: (err as Error).message,
      });
    } finally {
      setTesting(false);
    }
  };

  const runPreflight = async () => {
    setPreflightOpen(true);
    setPreflightLoading(true);
    try {
      setPreflightReport(await api.runPreflight());
    } catch (err) {
      setPreflightReport(null);
      message.error(`系统检查失败：${(err as Error).message}`);
    } finally {
      setPreflightLoading(false);
    }
  };

  if (!config) return <Card loading />;

  return (
    <Card
      title="设置"
      extra={
        <Button icon={<MedicineBoxOutlined />} onClick={() => void runPreflight()}>
          系统检查
        </Button>
      }
    >
      {config.translate.api_key_resolved && (
        <Alert
          type="success"
          showIcon
          message="API 密钥已通过环境变量配置"
          style={{ marginBottom: 16 }}
        />
      )}
      <Form form={form} layout="vertical" onFinish={save} style={{ maxWidth: 720 }}>
        <Tabs
          items={[
            {
              key: "asr",
              label: "ASR（转录）",
              children: (
                <>
                  <Form.Item label="后端" name={["asr", "backend"]}>
                    <Select
                      options={[
                        { value: "vllm", label: "vLLM（优先）" },
                        { value: "transformers", label: "transformers（兜底）" },
                      ]}
                      style={{ width: 240 }}
                    />
                  </Form.Item>
                  <Form.Item label="转录模型" name={["asr", "model"]}>
                    <Input />
                  </Form.Item>
                  <Form.Item label="对齐模型" name={["asr", "aligner_model"]}>
                    <Input />
                  </Form.Item>
                  <Space size="large" wrap>
                    <Form.Item label="设备" name={["asr", "device"]}>
                      <Input style={{ width: 120 }} />
                    </Form.Item>
                    <Form.Item label="精度" name={["asr", "dtype"]}>
                      <Input style={{ width: 120 }} />
                    </Form.Item>
                    <Form.Item
                      label="切块上限（秒）"
                      name={["asr", "chunk_max_seconds"]}
                      tooltip="ForcedAligner 单次 ≤5 分钟"
                    >
                      <InputNumber min={30} max={300} style={{ width: 120 }} />
                    </Form.Item>
                  </Space>
                  <Form.Item
                    label="源语言"
                    name={["asr", "language"]}
                    tooltip="留空自动检测"
                  >
                    <Input style={{ width: 200 }} placeholder="自动检测" />
                  </Form.Item>
                  <Form.Item
                    label="ffmpeg 路径"
                    name={["asr", "ffmpeg_path"]}
                    tooltip="留空自动探测（PATH → imageio-ffmpeg）"
                  >
                    <Input placeholder="自动探测" />
                  </Form.Item>
                </>
              ),
            },
            {
              key: "translate",
              label: "翻译",
              children: (
                <>
                  <Form.Item label="Base URL" name={["translate", "base_url"]}>
                    <Input placeholder="http://127.0.0.1:8000/v1" />
                  </Form.Item>
                  <Form.Item
                    label="API Key"
                    name={["translate", "api_key"]}
                    tooltip="留空表示不修改；也可用 api_key_env 环境变量引用"
                  >
                    <Input.Password
                      placeholder={
                        config.translate.api_key
                          ? `当前：${config.translate.api_key}（留空不修改）`
                          : "留空不修改，可改用下方 api_key_env"
                      }
                    />
                  </Form.Item>
                  <Form.Item
                    label="API Key 环境变量名"
                    name={["translate", "api_key_env"]}
                    tooltip="优先于明文 api_key，避免密钥落盘"
                  >
                    <Input style={{ width: 280 }} placeholder="如 MY_API_KEY" />
                  </Form.Item>
                  <Form.Item label="模型" name={["translate", "model"]}>
                    <Input />
                  </Form.Item>
                  <Space size="large" wrap>
                    <Form.Item
                      label="temperature"
                      name={["translate", "temperature"]}
                    >
                      <InputNumber min={0} max={2} step={0.1} style={{ width: 100 }} />
                    </Form.Item>
                    <Form.Item
                      label="历史条数"
                      name={["translate", "history_count"]}
                    >
                      <InputNumber min={0} max={100} style={{ width: 100 }} />
                    </Form.Item>
                    <Form.Item
                      label="前瞻条数"
                      name={["translate", "forward_count"]}
                    >
                      <InputNumber min={0} max={20} style={{ width: 100 }} />
                    </Form.Item>
                    <Form.Item
                      label="术语上限"
                      name={["translate", "glossary_max_entries"]}
                    >
                      <InputNumber min={1} max={500} style={{ width: 100 }} />
                    </Form.Item>
                  </Space>
                  <Form.Item
                    label="目标语言"
                    name={["translate", "target_language"]}
                  >
                    <Input style={{ width: 200 }} />
                  </Form.Item>
                  <Form.Item
                    label="附加提示"
                    name={["translate", "additional_prompt"]}
                  >
                    <Input.TextArea autoSize={{ minRows: 2 }} />
                  </Form.Item>
                  <Space size="large" wrap>
                    <Form.Item
                      label="请求超时（秒）"
                      name={["translate", "request_timeout"]}
                    >
                      <InputNumber min={5} max={3600} style={{ width: 110 }} />
                    </Form.Item>
                    <Form.Item
                      label="HTTP 重试次数"
                      name={["translate", "max_retries"]}
                    >
                      <InputNumber min={0} max={20} style={{ width: 100 }} />
                    </Form.Item>
                    <Form.Item
                      label="术语解析重试"
                      name={["translate", "glossary_max_retries"]}
                    >
                      <InputNumber min={0} max={10} style={{ width: 100 }} />
                    </Form.Item>
                  </Space>
                  <Form.Item label="连通性">
                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                      <Button
                        icon={<CheckCircleOutlined />}
                        loading={testing}
                        onClick={() => void testConnection()}
                      >
                        测试连接
                      </Button>
                      {testResult &&
                        (testResult.ok ? (
                          <Alert
                            type="success"
                            showIcon
                            message={`连接成功，延迟 ${Math.round(testResult.latency_ms)} ms`}
                            description={
                              testResult.response_preview
                                ? `响应预览：${testResult.response_preview}`
                                : undefined
                            }
                          />
                        ) : (
                          <Alert
                            type="error"
                            showIcon
                            message="连接失败"
                            description={testResult.error ?? "未知错误"}
                          />
                        ))}
                    </Space>
                  </Form.Item>
                </>
              ),
            },
            {
              key: "ui",
              label: "界面",
              children: (
                <>
                  <Space size="large" wrap>
                    <Form.Item label="监听地址" name={["ui", "host"]}>
                      <Input style={{ width: 160 }} />
                    </Form.Item>
                    <Form.Item label="端口" name={["ui", "port"]}>
                      <InputNumber min={1} max={65535} style={{ width: 120 }} />
                    </Form.Item>
                  </Space>
                  <Form.Item
                    label="上传存储目录"
                    name={["ui", "upload_dir"]}
                    tooltip="网页上传文件的落盘位置；留空为 ~/.subtitle_translator/uploads"
                  >
                    <Input placeholder="~/.subtitle_translator/uploads" />
                  </Form.Item>
                </>
              ),
            },
          ]}
        />
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={saving}>
            保存配置
          </Button>
        </Form.Item>
      </Form>

      <Modal
        title="系统检查（启动预检）"
        open={preflightOpen}
        onCancel={() => setPreflightOpen(false)}
        footer={
          <Button onClick={() => setPreflightOpen(false)}>关闭</Button>
        }
        width={640}
      >
        {preflightLoading ? (
          <Typography.Text type="secondary">
            检查中（翻译端点会发一条最小请求实测，可能需十几秒）…
          </Typography.Text>
        ) : preflightReport ? (
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Alert
              type={preflightReport.ok ? "success" : "error"}
              showIcon
              message={
                preflightReport.ok
                  ? "预检通过，服务 ready-to-use"
                  : "预检未通过：存在失败项，请先解决"
              }
            />
            <List
              size="small"
              dataSource={preflightReport.checks}
              renderItem={(check) => (
                <List.Item>
                  <Space direction="vertical" size={2} style={{ width: "100%" }}>
                    <Space>
                      <Tag color={CHECK_TAG[check.status]?.color ?? "default"}>
                        {CHECK_TAG[check.status]?.label ?? check.status}
                      </Tag>
                      <Typography.Text strong>{check.name}</Typography.Text>
                    </Space>
                    {check.message && (
                      <Typography.Text
                        type={check.status === "fail" ? "danger" : "secondary"}
                        style={{ fontSize: 12, whiteSpace: "pre-wrap" }}
                      >
                        {check.message}
                      </Typography.Text>
                    )}
                  </Space>
                </List.Item>
              )}
            />
          </Space>
        ) : (
          <Typography.Text type="secondary">未能获取检查结果</Typography.Text>
        )}
      </Modal>
    </Card>
  );
}
