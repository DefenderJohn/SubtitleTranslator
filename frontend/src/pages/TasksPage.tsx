/** 任务页（默认页）：新建任务 + 任务列表（SSE 实时进度）。 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  Popconfirm,
  Progress,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { FolderOpenOutlined, ReloadOutlined } from "@ant-design/icons";
import { Link } from "react-router-dom";
import {
  api,
  basename,
  type TaskEvent,
  type TaskSnapshot,
  type TaskStatus,
} from "../api";
import DirectoryBrowser from "../components/DirectoryBrowser";

export const STATUS_TAG: Record<TaskStatus, { color: string; label: string }> = {
  pending: { color: "gold", label: "排队中" },
  running: { color: "processing", label: "运行中" },
  waiting_confirm: { color: "warning", label: "待确认术语" },
  done: { color: "success", label: "完成" },
  failed: { color: "error", label: "失败" },
  cancelled: { color: "default", label: "已取消" },
};

const ACTIVE_STATES: TaskStatus[] = ["pending", "running", "waiting_confirm"];

interface FormValues {
  path: string;
  auto_confirm: boolean;
  bilingual: boolean;
  transcribe_only: boolean;
  language?: string;
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm<FormValues>();
  // taskId -> EventSource，组件卸载时统一关闭
  const sourcesRef = useRef<Map<string, EventSource>>(new Map());

  const applyEvent = useCallback((event: TaskEvent) => {
    setTasks((prev) =>
      prev.map((t) =>
        t.id === event.task_id
          ? { ...t, status: event.status, progress: event }
          : t,
      ),
    );
  }, []);

  /** 给活跃任务开 SSE；终态任务关流 */
  const syncSubscriptions = useCallback(
    (list: TaskSnapshot[]) => {
      const sources = sourcesRef.current;
      for (const task of list) {
        if (ACTIVE_STATES.includes(task.status) && !sources.has(task.id)) {
          sources.set(task.id, api.subscribeTaskEvents(task.id, applyEvent));
        }
      }
      for (const [id, source] of sources) {
        const task = list.find((t) => t.id === id);
        if (!task || !ACTIVE_STATES.includes(task.status)) {
          source.close();
          sources.delete(id);
        }
      }
    },
    [applyEvent],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.listTasks();
      setTasks(list);
      syncSubscriptions(list);
    } catch (err) {
      message.error(`加载任务列表失败：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [syncSubscriptions]);

  useEffect(() => {
    void refresh();
    const sources = sourcesRef.current;
    return () => {
      for (const source of sources.values()) source.close();
      sources.clear();
    };
  }, [refresh]);

  // SSE 把任务推到终态后关流
  useEffect(() => {
    syncSubscriptions(tasks);
  }, [tasks, syncSubscriptions]);

  const createTask = async (values: FormValues) => {
    setSubmitting(true);
    try {
      const task = await api.createTask(values.path, {
        auto_confirm: values.auto_confirm,
        bilingual: values.bilingual,
        transcribe_only: values.transcribe_only,
        language: values.language?.trim() || null,
      });
      message.success(`任务已创建：${basename(values.path)}`);
      form.resetFields();
      setTasks((prev) => [...prev, task]);
      syncSubscriptions([...tasks, task]);
    } catch (err) {
      message.error(`创建任务失败：${(err as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const cancelTask = async (id: string) => {
    try {
      await api.cancelTask(id);
      message.success("取消请求已发送（运行中的任务在当前条目完成后停止）");
      void refresh();
    } catch (err) {
      message.error(`取消失败：${(err as Error).message}`);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <Card title="新建任务">
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            path: "",
            auto_confirm: false,
            bilingual: true,
            transcribe_only: false,
            language: "",
          }}
          onFinish={createTask}
        >
          <Form.Item
            label="媒体文件或目录路径"
            name="path"
            rules={[{ required: true, message: "请输入路径或点浏览选择" }]}
          >
            <Space.Compact style={{ width: "100%" }}>
              <Input placeholder="/path/to/movie.mp4 或目录" />
              <Button
                icon={<FolderOpenOutlined />}
                onClick={() => setBrowserOpen(true)}
              >
                浏览
              </Button>
            </Space.Compact>
          </Form.Item>
          <Space size="large" wrap>
            <Form.Item
              label="自动确认术语表"
              name="auto_confirm"
              valuePropName="checked"
              tooltip="开启后跳过术语表人工确认检查点"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              label="双语导出"
              name="bilingual"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              label="仅转录"
              name="transcribe_only"
              valuePropName="checked"
              tooltip="只转录并导出原文 SRT，不翻译"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              label="源语言"
              name="language"
              tooltip="留空自动检测，如 en / zh / ja"
            >
              <Input style={{ width: 120 }} placeholder="自动检测" />
            </Form.Item>
            <Form.Item label=" ">
              <Button type="primary" htmlType="submit" loading={submitting}>
                创建任务
              </Button>
            </Form.Item>
          </Space>
        </Form>
      </Card>

      <Card
        title="任务列表"
        extra={
          <Button icon={<ReloadOutlined />} onClick={() => void refresh()}>
            刷新
          </Button>
        }
      >
        <Table<TaskSnapshot>
          rowKey="id"
          loading={loading}
          dataSource={tasks}
          pagination={false}
          locale={{ emptyText: "暂无任务" }}
          columns={[
            {
              title: "媒体文件",
              dataIndex: "media",
              render: (media: string[], task) => (
                <Space direction="vertical" size={0}>
                  {media.map((m) => (
                    <Typography.Text key={m}>{basename(m)}</Typography.Text>
                  ))}
                  {media.length > 1 && (
                    <Typography.Text type="secondary">
                      共 {media.length} 个文件（目录任务 {task.path}）
                    </Typography.Text>
                  )}
                </Space>
              ),
            },
            {
              title: "状态",
              dataIndex: "status",
              width: 120,
              render: (status: TaskStatus, task) => (
                <Space direction="vertical" size={4}>
                  <Tag color={STATUS_TAG[status].color}>
                    {STATUS_TAG[status].label}
                  </Tag>
                  {task.error && (
                    <Typography.Text type="danger" style={{ fontSize: 12 }}>
                      {task.error}
                    </Typography.Text>
                  )}
                </Space>
              ),
            },
            {
              title: "进度",
              width: 240,
              render: (_, task) => {
                const p = task.progress;
                if (!p || !p.total) {
                  return (
                    <Typography.Text type="secondary">
                      {p?.message || "—"}
                    </Typography.Text>
                  );
                }
                return (
                  <Space direction="vertical" size={0} style={{ width: "100%" }}>
                    <Progress
                      percent={Math.round((p.done / p.total) * 100)}
                      size="small"
                      status={task.status === "failed" ? "exception" : "active"}
                    />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {p.message || `${p.stage} ${p.done}/${p.total}`}
                    </Typography.Text>
                  </Space>
                );
              },
            },
            {
              title: "创建时间",
              dataIndex: "created_at",
              width: 170,
              render: (ts: number) => new Date(ts * 1000).toLocaleString(),
            },
            {
              title: "操作",
              width: 160,
              render: (_, task) => (
                <Space>
                  <Link to={`/tasks/${task.id}`}>详情</Link>
                  {(task.status === "pending" ||
                    task.status === "running" ||
                    task.status === "waiting_confirm") && (
                    <Popconfirm
                      title="确认取消该任务？"
                      description="运行中的任务会在当前条目完成后停止，进度已保存"
                      onConfirm={() => void cancelTask(task.id)}
                    >
                      <Typography.Link type="danger">取消</Typography.Link>
                    </Popconfirm>
                  )}
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <DirectoryBrowser
        open={browserOpen}
        onCancel={() => setBrowserOpen(false)}
        onSelect={(path) => {
          form.setFieldValue("path", path);
          setBrowserOpen(false);
        }}
      />
    </Space>
  );
}
