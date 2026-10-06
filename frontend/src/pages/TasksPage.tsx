/** 任务页（默认页）：新建任务（服务器路径 / 浏览器上传两种模式）+ 任务列表（SSE 实时进度）。 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Form,
  Input,
  Popconfirm,
  Progress,
  Segmented,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from "antd";
import type { UploadFile, UploadProps } from "antd";
import {
  FolderOpenOutlined,
  InboxOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
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

// 与后端 MEDIA_EXTENSIONS 对齐（文件选择器过滤用；真正校验在服务端）
const MEDIA_ACCEPT =
  ".flac,.m4a,.mp3,.mp4,.mpeg,.mpga,.oga,.ogg,.wav,.webm,.mkv,.mov,.avi,.m4v";

type SourceMode = "path" | "upload";

/** 上传成功的文件：uid（antd Upload）→ 服务器侧路径 */
interface UploadedFile {
  uid: string;
  path: string;
}

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
  const [mode, setMode] = useState<SourceMode>("path");
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploaded, setUploaded] = useState<UploadedFile[]>([]);
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

  /** 直传 /api/upload（自定义请求，逐文件显示进度），成功后记下服务器侧路径 */
  const uploadRequest: UploadProps["customRequest"] = async (options) => {
    const { file, onProgress, onSuccess, onError } = options;
    const uploadFile = file as UploadFile;
    try {
      const resp = await api.uploadFile(file as File, (percent) =>
        onProgress?.({ percent }),
      );
      setUploaded((prev) => [
        ...prev,
        { uid: uploadFile.uid, path: resp.paths[0] },
      ]);
      onSuccess?.(resp);
    } catch (err) {
      message.error(`上传失败：${(err as Error).message}`);
      onError?.(err as Error);
    }
  };

  const createTask = async (values: FormValues) => {
    // 上传模式：每个已上传文件一个任务；路径模式：单路径（文件或目录）
    const paths =
      mode === "upload" ? uploaded.map((u) => u.path) : [values.path.trim()];
    if (!paths.length || !paths[0]) {
      message.warning(
        mode === "upload" ? "请先上传文件" : "请输入路径或点浏览选择",
      );
      return;
    }
    setSubmitting(true);
    try {
      const created: TaskSnapshot[] = [];
      for (const path of paths) {
        created.push(
          await api.createTask(path, {
            auto_confirm: values.auto_confirm,
            bilingual: values.bilingual,
            transcribe_only: values.transcribe_only,
            language: values.language?.trim() || null,
          }),
        );
      }
      message.success(
        `已创建 ${created.length} 个任务：${paths.map(basename).join("、")}`,
      );
      form.resetFields();
      setFileList([]);
      setUploaded([]);
      setTasks((prev) => [...prev, ...created]);
      syncSubscriptions([...tasks, ...created]);
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
          <Form.Item label="媒体来源">
            <Segmented<SourceMode>
              value={mode}
              onChange={setMode}
              options={[
                { label: "服务器路径", value: "path" },
                { label: "上传文件", value: "upload" },
              ]}
            />
          </Form.Item>
          {mode === "path" ? (
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
          ) : (
            <Form.Item
              label="上传本地文件"
              extra={
                uploaded.length > 0
                  ? `已上传 ${uploaded.length} 个文件，提交后每个文件创建一个任务`
                  : undefined
              }
            >
              <Upload.Dragger
                multiple
                accept={MEDIA_ACCEPT}
                customRequest={uploadRequest}
                fileList={fileList}
                onChange={({ fileList }) => setFileList(fileList)}
                onRemove={(file) => {
                  setUploaded((prev) => prev.filter((u) => u.uid !== file.uid));
                }}
              >
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                <p className="ant-upload-hint">
                  支持常见音视频格式（mp4 / mkv / mp3 / ogg 等），可多选；上传完成后在下方选选项并创建任务
                </p>
              </Upload.Dragger>
            </Form.Item>
          )}
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
