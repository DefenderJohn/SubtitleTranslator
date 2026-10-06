/** 任务页（默认页）：新建任务（服务器路径 / 浏览器上传两种模式）+ 任务列表（SSE 实时进度）。
 *
 * 新建任务的草稿（媒体来源 / 路径 / 上传列表 / 选项）放在全局 UploadDraftProvider
 * （src/uploadDraft.tsx），切换路由不丢；server 重启后上传批次已被清理，挂载时
 * 按批次目录探测（/api/media 404），失效给提示并允许一键清空，创建任务的 404 兜底。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
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
  ApiError,
  basename,
  type TaskEvent,
  type TaskSnapshot,
  type TaskStatus,
} from "../api";
import DirectoryBrowser from "../components/DirectoryBrowser";
import { useUploadDraft, type SourceMode } from "../uploadDraft";

export const STATUS_TAG: Record<TaskStatus, { color: string; label: string }> = {
  pending: { color: "gold", label: "排队中" },
  running: { color: "processing", label: "运行中" },
  waiting_confirm: { color: "warning", label: "待确认术语" },
  done: { color: "success", label: "完成" },
  failed: { color: "error", label: "失败" },
  cancelled: { color: "default", label: "已取消" },
};

const ACTIVE_STATES: TaskStatus[] = ["pending", "running", "waiting_confirm"];

/** pipeline 事件 stage → 中文阶段名（进度展示用） */
export const STAGE_LABEL: Record<string, string> = {
  transcribe: "转录",
  translate: "翻译",
  export: "导出",
};

// 与后端 MEDIA_EXTENSIONS 对齐（文件选择器过滤用；真正校验在服务端）
const MEDIA_ACCEPT =
  ".flac,.m4a,.mp3,.mp4,.mpeg,.mpga,.oga,.ogg,.wav,.webm,.mkv,.mov,.avi,.m4v";

interface FormValues {
  path: string;
  auto_confirm: boolean;
  bilingual: boolean;
  transcribe_only: boolean;
  language?: string;
}

/** 任务列表「进度」列：按状态给出进度条 + 阶段文字 / 结果提示 */
export function TaskProgressCell({ task }: { task: TaskSnapshot }) {
  const p = task.progress;
  switch (task.status) {
    case "pending":
      return <Typography.Text type="secondary">排队等待中</Typography.Text>;
    case "waiting_confirm":
      return <Typography.Text type="warning">等待术语确认</Typography.Text>;
    case "done":
      return <Typography.Text type="success">已完成</Typography.Text>;
    case "failed":
      return (
        <Typography.Text type="danger">
          {p?.message || "失败（详见任务日志）"}
        </Typography.Text>
      );
    case "cancelled":
      return <Typography.Text type="secondary">已取消，进度已保存</Typography.Text>;
    case "running": {
      if (!p) {
        return <Typography.Text type="secondary">准备中…</Typography.Text>;
      }
      if (!p.total) {
        return (
          <Typography.Text type="secondary">
            {p.message || "准备中…"}
          </Typography.Text>
        );
      }
      const stage = STAGE_LABEL[p.stage] ?? p.stage;
      return (
        <Space direction="vertical" size={0} style={{ width: "100%" }}>
          <Progress
            percent={Math.round((p.done / p.total) * 100)}
            size="small"
            status="active"
          />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {stage}中 {p.done}/{p.total}
            {p.message && p.message !== `${stage} ${p.done}/${p.total}`
              ? `（${p.message}）`
              : ""}
          </Typography.Text>
        </Space>
      );
    }
  }
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [staleUploads, setStaleUploads] = useState(false);
  const { draft, dispatch } = useUploadDraft();
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

  // 失效批次探测：草稿里的上传文件可能已随 server 重启被清理，
  // 逐批次目录查 /api/media（目录不存在 404），失效则提示一键清空
  useEffect(() => {
    if (!draft.uploaded.length) return;
    const batches = Array.from(new Set(draft.uploaded.map((u) => u.batch)));
    let cancelled = false;
    void (async () => {
      for (const batch of batches) {
        try {
          await api.browseMedia(batch);
        } catch (err) {
          if (!cancelled && err instanceof ApiError && err.status === 404) {
            setStaleUploads(true);
            return;
          }
          // 网络类错误不误判为失效，下次进页面再探
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [draft.uploaded]);

  /** 直传 /api/upload（自定义请求，逐文件显示进度），成功后记下服务器侧路径 */
  const uploadRequest: UploadProps["customRequest"] = async (options) => {
    const { file, onProgress, onSuccess, onError } = options;
    const uploadFile = file as UploadFile;
    try {
      const resp = await api.uploadFile(file as File, (percent) =>
        onProgress?.({ percent }),
      );
      dispatch({
        type: "addUploaded",
        entry: { uid: uploadFile.uid, path: resp.paths[0], batch: resp.batch },
      });
      setStaleUploads(false);
      onSuccess?.(resp);
    } catch (err) {
      message.error(`上传失败：${(err as Error).message}`);
      onError?.(err as Error);
    }
  };

  const createTask = async (values: FormValues) => {
    // 上传模式：每个已上传文件一个任务；路径模式：单路径（文件或目录）
    const paths =
      draft.mode === "upload"
        ? draft.uploaded.map((u) => u.path)
        : [values.path.trim()];
    if (!paths.length || !paths[0]) {
      message.warning(
        draft.mode === "upload" ? "请先上传文件" : "请输入路径或点浏览选择",
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
      // 清空草稿并同步表单（initialValues 只在挂载时生效，需显式回写默认值）
      dispatch({ type: "reset" });
      form.setFieldsValue({
        path: "",
        auto_confirm: false,
        bilingual: true,
        transcribe_only: false,
        language: "",
      });
      setStaleUploads(false);
      setTasks((prev) => [...prev, ...created]);
      syncSubscriptions([...tasks, ...created]);
    } catch (err) {
      // 兜底：上传批次已在 server 侧被清理（重启/退出清理），草稿路径失效
      if (
        draft.mode === "upload" &&
        err instanceof ApiError &&
        err.status === 404
      ) {
        Modal.confirm({
          title: "上传的文件已不在服务器上",
          content:
            "服务重启后上传的临时副本会被清理，当前草稿里的文件路径已失效。清空上传列表后请重新上传。",
          okText: "清空上传列表",
          cancelText: "保留",
          onOk: () => {
            dispatch({ type: "clearUploads" });
            setStaleUploads(false);
          },
        });
      } else {
        message.error(`创建任务失败：${(err as Error).message}`);
      }
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
            path: draft.path,
            auto_confirm: draft.options.auto_confirm,
            bilingual: draft.options.bilingual,
            transcribe_only: draft.options.transcribe_only,
            language: draft.options.language,
          }}
          onValuesChange={(changed: Partial<FormValues>) => {
            // 草稿随表单输入同步到全局，切换路由回来不丢
            if (changed.path !== undefined) {
              dispatch({ type: "setPath", path: changed.path });
            }
            const { path: _path, ...opts } = changed;
            if (Object.keys(opts).length) {
              dispatch({ type: "setOptions", options: opts });
            }
          }}
          onFinish={createTask}
        >
          <Form.Item label="媒体来源">
            <Segmented<SourceMode>
              value={draft.mode}
              onChange={(mode) => dispatch({ type: "setMode", mode })}
              options={[
                { label: "服务器路径", value: "path" },
                { label: "上传文件", value: "upload" },
              ]}
            />
          </Form.Item>
          {draft.mode === "path" ? (
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
                draft.uploaded.length > 0
                  ? `已上传 ${draft.uploaded.length} 个文件，提交后每个文件创建一个任务`
                  : undefined
              }
            >
              {staleUploads && (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="之前上传的文件已不在服务器上"
                  description="服务重启后上传的临时副本会被清理，草稿里的文件路径已失效，请重新上传。"
                  action={
                    <Button
                      size="small"
                      onClick={() => {
                        dispatch({ type: "clearUploads" });
                        setStaleUploads(false);
                      }}
                    >
                      清空上传列表
                    </Button>
                  }
                />
              )}
              <Upload.Dragger
                multiple
                accept={MEDIA_ACCEPT}
                customRequest={uploadRequest}
                fileList={draft.fileList}
                onChange={({ fileList }) =>
                  dispatch({ type: "setFileList", fileList })
                }
                onRemove={(file) => dispatch({ type: "removeUploaded", uid: file.uid })}
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
              width: 260,
              render: (_, task) => <TaskProgressCell task={task} />,
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
          dispatch({ type: "setPath", path });
          setBrowserOpen(false);
        }}
      />
    </Space>
  );
}
