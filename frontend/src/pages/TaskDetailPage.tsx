/** 任务详情页：SSE 事件日志（历史回放 + 实时追加）、术语确认、导出、字幕校对。 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Space,
  Tabs,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  ExportOutlined,
} from "@ant-design/icons";
import { Link, useParams } from "react-router-dom";
import {
  api,
  basename,
  type TaskEvent,
  type TaskSnapshot,
} from "../api";
import { STATUS_TAG } from "./TasksPage";
import GlossaryPanel from "../components/GlossaryPanel";
import CueTable from "../components/CueTable";

const TERMINAL: string[] = ["done", "failed", "cancelled"];

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [notFound, setNotFound] = useState<string | null>(null);
  const [srtPath, setSrtPath] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!taskId) return;
    let source: EventSource | null = null;
    let cancelled = false;

    (async () => {
      try {
        const snapshot = await api.getTask(taskId);
        if (cancelled) return;
        setTask(snapshot);
        // SSE 先回放历史事件再实时推送；终态任务后端会直接关流
        source = api.subscribeTaskEvents(taskId, (event) => {
          setEvents((prev) => [...prev, event]);
          setTask((prev) =>
            prev
              ? {
                  ...prev,
                  status: event.status,
                  progress: event,
                }
              : prev,
          );
          if (TERMINAL.includes(event.status)) source?.close();
        });
      } catch (err) {
        if (!cancelled) setNotFound((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      source?.close();
    };
  }, [taskId]);

  // 日志自动滚到底部
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [events]);

  const doExport = useCallback(async () => {
    if (!task) return;
    setExporting(true);
    try {
      const resp = await api.exportProject(task.media[0] ?? task.path, true);
      setSrtPath(resp.srt_path);
      message.success("已导出 SRT");
    } catch (err) {
      message.error(`导出失败：${(err as Error).message}`);
    } finally {
      setExporting(false);
    }
  }, [task]);

  if (notFound) {
    return <Alert type="error" message={`任务加载失败：${notFound}`} />;
  }
  if (!task || !taskId) {
    return <Card loading />;
  }

  const projectPath = task.media[0] ?? task.path;
  const tag = STATUS_TAG[task.status];
  // 下载列表：pipeline 导出的产物（快照 artifacts）+ 本次手动导出的结果
  const downloadPaths = Array.from(
    new Set([...(task.artifacts ?? []), ...(srtPath ? [srtPath] : [])]),
  );

  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <Card>
        <Space direction="vertical" size={4} style={{ width: "100%" }}>
          <Space size="middle" wrap>
            <Link to="/">
              <Button size="small" icon={<ArrowLeftOutlined />}>
                返回列表
              </Button>
            </Link>
            <Typography.Title level={4} style={{ margin: 0 }}>
              {task.media.map(basename).join("、") || basename(task.path)}
            </Typography.Title>
            <Tag color={tag.color}>{tag.label}</Tag>
          </Space>
          <Typography.Text type="secondary" copyable>
            {task.path}
          </Typography.Text>
          {task.error && (
            <Alert type="error" message={task.error} style={{ marginTop: 8 }} />
          )}
          {task.status === "done" && (
            <Space style={{ marginTop: 8 }} wrap>
              <Button
                type="primary"
                icon={<ExportOutlined />}
                loading={exporting}
                onClick={() => void doExport()}
              >
                导出 SRT
              </Button>
              {downloadPaths.map((p) => (
                <Button
                  key={p}
                  icon={<DownloadOutlined />}
                  href={api.downloadUrl(p)}
                >
                  下载 {basename(p)}
                </Button>
              ))}
              {srtPath && (
                <Typography.Text type="success" copyable>
                  结果：{srtPath}
                </Typography.Text>
              )}
            </Space>
          )}
        </Space>
      </Card>

      {task.status === "waiting_confirm" && (
        <Card title="术语表确认">
          <GlossaryPanel
            path={projectPath}
            taskId={taskId}
            onResumed={() => {
              // resume 后状态经 SSE 推送更新，这里无需手动刷新
            }}
          />
        </Card>
      )}

      <Card>
        <Tabs
          items={[
            {
              key: "cues",
              label: "字幕校对",
              children: (
                <CueTable path={projectPath} videoPath={task.media[0]} />
              ),
            },
            {
              key: "events",
              label: `事件日志（${events.length}）`,
              children: (
                <div
                  ref={logRef}
                  style={{
                    maxHeight: 420,
                    overflow: "auto",
                    background: "#0d1117",
                    borderRadius: 6,
                    padding: 12,
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#c9d1d9",
                  }}
                >
                  {events.length === 0 && (
                    <Typography.Text type="secondary">
                      暂无事件
                    </Typography.Text>
                  )}
                  {events.map((event, i) => (
                    <div key={i} style={{ whiteSpace: "pre-wrap" }}>
                      <span style={{ color: "#8b949e" }}>
                        [{event.stage || "status"}]
                      </span>{" "}
                      {event.total > 0 && (
                        <span style={{ color: "#58a6ff" }}>
                          {event.done}/{event.total}{" "}
                        </span>
                      )}
                      {event.message}
                    </div>
                  ))}
                </div>
              ),
            },
          ]}
        />
      </Card>
    </Space>
  );
}
