/** 任务详情页 v2：以任务状态 + SSE 事件流为主体，project JSON 是可选增强。
 *
 * - JSON 尚未生成（任务早期 404）时各区块显示空态，绝不弹错误；
 *   阶段事件推进（阶段切换 / 完成 / 状态变更）时重新拉取 JSON 刷新；
 * - 头部：文件名、状态 Tag、媒体路径（上传任务标注「上传副本」）、操作组
 *   （取消 / 恢复 / 导出 / 下载 SRT / 查看日志），failed 直接展示错误摘要；
 * - 进度区：Steps 五阶段（转录 → 建档 → 术语确认 → 翻译 → 导出，仅转录任务
 *   为两阶段），当前阶段下进度条 + done/total + 当前消息 + ETA（事件时间戳
 *   估算，进度 <2% 显示「估算中」）；
 * - 下方 Tabs：字幕校对 / 术语表 / 摘要 / 运行信息；日志走头部抽屉。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  Empty,
  List,
  Popconfirm,
  Progress,
  Space,
  Steps,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  ExportOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { Link, useParams } from "react-router-dom";
import {
  api,
  ApiError,
  basename,
  type Cue,
  type Project,
  type TaskEvent,
  type TaskSnapshot,
} from "../api";
import { STATUS_TAG } from "./TasksPage";
import GlossaryPanel from "../components/GlossaryPanel";
import CueTable from "../components/CueTable";

const TERMINAL: string[] = ["done", "failed", "cancelled"];

/** 上传批次目录命名（与 server BATCH_DIR_PATTERN 一致），命中即「上传副本」 */
const UPLOAD_BATCH_PATTERN = /\d{8}-\d{6}-[0-9a-f]{6}/;

const FULL_STEPS = ["转录", "建档", "术语确认", "翻译", "导出"];
const TRANSCRIBE_ONLY_STEPS = ["转录", "导出"];

type StepStatus = "wait" | "process" | "finish" | "error";

interface StepsState {
  titles: string[];
  current: number; // -1 = 排队中（尚未开始）
  statuses: StepStatus[];
  /** 当前进行中的 pipeline stage（transcribe/translate/export），无则 null */
  activeStage: string | null;
}

/** 由事件流 + 任务状态推导 Steps 展示状态 */
export function analyzeSteps(
  task: TaskSnapshot,
  events: TaskEvent[],
): StepsState {
  const transcribeOnly = !!task.options?.transcribe_only;
  const titles = transcribeOnly ? TRANSCRIBE_ONLY_STEPS : FULL_STEPS;
  const lastStage = [...events].reverse().find((e) => e.stage !== "");

  let current: number;
  if (task.status === "done") {
    current = titles.length; // 全部完成
  } else if (task.status === "pending") {
    current = -1;
  } else if (task.status === "waiting_confirm") {
    current = transcribeOnly ? 1 : 2;
  } else if (!lastStage) {
    current = 0; // running 但还没有阶段事件
  } else if (lastStage.stage === "transcribe") {
    current = /完成/.test(lastStage.message) ? 1 : 0;
  } else if (lastStage.stage === "translate") {
    if (transcribeOnly) {
      current = 1; // 不会出现，兜底
    } else if (lastStage.total > 0) {
      current = 3;
    } else if (/待人工确认|待确认/.test(lastStage.message)) {
      current = 2;
    } else {
      current = 1;
    }
  } else {
    // export
    current = transcribeOnly ? 1 : 4;
  }

  const statuses: StepStatus[] = titles.map((_, i) => {
    if (current >= titles.length) return "finish";
    if (i < current) return "finish";
    if (i > current || current < 0) return "wait";
    if (task.status === "failed") return "error";
    if (task.status === "cancelled") return "wait";
    return "process";
  });

  // 当前阶段对应的 pipeline stage（进度条数据从该 stage 的事件里取）
  const activeStage =
    current < 0 || current >= titles.length
      ? null
      : transcribeOnly
        ? current === 0
          ? "transcribe"
          : "export"
        : (["transcribe", null, null, "translate", "export"] as const)[current];

  return { titles, current, statuses, activeStage };
}

/** 秒 → 人可读时长（x 秒 / x 分 y 秒 / x 小时 y 分） */
function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.floor((s % 3600) / 60)} 分`;
}

/** ETA：已耗时 ÷ 完成比例估算剩余；进度 <2% 或样本不足返回 null（显示「估算中」） */
function estimateEta(stageEvents: TaskEvent[]): number | null {
  const progressEvents = stageEvents.filter((e) => e.total > 0);
  const last = progressEvents[progressEvents.length - 1];
  if (!last || last.done >= last.total) return null;
  const ratio = last.done / last.total;
  if (ratio < 0.02 || progressEvents.length < 2) return null;
  const elapsed = last.time - progressEvents[0].time;
  if (elapsed <= 0) return null;
  return (elapsed / ratio) * (1 - ratio);
}

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [notFound, setNotFound] = useState<string | null>(null);
  const [srtPath, setSrtPath] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [logText, setLogText] = useState<string | null>(null);
  const [logLoading, setLogLoading] = useState(false);
  const projectPathRef = useRef<string | null>(null);
  const lastStageRef = useRef<string>("");
  const refreshTimerRef = useRef<number | null>(null);

  /** 拉取工程 JSON：404 = 尚未生成 → 空态（绝不弹错误）；其他错误才提示 */
  const loadProject = useCallback(async (path: string) => {
    try {
      setProject(await api.getProject(path));
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setProject(null);
      } else {
        message.error(`加载工程文件失败：${(err as Error).message}`);
      }
    }
  }, []);

  /** 阶段事件推进后防抖刷新一次 JSON（历史回放会一次性涌入多条事件） */
  const scheduleProjectRefresh = useCallback(() => {
    const path = projectPathRef.current;
    if (!path) return;
    if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
    refreshTimerRef.current = window.setTimeout(() => {
      void loadProject(path);
    }, 300);
  }, [loadProject]);

  useEffect(() => {
    if (!taskId) return;
    let source: EventSource | null = null;
    let cancelled = false;

    (async () => {
      try {
        const snapshot = await api.getTask(taskId);
        if (cancelled) return;
        setTask(snapshot);
        projectPathRef.current = snapshot.media[0] ?? snapshot.path;
        void loadProject(projectPathRef.current);
        // SSE 先回放历史事件再实时推送；终态任务后端会直接关流
        source = api.subscribeTaskEvents(taskId, (event) => {
          setEvents((prev) => [...prev, event]);
          setTask((prev) =>
            prev ? { ...prev, status: event.status, progress: event } : prev,
          );
          // 阶段推进（阶段切换 / 阶段完成 / 状态变更）时刷新一次工程 JSON
          const stageAdvanced =
            event.stage === "" ||
            (event.stage !== "" && event.stage !== lastStageRef.current) ||
            (event.total > 0 && event.done === event.total);
          if (event.stage) lastStageRef.current = event.stage;
          if (stageAdvanced) scheduleProjectRefresh();
          if (TERMINAL.includes(event.status)) source?.close();
        });
      } catch (err) {
        if (!cancelled) setNotFound((err as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      source?.close();
      if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
    };
  }, [taskId, loadProject, scheduleProjectRefresh]);

  const doExport = useCallback(async () => {
    const path = projectPathRef.current;
    if (!path) return;
    setExporting(true);
    try {
      const resp = await api.exportProject(path, true);
      setSrtPath(resp.srt_path);
      message.success("已导出 SRT");
    } catch (err) {
      message.error(`导出失败：${(err as Error).message}`);
    } finally {
      setExporting(false);
    }
  }, []);

  const doCancel = useCallback(async () => {
    if (!taskId) return;
    try {
      await api.cancelTask(taskId);
      message.success("取消请求已发送（运行中的任务在当前条目完成后停止）");
    } catch (err) {
      message.error(`取消失败：${(err as Error).message}`);
    }
  }, [taskId]);

  const doResume = useCallback(async () => {
    if (!taskId) return;
    setResuming(true);
    try {
      await api.resumeTask(taskId);
      message.success("任务已重新排队");
    } catch (err) {
      message.error(`恢复失败：${(err as Error).message}`);
    } finally {
      setResuming(false);
    }
  }, [taskId]);

  const openLog = useCallback(async () => {
    if (!taskId) return;
    setLogOpen(true);
    setLogLoading(true);
    try {
      setLogText(await api.getTaskLog(taskId));
    } catch (err) {
      setLogText(null);
      message.warning(`日志不可用：${(err as Error).message}`);
    } finally {
      setLogLoading(false);
    }
  }, [taskId]);

  if (notFound) {
    return <Alert type="error" message={`任务加载失败：${notFound}`} />;
  }
  if (!task || !taskId) {
    return <Card loading />;
  }

  const projectPath = task.media[0] ?? task.path;
  const tag = STATUS_TAG[task.status];
  const isUpload = UPLOAD_BATCH_PATTERN.test(task.path);
  // 下载列表：pipeline 导出的产物（快照 artifacts）+ 本次手动导出的结果
  const downloadPaths = Array.from(
    new Set([...(task.artifacts ?? []), ...(srtPath ? [srtPath] : [])]),
  );

  // ---------------------------------------------------------- 进度区
  const steps = analyzeSteps(task, events);
  const stageEvents = steps.activeStage
    ? events.filter((e) => e.stage === steps.activeStage)
    : [];
  const lastProgress = [...stageEvents].reverse().find((e) => e.total > 0);
  const eta = lastProgress ? estimateEta(stageEvents) : null;
  const currentMessage = [...events].reverse().find((e) => e.message)?.message;

  // ---------------------------------------------------------- 运行信息
  interface StageRow {
    key: string;
    label: string;
    start: string;
    duration: string;
  }
  const firstEventTime = events.length ? events[0].time : null;
  const stageDurations: StageRow[] = (
    [
      ["transcribe", "转录"],
      ["translate", "翻译（含建档）"],
      ["export", "导出"],
    ] as const
  )
    .map(([stage, label]): StageRow | null => {
      const evts = events.filter((e) => e.stage === stage);
      if (!evts.length) return null;
      return {
        key: stage,
        label,
        start: new Date(evts[0].time * 1000).toLocaleTimeString(),
        duration: formatDuration(evts[evts.length - 1].time - evts[0].time),
      };
    })
    .filter((r): r is StageRow => r !== null);
  if (firstEventTime) {
    stageDurations.unshift({
      key: "queue",
      label: "排队",
      start: new Date(task.created_at * 1000).toLocaleTimeString(),
      duration: formatDuration(firstEventTime - task.created_at),
    });
  }
  const failedCues: Cue[] = (project?.cues ?? []).filter((c) =>
    (c.flags ?? []).includes("translation_failed"),
  );

  const canAct =
    task.status === "pending" ||
    task.status === "running" ||
    task.status === "waiting_confirm";

  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      {/* ------------------------------------------------ 头部 */}
      <Card>
        <Space direction="vertical" size={8} style={{ width: "100%" }}>
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
            {isUpload && (
              <Tag color="orange">上传副本（服务退出时清理）</Tag>
            )}
          </Space>
          <Typography.Text type="secondary" copyable>
            {task.path}
          </Typography.Text>
          <Space wrap>
            {canAct && (
              <Popconfirm
                title="确认取消该任务？"
                description="运行中的任务会在当前条目完成后停止，进度已保存"
                onConfirm={() => void doCancel()}
              >
                <Button size="small" danger icon={<StopOutlined />}>
                  取消
                </Button>
              </Popconfirm>
            )}
            {task.status === "waiting_confirm" && (
              <Popconfirm
                title="直接恢复任务？"
                description="若术语表尚未全部确认，任务会再次停在检查点；建议先到「术语表」页签确认"
                onConfirm={() => void doResume()}
              >
                <Button
                  size="small"
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  loading={resuming}
                >
                  恢复
                </Button>
              </Popconfirm>
            )}
            <Button
              size="small"
              icon={<ExportOutlined />}
              loading={exporting}
              disabled={!project?.cues?.length}
              onClick={() => void doExport()}
            >
              导出 SRT
            </Button>
            {downloadPaths.map((p) => (
              <Button
                key={p}
                size="small"
                icon={<DownloadOutlined />}
                href={api.downloadUrl(p)}
              >
                下载 {basename(p)}
              </Button>
            ))}
            <Button
              size="small"
              icon={<FileTextOutlined />}
              onClick={() => void openLog()}
            >
              查看日志
            </Button>
          </Space>
          {task.status === "failed" && task.error && (
            <Alert type="error" showIcon message="任务失败" description={task.error} />
          )}
          {task.status === "waiting_confirm" && (
            <Alert
              type="warning"
              showIcon
              message="任务在术语确认检查点暂停，请到下方「术语表」页签核对并确认后继续"
            />
          )}
        </Space>
      </Card>

      {/* ------------------------------------------------ 进度区 */}
      <Card>
        <Steps
          size="small"
          current={Math.max(steps.current, 0)}
          items={steps.titles.map((title, i) => ({
            title,
            status: steps.statuses[i],
          }))}
        />
        {steps.current < 0 && (
          <Typography.Text type="secondary" style={{ marginTop: 12, display: "block" }}>
            排队等待中（单并发，前面的任务跑完即开始）
          </Typography.Text>
        )}
        {steps.current >= 0 && steps.current < steps.titles.length && (
          <div style={{ marginTop: 16, maxWidth: 560 }}>
            {lastProgress ? (
              <Space direction="vertical" size={4} style={{ width: "100%" }}>
                <Progress
                  percent={Math.round((lastProgress.done / lastProgress.total) * 100)}
                  size="small"
                  status={
                    task.status === "failed"
                      ? "exception"
                      : task.status === "done"
                        ? "success"
                        : "active"
                  }
                />
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {lastProgress.done}/{lastProgress.total}
                  {currentMessage ? ` · ${currentMessage}` : ""}
                  {" · 预计剩余 "}
                  {eta != null ? formatDuration(eta) : "估算中…"}
                </Typography.Text>
              </Space>
            ) : (
              currentMessage && (
                <Typography.Text type="secondary">
                  {currentMessage}
                  {steps.titles[steps.current] === "术语确认"
                    ? ""
                    : "（该阶段无细分进度）"}
                </Typography.Text>
              )
            )}
          </div>
        )}
      </Card>

      {/* ------------------------------------------------ Tabs */}
      <Card>
        <Tabs
          items={[
            {
              key: "cues",
              label: "字幕校对",
              children: (
                <CueTable
                  path={projectPath}
                  videoPath={task.media[0]}
                  project={project}
                  onProjectChange={setProject}
                />
              ),
            },
            {
              key: "glossary",
              label: "术语表",
              children: (
                <GlossaryPanel
                  path={projectPath}
                  taskId={taskId}
                  taskStatus={task.status}
                  project={project}
                  onProjectChange={setProject}
                />
              ),
            },
            {
              key: "summary",
              label: "摘要",
              children: project?.summary ? (
                <Typography.Paragraph
                  style={{ background: "#fafafa", padding: 12, borderRadius: 6 }}
                  ellipsis={{ rows: 8, expandable: true, symbol: "展开" }}
                >
                  {project.summary}
                </Typography.Paragraph>
              ) : (
                <Empty description="尚未生成（建档完成后可见）" />
              ),
            },
            {
              key: "info",
              label: "运行信息",
              children: (
                <Space direction="vertical" size="large" style={{ width: "100%" }}>
                  <Descriptions title="任务选项" size="small" column={4}>
                    <Descriptions.Item label="自动确认术语">
                      {task.options?.auto_confirm ? "是" : "否"}
                    </Descriptions.Item>
                    <Descriptions.Item label="双语导出">
                      {task.options?.bilingual ? "是" : "否"}
                    </Descriptions.Item>
                    <Descriptions.Item label="仅转录">
                      {task.options?.transcribe_only ? "是" : "否"}
                    </Descriptions.Item>
                    <Descriptions.Item label="源语言">
                      {task.options?.language || "自动检测"}
                    </Descriptions.Item>
                  </Descriptions>
                  {project ? (
                    <Descriptions title="工程信息" size="small" column={2}>
                      <Descriptions.Item label="ASR 模型">
                        {project.models?.asr || "—"}
                      </Descriptions.Item>
                      <Descriptions.Item label="翻译模型">
                        {project.models?.translator || "—"}
                      </Descriptions.Item>
                      <Descriptions.Item label="媒体时长">
                        {project.source?.duration
                          ? formatDuration(project.source.duration)
                          : "—"}
                      </Descriptions.Item>
                      <Descriptions.Item label="语种">
                        {project.source?.language || "—"}
                      </Descriptions.Item>
                    </Descriptions>
                  ) : (
                    <Alert type="info" showIcon message="工程文件尚未生成，模型 / 时长等信息暂不可用" />
                  )}
                  {stageDurations.length > 0 && (
                    <Table
                      title={() => "阶段耗时"}
                      rowKey="key"
                      size="small"
                      pagination={false}
                      dataSource={stageDurations}
                      columns={[
                        { title: "阶段", dataIndex: "label" },
                        { title: "开始时间", dataIndex: "start" },
                        { title: "耗时", dataIndex: "duration" },
                      ]}
                    />
                  )}
                  {project && failedCues.length > 0 && (
                    <List
                      size="small"
                      header={
                        <Typography.Text type="danger">
                          翻译失败的 cue（{failedCues.length} 条，保留原文占位）
                        </Typography.Text>
                      }
                      bordered
                      dataSource={failedCues}
                      renderItem={(cue) => (
                        <List.Item>
                          <Typography.Text>
                            #{cue.id} {cue.text}
                          </Typography.Text>
                        </List.Item>
                      )}
                    />
                  )}
                  <div>
                    <Typography.Text strong>事件记录（{events.length}）</Typography.Text>
                    <div
                      style={{
                        marginTop: 8,
                        maxHeight: 320,
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
                        <Typography.Text type="secondary">暂无事件</Typography.Text>
                      )}
                      {events.map((event, i) => (
                        <div key={i} style={{ whiteSpace: "pre-wrap" }}>
                          <span style={{ color: "#8b949e" }}>
                            {new Date(event.time * 1000).toLocaleTimeString()} [
                            {event.stage || "status"}]
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
                  </div>
                </Space>
              ),
            },
          ]}
        />
      </Card>

      <Drawer
        title="任务日志"
        placement="right"
        width={720}
        open={logOpen}
        onClose={() => setLogOpen(false)}
        extra={
          <Button size="small" onClick={() => void openLog()} loading={logLoading}>
            刷新
          </Button>
        }
      >
        {logLoading && !logText ? (
          <Typography.Text type="secondary">加载中…</Typography.Text>
        ) : logText ? (
          <pre
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
              fontFamily: "monospace",
              fontSize: 12,
              margin: 0,
            }}
          >
            {logText}
          </pre>
        ) : (
          <Typography.Text type="secondary">
            暂无日志（任务尚未开始执行，或日志文件已被清理）
          </Typography.Text>
        )}
      </Drawer>
    </Space>
  );
}
