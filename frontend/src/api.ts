/** API 薄封装：类型与后端 JSON 对齐（docs/DESIGN.md §8.2）。
 *
 * 所有请求走同源 /api（vite dev server 代理到 127.0.0.1:7860）。
 */

// ---------------------------------------------------------------- 类型

export type TaskStatus =
  | "pending"
  | "running"
  | "waiting_confirm"
  | "done"
  | "failed"
  | "cancelled";

/** pipeline 事件协议五键 + task_id + status + time（SSE 与任务快照共用） */
export interface TaskEvent {
  task_id: string;
  status: TaskStatus;
  media: string;
  stage: string; // transcribe | translate | export | ""（状态变更事件）
  done: number;
  total: number;
  message: string;
  time: number; // epoch 秒（server 记录，ETA / 阶段耗时估算用）
}

export interface TaskSnapshot {
  id: string;
  path: string;
  media: string[];
  options: TaskOptions; // 创建任务时的选项快照
  status: TaskStatus;
  created_at: number;
  error: string | null;
  progress: TaskEvent | null;
  artifacts: string[]; // 已存在的导出产物（SRT）路径，可经 /api/download 下载
}

export interface TaskOptions {
  auto_confirm: boolean;
  bilingual: boolean;
  transcribe_only: boolean;
  language: string | null;
}

export interface WordTiming {
  text: string;
  start: number;
  end: number;
}

export interface Cue {
  id: number;
  start: number;
  end: number;
  text: string;
  /** 未翻译的 cue 为 null（后端 models.py Cue.translation 是 Optional[str]），
   * 渲染前必须 ?? ""，不能当 string 直接用 */
  translation: string | null;
  words: WordTiming[];
  flags: string[]; // translation_failed / glossary_miss:<src>
}

export interface GlossaryEntry {
  src: string;
  dst: string;
  count: number;
  confirmed: boolean;
}

export interface Project {
  version: number;
  source: { file: string; duration: number; language: string | null };
  models: { asr: string; aligner: string; translator: string };
  summary: string;
  glossary: GlossaryEntry[];
  cues: Cue[];
  stage: "empty" | "transcribed" | "contexted" | "translated";
}

export interface MediaBrowse {
  path: string;
  parent: string | null;
  directories: string[];
  media: string[];
}

export interface AsrConfigPayload {
  backend: string;
  model: string;
  aligner_model: string;
  device: string;
  dtype: string;
  chunk_max_seconds: number;
  language: string | null;
  ffmpeg_path: string;
}

export interface TranslateConfigPayload {
  base_url: string;
  api_key: string | null; // masked，如 sk-...***
  api_key_env: string;
  api_key_resolved: boolean; // 密钥是否已可用（可能来自环境变量）
  model: string;
  temperature: number;
  history_count: number;
  forward_count: number;
  glossary_max_entries: number;
  additional_prompt: string;
  target_language: string;
  request_timeout: number;
  max_retries: number;
  glossary_max_retries: number;
}

export interface UiConfigPayload {
  host: string;
  port: number;
  upload_dir: string; // 空 = ~/.subtitle_translator/uploads
}

export interface LogConfigPayload {
  level: string; // DEBUG / INFO / WARNING / ERROR / CRITICAL（控制台级别；文件始终 DEBUG）
  dir: string; // 空 = ~/.subtitle_translator/logs
}

/** POST /api/upload 响应：批次目录 + 服务器侧文件路径（拿去建任务） */
export interface UploadResponse {
  batch: string;
  paths: string[];
}

/** POST /api/config/test 响应：翻译端点连通性测试 */
export interface ConfigTestResult {
  ok: boolean;
  latency_ms: number;
  response_preview: string; // 前 50 字符
  error: string | null;
}

/** POST /api/preflight 响应：启动预检报告（有 fail 即 ok=false） */
export interface PreflightCheck {
  name: string;
  status: "ok" | "warn" | "fail";
  message: string;
}

export interface PreflightReport {
  ok: boolean;
  checks: PreflightCheck[];
}

export interface ConfigPayload {
  asr: AsrConfigPayload;
  translate: TranslateConfigPayload;
  ui: UiConfigPayload;
  log: LogConfigPayload;
}

// ---------------------------------------------------------------- 请求封装

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // 保留默认 detail
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

function jsonBody(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// ---------------------------------------------------------------- 任务

export const api = {
  createTask: (path: string, options: TaskOptions) =>
    request<TaskSnapshot>("/api/tasks", jsonBody("POST", { path, options })),
  listTasks: () => request<TaskSnapshot[]>("/api/tasks"),
  getTask: (id: string) => request<TaskSnapshot>(`/api/tasks/${id}`),
  cancelTask: (id: string) =>
    request<TaskSnapshot>(`/api/tasks/${id}/cancel`, { method: "POST" }),
  resumeTask: (id: string) =>
    request<TaskSnapshot>(`/api/tasks/${id}/resume`, { method: "POST" }),

  /** 任务日志（GET /api/tasks/{id}/log，纯文本，尾部优先） */
  getTaskLog: async (id: string): Promise<string> => {
    const resp = await fetch(`/api/tasks/${id}/log`);
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = await resp.json();
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        // 保留默认 detail
      }
      throw new ApiError(resp.status, detail);
    }
    return resp.text();
  },

  /** SSE 订阅（历史回放 + 实时推送），返回 EventSource 由调用方关闭 */
  subscribeTaskEvents: (
    id: string,
    onEvent: (event: TaskEvent) => void,
  ): EventSource => {
    const source = new EventSource(`/api/tasks/${id}/events`);
    source.onmessage = (msg) => onEvent(JSON.parse(msg.data) as TaskEvent);
    return source;
  },

  // ---------------------------------------------------------------- 媒体

  browseMedia: (path: string) =>
    request<MediaBrowse>(`/api/media?path=${encodeURIComponent(path)}`),
  videoUrl: (path: string) => `/api/video?path=${encodeURIComponent(path)}`,

  // ---------------------------------------------------------------- 上传与下载

  /** 单文件上传到 /api/upload（XHR 才有上传进度事件）；多文件由调用方逐个调 */
  uploadFile: (
    file: File,
    onProgress?: (percent: number) => void,
  ): Promise<UploadResponse> =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("files", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
      xhr.onload = () => {
        let body: { detail?: string } & Partial<UploadResponse> = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          // 保留空 body
        }
        if (xhr.status >= 200 && xhr.status < 300 && body.paths) {
          resolve(body as UploadResponse);
        } else {
          reject(new ApiError(xhr.status, body.detail ?? `HTTP ${xhr.status}`));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "网络错误，上传失败"));
      xhr.send(form);
    }),
  downloadUrl: (path: string) => `/api/download?path=${encodeURIComponent(path)}`,

  // ---------------------------------------------------------------- 工程数据

  getProject: (path: string) =>
    request<Project>(`/api/project?path=${encodeURIComponent(path)}`),
  patchCue: (
    path: string,
    cueId: number,
    fields: { text?: string; translation?: string },
  ) =>
    request<Cue>(
      "/api/project/cues",
      jsonBody("PATCH", { path, cue_id: cueId, ...fields }),
    ),
  getGlossary: (path: string) =>
    request<{ path: string; glossary: GlossaryEntry[] }>(
      `/api/project/glossary?path=${encodeURIComponent(path)}`,
    ),
  patchGlossary: (
    path: string,
    body: {
      updates?: { src: string; new_src?: string; dst?: string }[];
      confirm?: string[];
      confirm_all?: boolean;
    },
  ) =>
    request<{ path: string; glossary: GlossaryEntry[] }>(
      "/api/project/glossary",
      jsonBody("PATCH", { path, ...body }),
    ),
  exportProject: (path: string, bilingual: boolean) =>
    request<{ srt_path: string; download_url: string }>(
      "/api/project/export",
      jsonBody("POST", { path, bilingual }),
    ),

  // ---------------------------------------------------------------- 配置

  getConfig: () => request<ConfigPayload>("/api/config"),
  /** 局部更新：只传改动节；api_key 传空字符串 / mask 值表示不修改 */
  updateConfig: (payload: Partial<ConfigPayload>) =>
    request<ConfigPayload>("/api/config", jsonBody("PUT", payload)),
  /** 翻译端点连通性测试；translate 携带未保存的表单值，缺省用磁盘 config */
  testConfig: (translate?: Record<string, unknown>) =>
    request<ConfigTestResult>(
      "/api/config/test",
      jsonBody("POST", translate ? { translate } : {}),
    ),
  /** 完整启动预检（设置页「系统检查」入口） */
  runPreflight: () =>
    request<PreflightReport>("/api/preflight", { method: "POST" }),
};

// ---------------------------------------------------------------- 工具

/** 秒 → HH:MM:SS.mmm（cue 时间轴显示） */
export function formatTime(seconds: number): string {
  const ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(milli, 3)}`;
}

/** 文件全路径 →  basename */
export function basename(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}
