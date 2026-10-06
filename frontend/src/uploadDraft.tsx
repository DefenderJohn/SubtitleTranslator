/** 「新建任务」草稿的全局状态（context + useReducer，不引状态库）。
 *
 * 修复用户反馈：上传文件后切到设置页再回来，上传列表被清空（组件 state
 * 随路由卸载丢失）。草稿提升到 App 级 provider，路由切换不丢。
 *
 * 注意失效语义：server 重启后上传批次目录会被清理（见 docs/DESIGN.md §8.1
 * 上传生命周期），草稿里记的服务器路径随之失效。TasksPage 挂载时按批次目录
 * 逐一探测（GET /api/media），失效时提示并允许一键清空；创建任务时的 404
 * 作为兜底。
 */
import { createContext, useContext, useMemo, useReducer } from "react";
import type { ReactNode } from "react";
import type { UploadFile } from "antd";
import type { TaskOptions } from "./api";

export type SourceMode = "path" | "upload";

/** 上传成功的文件：uid（antd Upload）→ 服务器侧路径 + 所属批次目录 */
export interface UploadedEntry {
  uid: string;
  path: string;
  batch: string;
}

export interface DraftState {
  mode: SourceMode;
  /** 服务器路径模式的输入值 */
  path: string;
  /** 任务选项（language 空串 = null 自动检测） */
  options: Omit<TaskOptions, "language"> & { language: string };
  /** antd Upload 的展示列表（含上传进度态） */
  fileList: UploadFile[];
  /** 已上传完成的文件 → 服务器路径 */
  uploaded: UploadedEntry[];
}

export const DEFAULT_DRAFT: DraftState = {
  mode: "path",
  path: "",
  options: {
    auto_confirm: false,
    bilingual: true,
    transcribe_only: false,
    language: "",
  },
  fileList: [],
  uploaded: [],
};

type DraftAction =
  | { type: "setMode"; mode: SourceMode }
  | { type: "setPath"; path: string }
  | { type: "setOptions"; options: Partial<DraftState["options"]> }
  | { type: "setFileList"; fileList: UploadFile[] }
  | { type: "addUploaded"; entry: UploadedEntry }
  | { type: "removeUploaded"; uid: string }
  | { type: "clearUploads" }
  | { type: "reset" };

function reducer(state: DraftState, action: DraftAction): DraftState {
  switch (action.type) {
    case "setMode":
      return { ...state, mode: action.mode };
    case "setPath":
      return { ...state, path: action.path };
    case "setOptions":
      return { ...state, options: { ...state.options, ...action.options } };
    case "setFileList":
      return { ...state, fileList: action.fileList };
    case "addUploaded":
      return { ...state, uploaded: [...state.uploaded, action.entry] };
    case "removeUploaded":
      return {
        ...state,
        uploaded: state.uploaded.filter((u) => u.uid !== action.uid),
      };
    case "clearUploads":
      return { ...state, fileList: [], uploaded: [] };
    case "reset":
      return DEFAULT_DRAFT;
  }
}

interface DraftContextValue {
  draft: DraftState;
  dispatch: React.Dispatch<DraftAction>;
}

const DraftContext = createContext<DraftContextValue | null>(null);

export function UploadDraftProvider({ children }: { children: ReactNode }) {
  const [draft, dispatch] = useReducer(reducer, DEFAULT_DRAFT);
  const value = useMemo(() => ({ draft, dispatch }), [draft]);
  return <DraftContext.Provider value={value}>{children}</DraftContext.Provider>;
}

export function useUploadDraft(): DraftContextValue {
  const ctx = useContext(DraftContext);
  if (!ctx) throw new Error("useUploadDraft 必须在 UploadDraftProvider 内使用");
  return ctx;
}
