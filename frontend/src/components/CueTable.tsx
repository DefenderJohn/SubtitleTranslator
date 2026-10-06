/** 字幕校对：cue 表格（译文可编辑）+ 视频预览（点行跳转播放）。
 *
 * project JSON 由详情页统一拉取传入（404 = 尚未生成 → 空态，不弹错误）；
 * 顶部统计行：共 N 条 / 已翻译 N / 术语标记 N（红，点击切换只看标记行）。
 * 编辑保存调 PATCH /api/project/cues 后回调 onProjectChange 同步上层。
 */
import { useState } from "react";
import {
  Button,
  Empty,
  Input,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { CheckOutlined, CloseOutlined, EditOutlined } from "@ant-design/icons";
import { api, basename, formatTime, type Cue, type Project } from "../api";

interface Props {
  /** 媒体路径或 .sub.json 路径（后端两种都接受） */
  path: string;
  /** 视频预览路径；不传（如纯音频 / from_json 工程）则隐藏预览面板 */
  videoPath?: string;
  /** 工程 JSON；null = 尚未生成（任务早期）→ 空态 */
  project: Project | null;
  onProjectChange: (project: Project) => void;
}

/** 术语后校验未命中的标记（glossary_miss:<src>） */
const cueFlags = (cue: Cue): string[] => cue.flags ?? [];

const isGlossaryMiss = (cue: Cue) =>
  cueFlags(cue).some((f) => f.startsWith("glossary_miss:"));

export default function CueTable({ path, videoPath, project, onProjectChange }: Props) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [onlyMarked, setOnlyMarked] = useState(false);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);

  const saveCue = async (cue: Cue) => {
    try {
      const updated = await api.patchCue(path, cue.id, { translation: draft });
      onProjectChange({
        ...project!,
        cues: project!.cues.map((c) => (c.id === cue.id ? updated : c)),
      });
      setEditingId(null);
      message.success(`已保存第 ${cue.id} 条`);
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    }
  };

  const seekTo = (cue: Cue) => {
    if (videoEl) {
      videoEl.currentTime = cue.start;
      void videoEl.play();
    }
  };

  const isVideo =
    videoPath != null &&
    /\.(mp4|webm|mkv|mov|avi|m4v|mpeg|mpga|ogg)$/i.test(videoPath);

  if (!project) {
    return <Empty description="工程文件尚未生成（转录完成后可校对）" />;
  }

  const cues = project.cues ?? [];
  // translation 在逐句翻译前是 null（后端 Optional[str]），统计时按空串处理
  const translatedCount = cues.filter((c) => (c.translation ?? "").trim()).length;
  const markedCount = cues.filter(isGlossaryMiss).length;
  const shown = onlyMarked ? cues.filter(isGlossaryMiss) : cues;

  const stats = (
    <Space size="large" style={{ marginBottom: 12 }} wrap>
      <Statistic title="共" value={cues.length} suffix="条" />
      <Statistic title="已翻译" value={translatedCount} suffix="条" />
      <span
        onClick={() => setOnlyMarked((v) => !v)}
        style={{ cursor: markedCount ? "pointer" : "default" }}
        title={markedCount ? "点击切换只看术语标记行" : undefined}
      >
        <Statistic
          title={onlyMarked ? "术语标记（只看标记行，点击取消）" : "术语标记"}
          value={markedCount}
          suffix="条"
          valueStyle={{ color: markedCount ? "#cf1322" : undefined }}
        />
      </span>
    </Space>
  );

  const table = (
    <Table<Cue>
      rowKey="id"
      size="small"
      dataSource={shown}
      pagination={{ pageSize: 50, showSizeChanger: true }}
      onRow={(cue) => ({
        onClick: () => seekTo(cue),
        style: { cursor: isVideo ? "pointer" : "default" },
      })}
      columns={[
        { title: "#", dataIndex: "id", width: 56 },
        {
          title: "起止时间",
          width: 190,
          render: (_, cue) => (
            <Typography.Text code style={{ fontSize: 12 }}>
              {formatTime(cue.start)} → {formatTime(cue.end)}
            </Typography.Text>
          ),
        },
        { title: "原文", dataIndex: "text" },
        {
          title: "译文",
          render: (_, cue) => {
            const misses = cueFlags(cue).filter((f) => f.startsWith("glossary_miss:"));
            if (editingId === cue.id) {
              return (
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Input.TextArea
                    autoSize={{ minRows: 2 }}
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <Space onClick={(e) => e.stopPropagation()}>
                    <Button
                      size="small"
                      type="primary"
                      icon={<CheckOutlined />}
                      onClick={() => void saveCue(cue)}
                    >
                      保存
                    </Button>
                    <Button
                      size="small"
                      icon={<CloseOutlined />}
                      onClick={() => setEditingId(null)}
                    >
                      取消
                    </Button>
                  </Space>
                </Space>
              );
            }
            return (
              <Space direction="vertical" size={2}>
                <Space size={4}>
                  <span>{cue.translation || "—"}</span>
                  <Button
                    size="small"
                    type="text"
                    icon={<EditOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditingId(cue.id);
                      setDraft(cue.translation ?? "");
                    }}
                  />
                </Space>
                {misses.map((flag) => (
                  <Tag key={flag} color="red" style={{ fontSize: 12 }}>
                    术语未命中：{flag.slice("glossary_miss:".length)}
                  </Tag>
                ))}
                {cueFlags(cue).includes("translation_failed") && (
                  <Tag color="red" style={{ fontSize: 12 }}>
                    翻译失败（保留原文占位）
                  </Tag>
                )}
              </Space>
            );
          },
        },
      ]}
    />
  );

  return (
    <div>
      {stats}
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div style={{ flex: 1, minWidth: 0 }}>{table}</div>
        {isVideo && videoPath && (
          <div style={{ width: 360, flexShrink: 0 }}>
            <video
              ref={setVideoEl}
              controls
              style={{ width: "100%", background: "#000", borderRadius: 6 }}
              src={api.videoUrl(videoPath)}
            />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {basename(videoPath)}（点击左侧 cue 行跳转播放）
            </Typography.Text>
          </div>
        )}
      </div>
    </div>
  );
}
