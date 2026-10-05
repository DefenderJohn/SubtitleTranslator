/** 字幕校对：cue 表格（译文可编辑）+ 视频预览（点行跳转播放）。
 *
 * 术语不一致标记（glossary_miss:<src>）在译文单元格红色提示；
 * 编辑保存调 PATCH /api/project/cues。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Button,
  Input,
  Space,
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
}

export default function CueTable({ path, videoPath }: Props) {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setProject(await api.getProject(path));
    } catch (err) {
      message.error(`加载工程失败：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveCue = async (cue: Cue) => {
    try {
      const updated = await api.patchCue(path, cue.id, { translation: draft });
      setProject((prev) =>
        prev
          ? {
              ...prev,
              cues: prev.cues.map((c) => (c.id === cue.id ? updated : c)),
            }
          : prev,
      );
      setEditingId(null);
      message.success(`已保存第 ${cue.id} 条`);
    } catch (err) {
      message.error(`保存失败：${(err as Error).message}`);
    }
  };

  const seekTo = (cue: Cue) => {
    const video = videoRef.current;
    if (video) {
      video.currentTime = cue.start;
      void video.play();
    }
  };

  const isVideo =
    videoPath != null &&
    /\.(mp4|webm|mkv|mov|avi|m4v|mpeg|mpga|ogg)$/i.test(videoPath);

  const table = (
    <Table<Cue>
      rowKey="id"
      size="small"
      loading={loading}
      dataSource={project?.cues ?? []}
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
            const misses = cue.flags.filter((f) => f.startsWith("glossary_miss:"));
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
                      setDraft(cue.translation);
                    }}
                  />
                </Space>
                {misses.map((flag) => (
                  <Tag key={flag} color="red" style={{ fontSize: 12 }}>
                    术语未命中：{flag.slice("glossary_miss:".length)}
                  </Tag>
                ))}
                {cue.flags.includes("translation_failed") && (
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
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <div style={{ flex: 1, minWidth: 0 }}>{table}</div>
      {isVideo && videoPath && (
        <div style={{ width: 360, flexShrink: 0 }}>
          <video
            ref={videoRef}
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
  );
}
