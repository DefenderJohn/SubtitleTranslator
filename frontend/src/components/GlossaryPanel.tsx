/** 术语表确认面板：waiting_confirm 任务的核心交互。
 *
 * 展示摘要 + 可编辑术语表（译文可改、逐条确认），
 * 「全部确认并继续」→ PATCH confirm_all → POST resume。
 */
import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Popconfirm,
  Space,
  Table,
  Typography,
  message,
} from "antd";
import { api, type GlossaryEntry } from "../api";

interface Props {
  /** 媒体路径或 .sub.json 路径（后端两种都接受） */
  path: string;
  taskId: string;
  onResumed: () => void;
}

export default function GlossaryPanel({ path, taskId, onResumed }: Props) {
  const [summary, setSummary] = useState<string>("");
  const [glossary, setGlossary] = useState<GlossaryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [resuming, setResuming] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [project, gl] = await Promise.all([
        api.getProject(path),
        api.getGlossary(path),
      ]);
      setSummary(project.summary);
      setGlossary(gl.glossary);
    } catch (err) {
      message.error(`加载术语表失败：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveEntry = async (entry: GlossaryEntry, dst: string) => {
    if (dst === entry.dst) return;
    try {
      const resp = await api.patchGlossary(path, {
        updates: [{ src: entry.src, dst }],
      });
      setGlossary(resp.glossary);
      message.success(`已更新术语：${entry.src}`);
    } catch (err) {
      message.error(`更新失败：${(err as Error).message}`);
    }
  };

  const confirmEntry = async (entry: GlossaryEntry) => {
    try {
      const resp = await api.patchGlossary(path, { confirm: [entry.src] });
      setGlossary(resp.glossary);
    } catch (err) {
      message.error(`确认失败：${(err as Error).message}`);
    }
  };

  const confirmAllAndResume = async () => {
    setResuming(true);
    try {
      await api.patchGlossary(path, { confirm_all: true });
      await api.resumeTask(taskId);
      message.success("术语表已全部确认，任务已重新排队");
      onResumed();
    } catch (err) {
      message.error(`操作失败：${(err as Error).message}`);
    } finally {
      setResuming(false);
    }
  };

  const confirmedCount = glossary.filter((g) => g.confirmed).length;

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Alert
        type="warning"
        showIcon
        message="任务在术语表人工确认检查点暂停"
        description="请核对下面的摘要与术语表（译文可直接修改），全部确认后任务会继续逐句翻译。"
      />
      {summary && (
        <Typography.Paragraph
          style={{ background: "#fafafa", padding: 12, borderRadius: 6 }}
          ellipsis={{ rows: 4, expandable: true, symbol: "展开" }}
        >
          <Typography.Text strong>摘要：</Typography.Text>
          {summary}
        </Typography.Paragraph>
      )}
      <Table<GlossaryEntry>
        rowKey="src"
        size="small"
        loading={loading}
        dataSource={glossary}
        pagination={false}
        columns={[
          { title: "原文", dataIndex: "src", width: "30%" },
          {
            title: "译文（点击编辑）",
            dataIndex: "dst",
            width: "40%",
            render: (dst: string, entry) => (
              <Typography.Text
                editable={{
                  text: dst,
                  onChange: (value) => void saveEntry(entry, value),
                  triggerType: ["text", "icon"],
                }}
              >
                {dst}
              </Typography.Text>
            ),
          },
          { title: "次数", dataIndex: "count", width: 80 },
          {
            title: "确认",
            dataIndex: "confirmed",
            width: 80,
            render: (confirmed: boolean, entry) => (
              // 后端确认是单向操作（无 unconfirm），已确认的条目禁止取消勾选
              <Checkbox
                checked={confirmed}
                disabled={confirmed}
                onChange={() => void confirmEntry(entry)}
              />
            ),
          },
        ]}
      />
      <Space>
        <Typography.Text type="secondary">
          已确认 {confirmedCount}/{glossary.length}
        </Typography.Text>
        <Popconfirm
          title="全部确认并继续？"
          description="将把全部条目标记为已确认并重新排队任务"
          onConfirm={() => void confirmAllAndResume()}
        >
          <Button type="primary" loading={resuming}>
            全部确认并继续
          </Button>
        </Popconfirm>
      </Space>
    </Space>
  );
}
