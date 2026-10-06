/** 术语表页签：建档完成（stage ≥ contexted）后随时可看可改，不再局限于 waiting_confirm。
 *
 * project JSON 由详情页统一拉取传入（尚未生成 → 空态）；
 * 保留确认流程：waiting_confirm 时「全部确认并继续」（confirm_all + resume），
 * 其他状态只给「全部确认」；逐条确认 / 译文编辑任何状态可用。
 * 「重翻受影响行」本版不做（只读 + 编辑 + 确认）。
 */
import { useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Popconfirm,
  Space,
  Table,
  Typography,
  message,
} from "antd";
import {
  api,
  type GlossaryEntry,
  type Project,
  type TaskStatus,
} from "../api";

interface Props {
  /** 媒体路径或 .sub.json 路径（后端两种都接受） */
  path: string;
  taskId: string;
  taskStatus: TaskStatus;
  /** 工程 JSON；null 或 stage 未到 contexted → 空态 */
  project: Project | null;
  onProjectChange: (project: Project) => void;
}

export default function GlossaryPanel({
  path,
  taskId,
  taskStatus,
  project,
  onProjectChange,
}: Props) {
  const [resuming, setResuming] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const saveEntry = async (entry: GlossaryEntry, dst: string) => {
    if (!project || dst === entry.dst) return;
    try {
      const resp = await api.patchGlossary(path, {
        updates: [{ src: entry.src, dst }],
      });
      onProjectChange({ ...project, glossary: resp.glossary });
      message.success(`已更新术语：${entry.src}`);
    } catch (err) {
      message.error(`更新失败：${(err as Error).message}`);
    }
  };

  const confirmEntry = async (entry: GlossaryEntry) => {
    if (!project) return;
    try {
      const resp = await api.patchGlossary(path, { confirm: [entry.src] });
      onProjectChange({ ...project, glossary: resp.glossary });
    } catch (err) {
      message.error(`确认失败：${(err as Error).message}`);
    }
  };

  const confirmAll = async () => {
    if (!project) return false;
    try {
      const resp = await api.patchGlossary(path, { confirm_all: true });
      onProjectChange({ ...project, glossary: resp.glossary });
      return true;
    } catch (err) {
      message.error(`确认失败：${(err as Error).message}`);
      return false;
    }
  };

  const confirmAllAndResume = async () => {
    setResuming(true);
    try {
      if (await confirmAll()) {
        await api.resumeTask(taskId);
        message.success("术语表已全部确认，任务已重新排队");
      }
    } catch (err) {
      message.error(`操作失败：${(err as Error).message}`);
    } finally {
      setResuming(false);
    }
  };

  // 建档（摘要+术语表）完成后才可用；之前一律空态，不弹错误
  if (!project || project.stage === "empty" || project.stage === "transcribed") {
    return <Empty description="术语表尚未生成（建档完成后可查看、编辑）" />;
  }

  const glossary = project.glossary;
  const confirmedCount = glossary.filter((g) => g.confirmed).length;
  const waiting = taskStatus === "waiting_confirm";

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      {waiting && (
        <Alert
          type="warning"
          showIcon
          message="任务在术语表人工确认检查点暂停"
          description="核对并修改下面的术语表（译文可直接点击编辑），全部确认后任务会继续逐句翻译。"
        />
      )}
      <Table<GlossaryEntry>
        rowKey="src"
        size="small"
        dataSource={glossary}
        pagination={false}
        locale={{ emptyText: "术语表为空" }}
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
        {waiting ? (
          <Popconfirm
            title="全部确认并继续？"
            description="将把全部条目标记为已确认并重新排队任务"
            onConfirm={() => void confirmAllAndResume()}
          >
            <Button type="primary" loading={resuming}>
              全部确认并继续
            </Button>
          </Popconfirm>
        ) : (
          <Button
            loading={confirming}
            disabled={confirmedCount === glossary.length}
            onClick={() => {
              setConfirming(true);
              void confirmAll().finally(() => setConfirming(false));
            }}
          >
            全部确认
          </Button>
        )}
      </Space>
    </Space>
  );
}
