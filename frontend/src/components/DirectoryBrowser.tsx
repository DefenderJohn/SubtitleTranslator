/** 目录浏览弹窗：调 GET /api/media?path= 逐级浏览，选中文件或目录。 */
import { useCallback, useEffect, useState } from "react";
import { Button, List, Modal, Space, Typography, message } from "antd";
import {
  ArrowUpOutlined,
  FolderOutlined,
  FileOutlined,
} from "@ant-design/icons";
import { api, basename, type MediaBrowse } from "../api";

interface Props {
  open: boolean;
  onSelect: (path: string) => void;
  onCancel: () => void;
}

export default function DirectoryBrowser({ open, onSelect, onCancel }: Props) {
  // 空字符串 = 让后端给默认起点（用户主目录）
  const [current, setCurrent] = useState<string>("");
  const [data, setData] = useState<MediaBrowse | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const result = await api.browseMedia(path);
      setData(result);
      setCurrent(result.path);
    } catch (err) {
      message.error(`无法浏览目录：${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void load(current);
    // 仅在弹窗打开时加载当前目录
  }, [open]);

  const entries = [
    ...(data?.parent != null
      ? [{ key: "..", name: "..（上级目录）", dir: true, path: data.parent }]
      : []),
    ...(data?.directories ?? []).map((name) => ({
      key: `d:${name}`,
      name,
      dir: true,
      path: `${current.replace(/\/$/, "")}/${name}`,
    })),
    ...(data?.media ?? []).map((path) => ({
      key: `f:${path}`,
      name: basename(path),
      dir: false,
      path,
    })),
  ];

  return (
    <Modal
      title="浏览选择媒体文件或目录"
      open={open}
      onCancel={onCancel}
      footer={[
        <Button key="pick-dir" onClick={() => onSelect(current)}>
          选择当前目录
        </Button>,
        <Button key="cancel" onClick={onCancel}>
          取消
        </Button>,
      ]}
      width={640}
    >
      <Typography.Text type="secondary" copyable>
        {current}
      </Typography.Text>
      <List
        loading={loading}
        size="small"
        style={{ maxHeight: 400, overflow: "auto", marginTop: 8 }}
        dataSource={entries}
        renderItem={(item) => (
          <List.Item
            style={{ cursor: "pointer" }}
            onClick={() => {
              if (item.dir) void load(item.path);
              else onSelect(item.path);
            }}
          >
            <Space>
              {item.dir ? (
                item.name.startsWith("..") ? (
                  <ArrowUpOutlined />
                ) : (
                  <FolderOutlined />
                )
              ) : (
                <FileOutlined />
              )}
              {item.name}
            </Space>
          </List.Item>
        )}
      />
    </Modal>
  );
}
