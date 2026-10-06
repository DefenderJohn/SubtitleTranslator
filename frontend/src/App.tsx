import { Layout, Menu } from "antd";
import { UnorderedListOutlined, SettingOutlined } from "@ant-design/icons";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import TasksPage from "./pages/TasksPage";
import TaskDetailPage from "./pages/TaskDetailPage";
import SettingsPage from "./pages/SettingsPage";
import ErrorBoundary from "./components/ErrorBoundary";
import { UploadDraftProvider } from "./uploadDraft";

const { Sider, Header, Content } = Layout;

export default function App() {
  const location = useLocation();
  const selectedKey = location.pathname.startsWith("/settings")
    ? "/settings"
    : "/";

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider theme="dark">
        <div
          style={{
            color: "#fff",
            padding: "16px",
            fontWeight: 600,
            fontSize: 15,
            whiteSpace: "nowrap",
            overflow: "hidden",
          }}
        >
          SubtitleTranslator
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selectedKey]}
          items={[
            {
              key: "/",
              icon: <UnorderedListOutlined />,
              label: <Link to="/">任务</Link>,
            },
            {
              key: "/settings",
              icon: <SettingOutlined />,
              label: <Link to="/settings">设置</Link>,
            },
          ]}
        />
      </Sider>
      <Layout>
        <Header style={{ background: "#fff", padding: "0 24px" }} />
        <Content style={{ margin: 24 }}>
          <UploadDraftProvider>
            {/* key 按路径重挂载：崩溃后点「返回任务列表」即恢复，不需整页刷新 */}
            <ErrorBoundary key={location.pathname}>
              <Routes>
                <Route path="/" element={<TasksPage />} />
                <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </ErrorBoundary>
          </UploadDraftProvider>
        </Content>
      </Layout>
    </Layout>
  );
}
