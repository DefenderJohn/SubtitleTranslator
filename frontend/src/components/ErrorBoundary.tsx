/** 路由级渲染兜底：任何子树渲染崩溃（如数据形态不符合预期）时，
 * 显示错误信息 + 组件堆栈摘要 + 「返回任务列表」出口，而不是无声白屏。
 * 堆栈与错误同时进 console，方便排查。
 */
import { Component, type ErrorInfo, type ReactNode } from "react";
import { Alert, Button, Typography } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { Link } from "react-router-dom";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  componentStack: string | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): State {
    return { error, componentStack: null };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary] 渲染崩溃:", error, info.componentStack);
    this.setState({ componentStack: info.componentStack ?? null });
  }

  render() {
    const { error, componentStack } = this.state;
    if (!error) return this.props.children;
    return (
      <div style={{ maxWidth: 720, margin: "48px auto" }}>
        <Alert
          type="error"
          showIcon
          message="页面渲染出错"
          description={
            <>
              <Typography.Paragraph style={{ marginBottom: 8 }}>
                {error.name}: {error.message}
              </Typography.Paragraph>
              {componentStack && (
                <pre
                  style={{
                    maxHeight: 240,
                    overflow: "auto",
                    background: "#fff1f0",
                    borderRadius: 6,
                    padding: 12,
                    fontSize: 12,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                  }}
                >
                  {componentStack.trim()}
                </pre>
              )}
              <Link to="/">
                <Button icon={<ArrowLeftOutlined />} style={{ marginTop: 8 }}>
                  返回任务列表
                </Button>
              </Link>
            </>
          }
        />
      </div>
    );
  }
}
