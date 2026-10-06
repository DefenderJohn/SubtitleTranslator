"""命令行薄壳：只做参数解析与转发，业务逻辑在 pipeline.py。

入口：``subtitle-translator``（见 pyproject.toml [project.scripts]）。

子命令：
- ``run``     跑流水线（转录 → 翻译 → 导出 SRT），断点续传
- ``export``  从已有 .sub.json 导出 SRT
- ``glossary`` 查看 / 确认术语表（人工确认检查点）
- ``serve``   启动网页服务（FastAPI + SSE，需 web extra）
- ``config init`` 生成默认 config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import pipeline, preflight
from .config import default_config, load_config, resolve_api_key, save_config
from .logsetup import register_secret, setup_logging
from .models import SubtitleProject
from .srt import export_srt
from .translate import GlossaryNotConfirmedError

DEFAULT_CONFIG_PATH = "config.yaml"

_STAGE_LABELS = {"transcribe": "转录", "translate": "翻译", "export": "导出"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-translator",
        description="音视频字幕转录与翻译工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑流水线（转录 → 翻译 → 导出 SRT）")
    run.add_argument("path", help="媒体文件 / 目录（--from-json 时为 .sub.json）")
    run.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"config.yaml 路径（默认 {DEFAULT_CONFIG_PATH}，不存在时用内置默认值）",
    )
    run.add_argument(
        "--transcribe-only",
        action="store_true",
        help="只转录并导出原文 SRT，不翻译",
    )
    run.add_argument(
        "--auto-confirm",
        action="store_true",
        help="自动确认术语表（跳过人工确认检查点）",
    )
    run.add_argument("--no-bilingual", action="store_true", help="导出单语 SRT（默认双语）")
    run.add_argument(
        "--from-json",
        action="store_true",
        help="直接对已有 .sub.json 跑后续阶段，不接触媒体文件（重翻/调术语后重跑）",
    )
    run.add_argument("--language", help="源语言（覆盖 asr.language，如 en；默认自动检测）")
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过启动预检（不推荐；预检失败本该在启动时暴露而不是运行时才炸）",
    )

    export = sub.add_parser("export", help="从已有 .sub.json 导出 SRT")
    export.add_argument("json_path", help="工程文件（xxx.sub.json）")
    export.add_argument("--bilingual", action="store_true", help="双语导出（译文在上、原文在下）")
    export.add_argument("-o", "--output", help="输出 SRT 路径（默认与工程文件同名）")

    glossary = sub.add_parser("glossary", help="查看 / 确认术语表")
    glossary.add_argument("json_path", help="工程文件（xxx.sub.json）")
    glossary.add_argument(
        "--confirm-all", action="store_true", help="确认全部条目并保存（confirmed=true）"
    )
    glossary.add_argument("--show", action="store_true", help="打印术语表")

    serve = sub.add_parser("serve", help="启动网页服务（FastAPI + SSE，需 web extra）")
    serve.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"config.yaml 路径（默认 {DEFAULT_CONFIG_PATH}）",
    )
    serve.add_argument("--host", help="监听地址（默认取 ui.host 配置）")
    serve.add_argument("--port", type=int, help="监听端口（默认取 ui.port 配置）")

    config = sub.add_parser("config", help="配置管理")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_init = config_sub.add_parser("init", help="生成默认 config.yaml")
    config_init.add_argument(
        "path", nargs="?", default=DEFAULT_CONFIG_PATH, help=f"输出路径（默认 {DEFAULT_CONFIG_PATH}）"
    )
    return parser


# ---------------------------------------------------------------------------
# 进度展示：stderr 简单打印（网页才是主交互，此处不引 tqdm）
# ---------------------------------------------------------------------------


def _stderr_progress(event: dict) -> None:
    stage = _STAGE_LABELS.get(event.get("stage", ""), event.get("stage", ""))
    media = Path(event.get("media", "")).name
    done, total = event.get("done", 0), event.get("total", 0)
    message = event.get("message", "")
    if total:
        line = f"\r[{media}] {stage}: {done}/{total}"
        print(line, end="\n" if done >= total else "", file=sys.stderr, flush=True)
    else:
        print(f"[{media}] {stage}: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 各子命令
# ---------------------------------------------------------------------------


def _preflight_progress(event: dict) -> None:
    """预检耗时步骤（模型下载等）的进度提示。"""
    print(f"[预检] {event.get('check', '')}: {event.get('message', '')}", file=sys.stderr, flush=True)


def _run_preflight_or_exit(cfg, *, need_translate: bool, need_transcribe: bool) -> int:
    """执行预检并打印报告；未通过返回退出码 2，通过返回 0。"""
    report = preflight.run_preflight(
        cfg,
        need_translate=need_translate,
        need_transcribe=need_transcribe,
        progress_cb=_preflight_progress,
    )
    print("启动预检：", file=sys.stderr)
    print(report.format_text(), file=sys.stderr)
    if not report.ok:
        return 2
    return 0


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.language:
        cfg.asr.language = args.language
    if not args.skip_preflight:
        # --from-json 不接触媒体/模型，跳过转录侧检查；transcribe-only 不查翻译端点
        rc = _run_preflight_or_exit(
            cfg,
            need_translate=not args.transcribe_only,
            need_transcribe=not args.from_json,
        )
        if rc != 0:
            return rc
    if args.transcribe_only:
        stages = ("transcribe", "export")
    else:
        stages = pipeline.PIPELINE_STAGES

    common = dict(
        stages=stages,
        auto_confirm=args.auto_confirm,
        bilingual=not args.no_bilingual and not args.transcribe_only,
        progress_cb=_stderr_progress,
    )
    try:
        if args.from_json:
            pipeline.run_pipeline(args.path, cfg, from_json=True, **common)
        else:
            path = Path(args.path)
            if path.is_dir():
                result = pipeline.run_batch(path, cfg, **common)
                print(
                    f"批量完成：成功 {len(result.succeeded)} 个，失败 {len(result.failed)} 个",
                    file=sys.stderr,
                )
                for media, err in result.failed:
                    print(f"  失败: {media} — {err}", file=sys.stderr)
                if result.failed:
                    return 1
            else:
                pipeline.run_pipeline(path, cfg, **common)
    except GlossaryNotConfirmedError as exc:
        json_path = (
            Path(args.path) if args.from_json else pipeline.project_json_path(args.path)
        )
        print(
            f"{exc}\n提示：可用 `subtitle-translator glossary {json_path} --show` 查看术语表，\n"
            f"确认后运行 `subtitle-translator glossary {json_path} --confirm-all`，"
            f"再重新 run 即可从逐句翻译续跑；或加 --auto-confirm 跳过人工确认。",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI 边界统一兜底，友好报错
        print(f"错误：处理失败 — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_export(args) -> int:
    json_path = Path(args.json_path)
    try:
        project = SubtitleProject.load(json_path)
    except FileNotFoundError:
        print(f"错误：工程文件不存在: {json_path}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"错误：工程文件无法解析 — {exc}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else pipeline.default_srt_path(json_path, from_json=True)
    export_srt(project, output, bilingual=args.bilingual)
    print(f"已导出 {output}", file=sys.stderr)
    return 0


def _cmd_glossary(args) -> int:
    json_path = Path(args.json_path)
    try:
        project = SubtitleProject.load(json_path)
    except FileNotFoundError:
        print(f"错误：工程文件不存在: {json_path}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"错误：工程文件无法解析 — {exc}", file=sys.stderr)
        return 1

    if args.confirm_all:
        for entry in project.glossary:
            entry.confirmed = True
        project.save(json_path)
        print(f"已确认全部 {len(project.glossary)} 条术语并保存到 {json_path}", file=sys.stderr)

    if args.show or not args.confirm_all:
        if not project.glossary:
            print("（术语表为空）")
        for entry in project.glossary:
            mark = "✓" if entry.confirmed else " "
            print(f"[{mark}] {entry.src} | {entry.dst} | {entry.count}")
    return 0


def _cmd_config_init(args) -> int:
    path = Path(args.path)
    if path.exists():
        print(f"错误：{path} 已存在，如需重新生成请先删除或另选路径", file=sys.stderr)
        return 1
    save_config(default_config(), path)
    print(
        f"已生成默认配置 {path}\n"
        "请按需修改 translate.base_url / model，并用 api_key_env 引用密钥环境变量。",
        file=sys.stderr,
    )
    return 0


def _cmd_serve(args) -> int:
    cfg = load_config(args.config)
    # 启动预检：ffmpeg / 模型问题硬错（进入服务即应 ready-to-use）；
    # 只做存在性检查与必要的模型下载，模型加载仍留在首次任务时（启动要快）
    rc = _run_preflight_or_exit(cfg, need_translate=False, need_transcribe=True)
    if rc != 0:
        return rc
    # 翻译端点只警告不阻止启动：serve 是长期进程，用户可能只转录或稍后配 key
    for check in preflight.check_translate(cfg):
        if check.status == preflight.STATUS_OK:
            continue
        logger_hint = check.message.splitlines()[0] if check.message else check.status
        print(f"警告：{check.name}: {logger_hint}", file=sys.stderr)
    host = args.host or cfg.ui.host
    port = args.port or cfg.ui.port
    try:
        import uvicorn
    except ImportError:
        print(
            "错误：未安装 uvicorn。网页服务依赖 web extra，请先执行\n"
            "  pip install -e .[web]",
            file=sys.stderr,
        )
        return 1
    from .server import create_app

    app = create_app(config_path=args.config)
    print(f"网页服务启动于 http://{host}:{port}（API 文档 /docs）", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # 统一日志配置（console + 滚动文件，均过脱敏过滤器）；
    # 已配置的 api_key 注册进脱敏过滤器做精确替换
    cfg = load_config(getattr(args, "config", DEFAULT_CONFIG_PATH))
    setup_logging(cfg)
    register_secret(resolve_api_key(cfg))
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "glossary":
        return _cmd_glossary(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "config":
        return _cmd_config_init(args)
    return 2  # pragma: no cover - argparse required=True 保证不可达


if __name__ == "__main__":
    sys.exit(main())
