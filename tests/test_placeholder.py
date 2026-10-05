"""骨架阶段的占位测试：验证包可导入、版本号存在。"""

import subtitle_translator


def test_importable():
    assert subtitle_translator.__version__


def test_placeholder_modules_importable():
    import subtitle_translator.cli  # noqa: F401
    import subtitle_translator.config  # noqa: F401
    import subtitle_translator.models  # noqa: F401
    import subtitle_translator.pipeline  # noqa: F401
    import subtitle_translator.segment  # noqa: F401
    import subtitle_translator.server  # noqa: F401
    import subtitle_translator.srt  # noqa: F401
    import subtitle_translator.transcribe  # noqa: F401
    import subtitle_translator.translate  # noqa: F401
