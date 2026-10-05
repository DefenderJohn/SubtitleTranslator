"""数据结构与 JSON schema（占位，后续阶段实现）。

每个视频一个 JSON 文件，是唯一事实来源；SRT 只是导出物。

Schema 字段：
- version
- source{file, duration, language}
- models{asr, aligner, translator}
- summary
- glossary[{src, dst, count, confirmed}]
- cues[{id, start, end, text, translation, words[{text, start, end}]}]
- stage（transcribed -> contexted -> translated，驱动断点续传）
"""
