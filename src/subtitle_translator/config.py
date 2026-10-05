"""配置加载与校验（占位，后续阶段实现）。

单份 config.yaml 为唯一配置存储，网页 / CLI / 脚本共用。
分 asr / translate / ui 三节；translate.api_key 支持
``api_key_env`` 环境变量引用，避免明文密钥入库。
"""
