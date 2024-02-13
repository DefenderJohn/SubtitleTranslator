import gradio as gr
import whisper
import pysrt
import os

def generate_subtitles(video_file, language, model_size):
    # 将上传的视频文件保存到本地
    video_path = video_file.name
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    # 加载 Whisper 模型
    model = whisper.load_model(model_size)

    # 生成字幕
    result = model.transcribe(video_path, language=language)
    subtitles = result["segments"]

    # 创建 SRT 格式的字幕文件
    srt_content = ""
    for i, segment in enumerate(subtitles):
        srt_content += "{}\n".format(i+1)
        srt_content += "{} --> {}\n".format(segment['start'], segment['end'])
        srt_content += "{}\n\n".format(segment['text'])

    srt_filename = "output.srt"
    with open(srt_filename, 'w') as f:
        f.write(srt_content)

    # 清理临时视频文件
    os.remove(video_path)

    return srt_filename

# Gradio 界面设置
iface = gr.Interface(
    fn=generate_subtitles,
    inputs=[
        gr.inputs.Video(label="Upload your video"),
        gr.inputs.Dropdown(choices=["English", "Spanish", "French", "German"], label="Select Video Language"),
        gr.inputs.Radio(choices=["base", "small", "medium", "large"], label="Select Model Size"),
    ],
    outputs="file"
)

# 启动 Gradio 界面
if __name__ == "__main__":
    iface.launch()
