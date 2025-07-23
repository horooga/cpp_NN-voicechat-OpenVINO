from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenVINO/distil-whisper-large-v3-int4-ov",
    local_dir="distil-whisper-large-v3-int4-ov",
)
snapshot_download(repo_id="llmware/tiny-llama-chat-ov", local_dir="tiny-llama-chat-ov")
