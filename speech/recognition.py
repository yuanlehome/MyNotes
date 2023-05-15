import paddle
from paddlespeech.cli.asr import ASRExecutor

asr_executor = ASRExecutor()
text = asr_executor(
    model='conformer_wenetspeech',
    lang='zh',
    sample_rate=16000,
    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
    ckpt_path=None,
    audio_file='./zh.wav',
    force_yes=True,
    device=paddle.get_device(),
)

print("ASR Result: \n{}".format(text))
