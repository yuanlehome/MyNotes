import paddle
from paddlespeech.cli.st import STExecutor

st_executor = STExecutor()
text = st_executor(
    model='fat_st_ted',
    src_lang='en',
    tgt_lang='zh',
    sample_rate=16000,
    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
    ckpt_path=None,
    audio_file='./en.wav',
    device=paddle.get_device())
print('ST Result: \n{}'.format(text))