
[オリジナルのREADMEはこちらです](README_ORIG.md)

# About this fork
後述のZ版4ファイルが追加されています。requirements.txt のバージョンが本家とちょっと違います。Windows の Python 3.10 の venv で動作確認しています。

## Z版

設定ファイルなどをモデルごとに分離して管理しやすくし、複数のモデルを並行して学習できるようにしました。生成コマンドも使いやすくしています。
- z_inference_main.py
- z_init_project.py
- z_preprocess.py
- z_train.py
の4ファイルが追加されています。InquirerPy が必要です。

```
pip install InquirerPy
```

https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
hubert フォルダに入れます。これは従来と同じです。

https://huggingface.co/innnky/sovits_pretrained/resolve/main/G_0.pth
https://huggingface.co/innnky/sovits_pretrained/resolve/main/D_0.pth
pretrained_models という名前のフォルダを作ってそこに入れます。従来と違います。

### 学習

1.初期化

```
python z_init_project.py
```

実行すると名前を訊いてくるので入力してください。projects/model_name フォルダが作成されます。データセットや設定などはすべてこのフォルダ内に作成されることになります。logs/model_name フォルダも作成され D_0.pth と G_0.pth がコピーされます。

2.準備

projects/model_name/raw/speaker_name フォルダが作成されているので、その中にwavファイルを入れます。サブフォルダを作っても認識されます。

```
python z_preprocess.py
```

実行するとプロジェクトの一覧を表示するのでプリプロセスしたいものを選択してください。選択すると projects/model_name 内の dataset/ に 32kHz 変換された wav が生成され、解析もされ、config.json と filelists/ も生成されます。従来の resample.py、preprocess_flist_config.py、preprocess_hubert_f0.py をまとめて実行するようなものです。

config.json の `batch_size` は 10 にしてあります（VRAM12GB向け）。必要があれば変更してください。

3.実行

```
python z_train.py
```

実行するとプロジェクトの一覧を表示するので学習したいものを選択してください。選択すると学習開始します。途中経過は従来と同じ logs/model_name フォルダに生成されます。引数を指定して train.py を実行するのと同等の処理をしているだけなので、train.py で代用することも可能です。途中でやめてから再開すると最後に生成された中間結果から再開されます。

### 生成

```
python z_inference_main.py
```

実行するとモデルやスピーカー、スライスのしきい値を訊いてくるので選択あるいは入力します。モデルは logs フォルダにあるものを一覧表示します。最も数字の大きい G_数字.pth が使用されます。変換したい wav ファイルを入力（エクスプローラからドラッグ＆ドロップ）すると変換を行います。結果は results フォルダに生成されます。

変換中にVRAM不足になったときは slice threshold db を -30 くらいにちょっと増やして試してみてください。

