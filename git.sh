#!/bin/bash

# コミットメッセージの引数がなければエラーメッセージを表示して終了
if [ $# -eq 0 ]; then
  echo "Usage: $0 commit message"
  exit 1
fi

# 引数をスペース区切りで結合して、1つのメッセージにする
commit_message="$*"

# 一連のGitコマンドを実行
git add .
git commit -m "$commit_message"
git push origin master
