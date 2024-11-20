#!/bin/bash

# コミットメッセージの引数がなければエラーメッセージを表示して終了
if [ -z "$1" ]; then
  echo "Usage: $0 \"commit message\""
  exit 1
fi

# 一連のGitコマンドを実行
git add .
git commit -m "$1"
git push origin master