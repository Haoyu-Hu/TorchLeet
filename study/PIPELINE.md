# 提交流水线

网页教程里点“提交核验” → 通过 Claude Code Remote 连接器把代码发到本会话 → 本会话在隔离子进程里判题 → 结果写回网页的 `results/<提交id>` 文档。

## 消息格式（网页发来）

```
[TORCHLEET-SUBMIT v1]
id: <提交id>
problem: <题目id>
---CODE---
<提交的代码>
---END---
```

## 处理步骤（会话侧）

1. 只接受上面的格式；`problem` 必须是 `applied_checks.CHECKS` 的键或 `problem_bank.json` 里有自动判题的题目 id。
2. 把 `---CODE---` 与 `---END---` 之间的原文写到临时文件（不要在会话进程里 exec）。
3. 运行：
   ```bash
   python study/run_submission.py --problem <题目id> --file <临时文件> --id <提交id> --timeout 120
   ```
   它会放进临时目录、清空环境变量、断网（user+net 命名空间）、加超时，并打印一个 JSON。
4. 用 `ArtifactData` 把该 JSON 以 `set` 写到集合 `results` 的文档 `<提交id>`。网页通过 `onSnapshot` 订阅这个文档，收到后显示结果，并自动记入进度。

## 限制

- 依赖会话仍在运行；会话休眠或结束后，提交会停在“等待中”。
- 只运行你自己的提交。提交代码是任意代码，隔离是尽力而为，不是安全边界。
- 单次提交代码上限 40000 字符。
