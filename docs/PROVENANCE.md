# 初始化来源与变更

本初始化以用户对话和已附启动文件为依据，不重新开展文献检索，不把历史软件文档当作本次已实测能力。

原始启动包的所有成员文件按原字节保存于 `archive/`；方法、数据合同、来源与 seed schema 同时复制到活动文档目录。新的根指令和配置落实远程仓库、完整项目授权、公开数据先行与实验暂缓。

用户实际资源 YAML 只发布脱敏副本。旧工程源代码、数据、权重、环境锁的真实文件尚未取得，不能声称已迁移。

来源清单：

```json
{
  "prepared_on": "2026-09-25",
  "imports": [
    {
      "source": "reaction_flow_agent_starter.zip",
      "sha256": "b58dd2e0e283d566e23562536f148b6486976e71360780ed5daea705577401f5",
      "handling": "all member files preserved byte-for-byte under archive/"
    },
    {
      "source": "public_first_agent_addendum.zip",
      "sha256": "848225d1671ae03f827debb5200d44f2a49140f3c5c24b371d0b33b835e99fdb",
      "handling": "all member files preserved byte-for-byte under archive/"
    },
    {
      "source": "user-provided resources YAML",
      "sha256": "7f026039eea451925762b8fc090253a622bd55f2cc45834f4fd8133a6809489d",
      "handling": "redacted historical copy only; original remains outside public repository"
    }
  ],
  "new_files": "root instructions, current configs, provenance/status, ignore rules and validator",
  "remote_intent": "git@github.com:askusk521-cyber/xtbflow.git",
  "legacy_full_repository_imported": false,
  "scientific_calculations_performed": false
}
```

`bootstrap_inventory.json` 记录本初始化快照的文件 SHA-256；它不是防篡改签名，也不证明科学结果。实际代码身份以 Git commit 为准。仓库自身不能在同一个提交里保存自己的最终 commit 哈希；运行时读取并写入运行元数据。
