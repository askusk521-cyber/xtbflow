# 初始化交付状态

日期：2026-09-25。

## 已直接核实

GitHub 接口可读取 `askusk521-cyber/xtbflow`，仓库可见性为 public、默认分支为 main，读取时为空。仓库元数据返回账户层面的 push/admin 等权限。

首次实际写入调用（创建 README.md）曾返回：

```text
403 Resource not accessible by integration
```

这说明当时的 GitHub 集成写入路径不可用；没有擅自更改应用连接、账户授权或其他仓库设置。随后在具备目标仓库实际 Git 写入凭据的执行环境中，bundle 已通过 SSH 推送成功。

## 当前远端状态

远端 `main` 已创建并核验：

```text
remote: git@github.com:askusk521-cyber/xtbflow.git
branch: main
sha: 329da0199d5ececd06978103ed4421bbc4536bed
```

验证命令 `git ls-remote origin refs/heads/main` 返回同一 SHA。此次及前一次推送均没有使用 force push。

## 本地交付

初始化文档、当前配置、原始材料归档、资源脱敏副本、完整性脚本和文件清单已在会话运行环境生成。交付时提供本地 Git commit 与 Git bundle；这些不构成已推送远端的证据。

旧工程、主机、GPU、Slurm、真实数据集和量化程序没有在本次连接或执行。没有产生模型性能或机理结论。

## 从 Git bundle 推送（已完成；供复核）

以下命令由已经对目标仓库拥有真实写入凭据的执行环境运行，不需要把凭据交给聊天：

```bash
git clone /path/to/xtbflow-bootstrap.bundle xtbflow
cd xtbflow
python scripts/validate_bootstrap.py
git remote set-url origin git@github.com:askusk521-cyber/xtbflow.git
git push -u origin main
```

不会使用 force push。若远端在这期间产生新提交，应先获取并审查，不覆盖。此次已按该流程完成，并将 `configs/resources.yaml` 的远端写入状态更新为 true。

单纯在聊天中再次批准不能保证消除 GitHub 返回的 403；需要执行环境或连接在 GitHub 一侧具有可用的仓库内容写入授权。
