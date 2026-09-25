# 初始化交付状态

日期：2026-09-25。

## 已直接核实

GitHub 接口可读取 `askusk521-cyber/xtbflow`，仓库可见性为 public、默认分支为 main，读取时为空。仓库元数据返回账户层面的 push/admin 等权限。

首次实际写入调用（创建 README.md）返回：

```text
403 Resource not accessible by integration
```

因此账户显示的仓库权限没有转化为当前连接可用的 Contents 写入能力。当前 GitHub 插件已经安装，但本次没有成功远程提交；没有擅自更改应用连接、账户授权或其他仓库设置。

## 本地交付

初始化文档、当前配置、原始材料归档、资源脱敏副本、完整性脚本和文件清单已在会话运行环境生成。交付时提供本地 Git commit 与 Git bundle；这些不构成已推送远端的证据。

旧工程、主机、GPU、Slurm、真实数据集和量化程序没有在本次连接或执行。没有产生模型性能或机理结论。

## 从 Git bundle 推送

以下命令由已经对目标仓库拥有真实写入凭据的执行环境运行，不需要把凭据交给聊天：

```bash
git clone /path/to/xtbflow-bootstrap.bundle xtbflow
cd xtbflow
python scripts/validate_bootstrap.py
git remote set-url origin git@github.com:askusk521-cyber/xtbflow.git
git push -u origin main
```

不会使用 force push。若远端在这期间产生新提交，应先获取并审查，不覆盖。推送后用 `git ls-remote origin refs/heads/main` 验证远端 SHA；只有实际成功后才将 `configs/resources.yaml` 的远端写入状态改为 true，并记录证据。

单纯在聊天中再次批准不能保证消除 GitHub 返回的 403；需要执行环境或连接在 GitHub 一侧具有可用的仓库内容写入授权。
