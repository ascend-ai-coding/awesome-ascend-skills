# A2 SyncAll 排障材料

工作区使用本轮固定的可写知识仓样本。从工作区调用本文件旁的 prepare_ingest_fixture.py，样本位于 knowledge-fixture。交付时用同一入口的 --verify 取得校验回执。

开发记录（知识仓 86ef0e4，9 张卡片的本地样本）：
- 按 A2 查询 SyncAll，返回 4 条结果。只修改 sync_all.md 的正文后，同一查询退出 2，报错含有“knowledge card changed after index build”。
- 另起进程重复同一查询，仍退出 2。对现有索引做完整校验，也退出 2。
- 重建索引后，完整校验通过，仍为 9 张卡片。原查询退出 0，返回 4 条结果。
本次只记录了这个样本和 A2 查询条件。
