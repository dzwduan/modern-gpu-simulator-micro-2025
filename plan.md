# True-Path 一致性重构实施计划

**状态**: 已完成

**Goal:** 基于 true-path-consistency-analysis.md 的发现，移除传统 scoreboard legacy 路径，简化 guard conditions，修复配置缺失，清理 pipeline 代码，全程 TDD。

## 完成的任务

| Task | 描述 | 状态 |
|------|------|------|
| 1 | 搭建 GoogleTest 测试框架 | ✅ |
| 2 | 提取 guard condition helper + 写测试 (10 tests) | ✅ |
| 3 | 移除 subcore.cc scoreboard legacy 路径 | ✅ |
| 4 | 移除 sm.cc scoreboard legacy 路径 | ✅ |
| 5 | 移除 functional_unit.cc scoreboard legacy 路径 | ✅ |
| 6 | 修复 is_loog_enabled 配置注册 | ✅ |
| 7 | INT/FP32 pipeline 路由测试 (7 tests) | ✅ |
| 8 | DP pipeline 共享路径验证 | ✅ |
| 9 | 清理未使用的 Scoreboard 相关代码 | ✅ |
| 10 | 更新设计文档 | ✅ |

## 修改的文件

| 文件 | 改动类型 |
|------|---------|
| `simulator-remodeled/.../dependency_path.h` | 新增：guard condition 逻辑提取 |
| `simulator-remodeled/.../pipeline_routing.h` | 新增：pipeline 路由逻辑提取 |
| `simulator-remodeled/.../shader.h` | 修改：添加 include 和 helper 方法 |
| `simulator-remodeled/.../remodeling/subcore.cc` | 修改：移除 scoreboard 分支 |
| `simulator-remodeled/.../remodeling/subcore.h` | 修改：移除 use_traditional_scoreboarding 参数 |
| `simulator-remodeled/.../remodeling/sm.cc` | 修改：移除 scoreboard 分支 |
| `simulator-remodeled/.../remodeling/sm.h` | 修改：移除 use_traditional_scoreboarding 参数 |
| `simulator-remodeled/.../remodeling/functional_unit.cc` | 修改：移除 scoreboard WAR release |
| `simulator-remodeled/.../gpu-sim-config.cc` | 修改：注册 is_loog_enabled |
| `docs/detailed-design/01-总览与假设集合.md` | 修改：更新文档 |
| `tests/CMakeLists.txt` | 新增：测试构建系统 |
| `tests/test_dependency_path.cc` | 新增：依赖路径测试 |
| `tests/test_pipeline_routing.cc` | 新增：pipeline 路由测试 |

## 测试结果

17 个测试全部通过，编译零错误。
