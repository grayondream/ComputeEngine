# P5 · 规则引擎(RuleEngine + 推理链可视化)

> 设计来源:`docs/readme.md` §1(规则算子)、§8.1(规则推理)、§13.1(规则链可视化)、§15(`rules/`)。

## 1. 目标与验收(DoD)

- **目标**:`RuleEngine` 可接收事实与规则,输出议程与触发链;推理链可被 `rules/rule_trace` 采集并可视化。
- **验收标准**:
  1. `RuleEngine::add_fact / add_rule / fire()` 返回 `Agenda`(§8.1);`trace()` 返回触发轨迹。
  2. 一组典型规则(条件/优先级/动作)按预期议程顺序触发;优先级生效。
  3. 推理链可导出为 JSON→Mermaid sequence(§13.1),可视化输入 `ie-graph --format` 等价路径可用。
  4. 规则算子 `RuleMatch/RuleFire/FactAssert`(OpType,§4.2)可在 IR 中表示。

## 2. 范围与边界

- **范围内**:RuleEngine 核心、Fact/Condition/Rule/WorkingMemory/Agenda、`rules/{fact,rule,agenda}`、rete(起步可选)、推理链追踪。
- **范围外**:复杂 Rete 网络优化(可后置);工具调用 ToolCall/LLMGenerate 编排(P14);多引擎统一调度。

## 3. 需实现/落地的模块

- `rules/fact/`(Fact,WorkingMemory)
- `rules/rule/`(Rule,Condition,RuleEngine)
- `rules/agenda/`(Agenda,优先级)
- `rules/rete/`(可选起步)
- `tools/rule_trace/`(推理链采集 → JSON / Mermaid sequence)

## 4. 接口要点

- `struct Fact/Condition/Rule`、`class RuleEngine`(fire/trace)(§8.1)
- 规则 IR 算子:OpType `RuleMatch/RuleFire/FactAssert` 可用并注册(§4.2)

## 5. 实现要点

- 议程演化要可复现、可排序(优先级→插入序),保证确定性(§10.4 思维).
- 触发链记录每次 fire 的规则名、命中事实、动作前的 WorkingMemory 快照变化(增量).
- 可视化输出先 JSON 结构化,再渲染 Mermaid sequence(§13.1).

## 6. 测试与验证

- 单测:议程顺序、优先级、冲突消解、trace 完整性。
- 端到端:一个 mid-size 规则集按预期出链并正确输出 JSON/Mermaid。

## 7. 依赖

- P0.5(接口与 IR 算子类型)。
- 可视化复用 `tools/` 采集管线(§13.1 TraceExporter 的规则分节)。

## 8. 决策点(默认建议 · 待实现时裁定)

> 记录于 2026-10-18,为**默认建议**而非硬约束;实现 P5 时在实现内裁定并回写。

| 决策点 | 默认建议 | 副作用/影响 |
|---|---|---|
| Rete 实现深度 | **先线性,rete 后置** | P5 用线性遍历 + 半 left-join 求值保证正确;完整 Rete(alpha/beta 内存、索引)后置;`rules/rete/` 目录保持占位 |