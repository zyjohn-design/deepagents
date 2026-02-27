# 跨年夜活动文档提取示例

## 示例1：核心区禁停

**原文文本**
> 12月31日下午,核心管控区,15时起禁止机动车停放

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "核心管控区",
      "type": "核心区",
      "boundaries": "沿江大道(民生路至一元路)、一元路(不含)、胜利街(一元路至黄石路,不含)、黄石路(不含)、中山大道(黄石路至民生路)、民生路(不含)合围区内道路",
      "control_measures": "禁止机动车停放"
    }
  ],
  "tasks": [
    {
      "phase": "启动准备阶段",
      "phase_code": "preparation",
      "start_time": "2025-12-31 15:00",
      "end_time": "2025-12-31 18:30",
      "control_type": "车辆禁停",
      "description": "核心管控区禁止机动车停放",
      "action": "禁止",
      "original_text": "12月31日下午,核心管控区,15时起禁止机动车停放",
      "affected_area": "核心管控区"
    }
  ]
}
```

## 示例2：疏导区分流

**原文文本**
> 预计,17时30分,启动疏导区分流

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "中端疏导区",
      "type": "疏导区",
      "boundaries": "黄石路、胜利街(一元路至黄石路)、一元路、沿江大道、三阳路、京汉大道...合围区",
      "control_measures": "减量分流、只出不进"
    },
    {
      "area_name": "远端分流区",
      "type": "分流区",
      "boundaries": "沿江大道、黄浦大街、解放大道、利济北路、武胜路、沿河大道...合围区",
      "control_measures": "承压保畅、梯次分流"
    }
  ],
  "tasks": [
    {
      "phase": "管控实施阶段",
      "phase_code": "implementation",
      "start_time": "2025-12-31 17:30",
      "end_time": "2025-12-31 18:30",
      "control_type": "交通管制",
      "description": "启动中端疏导区分流措施",
      "action": "启动",
      "original_text": "预计,17时30分,启动疏导区分流",
      "affected_area": "中端疏导区"
    },
    {
      "phase": "管控实施阶段",
      "phase_code": "implementation",
      "start_time": "2025-12-31 17:30",
      "end_time": "2025-12-31 18:30",
      "control_type": "交通管制",
      "description": "启动远端分流区分流措施",
      "action": "启动",
      "original_text": "预计,17时30分,启动疏导区分流",
      "affected_area": "远端分流区"
    }
  ]
}
```

## 示例3：特殊车辆管控

**原文文本**
> 智能网联车类(萝卜快跑及无人快递车):按黄浦大街-二环线-发展大道-青年路-武胜路-沿河大道-沿江大道合围区设置临时禁行区。控制时间:12月31日17时整-1月1日3时整。

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "智能网联车临时禁行区",
      "type": "管控区",
      "boundaries": "黄浦大街-二环线-发展大道-青年路-武胜路-沿河大道-沿江大道合围区",
      "control_measures": "智能网联车(萝卜快跑及无人快递车)临时禁行"
    }
  ],
  "tasks": [
    {
      "phase": "管控实施阶段",
      "phase_code": "implementation",
      "start_time": "2025-12-31 17:00",
      "end_time": "2026-01-01 03:00",
      "control_type": "全封闭",
      "description": "智能网联车(萝卜快跑及无人快递车)临时禁行",
      "action": "禁止",
      "original_text": "智能网联车类(萝卜快跑及无人快递车):按黄浦大街-二环线-发展大道-青年路-武胜路-沿河大道-沿江大道合围区设置临时禁行区。控制时间:12月31日17时整-1月1日3时整。",
      "affected_area": "智能网联车临时禁行区"
    }
  ]
}
```

## 示例4：收尾恢复

**原文文本**
> 元旦2时起,根据路面人群疏散情况,视情逐段恢复沿江大道车辆放行

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "沿江大道",
      "type": "路段",
      "control_measures": "视情逐段恢复车辆放行"
    }
  ],
  "tasks": [
    {
      "phase": "疏散收尾阶段",
      "phase_code": "dispersal",
      "start_time": "2026-01-01 02:00",
      "end_time": "2026-01-01 03:00",
      "control_type": "交通管制",
      "description": "视情逐段恢复沿江大道车辆放行",
      "action": "恢复",
      "original_text": "元旦2时起,根据路面人群疏散情况,视情逐段恢复沿江大道车辆放行",
      "conditions": "根据路面人群疏散情况",
      "affected_area": "沿江大道"
    }
  ]
}
```
