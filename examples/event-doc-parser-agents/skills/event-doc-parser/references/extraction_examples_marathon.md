# 马拉松活动文档提取示例

## 示例1：赛前禁停

**原文文本**
> 3月22日14时起至比赛结束，以下道路禁止车辆停放：（一）所有比赛线路

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "全程马拉松路线",
      "type": "线路",
      "boundaries": "沿江大道三阳路口→...→欢乐大道辅道→全程终点(武汉欢乐谷)",
      "control_measures": "禁止车辆停放"
    },
    {
      "area_name": "半程马拉松路线",
      "type": "线路",
      "control_measures": "禁止车辆停放"
    },
    {
      "area_name": "13公里跑路线",
      "type": "线路",
      "control_measures": "禁止车辆停放"
    }
  ],
  "tasks": [
    {
      "phase": "赛前准备阶段",
      "phase_code": "pre_race",
      "start_time": "2025-03-22 14:00",
      "end_time": "比赛结束",
      "control_type": "车辆禁停",
      "description": "所有比赛线路禁止车辆停放",
      "action": "禁止",
      "original_text": "3月22日14时起至比赛结束，以下道路禁止车辆停放：（一）所有比赛线路",
      "affected_area": "全程马拉松路线"
    },
    {
      "phase": "赛前准备阶段",
      "phase_code": "pre_race",
      "start_time": "2025-03-22 14:00",
      "end_time": "比赛结束",
      "control_type": "车辆禁停",
      "description": "所有比赛线路禁止车辆停放",
      "action": "禁止",
      "original_text": "3月22日14时起至比赛结束，以下道路禁止车辆停放：（一）所有比赛线路",
      "affected_area": "半程马拉松路线"
    },
    {
      "phase": "赛前准备阶段",
      "phase_code": "pre_race",
      "start_time": "2025-03-22 14:00",
      "end_time": "比赛结束",
      "control_type": "车辆禁停",
      "description": "所有比赛线路禁止车辆停放",
      "action": "禁止",
      "original_text": "3月22日14时起至比赛结束，以下道路禁止车辆停放：（一）所有比赛线路",
      "affected_area": "13公里跑路线"
    }
  ]
}
```

## 示例2：隧道封闭

**原文文本**
> 3月23日1时起至本区域比赛结束，长江隧道所有进出口、长江公铁隧道五福路出口均临时封闭。

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "长江隧道",
      "type": "隧道",
      "boundaries": "所有进出口",
      "control_measures": "临时封闭"
    },
    {
      "area_name": "长江公铁隧道",
      "type": "隧道",
      "boundaries": "五福路出口",
      "control_measures": "临时封闭"
    }
  ],
  "tasks": [
    {
      "phase": "赛前准备阶段",
      "phase_code": "pre_race",
      "start_time": "2025-03-23 01:00",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "长江隧道所有进出口、长江公铁隧道五福路出口临时封闭",
      "action": "封闭",
      "original_text": "3月23日1时起至本区域比赛结束，长江隧道所有进出口、长江公铁隧道五福路出口均临时封闭。",
      "affected_area": "长江隧道"
    },
    {
      "phase": "赛前准备阶段",
      "phase_code": "pre_race",
      "start_time": "2025-03-23 01:00",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "长江隧道所有进出口、长江公铁隧道五福路出口临时封闭",
      "action": "封闭",
      "original_text": "3月23日1时起至本区域比赛结束，长江隧道所有进出口、长江公铁隧道五福路出口均临时封闭。",
      "affected_area": "长江公铁隧道"
    }
  ]
}
```

## 示例3：道路管控

**原文文本**
> 6时30分起至本区域比赛结束，以下道路禁止车辆通行：中山大道(黄石路至崇仁路)；武胜路(中山大道至沿河大道)；武胜路高架桥；江汉桥。

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "中山大道",
      "type": "路段",
      "boundaries": "黄石路至崇仁路",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "武胜路",
      "type": "路段",
      "boundaries": "中山大道至沿河大道",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "武胜路高架桥",
      "type": "桥梁",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "江汉桥",
      "type": "桥梁",
      "control_measures": "禁止车辆通行"
    }
  ],
  "tasks": [
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:30",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "中山大道、武胜路、江汉桥禁止车辆通行",
      "action": "禁止",
      "original_text": "6时30分起至本区域比赛结束，以下道路禁止车辆通行：中山大道(黄石路至崇仁路)；武胜路(中山大道至沿河大道)；武胜路高架桥；江汉桥。",
      "affected_area": "中山大道"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:30",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "中山大道、武胜路、江汉桥禁止车辆通行",
      "action": "禁止",
      "original_text": "6时30分起至本区域比赛结束，以下道路禁止车辆通行：中山大道(黄石路至崇仁路)；武胜路(中山大道至沿河大道)；武胜路高架桥；江汉桥。",
      "affected_area": "武胜路"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:30",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "中山大道、武胜路、江汉桥禁止车辆通行",
      "action": "禁止",
      "original_text": "6时30分起至本区域比赛结束，以下道路禁止车辆通行：中山大道(黄石路至崇仁路)；武胜路(中山大道至沿河大道)；武胜路高架桥；江汉桥。",
      "affected_area": "武胜路高架桥"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:30",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "中山大道、武胜路、江汉桥禁止车辆通行",
      "action": "禁止",
      "original_text": "6时30分起至本区域比赛结束，以下道路禁止车辆通行：中山大道(黄石路至崇仁路)；武胜路(中山大道至沿河大道)；武胜路高架桥；江汉桥。",
      "affected_area": "江汉桥"
    }
  ]
}
```

## 示例4：桥梁封闭

**原文文本**
> 6时45分起至本区域比赛结束，以下道路禁止车辆通行：...龟山南路；长江大桥；首义广场地下通道；黄鹤楼东路；黄鹤楼南路；武昌路。

**提取输出**
```json
{
  "affected_areas": [
    {
      "area_name": "鹦鹉大道",
      "type": "路段",
      "boundaries": "江汉桥至汉阳大道",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "龟山南路",
      "type": "路段",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "长江大桥",
      "type": "桥梁",
      "control_measures": "禁止车辆通行"
    },
    {
      "area_name": "首义广场地下通道",
      "type": "隧道",
      "control_measures": "禁止车辆通行"
    }
  ],
  "tasks": [
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:45",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "鹦鹉大道禁止车辆通行",
      "action": "禁止",
      "original_text": "6时45分起至本区域比赛结束，以下道路禁止车辆通行：...龟山南路；长江大桥；首义广场地下通道；黄鹤楼东路；黄鹤楼南路；武昌路。",
      "affected_area": "鹦鹉大道"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:45",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "龟山南路禁止车辆通行",
      "action": "禁止",
      "original_text": "6时45分起至本区域比赛结束，以下道路禁止车辆通行：...龟山南路；长江大桥；首义广场地下通道；黄鹤楼东路；黄鹤楼南路；武昌路。",
      "affected_area": "龟山南路"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:45",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "长江大桥禁止车辆通行",
      "action": "禁止",
      "original_text": "6时45分起至本区域比赛结束，以下道路禁止车辆通行：...龟山南路；长江大桥；首义广场地下通道；黄鹤楼东路；黄鹤楼南路；武昌路。",
      "affected_area": "长江大桥"
    },
    {
      "phase": "赛事进行阶段",
      "phase_code": "racing",
      "start_time": "2025-03-23 06:45",
      "end_time": "本区域比赛结束",
      "control_type": "全封闭",
      "description": "首义广场地下通道禁止车辆通行",
      "action": "禁止",
      "original_text": "6时45分起至本区域比赛结束，以下道路禁止车辆通行：...龟山南路；长江大桥；首义广场地下通道；黄鹤楼东路；黄鹤楼南路；武昌路。",
      "affected_area": "首义广场地下通道"
    }
  ]
}
```
