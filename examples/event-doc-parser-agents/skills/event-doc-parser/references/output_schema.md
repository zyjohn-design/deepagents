# 输出JSON Schema定义

## 完整Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EventDocExtractionResult",
  "description": "大型活动文档提取结果的统一格式",
  "type": "object",
  "properties": {
    "event_name": {
      "type": "string",
      "description": "活动名称，从文档标题提取"
    },
    "event_type": {
      "type": "string",
      "description": "活动类型中文描述",
      "enum": ["跨年夜", "马拉松", "演唱会", "体育赛事", "大型集会", "其他"]
    },
    "scenario_detected": {
      "type": "string",
      "description": "识别到的场景类型代码",
      "enum": ["newyear", "marathon", "concert", "sports", "general"]
    },
    "event_date": {
      "type": "string",
      "description": "活动日期，ISO 8601格式 (YYYY-MM-DD)"
    },
    "affected_areas": {
      "type": "array",
      "description": "活动涉及的所有封控区域列表",
      "items": {
        "$ref": "#/definitions/AffectedArea"
      }
    },
    "tasks": {
      "type": "array",
      "description": "按时间顺序排列的管控任务列表",
      "items": {
        "$ref": "#/definitions/Task"
      }
    }
  },
  "required": ["event_name", "event_type", "scenario_detected", "affected_areas", "tasks"],
  "definitions": {
    "Task": {
      "type": "object",
      "description": "单个管控任务",
      "properties": {
        "phase": {
          "type": "string",
          "description": "所处阶段名称",
          "enum": ["启动准备阶段", "管控实施阶段", "疏散收尾阶段", "全时段保障"]
        },
        "start_time": {
          "type": "string",
          "description": "开始时间 (YYYY-MM-DD HH:MM)"
        },
        "end_time": {
          "type": "string",
          "description": "结束时间"
        },
        "control_type": {
          "type": "string",
          "description": "封控类型",
          "enum": ["全封闭", "半封闭", "交通管制", "人流管控", "车辆禁停", "临时管控"]
        },
        "description": {
          "type": "string",
          "description": "简洁任务描述",
          "maxLength": 200
        },
        "action": {
          "type": "string",
          "description": "执行动作",
          "enum": ["启动", "禁止", "恢复", "部署", "疏散", "封闭", "开放", "鸣枪", "完成", "检查"]
        },
        "affected_area": {
          "type": "string",
          "description": "受影响的区域名称（必须引用affected_areas中的area_name）"
        }
      },
      "required": ["phase", "start_time", "control_type", "description", "action", "affected_area"]
    },
    "AffectedArea": {
      "type": "object",
      "description": "受影响的区域或线路",
      "properties": {
        "area_name": {
          "type": "string",
          "description": "区域或道路名称"
        },
        "type": {
          "type": "string",
          "description": "区域类型（严格遵守知识库定义）",
          "enum": ["核心管控区", "中端疏导区", "远端分流区", "管控区"]
        },
        "phase": {
          "type": "string",
          "description": "该区域/线路管控生效的阶段。如果全时段不变，则使用'全时段保障'",
          "enum": ["启动准备阶段", "管控实施阶段", "疏散收尾阶段", "全时段保障"]
        },
        "start_time": {
          "type": "string",
          "description": "管控开始时间 (YYYY-MM-DD HH:MM)"
        },
        "end_time": {
          "type": "string",
          "description": "管控结束时间"
        },
        "boundaries": {
          "type": "string",
          "description": "边界描述"
        },
        "control_measures": {
          "type": "string",
          "description": "具体管控措施"
        }
      },
      "required": ["area_name", "type", "phase"]
    }
  }
}
```

## 变更说明
- 精简了冗余字段 (phase_code, original_text, conditions, responsible_dept, extraction_metadata)。
- 严格限制 `phase` 枚举为 `["启动准备阶段", "管控实施阶段", "疏散收尾阶段", "全时段保障"]`。
- 严格限制 `AffectedArea.type` 枚举为 `["核心管控区", "中端疏导区", "远端分流区", "管控区"]`。
- `AffectedArea` 新增 `phase`, `start_time`, `end_time` 字段，支持分阶段定义管控区域。
