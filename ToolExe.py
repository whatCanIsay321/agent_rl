import json
import re
from typing import Generic, TypeVar, Type, Any
from pydantic import BaseModel, RootModel, ValidationError


# ============================================================
# ToolBase 基类（包含：strip_title / generate_schema）
# ============================================================
IN = TypeVar("IN", bound=BaseModel)
OUT = TypeVar("OUT", bound=BaseModel)


class ToolBase(Generic[IN, OUT]):
    """
    通用工具基类：
    - 子类定义 name / description
    - 子类实现 Input / Output 模型
    - 子类实现 run(data: Input) -> Output
    """

    name: str = "tool_name"
    description: str = "tool_description"

    # 子类中覆盖
    Input: Type[IN] = BaseModel
    Output: Type[OUT] = BaseModel

    # ============================================================
    # 工具：递归删除 title
    # ============================================================
    @staticmethod
    def _strip_title(schema):
        if isinstance(schema, dict):
            schema.pop("title", None)
            for v in schema.values():
                ToolBase._strip_title(v)
        elif isinstance(schema, list):
            for item in schema:
                ToolBase._strip_title(item)
        return schema

    # ============================================================
    # 生成 Pydantic Schema（支持 RootModel）
    # ============================================================
    @classmethod
    def _generate_schema(cls, model: Type[BaseModel | RootModel]) -> dict:
        schema = model.model_json_schema()
        cls._strip_title(schema)

        # BaseModel 自动添加 required（RootModel 不需要）
        if issubclass(model, BaseModel) and not issubclass(model, RootModel):
            required_fields = [
                name for name, field in model.model_fields.items()
                if field.is_required()
            ]
            if required_fields:
                schema["required"] = required_fields

        return schema

    # ------------ JSON Schema 导出 ------------
    def json_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._generate_schema(self.Input),
                "returns": self._generate_schema(self.Output),
            }
        }

    # ------------ 执行工具（带 try/except）------------
    def execute(self, raw_args: dict) -> dict | Any:
        # 入参校验
        try:
            input_obj = self.Input.model_validate(raw_args)
        except ValidationError as e:
            return {"error": "INVALID_ARGUMENTS", "detail": json.loads(e.json())}

        # 运行工具
        try:
            result = self.run(input_obj)
        except Exception as e:
            return {"error": "TOOL_EXEC_FAILED", "detail": str(e)}

        # 输出校验
        try:
            output_obj = self.Output.model_validate(result)
        except ValidationError as e:
            return {"error": "INVALID_TOOL_RETURN", "detail": json.loads(e.json())}

        # ---- 关键：RootModel 支持 ----
        if issubclass(self.Output, RootModel):
            return output_obj.root

        return output_obj.model_dump()

    # 子类必须实现
    def run(self, data: IN) -> OUT:
        raise NotImplementedError("Tool must implement run()")


# ============================================================
# ToolExe（解析 <tool_call> 并执行对应工具）
# ============================================================
class ToolExe:
    TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

    def __init__(self):
        self.tools = {}

    def register(self, tool: ToolBase):
        self.tools[tool.name] = tool


    def export_schemas(self) -> str:
        """
        导出所有工具 schema，每个 JSON 独立一行，不是数组。
        """
        lines = []
        for tool in self.tools.values():
            schema_str = json.dumps(tool.json_schema(), ensure_ascii=False,indent=1)
            lines.append(schema_str)
        return "\n".join(lines)

    def parse_tool_call(self, text: str):
        try:
            m = self.TOOL_RE.search(text)
            return json.loads(m.group(1))
        except Exception as e:
            raise ValueError(f"{e}")

    def run(self, model_output: str):
        try:
            call = self.parse_tool_call(model_output)
        except Exception as e:
            return {"error": "TOOL_CALL_PARSE_ERROR", "detail": str(e)}

        if call is None:
            return {"error": "NO_TOOL_CALL_DETECTED"}

        name = call.get("name")
        args = call.get("arguments", {})

        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}

        tool = self.tools[name]
        return tool.execute(args)


# ============================================================
# 示例工具：自我介绍
# ============================================================


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    class IntroInput(BaseModel):
        name: str
        lang: str = "zh"


    class IntroOutput(RootModel[str]):
        """你好"""
        pass


    class IntroTool(ToolBase[IntroInput, IntroOutput]):
        name = "自我介绍"
        description = "返回自我介绍文本"

        Input = IntroInput
        Output = IntroOutput

        def run(self, data: IntroInput):
            if data.lang == "zh":
                return f"你好，{data.name}！我是天智AI。"
            return f"Hello, {data.name}! I'm Tianzhi AI."


    class ListUsersInput(BaseModel):
        limit: int = 5


    from typing import List


    class ListUsersOutput(RootModel[List[str]]):
        """返回用户列表"""
        pass


    class ListUsersTool(ToolBase[ListUsersInput, ListUsersOutput]):
        name = "获取用户列表"
        description = "返回一个字符串列表，表示用户昵称"

        Input = ListUsersInput
        Output = ListUsersOutput

        def run(self, data: ListUsersInput):
            # 简单模拟：返回 limit 个用户名字
            users = [f"用户{i + 1}" for i in range(data.limit)]
            return users  # RootModel[List[str]] 的正确返回方式
    tool_exe = ToolExe()
    tool_exe.register(IntroTool())
    tool_exe.register(ListUsersTool())
    print("=== 工具 JSON Schema ===")
    x  = tool_exe.export_schemas()
    print(tool_exe.export_schemas())

    model_output = """
    <tool_call>
    {"name": "自我介绍", "arguments": {"name": "小明", "lang": "zh"}}
    </tool_call>
    """

    print("\n=== 工具执行结果 ===")
    print(tool_exe.run(model_output))
