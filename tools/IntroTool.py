from pydantic import BaseModel, RootModel
from tools.base import ToolBase
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
