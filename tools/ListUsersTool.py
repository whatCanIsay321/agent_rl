from pydantic import BaseModel, RootModel
from tools.base import ToolBase
from typing import List

class ListUsersInput(BaseModel):
    limit: int = 5

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