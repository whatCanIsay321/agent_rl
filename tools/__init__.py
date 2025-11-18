import os
import pkgutil
import importlib
from tools.base import ToolBase

ALL_TOOLS = []

package_dir = os.path.dirname(__file__)

for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if is_pkg:                # 跳过子包（如果有）
        continue
    if module_name == "base": # 跳过工具基类文件
        continue
    if module_name.startswith("_"):
        continue

    module = importlib.import_module(f"{__name__}.{module_name}")

    # 扫描 ToolBase 子类
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        try:
            if isinstance(attr, type) and issubclass(attr, ToolBase) and attr is not ToolBase:
                ALL_TOOLS.append(attr)
        except Exception:
            pass
