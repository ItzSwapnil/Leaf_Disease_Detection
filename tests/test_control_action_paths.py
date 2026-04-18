import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


APP_FILE = PROJECT_ROOT / "app.py"


def _string_assignments(module: ast.Module) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                target = node.targets[0].id
                values[target] = node.value.value
    return values


def _resolve_string(node: ast.AST, known: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return known.get(node.id) if isinstance(node, ast.Name) else None


def _extract_control_actions() -> dict[str, str]:
    source = APP_FILE.read_text(encoding="utf-8")
    module = ast.parse(source)
    known = _string_assignments(module)

    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if (
            not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id != "CONTROL_ACTIONS"
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        actions: dict[str, str] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            action_key = _resolve_string(key_node, known)
            if action_key is None or not isinstance(value_node, ast.Dict):
                continue

            script_value = next(
                (
                    _resolve_string(inner_value, known)
                    for inner_key, inner_value in zip(
                        value_node.keys, value_node.values
                    )
                    if _resolve_string(inner_key, known) == "script"
                ),
                None,
            )

            if script_value is not None:
                actions[action_key] = script_value

        return actions

    raise AssertionError("CONTROL_ACTIONS assignment was not found in app.py")


def test_control_actions_reference_existing_scripts():
    actions = _extract_control_actions()

    expected_actions = {"train", "fine_tune", "evaluate", "generate_figures"}
    missing = expected_actions - set(actions)
    assert not missing, f"Missing expected CONTROL_ACTIONS keys: {sorted(missing)}"

    for action_key, relative_path in actions.items():
        script_path = PROJECT_ROOT / relative_path
        assert script_path.exists(), (
            f"Action '{action_key}' points to missing script: {relative_path}"
        )
        assert script_path.is_file(), (
            f"Action '{action_key}' path is not a file: {relative_path}"
        )
