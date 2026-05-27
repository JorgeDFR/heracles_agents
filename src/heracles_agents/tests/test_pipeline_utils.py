from types import SimpleNamespace

import pytest
import yaml

from heracles_agents.pipelines import codegen_utils, in_context_utils, prompt_utils


def test_answer_formatting_guidance_by_output_type():
    question = SimpleNamespace(solution="<1, 2>")

    sldp = prompt_utils.get_answer_formatting_guidance_helper(
        SimpleNamespace(output_type="SLDP", answer_type_hint=True),
        question,
    )
    assert "SLDP Equality Language" in sldp
    assert "<answer>" in sldp
    assert "Your answer should be an SLDP set" in sldp

    sldp_tool = prompt_utils.get_answer_formatting_guidance_helper(
        SimpleNamespace(output_type="SLDP_TOOL", sldp_answer_type_hint=True),
        question,
    )
    assert "Call the tool sldp_answer_tool" in sldp_tool

    pddl = prompt_utils.get_answer_formatting_guidance_helper(
        SimpleNamespace(output_type="PDDL", sldp_answer_type_hint=False),
        question,
    )
    assert "Return the PDDL goal" in pddl

    assert (
        prompt_utils.get_answer_formatting_guidance_helper(
            SimpleNamespace(output_type=None, sldp_answer_type_hint=False),
            question,
        )
        is None
    )
    with pytest.raises(NotImplementedError):
        prompt_utils.get_answer_formatting_guidance_helper(
            SimpleNamespace(output_type="PDDL_TOOL", sldp_answer_type_hint=False),
            question,
        )
    with pytest.raises(ValueError, match="Unknown output type"):
        prompt_utils.get_answer_formatting_guidance_helper(
            SimpleNamespace(output_type="BAD", sldp_answer_type_hint=False),
            question,
        )


def test_get_answer_formatting_guidance_reads_nested_agent_config():
    agent_config = SimpleNamespace(
        agent_info=SimpleNamespace(
            prompt_settings=SimpleNamespace(
                output_type="PDDL",
                sldp_answer_type_hint=False,
            )
        )
    )

    assert "Return the PDDL goal" in prompt_utils.get_answer_formatting_guidance(
        agent_config, SimpleNamespace(solution="(and)")
    )


def test_codegen_format_callable_api_includes_docstrings_and_examples():
    rendered = codegen_utils.format_callable_api(
        {
            "name": "lookup",
            "description": "Find a thing.",
            "inputs": [{"name": "query", "type": "str", "description": "Search text"}],
            "output": {"type": "int", "description": "Count"},
            "example": "x = self.lookup('chair')",
        },
        "Graph",
        is_constructor=False,
        include_descriptions=True,
        include_examples=True,
    )

    assert "def lookup(self, query: str) -> int:" in rendered
    assert '"""Find a thing.' in rendered
    assert "query (str): Search text" in rendered
    assert "Returns:" in rendered
    assert "# x = self.lookup('chair')" in rendered

    constructor = codegen_utils.format_callable_api(
        {"inputs": [{"name": "path", "type": "str"}]},
        "Graph",
        is_constructor=True,
        include_descriptions=False,
        include_examples=False,
    )
    assert constructor == "  def __init__(self, path: str):"


def test_load_dsg_api_prompt_filters_and_formats_api_yaml(tmp_path):
    api_file = tmp_path / "api.yaml"
    api_file.write_text(
        yaml.safe_dump(
            {
                "api": {
                    "name": "SceneGraph",
                    "version": "1",
                    "description": "Graph API",
                    "classes": [
                        {
                            "name": "Graph",
                            "include": True,
                            "description": "Graph class",
                            "constructor": {
                                "include": True,
                                "inputs": [{"name": "path", "type": "str"}],
                            },
                            "properties": [
                                {
                                    "name": "nodes",
                                    "type": "list",
                                    "description": "All nodes",
                                }
                            ],
                            "methods": [
                                {
                                    "name": "count",
                                    "include": True,
                                    "inputs": [],
                                    "output": {"type": "int", "description": "Count"},
                                },
                                {"name": "hidden", "include": False},
                            ],
                            "enums": [{"name": "Layer", "values": ["OBJECTS"]}],
                        },
                        {"name": "Ignored", "include": False},
                    ],
                    "enums": [
                        {
                            "name": "Status",
                            "description": "State",
                            "values": [
                                "OK",
                                {"name": "BAD", "description": "Failure"},
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    rendered = codegen_utils.load_dsg_api_prompt(
        api_file,
        include_descriptions=True,
        include_examples=False,
    )

    assert "# SceneGraph API Reference (Version 1)" in rendered
    assert "Graph API" in rendered
    assert "class Graph" in rendered
    assert "def __init__(self, path: str):" in rendered
    assert "nodes: list  # All nodes" in rendered
    assert "def count(self) -> int:" in rendered
    assert "hidden" not in rendered
    assert "class Status" in rendered
    assert "BAD = ...  # Failure" in rendered


def test_load_dsg_uses_cache(monkeypatch):
    codegen_utils.dsg_cache.clear()
    loads = []

    def load(path):
        graph = object()
        loads.append((path, graph))
        return graph

    monkeypatch.setattr(codegen_utils.spark_dsg.DynamicSceneGraph, "load", load)

    first = codegen_utils.load_dsg("scene.dsg")
    second = codegen_utils.load_dsg("scene.dsg")
    third = codegen_utils.load_dsg("other.dsg")

    assert first is second
    assert third is not first
    assert [path for path, graph in loads] == ["scene.dsg", "other.dsg"]

    codegen_utils.dsg_cache.clear()


def test_codegen_execute_generated_code_paths(monkeypatch):
    assert codegen_utils.execute_generated_code(
        "def solve_task(G):\n    return len(G)",
        [1, 2, 3],
    ) == (True, 3)
    assert codegen_utils.execute_generated_code("x = 1", []) == (
        False,
        "'solve_task' function not found in the generated code.",
    )
    ok, message = codegen_utils.execute_generated_code("raise RuntimeError('boom')", [])
    assert not ok
    assert "boom" in message

    monkeypatch.setattr(
        codegen_utils,
        "run_with_timeout",
        lambda fn, args, timeout: (_ for _ in ()).throw(
            codegen_utils.FunctionTimeoutError("timeout")
        ),
    )
    assert codegen_utils.execute_generated_code_timed("code", []) == (
        "Your code timed out. In 60 seconds."
    )


class FakeId:
    def __init__(self, value):
        self.value = value

    def str(self, include_prefix):
        return self.value


class FakeNode:
    def __init__(self, node_id, label=1, position=(1, 2, 3), parents=None, siblings=None):
        self.id = FakeId(node_id)
        self.attributes = SimpleNamespace(semantic_label=label, position=position)
        self._parents = parents or []
        self._siblings = siblings or []

    def parents(self):
        return self._parents

    def siblings(self):
        return self._siblings


class FakeLabelspace:
    def __init__(self, prefix):
        self.prefix = prefix

    def get_category(self, label):
        return f"{self.prefix}-{label}"


class FakeSceneGraph:
    def __init__(self):
        self.object = FakeNode("O1", parents=["P1"])
        self.place = FakeNode("P1", parents=["R1"])
        self.room = FakeNode("R1", position=(4, 5, 6), siblings=[])
        self.nodes = {"O1": self.object, "P1": self.place, "R1": self.room}

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_labelspace(self, layer, index):
        if layer == 2:
            return FakeLabelspace("object")
        if layer == 4:
            return FakeLabelspace("room")
        return None

    def get_layer(self, layer):
        if layer == in_context_utils.spark_dsg.DsgLayers.OBJECTS:
            return SimpleNamespace(nodes=[self.object])
        if layer == in_context_utils.spark_dsg.DsgLayers.ROOMS:
            return SimpleNamespace(nodes=[self.room])
        if layer == "places":
            return SimpleNamespace(nodes=[self.place])
        return SimpleNamespace(nodes=[])


def test_in_context_scene_graph_formatting_with_fake_graph():
    graph = FakeSceneGraph()

    assert in_context_utils.get_position_string(graph.object.attributes) == "(1.00,2.00,3.00)"
    assert in_context_utils.get_room_parents_of_object(graph.object, graph) == {"R1"}
    assert "type=room-1" in in_context_utils.room_to_string(graph.room, graph)
    assert "parent_rooms={'R1'}" in in_context_utils.object_to_string_room_parent(
        graph.object, graph
    )

    compact = in_context_utils.scene_graph_to_prompt(graph)
    assert "<Scene Graph>" in compact
    assert "Objects:" in compact
    assert "Rooms:" in compact

    full = in_context_utils.scene_graph_to_prompt_full(graph, "places")
    assert "Places:" in full
    assert "parent_places={'P1'}" in full
    assert "parent_rooms={'R1'}" in full


def test_in_context_scene_graph_errors_and_empty_parents():
    graph = FakeSceneGraph()
    graph.object._parents = []
    assert in_context_utils.get_room_parents_of_object(graph.object, graph) == "none"

    graph.get_labelspace = lambda layer, index: None
    with pytest.raises(in_context_utils.PromptingFailure, match="object labelspace"):
        in_context_utils.object_to_string_room_parent(graph.object, graph)
    with pytest.raises(in_context_utils.PromptingFailure, match="room labelspace"):
        in_context_utils.room_to_string(graph.room, graph)
