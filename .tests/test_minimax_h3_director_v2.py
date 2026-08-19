from nodes.nodes_minimax_h3_director_v2 import MiniMaxH3DirectorV2


def test_director_v2_is_a_distinct_internal_execution_node():
    schema = MiniMaxH3DirectorV2.INPUT_TYPES()

    assert "clip" in schema["optional"]
    assert "vae" in schema["optional"]
    assert "fl2va_clip" not in schema["optional"]
    assert "ref2va_clip" not in schema["optional"]
    assert "patched_model" not in schema["optional"]
    assert "patched_clip" not in schema["optional"]
    assert schema["optional"]["fl2va_model"][0] == "MODEL"
    assert schema["optional"]["ref2va_model"][0] == "MODEL"
    assert MiniMaxH3DirectorV2.OUTPUT_NODE is True
    assert MiniMaxH3DirectorV2.RETURN_TYPES == ("FLOAT", "INT", "IMAGE")
    assert MiniMaxH3DirectorV2.RETURN_NAMES == ("frame_rate", "duration", "images")


def test_director_v2_prefers_the_returned_lora_model():
    assert MiniMaxH3DirectorV2.select_execution_model("REF2VA", "fl2", "ref") == "ref"
    assert MiniMaxH3DirectorV2.select_execution_model("I2VA", "fl2", "ref") == "fl2"
