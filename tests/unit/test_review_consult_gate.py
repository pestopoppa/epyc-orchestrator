from src.orchestration.review_consult_gate import review_before_commit_targeted_gate


def test_review_before_commit_gate_triggers_on_parser_data_contract_edge() -> None:
    decision = review_before_commit_targeted_gate(
        task_prompt="Fix the YAML parser so comments and quoted values round-trip correctly.",
        current_paths=["src/config_loader.py"],
        draft_paths=["src/config_loader.py"],
        delete_paths=[],
        raw_model_output="<<<FILE: src/config_loader.py>>>\n...\n<<<END>>>",
    )

    assert decision.enabled is True
    assert "parser_data_contract_or_compatibility" in decision.reasons
    assert "hidden_verifier_or_transaction_risk" in decision.reasons


def test_review_before_commit_gate_triggers_on_public_registry_surface() -> None:
    decision = review_before_commit_targeted_gate(
        task_prompt="Add a plugin entry.",
        current_paths=["src/plugins/registry.py"],
        draft_paths=["src/plugins/registry.py"],
        delete_paths=[],
        raw_model_output="<<<FILE: src/plugins/registry.py>>>\n...\n<<<END>>>",
    )

    assert decision.enabled is True
    assert "public_api_registry_or_config_surface" in decision.reasons


def test_review_before_commit_gate_skips_plain_single_file_edit() -> None:
    decision = review_before_commit_targeted_gate(
        task_prompt="Rename the greeting string.",
        current_paths=["hello.py"],
        draft_paths=["hello.py"],
        delete_paths=[],
        raw_model_output="<<<FILE: hello.py>>>\nGREETING = 'hi'\n<<<END>>>",
    )

    assert decision.enabled is False
    assert decision.reasons == ()


def test_review_before_commit_gate_skips_unparsed_draft() -> None:
    decision = review_before_commit_targeted_gate(
        task_prompt="Fix parser comments.",
        current_paths=["parser.py"],
        draft_paths=[],
        delete_paths=[],
        raw_model_output="I would change the parser.",
    )

    assert decision.enabled is False
    assert decision.reasons == ("no_parsed_file_blocks",)
