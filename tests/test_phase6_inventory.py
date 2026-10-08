"""Phase 6 inventory test: every required check is registered."""

from checks.run_all import CHECK_MODULES


def test_required_check_inventory_has_all_tasks() -> None:
    expected = {
        "checks.task1_single_chapter",
        "checks.task1_two_chapter",
        "checks.task1_before_after",
        "checks.task1_unsupported",
        "checks.task1_malformed_max_chapter",
        "checks.task1_alias_reference",
        "checks.task1_contradiction",
        "checks.task2_success",
        "checks.task2_reject",
        "checks.task2_invalid_args",
        "checks.task2_duplicate_save",
        "checks.task2_injection",
        "checks.task2_direct_guards",
        "checks.task2_inert_tool_call_syntax",
        "checks.task3_R1",
        "checks.task3_R2",
        "checks.task3_R3",
        "checks.task3_R4_4096",
        "checks.task3_R4_alt_threshold",
        "checks.task3_R5",
        "checks.task3_R6",
        "checks.task4_repair",
    }
    assert set(CHECK_MODULES) == expected
    assert len(CHECK_MODULES) == 22
