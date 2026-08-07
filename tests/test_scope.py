"""Tests for memory scopes and the bar for promoting a claim to shared knowledge.

The split these enforce: personal memories are observations about one owner and
never travel; knowledge is a claim about the world and is the only shareable
scope. Mixing them is what makes a shared layer unshareable — it ends up full of
one person's preferences — and a personal layer impersonal.
"""

import unittest

from fibmind import FibMind, MemoryScope, RelationType
from fibmind.graph import DEFAULT_PROMOTION_THRESHOLD


def _observations(memory: FibMind, owner: str, count: int) -> list[str]:
    return [
        memory.append(
            "math",
            f"{owner} missed the discriminant #{index}",
            "solved a quadratic without checking whether the discriminant is negative",
            scope=MemoryScope.PERSONAL,
            owner=owner,
        )
        for index in range(count)
    ]


class ScopeTests(unittest.TestCase):
    def test_personal_memories_do_not_cross_owners(self) -> None:
        memory = FibMind()
        mine = memory.append(
            "prefs", "Prefers terse answers", "wants short replies", owner="student-a"
        )
        theirs = memory.append(
            "prefs", "Prefers worked examples", "wants long replies", owner="student-b"
        )

        visible = {hit.node.id for hit in memory.search("prefers replies", owner="student-a")}

        self.assertIn(mine, visible)
        self.assertNotIn(theirs, visible)

    def test_missing_owner_does_not_expose_named_personal_memory(self) -> None:
        memory = FibMind()
        named = memory.append(
            "prefs", "Private preference", "student-a prefers terse replies", owner="student-a"
        )

        visible = {hit.node.id for hit in memory.search("private preference terse")}

        self.assertNotIn(named, visible)

    def test_named_owner_does_not_inherit_unowned_personal_memory(self) -> None:
        memory = FibMind()
        legacy = memory.append("prefs", "Legacy preference", "prefers long replies")

        visible = {hit.node.id for hit in memory.search("legacy preference", owner="student-a")}

        self.assertNotIn(legacy, visible)

    def test_knowledge_is_visible_to_everyone(self) -> None:
        memory = FibMind()
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD)
        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students solving quadratics skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        visible = {hit.node.id for hit in memory.search("discriminant", owner="student-b")}

        self.assertIn(claim, visible)
        self.assertTrue(visible.isdisjoint(set(supporting)))

    def test_scope_filter_restricts_results(self) -> None:
        memory = FibMind()
        personal = memory.append("math", "Missed a sign", "dropped a minus", owner="student-a")
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD)
        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        knowledge_only = {
            hit.node.id
            for hit in memory.search(
                "discriminant minus sign", scopes={MemoryScope.KNOWLEDGE}, top_k=50
            )
        }

        self.assertIn(claim, knowledge_only)
        self.assertNotIn(personal, knowledge_only)


class PromotionGateTests(unittest.TestCase):
    def test_promotion_requires_enough_supporting_observations(self) -> None:
        """Whether something generalizes is a claim about a population."""
        memory = FibMind()
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD - 1)

        with self.assertRaises(ValueError) as caught:
            memory.promote_to_knowledge(
                title="Everyone skips the discriminant",
                content="all students skip the discriminant check",
                supporting_node_ids=supporting,
            )

        self.assertIn(str(DEFAULT_PROMOTION_THRESHOLD), str(caught.exception))

    def test_duplicate_supporters_do_not_count_twice(self) -> None:
        memory = FibMind()
        one = _observations(memory, "student-a", 1)[0]

        with self.assertRaises(ValueError):
            memory.promote_to_knowledge(
                title="Everyone skips the discriminant",
                content="all students skip the discriminant check",
                supporting_node_ids=[one] * DEFAULT_PROMOTION_THRESHOLD,
            )

    def test_promotion_keeps_provenance_pointers(self) -> None:
        """A claim that turns out wrong has to be traceable to what produced it."""
        memory = FibMind()
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD)

        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        derived_from = {
            edge.to_node_id
            for edge in memory.edges.values()
            if edge.from_node_id == claim and edge.relation_type == RelationType.DERIVED_FROM
        }
        self.assertEqual(set(supporting), derived_from)
        self.assertEqual(
            memory.nodes[claim].metadata["supporting_node_ids"], list(supporting)
        )

    def test_promoted_claims_start_unproven(self) -> None:
        memory = FibMind()
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD)

        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        self.assertEqual(memory.nodes[claim].scope, MemoryScope.KNOWLEDGE)
        self.assertEqual(memory.nodes[claim].confidence, 0.0)

    def test_forgetting_a_claim_leaves_its_observations(self) -> None:
        memory = FibMind()
        supporting = _observations(memory, "student-a", DEFAULT_PROMOTION_THRESHOLD)
        claim = memory.promote_to_knowledge(
            title="Discriminant sign is routinely skipped",
            content="Students skip checking whether the discriminant is negative",
            supporting_node_ids=supporting,
        )

        memory.forget(claim, reason="refuted by a wider sample")

        self.assertNotIn(claim, memory.nodes)
        for node_id in supporting:
            self.assertIn(node_id, memory.nodes)


if __name__ == "__main__":
    unittest.main()
