from mesa import Agent


class AudienceAgent(Agent):
    """One audience member in Miller and Page's Standing Ovation Problem."""

    def __init__(self, model, pos, quality, threshold=0.5):
        super().__init__(model)
        self.pos = pos
        self.quality = quality
        self.threshold = threshold

        # Period 0 in the paper: each agent observes a private noisy signal
        # of the performance quality and stands if and only if that signal
        # weakly exceeds the common threshold. This is the only step in which
        # private quality information influences behavior; from period 1 onward
        # agents respond only to visible neighbors (see decide_from_visible_neighbors).
        self.standing = self.quality >= self.threshold
        self.next_standing = self.standing

    def decide_from_visible_neighbors(self, visible_neighbors):
        """Compute the next action using the post-period-0 majority rule.

        Miller and Page's computational model makes a deliberately sharp
        transition: after the initial quality-based decision, agents ignore
        their own quality signal entirely and stand if and only if a strict
        majority of their visible neighbors are standing.

        INCONSISTENCY NOTE: The paper does not specify behavior for agents
        with no visible neighbors (e.g., front-row agents under the Five
        Neighbors structure who have no row ahead of them). Retaining the
        current action is the most neutral choice and avoids forcing
        arbitrary behavior for corner and front-row seats.

        INCONSISTENCY NOTE: The paper does not explicitly state how ties
        are broken when exactly half of visible neighbors are standing.
        A 50/50 random choice is the most natural symmetric resolution
        and is consistent with the paper's general treatment of indifference.
        """
        if not visible_neighbors:
            self.next_standing = self.standing
            return

        standing_count = sum(neighbor.standing for neighbor in visible_neighbors)
        total = len(visible_neighbors)

        if standing_count > total / 2:
            self.next_standing = True
        elif standing_count < total / 2:
            self.next_standing = False
        else:
            # Tie: stand or sit with equal probability.
            self.next_standing = self.random.random() < 0.5

    def advance(self):
        """Commit the decision computed in decide_from_visible_neighbors.

        Separating decide and advance into two methods supports synchronous
        updating in Mesa: all agents call decide first (reading the current
        state), then all call advance (writing the new state), so no agent
        sees a partially updated grid.
        """
        self.standing = self.next_standing