from mesa import Agent


class AudienceAgent(Agent):
    """One audience member in Miller and Page's Standing Ovation Problem."""

    def __init__(self, model, pos, quality, threshold=0.5):
        super().__init__(model)
        self.pos = pos
        self.quality = quality
        self.threshold = threshold

        # Period 0 in the paper: agents stand only if their private perceived quality weakly exceeds the threshold.
        self.standing = self.quality >= self.threshold
        self.next_standing = self.standing

    def decide_from_visible_neighbors(self, visible_neighbors):
        """Compute the next action using the post-period-0 majority rule.

        Miller and Page's computational model makes a deliberately sharp
        transition: after the initial quality-based decision, agents ignore
        their own quality and stand if a majority of visible neighbors stand.
        In a tie, they stand or sit with equal probability.
        """
        if not visible_neighbors:
            # The paper is vague for agents with no visible neighbors. Keeping the current action avoids forcing arbitrary behavior in corners.
            self.next_standing = self.standing
            return

        standing_count = sum(neighbor.standing for neighbor in visible_neighbors)
        total = len(visible_neighbors)

        if standing_count > total / 2:
            self.next_standing = True
        elif standing_count < total / 2:
            self.next_standing = False
        else:
            self.next_standing = self.random.random() < 0.5

    def advance(self):
        self.standing = self.next_standing
