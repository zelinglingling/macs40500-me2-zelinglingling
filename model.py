import math
from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from agents import AudienceAgent


def standing_count(model):
    return sum(agent.standing for agent in model.agents)


def standing_proportion(model):
    return standing_count(model) / len(model.agents)


def stick_in_the_muds(model):
    """Share of agents taking the minority action in the current state.

    Miller and Page use this as a summary statistic for how polarized the
    audience remains at equilibrium. When p < 0.5 (most people sitting),
    this equals p; when p > 0.5 (most people standing), this equals 1 - p.
    A value near 0 means near-consensus; a value near 0.5 means the audience
    is evenly split.
    """
    p = standing_proportion(model)
    return min(p, 1 - p)


class StandingOvationModel(Model):
    """Mesa implementation of Miller and Page's Standing Ovation Problem.

    Main paper-matching defaults:
    - 20 x 20 square auditorium = 400 seats
    - common standing threshold = 0.5
    - initial decision based only on perceived quality
    - later decisions based only on visible neighbors' majority behavior
    - five-neighbor and cone information structures
    - synchronous, asynchronous-random, and asynchronous-incentive updating
    """

    def __init__(
        self,
        width=20,
        height=20,
        threshold=0.5,
        neighborhood="Five Neighbors",
        update_mode="Synchronous",
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.neighborhood = neighborhood
        self.update_mode = update_mode
        self.running = True

        # y=0 is treated as the front row (closest to the stage). Agents can
        # see people beside them and people in rows ahead of them (lower y),
        # but not behind them (higher y). This matches the paper's assumption
        # that audience members face the stage and cannot see rows behind them.
        self.grid = SingleGrid(width, height, torus=False)

        for y in range(height):
            for x in range(width):
                # The paper's baseline draws each agent's perceived quality
                # independently from a uniform distribution on [0, 1].
                quality = self.random.random()
                agent = AudienceAgent(
                    model=self,
                    quality=quality,
                    threshold=threshold,
                )
                self.grid.place_agent(agent, (x, y))

        self.initial_majority_standing = standing_proportion(self) >= 0.5
        self.previous_state = self._state_signature()
        self.stable_steps = 0

        self.datacollector = DataCollector(
            model_reporters={
                "Standing Count": standing_count,
                "Standing Proportion": standing_proportion,
                "Stick in the Muds": stick_in_the_muds,
            }
        )
        self.datacollector.collect(self)

    def _agent_at(self, x, y):
        """Return the agent at (x, y), or None if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid.get_cell_list_contents([(x, y)])[0]
        return None

    def visible_neighbors(self, agent):
        """Return the list of agents visible to this agent.

        The paper defines two information structures:

        Five Neighbors: each agent sees the two agents beside them in the same
        row and the three agents directly ahead in the next row (centered on
        their column). Interior agents have exactly 5 visible neighbors; edge
        and corner agents have fewer because some cells are out of bounds.

        Cones: each agent sees the two same-row side neighbors plus an
        expanding triangular region ahead. At distance d rows ahead, the
        visible window is 2d + 1 cells wide, centered on the agent's column.
        This models the widening field of view as one looks further ahead.

        INCONSISTENCY NOTE: The paper does not specify what happens for agents
        with no visible neighbors (e.g., front-row agents under Five Neighbors
        who have no row ahead of them). I retain the agent's current action
        in this case rather than forcing a default stand or sit, which is the
        most neutral choice and avoids biasing front-row behavior.
        """
        x, y = agent.pos
        neighbors = []

        # Two same-row side neighbors are included in both structures.
        for dx in (-1, 1):
            neighbor = self._agent_at(x + dx, y)
            if neighbor is not None:
                neighbors.append(neighbor)

        if self.neighborhood == "Cones":
            # At distance d ahead, include all cells within d columns of x.
            # The window width is 2d + 1 and grows with each row further ahead.
            for distance in range(1, y + 1):
                row = y - distance
                for dx in range(-distance, distance + 1):
                    neighbor = self._agent_at(x + dx, row)
                    if neighbor is not None:
                        neighbors.append(neighbor)
        else:
            # Five-neighbor structure: only the immediately preceding row,
            # three cells wide (left diagonal, directly ahead, right diagonal).
            row = y - 1
            for dx in (-1, 0, 1):
                neighbor = self._agent_at(x + dx, row)
                if neighbor is not None:
                    neighbors.append(neighbor)

        return neighbors

    def disagreement_score(self, agent):
        """Compute how unlike an agent is from its visible neighbors.

        Used for incentive-based asynchronous updating. Miller and Page
        describe agents who are "least like the people around them" as moving
        first. I operationalize this as the fraction of visible neighbors
        currently taking the opposite action: a score of 1.0 means all
        neighbors disagree; 0.0 means all neighbors agree.

        INCONSISTENCY NOTE: The paper does not give an exact formula for this
        score. The fraction-of-disagreeing-neighbors interpretation is the
        most natural reading and produces a value in [0, 1] that can be
        consistently ranked across agents.

        Agents with no visible neighbors receive -inf so they are always
        updated last, since their disagreement score is undefined.
        """
        neighbors = self.visible_neighbors(agent)
        if not neighbors:
            return -math.inf
        opposite = sum(neighbor.standing != agent.standing for neighbor in neighbors)
        return opposite / len(neighbors)

    def _state_signature(self):
        """Return a tuple of all agents' standing states, sorted by position.

        Used to detect convergence: if the state is unchanged for two
        consecutive steps, the model has reached a fixed point or is in a
        2-cycle and is stopped.
        """
        return tuple(agent.standing for agent in sorted(self.agents, key=lambda a: a.pos))

    def _synchronous_step(self):
        """Update all agents simultaneously.

        All agents first compute their next action based on the current state
        of their neighbors (decide phase), then all commit to the new action
        at once (advance phase). This two-phase update matches the paper's
        default synchronous updating rule.

        INCONSISTENCY NOTE: Synchronous updating can produce 2-cycles in which
        the population oscillates indefinitely between two states. The paper
        acknowledges this possibility but does not prescribe a resolution.
        I stop the model after two consecutive identical states, which catches
        fixed points; persistent 2-cycles will keep the model running until
        the user stops it manually.
        """
        for agent in self.agents:
            agent.decide_from_visible_neighbors(self.visible_neighbors(agent))
        for agent in self.agents:
            agent.advance()

    def _asynchronous_random_step(self):
        """Update agents one at a time in a randomly shuffled order.

        Each agent both decides and commits immediately before the next agent
        acts, so later agents in the sequence see the updated states of earlier
        ones. The paper presents this as a robustness check on synchronous
        updating.
        """
        agents = list(self.agents)
        self.random.shuffle(agents)
        for agent in agents:
            agent.decide_from_visible_neighbors(self.visible_neighbors(agent))
            agent.advance()

    def _asynchronous_incentive_step(self):
        """Update agents in descending order of disagreement with neighbors.

        Agents who are most unlike their visible neighbors act first, modeling
        the intuition that social pressure is strongest for those who stand out
        the most. As with random asynchronous updating, each agent commits
        immediately so subsequent agents see the updated state.
        """
        agents = sorted(
            list(self.agents),
            key=lambda agent: self.disagreement_score(agent),
            reverse=True,
        )
        for agent in agents:
            agent.decide_from_visible_neighbors(self.visible_neighbors(agent))
            agent.advance()

    def step(self):
        if self.update_mode == "Asynchronous-Random":
            self._asynchronous_random_step()
        elif self.update_mode == "Asynchronous-Incentive-Based":
            self._asynchronous_incentive_step()
        else:
            self._synchronous_step()

        new_state = self._state_signature()
        if new_state == self.previous_state:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        self.previous_state = new_state

        # Stop after two consecutive unchanged states. Two steps (rather than
        # one) are required because synchronous updating can produce alternating
        # states that look stable for one step but are actually cycling.
        # Users can continue stepping manually if the model stops prematurely.
        if self.stable_steps >= 2:
            self.running = False

        self.datacollector.collect(self)