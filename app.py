import solara
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model import StandingOvationModel


# Agent visualization

def agent_portrayal(agent):
    """Map each agent's state to a color for the grid display.

    Red = standing, blue = sitting. This gives an immediate visual read of
    how standing clusters form and dissolve across the auditorium over time.
    """
    if agent.standing:
        color = "red"
    else:
        color = "blue"

    return {
        "color": color,
        "size": 20,
    }


# Model parameters

model_params = {
    # Width and height are fixed at 20x20 to match the paper's default
    # auditorium size of 400 seats.
    "width": 20,
    "height": 20,

    # Threshold: the minimum perceived quality required for an agent to stand
    # in period 0. The paper's baseline is 0.5.
    "threshold": {
        "type": "SliderFloat",
        "label": "Threshold",
        "value": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },

    # Neighborhood structure controlling which agents are visible to each other.
    # "Five Neighbors" is the paper's default; "Cones" expands the visible
    # region with each additional row ahead.
    "neighborhood": {
        "type": "Select",
        "label": "Neighborhood",
        "value": "Five Neighbors",
        "values": ["Five Neighbors", "Cones"],
    },

    # Updating scheme. "Synchronous" matches the paper's primary model.
    # The two asynchronous modes are presented in the paper as robustness
    # checks. String values here must exactly match the if/elif branches
    # in model.py's step() method.
    "update_mode": {
        "type": "Select",
        "label": "Update Mode",
        "value": "Synchronous",
        "values": [
            "Synchronous",
            "Asynchronous-Random",
            "Asynchronous-Incentive-Based",
        ],
    },

    "seed": 42,
}


# Visualization components

space = make_space_component(agent_portrayal)

plot1 = make_plot_component("Standing Proportion")  # fraction of audience standing
plot2 = make_plot_component("Standing Count")        # raw number standing
plot3 = make_plot_component("Stick in the Muds")     # minority-action share


# Initial model instance

model = StandingOvationModel(
    threshold=0.5,
    neighborhood="Five Neighbors",
    update_mode="Synchronous",
    seed=42,
)


# Page

page = SolaraViz(
    model,
    components=[
        space,   # auditorium grid: spatial distribution of standing/sitting
        plot1,   # proportion standing over time
        plot2,   # raw count standing over time
        plot3,   # minority-action share (stick-in-the-muds) over time
    ],
    model_params=model_params,
    name="Standing Ovation Model",
)

page