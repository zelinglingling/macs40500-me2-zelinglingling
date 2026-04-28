import solara
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model import StandingOvationModel

# Agent visualization
def agent_portrayal(agent):
    # Red = standing, blue = sitting.
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
    "width": 20,
    "height": 20,

    "threshold": {
        "type": "SliderFloat",
        "label": "Threshold",
        "value": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },

    "neighborhood": {
        "type": "Select",
        "label": "Neighborhood",
        "value": "Five Neighbors",
        "values": ["Five Neighbors", "Cones"],
    },

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

    "quality_distribution": {
        "type": "Select",
        "label": "Quality Distribution",
        "value": "uniform",
        "values": ["uniform", "normal"],
    },

    "mean_quality": {
        "type": "SliderFloat",
        "label": "Mean Quality (normal only)",
        "value": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },

    "quality_noise": {
        "type": "SliderFloat",
        "label": "Quality Noise (normal only)",
        "value": 0.15,
        "min": 0.0,
        "max": 0.5,
        "step": 0.01,
    },

    "seed": 42,
}


# Visualization components
space = make_space_component(agent_portrayal)

plot1 = make_plot_component("Standing Proportion")
plot2 = make_plot_component("Standing Count")
plot3 = make_plot_component("Stick in the Muds")


# Initial model instance
model = StandingOvationModel(
    threshold=0.5,
    neighborhood="Five Neighbors",
    update_mode="Synchronous",
    quality_distribution="uniform",
    mean_quality=0.5,
    quality_noise=0.15,
    seed=42,
)

# Page
page = SolaraViz(
    model,
    components=[
        space,   # auditorium grid
        plot1,   # proportion standing over time
        plot2,   # raw count standing over time
        plot3,   # minority-action share (stick-in-the-muds) over time
    ],
    model_params=model_params,
    name="Standing Ovation Model",
)

page
