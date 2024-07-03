import pandas as pd
import seaborn as sns
from functools import partial

# Import data from shared.py
from shared import data, ratio
from prediction import predict
from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import page_navbar

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Page title (with some additional top padding)
ui.page_opts(
    title="Aleksandra Wojcik",
    page_fn=partial(page_navbar, id="page"),
)

with ui.nav_panel("Distributions"):
    with ui.layout_columns(col_widths=(8, 4)):
        with ui.card():
            @render.plot
            def dist():
                p = sns.displot(data, x=input.dist_var(), hue=data["stroke"])
                return p.set(xlabel=None)
        with ui.card():
            nd = data[data["stroke"] == 1]
            nd = nd[nd["gender"] == 0]
            @render.text()
            def dist_ratio():
                txt = ""
                for value in data[input.dist_var()].unique():
                    txt += f"{value}: {str(ratio(input.dist_var(), value))}%\n"
                return txt

    ui.input_select("dist_var", "Select variable", choices=["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","bmi","smoking_status"])

with ui.nav_panel("Statistical relationships"):
    @render.plot
    def stat():
        p = sns.relplot(data, x=input.stat_var(), y="stroke", kind="line")
        return p.set(xlabel=None)

    ui.input_select("stat_var", "Select variable", choices=["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","bmi","smoking_status"])

with ui.nav_panel("Prediction"):
    with ui.layout_columns(col_widths=(8, 4)):
        with ui.card():
            with ui.layout_column_wrap(width=1 / 2):
                with ui.card():
                    ui.input_select(
                        "gender",
                        "Gender",
                        {"Male": "Male", "Female": "Female"},
                    )
                    ui.input_numeric("age", "Age", 1, min=0, max=100)
                    ui.input_select(
                        "hypertension",
                        "Hypertension",
                        {0: "No", 1: "Yes"},
                    )
                    ui.input_select(
                        "heart_disease",
                        "Heart disease",
                        {0: "No", 1: "Yes"},
                    )
                    ui.input_select(
                        "ever_married",
                        "Ever married",
                        {"No": "No", "Yes": "Yes"},
                    )
                with ui.card():
                    ui.input_select(
                        "work_type",
                        "Work type",
                        {'Never_worked': 'Never worked', 'Private': 'Private', 'Self-employed': 'Self-employed', 'Govt_job': 'Govt job', 'Children': 'Children'},
                    )
                    ui.input_select(
                        "Residence_type",
                        "Residence type",
                        {'Rural': 'Rural', 'Urban': 'Urban'},
                    )
                    ui.input_numeric("bmi", "BMI", 25, min=0, max=100)
                    ui.input_select(
                        "smoking_status",
                        "Smoking status",
                        {'formerly smoked': 'Formerly smoked', 'never smoked': 'Never smoked', 'smokes': 'Smokes'},
                    )
        with ui.card():
            ui.input_action_button("action_button", "Predict")  

            @render.text()
            @reactive.event(input.action_button)
            def counter():
                new_data = pd.DataFrame([[input.gender(), float(input.age()), int(input.hypertension()), int(input.heart_disease()), input.ever_married(), input.work_type(), input.Residence_type(), float(input.bmi()), input.smoking_status()]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'bmi', 'smoking_status'])
                predicted = predict(new_data)
                return f"Predicted: {'Stroke' if predicted >= 1 else 'No stroke'}"
