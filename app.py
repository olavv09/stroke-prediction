import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

# Import data from shared.py
from shared import df
from prediction import data, predict
from shiny import reactive
from shiny.express import input, render, ui
from shiny.ui import page_navbar

# Page title (with some additional top padding)
ui.page_opts(
    title="Aleksandra Wojcik",
    page_fn=partial(page_navbar, id="page"),
)

with ui.nav_panel("Distributions"):
    @render.plot
    def dist():
        p = sns.histplot(df, x=input.dist_var(), hue="stroke", multiple="stack")
        return p.set(xlabel=None)

    ui.input_select("dist_var", "Select variable", choices=["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","bmi","smoking_status"])

with ui.nav_panel("Statistical relationships"):
    @render.plot
    def stat():
        p = sns.relplot(df, x=input.stat_var(), y="stroke", kind="line")
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
                        {0: "Male", 1: "Female"},
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
                        {0: "No", 1: "Yes"},
                    )
                with ui.card():
                    ui.input_select(
                        "work_type",
                        "Work type",
                        {0: 'Never worked', 1: 'Private', 2: 'Self-employed', 3: 'Govt job', 4: 'Children'},
                    )
                    ui.input_select(
                        "Residence_type",
                        "Residence type",
                        {0: 'Rural', 1: 'Urban'},
                    )
                    ui.input_numeric("bmi", "BMI", 25, min=0, max=100)
                    ui.input_select(
                        "smoking_status",
                        "Smoking status",
                        {1: 'Formerly smoked', 2: 'Never smoked', 3: 'Smokes'},
                    )
        with ui.card():
            ui.input_action_button("action_button", "Predict")  

            @render.text()
            @reactive.event(input.action_button)
            def counter():
                new_data = pd.DataFrame([[int(input.gender()), int(input.age()), int(input.hypertension()), int(input.heart_disease()), int(input.ever_married()), int(input.work_type()), int(input.Residence_type()), float(input.bmi()), int(input.smoking_status())]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'bmi', 'smoking_status'])
                predicted = 'stroke' if predict(new_data) > 0.5 else 'no stroke'
                return (f"Prediction #{input.action_button()}: {predicted}")