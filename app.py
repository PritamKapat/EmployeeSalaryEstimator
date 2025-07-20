import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load all models
models = {
    "Linear Regression": joblib.load("model_files/LinearRegression_model.pkl"),
    "Random Forest": joblib.load("model_files/RandomForest_model.pkl"),
    "GradientBoost": joblib.load("model_files/GradientBoosting_model.pkl"),
    "KNN":  joblib.load("model_files/KNN_model.pkl"),
}

city_mapping = {
    "Bangalore": 0,
    "Delhi": 1,
    "Pune": 2
}

Edu_mapping = {
    "Bachelors": 0,
    "Masters": 1,
    "PhD": 2
}
role_mapping = {
    "Business Analyst": 0,
    "Data Analyst": 1,
    "Full Stack Developer": 2,
    "QA Tester": 3,
    "Research Scientist": 4,
    "Software Engineer": 5,
    "Support Engineer": 6,
}
role_bonus_percent = {
    "Business Analyst": 5,     # 5%
    "Data Analyst": 4,
    "Full Stack Developer": 10,
    "QA Tester": 3,
    "Research Scientist": 12,
    "Software Engineer": 8,
    "Support Engineer": 4,
}
results = {
    "RF": 0.9980,
    "LR": 0.9966,
    "KNN": 0.9862,
    "GB": 0.9966 
}
st.info("You can change model from the sidebar")

st.title("💼 Compensation Prediction for Employees")

# Sidebar model selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))


# Input form
st.header("Enter Employee Details")

selected_edu_name = st.selectbox("Education", list(Edu_mapping.keys()))
Edu_code = Edu_mapping[selected_edu_name]

age = st.slider("Age", 18, 60)

years = list(range(2012, 2018))
selected_year = st.selectbox("Select your joining year", years)

selected_city_name = st.segmented_control(
    "Select City",
    options=list(city_mapping.keys()),
    default="Bangalore", 
    help="Choose your city from the options"
)


city_code = city_mapping[selected_city_name]

tier = st.pills("Payment Tier", [1, 2, 3])

gender = st.selectbox("Gender", ["Male", "Female"])
benched= st.checkbox("Ever Benched", value=False)

experience = st.number_input("Experience in Domain (years)", min_value=0, max_value=7)
leave_or_not = st.checkbox("Have taken Leaves?", value=False)

selected_role = st.selectbox(
    "Select Role",
    options=list(role_mapping.keys()) 
) 
role_code = role_mapping[selected_role]

gender_val = 1 if gender == "Male" else 0
benched = 1 if benched else 0  
leave_val = 1 if leave_or_not else 0 

if st.button("Predict Salary"):
    if tier is None or experience is None:
        st.warning("⚠️ Please fill in all the fields before predicting.")
    else:
        input_data = np.array([[Edu_code, selected_year, city_code, tier, age,
                                gender_val, benched, experience, leave_val, role_code]])

        model = models[model_choice]
        base_salary = model.predict(input_data)[0]

        
        bonus_percent = role_bonus_percent[selected_role]
        bonus_amount = base_salary * (bonus_percent / 100)
        adjusted_salary = base_salary + bonus_amount

        st.success(f"💰 Predicted Base Salary: ₹{base_salary:.2f}")
        st.info(f"🎁 Role-Based Bonus ({bonus_percent}%): ₹{bonus_amount:.2f}")
        st.success(f"💼 Final Adjusted Salary: ₹{adjusted_salary:.2f}")





#plot   
with st.sidebar:
    show_plot = st.button("Compare Model")


    if show_plot:
        
        fig = go.Figure(
            data=[go.Bar(
                x=list(results.keys()),
                y=[0]*len(results),  
                marker_color='red'
            )],
            layout=go.Layout(
                yaxis=dict(range=[0.8, 1.0]),
                title='Accuracy Scores'
            )
        )

        
        frames = []
        for step in np.linspace(0, 1, 40):
            frames.append(go.Frame(
                data=[go.Bar(
                    x=list(results.keys()),
                    y=[val * step for val in results.values()]
                )]
            ))

        fig.frames = frames

        # Play button
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Click",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 80, "redraw": True}}],
                }],
                "x": 0.3,
                "y": -0.2,
                "font": {"color": "black"},     # ✅ set text color here
                "bgcolor": "black",             # optional: button background
                "bordercolor": "gray",          # optional: border color
                "showactive": True              # optional: highlight selected
            }],
            modebar=dict(remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'lasso2d'])
        )




        # Show chart in main area
        st.plotly_chart(fig, use_container_width=True)
