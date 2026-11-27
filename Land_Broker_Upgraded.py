# ============================================================
# LAND BROKER: LAND PRICE PREDICTION USING MACHINE LEARNING
# B.Tech CSE - 3rd Semester
# Algorithm: Random Forest Regression
# Dataset: land_data.csv
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

if os.path.exists("land_data.csv"):
    file_path = "land_data.csv"
elif os.path.exists("land_data(1).csv"):
    file_path = "land_data(1).csv"
else:
    raise FileNotFoundError(
        "Dataset not found! Place land_data.csv in the same folder "
        "as this Python file."
    )

df = pd.read_csv(file_path)

print("\n========================================")
print("       LAND BROKER ML PROJECT")
print("========================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK AND CLEAN THE DATASET
# ============================================================

print("\nMissing values in each column:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Check required columns
required_columns = [
    "city",
    "locality",
    "land_type",
    "area_sqft",
    "road_width_ft",
    "distance_from_center_km",
    "near_highway_or_metro",
    "water_electricity",
    "corner_plot",
    "price_lakh"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Convert numerical columns to numeric values
numeric_columns = [
    "area_sqft",
    "road_width_ft",
    "distance_from_center_km",
    "near_highway_or_metro",
    "water_electricity",
    "corner_plot",
    "price_lakh"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

# Remove records with invalid land area or price
df = df[
    (df["area_sqft"] > 0) &
    (df["price_lakh"] >= 0)
].copy()

if len(df) < 2:
    raise ValueError("Not enough valid records to train the model.")

# Clean categorical values
categorical_columns = ["city", "locality", "land_type"]

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()

print("\nCleaned dataset shape:", df.shape)


# ============================================================
# 3. LOCATION-BASED ANALYSIS
# ============================================================

print("\n========================================")
print("       LOCATION-BASED ANALYSIS")
print("========================================")

# Average price by city
city_analysis = df.groupby("city")["price_lakh"].agg(
    ["count", "mean", "min", "max"]
).round(2)

city_analysis.columns = [
    "Number_of_Lands",
    "Average_Price_Lakh",
    "Minimum_Price_Lakh",
    "Maximum_Price_Lakh"
]

print("\nAverage land price by city:")
print(city_analysis.sort_values(
    by="Average_Price_Lakh",
    ascending=False
))

# Average price by locality
locality_analysis = df.groupby(
    ["city", "locality"]
)["price_lakh"].agg(
    ["count", "mean", "min", "max"]
).round(2)

locality_analysis.columns = [
    "Number_of_Lands",
    "Average_Price_Lakh",
    "Minimum_Price_Lakh",
    "Maximum_Price_Lakh"
]

print("\nAverage land price by locality:")
print(locality_analysis.sort_values(
    by="Average_Price_Lakh",
    ascending=False
))

# Average price by land type
print("\nAverage price by land type:")
print(
    df.groupby("land_type")["price_lakh"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

# Save location analysis to CSV files
city_analysis.to_csv("city_price_analysis.csv")
locality_analysis.to_csv("locality_price_analysis.csv")

print("\nLocation analysis saved successfully!")


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

# Input features
features = [
    "city",
    "locality",
    "land_type",
    "area_sqft",
    "road_width_ft",
    "distance_from_center_km",
    "near_highway_or_metro",
    "water_electricity",
    "corner_plot"
]

# Target variable
target = "price_lakh"

X = df[features]
y = df[target]

categorical_features = [
    "city",
    "locality",
    "land_type"
]

numerical_features = [
    "area_sqft",
    "road_width_ft",
    "distance_from_center_km",
    "near_highway_or_metro",
    "water_electricity",
    "corner_plot"
]

print("\nInput features:")
print(features)

print("\nTarget variable:", target)


# ============================================================
# 5. SPLIT THE DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. PREPROCESS CATEGORICAL DATA
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. CREATE THE RANDOM FOREST MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

print("\n========================================")
print("          TRAINING THE MODEL")
print("========================================")

model.fit(X_train, y_train)

print("Model trained successfully!")


# ============================================================
# 9. EVALUATE THE MODEL
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========================================")
print("          MODEL EVALUATION")
print("========================================")

print(f"Mean Absolute Error (MAE): {mae:.2f} lakh")
print(f"Mean Squared Error (MSE):  {mse:.2f}")
print(f"Root Mean Squared Error:   {rmse:.2f} lakh")
print(f"R2 Score:                  {r2:.4f}")

print("\nInterpretation:")
print(f"Average absolute error: approximately Rs. {mae * 100000:,.0f}")

print(
    "R2 measures how much of the variation in test-set prices "
    "is explained by the model."
)


# ============================================================
# 10. SAVE THE TRAINED MODEL
# ============================================================

joblib.dump(model, "land_price_model.joblib")

print("\nTrained model saved as land_price_model.joblib")


# ============================================================

# ============================================================
# 11. UPGRADED GUI - LAND BROKER (TKINTER)
# Features: Dashboard, prediction form, analytics, history, CSV export
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

# Store every prediction made during the current session.
prediction_history = []

# --------------------------
# Main window and theme
# --------------------------
root = tk.Tk()
root.title("Land Broker | Smart Land Price Estimator")
root.geometry("1100x780")
root.minsize(950, 680)
root.configure(bg="#eef2f7")

NAVY = "#172554"
BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"
BG = "#eef2f7"
WHITE = "#ffffff"
TEXT = "#1e293b"
MUTED = "#64748b"
GREEN = "#15803d"
RED = "#b91c1c"

style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass
style.configure("TNotebook", background=BG, borderwidth=0)
style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 10, "bold"), background="#e2e8f0")
style.map("TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", BLUE)])
style.configure("TCombobox", padding=7, font=("Segoe UI", 10))
style.configure("Treeview", font=("Segoe UI", 9), rowheight=28, background=WHITE, fieldbackground=WHITE)
style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#e2e8f0")
style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)

# Header
header = tk.Frame(root, bg=NAVY, height=92)
header.pack(fill="x")
header.pack_propagate(False)
header_inner = tk.Frame(header, bg=NAVY)
header_inner.pack(fill="both", expand=True, padx=28, pady=13)
tk.Label(header_inner, text="LAND BROKER", bg=NAVY, fg=WHITE,
         font=("Segoe UI", 23, "bold")).pack(anchor="w")
tk.Label(header_inner, text="Machine Learning Based Land Price Prediction System",
         bg=NAVY, fg="#bfdbfe", font=("Segoe UI", 10)).pack(anchor="w")

# Notebook / pages
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=20, pady=(16, 8))
prediction_tab = tk.Frame(notebook, bg=BG)
analytics_tab = tk.Frame(notebook, bg=BG)
history_tab = tk.Frame(notebook, bg=BG)
notebook.add(prediction_tab, text="  Predict Price  ")
notebook.add(analytics_tab, text="  Location Analysis  ")
notebook.add(history_tab, text="  Prediction History  ")

# --------------------------
# Reusable UI helpers
# --------------------------
def card(parent, padx=18, pady=15):
    return tk.Frame(parent, bg=WHITE, padx=padx, pady=pady,
                    highlightbackground="#dbe2ea", highlightthickness=1)

def field_label(parent, text, row, column=0):
    tk.Label(parent, text=text, bg=WHITE, fg=TEXT,
             font=("Segoe UI", 9, "bold")).grid(
        row=row, column=column, sticky="w", padx=(0, 14), pady=9
    )

def make_entry(parent, row, column=1):
    entry = ttk.Entry(parent, font=("Segoe UI", 10), width=28)
    entry.grid(row=row, column=column, sticky="ew", pady=7, ipady=4)
    return entry

def make_combo(parent, variable, values, row, column=1):
    combo = ttk.Combobox(parent, textvariable=variable, values=list(values),
                         state="readonly", font=("Segoe UI", 10), width=26)
    combo.grid(row=row, column=column, sticky="ew", pady=7, ipady=3)
    return combo

def format_inr_lakh(value):
    return f"₹{value:,.2f} lakh"

# --------------------------
# Prediction page
# --------------------------
prediction_tab.grid_columnconfigure(0, weight=3)
prediction_tab.grid_columnconfigure(1, weight=2)
prediction_tab.grid_rowconfigure(0, weight=1)

form_card = card(prediction_tab, padx=24, pady=20)
form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
result_card = card(prediction_tab, padx=22, pady=20)
result_card.grid(row=0, column=1, sticky="new", padx=(10, 0), pady=5)

tk.Label(form_card, text="Property Details", bg=WHITE, fg=NAVY,
         font=("Segoe UI", 16, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
tk.Label(form_card, text="Enter the property information to estimate its market price.",
         bg=WHITE, fg=MUTED, font=("Segoe UI", 9)).grid(
    row=1, column=0, columnspan=2, sticky="w", pady=(0, 12)
)

city_var = tk.StringVar()
locality_var = tk.StringVar()
land_type_var = tk.StringVar()
metro_var = tk.StringVar(value="No")
water_var = tk.StringVar(value="Yes")
corner_var = tk.StringVar(value="No")

cities = sorted(df["city"].astype(str).unique())
land_types = sorted(df["land_type"].astype(str).unique())

field_label(form_card, "City", 2)
city_combo = make_combo(form_card, city_var, cities, 2)
field_label(form_card, "Locality", 3)
locality_combo = make_combo(form_card, locality_var, [], 3)
field_label(form_card, "Land Type", 4)
land_type_combo = make_combo(form_card, land_type_var, land_types, 4)
field_label(form_card, "Land Area (sq. ft.)", 5)
area_entry = make_entry(form_card, 5)
field_label(form_card, "Road Width (feet)", 6)
road_entry = make_entry(form_card, 6)
field_label(form_card, "Distance from Centre (km)", 7)
distance_entry = make_entry(form_card, 7)
field_label(form_card, "Highway / Metro Nearby", 8)
metro_combo = make_combo(form_card, metro_var, ["Yes", "No"], 8)
field_label(form_card, "Water & Electricity", 9)
water_combo = make_combo(form_card, water_var, ["Yes", "No"], 9)
field_label(form_card, "Corner Plot", 10)
corner_combo = make_combo(form_card, corner_var, ["Yes", "No"], 10)
form_card.grid_columnconfigure(1, weight=1)


def update_localities(event=None):
    selected_city = city_var.get()
    choices = sorted(df.loc[df["city"] == selected_city, "locality"].astype(str).unique())
    locality_combo["values"] = choices
    locality_var.set(choices[0] if choices else "")

city_combo.bind("<<ComboboxSelected>>", update_localities)
if cities:
    city_var.set(cities[0])
    update_localities()
if land_types:
    land_type_var.set(land_types[0])

# Result panel
tk.Label(result_card, text="PREDICTION RESULT", bg=WHITE, fg=NAVY,
         font=("Segoe UI", 14, "bold")).pack(anchor="w")
tk.Label(result_card, text="Your estimated land value will appear here.",
         bg=WHITE, fg=MUTED, font=("Segoe UI", 9), wraplength=260,
         justify="left").pack(anchor="w", pady=(5, 18))
result_price = tk.Label(result_card, text="₹ --", bg=LIGHT_BLUE, fg=NAVY,
                        font=("Segoe UI", 21, "bold"), padx=12, pady=18,
                        wraplength=270, justify="center")
result_price.pack(fill="x", pady=(0, 10))
result_lakh = tk.Label(result_card, text="Price in lakh: --", bg=WHITE, fg=TEXT,
                       font=("Segoe UI", 11, "bold"))
result_lakh.pack(anchor="w", pady=5)
result_crore = tk.Label(result_card, text="Price in crore: --", bg=WHITE, fg=TEXT,
                        font=("Segoe UI", 11, "bold"))
result_crore.pack(anchor="w", pady=5)
result_sqft = tk.Label(result_card, text="Approx. price / sq. ft.: --", bg=WHITE,
                       fg=TEXT, font=("Segoe UI", 10))
result_sqft.pack(anchor="w", pady=5)
result_status = tk.Label(result_card, text="Waiting for property details", bg=WHITE,
                         fg=MUTED, font=("Segoe UI", 9), wraplength=270)
result_status.pack(anchor="w", pady=(14, 8))


def reset_form():
    for entry in (area_entry, road_entry, distance_entry):
        entry.delete(0, tk.END)
    metro_var.set("No")
    water_var.set("Yes")
    corner_var.set("No")
    if cities:
        city_var.set(cities[0])
        update_localities()
    if land_types:
        land_type_var.set(land_types[0])
    result_price.config(text="₹ --")
    result_lakh.config(text="Price in lakh: --")
    result_crore.config(text="Price in crore: --")
    result_sqft.config(text="Approx. price / sq. ft.: --")
    result_status.config(text="Form cleared. Enter new property details.", fg=MUTED)


def predict_price():
    try:
        selected_city = city_var.get().strip()
        selected_locality = locality_var.get().strip()
        selected_land_type = land_type_var.get().strip()
        if not all([selected_city, selected_locality, selected_land_type]):
            raise ValueError("Please select city, locality and land type.")

        try:
            area = float(area_entry.get().strip())
            road_width = float(road_entry.get().strip())
            distance = float(distance_entry.get().strip())
        except ValueError:
            raise ValueError("Area, road width and distance must be valid numbers.")

        if not np.isfinite(area) or area <= 0:
            raise ValueError("Land area must be greater than zero.")
        if not np.isfinite(road_width) or road_width < 0:
            raise ValueError("Road width cannot be negative.")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Distance cannot be negative.")

        values = {
            "city": selected_city,
            "locality": selected_locality,
            "land_type": selected_land_type,
            "area_sqft": area,
            "road_width_ft": road_width,
            "distance_from_center_km": distance,
            "near_highway_or_metro": 1 if metro_var.get() == "Yes" else 0,
            "water_electricity": 1 if water_var.get() == "Yes" else 0,
            "corner_plot": 1 if corner_var.get() == "Yes" else 0
        }
        new_land = pd.DataFrame([values], columns=features)
        predicted_lakh = max(0.0, float(model.predict(new_land)[0]))
        predicted_rupees = predicted_lakh * 100000
        per_sqft = predicted_rupees / area

        result_price.config(text=f"₹{predicted_rupees:,.0f}")
        result_lakh.config(text=f"Price in lakh: ₹{predicted_lakh:,.2f}")
        result_crore.config(text=f"Price in crore: ₹{predicted_lakh / 100:,.3f}")
        result_sqft.config(text=f"Approx. price / sq. ft.: ₹{per_sqft:,.2f}")
        result_status.config(text="Prediction generated successfully. This is an estimate, not a formal valuation.", fg=GREEN)

        prediction_history.append({
            "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "City": selected_city,
            "Locality": selected_locality,
            "Land_Type": selected_land_type,
            "Area_sqft": area,
            "Predicted_Price_Lakh": round(predicted_lakh, 2),
            "Predicted_Price_Rupees": round(predicted_rupees, 2)
        })
        refresh_history()

    except ValueError as error:
        messagebox.showerror("Check Input", str(error))
    except Exception as error:
        messagebox.showerror("Prediction Error", f"Unable to generate prediction.\n{error}")

button_row = tk.Frame(form_card, bg=WHITE)
button_row.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(18, 2))
tk.Button(button_row, text="PREDICT PRICE", command=predict_price,
          bg=BLUE, fg=WHITE, activebackground="#1d4ed8", activeforeground=WHITE,
          relief="flat", cursor="hand2", font=("Segoe UI", 10, "bold"),
          padx=20, pady=11).pack(side="left", padx=(0, 10))
tk.Button(button_row, text="RESET", command=reset_form,
          bg="#e2e8f0", fg=TEXT, activebackground="#cbd5e1",
          relief="flat", cursor="hand2", font=("Segoe UI", 10, "bold"),
          padx=20, pady=11).pack(side="left")

# --------------------------
# Analytics page
# --------------------------
analytics_tab.grid_columnconfigure(0, weight=1)
analytics_tab.grid_rowconfigure(1, weight=1)
summary = card(analytics_tab)
summary.grid(row=0, column=0, sticky="ew", pady=5)
tk.Label(summary, text="Dataset Overview", bg=WHITE, fg=NAVY,
         font=("Segoe UI", 15, "bold")).pack(anchor="w")
tk.Label(summary,
         text=f"Valid records: {len(df):,}     |     Cities: {df['city'].nunique()}     |     Localities: {df['locality'].nunique()}     |     Land types: {df['land_type'].nunique()}",
         bg=WHITE, fg=TEXT, font=("Segoe UI", 10)).pack(anchor="w", pady=(8, 0))

analysis_card = card(analytics_tab)
analysis_card.grid(row=1, column=0, sticky="nsew", pady=(10, 5))
analysis_card.grid_rowconfigure(1, weight=1)
analysis_card.grid_columnconfigure(0, weight=1)
tk.Label(analysis_card, text="Average Land Price by City (in lakh)", bg=WHITE,
         fg=NAVY, font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
city_tree = ttk.Treeview(analysis_card,
                         columns=("city", "count", "average", "minimum", "maximum"),
                         show="headings", height=10)
for col, title, width in [
    ("city", "City", 220), ("count", "Properties", 110),
    ("average", "Average (Lakh)", 150), ("minimum", "Minimum (Lakh)", 150),
    ("maximum", "Maximum (Lakh)", 150)
]:
    city_tree.heading(col, text=title)
    city_tree.column(col, width=width, anchor="center")
city_tree.grid(row=1, column=0, sticky="nsew")
city_scroll = ttk.Scrollbar(analysis_card, orient="vertical", command=city_tree.yview)
city_tree.configure(yscrollcommand=city_scroll.set)
city_scroll.grid(row=1, column=1, sticky="ns")
for city_name, row in city_analysis.sort_values("Average_Price_Lakh", ascending=False).iterrows():
    city_tree.insert("", "end", values=(city_name, int(row["Number_of_Lands"]),
                     f"{row['Average_Price_Lakh']:.2f}", f"{row['Minimum_Price_Lakh']:.2f}",
                     f"{row['Maximum_Price_Lakh']:.2f}"))

# --------------------------
# History page
# --------------------------
history_tab.grid_columnconfigure(0, weight=1)
history_tab.grid_rowconfigure(1, weight=1)
history_top = card(history_tab)
history_top.grid(row=0, column=0, sticky="ew", pady=5)
tk.Label(history_top, text="Your Prediction History", bg=WHITE, fg=NAVY,
         font=("Segoe UI", 15, "bold")).pack(side="left")

history_table_card = card(history_tab)
history_table_card.grid(row=1, column=0, sticky="nsew", pady=(10, 5))
history_table_card.grid_rowconfigure(0, weight=1)
history_table_card.grid_columnconfigure(0, weight=1)
history_tree = ttk.Treeview(history_table_card,
    columns=("date", "city", "locality", "type", "area", "price"), show="headings")
for col, title, width in [
    ("date", "Date / Time", 155), ("city", "City", 130),
    ("locality", "Locality", 150), ("type", "Land Type", 120),
    ("area", "Area (sq. ft.)", 120), ("price", "Predicted Price (Lakh)", 170)
]:
    history_tree.heading(col, text=title)
    history_tree.column(col, width=width, anchor="center")
history_tree.grid(row=0, column=0, sticky="nsew")
history_scroll = ttk.Scrollbar(history_table_card, orient="vertical", command=history_tree.yview)
history_tree.configure(yscrollcommand=history_scroll.set)
history_scroll.grid(row=0, column=1, sticky="ns")


def refresh_history():
    for item in history_tree.get_children():
        history_tree.delete(item)
    for record in reversed(prediction_history):
        history_tree.insert("", "end", values=(
            record["Date_Time"], record["City"], record["Locality"],
            record["Land_Type"], f"{record['Area_sqft']:,.0f}",
            f"{record['Predicted_Price_Lakh']:,.2f}"
        ))


def export_history():
    if not prediction_history:
        messagebox.showinfo("No History", "Generate at least one prediction before exporting.")
        return
    destination = filedialog.asksaveasfilename(
        title="Save Prediction History", defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")], initialfile="land_prediction_history.csv"
    )
    if destination:
        pd.DataFrame(prediction_history).to_csv(destination, index=False)
        messagebox.showinfo("Export Complete", f"Prediction history saved to:\n{destination}")

history_buttons = tk.Frame(history_top, bg=WHITE)
history_buttons.pack(side="right")
tk.Button(history_buttons, text="EXPORT CSV", command=export_history,
          bg=BLUE, fg=WHITE, relief="flat", cursor="hand2",
          font=("Segoe UI", 9, "bold"), padx=12, pady=7).pack(side="left", padx=4)
tk.Button(history_buttons, text="CLEAR HISTORY",
          command=lambda: (prediction_history.clear(), refresh_history()),
          bg="#fee2e2", fg=RED, relief="flat", cursor="hand2",
          font=("Segoe UI", 9, "bold"), padx=12, pady=7).pack(side="left", padx=4)

# Footer
footer = tk.Label(root,
    text=f"Random Forest Regression  |  Test MAE: {mae:.2f} lakh  |  R² Score: {r2:.3f}  |  Dataset: {os.path.basename(file_path)}",
    bg=BG, fg=MUTED, font=("Segoe UI", 8))
footer.pack(fill="x", padx=22, pady=(0, 10))

root.mainloop()
