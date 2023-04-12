import sys
import pandas as pd

# Get the command-line arguments
patient_id = sys.argv[1]
systolic_bp = float(sys.argv[2])
diastolic_bp = float(sys.argv[3])
total_cholesterol = float(sys.argv[4])
triglycerides = float(sys.argv[5])

# Load the dataset
df = pd.read_csv("Dataset_HD.csv")

# Define the rules for each attribute


def systolic_bp_rule(value):
    if value >= 140:
        return 1
    elif value <= 90:
        return 1
    else:
        return 0


def diastolic_bp_rule(value):
    if value >= 90:
        return 1
    elif value <= 60:
        return 1
    else:
        return 0


def total_cholesterol_rule(value):
    if value >= 240:
        return 1
    elif value <= 200:
        return 1
    else:
        return 0


def triglycerides_rule(value):
    if value >= 200:
        return 1
    elif value <= 150:
        return 1
    else:
        return 0


# Create a new row for the new patient
new_row = {"Patient ID": patient_id,
           "Systolic BP": systolic_bp,
           "Diastolic BP": diastolic_bp,
           "Total Cholesterol": total_cholesterol,
           "Triglycerides": triglycerides,
           "CVD Diagnosis": None}

# Add the new row to the DataFrame
df = df.append(new_row, ignore_index=True)

# Apply the rules to each row in the dataset
df["CVD Diagnosis"] = df.apply(lambda row:
                               systolic_bp_rule(row["Systolic BP"])
                               if pd.isna(row["CVD Diagnosis"])
                               else row["CVD Diagnosis"], axis=1)

df["CVD Diagnosis"] = df.apply(lambda row:
                               diastolic_bp_rule(row["Diastolic BP"])
                               if pd.isna(row["CVD Diagnosis"])
                               else row["CVD Diagnosis"], axis=1)

df["CVD Diagnosis"] = df.apply(lambda row:
                               total_cholesterol_rule(row["Total Cholesterol"])
                               if pd.isna(row["CVD Diagnosis"])
                               else row["CVD Diagnosis"], axis=1)

df["CVD Diagnosis"] = df.apply(lambda row:
                               triglycerides_rule(row["Triglycerides"])
                               if pd.isna(row["CVD Diagnosis"])
                               else row["CVD Diagnosis"], axis=1)

# Save the updated dataset
df.to_csv("cvd_dataset_with_rules_binary.csv", index=False)
# Get the CVD diagnosis for the new user
new_user_diagnosis = df[df["Patient ID"] ==
                        patient_id]["CVD Diagnosis"].values[0]
print(
    f"CVD diagnosis for new user with patient ID {patient_id}: {new_user_diagnosis}")
