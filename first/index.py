import pandas as pd


csv_file = r"C:\Users\krish\OneDrive\Desktop\python\ml\sample_data.csv"


data_csv = pd.read_csv(csv_file)
print("CSV file data: ")
print(data_csv)

print("\nData Dscriptoin")
print("CSV data Dscriptoin")
print(data_csv.describe())

