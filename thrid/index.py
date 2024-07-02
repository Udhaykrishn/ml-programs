import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


data = {
    'Age':[25,30,None,28,35],
    "Gender":["Female","Male","Male","Female","Male"],
    "Income":[50000,60000,45000,None,70000]
}

df  = pd.DataFrame(data)

imputer = SimpleImputer(strategy="mean")
df[["Age","Incom"]] = imputer.fit_transform(df[["Age","Income"]])

print("Data After Handling Missing values: ")
print(df)

encoder = OneHotEncoder()
encoder_data = encoder.fit_transform(df[["Gender"]]).toarray()

encoded_df = pd.DataFrame(encoder_data,columns=encoder.get_feature_names_out(["Gender"]))
print("\nData After categorical encoding: ")
print(encoded_df)

scaler = StandardScaler()
scaler_data = scaler.fit_transform(df[["Age","Income"]])

scaled_df = pd.DataFrame(scaler_data,columns=["Scaled Age","Scaled Income"])
print("\nData After feature scaling: ")
print(scaled_df)