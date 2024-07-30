import pandas as pd
from sklearn.preprocessing import StandardScaler

from anon import anonymization
from file_utils import get_file


def main():
    data = get_file()
    if data is not None:
        column_names = data.columns

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)

        anonymized_data = anonymization(normalized_data)
        anonymized_df = pd.DataFrame(anonymized_data, columns=column_names)
        # Save the data anonimized in CSV
        anonymized_df.to_csv('anonymized_data.csv', index=False)
        print("Saved data in 'anonymized_data.csv'.")
    else:
        print("Failed to load data.")
   
if __name__ == "__main__":
    main()
