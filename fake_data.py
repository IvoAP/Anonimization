import os
import random

import pandas as pd
from faker import Faker

fake = Faker('pt_BR')

def generate_fake_data(num_rows):
    data = {
        "idade": [],
        "cpf": [],
        "rg": [],
        "altura": [],
        "peso": []
    }

    for _ in range(num_rows):
        data["idade"].append(random.randint(18, 90))
        data["cpf"].append(fake.cpf().replace('.', '').replace('-', ''))
        data["rg"].append(random.randint(10000000, 99999999))  # RG is ofthe a number of eight digits
        data["altura"].append(random.randint(150, 200))  # Height em cm
        data["peso"].append(random.randint(50, 100))  # Weight em kg

    return pd.DataFrame(data)

def main():
    num_rows = 50000
    df = generate_fake_data(num_rows)

    # Create the dict if it's not exist
    output_dir = 'datas'
    os.makedirs(output_dir, exist_ok=True)

    # name of the file
    file_name = f'fake_csv_{num_rows}.csv'
    file_path = os.path.join(output_dir, file_name)

    # Salvar em um arquivo CSV
    df.to_csv(file_path, index=False)
    print(f'Dataset save in: {file_path}')

if __name__ == "__main__":
    main()

