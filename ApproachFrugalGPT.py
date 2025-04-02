# Import necessary libraries
# pip install azure-ai-inference
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, dotenv_values
import sys
import pandas as pd
from IPython.display import display
import numpy
from tqdm import tqdm
import shutil
import copy

# Add the source directory to the system path
sys.path.append("src/")
import FrugalGPT

# Load environment variables from .env file
load_dotenv()

# Get the list of supported LLM services
supported_LLM = FrugalGPT.getservicename()

# Function to convert a list of lists into a pandas DataFrame
def list_to_dataframe(data_list):
    # The first sublist is the header
    headers = data_list[0]
    # The rest are the data rows
    data = data_list[1:]
    # Create the dataframe
    df = pd.DataFrame(data, columns=headers)
    return df

# Function to convert and merge train and test DataFrames
def convert_and_merge_dataframes(train_df, test_df):
    def extract_last_query_part(query):
        # Extract the last part of the query after splitting by '\n\n'
        return query.split('\n\n')[-1]

    def create_converted_df(df, start_query_id=1):
        # Extract the new 'query' and keep 'ref_answer' the same
        df['new_query'] = df['query'].apply(extract_last_query_part)

        # Group by 'new_query' and 'ref_answer' to merge identical queries
        merged_df = df.groupby(['new_query', 'ref_answer'], as_index=False).first()

        # Create a new dataframe with the three columns
        converted_df = pd.DataFrame({
            'query': merged_df['new_query'],
            'ref_answer': merged_df['ref_answer'],
            'query_id': range(start_query_id, start_query_id + len(merged_df))
        })

        return converted_df

    # Convert and merge the train dataframe
    converted_train_df = create_converted_df(train_df)

    # Find the last query_id from the training data
    last_train_query_id = converted_train_df['query_id'].max()

    # Convert and merge the test dataframe, starting query_id from the last training id + 1
    converted_test_df = create_converted_df(test_df, start_query_id=last_train_query_id + 1)

    return converted_train_df, converted_test_df  

# Load raw train and test data
train_raw = FrugalGPT.loadcsvdata("data\\AGNEWS\\AGNEWS_train.csv")
test_raw = FrugalGPT.loadcsvdata("data\\AGNEWS\\AGNEWS_test.csv")

# Convert raw data to DataFrames
train_df = list_to_dataframe(train_raw)
test_df = list_to_dataframe(test_raw)

# Convert and merge train and test DataFrames
converted_train, converted_test = convert_and_merge_dataframes(train_df, test_df)

# Save the converted DataFrames to CSV files
columns_to_save = ['query', 'ref_answer', 'query_id']
converted_train[columns_to_save].to_csv("data\\AGNEWS\\train.csv", index=False, header=False)
converted_test[columns_to_save].to_csv("data\\AGNEWS\\test.csv", index=False, header=False)

# Function to generate a DataFrame with model performance metrics
def generate_dataframe(service_names, train_data, test_data, genparams, db_path="db\\AGNEWS.sqlite", max_workers=2):
    # Initialize an empty list to store the rows for the DataFrame
    data = []
    MyLLMforAll = FrugalGPT.LLMforAll(
        db_path=db_path,
        max_workers=max_workers,
    )
    # Dictionary to keep track of markers for each provider
    provider_marker = {}

    # Iterate through the service names
    for name in service_names:
        # Extract provider and method
        provider = name.split('/')[0]
        method = name.split('/')[-1]

        # If the provider is seen for the first time, initialize its marker
        if provider not in provider_marker:
            provider_marker[provider] = 1
        else:
            provider_marker[provider] += 1

        # Get the completion batch for train and test data
        r1_train = MyLLMforAll.get_completion_batch(queries=train_data, genparams=genparams, service_name=name)
        r2_train = FrugalGPT.compute_score(r1_train)
        r1_test = MyLLMforAll.get_completion_batch(queries=test_data, genparams=genparams, service_name=name)
        r2_test = FrugalGPT.compute_score(r1_test)

        # Extract accuracy and cost
        train_acc = r2_train['em']
        train_cost = r2_train['cost']
        test_acc = r2_test['em']
        test_cost = r2_test['cost']

        # Create a row with the schema
        row = {
            "Test_acc": test_acc,
            "Test_cost": test_cost,
            "Test_size": len(test_data),
            "Train_acc": train_acc,
            "Train_cost": train_cost,
            "Train_size": len(train_data),
            "Budget": 10,
            "Method": method,
            "Provider": provider,
            "Marker": provider_marker[provider],
        }

        # Append the row to the data list
        data.append(row)

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df

# Define dataset name and service names
dataname = "AGNEWS"
service_names = [
    'azure_openai/gpt-3.5-turbo',
    'azure_llama/llama-3-70b-instruct',
    'azure_gpt4o/gpt-4o',
    'azure_mistral_nemo/mistral-nemo',
    'azure_ministral/ministral',
    'azure_gpt4o_mini/gpt-4o-mini',
]

# Define generation parameters
genparams = FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])

# Load and format test data
test_data = FrugalGPT.loadcsvdata(f"data\\{dataname}\\train.csv")
prefix = open(f'config\\prompt\\{dataname}\\prefix_e8.txt').read()
test_data = FrugalGPT.formatdata(test_data, prefix)

# Load and format train data
train_data = FrugalGPT.loadcsvdata(f"data\\{dataname}\\test.csv")
prefix = open(f'config\\prompt\\{dataname}\\prefix_e8.txt').read()
train_data = FrugalGPT.formatdata(train_data, prefix)

# Generate DataFrame with model performance metrics
sample_size = 100
individualmodel_df = generate_dataframe(service_names,
                                        train_data[0:sample_size], test_data[0:sample_size],
                                        genparams,
                                        db_path=f"db\\{dataname}.sqlite",
                                        max_workers=2)
display(individualmodel_df)

# Function to compute tradeoffs for different budgets
def compute_tradeoffs(
    train_data,
    budget_list,
    name="test",
    service_names=[
        'azure_gpt4o/gpt-4o',
        'azure_openai/gpt-3.5-turbo',
        'azure_llama/llama-3-70b-instruct',
        'azure_mistral_nemo/mistral-nemo',
        'azure_ministral/ministral',
        'azure_gpt4o_mini/gpt-4o-mini',
    ],
    prefix="",
    skip=0,
    MyCascade=FrugalGPT.LLMCascade(
        score_noise_injection=False,
        db_path="db\\AGNEWS.sqlite"
    ),
    cascade_depth=3,
    score_test_size=0.55,
):
    for idx, budget in tqdm(enumerate(budget_list)):
        # Train the model
        user_budget = budget
        try:
            MyCascade.load(loadpath=f"strategy/{name}/", budget=user_budget)
            print("Already trained. Skipped.")
            continue
        except:
            print("cannot find, start new training")
        if idx < skip:
            continue
        if idx == 0:
            result = MyCascade.train(train_data[0:100], budget=user_budget,
                                     service_names=service_names,
                                     no_scorer_train=False,
                                     prefix=prefix,
                                     cascade_depth=cascade_depth,
                                     score_test_size=0.55,
                                     )
        else:
            result = MyCascade.train(train_data[0:100], budget=user_budget,
                                     service_names=service_names,
                                     no_scorer_train=True,
                                     prefix=prefix,
                                     cascade_depth=cascade_depth,
                                     score_test_size=0.55,
                                     )
        MyCascade.save(savepath=f"strategy/{name}/")
    return

# Define budget range and number of evaluations
start_budget = 0.00002
end_budget = 0.0050
num_eval = 10

# Define strategy name
name = f'{dataname}_Model20252602'

# Generate budget list
budget_list = numpy.linspace(start_budget, end_budget, num_eval)
budget_list[0] = 0.00001

# Load and format train data
dev = FrugalGPT.loadcsvdata(f"data\\{dataname}\\train.csv")
train_data = FrugalGPT.formatdata(dev, prefix)
MyCascade = FrugalGPT.LLMCascade(
    score_noise_injection=False,
    db_path=f"db\\{dataname}.sqlite",
    batch_build=True,
)

# Define service names for training
service_names_train = [
    'azure_gpt4o/gpt-4o',
    'azure_llama/llama-3-70b-instruct',
    'azure_openai/gpt-3.5-turbo',
    'azure_mistral_nemo/mistral-nemo',
    'azure_ministral/ministral',
    'azure_gpt4o_mini/gpt-4o-mini',
]

# Compute tradeoffs for different budgets
compute_tradeoffs(train_data=train_data[0:100],
                  budget_list=budget_list,
                  name=name,
                  service_names=service_names_train,
                  prefix=prefix,
                  skip=0,
                  MyCascade=MyCascade,
                  cascade_depth=3,
                  score_test_size=0.55,
                  )

# Function to generate a DataFrame from cascade results
def generate_dataframe_from_cascade(MyCascade, budget_list, train_data, test_data, genparams, name):
    # Initialize an empty list to store the rows for the DataFrame
    data = []

    # Iterate through the budget list
    for budget in tqdm(budget_list):
        # Load the strategy for the given budget
        MyCascade.load(loadpath=f"strategy/{name}/", budget=budget)

        # Get the completion batch for train data
        train_result = MyCascade.get_completion_batch(queries=train_data, genparams=genparams)

        # Compute the ACC and cost for train data
        train_acc_cost = FrugalGPT.compute_score(train_result)

        # Get the completion batch for test data
        test_result = MyCascade.get_completion_batch(queries=test_data, genparams=genparams)

        # Compute the ACC and cost for test data
        test_acc_cost = FrugalGPT.compute_score(test_result)

        # Create a row with the schema
        row = {
            "Test_acc": test_acc_cost['em'],
            "Test_cost": test_acc_cost['cost'],
            "Test_size": len(test_data),
            "Train_acc": train_acc_cost['em'],
            "Train_cost": train_acc_cost['cost'],
            "Train_size": len(train_data),
            "Budget": budget,
            "Method": "FrugalGPT",
            "Provider": "FrugalGPT",
            "Marker": 1,  # Marker is always 1 for this function
        }

        # Append the row to the data list
        data.append(row)
        display(row)

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df

# Generate DataFrame from cascade results
MyCascade_eval = FrugalGPT.LLMCascade()
MyCascade_eval.prefix = prefix
frugalgpt_df = generate_dataframe_from_cascade(MyCascade_eval,
                                               budget_list, train_data[0:100], test_data[0:100], genparams,
                                               name)
display(frugalgpt_df)

# Combine individual model and FrugalGPT DataFrames
individualmodel_df2 = copy.copy(individualmodel_df)
full_pd = pd.concat([frugalgpt_df, individualmodel_df2])
display(full_pd)

"""
# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Create the plot
plt.figure(figsize=(12, 8))

# Plot FrugalGPT points with split data
frugal_data = full_pd[full_pd['Method'] == 'FrugalGPT'].sort_values('Budget')

# First two points with Test_cost
first_two = frugal_data.iloc[:2]
plt.scatter(first_two['Test_cost'], first_two['Test_acc'], 
           label='FrugalGPT', color='blue', s=100)

# Remaining points with Budget
remaining = frugal_data.iloc[2:]
plt.scatter(remaining['Budget'], remaining['Test_acc'], 
           label='', color='blue', s=100)

# Add connecting lines for FrugalGPT points
x_values = list(first_two['Test_cost']) + list(remaining['Budget'])
y_values = list(first_two['Test_acc']) + list(remaining['Test_acc'])
plt.plot(x_values, y_values, color='blue', linestyle='-', alpha=0.5)

# Plot individual models
other_data = full_pd[full_pd['Method'] != 'FrugalGPT']
for Method in other_data['Method'].unique():
    Method_data = other_data[other_data['Method'] == Method]
    plt.scatter(Method_data['Test_cost'], Method_data['Test_acc'], 
               label=Method, s=100, alpha=0.7)

# Customize the plot
plt.xlabel('Cost per query ($)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance: Accuracy vs Cost', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
"""

