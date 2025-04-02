from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from dotenv import load_dotenv
import tiktoken
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime
import time  # Import the time module for latency measurement

# Add the src directory to the Python path
sys.path.insert(0, 'src/')
import FrugalGPT
import service
from service.modelservice import AzureGPT4oModelProvider, GenerationParameter

# Configure logging to display information-level logs
logging.basicConfig(level=logging.INFO)

# Initialize the FastAPI application
app = FastAPI()

# Command to run the application: uvicorn main:app --reload

# Configure CORS to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model for cascade-related endpoints
class CascadeRequest(BaseModel):
    prompt: str
    conversation_history: list = []  # Optional conversation history
    system_prompt: str = ""  # Optional system prompt
    few_shots: list = []  # Optional few-shot examples
    content: str
    query_prompt_template: str = ""  # Add query_prompt_template field

# Load environment variables from a .env file
load_dotenv()

# Function to execute the cascade logic
def execute_cascade_logic(prompt: str, content: str, system_prompt: str, few_shots: list, query_prompt_template: str):
    logging.info("Starting cascade logic execution")
    MyCascade = FrugalGPT.LLMCascade_cache()
    strategy_path = 'strategy/cascade_strategy.json'
    
    try:
        # Measure start time
        start_time = time.time()

        # Define the budget for the cascade
        budget = 0.0005733333333333334
        logging.info(f"Loading cascade strategy from: {strategy_path} with budget: {budget}")
        MyCascade.load(loadpath=strategy_path, budget=budget)
        
        # Define generation parameters
        genparams = FrugalGPT.GenerationParameter(
            max_tokens=200,
            temperature=0.1,
            stop=['\n']
        )
        logging.info(f"Generation parameters: {genparams}")

        # Get the completion from the cascade
        answer, model_used = MyCascade.get_completion(
            query=prompt,
            genparams=genparams,
            system_prompt=system_prompt,
            content=content,
            few_shots=few_shots,
            query_prompt_template=query_prompt_template  # Pass query_prompt_template
        )
        logging.info(f"Cascade answer: {answer}, Model used: {model_used}")

        # Calculate the cost of the cascade
        cost = MyCascade.get_cost()
        logging.info(f"Cascade cost: {cost}")

        # Measure end time and calculate latency
        end_time = time.time()
        latency = end_time - start_time
        logging.info(f"Cascade latency: {latency:.6f} seconds")

        return {
            "answer": answer,
            "cost": cost,
            "model_used": model_used,
            "latency": latency  # Include latency in the response
        }
    except Exception as e:
        logging.error(f"Error in cascade logic: {str(e)}")
        raise Exception(f"Error during cascade execution: {str(e)}")

# Function to count the number of tokens in a text for a specific model
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to execute GPT-4o logic
def execute_gpt4o(prompt: str, content: str, system_prompt: str, few_shots: list):
    try:
        # Measure start time
        start_time = time.time()

        # Initialize the GPT-4o model provider
        provider = AzureGPT4oModelProvider("gpt-4o-mini")

        # Define generation parameters
        genparams = GenerationParameter(
            max_tokens=200,
            temperature=0.1,
            stop=['\n']
        )

        # Calculate input tokens
        input_text = prompt + system_prompt + content + " ".join(map(str, few_shots))
        input_tokens = count_tokens(input_text)

        # Get the model completion
        completion = provider.getcompletiongpt4o(
            query=prompt,
            genparams=genparams,
            content=content,
            system_prompt=system_prompt,
            few_shots=few_shots
        )

        # Calculate output tokens
        output_tokens = count_tokens(completion)

        # Define token costs
        COST_PER_INPUT_TOKENS = 0.0000025  
        COST_PER_OUTPUT_TOKENS = 0.00001

        # Calculate the total cost
        cost = (input_tokens) * COST_PER_INPUT_TOKENS + (output_tokens) * COST_PER_OUTPUT_TOKENS

        # Measure end time and calculate latency
        end_time = time.time()
        latency = end_time - start_time
        logging.info(f"GPT-4o latency: {latency:.6f} seconds")

        return {
            "answer": completion,
            "cost": cost,
            "model_used": "gpt-4o-mini",
            "latency": latency  # Include latency in the response
        }

    except Exception as e:
        raise Exception(f"Error executing GPT-4o request: {str(e)}")

# Endpoint to execute cascade logic
@app.post("/execute_cascade")
async def execute_cascade(request: CascadeRequest):
    try:
        result = execute_cascade_logic(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content=request.content,
            few_shots=request.few_shots,
            query_prompt_template=request.query_prompt_template  # Pass query_prompt_template
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint to execute GPT-4o logic
@app.post("/execute_gpt4o")
async def execute_gpt4o_endpoint(request: CascadeRequest):
    try:
        result = execute_gpt4o(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            few_shots=request.few_shots,
            content=request.content
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint to compare costs between cascade and GPT-4o
@app.post("/compare_costs")
async def compare_costs(request: CascadeRequest):
    try:
        # Log the incoming request
        logging.info("Received request for /compare_costs")

        # Execute cascade logic
        cascade_result = execute_cascade_logic(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content=request.content,
            few_shots=request.few_shots,
            query_prompt_template=request.query_prompt_template
        )
        cascade_cost = cascade_result["cost"]
        cascade_latency = cascade_result["latency"]
        logging.info(f"Cascade cost calculated: {cascade_cost}, Latency: {cascade_latency:.6f} seconds")

        # Execute GPT-4o logic
        gpt4o_result = execute_gpt4o(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            content=request.content,
            few_shots=request.few_shots
        )
        gpt4o_cost = gpt4o_result["cost"]
        gpt4o_latency = gpt4o_result["latency"]
        logging.info(f"GPT-4o cost calculated: {gpt4o_cost}, Latency: {gpt4o_latency:.6f} seconds")

        # Create the comparison graph
        labels = ['Cascade', 'GPT-4o']
        costs = [cascade_cost, gpt4o_cost]
        latencies = [cascade_latency, gpt4o_latency]

        plt.figure(figsize=(8, 6))

        # Plot cost comparison
        plt.subplot(2, 1, 1)
        bars = plt.bar(labels, costs, color=['blue', 'orange'])
        plt.title('Cost Comparison')
        plt.ylabel('Cost (USD)')
        plt.tight_layout()
        for bar, cost in zip(bars, costs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${cost:.6f}", ha='center', va='bottom')

        # Plot latency comparison
        plt.subplot(2, 1, 2)
        bars = plt.bar(labels, latencies, color=['blue', 'orange'])
        plt.title('Latency Comparison')
        plt.ylabel('Latency (seconds)')
        plt.xlabel('Model')
        plt.tight_layout()
        for bar, latency in zip(bars, latencies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{latency:.6f}s", ha='center', va='bottom')

        # Ensure the 'plots' directory exists
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"cost_latency_comparison_{timestamp}.png")

        # Save the graph to the file
        plt.savefig(plot_path)
        logging.info(f"Graph saved to file: {plot_path}")

        # Encode the graph as base64 for the JSON response
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        logging.info("Graph generated and encoded successfully")

        # Return the JSON response
        return JSONResponse(content={
            "status": "success",
            "graph": graph_base64,
            "details": {
                "cascade_cost": cascade_cost,
                "cascade_latency": cascade_latency,
                "gpt4o_cost": gpt4o_cost,
                "gpt4o_latency": gpt4o_latency,
                "graph_path": plot_path  # Include the file path in the response
            }
        })
    except Exception as e:
        logging.error(f"Error in /compare_costs: {str(e)}")
        return {"status": "error", "message": str(e)}

