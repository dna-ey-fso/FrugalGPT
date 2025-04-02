# Set the Python executable path if not in PATH
$pythonPath = "python"

# Navigate to the script directory
Set-Location -Path "c:\Users\UH316ZA\OneDrive - EY\Documents\GitHub\FrugalGPT"

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..."
& $pythonPath -m pip install -r requirements.txt

# Run ApproachFrugalGPT.py
Write-Host "Running ApproachFrugalGPT.py..."
& $pythonPath "ApproachFrugalGPT.py"

# Run uvicorn to start the app
Write-Host "Starting Uvicorn server..."
& $pythonPath -c "import uvicorn; uvicorn.run('main:app', reload=True)"
