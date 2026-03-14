# Motor Maintenance Prediction API

This project implements a predictive maintenance API for industrial motors, utilizing machine learning to assess failure risks based on real-time sensor telemetry data. Developed with FastAPI for high-performance web services and scikit-learn for robust model training, it provides scalable and reliable maintenance predictions.

## Features

- **Synthetic Data Generation**: Create realistic motor telemetry datasets with configurable noise and sensor drift
- **Refined Machine Learning Model**: Random Forest classifier with class balancing, feature engineering (voltage deviation, power usage, thermal stress), achieving 99.98% accuracy, 100% ROC-AUC, and 99.97% balanced accuracy
- **RESTful API**: FastAPI-based endpoints for real-time failure predictions with input validation and confidence scoring
- **Production Ready**: Includes model serialization, request logging, timestamps, and comprehensive testing
- **CI/CD**: GitHub Actions for automated testing, model training, and Docker deployment
- **Containerized**: Docker and Docker Compose support for easy deployment

## Requirements

- Python 3.8+
- Docker (optional, for containerized deployment)
- Dependencies listed in `requirements.txt`

## Installation

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd motor-maintenance-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd motor-maintenance-api
   ```

2. **Build and run with Docker**
   ```bash
   # Build the Docker image
   docker build -t motor-maintenance-api .

   # Run the container
   docker run -d -p 8000:8000 --name motor-api motor-maintenance-api
   ```

3. **Or use Docker Compose**
   ```bash
   # If you have docker-compose installed
   docker-compose up -d
   ```

The API will be available at `http://localhost:8000`

## Data Generation

Generate synthetic motor telemetry data for training and testing:

```bash
# Generate default dataset (100k rows)
python data_generator/data_generator.py

# Customize parameters
python data_generator/data_generator.py --rows 50000 --noise-scale 1.2 --drift-amount 5.0
```

**Options:**
- `--rows`: Number of data points to generate
- `--noise-scale`: Add measurement noise (1.0 = no extra noise)
- `--drift-amount`: Simulate sensor drift over time (0.0 = no drift)

## Model Training

Train the refined Random Forest model on generated data:

```bash
python src/train_model.py
```

This will:
- Load data from `data/motor_maintenance_data.csv`
- Apply feature engineering (voltage deviation, power usage, thermal stress)
- Train a balanced Random Forest classifier
- Evaluate with accuracy, ROC-AUC, F1-score, and balanced accuracy
- Save the model to `models/motor_model.pkl`
- Save metrics to `models/metrics.pkl`

**Sample Output:**
```
Training Complete!
Accuracy: 99.98%
ROC-AUC: 100.00%
F1-Score: 99.94%
Balanced Accuracy: 99.97%
```

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Usage

### Endpoints

#### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "API is Online",
  "model": "Random Forest v1"
}
```

#### POST `/predict`
Predict motor failure risk based on sensor data.

**Request Body:**
```json
{
  "voltage_v": 230.0,
  "current_a": 12.0,
  "temp_c": 65.0,
  "vibration_g": 0.08
}
```

**Response:**
```json
{
  "failure_prediction": 0,
  "status": "NORMAL",
  "confidence": 0.96,
  "timestamp": "2026-03-15T10:30:00.000Z"
}
```

**Validation:**
- `voltage_v`: 180-280 V
- `current_a`: 0-30 A
- `temp_c`: -10-120 °C
- `vibration_g`: 0-1 g

### Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"voltage_v": 230, "current_a": 12, "temp_c": 65, "vibration_g": 0.08}'
```

Using Python:
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "voltage_v": 230,
    "current_a": 12,
    "temp_c": 65,
    "vibration_g": 0.08
})
print(response.json())
```

## Testing

Run the unit tests to verify functionality:

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
python -m pytest tests/
```

Tests include:
- Health check endpoint
- Normal operation predictions
- Failure risk predictions

## Deployment

### GitHub Actions

The project includes GitHub Actions workflows for CI/CD:

- **CI** (`.github/workflows/ci.yml`): Runs on every push/PR, executes tests and retrains the model
- **Deploy** (`.github/workflows/deploy.yml`): Runs on releases, builds and pushes Docker image

To enable deployment:
1. Set up Docker Hub account
2. Add repository secrets: `DOCKER_USERNAME` and `DOCKER_PASSWORD`
3. Create a release to trigger deployment

### Manual Docker Deployment

```bash
# Build image
docker build -t your-registry/motor-maintenance-api:latest .

# Push to registry
docker push your-registry/motor-maintenance-api:latest

# Run on server
docker run -d -p 8000:8000 your-registry/motor-maintenance-api:latest
```

## Project Structure

```
motor-maintenance-api/
├── app/
│   └── main.py              # FastAPI application
├── data/
│   └── motor_maintenance_data.csv  # Training data
├── data_generator/
│   └── data_generator.py    # Synthetic data generation
├── models/
│   ├── motor_model.pkl      # Trained model
│   └── metrics.pkl          # Model performance metrics
├── src/
│   └── train_model.py       # Model training script
├── tests/
│   └── test_api.py          # Unit tests
├── .github/
│   └── workflows/           # GitHub Actions
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### GET /
Health check endpoint.

**Response:**
```json
{
  "status": "API is Online",
  "model": "Random Forest v1"
}
```

#### POST /predict
Predict motor failure risk from sensor readings.

**Request Body:**
```json
{
  "voltage_v": 240.5,
  "current_a": 14.2,
  "temp_c": 75.0,
  "vibration_g": 0.15
}
```

**Response:**
```json
{
  "failure_prediction": 1,
  "status": "FAILURE RISK",
  "confidence": 0.8923
}
```

**Fields:**
- `failure_prediction`: 0 (normal) or 1 (failure risk)
- `status`: Human-readable status message
- `confidence`: Model's confidence score (0.0-1.0)

## Docker Deployment

The application is fully containerized and can be deployed using Docker:

### Build the Image
```bash
docker build -t motor-maintenance-api .
```

### Run the Container
```bash
# Run in detached mode
docker run -d -p 8000:8000 --name motor-api motor-maintenance-api

# Check logs
docker logs motor-api

# Stop the container
docker stop motor-api
```

### Docker Compose
For easier management, use the provided `docker-compose.yml`:

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Container Features
- **Automatic Setup**: Data generation and model training happen during build
- **Port Mapping**: API exposed on port 8000
- **Health Checks**: Built-in health monitoring
- **Volume Mounting**: Models and data directories are mounted for persistence

## Project Structure

```
motor_maintanance_api/
├── data_generator/
│   ├── data_generator.py      # Synthetic data generation script
│   └── data/                  # Generated CSV files
├── src/
│   └── train_model.py         # Model training pipeline
├── app/
│   └── main.py                # FastAPI application
├── models/                    # Trained model files
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open a GitHub issue or contact the maintainers.