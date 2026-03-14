# Motor Maintenance Prediction API

A machine learning-powered API for predicting motor failure risks using real-time sensor telemetry data. Built with FastAPI and scikit-learn for reliable, scalable predictive maintenance.

## Features

- **Synthetic Data Generation**: Create realistic motor telemetry datasets with configurable noise and sensor drift
- **Machine Learning Model**: Random Forest classifier trained on voltage, current, temperature, and vibration sensors
- **RESTful API**: FastAPI-based endpoints for real-time failure predictions
- **High Accuracy**: Achieves ~95% accuracy on test data with proper feature engineering
- **Production Ready**: Includes model serialization, input validation, and confidence scoring

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd motor_maintanance_api
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
   cd motor_maintanance_api
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

Train the Random Forest model on generated data:

```bash
python src/train_model.py
```

This will:
- Load data from `data/motor_maintenance_data.csv`
- Train a Random Forest classifier
- Save the model to `models/motor_model.pkl`
- Display accuracy metrics and classification report

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
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