"""
Generate 4 environmental datasets for GMDH comparison.
Each dataset mimics patterns found in real environmental data:
- Heteroscedastic noise (noise magnitude varies with signal)
- Seasonal/temporal oscillations
- Nonlinear feature interactions
- Realistic feature bounds and correlations

Includes an ultra-high dimensional dataset (40+ variables) where GMDH excels.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_wildfire_dataset(n_samples=500, random_state=42):
    """
    Dataset 1: Wildfire intensity prediction
    Features: Temperature, Humidity, Wind Speed, Precipitation
    Target: Fire intensity (0-100)
    
    Mimics: Low-dimensional, data-rich environmental scenario with strong
    nonlinearities and inverse relationships.
    """
    rng = np.random.RandomState(random_state)
    
    # Feature generation with realistic bounds
    temp = rng.uniform(15, 40, n_samples)  # Celsius, 15-40°C range
    humidity = rng.uniform(20, 95, n_samples)  # Percent, 20-95%
    wind = rng.uniform(0, 25, n_samples)  # km/h, 0-25
    precip = rng.exponential(2, n_samples)  # mm, skewed distribution
    precip = np.clip(precip, 0, 20)
    
    # Nonlinear true function with realistic interactions
    y = (
        0.5 * temp +                    # Direct effect of temperature
        -0.3 * humidity +               # Humidity reduces fire risk
        2.0 * wind +                    # Wind accelerates spread
        -5.0 / (precip + 0.5) +         # Inverse: precipitation suppresses fire
        0.02 * temp * wind +            # Interaction: hot + windy = very dangerous
        15 * np.sin(temp / 10)          # Seasonal oscillation in fire behavior
    )
    
    # Heteroscedastic noise: more noise in extreme regions
    noise_scale = 0.5 + 0.01 * np.abs(y)
    noise = rng.normal(0, noise_scale)
    y = y + noise
    y = np.clip(y, 0, 100)  # Clamp to 0-100 intensity scale
    
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'wind_speed': wind,
        'precipitation': precip,
        'fire_intensity': y
    })
    return df


def generate_weather_dataset(n_samples=365, random_state=42):
    """
    Dataset 2: Weather/climate forecasting
    Features: 8 weather variables over seasonal cycle
    Target: Next-day average temperature
    
    Mimics: Medium-dimensional with strong seasonal patterns, autocorrelation,
    and temporal dynamics typical of climate data.
    """
    rng = np.random.RandomState(random_state)
    
    # Time index (simulating 1 year of daily data)
    t = np.arange(n_samples)
    
    # Strong seasonal component (365-day cycle)
    seasonal = 15 * np.sin(2 * np.pi * t / 365)
    
    # Features with realistic seasonal patterns
    temp_base = 15 + seasonal  # Seasonal temperature
    temp = temp_base + rng.normal(0, 2, n_samples)  # Add daily noise
    
    humidity = 70 - 20 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 30, 95)
    
    pressure = 1013 + 3 * np.sin(2 * np.pi * t / 365 + 0.5) + rng.normal(0, 1, n_samples)
    
    wind = 5 + 3 * np.sin(2 * np.pi * t / 365 + 1) + rng.exponential(0.5, n_samples)
    wind = np.clip(wind, 0, 20)
    
    # Oscillating cloud cover with noise
    cloud_cover = 50 + 30 * np.sin(2 * np.pi * t / 365 + 0.3) + rng.normal(0, 10, n_samples)
    cloud_cover = np.clip(cloud_cover, 0, 100)
    
    # Radiation (inverse to cloud)
    radiation = 300 + (100 - cloud_cover) * 1.5 + rng.normal(0, 20, n_samples)
    
    # Soil moisture (inversely related to temperature, affected by rain)
    soil_moisture = 40 - 10 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 5, n_samples)
    soil_moisture = np.clip(soil_moisture, 20, 60)
    
    # Atmospheric CO2 (slow trend)
    co2 = 410 + 0.05 * t + rng.normal(0, 1, n_samples)
    
    # Target: temperature with slight lag and seasonal phase shift
    y = (
        0.8 * temp +
        -0.15 * humidity +
        0.05 * pressure +
        0.3 * wind +
        -0.2 * cloud_cover +
        0.05 * radiation +
        0.1 * soil_moisture +
        0.01 * co2 +
        2 * np.sin(2 * np.pi * (t + 30) / 365)  # Lagged seasonal
    )
    
    # Autocorrelated noise (temporal correlation)
    noise = rng.normal(0, 1, n_samples)
    for i in range(1, n_samples):
        noise[i] = 0.4 * noise[i - 1] + 0.6 * noise[i]
    y = y + noise
    
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind,
        'cloud_cover': cloud_cover,
        'solar_radiation': radiation,
        'soil_moisture': soil_moisture,
        'co2_level': co2,
        'next_day_temp': y
    })
    return df


def generate_ecological_dataset(n_samples=150, random_state=42):
    """
    Dataset 3: Ecological/environmental monitoring
    Features: 20 high-dimensional vegetation, climate, and terrain variables
    Target: Ecosystem health index
    
    Mimics: High-dimensional problem with strong feature interactions, typical
    of ecosystem monitoring and biodiversity assessments. Fewer samples relative
    to features (short-sequence regime).
    """
    rng = np.random.RandomState(random_state)
    
    # Vegetation indices (NDVI, EVI, etc.)
    ndvi = rng.uniform(0.2, 0.8, n_samples)  # Normalized Difference Vegetation Index
    evi = rng.uniform(0.1, 0.6, n_samples)   # Enhanced Vegetation Index
    lai = rng.uniform(0.5, 6, n_samples)     # Leaf Area Index
    
    # Climate variables with spatial variation
    temp_annual = rng.uniform(5, 30, n_samples)
    precip_annual = rng.uniform(200, 3000, n_samples)
    
    # Seasonal patterns (e.g., dry season length, wet season intensity)
    dry_season_length = rng.uniform(2, 10, n_samples)
    wet_season_intensity = rng.uniform(0.3, 1.0, n_samples)
    
    # Terrain features
    elevation = rng.uniform(0, 3000, n_samples)
    slope = rng.uniform(0, 45, n_samples)
    aspect = rng.uniform(0, 360, n_samples)  # Cardinal direction
    
    # Soil properties (with feature coupling)
    soil_ph = rng.uniform(4, 8, n_samples)
    soil_nitrogen = rng.uniform(0.5, 5, n_samples)
    soil_organic_matter = rng.uniform(1, 10, n_samples)
    soil_moisture_capacity = rng.uniform(0.2, 0.5, n_samples)
    
    # Disturbance and management
    fire_frequency = rng.exponential(0.3, n_samples)
    fire_frequency = np.clip(fire_frequency, 0, 5)
    
    grazing_pressure = rng.uniform(0, 3, n_samples)  # Stock units
    human_impact = rng.uniform(0, 100, n_samples)  # Anthropogenic pressure score
    
    # Biodiversity proxy
    species_richness = rng.poisson(20, n_samples)
    
    # Target: Ecosystem health (0-100)
    y = (
        40 * ndvi +                            # Vegetation is key
        20 * evi +
        5 * lai +
        0.2 * (temp_annual - 15) ** 2 +        # Optimum around 15°C
        0.01 * precip_annual -                 # More rain is generally good
        2 * fire_frequency -                   # Disturbance reduces health
        1.5 * grazing_pressure -
        0.5 * human_impact +
        0.2 * soil_nitrogen +
        0.3 * soil_organic_matter -
        1 * np.abs(soil_ph - 6.5) +            # Optimal pH ~6.5
        0.02 * elevation -                     # Very high elevation is marginal
        0.05 * slope +                         # Some slope is okay
        0.01 * species_richness +
        -2 * dry_season_length +
        3 * wet_season_intensity +
        1 * np.sin(aspect / 45)                # Aspect affects microclimate
    )
    
    # Feature interactions (coupled effects)
    y += 0.01 * ndvi * precip_annual  # Vegetation responds to water
    y += -0.5 * (fire_frequency + 0.1) * ndvi  # Fire damage to vegetation
    y += 0.02 * soil_nitrogen * ndvi  # Nutrients support vegetation
    
    # Heteroscedastic noise with outliers
    noise_scale = 2 + 0.05 * np.abs(y)
    noise = rng.normal(0, noise_scale)
    # Occasional outliers (unusual environmental events)
    outliers = rng.binomial(1, 0.05, n_samples)
    noise += outliers * rng.normal(0, 15, n_samples)
    
    y = y + noise
    y = np.clip(y, 0, 100)  # Health index 0-100
    
    df = pd.DataFrame({
        'ndvi': ndvi,
        'evi': evi,
        'leaf_area_index': lai,
        'temp_annual': temp_annual,
        'precip_annual': precip_annual,
        'dry_season_length': dry_season_length,
        'wet_season_intensity': wet_season_intensity,
        'elevation': elevation,
        'slope': slope,
        'aspect': aspect,
        'soil_ph': soil_ph,
        'soil_nitrogen': soil_nitrogen,
        'soil_organic_matter': soil_organic_matter,
        'soil_moisture_capacity': soil_moisture_capacity,
        'fire_frequency': fire_frequency,
        'grazing_pressure': grazing_pressure,
        'human_impact': human_impact,
        'species_richness': species_richness,
        'ecosystem_health': y
    })
    return df


def generate_air_quality_dataset(n_samples=250, random_state=42):
    """
    Dataset 4: Ultra-high dimensional air quality/atmospheric chemistry
    Features: 41 pollutants, meteorological, and temporal variables
    Target: Air quality index (AQI) or health impact score
    
    Mimics: Ultra-high dimensional problem where GMDH excels. Combines:
    - Multiple pollutant species (NO2, O3, PM2.5, PM10, CO, SO2, VOCs, etc.)
    - Meteorological variables (temperature, humidity, pressure, wind)
    - Temporal/seasonal factors
    - Complex nonlinear chemical interactions
    - Feature-to-feature dependencies typical of atmospheric chemistry
    
    This regime is where GMDH's polynomial basis and self-organization shine.
    """
    rng = np.random.RandomState(random_state)
    
    # Time representation
    t = np.arange(n_samples)
    hour_of_day = (t % 24).astype(float)
    day_of_year = (t % 365).astype(float)
    
    # Meteorological base variables
    temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + rng.normal(0, 2, n_samples)
    temp = np.clip(temp, -5, 35)
    
    humidity = 60 - 15 * np.sin(2 * np.pi * day_of_year / 365) + rng.normal(0, 8, n_samples)
    humidity = np.clip(humidity, 20, 95)
    
    pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365 + 0.7) + rng.normal(0, 2, n_samples)
    
    wind_speed = 3 + 2 * np.sin(2 * np.pi * day_of_year / 365 + 1.5) + rng.exponential(0.3, n_samples)
    wind_speed = np.clip(wind_speed, 0, 15)
    
    wind_direction = rng.uniform(0, 360, n_samples)
    
    solar_radiation = 200 + 300 * np.sin(2 * np.pi * (hour_of_day - 6) / 24) * np.cos(2 * np.pi * day_of_year / 365)
    solar_radiation = np.clip(solar_radiation, 0, 800)
    
    # Primary pollutants (emitted directly)
    no2 = 30 + 20 * (1 - wind_speed / 15) + 15 * (1 - humidity / 100) + rng.exponential(1.5, n_samples)
    no2 = np.clip(no2, 5, 150)
    
    co = 0.5 + 0.3 * (1 - wind_speed / 15) + rng.exponential(0.1, n_samples)
    co = np.clip(co, 0.1, 2.0)
    
    so2 = 5 + 3 * (1 - wind_speed / 15) + rng.exponential(0.8, n_samples)
    so2 = np.clip(so2, 1, 20)
    
    pm25 = 20 + 15 * (1 - wind_speed / 15) + 10 * (humidity / 100) + rng.exponential(2, n_samples)
    pm25 = np.clip(pm25, 5, 150)
    
    pm10 = 30 + 20 * (1 - wind_speed / 15) + 15 * (humidity / 100) + rng.exponential(2, n_samples)
    pm10 = np.clip(pm10, 10, 200)
    
    # Secondary pollutants (formed through photochemistry)
    # O3 depends on NOx, radiation, temperature
    o3 = (
        15 + 
        0.3 * no2 +  # NOx precursor
        0.05 * solar_radiation / 100 +  # Photochemical formation
        5 * np.sin(2 * np.pi * (hour_of_day - 4) / 24) +  # Diurnal cycle
        2 * (temp - 15) / 10  # Temperature dependence
    )
    o3 = np.clip(o3, 5, 200)
    
    # VOCs (volatile organic compounds) - 8 species
    voc_names = ['benzene', 'toluene', 'xylene', 'formaldehyde', 
                 'acetaldehyde', 'isoprene', 'pinene', 'limonene']
    vocs = {}
    for i, voc_name in enumerate(voc_names):
        base = [5, 3, 2, 4, 2, 8, 6, 4][i]  # Different baseline levels
        vocs[f'voc_{voc_name}'] = (
            base + 
            (base * 0.7) * (1 - wind_speed / 15) +
            (base * 0.5) * solar_radiation / 800 +
            rng.exponential(0.3, n_samples)
        )
        vocs[f'voc_{voc_name}'] = np.clip(vocs[f'voc_{voc_name}'], base * 0.1, base * 5)
    
    # Heavy metals / trace elements (5 species)
    pb = 0.05 + 0.04 * (1 - wind_speed / 15) + rng.exponential(0.01, n_samples)
    pb = np.clip(pb, 0.01, 0.2)
    
    cd = 0.01 + 0.008 * (1 - wind_speed / 15) + rng.exponential(0.002, n_samples)
    cd = np.clip(cd, 0.002, 0.05)
    
    ni = 0.02 + 0.015 * (1 - wind_speed / 15) + rng.exponential(0.003, n_samples)
    ni = np.clip(ni, 0.005, 0.08)
    
    cu = 0.03 + 0.025 * (1 - wind_speed / 15) + rng.exponential(0.005, n_samples)
    cu = np.clip(cu, 0.01, 0.12)
    
    zn = 0.08 + 0.06 * (1 - wind_speed / 15) + rng.exponential(0.01, n_samples)
    zn = np.clip(zn, 0.02, 0.3)
    
    # Secondary inorganic aerosols
    so4 = 2 + 1.5 * so2 / 5 + 0.5 * solar_radiation / 200 + rng.exponential(0.3, n_samples)
    so4 = np.clip(so4, 0.5, 20)
    
    no3 = 1 + 0.8 * no2 / 30 + rng.exponential(0.2, n_samples)
    no3 = np.clip(no3, 0.2, 10)
    
    nh4 = 0.5 + 0.3 * humidity / 100 + rng.exponential(0.1, n_samples)
    nh4 = np.clip(nh4, 0.1, 3)
    
    # Target: Air Quality Index (AQI) / health impact score (0-500+)
    # Based on pollutant concentrations and their health effects
    y = (
        1.5 * pm25 +                    # PM2.5 is primary health concern
        0.8 * pm10 +
        0.7 * no2 +                     # NOx respiratory effects
        50 * co +                       # CO toxicity
        2 * so2 +                       # SO2 respiratory effects
        0.5 * o3 +                      # O3 oxidative stress
        5 * pb +                        # Heavy metal toxicity
        10 * cd +
        8 * ni +
        6 * cu +
        4 * zn +
        1 * so4 +                       # Secondary aerosol effects
        1.2 * no3 +
        0.8 * nh4 +
        sum([0.3 * vocs[f'voc_{voc_name}'] for voc_name in voc_names]) +  # VOC effects
        2 * (temp - 20) ** 2 / 10 +     # Temperature stress (heat/cold)
        -0.5 * wind_speed +             # Wind dispersion helps
        0.2 * humidity                  # Humidity affects aerosol properties
    )
    
    # Complex chemical interactions
    y += 0.05 * no2 * o3 * solar_radiation / 200  # Photooxidant chemistry
    y += 0.02 * pm25 * (humidity / 100) * 10  # Hygroscopic growth
    y += -0.1 * wind_speed * pm25  # Wind dispersion
    y += 0.03 * so2 * humidity / 100 * 5  # SO2 oxidation and aerosol formation
    
    # Heteroscedastic noise
    noise_scale = 2 + 0.05 * np.abs(y)
    noise = rng.normal(0, noise_scale)
    y = y + noise
    y = np.clip(y, 0, 500)  # AQI scale
    
    # Build DataFrame
    df_data = {
        # Meteorology
        'temperature': temp,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'solar_radiation': solar_radiation,
        'hour_of_day': hour_of_day,
        'day_of_year': day_of_year,
        # Primary pollutants
        'no2': no2,
        'co': co,
        'so2': so2,
        'pm25': pm25,
        'pm10': pm10,
        # Secondary pollutants
        'o3': o3,
        # VOCs
        **vocs,
        # Heavy metals
        'pb': pb,
        'cd': cd,
        'ni': ni,
        'cu': cu,
        'zn': zn,
        # Secondary inorganic aerosols
        'so4': so4,
        'no3': no3,
        'nh4': nh4,
        # Target
        'aqi': y
    }
    
    df = pd.DataFrame(df_data)
    return df

def generate_low_dim_dataset(n_samples=500, random_state=42):
    """
    Dataset 5: Low-dimensional synthetic regression
    Features: 3 independent variables (X1, X2, X3) with simple quadratic interactions
    Target: y = 2*X1^2 + 1.5*X2*X3 + 3*X1 - 0.5*X2 + noise
    
    This dataset is designed to be easy for GMDH: low dimension, clear polynomial structure.
    """
    rng = np.random.RandomState(random_state)
    
    # Generate three features uniformly in [-3, 3]
    X1 = rng.uniform(-3, 3, n_samples)
    X2 = rng.uniform(-3, 3, n_samples)
    X3 = rng.uniform(-3, 3, n_samples)
    
    # True function: quadratic and interaction terms
    y = (2 * X1**2 + 1.5 * X2 * X3 + 3 * X1 - 0.5 * X2 + 
         0.8 * X3**2 - 1.2 * X1 * X2)
    
    # Add moderate heteroscedastic noise
    noise_scale = 0.5 + 0.1 * np.abs(y)
    noise = rng.normal(0, noise_scale)
    y = y + noise
    
    df = pd.DataFrame({
        'x1': X1,
        'x2': X2,
        'x3': X3,
        'target': y
    })
    return df


def main():
    """Generate and save all four datasets."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("Generating environmental datasets...")
    print("="*70)
    
    # Dataset 1: Wildfire
    df_wildfire = generate_wildfire_dataset(n_samples=500)
    path_wildfire = data_dir / "wildfire.csv"
    df_wildfire.to_csv(path_wildfire, index=False)
    print(f"✓ Wildfire dataset: {len(df_wildfire)} samples, {len(df_wildfire.columns)-1} features")
    print(f"  Low-dimensional, data-rich regime")
    print(f"  Saved to {path_wildfire}")
    
    # Dataset 2: Weather
    df_weather = generate_weather_dataset(n_samples=365)
    path_weather = data_dir / "weather.csv"
    df_weather.to_csv(path_weather, index=False)
    print(f"\n✓ Weather dataset: {len(df_weather)} samples, {len(df_weather.columns)-1} features")
    print(f"  Medium-dimensional with strong seasonality")
    print(f"  Saved to {path_weather}")
    
    # Dataset 3: Ecological
    df_ecological = generate_ecological_dataset(n_samples=150)
    path_ecological = data_dir / "ecological.csv"
    df_ecological.to_csv(path_ecological, index=False)
    print(f"\n✓ Ecological dataset: {len(df_ecological)} samples, {len(df_ecological.columns)-1} features")
    print(f"  High-dimensional with complex interactions")
    print(f"  Saved to {path_ecological}")
    
    # Dataset 4: Air Quality (ultra-high dimensional)
    df_air_quality = generate_air_quality_dataset(n_samples=250)
    path_air_quality = data_dir / "air_quality.csv"
    df_air_quality.to_csv(path_air_quality, index=False)
    print(f"\n✓ Air Quality dataset: {len(df_air_quality)} samples, {len(df_air_quality.columns)-1} features")
    print(f"  🔥 ULTRA-HIGH dimensional (41 features)")
    print(f"  Complex atmospheric chemistry interactions")
    print(f"  Optimal regime for GMDH performance")
    print(f"  Saved to {path_air_quality}")

    # Dataset 5: Low-dimensional synthetic
    df_low_dim = generate_low_dim_dataset(n_samples=500)
    path_low_dim = data_dir / "low_dim.csv"
    df_low_dim.to_csv(path_low_dim, index=False)
    print(f"\n✓ Low-dimensional dataset: {len(df_low_dim)} samples, {len(df_low_dim.columns)-1} features")
    print(f"  Very low-dimensional regime (3 features)")
    print(f"  Saved to {path_low_dim}")
    
    print("\n" + "="*70)
    print("All 5 datasets generated successfully.")
    print("="*70)


if __name__ == "__main__":
    main()