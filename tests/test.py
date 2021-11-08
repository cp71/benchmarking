import pytest
import sys
import pathlib
import numpy as np
import pandas as pd
import statsmodels.tsa.interp

ROOT = pathlib.Path(__file__).resolve().parent / "benchmark"
sys.path.append(str(ROOT))
import benchmark

@pytest.fixture
def get_data():
    return pd.read_csv(ROOT/ "documentation/GDP.csv")

def test_imf():
    #input data
    high_frequency_data = np.array([98.2, 100.8, 102.2, 100.8, 99.0, 101.6, 102.7, 101.5, 100.5, 103.0, 103.5, 101.5])
    low_frequency_data = np.array([4000.,4161.4])
    truncation = 8

    frequency = len(high_frequency_data[:truncation]) // len(low_frequency_data)
    linear_constraint = np.ones(frequency)
    
    estimator = benchmark.Estimator(linear_constraint=linear_constraint)
    estimator.fit(high_frequency_data[:truncation], low_frequency_data)
    
    interpolant = estimator.predict()
    
    imf_interpolant = np.array([969.8, 998.4, 1018.3, 1013.4, 1007.2, 1042.9, 1060.3, 1051.0, 1040.6, 1066.5, 1071.7, 1051.0])
    np.testing.assert_almost_equal(imf_interpolant[:truncation], interpolant, 1)
    
    ratio = interpolant[-1] / high_frequency_data[-1]
    extrapolant = ratio * high_frequency_data[truncation:]
    np.testing.assert_almost_equal(imf_interpolant[truncation:], extrapolant, 1)
    
def test_denton():
   #input data
    high_frequency_data = np.array([50,100,150,100] * 5)
    low_frequency_data = np.array([500,400,300,400,500])

    frequency = len(high_frequency_data) // len(low_frequency_data)
    linear_constraint = np.ones(frequency)

    estimator = benchmark.Estimator(linear_constraint=linear_constraint)
    estimator.fit(high_frequency_data, low_frequency_data)

    interpolant = estimator.predict()

    denton_interpolant = np.array([64.334796,127.80616,187.82379,120.03526,56.563894,105.97568,147.50144,89.958987,40.547201,74.445963, 108.34473,76.66211,42.763347,94.14664,153.41596, 109.67405,58.290761,122.62556,190.41409,128.66959])
    np.testing.assert_almost_equal(denton_interpolant, interpolant, 1)
    
@pytest.mark.slow
def test_statsmodels(get_data):
    data = get_data

    data = data.loc[data["Country Name"].isin(["Australia", "Brazil"])].copy()
    data = data.drop(columns = ["Country Code", "Indicator Name", "Indicator Code"]).set_index("Country Name").T.copy()

    data.index = pd.to_datetime(data.index).year
    data.index.name = "Year"
    data.columns.name = "Country"

    data = data[:-1].copy()
    
    data["Brazil_Downsample"] = data["Brazil"].copy()
    data["Brazil_Downsample"][1::2] = np.NaN
    
    high_frequency_data = data["Australia"].values
    low_frequency_data = data["Brazil_Downsample"].dropna().values
    
    aud_gdp, brazil_downsample_gdp = data["Australia"].values, data["Brazil_Downsample"].dropna().values
    brazil_upsample = statsmodels.tsa.interp.dentonm(aud_gdp, brazil_downsample_gdp, freq = "other", k = 2)

    data["Brazil_Upsample"] = brazil_upsample * 2
    data[["Brazil","Brazil_Upsample"]].corr().iloc[0,1]

    estimator = benchmark.Estimator()
    estimator.fit(high_frequency_data, low_frequency_data)
    
    upsampled = estimator.predict()
    np.testing.assert_almost_equal(data["Brazil_Upsample"].values, upsampled)