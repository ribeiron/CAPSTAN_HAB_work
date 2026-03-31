"""
Utility functions to be reused in notebooks.
"""
import os

from glob import glob
import fsspec
import s3fs
import pandas as pd
import xarray as xr
from dask import bag as db
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Default location to store local copy of data
DATA_DIR = "../Datasets/"

# List of moorings and corresponding regions to build S3 paths
DEFAULT_MOORINGS = [
    ("NRS", "NRSKAI"),
    ("SA", "SAM8SG"),
    ("SA", "SAM5CB"),
    ("SA", "SAM2CP"),
    ("SA", "SAM6IS"),
    ("SA", "SAM3MS"),
    ("SA", "SAM7DS")
]


def extract_file_id_from_filename(filename):
    """
    Extract the file ID from a filename.

    Filename must follow the structure of IMOS data in S3 buckets.
    """
    return filename.split("/")[6].split("-")[2]


def load_file_urls(path="s3://imos-data/IMOS/ANMN/NRS/NRSKAI/Temperature/", pattern="*.nc", get_file_ids=False, get_first_file_only=False):
    """Load files from an S3 bucket that match a pattern.

    Parameters
    ----------
    path : str
        Path to the directory containing the files.
    pattern : str
        Pattern to match the files.
    get_file_ids : bool
        If turned on, create a list of lists where each list has a specific file ID.

    Returns
    -------
    files : list
        List of files that match the path and pattern.
    """
    fs = fsspec.filesystem(
        "s3",
        use_listings_cache=False,
        anon=True,
    )
    if not path.endswith("/"):
        path = path + "/"
    files = sorted(fs.glob(f"{path}{pattern}"))
    
    if get_file_ids:
        file_ids = dict()
        for file in files:
            file_id = extract_file_id_from_filename(file)
            if not file_ids.get(file_id, False):
                file_ids[file_id] = []
            else:
                file_ids[file_id].append(file)
                
        files = sorted(list(file_ids.values()))
        
    if get_first_file_only:
        files = [file[0] if isinstance(file, list) else file for file in files]

    return files


def open_nc(url_or_path, variable=None, remote=True):
    """
    Open an nc file from an S3 bucket or locally.

    Parameters
    ----------
    url_or_path : str
        URL or path to the file.
    remote : bool
        Whether to load from S3 or locally

    Returns
    -------
    data : xarray.Dataset
    """
    try:
        if remote:
            data = xr.open_dataset(url_or_path, engine="h5netcdf").load().squeeze()
        else:
            data = xr.open_dataset(url_or_path, engine="h5netcdf").load().squeeze()
    except Exception as e:
        print(f"Failed to open {url_or_path}: {e}")
        return None
    return data


def open_files_with_dask(files):
    """
    Open files with dask bag. Requires a running Dask client.

    Parameters
    ----------
    files : list
        List of file URLs to open.

    Returns
    -------
    bag : dask.bag
    cast : list of xarray Datasets.
    """
    bag = db.from_sequence(files)
    cast = db.map(open_nc, bag).compute()
    return cast


def get_shared_coordinates(list_of_xr_datasets):
    """
    Get shared coordinates between a list of xarray datasets.

    Parameters
    ----------
    list_of_xr_datasets : list
        List of xarray datasets.

    Returns
    -------
    commonvars: list
        List of shared coordinates.
    """
    return list(
        set.intersection(
            *list(
                (
                    map(
                        lambda ds: set([var for var in ds.data_vars]),
                        list_of_xr_datasets,
                    )
                )
            )
        )
    )


# +
import os
import xarray as xr
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Map moorings to their regions
MOORING_REGIONS = {
    "NRSKAI": "NRS",
    "SAM8SG": "SA",
    "SAM5CB": "SA",
    "SAM2CP": "SA",
    "SAM6IS": "SA",
    "SAM3MS": "SA",
    "SAM7DS": "SA"
}

def load_data_products(
    moorings_list,
    data_type="hourly_timeseries",
    local_base="imos-data",
    cache=True
):
    """
    Load IMOS mooring data for a list of moorings from the S3 bucket.

    Parameters
    ----------
    moorings_list : list
        List of mooring names (e.g., ["NRSKAI", "SAM8SG"]).
    data_type : str
        Type of data folder (default "hourly_timeseries").
    local_base : str
        Local folder to store downloaded data.
    cache : bool
        Whether to save files locally.

    Returns
    -------
    files_dict : dict
        Paths to downloaded NetCDF files.
    datasets_dict : dict
        Loaded xarray Datasets.
    """
    files_dict = {}
    datasets_dict = {}

    # Ensure local folder exists
    os.makedirs(local_base, exist_ok=True)

    # Connect to S3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    bucket = "imos-data"

    for mooring in moorings_list:
        region = MOORING_REGIONS.get(mooring)
        if region is None:
            print(f"Unknown mooring: {mooring}, skipping.")
            continue

        # Construct the filename (S3 standard naming)
        filename = f"IMOS_ANMN-{region}_{mooring}_FV02_{data_type}_END-20240923_C-20250125.nc"
        local_path = os.path.join(local_base, f"{region}_{mooring}_{filename}")

        # Download if not cached
        if not os.path.exists(local_path) or not cache:
            try:
                s3.download_file(bucket, f"IMOS/ANMN/{region}/{mooring}/{data_type}/{filename}", local_path)
                print(f"Downloaded {mooring} from S3.")
            except Exception as e:
                print(f"Failed to download {mooring}: {e}")
                continue

        # Load dataset
        try:
            ds = xr.open_dataset(local_path)
            datasets_dict[mooring] = ds
            files_dict[mooring] = local_path
        except Exception as e:
            print(f"Failed to load {mooring}: {e}")

    if not datasets_dict:
        print("No datasets were loaded. Check your moorings or S3 paths.")

    return files_dict, datasets_dict


# -

def extract_timeseries_df(ds: xr.Dataset, sigclip=5, save=False):
    """From the given hourly-timeseries Dataset, extract a timeseries of temperature
    Filter out only values that are
    * Not from ADCP instruments
    * Within 10m of the deepest nominal depth in the dataset

    Parameters
    ----------
    ds: xarray.Dataset
        The input dataset
    sigclip: bool or numeric
        If numeric, clip timeseries to this number of standard deviations from the mean.
        If True, use 5 x stddev
        If None, False or 0, don't clip
    save: bool
        If True, save timeseries to a CSV file in the default local data directory

    Return a pandas DataFrame containing TIME, TEMP and DEPTH
    """

    # Find the index of all the non-ADCP instruments
    is_adcp = ds.instrument_id.str.find("ADCP") > 0
    i_adcp = [i for i in range(len(ds.INSTRUMENT)) if is_adcp[i]]

    # Boolean to select OBSERVATIONs from non-ADCP instruments
    inst_filter = ~ds.instrument_index.isin(i_adcp)

    # Boolean to select deep measurements
    dmax = ds.NOMINAL_DEPTH.values.max()
    dmin = dmax - 10.
    depth_filter = ds.DEPTH > dmin

    ii = inst_filter & depth_filter
    df = pd.DataFrame({"TIME": ds.TIME[ii],
                       "TEMP": ds.TEMP[ii],
                       "DEPTH": ds.DEPTH[ii]})

    # Apply sigma clipping
    if sigclip:
        nsig = 5 if sigclip is True else sigclip
        mean = df.TEMP.mean()
        std = df.TEMP.std()
        df.TEMP.mask(abs(df.TEMP-mean) >= nsig*std, inplace=True)

    # Save to a CSV file
    if save:
        csv_path = os.path.join(DATA_DIR, f"{ds.site_code}_TEMP_{dmin:.0f}-{dmax:.0f}m.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved timeseries to {csv_path}")

    return df


def load_all_timeseries():
    """Load hourly data and extract timeseries for all sites, saving to CSV locally.
    Return a dictionary mapping site => timeseries (pd.DataFrame).
    """

    hourly_files, hourly_datasets = load_data_products()
    temp_timeseries = dict()
    for mooring in hourly_datasets.keys():
        ds = hourly_datasets[mooring]
        df = extract_timeseries_df(ds, save=True)
        temp_timeseries[mooring] = df

    return temp_timeseries


def create_modelling_data(mooring_csv):
    """
    Preprocesses the mooring temp series CSV data along with the indexes data.
    """
    dataframes = {}
    dataframes["Upwelling"] = pd.read_csv(mooring_csv)
    dataframes["Upwelling"]["date"] = pd.to_datetime(dataframes["Upwelling"]["TIME"])
    dataframes["Upwelling"] = dataframes["Upwelling"][["TEMP", "DEPTH", "date"]].rename(columns={"TEMP": "Mooring Temp", "DEPTH": "Mooring Depth"})

    files = {
        "SAM": "../Datasets/SAM_index.csv",
        "ENSO": "../Datasets/SOI_index.csv",
        "IOD": "../Datasets/iod_index.csv",
        "Polar_vortex": "../Datasets/Vortex_datasets.csv"
    }

    for k, v in files.items():
        dataframes[k] = pd.read_csv(v)

    for df in "Upwelling SAM ENSO IOD".split():
        dataframes[df]["date"] = pd.to_datetime(dataframes[df]["date"], dayfirst=True)
        dataframes[df]["Year"] = dataframes[df]["date"].dt.year
        dataframes[df]["Month"] = dataframes[df]["date"].dt.to_period("M")
        dataframes[df] = dataframes[df].groupby("Month").mean(numeric_only=True)

    dataframes["Polar_vortex"] = dataframes["Polar_vortex"].dropna().copy()
    dataframes["Polar_vortex"]["Year"] = dataframes["Polar_vortex"]["Year"].astype(int)
    dataframes["Polar_vortex"] = dataframes["Polar_vortex"].set_index("Year")

    data = pd.DataFrame(index=dataframes["Upwelling"].index)
    for k, v in dataframes.items():
        if k != "Polar_vortex":
            data = data.merge(v.drop("Year", axis=1), left_index=True, right_index=True)

    rename_dict = {'Mooring Temp': "Mooring Temp",
                   'Mooring Depth': "Mooring Depth",
                   'sam_index': "SAM",
                   'soi_index': "ENSO",
                   'iod_index': "IOD",
                   'S-Tmode_Lim_et_al_2018': "Polar vortex 1",
                   'Sep-Nov[U]_60S10hPa_JRA55': "Polar vortex 2"}

    data = data.rename(columns=rename_dict)
    data = data.dropna(subset=["Mooring Temp"])
    # Separate features (climatic indices) and target (upwelling) variables
    X = data[['SAM', 'ENSO', 'IOD',]] # 'Polar vortex 1', 'Polar vortex 2']]
    y = data['Mooring Temp']

    return data, X, y


def create_regression_model(X, y, test_size=0.35, random_state=42, model=LinearRegression, mooring_id=None):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a linear regression model
    model = model()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model:", str(model).replace("()", ""))
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    # Plot the predicted vs. actual upwelling values
    # plt.scatter(y_test, y_pred)
    sns.regplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Upwelling")
    plt.ylabel("Predicted Upwelling")
    title = "Actual vs. Predicted Upwelling"
    if mooring_id:
        title = mooring_id + " " + title
    plt.title(title)
    plt.show()

    return model, mse, r2
