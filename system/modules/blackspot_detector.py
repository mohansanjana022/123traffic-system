
# import pandas as pd

# def detect_blackspots(file):

#     df = pd.read_csv(file)

#     # check coordinates exist
#     if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
#         raise ValueError("Dataset must contain LATITUDE and LONGITUDE columns")

#     # group by location
#     grouped = (
#         df.groupby(["LATITUDE","LONGITUDE"])
#         .size()
#         .reset_index(name="Accident_Count")
#         .sort_values("Accident_Count", ascending=False)
#     )

#     return grouped.head(10)

from geopy.geocoders import Nominatim
import pandas as pd

geolocator = Nominatim(user_agent="traffic_ai")

def get_location(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        return location.address
    except:
        return "Unknown"

def detect_blackspots(file):

    df = pd.read_csv(file)
    df = df[(df["LATITUDE"] != 0) & (df["LONGITUDE"] != 0)]

    grouped = (
        df.groupby(["LATITUDE","LONGITUDE"])
        .size()
        .reset_index(name="Accident_Count")
        .sort_values("Accident_Count", ascending=False)
        .head(10)
    )

    # convert coordinates → place name
    grouped["Location_Name"] = grouped.apply(
        lambda x: get_location(x["LATITUDE"], x["LONGITUDE"]), axis=1
    )

    return grouped