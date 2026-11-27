from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def nyc_addr_convert(in_addr:str):
    geolocator = Nominatim(user_agent="nyc_address_converter_app", timeout=10)
    try:
        location = geolocator.geocode(in_addr, country_codes="us")
        if location:
            print(f"{in_addr} was found with success")
            print(f"Parsed address: {location.address}")
            print(f"Latitude:  {location.latitude}")
            print(f"Longitude: {location.longitude}")
            return location.latitude, location.longitude
        else:
            print(f"{in_addr} was found with failure")
            return None
    except GeocoderTimedOut:
        print("Request timeout.")
    except GeocoderServiceError as e:
        print(f"Error from server: {e}")
    return None
