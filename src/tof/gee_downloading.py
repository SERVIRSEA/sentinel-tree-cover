import ee
import datetime
import numpy as np
import copy
from typing import List, Tuple
from google.auth import default
from google.api_core import retry, exceptions
import requests
import os
import io
import google.auth
from numpy.lib.recfunctions import structured_to_unstructured
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
#import cloud_removal
import hickle as hkl
from skimage.transform import resize
import math
from src.downloading.utils import calculate_and_save_best_images
from src.preprocessing.slope import calcSlope

# Initialize the Earth Engine module.
ee.Initialize()

def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    credentials, project = default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(image, roi, patch_size) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.
    
    Args:
        image: Image to get the patch from.
        roi: Region of interest as an ee.Geometry.
        patch_size: Size in pixels of the surrounding square patch.
    
    Raises:
        requests.exceptions.RequestException
    
    Returns: 
        The requested patch of pixels as a NumPy array with shape (width, height, bands).
    """
    url = image.getDownloadURL(
        {
            "region": roi,
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        }
    )

    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True)

def to_int16(array: np.array) -> np.array:
    '''Converts a float32 array to uint16'''
    assert np.min(array) >= 0, np.min(array)
    assert np.max(array) <= 1, np.max(array)

    array = np.clip(array, 0, 1)
    array = np.trunc(array * 65535)
    assert np.min(array >= 0)
    assert np.max(array <= 65535)

    return array.astype(np.uint16)

def extract_dates(image_collection):
    # Extract dates from image collection
    dates = image_collection.aggregate_array('system:time_start').getInfo()
    dates = [datetime.datetime.utcfromtimestamp(ms / 1000).strftime('%Y-%m-%d') for ms in dates]
    filenames = image_collection.aggregate_array('system:index').getInfo()
    return dates, filenames

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Makes a (min_x, min_y, max_x, max_y) bounding box that
       is expanded by a given number of pixels.

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): Number of pixels to expand by

       Returns:
            bbx (list): Expanded [min_x, min_y, max_x, max_y]
    """
    multiplier = 1 / 360  # Sentinel-2 pixel size in decimal degrees (adjust if necessary)
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def _check_for_alt_img(probs, dates, date):
    # Checks to see if there is an image within win days that has
    # Less local cloud cover. If so, remove the higher CC image.
    # This is done to avoid having to remove (interpolate)
    # All of an image for a given tile, as this can cause artifacts
    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341, 410]

    begins = (end - date)
    begins[begins < 0] = 999
    month_start = begin[np.argmin(begins)]
    month_end = end[np.argmin(begins)]
    lower = month_start
    upper = month_end
    upper = np.maximum(date + 28, upper)
    lower = np.minimum(date - 28, lower)

    candidate_idx = np.argwhere(
        np.logical_and(np.logical_and(dates >= lower, dates <= upper),
                       dates != date))
    candidate_probs = probs[candidate_idx]
    if len(candidate_probs) == 0:
        return False

    idx = np.argwhere(dates == date).flatten()
    begin_prob = probs[idx]
    if np.min(candidate_probs) < (begin_prob - 0.20):
        return True
    else:
        return False


def downsample_to_40m(bands_20m, nSteps):
    """
    Downsample 20m resolution bands to 40m.
    
    Parameters:
    bands_20m (numpy.ndarray): Input array of shape (steps, height, width, n_bands)
    nSteps (int): Number of time steps
    
    Returns:
    numpy.ndarray: Downsampled array of shape (steps, height//2, width//2, n_bands)
    """
    steps, height, width, n_bands = bands_20m.shape
    bands_40m = np.zeros((steps, height // 2, width // 2, n_bands), dtype=bands_20m.dtype)
    
    for step in range(steps):
        for band in range(n_bands):
            bands_40m[step, :, :, band] = resize(bands_20m[step, :, :, band], (height // 2, width // 2), order=0, preserve_range=True)
    
    return bands_40m

def adjust_dimensions(s220img, s240img):
    """
    Adjust dimensions of downsampled 40m bands to match 20m bands.
    
    Parameters:
    s220img (numpy.ndarray): Original 20m bands of shape (steps, height, width, n_bands)
    s240img (numpy.ndarray): Downsampled 40m bands of shape (steps, height//2, width//2, n_bands)
    
    Returns:
    numpy.ndarray: Adjusted 40m bands of shape (steps, height, width, n_bands)
    """
    img_40 = s240img.repeat(2, axis=1).repeat(2, axis=2)

    if (s220img.shape[1] > img_40.shape[1]) or (s220img.shape[2] > img_40.shape[2]):
        img_40 = resize(img_40, (s220img.shape[0], s220img.shape[1], s220img.shape[2], img_40.shape[-1]), order=0)

    if img_40.shape[1] > s220img.shape[1]:
        to_remove = (img_40.shape[1] - s220img.shape[1])
        if to_remove == 2:
            img_40 = img_40[:, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis=1).repeat(2, axis=2)
            img_40 = img_40[:, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, img_40.shape[-1]))
            img_40 = np.mean(img_40, axis=(2, 4))

    if img_40.shape[2] > s220img.shape[2]:
        to_remove = (img_40.shape[2] - s220img.shape[2])
        if to_remove == 2:
            img_40 = img_40[:, :, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis=1).repeat(2, axis=2)
            img_40 = img_40[:, :, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, img_40.shape[-1]))
            img_40 = np.mean(img_40, axis=(2, 4))

    return img_40


def download_sentinel_2_new(fnames,cloud_bbx, dates, year, maxclouds=0.5):
    """ Downloads the L2A sentinel layer with 10 and 20 meter bands

        Parameters:
         bbox (list): output of calc_bbox
         clean_steps (list): list of steps to filter download request
         epsg (float): EPSG associated with bbox
         time (tuple): YY-MM-DD - YY-MM-DD bounds for downloading

        Returns:
         img (arr):
         img_request (obj):
    """
    start_date, end_date = dates
    QA_BAND = 'cs_cdf'

    # Define the region of interest
    initial_bbx = cloud_bbx #[cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)

    roi = ee.Geometry.Polygon([
        [
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]]
        ]
    ])

    # Load Sentinel-2 image collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxclouds * 100))\
        .filter(ee.Filter.inList("system:index",ee.List(fnames.tolist())))
        
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.map(lambda img: img.addBands(csPlus.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
    
    
    s210 = s2.select(["B2", "B3", "B4", "B8"])
    s220 = s2.select(["B5", "B6", "B7", "B8A", "B11", "B12"])
    s2cloud = s2.select(['cs_cdf'])
    
    nSteps  = int(s2.size().getInfo())
    
    #print(ee.Image(s210.toBands()).bandNames().getInfo())




    patchsize10 = 600
    patch = get_patch(s210.toBands(), roi, patchsize10)    
    s210img = structured_to_unstructured(patch)
    #print(s210img.shape)
    num_bands = 4
    new_shape = (nSteps, patchsize10, patchsize10, num_bands)
    result = np.empty(new_shape)

    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands):
            result[i, :, :, j] = s210img[:, :, i * num_bands + j]
    s210img = result / 10000
    #print("new shape =====================",s210img.shape)
    #s210img = s210img.reshape((patchsize10, patchsize10, 4, nSteps))
    #s210img = s210img.transpose(3, 0, 1, 2) / 10000
    
    
    
    patchsize20 = 300
    patch = get_patch(s220.toBands(), roi, patchsize20)    
    s220img = structured_to_unstructured(patch)

    #s220img = s220img.reshape((patchsize20, patchsize20, 6, nSteps))
    #s220img = s220img.transpose(3, 0, 1, 2) / 10000
    num_bands = 6
    new_shape = (nSteps, patchsize20, patchsize20, num_bands)
    result = np.empty(new_shape)

    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands):
            result[i, :, :, j] = s220img[:, :, i * num_bands + j]
    s220img = result / 10000



    patch = get_patch(s2cloud.toBands(), roi, patchsize20)    
    s2cloud = structured_to_unstructured(patch)
    #print("s2clouds:",s2cloud.shape)
    s2cloud = s2cloud.reshape((patchsize20, patchsize20, 1, nSteps))
    s2cloud = s2cloud.transpose(3, 0, 1, 2)
    s2cloud = np.where(s2cloud > maxclouds, 0, 1)
    
    # Downsample to 40m
    s240img = downsample_to_40m(s220img, nSteps)
    #print("Shape of 40m bands before adjustment:", s240img.shape)

    # Adjust dimensions to match 20m bands
    s240img = adjust_dimensions(s220img, s240img)
    #print("Shape of 40m bands after adjustment:", s240img.shape)
	
    s220img = np.concatenate([s220img, s240img], axis=-1)
    #print("Shape of 40m bands after adjustment:", s220img.shape)

    # Convert 10m bands to np.float32, ensure correct dimensions
    if not isinstance(s210img.flat[0], np.floating):
        assert np.max(s210img) > 1
        s210img = np.float32(s210img) / 65535.
        assert np.max(s210img) <= 1
        assert s210img.dtype == np.float32

    #print(s210img)
    # Convert 10m bands to np.float32, ensure correct dimensions
    if not isinstance(s220img.flat[0], np.floating):
        assert np.max(s220img) > 1
        s220img = np.float32(s220img) / 65535.
        assert np.max(s220img) <= 1
        assert s220img.dtype == np.float32
    #print(np.max(s210img))
    
    s210img = np.clip(s210img, 0, 1)
    s220img = np.clip(s220img, 0, 1)
    
    """
    first_timestep_data = s210img[1,:,:] 
    plt.figure(figsize=(10, 8))
    plt.imshow(first_timestep_data[:,:,0:3], cmap='gray')
    #plt.imshow(first_timestep_data, cmap='gray')
    plt.colorbar()
    plt.show()
    """

    return s210img, s220img, s2cloud, dates
    


def identify_clouds_big_bbx(cloud_bbx, dates, year, maxclouds=0.5):
    """
    Downloads and calculates cloud cover and shadow

    Parameters:
     cloud_bbx (list): Bounding box coordinates [minLng, minLat, maxLng, maxLat]
     dates (tuple): Start and end date for filtering images ('YYYY-MM-DD', 'YYYY-MM-DD')
     year (int): Year for date calculations
     maxclouds (float): Maximum cloud cover percentage

    Returns:
     cloudImage (np.ndarray): Cloud images
     cloud_percent (np.ndarray): Cloud percentage
     cloud_dates (np.ndarray): Dates of the cloud images
     local_clouds (np.ndarray): Local cloud data
    """
    start_date, end_date = dates
    QA_BAND = 'cs_cdf'

    # Define the region of interest
    initial_bbx = cloud_bbx #[cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)
    print(cloud_bbx_expanded)
    roi = ee.Geometry.Polygon([
        [
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]]
        ]
    ])

    # Load Sentinel-2 image collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxclouds * 100))

    print("size",s2.size().getInfo())
    # Extract dates of the images
    cloud_dates, filenames = extract_dates(s2)

    base_date = f'{year - 1}-12-31'  # Base date to calculate day of year
    cloud_dates = convert_to_day_of_year(cloud_dates, base_date)
    
    # Apply cloud mask
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.map(lambda img: img.addBands(csPlus.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
    clouds = ee.Image(s2.select([QA_BAND]).toBands())
    patch = get_patch(clouds, roi, 160)

    cloudImage = structured_to_unstructured(patch)

    # Filter out arrays with -inf or nan values
    valid_indices = ~np.isnan(cloudImage).any(axis=(0, 1)) & ~np.isinf(cloudImage).any(axis=(0, 1))
    cloud_dates = np.array(cloud_dates)[valid_indices]
    filenames = np.array(filenames)[valid_indices]
    print("file names",filenames)

    cloudImage = cloudImage[:, :, valid_indices]
    cloudImage = np.transpose(cloudImage, (2, 0, 1))

    mid_idx = cloudImage.shape[1] // 2
    mid_idx_y = cloudImage.shape[2] // 2

    # Apply the clear threshold
    cloudImage[cloudImage > 0.6] = np.nan
    cloudImage[cloudImage <= 0.6] = 0

    cloud_percent = np.nanmean(cloudImage, axis=(1, 2))
    print("cloud percent",cloud_percent)

    local_clouds = np.copy(cloudImage[:, mid_idx - 15:mid_idx + 15, mid_idx_y - 15:mid_idx_y + 15])
    for i in range(cloudImage.shape[0]):
        clouds = local_clouds[i]
        local_clouds[i] = binary_dilation(clouds)

    local_clouds = np.nanmean(local_clouds, axis=(1, 2))

    cloudImage[np.isnan(cloudImage)] = 1

    cloud_steps = np.argwhere(cloud_percent > 0.5)
    cloudImage = np.delete(cloudImage, cloud_steps, 0)
    cloud_percent = np.delete(cloud_percent, cloud_steps)
    cloud_dates = np.delete(cloud_dates, cloud_steps)
    filenames = np.delete(filenames, cloud_steps)
    local_clouds = np.delete(local_clouds, cloud_steps)

    print("steps",cloud_steps)
    print("dates",cloud_dates)

    cloud_percent[cloud_percent > 0.4] = ((0.25 * cloud_percent[cloud_percent > 0.4] +
                                           0.75 * local_clouds[cloud_percent > 0.4]))

    cloud_steps = np.argwhere(cloud_percent > maxclouds)
    cloudImage = np.delete(cloudImage, cloud_steps, 0)
    cloud_percent = np.delete(cloud_percent, cloud_steps)
    cloud_dates = np.delete(cloud_dates, cloud_steps)
    filenames = np.delete(filenames, cloud_steps)
    local_clouds = np.delete(local_clouds, cloud_steps)

    to_remove = []
    for i, x, l in zip(cloud_dates, local_clouds, range(len(cloud_dates))):
        if x < 0.60 and _check_for_alt_img(local_clouds, cloud_dates, i):
            to_remove.append(l)

    if to_remove:
        cloud_dates = np.delete(cloud_dates, to_remove)
        filenames = np.delete(filenames, to_remove)
        cloudImage = np.delete(cloudImage, to_remove, 0)
        cloud_percent = np.delete(cloud_percent, to_remove)
        local_clouds = np.delete(local_clouds, to_remove)

    cloudImage = cloudImage.astype(np.float32)
    assert np.max(cloudImage) <= 1, np.max(cloudImage)
    assert cloudImage.dtype == np.float32

    return cloudImage, cloud_percent, np.array(cloud_dates), local_clouds, filenames

def generate_date_range(year):
    start_date = f'{year - 1}-11-15'
    end_date = f'{year + 1}-02-15'
    return start_date, end_date

def convert_to_day_of_year(dates, base_date):
    base_date = datetime.datetime.strptime(base_date, '%Y-%m-%d')
    day_of_year = [(datetime.datetime.strptime(date, '%Y-%m-%d') - base_date).days for date in dates]
    return np.array(day_of_year)


"""
def process_cloud_data(cloud_bbx, dates, year, maxclouds=0.5):
    cloud_probs, cloud_percent, all_dates, all_local_clouds, filenames = identify_clouds_big_bbx(cloud_bbx, dates, year, maxclouds)

    print("final cloud image\n", cloud_probs)
    print("final cloud percent\n", cloud_percent)
    print("final local clouds\n", all_local_clouds)
    print("final doy\n", all_dates)
    print("filenames\n", filenames)

    cloud_probs = cloud_probs * 100
    cloud_probs[cloud_probs > 100] = np.nan
    cloud_percent = np.nanmean(cloud_probs, axis=(1, 2))
    cloud_percent = cloud_percent / 100

    print("cloud percent\n\n", cloud_percent)

    local_clouds = np.copy(all_local_clouds)
    image_dates = np.copy(all_dates)
    image_names = np.copy(filenames)
    print("dates\n\n", image_dates)

    to_remove = cloud_removal.subset_contiguous_sunny_dates(image_dates, cloud_percent)
    print("to remove\n\n", to_remove)

    if len(to_remove) > 0:
        clean_dates = np.delete(image_dates, to_remove)
        clean_filenames = np.delete(filenames, to_remove)
        cloud_probs = np.delete(cloud_probs, to_remove, 0)
        cloud_percent = np.delete(cloud_percent, to_remove)
        local_clouds = np.delete(local_clouds, to_remove)
    else:
        clean_dates = image_dates
        clean_filenames = filenames

    if len(clean_dates) >= 11:
        clean_dates = np.delete(clean_dates, 5)
        clean_filenames = np.delete(clean_filenames, 5)
        cloud_probs = np.delete(cloud_probs, 5, 0)
        cloud_percent = np.delete(cloud_percent, 5)
        local_clouds = np.delete(local_clouds, 5)

        _ = cloud_removal.print_dates(clean_dates, cloud_percent)


    lowest_three_local = np.argpartition(all_local_clouds, 3)[:3]
    lowest_four_local = np.argpartition(all_local_clouds, 4)[:4]

    criteria1 = (np.sum((local_clouds <= 0.3)) < 3)
    criteria2 = (np.sum((local_clouds <= 0.4)) < 4)
    criteria3 = len(local_clouds) <= 8
    criteria2 = np.logical_or(criteria2, criteria3)

    if criteria1 or criteria2:
        if len(clean_dates) <= 9:
            lowest = lowest_four_local if criteria2 else lowest_three_local
            lowest_dates = image_dates[lowest]
            lowest_filenames = filenames[lowest]
            existing_imgs_in_local = [x for x in clean_dates if x in image_dates[lowest]]
            images_to_add = [x for x in lowest_dates if x not in clean_dates]
            filenames_to_add = [x for x in lowest_filenames if x not in clean_filenames]
            print(f"Adding these images: {images_to_add}")
            clean_dates = np.concatenate([clean_dates, images_to_add])
            clean_filenames = np.concatenate([clean_filenames, filenames_to_add])
            clean_dates = np.sort(clean_dates)
        if len(clean_dates) <= 9:
            imgs_to_add = 9 - len(clean_dates)
            lowest_five_local = np.argpartition(all_local_clouds, 5)[:5]
            images_to_add = [x for x in image_dates[lowest_five_local] if x not in clean_dates][:imgs_to_add]
            filenames_to_add = [x for x in filenames[lowest_five_local] if x not in clean_filenames][:imgs_to_add]
            clean_dates = np.concatenate([clean_dates, images_to_add])
            clean_filenames = np.concatenate([clean_filenames, filenames_to_add])
            clean_dates = np.sort(clean_dates)

        for i, x, y in zip(clean_dates, cloud_percent, local_clouds):
            print(i, x, y)


    return cloud_probs, clean_dates, clean_filenames
"""

def toNatural(img):
  """Function to convert from dB to natural"""
  return ee.Image(10.0).pow(img.select(0).divide(10.0));

def toDB(img):
  """ Function to convert from natural to dB """
  return ee.Image(img).log10().multiply(10.0);

def addRatio(img):
  geom = img.geometry()
  vv = toNatural(img.select(['VV'])).rename(['VV']);
  vh = toNatural(img.select(['VH'])).rename(['VH']);
  ratio = vh.divide(vv).rename(['ratio']);
  return ee.Image(ee.Image.cat(vh,vv).copyProperties(img,['system:time_start'])).clip(geom).copyProperties(img);


def terrainCorrection(image):
    date = ee.Date(image.get('system:time_start'))
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom)    # 30m srtm
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

    #Article ( numbers relate to chapters)
    #2.1.1 Radar geometry
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000).get('aspect')

    #2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    # 2.1.3 Model geometry
    # reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    #convert all to radians
    phi_rRad = phi_r.multiply(math.pi / 180)
    alpha_sRad = alpha_s.multiply(math.pi / 180)
    theta_iRad = theta_i.multiply(math.pi / 180)
    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

    # slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

    # slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

    # local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
    theta_liaDeg = theta_lia.multiply(180 / math.pi)

    # 2.2
    # Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos())
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

    # Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
    denominator = (ninetyRad.subtract(theta_iRad)).tan()
    volModel = (nominator.divide(denominator)).abs()

    # apply model
    gamma0_Volume = gamma0.divide(volModel)
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

    # we add a layover/shadow maskto the original implmentation
    # layover, where slope > radar viewing angle
    alpha_rDeg = alpha_r.multiply(180 / math.pi)
    layover = alpha_rDeg.lt(theta_i);

    # shadow where LIA > 90
    shadow = theta_liaDeg.lt(85)

    # calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad)\
			    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

    return output.select(['VH', 'VV'], ['VH', 'VV']).set("system:time_start",date).clip(imgGeom ).copyProperties(image)


def process_sentinel_1_tile(sentinel1: np.ndarray,
                            dates: np.ndarray) -> np.ndarray:
    """Converts a (?, X, Y, 2) Sentinel 1 array to a regular monthly grid

        Parameters:
         sentinel1 (np.array):
         dates (np.array):

        Returns:
         s1 (np.array)
    """

    s1, _ = calculate_and_save_best_images(sentinel1, dates)
    monthly = np.zeros((12, sentinel1.shape[1], sentinel1.shape[2], 2), dtype = np.float32)
    index = 0
    for start, end in zip(
            range(0, 24 + 2, 24 // 12),  #0, 72, 6
            range(24 // 12, 24 + 2, 24 // 12)):  # 6, 72, 6
        monthly[index] = np.median(s1[start:end], axis=0)
        index += 1

    return monthly

# Function to get the Julian day for the 15th of each month in a given year
def get_mid_month_julian_days(year):
    mid_month_days = []
    for month in range(1, 13):
        date = datetime.datetime(year, month, 15)
        julian_day = date.timetuple().tm_yday
        mid_month_days.append(julian_day)
    return mid_month_days

def download_sentinel_1_composite(cloud_bbx, year):

    # Define the region of interest
    initial_bbx = cloud_bbx #[cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)

    roi = ee.Geometry.Polygon([
        [
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]]
        ]
    ])

    months = ee.List.sequence(1,12,1)
	
    start = ee.Date.fromYMD(year,1,1)
    end	 = ee.Date.fromYMD(year,12,31)
	
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi)\
												.filterDate(start,end)\
												.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
												.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
												.filter(ee.Filter.eq('instrumentMode', 'IW'))\
												.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
												
    s1 = s1.map(terrainCorrection).map(addRatio).select(["VV","VH"])
    
    # Function to create a median composite for a given month
    def get_monthly_median(month):
        start_month = ee.Date.fromYMD(year, month, 1)
        end_month = start_month.advance(1, 'month')
        monthly_composite = s1.filterDate(start_month, end_month).median()
        return monthly_composite.set('month', month)

    # Create monthly median composites
    monthly_composites = ee.ImageCollection(months.map(lambda m: get_monthly_median(ee.Number(m))).flatten())
    
    patchsize = 300
    patch = get_patch(monthly_composites.toBands(), roi, patchsize)    
    s1img = structured_to_unstructured(patch)
    
    num_bands = 2
    nSteps = 12
    new_shape = (nSteps, patchsize, patchsize, num_bands)
    result = np.empty(new_shape)

    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands):
            result[i, :, :, j] = s1img[:, :, i * num_bands + j]
    s1img = result.clip(0,1) 
    
    s1img  = s1img.repeat(2, axis=1).repeat(2, axis=2)
    
    dates = get_mid_month_julian_days(year)

    s1img = process_sentinel_1_tile(s1img, dates)

    return s1img, dates
    #print(monthly_composites.size().getInfo())


def download_dem(cloud_bbx):
    """ Downloads the DEM layer from Sentinel hub

        Parameters:
         bbox (list): output of calc_bbox
         epsg (float): EPSG associated with bbox

        Returns:
         dem_image (arr):
    """
    # Define the region of interest
    initial_bbx = [cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)

    roi = ee.Geometry.Polygon([
        [
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[1]],
            [cloud_bbx_expanded[2], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[3]],
            [cloud_bbx_expanded[0], cloud_bbx_expanded[1]]
        ]
    ])
    
    dem = ee.Image("CGIAR/SRTM90_V4")
	
    patchsize = 202
    patch = get_patch(dem, roi, patchsize)    
    dem_image = structured_to_unstructured(patch).squeeze()


    # Convert the uint16 data to float32
    dem_image = dem_image - 12000
    dem_image = dem_image.astype(np.float32)
    width = dem_image.shape[0]
    height = dem_image.shape[1]

    # Apply median filter, calculate slope
    #dem_image = median_filter(dem_image, size=5)
    dem_image = calcSlope(dem_image.reshape((1, width, height)),
                          np.full((width, height), 10),
                          np.full((width, height), 10),
                          zScale=1,
                          minSlope=0.02)
    dem_image = dem_image.reshape((width, height, 1))

    dem_image = dem_image[1:width - 1, 1:height - 1, :]
    dem_image = dem_image.squeeze()

    dem_image = dem_image.repeat(3, axis=0).repeat(3, axis=1)

    return dem_image

"""

# Example usage
year = 2023
dates = generate_date_range(year)
cloud_bbx = [94.53938593874919,20.702644427517065]  # Define your bounding box coordinates
maxclouds = 0.7


fpath = "/home/ate-laptop/sig/treecover/sentinel_original/sentinel-tree-cover/project-monitoring/tiles/2022/1/1/"

clouds_file = fpath + "raw/clouds/clouds_1X1Y.hkl"
clean_steps_file = fpath + "raw/clouds/clean_steps_1X1Y.hkl"
clean_filename = fpath + "raw/clouds/clean_fname_1X1Y.hkl"


#if os.path.exists(clouds_file) and os.path.exists(clean_steps_file):
#    cloud_probs = hkl.load(clouds_file)
#    clean_dates = hkl.load(clean_steps_file)
#    clean_filenames = np.array(hkl.load(clean_filename))
#    print(f"Loaded data from {clouds_file} and {clean_steps_file}.")
#else:
cloud_probs, clean_dates, clean_filenames = process_cloud_data(cloud_bbx, dates, year, maxclouds)

hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
hkl.dump(clean_dates, clean_steps_file, mode='w', compression='gzip')
hkl.dump(clean_filenames, clean_filename, mode='w', compression='gzip')
# print(f"Downloading {len(clean_dates)} of {len(clean_dates)} total steps")

s2_10_file = fpath + "raw/s2_10/1X1Y.hkl"
s2_20_file = fpath + "raw/s2_20/1X1Y.hkl"
cloud_mask_file = fpath + "raw/clouds/cloudmask_1X1Y.hkl"
s2_dates_file = fpath + "raw/misc/s2_dates_1X1Y.hkl"


#if os.path.exists(s2_10_file) and os.path.exists(s2_20_file) and os.path.exists(cloud_mask_file):
#    s2_10 = hkl.load(s2_10_file)
#    s2_20 = hkl.load(s2_20_file)
#    clm = hkl.load(cloud_mask_file)

#else:
s2_10, s2_20, clm,s2_dates  = download_sentinel_2_new(clean_filenames,cloud_bbx, dates, year, maxclouds)

hkl.dump(to_int16(s2_10), s2_10_file, mode='w', compression='gzip')
hkl.dump(to_int16(s2_20), s2_20_file, mode='w', compression='gzip')
hkl.dump(clm, cloud_mask_file, mode='w', compression='gzip')
hkl.dump(clean_dates, s2_dates_file, mode='w', compression='gzip')


s1_file = fpath +"raw/s1/1X1Y.hkl"
s1_dates_file = fpath +"raw/misc/s1_dates_1X1Y.hkl"

#if os.path.exists(s1_file):
#s1 = hkl.load(s1_file)
#else:
s1, s1_dates = download_sentinel_1_composite(cloud_bbx,  year)
hkl.dump(to_int16(s1), s1_file, mode='w', compression='gzip')
hkl.dump(s1_dates, s1_dates_file, mode='w', compression='gzip')

dem_file = fpath +"raw/misc/dem_1X1Y.hkl"
dem = download_dem(cloud_bbx)
hkl.dump(dem, dem_file, mode='w', compression='gzip')
"""
