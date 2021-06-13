from coastsat.SDS_download import *

from utils.print_utils import printProgress, printError, printSuccess
from utils.name_utils import geotifFileName, pickleDumpName


def retrieve_median_sar(settings):

    # initialise connection with GEE server
    ee.Initialize()
    printProgress('connected to GEE')

    pixel_size = settings['pixel_size']
    sat_name = settings['sat_name']
    site_name = settings['site_name']
    median_dir_path = settings['median_dir_path']
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]
    polygon = ee.Geometry.Polygon(settings['polygon'])

    if date_start > date_end:
        printError('you cannot have end date before the start date')

    # create a new directories as required
    if not os.path.exists(median_dir_path):
        os.makedirs(median_dir_path)

    # create directories store the image geotiffs and metadata
    sar_dir_path = os.path.join(median_dir_path, sat_name)
    if not os.path.exists(sar_dir_path):
        os.makedirs(sar_dir_path)
    meta_dir_path = os.path.join(sar_dir_path, 'meta')
    if not os.path.exists(meta_dir_path):
        os.makedirs(meta_dir_path)

    median_images_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
                   .filterBounds(polygon) \
                   .filterDate(date_start, date_end) \
                   .filter(ee.Filter.eq('instrumentMode', 'IW'))
    median_images_list = median_images_collection.toList(500)
    number_images = len(median_images_list.getInfo())

    printProgress(f'found {number_images} images')

    printProgress('downloading median image')
    median_image = median_images_collection.median()
    download_GEE_image(median_image, ee.Number(pixel_size),
                       settings['polygon'], sar_dir_path)

    # rename the file as the image is downloaded as 'data.tif'
    # locate download
    local_data = sar_dir_path + '\\data.tif'
    median_filename = geotifFileName(site_name, date_start, date_end, None)
    local_file_path = os.path.join(sar_dir_path, median_filename)

    try:
        os.rename(local_data, local_file_path)
    except:  # overwrite if already exists
        os.remove(local_file_path)
        os.rename(local_data, local_file_path)

    printProgress('writing metadata')
    gee_metadata = median_image.getInfo()
    epsg = int(gee_metadata['bands'][0]['crs'][5:])
    # metadata for .txt file
    txt_file_name = median_filename.replace('tif', 'txt')
    metadata_dict = {'file_name': median_filename,
                     'epsg': epsg,
                     'date_start': date_start,
                     'date_end': date_end,
                     'number_images': number_images}

    # write metadata as text file
    with open(os.path.join(meta_dir_path, txt_file_name), 'w') as f:
        for key in metadata_dict.keys():
            f.write('%s\t%s\n' % (key, metadata_dict[key]))

    printProgress('GEE connection closed')
    printSuccess('median image downloaded')

def retrieve_sar(settings):

    # initialise connection with GEE server
    ee.Initialize()
    printProgress('connected to GEE')

    pixel_size = settings['pixel_size']
    sat_name = settings['sat_name']
    site_name = settings['site_name']
    median_dir_path = settings['median_dir_path']
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]
    polygon = ee.Geometry.Polygon(settings['polygon'])

    if date_start > date_end:
        printError('you cannot have end date before the start date')

    # create a new directories as required
    head, tail = os.path.split(median_dir_path)
    images_dir_path = os.path.join(head, 'images')
    if not os.path.exists(images_dir_path):
        os.makedirs(images_dir_path)
    sar_dir_path = os.path.join(images_dir_path, sat_name)
    if not os.path.exists(sar_dir_path):
        os.makedirs(sar_dir_path)
    meta_dir_path = os.path.join(images_dir_path, 'meta')
    if not os.path.exists(meta_dir_path):
        os.makedirs(meta_dir_path)

    sar_images_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
                   .filterBounds(polygon) \
                   .filterDate(date_start, date_end) \
                   .filter(ee.Filter.eq('instrumentMode', 'IW'))

    sar_images_list = sar_images_collection.getInfo().get('features')
    number_images = len(sar_images_list)

    printProgress(f'downloading {number_images} sar images')

    for image_index in range(len(sar_images_list)):

        image_metadata = sar_images_list[image_index]
        epsg = int(image_metadata['bands'][0]['crs'][5:])

        # get time of acquisition (UNIX time) and convert to datetime
        image_time = image_metadata['properties']['system:time_start']
        image_timestamp = datetime.fromtimestamp(image_time / 1000, tz=pytz.utc)
        image_dt = image_timestamp.strftime('%Y%m%dT%H%M%S')

        sar_image = ee.Image(image_metadata['id'])

        image_filename = 'S1' + image_metadata['properties']['platform_number'] + \
                         '_' + image_metadata['properties']['orbitProperties_pass'][:3] + \
                         '_' + image_dt + '.tif'

        download_GEE_image(sar_image, ee.Number(pixel_size),
                           settings['polygon'], sar_dir_path)

        # rename the file as the image is downloaded as 'data.tif'
        # locate download
        local_data = sar_dir_path + '\\data.tif'
        local_file_path = os.path.join(sar_dir_path, image_filename)

        try:
            os.rename(local_data, local_file_path)
        except:  # overwrite if already exists
            os.remove(local_file_path)
            os.rename(local_data, local_file_path)

        # metadata for .txt file
        txt_file_name = image_filename.replace('tif', 'txt')
        metadata_dict = {'file_name': image_filename,
                         'epsg': epsg,
                         'date_start': date_start,
                         'date_end': date_end,
                         }

        # write metadata as text file
        with open(os.path.join(meta_dir_path, txt_file_name), 'w') as f:
            for key in metadata_dict.keys():
                f.write('%s\t%s\n' % (key, metadata_dict[key]))

    printProgress('GEE connection closed')
    printSuccess('images downloaded')


def retrieve_median_optical(settings):

    ee.Initialize()
    printProgress('connected to GEE')

    sat_name = settings['sat_name']
    polygon = settings['polygon']
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]
    site_name = settings['site_name']

    band_list = settings['bands'][sat_name]

    median_dir_path = settings['median_dir_path']
    if not os.path.exists(median_dir_path):
        os.makedirs(median_dir_path)
    filepaths = create_folder_structure(median_dir_path, sat_name)

    image_filename = {}

    if sat_name == 'L8':
        GEE_collection = 'LANDSAT/LC08/C01/T1_TOA'
    elif sat_name == 'L5':
        GEE_collection = 'LANDSAT/LT05/C01/T1_TOA'
    elif sat_name == 'S2':
        GEE_collection = 'COPERNICUS/S2'
    else:
        printError(f'unknown satellite: {sat_name}')

    median_image, number_images = get_median_image_optical(GEE_collection, settings)

    printProgress(f'found {number_images} images')

    image_metadata = median_image.getInfo()
    image_epsg = image_metadata['bands'][0]['crs'][5:]

    for band_key in band_list.keys():
        image_filename[band_key] = geotifFileName(site_name, date_start, date_end, band_key)

    printProgress('downloading median data for:')

    if sat_name[0] == 'L' and settings['coregistration'] is True:

#            displacement = Landsat_Coregistration(setting)
            # Apply XY displacement values from overlapping images to the median composite
#            median_image = median_image.displace(displacement, mode="bicubic")
            printProgress('co-registered')


    for index, band_key in enumerate(band_list.keys()):

        band_names = band_list[band_key][0]
        band_scale = band_list[band_key][1]
        band_number = index + 1
        band_file_path = filepaths[band_number]
        image_file_name = image_filename[band_key]

        local_data = filepaths[band_number] + '\\data.tif'
        local_file_path = os.path.join(band_file_path, image_file_name)

        printProgress(f'\t"{band_key}" bands:\t{band_names}')
        download_GEE_image(median_image, ee.Number(band_scale),
                           ee.Geometry.Polygon(polygon), band_file_path, bands=band_names)

        try:
            os.rename(local_data, local_file_path)
        except:  # overwrite if already exists
            os.remove(local_file_path)
            os.rename(local_data, local_file_path)

    base_file_name = image_file_name.replace('_'+band_key+'.tif', '')
    txt_file_name = base_file_name + '.txt'

    metadata_dict = {'file_name': base_file_name,
                     'epsg': image_epsg,
                     'date_start': date_start,
                     'date_end': date_end,
                     'number_images': number_images}

    with open(os.path.join(filepaths[0], txt_file_name), 'w') as f:
        for key in metadata_dict.keys():
            f.write('%s\t%s\n' % (key, metadata_dict[key]))

    printProgress('GEE connection closed')
    printSuccess('median image downloaded')

    return metadata_dict

def get_median_image_optical(collection, settings):

    sat_name = settings['sat_name']
    polygon = ee.Geometry.Polygon(settings['polygon'])
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]

    bands_list = []
    bands_dict = settings['bands'][sat_name]

    for band_key in bands_dict.keys():
        bands_list += bands_dict[band_key][0]

    bands_list = list(set(bands_list))

    def LandsatCloudScore(image):
        # Compute a cloud score band.
        cloud = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud')
        cloudiness = cloud.reduceRegion(
            reducer='mean',
            geometry=polygon,
            scale=30)
        return image.set(cloudiness)

    def LandsatCloudMask(image):
        # Compute a cloud score band.
        cloud = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud')
        cloudmask = cloud.lt(settings['LCloudThreshold'])
        masked = image.updateMask(cloudmask)
        return masked

    collect = ee.ImageCollection(collection)

    # Set bands to be extracted
    LC8_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'BQA']  ## Landsat 8
    LC7_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6_VCID_2', 'BQA']  ## Landsat 7
    LC5_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'BQA']  ## Landsat 5
    STD_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'temp', 'BQA']

    if sat_name == 'L5':
        ## Filter by time range and location
        collection = (collect.filterDate(date_start, date_end)
                      .filterBounds(polygon))

        # Apply Cloud Score layer to each image, then filter collection
        withCloudiness = collection.map(LandsatCloudScore)
        filteredCollection = (withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore']))
                              .select(LC5_BANDS, STD_NAMES))

        # Apply cloud masking to all images within collection then select bands
        maskedcollection = filteredCollection.map(LandsatCloudMask)
        filteredCollection_masked = maskedcollection.select(LC7_BANDS, STD_NAMES)

        ##Print Images in Collection
        # List Images in Collection
        image_list = filteredCollection.toList(500)
        image_count = len(image_list.getInfo())

        if settings['add_L7_to_L5'] == True:

            # Add Landsat 7 to Collection
            collect = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA')
            ## Filter by time range and location
            L7_collection = (collect.filterDate(date_start, date_end)
                             .filterBounds(polygon))


            # Apply Cloud Score layer to each image, then filter collection
            L7_withCloudiness = L7_collection.map(LandsatCloudScore)
            L7_filteredCollection = (L7_withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore']))
                                     .select(LC7_BANDS, STD_NAMES))


            # Apply cloud masking to all images within collection then select bands
            maskedcollection = filteredCollection.map(LandsatCloudMask)
            filteredCollection_masked = maskedcollection.select(LC7_BANDS, STD_NAMES)

            ##Print Images in Collection
            # List Images in Collection
            L7_image_list = L7_filteredCollection.toList(500)
            L7_count = len(L7_image_list.getInfo())

            # Merge collection with Landsat 5
            combined_collection = filteredCollection.merge(L7_filteredCollection)
            image_median = combined_collection.median()
            median_number = image_count + L7_count

        else:
            # Take median of Collection
            image_median = filteredCollection.median()
            median_number = image_count

        return image_median, median_number

    if sat_name == 'L7':
        ## Filter by time range and location
        collection = (collect.filterDate(date_start, date_end)
                      .filterBounds(polygon))

        # Apply Cloud Score layer to each image, then filter collection
        withCloudiness = collection.map(LandsatCloudScore)
        filteredCollection_no_pan = (withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore'])))

        # Apply cloud masking to all images within collection then select bands
        maskedcollection = filteredCollection_no_pan.map(LandsatCloudMask)
        filteredCollection_masked = maskedcollection.select(LC7_BANDS, STD_NAMES)

        ##Print Images in Collection
        # List Images in Collection
        image_list = filteredCollection_no_pan.toList(500)
        image_count = len(image_list.getInfo())

        if settings['add_L5_to_L7'] == True:

            # Add Landsat 7 to Collection
            collect = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')
            ## Filter by time range and location
            L5_collection = (collect.filterDate(date_start, date_end)
                             .filterBounds(polygon))


            L5_withCloudiness = L5_collection.map(LandsatCloudScore)
            L5_filteredCollection = (L5_withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore']))
                                     .select(LC5_BANDS, STD_NAMES))
            ##Print Images in Collection
            # List Images in Collection
            L5_image_list = L5_filteredCollection.toList(500)
            L5_count = len(L5_image_list.getInfo())

            if L5_count > 0:

                # Apply cloud masking to all images within collection then select bands
                maskedcollection = filteredCollection_no_pan.map(LandsatCloudMask)
                filteredCollection_masked = maskedcollection.select(LC7_BANDS, STD_NAMES)

                # Merge Collections
                filteredCollection_masked = filteredCollection_masked.merge(L5_filteredCollection)
                median_number = image_count + L5_count
            else:
                median_number = image_count
                pass
        else:
            median_number = image_count

        # Take median of Collection
        image_median_no_pan = filteredCollection_masked.median()

        ## Add panchromatic band to collection from Landsat 7
        # Add Panchromatic Band
        panchromatic = ['B8']
        panchromatic_name = ['pan']
        filteredCollection_pan = (withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore'])))

        # Repeat masking process and take median
        maskedcollection_pan = filteredCollection_pan.map(LandsatCloudMask)
        filteredCollection_masked_pan = maskedcollection_pan.select(panchromatic, panchromatic_name)
        image_median_pan = filteredCollection_masked_pan.median()

        # Combine multiplspectral and panchromatic bands
        image_median = image_median_no_pan.addBands(image_median_pan)

        return image_median, median_number

    if sat_name == 'L8':
        ## Filter by time range and location
        L8_collection = (collect.filterDate(date_start, date_end)
                         .filterBounds(polygon))

        # Apply cloud masking to all images within collection then select bands
        L8_withCloudiness = L8_collection.map(LandsatCloudScore)
        L8_filteredCollection = (L8_withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore'])))

        ## Need to add panchromatic band to collection
        # Add Panchromatic Band
        panchromatic = ['B8']
        panchromatic_name = ['pan']

        # Apply cloud masking to all images within collection then select bands
        maskedcollection = L8_filteredCollection.map(LandsatCloudMask)
        filteredCollection_masked = maskedcollection.select(LC8_BANDS + panchromatic, STD_NAMES + panchromatic_name)

        ##Print Images in Collection
        # List Images in Collection
        image_list = L8_filteredCollection.toList(500)
        image_count = len(image_list.getInfo())

        if settings['add_L7_to_L8'] == True:

            # Add Landsat 7 to Collection
            collect = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA')
            ## Filter by time range and location
            L7_collection = (collect.filterDate(date_start, date_end)
                             .filterBounds(polygon))


            # Apply cloud masking to all images within collection then select bands
            L7_withCloudiness = L7_collection.map(LandsatCloudScore)
            L7_filteredCollection = (L7_withCloudiness.filter(ee.Filter.lt('cloud', settings['LCloudScore'])))

            # Apply cloud masking to all images within collection then select bands
            maskedcollection = L7_filteredCollection.map(LandsatCloudMask)
            L7_filteredCollection_masked = maskedcollection.select(LC7_BANDS + panchromatic,
                                                                   STD_NAMES + panchromatic_name)

            ## Print Images in Collection
            # List Images in Collection
            L7_image_list = L7_filteredCollection.toList(500)
            L7_count = len(L7_image_list.getInfo())

            # Merge Collections
            L8_filteredCollection = filteredCollection_masked.merge(L7_filteredCollection_masked)
            median_number = image_count + L7_count

        else:
            median_number = image_count
            pass

        # Take median of Collection
        image_median = L8_filteredCollection.median()

        return image_median, median_number

    elif sat_name == 'S2':

        def add_cloud_bands(img):
            """
            Cloud components
            Define a function to add the s2cloudless probability layer and
            derived cloud mask as bands to an S2 SR image input.

            Parameters
            ----------
            img : TYPE
                DESCRIPTION.

            Returns
            -------
            TYPE
                DESCRIPTION.

            """
            # Get s2cloudless image, subset the probability band.
            cloud_prb = ee.Image(img.get('s2cloudless')).select('probability')

            # Condition s2cloudless by the probability threshold value.
            is_cloud = cloud_prb.gt(settings['CLD_PRB_THRESH']).rename('clouds')

            # Add the cloud probability layer and cloud mask as image bands.
            return img.addBands(ee.Image([cloud_prb, is_cloud]))

        def add_shadow_bands(img):
            """
            #### Cloud shadow components

            Define a function to add dark pixels, cloud projection, and identified
            shadows as bands to an S2 SR image input. Note that the image input needs
            to be the result of the above `add_cloud_bands` function because it
            relies on knowing which pixels are considered cloudy (`'clouds'` band).

            Parameters
            ----------
            img : TYPE
                DESCRIPTION.

            Returns
            -------
            TYPE
                DESCRIPTION.

            """
            # Identify water pixels from the SCL band.
            not_water = img.select('SCL').neq(6)

            # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
            SR_BAND_SCALE = 1e4
            dark_pixels = img.select('B8').lt(settings['NIR_DRK_THRESH'] * SR_BAND_SCALE).multiply(not_water).rename(
                'dark_pixels')

            # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
            shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

            # Project shadows from clouds for the distance specified by the cloud_PRJ_DIST input.
            cloud_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, settings['CLD_PRJ_DIST'] * 10)
                        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                        .select('distance')
                        .mask()
                        .rename('cloud_transform'))

            # Identify the intersection of dark pixels with cloud shadow projection.
            shadows = cloud_proj.multiply(dark_pixels).rename('shadows')

            # Add dark pixels, cloud projection, and identified shadows as image bands.
            return img.addBands(ee.Image([dark_pixels, cloud_proj, shadows]))

        def add_cloud_shadow_mask(img):
            """
            #### Final cloud-shadow mask

            Define a function to assemble all of the cloud and cloud shadow components and produce the final mask.

            """
            # Add cloud component bands.
            image_cloud = add_cloud_bands(img)

            # End date from user input range
            user_end = date_start.split("-")
            # Period of Sentinel 2 data before Surface reflectance data is available
            start = datetime(2015, 6, 23)
            end = datetime(2021, 1, 28)

            def time_in_range(start, end, x):
                if start <= end:
                    return start <= x <= end
                else:
                    return start <= x or x <= end

            # Is start date within pre S2_SR period?
            if time_in_range(start, end, datetime(int(user_end[0]), int(user_end[1]), int(user_end[2]))) == False:

                # Add cloud shadow component bands.
                image_cloud_shadow = add_shadow_bands(image_cloud)
                # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
                is_cloud_shadow = image_cloud_shadow.select('clouds').add(image_cloud_shadow.select('shadows')).gt(0)

            else:
                # Add cloud shadow component bands.
                image_cloud_shadow = image_cloud
                # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
                is_cloud_shadow = image_cloud.select('clouds').gt(0)

            # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
            # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
            is_cloud_shadow = (is_cloud_shadow.focal_min(2).focal_max(settings['BUFFER'] * 2 / 20)
                           .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                           .rename('cloudmask'))

            # Add the final cloud-shadow mask to the image.
            return image_cloud_shadow.addBands(is_cloud_shadow)

        def apply_cloud_shadow_mask(img):
            """
            ### Define cloud mask application function

            Define a function to apply the cloud mask to each image in the collection.

            """
            # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
            not_cloud_shadow = img.select('cloudmask').Not()

            # Subset reflectance bands and update their masks, return the result.
            return img.select('B.*').updateMask(not_cloud_shadow)

        # Build masks and apply to S2 image
        s2_sr_cloud_col, median_number = get_S2_SR_cloud_col(settings)

        image_median = (s2_sr_cloud_col.map(add_cloud_shadow_mask)
                        .map(apply_cloud_shadow_mask)
                        .median())

        return image_median, median_number


def download_GEE_image(image, scale, region, file_path, bands=['VV', 'VH']):

    path = image.getDownloadURL({
        'name': 'data',
        'scale': scale,
        'region': region,
        'filePerBand': False,
        'bands': bands
    })

    local_zip, headers = urlretrieve(path)
    with zipfile.ZipFile(local_zip) as local_zipfile:
        return local_zipfile.extractall(path=str(file_path))


def save_metadata(settings):

    sat_name = settings['sat_name']
    site_name = settings['site_name']
    median_dir_path = settings['median_dir_path']

    # initialize metadata dict
    metadata = {}

    # if a dir has been created for the given satellite mission
    if sat_name in os.listdir(median_dir_path):

        meta_dir_path = os.path.join(median_dir_path, sat_name, 'meta')
        metadata_files = os.listdir(meta_dir_path)
        # update the metadata dict
        metadata[sat_name] = {'file_names':[], 'epsg':[], 'date_start':[],
                              'date_end':[], 'number_images':[]}

        text_files = [file_name for file_name in metadata_files if file_name[-4:] == '.txt']

        # loop through the .txt files
        for image_meta in text_files:

            # read them and extract the metadata info
            with open(os.path.join(meta_dir_path, image_meta), 'r') as f:

                filename = f.readline().split('\t')[1].replace('\n','')
                epsg = int(f.readline().split('\t')[1].replace('\n',''))
                date_start = f.readline().split('\t')[1].replace('\n','')
                date_end = f.readline().split('\t')[1].replace('\n','')
                number_images = int(f.readline().split('\t')[1].replace('\n', ''))

            # store the information in the metadata dict
            metadata[sat_name]['file_names'].append(filename)
            metadata[sat_name]['epsg'].append(epsg)
            metadata[sat_name]['date_start'].append(date_start)
            metadata[sat_name]['date_end'].append(date_end)
            metadata[sat_name]['number_images'].append(number_images)

    # save a .pkl file containing the metadata dict
    metadata_file_name = pickleDumpName('metadata', site_name, sat_name)
    with open(os.path.join(median_dir_path, metadata_file_name), 'wb') as f:
        pickle.dump(metadata, f)

    printProgress('metadata saved')

    settings['metadata'] = metadata

def load_metadata(settings):

    sat_name = settings['sat_name']
    site_name = settings['site_name']
    median_dir_path = settings['median_dir_path']
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]

    metadata_file_name = pickleDumpName('metadata', site_name, sat_name)
    with open(os.path.join(median_dir_path,  metadata_file_name), 'rb') as f:
        metadata_dict = pickle.load(f)

    metadata_sat = metadata_dict[sat_name]
    file_names = metadata_sat['file_names']

    # initialize metadata dict
    metadata = {}
    for file_index, file_name in enumerate(file_names):

        if date_start == metadata_sat['date_start'][file_index] and \
             date_end == metadata_sat['date_end'][file_index]:

            file_names = []

            if sat_name == 'S1':
                file_names.append(file_name)
            else:
                band_dict = settings['bands'][sat_name]
                for band_key in band_dict.keys():
                    file_names.append(file_name + '_' + band_key + '.tif')

            metadata['file_names'] = file_names
            metadata['epsg'] = int(metadata_sat['epsg'][file_index])
            metadata['date_start'] = date_start
            metadata['date_end'] = date_end
            metadata['number_images'] = metadata_sat['number_images'][file_index]

            break

    printProgress('metadata loaded')

    return metadata


def get_S2_SR_cloud_col(settings):

    def time_in_range(start, end, x):
        if start <= end:
            return start <= x <= end
        else:
            return start <= x or x <= end


    polygon = ee.Geometry.Polygon(settings['polygon'])
    date_start = settings['dates'][0]
    date_end = settings['dates'][1]
    CLOUD_FILTER = settings['CLOUD_FILTER']

    # End date from user input range
    user_end = date_start.split("-")
    # Period of Sentinel 2 data before Surface reflectance data is available
    start = datetime(2015, 6, 23)
    end = datetime(2021, 1, 28)

    # Is end date within pre S2_SR period?
    if time_in_range(start, end, datetime(int(user_end[0]), int(user_end[1]), int(user_end[2]))) == False:
        # Import and filter S2 SR.

        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2')
                     .filterBounds(polygon)
                     .filterDate(date_start, date_end)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    else:
        # Import and filter S2 SR.
        s2_sr_col = (ee.ImageCollection('COPERNICUS/S2')
                     .filterBounds(polygon)
                     .filterDate(date_start, date_end)
                     .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(polygon)
                        .filterDate(date_start, date_end))

    ##Print Images in Collection
    # List Images in Collection
    image_list = s2_sr_col.toList(500)
    median_number = len(image_list.getInfo())

    median_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return median_collection, median_number

