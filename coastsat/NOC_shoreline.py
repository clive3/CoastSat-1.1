from coastsat.SDS_shoreline import *

from coastsat import NOC_preprocess, NOC_tools


def classify_image_NN_5classes(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    im_extra:
        only used for Landsat 7 and 8 where im_extra is the panchromatic band
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_land1 = im_classif == 1
    im_land2 = im_classif == 2
    im_land3 = im_classif == 3
    im_ww = im_classif == 4
    im_water = im_classif == 5

    # remove small patches of sand or water that could be around the image (usually noise)
    im_land1 = morphology.remove_small_objects(im_land1, min_size=min_beach_area, connectivity=2)
    im_land2 = morphology.remove_small_objects(im_land2, min_size=min_beach_area, connectivity=2)
    im_land3 = morphology.remove_small_objects(im_land3, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_land1, im_land2, im_land3, im_ww, im_water), axis=-1)

    return im_classif, im_labels

def classify_image_NN_4classes(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    im_extra:
        only used for Landsat 7 and 8 where im_extra is the panchromatic band
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_land1 = im_classif == 1
    im_land2 = im_classif == 2
    im_land3 = im_classif == 3
    im_ww = im_classif == 4
    im_water = im_classif == 5

    # remove small patches of sand or water that could be around the image (usually noise)
    im_land1 = morphology.remove_small_objects(im_land1, min_size=min_beach_area, connectivity=2)
    im_land2 = morphology.remove_small_objects(im_land2, min_size=min_beach_area, connectivity=2)
    im_land3 = morphology.remove_small_objects(im_land3, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_land1, im_land2, im_land3, im_ww, im_water), axis=-1)

    return im_classif, im_labels


def find_wl_contours2_5classes(im_ms, im_labels, cloud_mask, buffer_size, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification
    component to refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    buffer_size: int
        size of the buffer around the sandy beach over which the pixels are considered in the
        thresholding algorithm.
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu sand/water threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_land1 = im_labels[:,:,0].reshape(ncols*nrows)
    vec_land2 = im_labels[:,:,1].reshape(ncols*nrows)
    vec_land3 = im_labels[:,:,2].reshape(ncols*nrows)
    vec_all_land = np.logical_or((np.logical_or(vec_land1, vec_land2)), vec_land3)

    vec_water = im_labels[:,:,4].reshape(ncols*nrows)
    vec_im_ref_buffer = im_ref_buffer.reshape(ncols*nrows)

    # select water/sand/swash pixels that are within the buffer
    int_land = vec_ind[np.logical_and(vec_im_ref_buffer,vec_all_land),:]
    int_sea = vec_ind[np.logical_and(vec_im_ref_buffer,vec_water),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_land) > 0 and len(int_sea) > 0:
        if np.argmin([int_land.shape[0],int_sea.shape[0]]) == 1:
            int_land = int_land[np.random.choice(int_land.shape[0],int_sea.shape[0], replace=False),:]
        else:
            int_sea = int_sea[np.random.choice(int_sea.shape[0],int_land.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_land,int_sea, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])

    # find contour with MS algorithm
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)
    # remove contour points that are NaNs (around clouds)
    contours_mwi = process_contours(contours_mwi)

    # only return MNDWI contours and threshold
    return contours_mwi, t_mwi


def extract_shorelines(metadata, settings):
    """
    Main function to extract shorelines from satellite images

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'buffer_size': int
            size of the buffer (m) around the sandy pixels over which the pixels
            are considered in the thresholding algorithm
        'min_beach_area': int
            minimum allowable object area (in metres^2) for the class 'sand',
            the area is converted to number of connected pixels
        'min_length_sl': int
            minimum length (in metres) of shoreline contour to be valid
        'sand_color': str
            default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
        'adjust_detection': bool
            if True, allows user to manually adjust the detected shoreline

    Returns:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates + metadata

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping shorelines:')

    # loop through satellite list
    for satname in metadata.keys():

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'], satname)
        filenames = metadata[satname]['filenames']

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = []  # filename of the images from which the shorelines where derived
        output_cloudcover = []  # cloud cover of the images
        output_geoaccuracy = []  # georeferencing accuracy of the images
        output_idxkeep = []  # index that were kept during the analysis (cloudy images are skipped)
        output_median_no = []

        # load classifiers
        if satname in ['L5', 'L7', 'L8']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark.pkl'))
            elif settings['sand_color'] == 'bright':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright.pkl'))
            else:
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat.pkl'))

        elif satname == 'S2':
            pixel_size = 10
            clf = joblib.load(os.path.join(filepath_models, 'NN_5classes_S2.pkl'))

        # convert settings['min_beach_area'] and settings['buffer_size'] from metres to pixels
        buffer_size_pixels = np.ceil(settings['buffer_size'] / pixel_size)
        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)

        # loop through the images
        for i in range(len(filenames)):

            print('\r%s:   %d%%' % (satname, int(((i + 1) / len(filenames)) * 100)))
            print()

            # get image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = \
                SDS_preprocess.preprocess_single(fn, satname, settings['cloud_mask_issue'])
            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                             (cloud_mask.shape[0] * cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
                continue
            # remove no data pixels from the cloud mask
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (cloud_mask.shape[0] * cloud_mask.shape[1]))
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                                    pixel_size, settings)

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = classify_image_NN_5classes(im_ms, cloud_mask,
                                                      min_beach_area_pixels, clf)

            # find the shoreline interactively
            date = filenames[i][:19]
            skip_image, shoreline = adjust_detection_5classes(im_ms, cloud_mask, im_labels,
                                                     im_ref_buffer, image_epsg, georef, settings, date,
                                                     satname, buffer_size_pixels)
            # if the user decides to skip the image, continue and do not save the mapped shoreline
            if skip_image:
                continue

            # append to output variables
            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            output_median_no.append(metadata[satname]['median_no'][i])

        # create dictionnary of output
        output[satname] = {
            'dates': output_timestamp,
            'shorelines': output_shoreline,
            'filename': output_filename,
            'cloud_cover': output_cloudcover,
            'geoaccuracy': output_geoaccuracy,
            'idx': output_idxkeep,
            'median_no': output_median_no
        }
        print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output


def extract_sar_shorelines(metadata, settings):

    inputs = settings['inputs']

    sitename = inputs['sitename']
    base_filepath = inputs['filepath']
    satname = inputs['sat_list'][0]
    pixel_size = inputs['pixel_size']

    # initialise output structure
    output = {}
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(base_filepath, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping shorelines:')

    # get images
    filepath = NOC_tools.get_filepath(settings['inputs'], satname)
    filenames = metadata[satname]['filenames']

    # initialise the output variables
    output_timestamp = []  # datetime at which the image was acquired (UTC time)
    output_shoreline = []  # vector of shoreline points
    output_filename = []  # filename of the images from which the shorelines where derived
    output_geoaccuracy = []  # georeferencing accuracy of the images
    output_median_no = []

    # loop through the images
    for i in range(len(filenames)):

        print('\r%s:   %d%%' % (satname, int(((i + 1) / len(filenames)) * 100)))
        print()

        # get image filename
        filename = filenames[i]

        sar_image, georef = NOC_preprocess.preprocess_sar(filename, satname)

        # get image spatial reference system (epsg code) from metadata dict
        image_epsg = metadata[satname]['epsg'][i]

        # calculate a buffer around the reference shoreline (if any has been digitised)
        im_ref_buffer = create_shoreline_buffer(sar_image.shape, georef, image_epsg,
                                                pixel_size, settings)


        # find the shoreline interactively
        date = filename[:19]
        skip_image, shoreline = adjust_detection_sar(sar_image,  im_ref_buffer, image_epsg,
                                                     georef, settings, date,  satname)
        # if the user decides to skip the image, continue and do not save the mapped shoreline
        if skip_image:
            continue

        # append to output variables
        output_timestamp.append(metadata[satname]['dates'][i])
        output_shoreline.append(shoreline)
        output_filename.append(filenames[i])
        output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
        output_median_no.append(metadata[satname]['median_no'][i])

    # create dictionnary of output
    output[satname] = {
        'dates': output_timestamp,
        'shorelines': output_shoreline,
        'filename': output_filename,
        'geoaccuracy': output_geoaccuracy,
        'median_no': output_median_no
    }
    print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(base_filepath, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output


def show_detection_4classes(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control.
    The user can accept/reject the detected shorelines  by using keep/skip
    buttons.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)
    colours = np.zeros((4,4))
    colours[0, :] = np.array([1, 0, 0, 1])
    colours[1, :] = np.array([0, 1, 0, 1])
    colours[2, :] = np.array([1, 1, 0, 1])
    colours[3, :] = np.array([0.8, 1, 1, 1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:,[0,1]], georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])

    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2,0], sharex=ax1, sharey=ax1)
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    patch0 = mpatches.Patch(color=[1,1,1], label='unclassified')
    patch1 = mpatches.Patch(color=colours[0,:], label='hard surface')
    patch2 = mpatches.Patch(color=colours[1,:], label='natural')
    patch3 = mpatches.Patch(color=colours[2,:], label='urban')
    patch4 = mpatches.Patch(color=colours[3, :], label='white-water')
    patch5 = mpatches.Patch(color=colours[4,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title('classification', fontweight='bold', fontsize=16)

    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    ax3.set_title('MNDWI', fontweight='bold', fontsize=16)

    # additional options
    #    ax1.set_anchor('W')
    #    ax2.set_anchor('W')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    cb.set_label('MNDWI values')
    ax3.set_anchor('W')

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, 'NOC_' + date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image

def show_detection_5classes(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control.
    The user can accept/reject the detected shorelines  by using keep/skip
    buttons.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)
    colours = np.zeros((5,4))
    colours[0, :] = np.array([1, 0, 0, 1])
    colours[1, :] = np.array([0, 1, 0, 1])
    colours[2, :] = np.array([1, 1, 0, 1])
    colours[3, :] = np.array([0.8, 1, 1, 1])
    colours[4, :] = np.array([0, 0.4, 1, 1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:,[0,1]], georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])

    if plt.get_fignums():
            # get open figure if it exists
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2,0], sharex=ax1, sharey=ax1)
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=16)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    patch0 = mpatches.Patch(color=[1,1,1], label='unclassified')
    patch1 = mpatches.Patch(color=colours[0,:], label='hard surface')
    patch2 = mpatches.Patch(color=colours[1,:], label='natural')
    patch3 = mpatches.Patch(color=colours[2,:], label='urban')
    patch4 = mpatches.Patch(color=colours[3, :], label='white-water')
    patch5 = mpatches.Patch(color=colours[4,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title('classification', fontweight='bold', fontsize=16)

    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    ax3.set_title('MNDWI', fontweight='bold', fontsize=16)

    # additional options
    #    ax1.set_anchor('W')
    #    ax2.set_anchor('W')
#    cb = plt.colorbar()
#    cb.ax.tick_params(labelsize=10)
#    cb.set_label('MNDWI values')
#    ax3.set_anchor('W')

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, 'NOC_' + date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image

def adjust_detection_4classes(im_ms, cloud_mask, im_labels, im_ref_buffer, image_epsg, georef,
                     settings, date, satname, buffer_size_pixels):
    """
    Advanced version of show detection where the user can adjust the detected
    shorelines with a slide bar.

    KV WRL 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    buffer_size_pixels: int
        buffer_size converted to number of pixels
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    # format date
    date_str = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d  %H:%M:%S')
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)

    colours = np.zeros((4,4))

    colours[0, :] = np.array([1, 0, 0, 1])
    colours[1, :] = np.array([0, 1, 0, 1])
    colours[2, :] = np.array([1, 1, 0, 1])
    colours[3, :] = np.array([0.8, 1, 1, 1])

    for k in range(0, im_labels.shape[2]):
        im_class[im_labels[:, :, k], 0] = colours[k, 0]
        im_class[im_labels[:, :, k], 1] = colours[k, 1]
        im_class[im_labels[:, :, k], 2] = colours[k, 2]

    # compute MNDWI grayscale image
    im_mndwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    im_mndwi_buffer = np.copy(im_mndwi)
    im_mndwi_buffer[~im_ref_buffer] = np.nan

    # get MNDWI pixel intensity in each class (for histogram plot)
    int_sand = im_mndwi[im_labels[:, :, 0]]
    int_ww = im_mndwi[im_labels[:, :, 1]]
    int_water = im_mndwi[im_labels[:, :, 2]]
    labels_other = np.logical_and(np.logical_and(~im_labels[:, :, 0], ~im_labels[:, :, 1]), ~im_labels[:, :, 2])
    int_other = im_mndwi[labels_other]

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # plot image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
    ax1.set_title('%s - %s' % (sitename, satname), fontsize=12)

    # plot image 2 (classification)
    ax2.imshow(im_class)
    ax2.axis('off')
    patch0 = mpatches.Patch(color=[1,1,1], label='unclassified')
    patch1 = mpatches.Patch(color=colours[0,:], label='hard surface')
    patch2 = mpatches.Patch(color=colours[1,:], label='natural')
    patch3 = mpatches.Patch(color=colours[2,:], label='urban')
    patch4 = mpatches.Patch(color=colours[3, :], label='white-water')
    patch5 = mpatches.Patch(color=colours[4,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title('classification', fontweight='bold', fontsize=16)

    # plot image 3 (MNDWI)
    ax3.imshow(im_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontsize=12)

    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF', yticklabels=[], xlim=[-1, 1])
    if len(int_sand) > 0:
        bins = np.arange(np.nanmin(int_sand), np.nanmax(int_sand) + binwidth, binwidth)
        ax4.hist(int_sand, bins=bins, density=True, color=colours[0, :], label='sand')
    if len(int_ww) > 0:
        bins = np.arange(np.nanmin(int_ww), np.nanmax(int_ww) + binwidth, binwidth)
        ax4.hist(int_ww, bins=bins, density=True, color=colours[1, :], label='whitewater', alpha=0.75)
    if len(int_water) > 0:
        bins = np.arange(np.nanmin(int_water), np.nanmax(int_water) + binwidth, binwidth)
        ax4.hist(int_water, bins=bins, density=True, color=colours[2, :], label='water', alpha=0.75)
    if len(int_other) > 0:
        bins = np.arange(np.nanmin(int_other), np.nanmax(int_other) + binwidth, binwidth)
        ax4.hist(int_other, bins=bins, density=True, color='C4', label='other', alpha=0.5)

        # automatically map the shoreline based on the classifier if enough sand pixels
    if sum(sum(im_labels[:, :, 0])) > 10:
        # use classification to refine threshold and extract the sand/water interface
        contours_mndwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask,
                                                    buffer_size_pixels, im_ref_buffer)
    else:
        # find water contours on MNDWI grayscale image
        contours_mndwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)

        # process the water contours into a shoreline
    shoreline = process_shoreline(contours_mndwi, cloud_mask, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_mndwi, ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # update the threshold value
            t_mndwi = pt[0][0]
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(t_mndwi) >= 1: continue
            # update the plot
            t_line.set_xdata([t_mndwi, t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(im_mndwi_buffer, t_mndwi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours)
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:, [0, 1]], georef)
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}

    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k', fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_skip.remove()
        btn_keep.remove()
        btn_esc.remove()

        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            break
        elif key_event.get('pressed') == 'left':
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline

def adjust_detection_5classes(im_ms, cloud_mask, im_labels, im_ref_buffer, image_epsg, georef,
                     settings, date, satname, buffer_size_pixels):
    """
    Advanced version of show detection where the user can adjust the detected
    shorelines with a slide bar.

    KV WRL 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    buffer_size_pixels: int
        buffer_size converted to number of pixels
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    # format date
    date_str = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d  %H:%M:%S')
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)

    colours = np.zeros((5,4))
    colours[0, :] = np.array([1, 0, 0, 1])
    colours[1, :] = np.array([0, 1, 0, 1])
    colours[2, :] = np.array([1, 1, 0, 1])
    colours[3, :] = np.array([0.8, 1, 1, 1])
    colours[4, :] = np.array([0, 0.4, 1, 1])
    for k in range(0, im_labels.shape[2]):
        im_class[im_labels[:, :, k], 0] = colours[k, 0]
        im_class[im_labels[:, :, k], 1] = colours[k, 1]
        im_class[im_labels[:, :, k], 2] = colours[k, 2]

    # compute MNDWI grayscale image
    im_mndwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    im_mndwi_buffer = np.copy(im_mndwi)
    im_mndwi_buffer[~im_ref_buffer] = np.nan

    # get MNDWI pixel intensity in each class (for histogram plot)
    int_sand = im_mndwi[im_labels[:, :, 0]]
    int_nature = im_mndwi[im_labels[:, :, 1]]
    int_urban = im_mndwi[im_labels[:, :, 2]]
    int_ww = im_mndwi[im_labels[:, :, 3]]
    int_water = im_mndwi[im_labels[:, :, 4]]

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # plot image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
    ax1.set_title('%s - %s' % (sitename, satname), fontsize=12)

    # plot image 2 (classification)
    ax2.imshow(im_class)
    ax2.axis('off')
    patch0 = mpatches.Patch(color=[1,1,1], label='unclassified')
    patch1 = mpatches.Patch(color=colours[0,:], label='sand/hard')
    patch2 = mpatches.Patch(color=colours[1,:], label='land1')
    patch3 = mpatches.Patch(color=colours[2,:], label='land2')
    patch4 = mpatches.Patch(color=colours[3, :], label='white-water')
    patch5 = mpatches.Patch(color=colours[4,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title('classification', fontweight='bold', fontsize=16)

    # plot image 3 (MNDWI)
    ax3.imshow(im_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontsize=12)

    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF', yticklabels=[], xlim=[-1, 1])
    if len(int_sand) > 0:
        bins = np.arange(np.nanmin(int_sand), np.nanmax(int_sand) + binwidth, binwidth)
        ax4.hist(int_sand, bins=bins, density=True, color=colours[0, :], label='sand/hard')
    if len(int_nature) > 0:
        bins = np.arange(np.nanmin(int_nature), np.nanmax(int_nature) + binwidth, binwidth)
        ax4.hist(int_nature, bins=bins, density=True, color=colours[1, :], label='land1', alpha=0.75)
    if len(int_urban) > 0:
        bins = np.arange(np.nanmin(int_urban), np.nanmax(int_urban) + binwidth, binwidth)
        ax4.hist(int_urban, bins=bins, density=True, color=colours[2, :], label='land2', alpha=0.75)
    if len(int_ww) > 0:
        bins = np.arange(np.nanmin(int_ww), np.nanmax(int_ww) + binwidth, binwidth)
        ax4.hist(int_ww, bins=bins, density=True, color=colours[3, :], label='white-water', alpha=0.75)
    if len(int_water) > 0:
        bins = np.arange(np.nanmin(int_water), np.nanmax(int_water) + binwidth, binwidth)
        ax4.hist(int_water, bins=bins, density=True, color=colours[4, :], label='water', alpha=0.75)

        # automatically map the shoreline based on the classifier if enough sand pixels
    if sum(sum(im_labels[:, :, 0])) > 10:
        # use classification to refine threshold and extract the sand/water interface
        contours_mndwi, t_mndwi = find_wl_contours2_5classes(im_ms, im_labels, cloud_mask,
                                                    buffer_size_pixels, im_ref_buffer)
    else:
        print('not enough sand pixels ... using alternative algorithm for shoreline')
        # find water contours on MNDWI grayscale image
        contours_mndwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)

    # process the water contours into a shoreline
    shoreline = process_shoreline(contours_mndwi, cloud_mask, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_mndwi, ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # update the threshold value
            t_mndwi = pt[0][0]
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(t_mndwi) >= 1: continue
            # update the plot
            t_line.set_xdata([t_mndwi, t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(im_mndwi_buffer, t_mndwi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours)
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:, [0, 1]], georef)
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}

    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k', fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_skip.remove()
        btn_keep.remove()
        btn_esc.remove()

        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            break
        elif key_event.get('pressed') == 'left':
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline


def adjust_detection_sar(sar_image, im_ref_buffer, image_epsg, georef,
                     settings, date, satname):


    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    # format date
    date_str = datetime.strptime(date, '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d  %H:%M:%S')

    # compute classified image
    im_class = np.copy(sar_image[:,:,1])

    colours = np.array([1, 1, 1, 1])

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
 #       ax2 = fig.axes[1]
 #       ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
#        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
#        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # plot image 1 (RGB)
    ax1.imshow(im_class)
    ax1.axis('off')
    ax1.set_title('%s - %s' % (sitename, satname), fontsize=12)


    # plot histogram of sigma values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF', yticklabels=[], xlim=[-1, 1])

    bins = np.arange(np.nanmin(im_class), np.nanmax(im_class) + binwidth, binwidth)
    ax4.hist(im_class, bins=bins, density=True, color=colours[0, :], label='sigma0')

    contours_sar, t_sar = find_sar_contours(im_class, im_ref_buffer)

    cloud_mask = np.ones(im_class.shape)
    # process the water contours into a shoreline
    shoreline = process_shoreline(contours_sar, cloud_mask, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
#    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
#    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_sar, ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        'Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # update the threshold value
            t_sar = pt[0][0]
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(t_sar) >= 1: continue
            # update the plot
            t_line.set_xdata([t_sar, t_sar])
            # map contours with new threshold
            contours = measure.find_contours(im_class, t_sar)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours)
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:, [0, 1]], georef)
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
#            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
#            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}

    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k', fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                           transform=ax1.transAxes,
                           bbox=dict(boxstyle="square", ec='k', fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_skip.remove()
        btn_keep.remove()
        btn_esc.remove()

        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            break
        elif key_event.get('pressed') == 'left':
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline


def find_sar_contours(im_sar, im_ref_buffer):

    t_sar = filters.threshold_otsu(im_sar)

    # find contour with MS algorithm
    im_sar_buffer = np.copy(im_sar)
    im_sar_buffer[~im_ref_buffer] = np.nan
    contours_sar = measure.find_contours(im_sar_buffer, t_sar)

    # only return MNDWI contours and threshold
    return contours_sar, t_sar
