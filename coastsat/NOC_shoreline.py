from scipy.ndimage import gaussian_filter

from coastsat.SDS_shoreline import *
from coastsat import NOC_preprocess, NOC_classify, NOC_tools

from utils.print_utils import printProgress, printSuccess, printWarning


def extract_shoreline_optical(metadata, settings):

    inputs = settings['inputs']
    classes = settings['classes']

    median_dir_path = inputs['median_dir_path']
    sat_name = inputs['sat_name']
    site_name = inputs['site_name']
    date_start = inputs['dates'][0]
    date_end = inputs['dates'][1]
    pansharpen = inputs['pansharpen']

    band_dict = settings['bands'][sat_name]
    first_key = next(iter(band_dict))
    pixel_size = band_dict[first_key][1]

    file_names = metadata['file_names']

    models_file_path = os.path.join(os.getcwd(), 'classification', 'models')

    bands_20m = band_dict['20m']
    for SWIR_index, band in enumerate(bands_20m):
        if settings['SWIR'] == band:
            break

    printProgress(f'extracting shoreline ')

    # create a subfolder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)
    # close all open figures
    plt.close('all')

    # load classifier
    if sat_name in ['L5', 'L7', 'L8']:
#            if settings['sand_color'] == 'dark':
#                classifier = joblib.load(os.path.join(models_file_path, 'NN_4classes_Landsat_dark.pkl'))
#            elif settings['sand_color'] == 'bright':
#                classifier = joblib.load(os.path.join(models_file_path, 'NN_4classes_Landsat_bright.pkl'))
#            else:
        classifier = joblib.load(os.path.join(models_file_path, 'NN_4classes_Landsat.pkl'))

    elif sat_name == 'S2':
        classifier = joblib.load(os.path.join(models_file_path, 'NN_6classes_S2.pkl'))

    file_paths = []
    for band_index, band_key in enumerate(band_dict.keys()):
        file_paths.append(os.path.join(median_dir_path, sat_name, band_key, file_names[band_index]))

    # convert settings['min_beach_area'] from  metres to pixels
    min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)

    # preprocess image (cloud mask + pansharpening/downsampling)
    image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata = \
            NOC_preprocess.preprocess_single(file_paths, sat_name, settings['cloud_mask_issue'],
                                             pansharpen=pansharpen, SWIR_index=SWIR_index)

    # get image spatial reference system (epsg code) from metadata dict
    image_epsg = int(metadata['epsg'])

    # compute cloud_cover percentage (with no data pixels)
    cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                     (cloud_mask.shape[0] * cloud_mask.shape[1]))
    if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
        return []

    # remove no data pixels from the cloud mask
    # (for example L7 bands of no data should not be accounted for)
    cloud_mask_adv = np.logical_xor(cloud_mask, image_nodata)
    # compute updated cloud cover percentage (without no data pixels)
    cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                            (cloud_mask.shape[0] * cloud_mask.shape[1]))
    # skip image if cloud cover is above user-defined threshold
    if cloud_cover > settings['cloud_thresh']:
        return []

    buffer_shape = cloud_mask.shape
    image_ref_buffer = create_shoreline_buffer(buffer_shape, georef, image_epsg,
                                                pixel_size, settings)

    printProgress('classifying image')
    # classify image with NN classifier
    _, image_labels = NOC_classify.classify_image_NN(image_ms, classes, cloud_mask,
                                                       min_beach_area_pixels, classifier)

    # find the shoreline interactively
    shoreline, _ = adjust_detection_optical(image_ms, cloud_mask, image_labels, image_ref_buffer,
                                          image_epsg, georef, settings, sat_name)

    gdf = NOC_tools.output_to_gdf(shoreline, metadata)
    file_string = f'{site_name}_shoreline_{sat_name}' + \
                  f'_S{date_start}_E{date_end}.geojson'
    if ~gdf.empty:
        gdf.crs = {'init': 'epsg:' + str(settings['output_epsg'])}  # set layer projection
        # save GEOJSON layer to file
        gdf.to_file(os.path.join(inputs['median_dir_path'],
                                 file_string),
                    driver='GeoJSON', encoding='utf-8')

        printSuccess('shoreline saved')
    else:
        printWarning('no shorelines to be seen ...')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()


def adjust_detection_optical(image_ms, cloud_mask, image_labels, image_ref_buffer, image_epsg, georef,
                     settings, sat_name):

    inputs = settings['inputs']
    site_name = inputs['site_name']
    date_start = inputs['dates'][0]
    date_end = inputs['dates'][1]
    median_dir_path = inputs['median_dir_path']
    reference_threshold = inputs['reference_threshold']

    #  image_classifiery will become filled with labels
    image_RGB = SDS_preprocess.rescale_image_intensity(image_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
    image_classified = np.copy(image_RGB)
    image_classified = np.where(np.isnan(image_classified), 1.0, image_classified)

    # compute MNDWI grayscale image
    image_mndwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    image_mndwi_buffer = np.copy(image_mndwi)
    image_mndwi_buffer[~image_ref_buffer] = np.nan


    print(f'@@@ MNDWI min: {np.nanmin(image_mndwi_buffer)}')
    print(f'@@@ MNDWI max: {np.nanmax(image_mndwi_buffer)}')

    # for each class add it to image_classified and
    # extract the MDWI values for all pixels in that class
    classes = settings['classes']
    class_keys = classes.keys()

    mndwi_pixels = {}
    for key in class_keys:
        class_label = classes[key][0]
        class_colour = classes[key][1]

        image_classified[image_labels[:, :, class_label-1], 0] = class_colour[0]
        image_classified[image_labels[:, :, class_label-1], 1] = class_colour[1]
        image_classified[image_labels[:, :, class_label-1], 2] = class_colour[2]

        mndwi_pixels[class_label] = image_mndwi[image_labels[:, :, class_label-1]]

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
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
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
    image_RGB = np.where(np.isnan(image_RGB), nan_color, image_RGB)
    image_class = np.where(np.isnan(image_classified), 1, image_classified)

    # plot image 1 (RGB)
    ax1.imshow(image_RGB)
    ax1.axis('off')
    ax1.set_title(site_name, fontweight='bold', fontsize=16)

    # plot image 2 (classification)
    ax2.imshow(image_class)
    ax2.axis('off')

    ax2.set_title(date_start + ' to ' + date_end, fontweight='bold', fontsize=12)

    # plot image 3 (MNDWI)
    ax3.imshow(image_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontweight='bold', fontsize=16)

    # plot histogram of MNDWI values
    bin_width = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='pixels', yticklabels=[], xlim=[-1, 1])

    for key in class_keys:
        class_label = classes[key][0]
        class_colour = classes[key][1]
        class_pixels = mndwi_pixels[class_label]

        if key in ['white-water', 'sand']:
            alpha = 0.75
        elif key in['nature', 'urban', 'water']:
            alpha = 0.9
        else:
            alpha = 1.0

        if len(class_pixels) > 0:
            bins = np.arange(np.nanmin(class_pixels), np.nanmax(class_pixels) + bin_width, bin_width)
            ax4.hist(class_pixels, bins=bins, density=True,  color=class_colour, label=key, alpha=alpha)

    # use classification to refine threshold and extract the sand/water interface
    contours_mndwi, t_mndwi = find_contours_optical(image_ms, image_labels, cloud_mask, image_ref_buffer)

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
    t_line = ax4.axvline(x=t_mndwi, ls='--', c='k', lw=1.5, label=f'threshold')
    thresh_label = ax4.text(t_mndwi+bin_width, 4, str(f'{t_mndwi:4.3f}'), rotation=90)
    if reference_threshold:
        ref_threshold = reference_threshold
    else:
        ref_threshold = t_mndwi

    ax4.axvline(x=ref_threshold, ls='--', c='r', lw=1.5, label=f'ref threshold {reference_threshold:4.3f}')

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
            thresh_label.set(x=t_mndwi+bin_width, text=str(f'{t_mndwi:4.3f}'))
            # map contours with new threshold
            contours = measure.find_contours(image_mndwi_buffer, level=t_mndwi,  mask=image_ref_buffer)
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
            # update the plotted shoreline
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])

            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break

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

    plt.close()
    
     # if save_figure is True, save a .jpg under /jpg_files/detection
    if not skip_image:
        jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
        fig.savefig(os.path.join(jpeg_file_path, sat_name + '_detection_S' + date_start + '_E' + date_end + '.jpg'), dpi=150)

    if reference_threshold:
        printProgress('shoreline extracted')

    return shoreline, t_mndwi


def find_contours_optical(image_ms, image_labels, cloud_mask, ref_shoreline_buffer):

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    image_mwi = SDS_tools.nd_index(image_ms[:, :, 4], image_ms[:, :, 1], cloud_mask)

    # calculate Normalized Difference Modified Water Index (NIR - G)
    image_wi = SDS_tools.nd_index(image_ms[:, :, 3], image_ms[:, :, 1], cloud_mask)
    # stack indices together
    image_ind = np.stack((image_wi, image_mwi), axis=-1)
    vec_ind = image_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_land1 = image_labels[:, :, 0].reshape(ncols * nrows)
    vec_land2 = image_labels[:, :, 1].reshape(ncols * nrows)
    vec_land3 = image_labels[:, :, 2].reshape(ncols * nrows)
    vec_sand = image_labels[:, :, 5].reshape(ncols * nrows)
    vec_all_land = np.logical_or(vec_land2, vec_land3)

    vec_water = image_labels[:, :, 4].reshape(ncols * nrows)
    vec_image_ref_buffer = ref_shoreline_buffer.reshape(ncols * nrows)

    # select land and water pixels that are within the buffer
    int_land = vec_ind[np.logical_and(vec_image_ref_buffer,vec_all_land),:]
    int_sea = vec_ind[np.logical_and(vec_image_ref_buffer,vec_water),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_land) > 0 and len(int_sea) > 0:
        if np.argmin([int_land.shape[0],int_sea.shape[0]]) == 1:
            int_land = int_land[np.random.choice(int_land.shape[0],int_sea.shape[0], replace=False),:]
        else:
            int_sea = int_sea[np.random.choice(int_sea.shape[0],int_land.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_land,int_sea, axis=0)
    int_all = int_all[~np.isnan(int_all)]
    t_mwi = filters.threshold_otsu(int_all)

    # find contour with MS algorithm
    image_mwi_buffer = np.copy(image_mwi)
    image_mwi_buffer[~ref_shoreline_buffer] = np.nan
    contours_mwi = measure.find_contours(image_mwi_buffer, level=t_mwi, mask=ref_shoreline_buffer)
    # remove contour points that are NaNs (around clouds)
    contours_mwi = process_contours(contours_mwi)

    return contours_mwi, t_mwi


def extract_shoreline_sar(metadata, settings):

    inputs = settings['inputs']

    median_dir_path = inputs['median_dir_path']
    sat_name = inputs['sat_name']
    site_name = inputs['site_name']
    pixel_size = inputs['pixel_size']
    date_start = inputs['dates'][0]
    date_end = inputs['dates'][1]

    file_names = metadata['file_names']

    # create a folder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)

    printProgress('mapping shoreline')

    image_epsg = int(metadata['epsg'])

    for file_index, file_name in enumerate(file_names):

        file_path = os.path.join(median_dir_path, sat_name, file_name)

        # close all open figures
        plt.close('all')

        ## read the geotiff
        sar_image, georef = NOC_preprocess.preprocess_sar(file_path)

        # calculate a buffer around the reference shoreline if it has already been generated
        buffer_shape = (sar_image.shape[0], sar_image.shape[1])
        image_ref_buffer = create_shoreline_buffer(buffer_shape, georef, image_epsg,
                                                    pixel_size, settings)

        shoreline, _ = adjust_detection_sar(sar_image, image_ref_buffer, image_epsg,
                                          georef, settings)

        gdf = NOC_tools.output_to_gdf(shoreline, metadata)
        file_string = f'{site_name}_shoreline_{inputs["polarisation"]}' + \
                      f'_S{date_start}_E{date_end}.geojson'
        if ~gdf.empty:
            gdf.crs = {'init':'epsg:'+str(settings['output_epsg'])} # set layer projection
            # save GEOJSON layer to file
            gdf.to_file(os.path.join(inputs['median_dir_path'],
                                     file_string),
                                     driver='GeoJSON', encoding='utf-8')

            printSuccess('shoreline saved')
        else:
            printWarning('no shorelines to be seen ...')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()



def adjust_detection_sar(sar_image, image_ref_buffer, image_epsg, georef, settings):

    inputs = settings['inputs']
    sat_name = inputs['sat_name']
    polarisation = inputs['polarisation']
    date_start = inputs['dates'][0]
    data_end = inputs['dates'][1]
    reference_threshold = inputs['reference_threshold']

    site_name = inputs['site_name']
    median_dir_path = inputs['median_dir_path']

    # compute classified image
    if polarisation == 'VV':
        image_pol = np.copy(sar_image[:,:,0])
        colour = [1, 0, 1]
    else:
        image_pol = np.copy(sar_image[:,:,1])
        colour = [0, 1, 1]

    if inputs['create_reference_shoreline']:
        image_pol = gaussian_filter(image_pol, sigma=inputs['sigma'], mode='reflect')

    # and the vectors needed for the histogram
    cols = sar_image.shape[0]
    rows = sar_image.shape[1]
    vec_pol = image_pol.reshape(cols*rows)

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

    # plot image 1 (RGB)
    ax1.imshow(image_pol)
    ax1.axis('off')
    ax1.set_title(site_name, fontweight='bold', fontsize=16)

    # plot image 1 (grey)
    ax2.imshow(image_pol, cmap='gray')
    ax2.axis('off')
    ax2.set_title(date_start + ' to ' + data_end, fontweight='bold', fontsize=16)

    # plot image 3 (blue/red)
    ax3.imshow(image_pol, cmap='bwr')
    ax3.axis('off')
    ax3.set_title(polarisation, fontweight='bold', fontsize=16)

    # plot histogram of sigma values
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='pixels', yticklabels=[], xlim=[np.nanmin(sar_image), np.nanmax(sar_image)])

    bin_width = 0.1
    bins = np.arange(np.nanmin(vec_pol), np.nanmax(vec_pol) + bin_width, bin_width)
    ax4.hist(vec_pol, bins=bins, density=True, color=colour, label=polarisation)

    t_sar = filters.threshold_otsu(image_pol)
    contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)

    # process the water contours into a shoreline
    shoreline = process_sar_shoreline(contours_sar, georef, image_epsg, settings)

    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg)[:, [0, 1]], georef)
    else:
        printWarning('no shoreline yet')
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], 'r', markersize=3)

    t_line = ax4.axvline(x=t_sar, ls='--', c='k', lw=1.5, label='threshold')
    thresh_label = ax4.text(t_sar + bin_width, 0.25, str(f'{t_sar:4.2f}'), rotation=90)
    if not reference_threshold: reference_threshold = t_sar
    ax4.axvline(x=reference_threshold, ls='--', c='r', lw=1.5, label=f'ref threshold {reference_threshold:4.2f}')

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

            # update the plot
            t_line.set_xdata([t_sar, t_sar])
            thresh_label.set(x=t_sar + bin_width, text=str(f'{t_sar:4.2f}'))
            # map contours with new threshold
            contours_sar = measure.find_contours(image_pol, level=t_sar, mask=image_ref_buffer)
            # process the water contours into a shoreline
            shoreline = process_sar_shoreline(contours_sar, georef, image_epsg, settings)

            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg)[:, [0, 1]], georef)
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shoreline
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])

            fig.canvas.draw_idle()
        else:
            ax4.set_title('sigma0 pixel intensities and threshold')
            break

    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    fig.savefig(os.path.join(jpeg_file_path, sat_name + '_detection_S' + date_start + \
                             '_E' + data_end + '.jpg'), dpi=150)

    ## if creating reference shoreline return the Otsu threshold
    if inputs['create_reference_shoreline']:
        t_sar = reference_threshold

    return shoreline, t_sar


def find_reference_threshold(settings):

    inputs = settings['inputs']

    median_dir_path = inputs['median_dir_path']
    sat_name = inputs['sat_name']
    site_name = inputs['site_name']

    models_file_path = os.path.join(os.getcwd(), 'classification', 'models')

    # create a subfolder to store the .jpg images showing the detection
    jpeg_file_path = os.path.join(median_dir_path, 'jpg_files', 'detection')
    if not os.path.exists(jpeg_file_path):
        os.makedirs(jpeg_file_path)

    if sat_name != 'S1':

        classes = settings['classes']
        band_dict = settings['bands'][sat_name]
        first_key = next(iter(band_dict))
        pixel_size = band_dict[first_key][1]
        cloud_mask_issue = settings['cloud_mask_issue']
        classifier = joblib.load(os.path.join(models_file_path, 'NN_6classes_S2.pkl'))
        min_beach_area_pixels = np.ceil(settings['min_beach_area'] / pixel_size ** 2)
        pansharpen = inputs['pansharpen']
        bands_20m = band_dict['20m']
        for SWIR_index, band in enumerate(bands_20m):
            if settings['SWIR'] == band:
                break

    with open(os.path.join(median_dir_path, site_name + '_metadata_' + sat_name + '.pkl'), 'rb') as f:
        metadata_dict = pickle.load(f)

    metadata_sat = metadata_dict[sat_name]
    file_names = metadata_sat['file_names']
    image_epsg = metadata_sat['epsg'][0]

    printProgress('file_names loaded')

    printProgress('displaying reference histogram')

    # close all open figures
    plt.close('all')

    for file_index, file_name in enumerate(file_names):

        if sat_name == 'S1':

            file_path = os.path.join(median_dir_path, sat_name, file_name)

            if file_index == 0:
                full_image, georef = NOC_preprocess.preprocess_sar(file_path)
                full_image = np.expand_dims(full_image, axis=3)

            else:
                sar_image, georef = NOC_preprocess.preprocess_sar(file_path)
                sar_image = np.expand_dims(sar_image, axis=3)
                full_image = np.append(full_image, sar_image, axis=3)

        else:

            file_paths = []

            for band_key in band_dict:
                file_paths.append(os.path.join(median_dir_path, sat_name,
                                               band_key, file_name + '_' + band_key + '.tif'))

            if file_index == 0:
                full_image, georef, cloud_mask, image_extra, image_QA, image_nodata = \
                    NOC_preprocess.preprocess_single(file_paths, sat_name, cloud_mask_issue,
                                                     pansharpen=pansharpen,
                                                     SWIR_index=SWIR_index)
                full_image = np.expand_dims(full_image, axis=3)

            else:
                optical_image, georef, cloud_mask, image_extra, image_QA, image_nodata = \
                    NOC_preprocess.preprocess_single(file_paths, sat_name, cloud_mask_issue,
                                                     pansharpen=pansharpen,
                                                     SWIR_index=SWIR_index)
                optical_image = np.expand_dims(optical_image, axis=3)
                full_image = np.append(full_image, optical_image, axis=3)

    full_image = np.median(full_image, axis=3)

    # calculate a buffer around the reference shoreline if it has already been generated
    buffer_shape = (full_image.shape[0], full_image.shape[1])
    image_ref_buffer = np.ones(buffer_shape, dtype=np.bool)

    if sat_name == 'S1':
        reference_shoreline, reference_threshold = adjust_detection_sar(full_image, image_ref_buffer, image_epsg,
                                         georef, settings)

        with open(os.path.join(median_dir_path, site_name + '_reference_shoreline.pkl'), 'wb') as f:
            pickle.dump(reference_shoreline, f)

    else:
        _, image_labels = NOC_classify.classify_image_NN(full_image, classes, cloud_mask,
                                                         min_beach_area_pixels, classifier)

        _, reference_threshold = adjust_detection_optical(full_image, cloud_mask, image_labels,
                                             image_ref_buffer, image_epsg, georef,
                                             settings, sat_name)

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    if sat_name == 'S1' and inputs['create_reference_shoreline']:
        printSuccess(f'reference shoreline saved, reference threshold: {reference_threshold:4.3f}')
    else:
        printSuccess(f'reference threshold: {reference_threshold:4.3f}')

    return reference_threshold


def process_sar_shoreline(contours, georef, image_epsg, settings):

    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    # remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        if a.length >= settings['min_length_shoreline']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    shoreline = np.transpose(np.array([x_points, y_points]))

    return shoreline
